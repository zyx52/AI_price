"""
FastAPI 服务 v3 (升级到 PricingEngineV3 全栈)

继承 v2 全部优势:
  P0-01: 全异步路由 + run_in_threadpool
  P2-01: Pydantic Field 强校验
  P2-02: 结构化错误码,无隐式吞咽
  P0-03: 启动时预热特征缓存
  P3-01: 配置热更新

v3 新增:
  ✓ FeatureService (特征中心) 自动注入
  ✓ QuantileEnsembleForecaster (P10/P50/P90 + VaR定价)
  ✓ EnhancedShiftDetector (Z-score + AutoEncoder 双重检测)
  ✓ IncrementalTrainingManager (异常自动入库 + 异步增量训练)
  ✓ ContinuousRLPricer (24维状态 + GNN嵌入)
  ✓ PredictionMonitor (预测偏差实时监控 + MAPE告警)

新增接口:
  GET  /monitor/status                     当前监控状态
  GET  /monitor/predictions                获取预测vs实际明细
  POST /monitor/record-actual              录入实际客流(供回归校准)
  GET  /monitor/alerts                     告警历史
  GET  /admin/anomaly-pool                 异常样本池状态
  POST /admin/force-retrain                运维手动触发增量训练
  POST /pricing/decide-with-quantiles      增强版定价(带分位)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator
from typing import Dict, Optional, List, Any
import requests

from data import DataLoader
from models import (
    QuantileEnsembleForecaster, ContinuousRLPricer,
    FeatureService, EnhancedShiftDetector, IncrementalTrainingManager,
    NLPReporter, ParkAttractionGraph,
)
from engine import (
    PricingEngineV3, BundleOptimizer,
    TimeSlotPricer, ChannelPricer, CapacityManager,
)
from monitor import AlertEngine, PredictionMonitor
from utils.date_utils import get_day_type
from utils.logger import get_logger
from utils.feature_cache import feature_cache
from config import settings

logger = get_logger("api_v3")


# ============================================================
# 生命周期: v3 全栈初始化
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"🚀 API v3 启动 | env={settings.env} | park={settings.park_name}")

    def _sync_init():
        # 1. 数据
        loader = DataLoader(source="mock")
        history = loader.load_history()

        # 2. 分位数预测器(替代单点Ensemble)
        forecaster = QuantileEnsembleForecaster()
        forecaster.train(history)

        # 3. 特征服务
        feature_service = FeatureService(history)

        # 4. 增强偏移检测 + 增量训练
        shift_detector = EnhancedShiftDetector(use_torch=False)
        shift_detector.fit(history)

        # 5. 增量训练管理器(异常自动入库)
        def _retrain_handler(records):
            logger.warning(f"📥 增量训练触发 | 收到 {len(records)} 条异常样本")
            # 真实场景: 把样本喂给 forecaster.partial_fit() 或重新训练子模型
            # 这里仅记录,生产可对接 Celery/Airflow

        incremental_mgr = IncrementalTrainingManager(
            storage_path="./data/anomalies",
            retrain_threshold=20,
            retrain_callback=_retrain_handler,
        )

        # 6. 连续表征RL(带GNN嵌入)
        graph = ParkAttractionGraph(); graph.build_default_park()
        rl = ContinuousRLPricer(
            forecaster=forecaster, history_df=history,
            attraction_graph=graph,
        )
        rl.train(total_timesteps=5000)

        # 7. 预测偏差监控
        monitor = PredictionMonitor(
            mape_threshold=0.12, consecutive_days=3,
            storage_path="./data/monitor",
        )

        return {
            "history": history,
            "forecaster": forecaster,
            "feature_service": feature_service,
            "shift_detector": shift_detector,
            "incremental_mgr": incremental_mgr,
            "rl_pricer": rl,
            "graph": graph,
            "monitor": monitor,
            "engine": PricingEngineV3(
                forecaster=forecaster, rl_pricer=rl,
                feature_service=feature_service,
                shift_detector=shift_detector,
                incremental_manager=incremental_mgr,
            ),
            "bundle_opt": BundleOptimizer(),
            "alert_engine": AlertEngine(),
            "reporter": NLPReporter(use_llm=False),
            "time_slot": TimeSlotPricer(),
            "channel": ChannelPricer(),
            "capacity": CapacityManager(),
        }

    state = await run_in_threadpool(_sync_init)
    for k, v in state.items():
        setattr(app.state, k, v)
    logger.info(f"✅ v3 全栈就绪 | 特征缓存: {feature_cache.health()}")
    yield
    logger.info("👋 API 关闭")


app = FastAPI(
    title="AI智能定价票务平台 API v3",
    version="3.0.0",
    description="深度升级版 · FeatureService · 分位数回归 · AutoEncoder OOD · PPO+GNN · 预测监控",
    lifespan=lifespan,
)


def _remote_inference_enabled() -> bool:
    return bool(getattr(settings, "inference_service_url", "").strip())


def _call_remote_decide(req: "PricingRequest", with_quantiles: bool = False) -> dict:
    base = settings.inference_service_url.strip().rstrip("/")
    endpoint = "/pricing/decide-with-quantiles" if with_quantiles else "/pricing/decide"
    url = f"{base}{endpoint}"
    timeout = float(getattr(settings, "inference_service_timeout_seconds", 3.0))
    resp = requests.post(url, json=req.model_dump(), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("远程推理返回结构非法")
    return data


def _monitor_record_from_decision_payload(date: str, payload: dict):
    try:
        pred = payload.get("predicted_visitors")
        if pred is not None:
            app.state.monitor.record_prediction(date, float(pred))
        p10 = payload.get("visitors_p10")
        p50 = payload.get("visitors_p50")
        p90 = payload.get("visitors_p90")
        if p10 is not None and p50 is not None and p90 is not None:
            shift_raw = payload.get("shift_detection")
            shift: Dict[str, Any] = shift_raw if isinstance(shift_raw, dict) else {}
            app.state.monitor.record_distribution(
                date=date,
                p10_visitors=float(p10),
                p50_visitors=float(p50),
                p90_visitors=float(p90),
                uncertainty_spread=float(payload.get("uncertainty_spread") or payload.get("uncertainty_ratio") or 0.0),
                shift_level=str(shift.get("level", "normal")),
                fallback_mode=bool(payload.get("fallback_mode", False)),
            )
    except Exception as e:
        logger.warning(f"监控记录失败(不影响主流程): {e}")


# ============================================================
# 请求模型 (P2-01 强校验,沿用v2)
# ============================================================
_ALLOWED_WEATHERS = {"晴好", "雨", "暴雨", "酷热", "严寒"}


class PricingRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-05-02"])
    weather: str = Field("晴好")
    temperature: float = Field(24.0, ge=-30.0, le=50.0)
    rainfall: float = Field(0.0, ge=0.0, le=500.0)
    competitor_prices: Dict[str, float] = Field(default={"A": 310, "B": 280, "C": 350})
    day_type: Optional[str] = Field(None, pattern=r"^(weekday|weekend|holiday|golden_week)$")
    prev_price: Optional[float] = Field(None, ge=1.0, le=10000.0)

    @field_validator("weather")
    @classmethod
    def _w(cls, v):
        if v not in _ALLOWED_WEATHERS:
            raise ValueError(f"weather必须是 {_ALLOWED_WEATHERS} 之一")
        return v

    @field_validator("competitor_prices")
    @classmethod
    def _c(cls, v):
        if not v:
            raise ValueError("竞品价字典不能为空")
        for n, p in v.items():
            if not (1.0 <= p <= 10000.0):
                raise ValueError(f"竞品{n}价格{p}超出[1,10000]")
        return v


class BundleRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-05-02"])
    weather: str = Field("晴好")
    temperature: float = Field(24.0, ge=-30.0, le=50.0)
    rainfall: float = Field(0.0, ge=0.0, le=500.0)
    load_rate: float = Field(0.6, ge=0.0, le=1.0)


class ActualVisitorsRequest(BaseModel):
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", examples=["2026-05-02"])
    actual_visitors: float = Field(..., ge=0, le=200000)


# ============================================================
# 基础路由
# ============================================================
@app.get("/")
async def root():
    return {
        "service": "AI Pricing Platform",
        "version": "3.0.0",
        "env": settings.env,
        "park": settings.park_name,
        "endpoints": {
            "定价": [
                "POST /pricing/decide",
                "POST /pricing/decide-with-quantiles  ★v3新",
                "POST /pricing/bundles",
                "POST /pricing/alerts",
                "POST /pricing/daily-report",
                "POST /pricing/time-slots",
                "POST /pricing/channels",
                "POST /pricing/capacity",
            ],
            "监控": [
                "GET  /monitor/status                  ★v3新",
                "GET  /monitor/predictions             ★v3新",
                "GET  /monitor/distributions           ★v3新",
                "GET  /monitor/ood                     ★v3新",
                "POST /monitor/record-actual           ★v3新",
                "GET  /monitor/alerts                  ★v3新",
            ],
            "运维": [
                "GET  /health",
                "POST /admin/reload-config",
                "POST /admin/invalidate-cache",
                "GET  /admin/anomaly-pool              ★v3新",
                "POST /admin/force-retrain             ★v3新",
            ],
        },
    }


@app.get("/health")
async def health():
    cache_info = feature_cache.health()
    return {
        "status": "ok" if cache_info["baseline_loaded"] else "degraded",
        "feature_cache": cache_info,
        "models_loaded": hasattr(app.state, "engine"),
        "engine_type": "PricingEngineV3",
        "inference_mode": "remote_service" if _remote_inference_enabled() else "in_process",
        "inference_service_url": settings.inference_service_url or None,
    }


# ============================================================
# 定价路由
# ============================================================
@app.post("/pricing/decide")
async def decide(req: PricingRequest):
    """综合定价决策 (v3全栈,自动P10/P50/P90 + VaR)"""
    try:
        if _remote_inference_enabled():
            remote = await run_in_threadpool(_call_remote_decide, req, False)
            _monitor_record_from_decision_payload(req.date, remote)
            return remote

        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices,
            day_type=day_type, prev_price=req.prev_price,
        )
        out = decision.to_dict()
        _monitor_record_from_decision_payload(req.date, out)
        return out
    except ValueError as e:
        raise HTTPException(400, {"code": "INVALID_INPUT", "msg": str(e)})
    except Exception as e:
        logger.exception("decide failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/decide-with-quantiles")
async def decide_with_quantiles(req: PricingRequest):
    """
    增强版定价 —— 返回完整的分位数信息和VaR策略
    适合需要风险展示的高级看板
    """
    try:
        if _remote_inference_enabled():
            remote = await run_in_threadpool(_call_remote_decide, req, True)
            decision_payload = remote.get("decision", remote)
            if isinstance(decision_payload, dict):
                _monitor_record_from_decision_payload(req.date, decision_payload)
            return remote

        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices,
            day_type=day_type, prev_price=req.prev_price,
        )
        _monitor_record_from_decision_payload(req.date, decision.to_dict())

        # 额外补充分位数详情
        feature_row = app.state.feature_service.build_for_engine(
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            day_type=day_type, base_price=decision.recommended_price,
            competitor_avg=float(sum(req.competitor_prices.values()) / len(req.competitor_prices)),
        )

        def _get_curve():
            return app.state.forecaster.demand_curve_with_quantiles(feature_row)

        curve = await run_in_threadpool(_get_curve)
        # 返回关键价格点的分位详情(精简,避免响应过大)
        sample = curve.iloc[::10]  # 每10个点取一个

        return {
            "decision": decision.to_dict(),
            "demand_curve_quantiles": [
                {
                    "price": float(r["price"]),
                    "p10_visitors": float(r["visitors_p10"]),
                    "p50_visitors": float(r["visitors_p50"]),
                    "p90_visitors": float(r["visitors_p90"]),
                    "uncertainty_ratio": float(r["uncertainty_ratio"]),
                    "uncertainty_spread": float(r["uncertainty_ratio"]),
                }
                for _, r in sample.iterrows()
            ],
        }
    except ValueError as e:
        raise HTTPException(400, {"code": "INVALID_INPUT", "msg": str(e)})
    except Exception as e:
        logger.exception("decide_with_quantiles failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/bundles")
async def bundles(req: BundleRequest):
    try:
        day_type = get_day_type(req.date)
        result = await run_in_threadpool(
            app.state.bundle_opt.suggest,
            day_type=day_type, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            load_rate=req.load_rate,
        )
        return [b.to_dict() for b in result]
    except Exception as e:
        logger.exception("bundles failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/alerts")
async def alerts(req: PricingRequest):
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices, day_type=day_type,
        )
        external = {
            "weather_forecast": {
                "rainfall_mm": req.rainfall,
                "temperature_high": req.temperature,
                "rain_probability": 0.6 if req.rainfall > 5 else 0.1,
            },
            "competitor_prices": req.competitor_prices,
        }
        alerts_list = await run_in_threadpool(
            app.state.alert_engine.check,
            external=external,
            predicted_visitors=decision.predicted_visitors,
            predicted_revenue=decision.predicted_revenue,
            history=app.state.history,
        )
        return [a.to_dict() for a in alerts_list]
    except Exception as e:
        logger.exception("alerts failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/time-slots")
async def time_slots(req: PricingRequest):
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices, day_type=day_type,
        )
        slots = await run_in_threadpool(
            app.state.time_slot.compute,
            base_price=decision.recommended_price,
            day_type=decision.day_type, weather=decision.weather,
            temperature=req.temperature,
        )
        return {
            "base_price": decision.recommended_price,
            "slots": [s.to_dict() for s in slots],
        }
    except Exception as e:
        logger.exception("time_slots failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/channels")
async def channels(req: PricingRequest):
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices, day_type=day_type,
        )
        strategies = await run_in_threadpool(
            app.state.channel.compute,
            base_price=decision.recommended_price,
            day_type=decision.day_type,
            predicted_load=decision.load_rate,
        )
        return [s.to_dict() for s in strategies]
    except Exception as e:
        logger.exception("channels failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/capacity")
async def capacity(req: PricingRequest):
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices, day_type=day_type,
        )
        plan = await run_in_threadpool(
            app.state.capacity.plan,
            date=req.date,
            predicted_visitors=int(decision.predicted_visitors),
            day_type=decision.day_type, weather=decision.weather,
        )
        return plan.to_dict()
    except Exception as e:
        logger.exception("capacity failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.post("/pricing/daily-report")
async def daily_report(req: PricingRequest):
    try:
        day_type = req.day_type or get_day_type(req.date)
        decision = await run_in_threadpool(
            app.state.engine.decide,
            date=req.date, weather=req.weather,
            temperature=req.temperature, rainfall=req.rainfall,
            competitor_prices=req.competitor_prices, day_type=day_type,
        )
        bundles_res = await run_in_threadpool(
            app.state.bundle_opt.suggest,
            day_type, req.weather, req.temperature, req.rainfall, decision.load_rate,
        )
        external = {
            "weather_forecast": {
                "rainfall_mm": req.rainfall,
                "temperature_high": req.temperature,
                "rain_probability": 0.6 if req.rainfall > 5 else 0.1,
            },
            "competitor_prices": req.competitor_prices,
        }
        alerts_list = await run_in_threadpool(
            app.state.alert_engine.check, external,
            decision.predicted_visitors, decision.predicted_revenue,
            app.state.history,
        )
        comp_avg = sum(req.competitor_prices.values()) / len(req.competitor_prices)
        brief = await run_in_threadpool(
            app.state.reporter.daily_brief,
            {
                "date": req.date,
                "recommended_price": decision.recommended_price,
                "predicted_visitors": decision.predicted_visitors,
                "predicted_revenue": decision.predicted_revenue,
                "load_rate": decision.load_rate,
                "weather": decision.weather, "day_type": decision.day_type,
                "competitor_avg": comp_avg,
                "bundle_suggestion": bundles_res[0].name if bundles_res else "—",
                "alerts": [a.title for a in alerts_list if a.level in ("critical", "warning")],
            },
        )
        return {
            "decision": decision.to_dict(),
            "bundles": [b.to_dict() for b in bundles_res],
            "alerts": [a.to_dict() for a in alerts_list],
            "narrative": brief,
        }
    except Exception as e:
        logger.exception("daily_report failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


# ============================================================
# 监控接口 ★v3 新增
# ============================================================
@app.get("/monitor/status")
async def monitor_status():
    """当前预测偏差监控状态"""
    return app.state.monitor.get_status()


@app.get("/monitor/predictions")
async def monitor_predictions(limit: int = 30):
    """获取预测vs实际明细 (供前端绘图)"""
    df = app.state.monitor.get_dataframe()
    if df.empty:
        return {"records": []}
    df_recent = df.tail(limit)
    return {
        "records": df_recent.to_dict(orient="records"),
        "total": len(df),
    }


@app.get("/monitor/distributions")
async def monitor_distributions(limit: int = 30):
    """获取分位数分布监控明细 (P10/P50/P90)"""
    df = app.state.monitor.get_distribution_dataframe()
    if df.empty:
        return {"records": []}
    df_recent = df.tail(limit)
    return {
        "records": df_recent.to_dict(orient="records"),
        "total": len(df),
    }


@app.get("/monitor/ood")
async def monitor_ood_status():
    """获取 OOD 触发次数与熔断率"""
    status = app.state.monitor.get_status()
    return {
        "ood_trigger_count": status.get("ood_trigger_count", 0),
        "fallback_count": status.get("fallback_count", 0),
        "fallback_rate": status.get("fallback_rate", 0.0),
        "distribution_records": status.get("distribution_records", 0),
    }


@app.post("/monitor/record-actual")
async def monitor_record_actual(req: ActualVisitorsRequest):
    """录入实际客流 (T+1批量调用)"""
    try:
        rec = await run_in_threadpool(
            app.state.monitor.record_actual,
            req.date, req.actual_visitors,
        )
        if rec is None:
            raise HTTPException(404, {"code": "NO_PREDICTION",
                                      "msg": f"日期{req.date}无对应预测记录"})
        return {
            "date": rec.date,
            "predicted": rec.predicted_visitors,
            "actual": rec.actual_visitors,
            "abs_pct_error": rec.abs_pct_error,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("record_actual failed")
        raise HTTPException(500, {"code": "INTERNAL_ERROR", "msg": str(e)})


@app.get("/monitor/alerts")
async def monitor_alerts(limit: int = 10):
    """告警历史"""
    alerts = app.state.monitor.get_alerts()
    return {
        "alerts": [a.to_dict() for a in alerts[-limit:]],
        "total": len(alerts),
    }


# ============================================================
# 运维接口 ★v3 新增
# ============================================================
@app.post("/admin/reload-config")
async def reload_config():
    try:
        settings.reload()
        logger.info(f"🔄 配置热更新 | env={settings.env}")
        return {
            "status": "reloaded",
            "park_name": settings.park_name,
            "pricing_base": settings.pricing.base_price,
            "alert_load_high": settings.alert_load_rate_high,
            "optimal_load": settings.optimal_load,
        }
    except Exception as e:
        logger.exception("reload failed")
        raise HTTPException(500, {"code": "RELOAD_FAILED", "msg": str(e)})


@app.post("/admin/invalidate-cache")
async def invalidate_cache():
    feature_cache.invalidate()
    return {"status": "ok", "msg": "特征缓存已清空"}


@app.get("/admin/anomaly-pool")
async def anomaly_pool():
    """异常样本池状态(P1-02闭环可视化)"""
    return app.state.incremental_mgr.get_status()


@app.post("/admin/force-retrain")
async def force_retrain():
    """运维手动触发增量训练"""
    triggered = app.state.incremental_mgr.force_retrain()
    return {
        "status": "triggered" if triggered else "skipped",
        "msg": "已启动异步训练" if triggered else "异常样本池为空",
    }
