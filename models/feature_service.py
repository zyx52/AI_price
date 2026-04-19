"""
FeatureService (特征中心)

废除 PricingEngine 中手动构造 feature_row 的做法,改为:
  1. 统一由 FeatureService 提供特征
  2. 自动计算各种窗口特征(7/14/30天rolling)
  3. 支持"实时特征"与"静态特征"分离
  4. 自动发现: 调用 feature_engineer 构建完整特征矩阵

核心理念:
  - PricingEngine 只关心"业务决策",不关心"特征怎么来"
  - 任何想加新特征的场景,只需在 FeatureService 注册一次,全局生效
  - 特征消费方(Engine / RL / Ensemble)调用同一接口,保证一致性
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any
import pandas as pd
import numpy as np

from utils.logger import get_logger
from utils.feature_cache import feature_cache

logger = get_logger("FeatureService")


@dataclass
class FeatureVector:
    """标准化的特征向量"""
    date: str
    features: Dict[str, float]
    # 元信息
    source: Dict[str, str]          # {feature_name: "cached" | "realtime" | "derived"}
    built_at: str                   # 构造时间戳
    static_keys: List[str]          # 来自缓存的滞后/滑动特征
    realtime_keys: List[str]        # 来自当前请求的实时特征
    derived_keys: List[str]         # 自动派生的窗口特征

    def to_dict(self) -> dict:
        return asdict(self)

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([self.features])


class FeatureService:
    """
    统一特征服务
    
    用法:
      svc = FeatureService(history_df)
      svc.register_derived("last_7d_avg_load", lambda hist, ctx: hist.tail(7)["load_rate"].mean())
      vec = svc.build(date="2026-05-02", realtime={"temperature": 24, "rainfall": 0})
      # vec.features → 完整的 feature dict,可直接喂给模型
    """

    def __init__(self, history_df: Optional[pd.DataFrame] = None):
        self.history = history_df
        self.derived_registry: Dict[str, Callable[[pd.DataFrame, dict], float]] = {}
        # 内置派生特征
        self._register_builtin_derived()

    # ========================================================
    # 派生特征注册 (需求1核心: 特征自动发现)
    # ========================================================
    def register_derived(self, name: str, fn: Callable[[pd.DataFrame, dict], float]):
        """注册一个派生特征函数 fn(history_df, realtime_context) -> float"""
        self.derived_registry[name] = fn
        logger.info(f"注册派生特征: {name}")

    def _register_builtin_derived(self):
        """
        内置的"窗口特征库" —— 上线时可无缝接入业务需要的窗口特征
        这些是示例常用特征,产品上线后可在配置中心按需开关
        """

        # 过去 7 天平均负载率(真实反映近期拥挤程度)
        self.register_derived(
            "last_7d_avg_load",
            lambda hist, ctx: (
                float(hist.tail(7)["visitors"].mean() / ctx.get("park_capacity", 40000))
                if "visitors" in hist.columns else 0.5
            )
        )

        # 过去 14 天平均客流
        self.register_derived(
            "last_14d_avg_visitors",
            lambda hist, ctx: float(hist.tail(14)["visitors"].mean())
            if "visitors" in hist.columns else 0.0
        )

        # 过去 30 天收入标准差(体现波动)
        self.register_derived(
            "last_30d_revenue_std",
            lambda hist, ctx: float(hist.tail(30)["revenue_total"].std())
            if "revenue_total" in hist.columns else 0.0
        )

        # 过去 7 天价格均值
        self.register_derived(
            "last_7d_avg_price",
            lambda hist, ctx: float(hist.tail(7)["price"].mean())
            if "price" in hist.columns else 299.0
        )

        # 过去 7 天同类型日期的平均客流(若是周末,就看过去周末;若工作日就看过去工作日)
        self.register_derived(
            "last_7d_same_daytype_visitors",
            lambda hist, ctx: (
                float(hist[hist["day_type"] == ctx.get("day_type", "weekday")].tail(7)["visitors"].mean())
                if ("day_type" in hist.columns and "visitors" in hist.columns) else 0.0
            )
        )

        # 7天价格变化率(趋势特征)
        self.register_derived(
            "price_trend_7d",
            lambda hist, ctx: (
                float((hist.tail(1)["price"].iloc[0] - hist.tail(7)["price"].mean())
                      / max(hist.tail(7)["price"].mean(), 1))
                if "price" in hist.columns and len(hist) >= 7 else 0.0
            )
        )

        # 竞品调价频率 (过去30天竞品价变化次数)
        self.register_derived(
            "competitor_adjustment_freq_30d",
            lambda hist, ctx: self._compute_competitor_freq(hist, ctx)
        )

        # 过去7天平均转化率
        self.register_derived(
            "past_7d_avg_conversion_rate",
            lambda hist, ctx: self._compute_avg_conversion_rate_7d(hist, ctx)
        )

        # 竞品连续调价动量
        self.register_derived(
            "competitor_price_momentum_7d",
            lambda hist, ctx: self._compute_competitor_momentum_7d(hist, ctx)
        )

        # 社交媒体天气恐慌指数
        self.register_derived(
            "social_weather_panic_index",
            lambda hist, ctx: self._estimate_weather_panic_index(hist, ctx)
        )

    @staticmethod
    def _compute_competitor_freq(hist: pd.DataFrame, ctx: dict) -> float:
        """竞品调价频率 = 过去30天竞品价变化>5%的次数"""
        if "competitor_avg_price" not in hist.columns or len(hist) < 30:
            return 2.0  # 默认值
        comp = hist.tail(30)["competitor_avg_price"].to_numpy(dtype=float, copy=False)
        diff = np.abs(np.diff(comp)) / (comp[:-1] + 1.0)
        return float(np.sum(diff > 0.05))

    @staticmethod
    def _compute_avg_conversion_rate_7d(hist: pd.DataFrame, ctx: dict) -> float:
        """过去7天平均转化率"""
        if "conversion_rate" in hist.columns:
            return float(hist.tail(7)["conversion_rate"].mean())
        if "ticket_orders" in hist.columns and "visitors" in hist.columns:
            sub = hist.tail(7)
            visitors = sub["visitors"].replace(0, np.nan)
            conv = (sub["ticket_orders"] / visitors).fillna(0.0)
            return float(conv.mean())
        return 0.05

    @staticmethod
    def _compute_competitor_momentum_7d(hist: pd.DataFrame, ctx: dict) -> float:
        """竞品连续调价动量(近7天线性斜率)"""
        if "competitor_avg_price" in hist.columns and len(hist) >= 7:
            y = hist.tail(7)["competitor_avg_price"].to_numpy(dtype=float, copy=False)
            x = np.arange(len(y), dtype=float)
            slope = np.polyfit(x, y, 1)[0]
            return float(slope)

        comp = float(ctx.get("competitor_avg", 0.0) or 0.0)
        if comp > 0:
            return float(comp - float(ctx.get("base_price", 299.0)))
        return 0.0

    @staticmethod
    def _estimate_weather_panic_index(hist: pd.DataFrame, ctx: dict) -> float:
        """社交媒体天气恐慌指数: 优先读历史字段,否则实时启发式估计"""
        if "social_weather_panic_index" in hist.columns:
            return float(hist.tail(7)["social_weather_panic_index"].mean())

        weather = str(ctx.get("weather", "晴好"))
        rainfall = float(ctx.get("rainfall", 0.0) or 0.0)
        if weather == "暴雨":
            base = 0.85
        elif weather in ("酷热", "严寒"):
            base = 0.55
        elif weather == "雨":
            base = 0.45
        else:
            base = 0.15
        return float(min(1.0, base + min(rainfall / 200.0, 0.25)))

    # ========================================================
    # 核心接口: build / build_batch
    # ========================================================
    def build(
        self,
        date: str,
        realtime: Dict[str, Any],
        feature_names: Optional[List[str]] = None,
        include_derived: bool = True,
    ) -> FeatureVector:
        """
        构建一个完整的特征向量
        
        realtime: 请求时提供的实时字段(temperature/rainfall/day_type等)
        feature_names: 指定消费方需要的特征名列表; None=返回全部
        """
        from datetime import datetime

        static_feats = self._get_static_features(feature_names)
        realtime_keys = list(realtime.keys())

        # 合并: static 被 realtime 覆盖
        merged = dict(static_feats)
        for k, v in realtime.items():
            merged[k] = v

        derived_keys = []
        source = {k: "cached" if k in static_feats else "realtime" for k in merged.keys()}

        if include_derived and self.history is not None:
            derived_ctx = dict(realtime)
            derived_ctx["date"] = date
            for name, fn in self.derived_registry.items():
                try:
                    merged[name] = float(fn(self.history, derived_ctx))
                    source[name] = "derived"
                    derived_keys.append(name)
                except Exception as e:
                    logger.warning(f"派生特征 {name} 计算失败: {e}")
                    merged[name] = 0.0
                    source[name] = "derived_fallback"

        # 按 feature_names 过滤(若指定)
        if feature_names:
            merged = {k: merged.get(k, 0.0) for k in feature_names}
            source = {k: source.get(k, "missing") for k in feature_names}

        return FeatureVector(
            date=date,
            features=merged,
            source=source,
            built_at=datetime.now().isoformat(),
            static_keys=[k for k, s in source.items() if s == "cached"],
            realtime_keys=realtime_keys,
            derived_keys=derived_keys,
        )

    def build_batch(
        self,
        dates: List[str],
        realtime_list: List[Dict[str, Any]],
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """批量构造(P0-02友好)"""
        vecs = [
            self.build(d, rt, feature_names).features
            for d, rt in zip(dates, realtime_list)
        ]
        return pd.DataFrame(vecs)

    # ========================================================
    # 内部: 静态特征(从 feature_cache 取)
    # ========================================================
    def _get_static_features(self, feature_names: Optional[List[str]] = None) -> Dict[str, float]:
        baseline = feature_cache.get_baseline_features()
        if baseline is None or len(baseline) == 0:
            return {}
        last_row = baseline.iloc[-1]
        # 只返回数值型字段
        static = {}
        for col in last_row.index:
            val = last_row[col]
            if isinstance(val, (int, float, np.integer, np.floating)):
                static[col] = float(val)
        if feature_names:
            static = {k: v for k, v in static.items() if k in feature_names}
        return static

    # ========================================================
    # 一键生成 Engine 需要的 feature_row
    # ========================================================
    def build_for_engine(
        self,
        date: str,
        weather: str,
        temperature: float,
        rainfall: float,
        day_type: str,
        base_price: float,
        competitor_avg: Optional[float] = None,
    ) -> dict:
        """
        给 PricingEngine 提供标准 feature_row
        替代原来在 decide() 里手写的逻辑
        """
        d = pd.to_datetime(date)
        realtime = {
            "price": base_price,
            "temperature": temperature,
            "rainfall": rainfall,
            "is_holiday": int(day_type in ("holiday", "golden_week")),
            "is_weekend": int(day_type == "weekend"),
            "day_of_week": int(d.dayofweek),
            "month": int(d.month),
            "day_of_month": int(d.day),
            "season_id": {3:0,4:0,5:0,6:1,7:1,8:1,9:2,10:2,11:2}.get(d.month, 3),
            "day_type_id": {"weekday":0,"weekend":1,"holiday":2,"golden_week":3}.get(day_type, 0),
            "weather_id": {"晴好":0,"雨":1,"暴雨":2,"酷热":3,"严寒":4}.get(weather, 0),
            # 给派生特征用的context
            "day_type": day_type,
            "weather": weather,
            "competitor_avg": competitor_avg if competitor_avg is not None else 0.0,
            "base_price": base_price,
        }
        vec = self.build(date, realtime)
        # 去掉非数值字段
        return {k: v for k, v in vec.features.items()
                if isinstance(v, (int, float, np.integer, np.floating))}

    # ========================================================
    # 数据更新
    # ========================================================
    def update_history(self, new_history: pd.DataFrame):
        """数据源变化后调用"""
        self.history = new_history
        # 同步刷新特征缓存
        feature_cache.preload_baseline(new_history)
        logger.info(f"FeatureService 历史已更新 | rows={len(new_history)}")
