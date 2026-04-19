"""
定价决策引擎 (Refactored with P1-02 动态降级)

融合三路信号:
  1) 业务规则(节假日/黄金周/天气系数)   —— 可解释、安全兜底
  2) ML需求预测曲线 → 寻找收入最大化点     —— 数据驱动
  3) RL智能体推荐                         —— 长期均衡客流+收入

改进:
  - P0-02: 调用 forecaster.demand_curve() 已是批处理实现
  - P1-02: 引入 DistributionShiftDetector,OOD输入自动降级权重
  - P2-02: 预测失败显式抛出或进入规则fallback,不再静默吞咽
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.date_utils import get_day_type
from config import settings

logger = get_logger("PricingEngine")


@dataclass
class PricingDecision:
    date: str
    recommended_price: float
    predicted_visitors: float
    predicted_revenue: float
    load_rate: float
    decision_weights: dict
    price_breakdown: dict
    business_rule_price: float
    ml_optimal_price: float
    rl_recommended_price: float
    day_type: str
    weather: str
    confidence: float
    reasoning: str
    # P1-02 新增字段
    shift_detection: Optional[dict] = None
    fallback_mode: bool = False

    def to_dict(self):
        return asdict(self)


class PricingEngine:
    """综合定价引擎"""

    DEFAULT_WEIGHTS = (0.30, 0.45, 0.25)

    def __init__(
        self,
        forecaster=None,
        rl_pricer=None,
        weights: Optional[Tuple[float, float, float]] = None,
        shift_detector=None,
    ):
        self.forecaster = forecaster
        self.rl_pricer = rl_pricer
        self.shift_detector = shift_detector
        self.default_weights = weights or self.DEFAULT_WEIGHTS

    # ---------- 1. 业务规则基线 ----------
    def _business_rule_price(self, day_type: str, weather: str) -> float:
        p = settings.pricing
        h = settings.holiday
        price = p.base_price
        if day_type == "golden_week":
            price *= h.golden_week_markup
        elif day_type == "holiday":
            price *= h.holiday_markup
        elif day_type == "weekend":
            price *= h.weekend_markup
        else:
            price *= h.weekday_discount

        if weather in ("雨", "暴雨"):
            price *= 0.88
        elif weather == "酷热":
            price *= 0.94
        elif weather == "严寒":
            price *= 0.92
        return float(np.clip(price, p.min_price, p.max_price).round(0))

    # ---------- 2. ML需求曲线寻优 ----------
    def _ml_optimal_price(self, feature_row: dict, target_date: Optional[str] = None) -> Tuple[float, dict]:
        if self.forecaster is None or not getattr(self.forecaster, "is_trained", False):
            return settings.pricing.base_price, {}

        # 兼容旧版 DemandForecaster (不支持 target_date) 与新版 EnsembleDemandForecaster
        import inspect
        sig = inspect.signature(self.forecaster.demand_curve)
        kwargs = {}
        if target_date and "target_date" in sig.parameters:
            kwargs["target_date"] = target_date
        curve = self.forecaster.demand_curve(feature_row, **kwargs)
        curve["secondary"] = curve["visitors"] * settings.secondary_consumption_ratio * 130
        curve["total_rev"] = curve["revenue"] + curve["secondary"]
        curve["load_rate"] = curve["visitors"] / settings.park_capacity
        curve["load_penalty"] = 80 * (curve["load_rate"] - settings.optimal_load) ** 2 * 1000
        curve["score"] = curve["total_rev"] - curve["load_penalty"]
        best = curve.loc[curve["score"].idxmax()]

        individual = {}
        if hasattr(self.forecaster, "individual_predictions"):
            try:
                individual = self.forecaster.individual_predictions(feature_row, target_date)
            except Exception as e:
                logger.warning(f"获取模型单独预测失败: {e}")

        return float(best["price"]), individual

    # ---------- 3. RL推荐 ----------
    def _rl_price(self, day_type, weather, competitor_avg, load_rate=0.5, prev_price=None) -> float:
        if self.rl_pricer is None:
            return settings.pricing.base_price
        try:
            if hasattr(self.rl_pricer, "env"):
                res = self.rl_pricer.recommend_price(day_type, weather, competitor_avg, load_rate, prev_price)
            else:
                try:
                    res = self.rl_pricer.recommend_price(day_type, weather, competitor_avg, load_rate, prev_price)
                except TypeError:
                    res = self.rl_pricer.recommend_price(day_type, weather, competitor_avg)
        except Exception as e:
            logger.error(f"RL推荐失败: {e}", exc_info=True)
            return settings.pricing.base_price
        return res["recommended_price"]

    # ---------- 核心决策 ----------
    def decide(
        self, date, weather, temperature, rainfall,
        competitor_prices, day_type=None, prev_price=None,
    ) -> PricingDecision:
        day_type = day_type or get_day_type(date)
        competitor_avg = float(np.mean(list(competitor_prices.values())))

        price_rule = self._business_rule_price(day_type, weather)

        d = pd.to_datetime(date)
        feature_row = {
            "price": settings.pricing.base_price,
            "temperature": temperature, "rainfall": rainfall,
            "is_holiday": int(day_type in ("holiday", "golden_week")),
            "is_weekend": int(day_type == "weekend"),
            "day_of_week": int(d.dayofweek), "month": int(d.month),
            "day_of_month": int(d.day),
            "season_id": {3:0,4:0,5:0,6:1,7:1,8:1,9:2,10:2,11:2}.get(d.month, 3),
            "day_type_id": {"weekday":0,"weekend":1,"holiday":2,"golden_week":3}.get(day_type, 0),
            "weather_id": {"晴好":0,"雨":1,"暴雨":2,"酷热":3,"严寒":4}.get(weather, 0),
        }

        try:
            price_ml, individual_preds = self._ml_optimal_price(feature_row, str(date))
        except Exception as e:
            logger.error(f"ML预测失败,启用业务规则fallback: {e}", exc_info=True)
            price_ml = price_rule
            individual_preds = {}

        # === P1-02: 分布偏移检测 (只用外部输入特征,不用price) ===
        shift_info = None
        w_rule, w_ml, w_rl = self.default_weights
        fallback_mode = False

        if self.shift_detector is not None:
            shift_info = self.shift_detector.detect(
                input_features={"temperature": temperature, "rainfall": rainfall},
                day_type=day_type, weather=weather,
                model_predictions=individual_preds,
            )
            w_rule, w_ml, w_rl = shift_info.adjusted_weights
            fallback_mode = shift_info.fallback_triggered

        try:
            price_rl = self._rl_price(day_type, weather, competitor_avg, 0.5, prev_price)
        except Exception as e:
            logger.error(f"RL推荐失败: {e}", exc_info=True)
            price_rl = price_rule

        final_price = w_rule * price_rule + w_ml * price_ml + w_rl * price_rl

        bound_low = settings.pricing.base_price * (1 - settings.pricing.max_daily_change)
        bound_high = settings.pricing.base_price * (1 + settings.pricing.max_daily_change) * 1.5
        final_price = float(np.clip(final_price,
                                    max(bound_low, settings.pricing.min_price),
                                    min(bound_high, settings.pricing.max_price)))
        final_price = round(final_price / 5) * 5

        feature_row["price"] = final_price
        try:
            if self.forecaster and getattr(self.forecaster, "is_trained", False):
                # 兼容两种 predict 接口
                import inspect
                sig = inspect.signature(self.forecaster.predict)
                if "target_date" in sig.parameters:
                    predicted_visitors = self.forecaster.predict(feature_row, target_date=str(date))
                else:
                    predicted_visitors = self.forecaster.predict(feature_row)
            else:
                predicted_visitors = settings.park_capacity * 0.5
        except Exception as e:
            logger.error(f"最终客流预测失败: {e}", exc_info=True)
            predicted_visitors = settings.park_capacity * 0.5

        load_rate = predicted_visitors / settings.park_capacity
        predicted_revenue = predicted_visitors * final_price * (1 + settings.secondary_consumption_ratio)

        prices = np.array([price_rule, price_ml, price_rl])
        base_confidence = float(np.clip(1 - prices.std() / max(prices.mean(), 1), 0.3, 0.99))
        if shift_info and shift_info.level.value != "normal":
            penalty = {"light": 0.15, "severe": 0.35, "critical": 0.60}[shift_info.level.value]
            base_confidence = max(0.1, base_confidence - penalty)

        reasoning = (
            f"业务规则¥{price_rule:.0f} (权重{w_rule:.0%}), "
            f"ML最优¥{price_ml:.0f} (权重{w_ml:.0%}), "
            f"RL推荐¥{price_rl:.0f} (权重{w_rl:.0%}), "
            f"综合建议¥{final_price:.0f}, "
            f"预计客流{predicted_visitors:,.0f}人(负载率{load_rate*100:.1f}%)。"
        )
        if fallback_mode:
            reasoning += " ⚠️ 检测到极端分布偏移,启用100%业务规则兜底模式。"
        elif shift_info and shift_info.level.value != "normal":
            reasoning += f" 检测到{shift_info.level.value}级分布偏移,动态提升规则权重。"

        return PricingDecision(
            date=str(date),
            recommended_price=final_price,
            predicted_visitors=float(predicted_visitors),
            predicted_revenue=float(predicted_revenue),
            load_rate=float(load_rate),
            decision_weights={"business_rule": w_rule, "ml": w_ml, "rl": w_rl},
            price_breakdown={"business_rule": price_rule,
                             "ml_optimal": price_ml,
                             "rl_recommended": price_rl},
            business_rule_price=price_rule,
            ml_optimal_price=price_ml,
            rl_recommended_price=price_rl,
            day_type=day_type, weather=weather,
            confidence=base_confidence,
            reasoning=reasoning,
            shift_detection=shift_info.to_dict() if shift_info else None,
            fallback_mode=fallback_mode,
        )
