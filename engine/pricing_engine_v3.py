"""
定价引擎 v3 —— 深度架构升级版

整合以下升级:
  1. FeatureService  —— 废除手动构造 feature_row,统一由特征中心提供
  2. QuantileForecaster —— 使用分位数预测,基于VaR做保守决策
  3. EnhancedShiftDetector —— AutoEncoder + Z-score 双重检测
  4. IncrementalTrainingManager —— CRITICAL偏移自动入库闭环

VaR 定价策略:
  - 不确定性低(uncertainty_ratio<0.3) → 选 P50 下的收入最大化价格
  - 不确定性高(>0.5) → 选 P10 下的"最差情况最优"价格(保守)
  - 中等 → 加权混合
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any
import importlib
import numpy as np
import pandas as pd

from utils.logger import get_logger
from utils.date_utils import get_day_type
settings = importlib.import_module("config").settings

logger = get_logger("PricingEngineV3")


@dataclass
class PricingDecisionV3:
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
    # v3 新增
    shift_detection: Optional[dict] = None
    fallback_mode: bool = False
    # 分位数信息
    visitors_p10: Optional[float] = None
    visitors_p50: Optional[float] = None
    visitors_p90: Optional[float] = None
    uncertainty_ratio: Optional[float] = None
    uncertainty_spread: Optional[float] = None
    risk_aware_mode: bool = False   # 是否走了VaR保守决策
    conservative_by_uncertainty: bool = False
    var_p10_price: Optional[float] = None  # P10下的最优价
    var_p50_price: Optional[float] = None  # P50下的最优价

    def to_dict(self):
        return asdict(self)


class PricingEngineV3:
    """v3版定价引擎"""

    DEFAULT_WEIGHTS = (0.30, 0.45, 0.25)

    # VaR相关阈值
    UNCERTAINTY_LOW = 0.30   # <0.30: 走P50最大化
    UNCERTAINTY_HIGH = 0.40  # 默认 >0.40: 强制走P10保守

    def __init__(
        self,
        forecaster=None,          # 期望是 QuantileEnsembleForecaster
        rl_pricer=None,
        feature_service=None,     # FeatureService
        shift_detector=None,      # EnhancedShiftDetector 或 旧版
        incremental_manager=None, # IncrementalTrainingManager
        weights=None,
    ):
        self.forecaster = forecaster
        self.rl_pricer = rl_pricer
        self.feature_service = feature_service
        self.shift_detector = shift_detector
        self.incremental_manager = incremental_manager
        self.default_weights = weights or self.DEFAULT_WEIGHTS
        self.uncertainty_high = float(getattr(settings, "uncertainty_spread_threshold", self.UNCERTAINTY_HIGH))

    # ============================================================
    # 1. 业务规则
    # ============================================================
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

    # ============================================================
    # 2. ML + VaR 定价
    # ============================================================
    def _ml_optimal_price_with_var(
        self, feature_row: dict, target_date: Optional[str] = None,
    ) -> Tuple[float, dict]:
        """
        基于分位数预测的 VaR 定价
        
        返回 (推荐价格, VaR相关元信息)
        """
        if self.forecaster is None or not getattr(self.forecaster, "is_trained", False):
            return settings.pricing.base_price, {}

        # 优先使用 demand_curve_with_quantiles (QuantileForecaster)
        if hasattr(self.forecaster, "demand_curve_with_quantiles"):
            curve = self.forecaster.demand_curve_with_quantiles(feature_row)
        else:
            # 降级: 用原来的 demand_curve,伪造 p10/p50/p90 都等于 p50
            import inspect
            sig = inspect.signature(self.forecaster.demand_curve)
            kwargs = {}
            if target_date and "target_date" in sig.parameters:
                kwargs["target_date"] = target_date
            curve = self.forecaster.demand_curve(feature_row, **kwargs)
            curve["visitors_p10"] = curve["visitors"] * 0.85
            curve["visitors_p50"] = curve["visitors"]
            curve["visitors_p90"] = curve["visitors"] * 1.15
            curve["revenue_p10"] = curve["price"] * curve["visitors_p10"]
            curve["revenue_p50"] = curve["price"] * curve["visitors_p50"]
            curve["revenue_p90"] = curve["price"] * curve["visitors_p90"]
            curve["uncertainty_ratio"] = 0.3

        # 各分位下的总收入(含二消)
        sec_ratio = settings.secondary_consumption_ratio
        curve["total_p10"] = curve["revenue_p10"] * (1 + sec_ratio)
        curve["total_p50"] = curve["revenue_p50"] * (1 + sec_ratio)
        curve["total_p90"] = curve["revenue_p90"] * (1 + sec_ratio)

        # 负载率惩罚(用P50客流计算)
        curve["load_rate"] = curve["visitors_p50"] / settings.park_capacity
        curve["load_penalty"] = 80 * (curve["load_rate"] - settings.optimal_load) ** 2 * 1000

        # === 三种定价策略的最优价 ===
        curve["score_p50"] = curve["total_p50"] - curve["load_penalty"]  # 期望最大化
        curve["score_p10"] = curve["total_p10"] - curve["load_penalty"]  # 悲观最大化(VaR)
        curve["score_var"] = 0.7 * curve["score_p10"] + 0.3 * curve["score_p50"]  # 风险厌恶混合

        best_p50 = curve.loc[curve["score_p50"].idxmax()]
        best_p10 = curve.loc[curve["score_p10"].idxmax()]
        best_var = curve.loc[curve["score_var"].idxmax()]

        # 根据整体不确定性挑策略
        avg_uncertainty = float(curve["uncertainty_ratio"].median())
        high_uncertainty_triggered = avg_uncertainty > self.uncertainty_high
        if avg_uncertainty < self.UNCERTAINTY_LOW:
            chosen_price = float(best_p50["price"])
            mode = "expected_max"
        elif high_uncertainty_triggered:
            chosen_price = float(best_p10["price"])
            mode = "p10_hard_floor"
        else:
            chosen_price = float(best_var["price"])
            mode = "risk_aware_blend"

        info = {
            "var_p10_price": float(best_p10["price"]),
            "var_p50_price": float(best_p50["price"]),
            "uncertainty_ratio": avg_uncertainty,
            "uncertainty_spread": avg_uncertainty,
            "risk_aware_mode": mode != "expected_max",
            "conservative_by_uncertainty": high_uncertainty_triggered,
            "strategy": mode,
        }
        return chosen_price, info

    # ============================================================
    # 3. RL
    # ============================================================
    def _rl_price(
        self,
        day_type,
        weather,
        competitor_avg,
        load_rate=0.5,
        prev_price=None,
        temperature: float = 22.0,
        rainfall: float = 0.0,
    ) -> float:
        if self.rl_pricer is None:
            return settings.pricing.base_price
        try:
            if hasattr(self.rl_pricer, "env"):
                res = self.rl_pricer.recommend_price(
                    day_type=day_type,
                    weather=weather,
                    competitor_avg=competitor_avg,
                    load_rate=load_rate,
                    prev_price=prev_price,
                    temperature=temperature,
                    rainfall=rainfall,
                )
            else:
                try:
                    res = self.rl_pricer.recommend_price(
                        day_type=day_type,
                        weather=weather,
                        competitor_avg=competitor_avg,
                        load_rate=load_rate,
                        prev_price=prev_price,
                        temperature=temperature,
                        rainfall=rainfall,
                    )
                except TypeError:
                    res = self.rl_pricer.recommend_price(day_type, weather, competitor_avg)
            return res["recommended_price"]
        except Exception as e:
            logger.error(f"RL推荐失败: {e}", exc_info=True)
            return settings.pricing.base_price

    # ============================================================
    # 4. 核心 decide
    # ============================================================
    def decide(
        self, date, weather, temperature, rainfall,
        competitor_prices, day_type=None, prev_price=None,
    ) -> PricingDecisionV3:
        forecaster: Any = self.forecaster
        day_type = day_type or get_day_type(date)
        competitor_avg = float(np.mean(list(competitor_prices.values())))

        price_rule = self._business_rule_price(day_type, weather)

        # FeatureService 是 v3 的强依赖,禁止在 decide 内手工拼接特征
        if self.feature_service is None:
            raise RuntimeError("PricingEngineV3 requires FeatureService for feature construction")
        feature_row = self.feature_service.build_for_engine(
            date=str(date), weather=weather,
            temperature=temperature, rainfall=rainfall,
            day_type=day_type, base_price=settings.pricing.base_price,
            competitor_avg=competitor_avg,
        )

        # === ML + VaR ===
        try:
            price_ml, var_info = self._ml_optimal_price_with_var(feature_row, str(date))
            individual_preds = {}
            if forecaster is not None and hasattr(forecaster, "individual_predictions"):
                try:
                    individual_preds = forecaster.individual_predictions(feature_row, str(date))
                except Exception:
                    pass
        except Exception as e:
            logger.error(f"ML预测失败: {e}", exc_info=True)
            price_ml = price_rule
            var_info = {}
            individual_preds = {}

        # === Shift Detection ===
        shift_info = None
        w_rule, w_ml, w_rl = self.default_weights
        fallback_mode = False

        if self.shift_detector is not None:
            shift_info = self.shift_detector.detect(
                input_features={"temperature": temperature, "rainfall": rainfall,
                                "price": price_rule, "visitors": 20000},  # AE需要4个特征
                day_type=day_type, weather=weather,
                model_predictions=individual_preds,
            )
            w_rule, w_ml, w_rl = shift_info.adjusted_weights
            fallback_mode = shift_info.fallback_triggered

            # === 需求3闭环: CRITICAL/SEVERE 自动打标 ===
            if self.incremental_manager is not None:
                self.incremental_manager.label_anomaly(
                    date=str(date),
                    features={"temperature": temperature, "rainfall": rainfall,
                              "day_type": day_type, "weather": weather},
                    shift_info=shift_info,
                )

        # === RL ===
        price_rl = self._rl_price(
            day_type,
            weather,
            competitor_avg,
            0.5,
            prev_price,
            temperature=temperature,
            rainfall=rainfall,
        )

        # 不确定性散布度超过阈值后,强制进入保守策略(限制RL影响,聚焦业务规则+P10基线)
        conservative_by_uncertainty = bool(var_info.get("conservative_by_uncertainty", False))
        if conservative_by_uncertainty and not fallback_mode:
            w_rule, w_ml, w_rl = (0.70, 0.30, 0.00)

        # === 加权融合 ===
        final_price = w_rule * price_rule + w_ml * price_ml + w_rl * price_rl

        # 安全约束
        bound_low = settings.pricing.base_price * (1 - settings.pricing.max_daily_change)
        bound_high = settings.pricing.base_price * (1 + settings.pricing.max_daily_change) * 1.5
        final_price = float(np.clip(final_price,
                                    max(bound_low, settings.pricing.min_price),
                                    min(bound_high, settings.pricing.max_price)))
        final_price = round(final_price / 5) * 5

        # === 最终客流(带分位) ===
        feature_row["price"] = final_price
        visitors_p10 = visitors_p50 = visitors_p90 = None
        uncertainty = None
        predicted_visitors: float = settings.park_capacity * 0.5
        try:
            if forecaster is not None and hasattr(forecaster, "predict") and callable(forecaster.predict):
                if hasattr(forecaster, "demand_curve_with_quantiles"):
                    # 分位预测
                    pred = forecaster.predict(feature_row)
                    p50 = getattr(pred, "p50", None)
                    if p50 is not None:
                        visitors_p10 = float(getattr(pred, "p10", p50))
                        visitors_p50 = float(p50)
                        visitors_p90 = float(getattr(pred, "p90", p50))
                        uncertainty_val = getattr(pred, "uncertainty_ratio", None)
                        uncertainty = float(uncertainty_val) if uncertainty_val is not None else None
                        predicted_visitors = float(p50)
                    else:
                        pred_value: Any = pred
                        predicted_visitors = float(pred_value)
                else:
                    import inspect
                    sig = inspect.signature(forecaster.predict)
                    if "target_date" in sig.parameters:
                        pred_value: Any = forecaster.predict(feature_row, target_date=str(date))
                        predicted_visitors = float(pred_value)
                    else:
                        pred_value: Any = forecaster.predict(feature_row)
                        predicted_visitors = float(pred_value)
        except Exception as e:
            logger.error(f"最终客流预测失败: {e}", exc_info=True)
            predicted_visitors = float(settings.park_capacity * 0.5)

        load_rate = predicted_visitors / settings.park_capacity
        predicted_revenue = predicted_visitors * final_price * (1 + settings.secondary_consumption_ratio)

        # === 置信度 ===
        prices = np.array([price_rule, price_ml, price_rl])
        base_conf = float(np.clip(1 - prices.std() / max(prices.mean(), 1), 0.3, 0.99))
        if shift_info and shift_info.level.value != "normal":
            penalty = {"light": 0.15, "severe": 0.35, "critical": 0.60}[shift_info.level.value]
            base_conf = max(0.1, base_conf - penalty)
        if uncertainty is not None and uncertainty > self.uncertainty_high:
            base_conf = max(0.1, base_conf - 0.15)

        reasoning = (
            f"业务规则¥{price_rule:.0f}(权重{w_rule:.0%}), "
            f"ML最优¥{price_ml:.0f}(权重{w_ml:.0%}), "
            f"RL推荐¥{price_rl:.0f}(权重{w_rl:.0%}), "
            f"综合¥{final_price:.0f}。"
        )
        if var_info.get("risk_aware_mode"):
            reasoning += f" [VaR模式: {var_info.get('strategy')}, 不确定性={var_info.get('uncertainty_ratio'):.2%}]"
        if conservative_by_uncertainty:
            reasoning += " [UncertaintySpread超阈值,已切换P10保守定价]"
        if fallback_mode:
            reasoning += " ⚠️ 检测到极端分布偏移,100%业务规则兜底。"

        return PricingDecisionV3(
            date=str(date),
            recommended_price=final_price,
            predicted_visitors=float(predicted_visitors),
            predicted_revenue=float(predicted_revenue),
            load_rate=float(load_rate),
            decision_weights={"business_rule": w_rule, "ml": w_ml, "rl": w_rl},
            price_breakdown={"business_rule": price_rule, "ml_optimal": price_ml, "rl_recommended": price_rl},
            business_rule_price=price_rule,
            ml_optimal_price=price_ml,
            rl_recommended_price=price_rl,
            day_type=day_type, weather=weather,
            confidence=base_conf,
            reasoning=reasoning,
            shift_detection=shift_info.to_dict() if shift_info else None,
            fallback_mode=fallback_mode,
            visitors_p10=visitors_p10,
            visitors_p50=visitors_p50,
            visitors_p90=visitors_p90,
            uncertainty_ratio=uncertainty,
            uncertainty_spread=uncertainty,
            risk_aware_mode=bool(var_info.get("risk_aware_mode")),
            conservative_by_uncertainty=conservative_by_uncertainty,
            var_p10_price=var_info.get("var_p10_price"),
            var_p50_price=var_info.get("var_p50_price"),
        )
