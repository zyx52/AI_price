"""
分位数回归 Ensemble (QuantileEnsembleForecaster)

相比 EnsembleDemandForecaster (只输出点预测) 的升级:
  - 对每个价格点输出 (P10, P50, P90) 三个分位数 → 完整的客流分布
  - P50 是中位数预测(相当于原标量预测)
  - P90-P10 的宽度反映"不确定性"
  - 下游定价可做 VaR 决策:不确定性高时自动收敛保守价

技术选型:
  - LightGBM 原生支持 quantile objective
  - 训练 3 个独立模型: alpha=0.1, 0.5, 0.9
  - 相当于训练开销 ×3,但换来完整的分布信息

VaR 定价思路:
  - 期望收入 = P50客流 × 价格
  - 悲观收入 = P10客流 × 价格   (最差情况,风险价值)
  - 不确定性 ratio = (P90-P10) / P50
  - 不确定性>阈值 → 选择 P10 客流下的"风险最小化"价格,而非 P50 下的"期望最大化"价格
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Protocol, cast
from dataclasses import dataclass, asdict

from utils.logger import get_logger
from .feature_engineer import AdvancedFeatureEngineer

logger = get_logger("QuantileEnsemble")

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    GradientBoostingRegressor = None

_HAS_LGB = lgb is not None


class _QuantileModel(Protocol):
    def fit(self, X: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        ...

    def predict(self, X: Any, *args: Any, **kwargs: Any) -> np.ndarray:
        ...


@dataclass
class QuantilePrediction:
    p10: float          # 悲观 (10% 分位,实际客流更低的概率只有10%)
    p50: float          # 中位数
    p90: float          # 乐观
    uncertainty_ratio: float   # (p90-p10)/p50,反映分布宽度
    mean_estimate: float       # = p50,便于向后兼容

    def to_dict(self):
        return asdict(self)


class QuantileEnsembleForecaster:
    """
    三分位回归 Ensemble
    
    输出:
        forecaster.predict(features) → QuantilePrediction(p10, p50, p90, uncertainty_ratio)
        forecaster.demand_curve_with_quantiles(features) → DataFrame with p10/p50/p90 per price
    """

    QUANTILES = [0.1, 0.5, 0.9]

    def __init__(self):
        self.models: Dict[float, object] = {}
        self.feature_list = AdvancedFeatureEngineer.ALL_FEATURES
        self.is_trained = False
        self.metrics: Dict[str, float] = {}

    # ---------- 训练 ----------
    def train(self, df: pd.DataFrame, target: str = "visitors") -> dict:
        logger.info("QuantileEnsemble 训练...")

        featured = AdvancedFeatureEngineer.build_features(df, is_training=True)
        # 预热特征缓存
        from utils.feature_cache import feature_cache
        feature_cache.preload_baseline(df)

        X = featured[self.feature_list]
        y = np.asarray(featured[target], dtype=float)

        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y[:split], y[split:]

        for q in self.QUANTILES:
            if _HAS_LGB and lgb is not None:
                m = lgb.LGBMRegressor(
                    objective="quantile", alpha=q,
                    n_estimators=600, learning_rate=0.05,
                    max_depth=6, num_leaves=31,
                    verbose=-1,
                )
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(30, verbose=False)])
            else:
                if GradientBoostingRegressor is None:
                    raise ImportError("缺少 sklearn，无法使用 GradientBoostingRegressor 训练分位数模型")
                # sklearn GBR 也支持 quantile
                m = GradientBoostingRegressor(
                    loss="quantile", alpha=q,
                    n_estimators=300, learning_rate=0.05, max_depth=5,
                )
                m.fit(X_tr, y_tr)
            self.models[q] = m
            logger.info(f"  quantile={q} 训练完成")

        # 评估: 覆盖率 (真值落在[p10,p90]的比例)
        preds = self.predict_batch_raw(X_val)
        coverage = float(np.mean((y_val >= preds[0.1]) & (y_val <= preds[0.9])))
        # MAPE 用 p50
        mape = float(np.mean(np.abs((preds[0.5] - y_val) / np.maximum(y_val, 1))) * 100)
        avg_width = float(np.mean((preds[0.9] - preds[0.1]) / np.maximum(preds[0.5], 1)))

        self.metrics = {
            "p50_mape": mape,
            "coverage_80pct": coverage,  # 理想值 0.80
            "avg_uncertainty_width": avg_width,
        }
        self.is_trained = True
        logger.info(f"  P50 MAPE={mape:.2f}% | 80%区间覆盖={coverage:.1%} | 平均区间宽度={avg_width:.2%}")
        return self.metrics

    # ---------- 批量预测(各分位同时输出) ----------
    def predict_batch_raw(self, X: pd.DataFrame) -> Dict[float, np.ndarray]:
        """对每个分位批量预测"""
        out = {}
        for q, m in self.models.items():
            model = cast(_QuantileModel, m)
            out[q] = np.clip(model.predict(X), 0, None)
        return out

    def predict(self, feature_row: dict) -> QuantilePrediction:
        """单次预测"""
        if not self.is_trained:
            raise RuntimeError("未训练")
        from utils.feature_cache import feature_cache
        X = feature_cache.merge_with_realtime(feature_row, self.feature_list)
        preds = self.predict_batch_raw(X)
        p10 = float(preds[0.1][0])
        p50 = float(preds[0.5][0])
        p90 = float(preds[0.9][0])
        # 防止 p10 > p50 的分位数交叉问题
        p10, p50, p90 = sorted([p10, p50, p90])
        uncertainty = (p90 - p10) / max(p50, 1)
        return QuantilePrediction(
            p10=p10, p50=p50, p90=p90,
            uncertainty_ratio=uncertainty,
            mean_estimate=p50,
        )

    # ---------- 批量版本 ----------
    def predict_batch(
        self, feature_rows: List[dict],
        target_dates: Optional[List[str]] = None,
    ) -> np.ndarray:
        """与 EnsembleDemandForecaster 兼容: 返回 P50 一维数组"""
        if not feature_rows:
            return np.array([])
        from utils.feature_cache import feature_cache
        X = feature_cache.merge_batch_with_realtime(feature_rows, self.feature_list)
        preds = self.predict_batch_raw(X)
        return preds[0.5]

    def predict_quantile_batch(
        self, feature_rows: List[dict],
    ) -> Dict[str, np.ndarray]:
        """批量输出三个分位"""
        from utils.feature_cache import feature_cache
        X = feature_cache.merge_batch_with_realtime(feature_rows, self.feature_list)
        raw = self.predict_batch_raw(X)
        return {
            "p10": raw[0.1], "p50": raw[0.5], "p90": raw[0.9],
        }

    # ---------- 兼容 Engine 的 demand_curve ----------
    def demand_curve(self, feature_row: dict, target_date: Optional[str] = None) -> pd.DataFrame:
        """返回各价格点的 P50 客流/收入 (为兼容老Engine接口)"""
        price_grid = np.linspace(80, 599, 80)
        batch = [dict(feature_row, price=float(p)) for p in price_grid]
        res = self.predict_quantile_batch(batch)
        return pd.DataFrame({
            "price": price_grid,
            "visitors": res["p50"],
            "revenue": price_grid * res["p50"],
        })

    # ---------- VaR 定价核心: 带分位的价格曲线 ----------
    def demand_curve_with_quantiles(
        self, feature_row: dict, price_grid: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        批量输出每个价格对应的 (p10, p50, p90) 客流与三档收入
        下游可据此做 VaR 定价决策
        """
        if price_grid is None:
            price_grid = np.linspace(80, 599, 80)
        batch = [dict(feature_row, price=float(p)) for p in price_grid]
        res = self.predict_quantile_batch(batch)

        df = pd.DataFrame({
            "price": price_grid,
            "visitors_p10": res["p10"],
            "visitors_p50": res["p50"],
            "visitors_p90": res["p90"],
            "revenue_p10": price_grid * res["p10"],
            "revenue_p50": price_grid * res["p50"],
            "revenue_p90": price_grid * res["p90"],
        })
        # 不确定性 ratio
        df["uncertainty_ratio"] = (df["visitors_p90"] - df["visitors_p10"]) / df["visitors_p50"].clip(lower=1)
        return df

    def individual_predictions(self, feature_row: dict, target_date: Optional[str] = None) -> dict:
        """返回三个分位的预测(供shift detector用)"""
        if not self.is_trained:
            return {}
        pred = self.predict(feature_row)
        return {"p10": pred.p10, "p50": pred.p50, "p90": pred.p90}
