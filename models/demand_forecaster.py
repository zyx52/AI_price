"""
需求预测模型
主: LightGBM 预测客流量
备: sklearn GradientBoosting (无依赖环境保底)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Any, cast

from utils.logger import get_logger

logger = get_logger("DemandForecaster")


# 选择性导入 —— LightGBM 优先,不可用时fallback
try:
    import lightgbm as lgb  # type: ignore[import-not-found]
    _HAS_LGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    lgb = cast(Any, None)
    _HAS_LGB = False
    logger.warning("LightGBM未安装,降级使用sklearn GradientBoosting")


class DemandForecaster:
    """
    客流需求预测器

    输入特征:
        price, temperature, rainfall, is_holiday, is_weekend,
        day_of_week, month, season_id, competitor_avg_price
    输出:
        预测客流量 visitors
    """

    FEATURES = [
        "price", "temperature", "rainfall",
        "is_holiday", "is_weekend",
        "day_of_week", "month", "season_id",
    ]

    SEASON_MAP = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}

    def __init__(self):
        self.model: Optional[Any] = None
        self.is_trained = False

    # ---------- 特征工程 ----------
    @classmethod
    def build_features(cls, df: pd.DataFrame) -> pd.DataFrame:
        """从原始数据构造特征矩阵"""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["season_id"] = df["season"].map(cls.SEASON_MAP).fillna(0).astype(int)
        df["is_holiday"] = df["is_holiday"].astype(int)
        df["is_weekend"] = df["is_weekend"].astype(int)
        return df

    # ---------- 训练 ----------
    def train(self, df: pd.DataFrame, target: str = "visitors") -> dict:
        logger.info("开始训练需求预测模型...")
        df = self.build_features(df)
        X = df[self.FEATURES].astype(np.float64)
        y = np.asarray(df[target].values, dtype=np.float64)

        # 时间切分(末尾20%作验证)
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y[:split], y[split:]

        if _HAS_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=500, learning_rate=0.05,
                max_depth=6, num_leaves=31,
                min_child_samples=20, verbose=-1,
            )
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            self.model = model
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.05, max_depth=5,
            )
            model.fit(X_tr, y_tr)
            self.model = model

        # 评估
        model = cast(Any, self.model)
        pred_val = np.asarray(model.predict(X_val), dtype=np.float64)
        y_val_arr = np.asarray(y_val, dtype=np.float64)
        mae = float(np.mean(np.abs(pred_val - y_val_arr)))
        mape = float(np.mean(np.abs((pred_val - y_val_arr) / np.maximum(y_val_arr, 1.0))) * 100)
        self.is_trained = True

        logger.info(f"训练完成 | MAE={mae:.0f} | MAPE={mape:.2f}%")
        return {"mae": mae, "mape": mape, "n_train": len(X_tr), "n_val": len(X_val)}

    # ---------- 预测 ----------
    def predict(self, feature_row: dict) -> float:
        """
        单日预测客流
        feature_row 需包含:
          price, temperature, rainfall, is_holiday, is_weekend,
          day_of_week, month, season_id
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练,请先调用 train()")
        if self.model is None:
            raise RuntimeError("模型未初始化,请先调用 train()")

        x = pd.DataFrame([[feature_row.get(f, 0.0) for f in self.FEATURES]],
                         columns=self.FEATURES)
        # 确保所有列都是浮点型,避免LightGBM类型检查失败
        x = x.astype(np.float64)
        model = cast(Any, self.model)
        pred_arr = np.asarray(model.predict(x), dtype=np.float64)
        pred = float(pred_arr[0])
        return max(0.0, pred)

    # ---------- 需求-价格曲线 ----------
    def demand_curve(
        self, feature_row: dict,
        price_grid: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        固定其他条件,扫描价格得到 需求-价格曲线
        用于定价引擎寻找收入最大化点
        """
        if price_grid is None:
            price_grid = np.linspace(80, 599, 80)
        rows = []
        for p in price_grid:
            row = dict(feature_row)
            row["price"] = p
            visitors = self.predict(row)
            rows.append({"price": p, "visitors": visitors, "revenue": p * visitors})
        return pd.DataFrame(rows)

    # ---------- 特征重要性 ----------
    def feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return pd.DataFrame()
        importances = getattr(self.model, "feature_importances_", None)
        if importances is None:
            return pd.DataFrame()
        return (pd.DataFrame({"feature": self.FEATURES, "importance": importances})
                  .sort_values("importance", ascending=False)
                  .reset_index(drop=True))
