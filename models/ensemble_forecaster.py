"""
Ensemble 需求预测器

融合三个模型:
  1. LightGBM   —— 树模型,擅长捕捉非线性交互
  2. XGBoost    —— 另一个树模型,不同偏差
  3. Prophet    —— 时序模型,擅长季节性+节假日

融合方式: 基于验证集MAPE的加权融合(表现越好的模型权重越高)
预期效果: 相比单一LightGBM,MAPE从7.65%降到4-5%
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any

from utils.logger import get_logger
from .feature_engineer import AdvancedFeatureEngineer

logger = get_logger("EnsembleForecaster")

# 选择性导入
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from prophet import Prophet
    _HAS_PROPHET = True
except ImportError:
    _HAS_PROPHET = False

from sklearn.ensemble import GradientBoostingRegressor


class EnsembleDemandForecaster:
    """三模型Ensemble需求预测器"""

    def __init__(self):
        self.models: Dict[str, object] = {}
        self.weights: Dict[str, float] = {}
        self.model_metrics: Dict[str, Dict] = {}
        self.is_trained = False
        self.feature_list = AdvancedFeatureEngineer.ALL_FEATURES
        self._history_tail: Optional[pd.DataFrame] = None  # 用于预测时的滞后特征

    # ---------- 训练 ----------
    def train(self, df: pd.DataFrame, target: str = "visitors") -> dict:
        logger.info("开始Ensemble训练...")

        # P2-03: 用 TimeSeriesSplit 滚动交叉验证替代单次切分
        from sklearn.model_selection import TimeSeriesSplit

        # 构造特征
        featured = AdvancedFeatureEngineer.build_features(df, is_training=True)
        self._history_tail = featured.tail(60).copy()

        # P0-03: 训练完后立刻预热特征缓存
        from utils.feature_cache import feature_cache
        feature_cache.preload_baseline(df)

        X = featured[self.feature_list]
        y = featured[target].values
        dates = featured["date"].values

        # ===== P2-03: 用 TimeSeriesSplit 做 3 折滚动验证,计算平均 MAPE =====
        n_splits = 3
        tscv = TimeSeriesSplit(n_splits=n_splits)
        logger.info(f"使用 TimeSeriesSplit 滚动CV (n_splits={n_splits})")

        fold_mapes: Dict[str, List[float]] = {}
        # 我们需要保留最后一折的模型作为生产模型
        final_predictions = {}
        final_y_val = None
        final_date_val = None

        fold_idx = 0
        for tr_idx, val_idx in tscv.split(X):
            fold_idx += 1
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y[tr_idx], y[val_idx]
            date_tr, date_val = dates[tr_idx], dates[val_idx]

            fold_preds: Dict[str, np.ndarray] = {}

            # ---- LightGBM ----
            if _HAS_LGB:
                m = lgb.LGBMRegressor(  # type: ignore[name-defined]
                    n_estimators=800, learning_rate=0.03,
                    max_depth=7, num_leaves=63,
                    min_child_samples=15, reg_alpha=0.1, reg_lambda=0.1,
                    verbose=-1,
                )
                m.fit(np.asarray(X_tr, dtype=np.float64), np.asarray(y_tr, dtype=np.float64), 
                      eval_set=[(np.asarray(X_val, dtype=np.float64), np.asarray(y_val, dtype=np.float64))],
                      callbacks=[lgb.early_stopping(50, verbose=False)])  # type: ignore[name-defined]
                if fold_idx == n_splits:
                    self.models["lightgbm"] = m
                fold_preds["lightgbm"] = np.asarray(m.predict(X_val))
            else:
                m = GradientBoostingRegressor(n_estimators=400, learning_rate=0.04, max_depth=6)
                m.fit(X_tr, np.asarray(y_tr, dtype=np.float64))
                if fold_idx == n_splits:
                    self.models["gbr"] = m
                fold_preds["gbr"] = np.asarray(m.predict(X_val))

            # ---- XGBoost ----
            if _HAS_XGB:
                m = xgb.XGBRegressor(  # type: ignore[name-defined]
                    n_estimators=800, learning_rate=0.03,
                    max_depth=6, min_child_weight=5,
                    reg_alpha=0.1, reg_lambda=0.1,
                    early_stopping_rounds=50, verbosity=0,
                )
                m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
                if fold_idx == n_splits:
                    self.models["xgboost"] = m
                fold_preds["xgboost"] = np.asarray(m.predict(X_val))

            # ---- Prophet ----
            if _HAS_PROPHET:
                try:
                    prophet_df = pd.DataFrame({"ds": date_tr, "y": y_tr})
                    p = Prophet(  # type: ignore[name-defined]
                        yearly_seasonality="auto", weekly_seasonality="auto",
                        changepoint_prior_scale=0.05,
                    )
                    import logging as _l
                    _l.getLogger("prophet").setLevel(_l.WARNING)
                    _l.getLogger("cmdstanpy").setLevel(_l.WARNING)
                    p.fit(prophet_df)
                    future = pd.DataFrame({"ds": date_val})
                    fold_preds["prophet"] = np.asarray(p.predict(future)["yhat"].values)
                    if fold_idx == n_splits:
                        self.models["prophet"] = p
                except Exception as e:
                    logger.warning(f"[fold {fold_idx}] Prophet训练失败: {e}")

            # 统计各模型本折MAPE
            for name, pred in fold_preds.items():
                pred_array: np.ndarray = np.asarray(pred)
                pred_clip = np.clip(pred_array, 0, None)
                y_val_array: np.ndarray = np.asarray(y_val)
                mape = float(np.mean(np.abs((pred_clip - y_val_array) / np.maximum(y_val_array, 1))) * 100)
                fold_mapes.setdefault(name, []).append(mape)

            # 最后一折保留预测结果,用于 ensemble 整体评估
            if fold_idx == n_splits:
                final_predictions = fold_preds
                final_y_val = y_val
                final_date_val = date_val

            y_val_array = np.asarray(y_val)
            logger.info(f"  fold {fold_idx}/{n_splits}: " +
                        " | ".join([f"{k}={np.mean(np.abs((np.clip(np.asarray(v),0,None)-y_val_array)/np.maximum(y_val_array,1)))*100:.2f}%"
                                    for k, v in fold_preds.items()]))

        # 平均MAPE作为每个模型的稳定评估
        for name, mapes in fold_mapes.items():
            avg_mape = float(np.mean(mapes))
            avg_mae_list = []  # 重新计算MAE(最后一折)
            if name in final_predictions:
                avg_mae_list.append(float(np.mean(np.abs(np.clip(final_predictions[name], 0, None) - final_y_val))))
            self.model_metrics[name] = {
                "mape": avg_mape,
                "mae": float(np.mean(avg_mae_list)) if avg_mae_list else 0.0,
                "fold_mapes": mapes,
            }
            logger.info(f"  {name}: 平均MAPE={avg_mape:.2f}% | 各折={[f'{m:.2f}' for m in mapes]}")

        # 用最后一折评估集做ensemble对比
        predictions = final_predictions
        y_val = final_y_val

        # ===== 融合权重: softmax(-MAPE) =====
        mapes = np.array([m["mape"] for m in self.model_metrics.values()])
        # 倒数归一化: MAPE越小权重越大
        inv = 1.0 / (mapes + 1e-6)
        weights = inv / inv.sum()
        for i, name in enumerate(self.model_metrics.keys()):
            self.weights[name] = float(weights[i])

        # ===== Ensemble MAPE =====
        y_val_array: np.ndarray = np.asarray(y_val)
        ensemble_pred = np.zeros(len(y_val_array))
        for name, pred in predictions.items():
            ensemble_pred += self.weights[name] * np.clip(np.asarray(pred), 0, None)
        ensemble_mape = float(np.mean(np.abs((ensemble_pred - y_val_array) / np.maximum(y_val_array, 1))) * 100)
        ensemble_mae = float(np.mean(np.abs(ensemble_pred - y_val_array)))

        self.is_trained = True
        logger.info(f"Ensemble融合完成 | MAPE={ensemble_mape:.2f}% | MAE={ensemble_mae:.0f}")
        logger.info(f"融合权重: {self.weights}")

        return {
            "ensemble_mape": ensemble_mape,
            "ensemble_mae": ensemble_mae,
            "individual_metrics": self.model_metrics,
            "weights": self.weights,
        }

    # ---------- 预测 ----------
    def predict(self, feature_row: dict, target_date: Optional[str] = None) -> float:
        """
        单次预测 (P2-02: 失败显式抛出,不使用伪造特征)
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")

        # 用批处理接口,保证与 demand_curve 的推理一致性(P0-02)
        dates = [target_date] if target_date else None
        result = self.predict_batch([feature_row], target_dates=dates)
        return float(result[0])

    # ---------- 批处理预测(P0-02) ----------
    def predict_batch(
        self,
        feature_rows: List[dict],
        target_dates: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        批处理预测 —— 一次计算多个场景,大幅降低推理开销
        
        feature_rows: List[dict],每个dict是一个预测场景
        target_dates: 可选,对应的日期列表(Prophet用)
        
        返回: np.ndarray shape=(len(feature_rows),)
        """
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        if not feature_rows:
            return np.array([])

        # 一次性构造完整特征矩阵
        try:
            X_batch = self._build_batch_features(feature_rows)
        except Exception as e:
            # 按P2-02要求,不隐式吞咽错误,记录并降级
            logger.error(f"批量特征构造失败: {e}", exc_info=True)
            raise

        n = len(feature_rows)
        predictions_sum = np.zeros(n)
        weights_sum = 0.0

        # ===== 各模型批量推理(一次predict全部样本) =====
        if "lightgbm" in self.models:
            model_lgb: Any = self.models["lightgbm"]
            preds = np.clip(model_lgb.predict(X_batch), 0, None)
            w = self.weights["lightgbm"]
            predictions_sum += w * preds
            weights_sum += w

        if "gbr" in self.models:
            model_gbr: Any = self.models["gbr"]
            preds = np.clip(model_gbr.predict(X_batch), 0, None)
            w = self.weights["gbr"]
            predictions_sum += w * preds
            weights_sum += w

        if "xgboost" in self.models:
            model_xgb: Any = self.models["xgboost"]
            preds = np.clip(model_xgb.predict(X_batch), 0, None)
            w = self.weights["xgboost"]
            predictions_sum += w * preds
            weights_sum += w

        # Prophet也可批处理(一次传入所有日期)
        if "prophet" in self.models and target_dates:
            try:
                pdate = pd.DataFrame({"ds": pd.to_datetime(target_dates)})
                model_prophet: Any = self.models["prophet"]
                prophet_preds = model_prophet.predict(pdate)["yhat"].values
                prophet_preds = np.clip(prophet_preds, 0, None)
                w = self.weights["prophet"]
                predictions_sum += w * prophet_preds
                weights_sum += w
            except Exception as e:
                logger.warning(f"Prophet批量推理失败,跳过: {e}")

        if weights_sum == 0:
            raise RuntimeError("所有模型都不可用")

        return np.clip(predictions_sum / weights_sum, 0, None)

    def _build_batch_features(self, feature_rows: List[dict]) -> pd.DataFrame:
        """
        批量构造特征矩阵 (P0-03: 走特征缓存,只做轻量级实时字段替换)
        """
        from utils.feature_cache import feature_cache

        # 优先走缓存
        cached_df = feature_cache.merge_batch_with_realtime(
            batch_realtime=feature_rows,
            feature_columns=self.feature_list,
        )
        return cached_df

    def demand_curve(
        self, feature_row: dict,
        price_grid: Optional[np.ndarray] = None,
        target_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """需求-价格曲线 (P0-02: 批处理实现,替代原for循环)"""
        if price_grid is None:
            price_grid = np.linspace(80, 599, 80)

        # 一次构造80个场景的feature_rows
        batch = []
        for p in price_grid:
            r = dict(feature_row)
            r["price"] = float(p)
            batch.append(r)

        # 一次批量推理(替代80次单点predict)
        dates = [target_date] * len(price_grid) if target_date else None
        visitors = self.predict_batch(batch, target_dates=dates)

        return pd.DataFrame({
            "price": price_grid,
            "visitors": visitors,
            "revenue": price_grid * visitors,
        })

    def individual_predictions(self, feature_row: dict, target_date: Optional[str] = None) -> Dict[str, float]:
        """返回各模型单独预测,便于调试"""
        if not self.is_trained:
            return {}
        try:
            X_pred = AdvancedFeatureEngineer.build_single_prediction_features(
                feature_row, self._history_tail or pd.DataFrame()
            )
        except Exception:
            X_pred = pd.DataFrame([[feature_row.get(f, 0) for f in self.feature_list]],
                                  columns=self.feature_list)
        preds = {}
        if "lightgbm" in self.models:
            model_lgb_pred: Any = self.models["lightgbm"]
            preds["lightgbm"] = float(model_lgb_pred.predict(X_pred)[0])
        if "xgboost" in self.models:
            model_xgb_pred: Any = self.models["xgboost"]
            preds["xgboost"] = float(model_xgb_pred.predict(X_pred)[0])
        if "prophet" in self.models and target_date:
            try:
                pdate = pd.DataFrame({"ds": [pd.to_datetime(target_date)]})
                model_prophet_pred: Any = self.models["prophet"]
                preds["prophet"] = float(model_prophet_pred.predict(pdate)["yhat"].iloc[0])
            except Exception:
                pass
        return preds
