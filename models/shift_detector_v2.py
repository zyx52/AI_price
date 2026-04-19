"""
需求3: AutoEncoder 分布偏移检测 + 增量训练闭环

旧版 shift_detector 的问题:
  - 只用单变量 Z-score,无法捕捉"多变量联合分布"偏移
    (例如 temperature=25, rainfall=10 单独都正常,但组合在一起训练集没见过)
  - 只有"检测+降级",没有"学习-迭代"闭环

本模块升级:
  1. 用 AutoEncoder 学习训练集的完整流形,用重建误差衡量 OOD 程度
  2. 引入 IncrementalTrainingManager: CRITICAL 偏移的数据自动打标入库
  3. 达到阈值数量后异步触发增量训练任务
  4. 与旧版 DistributionShiftDetector 组合使用(双重检测)

无需深度学习依赖:
  用 PCA 作为 "线性 AutoEncoder",在 scikit-learn 上就能跑。
  保留 torch 可选接口,装了 torch 自动启用深层 AutoEncoder。
"""
from __future__ import annotations
import time
import threading
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Callable, Any, cast
from pathlib import Path
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from utils.logger import get_logger
from models.shift_detector import DistributionShiftDetector, ShiftLevel, ShiftDetection
from config import settings

logger = get_logger("AutoEncoderShift")

try:
    import torch  # type: ignore[import-not-found]
    import torch.nn as nn  # type: ignore[import-not-found]
    _HAS_TORCH = True
except ImportError:
    torch = cast(Any, None)
    nn = cast(Any, None)
    _HAS_TORCH = False

try:
    import redis  # type: ignore[import-not-found]
except ImportError:
    redis = cast(Any, None)


# ============================================================
# 深层 AutoEncoder (torch可用时)
# ============================================================
if _HAS_TORCH:
    class _DeepAutoEncoder(nn.Module):  # type: ignore[name-defined]
        def __init__(self, input_dim: int, hidden_dim: int = 8):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32), nn.ReLU(),
                nn.Linear(32, 16), nn.ReLU(),
                nn.Linear(16, hidden_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, 16), nn.ReLU(),
                nn.Linear(16, 32), nn.ReLU(),
                nn.Linear(32, input_dim),
            )

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)


# ============================================================
# AutoEncoder 偏移检测
# ============================================================
class AutoEncoderShiftDetector:
    """用重建误差检测联合分布偏移"""

    FEATURES = ["temperature", "rainfall", "price", "visitors"]
    # 分级阈值(基于训练集重建误差分布的分位数)
    LIGHT_PERCENTILE = 90
    SEVERE_PERCENTILE = 97
    CRITICAL_PERCENTILE = 99.5

    def __init__(self, use_torch: bool = False, hidden_dim: int = 8):
        self.use_torch = use_torch and _HAS_TORCH
        self.hidden_dim = hidden_dim
        self.scaler: Optional[StandardScaler] = None
        self.model: Optional[Any] = None
        # 分级阈值
        self.light_threshold = 0.0
        self.severe_threshold = 0.0
        self.critical_threshold = 0.0
        self.fitted = False

    # ---------- 训练 ----------
    def fit(self, history_df: pd.DataFrame):
        X = history_df[self.FEATURES].fillna(0).values
        self.scaler = StandardScaler()
        X_std = self.scaler.fit_transform(X)

        if self.use_torch:
            self._fit_torch(X_std)
        else:
            # 线性 AE = PCA
            self.model = PCA(n_components=min(self.hidden_dim, X_std.shape[1] - 1))
            self.model.fit(X_std)

        # 计算训练集重建误差,得到阈值
        errors = self._reconstruction_error(X_std)
        self.light_threshold = float(np.percentile(errors, self.LIGHT_PERCENTILE))
        self.severe_threshold = float(np.percentile(errors, self.SEVERE_PERCENTILE))
        self.critical_threshold = float(np.percentile(errors, self.CRITICAL_PERCENTILE))
        self.fitted = True
        logger.info(f"AutoEncoder训练完成 (torch={self.use_torch}) | "
                    f"阈值 light={self.light_threshold:.3f} / "
                    f"severe={self.severe_threshold:.3f} / "
                    f"critical={self.critical_threshold:.3f}")

    def _fit_torch(self, X_std: np.ndarray, epochs: int = 100, lr: float = 1e-3):
        if not self.use_torch:
            raise RuntimeError("Torch 模式未启用")
        X_t = torch.FloatTensor(X_std)
        model = _DeepAutoEncoder(X_std.shape[1], self.hidden_dim)
        self.model = model
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        crit = nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            out = model(X_t)
            loss = crit(out, X_t)
            loss.backward()
            opt.step()

    def _reconstruction_error(self, X_std: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("AutoEncoder 模型尚未训练")

        if self.use_torch:
            model = cast(Any, self.model)
            with torch.no_grad():
                X_t = torch.FloatTensor(X_std)
                rec = model(X_t).numpy()
        else:
            model = cast(PCA, self.model)
            z = model.transform(X_std)
            rec = model.inverse_transform(z)
        return np.mean((X_std - rec) ** 2, axis=1)

    # ---------- 检测 ----------
    def detect(self, input_features: Dict[str, float]) -> dict:
        if not self.fitted:
            return {"level": "unknown", "error": 0.0, "threshold": 0.0}
        if self.scaler is None:
            return {"level": "unknown", "error": 0.0, "threshold": 0.0}

        # 使用 FEATURES 中实际提供的字段,缺失用0
        vec = np.array([[input_features.get(f, 0.0) for f in self.FEATURES]])
        x_std = self.scaler.transform(vec)
        err = float(self._reconstruction_error(x_std)[0])

        if err >= self.critical_threshold:
            level = "critical"
        elif err >= self.severe_threshold:
            level = "severe"
        elif err >= self.light_threshold:
            level = "light"
        else:
            level = "normal"

        return {
            "level": level,
            "reconstruction_error": err,
            "threshold_light": self.light_threshold,
            "threshold_severe": self.severe_threshold,
            "threshold_critical": self.critical_threshold,
        }


# ============================================================
# 联合检测器: Z-score + AutoEncoder 双重检测
# ============================================================
class EnhancedShiftDetector:
    """
    组合了 统计检测(Z-score) + AutoEncoder(流形重建)
    输出: 两个检测器的综合结果,取更严格的一级
    """
    def __init__(self, use_torch: bool = False):
        self.zscore_detector = DistributionShiftDetector()
        self.autoencoder_detector = AutoEncoderShiftDetector(use_torch=use_torch)
        self.fitted = False

    def fit(self, history_df: pd.DataFrame):
        self.zscore_detector.fit(history_df)
        self.autoencoder_detector.fit(history_df)
        self.fitted = True

    def detect(
        self,
        input_features: Dict[str, float],
        day_type: str,
        weather: str,
        model_predictions: Optional[Dict[str, float]] = None,
    ) -> ShiftDetection:
        """返回与旧版兼容的 ShiftDetection,额外信息塞到 reasons 里"""
        zs = self.zscore_detector.detect(input_features, day_type, weather, model_predictions)
        ae = self.autoencoder_detector.detect(input_features)

        # 综合两者:取更严格的级别
        level_order = [ShiftLevel.NORMAL, ShiftLevel.LIGHT, ShiftLevel.SEVERE, ShiftLevel.CRITICAL]
        ae_level = {
            "normal": ShiftLevel.NORMAL, "light": ShiftLevel.LIGHT,
            "severe": ShiftLevel.SEVERE, "critical": ShiftLevel.CRITICAL,
        }.get(ae["level"], ShiftLevel.NORMAL)

        final_level = zs.level if level_order.index(zs.level) >= level_order.index(ae_level) else ae_level
        zs.level = final_level
        zs.adjusted_weights = DistributionShiftDetector.WEIGHTS_BY_LEVEL[final_level]
        zs.fallback_triggered = (final_level == ShiftLevel.CRITICAL)
        # 附加AE信息
        zs.reasons.append(
            f"AE重建误差={ae['reconstruction_error']:.3f} (critical>={ae['threshold_critical']:.3f}) → {ae['level']}"
        )
        return zs


class AnomalyEventPublisher:
    """将 CRITICAL 熔断上下文推送到消息队列(默认 Redis Stream)。"""

    def __init__(
        self,
        redis_url: Optional[str] = None,
        stream_name: Optional[str] = None,
        enabled: Optional[bool] = None,
    ):
        self.redis_url = redis_url or settings.redis_url
        self.stream_name = stream_name or settings.redis_stream_name
        self.enabled = bool(settings.redis_enabled) if enabled is None else bool(enabled)
        self._client = None

    def _get_client(self):
        if not self.enabled or redis is None:
            return None
        if self._client is not None:
            return self._client
        try:
            self._client = redis.from_url(  # type: ignore[attr-defined]
                self.redis_url,
                decode_responses=True,
                socket_timeout=float(settings.redis_socket_timeout_seconds),
            )
            self._client.ping()
            return self._client
        except Exception as e:
            logger.warning(f"Redis消息队列连接失败,降级为本地日志: {e}")
            self._client = None
            return None

    def publish(self, payload: Dict[str, Any]) -> bool:
        client = self._get_client()
        if client is None:
            logger.warning(f"[ANOMALY_EVENT_FALLBACK] {json.dumps(payload, ensure_ascii=False)}")
            return False
        try:
            client.xadd(  # type: ignore[attr-defined]
                self.stream_name,
                {
                    "event_type": "ood_fallback_critical",
                    "payload": json.dumps(payload, ensure_ascii=False),
                },
                maxlen=10000,
                approximate=True,
            )
            return True
        except Exception as e:
            logger.error(f"异常上下文推送消息队列失败: {e}")
            return False


# ============================================================
# 增量训练数据池 + 触发器
# ============================================================
@dataclass
class AnomalyRecord:
    timestamp: float
    date: str
    features: Dict[str, float]
    shift_level: str
    reasons: List[str]


class IncrementalTrainingManager:
    """
    当 CRITICAL 偏移数据累积到阈值时,
    自动触发异步增量训练任务
    """

    def __init__(
        self,
        storage_path: str = "./data/anomalies",
        retrain_threshold: int = 50,
        retrain_callback: Optional[Callable[[List[AnomalyRecord]], None]] = None,
        event_publisher: Optional[AnomalyEventPublisher] = None,
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.retrain_threshold = retrain_threshold
        self.retrain_callback = retrain_callback
        self.event_publisher = event_publisher or AnomalyEventPublisher()
        self.pending_pool: List[AnomalyRecord] = []
        self.published_events = 0
        self._lock = threading.Lock()
        self._retraining = False

    def label_anomaly(
        self,
        date: str,
        features: Dict[str, float],
        shift_info: ShiftDetection,
    ):
        """
        把异常样本打标入库 (闭环核心)
        """
        if shift_info.level not in (ShiftLevel.SEVERE, ShiftLevel.CRITICAL):
            return

        record = AnomalyRecord(
            timestamp=time.time(),
            date=date,
            features=dict(features),
            shift_level=shift_info.level.value,
            reasons=shift_info.reasons,
        )
        with self._lock:
            self.pending_pool.append(record)
            # 持久化(以防进程重启丢失)
            self._persist(record)

            # CRITICAL 熔断样本强制推送消息队列,供在线增量训练消费
            if shift_info.level == ShiftLevel.CRITICAL:
                published = self.event_publisher.publish(
                    {
                        "timestamp": record.timestamp,
                        "date": record.date,
                        "shift_level": record.shift_level,
                        "features": record.features,
                        "reasons": record.reasons,
                        "fallback_triggered": True,
                    }
                )
                if published:
                    self.published_events += 1

            pool_size = len(self.pending_pool)
            logger.warning(
                f"📌 异常样本打标 | level={record.shift_level} | "
                f"pool={pool_size}/{self.retrain_threshold}"
            )

            # 触发增量训练
            if pool_size >= self.retrain_threshold and not self._retraining:
                self._trigger_retrain()

    def _persist(self, record: AnomalyRecord):
        fn = self.storage_path / f"anomaly_{int(record.timestamp)}.json"
        try:
            with open(fn, "w") as f:
                json.dump(asdict(record), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"异常样本持久化失败: {e}")

    def _trigger_retrain(self):
        """异步启动增量训练"""
        self._retraining = True
        records = list(self.pending_pool)
        self.pending_pool.clear()

        def _run():
            try:
                logger.warning(f"🔄 触发增量训练 | 样本数={len(records)}")
                if self.retrain_callback:
                    self.retrain_callback(records)
                else:
                    logger.info("  [未配置retrain_callback,本次仅记录]")
                logger.warning(f"✅ 增量训练完成")
            except Exception as e:
                logger.error(f"增量训练失败: {e}", exc_info=True)
            finally:
                self._retraining = False

        threading.Thread(target=_run, daemon=True).start()

    def get_status(self) -> dict:
        with self._lock:
            return {
                "pending_anomalies": len(self.pending_pool),
                "threshold": self.retrain_threshold,
                "is_retraining": self._retraining,
                "stored_count": len(list(self.storage_path.glob("anomaly_*.json"))),
                "published_events": self.published_events,
                "event_stream": self.event_publisher.stream_name,
                "event_stream_enabled": self.event_publisher.enabled,
            }

    def force_retrain(self):
        """运维手动触发"""
        with self._lock:
            if len(self.pending_pool) == 0:
                logger.info("异常池为空,无需训练")
                return False
            self._trigger_retrain()
            return True
