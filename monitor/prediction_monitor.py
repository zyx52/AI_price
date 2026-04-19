"""
预测偏差实时监控 (PredictionMonitor)

需求5:
  - 记录每日【预测值 vs 实际值】
  - 持续计算滚动 MAPE
  - MAPE 连续 N 天超标 → 自动告警
  - 提供 Streamlit 可视化接口
"""
from __future__ import annotations
import json
import time
import threading
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Callable, Dict
import numpy as np
import pandas as pd

from utils.logger import get_logger

logger = get_logger("PredictionMonitor")


@dataclass
class PredictionRecord:
    date: str
    predicted_visitors: float
    actual_visitors: float
    abs_pct_error: float          # |predicted - actual| / actual
    recorded_at: float            # timestamp


@dataclass
class MonitorAlert:
    alert_id: str
    alert_time: float
    rolling_mape: float           # 当前滚动MAPE
    threshold: float              # MAPE告警阈值
    consecutive_breach_days: int  # 连续超标天数
    recent_records: List[PredictionRecord]
    message: str

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class DistributionRecord:
    date: str
    p10_visitors: float
    p50_visitors: float
    p90_visitors: float
    uncertainty_spread: float
    shift_level: str
    fallback_mode: bool
    recorded_at: float


class PredictionMonitor:
    """
    预测偏差实时监控
    
    使用:
      mon = PredictionMonitor(mape_threshold=0.12, consecutive_days=3)
      mon.record_prediction(date, predicted)
      ...当天结束后有实际数据再...
      mon.record_actual(date, actual_visitors)
      
      status = mon.get_status()   # 当前状态
      df = mon.get_dataframe()    # Streamlit用
    """

    def __init__(
        self,
        mape_threshold: float = 0.12,       # MAPE超过12%视为超标
        consecutive_days: int = 3,          # 连续超标3天触发告警
        window_size: int = 14,              # 滚动窗口14天
        storage_path: str = "./data/monitor",
        alert_callback: Optional[Callable[[MonitorAlert], None]] = None,
    ):
        self.mape_threshold = mape_threshold
        self.consecutive_days = consecutive_days
        self.window_size = window_size
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.alert_callback = alert_callback

        self.records: Dict[str, PredictionRecord] = {}
        self.distribution_records: Dict[str, DistributionRecord] = {}
        self.pending_predictions: Dict[str, float] = {}   # date -> predicted
        self.alerts_log: List[MonitorAlert] = []
        self._lock = threading.Lock()
        self._load_persisted()

    def _load_persisted(self):
        """启动时读取历史记录"""
        fn = self.storage_path / "records.json"
        if fn.exists():
            try:
                with open(fn) as f:
                    data = json.load(f)
                for item in data:
                    rec = PredictionRecord(**item)
                    self.records[rec.date] = rec
                logger.info(f"加载历史监控记录: {len(self.records)}条")
            except Exception as e:
                logger.warning(f"读取监控记录失败: {e}")

        dist_fn = self.storage_path / "distribution_records.json"
        if dist_fn.exists():
            try:
                with open(dist_fn) as f:
                    data = json.load(f)
                for item in data:
                    rec = DistributionRecord(**item)
                    self.distribution_records[rec.date] = rec
                logger.info(f"加载分布监控记录: {len(self.distribution_records)}条")
            except Exception as e:
                logger.warning(f"读取分布监控记录失败: {e}")

    def _persist(self):
        """保存记录(完整重写)"""
        fn = self.storage_path / "records.json"
        try:
            with open(fn, "w") as f:
                json.dump([asdict(r) for r in self.records.values()], f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"监控记录持久化失败: {e}")

    def _persist_distribution(self):
        fn = self.storage_path / "distribution_records.json"
        try:
            with open(fn, "w") as f:
                json.dump([asdict(r) for r in self.distribution_records.values()], f,
                          ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"分布监控记录持久化失败: {e}")

    # ========== 核心接口 ==========
    def record_prediction(self, date: str, predicted_visitors: float):
        """记录对某日的预测(当日决策时调用)"""
        with self._lock:
            self.pending_predictions[date] = float(predicted_visitors)
            logger.info(f"预测记录 | {date} → {predicted_visitors:.0f}人")

    def record_distribution(
        self,
        date: str,
        p10_visitors: float,
        p50_visitors: float,
        p90_visitors: float,
        uncertainty_spread: float,
        shift_level: str = "normal",
        fallback_mode: bool = False,
    ):
        with self._lock:
            rec = DistributionRecord(
                date=date,
                p10_visitors=float(p10_visitors),
                p50_visitors=float(p50_visitors),
                p90_visitors=float(p90_visitors),
                uncertainty_spread=float(uncertainty_spread),
                shift_level=str(shift_level),
                fallback_mode=bool(fallback_mode),
                recorded_at=time.time(),
            )
            self.distribution_records[date] = rec
            self._persist_distribution()

    def record_actual(self, date: str, actual_visitors: float):
        """记录某日实际客流(第二天获取结算数据时调用)"""
        actual = float(actual_visitors)
        with self._lock:
            predicted = self.pending_predictions.get(date)
            if predicted is None:
                logger.warning(f"无对应预测记录: {date}")
                return None
            abs_err = abs(predicted - actual) / max(actual, 1)
            rec = PredictionRecord(
                date=date,
                predicted_visitors=predicted,
                actual_visitors=actual,
                abs_pct_error=abs_err,
                recorded_at=time.time(),
            )
            self.records[date] = rec
            self.pending_predictions.pop(date, None)
            self._persist()

            # 检查告警
            alert = self._check_alert()
            if alert:
                self.alerts_log.append(alert)
                logger.warning(f"🚨 MAPE监控告警: {alert.message}")
                if self.alert_callback:
                    try:
                        self.alert_callback(alert)
                    except Exception as e:
                        logger.error(f"alert_callback 失败: {e}")
            return rec

    # ========== 告警检测 ==========
    def _check_alert(self) -> Optional[MonitorAlert]:
        """检查连续N天MAPE超标"""
        sorted_dates = sorted(self.records.keys())
        if len(sorted_dates) < self.consecutive_days:
            return None

        # 取最后 consecutive_days 天
        recent_dates = sorted_dates[-self.consecutive_days:]
        recent_records = [self.records[d] for d in recent_dates]

        # 所有最近N天MAPE均超阈值
        all_exceed = all(r.abs_pct_error > self.mape_threshold for r in recent_records)
        if not all_exceed:
            return None

        # 滚动MAPE(最近window_size天)
        window_dates = sorted_dates[-self.window_size:]
        window_records = [self.records[d] for d in window_dates]
        rolling_mape = float(np.mean([r.abs_pct_error for r in window_records]))

        alert_id = f"mape_alert_{int(time.time())}"
        msg = (f"连续{self.consecutive_days}天MAPE超标 "
               f"(阈值{self.mape_threshold:.1%}) | "
               f"滚动{self.window_size}天MAPE={rolling_mape:.2%}")

        return MonitorAlert(
            alert_id=alert_id,
            alert_time=time.time(),
            rolling_mape=rolling_mape,
            threshold=self.mape_threshold,
            consecutive_breach_days=self.consecutive_days,
            recent_records=recent_records,
            message=msg,
        )

    # ========== 查询接口 ==========
    def get_status(self) -> dict:
        with self._lock:
            ood_trigger_count = sum(
                1 for r in self.distribution_records.values()
                if (r.shift_level in ("severe", "critical") or r.fallback_mode)
            )
            fallback_count = sum(1 for r in self.distribution_records.values() if r.fallback_mode)
            dist_total = len(self.distribution_records)

            if not self.records:
                return {"n_records": 0, "rolling_mape": None,
                        "latest_date": None, "alerts": len(self.alerts_log),
                        "pending": len(self.pending_predictions),
                        "distribution_records": dist_total,
                        "ood_trigger_count": ood_trigger_count,
                        "fallback_count": fallback_count,
                        "fallback_rate": (fallback_count / dist_total) if dist_total else 0.0}

            sorted_recs = sorted(self.records.values(), key=lambda r: r.date)
            window = sorted_recs[-self.window_size:]
            rolling_mape = float(np.mean([r.abs_pct_error for r in window]))
            return {
                "n_records": len(self.records),
                "rolling_mape": rolling_mape,
                "rolling_window": self.window_size,
                "mape_threshold": self.mape_threshold,
                "latest_date": sorted_recs[-1].date,
                "alerts_total": len(self.alerts_log),
                "last_alert": self.alerts_log[-1].to_dict() if self.alerts_log else None,
                "pending_predictions": len(self.pending_predictions),
                "threshold_breach_now": rolling_mape > self.mape_threshold,
                "distribution_records": dist_total,
                "ood_trigger_count": ood_trigger_count,
                "fallback_count": fallback_count,
                "fallback_rate": (fallback_count / dist_total) if dist_total else 0.0,
            }

    def get_dataframe(self) -> pd.DataFrame:
        """供 Streamlit 可视化"""
        with self._lock:
            rows = []
            for r in sorted(self.records.values(), key=lambda x: x.date):
                rows.append({
                    "date": r.date,
                    "predicted": r.predicted_visitors,
                    "actual": r.actual_visitors,
                    "abs_pct_error": r.abs_pct_error,
                    "exceed_threshold": r.abs_pct_error > self.mape_threshold,
                })
            return pd.DataFrame(rows)

    def get_alerts(self) -> List[MonitorAlert]:
        with self._lock:
            return list(self.alerts_log)

    def get_distribution_dataframe(self) -> pd.DataFrame:
        with self._lock:
            rows = []
            for r in sorted(self.distribution_records.values(), key=lambda x: x.date):
                rows.append({
                    "date": r.date,
                    "p10_visitors": r.p10_visitors,
                    "p50_visitors": r.p50_visitors,
                    "p90_visitors": r.p90_visitors,
                    "uncertainty_spread": r.uncertainty_spread,
                    "shift_level": r.shift_level,
                    "fallback_mode": r.fallback_mode,
                })
            return pd.DataFrame(rows)
