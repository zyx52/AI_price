"""
时间滑窗特征引擎

为 PPO 强化学习模型提供时间序列感知特征:

  1. 客流增长速率 (一阶导数) —— 感知"爆发期" vs "平缓期"
  2. 加速度 (二阶导数) —— 感知变化趋势的拐点
  3. 多时间尺度窗口统计 —— 15min / 30min / 1h / 2h
  4. 周期性对比 —— 与昨日同时段/上周同时段的偏离

所有特征自动计算,喂给 models/ppo_pricer.py 的 ParkPricingEnv。

与 FeatureService 集成:
  FeatureService.register_derived() 可直接引用此模块的函数
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from utils.feature_cache import feature_cache
from config import settings

logger = get_logger("SlidingWindow")


# ============================================================
# 数据结构
# ============================================================
@dataclass
class WindowedFeatures:
    """滑窗特征输出"""
    timestamp: str
    # 原始值
    current_visitors: int
    current_load_rate: float
    current_entry_rate: float          # 人/分钟

    # 一阶导数 (增长速率)
    growth_rate_15min: float           # 15分钟增长率 (人/分钟)
    growth_rate_30min: float
    growth_rate_1h: float
    growth_rate_2h: float

    # 二阶导数 (加速度) —— 感知"爆发期"
    acceleration_30min: float          # 增长率的变化率
    acceleration_1h: float

    # 窗口统计
    visitors_mean_2h: float
    visitors_std_2h: float
    load_rate_mean_2h: float
    entry_rate_mean_2h: float

    # 周期性对比
    vs_yesterday_same_time: float      # 与昨日同时段偏差 (ratio)
    vs_last_week_same_time: float      # 与上周同时段偏差 (ratio)

    # 趋势标签(离散)
    trend_label: str = "stable"        # "surging" | "rising" | "stable" | "declining" | "crashing"

    def to_dict(self) -> dict:
        return asdict(self)

    def as_feature_vector(self) -> Dict[str, float]:
        """转换为可直接喂给模型的 feature dict"""
        return {
            "growth_rate_15min": self.growth_rate_15min,
            "growth_rate_30min": self.growth_rate_30min,
            "growth_rate_1h": self.growth_rate_1h,
            "growth_rate_2h": self.growth_rate_2h,
            "acceleration_30min": self.acceleration_30min,
            "acceleration_1h": self.acceleration_1h,
            "visitors_mean_2h": self.visitors_mean_2h,
            "visitors_std_2h": self.visitors_std_2h,
            "entry_rate_mean_2h": self.entry_rate_mean_2h,
            "vs_yesterday_ratio": self.vs_yesterday_same_time,
            "vs_last_week_ratio": self.vs_last_week_same_time,
            "trend_surging": 1.0 if self.trend_label == "surging" else 0.0,
            "trend_rising": 1.0 if self.trend_label == "rising" else 0.0,
            "trend_declining": 1.0 if self.trend_label == "declining" else 0.0,
        }


# ============================================================
# 滑窗计算引擎
# ============================================================
class SlidingWindowEngine:
    """
    时间滑窗特征计算引擎

    内部维护一个时间戳 -> (visitors, load_rate) 的环形缓冲区,
    支持任意时间窗口的聚合计算。

    用法:
      engine = SlidingWindowEngine(max_history_hours=4)
      engine.push(visitors=15000, load_rate=0.38)
      features = engine.compute()  # → WindowedFeatures
    """

    def __init__(self, max_history_hours: float = 4.0):
        self.max_history_hours = max_history_hours
        self._buffer: Deque[Tuple[float, int, float]] = deque()  # (timestamp, visitors, load_rate)
        self._yesterday_buffer: Dict[int, int] = {}   # hour → visitors (昨日)
        self._last_week_buffer: Dict[int, int] = {}    # hour → visitors (上周)
        self._last_compute: Optional[WindowedFeatures] = None

    def push(self, visitors: int, load_rate: float, entry_rate: float = 0.0):
        """推入一个新数据点"""
        now = time.time()
        self._buffer.append((now, visitors, load_rate, entry_rate))

        # 清理过期数据
        cutoff = now - self.max_history_hours * 3600
        while self._buffer and self._buffer[0][0] < cutoff:
            self._buffer.popleft()

    def set_historical_baseline(self, yesterday_by_hour: Dict[int, int],
                                 last_week_by_hour: Dict[int, int]):
        """设置历史基线(用于周期性对比)"""
        self._yesterday_buffer = yesterday_by_hour
        self._last_week_buffer = last_week_by_hour

    def compute(self) -> Optional[WindowedFeatures]:
        """计算当前滑窗特征"""
        if len(self._buffer) < 2:
            return None

        now = time.time()
        timestamps = np.array([t for t, _, _, _ in self._buffer])
        visitors_arr = np.array([v for _, v, _, _ in self._buffer])
        load_arr = np.array([l for _, _, l, _ in self._buffer])
        entry_arr = np.array([e for _, _, _, e in self._buffer])

        current_visitors = int(visitors_arr[-1])
        current_load = float(load_arr[-1])
        current_entry = float(entry_arr[-1])

        # 一阶导数: 不同窗口的增长率
        def growth_rate(window_min: float) -> float:
            cutoff = now - window_min * 60
            mask = timestamps >= cutoff
            if mask.sum() < 2:
                return 0.0
            vals = visitors_arr[mask]
            times = timestamps[mask]
            dt = (times[-1] - times[0]) / 60.0  # 分钟
            if dt <= 0:
                return 0.0
            return (vals[-1] - vals[0]) / dt

        gr_15 = growth_rate(15)
        gr_30 = growth_rate(30)
        gr_1h = growth_rate(60)
        gr_2h = growth_rate(120)

        # 二阶导数: 加速度
        acc_30 = (gr_30 - growth_rate(15)) / 15.0 if len(self._buffer) >= 3 else 0.0
        acc_1h = (gr_1h - gr_30) / 30.0 if len(self._buffer) >= 4 else 0.0

        # 2小时窗口统计
        cutoff_2h = now - 7200
        mask_2h = timestamps >= cutoff_2h
        v_2h = visitors_arr[mask_2h]
        e_2h = entry_arr[mask_2h]
        l_2h = load_arr[mask_2h]

        visitors_mean_2h = float(np.mean(v_2h)) if len(v_2h) > 0 else float(current_visitors)
        visitors_std_2h = float(np.std(v_2h)) if len(v_2h) > 1 else 0.0
        load_mean_2h = float(np.mean(l_2h)) if len(l_2h) > 0 else float(current_load)
        entry_mean_2h = float(np.mean(e_2h)) if len(e_2h) > 0 else float(current_entry)

        # 周期性对比
        from datetime import datetime
        current_hour = datetime.now().hour
        yesterday = self._yesterday_buffer.get(current_hour, current_visitors)
        last_week = self._last_week_buffer.get(current_hour, current_visitors)
        vs_yesterday = (current_visitors / yesterday - 1.0) if yesterday > 0 else 0.0
        vs_last_week = (current_visitors / last_week - 1.0) if last_week > 0 else 0.0

        # 趋势标签
        trend = self._classify_trend(gr_30, acc_30)

        features = WindowedFeatures(
            timestamp=datetime.now().isoformat(),
            current_visitors=current_visitors,
            current_load_rate=current_load,
            current_entry_rate=current_entry,
            growth_rate_15min=round(gr_15, 2),
            growth_rate_30min=round(gr_30, 2),
            growth_rate_1h=round(gr_1h, 2),
            growth_rate_2h=round(gr_2h, 2),
            acceleration_30min=round(acc_30, 4),
            acceleration_1h=round(acc_1h, 4),
            visitors_mean_2h=round(visitors_mean_2h, 1),
            visitors_std_2h=round(visitors_std_2h, 1),
            load_rate_mean_2h=round(load_mean_2h, 4),
            entry_rate_mean_2h=round(entry_mean_2h, 2),
            vs_yesterday_same_time=round(vs_yesterday, 4),
            vs_last_week_same_time=round(vs_last_week, 4),
            trend_label=trend,
        )

        self._last_compute = features
        return features

    @staticmethod
    def _classify_trend(growth_rate_30min: float, acceleration: float) -> str:
        """根据增长率和加速度判定趋势"""
        if growth_rate_30min > 100 and acceleration > 0.5:
            return "surging"
        if growth_rate_30min > 30:
            return "rising"
        if growth_rate_30min < -50:
            return "crashing"
        if growth_rate_30min < -10:
            return "declining"
        return "stable"


# ============================================================
# 全局单例 (供 FeatureService 集成使用)
# ============================================================
_sliding_engine: Optional[SlidingWindowEngine] = None


def get_sliding_engine() -> SlidingWindowEngine:
    global _sliding_engine
    if _sliding_engine is None:
        _sliding_engine = SlidingWindowEngine(max_history_hours=4)
    return _sliding_engine


# ============================================================
# FeatureService 派生特征注册函数
# ============================================================
def register_sliding_window_features(feature_service):
    """
    将滑窗特征注册到 FeatureService

    用法:
      from models.sliding_window import register_sliding_window_features
      svc = FeatureService(history_df)
      register_sliding_window_features(svc)
    """
    engine = get_sliding_engine()

    feature_service.register_derived(
        "sw_growth_rate_2h",
        lambda hist, ctx: engine._last_compute.growth_rate_2h if engine._last_compute else 0.0
    )
    feature_service.register_derived(
        "sw_acceleration_1h",
        lambda hist, ctx: engine._last_compute.acceleration_1h if engine._last_compute else 0.0
    )
    feature_service.register_derived(
        "sw_trend_surging",
        lambda hist, ctx: 1.0 if (engine._last_compute and engine._last_compute.trend_label == "surging") else 0.0
    )
    feature_service.register_derived(
        "sw_vs_yesterday_ratio",
        lambda hist, ctx: engine._last_compute.vs_yesterday_same_time if engine._last_compute else 0.0
    )
    feature_service.register_derived(
        "sw_visitors_std_2h",
        lambda hist, ctx: engine._last_compute.visitors_std_2h if engine._last_compute else 0.0
    )

    logger.info("滑窗特征已注册到 FeatureService: growth_rate, acceleration, trend, vs_yesterday, std_2h")
