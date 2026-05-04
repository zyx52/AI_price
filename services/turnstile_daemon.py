"""
闸机/客流数据守护进程 —— 5分钟实时轮询

职责:
  1. 每5分钟对接票务系统API,拉取"已检票入园人数"和"已购票未入园人数"
  2. 发布到 Redis Pub/Sub park:live_data:turnstile
  3. 脏数据过滤(负数、突增突降 > 3σ)
  4. 前向填充 + 近期均值插值
  5. 85%容量预警

启动:
  python services/turnstile_daemon.py --source mock

真实票务系统接入:
  在 TurnstileAPIClient 中实现 _fetch_from_ticketing_system()
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger
from utils.feature_cache import feature_cache
from config import settings
from services.message_bus import bus, Channel

logger = get_logger("TurnstileDaemon")


# ============================================================
# 数据结构
# ============================================================
@dataclass
class TurnstileSnapshot:
    """闸机数据快照"""
    timestamp: str                      # ISO 8601
    date: str                           # YYYY-MM-DD
    # 入园数据
    checked_in_count: int               # 已检票入园
    not_entered_count: int              # 已购票未入园
    total_tickets_sold: int             # 总售票数(截止当前)
    # 实时衍生
    current_load_rate: float            # 当前负载率 (checked_in / capacity)
    effective_capacity: int             # 有效容量 (capacity - checked_in)
    entry_rate_per_min: float           # 近5分钟入园速率(人/分钟)
    # 预测
    estimated_end_of_day: int           # 预估全日入园总量
    load_warning: str                   # "normal" | "warning_85%" | "critical_90%"
    # 数据质量
    is_forward_filled: bool = False
    is_interpolated: bool = False
    consecutive_failures: int = 0
    data_quality_flags: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 闸机API客户端 (抽象基类 + Mock实现)
# ============================================================
class TurnstileAPIClient:
    """闸机API客户端基类 —— 真实上线时继承此类实现"""

    def fetch(self) -> Dict[str, Any]:
        raise NotImplementedError


class MockTurnstileClient(TurnstileAPIClient):
    """Mock闸机客户端 —— 模拟客流入园曲线"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self._base_visitors = 20000
        # 模拟时间相关的客流曲线
        self._hour_profile = {
            8: 0.02, 9: 0.08, 10: 0.15, 11: 0.18,
            12: 0.12, 13: 0.10, 14: 0.08, 15: 0.07,
            16: 0.06, 17: 0.05, 18: 0.04, 19: 0.03, 20: 0.02,
        }

    def fetch(self) -> Dict[str, Any]:
        now = datetime.now()
        hour = now.hour
        # 根据小时确定当日累计入园比例
        cumulative_ratio = sum(
            v for h, v in self._hour_profile.items() if h <= hour
        )
        # 加一些随机波动
        noise = self.rng.normal(0, 0.03)
        cumulative_ratio = max(0.01, min(0.98, cumulative_ratio + noise))

        total = int(self._base_visitors * (1 + self.rng.normal(0, 0.1)))
        checked_in = int(total * cumulative_ratio)
        not_entered = total - checked_in

        # 偶尔模拟数据异常(测试脏数据过滤)
        if self.rng.random() < 0.02:  # 2%概率模拟异常
            if self.rng.random() < 0.5:
                checked_in = -100  # 负数异常
            else:
                checked_in = int(checked_in * 5)  # 突增异常

        return {
            "checked_in_count": checked_in,
            "not_entered_count": not_entered,
            "total_tickets_sold": total,
            "timestamp": now.isoformat(),
        }


# ============================================================
# 脏数据过滤器
# ============================================================
class DataSanitizer:
    """
    脏数据过滤 + 前向填充 + 插值

    检测规则:
      1. 负数 → 标记为脏数据
      2. 突增/突降超过3σ → 标记为可疑
      3. 短时间内重复相同值 → 可能是系统卡死
    """

    def __init__(self, window_size: int = 20, sigma_threshold: float = 3.0):
        self.window_size = window_size
        self.sigma_threshold = sigma_threshold
        self._history: List[TurnstileSnapshot] = []
        self._last_valid: Optional[TurnstileSnapshot] = None

    def sanitize(self, raw: Dict[str, Any]) -> TurnstileSnapshot:
        """清洗一份原始数据,返回干净快照"""
        flags: List[str] = []

        checked_in = raw.get("checked_in_count", 0)
        not_entered = raw.get("not_entered_count", 0)
        total = raw.get("total_tickets_sold", 0)

        # === 规则1: 负数检测 ===
        hard_corrected = False
        if checked_in < 0:
            flags.append("negative_checked_in")
            checked_in = 0
            hard_corrected = True
        if not_entered < 0:
            flags.append("negative_not_entered")
            not_entered = 0
        if total < 0:
            flags.append("negative_total")
            total = 0

        # === 规则2: 突增/突降 > 3σ 检测 ===
        # 跳过已被规则1硬纠正的值(避免0被误判为spike)
        if not hard_corrected and len(self._history) >= 5:
            recent_vals = [s.checked_in_count for s in self._history[-self.window_size:]]
            mean = np.mean(recent_vals)
            std = np.std(recent_vals) if len(recent_vals) > 1 else 1.0

            if std > 0:
                z_score = abs(checked_in - mean) / std
                if z_score > self.sigma_threshold:
                    flags.append(f"spike_detected_z{z_score:.1f}")
                    # 用近期均值替代
                    checked_in = int(mean)
                    not_entered = max(0, total - checked_in)

        # === 规则3: 相同值重复(系统卡死) ===
        if len(self._history) >= 3:
            last_3 = [s.checked_in_count for s in self._history[-3:]]
            if len(set(last_3)) == 1 and last_3[0] == checked_in:
                flags.append("stale_data_possible")

        # === 前向填充: 如果数据全部异常 ===
        is_forward_filled = False
        is_interpolated = len(flags) > 0 and "spike_detected" in str(flags)

        if checked_in == 0 and not_entered == 0 and total == 0:
            if self._last_valid is not None:
                # 前向填充: 使用上次有效值,并随时间衰减
                decay = 0.98
                checked_in = int(self._last_valid.checked_in_count * decay)
                not_entered = int(self._last_valid.not_entered_count * decay)
                total = checked_in + not_entered
                is_forward_filled = True
                flags.append("forward_filled")

        # 构造快照
        capacity = settings.park_capacity
        load_rate = checked_in / capacity if capacity > 0 else 0.0
        entry_rate = 0.0
        if self._history:
            last = self._history[-1]
            time_delta_min = 5.0  # 5分钟轮询间隔
            if time_delta_min > 0:
                entry_rate = max(0, (checked_in - last.checked_in_count) / time_delta_min)

        # 负载预警
        if load_rate >= 0.90:
            load_warning = "critical_90%"
        elif load_rate >= 0.85:
            load_warning = "warning_85%"
        else:
            load_warning = "normal"

        snapshot = TurnstileSnapshot(
            timestamp=raw.get("timestamp", datetime.now().isoformat()),
            date=datetime.now().strftime("%Y-%m-%d"),
            checked_in_count=checked_in,
            not_entered_count=not_entered,
            total_tickets_sold=total,
            current_load_rate=round(load_rate, 4),
            effective_capacity=capacity - checked_in,
            entry_rate_per_min=round(entry_rate, 2),
            estimated_end_of_day=int(total * 1.05),  # 简单预估: 总售票 * 1.05
            load_warning=load_warning,
            is_forward_filled=is_forward_filled,
            is_interpolated=is_interpolated,
            data_quality_flags=flags,
        )

        # 更新历史
        self._history.append(snapshot)
        if len(self._history) > self.window_size * 2:
            self._history = self._history[-self.window_size:]

        if not is_forward_filled:
            self._last_valid = snapshot
            snapshot.consecutive_failures = 0
        else:
            snapshot.consecutive_failures = (
                self._last_valid.consecutive_failures + 1 if self._last_valid else 1
            )

        return snapshot


# ============================================================
# 闸机守护进程核心
# ============================================================
class TurnstileDaemon:
    """闸机轮询守护进程"""

    CACHE_KEY = "turnstile:latest"

    def __init__(self, source: str = "mock"):
        self.source = source
        self._running = False

        # 初始化API客户端
        if source == "mock":
            self._client = MockTurnstileClient()
        else:
            # TODO: 实现 TurnstileAPIClient 的真实子类
            logger.warning(f"未知闸机数据源: {source}, 降级为mock")
            self._client = MockTurnstileClient()

        self._sanitizer = DataSanitizer()

        # 信号处理
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"收到信号 {signum}, 正在停止...")
        self._running = False

    # ============================================================
    # 主循环
    # ============================================================
    def run(self, interval_seconds: int = 300):
        """
        主循环: 每5分钟轮询一次

        interval_seconds: 轮询间隔(默认300=5分钟)
        """
        self._running = True
        logger.info(f"🚪 闸机守护进程启动 | source={self.source} "
                     f"| interval={interval_seconds}s | bus={'healthy' if bus.is_healthy else 'no-op'}")

        while self._running:
            start_time = time.time()

            try:
                raw = self._client.fetch()
                snapshot = self._sanitizer.sanitize(raw)

                # 发布到消息总线
                bus.publish_turnstile(snapshot.to_dict())

                # 缓存最新快照
                feature_cache.set(self.CACHE_KEY, snapshot.to_dict(), ttl=600)

                # 85%容量预警 → 额外发布到异常频道
                if snapshot.load_warning != "normal":
                    bus.publish_anomaly({
                        "type": "capacity_warning",
                        "level": snapshot.load_warning,
                        "load_rate": snapshot.current_load_rate,
                        "checked_in": snapshot.checked_in_count,
                        "message": f"园区负载率达到 {snapshot.current_load_rate:.1%}, 建议上调价格控制客流",
                    })
                    logger.warning(
                        f"⚠️ {snapshot.load_warning}: "
                        f"负载率={snapshot.current_load_rate:.1%} | "
                        f"入园={snapshot.checked_in_count} | "
                        f"速率={snapshot.entry_rate_per_min}人/分钟"
                    )

                flag_str = f" flags={snapshot.data_quality_flags}" if snapshot.data_quality_flags else ""
                logger.info(
                    f"闸机已发布: 入园={snapshot.checked_in_count} | "
                    f"未入园={snapshot.not_entered_count} | "
                    f"负载率={snapshot.current_load_rate:.1%} | "
                    f"速率={snapshot.entry_rate_per_min}人/分钟"
                    f"{flag_str}"
                )

            except Exception as e:
                logger.error(f"闸机守护进程异常: {e}", exc_info=True)

            # 等待到下一次轮询
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed)
            if sleep_time > 0 and self._running:
                for _ in range(int(sleep_time)):
                    if not self._running:
                        break
                    time.sleep(1)

        logger.info("闸机守护进程已停止")


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="闸机/客流数据守护进程")
    parser.add_argument("--source", default="mock", choices=["mock", "api"],
                        help="数据源 (mock=模拟数据, api=真实票务系统)")
    parser.add_argument("--interval", type=int, default=300,
                        help="轮询间隔(秒), 默认300(5分钟)")
    args = parser.parse_args()

    daemon = TurnstileDaemon(source=args.source)
    daemon.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
