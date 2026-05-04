"""
全局熔断中心 (Circuit Breaker) + 自动降级策略

熔断触发条件 (任一满足):
  1. RiskGuard 连续 3 次触碰硬性价格边界
  2. InputValidator 检测到关键输入特征大面积失真
  3. 业务人员手动按下 Kill-switch
  4. 模型推理连续超时/异常超过阈值

降级策略:
  OPEN(熔断) → 切换到静态时间表定价 (TimeSlotPricer)
  HALF_OPEN → 部分流量走 AI,部分走静态(试探恢复)
  CLOSED → 正常 AI 定价

状态机:
  CLOSED ──(触发条件)──→ OPEN ──(冷却期过后)──→ HALF_OPEN
  HALF_OPEN ──(成功)──→ CLOSED
  HALF_OPEN ──(失败)──→ OPEN (重置冷却期)
"""
from __future__ import annotations

import time
import threading
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from utils.logger import get_logger
from config import settings
from services.message_bus import bus, Channel

logger = get_logger("CircuitBreaker")


# ============================================================
# 数据结构
# ============================================================
class BreakerState(str, Enum):
    CLOSED = "closed"            # 正常: AI定价
    OPEN = "open"                # 熔断: 静态定价
    HALF_OPEN = "half_open"      # 试探: 部分AI+部分静态


@dataclass
class BreakerEvent:
    """熔断事件记录"""
    timestamp: str
    from_state: str
    to_state: str
    trigger_reason: str
    trigger_detail: dict = field(default_factory=dict)
    operator: str = "system"     # "system" | "manual:{name}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass 
class FallbackDecision:
    """降级定价决策"""
    date: str
    base_price: float
    source: str                 # "ai" | "static_timeslot" | "manual"
    breaker_state: str
    reason: str
    time_slots: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 熔断中心
# ============================================================
class CircuitBreaker:
    """
    全局熔断中心

    用法:
      cb = CircuitBreaker()

      # 定价入口
      decision = cb.route(suggested_price, prev_price, day_type, weather)

      # 手动控制
      cb.manual_open("管理员看到异常,紧急熔断")

      # 状态查询
      print(cb.get_status())
    """

    COOLDOWN_SECONDS = 300          # 5分钟冷却期后进入 HALF_OPEN
    HALF_OPEN_TRIAL_COUNT = 5       # HALF_OPEN 期间需要成功 N 次才恢复 CLOSED
    CONSECUTIVE_FAILURE_THRESHOLD = 3  # 连续失败 N 次 → 熔断

    def __init__(self):
        self._state: BreakerState = BreakerState.CLOSED
        self._lock = threading.RLock()
        self._opened_at: float = 0.0
        self._consecutive_failures = 0
        self._half_open_successes = 0
        self._event_log: List[BreakerEvent] = []
        self._last_ai_decision: Optional[dict] = None
        self._last_fallback_decision: Optional[FallbackDecision] = None

        # 外部回调: 熔断/恢复时通知
        self._on_open_callbacks: List[Callable] = []
        self._on_close_callbacks: List[Callable] = []

    # ============================================================
    # 状态查询
    # ============================================================
    @property
    def state(self) -> BreakerState:
        return self._state

    @property
    def is_open(self) -> bool:
        return self._state == BreakerState.OPEN

    @property
    def is_closed(self) -> bool:
        return self._state == BreakerState.CLOSED

    # ============================================================
    # 触发熔断
    # ============================================================
    def trip(self, reason: str, detail: dict = None, operator: str = "system"):
        """触发熔断 → OPEN"""
        if self._state == BreakerState.OPEN:
            return

        with self._lock:
            old = self._state
            self._state = BreakerState.OPEN
            self._opened_at = time.monotonic()
            self._half_open_successes = 0

            event = BreakerEvent(
                timestamp=datetime.now().isoformat(),
                from_state=old.value,
                to_state=BreakerState.OPEN.value,
                trigger_reason=reason,
                trigger_detail=detail or {},
                operator=operator,
            )
            self._event_log.append(event)

            logger.critical(f"🔴 熔断触发: {reason} (operator={operator})")

            # 发布告警
            bus.publish(Channel.ANOMALY, {
                "type": "circuit_breaker_open",
                "level": "critical",
                "reason": reason,
                "operator": operator,
                "message": f"定价系统已熔断 | 原因: {reason}",
                "suggested_action": (
                    "系统已切换到静态时间表定价模式。\n"
                    "请检查: 1)模型健康状态 2)输入数据源 3)API连通性\n"
                    "确认无误后调用 cb.attempt_reset() 恢复"
                ),
            }, source="circuit_breaker")

            # 执行回调
            for cb_fn in self._on_open_callbacks:
                try:
                    cb_fn(event)
                except Exception as e:
                    logger.error(f"熔断回调异常: {e}")

    def manual_open(self, reason: str, operator: str = "unknown"):
        """手动触发熔断 (Kill-switch)"""
        self.trip(reason, {"manual": True}, operator=f"manual:{operator}")

    # ============================================================
    # 恢复
    # ============================================================
    def attempt_reset(self) -> bool:
        """
        尝试重置熔断 → HALF_OPEN (如果冷却期已过)
        """
        with self._lock:
            if self._state != BreakerState.OPEN:
                return False

            elapsed = time.monotonic() - self._opened_at
            if elapsed < self.COOLDOWN_SECONDS:
                remaining = self.COOLDOWN_SECONDS - elapsed
                logger.info(f"冷却期未过,剩余{remaining:.0f}秒")
                return False

            self._state = BreakerState.HALF_OPEN
            self._half_open_successes = 0
            self._consecutive_failures = 0

            event = BreakerEvent(
                timestamp=datetime.now().isoformat(),
                from_state=BreakerState.OPEN.value,
                to_state=BreakerState.HALF_OPEN.value,
                trigger_reason="冷却期结束,进入试探恢复阶段",
            )
            self._event_log.append(event)

            logger.info("🟡 熔断降级为 HALF_OPEN: 开始试探恢复")
            return True

    def report_success(self):
        """报告 AI 定价成功 → 累计 HALF_OPEN 成功次数"""
        with self._lock:
            if self._state == BreakerState.HALF_OPEN:
                self._half_open_successes += 1
                self._consecutive_failures = 0

                if self._half_open_successes >= self.HALF_OPEN_TRIAL_COUNT:
                    self._state = BreakerState.CLOSED
                    self._consecutive_failures = 0
                    event = BreakerEvent(
                        timestamp=datetime.now().isoformat(),
                        from_state=BreakerState.HALF_OPEN.value,
                        to_state=BreakerState.CLOSED.value,
                        trigger_reason=f"HALF_OPEN阶段连续成功{self._half_open_successes}次,恢复正常",
                    )
                    self._event_log.append(event)
                    logger.info("✅ 熔断已恢复: CLOSED")

                    for cb_fn in self._on_close_callbacks:
                        try:
                            cb_fn()
                        except Exception as e:
                            logger.error(f"恢复回调异常: {e}")

    def report_failure(self):
        """报告 AI 定价失败 → HALF_OPEN 下失败立即回到 OPEN"""
        with self._lock:
            self._consecutive_failures += 1

            if self._consecutive_failures >= self.CONSECUTIVE_FAILURE_THRESHOLD:
                if self._state != BreakerState.OPEN:
                    self.trip(
                        f"连续{self._consecutive_failures}次AI定价失败",
                        {"consecutive_failures": self._consecutive_failures},
                    )

    # ============================================================
    # 核心路由: AI or 静态?
    # ============================================================
    def route(
        self,
        suggested_price: float,
        prev_price: float,
        day_type: str = "weekday",
        weather: str = "晴好",
        temperature: float = 24.0,
        rainfall: float = 0.0,
        load_rate: float = 0.5,
    ) -> FallbackDecision:
        """
        定价路由: 根据熔断状态决定使用 AI 还是静态定价

        Returns:
          FallbackDecision with source="ai" or "static_timeslot"
        """
        date_str = datetime.now().strftime("%Y-%m-%d")

        with self._lock:
            # HALF_OPEN: 检查是否可试探
            if self._state == BreakerState.HALF_OPEN:
                # 冷却期检查(从OPEN恢复后)
                pass  # attempt_reset已处理

            # OPEN or HALF_OPEN: 使用静态定价
            if self._state in (BreakerState.OPEN, BreakerState.HALF_OPEN):
                return self._compute_static_fallback(
                    date_str, prev_price, day_type, weather,
                    temperature, rainfall, load_rate,
                )

            # CLOSED: 正常走AI
            decision = FallbackDecision(
                date=date_str,
                base_price=suggested_price,
                source="ai",
                breaker_state=self._state.value,
                reason="正常AI定价模式",
            )
            self._last_ai_decision = decision.to_dict()
            return decision

    def _compute_static_fallback(
        self, date: str, prev_price: float, day_type: str,
        weather: str, temperature: float, rainfall: float, load_rate: float,
    ) -> FallbackDecision:
        """
        静态降级: 使用时间表定价 (TimeSlotPricer)

        这是安全兜底: 价格死板但绝不会出错
        """
        from engine.time_slot_pricing import TimeSlotPricer

        pricer = TimeSlotPricer()
        slots = pricer.compute(prev_price, day_type, weather, temperature)

        # 取正常场价格作为 base_price
        base = prev_price
        for s in slots:
            if s.slot_name == "正常场":
                base = s.price
                break

        decision = FallbackDecision(
            date=date,
            base_price=base,
            source="static_timeslot",
            breaker_state=self._state.value,
            reason=f"熔断降级({self._state.value}): 使用静态时间表定价",
            time_slots=[s.to_dict() for s in slots],
        )
        self._last_fallback_decision = decision
        return decision

    # ============================================================
    # 回调注册
    # ============================================================
    def on_open(self, callback: Callable):
        self._on_open_callbacks.append(callback)

    def on_close(self, callback: Callable):
        self._on_close_callbacks.append(callback)

    # ============================================================
    # 状态
    # ============================================================
    def get_status(self) -> dict:
        cooldown_remaining = 0.0
        if self._state == BreakerState.OPEN:
            elapsed = time.monotonic() - self._opened_at
            cooldown_remaining = max(0, self.COOLDOWN_SECONDS - elapsed)

        return {
            "state": self._state.value,
            "consecutive_failures": self._consecutive_failures,
            "half_open_successes": self._half_open_successes,
            "cooldown_remaining_seconds": round(cooldown_remaining, 1),
            "can_attempt_reset": (
                self._state == BreakerState.OPEN and cooldown_remaining <= 0
            ),
            "recent_events": [e.to_dict() for e in self._event_log[-5:]],
        }


# 全局单例
breaker = CircuitBreaker()
