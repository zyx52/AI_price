"""
硬性价格边界拦截器 (Risk Guard / Kill-switch 第一道防线)

策略:
  在任何价格下发到渠道之前,强制经过此拦截器校验:
  
  1. 底价保护 (Price Floor)
     价格不能低于: 边际运营成本 + 最低利润率
     触发 → 强制修正为底线价 + 记录触底告警

  2. 天花板保护 (Price Ceiling)  
     价格不能超过: 基础价 × 最大上浮系数 (物价局备案约束)
     触发 → 强制修正为天花板价 + 记录触顶告警

  3. 单次幅度限制 (Step Limit)
     相邻两次调价涨跌幅 ≤ 15%
     触发 → 强制修正为最大允许幅度 + 记录超幅告警

  4. 熔断信号检测
     连续 3 次触碰硬边界 → 触发熔断信号

集成位置:
  PriceSyncService.decide() → RiskGuard.validate() → Adapter.push_price()
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

from utils.logger import get_logger
from config import settings
from services.message_bus import bus, Channel

logger = get_logger("RiskGuard")


# ============================================================
# 数据结构
# ============================================================
class GuardViolation(str, Enum):
    FLOOR = "floor_breach"           # 低于底价
    CEILING = "ceiling_breach"       # 超过天花板
    STEP = "step_breach"             # 单次变动超幅
    OK = "ok"


@dataclass
class GuardResult:
    """拦截器校验结果"""
    original_price: float            # 模型原始建议价
    corrected_price: float           # 经拦截器修正后的价格
    violation: GuardViolation        # 违规类型
    correction_detail: str           # 修正说明
    consecutive_boundary_hits: int   # 连续触碰硬边界次数
    circuit_breaker_signal: bool     # 是否触发熔断信号
    floor_price: float               # 当前底价
    ceiling_price: float             # 当前天花板
    prev_price: float                # 上次价格
    max_step_pct: float              # 最大幅度限制
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["violation"] = self.violation.value
        return d


# ============================================================
# 价格边界拦截器
# ============================================================
class RiskGuard:
    """
    硬性价格边界拦截器

    配置来源:
      - config/settings.py 中的 pricing 段
      - 可运行时动态更新

    用法:
      guard = RiskGuard()
      result = guard.validate(
          suggested_price=350.0,
          prev_price=299.0,
          force=False,
      )
      if result.violation != GuardViolation.OK:
          logger.warning(f"价格被拦截: {result.correction_detail}")
    """

    # 硬编码安全默认值(当config不可用时兜底)
    HARD_FLOOR = 50.0            # 绝对底价: 低于此说明系统bug
    HARD_CEILING_RATIO = 1.50    # 绝对天花板: 基础价的1.5倍
    MAX_STEP_CHANGE = 0.15       # 最大单次变动15%
    CIRCUIT_BREAKER_THRESHOLD = 3  # 连续3次触边 → 熔断

    def __init__(self):
        self._consecutive_hits = 0
        self._last_violation_type: Optional[GuardViolation] = None
        self._violation_history: List[GuardResult] = []
        self._killed = False  # 手动 Kill-switch

    # ============================================================
    # 主校验入口
    # ============================================================
    def validate(
        self,
        suggested_price: float,
        prev_price: float,
        force: bool = False,
    ) -> GuardResult:
        """
        对模型建议价进行多道拦截

        Args:
          suggested_price: 模型建议价
          prev_price: 上次实际执行价格
          force: 是否绕过所有检查(运维强制)

        Returns:
          GuardResult with corrected_price
        """
        # 手动 Kill-switch 生效
        if self._killed and not force:
            return GuardResult(
                original_price=suggested_price,
                corrected_price=prev_price,  # 保持原价不变
                violation=GuardViolation.STEP,
                correction_detail="Kill-switch已激活,拒绝所有价格变更",
                consecutive_boundary_hits=self._consecutive_hits,
                circuit_breaker_signal=True,
                floor_price=self._get_floor(),
                ceiling_price=self._get_ceiling(),
                prev_price=prev_price,
                max_step_pct=self.MAX_STEP_CHANGE,
            )

        if force:
            # 强制模式: 绕过检查,重置计数器
            self._consecutive_hits = 0
            return GuardResult(
                original_price=suggested_price,
                corrected_price=suggested_price,
                violation=GuardViolation.OK,
                correction_detail="Force模式,跳过所有拦截",
                consecutive_boundary_hits=0,
                circuit_breaker_signal=False,
                floor_price=self._get_floor(),
                ceiling_price=self._get_ceiling(),
                prev_price=prev_price,
                max_step_pct=self.MAX_STEP_CHANGE,
            )

        price = suggested_price
        violation = GuardViolation.OK
        details: List[str] = []

        # === 检查1: 底价保护 ===
        floor = self._get_floor()
        if price < floor:
            details.append(f"底价保护: ¥{price} < ¥{floor}(底线)")
            price = floor
            violation = GuardViolation.FLOOR
            self._consecutive_hits += 1
            self._emit_alert("FLOOR", suggested_price, price, floor)

        # === 检查2: 天花板保护 ===
        ceiling = self._get_ceiling()
        if price > ceiling:
            details.append(f"天花板保护: ¥{price} > ¥{ceiling}(上限)")
            price = ceiling
            if violation == GuardViolation.OK:
                violation = GuardViolation.CEILING
            self._consecutive_hits += 1
            self._emit_alert("CEILING", suggested_price, price, ceiling)

        # === 检查3: 单次幅度限制 ===
        if prev_price > 0:
            step_pct = abs(price - prev_price) / prev_price
            if step_pct > self.MAX_STEP_CHANGE:
                # 修正到最大允许幅度
                direction = 1 if price > prev_price else -1
                max_allowed = prev_price * (1 + direction * self.MAX_STEP_CHANGE)
                details.append(
                    f"幅度限制: {step_pct:.1%} > {self.MAX_STEP_CHANGE:.0%}, "
                    f"修正为 ¥{max_allowed:.0f}"
                )
                price = round(max_allowed, 2)
                if violation == GuardViolation.OK:
                    violation = GuardViolation.STEP
                self._consecutive_hits += 1
                self._emit_alert("STEP", suggested_price, price,
                                 prev_price * (1 + self.MAX_STEP_CHANGE))

        # 没有违规: 重置计数器
        if violation == GuardViolation.OK:
            self._consecutive_hits = 0

        # === 熔断信号 ===
        circuit_breaker = self._consecutive_hits >= self.CIRCUIT_BREAKER_THRESHOLD
        if circuit_breaker:
            logger.critical(
                f"🔴 熔断信号触发! 连续{self._consecutive_hits}次触碰硬边界, "
                f"建议切换到静态定价模式"
            )
            bus.publish(Channel.ANOMALY, {
                "type": "circuit_breaker_signal",
                "level": "critical",
                "consecutive_hits": self._consecutive_hits,
                "last_violation": violation.value,
                "message": f"连续{self._consecutive_hits}次触碰硬性价格边界,建议立即熔断",
                "suggested_action": "执行熔断降级 → 切换为静态时间表定价",
            }, source="risk_guard")

        result = GuardResult(
            original_price=suggested_price,
            corrected_price=round(price, 2),
            violation=violation,
            correction_detail="; ".join(details) if details else "通过所有拦截检查",
            consecutive_boundary_hits=self._consecutive_hits,
            circuit_breaker_signal=circuit_breaker,
            floor_price=floor,
            ceiling_price=ceiling,
            prev_price=prev_price,
            max_step_pct=self.MAX_STEP_CHANGE,
        )

        self._violation_history.append(result)
        if len(self._violation_history) > 100:
            self._violation_history = self._violation_history[-50:]

        return result

    # ============================================================
    # Kill-switch
    # ============================================================
    def kill(self, operator: str = "unknown"):
        """手动激活 Kill-switch"""
        self._killed = True
        logger.critical(f"🛑 Kill-switch 手动激活! operator={operator}")
        bus.publish(Channel.ANOMALY, {
            "type": "kill_switch_activated",
            "level": "critical",
            "operator": operator,
            "message": f"人工紧急停止: {operator} 手动激活了 Kill-switch",
            "suggested_action": "系统已冻结所有价格变更,请联系管理员恢复",
        }, source="risk_guard")

    def revive(self, operator: str = "unknown"):
        """手动恢复 Kill-switch"""
        self._killed = False
        self._consecutive_hits = 0
        logger.info(f"✅ Kill-switch 已解除 operator={operator}")

    @property
    def is_killed(self) -> bool:
        return self._killed

    # ============================================================
    # 边界计算
    # ============================================================
    def _get_floor(self) -> float:
        """计算底价: max(配置底价, 硬编码兜底)"""
        try:
            config_floor = settings.pricing.min_price
        except Exception:
            config_floor = 80.0
        return max(config_floor, self.HARD_FLOOR)

    def _get_ceiling(self) -> float:
        """计算天花板: min(配置最高价, 基础价×硬编码倍数)"""
        try:
            config_ceiling = settings.pricing.max_price
            base = settings.pricing.base_price
        except Exception:
            config_ceiling = 599.0
            base = 299.0
        hard_ceiling = base * self.HARD_CEILING_RATIO
        return min(config_ceiling, hard_ceiling)

    # ============================================================
    # 告警
    # ============================================================
    def _emit_alert(self, boundary_type: str, original: float,
                    corrected: float, boundary: float):
        logger.warning(
            f"⚠️ [{boundary_type}] 价格边界触发: "
            f"建议价¥{original} → 修正为¥{corrected} (边界¥{boundary})"
        )

    def get_status(self) -> dict:
        return {
            "killed": self._killed,
            "consecutive_hits": self._consecutive_hits,
            "threshold": self.CIRCUIT_BREAKER_THRESHOLD,
            "floor": self._get_floor(),
            "ceiling": self._get_ceiling(),
            "max_step_pct": self.MAX_STEP_CHANGE,
            "recent_violations": [
                r.to_dict() for r in self._violation_history[-5:]
            ],
        }
