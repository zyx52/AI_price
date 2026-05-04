"""
输入特征校验器 —— 防御性输入与脏数据阻断

在数据喂给模型前进行多层异常检测:
  1. 物理合理性校验: 温度/降水/人数必须在物理可行范围内
  2. 分布偏移检测: 利用 shift_detector 判别OOD
  3. 数据完整性校验: 关键字段不缺失
  4. 数据时效性校验: 实时数据不能过期太久

当检测通过 → 返回 validated features
当检测失败 → 根据严重级别:
  - WARNING: 记录日志,继续推理(降级权重)
  - CRITICAL: 拒绝推理,触发熔断信号
"""
from __future__ import annotations

import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from config import settings
from services.message_bus import bus, Channel

logger = get_logger("InputValidator")


# ============================================================
# 数据结构
# ============================================================
class ValidationLevel(str, Enum):
    PASS = "pass"            # 全部通过
    WARNING = "warning"      # 部分可疑,降级权重,继续推理
    CRITICAL = "critical"    # 关键字段异常,拒绝推理,触发熔断


@dataclass
class ValidationResult:
    """输入校验结果"""
    level: ValidationLevel
    passed: bool              # True=可以继续推理
    blocked: bool             # True=必须阻断推理
    issues: List[dict] = field(default_factory=list)
    # 降级建议
    suggested_fallback: bool = False
    fallback_reason: str = ""
    # 详情
    field_checks: Dict[str, str] = field(default_factory=dict)
    shift_detection: Optional[dict] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["level"] = self.level.value
        return d


# ============================================================
# 物理边界定义
# ============================================================
PHYSICAL_BOUNDS = {
    "temperature":     (-30.0, 50.0),       # ℃
    "rainfall":        (0.0, 500.0),        # mm
    "checked_in_count": (0, 200000),        # 不可能超过20万人
    "load_rate":       (0.0, 1.5),          # 负载率(允许微超)
    "entry_rate":      (0.0, 5000.0),       # 人/分钟
    "price":           (10.0, 10000.0),     # 元
}

# 关键字段: 任何一个异常就触发 CRITICAL
CRITICAL_FIELDS = {"temperature", "rainfall", "checked_in_count", "load_rate"}

# 数据时效性: 超过此秒数视为过期
MAX_DATA_AGE_SECONDS = {
    "turnstile": 600,    # 闸机数据超过10分钟视为过期
    "weather": 3600,     # 天气数据超过1小时视为过期
}


class InputValidator:
    """
    输入特征多层校验器

    用法:
      validator = InputValidator(shift_detector)

      result = validator.validate({
          "temperature": 24.0,
          "rainfall": 0.0,
          "checked_in_count": 15000,
          "load_rate": 0.38,
      })

      if result.blocked:
          raise CircuitBreakerTrip("输入特征异常")
      if result.level == ValidationLevel.WARNING:
          use_degraded_weights = True
    """

    def __init__(self, shift_detector=None):
        """
        shift_detector: DistributionShiftDetector 或 EnhancedShiftDetector 实例
        """
        self._shift_detector = shift_detector
        self._last_valid_input: Optional[Dict[str, float]] = None
        self._validation_history: List[ValidationResult] = []

    def validate(
        self,
        inputs: Dict[str, Any],
        data_ages: Optional[Dict[str, float]] = None,
    ) -> ValidationResult:
        """
        执行多层输入校验

        Args:
          inputs: 即将喂给模型的输入特征 dict
          data_ages: {"turnstile": age_seconds, "weather": age_seconds}
        """
        issues: List[dict] = []
        field_checks: Dict[str, str] = {}
        critical_count = 0

        # === 第1层: 物理合理性 ===
        for field, (low, high) in PHYSICAL_BOUNDS.items():
            if field not in inputs:
                continue

            value = inputs[field]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                field_checks[field] = "MISSING_OR_NAN"
                if field in CRITICAL_FIELDS:
                    critical_count += 1
                    issues.append({
                        "field": field, "value": value, "layer": "physical",
                        "reason": f"关键字段缺失或NaN",
                        "severity": "critical",
                    })
                continue

            value = float(value)
            if value < low or value > high:
                field_checks[field] = f"OUT_OF_BOUNDS({value})"
                severity = "critical" if field in CRITICAL_FIELDS else "warning"
                if field in CRITICAL_FIELDS:
                    critical_count += 1
                issues.append({
                    "field": field, "value": value, "layer": "physical",
                    "reason": f"值 {value} 超出物理范围 [{low}, {high}]",
                    "severity": severity,
                })
            else:
                field_checks[field] = "OK"

        # === 第2层: 数据时效性 ===
        if data_ages:
            for source, max_age in MAX_DATA_AGE_SECONDS.items():
                age = data_ages.get(source, 0)
                if age > max_age:
                    severity = "critical" if source == "turnstile" else "warning"
                    if source == "turnstile":
                        critical_count += 1
                    issues.append({
                        "field": source, "value": f"{age:.0f}s old",
                        "layer": "freshness",
                        "reason": f"{source}数据过期({age:.0f}s > {max_age}s)",
                        "severity": severity,
                    })

        # === 第3层: 分布偏移检测 ===
        shift_result = None
        if self._shift_detector is not None:
            try:
                # 尝试用 shift_detector 检测
                if hasattr(self._shift_detector, 'detect'):
                    shift_result = self._shift_detector.detect(inputs)
                elif hasattr(self._shift_detector, 'predict'):
                    shift_result = self._shift_detector.predict(inputs)

                if shift_result:
                    shift_level = getattr(shift_result, 'level', None)
                    if shift_level:
                        shift_level_str = str(shift_level)
                        if "critical" in shift_level_str.lower():
                            critical_count += 1
                            issues.append({
                                "field": "distribution", "layer": "shift",
                                "reason": f"分布偏移: {shift_level_str}",
                                "severity": "critical",
                            })
                        elif "severe" in shift_level_str.lower():
                            issues.append({
                                "field": "distribution", "layer": "shift",
                                "reason": f"分布偏移: {shift_level_str}",
                                "severity": "warning",
                            })
            except Exception as e:
                logger.warning(f"分布偏移检测异常(非致命): {e}")

        # === 判定 ===
        if critical_count > 0:
            level = ValidationLevel.CRITICAL
            passed = False
            blocked = True
            suggested_fallback = True
            fallback_reason = f"CRITICAL: {critical_count}个关键字段异常"

            # 发布阻断告警
            bus.publish(Channel.ANOMALY, {
                "type": "input_validation_blocked",
                "level": "critical",
                "critical_count": critical_count,
                "issues": issues,
                "message": f"输入特征校验失败({critical_count}个关键异常),已阻断模型推理",
                "suggested_action": "检查数据源: 闸机API/天气API是否正常",
            }, source="input_validator")

        elif len(issues) > 0:
            level = ValidationLevel.WARNING
            passed = True
            blocked = False
            suggested_fallback = False
            fallback_reason = "WARNING: 部分特征可疑,建议降级权重"

        else:
            level = ValidationLevel.PASS
            passed = True
            blocked = False
            suggested_fallback = False
            fallback_reason = ""
            self._last_valid_input = {k: float(v) for k, v in inputs.items()
                                       if isinstance(v, (int, float))}

        result = ValidationResult(
            level=level,
            passed=passed,
            blocked=blocked,
            issues=issues,
            suggested_fallback=suggested_fallback,
            fallback_reason=fallback_reason,
            field_checks=field_checks,
            shift_detection=shift_result,
        )

        self._validation_history.append(result)
        if len(self._validation_history) > 50:
            self._validation_history = self._validation_history[-30:]

        return result

    # ============================================================
    # 前向填充: 用上次有效值替代异常值
    # ============================================================
    def safe_fill(self, inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        安全填充: 将异常值替换为上次有效值或默认值

        用于 WARNING 级别时,不影响推理但用安全值替代可疑字段
        """
        safe = {}

        for field in CRITICAL_FIELDS:
            if field in inputs:
                val = inputs[field]
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    # 用上次有效值或默认值
                    if self._last_valid_input and field in self._last_valid_input:
                        safe[field] = self._last_valid_input[field]
                    else:
                        # 气候学/运营默认值
                        defaults = {
                            "temperature": 22.0, "rainfall": 0.0,
                            "checked_in_count": 15000, "load_rate": 0.38,
                        }
                        safe[field] = defaults.get(field, 0.0)
                else:
                    # 在物理边界内
                    low, high = PHYSICAL_BOUNDS.get(field, (-1e9, 1e9))
                    safe[field] = max(low, min(high, float(val)))

        # 非关键字段直传
        for k, v in inputs.items():
            if k not in safe and v is not None:
                try:
                    safe[k] = float(v)
                except (ValueError, TypeError):
                    pass

        return safe

    def get_status(self) -> dict:
        return {
            "recent_validations": [
                r.to_dict() for r in self._validation_history[-5:]
            ],
            "has_last_valid": self._last_valid_input is not None,
            "blocked_count": sum(
                1 for r in self._validation_history[-20:] if r.blocked
            ),
        }
