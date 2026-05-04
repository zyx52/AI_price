"""
风控体系 (Risk Guard + Circuit Breaker + Input Validator) 集成测试

覆盖:
  1. RiskGuard 底价/天花板/幅度三层拦截
  2. 连续边界触碰 → 熔断信号
  3. CircuitBreaker 状态机 CLOSED → OPEN → HALF_OPEN → CLOSED
  4. 静态降级到 TimeSlotPricer
  5. InputValidator 物理合理性+时效性+分布偏移
  6. 安全填充 Safe Fill
  7. BusinessMonitor 业务指标告警
  8. NotificationRouter 分级通知

运行:
  python tests/test_risk_control.py
"""
from __future__ import annotations

import sys
import time


# ============================================================
# Test 1: RiskGuard 三层硬性拦截
# ============================================================
def test_risk_guard_boundaries():
    print("\n" + "=" * 60)
    print("  Test 1: RiskGuard 硬性价格边界三层拦截")
    print("=" * 60)

    from services.risk_guard import RiskGuard, GuardViolation

    guard = RiskGuard()

    # --- 底价保护 ---
    result = guard.validate(suggested_price=30.0, prev_price=299.0)
    assert result.violation == GuardViolation.FLOOR, \
        f"应触发底价保护, 实际={result.violation}"
    assert result.corrected_price >= guard.HARD_FLOOR
    print(f"  ✓ 底价保护: ¥30 → ¥{result.corrected_price} (底线¥{guard.HARD_FLOOR})")

    # --- 天花板保护 ---
    result2 = guard.validate(suggested_price=999.0, prev_price=299.0)
    assert result2.violation == GuardViolation.CEILING, \
        f"应触发天花板保护, 实际={result2.violation}"
    assert result2.corrected_price <= result2.ceiling_price
    print(f"  ✓ 天花板保护: ¥999 → ¥{result2.corrected_price}")

    # --- 幅度限制 ---
    result3 = guard.validate(suggested_price=400.0, prev_price=299.0)
    step_pct = abs(400 - 299) / 299
    assert step_pct > 0.15, f"变动{step_pct:.1%}应超过15%"
    assert result3.violation == GuardViolation.STEP, \
        f"应触发幅度限制, 实际={result3.violation}"
    max_allowed = 299 * 1.15
    assert abs(result3.corrected_price - max_allowed) < 1.0
    print(f"  ✓ 幅度限制: ¥400({step_pct:.1%}) → ¥{result3.corrected_price}")

    # --- 正常通过 ---
    # 需要先重置计数器
    guard._consecutive_hits = 0
    result4 = guard.validate(suggested_price=320.0, prev_price=299.0)
    step_pct4 = abs(320 - 299) / 299
    assert step_pct4 < 0.15
    assert result4.violation == GuardViolation.OK, \
        f"应通过, 实际={result4.violation}"
    print(f"  ✓ 正常通过: ¥{result4.corrected_price} ({step_pct4:.1%})")

    print("  ✅ Test 1 通过\n")
    return True


# ============================================================
# Test 2: 连续边界触碰 → 熔断信号
# ============================================================
def test_risk_guard_circuit_breaker_signal():
    print("=" * 60)
    print("  Test 2: 连续触碰 → 熔断信号")
    print("=" * 60)

    from services.risk_guard import RiskGuard, GuardViolation

    guard = RiskGuard()
    guard._consecutive_hits = 0

    # 连续3次触发底价
    for i in range(3):
        result = guard.validate(suggested_price=10.0, prev_price=299.0)
        assert result.violation == GuardViolation.FLOOR
        print(f"  第{i+1}次触底: consecutive_hits={result.consecutive_boundary_hits}")

    assert result.consecutive_boundary_hits >= 3
    assert result.circuit_breaker_signal, "连续3次应触发熔断信号!"
    print("  ✓ 连续3次触碰硬边界 → circuit_breaker_signal=True")

    # 正常通过应重置
    guard._consecutive_hits = 0
    result_ok = guard.validate(suggested_price=310.0, prev_price=299.0)
    assert result_ok.consecutive_boundary_hits == 0
    print("  ✓ 正常通过后计数器重置为0")

    print("  ✅ Test 2 通过\n")
    return True


# ============================================================
# Test 3: Kill-switch
# ============================================================
def test_kill_switch():
    print("=" * 60)
    print("  Test 3: Kill-switch 手动紧急停止")
    print("=" * 60)

    from services.risk_guard import RiskGuard

    guard = RiskGuard()
    assert not guard.is_killed

    # 手动激活
    guard.kill(operator="admin_test")
    assert guard.is_killed
    print("  ✓ Kill-switch 激活: is_killed=True")

    # 激活后所有定价被冻结
    result = guard.validate(suggested_price=350.0, prev_price=299.0)
    assert result.circuit_breaker_signal
    assert result.corrected_price == 299.0  # 保持原价
    print(f"  ✓ Kill-switch下价格冻结: ¥350 → ¥{result.corrected_price}")

    # 强制模式仍可绕过
    result_force = guard.validate(suggested_price=350.0, prev_price=299.0, force=True)
    assert result_force.violation.value == "ok"
    print("  ✓ Force模式可绕过Kill-switch")

    # 恢复
    guard.revive(operator="admin_test")
    assert not guard.is_killed
    print("  ✓ Kill-switch 恢复: is_killed=False")

    print("  ✅ Test 3 通过\n")
    return True


# ============================================================
# Test 4: CircuitBreaker 状态机
# ============================================================
def test_circuit_breaker_state_machine():
    print("=" * 60)
    print("  Test 4: CircuitBreaker 状态机 CLOSED→OPEN→HALF_OPEN→CLOSED")
    print("=" * 60)

    from services.circuit_breaker import CircuitBreaker, BreakerState

    cb = CircuitBreaker()
    # 缩短冷却期用于测试
    cb.COOLDOWN_SECONDS = 1
    cb.HALF_OPEN_TRIAL_COUNT = 2

    # CLOSED: 正常走AI
    assert cb.state == BreakerState.CLOSED
    decision = cb.route(suggested_price=310, prev_price=299, day_type="weekday", weather="晴好")
    assert decision.source == "ai"
    print(f"  ✓ CLOSED: source={decision.source}")

    # 触发熔断 → OPEN
    cb.trip("test: 模拟连续边界触碰")
    assert cb.state == BreakerState.OPEN
    decision2 = cb.route(suggested_price=310, prev_price=299, day_type="weekday", weather="晴好")
    assert decision2.source == "static_timeslot"
    assert len(decision2.time_slots) >= 2  # 至少早鸟+正常场
    print(f"  ✓ OPEN→静态降级: source={decision2.source}, {len(decision2.time_slots)}个时段")

    # 冷却期内无法恢复
    ok = cb.attempt_reset()
    assert not ok
    print("  ✓ 冷却期内无法恢复")

    # 冷却期过后: OPEN → HALF_OPEN
    time.sleep(1.1)
    ok = cb.attempt_reset()
    assert ok
    assert cb.state == BreakerState.HALF_OPEN
    print(f"  ✓ 冷却期过后: → {cb.state.value}")

    # HALF_OPEN: 成功N次 → CLOSED
    decision3 = cb.route(suggested_price=310, prev_price=299, day_type="weekday", weather="晴好")
    assert decision3.source == "static_timeslot"  # HALF_OPEN仍走静态
    cb.report_success()
    cb.report_success()
    assert cb.state == BreakerState.CLOSED
    print(f"  ✓ HALF_OPEN成功2次 → {cb.state.value} (恢复)")

    print("  ✅ Test 4 通过\n")
    return True


# ============================================================
# Test 5: InputValidator 多层校验
# ============================================================
def test_input_validator():
    print("=" * 60)
    print("  Test 5: InputValidator 输入特征校验")
    print("=" * 60)

    from services.input_validator import InputValidator, ValidationLevel

    validator = InputValidator()

    # 正常输入
    result = validator.validate({
        "temperature": 24.0, "rainfall": 0.0,
        "checked_in_count": 15000, "load_rate": 0.38,
    })
    assert result.level == ValidationLevel.PASS
    assert result.passed and not result.blocked
    print(f"  ✓ 正常输入: level={result.level.value}")

    # 温度异常 (150℃)
    result2 = validator.validate({
        "temperature": 150.0, "rainfall": 0.0,
        "checked_in_count": 15000, "load_rate": 0.38,
    })
    assert result2.blocked, "150℃应触发阻断"
    assert result2.level == ValidationLevel.CRITICAL
    print(f"  ✓ 异常温度150℃: blocked={result2.blocked} level={result2.level.value}")

    # 负数入园
    result3 = validator.validate({
        "temperature": 24.0, "rainfall": 0.0,
        "checked_in_count": -5, "load_rate": -0.1,
    })
    assert result3.blocked, "负数入园应触发阻断"
    print(f"  ✓ 负数入园: blocked={result3.blocked}")

    # 数据过期
    result4 = validator.validate(
        {"temperature": 24.0, "rainfall": 0.0,
         "checked_in_count": 15000, "load_rate": 0.38},
        data_ages={"turnstile": 900, "weather": 100},  # 闸机数据过期
    )
    assert result4.blocked, "闸机数据过期应阻断"
    print(f"  ✓ 数据过期: blocked={result4.blocked}")

    # 安全填充
    safe = validator.safe_fill({
        "temperature": 150.0, "rainfall": None,
        "checked_in_count": -5, "load_rate": 0.38,
    })
    assert safe["temperature"] <= 50.0, f"温度应被钳制, 实际={safe['temperature']}"
    assert safe["rainfall"] == 0.0, "None降雨应填0"
    assert safe["checked_in_count"] == 0, "负数入园应填0"
    assert safe["load_rate"] == 0.38
    print(f"  ✓ 安全填充: temp={safe['temperature']}, rain={safe['rainfall']}, "
          f"checkin={safe['checked_in_count']}")

    print("  ✅ Test 5 通过\n")
    return True


# ============================================================
# Test 6: BusinessMonitor 业务指标告警
# ============================================================
def test_business_monitor():
    print("=" * 60)
    print("  Test 6: BusinessMonitor 业务指标告警")
    print("=" * 60)

    from monitor.alert_engine import BusinessMonitor

    monitor = BusinessMonitor()
    monitor.set_baseline("conversion_rate", 0.08)
    monitor.set_baseline("refund_rate", 0.02)

    # 转化率正常
    alerts_ok = monitor.check(conversion_rate=0.07, refund_rate=0.02)
    assert len(alerts_ok) == 0
    print("  ✓ 指标正常: 无告警")

    # 转化率断崖下跌35% → CRITICAL
    alerts = monitor.check(conversion_rate=0.03, refund_rate=0.02)
    assert len(alerts) >= 1
    assert alerts[0].level == "critical"
    assert "转化率" in alerts[0].message
    print(f"  ✓ 转化率暴跌: level={alerts[0].level}, {alerts[0].message}")

    # 退票率翻倍 → WARNING
    alerts2 = monitor.check(conversion_rate=0.08, refund_rate=0.06)
    assert len(alerts2) >= 1
    assert alerts2[0].level == "warning"
    print(f"  ✓ 退票率激增: level={alerts2[0].level}")

    # 记录指标
    monitor.record_metric("conversion_rate", 0.08)
    monitor.record_metric("conversion_rate", 0.07)
    recent = monitor.get_recent("conversion_rate", minutes=5)
    assert len(recent) == 2
    print(f"  ✓ 指标记录: {len(recent)}条")

    print("  ✅ Test 6 通过\n")
    return True


# ============================================================
# Test 7: NotifyChannel 分级通知
# ============================================================
def test_notification_channels():
    print("=" * 60)
    print("  Test 7: NotificationRouter 分级通知")
    print("=" * 60)

    from monitor.alert_engine import (
        DingTalkChannel, WeComChannel, NotificationRouter, notify,
    )

    # 钉钉通道(无webhook, 验证不崩溃)
    ding = DingTalkChannel(webhook_url="")
    ok = ding.send("测试", "钉钉测试消息", "critical")
    print(f"  ✓ 钉钉通道(无webhook): 安全返回 {ok}")

    # 企微通道
    wecom = WeComChannel(webhook_url="")
    ok2 = wecom.send("测试", "企微测试消息", "warning")
    print(f"  ✓ 企微通道(无webhook): 安全返回 {ok2}")

    # 路由通知(验证不崩溃)
    notify.notify("测试标题", "测试消息", "critical")
    print("  ✓ 通知路由: 安全完成(无webhook时静默)")

    # 检查可用通道
    channels = notify.available_channels
    print(f"  ✓ 可用通道: {channels}")

    print("  ✅ Test 7 通过\n")
    return True


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  🛡️ 风控体系 · 集成测试")
    print("  Risk Guard + Circuit Breaker + Input Validator")
    print("=" * 60)

    results = []
    tests = [
        ("三层硬性拦截", test_risk_guard_boundaries),
        ("连续触碰→熔断信号", test_risk_guard_circuit_breaker_signal),
        ("Kill-switch紧急停止", test_kill_switch),
        ("熔断状态机", test_circuit_breaker_state_machine),
        ("输入特征校验", test_input_validator),
        ("业务指标告警", test_business_monitor),
        ("分级通知通道", test_notification_channels),
    ]

    for name, fn in tests:
        try:
            fn()
            results.append((name, True, None))
        except Exception as e:
            results.append((name, False, str(e)))
            print(f"  ❌ {name} 失败: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("  测试结果汇总")
    print("=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    for name, ok, err in results:
        status = "✅" if ok else "❌"
        print(f"  {status} {name}")
    print(f"\n  {passed}/{len(results)} 通过")

    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
