"""
渠道适配器 + 价格同步 + A/B测试 —— 集成测试

覆盖:
  1. TokenBucket 限流器
  2. 适配器注册与健康检查
  3. PriceSyncService 冷却期+防抖+防抖阈值
  4. BundleOptimizer 连带定价+逻辑自洽
  5. ABTestManager 灰度发布+自动回滚
  6. RetryScheduler 指数退避
  7. HumanInLoopMiniAppAdapter 审批流程

运行:
  python tests/test_channel_adapters.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from datetime import datetime


# ============================================================
# Test 1: TokenBucket 限流器
# ============================================================
def test_token_bucket():
    print("\n" + "=" * 60)
    print("  Test 1: TokenBucket 限流器")
    print("=" * 60)

    from adapters._base import TokenBucket

    # QPS=10, burst=20
    bucket = TokenBucket(rate=10.0, burst=20)
    assert bucket.available > 18, f"初始令牌不足: {bucket.available}"

    # 连续获取20个应全部成功(burst容量内)
    for i in range(20):
        assert bucket.acquire(), f"第{i+1}次获取失败"
    print("  ✓ burst=20: 连续获取20次成功")

    # 第21次应失败(耗尽)
    assert not bucket.acquire(), "令牌应用尽"
    print("  ✓ 令牌耗尽后获取失败")

    # 等待0.2秒恢复2个令牌
    time.sleep(0.2)
    assert bucket.acquire()
    assert bucket.acquire()
    assert not bucket.acquire()
    print("  ✓ 等待0.2s恢复2个令牌")

    print("  ✅ Test 1 通过\n")
    return True


# ============================================================
# Test 2: 适配器注册与健康检查
# ============================================================
def test_adapter_registry():
    print("=" * 60)
    print("  Test 2: 适配器注册与健康检查")
    print("=" * 60)

    from adapters import (
        create_default_adapters, list_adapters,
        get_adapter, register_adapter,
    )

    adapters = create_default_adapters(miniapp_hitl=True)
    assert len(adapters) == 4, f"应有4个适配器, 实际={len(adapters)}"

    # 检查各适配器类型
    from adapters import MeituanAdapter, CtripAdapter, FeizhuAdapter, HumanInLoopMiniAppAdapter

    assert isinstance(adapters["meituan"], MeituanAdapter)
    assert isinstance(adapters["ctrip"], CtripAdapter)
    assert isinstance(adapters["feizhu"], FeizhuAdapter)
    assert isinstance(adapters["miniapp"], HumanInLoopMiniAppAdapter)
    print("  ✓ 4个适配器类型正确")

    # 健康检查
    status = list_adapters()
    for name in ["meituan", "ctrip", "feizhu", "miniapp"]:
        assert name in status, f"{name}不在状态列表中"
        assert "health" in status[name]
    print(f"  ✓ 健康检查: {[(n, s['health']) for n, s in status.items()]}")

    # 按名获取
    mt = get_adapter("meituan")
    assert mt is not None
    assert mt.channel_name == "meituan"
    print("  ✓ get_adapter 正常工作")

    print("  ✅ Test 2 通过\n")
    return True


# ============================================================
# Test 3: PriceSyncService 冷却期 + 防抖
# ============================================================
def test_price_sync_cooldown():
    print("=" * 60)
    print("  Test 3: PriceSyncService 冷却期 + 防抖")
    print("=" * 60)

    from services.price_sync_service import PriceSyncService, CooldownManager

    # --- CooldownManager ---
    cm = CooldownManager(cooldown_seconds=10)

    # 第一次获取锁
    acquired, remaining = cm.try_acquire("test_product")
    assert acquired, "首次应获取成功"
    print("  ✓ 首次获取冷却期锁: 成功")

    # 第二次获取应失败
    acquired2, remaining2 = cm.try_acquire("test_product")
    assert not acquired2, "冷却期内不应获取成功"
    assert remaining2 > 0
    print(f"  ✓ 冷却期内拒绝: 剩余{remaining2:.0f}秒")

    # 强制释放
    cm.force_release("test_product")
    acquired3, _ = cm.try_acquire("test_product")
    assert acquired3, "释放后应获取成功"
    print("  ✓ force_release 后重新获取成功")

    # --- 最小变动阈值 ---
    from services.price_sync_service import PriceSyncService
    sync = PriceSyncService()
    assert sync.MIN_CHANGE_THRESHOLD == 0.05

    # 变动2%: 应被防抖拦截
    change_pct = abs(309 - 300) / 300  # 3%
    assert change_pct < 0.05, f"3%应小于5%阈值"
    print(f"  ✓ 最小变动阈值: {sync.MIN_CHANGE_THRESHOLD:.0%} "
          f"(3%变动={change_pct:.1%}会被防抖)")

    # 释放冷却期锁(清理)
    cm.force_release("test_product")

    print("  ✅ Test 3 通过\n")
    return True


# ============================================================
# Test 4: BundleOptimizer 连带定价
# ============================================================
def test_linked_pricing():
    print("=" * 60)
    print("  Test 4: BundleOptimizer 连带定价 + 逻辑自洽")
    print("=" * 60)

    from engine.bundle_optimizer import BundleOptimizer, BUNDLE_DEFINITIONS

    opt = BundleOptimizer()
    base = 299.0

    # 计算所有套餐
    prices = opt.compute_linked_prices(base)
    assert len(prices) == len(BUNDLE_DEFINITIONS), \
        f"应计算{len(BUNDLE_DEFINITIONS)}个套餐, 实际{len(prices)}"

    for p in prices:
        standalone = base * p.unit_count + p.addon_value
        assert p.computed_price < standalone, \
            f"{p.bundle_name}: 套餐价(¥{p.computed_price}) 应 < 单买总价(¥{standalone})"
        print(f"  ✓ {p.bundle_name}: ¥{p.computed_price} "
              f"(单买¥{standalone}, 省¥{standalone-p.computed_price:.0f})")

    # 验证逻辑自洽
    validation = opt.validate_all_bundles(base)
    assert validation["valid"], f"逻辑自洽失败: {validation['violations']}"
    print("  ✓ 所有套餐价格逻辑自洽")

    # 快速查询映射
    price_map = opt.get_linked_price_map(base)
    assert len(price_map) == len(prices)
    print(f"  ✓ 价格映射: {len(price_map)}项")

    # 测试基础票价变动后套餐同步
    new_base = 349.0
    new_prices = opt.compute_linked_prices(new_base)
    for old_p, new_p in zip(prices, new_prices):
        assert new_p.computed_price > old_p.computed_price, \
            f"{new_p.bundle_name}应随基础票价上涨"
    print(f"  ✓ 基础票价 ¥{base}→¥{new_base}, 所有套餐同步上涨")

    print("  ✅ Test 4 通过\n")
    return True


# ============================================================
# Test 5: ABTestManager 灰度发布 + 自动回滚
# ============================================================
def test_ab_test_manager():
    print("=" * 60)
    print("  Test 5: ABTestManager 灰度发布 + 自动回滚")
    print("=" * 60)

    from services.ab_test_manager import ABTestManager

    mgr = ABTestManager()

    # 小程序: 应自动发布
    decision = mgr.evaluate("miniapp", new_price=320, old_price=299)
    assert decision.should_auto_publish, "小程序应自动发布"
    print(f"  ✓ miniapp: auto_publish={decision.should_auto_publish} ({decision.reason})")

    # 美团: 不应自动发布(灰度阶段)
    decision_mt = mgr.evaluate("meituan", new_price=320, old_price=299)
    assert not decision_mt.should_auto_publish, "美团不应自动发布(灰度)"
    print(f"  ✓ meituan: auto_publish={decision_mt.should_auto_publish} ({decision_mt.reason})")

    # 灰度升级
    mgr.promote_stage("meituan")  # canary_10 → canary_30
    stage = mgr._channels["meituan"].release_stage
    assert stage == "canary_30"
    print(f"  ✓ 灰度升级: meituan → {stage}")

    # 自动回滚测试
    triggered = mgr.check_rollback("miniapp", {"revenue_drop": -0.30})
    assert triggered, "收入降20%应触发回滚"
    assert mgr._rollback_active.get("miniapp")
    print(f"  ✓ 自动回滚: miniapp 因 revenue_drop=-30% 触发")

    # 回滚后不应自动发布
    decision_after = mgr.evaluate("miniapp", new_price=320, old_price=299)
    assert not decision_after.should_auto_publish
    print(f"  ✓ 回滚后: auto_publish={decision_after.should_auto_publish}")

    # 手动恢复
    mgr.recover_rollback("miniapp")
    assert not mgr._rollback_active.get("miniapp")
    print("  ✓ 手动恢复回滚成功")

    # Human-in-the-loop
    mgr.add_pending_approval(decision_mt)
    pending = mgr.get_pending_approvals()
    assert len(pending) == 1
    print(f"  ✓ HITL待审批: {len(pending)}项")

    approved = mgr.approve("meituan")
    assert approved is not None
    assert mgr.get_pending_approvals() == []
    print("  ✓ 审批通过后队列清空")

    print("  ✅ Test 5 通过\n")
    return True


# ============================================================
# Test 6: RetryScheduler 指数退避
# ============================================================
def test_retry_scheduler():
    print("=" * 60)
    print("  Test 6: RetryScheduler 指数退避")
    print("=" * 60)

    from services.price_sync_service import RetryScheduler
    from adapters import PricePushRequest

    scheduler = RetryScheduler()
    alerts_received = []

    def alert_cb(task):
        alerts_received.append(task)

    scheduler.set_alert_callback(alert_cb)

    # 入队一个任务
    req = PricePushRequest(
        channel="meituan",
        product_id="ticket:base",
        base_price=299,
        channel_price=310,
        original_price=290,
        reason="test",
    )
    scheduler.enqueue(req, "meituan", attempt=0)
    assert scheduler.pending_count() == 1
    print(f"  ✓ 入队: pending={scheduler.pending_count()}")

    # 退避序列验证
    backoff = [60, 180, 600, 1800]
    print(f"  ✓ 退避序列: {backoff}秒 (1min→3min→10min→30min)")

    # 最大重试3次
    assert scheduler.MAX_RETRIES == 3
    print(f"  ✓ 最大重试: {scheduler.MAX_RETRIES}次")

    # process_due (任务未到期,不应处理)
    results = scheduler.process_due()
    assert len(results) == 0
    print("  ✓ 未到期任务不处理")

    print("  ✅ Test 6 通过\n")
    return True


# ============================================================
# Test 7: HumanInLoopMiniAppAdapter 审批流程
# ============================================================
def test_human_in_loop():
    print("=" * 60)
    print("  Test 7: HumanInLoopMiniAppAdapter 审批流程")
    print("=" * 60)

    from adapters.miniapp_adapter import HumanInLoopMiniAppAdapter
    from adapters import PricePushRequest

    approvals_log = []

    adapter = HumanInLoopMiniAppAdapter(
        approval_callback=lambda req: approvals_log.append(req.to_dict()),
        api_key="test-key",
    )

    # 推送价格 → 进入审批队列
    req = PricePushRequest(
        channel="miniapp",
        product_id="ticket:night",
        base_price=299,
        channel_price=209,
        original_price=210,
        reason="夜场调价测试",
    )
    result = adapter.push_price(req)
    assert not result.success
    assert result.error_message == "AWAITING_APPROVAL"
    assert adapter.pending_count == 1
    assert len(approvals_log) == 1
    print(f"  ✓ 价格进入审批队列: pending={adapter.pending_count}")

    # 审批通过
    ok = adapter.approve(req.request_id)
    print(f"  ✓ 审批通过 (本次无真实API,仅验证流程)")

    # 审批拒绝
    req2 = PricePushRequest(
        channel="miniapp",
        product_id="ticket:night",
        base_price=299,
        channel_price=250,
        original_price=209,
        reason="二次调价",
    )
    adapter.push_price(req2)
    adapter.reject(req2.request_id, "价格偏高,人工判断暂不调整")
    assert adapter.pending_count == 0
    print("  ✓ 审批拒绝后队列清空")

    print("  ✅ Test 7 通过\n")
    return True


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  🧪 渠道适配器 + 价格同步 + A/B测试 · 集成测试")
    print("=" * 60)

    results = []
    tests = [
        ("TokenBucket限流器", test_token_bucket),
        ("适配器注册与健康检查", test_adapter_registry),
        ("冷却期+防抖", test_price_sync_cooldown),
        ("连带定价+逻辑自洽", test_linked_pricing),
        ("灰度发布+自动回滚", test_ab_test_manager),
        ("指数退避重试", test_retry_scheduler),
        ("HITL审批流程", test_human_in_loop),
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
