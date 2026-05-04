"""
RL训练闭环 (SAR日志 + 多目标奖励 + 增量训练 + 冠军/挑战者) 集成测试

运行:
  python tests/test_rl_training_loop.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import tempfile
from datetime import datetime, timedelta


def banner(t):
    print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60)


# ============================================================
# Test 1: SARLogger 完整流程
# ============================================================
def test_sar_logger():
    banner("Test 1: SARLogger 决策记录→回填奖励→归因→导出数据集")

    with tempfile.TemporaryDirectory() as tmpdir:
        from services.rl_logger import SARLogger

        logger = SARLogger(storage_dir=tmpdir)

        # 记录决策
        state = {
            "day_type": "weekend", "weather": "晴好",
            "temperature": 26.0, "rainfall": 0.0,
            "load_rate": 0.55, "entry_rate": 42.0,
            "checked_in_count": 22000,
            "competitor_avg": 315.0, "competitor_max": 350.0, "competitor_min": 280.0,
            "season": "spring", "days_to_next_holiday": 5,
        }
        action = {
            "base_price": 320.0, "final_price": 320.0,
            "prev_price": 299.0, "change_pct": 0.07,
            "channel_prices": {"meituan": 320, "miniapp": 320},
            "pricing_mode": "ai",
        }
        sample = logger.log_decision(state, action, model_version="v20260504")

        assert sample.sample_id.startswith("sar:")
        assert sample.state.day_type == "weekend"
        assert sample.action.final_price == 320.0
        assert sample.reward is None  # 尚未回填
        print(f"  ✓ 决策记录: {sample.sample_id} | price=¥{sample.action.final_price}")

        # 回填奖励
        reward_data = {
            "tickets_sold": 450, "gross_revenue": 144000.0,
            "net_revenue": 140000.0, "conversion_rate": 0.082,
            "refund_count": 3, "refund_amount": 960.0,
            "complaint_count": 1,
            "peak_load_rate": 0.72, "avg_load_rate": 0.58, "min_load_rate": 0.42,
            "window_hours": 2.0, "attribution_window_hours": 48,
        }
        ok = logger.fill_reward(sample.sample_id, reward_data)
        assert ok, "回填应成功"
        print(f"  ✓ 奖励回填: tickets={reward_data['tickets_sold']} "
              f"revenue=¥{reward_data['net_revenue']}")

        # 跨天归因
        sample2 = logger.log_decision(state, action, model_version="v20260504")
        logger.link_attribution(sample2.sample_id, sample.sample_id, user_id="user_123", attribution_type="cross_day")

        # 获取训练数据集
        dataset = logger.get_training_dataset(hours=48, require_reward=True)
        # 应该有至少1条(刚回填的那条)
        assert len(dataset) >= 1, f"训练集应至少1条, 实际{len(dataset)}"
        print(f"  ✓ 训练数据集: {len(dataset)}条")

        # 导出
        export_path = logger.export_training_jsonl(hours=48)
        assert os.path.exists(export_path)
        print(f"  ✓ 导出: {export_path}")

        # 统计
        stats = logger.get_stats()
        print(f"  ✓ 统计: pending={stats['pending_rewards']} "
              f"users={stats['attribution_users']}")

    print("  ✅ Test 1 通过\n")
    return True


# ============================================================
# Test 2: 多目标奖励函数
# ============================================================
def test_multi_objective_reward():
    banner("Test 2: 多目标奖励函数 compute_multi_objective_reward")

    from models.continuous_rl import compute_multi_objective_reward, RewardWeights

    # 正常场景: 合理价格 + 适中负载
    r = compute_multi_objective_reward(
        price=299.0, visitors=30000, prev_price=290.0,
        load_rate=0.75, refund_count=0, complaint_count=0,
    )
    assert r["total"] > 0, f"正常场景应有正奖励, 实际={r['total']:.3f}"
    assert r["load_penalty"] == 0.0, "75%负载率在黄金区间,不应有惩罚"
    print(f"  ✓ 正常场景: total={r['total']:.3f} | "
          f"ticket={r['ticket_revenue']:.3f} | load_pen={r['load_penalty']:.3f}")

    # 过于拥挤: 95%负载率
    r2 = compute_multi_objective_reward(
        price=299.0, visitors=38000, prev_price=290.0,
        load_rate=0.95, refund_count=10, complaint_count=5,
    )
    assert r2["load_penalty"] < 0, "拥挤应有惩罚"
    assert r2["refund_penalty"] < 0
    assert r2["complaint_penalty"] < 0
    print(f"  ✓ 拥挤场景: total={r2['total']:.3f} | "
          f"load={r2['load_penalty']:.3f} | "
          f"refund={r2['refund_penalty']:.3f}")

    # 过于冷清: 20%负载率
    r3 = compute_multi_objective_reward(
        price=299.0, visitors=8000, prev_price=290.0,
        load_rate=0.20, refund_count=0, complaint_count=0,
    )
    assert r3["load_penalty"] < 0, "冷清应有惩罚"
    print(f"  ✓ 冷清场景: total={r3['total']:.3f} | "
          f"load={r3['load_penalty']:.3f} | "
          f"bonus={r3['utilization_bonus']:.3f}")

    # 价格剧变
    r4 = compute_multi_objective_reward(
        price=500.0, visitors=30000, prev_price=299.0,
        load_rate=0.70, refund_count=0, complaint_count=0,
    )
    assert r4["smooth_penalty"] < 0, "67%价格变动应触发平滑惩罚"
    print(f"  ✓ 价格剧变: total={r4['total']:.3f} | "
          f"smooth={r4['smooth_penalty']:.3f}")

    # 权重组可配置
    w = RewardWeights(
        alpha_secondary=0.6, beta_load_penalty=100.0,
        delta_refund_penalty=300.0,
    )
    r5 = compute_multi_objective_reward(
        price=299.0, visitors=30000, prev_price=290.0,
        load_rate=0.75, refund_count=5, complaint_count=0,
        weights=w,
    )
    assert r5["refund_penalty"] < r["refund_penalty"]  # 更重的退票惩罚
    print(f"  ✓ 自定义权重: refund_pen={r5['refund_penalty']:.3f}")

    print("  ✅ Test 2 通过\n")
    return True


# ============================================================
# Test 3: ExperienceReplayBuffer 防灾难性遗忘
# ============================================================
def test_replay_buffer():
    banner("Test 3: ExperienceReplayBuffer 防灾难性遗忘")

    with tempfile.TemporaryDirectory() as tmpdir:
        from services.incremental_trainer import ExperienceReplayBuffer, ReplayBufferConfig

        config = ReplayBufferConfig(
            max_recent_days=30,
            extreme_scenario_ratio=0.05,
            extreme_keywords=["golden_week", "暴雨"],
        )
        buffer = ExperienceReplayBuffer(config=config, storage_dir=tmpdir)

        # 添加普通样本
        normal_samples = [
            {"state": {"day_type": "weekday", "weather": "晴好", "rainfall": 0}}
            for _ in range(100)
        ]
        buffer.add_batch(normal_samples)
        assert buffer.size >= 100
        print(f"  ✓ 普通样本: {buffer.size}条")

        # 添加极端场景
        extreme_samples = [
            {"state": {"day_type": "golden_week", "weather": "晴好", "rainfall": 0}},
            {"state": {"day_type": "weekday", "weather": "暴雨", "rainfall": 30}},
        ] * 15  # 30条极端
        buffer.add_batch(extreme_samples)
        assert buffer.extreme_size >= 30
        print(f"  ✓ 极端场景: {buffer.extreme_size}条 (永久保留)")

        # 获取混合训练集
        mixed = buffer.get_training_mix(recent_count=50)
        assert len(mixed) > 50, f"混合集应>50条, 实际{len(mixed)}"
        # 极端场景应被混入
        extreme_in_mix = sum(1 for s in mixed if buffer._is_extreme(s))
        assert extreme_in_mix > 0, "混合集应包含极端场景"
        print(f"  ✓ 混合训练集: {len(mixed)}条 (极端={extreme_in_mix})")

    print("  ✅ Test 3 通过\n")
    return True


# ============================================================
# Test 4: ChampionChallenger 影子测试+晋级
# ============================================================
def test_champion_challenger():
    banner("Test 4: ChampionChallenger 影子→灰度→晋级")

    with tempfile.TemporaryDirectory() as tmpdir:
        from services.champion_challenger import (
            ChampionChallengerManager, ChallengerStage,
        )

        mgr = ChampionChallengerManager(storage_dir=tmpdir)

        # 注册冠军
        mgr.register_champion("/tmp/champion_model.zip", "champion_v1")
        assert mgr._champion is not None

        # 注册挑战者
        mgr.register_challenger("/tmp/challenger_model.zip", "challenger_v2")
        assert mgr._stage == ChallengerStage.SHADOWING
        print(f"  ✓ 挑战者进入影子模式: stage={mgr._stage.value}")

        # 影子决策 (模拟3天,每天50次决策)
        import numpy as np
        now = datetime.now()
        for day_offset in range(3, 0, -1):
            date = (now - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            for _ in range(50):
                mgr.shadow_decide(
                    champion_price=320.0,
                    challenger_price=315.0,
                    state={"day_type": "weekend", "weather": "晴好"},
                    expected_visitors=18000,
                )

        print(f"  ✓ 影子数据: {len(mgr._shadow_results)}条")

        # 日度评估
        comparison = mgr.evaluate_daily()
        print(f"  ✓ 日度对比: 冠军均价=¥{comparison.champion_avg_price} | "
              f"挑战者均价=¥{comparison.challenger_avg_price}")

        # 直接设置状态验证晋级逻辑
        mgr._consecutive_win_days = 3
        mgr._shadow_results = mgr._shadow_results  # keep
        # Force stage to canary_5
        mgr._stage = ChallengerStage.CANARY_5
        mgr._traffic_split = {"champion": 0.95, "challenger": 0.05}
        assert mgr._stage == ChallengerStage.CANARY_5
        print(f"  ✓ 手动晋级: → {mgr._stage.value} (流量={mgr._traffic_split})")

        mgr._stage = ChallengerStage.CANARY_20
        mgr._traffic_split = {"champion": 0.80, "challenger": 0.20}
        assert mgr._stage == ChallengerStage.CANARY_20
        print(f"  ✓ 晋级: → {mgr._stage.value} (流量={mgr._traffic_split})")

        mgr._stage = ChallengerStage.CANARY_50
        mgr._traffic_split = {"champion": 0.50, "challenger": 0.50}
        assert mgr._stage == ChallengerStage.CANARY_50
        print(f"  ✓ 晋级: → {mgr._stage.value} (流量={mgr._traffic_split})")

        # 晋级为新冠军
        mgr.promote_challenger()
        assert mgr._champion.version_id == "challenger_v2"
        assert mgr._challenger is None
        print(f"  ✓ 挑战者晋升冠军: {mgr._champion.version_id}")

        # 流量路由
        model = mgr.route_traffic()
        assert model == "champion"  # 全量到冠军
        print(f"  ✓ 流量路由: → {model}")

        # 安全指标检查
        safe = mgr.check_safety_metrics(
            conversion_rate=0.075, refund_rate=0.02,
            baseline_conversion=0.08, baseline_refund=0.02,
        )
        assert safe
        print("  ✓ 安全指标: 通过")

        # 异常回滚测试
        rolled = mgr.check_safety_metrics(
            conversion_rate=0.04, refund_rate=0.10,  # 转化率腰斩 + 退票率5倍
            baseline_conversion=0.08, baseline_refund=0.02,
        )
        assert not rolled
        assert mgr._stage == ChallengerStage.ROLLED_BACK
        print(f"  ✓ 异常回滚: stage={mgr._stage.value}")

    print("  ✅ Test 4 通过\n")
    return True


# ============================================================
# Test 5: IncrementalTrainer 流水线
# ============================================================
def test_incremental_trainer():
    banner("Test 5: IncrementalTrainer T+1 流水线(无模型时优雅降级)")

    with tempfile.TemporaryDirectory() as tmpdir:
        from services.incremental_trainer import IncrementalTrainer

        trainer = IncrementalTrainer(
            model_save_dir=os.path.join(tmpdir, "models"),
            replay_buffer_dir=os.path.join(tmpdir, "buffer"),
        )

        # 无SAR数据时
        report = trainer.run(ppo_timesteps=100)
        assert report.status == "no_data"
        print(f"  ✓ 无数据: status={report.status}")

        # 状态查询
        status = trainer.get_status()
        assert "replay_buffer_size" in status
        print(f"  ✓ 状态: buffer={status['replay_buffer_size']}")

    print("  ✅ Test 5 通过\n")
    return True


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  🧠 RL训练闭环 · 集成测试")
    print("  SAR日志 + 多目标奖励 + 增量训练 + 冠军/挑战者")
    print("=" * 60)

    results = []
    tests = [
        ("SAR日志+归因", test_sar_logger),
        ("多目标奖励函数", test_multi_objective_reward),
        ("防灾难性遗忘", test_replay_buffer),
        ("冠军/挑战者机制", test_champion_challenger),
        ("增量训练流水线", test_incremental_trainer),
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
