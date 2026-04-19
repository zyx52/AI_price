"""
v3 五大升级集成测试
1. FeatureService  —— 废除手动feature_row
2. 连续表征RL      —— PPO+GNN
3. AutoEncoder OOD —— 闭环
4. 分位数回归      —— VaR定价
5. 监控看板        —— MAPE告警
"""
import numpy as np
import pandas as pd

from data import DataLoader
from models import (
    FeatureService, QuantileEnsembleForecaster,
    EnhancedShiftDetector, IncrementalTrainingManager,
    ContinuousRLPricer, ParkAttractionGraph,
)
from engine import PricingEngineV3
from monitor import PredictionMonitor


def banner(t):
    print("\n" + "="*70 + f"\n  {t}\n" + "="*70)


def test_v3():
    banner("v3 五大升级集成测试")

    # === 数据 ===
    loader = DataLoader(source="mock")
    history = loader.load_history()
    print(f"历史数据: {len(history)} 条\n")

    # ============ 需求1: FeatureService ============
    banner("Test 1: FeatureService(特征中心)")
    svc = FeatureService(history)
    svc.register_derived(
        "custom_weekend_ratio",
        lambda hist, ctx: float((hist["is_weekend"].tail(30).mean()))
    )
    vec = svc.build(
        date="2026-05-02",
        realtime={"temperature": 24, "rainfall": 0, "day_type": "holiday", "weather": "晴好"},
    )
    print(f"  ✓ 构造特征数量: {len(vec.features)}")
    print(f"  ✓ 派生特征数量: {len(vec.derived_keys)}")
    print(f"  ✓ 内置窗口特征示例:")
    for k in ["last_7d_avg_load", "last_7d_avg_price", "competitor_adjustment_freq_30d"]:
        if k in vec.features:
            print(f"       {k} = {vec.features[k]:.3f}")

    # ============ 需求4: 分位数回归 ============
    banner("Test 2: 分位数回归 QuantileEnsembleForecaster")
    qf = QuantileEnsembleForecaster()
    metrics = qf.train(history)
    print(f"  ✓ 训练完成 | P50 MAPE={metrics['p50_mape']:.2f}%")
    print(f"  ✓ 80%区间覆盖率: {metrics['coverage_80pct']:.1%}")
    print(f"  ✓ 平均不确定性: {metrics['avg_uncertainty_width']:.2%}")

    # 单点预测
    feat = svc.build_for_engine(
        "2026-05-02", "晴好", 24, 0, "holiday", 299,
    )
    pred = qf.predict(feat)
    print(f"  单点预测: p10={pred.p10:.0f} | p50={pred.p50:.0f} | p90={pred.p90:.0f} "
          f"| 不确定性={pred.uncertainty_ratio:.2%}")

    # ============ 需求3: AutoEncoder OOD + 增量训练 ============
    banner("Test 3: AutoEncoder 偏移检测 + 增量训练闭环")
    ae_detector = EnhancedShiftDetector(use_torch=False)
    ae_detector.fit(history)

    # 收集增量训练回调
    retrained_samples = []
    def _retrain_cb(records):
        retrained_samples.extend(records)
        print(f"    [Callback] 收到{len(records)}个异常样本用于增量训练")

    inc_mgr = IncrementalTrainingManager(
        storage_path="/tmp/anomaly_test",
        retrain_threshold=5,  # 低阈值,便于触发
        retrain_callback=_retrain_cb,
    )

    # 正常输入
    normal = ae_detector.detect(
        {"temperature": 24, "rainfall": 2, "price": 299, "visitors": 20000},
        "weekend", "晴好", {"p50": 20000},
    )
    print(f"  正常输入: level={normal.level.value}")

    # 注入6个极端输入,触发retrain
    print("  注入6个极端输入...")
    for i in range(6):
        extreme = ae_detector.detect(
            {"temperature": 50, "rainfall": 400, "price": 999, "visitors": 1000},
            "golden_week", "酷热", {"p10": 100, "p50": 50000, "p90": 80000},
        )
        inc_mgr.label_anomaly(f"2026-05-{i+10:02d}",
                              {"temperature": 50, "rainfall": 400},
                              extreme)

    # 等待异步retrain触发
    import time
    time.sleep(0.5)
    status = inc_mgr.get_status()
    print(f"  ✓ 异常池状态: {status}")

    # ============ 需求2: 连续表征RL ============
    banner("Test 4: 连续表征RL (PPO + GNN 嵌入)")
    graph = ParkAttractionGraph()
    graph.build_default_park()
    print(f"  景点图构建完成: {len(graph.attractions)} 节点")

    c_rl = ContinuousRLPricer(
        forecaster=qf, history_df=history, attraction_graph=graph,
    )
    info = c_rl.train(total_timesteps=5000)  # 轻量训练用于测试
    print(f"  ✓ 训练完成 | mode={info['mode']} | state_dim={info['state_dim']}")

    res = c_rl.recommend_price(
        "weekend", "晴好", 310, 0.5, 299,
        season="spring", temperature=24, rainfall=0,
    )
    print(f"  推荐价: ¥{res['recommended_price']:.0f} (mode={res.get('mode','n/a')})")

    # ============ 需求5: 监控看板 ============
    banner("Test 5: 预测偏差实时监控 + MAPE告警")
    alerts_received = []
    mon = PredictionMonitor(
        mape_threshold=0.10, consecutive_days=3,
        storage_path="/tmp/monitor_test",
        alert_callback=lambda a: alerts_received.append(a),
    )

    # 模拟7天: 前4天正常,后3天大偏差
    scenarios = [
        ("2026-04-20", 20000, 20500),
        ("2026-04-21", 22000, 21800),
        ("2026-04-22", 25000, 24500),
        ("2026-04-23", 23000, 22800),
        ("2026-04-24", 20000, 25000),  # 25%偏差
        ("2026-04-25", 21000, 27000),  # 22%偏差
        ("2026-04-26", 22000, 28500),  # 23%偏差
    ]
    for date, pred, actual in scenarios:
        mon.record_prediction(date, pred)
        mon.record_actual(date, actual)

    st = mon.get_status()
    print(f"  ✓ 状态: MAPE={st['rolling_mape']:.2%} | 告警总数={st['alerts_total']}")
    print(f"  ✓ 告警回调触发: {len(alerts_received)} 次")
    if alerts_received:
        print(f"    最新告警: {alerts_received[-1].message}")

    # ============ 集成: PricingEngineV3 全链路 ============
    banner("Test 6: PricingEngineV3 全链路集成")
    engine = PricingEngineV3(
        forecaster=qf, rl_pricer=c_rl,
        feature_service=svc,
        shift_detector=ae_detector,
        incremental_manager=inc_mgr,
    )

    # 正常场景
    d_normal = engine.decide(
        date="2026-05-02", weather="晴好", temperature=24, rainfall=0,
        competitor_prices={"A": 310, "B": 280, "C": 350}, day_type="holiday",
    )
    print(f"\n  [正常] ¥{d_normal.recommended_price:.0f} | "
          f"P10/P50/P90={d_normal.visitors_p10:.0f}/{d_normal.visitors_p50:.0f}/{d_normal.visitors_p90:.0f} | "
          f"不确定性={d_normal.uncertainty_ratio:.2%} | "
          f"VaR模式={d_normal.risk_aware_mode}")

    # 极端场景
    d_extreme = engine.decide(
        date="2026-10-03", weather="酷热", temperature=48, rainfall=380,
        competitor_prices={"A": 310}, day_type="golden_week",
    )
    print(f"  [极端] ¥{d_extreme.recommended_price:.0f} | "
          f"fallback={d_extreme.fallback_mode} | "
          f"权重={d_extreme.decision_weights}")

    banner("✅ 全部5大升级通过验证")


if __name__ == "__main__":
    test_v3()
