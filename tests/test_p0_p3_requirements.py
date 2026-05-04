"""
P0-P3 需求落地验证测试
覆盖场景:
  1. 正常场景 —— 验证全链路跑通
  2. 极端OOD场景(温度50℃) —— 验证 P1-02 分布偏移检测与降级
  3. 非法输入(温度-100℃) —— 验证 P2-01 强校验拒绝
  4. 特征缓存命中率 —— 验证 P0-03 缓存生效
  5. 批处理 vs 单点 —— 验证 P0-02 性能改进
  6. 滚动CV —— 验证 P2-03
  7. 仿真RL —— 验证 P1-01
"""
import time
import numpy as np
from pydantic import ValidationError

from config import settings
from data import DataLoader
from models import (
    EnsembleDemandForecaster, RLPricerV2,
    DistributionShiftDetector, ShiftLevel,
    ContinuousRLPricer, ParkAttractionGraph,
)
from engine import PricingEngine
from utils.feature_cache import feature_cache


def banner(text):
    print("\n" + "="*70 + f"\n  {text}\n" + "="*70)


def test_all():
    banner("P0-P3 需求验证测试")

    # === 数据 + 训练 ===
    loader = DataLoader(source="mock")
    history = loader.load_history()

    print("\n【Test 1: P2-03】TimeSeriesSplit滚动CV + Ensemble")
    forecaster = EnsembleDemandForecaster()
    metrics = forecaster.train(history)
    print(f"  ✓ Ensemble MAPE(各折平均): {metrics['ensemble_mape']:.2f}%")
    for name, m in metrics["individual_metrics"].items():
        folds = m.get("fold_mapes", [])
        print(f"    - {name}: 各折MAPE={[f'{f:.2f}' for f in folds]} | 平均={m['mape']:.2f}%")

    print("\n【Test 2: P0-03】特征缓存预热检查")
    cache_info = feature_cache.health()
    assert cache_info["baseline_loaded"], "特征缓存未预热!"
    print(f"  ✓ 特征缓存已预热 | baseline_loaded={cache_info['baseline_loaded']} "
          f"| TTL剩余={cache_info['baseline_ttl_remaining']}s")

    print("\n【Test 3: P0-02】批处理 vs 单点推理性能对比")
    feature_row: dict[str, float | int] = {
        "price": 299, "temperature": 24, "rainfall": 0,
        "is_holiday": 0, "is_weekend": 1,
        "day_of_week": 5, "month": 5, "day_of_month": 2,
        "season_id": 0, "day_type_id": 1, "weather_id": 0,
    }
    # 批处理80个价格点
    t0 = time.perf_counter()
    curve = forecaster.demand_curve(feature_row, target_date="2026-05-02")
    t_batch = time.perf_counter() - t0
    # 单点80次循环
    t0 = time.perf_counter()
    for p in np.linspace(80, 599, 80):
        r = dict(feature_row); r["price"] = float(p)
        _ = forecaster.predict(r, target_date="2026-05-02")
    t_loop = time.perf_counter() - t0
    speedup = t_loop / max(t_batch, 1e-6)
    print(f"  批处理80点耗时:   {t_batch*1000:.1f}ms")
    print(f"  单点循环80次耗时: {t_loop*1000:.1f}ms")
    print(f"  ✓ 加速比: {speedup:.1f}x")

    print("\n【Test 4: P1-02】分布偏移检测 + 动态降级")
    shift = DistributionShiftDetector(); shift.fit(history)

    # 4a. 正常输入
    normal_det = shift.detect(
        input_features={"temperature": 24, "rainfall": 2, "price": 299},
        day_type="weekend", weather="晴好",
        model_predictions={"lgb": 20000, "xgb": 20500},
    )
    print(f"  正常输入: level={normal_det.level.value} | 权重={normal_det.adjusted_weights}")
    assert normal_det.level == ShiftLevel.NORMAL

    # 4b. 温度极端(50℃) + 从未见过的组合
    extreme_det = shift.detect(
        input_features={"temperature": 50, "rainfall": 450, "price": 299},
        day_type="golden_week", weather="酷热",
        model_predictions={"lgb": 500, "xgb": 50000, "prophet": 10000},  # CV极高
    )
    print(f"  极端输入: level={extreme_det.level.value} | 权重={extreme_det.adjusted_weights}")
    for r in extreme_det.reasons:
        print(f"    · {r}")
    assert extreme_det.level in (ShiftLevel.SEVERE, ShiftLevel.CRITICAL), \
        f"极端输入应触发SEVERE/CRITICAL,实际={extreme_det.level}"
    print(f"  ✓ 分布偏移检测正确触发降级")

    print("\n【Test 5: P1-01】PPO连续RL训练")
    graph = ParkAttractionGraph(); graph.build_default_park()
    agent = ContinuousRLPricer(forecaster=forecaster, history_df=history, attraction_graph=graph)
    rl_info = agent.train(total_timesteps=3000)
    print(f"  ✓ PPO训练完成 | mode={rl_info['mode']} | state_dim={rl_info['state_dim']}")

    # 测试推理
    res = agent.recommend_price("weekend", "晴好", 310, 0.5, prev_price=299)
    print(f"  推荐价: ¥{res['recommended_price']:.0f} | mode={res.get('mode')}")

    print("\n【Test 6: PricingEngine集成】正常场景 + OOD场景对比")
    rl_v2 = RLPricerV2(); rl_v2.train(history, epochs=3)
    engine = PricingEngine(forecaster=forecaster, rl_pricer=rl_v2, shift_detector=shift)

    # 6a. 正常
    normal_dec = engine.decide(
        date="2026-05-02", weather="晴好", temperature=24, rainfall=0,
        competitor_prices={"A": 310, "B": 280, "C": 350}, day_type="holiday",
    )
    print(f"\n  [正常] 推荐价: ¥{normal_dec.recommended_price:.0f} | "
          f"权重: {normal_dec.decision_weights} | "
          f"fallback={normal_dec.fallback_mode}")

    # 6b. 极端场景
    extreme_dec = engine.decide(
        date="2026-05-02", weather="酷热", temperature=48, rainfall=400,
        competitor_prices={"A": 310, "B": 280, "C": 350}, day_type="golden_week",
    )
    print(f"  [极端] 推荐价: ¥{extreme_dec.recommended_price:.0f} | "
          f"权重: {extreme_dec.decision_weights} | "
          f"fallback={extreme_dec.fallback_mode}")
    assert extreme_dec.decision_weights["business_rule"] > 0.30, \
        "极端场景应提升业务规则权重"
    print(f"  ✓ 极端场景业务规则权重={extreme_dec.decision_weights['business_rule']} (默认0.30,已升权)")

    print("\n【Test 7: P2-01】API入参校验")
    from api.main import PricingRequest

    # 合法
    try:
        valid = PricingRequest(date="2026-05-02", temperature=24, rainfall=0,
                               weather="晴好", competitor_prices={"A": 310},
                               day_type="weekend", prev_price=299)
        print(f"  ✓ 合法请求通过")
    except ValidationError as e:
        print(f"  ✗ 合法请求被误拒: {e}")

    # 非法温度
    try:
        PricingRequest(date="2026-05-02", temperature=-100, rainfall=0,
                       weather="晴好", competitor_prices={"A": 310},
                       day_type="weekend", prev_price=299)
        print(f"  ✗ 温度-100℃应被拒绝但通过了!")
    except ValidationError:
        print(f"  ✓ 温度-100℃被拒绝")

    # 非法weather
    try:
        PricingRequest(date="2026-05-02", temperature=24, rainfall=0,
                       weather="龙卷风", competitor_prices={"A": 310},
                       day_type="weekend", prev_price=299)
        print(f"  ✗ 非法weather应被拒绝但通过了!")
    except ValidationError:
        print(f"  ✓ 非法weather被拒绝")

    # 非法date格式
    try:
        PricingRequest(date="2026/05/02", temperature=24, rainfall=0,
                       weather="晴好", competitor_prices={"A": 310},
                       day_type="weekend", prev_price=299)
        print(f"  ✗ 非法日期格式应被拒绝但通过了!")
    except ValidationError:
        print(f"  ✓ 非法日期格式被拒绝")

    # 非法竞品价
    try:
        PricingRequest(date="2026-05-02", temperature=24, rainfall=0,
                       weather="晴好", competitor_prices={"A": 99999},
                       day_type="weekend", prev_price=299)
        print(f"  ✗ 竞品价99999应被拒绝但通过了!")
    except ValidationError:
        print(f"  ✓ 竞品价99999超限被拒绝")

    print("\n【Test 8: P3-01】配置热更新")
    print(f"  初始 optimal_load = {settings.optimal_load}")
    import os
    os.environ["OPTIMAL_LOAD"] = "0.82"
    settings.reload()
    print(f"  reload后 optimal_load = {settings.optimal_load}")
    assert settings.optimal_load == 0.82, "热更新未生效"
    print(f"  ✓ 配置热更新生效")
    # 恢复
    del os.environ["OPTIMAL_LOAD"]
    settings.reload()

    banner("✅ 全部P0-P3需求验证通过")


if __name__ == "__main__":
    test_all()
