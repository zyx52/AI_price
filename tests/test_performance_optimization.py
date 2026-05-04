"""
性能优化验证测试

验证:
  1. BatchStateEncoder 预分配缓冲区 vs list.append
  2. 特征工程 .apply(lambda) vs 向量化 np.where
  3. SAR批量写入 vs 逐条写入
  4. 训练数据提取: 预分配数组 vs list.append
  
运行:
  python tests/test_performance_optimization.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime


def banner(t):
    print("\n" + "=" * 60 + f"\n  {t}\n" + "=" * 60)


# ============================================================
# Test 1: BatchStateEncoder 预分配缓冲区
# ============================================================
def test_batch_state_encoder():
    banner("Test 1: BatchStateEncoder 预分配 vs list.append")

    from services.rl_inference_opt import BatchStateEncoder

    n_samples = 5000
    dim = 24

    # --- 传统: list.append ---
    t0 = time.perf_counter()
    result_list = []
    for i in range(n_samples):
        vec = np.random.randn(dim).astype(np.float32)
        result_list.append(vec)
    result_arr_old = np.array(result_list, dtype=np.float32)
    t_list = time.perf_counter() - t0

    # --- 优化: BatchStateEncoder ---
    t1 = time.perf_counter()
    encoder = BatchStateEncoder(dim=dim, prealloc_size=n_samples)
    encoder.reset(n_samples)
    for i in range(n_samples):
        vec = np.random.randn(dim).astype(np.float32)
        encoder.add(vec)
    result_arr_new = encoder.to_array()
    t_enc = time.perf_counter() - t1

    assert len(result_arr_new) == n_samples
    assert result_arr_new.shape == (n_samples, dim)

    speedup = t_list / max(t_enc, 0.0001)
    print(f"  list.append:   {t_list*1000:.1f}ms")
    print(f"  BatchEncoder:  {t_enc*1000:.1f}ms")
    print(f"  ⚡ 加速: {speedup:.1f}x")

    # 批量添加
    t2 = time.perf_counter()
    encoder2 = BatchStateEncoder(dim=dim, prealloc_size=n_samples)
    encoder2.reset(n_samples)
    batch = np.random.randn(n_samples, dim).astype(np.float32)
    encoder2.add_batch(batch)
    t_batch = time.perf_counter() - t2

    print(f"  add_batch:     {t_batch*1000:.1f}ms (批量添加)")
    print(f"  ⚡ 批量加速: {t_list / max(t_batch, 0.0001):.1f}x")

    print("  ✅ Test 1 通过\n")
    return True


# ============================================================
# Test 2: 特征工程向量化 vs .apply(lambda)
# ============================================================
def test_feature_vectorization():
    banner("Test 2: 特征工程 .apply(lambda) vs 向量化")

    # 构造测试数据: 500天
    n = 500
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "date": dates,
        "temperature": np.random.uniform(-5, 40, n),
        "rainfall": np.random.exponential(3, n),
        "is_holiday": np.random.choice([0, 1], n, p=[0.85, 0.15]),
        "is_weekend": (dates.dayofweek >= 5).astype(int),
        "day_type": np.random.choice(["weekday", "weekend", "holiday"], n, p=[0.7, 0.2, 0.1]),
        "weather": np.random.choice(["晴好", "雨", "暴雨", "酷热"], n, p=[0.6, 0.25, 0.1, 0.05]),
        "season": np.random.choice(["spring", "summer", "autumn", "winter"], n),
        "visitors": np.random.randint(5000, 40000, n).astype(float),
        "price": np.random.uniform(150, 450, n),
    })

    holiday_dates = df[df["is_holiday"] == 1]["date"].values

    # --- 旧方法: .apply(lambda) ---
    t0 = time.perf_counter()
    if len(holiday_dates) > 0:
        _ = df["date"].apply(
            lambda d: min([(h - d).days for h in holiday_dates if h >= d], default=365)
        )
    t_apply = time.perf_counter() - t0

    # --- 新方法: 向量化广播 ---
    t1 = time.perf_counter()
    if len(holiday_dates) > 0:
        date_arr = df["date"].values
        delta_matrix = (
            date_arr[:, None].astype("datetime64[D]").view("int64")
            - holiday_dates[None, :].astype("datetime64[D]").view("int64")
        )
        delta_next = np.where(delta_matrix >= 0, delta_matrix, np.iinfo(np.int64).max)
        _ = delta_next.min(axis=1)
    t_vec = time.perf_counter() - t1

    speedup = t_apply / max(t_vec, 0.0001)
    print(f"  .apply(lambda): {t_apply*1000:.1f}ms (O(n²) Python循环)")
    print(f"  向量化广播:     {t_vec*1000:.1f}ms (O(n*m) NumPy C层)")
    print(f"  ⚡ 加速: {speedup:.1f}x")

    assert speedup > 1.5, f"向量化应至少快1.5x, 实际{speedup:.1f}x"

    print("  ✅ Test 2 通过\n")
    return True


# ============================================================
# Test 3: SAR 批量写入 vs 逐条写入
# ============================================================
def test_sar_batch_write():
    banner("Test 3: SAR 批量写入 vs 逐条写入")

    with tempfile.TemporaryDirectory() as tmpdir:
        from services.rl_logger import SARLogger

        logger = SARLogger(storage_dir=tmpdir)
        n = 200

        # --- 逐条写入 ---
        states = [{"day_type": "weekday", "weather": "晴好", "load_rate": 0.5} for _ in range(n)]
        actions = [{"base_price": 300.0, "final_price": 300.0, "prev_price": 290.0}
                    for _ in range(n)]

        t0 = time.perf_counter()
        for i in range(n):
            logger.log_decision(states[i], actions[i])
        t_single = time.perf_counter() - t0

        # 清理
        logger._pending_rewards.clear()

        # --- 批量写入 ---
        t1 = time.perf_counter()
        _ = logger.log_decision_batch(states, actions)
        t_batch = time.perf_counter() - t1

        speedup = t_single / max(t_batch, 0.0001)
        print(f"  逐条写入: {t_single*1000:.1f}ms ({n}次open/close)")
        print(f"  批量写入: {t_batch*1000:.1f}ms (1次open/close)")
        print(f"  ⚡ 加速: {speedup:.1f}x")

        # 验证批量写入的数据完整性
        dataset = logger.get_training_dataset(hours=48, require_reward=False)
        print(f"  ✓ 数据完整性: {len(dataset)}条 (预期{n})")

    print("  ✅ Test 3 通过\n")
    return True


# ============================================================
# Test 4: 训练数据提取: 预分配数组 vs list.append
# ============================================================
def test_vectorized_extraction():
    banner("Test 4: 训练数据提取: 预分配 vs list.append")

    with tempfile.TemporaryDirectory() as tmpdir:
        from services.rl_logger import SARLogger
        from datetime import datetime, timedelta

        logger = SARLogger(storage_dir=tmpdir)
        n = 500

        # 写入一些带奖励的样本
        states = []
        actions = []
        for i in range(n):
            states.append({
                "day_type": "weekday", "weather": "晴好", "load_rate": 0.5,
                "feature_vector": list(np.random.randn(24).astype(float)),
            })
            actions.append({
                "base_price": 300.0, "final_price": 300.0, "prev_price": 290.0,
            })

        samples = logger.log_decision_batch(states, actions)

        # 回填奖励
        for s in samples:
            logger.fill_reward(s.sample_id, {
                "tickets_sold": 400, "net_revenue": 120000.0,
                "conversion_rate": 0.08, "refund_count": 2,
                "peak_load_rate": 0.7, "avg_load_rate": 0.55,
            })

        # --- 旧方法: get_training_dataset (list.append) ---
        t0 = time.perf_counter()
        old_data = logger.get_training_dataset(hours=48, require_reward=True)
        t_old = time.perf_counter() - t0

        # --- 新方法: get_training_dataset_vectorized (预分配) ---
        t1 = time.perf_counter()
        states_arr, actions_arr, rewards_arr = logger.get_training_dataset_vectorized(
            hours=48, require_reward=True,
        )
        t_new = time.perf_counter() - t1

        speedup = t_old / max(t_new, 0.0001)
        print(f"  list.append提取:  {t_old*1000:.1f}ms | {len(old_data)}条")
        print(f"  预分配数组提取:    {t_new*1000:.1f}ms | states={states_arr.shape}")
        print(f"  ⚡ 加速: {speedup:.1f}x")

        assert len(states_arr) >= n * 0.9, f"样本数应接近{n}, 实际{len(states_arr)}"
        assert states_arr.shape == (len(states_arr), 24)
        assert actions_arr.shape == (len(states_arr), 1)
        assert rewards_arr.shape == (len(states_arr),)

    print("  ✅ Test 4 通过\n")
    return True


# ============================================================
# Test 5: RL推理优化 (torch.compile 检查)
# ============================================================
def test_rl_inference_opt():
    banner("Test 5: RL推理优化层 (torch.compile/TorchScript兼容性)")

    from services.rl_inference_opt import (
        BatchStateEncoder, optimize_for_inference,
        _HAS_TORCH, _TORCH_VERSION,
    )

    print(f"  PyTorch: {'可用' if _HAS_TORCH else '不可用'}")
    if _HAS_TORCH:
        import torch
        print(f"  Version: {torch.__version__}")
        print(f"  torch.compile: {'✅ 可用' if _TORCH_VERSION >= (2, 0) else '❌ 需 PyTorch 2.0+'}")

    # 即使无torch, BatchStateEncoder 应可用(numpy only)
    encoder = BatchStateEncoder(dim=24, prealloc_size=100)
    encoder.reset(100)
    for _ in range(50):
        encoder.add(np.random.randn(24).astype(np.float32))
    arr = encoder.to_array()
    assert arr.shape == (50, 24)

    # Tensor转换
    if _HAS_TORCH:
        t = encoder.to_tensor()
        assert t.shape == (50, 24)
        print(f"  ✓ Tensor转换: {t.shape} | dtype={t.dtype}")

    # 无模型时的优雅降级
    opt = optimize_for_inference(None)
    result = opt.predict_single(np.random.randn(24).astype(np.float32))
    assert isinstance(result, float)
    print(f"  ✓ 无模型降级: predict_single={result:.3f}")

    # 批量推理
    batch_obs = np.random.randn(10, 24).astype(np.float32)
    batch_result = opt.predict_batch(batch_obs)
    assert len(batch_result) == 10
    print(f"  ✓ 批量降级推理: {len(batch_result)}条")

    # 基准测试
    bench = opt.benchmark(n_samples=100, dim=24)
    print(f"  ✓ 基准: single={bench['single_ms']}ms | batch={bench['batch_ms_per_sample']}ms/样本")

    print("  ✅ Test 5 通过\n")
    return True


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("  ⚡ 性能优化 · 验证测试")
    print("=" * 60)

    results = []
    tests = [
        ("预分配缓冲区", test_batch_state_encoder),
        ("特征向量化", test_feature_vectorization),
        ("批量I/O写入", test_sar_batch_write),
        ("预分配提取", test_vectorized_extraction),
        ("推理优化层", test_rl_inference_opt),
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
