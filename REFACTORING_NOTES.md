# P0-P3 重构说明 (v3.1)

更新日期: 2026-04-18

本次重构面向 P0-P3 优先级需求，核心改动已完成并通过集成验证。
在 v3 基础上，2026-04-18 增加了一轮稳定性与可配置性修复。

## 验证方式

```bash
cd ai_pricing_platform
python test_requirements.py
```

## P0 · 核心性能与并发架构

### P0-01 API 全异步化

文件: `api/main.py`

- 所有路由改为 `async def`
- CPU 密集型调用通过 `run_in_threadpool(...)` 下沉线程池
- 使用 `@asynccontextmanager` 进行启动预加载
- 统一结构化错误码: `INVALID_INPUT` / `INTERNAL_ERROR` / `RELOAD_FAILED`

### P0-02 批处理预测

文件: `models/ensemble_forecaster.py`

- 新增 `predict_batch(feature_rows, target_dates)`
- `demand_curve(...)` 从逐点循环改为批量 DataFrame 推理
- 实测批处理显著快于单点循环

### P0-03 特征工程解耦与缓存

文件: `utils/feature_cache.py`, `models/ensemble_forecaster.py`

- 引入 `FeatureCache` (TTL + 线程安全)
- 训练后自动预热: `feature_cache.preload_baseline(history)`
- 热路径改为轻量 `merge_with_realtime`
- 支持 `/admin/invalidate-cache` 手动刷新

## P1 · 算法安全性与定价风控

### P1-01 RL 仿真环境

文件: `models/park_env.py`

- 新增 `ParkEnv`，将需求预测能力包装为仿真环境
- 新增 `SimulatorBasedQLearning`，在环境中探索完整动作空间
- 相比旧版仅基于历史动作的 Q-learning，探索性和泛化性更强

### P1-02 分布偏移检测与动态降级

文件: `models/shift_detector.py`, `engine/pricing_engine.py`

- 新增 `DistributionShiftDetector` (Z-score + 未见组合 + 预测分歧)
- 四级响应策略:
  - `NORMAL` -> `(0.30, 0.45, 0.25)`
  - `LIGHT` -> `(0.50, 0.35, 0.15)`
  - `SEVERE` -> `(0.80, 0.15, 0.05)`
  - `CRITICAL` -> `(1.00, 0.00, 0.00)` (规则兜底)
- `PricingEngine.decide()` 输出 `shift_detection`，便于追溯

## P2 · 接口鲁棒性

### P2-01 API 入参强制校验

文件: `api/main.py`

- `date` 使用正则约束 `^\d{4}-\d{2}-\d{2}$`
- `temperature`: `ge=-30, le=50`
- `rainfall`: `ge=0, le=500`
- `weather` 使用枚举校验
- `competitor_prices` 与 `prev_price` 使用范围约束
- 非法输入统一返回 `400` + `INVALID_INPUT`

### P2-02 移除隐式错误吞噬

文件: `models/ensemble_forecaster.py`, `engine/pricing_engine.py`

- 移除 `except Exception: pass`
- 构建失败显式日志并抛出
- Engine 捕获后进入业务规则 fallback 并保留可追踪日志

### P2-03 TimeSeriesSplit 滚动验证

文件: `models/ensemble_forecaster.py`

- 改用 `TimeSeriesSplit(n_splits=3)`
- 融合权重基于 3 折平均 MAPE 反向加权
- 降低时间泄漏和阶段性过拟合风险

## P3 · 工程规范化

### P3-01 全局配置动态化

文件: `config/settings.py`, `.env.example`

- 采用 `pydantic-settings`
- 支持优先级: 默认值 < `.env` < 系统环境变量
- 支持嵌套变量写法: `PRICING__MIN_PRICE=80`
- 新增 `settings.reload()` 运行时热更新
- `/admin/reload-config` 支持运维触发

## v3.1 增量修复 (2026-04-18)

### A. Problems 定位与类型修复

文件: `models/feature_service.py`

- 修复 `np.diff(...)` 入参类型冲突
- `Series.values` 改为 `.to_numpy(dtype=float, copy=False)`

文件: `utils/weather_client.py`

- 修复 `requests` 可选导入的“可能未绑定”
- 导入失败分支增加显式占位，保持降级能力

文件: `models/park_env.py`

- 修复 `current_scenario` 的 Optional 下标访问告警
- 在 `step()` 与 `_observation()` 增加空值守卫和类型收敛

文件: `models/ppo_pricer.py`

- 修复条件导入下 `gym.Env` 继承基类告警
- 通过 `GymEnvBase = cast(type[Any], gym.Env)` 收敛后再继承

### B. NLP 报告模块支持多 API 切换

文件: `models/nlp_reporter.py`

- 新增可配置参数: `provider`, `model`, `base_url`, `api_key`
- 支持后端:
  - `anthropic`
  - `openai`
  - `deepseek` / `moonshot` / `qwen` (OpenAI 兼容方式)
- 支持环境变量:
  - `NLP_API_PROVIDER`
  - `NLP_API_MODEL`
  - `NLP_API_BASE_URL`
  - `NLP_API_KEY`
- SDK 缺失或调用失败时自动降级到模板模式

### C. 依赖与文档同步

文件: `requirements.txt`

- 固定 `lightgbm==4.6.0` 以提升可复现性

文件: `README.md`

- 重写“快速开始”
- 补充 Windows 虚拟环境与自检流程
- 增加 `main_v2.py` 与可选依赖安装说明

### D. 环境与功能验证

- 已安装并验证导入:
  - `lightgbm==4.6.0`
  - `stable-baselines3==2.8.0`
  - `gymnasium==1.2.3`
  - `torch==2.11.0+cpu`
  - `torch-geometric==2.7.0`
- 验证脚本结果:
  - `python test_requirements.py`
  - 输出: `全部P0-P3需求验证通过`

## 测试结果摘要

| Test | 需求 | 关键结果 |
| --- | --- | --- |
| 1 | P2-03 | Ensemble 3 折平均 MAPE 约 8% |
| 2 | P0-03 | 特征缓存预热生效，TTL 正常 |
| 3 | P0-02 | 批处理明显快于单点循环 |
| 4 | P1-02 | 极端输入触发 CRITICAL 规则兜底 |
| 5 | P1-01 | 仿真 RL 训练完成，状态覆盖稳定 |
| 6 | P1-02 | Engine 集成下正常与极端场景行为符合预期 |
| 7 | P2-01 | 非法温度/日期/weather/竞品价均被拒绝 |
| 8 | P3-01 | `settings.reload()` 热更新生效 |

## 兼容性说明

- `main.py` 仍可独立运行 (旧版 DemandForecaster + RLPricer)
- `main_v2.py` 使用新版 Ensemble + RLPricerV2
- `models.__init__` 兼容原有导出
- `settings.alert_thresholds` 字典属性保留，兼容旧代码
