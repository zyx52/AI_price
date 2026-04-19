# AI智能定价票务平台 (AI-Pricing-Platform)

> 面向乐园运营管理层的动态定价决策系统

## 一、项目背景

针对乐园运营中的**季节性波动、天气敏感性、节假日高峰、客流不均衡**等核心场景,构建一套基于AI的智能定价与票务决策平台,目标:

- 提升游客体验(减少排队、优化入园节奏、个性化服务)
- 最大化园区整体收入(门票 + 二次消费 + 套餐组合)

## 二、系统架构

```text
┌─────────────────────────────────────────────────────────────┐
│                    Dashboard (可视化看板)                     │
│          数据看板 │ 定价建议 │ 市场预警 │ 策略复盘            │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                    Pricing Engine (定价引擎)                 │
│   业务规则  +  ML需求预测  +  RL动态定价  +  套餐组合优化     │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                Monitor (市场监控与预警模块)                  │
│        天气预报 │ 竞品价格 │ 节假日 │ 异常流量检测            │
└─────────────────────────────────────────────────────────────┘
                            ▲
                            │
┌─────────────────────────────────────────────────────────────┐
│                  Data Layer (数据层)                         │
│     历史销售 │ 客流数据 │ 二次消费 │ 外部数据(天气/节假日)    │
└─────────────────────────────────────────────────────────────┘
```

## 三、核心模块

### 1. 数据看板与叙事模块 (`dashboard/`)

将复杂销售/客流数据转化为趋势图、机会点提示。

### 2. 定价与产品决策模块 (`engine/`)

基于数据洞察自动提出票价调整、套餐组合方案(如"雨天特惠+室内项目优先券")。

当前版本新增:

- 概率分布预测: 返回 P10/P50/P90 与 `Uncertainty Spread`
- 高不确定性保守策略: `Uncertainty Spread > 0.40` 自动切换 P10 底线策略
- 连续RL: PPO 连续动作 + 平滑映射 + 剧烈调价惩罚
- OOD拦截: AutoEncoder 重建误差触发 CRITICAL 时熔断到业务规则

### 3. 市场监控与预警模块 (`monitor/`)

结合外部数据(天气、竞品、节假日)识别潜在市场变化与风险。

当前版本新增:

- 分布水位监控: P10-P90 区间实时可视化
- OOD 触发次数/熔断率实时统计
- CRITICAL 异常上下文推送消息队列 (Redis Stream)

### 4. AI模型模块 (`models/`)

- **需求预测**: LightGBM/XGBoost 预测未来客流
- **动态定价**: 强化学习(PPO) 在收入与客流均衡间寻优
- **NLP报告生成**: 自动生成运营日报
- **图神经网络(预留)**: 处理多维影响因素之间的关联

## 四、动态定价核心维度 (3-5个关键维度)

经benchmark与业务分析,筛选出以下**5个核心定价维度**:

| 维度 | 业务含义 | 数据来源 |
| --- | --- | --- |
| **时间维度** | 工作日/周末、节假日、季节 | 日历+历史销售 |
| **天气维度** | 温度、降水、极端天气 | 气象API |
| **客群维度** | 家庭客/年轻客/团体/亲子 | 订单画像 |
| **需求弹性** | 历史价格-销量曲线 | 自建弹性矩阵 |
| **竞品与宏观** | 周边乐园价格、区域景气度 | 竞品爬虫+宏观数据 |

## 五、技术路径

| 技术 | 应用场景 |
| --- | --- |
| LightGBM / XGBoost | 客流量、二消金额需求预测 |
| 强化学习 (PPO) | 动态定价策略学习 |
| Prophet | 季节性/节假日基线预测 |
| NLP (LLM API) | 运营日报、机会点叙事生成 |
| 图神经网络 GNN | 景区多维影响因素关联建模(预留) |
| FastAPI | 对外API服务 |
| Streamlit | 运营管理看板前端 |

## 六、GAI应用畅想

### 场景: AI运营助理

- 每日自动推送: "明日预计降雨概率70%,建议启动'雨天特惠+室内项目优先券'套餐,预计可挽回15%客流"
- 支持管理层自然语言提问: "下周客流高峰日哪些时段该加价?"
- 自动生成周报/月报、投诉洞察、机会点摘要

## 七、目录结构

```text
ai_pricing_platform/
├── config/          # 全局配置
├── data/            # 数据加载层 (真实数据接入点)
├── models/          # AI模型: 需求预测、RL定价、NLP
├── engine/          # 定价决策引擎
├── monitor/         # 市场监控与预警
├── dashboard/       # Streamlit可视化看板
├── api/             # FastAPI服务
├── utils/           # 工具函数
├── main.py          # 项目主入口
├── requirements.txt
└── README.md
```

## 八、快速开始

### 1. 环境准备

- Python 3.10-3.12 (推荐)
- 建议使用虚拟环境隔离依赖

### 2. 安装依赖

```bash
# Windows PowerShell: 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 升级pip并安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt

# 运行依赖与核心能力自检
python test_requirements.py
```

> 说明: `test_requirements.py` 通过即表示 P0-P3 关键能力可运行。

### 3. 运行主流程

```bash
# 方式1: 运行主流程(命令行演示)
python main.py

# 方式1-增强版: 运行全功能演示 v2
python main_v2.py

# 老终端纯 ASCII 输出模式(避免 emoji/中文导致显示或编码问题)
$env:AI_PRICING_ASCII="1"; python main_v2.py
```

### 4. 启动看板与API

```bash
# 方式2: 启动可视化看板
streamlit run dashboard/app.py

# 方式3: 启动API服务
uvicorn api.main:app --reload --port 8000
```

### 5. 运行 30 天 Shadow Backtest

对照策略:

- 新版概率决策: `PricingEngineV3` (P10/P50/P90 + uncertainty spread)
- 旧版固定权重: `PricingEngine` (固定权重加权)

```bash
python scripts/shadow_backtest_30d.py --window-days 30 --ppo-timesteps 3000
```

输出文件默认写入 `data/backtests/`:

- `shadow_backtest_30d_detail_*.csv`
- `shadow_backtest_30d_summary_*.json`

### 6. 独立 Inference Service 示例

HTTP 服务 (可直接用于网关转发):

```bash
uvicorn services.inference_http_service:app --host 0.0.0.0 --port 9001
```

gRPC 服务:

1. 先生成 Python 代码

```bash
python -m grpc_tools.protoc \
    -I services/grpc/proto \
    --python_out=services/grpc \
    --grpc_python_out=services/grpc \
    services/grpc/proto/inference.proto
```

1. 启动 gRPC server

```bash
python services/grpc/server.py --host 0.0.0.0 --port 50051
```

### 7. API 网关转发到独立推理服务

在 API 网关进程设置:

```bash
INFERENCE_SERVICE_URL=http://127.0.0.1:9001
INFERENCE_SERVICE_TIMEOUT_SECONDS=3
```

此时 `api/main.py` 会将 `/pricing/decide` 与 `/pricing/decide-with-quantiles` 转发到独立推理服务。

### 8. 可选依赖(按需安装)

```bash
# PPO强化学习(必需)
pip install stable-baselines3 gymnasium

# GNN增强模块(可选)
pip install torch torch-geometric

# 分布式缓存与异常事件流
pip install redis

# gRPC inference (optional)
pip install grpcio grpcio-tools
```

> 说明: 连续RL路径已强制使用 PPO,不再降级到离散Q-learning。

### 9. NFR 运行开关

可通过环境变量启用分布式缓存和推理服务解耦:

```bash
# Redis分布式特征缓存 + 异常事件流
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379/0
REDIS_KEY_PREFIX=ai_pricing
REDIS_STREAM_NAME=pricing.anomaly.events

# 推理解耦(可选): API网关转发到独立模型推理服务
INFERENCE_SERVICE_URL=http://127.0.0.1:9001
INFERENCE_SERVICE_TIMEOUT_SECONDS=3
```

### 10. Redis 一键验收

执行以下命令即可完成 Redis 全链路验收（PING、Stream 连通、FeatureCache 命中、OOD 事件推送）：

```bash
python scripts/redis_e2e_check.py
```

可选：临时覆盖 Redis 地址

```bash
python scripts/redis_e2e_check.py --redis-url redis://127.0.0.1:6379/0
```

脚本会输出结构化 JSON，并在末尾打印：

- `REDIS_E2E_RESULT=PASS`（退出码 0）
- `REDIS_E2E_RESULT=FAIL`（退出码 1）

PowerShell 快捷命令（当前终端会话生效）：

```powershell
. .\scripts\dev_commands.ps1
check-redis-e2e
```

使用短别名：

```powershell
cr2e
```

可选：覆盖 Redis 地址

```powershell
check-redis-e2e -RedisUrl redis://127.0.0.1:6379/0
```

## 九、真实数据接入

所有真实数据接入点已在 `data/data_loader.py` 中用 `# TODO: 接入真实数据` 标注。
当前使用模拟数据跑通全流程,替换为真实数据源即可投入使用。
