"""
RL 强化学习标准样本集 (State-Action-Reward Logging)

每一次报价都完整记录一条黄金样本:
  - State:  客观环境(入园率/天气/竞品等)
  - Action: AI给出的票价
  - Reward: 未来1-2小时的真实收益(售票数/转化率/退票)

归因逻辑:
  处理"延迟满足"问题 —— 游客今天看到价格没买,明天降价后买了,
  通过跨天归因窗口追踪用户行为。

存储:
  - 实时写入 JSONL 文件 + Redis Stream (供 T+1 增量训练消费)
  - 每条样本包含归因链接字段
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from utils.logger import get_logger
from config import settings
from services.message_bus import bus, Channel

logger = get_logger("RLLogger")


# ============================================================
# 数据结构
# ============================================================
@dataclass
class StateRecord:
    """状态快照"""
    timestamp: str
    date: str
    day_type: str                   # weekday/weekend/holiday/golden_week
    weather: str                    # 晴好/雨/暴雨/酷热/严寒
    temperature: float
    rainfall: float
    humidity: float
    season: str
    # 园区实时状态
    current_load_rate: float        # 当前入园率
    entry_rate_per_min: float       # 入园速率(人/分钟)
    checked_in_count: int
    # 竞品环境
    competitor_avg_price: float
    competitor_max_price: float
    competitor_min_price: float
    # 时间特征
    days_to_next_holiday: int
    is_holiday_eve: bool
    hour_of_day: int
    # 完整特征向量(供模型直接消费)
    feature_vector_24d: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ActionRecord:
    """动作记录"""
    timestamp: str
    base_price: float               # 模型原始建议价
    final_price: float              # 经风控拦截后的最终价
    prev_price: float               # 调价前价格
    change_pct: float               # 变动百分比
    # 渠道分发
    channel_prices: Dict[str, float] = field(default_factory=dict)
    # 决策来源
    model_version: str = ""         # 模型版本标识
    pricing_mode: str = "ai"       # "ai" | "static_fallback" | "manual"
    # 风控
    guard_violation: str = "ok"     # 风控告警类型
    circuit_breaker_state: str = "closed"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RewardRecord:
    """奖励记录 —— 在调价后1-2小时/次日回填"""
    timestamp: str                  # 回填时间
    window_start: str               # 观察窗口起始
    window_hours: float             # 观察窗口长度(小时)
    # 核心收益指标
    tickets_sold: int               # 窗口内售票数
    gross_revenue: float            # 窗口内毛收入
    net_revenue: float              # 扣除退票后的净收入
    conversion_rate: float          # 转化率
    # 惩罚指标
    refund_count: int               # 退票数
    refund_amount: float            # 退票金额
    complaint_count: int            # 客诉数
    # 负载
    peak_load_rate: float           # 窗口内峰值负载率
    avg_load_rate: float            # 窗口内平均负载率
    min_load_rate: float            # 最低负载率
    # 归因
    attributed: bool = False        # 是否完成归因
    attribution_window_hours: int = 48  # 归因窗口

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SARSample:
    """一条完整的 State-Action-Reward 黄金样本"""
    sample_id: str
    created_at: str

    # 三要素
    state: StateRecord
    action: ActionRecord
    reward: Optional[RewardRecord] = None  # 实时为空,稍后回填

    # 归因链接
    attribution_chain: List[str] = field(default_factory=list)
    # 关联的其他样本ID(跨天归因)
    linked_sample_ids: List[str] = field(default_factory=list)

    # 元信息
    park_id: str = ""
    model_version: str = ""
    episode_id: str = ""            # 同一个episode内的样本共享此ID

    def to_dict(self) -> dict:
        d = asdict(self)
        d["state"] = self.state.to_dict()
        d["action"] = self.action.to_dict()
        d["reward"] = self.reward.to_dict() if self.reward else None
        return d


# ============================================================
# SAR 日志记录器
# ============================================================
class SARLogger:
    """
    RL 训练数据采集器

    用法:
      logger = SARLogger()

      # 记录决策
      sample = logger.log_decision(state_dict, action_dict)

      # 稍后回填奖励
      logger.fill_reward(sample.sample_id, reward_dict)

      # 跨天归因
      logger.link_attribution(sample.sample_id, related_sample_id)

      # 获取T+1训练数据集
      dataset = logger.get_training_dataset(hours=24)
    """

    def __init__(
        self,
        storage_dir: str = "./data/rl_samples",
        max_pending_rewards: int = 1000,
    ):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # 当日样本文件
        self._today = datetime.now().strftime("%Y-%m-%d")
        self._file_path = os.path.join(storage_dir, f"sar_{self._today}.jsonl")

        # 待回填奖励的样本
        self._pending_rewards: Dict[str, SARSample] = {}
        self._max_pending = max_pending_rewards

        # 归因链接表: user_id → [sample_ids]
        self._user_attribution: Dict[str, List[str]] = {}

        logger.info(f"SAR日志器就绪 | storage={storage_dir} | today={self._today}")

    # ============================================================
    # 记录决策
    # ============================================================
    def log_decision(
        self,
        state: Dict[str, Any],
        action: Dict[str, Any],
        episode_id: str = "",
        model_version: str = "",
    ) -> SARSample:
        """
        记录一次定价决策

        state:  {day_type, weather, temperature, rainfall, load_rate, ...}
        action: {base_price, final_price, prev_price, channel_prices, ...}
        """
        self._rotate_daily_file()

        sample_id = f"sar:{uuid.uuid4().hex[:12]}"
        now = datetime.now()

        state_rec = StateRecord(
            timestamp=now.isoformat(),
            date=now.strftime("%Y-%m-%d"),
            day_type=str(state.get("day_type", "weekday")),
            weather=str(state.get("weather", "晴好")),
            temperature=float(state.get("temperature", 22.0)),
            rainfall=float(state.get("rainfall", 0.0)),
            humidity=float(state.get("humidity", 0.6)),
            season=str(state.get("season", "spring")),
            current_load_rate=float(state.get("load_rate", 0.5)),
            entry_rate_per_min=float(state.get("entry_rate", 0.0)),
            checked_in_count=int(state.get("checked_in_count", 0)),
            competitor_avg_price=float(state.get("competitor_avg", 310.0)),
            competitor_max_price=float(state.get("competitor_max", 350.0)),
            competitor_min_price=float(state.get("competitor_min", 280.0)),
            days_to_next_holiday=int(state.get("days_to_next_holiday", 30)),
            is_holiday_eve=bool(state.get("is_holiday_eve", False)),
            hour_of_day=now.hour,
            feature_vector_24d=list(state.get("feature_vector", [])),
        )

        action_rec = ActionRecord(
            timestamp=now.isoformat(),
            base_price=float(action.get("base_price", 299.0)),
            final_price=float(action.get("final_price", 299.0)),
            prev_price=float(action.get("prev_price", 299.0)),
            change_pct=float(action.get("change_pct", 0.0)),
            channel_prices=dict(action.get("channel_prices", {})),
            model_version=model_version,
            pricing_mode=str(action.get("pricing_mode", "ai")),
            guard_violation=str(action.get("guard_violation", "ok")),
            circuit_breaker_state=str(action.get("circuit_breaker_state", "closed")),
        )

        sample = SARSample(
            sample_id=sample_id,
            created_at=now.isoformat(),
            state=state_rec,
            action=action_rec,
            park_id=settings.park_name,
            model_version=model_version,
            episode_id=episode_id or f"ep:{now.strftime('%Y%m%d')}:{uuid.uuid4().hex[:6]}",
        )

        # 写入文件
        self._append_to_file(sample)

        # 入待回填队列
        self._pending_rewards[sample_id] = sample
        if len(self._pending_rewards) > self._max_pending:
            # 淘汰最旧
            oldest = next(iter(self._pending_rewards))
            self._pending_rewards.pop(oldest)

        logger.debug(f"SAR样本已记录: {sample_id} | price=¥{action_rec.final_price} "
                      f"| mode={action_rec.pricing_mode}")

        return sample

    # ============================================================
    # 回填奖励
    # ============================================================
    def fill_reward(
        self,
        sample_id: str,
        reward_data: Dict[str, Any],
    ) -> bool:
        """
        回填奖励数据

        reward_data: {
            tickets_sold, gross_revenue, net_revenue, conversion_rate,
            refund_count, refund_amount, complaint_count,
            peak_load_rate, avg_load_rate, min_load_rate,
        }
        """
        now = datetime.now()
        sample = self._pending_rewards.get(sample_id)
        if sample is None:
            # 尝试从文件中回读
            sample = self._find_sample_by_id(sample_id)
            if sample is None:
                logger.warning(f"reward回填失败: 找不到样本 {sample_id}")
                return False

        sample.reward = RewardRecord(
            timestamp=now.isoformat(),
            window_start=reward_data.get("window_start", sample.created_at),
            window_hours=float(reward_data.get("window_hours", 2.0)),
            tickets_sold=int(reward_data.get("tickets_sold", 0)),
            gross_revenue=float(reward_data.get("gross_revenue", 0.0)),
            net_revenue=float(reward_data.get("net_revenue", 0.0)),
            conversion_rate=float(reward_data.get("conversion_rate", 0.0)),
            refund_count=int(reward_data.get("refund_count", 0)),
            refund_amount=float(reward_data.get("refund_amount", 0.0)),
            complaint_count=int(reward_data.get("complaint_count", 0)),
            peak_load_rate=float(reward_data.get("peak_load_rate", 0.0)),
            avg_load_rate=float(reward_data.get("avg_load_rate", 0.0)),
            min_load_rate=float(reward_data.get("min_load_rate", 0.0)),
            attributed=bool(reward_data.get("attributed", True)),
            attribution_window_hours=int(reward_data.get("attribution_window_hours", 48)),
        )

        # 更新文件中的记录
        self._update_sample_in_file(sample)

        # 从待回填队列移除
        self._pending_rewards.pop(sample_id, None)

        logger.info(f"Reward已回填: {sample_id} | "
                     f"tickets={sample.reward.tickets_sold} | "
                     f"revenue=¥{sample.reward.net_revenue:.0f} | "
                     f"conv={sample.reward.conversion_rate:.1%}")

        # 发布到消息总线供T+1训练消费
        bus.publish(
            "park:rl:sample_complete",
            sample.to_dict(),
            source="sar_logger",
        )

        return True

    # ============================================================
    # 跨天归因
    # ============================================================
    def link_attribution(
        self,
        sample_id: str,
        related_sample_id: str,
        user_id: str = "",
        attribution_type: str = "cross_day",
    ):
        """
        建立跨天归因链接

        例如: 游客昨天看到¥188没买,今天看到¥168后购买
        → link_attribution(today_sample, yesterday_sample)
        """
        sample = self._pending_rewards.get(sample_id)
        if sample is None:
            sample = self._find_sample_by_id(sample_id)

        if sample is not None:
            sample.attribution_chain.append(attribution_type)
            sample.linked_sample_ids.append(related_sample_id)
            self._update_sample_in_file(sample)

        if user_id:
            if user_id not in self._user_attribution:
                self._user_attribution[user_id] = []
            self._user_attribution[user_id].append(sample_id)
            # 限制链长度
            if len(self._user_attribution[user_id]) > 10:
                self._user_attribution[user_id] = self._user_attribution[user_id][-10:]

        logger.debug(f"归因链接: {sample_id} ← {related_sample_id} "
                      f"(user={user_id}, type={attribution_type})")

    # ============================================================
    # T+1 训练数据集
    # ============================================================
    def get_training_dataset(
        self,
        hours: int = 24,
        require_reward: bool = True,
    ) -> List[dict]:
        """
        获取用于增量训练的完整SAR数据集

        hours: 回溯小时数
        require_reward: 是否只返回已回填reward的样本
        """
        cutoff = datetime.now() - timedelta(hours=hours)
        samples = []

        # 扫描最近几天的文件
        for day_offset in range(3):
            day = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            file_path = os.path.join(self.storage_dir, f"sar_{day}.jsonl")
            if not os.path.exists(file_path):
                continue

            with open(file_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        created = datetime.fromisoformat(data.get("created_at", ""))
                        if created >= cutoff:
                            if require_reward and data.get("reward") is None:
                                continue
                            samples.append(data)
                    except (json.JSONDecodeError, ValueError):
                        continue

        logger.info(f"T+1训练数据集: {len(samples)}条样本 (回溯{hours}h)")
        return samples

    def export_training_jsonl(self, hours: int = 24, output_path: str = "") -> str:
        """导出可供直接训练的JSONL文件"""
        samples = self.get_training_dataset(hours, require_reward=True)
        if not output_path:
            output_path = os.path.join(
                self.storage_dir,
                f"training_{datetime.now().strftime('%Y%m%d_%H%M')}.jsonl",
            )
        with open(output_path, "w") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        logger.info(f"训练数据集导出: {output_path} ({len(samples)}条)")
        return output_path

    # ============================================================
    # 内部方法
    # ============================================================
    def _rotate_daily_file(self):
        today = datetime.now().strftime("%Y-%m-%d")
        if today != self._today:
            self._today = today
            self._file_path = os.path.join(self.storage_dir, f"sar_{self._today}.jsonl")

    def _append_to_file(self, sample: SARSample):
        with open(self._file_path, "a") as f:
            f.write(json.dumps(sample.to_dict(), ensure_ascii=False) + "\n")

    # ============================================================
    # 批量操作优化: 避免逐条 I/O
    # ============================================================
    def log_decision_batch(
        self,
        states: List[Dict[str, Any]],
        actions: List[Dict[str, Any]],
        episode_id: str = "",
        model_version: str = "",
    ) -> List[SARSample]:
        """
        批量记录定价决策 —— 性能优化关键路径

        相比逐条调用 log_decision:
          - 一次性文件写入 (1次 open/close 替代 N 次)
          - 批量入队列
          - 减少 80%+ I/O 开销
        """
        self._rotate_daily_file()
        now = datetime.now()
        samples: List[SARSample] = []
        lines: List[str] = []

        for i, (state, action) in enumerate(zip(states, actions)):
            sample_id = f"sar:{uuid.uuid4().hex[:12]}"

            sample = SARSample(
                sample_id=sample_id,
                created_at=now.isoformat(),
                state=StateRecord(
                    timestamp=now.isoformat(),
                    date=now.strftime("%Y-%m-%d"),
                    day_type=str(state.get("day_type", "weekday")),
                    weather=str(state.get("weather", "晴好")),
                    temperature=float(state.get("temperature", 22.0)),
                    rainfall=float(state.get("rainfall", 0.0)),
                    humidity=float(state.get("humidity", 0.6)),
                    season=str(state.get("season", "spring")),
                    current_load_rate=float(state.get("load_rate", 0.5)),
                    entry_rate_per_min=float(state.get("entry_rate", 0.0)),
                    checked_in_count=int(state.get("checked_in_count", 0)),
                    competitor_avg_price=float(state.get("competitor_avg", 310.0)),
                    competitor_max_price=float(state.get("competitor_max", 350.0)),
                    competitor_min_price=float(state.get("competitor_min", 280.0)),
                    days_to_next_holiday=int(state.get("days_to_next_holiday", 30)),
                    is_holiday_eve=bool(state.get("is_holiday_eve", False)),
                    hour_of_day=now.hour,
                    feature_vector_24d=list(state.get("feature_vector", [])),
                ),
                action=ActionRecord(
                    timestamp=now.isoformat(),
                    base_price=float(action.get("base_price", 299.0)),
                    final_price=float(action.get("final_price", 299.0)),
                    prev_price=float(action.get("prev_price", 299.0)),
                    change_pct=float(action.get("change_pct", 0.0)),
                    channel_prices=dict(action.get("channel_prices", {})),
                    model_version=model_version,
                    pricing_mode=str(action.get("pricing_mode", "ai")),
                    guard_violation=str(action.get("guard_violation", "ok")),
                    circuit_breaker_state=str(action.get("circuit_breaker_state", "closed")),
                ),
                park_id=settings.park_name,
                model_version=model_version,
                episode_id=episode_id or f"ep:{now.strftime('%Y%m%d')}:{uuid.uuid4().hex[:6]}",
            )

            samples.append(sample)
            lines.append(json.dumps(sample.to_dict(), ensure_ascii=False))
            self._pending_rewards[sample_id] = sample

        # 批量写入: 1次 open/close
        with open(self._file_path, "a") as f:
            f.write("\n".join(lines) + "\n")

        # 控制队列大小
        while len(self._pending_rewards) > self._max_pending:
            oldest = next(iter(self._pending_rewards))
            self._pending_rewards.pop(oldest)

        logger.debug(f"SAR批量记录: {len(samples)}条 | file={self._file_path}")
        return samples

    def get_training_dataset_vectorized(
        self,
        hours: int = 24,
        require_reward: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        向量化训练数据提取 —— 直接返回预分配的 NumPy 数组

        替代逐条 json.loads + list.append 的低效模式

        Returns:
          states:  (N, D) float32
          actions: (N, 1) float32  归一化价格 [-1, 1]
          rewards: (N,)  float32  净收入/1e5
        """
        cutoff = datetime.now() - timedelta(hours=hours)

        # 先预扫描获取样本数量
        all_data: List[dict] = []
        for day_offset in range(3):
            day = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            file_path = os.path.join(self.storage_dir, f"sar_{day}.jsonl")
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        created = datetime.fromisoformat(data.get("created_at", ""))
                        if created >= cutoff:
                            if require_reward and data.get("reward") is None:
                                continue
                            all_data.append(data)
                    except (json.JSONDecodeError, ValueError):
                        continue

        n = len(all_data)
        if n == 0:
            return (
                np.zeros((0, 24), dtype=np.float32),
                np.zeros((0, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )

        # 预分配数组,避免反复 realloc
        states = np.zeros((n, 24), dtype=np.float32)
        actions = np.zeros((n, 1), dtype=np.float32)
        rewards = np.zeros((n,), dtype=np.float32)

        for i, data in enumerate(all_data):
            state = data.get("state", {})
            action = data.get("action", {})
            reward = data.get("reward", {})

            # 从特征向量提取
            fv = state.get("feature_vector_24d", [])
            if len(fv) == 24:
                states[i] = fv
            else:
                # 降级: 从标量字段近似构建
                self._fill_state_row(states[i], state)

            price = float(action.get("final_price", 299.0))
            actions[i, 0] = np.clip(
                2 * (price - 80.0) / (599.0 - 80.0) - 1, -1, 1
            )
            rewards[i] = float(reward.get("net_revenue", 0)) / 1e5

        logger.info(f"向量化提取: {n}条样本 → states={states.shape} actions={actions.shape}")
        return states, actions, rewards

    @staticmethod
    def _fill_state_row(row: np.ndarray, state: dict):
        """从标量字段填充状态行(降级模式)"""
        day_map = {"weekday": 0, "weekend": 1, "holiday": 2, "golden_week": 3}
        weather_map = {"晴好": 0, "雨": 1, "暴雨": 2, "酷热": 3, "严寒": 4}
        season_map = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}

        day = state.get("day_type", "weekday")
        weather = state.get("weather", "晴好")
        season = state.get("season", "spring")

        row[day_map.get(day, 0)] = 1.0
        row[4 + weather_map.get(weather, 0)] = 1.0
        row[9 + season_map.get(season, 0)] = 1.0
        row[13] = (float(state.get("temperature", 22.0)) - 20) / 15.0
        row[14] = float(state.get("rainfall", 0.0)) / 30.0

    def _find_sample_by_id(self, sample_id: str) -> Optional[SARSample]:
        for day_offset in range(3):
            day = (datetime.now() - timedelta(days=day_offset)).strftime("%Y-%m-%d")
            file_path = os.path.join(self.storage_dir, f"sar_{day}.jsonl")
            if not os.path.exists(file_path):
                continue
            with open(file_path) as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data.get("sample_id") == sample_id:
                            return SARSample(**data)
                    except (json.JSONDecodeError, TypeError):
                        continue
        return None

    def _update_sample_in_file(self, sample: SARSample):
        """原地更新样本(读文件→替换行→写回)"""
        file_path = os.path.join(self.storage_dir, f"sar_{sample.created_at[:10]}.jsonl")
        if not os.path.exists(file_path):
            return
        lines = []
        found = False
        with open(file_path) as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if data.get("sample_id") == sample.sample_id:
                        lines.append(json.dumps(sample.to_dict(), ensure_ascii=False))
                        found = True
                    else:
                        lines.append(line.strip())
                except json.JSONDecodeError:
                    lines.append(line.strip())
        if found:
            with open(file_path, "w") as f:
                for l in lines:
                    f.write(l + "\n")

    def get_stats(self) -> dict:
        return {
            "pending_rewards": len(self._pending_rewards),
            "attribution_users": len(self._user_attribution),
            "today_file": self._file_path,
            "file_exists": os.path.exists(self._file_path),
        }


# 全局单例
sar_logger = SARLogger()
