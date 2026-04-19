"""
强化学习定价模块 (简化版,无需 stable-baselines3)

思路:
  状态 s = (day_type, weather, competitor_price, last_load_rate)
  动作 a = 价格档位(离散化为若干档位)
  回报 r = 门票收入 + α·二消收入 - β·(|load - 0.75|)²   (兼顾收入与客流均衡)

为保证无深度学习依赖也能跑通,这里实现一个 Q-learning 定价器。
真实项目可替换为 PPO/DQN (stable-baselines3)。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from collections import defaultdict

from utils.logger import get_logger
from config import settings

logger = get_logger("RLPricer")


class RLPricer:
    """
    Q-Learning 动态定价智能体
    离散价格动作 × 离散状态空间
    """

    DAY_TYPES = ["weekday", "weekend", "holiday", "golden_week"]
    WEATHER_TYPES = ["晴好", "雨", "暴雨", "酷热", "严寒"]

    def __init__(
        self,
        n_price_levels: int = 11,
        alpha: float = 0.45,    # 二消权重
        beta: float = 80.0,     # 客流均衡惩罚权重
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 0.2,
    ):
        p = settings.pricing
        self.price_grid = np.linspace(p.min_price, p.max_price, n_price_levels).round(0)
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        # Q[state_key][action_idx] = value
        self.Q: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(n_price_levels)
        )
        self.n_actions = n_price_levels

    # ---------- 状态离散化 ----------
    @staticmethod
    def _state_key(day_type: str, weather: str, competitor_avg: float) -> str:
        # 竞品价分3档
        if competitor_avg < 280:
            comp_level = "low"
        elif competitor_avg < 330:
            comp_level = "mid"
        else:
            comp_level = "high"
        return f"{day_type}|{weather}|{comp_level}"

    # ---------- 回报函数 ----------
    def _reward(self, price: float, visitors: float) -> Tuple[float, dict]:
        """
        r = 门票收入 + α·二消收入 - β·(负载率偏离最优)²
        """
        ticket_rev = price * visitors
        secondary_rev = visitors * settings.secondary_consumption_ratio * 130  # 人均二消≈130
        load_rate = visitors / settings.park_capacity
        load_penalty = self.beta * (load_rate - settings.optimal_load) ** 2 * 1000
        reward = ticket_rev + self.alpha * secondary_rev - load_penalty
        return reward, {
            "ticket_revenue": ticket_rev,
            "secondary_revenue": secondary_rev,
            "load_rate": load_rate,
            "load_penalty": load_penalty,
        }

    # ---------- 训练(离线历史数据) ----------
    def train(self, history: pd.DataFrame, epochs: int = 3) -> dict:
        """
        用历史数据做 off-policy 学习
        history 必须包含: date, price, visitors, day_type, weather,
                         revenue_ticket, (可选) competitor_avg_price
        """
        logger.info(f"开始RL定价训练 | epochs={epochs}")
        df = history.copy()
        if "competitor_avg_price" not in df.columns:
            df["competitor_avg_price"] = 310.0  # 默认竞品均价

        total_samples = 0
        for epoch in range(epochs):
            for i in range(len(df) - 1):
                row = df.iloc[i]
                next_row = df.iloc[i + 1]
                s = self._state_key(row["day_type"], row["weather"],
                                    row["competitor_avg_price"])
                # 用最接近的价格档位作为动作
                a = int(np.argmin(np.abs(self.price_grid - row["price"])))
                r, _ = self._reward(row["price"], row["visitors"])
                s_next = self._state_key(next_row["day_type"], next_row["weather"],
                                         next_row["competitor_avg_price"])

                # Q-learning 更新
                td_target = r + self.gamma * self.Q[s_next].max()
                td_error = td_target - self.Q[s][a]
                self.Q[s][a] += self.lr * td_error
                total_samples += 1

        logger.info(f"训练完成 | 状态数={len(self.Q)} | 样本数={total_samples}")
        return {"n_states": len(self.Q), "n_samples": total_samples}

    # ---------- 推理(给定状态,返回最优价格) ----------
    def recommend_price(
        self,
        day_type: str,
        weather: str,
        competitor_avg: float,
        explore: bool = False,
    ) -> dict:
        s = self._state_key(day_type, weather, competitor_avg)
        q_values = self.Q[s]

        # 若该状态从未见过(全0) → 用业务规则兜底价格
        if np.all(q_values == 0):
            # 粗略映射到业务规则价格
            base = settings.pricing.base_price
            markup = {
                "golden_week": settings.holiday.golden_week_markup,
                "holiday": settings.holiday.holiday_markup,
                "weekend": settings.holiday.weekend_markup,
                "weekday": settings.holiday.weekday_discount,
            }.get(day_type, 1.0)
            fallback_price = base * markup
            action = int(np.argmin(np.abs(self.price_grid - fallback_price)))
            return {
                "state": s,
                "recommended_price": float(self.price_grid[action]),
                "q_value": 0.0,
                "q_distribution": {
                    float(self.price_grid[i]): 0.0 for i in range(self.n_actions)
                },
                "fallback": True,
            }

        if explore and np.random.rand() < self.epsilon:
            action = int(np.random.randint(self.n_actions))
        else:
            action = int(np.argmax(q_values))
        price = float(self.price_grid[action])
        return {
            "state": s,
            "recommended_price": price,
            "q_value": float(q_values[action]),
            "q_distribution": {
                float(self.price_grid[i]): float(q_values[i])
                for i in range(self.n_actions)
            },
            "fallback": False,
        }
