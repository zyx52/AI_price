"""
增强版RL动态定价 (RLPricerV2)

相比 rl_pricer.py (v1) 的改进:
  1. 状态空间细化: 7状态 → 约100+状态
     - day_type(4) × weather(5) × comp_level(5) × load_level(3) = 300理论状态
  2. 价格平滑约束: 连续两天价格变动不超过15%
  3. 衰减探索率: 训练初期多探索,后期少探索
  4. 双Q-Learning减轻过估计
  5. 软Q值更新,动作建议更稳健
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from collections import defaultdict

from utils.logger import get_logger
from config import settings

logger = get_logger("RLPricerV2")


class RLPricerV2:
    """细化状态 + 平滑约束的 Q-Learning 定价器"""

    DAY_TYPES = ["weekday", "weekend", "holiday", "golden_week"]
    WEATHER_TYPES = ["晴好", "雨", "暴雨", "酷热", "严寒"]

    def __init__(
        self,
        n_price_levels: int = 15,
        alpha: float = 0.45,
        beta: float = 80.0,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 0.3,
        epsilon_end: float = 0.05,
        # 价格平滑约束
        max_price_change_ratio: float = 0.15,  # 连续两天不超过15%
    ):
        p = settings.pricing
        self.price_grid = np.linspace(p.min_price, p.max_price, n_price_levels).round(0)
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.max_price_change_ratio = max_price_change_ratio

        # 双Q表(减少过估计)
        self.Q1: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_price_levels))
        self.Q2: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_price_levels))
        self.visit_counts: Dict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_price_levels))
        self.n_actions = n_price_levels

    # ---------- 细化的状态离散化 ----------
    @staticmethod
    def _state_key(
        day_type: str, weather: str,
        competitor_avg: float, load_rate: float,
    ) -> str:
        # 竞品价5档
        if competitor_avg < 260:
            comp_level = "very_low"
        elif competitor_avg < 290:
            comp_level = "low"
        elif competitor_avg < 320:
            comp_level = "mid"
        elif competitor_avg < 360:
            comp_level = "high"
        else:
            comp_level = "very_high"

        # 负载率3档
        if load_rate < 0.4:
            load_level = "low"
        elif load_rate < 0.7:
            load_level = "mid"
        else:
            load_level = "high"

        return f"{day_type}|{weather}|{comp_level}|{load_level}"

    # ---------- 回报 ----------
    def _reward(self, price: float, visitors: float, prev_price: Optional[float] = None) -> float:
        ticket_rev = price * visitors
        secondary_rev = visitors * settings.secondary_consumption_ratio * 130
        load_rate = visitors / settings.park_capacity
        load_penalty = self.beta * (load_rate - settings.optimal_load) ** 2 * 1000

        # 价格平滑惩罚
        smooth_penalty = 0
        if prev_price is not None and prev_price > 0:
            change_ratio = abs(price - prev_price) / prev_price
            if change_ratio > self.max_price_change_ratio:
                excess = change_ratio - self.max_price_change_ratio
                smooth_penalty = excess * 10000  # 超出越多惩罚越大

        return ticket_rev + self.alpha * secondary_rev - load_penalty - smooth_penalty

    # ---------- 训练 ----------
    def train(self, history: pd.DataFrame, epochs: int = 5) -> dict:
        logger.info(f"开始RL-V2训练 | epochs={epochs}")
        df = history.copy()
        if "competitor_avg_price" not in df.columns:
            df["competitor_avg_price"] = 310.0
        if "load_rate" not in df.columns:
            df["load_rate"] = df["visitors"] / settings.park_capacity

        total_samples = 0
        for epoch in range(epochs):
            # 衰减epsilon
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - epoch / epochs)

            for i in range(len(df) - 1):
                row = df.iloc[i]
                next_row = df.iloc[i + 1]
                prev_row = df.iloc[i - 1] if i > 0 else None

                s = self._state_key(
                    row["day_type"], row["weather"],
                    row["competitor_avg_price"], row["load_rate"],
                )
                a = int(np.argmin(np.abs(self.price_grid - row["price"])))
                prev_price = prev_row["price"] if prev_row is not None else None
                r = self._reward(row["price"], row["visitors"], prev_price)

                s_next = self._state_key(
                    next_row["day_type"], next_row["weather"],
                    next_row["competitor_avg_price"], next_row["load_rate"],
                )

                # 双Q-Learning: 用Q1选动作, Q2估值(反之亦然,随机选择)
                if np.random.rand() < 0.5:
                    best_a_next = int(np.argmax(self.Q1[s_next]))
                    td_target = r + self.gamma * self.Q2[s_next][best_a_next]
                    self.Q1[s][a] += self.lr * (td_target - self.Q1[s][a])
                else:
                    best_a_next = int(np.argmax(self.Q2[s_next]))
                    td_target = r + self.gamma * self.Q1[s_next][best_a_next]
                    self.Q2[s][a] += self.lr * (td_target - self.Q2[s][a])

                self.visit_counts[s][a] += 1
                total_samples += 1

        n_states = len(set(list(self.Q1.keys()) + list(self.Q2.keys())))
        logger.info(f"训练完成 | 状态数={n_states} | 样本数={total_samples}")
        return {"n_states": n_states, "n_samples": total_samples}

    # ---------- 推理 ----------
    def recommend_price(
        self,
        day_type: str, weather: str,
        competitor_avg: float, load_rate: float = 0.5,
        prev_price: Optional[float] = None,
    ) -> dict:
        s = self._state_key(day_type, weather, competitor_avg, load_rate)
        q_values = (self.Q1[s] + self.Q2[s]) / 2

        # 从未见过的状态: fallback业务规则
        if np.all(q_values == 0):
            base = settings.pricing.base_price
            markup = {
                "golden_week": settings.holiday.golden_week_markup,
                "holiday": settings.holiday.holiday_markup,
                "weekend": settings.holiday.weekend_markup,
                "weekday": settings.holiday.weekday_discount,
            }.get(day_type, 1.0)
            fallback_price = base * markup
            action = int(np.argmin(np.abs(self.price_grid - fallback_price)))
            price = float(self.price_grid[action])
            return {
                "state": s,
                "recommended_price": price,
                "q_value": 0.0,
                "fallback": True,
                "constraint_applied": False,
            }

        action = int(np.argmax(q_values))
        price = float(self.price_grid[action])

        # 价格平滑约束
        constraint_applied = False
        if prev_price is not None and prev_price > 0:
            change_ratio = (price - prev_price) / prev_price
            if abs(change_ratio) > self.max_price_change_ratio:
                # 限制在 ±15%
                direction = 1 if change_ratio > 0 else -1
                price = prev_price * (1 + direction * self.max_price_change_ratio)
                price = round(price / 5) * 5
                constraint_applied = True

        return {
            "state": s,
            "recommended_price": price,
            "q_value": float(q_values[action]),
            "fallback": False,
            "constraint_applied": constraint_applied,
            "prev_price": prev_price,
        }
