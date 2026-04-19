"""
P1-01: 乐园RL仿真环境 (ParkEnv) + 基于仿真的RL训练器

旧版 RLPricerV2 的问题:
  - 直接在历史DataFrame上做 Q-Learning,智能体永远只能"看到"历史上实际用过的价格,
    对未见价格(比如¥450这种历史从没定过的档位)无法探索,
    Q值全靠在少数动作上反复更新,容易过估计。

本模块的解决方案:
  - 构造 ParkEnv: 将已训练好的需求预测模型(ensemble)包装成环境的dynamics
  - 智能体给出任意价格档位 → 环境用需求模型反馈"那个价格下的客流+收益"
  - 智能体在一个可控的"虚拟乐园"里反复试验,真正做到 on-policy 探索
  - 训练出来的策略泛化性远高于纯离线 Q-Learning

可选:
  - 支持 Q-Learning (轻量,无深度学习依赖)
  - 支持 PPO (通过 stable-baselines3,需要额外安装)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from collections import defaultdict

from utils.logger import get_logger
from config import settings

logger = get_logger("ParkEnv")


# ============================================================
# 仿真环境 ParkEnv
# ============================================================
class ParkEnv:
    """
    乐园定价仿真环境
    
    状态: (day_type, weather, competitor_level, load_level, prev_price)
    动作: 价格档位(从n_price_levels个候选中选一个)
    转移: 基于需求模型预测客流,计算奖励,随机采样下一日情境
    奖励: 门票收入 + α·二消 - β·负载偏离 - γ·价格剧变
    """

    def __init__(
        self,
        demand_forecaster,            # 已训练好的预测模型(必须有predict_batch)
        history_sample: pd.DataFrame, # 用于采样"场景" (day_type/weather/temp/rain等分布)
        n_price_levels: int = 15,
        alpha: float = 0.45,
        beta: float = 80.0,
        gamma_smooth: float = 0.15,
        max_change_ratio: float = 0.15,
        episode_length: int = 30,    # 一个episode模拟30天
        seed: int = 42,
    ):
        self.forecaster = demand_forecaster
        self.history = history_sample.reset_index(drop=True)
        self.n_price_levels = n_price_levels
        self.price_grid = np.linspace(
            settings.pricing.min_price, settings.pricing.max_price, n_price_levels
        ).round(0)
        self.alpha = alpha
        self.beta = beta
        self.gamma_smooth = gamma_smooth
        self.max_change_ratio = max_change_ratio
        self.episode_length = episode_length
        self.rng = np.random.default_rng(seed)

        self.current_step = 0
        self.current_scenario: Optional[dict] = None
        self.prev_price: Optional[float] = None
        self.episode_reward = 0.0

    # ---------- reset / step ----------
    def reset(self) -> dict:
        self.current_step = 0
        self.prev_price = settings.pricing.base_price
        self.episode_reward = 0.0
        self.current_scenario = self._sample_scenario()
        return self._observation()

    def step(self, action: int) -> Tuple[dict, float, bool, dict]:
        """
        action: 0 ~ n_price_levels-1
        返回 (next_obs, reward, done, info)
        """
        assert 0 <= action < self.n_price_levels, f"无效action: {action}"
        if self.current_scenario is None:
            raise RuntimeError("环境尚未 reset，无法 step")

        scenario = self.current_scenario
        price = float(self.price_grid[action])

        # ========= 用需求模型仿真当日客流 =========
        feature_row = self._scenario_to_features(scenario, price)
        visitors = self.forecaster.predict(
            feature_row, target_date=scenario["date"]
        )
        visitors = max(100.0, min(float(visitors), settings.park_capacity))

        # ========= 计算奖励 =========
        ticket_rev = price * visitors
        secondary_rev = visitors * settings.secondary_consumption_ratio * 130
        load_rate = visitors / settings.park_capacity
        load_pen = self.beta * (load_rate - settings.optimal_load) ** 2 * 1000
        smooth_pen = 0.0
        if self.prev_price is not None:
            change = abs(price - self.prev_price) / max(self.prev_price, 1)
            if change > self.max_change_ratio:
                smooth_pen = (change - self.max_change_ratio) * 10000

        raw_reward = ticket_rev + self.alpha * secondary_rev - load_pen - smooth_pen
        # 归一化到合理量级(便于学习)
        reward = raw_reward / 1e5

        self.prev_price = price
        self.episode_reward += reward
        self.current_step += 1
        done = self.current_step >= self.episode_length

        # 采样下一日场景
        if not done:
            self.current_scenario = self._sample_scenario()

        info = {
            "price": price,
            "visitors": visitors,
            "load_rate": load_rate,
            "ticket_revenue": ticket_rev,
            "secondary_revenue": secondary_rev,
            "load_penalty": load_pen,
            "smooth_penalty": smooth_pen,
            "raw_reward": raw_reward,
        }
        return self._observation(), reward, done, info

    # ---------- 场景采样 ----------
    def _sample_scenario(self) -> dict:
        """从历史分布随机采样一天的外部情境(保留真实相关性)"""
        row = self.history.iloc[self.rng.integers(0, len(self.history))]
        return {
            "date": str(row["date"]),
            "day_type": row["day_type"],
            "weather": row["weather"],
            "temperature": float(row["temperature"]),
            "rainfall": float(row["rainfall"]),
            "is_holiday": bool(row["is_holiday"]),
            "is_weekend": bool(row["is_weekend"]),
            "season": row["season"],
            "competitor_avg": float(self.rng.normal(310, 25)),
        }

    def _scenario_to_features(self, sc: dict, price: float) -> dict:
        """把场景+价格转成预测模型需要的feature_row"""
        d = pd.to_datetime(sc["date"])
        return {
            "price": price,
            "temperature": sc["temperature"],
            "rainfall": sc["rainfall"],
            "is_holiday": int(sc["is_holiday"]),
            "is_weekend": int(sc["is_weekend"]),
            "day_of_week": int(d.dayofweek),
            "month": int(d.month),
            "day_of_month": int(d.day),
            "season_id": {"spring":0,"summer":1,"autumn":2,"winter":3}.get(sc["season"], 0),
            "day_type_id": {"weekday":0,"weekend":1,"holiday":2,"golden_week":3}.get(sc["day_type"], 0),
            "weather_id": {"晴好":0,"雨":1,"暴雨":2,"酷热":3,"严寒":4}.get(sc["weather"], 0),
        }

    def _observation(self) -> dict:
        """当前观测(给智能体看的状态)"""
        sc = self.current_scenario
        if sc is None:
            raise RuntimeError("环境尚未 reset，无法获取 observation")

        return {
            "day_type": sc["day_type"],
            "weather": sc["weather"],
            "competitor_avg": sc["competitor_avg"],
            "load_rate": (self.rng.uniform(0.3, 0.8)
                          if self.prev_price is None else 0.5),
            "prev_price": self.prev_price or settings.pricing.base_price,
        }


# ============================================================
# 基于仿真环境的Q-Learning智能体
# (相比旧版RLPricerV2的区别: 在ParkEnv里探索,而不是在历史DataFrame上)
# ============================================================
class SimulatorBasedQLearning:
    """基于ParkEnv的 Q-Learning 智能体"""

    def __init__(
        self,
        env: ParkEnv,
        lr: float = 0.1,
        gamma: float = 0.9,
        epsilon_start: float = 0.4,
        epsilon_end: float = 0.05,
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.Q: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(env.n_price_levels)
        )

    # 状态离散化(细化版,与 RLPricerV2 对齐)
    @staticmethod
    def _state_key(obs: dict) -> str:
        comp = obs["competitor_avg"]
        if comp < 260: cl = "very_low"
        elif comp < 290: cl = "low"
        elif comp < 320: cl = "mid"
        elif comp < 360: cl = "high"
        else: cl = "very_high"
        load = obs["load_rate"]
        if load < 0.4: ll = "low"
        elif load < 0.7: ll = "mid"
        else: ll = "high"
        return f"{obs['day_type']}|{obs['weather']}|{cl}|{ll}"

    def train(self, n_episodes: int = 300) -> dict:
        """
        在仿真环境中训练
        默认: 300 episodes × 30 steps = 9000 次交互,智能体能覆盖大部分状态×动作组合
        """
        logger.info(f"基于仿真的Q-Learning训练 | episodes={n_episodes}")
        episode_rewards = []
        for ep in range(n_episodes):
            epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * (1 - ep / n_episodes)
            obs = self.env.reset()
            done = False
            ep_reward = 0.0

            while not done:
                s = self._state_key(obs)
                # ε-greedy 在仿真环境中自由探索!
                if np.random.rand() < epsilon:
                    a = np.random.randint(self.env.n_price_levels)
                else:
                    a = int(np.argmax(self.Q[s]))

                next_obs, r, done, info = self.env.step(a)
                s_next = self._state_key(next_obs)
                td_target = r + self.gamma * self.Q[s_next].max() * (not done)
                self.Q[s][a] += self.lr * (td_target - self.Q[s][a])

                obs = next_obs
                ep_reward += r

            episode_rewards.append(ep_reward)
            if (ep + 1) % 50 == 0:
                avg_r = np.mean(episode_rewards[-50:])
                logger.info(f"  episode {ep+1}/{n_episodes} | avg_reward(last50)={avg_r:.2f}")

        logger.info(f"训练完成 | 状态数={len(self.Q)} | 最终平均奖励={np.mean(episode_rewards[-50:]):.2f}")
        return {
            "n_states": len(self.Q),
            "n_episodes": n_episodes,
            "final_avg_reward": float(np.mean(episode_rewards[-50:])),
            "episode_rewards": episode_rewards,
        }

    def recommend_price(
        self,
        day_type: str, weather: str,
        competitor_avg: float, load_rate: float = 0.5,
        prev_price: Optional[float] = None,
    ) -> dict:
        obs = {
            "day_type": day_type, "weather": weather,
            "competitor_avg": competitor_avg, "load_rate": load_rate,
            "prev_price": prev_price or settings.pricing.base_price,
        }
        s = self._state_key(obs)
        q_values = self.Q[s]
        if np.all(q_values == 0):
            # Fallback业务规则
            base = settings.pricing.base_price
            markup = {
                "golden_week": settings.holiday.golden_week_markup,
                "holiday": settings.holiday.holiday_markup,
                "weekend": settings.holiday.weekend_markup,
                "weekday": settings.holiday.weekday_discount,
            }.get(day_type, 1.0)
            fallback = base * markup
            action = int(np.argmin(np.abs(self.env.price_grid - fallback)))
            return {
                "state": s,
                "recommended_price": float(self.env.price_grid[action]),
                "fallback": True,
            }

        action = int(np.argmax(q_values))
        price = float(self.env.price_grid[action])

        # 价格平滑约束
        constraint_applied = False
        if prev_price is not None:
            change = (price - prev_price) / max(prev_price, 1)
            if abs(change) > self.env.max_change_ratio:
                direction = 1 if change > 0 else -1
                price = prev_price * (1 + direction * self.env.max_change_ratio)
                price = round(price / 5) * 5
                constraint_applied = True

        return {
            "state": s,
            "recommended_price": price,
            "q_value": float(q_values[action]),
            "fallback": False,
            "constraint_applied": constraint_applied,
        }
