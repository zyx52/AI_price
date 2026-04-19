"""
PPO 版动态定价智能体 (深度强化学习升级版)

相比 Q-Learning 版本的优势:
  ✓ 连续状态空间  —— 不再依赖离散化(天气/竞品/负载率可任意取值)
  ✓ 神经网络策略  —— 泛化能力强,少见状态也能给出合理决策
  ✓ 策略梯度优化  —— 更稳定、收敛更快
  ✓ 支持多维特征  —— 同时学习 10+ 维特征对定价的影响

架构:
  [Actor 策略网络] — 输入状态 → 输出价格动作概率分布
  [Critic 价值网络] — 输入状态 → 输出状态价值估计
  Actor-Critic 联合更新,用 PPO 裁剪的目标函数防止策略更新过激

使用:
  需要安装: pip install stable-baselines3 gymnasium
  若未安装,会自动降级到 rl_pricer.py 的 Q-Learning 版本。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Any, cast
from dataclasses import dataclass

from utils.logger import get_logger
from config import settings

logger = get_logger("PPOPricer")


# 选择性导入
try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]
    from stable_baselines3 import PPO  # type: ignore[import-not-found]
    from stable_baselines3.common.env_util import make_vec_env  # type: ignore[import-not-found]
    from stable_baselines3.common.callbacks import BaseCallback  # type: ignore[import-not-found]
    _HAS_SB3 = True
except ImportError:
    gym = cast(Any, None)
    spaces = cast(Any, None)
    PPO = cast(Any, None)
    make_vec_env = cast(Any, None)
    BaseCallback = cast(Any, None)
    _HAS_SB3 = False
    logger.warning("stable-baselines3 未安装,PPOPricer不可用,请使用 rl_pricer.RLPricer")


# ============================================================
# 乐园定价仿真环境 (Gymnasium接口)
# ============================================================
if _HAS_SB3:
    GymEnvBase = cast(type[Any], gym.Env)

    class ParkPricingEnv(GymEnvBase):
        """
        乐园定价强化学习环境

        状态空间 (连续, 10维):
          [日期类型one-hot(4), 温度归一化, 降水归一化,
           竞品价归一化, 上日负载率, 季节one-hot(4)]

        动作空间 (连续, 1维):
          [-1, 1] 映射到 [min_price, max_price]

        奖励:
          r = 门票收入 + α·二消 - β·(负载率-最优)² - γ·价格剧烈变动惩罚
        """
        metadata = {"render_modes": []}

        def __init__(self, history_df: pd.DataFrame, demand_forecaster=None):
            super().__init__()
            self.history = history_df.reset_index(drop=True).copy()
            self.demand_forecaster = demand_forecaster

            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            # 状态: 4(day_type) + 1(temp) + 1(rain) + 1(comp) + 1(last_load) + 4(season) = 12
            self.observation_space = spaces.Box(
                low=-3.0, high=3.0, shape=(12,), dtype=np.float32
            )

            self.min_price = settings.pricing.min_price
            self.max_price = settings.pricing.max_price
            self.base_price = settings.pricing.base_price
            self.optimal_load = settings.optimal_load
            self.capacity = settings.park_capacity
            self.alpha = 0.45
            self.beta = 80.0
            self.gamma_smooth = 0.15  # 价格剧烈变动惩罚

            self.current_idx = 0
            self.last_price = self.base_price
            self.last_load = 0.5

        def _build_obs(self, row) -> np.ndarray:
            dt_map = {"weekday": 0, "weekend": 1, "holiday": 2, "golden_week": 3}
            sn_map = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
            day_oh = np.zeros(4); day_oh[dt_map.get(row["day_type"], 0)] = 1
            season_oh = np.zeros(4); season_oh[sn_map.get(row["season"], 0)] = 1
            return np.concatenate([
                day_oh,
                [(row["temperature"] - 20) / 15.0],
                [row["rainfall"] / 30.0],
                [0.0],  # 竞品价(训练时默认,真实用时传入)
                [self.last_load * 2 - 1],
                season_oh,
            ]).astype(np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_idx = np.random.randint(0, len(self.history) - 50)
            self.last_price = self.base_price
            self.last_load = 0.5
            row = self.history.iloc[self.current_idx]
            return self._build_obs(row), {}

        def _action_to_price(self, action: np.ndarray) -> float:
            # [-1,1] → [min_price, max_price]
            a = float(np.clip(action[0], -1, 1))
            price = self.min_price + (a + 1) / 2 * (self.max_price - self.min_price)
            return round(price / 5) * 5

        def _simulate_visitors(self, row, price: float) -> float:
            """用需求模型(或简单弹性)估算该价格下的客流"""
            if self.demand_forecaster is not None and self.demand_forecaster.is_trained:
                d = pd.to_datetime(row["date"])
                f = {
                    "price": price,
                    "temperature": row["temperature"],
                    "rainfall": row["rainfall"],
                    "is_holiday": int(row["is_holiday"]),
                    "is_weekend": int(row["is_weekend"]),
                    "day_of_week": int(d.dayofweek),
                    "month": int(d.month),
                    "season_id": {"spring":0,"summer":1,"autumn":2,"winter":3}[row["season"]],
                }
                return self.demand_forecaster.predict(f)
            # fallback: 基于价格弹性的简单模型
            base_visitors = row["visitors"]
            elasticity = (row["price"] / price) ** 0.8
            return float(base_visitors * elasticity)

        def step(self, action: np.ndarray):
            row = self.history.iloc[self.current_idx]
            price = self._action_to_price(action)
            visitors = self._simulate_visitors(row, price)
            load_rate = visitors / self.capacity

            # 奖励三要素
            ticket = price * visitors
            secondary = visitors * settings.secondary_consumption_ratio * 130
            load_pen = self.beta * (load_rate - self.optimal_load) ** 2 * 1000
            smooth_pen = self.gamma_smooth * abs(price - self.last_price) * 10
            reward = ticket + self.alpha * secondary - load_pen - smooth_pen

            # 归一化奖励
            reward = float(reward / 1e5)

            self.last_price = price
            self.last_load = load_rate
            self.current_idx += 1
            done = self.current_idx >= len(self.history) - 1
            truncated = False

            next_row = self.history.iloc[self.current_idx] if not done else row
            obs = self._build_obs(next_row)
            info = {
                "price": price, "visitors": visitors, "load_rate": load_rate,
                "ticket_revenue": ticket, "total_reward_components": {
                    "ticket": ticket, "secondary": secondary,
                    "load_penalty": load_pen, "smooth_penalty": smooth_pen,
                }
            }
            return obs, reward, done, truncated, info


# ============================================================
# PPO Pricer
# ============================================================
class PPOPricer:
    """PPO 动态定价智能体"""

    def __init__(self, demand_forecaster=None, total_timesteps: int = 50_000):
        if not _HAS_SB3:
            raise ImportError(
                "PPOPricer 需要 stable-baselines3 和 gymnasium。\n"
                "请运行: pip install stable-baselines3 gymnasium\n"
                "或降级使用 rl_pricer.RLPricer(Q-learning版本)。"
            )
        self.demand_forecaster = demand_forecaster
        self.total_timesteps = total_timesteps
        self.model: Optional[Any] = None
        self.env = None
        self.is_trained = False

    def train(self, history: pd.DataFrame) -> dict:
        logger.info(f"开始PPO训练 | timesteps={self.total_timesteps:,}")
        self.env = ParkPricingEnv(history, demand_forecaster=self.demand_forecaster)

        model = PPO(
            "MlpPolicy", self.env,
            learning_rate=3e-4, n_steps=2048, batch_size=64,
            n_epochs=10, gamma=0.99, gae_lambda=0.95,
            clip_range=0.2, ent_coef=0.01, verbose=0,
            policy_kwargs=dict(net_arch=[128, 128]),
        )
        model.learn(total_timesteps=self.total_timesteps)
        self.model = model
        self.is_trained = True
        logger.info("PPO训练完成")
        return {"timesteps": self.total_timesteps}

    def recommend_price(
        self,
        day_type: str, weather: str,
        temperature: float, rainfall: float,
        competitor_avg: float, last_load: float = 0.5,
        season: str = "spring",
    ) -> dict:
        if not self.is_trained:
            raise RuntimeError("模型未训练")
        if self.model is None:
            raise RuntimeError("模型未加载或未训练")

        # 构造状态
        dt_map = {"weekday": 0, "weekend": 1, "holiday": 2, "golden_week": 3}
        sn_map = {"spring": 0, "summer": 1, "autumn": 2, "winter": 3}
        day_oh = np.zeros(4); day_oh[dt_map.get(day_type, 0)] = 1
        season_oh = np.zeros(4); season_oh[sn_map.get(season, 0)] = 1
        obs = np.concatenate([
            day_oh,
            [(temperature - 20) / 15.0],
            [rainfall / 30.0],
            [(competitor_avg - 310) / 50.0],
            [last_load * 2 - 1],
            season_oh,
        ]).astype(np.float32)

        action, _ = self.model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1, 1))
        price = settings.pricing.min_price + (a + 1) / 2 * (
            settings.pricing.max_price - settings.pricing.min_price
        )
        price = round(price / 5) * 5

        return {
            "recommended_price": float(price),
            "raw_action": a,
            "model": "PPO",
            "fallback": False,
        }

    def save(self, path: str):
        if self.model:
            self.model.save(path)
            logger.info(f"模型已保存: {path}")

    def load(self, path: str):
        if not _HAS_SB3:
            raise ImportError(
                "PPOPricer 需要 stable-baselines3 和 gymnasium。\n"
                "请运行: pip install stable-baselines3 gymnasium"
            )
        self.model = PPO.load(path)
        self.is_trained = True
        logger.info(f"模型已加载: {path}")
