"""
需求2: RL 从"离散查表"升级到"连续表征"

升级点:
  1. 状态从字符串拼接("weekday|晴好|mid|low") 升级为密集向量
  2. 引入 GNN 节点嵌入(来自 park_attraction_graph),让 RL 理解项目间关联
     例如 A 项目修缮时,B/C 项目客流会被带动,定价弹性改变
    3. 使用 stable-baselines3 的 PPO (强制要求,不再允许离散Q-learning降级)

状态向量设计 (24维):
  [4]  day_type one-hot
  [5]  weather one-hot
  [4]  season one-hot
  [1]  temperature(标准化)
  [1]  rainfall(标准化)
  [1]  competitor_avg(标准化)
  [1]  last_load_rate
  [1]  prev_price(标准化)
  [6]  GNN嵌入(从景点图聚合的园区表征)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Optional, Any, cast

from utils.logger import get_logger
from config import settings

logger = get_logger("ContinuousRL")


try:
    import gymnasium as gym  # type: ignore[import-not-found]
    from gymnasium import spaces  # type: ignore[import-not-found]
    from stable_baselines3 import PPO  # type: ignore[import-not-found]
    _HAS_SB3 = True
except ImportError:
    gym = cast(Any, None)
    spaces = cast(Any, None)
    PPO = cast(Any, None)
    _HAS_SB3 = False


# ============================================================
# 连续状态编码器 (带 GNN 嵌入融合)
# ============================================================
class ContinuousStateEncoder:
    """
    把 (day_type, weather, comp_avg, load, prev_price) 编码为24维向量
    并融合 GNN 节点嵌入
    """

    DAY_TYPES = ["weekday", "weekend", "holiday", "golden_week"]
    WEATHERS = ["晴好", "雨", "暴雨", "酷热", "严寒"]
    SEASONS = ["spring", "summer", "autumn", "winter"]

    GNN_DIM = 6
    TOTAL_DIM = 4 + 5 + 4 + 5 + 6  # =24

    def __init__(self, attraction_graph=None):
        """
        attraction_graph: ParkAttractionGraph 实例(可选)
        """
        self.graph = attraction_graph
        self._gnn_embedding_cache: Optional[np.ndarray] = None   # 缺少实时排队数据时的默认嵌入

    def fit_gnn_embedding(self):
        """
        从景点图计算一次"园区全局嵌入",用于注入RL状态
        如果没有 GNN 模型,用简单的统计特征替代:
          [室内项目占比, 家庭友好占比, 平均容量, 项目种类数, ride项目占比, food项目占比]
        """
        if self.graph is None or not hasattr(self.graph, 'attractions'):
            self._gnn_embedding_cache = np.zeros(self.GNN_DIM)
            return

        atts = list(self.graph.attractions.values())
        if not atts:
            self._gnn_embedding_cache = np.zeros(self.GNN_DIM)
            return

        n = len(atts)
        indoor_ratio = sum(1 for a in atts if a.indoor) / n
        family_ratio = sum(1 for a in atts if a.family_friendly) / n
        avg_cap = np.mean([a.capacity_per_hour for a in atts]) / 2000.0
        category_diversity = len(set(a.category for a in atts)) / 5.0
        ride_ratio = sum(1 for a in atts if a.category == "ride") / n
        food_ratio = sum(1 for a in atts if a.category == "food") / n

        # 如果有训练过的 GNN,用其 node_embeddings 平均
        if hasattr(self.graph, "node_embeddings") and self.graph.node_embeddings is not None:
            mean_emb = self.graph.node_embeddings.mean(axis=0)
            # 截取/填充到 GNN_DIM
            if len(mean_emb) >= self.GNN_DIM:
                self._gnn_embedding_cache = mean_emb[:self.GNN_DIM].astype(np.float32)
            else:
                pad = np.zeros(self.GNN_DIM - len(mean_emb))
                self._gnn_embedding_cache = np.concatenate([mean_emb, pad]).astype(np.float32)
        else:
            self._gnn_embedding_cache = np.array([
                indoor_ratio, family_ratio, avg_cap, category_diversity, ride_ratio, food_ratio
            ], dtype=np.float32)
        logger.info(f"GNN嵌入已计算 | embedding={self._gnn_embedding_cache}")

    def encode(
        self, day_type: str, weather: str,
        season: str = "spring",
        temperature: float = 22.0, rainfall: float = 0.0,
        competitor_avg: float = 310.0, load_rate: float = 0.5,
        prev_price: float = 299.0,
        current_queues: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """编码为 24 维向量"""
        if self._gnn_embedding_cache is None:
            self.fit_gnn_embedding()

        day_oh = np.zeros(4)
        if day_type in self.DAY_TYPES:
            day_oh[self.DAY_TYPES.index(day_type)] = 1
        weather_oh = np.zeros(5)
        if weather in self.WEATHERS:
            weather_oh[self.WEATHERS.index(weather)] = 1
        season_oh = np.zeros(4)
        if season in self.SEASONS:
            season_oh[self.SEASONS.index(season)] = 1

        numeric = np.array([
            (temperature - 20) / 15.0,
            rainfall / 30.0,
            (competitor_avg - 310) / 50.0,
            load_rate * 2 - 1,
            (prev_price - settings.pricing.base_price) / 100.0,
        ])
        gnn_emb = self._gnn_embedding_cache
        if self.graph is not None and hasattr(self.graph, "queue_heat_embedding"):
            try:
                dynamic_emb = self.graph.queue_heat_embedding(
                    current_queues=current_queues,
                    weather=weather,
                    dim=self.GNN_DIM,
                )
                gnn_emb = dynamic_emb
            except Exception as e:
                logger.warning(f"queue_heat_embedding 计算失败,回退默认嵌入: {e}")
        if gnn_emb is None:
            gnn_emb = np.zeros(self.GNN_DIM, dtype=np.float32)

        return np.concatenate([
            day_oh, weather_oh, season_oh, numeric, gnn_emb
        ]).astype(np.float32)


# ============================================================
# PPO 定价智能体 (基于仿真环境)
# ============================================================
if _HAS_SB3:
    class _PPOParkEnv(gym.Env):  # type: ignore[name-defined]
        """专为 PPO 设计的 Gymnasium 环境"""
        metadata = {"render_modes": []}

        def __init__(self, forecaster, history_df, encoder: ContinuousStateEncoder,
                     n_price_levels: int = 15, episode_length: int = 30):
            super().__init__()
            self.forecaster = forecaster
            self.history = history_df.reset_index(drop=True)
            self.encoder = encoder
            self.n_price_levels = n_price_levels
            self.episode_length = episode_length

            self.price_grid = np.linspace(
                settings.pricing.min_price, settings.pricing.max_price, n_price_levels
            ).round(0)

            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
            self.observation_space = spaces.Box(
                low=-5.0, high=5.0,
                shape=(ContinuousStateEncoder.TOTAL_DIM,),
                dtype=np.float32,
            )
            self.current_step = 0
            self.prev_price = settings.pricing.base_price
            self.current_scenario: Optional[Dict[str, Any]] = None
            self.rng = np.random.default_rng(42)

        def _sample_scenario(self):
            row = self.history.iloc[self.rng.integers(0, len(self.history))]
            return {
                "date": str(row["date"]),
                "day_type": row["day_type"],
                "weather": row["weather"],
                "temperature": float(row["temperature"]),
                "rainfall": float(row["rainfall"]),
                "season": row.get("season", "spring"),
                "competitor_avg": float(self.rng.normal(310, 25)),
                "queue_snapshot": self._sample_queue_snapshot(str(row["weather"])),
            }

        def _sample_queue_snapshot(self, weather: str) -> Dict[str, float]:
            g = self.encoder.graph
            if g is None or not hasattr(g, "attractions"):
                return {}
            base = 22.0 if weather not in ("雨", "暴雨") else 35.0
            return {
                aid: float(max(5.0, self.rng.normal(base, 6.0)))
                for aid in g.attractions.keys()
            }

        def _obs(self, load_rate: float = 0.5):
            sc = self.current_scenario
            if sc is None:
                sc = self._sample_scenario()
                self.current_scenario = sc
            return self.encoder.encode(
                day_type=str(sc["day_type"]), weather=str(sc["weather"]),
                season=str(sc.get("season", "spring")),
                temperature=float(sc["temperature"]), rainfall=float(sc["rainfall"]),
                competitor_avg=float(sc["competitor_avg"]),
                load_rate=load_rate,
                prev_price=self.prev_price,
                current_queues=cast(Optional[Dict[str, float]], sc.get("queue_snapshot")),
            )

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.current_step = 0
            self.prev_price = settings.pricing.base_price
            self.current_scenario = self._sample_scenario()
            return self._obs(), {}

        def step(self, action):
            a = float(np.clip(action[0], -1, 1))
            price = settings.pricing.min_price + (a + 1) / 2 * (
                settings.pricing.max_price - settings.pricing.min_price
            )
            price = round(price / 5) * 5

            # 用 forecaster 仿真客流
            sc = self.current_scenario
            if sc is None:
                sc = self._sample_scenario()
                self.current_scenario = sc
            d = pd.to_datetime(sc["date"])
            feature_row = {
                "price": price, "temperature": float(sc["temperature"]), "rainfall": float(sc["rainfall"]),
                "is_holiday": int(sc["day_type"] in ("holiday", "golden_week")),
                "is_weekend": int(sc["day_type"] == "weekend"),
                "day_of_week": int(d.dayofweek), "month": int(d.month),
                "day_of_month": int(d.day),
                "season_id": {"spring":0,"summer":1,"autumn":2,"winter":3}.get(sc.get("season","spring"), 0),
                "day_type_id": {"weekday":0,"weekend":1,"holiday":2,"golden_week":3}.get(sc["day_type"], 0),
                "weather_id": {"晴好":0,"雨":1,"暴雨":2,"酷热":3,"严寒":4}.get(sc["weather"], 0),
            }
            try:
                if hasattr(self.forecaster, "predict_batch"):
                    visitors = float(self.forecaster.predict_batch([feature_row])[0])
                else:
                    visitors = float(self.forecaster.predict(feature_row))
            except Exception:
                visitors = settings.park_capacity * 0.5
            visitors = max(100, min(visitors, settings.park_capacity))

            ticket_rev = price * visitors
            secondary_rev = visitors * settings.secondary_consumption_ratio * 130
            load_rate = visitors / settings.park_capacity
            load_pen = 80 * (load_rate - settings.optimal_load) ** 2 * 1000
            smooth_pen = 0.0
            if abs(price - self.prev_price) / max(self.prev_price, 1) > 0.15:
                smooth_pen = abs(price - self.prev_price) * 10
            raw = ticket_rev + 0.45 * secondary_rev - load_pen - smooth_pen
            reward = float(raw / 1e5)

            self.prev_price = price
            self.current_step += 1
            done = self.current_step >= self.episode_length
            if not done:
                self.current_scenario = self._sample_scenario()

            return self._obs(load_rate), reward, done, False, {
                "price": price, "visitors": visitors, "load_rate": load_rate,
            }


class ContinuousRLPricer:
    """连续表征RL定价器 — 仅支持 PPO 连续策略"""

    def __init__(self, forecaster, history_df, attraction_graph=None):
        if not _HAS_SB3:
            raise ImportError(
                "ContinuousRLPricer requires stable-baselines3 and gymnasium. "
                "Please install required dependencies before startup."
            )
        self.forecaster = forecaster
        self.history = history_df
        self.encoder = ContinuousStateEncoder(attraction_graph)
        self.encoder.fit_gnn_embedding()
        self.model: Optional[Any] = None
        self.is_trained = False
        self.mode = "ppo"

    def train(self, total_timesteps: int = 20_000) -> dict:
        return self._train_ppo(total_timesteps)

    def _train_ppo(self, total_timesteps: int) -> dict:
        logger.info(f"PPO训练(连续表征+GNN嵌入) | timesteps={total_timesteps}")
        env = _PPOParkEnv(self.forecaster, self.history, self.encoder)
        model = PPO(
            "MlpPolicy", env,
            learning_rate=3e-4, n_steps=1024, batch_size=64,
            n_epochs=10, gamma=0.99, clip_range=0.2,
            policy_kwargs=dict(net_arch=[64, 64]),
            verbose=0,
        )
        model.learn(total_timesteps=total_timesteps)
        self.model = model
        self.is_trained = True
        return {"mode": "ppo", "timesteps": total_timesteps,
                "state_dim": ContinuousStateEncoder.TOTAL_DIM}

    def recommend_price(
        self, day_type: str, weather: str,
        competitor_avg: float, load_rate: float = 0.5,
        prev_price: Optional[float] = None, season: str = "spring",
        temperature: float = 22.0, rainfall: float = 0.0,
        current_queues: Optional[Dict[str, float]] = None,
    ) -> dict:
        if not self.is_trained:
            raise RuntimeError("未训练")
        if self.model is None:
            raise RuntimeError("模型未初始化")

        model = cast(Any, self.model)
        prev = float(prev_price if prev_price is not None else settings.pricing.base_price)
        obs = self.encoder.encode(
            day_type, weather, season,
            temperature, rainfall, competitor_avg, load_rate,
            prev,
            current_queues=current_queues,
        )
        action, _ = model.predict(obs, deterministic=True)
        a = float(np.clip(action[0], -1, 1))
        raw_price = settings.pricing.min_price + (a + 1) / 2 * (
            settings.pricing.max_price - settings.pricing.min_price
        )

        # 动作平滑: 限制相对上一价格的单次变动幅度
        max_delta = max(prev * settings.pricing.max_daily_change, 5.0)
        smooth_price = float(np.clip(raw_price, prev - max_delta, prev + max_delta))
        price = round(smooth_price / 5) * 5

        return {
            "recommended_price": float(price),
            "raw_action": a,
            "raw_price": float(raw_price),
            "smoothed_price": float(price),
            "smooth_penalty": float(abs(raw_price - smooth_price)),
            "mode": "ppo_continuous",
            "fallback": False,
            "state_dim": len(obs),
        }
