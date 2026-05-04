"""
定价引擎订阅者 —— 实时消费消息总线,驱动定价决策

职责:
  1. 订阅 park:live_data:weather + park:live_data:turnstile
  2. 实时维护"当前情境"状态快照
  3. 当状态变化超过阈值(天气骤变/负载率突破85%) → 触发即时定价决策
  4. 发布决策结果到 park:decision:pricing
  5. 更新 FeatureService 实时特征

启动:
  python services/pricing_subscriber.py
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from utils.logger import get_logger
from utils.feature_cache import feature_cache
from utils.date_utils import get_day_type
from config import settings
from services.message_bus import bus, Channel, BusMessage

logger = get_logger("PricingSubscriber")


# ============================================================
# 当前情境快照
# ============================================================
@dataclass
class LiveContext:
    """实时情境快照 —— 由消息总线不断更新"""
    date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))

    # 天气
    weather_label: str = "晴好"
    temperature: float = 22.0
    rainfall: float = 0.0
    rain_probability: float = 0.0
    weather_timestamp: str = ""

    # 客流
    checked_in: int = 0
    not_entered: int = 0
    load_rate: float = 0.0
    entry_rate: float = 0.0
    load_warning: str = "normal"
    turnstile_timestamp: str = ""

    # 定价
    recommended_price: float = 299.0
    last_decision_timestamp: str = ""
    prev_price: float = 299.0

    def has_weather_changed(self, new_label: str, new_rainfall: float) -> bool:
        return self.weather_label != new_label or abs(self.rainfall - new_rainfall) > 3.0

    def has_load_breached(self) -> bool:
        return self.load_warning in ("warning_85%", "critical_90%")

    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "weather_label": self.weather_label,
            "temperature": self.temperature,
            "rainfall": self.rainfall,
            "rain_probability": self.rain_probability,
            "checked_in": self.checked_in,
            "not_entered": self.not_entered,
            "load_rate": self.load_rate,
            "entry_rate": self.entry_rate,
            "load_warning": self.load_warning,
            "recommended_price": self.recommended_price,
        }


# ============================================================
# 定价引擎订阅者
# ============================================================
class PricingSubscriber:
    """
    定价引擎订阅者

    核心逻辑:
      1. 监听天气+闸机频道
      2. 维护 LiveContext
      3. 触发条件:
         a) 天气骤变 (晴→暴雨 / 雨量突增 > 3mm)
         b) 负载率突破 85%
         c) 定时触发 (每30分钟检查一次)
      4. 调用 PricingEngineV3 做决策
      5. 发布决策结果
    """

    TRIGGER_INTERVAL_SECONDS = 1800   # 30分钟定时触发
    WEATHER_RAIN_SPIKE_THRESHOLD = 3.0  # 降雨突增阈值(mm)

    def __init__(self):
        self._running = False
        self._context = LiveContext()
        self._last_trigger_time = 0.0
        self._engine = None             # 延迟加载
        self._feature_service = None    # 延迟加载
        self._sliding_engine = None     # 延迟加载

    # ============================================================
    # 消息处理器
    # ============================================================
    def _on_weather(self, msg: BusMessage):
        """处理天气消息"""
        payload = msg.payload
        new_label = payload.get("weather_label", "晴好")
        new_temp = float(payload.get("current_temperature", 22.0))
        new_rain = float(payload.get("rainfall_mm", 0.0))
        new_rain_prob = float(payload.get("rain_probability", 0.0))
        is_ff = payload.get("is_forward_filled", False)

        # 更新情境
        old_label = self._context.weather_label
        self._context.weather_label = new_label
        self._context.temperature = new_temp
        self._context.rainfall = new_rain
        self._context.rain_probability = new_rain_prob
        self._context.weather_timestamp = payload.get("timestamp", "")

        # 天气骤变触发
        if self._context.has_weather_changed(new_label, new_rain):
            logger.info(f"🌧️ 天气变化触发决策: {old_label} → {new_label} "
                         f"(降雨={new_rain}mm, 前向填充={is_ff})")
            self._trigger_decision("weather_change")

    def _on_turnstile(self, msg: BusMessage):
        """处理闸机消息"""
        payload = msg.payload
        checked_in = int(payload.get("checked_in_count", 0))
        load_rate = float(payload.get("current_load_rate", 0.0))
        entry_rate = float(payload.get("entry_rate_per_min", 0.0))
        load_warning = payload.get("load_warning", "normal")
        is_ff = payload.get("is_forward_filled", False)

        # 更新情境
        self._context.checked_in = checked_in
        self._context.not_entered = int(payload.get("not_entered_count", 0))
        self._context.load_rate = load_rate
        self._context.entry_rate = entry_rate
        self._context.load_warning = load_warning
        self._context.turnstile_timestamp = payload.get("timestamp", "")

        # 更新滑窗引擎
        if self._sliding_engine is not None:
            self._sliding_engine.push(checked_in, load_rate, entry_rate)

        # 负载率突破阈值触发
        if load_warning != "normal":
            logger.info(f"🚨 负载率突破触发决策: {load_warning} "
                         f"(load={load_rate:.1%}, 入园={checked_in})")
            self._trigger_decision("load_breach")

    def _on_anomaly(self, msg: BusMessage):
        """处理异常事件"""
        payload = msg.payload
        logger.warning(f"⚠️ 收到异常事件: {payload.get('type')} - {payload.get('message')}")

    # ============================================================
    # 定价决策触发
    # ============================================================
    def _trigger_decision(self, reason: str):
        """触发一次定价决策"""
        # 防抖: 至少间隔60秒
        now = time.time()
        if now - self._last_trigger_time < 60:
            logger.debug(f"跳过决策(防抖): {reason}")
            return
        self._last_trigger_time = now

        try:
            self._ensure_engine()
            decision = self._execute_decision(reason)

            # 发布决策结果
            bus.publish(
                Channel.PRICING_DECISION,
                decision,
                source="pricing_subscriber",
                ttl_seconds=1800,
            )

            # 更新缓存
            feature_cache.set("pricing:latest_decision", decision, ttl=1800)

            logger.info(
                f"💰 定价决策完成 [{reason}]: "
                f"价格=¥{decision.get('recommended_price', 0)} | "
                f"客流预测={decision.get('predicted_visitors', 0)} | "
                f"置信度={decision.get('confidence', 0):.2%}"
            )

        except Exception as e:
            logger.error(f"定价决策失败 [{reason}]: {e}", exc_info=True)

    def _ensure_engine(self):
        """延迟初始化引擎(避免启动时加载所有模型)"""
        if self._engine is not None:
            return

        logger.info("初始化定价引擎...")
        from data import DataLoader
        from models import (
            QuantileEnsembleForecaster, ContinuousRLPricer,
            FeatureService, EnhancedShiftDetector,
            ParkAttractionGraph,
        )
        from engine import PricingEngineV3
        from models.sliding_window import get_sliding_engine, register_sliding_window_features

        loader = DataLoader(source="mock")
        history = loader.load_history()

        forecaster = QuantileEnsembleForecaster()
        forecaster.train(history)

        self._feature_service = FeatureService(history)
        register_sliding_window_features(self._feature_service)

        shift_detector = EnhancedShiftDetector(use_torch=False)
        shift_detector.fit(history)

        graph = ParkAttractionGraph()
        graph.build_default_park()

        rl_pricer = ContinuousRLPricer(
            forecaster=forecaster,
            history_df=history,
            attraction_graph=graph,
        )
        rl_pricer.train(total_timesteps=3000)

        self._engine = PricingEngineV3(
            forecaster=forecaster,
            rl_pricer=rl_pricer,
            feature_service=self._feature_service,
            shift_detector=shift_detector,
        )

        self._sliding_engine = get_sliding_engine()
        logger.info("定价引擎初始化完成")

    def _execute_decision(self, reason: str) -> Dict[str, Any]:
        """执行定价决策"""
        ctx = self._context
        day_type = get_day_type(ctx.date)

        decision = self._engine.decide(
            date=ctx.date,
            weather=ctx.weather_label,
            temperature=ctx.temperature,
            rainfall=ctx.rainfall,
            competitor_prices={"A": 310, "B": 280, "C": 350},
            day_type=day_type,
        )

        result = decision.to_dict() if hasattr(decision, 'to_dict') else dict(decision)
        result["trigger_reason"] = reason
        result["live_load_rate"] = ctx.load_rate
        result["live_entry_rate"] = ctx.entry_rate
        result["weather_is_forward_filled"] = bool(ctx.weather_timestamp and "前向填充" in ctx.weather_timestamp)

        return result

    # ============================================================
    # 主循环
    # ============================================================
    def run(self):
        """启动订阅者"""
        self._running = True

        # 注册频道处理器
        bus.subscribe(Channel.WEATHER, self._on_weather)
        bus.subscribe(Channel.TURNSTILE, self._on_turnstile)
        bus.subscribe(Channel.ANOMALY, self._on_anomaly)

        # 在后台线程启动消息监听
        bus.listen(blocking=False)

        logger.info("🎯 定价引擎订阅者已启动 | 监听频道: weather, turnstile, anomaly")

        # 主循环: 定时触发 + 健康检查
        while self._running:
            try:
                now = time.time()

                # 定时触发
                if now - self._last_trigger_time >= self.TRIGGER_INTERVAL_SECONDS:
                    self._trigger_decision("scheduled")

                # 健康检查
                health = bus.health_check()
                if not health["healthy"]:
                    logger.warning("MessageBus 不健康,等待恢复...")

                time.sleep(10)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(30)

        self._shutdown()

    def _shutdown(self):
        self._running = False
        bus.stop()
        logger.info("定价引擎订阅者已停止")


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="定价引擎订阅者")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="使用mock数据(默认)")
    args = parser.parse_args()

    subscriber = PricingSubscriber()
    subscriber.run()


if __name__ == "__main__":
    main()
