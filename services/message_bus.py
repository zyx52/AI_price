"""
Redis Pub/Sub 消息总线 —— 高可用实时数据动脉

架构:
  ┌──────────────────────────────────────────────────────┐
  │                  Redis Pub/Sub                        │
  │                                                      │
  │  Publishers:                                         │
  │    weather_daemon  ──→ park:live_data:weather        │
  │    turnstile_daemon ──→ park:live_data:turnstile     │
  │    competitor_crawler → park:live_data:competitor    │
  │                                                      │
  │  Subscribers:                                        │
  │    pricing_subscriber ←── 消费所有 live_data 频道    │
  │    feature_service    ←── 实时特征更新               │
  │    alert_engine       ←── 异常事件频道               │
  │    capacity_manager   ←── 客流实时推送               │
  └──────────────────────────────────────────────────────┘

频道命名规范:
  park:live_data:{source}       实时数据流
  park:anomaly:events           异常事件
  park:control:{command}        控制指令
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set
from enum import Enum

from utils.logger import get_logger
from config import settings

logger = get_logger("MessageBus")

try:
    import redis as _redis_lib
    _HAS_REDIS = True
except ImportError:
    _redis_lib = None  # type: ignore[assignment]
    _HAS_REDIS = False


# ============================================================
# 频道定义
# ============================================================
class Channel(str, Enum):
    """标准 Pub/Sub 频道"""
    # 实时数据
    WEATHER = "park:live_data:weather"
    TURNSTILE = "park:live_data:turnstile"
    COMPETITOR = "park:live_data:competitor"
    # 异常事件
    ANOMALY = "park:anomaly:events"
    # 控制指令
    CONTROL_RETRAIN = "park:control:retrain"
    CONTROL_FLUSH_CACHE = "park:control:flush_cache"
    # 定价决策结果
    PRICING_DECISION = "park:decision:pricing"


# ============================================================
# 消息结构
# ============================================================
@dataclass
class BusMessage:
    """标准消息信封"""
    channel: str
    payload: Dict[str, Any]
    timestamp: str          # ISO 8601
    source: str             # 发布者标识
    msg_id: str             # 唯一消息ID
    ttl_seconds: int        # 消息生存时间(0=永久)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "BusMessage":
        return cls(**json.loads(raw))

    @classmethod
    def create(cls, channel: str, payload: Dict[str, Any],
               source: str = "unknown", ttl_seconds: int = 300) -> "BusMessage":
        return cls(
            channel=channel,
            payload=payload,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=source,
            msg_id=f"{source}:{int(time.time() * 1000)}:{hash(json.dumps(payload, sort_keys=True)) & 0xFFFF:04x}",
            ttl_seconds=ttl_seconds,
        )


# ============================================================
# 消息总线核心
# ============================================================
class MessageBus:
    """
    Redis Pub/Sub 消息总线

    使用方式:
      bus = MessageBus()

      # 发布
      bus.publish(Channel.WEATHER, {"temperature": 24, "rainfall": 0})

      # 订阅
      def on_weather(msg: BusMessage):
          print(f"收到天气: {msg.payload}")

      bus.subscribe(Channel.WEATHER, on_weather)
      bus.listen()  # 阻塞监听(通常在独立线程中运行)
    """

    _instance: Optional["MessageBus"] = None
    _lock = threading.Lock()

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_url = redis_url or settings.redis_url
        self._publisher: Any = None
        self._subscriber: Any = None
        self._subscribed_channels: Set[str] = set()
        self._handlers: Dict[str, List[Callable[[BusMessage], None]]] = {}
        self._listening = False
        self._listener_thread: Optional[threading.Thread] = None
        self._healthy = False

        self._init_clients()

    def _init_clients(self):
        if not _HAS_REDIS:
            logger.warning("redis 包未安装, MessageBus 降级为 no-op 模式")
            self._healthy = False
            return

        try:
            self._publisher = _redis_lib.from_url(
                self.redis_url,
                socket_timeout=settings.redis_socket_timeout_seconds,
                socket_connect_timeout=settings.redis_socket_timeout_seconds,
                decode_responses=True,
            )
            self._publisher.ping()
            self._subscriber = _redis_lib.from_url(
                self.redis_url,
                socket_timeout=settings.redis_socket_timeout_seconds,
                socket_connect_timeout=settings.redis_socket_timeout_seconds,
                decode_responses=True,
            )
            self._subscriber.ping()
            self._healthy = True
            logger.info(f"MessageBus 已连接 Redis: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis 不可用, MessageBus 降级为 no-op 模式: {e}")
            self._healthy = False

    @classmethod
    def get_instance(cls) -> "MessageBus":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @property
    def is_healthy(self) -> bool:
        return self._healthy

    # ============================================================
    # 发布
    # ============================================================
    def publish(self, channel: str, payload: Dict[str, Any],
                source: str = "unknown", ttl_seconds: int = 300) -> bool:
        """发布消息到指定频道"""
        msg = BusMessage.create(channel, payload, source, ttl_seconds)

        if not self._healthy:
            logger.debug(f"[no-op] publish to {channel}: {payload}")
            return False

        try:
            count = self._publisher.publish(channel, msg.to_json())
            logger.debug(f"Published to {channel} | receivers={count} | {payload}")
            return count > 0
        except Exception as e:
            logger.warning(f"发布消息失败 [{channel}]: {e}")
            return False

    # ============================================================
    # 订阅
    # ============================================================
    def subscribe(self, channel: str, handler: Callable[[BusMessage], None]):
        """注册频道处理器"""
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)
        self._subscribed_channels.add(channel)
        logger.info(f"订阅频道: {channel} (handlers={len(self._handlers[channel])})")

    def unsubscribe(self, channel: str):
        self._handlers.pop(channel, None)
        self._subscribed_channels.discard(channel)

    # ============================================================
    # 监听循环
    # ============================================================
    def listen(self, blocking: bool = True):
        """
        开始监听已订阅的频道
        blocking=True:  阻塞当前线程
        blocking=False: 在后台线程中启动
        """
        if not self._healthy:
            logger.warning("MessageBus 不健康,跳过监听")
            return

        if blocking:
            self._listen_loop()
        else:
            self._listener_thread = threading.Thread(
                target=self._listen_loop, daemon=True, name="messagebus-listener"
            )
            self._listener_thread.start()
            logger.info("MessageBus 后台监听线程已启动")

    def _listen_loop(self):
        """主监听循环"""
        self._listening = True
        pubsub = self._subscriber.pubsub(ignore_subscribe_messages=True)

        channels = list(self._subscribed_channels)
        if not channels:
            logger.warning("没有订阅任何频道,监听循环退出")
            return
        pubsub.subscribe(*channels)
        logger.info(f"开始监听 {len(channels)} 个频道: {channels}")

        try:
            while self._listening:
                message = pubsub.get_message(timeout=1.0)
                if message is None:
                    continue

                channel = message.get("channel", "")
                data_raw = message.get("data", "")

                if not data_raw or data_raw == "":
                    continue

                try:
                    msg = BusMessage.from_json(data_raw)
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"消息解析失败 [{channel}]: {e}")
                    continue

                # 分发到对应 handlers
                handlers = self._handlers.get(channel, [])
                for handler in handlers:
                    try:
                        handler(msg)
                    except Exception as e:
                        logger.error(f"Handler 执行异常 [{channel}]: {e}", exc_info=True)

        except Exception as e:
            logger.error(f"监听循环异常: {e}", exc_info=True)
        finally:
            try:
                pubsub.unsubscribe(*channels)
                pubsub.close()
            except Exception:
                pass
            logger.info("MessageBus 监听循环已退出")

    def stop(self):
        """停止监听"""
        self._listening = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=3.0)

    # ============================================================
    # 便捷方法: 发布实时数据
    # ============================================================
    def publish_weather(self, data: Dict[str, Any]) -> bool:
        return self.publish(Channel.WEATHER, data, source="weather_daemon", ttl_seconds=1800)

    def publish_turnstile(self, data: Dict[str, Any]) -> bool:
        return self.publish(Channel.TURNSTILE, data, source="turnstile_daemon", ttl_seconds=300)

    def publish_competitor(self, data: Dict[str, Any]) -> bool:
        return self.publish(Channel.COMPETITOR, data, source="competitor_crawler", ttl_seconds=3600)

    def publish_anomaly(self, data: Dict[str, Any]) -> bool:
        return self.publish(Channel.ANOMALY, data, source="anomaly_detector", ttl_seconds=86400)

    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        status = {
            "healthy": self._healthy,
            "redis_url": self.redis_url,
            "subscribed_channels": list(self._subscribed_channels),
            "handler_count": sum(len(h) for h in self._handlers.values()),
            "listening": self._listening,
        }
        if self._healthy:
            try:
                status["redis_ping_ms"] = self._publisher.ping()
            except Exception:
                status["redis_ping_ms"] = None
        return status


# ============================================================
# 全局单例
# ============================================================
bus = MessageBus.get_instance()
