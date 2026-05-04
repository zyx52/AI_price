"""
渠道适配器基类 —— 标准化的渠道发布接口

所有 OTA / 自营渠道的 Adapter 必须实现此接口。
"""
from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from utils.logger import get_logger

logger = get_logger("ChannelAdapter")


class AdapterHealth(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    RATE_LIMITED = "rate_limited"
    UNAVAILABLE = "unavailable"


@dataclass
class PricePushRequest:
    channel: str
    product_id: str
    base_price: float
    channel_price: float
    original_price: float
    currency: str = "CNY"
    effective_time: str = ""
    expire_time: str = ""
    reason: str = ""
    request_id: str = ""

    def __post_init__(self):
        if not self.request_id:
            self.request_id = f"{self.channel}:{self.product_id}:{int(time.time()*1000)}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PricePushResult:
    request_id: str
    channel: str
    success: bool
    http_status: Optional[int] = None
    channel_ack_id: Optional[str] = None
    error_message: str = ""
    latency_ms: float = 0.0
    retry_count: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return asdict(self)


class TokenBucket:
    """线程安全的令牌桶限流器"""

    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: int = 1) -> bool:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_refill = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    @property
    def available(self) -> float:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            return min(self.burst, self._tokens + elapsed * self.rate)


class BaseChannelAdapter(ABC):
    """渠道适配器抽象基类"""

    def __init__(
        self,
        channel_name: str,
        base_url: str,
        rate_limit_qps: float = 5.0,
        timeout_seconds: float = 10.0,
    ):
        self.channel_name = channel_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout_seconds
        self._rate_limiter = TokenBucket(rate=rate_limit_qps, burst=int(rate_limit_qps * 2))
        self._health: AdapterHealth = AdapterHealth.HEALTHY
        self._consecutive_failures = 0
        self._last_success_time: float = 0.0

    @abstractmethod
    def _build_auth_headers(self) -> Dict[str, str]:
        ...

    @abstractmethod
    def _push_price_impl(self, req: PricePushRequest) -> Tuple[int, Optional[str], str]:
        ...

    @abstractmethod
    def _verify_price(self, product_id: str, expected_price: float) -> bool:
        ...

    def push_price(self, req: PricePushRequest) -> PricePushResult:
        start = time.monotonic()

        if not self._rate_limiter.acquire():
            self._health = AdapterHealth.RATE_LIMITED
            return PricePushResult(
                request_id=req.request_id, channel=self.channel_name,
                success=False, error_message="Rate limited",
                latency_ms=(time.monotonic() - start) * 1000,
            )

        try:
            http_status, ack_id, err = self._push_price_impl(req)
            latency = (time.monotonic() - start) * 1000

            if http_status and 200 <= http_status < 300:
                self._consecutive_failures = 0
                self._last_success_time = time.monotonic()
                self._health = AdapterHealth.HEALTHY
                return PricePushResult(
                    request_id=req.request_id, channel=self.channel_name,
                    success=True, http_status=http_status,
                    channel_ack_id=ack_id, latency_ms=latency,
                )
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= 3:
                    self._health = AdapterHealth.DEGRADED
                if self._consecutive_failures >= 10:
                    self._health = AdapterHealth.UNAVAILABLE
                return PricePushResult(
                    request_id=req.request_id, channel=self.channel_name,
                    success=False, http_status=http_status,
                    error_message=err or f"HTTP {http_status}", latency_ms=latency,
                )
        except Exception as e:
            self._consecutive_failures += 1
            return PricePushResult(
                request_id=req.request_id, channel=self.channel_name,
                success=False, error_message=str(e),
                latency_ms=(time.monotonic() - start) * 1000,
            )

    def health_check(self) -> Dict[str, Any]:
        return {
            "channel": self.channel_name,
            "health": self._health.value,
            "consecutive_failures": self._consecutive_failures,
            "last_success_seconds_ago": (
                time.monotonic() - self._last_success_time
                if self._last_success_time > 0 else None
            ),
            "rate_limiter_tokens": round(self._rate_limiter.available, 1),
        }

    def is_available(self) -> bool:
        return self._health in (AdapterHealth.HEALTHY, AdapterHealth.DEGRADED)
