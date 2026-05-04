"""
美团门票渠道适配器

鉴权: HMAC-SHA256 Token
API:  https://open.meituan.com/api/ticket/price/update
限流: 10 QPS (企业版)
"""
from __future__ import annotations

import os
import time
import hmac
import hashlib
import json
from typing import Dict, Optional, Tuple

from adapters import BaseChannelAdapter, PricePushRequest
from utils.logger import get_logger

logger = get_logger("MeituanAdapter")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


class MeituanAdapter(BaseChannelAdapter):
    """美团门票渠道适配器"""

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_url: str = "https://open.meituan.com/api",
        rate_limit_qps: float = 10.0,
    ):
        super().__init__(
            channel_name="meituan",
            base_url=base_url,
            rate_limit_qps=rate_limit_qps,
            timeout_seconds=8.0,
        )
        self.app_key = app_key or os.getenv("MEITUAN_APP_KEY", "")
        self.app_secret = app_secret or os.getenv("MEITUAN_APP_SECRET", "")

    def _build_auth_headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        sign_str = f"{self.app_key}{timestamp}"
        signature = hmac.new(
            self.app_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return {
            "Content-Type": "application/json",
            "X-MT-AppKey": self.app_key,
            "X-MT-Timestamp": timestamp,
            "X-MT-Signature": signature,
        }

    def _push_price_impl(self, req: PricePushRequest) -> Tuple[int, Optional[str], str]:
        if not _HAS_REQUESTS:
            return 0, None, "requests 库未安装"

        url = f"{self.base_url}/ticket/price/update"

        payload = {
            "productId": req.product_id,
            "price": int(req.channel_price * 100),  # 美团以分为单位
            "originalPrice": int(req.base_price * 100),
            "effectiveTime": req.effective_time or "",
            "reason": req.reason or "AI动态定价",
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._build_auth_headers(),
                timeout=self.timeout,
            )
            data = resp.json()

            if data.get("code") == 0:
                ack_id = data.get("data", {}).get("updateId", "")
                return resp.status_code, ack_id, ""
            else:
                return resp.status_code, None, data.get("msg", "未知错误")

        except requests.exceptions.Timeout:
            return 0, None, "Request timeout"
        except requests.exceptions.ConnectionError:
            return 0, None, "Connection failed"

    def _verify_price(self, product_id: str, expected_price: float) -> bool:
        if not _HAS_REQUESTS:
            return False
        try:
            url = f"{self.base_url}/ticket/price/query"
            resp = requests.get(
                url,
                params={"productId": product_id},
                headers=self._build_auth_headers(),
                timeout=5,
            )
            data = resp.json()
            current = data.get("data", {}).get("price", 0) / 100.0
            return abs(current - expected_price) < 0.01
        except Exception:
            return False
