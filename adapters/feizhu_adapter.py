"""
飞猪门票渠道适配器

鉴权: OAuth 2.0 + 阿里系签名
API:  https://open.fliggy.com/api/ticket/price/adjust
限流: 8 QPS
"""
from __future__ import annotations

import os
import time
import hmac
import hashlib
import base64
from typing import Dict, Optional, Tuple

from adapters import BaseChannelAdapter, PricePushRequest
from utils.logger import get_logger

logger = get_logger("FeizhuAdapter")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


class FeizhuAdapter(BaseChannelAdapter):
    """飞猪门票渠道适配器"""

    def __init__(
        self,
        app_key: Optional[str] = None,
        app_secret: Optional[str] = None,
        base_url: str = "https://open.fliggy.com/api",
        rate_limit_qps: float = 8.0,
    ):
        super().__init__(
            channel_name="feizhu",
            base_url=base_url,
            rate_limit_qps=rate_limit_qps,
            timeout_seconds=10.0,
        )
        self.app_key = app_key or os.getenv("FEIZHU_APP_KEY", "")
        self.app_secret = app_secret or os.getenv("FEIZHU_APP_SECRET", "")

    def _build_auth_headers(self) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        nonce = hashlib.md5(timestamp.encode()).hexdigest()[:16]

        # 阿里系签名: base64(hmac-sha256(app_secret, app_key+timestamp+nonce))
        sign_raw = f"{self.app_key}{timestamp}{nonce}"
        signature = base64.b64encode(
            hmac.new(
                self.app_secret.encode("utf-8"),
                sign_raw.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        ).decode("utf-8")

        return {
            "Content-Type": "application/json",
            "X-FZ-AppKey": self.app_key,
            "X-FZ-Timestamp": timestamp,
            "X-FZ-Nonce": nonce,
            "X-FZ-Signature": signature,
        }

    def _push_price_impl(self, req: PricePushRequest) -> Tuple[int, Optional[str], str]:
        if not _HAS_REQUESTS:
            return 0, None, "requests 库未安装"

        url = f"{self.base_url}/ticket/price/adjust"

        payload = {
            "itemId": req.product_id,
            "newPrice": req.channel_price,
            "costPrice": req.base_price,
            "effectiveTime": req.effective_time or "",
            "changeReason": req.reason or "AI动态定价",
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._build_auth_headers(),
                timeout=self.timeout,
            )
            data = resp.json()

            if data.get("success"):
                ack_id = data.get("result", {}).get("traceId", "")
                return resp.status_code, ack_id, ""
            else:
                return resp.status_code, None, data.get("errorMessage", "未知错误")

        except requests.exceptions.Timeout:
            return 0, None, "Request timeout"
        except requests.exceptions.ConnectionError:
            return 0, None, "Connection failed"

    def _verify_price(self, product_id: str, expected_price: float) -> bool:
        if not _HAS_REQUESTS:
            return False
        try:
            url = f"{self.base_url}/ticket/price/get"
            resp = requests.get(
                url,
                params={"itemId": product_id},
                headers=self._build_auth_headers(),
                timeout=5,
            )
            data = resp.json()
            current = float(data.get("result", {}).get("price", 0))
            return abs(current - expected_price) < 0.01
        except Exception:
            return False
