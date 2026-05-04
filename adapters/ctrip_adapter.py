"""
携程门票渠道适配器

鉴权: OAuth 2.0 (Client Credentials)
API:  https://open.ctrip.com/api/ticket/price/sync
限流: 5 QPS
"""
from __future__ import annotations

import os
import time
from typing import Dict, Optional, Tuple

from adapters import BaseChannelAdapter, PricePushRequest
from utils.logger import get_logger

logger = get_logger("CtripAdapter")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


class CtripAdapter(BaseChannelAdapter):
    """携程门票渠道适配器 (OAuth 2.0)"""

    TOKEN_URL = "https://open.ctrip.com/oauth/token"

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        base_url: str = "https://open.ctrip.com/api",
        rate_limit_qps: float = 5.0,
    ):
        super().__init__(
            channel_name="ctrip",
            base_url=base_url,
            rate_limit_qps=rate_limit_qps,
            timeout_seconds=10.0,
        )
        self.client_id = client_id or os.getenv("CTRIP_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("CTRIP_CLIENT_SECRET", "")
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0.0

    def _get_access_token(self) -> str:
        """获取/刷新 OAuth Token"""
        now = time.monotonic()
        if self._access_token and now < self._token_expires_at - 60:
            return self._access_token

        if not _HAS_REQUESTS:
            return ""

        resp = requests.post(
            self.TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=10,
        )
        data = resp.json()
        self._access_token = data.get("access_token", "")
        expires_in = data.get("expires_in", 3600)
        self._token_expires_at = now + expires_in
        logger.info("携程 OAuth Token 已刷新")
        return self._access_token

    def _build_auth_headers(self) -> Dict[str, str]:
        token = self._get_access_token()
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Client-Id": self.client_id,
        }

    def _push_price_impl(self, req: PricePushRequest) -> Tuple[int, Optional[str], str]:
        if not _HAS_REQUESTS:
            return 0, None, "requests 库未安装"

        url = f"{self.base_url}/ticket/price/sync"

        payload = {
            "resourceId": req.product_id,
            "sellPrice": req.channel_price,
            "costPrice": req.base_price,
            "effectiveDate": req.effective_time or "",
            "updateReason": req.reason or "AI动态调价",
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._build_auth_headers(),
                timeout=self.timeout,
            )
            data = resp.json()

            if data.get("status") == "SUCCESS":
                ack_id = data.get("syncId", "")
                return resp.status_code, ack_id, ""
            else:
                return resp.status_code, None, data.get("errorMsg", "未知错误")

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
                params={"resourceId": product_id},
                headers=self._build_auth_headers(),
                timeout=5,
            )
            data = resp.json()
            current = float(data.get("sellPrice", 0))
            return abs(current - expected_price) < 0.01
        except Exception:
            return False
