"""
自建小程序渠道适配器

鉴权: API Key (内部系统,最简单)
API:  内部服务 /api/internal/price/update
限流: 无限制(自有渠道)
特点: 
  - 灰度测试首选渠道
  - Human-in-the-loop 可配置
  - 更新即时生效
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

from adapters import BaseChannelAdapter, PricePushRequest
from utils.logger import get_logger

logger = get_logger("MiniAppAdapter")

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False


class MiniAppAdapter(BaseChannelAdapter):
    """自建小程序渠道适配器"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "http://localhost:8001/api/internal",
        rate_limit_qps: float = 50.0,
    ):
        super().__init__(
            channel_name="miniapp",
            base_url=base_url,
            rate_limit_qps=rate_limit_qps,
            timeout_seconds=5.0,
        )
        self.api_key = api_key or os.getenv("MINIAPP_API_KEY", "dev-internal-key")

    def _build_auth_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "X-Channel": "miniapp",
        }

    def _push_price_impl(self, req: PricePushRequest) -> Tuple[int, Optional[str], str]:
        if not _HAS_REQUESTS:
            return 0, None, "requests 库未安装"

        url = f"{self.base_url}/price/update"

        payload = {
            "productId": req.product_id,
            "price": req.channel_price,
            "basePrice": req.base_price,
            "originalPrice": req.original_price,
            "effectiveTime": req.effective_time or "",
            "reason": req.reason or "AI动态定价",
            "requestId": req.request_id,
        }

        try:
            resp = requests.post(
                url,
                json=payload,
                headers=self._build_auth_headers(),
                timeout=self.timeout,
            )
            data = resp.json()

            if data.get("ok"):
                ack_id = data.get("updateId", req.request_id)
                return resp.status_code, ack_id, ""
            else:
                return resp.status_code, None, data.get("error", "未知错误")

        except requests.exceptions.Timeout:
            return 0, None, "Request timeout"
        except requests.exceptions.ConnectionError as e:
            return 0, None, f"Connection failed: {e}"

    def _verify_price(self, product_id: str, expected_price: float) -> bool:
        if not _HAS_REQUESTS:
            return False
        try:
            url = f"{self.base_url}/price/get"
            resp = requests.get(
                url,
                params={"productId": product_id},
                headers=self._build_auth_headers(),
                timeout=3,
            )
            data = resp.json()
            current = float(data.get("price", 0))
            return abs(current - expected_price) < 0.01
        except Exception:
            return False


class HumanInLoopMiniAppAdapter(MiniAppAdapter):
    """
    Human-in-the-loop 模式小程序适配器

    价格变更需经过人工审批后才真正推送。
    用于灰度阶段的安全过渡。
    """

    def __init__(self, approval_callback=None, **kwargs):
        super().__init__(**kwargs)
        self._pending_approvals: Dict[str, PricePushRequest] = {}
        self._approval_callback = approval_callback

    def push_price(self, req: PricePushRequest):
        """拦截推送, 进入审批队列"""
        self._pending_approvals[req.request_id] = req
        logger.info(f"[miniapp:HITL] 价格变更待审批: {req.product_id} "
                     f"¥{req.original_price}→¥{req.channel_price}")

        # 通知审批人
        if self._approval_callback:
            self._approval_callback(req)

        from adapters import PricePushResult
        return PricePushResult(
            request_id=req.request_id,
            channel="miniapp",
            success=False,  # 暂未执行,等待审批
            error_message="AWAITING_APPROVAL",
        )

    def approve(self, request_id: str) -> bool:
        """审批通过,执行推送"""
        req = self._pending_approvals.pop(request_id, None)
        if req is None:
            return False
        logger.info(f"[miniapp:HITL] 审批通过,执行推送: {request_id}")
        return super().push_price(req).success

    def reject(self, request_id: str, reason: str = ""):
        """拒绝"""
        req = self._pending_approvals.pop(request_id, None)
        if req:
            logger.info(f"[miniapp:HITL] 审批拒绝: {request_id} reason={reason}")

    @property
    def pending_count(self) -> int:
        return len(self._pending_approvals)
