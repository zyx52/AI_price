"""
渠道适配器包 —— 统一的OTA/自营渠道发布接口

架构:
  PricingEngine → ChannelPricer → [Adapter1, Adapter2, ...] → OTA APIs
  
  每个Adapter负责:
    1. 鉴权 (OAuth/HMAC/API Key)
    2. 限流 (Token Bucket)
    3. 价格推送
    4. 健康检查
    5. 价格回查

支持的渠道:
  - meituan  : 美团  (HMAC-SHA256)
  - ctrip    : 携程  (OAuth 2.0)
  - feizhu   : 飞猪  (OAuth 2.0 + 阿里签名)
  - miniapp  : 自建小程序 (API Key)
"""

from typing import Dict, Optional

from adapters._base import (
    BaseChannelAdapter,
    AdapterHealth,
    PricePushRequest,
    PricePushResult,
    TokenBucket,
)

from adapters.meituan_adapter import MeituanAdapter
from adapters.ctrip_adapter import CtripAdapter
from adapters.feizhu_adapter import FeizhuAdapter
from adapters.miniapp_adapter import MiniAppAdapter, HumanInLoopMiniAppAdapter

# ============================================================
# 适配器注册表
# ============================================================
_ADAPTER_REGISTRY: Dict[str, BaseChannelAdapter] = {}


def register_adapter(name: str, adapter: BaseChannelAdapter):
    """注册一个渠道适配器"""
    _ADAPTER_REGISTRY[name] = adapter
    from utils.logger import get_logger
    get_logger("AdapterRegistry").info(f"注册渠道适配器: {name}")


def get_adapter(name: str) -> Optional[BaseChannelAdapter]:
    """获取已注册的适配器"""
    return _ADAPTER_REGISTRY.get(name)


def list_adapters() -> Dict[str, Dict]:
    """列出所有已注册适配器及其健康状态"""
    return {
        name: adapter.health_check()
        for name, adapter in _ADAPTER_REGISTRY.items()
    }


def create_default_adapters(miniapp_hitl: bool = True) -> Dict[str, BaseChannelAdapter]:
    """
    创建默认的渠道适配器组

    miniapp_hitl: 小程序是否使用 Human-in-the-loop 模式
    """
    adapters = {
        "meituan": MeituanAdapter(),
        "ctrip": CtripAdapter(),
        "feizhu": FeizhuAdapter(),
    }

    if miniapp_hitl:
        adapters["miniapp"] = HumanInLoopMiniAppAdapter()
    else:
        adapters["miniapp"] = MiniAppAdapter()

    for name, adp in adapters.items():
        register_adapter(name, adp)

    return adapters


__all__ = [
    # 基类
    "BaseChannelAdapter", "AdapterHealth",
    "PricePushRequest", "PricePushResult", "TokenBucket",
    # 具体适配器
    "MeituanAdapter", "CtripAdapter", "FeizhuAdapter",
    "MiniAppAdapter", "HumanInLoopMiniAppAdapter",
    # 注册表
    "register_adapter", "get_adapter", "list_adapters",
    "create_default_adapters",
]
