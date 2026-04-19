"""
渠道差异化定价引擎

不同销售渠道的特性差异:
  - OTA (携程/美团/飞猪): 流量大,价格敏感,需低价引流;平台抽佣
  - 官网直销:          毛利最高,用会员/积分绑定,可做深度套餐
  - 线下窗口:          即时游客,价格敏感度最低,正价销售
  - 企业团体:          大宗采购,走专属折扣通道

策略:
  1. 基础票价一致 —— 避免"大数据杀熟"合规风险
  2. 差异化在"附加权益" —— OTA送券、官网送积分、线下提供快速通道
  3. 动态渠道预算 —— 每日分配各渠道库存上限
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from enum import Enum

from utils.logger import get_logger
from config import settings

logger = get_logger("ChannelPricing")


class Channel(str, Enum):
    OTA = "OTA"                # 携程/美团/飞猪
    OFFICIAL = "OFFICIAL"      # 官网直销
    OFFLINE = "OFFLINE"        # 线下窗口
    GROUP = "GROUP"            # 企业团体
    MEMBER = "MEMBER"          # 会员专享


@dataclass
class ChannelPriceStrategy:
    channel: str
    display_price: float                  # 面价(各渠道一致,防合规风险)
    actual_net_price: float               # 扣除佣金/券后的实收
    commission_rate: float                # 渠道佣金率
    value_add: str                        # 附加权益
    allocation_ratio: float               # 建议分配库存比例
    target_segment: str
    marketing_message: str                # 营销文案(可被LLM重写)
    reasoning: str

    def to_dict(self):
        return asdict(self)


class ChannelPricer:
    """渠道差异化定价决策器"""

    # 各渠道基础参数(可从数据库读取,这里简化为常量)
    CHANNEL_CONFIG = {
        Channel.OTA: {
            "commission_rate": 0.10,     # 平台抽佣10%
            "target_segment": "价格敏感型游客",
            "strength": "流量大、转化高",
            "price_adjustment": 0.0,     # 面价不变
        },
        Channel.OFFICIAL: {
            "commission_rate": 0.02,     # 支付通道+营销费
            "target_segment": "品牌忠诚/会员",
            "strength": "毛利最高、数据闭环",
            "price_adjustment": 0.0,
        },
        Channel.OFFLINE: {
            "commission_rate": 0.00,
            "target_segment": "临时到访游客",
            "strength": "即时成交、无佣金",
            "price_adjustment": 0.05,    # 线下可小幅上浮(无折扣心智)
        },
        Channel.GROUP: {
            "commission_rate": 0.00,
            "target_segment": "企业/学校团体",
            "strength": "大宗订单",
            "price_adjustment": -0.15,   # 团体通常15%折扣
        },
        Channel.MEMBER: {
            "commission_rate": 0.02,
            "target_segment": "年卡/付费会员",
            "strength": "高LTV、高二消",
            "price_adjustment": -0.10,   # 会员10%折扣
        },
    }

    def compute(
        self,
        base_price: float,
        day_type: str,
        predicted_load: float,  # 预测负载率
    ) -> List[ChannelPriceStrategy]:
        """为每个渠道计算策略"""

        # 动态库存分配 —— 根据负载率调整
        # 低负载时OTA多分配(引流),高负载时官网多分配(保毛利)
        allocations = self._compute_allocations(predicted_load, day_type)

        # 各渠道附加权益(与负载率挂钩)
        value_adds = self._compute_value_adds(predicted_load, day_type)

        strategies = []

        for channel, config in self.CHANNEL_CONFIG.items():
            # 面价(OTA/官网/线下一致,团体和会员有折扣)
            if channel in (Channel.GROUP, Channel.MEMBER):
                display_price = round(base_price * (1 + config["price_adjustment"]) / 5) * 5
            else:
                display_price = round(base_price * (1 + config["price_adjustment"]) / 5) * 5

            # 净收(扣除佣金)
            net_price = display_price * (1 - config["commission_rate"])

            strategies.append(ChannelPriceStrategy(
                channel=channel.value,
                display_price=float(display_price),
                actual_net_price=round(float(net_price), 2),
                commission_rate=config["commission_rate"],
                value_add=value_adds[channel],
                allocation_ratio=allocations[channel],
                target_segment=config["target_segment"],
                marketing_message=self._marketing_message(channel, day_type, predicted_load),
                reasoning=self._reasoning(channel, config, predicted_load),
            ))
        return strategies

    # ---------- 动态分配 ----------
    def _compute_allocations(self, load: float, day_type: str) -> Dict[Channel, float]:
        """根据预测负载动态分配各渠道库存比例"""
        if load < 0.4:
            # 低负载: OTA引流为主
            return {
                Channel.OTA: 0.50, Channel.OFFICIAL: 0.20,
                Channel.OFFLINE: 0.15, Channel.GROUP: 0.10, Channel.MEMBER: 0.05,
            }
        elif load < 0.7:
            # 中等负载: 平衡
            return {
                Channel.OTA: 0.35, Channel.OFFICIAL: 0.30,
                Channel.OFFLINE: 0.20, Channel.GROUP: 0.08, Channel.MEMBER: 0.07,
            }
        else:
            # 高负载: 官网+线下为主(毛利高)
            return {
                Channel.OTA: 0.20, Channel.OFFICIAL: 0.40,
                Channel.OFFLINE: 0.28, Channel.GROUP: 0.05, Channel.MEMBER: 0.07,
            }

    # ---------- 附加权益 ----------
    def _compute_value_adds(self, load: float, day_type: str) -> Dict[Channel, str]:
        if load < 0.4:
            return {
                Channel.OTA:      "赠¥30园内代金券 + 免费储物柜",
                Channel.OFFICIAL: "双倍积分 + 会员专属冷饮券",
                Channel.OFFLINE:  "赠游园地图 + 项目推荐手册",
                Channel.GROUP:    "15%团体折扣 + 专属导览",
                Channel.MEMBER:   "会员8折 + 优先项目预约",
            }
        elif load < 0.7:
            return {
                Channel.OTA:      "赠¥20代金券",
                Channel.OFFICIAL: "积分10倍 + 免费停车",
                Channel.OFFLINE:  "赠园内WiFi连接",
                Channel.GROUP:    "15%团体折扣",
                Channel.MEMBER:   "会员9折",
            }
        else:  # 高负载
            return {
                Channel.OTA:      "标准权益",
                Channel.OFFICIAL: "官网尊享·快速通道¥80(旺日限定)",
                Channel.OFFLINE:  "赠避堵路线图",
                Channel.GROUP:    "10%团体折扣(旺日限)",
                Channel.MEMBER:   "会员9.5折 + VIP休息区",
            }

    # ---------- 营销文案 ----------
    def _marketing_message(self, channel: Channel, day_type: str, load: float) -> str:
        if channel == Channel.OTA:
            if load < 0.4:
                return "🎢 限时特惠!平日游园最舒适,现购门票再送¥30代金券"
            return "🎢 热门游乐园 · 品质保障 · 立即预订"
        if channel == Channel.OFFICIAL:
            return "✨ 官方直销·放心游园 · 下单享双倍积分+专属冷饮券"
        if channel == Channel.OFFLINE:
            return "今日门票发售中 · 即买即入园"
        if channel == Channel.GROUP:
            return "企业/学校团建专线 · 15人以上享团体价 + 专属服务"
        if channel == Channel.MEMBER:
            return "会员专享折扣 + 优先预约 · 一年畅游"
        return ""

    # ---------- 决策解释 ----------
    def _reasoning(self, channel: Channel, config: dict, load: float) -> str:
        base = f"{config['strength']}。佣金率{config['commission_rate']*100:.0f}%。"
        if load < 0.4 and channel == Channel.OTA:
            return base + " 低负载期加大投放引流。"
        if load > 0.7 and channel == Channel.OFFICIAL:
            return base + " 高负载期提升官网分配比,保护毛利。"
        return base

    # ---------- 汇总 ----------
    def format_summary(self, strategies: List[ChannelPriceStrategy]) -> str:
        lines = ["各渠道策略:"]
        for s in strategies:
            lines.append(
                f"  【{s.channel:8s}】¥{s.display_price:.0f} (净¥{s.actual_net_price:.0f}) | "
                f"分配 {s.allocation_ratio*100:.0f}% | {s.target_segment}"
            )
            lines.append(f"             附加权益: {s.value_add}")
        return "\n".join(lines)
