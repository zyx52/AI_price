"""
套餐组合优化器

场景:
  - 雨天 → 室内项目优先券 + 门票折扣
  - 高温 → 夜场票 + 冷饮券
  - 淡季工作日 → 学生/亲子套票
  - 黄金周 → 高附加值VIP套餐(快速通道+餐饮)

逻辑:
  根据当日外部信号,从套餐规则库中匹配若干候选套餐,
  并估算每个套餐的预期挽回客流/提升二消。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List
import numpy as np

from utils.logger import get_logger
from config import settings

logger = get_logger("BundleOptimizer")


@dataclass
class BundleSuggestion:
    name: str
    description: str
    discount: float            # 相对标准票的折扣(0.85 = 85折)
    target_segment: str        # 目标客群
    expected_uplift: float     # 预期客流/二消提振比例
    reasoning: str             # 推荐理由

    def to_dict(self):
        return asdict(self)


class BundleOptimizer:
    """套餐组合规则引擎(可进一步升级为RL/组合优化)"""

    def suggest(
        self,
        day_type: str,
        weather: str,
        temperature: float,
        rainfall: float,
        load_rate: float,
    ) -> List[BundleSuggestion]:
        """
        根据当日情境给出 1-3 个候选套餐
        """
        suggestions: List[BundleSuggestion] = []

        # === 天气驱动 ===
        if rainfall > 5 or weather in ("雨", "暴雨"):
            suggestions.append(BundleSuggestion(
                name="雨天特惠 · 室内项目优先券",
                description="门票88折 + 室内场馆优先入场 + 免费雨衣",
                discount=0.88,
                target_segment="家庭亲子",
                expected_uplift=0.18,
                reasoning="雨天客流通常下降30%+,通过室内体验+价格折扣挽回部分客流",
            ))

        if temperature >= settings.weather.heat_threshold_c:
            suggestions.append(BundleSuggestion(
                name="夜场避暑套票",
                description="17:00后入园 + 冷饮券 + 水上项目优先",
                discount=0.70,
                target_segment="年轻客群",
                expected_uplift=0.25,
                reasoning="高温日错峰运营,提升夜场使用率",
            ))

        if temperature <= settings.weather.cold_threshold_c:
            suggestions.append(BundleSuggestion(
                name="暖冬亲子包",
                description="门票 + 热饮2杯 + 室内互动展",
                discount=0.85,
                target_segment="家庭亲子",
                expected_uplift=0.15,
                reasoning="严寒天气客流下滑,通过取暖型增值提升吸引力",
            ))

        # === 日期驱动 ===
        if day_type == "weekday" and load_rate < 0.4:
            suggestions.append(BundleSuggestion(
                name="工作日学生/家庭早鸟票",
                description="门票75折 + 免费储物柜 + 15:00前入园",
                discount=0.75,
                target_segment="家庭亲子/学生",
                expected_uplift=0.30,
                reasoning="工作日客流偏低,通过价格激励提升上座率",
            ))

        if day_type == "golden_week":
            suggestions.append(BundleSuggestion(
                name="黄金周VIP尊享套餐",
                description="门票 + 快速通道 + 主题餐厅 + 纪念周边",
                discount=1.35,  # 溢价套餐
                target_segment="商务/VIP",
                expected_uplift=0.12,  # 单客消费提升
                reasoning="黄金周高需求,高附加值套餐提升单客毛利",
            ))

        if day_type in ("holiday", "golden_week") and load_rate > 0.85:
            suggestions.append(BundleSuggestion(
                name="错峰入园优选",
                description="下午场门票 + 夜间烟花 + 优先退场通道",
                discount=0.80,
                target_segment="年轻客群",
                expected_uplift=0.10,
                reasoning="旺日高负载,通过错峰套餐分流早晨排队压力",
            ))

        # 兜底: 保证至少返回一条标准建议
        if not suggestions:
            suggestions.append(BundleSuggestion(
                name="标准日常套餐",
                description="门票 + 园内代币¥50 + 一次免费项目",
                discount=0.95,
                target_segment="全客群",
                expected_uplift=0.05,
                reasoning="常规日无强烈外部信号,以小幅激励维持客流稳态",
            ))

        # 按预期提振排序
        suggestions.sort(key=lambda s: s.expected_uplift, reverse=True)
        return suggestions[:3]

    def format_summary(self, suggestions: List[BundleSuggestion]) -> str:
        """把候选套餐拼成一段运营友好的文字"""
        if not suggestions:
            return "无特别推荐套餐"
        lines = []
        for i, s in enumerate(suggestions, 1):
            price_hint = f"{s.discount*100:.0f}折" if s.discount < 1 else f"溢价{(s.discount-1)*100:.0f}%"
            lines.append(
                f"{i}. 【{s.name}】({price_hint}, 目标: {s.target_segment})\n"
                f"     {s.description}\n"
                f"     预期提振: {s.expected_uplift*100:.0f}% | {s.reasoning}"
            )
        return "\n".join(lines)
