"""
分时定价引擎

把单日一个价格细化到3个时段:
  - 早鸟场 (07:00-11:00):  吸引亲子/老年客群,折扣价
  - 正常场 (11:00-17:00):  主力时段,标准价上浮
  - 夜场   (17:00-22:00):  年轻客群/避暑,差异化价格

收益:
  1. 错峰引流 —— 平滑客流波峰,减少热门时段拥堵
  2. 客群细分 —— 不同时段匹配不同客群的价格敏感度
  3. 容量利用率提升 —— 夜场通常利用率低,低价激活
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from utils.logger import get_logger
from config import settings

logger = get_logger("TimeSlotPricing")


@dataclass
class TimeSlotPrice:
    slot_name: str              # "早鸟场" / "正常场" / "夜场"
    start_hour: int
    end_hour: int
    price: float
    ratio_to_base: float        # 相对全日票价的系数
    target_segment: str
    expected_visitor_share: float  # 预计该时段客流占比
    reasoning: str

    def to_dict(self):
        return asdict(self)


class TimeSlotPricer:
    """分时定价决策器"""

    # 默认3段式
    DEFAULT_SLOTS = [
        ("早鸟场", 7, 11),
        ("正常场", 11, 17),
        ("夜场", 17, 22),
    ]

    # 不同场景的时段价格系数
    # key: (day_type, weather)
    SLOT_RATIOS = {
        # 工作日: 早鸟优惠引流、夜场低价
        ("weekday", "晴好"):    [0.70, 1.00, 0.75],
        ("weekday", "雨"):      [0.65, 0.90, 0.70],
        ("weekday", "酷热"):    [0.75, 1.05, 0.65],  # 酷热时夜场更便宜
        # 周末: 正常场溢价明显
        ("weekend", "晴好"):    [0.85, 1.10, 0.85],
        ("weekend", "雨"):      [0.80, 0.95, 0.80],
        ("weekend", "酷热"):    [0.85, 1.10, 0.75],
        # 节假日: 全天高价,正常场峰值
        ("holiday", "晴好"):    [0.95, 1.15, 0.95],
        ("holiday", "雨"):      [0.85, 1.00, 0.90],
        # 黄金周: 全天高溢价,正常场极致定价
        ("golden_week", "晴好"): [1.00, 1.20, 1.00],
        ("golden_week", "雨"):   [0.90, 1.05, 0.95],
    }

    # 各时段客流份额(每种场景略不同)
    SLOT_SHARES_DEFAULT = [0.25, 0.50, 0.25]
    SLOT_SHARES_HOT = [0.15, 0.40, 0.45]     # 酷热时夜场占比高
    SLOT_SHARES_GOLDEN = [0.35, 0.45, 0.20]  # 黄金周早鸟排队入园

    # 各时段主要客群
    SLOT_SEGMENTS = {
        "早鸟场": "家庭亲子/老年客群",
        "正常场": "主力客群(全客群)",
        "夜场":   "年轻客群/情侣",
    }

    def compute(
        self,
        base_price: float,
        day_type: str,
        weather: str,
        temperature: float = 25.0,
    ) -> List[TimeSlotPrice]:
        """
        计算3个时段的定价
        base_price: 由定价引擎算出的全日基准价
        """
        # 查找系数
        key = (day_type, weather)
        ratios = self.SLOT_RATIOS.get(key)
        if ratios is None:
            # 兜底: 按day_type查晴好的
            ratios = self.SLOT_RATIOS.get((day_type, "晴好"), [0.80, 1.00, 0.85])

        # 客流份额
        if temperature >= settings.weather.heat_threshold_c:
            shares = self.SLOT_SHARES_HOT
        elif day_type == "golden_week":
            shares = self.SLOT_SHARES_GOLDEN
        else:
            shares = self.SLOT_SHARES_DEFAULT

        results = []
        for i, (slot_name, start, end) in enumerate(self.DEFAULT_SLOTS):
            price = round(base_price * ratios[i] / 5) * 5  # 凑整到5元
            price = max(settings.pricing.min_price,
                        min(settings.pricing.max_price, price))

            reasoning = self._build_reasoning(slot_name, day_type, weather, ratios[i], temperature)

            results.append(TimeSlotPrice(
                slot_name=slot_name,
                start_hour=start, end_hour=end,
                price=price,
                ratio_to_base=ratios[i],
                target_segment=self.SLOT_SEGMENTS[slot_name],
                expected_visitor_share=shares[i],
                reasoning=reasoning,
            ))
        return results

    @staticmethod
    def _build_reasoning(slot, day_type, weather, ratio, temperature) -> str:
        if slot == "早鸟场":
            if day_type == "weekday":
                return "工作日早鸟引流,吸引亲子/老年避峰客群"
            if day_type == "golden_week":
                return "黄金周早鸟入园,错峰减少排队"
            return f"早鸟场优惠{(1-ratio)*100:.0f}%,平衡早高峰压力"
        elif slot == "正常场":
            if day_type in ("holiday", "golden_week"):
                return "节假日主力时段,承担最大客流,溢价提升单客毛利"
            if weather in ("雨", "暴雨"):
                return "雨天正常场折扣,维持基础客流"
            return "全日主力时段,标准价位"
        else:  # 夜场
            if temperature >= 35:
                return "酷热避暑夜场,错峰+低价组合,预计客流占45%"
            if day_type == "weekday":
                return "工作日夜场,低价激活年轻客群"
            if day_type == "golden_week":
                return "黄金周夜场,稳定价位承接白天延续客流"
            return "夜场差异化定价,适合情侣/年轻游客"

    def format_summary(self, prices: List[TimeSlotPrice]) -> str:
        """格式化输出"""
        lines = []
        for p in prices:
            lines.append(
                f"  【{p.slot_name}】{p.start_hour:02d}:00-{p.end_hour:02d}:00  "
                f"¥{p.price:.0f} ({p.ratio_to_base*100:.0f}%) | "
                f"目标客群: {p.target_segment} | "
                f"预计占比: {p.expected_visitor_share*100:.0f}%"
            )
        return "\n".join(lines)
