"""
动态容量管理模块

不只调价格,还要根据预测客流动态调整:
  1. 园区运营时间(早开/延长闭园)
  2. 各项目运营人员配置
  3. 餐饮点开放数量
  4. 安保/清洁人员调度
  5. 各项目分时预约名额
  6. 应急限流措施

这是"AI参与完整运营决策循环"的关键一环。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional

from utils.logger import get_logger
from config import settings

logger = get_logger("CapacityManager")


@dataclass
class CapacityPlan:
    """容量管理计划"""
    date: str
    predicted_visitors: int
    load_rate: float
    load_level: str                         # "low" | "normal" | "high" | "critical"
    # 运营时间
    opening_time: str                       # "09:00"
    closing_time: str                       # "22:00"
    total_operating_hours: float
    # 人员配置
    staff_allocation: Dict[str, int]        # {"ride_ops": 45, "food_service": 30, ...}
    total_staff: int
    # 分项目容量
    ride_capacity_adjustment: Dict[str, str]   # {"过山车": "+15%", "摩天轮": "正常"}
    # 应急措施
    emergency_actions: List[str]
    # 估算成本/收益
    operational_cost_delta: float           # 相比基准日的成本增减
    reasoning: str

    def to_dict(self):
        return asdict(self)


class CapacityManager:
    """动态容量管理器"""

    # 基准运营(负载50%)的标准配置
    BASELINE = {
        "opening_time": "09:00",
        "closing_time": "21:00",
        "operating_hours": 12,
        "staff": {
            "ride_ops": 50,        # 游乐项目运营
            "food_service": 35,    # 餐饮服务
            "cleaning": 20,        # 清洁
            "security": 15,        # 安保
            "guest_service": 12,   # 客户服务
            "management": 8,       # 管理
        },
        "staff_cost_per_person_day": 350,   # 元/人·天
        "total_staff": 140,
    }

    def plan(
        self,
        date: str,
        predicted_visitors: int,
        day_type: str,
        weather: str,
    ) -> CapacityPlan:
        load_rate = predicted_visitors / settings.park_capacity
        load_level = self._load_level(load_rate)

        # 运营时间调整
        opening, closing, hours = self._adjust_operating_hours(day_type, weather, load_rate)

        # 人员配置
        staff = self._adjust_staff(load_rate, weather, day_type)
        total_staff = sum(staff.values())

        # 各项目容量调整
        ride_adj = self._adjust_ride_capacity(load_rate, weather)

        # 应急措施
        emergency = self._emergency_actions(load_rate, weather, day_type)

        # 成本变化
        cost_delta = (total_staff - self.BASELINE["total_staff"]) * self.BASELINE["staff_cost_per_person_day"]
        # 运营时间变化额外成本
        cost_delta += (hours - self.BASELINE["operating_hours"]) * 2500   # 每小时水电+维护成本

        return CapacityPlan(
            date=date,
            predicted_visitors=int(predicted_visitors),
            load_rate=float(load_rate),
            load_level=load_level,
            opening_time=opening,
            closing_time=closing,
            total_operating_hours=hours,
            staff_allocation=staff,
            total_staff=total_staff,
            ride_capacity_adjustment=ride_adj,
            emergency_actions=emergency,
            operational_cost_delta=float(cost_delta),
            reasoning=self._reasoning(load_level, weather, day_type, load_rate),
        )

    # ---------- 负载分级 ----------
    @staticmethod
    def _load_level(load: float) -> str:
        if load >= 0.90:
            return "critical"
        if load >= 0.75:
            return "high"
        if load >= 0.40:
            return "normal"
        return "low"

    # ---------- 运营时间 ----------
    def _adjust_operating_hours(self, day_type: str, weather: str, load: float) -> tuple:
        opening = "09:00"
        closing = "21:00"

        # 节假日/黄金周早开门
        if day_type in ("holiday", "golden_week"):
            opening = "08:00"
            closing = "22:00"
        # 酷热天延长夜场
        if weather == "酷热":
            opening = "10:00"  # 晚开门避开早高温
            closing = "22:30"
        # 暴雨可能提前闭园
        if weather == "暴雨":
            closing = "19:00"
        # 极低负载的工作日可提前闭园节约成本
        if day_type == "weekday" and load < 0.25 and weather not in ("晴好",):
            closing = "20:00"

        from datetime import datetime
        o = datetime.strptime(opening, "%H:%M")
        c = datetime.strptime(closing, "%H:%M")
        hours = (c - o).total_seconds() / 3600
        return opening, closing, hours

    # ---------- 人员配置 ----------
    def _adjust_staff(self, load: float, weather: str, day_type: str) -> Dict[str, int]:
        base = self.BASELINE["staff"].copy()

        # 按负载率线性调整
        if load >= 0.90:
            multipliers = {"ride_ops": 1.4, "food_service": 1.5, "cleaning": 1.6,
                           "security": 1.8, "guest_service": 1.5, "management": 1.2}
        elif load >= 0.75:
            multipliers = {"ride_ops": 1.25, "food_service": 1.3, "cleaning": 1.4,
                           "security": 1.5, "guest_service": 1.3, "management": 1.1}
        elif load >= 0.40:
            multipliers = {k: 1.0 for k in base}
        else:
            # 低负载时适度减员节约成本(但保留最小服务人员)
            multipliers = {"ride_ops": 0.75, "food_service": 0.65, "cleaning": 0.75,
                           "security": 0.80, "guest_service": 0.85, "management": 1.0}

        # 天气调整
        if weather == "雨" or weather == "暴雨":
            multipliers["cleaning"] = multipliers.get("cleaning", 1.0) * 1.3
            multipliers["security"] = multipliers.get("security", 1.0) * 1.2
        if weather == "酷热":
            multipliers["guest_service"] = multipliers.get("guest_service", 1.0) * 1.3  # 中暑/补水应对

        return {k: max(4, int(round(v * multipliers.get(k, 1.0)))) for k, v in base.items()}

    # ---------- 项目容量 ----------
    def _adjust_ride_capacity(self, load: float, weather: str) -> Dict[str, str]:
        adj = {}
        if load >= 0.75:
            adj["过山车·极速"] = "开启所有车组,+20%吞吐"
            adj["摩天轮"] = "缩短单圈时间,+15%吞吐"
            adj["旋转木马"] = "正常"
            adj["4D影院"] = "加开1场,+1场次"
            adj["室内演艺秀"] = "加开1场"
        elif load >= 0.40:
            adj["过山车·极速"] = "正常运营"
            adj["摩天轮"] = "正常"
            adj["4D影院"] = "按常规场次"
        else:
            adj["过山车·极速"] = "开启2/3车组,节约成本"
            adj["摩天轮"] = "正常"
            adj["4D影院"] = "按常规场次,人少时可取消部分场次"

        if weather in ("雨", "暴雨"):
            adj["过山车·极速"] = "暂停(安全原因)"
            adj["摩天轮"] = "暂停(强风/雷电)"
            adj["水世界"] = "暂停"
            adj["4D影院"] = "加开2场,分流雨天客流"
            adj["室内演艺秀"] = "加开1-2场"
        return adj

    # ---------- 应急措施 ----------
    def _emergency_actions(self, load: float, weather: str, day_type: str) -> List[str]:
        actions = []
        if load >= 0.95:
            actions.append("🔴 启动分时预约强制限流,官网/OTA停止当日售票")
            actions.append("🔴 开通紧急疏散动线,所有出入口增派引导员")
            actions.append("🔴 周边交通协调:联系交警启动外围分流")
        elif load >= 0.85:
            actions.append("🟡 热门项目启动免费快速通道(当日限量发放)")
            actions.append("🟡 各餐饮点准备快餐包,减少排队时长")
            actions.append("🟡 开启应急休息区空调/遮阳")
        elif load < 0.30:
            actions.append("🔵 通过APP/公众号推送当日限时二消优惠,刺激消费")
            actions.append("🔵 员工可轮休,降低人力成本")

        if weather == "暴雨":
            actions.append("⛈ 所有户外项目紧急暂停,清园室外区域")
            actions.append("⛈ 室内场馆加开场次,免费雨衣/毛巾发放点开启")
            actions.append("⛈ 启动天气险/延期入园凭证发放")
        elif weather == "酷热":
            actions.append("🌡 全园增加免费饮水点20个+冰敷站")
            actions.append("🌡 医务室加强中暑应急响应")
        return actions

    # ---------- 决策解释 ----------
    @staticmethod
    def _reasoning(level: str, weather: str, day_type: str, load: float) -> str:
        level_map = {
            "critical": "极高负载",
            "high":     "高负载",
            "normal":   "正常负载",
            "low":      "低负载",
        }
        desc = f"预计负载率{load*100:.0f}%({level_map[level]})。"
        if level == "critical":
            desc += "需要全力保障服务与安全,人员+时间全面扩张。"
        elif level == "high":
            desc += "合理扩充人力+延长营业,提升单日峰值吞吐。"
        elif level == "low":
            desc += "适度减员控成本,通过运营活动挖掘二消。"
        else:
            desc += "按基准配置,无需特别调整。"
        if weather == "暴雨":
            desc += "暴雨天户外项目停运,资源向室内场馆倾斜。"
        return desc
