"""
市场监控与预警引擎

结合外部数据(天气/竞品/节假日)+内部数据(客流/收入),
识别潜在市场变化与风险,输出分级预警。

预警级别:
  🔴 critical  —— 必须立即响应
  🟡 warning   —— 需要关注
  🔵 info      —— 运营提示
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd

from utils.logger import get_logger
from config import settings

logger = get_logger("AlertEngine")


@dataclass
class Alert:
    level: str            # 'critical' | 'warning' | 'info'
    category: str         # 'capacity' | 'revenue' | 'weather' | 'competitor'
    title: str
    message: str
    suggested_action: str

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(self.level, "⚪")
        return f"{icon} [{self.category}] {self.title} —— {self.message}"


class AlertEngine:
    """外部+内部信号融合预警引擎"""

    def check(
        self,
        external: dict,
        predicted_visitors: float,
        predicted_revenue: float,
        history: Optional[pd.DataFrame] = None,
    ) -> List[Alert]:
        alerts: List[Alert] = []
        th = settings.alert_thresholds

        # === 1. 容量预警 ===
        load_rate = predicted_visitors / settings.park_capacity
        if load_rate >= th["load_rate_high"]:
            alerts.append(Alert(
                level="critical",
                category="capacity",
                title="客流预警·高负载",
                message=f"预计负载率 {load_rate*100:.1f}% 超过{th['load_rate_high']*100:.0f}%,存在拥堵与体验下降风险",
                suggested_action="启动分时预约、提高票价上浮幅度、加开人工引导通道",
            ))
        elif load_rate <= th["load_rate_low"]:
            alerts.append(Alert(
                level="warning",
                category="capacity",
                title="客流预警·低利用",
                message=f"预计负载率仅 {load_rate*100:.1f}%,园区利用率偏低",
                suggested_action="推出当日促销(早鸟/学生/家庭套票),联动OTA/本地生活平台引流",
            ))

        # === 2. 天气预警 ===
        weather = external.get("weather_forecast", {})
        rainfall = weather.get("rainfall_mm", 0)
        rain_prob = weather.get("rain_probability", 0)
        temp_high = weather.get("temperature_high", 20)

        if rainfall > 15 or (rain_prob > 0.7 and rainfall > 5):
            alerts.append(Alert(
                level="warning",
                category="weather",
                title="天气预警·强降雨",
                message=f"降雨量预计{rainfall:.1f}mm,降雨概率{rain_prob*100:.0f}%",
                suggested_action="上线【雨天特惠】套餐,室内项目优先引导,准备应急雨具",
            ))
        if temp_high >= settings.weather.heat_threshold_c:
            alerts.append(Alert(
                level="warning",
                category="weather",
                title="天气预警·极端高温",
                message=f"最高气温{temp_high:.1f}℃,预计户外客流下降",
                suggested_action="开放夜场优惠、强化遮阳与饮水点、推广水上项目",
            ))

        # === 3. 竞品预警 ===
        comp = external.get("competitor_prices", {})
        if comp:
            comp_avg = sum(comp.values()) / len(comp)
            base = settings.pricing.base_price
            comp_delta = (comp_avg - base) / base
            if comp_delta <= th["competitor_cut"]:
                alerts.append(Alert(
                    level="warning",
                    category="competitor",
                    title="竞品预警·降价冲击",
                    message=f"周边竞品均价¥{comp_avg:.0f},低于本园{abs(comp_delta)*100:.1f}%",
                    suggested_action="评估价格弹性,考虑推出差异化套餐(体验/服务而非纯降价)",
                ))

        # === 4. 收入同比预警 ===
        if history is not None and len(history) > 30:
            recent = history.tail(7)["revenue_total"].mean()
            prior = history.tail(30).head(23)["revenue_total"].mean()
            drop = (recent - prior) / prior if prior > 0 else 0
            if drop <= th["revenue_drop"]:
                alerts.append(Alert(
                    level="critical",
                    category="revenue",
                    title="收入预警·近期下滑",
                    message=f"近7日平均收入较前3周下降{abs(drop)*100:.1f}%",
                    suggested_action="启动收入归因分析(客流/客单/二消),评估是否进入淡季或竞争加剧",
                ))

        # === 5. 无预警兜底信息 ===
        if not alerts:
            alerts.append(Alert(
                level="info",
                category="capacity",
                title="运营状态正常",
                message="当前预测各项指标处于健康区间",
                suggested_action="维持当前定价与套餐策略,持续监控",
            ))

        # 按级别排序
        order = {"critical": 0, "warning": 1, "info": 2}
        alerts.sort(key=lambda a: order.get(a.level, 9))
        return alerts
