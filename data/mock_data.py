"""
模拟数据生成器 —— 用于框架验证
真实项目接入后可直接废弃此文件
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from utils.date_utils import is_holiday, is_golden_week, is_weekend, get_season


def generate_mock_history(
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    park_capacity: int = 40000,
    base_price: float = 299.0,
    seed: int = 42,
) -> pd.DataFrame:
    """
    生成逼近真实的乐园历史销售数据
    列:
      date, price, visitors, revenue_ticket, revenue_secondary,
      weather, temperature, rainfall, is_holiday, is_weekend, day_type
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start_date, end_date, freq="D")
    n = len(dates)

    # === 1. 天气(温度、降水) ===
    doy = dates.dayofyear.values
    temperature = 15 + 12 * np.sin(2 * np.pi * (doy - 80) / 365) + rng.normal(0, 3, n)
    rainfall = np.clip(rng.gamma(0.8, 3, n) - 1.0, 0, None)
    # 夏季雨水增多
    summer_mask = (dates.month >= 6) & (dates.month <= 8)
    rainfall[summer_mask] *= 1.8

    # === 2. 标识日期类型 ===
    holiday_flag = np.array([is_holiday(d) for d in dates])
    weekend_flag = np.array([is_weekend(d) for d in dates])
    golden_week_flag = np.array([is_golden_week(d) for d in dates])

    # === 3. 模拟历史定价(带一定策略) ===
    price = np.full(n, base_price)
    price[weekend_flag] *= 1.08
    price[holiday_flag] *= 1.22
    price[golden_week_flag] *= 1.38
    # 淡季(12-2月非假期)适度降价
    low_season = ((dates.month == 12) | (dates.month <= 2)) & ~holiday_flag & ~weekend_flag
    price[low_season] *= 0.88
    price += rng.normal(0, 5, n)  # 市场噪声
    price = np.clip(price, 80, 599).round(0)

    # === 4. 模拟客流 —— 综合多因素 ===
    base_visitors = park_capacity * 0.45  # 日均基线
    # 日期因子
    date_factor = np.ones(n)
    date_factor[weekend_flag] = 1.55
    date_factor[holiday_flag] = 1.95
    date_factor[golden_week_flag] = 2.30
    # 天气因子
    weather_factor = np.ones(n)
    weather_factor[rainfall > 5] *= 0.65          # 雨天 -35%
    weather_factor[rainfall > 20] *= 0.75         # 暴雨再降
    weather_factor[temperature > 35] *= 0.80      # 高温 -20%
    weather_factor[temperature < 5] *= 0.85       # 低温 -15%
    # 价格弹性(当前价 vs 基准价)
    price_factor = (base_price / price) ** 0.8    # 弹性系数 0.8
    # 季节性长周期
    season_factor = 1 + 0.15 * np.sin(2 * np.pi * (doy - 100) / 365)

    visitors = base_visitors * date_factor * weather_factor * price_factor * season_factor
    visitors *= rng.normal(1.0, 0.08, n)  # 随机扰动
    visitors = np.clip(visitors, 500, park_capacity).astype(int)

    # === 5. 收入 ===
    revenue_ticket = visitors * price
    # 二消(餐饮/周边)—— 和客群结构弱相关,此处简化
    secondary_per_head = rng.normal(130, 20, n).clip(60, 260)
    revenue_secondary = (visitors * secondary_per_head).round(0)

    # === 6. 天气标签 ===
    weather_label = np.where(
        rainfall > 20, "暴雨",
        np.where(rainfall > 5, "雨",
        np.where(temperature > 35, "酷热",
        np.where(temperature < 5, "严寒", "晴好"))))

    day_type = np.where(golden_week_flag, "golden_week",
                np.where(holiday_flag, "holiday",
                np.where(weekend_flag, "weekend", "weekday")))

    df = pd.DataFrame({
        "date": dates,
        "price": price,
        "visitors": visitors,
        "revenue_ticket": revenue_ticket.round(0),
        "revenue_secondary": revenue_secondary,
        "revenue_total": (revenue_ticket + revenue_secondary).round(0),
        "weather": weather_label,
        "temperature": temperature.round(1),
        "rainfall": rainfall.round(1),
        "is_holiday": holiday_flag,
        "is_weekend": weekend_flag,
        "day_type": day_type,
        "season": [get_season(d) for d in dates],
        "load_rate": (visitors / park_capacity).round(3),
    })
    return df


def generate_external_signal(target_date: str, seed: int | None = None) -> dict:
    """
    生成某一天的外部信号(天气预报/竞品价/节假日标记)
    真实场景替换为天气API、竞品爬虫
    """
    rng = np.random.default_rng(seed)
    d = pd.to_datetime(target_date)

    return {
        "date": target_date,
        "weather_forecast": {
            "temperature_high": round(float(rng.normal(22, 8)), 1),
            "temperature_low": round(float(rng.normal(14, 6)), 1),
            "rainfall_mm": round(float(np.clip(rng.gamma(0.6, 3), 0, None)), 1),
            "rain_probability": round(float(rng.uniform(0, 1)), 2),
        },
        "competitor_prices": {
            "乐园A": round(float(rng.normal(310, 25)), 0),
            "乐园B": round(float(rng.normal(280, 20)), 0),
            "乐园C": round(float(rng.normal(350, 30)), 0),
        },
        "is_holiday": is_holiday(d.date()),
        "is_golden_week": is_golden_week(d.date()),
        "is_weekend": is_weekend(d.date()),
        "day_type": ("golden_week" if is_golden_week(d.date())
                     else "holiday" if is_holiday(d.date())
                     else "weekend" if is_weekend(d.date())
                     else "weekday"),
    }
