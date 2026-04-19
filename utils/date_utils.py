"""
日期工具 —— 判断节假日、周末、季节
后续接入真实节假日API即可替换 is_holiday 实现
"""
from datetime import date, datetime
from typing import Union

DateLike = Union[date, datetime, str]


# 2026年中国法定节假日(示例,真实项目应读取 data/holidays_cn.json)
_HOLIDAYS_2026 = {
    # 元旦
    "2026-01-01",
    # 春节
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-21", "2026-02-22",
    # 清明
    "2026-04-04", "2026-04-05", "2026-04-06",
    # 劳动节
    "2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04", "2026-05-05",
    # 端午
    "2026-06-19", "2026-06-20", "2026-06-21",
    # 中秋
    "2026-09-25", "2026-09-26", "2026-09-27",
    # 国庆(黄金周)
    "2026-10-01", "2026-10-02", "2026-10-03", "2026-10-04",
    "2026-10-05", "2026-10-06", "2026-10-07",
}

_GOLDEN_WEEK = {
    "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19",
    "2026-02-20", "2026-02-21", "2026-02-22",
    "2026-10-01", "2026-10-02", "2026-10-03",
    "2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07",
}


def _to_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return datetime.strptime(d, "%Y-%m-%d").date()


def is_holiday(d: DateLike) -> bool:
    """是否法定节假日"""
    return _to_date(d).isoformat() in _HOLIDAYS_2026


def is_golden_week(d: DateLike) -> bool:
    """是否黄金周(春节/国庆)"""
    return _to_date(d).isoformat() in _GOLDEN_WEEK


def is_weekend(d: DateLike) -> bool:
    return _to_date(d).weekday() >= 5


def get_season(d: DateLike) -> str:
    """返回 spring/summer/autumn/winter"""
    m = _to_date(d).month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"


def get_day_type(d: DateLike) -> str:
    """
    返回综合日期类型,供定价引擎使用
    优先级: 黄金周 > 节假日 > 周末 > 工作日
    """
    if is_golden_week(d):
        return "golden_week"
    if is_holiday(d):
        return "holiday"
    if is_weekend(d):
        return "weekend"
    return "weekday"
