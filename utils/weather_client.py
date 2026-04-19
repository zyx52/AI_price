"""
真实天气API客户端

支持三个主流天气服务:
  1. 和风天气 (QWeather)     —— 国内乐园首选,免费额度高,中文描述准确
  2. OpenWeather              —— 国际通用
  3. 中国天气网 (weatherapi)  —— 备用

使用:
  export QWEATHER_API_KEY="your_key"  # 推荐

  from utils.weather_client import WeatherClient
  client = WeatherClient(provider="qweather")
  forecast = client.get_forecast(location="上海", days=7)

所有调用结果会统一成标准 schema,便于上层模块消费。
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime

from utils.logger import get_logger

logger = get_logger("WeatherClient")

try:
    import requests as _requests
    requests: Any = _requests
    _HAS_REQUESTS = True
except ImportError:
    requests: Any = None
    _HAS_REQUESTS = False
    logger.warning("requests 未安装,WeatherClient 不可用")


# ============================================================
# 统一数据结构
# ============================================================
@dataclass
class DailyForecast:
    """标准化的日度预报数据"""
    date: str                      # YYYY-MM-DD
    temperature_high: float        # ℃
    temperature_low: float
    rainfall_mm: float             # 降水量
    rain_probability: float        # 0-1
    weather_text: str              # "多云" / "雷阵雨"
    wind_speed_kmh: float
    humidity: float                # 0-1

    # 派生标签(与定价引擎对齐)
    weather_label: str = ""        # "晴好" / "雨" / "暴雨" / "酷热" / "严寒"

    def __post_init__(self):
        if not self.weather_label:
            self.weather_label = self._compute_label()

    def _compute_label(self) -> str:
        if self.rainfall_mm > 20:
            return "暴雨"
        if self.rainfall_mm > 5 or self.rain_probability > 0.6:
            return "雨"
        if self.temperature_high > 35:
            return "酷热"
        if self.temperature_high < 5:
            return "严寒"
        return "晴好"

    def to_dict(self):
        return asdict(self)


# ============================================================
# 基类
# ============================================================
class BaseWeatherClient:
    """天气客户端基类"""
    def get_forecast(self, location: str, days: int = 7) -> List[DailyForecast]:
        raise NotImplementedError

    def get_current(self, location: str) -> DailyForecast:
        return self.get_forecast(location, days=1)[0]


# ============================================================
# 和风天气 (QWeather)
# 文档: https://dev.qweather.com/docs/api/
# ============================================================
class QWeatherClient(BaseWeatherClient):
    """和风天气客户端"""

    BASE = "https://devapi.qweather.com/v7"   # 免费版
    # 付费商业版: "https://api.qweather.com/v7"
    GEO = "https://geoapi.qweather.com/v2/city/lookup"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("QWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "需要和风天气 API Key。请到 https://dev.qweather.com/ 免费申请后,"
                "设置环境变量 QWEATHER_API_KEY"
            )

    def _lookup_location_id(self, location: str) -> str:
        """将城市名转为和风天气的 Location ID"""
        resp = requests.get(
            self.GEO,
            params={"location": location, "key": self.api_key},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != "200":
            raise RuntimeError(f"地理编码失败: {data}")
        return data["location"][0]["id"]

    def get_forecast(self, location: str, days: int = 7) -> List[DailyForecast]:
        loc_id = self._lookup_location_id(location)
        endpoint = f"{self.BASE}/weather/{days}d"
        resp = requests.get(
            endpoint, params={"location": loc_id, "key": self.api_key}, timeout=10,
        )
        data = resp.json()
        if data.get("code") != "200":
            raise RuntimeError(f"天气查询失败: {data}")

        results = []
        for day in data["daily"]:
            results.append(DailyForecast(
                date=day["fxDate"],
                temperature_high=float(day["tempMax"]),
                temperature_low=float(day["tempMin"]),
                rainfall_mm=float(day.get("precip", 0)),
                rain_probability=self._probability_from_text(day.get("textDay", "")),
                weather_text=day.get("textDay", ""),
                wind_speed_kmh=float(day.get("windSpeedDay", 0)),
                humidity=float(day.get("humidity", 60)) / 100,
            ))
        return results

    @staticmethod
    def _probability_from_text(text: str) -> float:
        """文字描述 → 降雨概率(和风免费版不直接给概率)"""
        if any(k in text for k in ["暴雨", "大雨"]): return 0.95
        if any(k in text for k in ["中雨", "小雨", "阵雨"]): return 0.80
        if any(k in text for k in ["雷阵雨"]): return 0.75
        if any(k in text for k in ["多云", "阴"]): return 0.25
        return 0.05


# ============================================================
# OpenWeather
# 文档: https://openweathermap.org/api
# ============================================================
class OpenWeatherClient(BaseWeatherClient):
    """OpenWeather One Call API 客户端"""

    BASE = "https://api.openweathermap.org/data/2.5"
    GEO = "https://api.openweathermap.org/geo/1.0/direct"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("需要 OpenWeather API Key (OPENWEATHER_API_KEY)")

    def _geocode(self, location: str) -> tuple:
        resp = requests.get(
            self.GEO, params={"q": location, "limit": 1, "appid": self.api_key},
            timeout=10,
        )
        results = resp.json()
        if not results:
            raise RuntimeError(f"找不到城市: {location}")
        return results[0]["lat"], results[0]["lon"]

    def get_forecast(self, location: str, days: int = 7) -> List[DailyForecast]:
        lat, lon = self._geocode(location)
        resp = requests.get(
            f"{self.BASE}/forecast",
            params={"lat": lat, "lon": lon, "appid": self.api_key,
                    "units": "metric", "cnt": min(days * 8, 40)},
            timeout=10,
        )
        data = resp.json()
        if data.get("cod") != "200":
            raise RuntimeError(f"天气查询失败: {data}")

        # 按日期聚合(OpenWeather免费版是3小时粒度)
        daily: Dict[str, dict] = {}
        for item in data["list"]:
            date = item["dt_txt"].split(" ")[0]
            d = daily.setdefault(date, {
                "temps": [], "rain": 0, "pop": [], "text": "",
                "wind": [], "humid": [],
            })
            d["temps"].append(item["main"]["temp"])
            d["rain"] += item.get("rain", {}).get("3h", 0)
            d["pop"].append(item.get("pop", 0))
            d["wind"].append(item["wind"]["speed"] * 3.6)
            d["humid"].append(item["main"]["humidity"] / 100)
            if not d["text"]:
                d["text"] = item["weather"][0]["description"]

        results = []
        for date, d in list(daily.items())[:days]:
            results.append(DailyForecast(
                date=date,
                temperature_high=max(d["temps"]),
                temperature_low=min(d["temps"]),
                rainfall_mm=d["rain"],
                rain_probability=max(d["pop"]) if d["pop"] else 0,
                weather_text=d["text"],
                wind_speed_kmh=sum(d["wind"]) / len(d["wind"]),
                humidity=sum(d["humid"]) / len(d["humid"]),
            ))
        return results


# ============================================================
# 统一入口
# ============================================================
class WeatherClient:
    """天气客户端工厂"""

    def __init__(self, provider: str = "qweather", api_key: Optional[str] = None):
        if not _HAS_REQUESTS:
            raise ImportError("请安装 requests: pip install requests")
        self.provider = provider
        if provider == "qweather":
            self.client = QWeatherClient(api_key)
        elif provider == "openweather":
            self.client = OpenWeatherClient(api_key)
        else:
            raise ValueError(f"未知provider: {provider}. 可选: qweather / openweather")

    def get_forecast(self, location: str, days: int = 7) -> List[DailyForecast]:
        return self.client.get_forecast(location, days)

    def get_current(self, location: str) -> DailyForecast:
        return self.client.get_current(location)

    def to_pricing_signal(self, forecast: DailyForecast) -> dict:
        """
        转换为定价引擎可消费的字典格式
        与 data/mock_data.py 的 generate_external_signal 输出保持一致
        """
        return {
            "date": forecast.date,
            "weather_forecast": {
                "temperature_high": forecast.temperature_high,
                "temperature_low": forecast.temperature_low,
                "rainfall_mm": forecast.rainfall_mm,
                "rain_probability": forecast.rain_probability,
                "weather_text": forecast.weather_text,
                "weather_label": forecast.weather_label,
            },
        }


# ============================================================
# 使用示例(也是 data_loader 接入模板)
# ============================================================
def demo():
    """演示如何在 data_loader 中接入真实天气"""
    # 1) 设置 API Key
    # export QWEATHER_API_KEY="xxx"

    # 2) 创建客户端
    client = WeatherClient(provider="qweather")

    # 3) 获取7天预报
    forecasts = client.get_forecast(location="上海", days=7)

    # 4) 转为定价信号
    for f in forecasts:
        signal = client.to_pricing_signal(f)
        print(f"{f.date}: {f.weather_label} | 最高{f.temperature_high}℃ | 降水{f.rainfall_mm}mm")
        # 把 signal 传入 PricingEngine.decide(...) 即可


if __name__ == "__main__":
    demo()
