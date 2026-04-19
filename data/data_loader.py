"""
数据加载器 —— 统一的数据接入接口

这一层是【真实数据接入点】,所有 TODO 都是需要你替换的位置。
目前使用模拟数据,保证框架可跑通。

接入真实数据时只需:
  1. 实现对应的 _load_from_xxx() 方法
  2. 修改 source 参数或配置文件
其他模块无需改动。
"""
import pandas as pd
from typing import Optional
from pathlib import Path

from utils.logger import get_logger
from config import settings
from .mock_data import generate_mock_history, generate_external_signal

logger = get_logger("DataLoader")


class DataLoader:
    """数据加载器 —— 支持多种数据源"""

    def __init__(self, source: str = "mock"):
        """
        source: 'mock' | 'csv' | 'database' | 'api'
        """
        self.source = source

    # ---------- 历史销售数据 ----------
    def load_history(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """加载历史销售数据 —— 定价模型的训练数据来源"""
        logger.info(f"加载历史数据 | source={self.source}")

        if self.source == "mock":
            df = generate_mock_history(
                start_date=start_date or "2024-01-01",
                end_date=end_date or "2025-12-31",
                park_capacity=settings.park_capacity,
                base_price=settings.pricing.base_price,
            )
        elif self.source == "csv":
            df = self._load_from_csv("data/raw/history.csv")
        elif self.source == "database":
            df = self._load_from_db(start_date, end_date)
        elif self.source == "api":
            df = self._load_from_api(start_date, end_date)
        else:
            raise ValueError(f"未知数据源: {self.source}")

        logger.info(f"历史数据加载完成 | rows={len(df)}")
        return df

    # ---------- 外部信号 ----------
    def load_external_signal(self, target_date: str) -> dict:
        """加载目标日期的外部信号(天气/竞品/节假日)"""
        if self.source == "mock":
            return generate_external_signal(target_date)
        return {
            "date": target_date,
            "weather_forecast": self._fetch_weather(target_date),
            "competitor_prices": self._fetch_competitor_prices(target_date),
            # ...
        }

    # ==========================================================
    # 以下方法为【真实数据接入点】——请在上线前实现
    # ==========================================================

    def _load_from_csv(self, path: str) -> pd.DataFrame:
        # TODO: 接入真实CSV
        p = Path(path)
        if not p.exists():
            logger.warning(f"CSV不存在,fallback到mock: {path}")
            return generate_mock_history()
        return pd.read_csv(p, parse_dates=["date"])

    def _load_from_db(self, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        # TODO: 接入真实数据库(MySQL/PostgreSQL/Hive)
        # 示例:
        # import sqlalchemy
        # engine = sqlalchemy.create_engine(settings.db_url)
        # sql = f"SELECT * FROM park_sales WHERE date BETWEEN '{start}' AND '{end}'"
        # return pd.read_sql(sql, engine)
        raise NotImplementedError("请在 _load_from_db 中接入真实数据库")

    def _load_from_api(self, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
        # TODO: 接入真实API (企业内部BI API / 订单系统API)
        raise NotImplementedError("请在 _load_from_api 中接入真实API")

    def _fetch_weather(self, target_date: str) -> dict:
        """
        接入真实天气API
        需要环境变量 QWEATHER_API_KEY 或 OPENWEATHER_API_KEY
        """
        from utils.weather_client import WeatherClient
        # 默认上海,实际项目从 settings 读取园区所在城市
        city = getattr(settings, "park_city", "上海")
        provider = getattr(settings, "weather_provider", "qweather")
        client = WeatherClient(provider=provider)
        forecasts = client.get_forecast(location=city, days=1)
        target = forecasts[0]
        return {
            "temperature_high": target.temperature_high,
            "temperature_low": target.temperature_low,
            "rainfall_mm": target.rainfall_mm,
            "rain_probability": target.rain_probability,
            "weather_text": target.weather_text,
            "weather_label": target.weather_label,
        }

    def _fetch_competitor_prices(self, target_date: str) -> dict:
        # TODO: 接入竞品价格爬虫/第三方服务
        raise NotImplementedError("请接入竞品价格服务")
