"""
全局配置 (基于 pydantic-settings)

支持从以下来源读取配置,优先级由低到高:
  1. 代码中的默认值
  2. .env 文件(项目根目录)
  3. 系统环境变量
  4. 运行时动态更新(settings.reload())

环境变量命名规范:
  PARK_NAME="xxx乐园"
  PARK_CAPACITY=40000
  PRICING__MIN_PRICE=80
  PRICING__MAX_PRICE=599
  WEATHER__RAIN_THRESHOLD_MM=5.0
  ALERT_LOAD_RATE_HIGH=0.90
"""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class PricingBounds(BaseModel):
    """定价边界约束"""

    min_price: float = Field(80.0, ge=1.0, le=10000.0)
    max_price: float = Field(599.0, ge=1.0, le=10000.0)
    base_price: float = Field(299.0, ge=1.0, le=10000.0)
    max_daily_change: float = Field(0.30, ge=0.0, le=1.0)

    @field_validator("max_price")
    @classmethod
    def _max_gt_min(cls, v, info):
        if "min_price" in info.data and v <= info.data["min_price"]:
            raise ValueError("max_price must be greater than min_price")
        return v


class WeatherConfig(BaseModel):
    rain_threshold_mm: float = Field(5.0, ge=0.0, le=500.0)
    heat_threshold_c: float = Field(35.0, ge=-50.0, le=60.0)
    cold_threshold_c: float = Field(5.0, ge=-50.0, le=60.0)
    rain_demand_coef: float = Field(-0.35, ge=-1.0, le=1.0)
    extreme_heat_coef: float = Field(-0.20, ge=-1.0, le=1.0)


class HolidayConfig(BaseModel):
    holiday_markup: float = Field(1.25, ge=0.1, le=3.0)
    weekend_markup: float = Field(1.10, ge=0.1, le=3.0)
    weekday_discount: float = Field(0.92, ge=0.1, le=3.0)
    golden_week_markup: float = Field(1.40, ge=0.1, le=3.0)


class CustomerSegment(BaseModel):
    name: str
    price_elasticity: float = Field(..., ge=-5.0, le=0.0)
    expected_share: float = Field(..., ge=0.0, le=1.0)


class Settings(BaseSettings):
    """全局配置对象"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
    )

    # 乐园基础信息
    park_name: str = "示例乐园"
    park_capacity: int = Field(40000, ge=1)
    optimal_load: float = Field(0.75, ge=0.0, le=1.0)
    park_city: str = "上海"

    pricing: PricingBounds = Field(default_factory=lambda: PricingBounds.model_validate({}))
    weather: WeatherConfig = Field(default_factory=lambda: WeatherConfig.model_validate({}))
    holiday: HolidayConfig = Field(default_factory=lambda: HolidayConfig.model_validate({}))

    segments: List[CustomerSegment] = Field(
        default_factory=lambda: [
            CustomerSegment(name="家庭亲子", price_elasticity=-0.8, expected_share=0.40),
            CustomerSegment(name="年轻客群", price_elasticity=-1.2, expected_share=0.30),
            CustomerSegment(name="团体游客", price_elasticity=-0.6, expected_share=0.20),
            CustomerSegment(name="商务VIP", price_elasticity=-0.3, expected_share=0.10),
        ]
    )

    secondary_consumption_ratio: float = Field(0.45, ge=0.0, le=5.0)

    # 预警阈值
    alert_load_rate_high: float = Field(0.90, ge=0.0, le=1.0)
    alert_load_rate_low: float = Field(0.30, ge=0.0, le=1.0)
    alert_revenue_drop: float = Field(-0.15, ge=-1.0, le=0.0)
    alert_competitor_cut: float = Field(-0.10, ge=-1.0, le=0.0)

    # 外部API
    weather_api_url: str = "https://api.example-weather.com/v1/forecast"
    weather_provider: str = "qweather"
    competitor_api_url: str = "https://api.example-competitor.com/prices"
    holiday_calendar_path: str = "data/holidays_cn.json"

    # 性能配置
    feature_cache_ttl_seconds: int = Field(3600, ge=0)
    api_timeout_seconds: float = Field(30.0, ge=0.1)
    threadpool_max_workers: int = Field(8, ge=1, le=128)

    # 风险控制
    uncertainty_spread_threshold: float = Field(0.40, ge=0.0, le=5.0)

    # 分布式缓存 / 消息队列 (Redis)
    redis_enabled: bool = Field(True)
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "ai_pricing"
    redis_stream_name: str = "pricing.anomaly.events"
    redis_socket_timeout_seconds: float = Field(1.5, ge=0.1, le=30.0)

    # 推理服务解耦开关 (可选)
    inference_service_url: str = ""
    inference_service_timeout_seconds: float = Field(3.0, ge=0.1, le=120.0)

    # 运行环境
    env: str = "dev"
    log_level: str = "INFO"

    @property
    def alert_thresholds(self) -> Dict[str, float]:
        return {
            "load_rate_high": self.alert_load_rate_high,
            "load_rate_low": self.alert_load_rate_low,
            "revenue_drop": self.alert_revenue_drop,
            "competitor_cut": self.alert_competitor_cut,
        }

    def reload(self):
        """热更新: 重读环境变量与 .env 文件"""
        fresh = Settings.model_validate({})
        for field_name in fresh.model_fields.keys():
            setattr(self, field_name, getattr(fresh, field_name))
        return self


settings = Settings.model_validate({})
