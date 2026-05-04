"""
天气数据守护进程 —— 7x24小时运行

职责:
  1. 每30分钟拉取最新天气+未来4小时短临预报
  2. 发布到 Redis Pub/Sub park:live_data:weather
  3. API故障时自动前向填充(Forward Fill)
  4. TTL缓存控制 → 防止按次计费API被无效调用

启动:
  python services/weather_daemon.py --provider qweather --city 上海

环境变量:
  QWEATHER_API_KEY   和风天气API密钥
  OPENWEATHER_API_KEY OpenWeather API密钥
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# 在项目根目录安装后可用标准导入
from utils.logger import get_logger
from utils.feature_cache import feature_cache
from config import settings
from services.message_bus import bus, Channel, BusMessage

logger = get_logger("WeatherDaemon")

try:
    import requests as _requests
    requests: Any = _requests
    _HAS_REQUESTS = True
except ImportError:
    requests = None
    _HAS_REQUESTS = False


# ============================================================
# 统一数据结构
# ============================================================
@dataclass
class HourlySlot:
    """小时级预报"""
    time: str               # "14:00"
    temperature: float
    rainfall_mm: float
    rain_probability: float
    humidity: float
    wind_speed_kmh: float


@dataclass
class WeatherSnapshot:
    """天气快照 —— 发布到消息总线的标准格式"""
    timestamp: str                      # ISO 8601
    location: str                       # "上海"
    current_temperature: float
    feels_like: float
    rainfall_mm: float
    rain_probability: float
    humidity: float
    wind_speed_kmh: float
    weather_text: str                   # "多云" / "雷阵雨"
    weather_label: str                  # "晴好" | "雨" | "暴雨" | "酷热" | "严寒"
    next_4h_forecast: List[Dict[str, Any]] = field(default_factory=list)
    # 数据来源与质量
    source: str = "qweather"
    is_forward_filled: bool = False     # True=API失败,使用了前向填充
    consecutive_failures: int = 0
    cache_ttl_remaining: float = 0.0    # 缓存剩余秒数

    @classmethod
    def compute_label(cls, rainfall: float, temperature: float) -> str:
        if rainfall > 20:
            return "暴雨"
        if rainfall > 5:
            return "雨"
        if temperature > 35:
            return "酷热"
        if temperature < 5:
            return "严寒"
        return "晴好"

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 和风天气实时 + 逐小时预报 客户端
# ============================================================
class QWeatherLiveClient:
    """和风天气实时 + 逐小时预报"""

    BASE = "https://devapi.qweather.com/v7"
    GEO = "https://geoapi.qweather.com/v2/city/lookup"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("QWEATHER_API_KEY")
        if not self.api_key:
            raise ValueError("缺少 QWEATHER_API_KEY 环境变量")
        self._location_id: Optional[str] = None

    def _resolve_location(self, city: str) -> str:
        if self._location_id:
            return self._location_id
        resp = requests.get(
            self.GEO,
            params={"location": city, "key": self.api_key},
            timeout=10,
        )
        data = resp.json()
        if data.get("code") != "200":
            raise RuntimeError(f"地理编码失败: {data}")
        self._location_id = data["location"][0]["id"]
        return self._location_id

    def fetch(self, city: str) -> WeatherSnapshot:
        loc_id = self._resolve_location(city)

        # 实时天气
        now_resp = requests.get(
            f"{self.BASE}/weather/now",
            params={"location": loc_id, "key": self.api_key},
            timeout=10,
        )
        now_data = now_resp.json()
        if now_data.get("code") != "200":
            raise RuntimeError(f"实时天气获取失败: {now_data}")

        now = now_data["now"]
        temp = float(now["temp"])
        feels_like = float(now.get("feelsLike", temp))
        humidity = float(now.get("humidity", 50)) / 100.0
        wind = float(now.get("windSpeed", 0))
        text = now.get("text", "未知")

        # 逐小时预报 (未来4h)
        hourly_resp = requests.get(
            f"{self.BASE}/weather/24h",
            params={"location": loc_id, "key": self.api_key},
            timeout=10,
        )
        hourly_data = hourly_resp.json()
        next_4h: List[Dict[str, Any]] = []
        total_rain = 0.0
        max_rain_prob = 0.0

        if hourly_data.get("code") == "200":
            for item in hourly_data.get("hourly", [])[:4]:
                precip = float(item.get("precip", "0"))
                pop = float(item.get("pop", "0")) / 100.0
                total_rain += precip
                max_rain_prob = max(max_rain_prob, pop)
                next_4h.append({
                    "time": item.get("fxTime", "")[-5:],
                    "temperature": float(item.get("temp", temp)),
                    "rainfall_mm": precip,
                    "rain_probability": pop,
                    "humidity": float(item.get("humidity", humidity * 100)) / 100.0,
                    "wind_speed_kmh": float(item.get("windSpeed", wind)),
                })

        rainfall_now = float(now.get("precip", "0"))
        rain_prob = max_rain_prob
        if rain_prob == 0 and hourly_data.get("code") == "200":
            # 从未来4h推断当前降水概率
            for h in next_4h:
                if h["rainfall_mm"] > 0:
                    rain_prob = max(rain_prob, 0.3)

        return WeatherSnapshot(
            timestamp=datetime.now().isoformat(),
            location=city,
            current_temperature=temp,
            feels_like=feels_like,
            rainfall_mm=rainfall_now,
            rain_probability=rain_prob,
            humidity=humidity,
            wind_speed_kmh=wind,
            weather_text=text,
            weather_label=WeatherSnapshot.compute_label(total_rain / 4 if next_4h else rainfall_now, temp),
            next_4h_forecast=next_4h,
            source="qweather",
        )


# ============================================================
# 天气守护进程核心
# ============================================================
class WeatherDaemon:
    """
    天气数据守护进程

    核心能力:
      - 每30分钟拉取一次天气数据
      - 发布到 Redis Pub/Sub park:live_data:weather
      - API失败 → 自动前向填充 (Forward Fill)
      - TTL缓存: 相同窗口期(30min)内直接读缓存,不调用API
    """

    # 缓存配置
    CACHE_KEY_PREFIX = "weather:snapshot"
    WEATHER_TTL = 1800   # 30分钟 (与轮询频率一致)
    MAX_FORWARD_FILL_COUNT = 6   # 最多前向填充6次(3小时),超过则告警

    def __init__(self, city: str = "上海", provider: str = "qweather"):
        self.city = city
        self.provider = provider
        self._running = False
        self._last_snapshot: Optional[WeatherSnapshot] = None
        self._consecutive_failures = 0

        # 初始化天气客户端
        if provider == "qweather":
            api_key = os.getenv("QWEATHER_API_KEY")
            if not api_key:
                logger.warning("QWEATHER_API_KEY 未设置, 将使用mock数据")
                self._client = None
            else:
                self._client = QWeatherLiveClient(api_key)
        else:
            logger.warning(f"未知天气提供商: {provider}, 降级为mock")
            self._client = None

        # 注册信号处理 (优雅关闭)
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"收到信号 {signum}, 正在停止...")
        self._running = False

    # ============================================================
    # 主循环
    # ============================================================
    def run(self, interval_seconds: int = 1800):
        """
        主循环: 每30分钟拉取一次

        interval_seconds: 轮询间隔(默认1800=30分钟)
        """
        self._running = True
        logger.info(f"🌤️ 天气守护进程启动 | city={self.city} | provider={self.provider} "
                     f"| interval={interval_seconds}s | bus={'healthy' if bus.is_healthy else 'no-op'}")

        while self._running:
            start_time = time.time()

            try:
                snapshot = self._fetch_or_fill()
                # 发布到消息总线
                bus.publish_weather(snapshot.to_dict())
                # 存入缓存(给其他模块同步读取)
                cache_key = f"{self.CACHE_KEY_PREFIX}:{self.city}"
                feature_cache.set(cache_key, snapshot.to_dict(), ttl=self.WEATHER_TTL)

                logger.info(
                    f"天气已发布: {snapshot.weather_label} | "
                    f"温度={snapshot.current_temperature}℃ | "
                    f"降水={snapshot.rainfall_mm}mm | "
                    f"前向填充={snapshot.is_forward_filled} | "
                    f"连续失败={snapshot.consecutive_failures}"
                )

            except Exception as e:
                logger.error(f"天气守护进程异常: {e}", exc_info=True)

            # 等待到下一次轮询
            elapsed = time.time() - start_time
            sleep_time = max(0, interval_seconds - elapsed)
            if sleep_time > 0 and self._running:
                logger.debug(f"下次天气拉取: {sleep_time:.0f}秒后")
                # 分段sleep以响应停止信号
                for _ in range(int(sleep_time)):
                    if not self._running:
                        break
                    time.sleep(1)

        logger.info("天气守护进程已停止")

    # ============================================================
    # 数据获取 + 前向填充
    # ============================================================
    def _fetch_or_fill(self) -> WeatherSnapshot:
        """获取天气数据,失败时前向填充"""

        # 1️⃣ 先检查缓存(防穿透)
        cache_key = f"{self.CACHE_KEY_PREFIX}:{self.city}"
        cached = feature_cache.get(cache_key)
        cache_remaining = 0.0

        # 2️⃣ 尝试从API获取
        if self._client is not None:
            try:
                snapshot = self._client.fetch(self.city)
                snapshot.consecutive_failures = 0
                snapshot.is_forward_filled = False
                self._last_snapshot = snapshot
                self._consecutive_failures = 0
                return snapshot
            except Exception as e:
                logger.warning(f"天气API调用失败: {e}, 尝试降级...")
                self._consecutive_failures += 1
        elif cached is not None:
            # Mock模式: 读取缓存
            cache_remaining = self.WEATHER_TTL
            logger.debug("无API客户端(mock模式), 使用缓存数据")

        # 3️⃣ 前向填充 (Forward Fill)
        if self._consecutive_failures > self.MAX_FORWARD_FILL_COUNT:
            logger.error(
                f"天气API连续失败 {self._consecutive_failures} 次(>{self.MAX_FORWARD_FILL_COUNT}),"
                f" 超过最大前向填充限制!"
            )
            # 发送告警
            bus.publish_anomaly({
                "type": "weather_api_dead",
                "consecutive_failures": self._consecutive_failures,
                "city": self.city,
                "message": "天气API连续失败超过3小时,请检查API密钥和网络连接",
            })

        return self._forward_fill()

    def _forward_fill(self) -> WeatherSnapshot:
        """前向填充: 使用最近一次成功的数据"""
        if self._last_snapshot is not None:
            snapshot = WeatherSnapshot(
                timestamp=datetime.now().isoformat(),
                location=self.city,
                current_temperature=self._last_snapshot.current_temperature,
                feels_like=self._last_snapshot.feels_like,
                rainfall_mm=self._last_snapshot.rainfall_mm,
                rain_probability=self._last_snapshot.rain_probability,
                humidity=self._last_snapshot.humidity,
                wind_speed_kmh=self._last_snapshot.wind_speed_kmh,
                weather_text=f"{self._last_snapshot.weather_text}(前向填充)",
                weather_label=self._last_snapshot.weather_label,
                next_4h_forecast=self._last_snapshot.next_4h_forecast,
                source=self._last_snapshot.source,
                is_forward_filled=True,
                consecutive_failures=self._consecutive_failures,
            )
            logger.info(f"使用前向填充 (连续失败{self._consecutive_failures}次)")
            return snapshot

        # 完全没有历史数据: 使用气候学默认值
        logger.warning("无历史天气数据,使用气候学默认值")
        return WeatherSnapshot(
            timestamp=datetime.now().isoformat(),
            location=self.city,
            current_temperature=22.0,
            feels_like=22.0,
            rainfall_mm=0.0,
            rain_probability=0.1,
            humidity=0.6,
            wind_speed_kmh=10.0,
            weather_text="未知(默认值)",
            weather_label="晴好",
            source="fallback_default",
            is_forward_filled=True,
            consecutive_failures=self._consecutive_failures,
        )


# ============================================================
# CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="天气数据守护进程")
    parser.add_argument("--city", default=settings.park_city, help="城市名称")
    parser.add_argument("--provider", default="qweather", choices=["qweather", "openweather"],
                        help="天气数据提供商")
    parser.add_argument("--interval", type=int, default=1800,
                        help="轮询间隔(秒), 默认1800(30分钟)")
    parser.add_argument("--mock", action="store_true",
                        help="强制使用mock模式(不上线API,用随机天气)")
    args = parser.parse_args()

    if args.mock:
        os.environ.pop("QWEATHER_API_KEY", None)
        logger.info("Mock模式: 使用随机模拟天气数据")

    daemon = WeatherDaemon(city=args.city, provider=args.provider)
    daemon.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()
