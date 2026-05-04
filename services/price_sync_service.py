"""
价格同步服务 —— 分布式事务 + 最终一致性保证

核心职责:
  1. 将调价指令异步推送到所有激活渠道
  2. 失败重试: 指数退避 (1min → 3min → 10min → 30min)
  3. 重试3次仍失败 → 发送最高级别告警
  4. 价格对账单: 完整记录 老价格→新价格→各渠道ACK
  5. Redis冷却期锁: 基础门票每2小时最多调价一次
  6. 最小变动阈值: <5%变动不触发下发

架构:
  PricingEngine → PriceSyncService.decide()
    ├── 冷却期检查 (Redis TTL锁)
    ├── 最小变动阈值检查 (防抖)
    ├── 构建 PricePushRequest[]
    ├── 并行推送到各 Adapter
    ├── 记录对账单 (audit log)
    └── 失败→重试队列 → 告警
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from utils.logger import get_logger
from utils.feature_cache import feature_cache
from config import settings
from services.message_bus import bus, Channel

logger = get_logger("PriceSyncService")

try:
    import redis as _redis_lib
    _HAS_REDIS = True
except ImportError:
    _redis_lib = None
    _HAS_REDIS = False

from adapters import (
    BaseChannelAdapter, PricePushRequest, PricePushResult,
    create_default_adapters, get_adapter, list_adapters,
)


# ============================================================
# 数据结构
# ============================================================
@dataclass
class PriceChangeRecord:
    """价格变更对账记录"""
    record_id: str
    timestamp: str
    base_price_old: float
    base_price_new: float
    change_pct: float
    reason: str
    # 各渠道推送结果
    channel_results: Dict[str, dict] = field(default_factory=dict)
    # 总体状态
    all_success: bool = False
    failed_channels: List[str] = field(default_factory=list)
    retry_pending: bool = False
    escalated_to_human: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetryTask:
    """重试任务"""
    request: PricePushRequest
    adapter_name: str
    attempt: int = 0
    next_retry_at: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================
# 重试编排器
# ============================================================
class RetryScheduler:
    """
    指数退避重试编排器

    退避序列: 60s → 180s → 600s → 1800s (1分钟→3分钟→10分钟→30分钟)
    最多重试 3 次, 第3次失败后触发告警升级
    """

    MAX_RETRIES = 3
    BACKOFF_SEQUENCE = [60, 180, 600, 1800]  # 秒

    def __init__(self):
        self._pending: List[RetryTask] = []
        self._lock = threading.Lock()
        self._alert_callback: Optional[Callable] = None

    def set_alert_callback(self, cb: Callable):
        self._alert_callback = cb

    def enqueue(self, request, adapter_name: str, attempt: int = 0):
        delay = self.BACKOFF_SEQUENCE[min(attempt, len(self.BACKOFF_SEQUENCE) - 1)]
        task = RetryTask(
            request=request,
            adapter_name=adapter_name,
            attempt=attempt,
            next_retry_at=time.monotonic() + delay,
        )
        with self._lock:
            self._pending.append(task)
        logger.info(f"[Retry] 入队: {adapter_name} attempt={attempt+1}/{self.MAX_RETRIES} "
                     f"下次重试={delay}s后")

    def process_due(self) -> List[Tuple[RetryTask, bool]]:
        """处理所有到期的重试任务, 返回 [(task, is_final_failure)]"""
        now = time.monotonic()
        due: List[RetryTask] = []

        with self._lock:
            remaining = []
            for t in self._pending:
                if t.next_retry_at <= now:
                    due.append(t)
                else:
                    remaining.append(t)
            self._pending = remaining

        results = []
        for task in due:
            adapter = get_adapter(task.adapter_name)
            if adapter is None:
                continue

            result = adapter.push_price(task.request)
            result.retry_count = task.attempt + 1

            if result.success:
                logger.info(f"[Retry] ✅ {task.adapter_name} 重试成功 "
                             f"(attempt={task.attempt+1})")
                results.append((task, False))
            elif task.attempt + 1 >= self.MAX_RETRIES:
                logger.error(f"[Retry] ❌ {task.adapter_name} 已达最大重试次数, "
                              f"升级为人工介入!")
                results.append((task, True))  # 最终失败
                if self._alert_callback:
                    self._alert_callback(task)
            else:
                # 继续重试
                self.enqueue(task.request, task.adapter_name, task.attempt + 1)
                logger.warning(f"[Retry] {task.adapter_name} 重试失败, "
                                f"将再次入队 (attempt={task.attempt+1})")

        return results

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)


# ============================================================
# Redis 冷却期锁
# ============================================================
class CooldownManager:
    """
    基于 Redis TTL 的冷却期管理器

    规则: 基础门票每 COOLDOWN_SECONDS 秒最多调价一次
    锁定期内,新价格只记录不执行
    """

    COOLDOWN_SECONDS = 7200       # 2小时
    LOCK_KEY_PREFIX = "pricing:cooldown"

    def __init__(self, cooldown_seconds: Optional[int] = None):
        self.cooldown = cooldown_seconds or self.COOLDOWN_SECONDS

    def try_acquire(self, product_type: str = "base_ticket") -> Tuple[bool, float]:
        """
        尝试获取调价锁

        Returns: (是否获得锁, 剩余冷却秒数)
        """
        key = f"{self.LOCK_KEY_PREFIX}:{product_type}"
        cached = feature_cache.get(key)

        if cached is not None:
            # 锁存在, 返回剩余时间
            return False, float(cached)

        # 获取锁: set key with TTL
        feature_cache.set(key, self.cooldown, ttl=self.cooldown)
        return True, 0.0

    def force_release(self, product_type: str = "base_ticket"):
        """强制释放锁(运维用)"""
        key = f"{self.LOCK_KEY_PREFIX}:{product_type}"
        feature_cache.invalidate(key)

    def remaining_cooldown(self, product_type: str = "base_ticket") -> float:
        key = f"{self.LOCK_KEY_PREFIX}:{product_type}"
        cached = feature_cache.get(key)
        return float(cached) if cached is not None else 0.0


# ============================================================
# 价格同步服务核心
# ============================================================
class PriceSyncService:
    """
    价格同步服务

    用法:
      sync = PriceSyncService(adapters_dict)
      decision = sync.decide(base_price=299.0, reason="AI动态调价")
      if decision.executed:
          print(f"价格已推送: {decision.channel_results}")
    """

    MIN_CHANGE_THRESHOLD = 0.05   # 5%最小变动阈值

    def __init__(
        self,
        adapters: Optional[Dict[str, BaseChannelAdapter]] = None,
        audit_log_path: str = "./data/price_audit_log.jsonl",
    ):
        self.adapters = adapters or create_default_adapters()
        self.audit_log_path = audit_log_path
        self.retry_scheduler = RetryScheduler()
        self.cooldown = CooldownManager()
        self._last_base_price: Dict[str, float] = {}

        # 设置重试告警回调
        self.retry_scheduler.set_alert_callback(self._on_retry_exhausted)

    # ============================================================
    # 主决策入口
    # ============================================================
    def decide(
        self,
        base_price: float,
        product_type: str = "base_ticket",
        reason: str = "AI动态调价",
        force: bool = False,
    ) -> PriceChangeRecord:
        """
        价格同步决策

        流程:
          1. 冷却期检查
          2. 最小变动阈值检查(防抖)
          3. 对每个激活渠道: ChannelPricer计算渠道价 → Adapter.push_price()
          4. 记录对账单
          5. 失败渠道 → 重试队列
        """
        old_price = self._last_base_price.get(product_type, base_price)
        record_id = f"price:{product_type}:{int(time.time())}"
        record = PriceChangeRecord(
            record_id=record_id,
            timestamp=datetime.now().isoformat(),
            base_price_old=old_price,
            base_price_new=base_price,
            change_pct=round((base_price - old_price) / old_price, 4) if old_price > 0 else 0.0,
            reason=reason,
        )

        # === 1. 冷却期检查 ===
        if not force:
            acquired, remaining = self.cooldown.try_acquire(product_type)
            if not acquired:
                logger.info(f"[防抖] 冷却期中(剩余{remaining:.0f}s),价格变更仅记录不执行")
                record.retry_pending = False
                self._write_audit_log(record)
                return record

        # === 2. 最小变动阈值检查 ===
        if old_price > 0 and not force:
            change_pct = abs(base_price - old_price) / old_price
            if change_pct < self.MIN_CHANGE_THRESHOLD:
                logger.info(f"[防抖] 变动幅度{change_pct:.1%}<{self.MIN_CHANGE_THRESHOLD:.0%}, "
                             f"不触发下发 (¥{old_price}→¥{base_price})")
                record.retry_pending = False
                # 释放冷却期锁(小变动不应该消耗调价机会)
                self.cooldown.force_release(product_type)
                self._write_audit_log(record)
                return record

        # === 3. 并行推送到各渠道 ===
        failed: List[str] = []
        for ch_name, adapter in self.adapters.items():
            if not adapter.is_available():
                logger.warning(f"[{ch_name}] 渠道不可用,跳过")
                failed.append(ch_name)
                continue

            # 渠道价计算: 使用 ChannelPricer
            channel_price = self._compute_channel_price(base_price, ch_name, old_price)

            req = PricePushRequest(
                channel=ch_name,
                product_id=f"{product_type}:{ch_name}",
                base_price=base_price,
                channel_price=channel_price,
                original_price=old_price,
                reason=reason,
            )
            result = adapter.push_price(req)
            record.channel_results[ch_name] = result.to_dict()

            if not result.success:
                failed.append(ch_name)
                # 入重试队列
                self.retry_scheduler.enqueue(req, ch_name)

        # === 4. 更新状态 ===
        record.all_success = len(failed) == 0
        record.failed_channels = failed
        record.retry_pending = len(failed) > 0

        if record.all_success:
            self._last_base_price[product_type] = base_price
            logger.info(f"✅ 价格同步成功: base=¥{base_price} "
                         f"({old_price}→{base_price}, {len(self.adapters)}渠道)")
        else:
            logger.warning(f"⚠️ 部分渠道失败: {failed}")

        # === 5. 记录对账单 ===
        self._write_audit_log(record)

        # 发布到消息总线
        bus.publish(
            Channel.PRICING_DECISION,
            record.to_dict(),
            source="price_sync_service",
        )

        return record

    # ============================================================
    # 后台重试循环
    # ============================================================
    def run_retry_loop(self, interval: float = 30.0):
        """
        后台重试循环(在独立线程中运行)

        interval: 检查间隔(秒)
        """
        logger.info("重试循环已启动")
        while True:
            try:
                results = self.retry_scheduler.process_due()
                for task, is_final in results:
                    if is_final:
                        self._on_retry_exhausted(task)
            except Exception as e:
                logger.error(f"重试循环异常: {e}", exc_info=True)
            time.sleep(interval)

    # ============================================================
    # 辅助方法
    # ============================================================
    def _compute_channel_price(self, base_price: float, channel: str,
                                old_price: float) -> float:
        """计算渠道最终售价"""
        from engine.channel_pricer import ChannelPricer, Channel as Ch
        pricer = ChannelPricer()

        ch_map = {
            "meituan": Ch.OTA, "ctrip": Ch.OTA, "feizhu": Ch.OTA,
            "miniapp": Ch.OFFICIAL,
        }
        ch_enum = ch_map.get(channel, Ch.OTA)

        strategies = pricer.compute(base_price, "weekday", 0.5)
        for s in strategies:
            if s.channel == ch_enum.value:
                return s.display_price

        return base_price

    def _write_audit_log(self, record: PriceChangeRecord):
        """写入价格对账单"""
        try:
            os.makedirs(os.path.dirname(self.audit_log_path) or ".", exist_ok=True)
            with open(self.audit_log_path, "a") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"对账单写入失败: {e}")

    def _on_retry_exhausted(self, task):
        """重试耗尽 → 发送最高级别告警"""
        alert_msg = (
            f"🔴 [CRITICAL] 渠道调价失败,需人工介入!\n"
            f"  渠道: {task.adapter_name}\n"
            f"  产品: {task.request.product_id}\n"
            f"  价格: ¥{task.request.original_price} → ¥{task.request.channel_price}\n"
            f"  重试: {task.attempt + 1}次全部失败\n"
            f"  请求ID: {task.request.request_id}"
        )
        logger.error(alert_msg)

        # 发布告警到消息总线
        bus.publish_anomaly({
            "type": "price_sync_exhausted",
            "level": "critical",
            "channel": task.adapter_name,
            "product_id": task.request.product_id,
            "price_new": task.request.channel_price,
            "retry_count": task.attempt + 1,
            "message": alert_msg,
            "suggested_action": (
                f"1. 检查{task.adapter_name} API连通性\n"
                f"2. 确认渠道侧鉴权Token是否过期\n"
                f"3. 手动登录渠道后台确认价格状态\n"
                f"4. 必要时回滚到原价 ¥{task.request.original_price}"
            ),
        })

    def get_status(self) -> Dict[str, Any]:
        return {
            "adapters": list_adapters() if list_adapters else {},
            "pending_retries": self.retry_scheduler.pending_count(),
            "cooldown_remaining": {
                "base_ticket": self.cooldown.remaining_cooldown("base_ticket"),
            },
            "last_prices": self._last_base_price,
        }
