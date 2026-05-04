"""
A/B 测试与灰度发布管理器

核心能力:
  1. 渠道白名单: 配置哪些渠道由AI自动接管
  2. 灰度比例: 逐步放开自动发布权限 (10%→30%→100%)
  3. Human-in-the-loop: 非白名单渠道仅生成建议,等待人工确认
  4. 自动回滚: 关键指标异常时自动切回人工模式

配置示例:
  {
    "miniapp": {"auto_publish": true,  "traffic_pct": 1.0},    # 100%自动
    "meituan": {"auto_publish": false, "traffic_pct": 0.0},    # 人工确认
    "ctrip":   {"auto_publish": false, "traffic_pct": 0.0},    # 人工确认
    "feizhu":  {"auto_publish": false, "traffic_pct": 0.0},    # 人工确认
  }
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from enum import Enum

from utils.logger import get_logger
from utils.feature_cache import feature_cache
from config import settings

logger = get_logger("ABTestManager")


# ============================================================
# 数据结构
# ============================================================
class ReleaseStage(str, Enum):
    CANARY_10 = "canary_10"       # 10%灰度
    CANARY_30 = "canary_30"       # 30%灰度
    CANARY_50 = "canary_50"       # 50%灰度
    FULL = "full"                 # 100%全量
    ROLLBACK = "rollback"         # 已回滚


@dataclass
class ChannelConfig:
    """单个渠道的发布配置"""
    channel: str
    auto_publish: bool = False          # AI是否自动发布
    traffic_pct: float = 0.0            # 流量比例 (0.0~1.0)
    release_stage: str = "canary_10"
    human_approval_required: bool = True
    max_daily_changes: int = 3          # 每日最大调价次数
    daily_change_count: int = 0
    last_change_date: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ABTestDecision:
    """A/B测试决策结果"""
    channel: str
    should_auto_publish: bool
    reason: str
    # 如果不自动发布,给出建议供人工审核
    suggested_price: Optional[float] = None
    old_price: Optional[float] = None
    change_pct: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RollbackRule:
    """自动回滚规则"""
    metric: str                        # "revenue_drop" | "complaint_spike" | "conversion_crash"
    threshold: float
    window_hours: int = 24
    enabled: bool = True


# ============================================================
# A/B 测试管理器
# ============================================================
class ABTestManager:
    """
    A/B 测试与灰度发布管理器

    用法:
      mgr = ABTestManager()
      mgr.load_config("data/ab_config.json")

      decision = mgr.evaluate("meituan", new_price=320, old_price=299)
      if decision.should_auto_publish:
          adapter.push_price(...)
      else:
          # 仅生成建议,等待人工审批
          mgr.add_pending_approval(decision)
    """

    CONFIG_CACHE_KEY = "ab_test:channel_config"

    # 默认: 仅小程序自动发布,其他渠道人工确认
    DEFAULT_CONFIG: Dict[str, dict] = {
        "miniapp": {"auto_publish": True,  "traffic_pct": 1.0, "release_stage": "full",
                     "human_approval_required": False, "max_daily_changes": 10},
        "meituan": {"auto_publish": False, "traffic_pct": 0.0, "release_stage": "canary_10",
                     "human_approval_required": True,  "max_daily_changes": 3},
        "ctrip":   {"auto_publish": False, "traffic_pct": 0.0, "release_stage": "canary_10",
                     "human_approval_required": True,  "max_daily_changes": 3},
        "feizhu":  {"auto_publish": False, "traffic_pct": 0.0, "release_stage": "canary_10",
                     "human_approval_required": True,  "max_daily_changes": 3},
    }

    # 自动回滚规则
    ROLLBACK_RULES: List[RollbackRule] = [
        RollbackRule("revenue_drop", -0.20, 24, True),        # 收入降20%
        RollbackRule("complaint_spike", 3.0, 4, True),        # 投诉突增3倍
        RollbackRule("conversion_crash", -0.50, 8, True),     # 转化率腰斩
    ]

    def __init__(self, config_path: Optional[str] = None):
        self._channels: Dict[str, ChannelConfig] = {}
        self._pending_approvals: List[ABTestDecision] = []
        self._rollback_active: Dict[str, bool] = {}

        # 加载默认配置
        for ch, cfg in self.DEFAULT_CONFIG.items():
            self._channels[ch] = ChannelConfig(channel=ch, **cfg)

        if config_path:
            self.load_config(config_path)

    # ============================================================
    # 配置管理
    # ============================================================
    def load_config(self, path: str):
        """从JSON文件加载渠道配置"""
        if not os.path.exists(path):
            logger.warning(f"配置文件不存在,使用默认配置: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        for ch, cfg in data.items():
            if ch in self._channels:
                for k, v in cfg.items():
                    setattr(self._channels[ch], k, v)
        logger.info(f"已加载 {len(data)} 个渠道的A/B配置")
        self._persist_to_cache()

    def save_config(self, path: str):
        """保存当前配置到JSON"""
        data = {ch: cfg.to_dict() for ch, cfg in self._channels.items()}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def _persist_to_cache(self):
        data = {ch: cfg.to_dict() for ch, cfg in self._channels.items()}
        feature_cache.set(self.CONFIG_CACHE_KEY, data, ttl=86400)

    # ============================================================
    # 核心决策
    # ============================================================
    def evaluate(
        self,
        channel: str,
        new_price: float,
        old_price: float,
        reason: str = "",
    ) -> ABTestDecision:
        """
        评估是否应该自动发布价格到指定渠道

        Returns: ABTestDecision with should_auto_publish flag
        """
        cfg = self._channels.get(channel)
        if cfg is None:
            return ABTestDecision(
                channel=channel,
                should_auto_publish=False,
                reason=f"未知渠道: {channel}",
            )

        # 检查是否已回滚
        if self._rollback_active.get(channel):
            return ABTestDecision(
                channel=channel,
                should_auto_publish=False,
                reason="渠道已触发自动回滚,需人工介入恢复",
                suggested_price=new_price,
                old_price=old_price,
                change_pct=(new_price - old_price) / old_price if old_price > 0 else 0,
            )

        # 检查每日调价次数
        today = datetime.now().strftime("%Y-%m-%d")
        if cfg.last_change_date != today:
            cfg.daily_change_count = 0
            cfg.last_change_date = today
        if cfg.daily_change_count >= cfg.max_daily_changes:
            return ABTestDecision(
                channel=channel,
                should_auto_publish=False,
                reason=f"已达每日调价上限({cfg.max_daily_changes}次)",
                suggested_price=new_price,
                old_price=old_price,
                change_pct=(new_price - old_price) / old_price if old_price > 0 else 0,
            )

        # 灰度流量判定
        if not cfg.auto_publish:
            return ABTestDecision(
                channel=channel,
                should_auto_publish=False,
                reason=f"渠道未开启自动发布(release_stage={cfg.release_stage})",
                suggested_price=new_price,
                old_price=old_price,
                change_pct=(new_price - old_price) / old_price if old_price > 0 else 0,
            )

        # 通过所有检查 → 可以自动发布
        cfg.daily_change_count += 1
        change_pct = (new_price - old_price) / old_price if old_price > 0 else 0

        return ABTestDecision(
            channel=channel,
            should_auto_publish=True,
            reason=f"自动发布(灰度{cfg.release_stage})",
            suggested_price=new_price,
            old_price=old_price,
            change_pct=change_pct,
        )

    # ============================================================
    # 灰度升级
    # ============================================================
    def promote_stage(self, channel: str) -> Optional[str]:
        """升级灰度阶段: canary_10 → canary_30 → canary_50 → full"""
        cfg = self._channels.get(channel)
        if cfg is None:
            return None

        stages = [ReleaseStage.CANARY_10, ReleaseStage.CANARY_30,
                   ReleaseStage.CANARY_50, ReleaseStage.FULL]
        try:
            idx = stages.index(ReleaseStage(cfg.release_stage))
        except ValueError:
            return None

        if idx < len(stages) - 1:
            next_stage = stages[idx + 1]
            cfg.release_stage = next_stage.value
            cfg.traffic_pct = {0: 0.1, 1: 0.3, 2: 0.5, 3: 1.0}[idx + 1]
            cfg.human_approval_required = (next_stage != ReleaseStage.FULL)
            self._persist_to_cache()
            logger.info(f"[{channel}] 灰度升级: → {next_stage.value} "
                         f"(traffic={cfg.traffic_pct:.0%})")
            return next_stage.value
        return None

    # ============================================================
    # 自动回滚
    # ============================================================
    def check_rollback(self, channel: str, metrics: Dict[str, float]) -> bool:
        """检查是否应触发自动回滚"""
        for rule in self.ROLLBACK_RULES:
            if not rule.enabled:
                continue
            current = metrics.get(rule.metric, 0.0)
            if rule.metric == "revenue_drop" and current < rule.threshold:
                self._trigger_rollback(channel, rule, current)
                return True
            if rule.metric == "complaint_spike" and current > rule.threshold:
                self._trigger_rollback(channel, rule, current)
                return True
            if rule.metric == "conversion_crash" and current < rule.threshold:
                self._trigger_rollback(channel, rule, current)
                return True
        return False

    def _trigger_rollback(self, channel: str, rule: RollbackRule, current_value: float):
        self._rollback_active[channel] = True
        cfg = self._channels[channel]
        cfg.auto_publish = False
        cfg.human_approval_required = True

        logger.error(f"🔴 [{channel}] 自动回滚触发! "
                      f"指标={rule.metric} 当前={current_value} 阈值={rule.threshold}")

        from services.message_bus import bus
        bus.publish_anomaly({
            "type": "ab_test_rollback",
            "level": "critical",
            "channel": channel,
            "metric": rule.metric,
            "current_value": current_value,
            "threshold": rule.threshold,
            "message": f"渠道{channel}因{rule.metric}异常自动回滚,转为人工确认模式",
            "suggested_action": "检查业务数据,确认无问题后手动调用 recover_rollback() 恢复",
        })

    def recover_rollback(self, channel: str):
        """手动恢复回滚"""
        self._rollback_active.pop(channel, None)
        logger.info(f"[{channel}] 回滚已恢复")

    # ============================================================
    # Human-in-the-loop
    # ============================================================
    def add_pending_approval(self, decision: ABTestDecision):
        self._pending_approvals.append(decision)
        logger.info(f"[HITL] 新增待审批: {decision.channel} "
                     f"¥{decision.old_price}→¥{decision.suggested_price}")

    def get_pending_approvals(self) -> List[ABTestDecision]:
        return list(self._pending_approvals)

    def approve(self, channel: str) -> Optional[ABTestDecision]:
        """人工审批通过"""
        for i, d in enumerate(self._pending_approvals):
            if d.channel == channel:
                self._pending_approvals.pop(i)
                logger.info(f"[HITL] 审批通过: {channel}")
                return d
        return None

    def reject(self, channel: str, reason: str = ""):
        self._pending_approvals = [
            d for d in self._pending_approvals if d.channel != channel
        ]
        logger.info(f"[HITL] 审批拒绝: {channel} reason={reason}")

    # ============================================================
    # 状态查询
    # ============================================================
    def get_status(self) -> Dict[str, Any]:
        return {
            "channels": {ch: cfg.to_dict() for ch, cfg in self._channels.items()},
            "rollback_active": dict(self._rollback_active),
            "pending_approvals": len(self._pending_approvals),
        }

    def is_auto_publish_enabled(self, channel: str) -> bool:
        cfg = self._channels.get(channel)
        return cfg is not None and cfg.auto_publish and not self._rollback_active.get(channel, False)
