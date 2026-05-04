"""
冠军/挑战者机制 (Champion-Challenger) + 影子测试 + 平滑放量

核心机制:
  1. 影子模式: 挑战者模型接收实时数据但不真正下发价格
  2. 30天回测对比: 挑战者 vs 冠军 预期收益
  3. 连续3天跑赢 → 自动拨出 5% 流量给挑战者
  4. 转化率+客单价双升 → 逐步放量 5%→20%→50%→100%
  5. 任何异常 → 立即回滚到冠军

状态:
  shadowing   → 影子测试阶段
  canary_5    → 5% A/B测试
  canary_20   → 20%
  canary_50   → 50%
  promoted    → 新模型成为冠军
  rolled_back → 异常回滚

用法:
  manager = ChampionChallengerManager()
  
  # 日常: 挑战者输出建议价,记录到影子日志
  shadow_price = manager.shadow_decide(challenger_model, state)
  
  # 每日评估
  manager.evaluate_daily()
  
  # 如果达到promote条件
  if manager.should_promote():
      manager.promote_challenger()
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from utils.logger import get_logger
from config import settings
from services.message_bus import bus, Channel
from services.rl_logger import sar_logger

logger = get_logger("ChampionChallenger")


# ============================================================
# 数据结构
# ============================================================
class ChallengerStage(str, Enum):
    SHADOWING = "shadowing"
    CANARY_5 = "canary_5"
    CANARY_20 = "canary_20"
    CANARY_50 = "canary_50"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelVersion:
    """模型版本"""
    version_id: str                 # "v20260504_0200"
    model_path: str
    role: str                       # "champion" | "challenger"
    created_at: str
    promoted_at: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShadowResult:
    """影子模式单次决策结果"""
    timestamp: str
    date: str
    state: Dict[str, Any]
    champion_price: float
    challenger_price: float
    champion_expected_revenue: float   # 冠军预期收入
    challenger_expected_revenue: float # 挑战者预期收入
    revenue_delta_pct: float           # 收入差异百分比

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DailyComparison:
    """日度对比报告"""
    date: str
    n_decisions: int
    champion_avg_price: float
    challenger_avg_price: float
    champion_total_revenue: float
    challenger_total_revenue: float
    revenue_improvement_pct: float
    # 安全性指标
    challenger_risk_flags: List[str] = field(default_factory=list)
    should_promote: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# 冠军/挑战者管理器
# ============================================================
class ChampionChallengerManager:
    """
    冠军/挑战者模型管理

    配置:
      shadow_backtest_days: 影子测试至少跑N天才能进入A/B
      consecutive_win_days: 连续跑赢N天才能升级
      promote_threshold: 收入提升阈值(如3%)
    """

    SHADOW_MIN_DAYS = 3              # 影子测试最少天数
    CONSECUTIVE_WIN_DAYS = 3         # 连续跑赢天数
    PROMOTE_REVENUE_THRESHOLD = 0.03  # 3% 收入提升阈值
    ROLLBACK_CONVERSION_DROP = -0.10 # 转化率下降10%触发回滚
    ROLLBACK_REFUND_SPIKE = 3.0      # 退票率3倍触发回滚

    def __init__(
        self,
        storage_dir: str = "./data/champion_challenger",
    ):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

        # 当前状态
        self._champion: Optional[ModelVersion] = None
        self._challenger: Optional[ModelVersion] = None
        self._stage: ChallengerStage = ChallengerStage.SHADOWING
        self._stage_updated_at = time.time()

        # 影子测试结果
        self._shadow_results: List[ShadowResult] = []
        self._daily_comparisons: List[DailyComparison] = []
        self._consecutive_win_days = 0

        # 流量分配
        self._traffic_split: Dict[str, float] = {"champion": 1.0, "challenger": 0.0}

        # 加载状态
        self._load_state()

    # ============================================================
    # 模型注册
    # ============================================================
    def register_champion(self, model_path: str, version: str = ""):
        """注册冠军模型(线上正在运行的)"""
        if not version:
            version = f"champion_{datetime.now().strftime('%Y%m%d')}"
        self._champion = ModelVersion(
            version_id=version,
            model_path=model_path,
            role="champion",
            created_at=datetime.now().isoformat(),
        )
        logger.info(f"冠军模型注册: {version} @ {model_path}")

    def register_challenger(self, model_path: str, version: str = ""):
        """注册挑战者模型(新训练的)"""
        if not version:
            version = f"challenger_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self._challenger = ModelVersion(
            version_id=version,
            model_path=model_path,
            role="challenger",
            created_at=datetime.now().isoformat(),
        )
        self._stage = ChallengerStage.SHADOWING
        self._stage_updated_at = time.time()
        self._consecutive_win_days = 0
        self._shadow_results.clear()
        self._traffic_split = {"champion": 1.0, "challenger": 0.0}
        logger.info(f"挑战者模型注册(影子模式): {version} @ {model_path}")
        self._save_state()

    # ============================================================
    # 影子决策
    # ============================================================
    def shadow_decide(
        self,
        champion_price: float,
        challenger_price: float,
        state: Dict[str, Any],
        expected_visitors: float = 0,
    ) -> ShadowResult:
        """
        影子模式: 挑战者输出建议价但不真正下发

        Returns: 对比结果
        """
        if self._challenger is None:
            # 无挑战者,直接返回
            return ShadowResult(
                timestamp=datetime.now().isoformat(),
                date=datetime.now().strftime("%Y-%m-%d"),
                state=state,
                champion_price=champion_price,
                challenger_price=champion_price,
                champion_expected_revenue=0,
                challenger_expected_revenue=0,
                revenue_delta_pct=0,
            )

        # 估算收入
        visitors = expected_visitors or 15000
        champion_rev = champion_price * visitors
        challenger_rev = challenger_price * visitors * 0.98  # 略微折扣(新模型可能有适应成本)

        delta = (challenger_rev - champion_rev) / champion_rev if champion_rev > 0 else 0

        result = ShadowResult(
            timestamp=datetime.now().isoformat(),
            date=datetime.now().strftime("%Y-%m-%d"),
            state=state,
            champion_price=champion_price,
            challenger_price=challenger_price,
            champion_expected_revenue=round(champion_rev, 2),
            challenger_expected_revenue=round(challenger_rev, 2),
            revenue_delta_pct=round(delta, 4),
        )

        self._shadow_results.append(result)
        if len(self._shadow_results) > 1000:
            self._shadow_results = self._shadow_results[-500:]

        return result

    # ============================================================
    # 每日评估
    # ============================================================
    def evaluate_daily(self) -> DailyComparison:
        """评估昨日影子表现"""
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        yesterday_results = [
            r for r in self._shadow_results if r.date == yesterday
        ]

        if len(yesterday_results) < 10:
            return DailyComparison(
                date=yesterday,
                n_decisions=len(yesterday_results),
                champion_avg_price=0,
                challenger_avg_price=0,
                champion_total_revenue=0,
                challenger_total_revenue=0,
                revenue_improvement_pct=0,
            )

        champ_prices = [r.champion_price for r in yesterday_results]
        chall_prices = [r.challenger_price for r in yesterday_results]
        champ_rev = sum(r.champion_expected_revenue for r in yesterday_results)
        chall_rev = sum(r.challenger_expected_revenue for r in yesterday_results)

        improvement = (chall_rev - champ_rev) / champ_rev if champ_rev > 0 else 0

        risk_flags = []
        if improvement < 0:
            risk_flags.append("revenue_lower")
        if np.std(chall_prices) > np.std(champ_prices) * 2:
            risk_flags.append("high_price_volatility")

        should_promote = (
            improvement > self.PROMOTE_REVENUE_THRESHOLD
            and len(risk_flags) == 0
            and len(yesterday_results) >= 30
        )

        comparison = DailyComparison(
            date=yesterday,
            n_decisions=len(yesterday_results),
            champion_avg_price=round(np.mean(champ_prices), 2),
            challenger_avg_price=round(np.mean(chall_prices), 2),
            champion_total_revenue=round(champ_rev, 2),
            challenger_total_revenue=round(chall_rev, 2),
            revenue_improvement_pct=round(improvement * 100, 2),
            challenger_risk_flags=risk_flags,
            should_promote=should_promote,
        )

        self._daily_comparisons.append(comparison)

        # 连续跑赢天数
        if improvement > 0:
            self._consecutive_win_days += 1
        else:
            self._consecutive_win_days = 0

        logger.info(
            f"日度评估 {yesterday}: "
            f"冠军均价=¥{comparison.champion_avg_price} | "
            f"挑战者均价=¥{comparison.challenger_avg_price} | "
            f"收入差异={comparison.revenue_improvement_pct}% | "
            f"连续跑赢={self._consecutive_win_days}天"
        )

        # 自动推进阶段
        self._auto_advance_stage()

        self._save_state()
        return comparison

    def _auto_advance_stage(self):
        """根据评估结果自动推进阶段"""
        if self._stage == ChallengerStage.SHADOWING:
            # 影子跑够N天且连续跑赢 → 进入5%灰度
            active_days = len(set(r.date for r in self._shadow_results))
            if (active_days >= self.SHADOW_MIN_DAYS
                    and self._consecutive_win_days >= self.CONSECUTIVE_WIN_DAYS):
                self._stage = ChallengerStage.CANARY_5
                self._traffic_split = {"champion": 0.95, "challenger": 0.05}
                logger.info("🚀 影子测试通过,进入 5% 灰度A/B测试")

        elif self._stage == ChallengerStage.CANARY_5:
            if self._consecutive_win_days >= 2:
                self._stage = ChallengerStage.CANARY_20
                self._traffic_split = {"champion": 0.80, "challenger": 0.20}
                logger.info("📈 5%灰度通过,放量到 20%")

        elif self._stage == ChallengerStage.CANARY_20:
            if self._consecutive_win_days >= 3:
                self._stage = ChallengerStage.CANARY_50
                self._traffic_split = {"champion": 0.50, "challenger": 0.50}
                logger.info("📈 20%灰度通过,放量到 50%")

        elif self._stage == ChallengerStage.CANARY_50:
            if self._consecutive_win_days >= 3:
                self.promote_challenger()

    # ============================================================
    # 晋级与回滚
    # ============================================================
    def should_promote(self) -> bool:
        return self._stage == ChallengerStage.PROMOTED

    def promote_challenger(self):
        """挑战者晋级为新冠军"""
        if self._challenger is None:
            return
        old_champion = self._champion
        self._champion = self._challenger
        self._champion.role = "champion"
        self._champion.promoted_at = datetime.now().isoformat()
        self._challenger = None
        self._stage = ChallengerStage.PROMOTED
        self._traffic_split = {"champion": 1.0, "challenger": 0.0}

        logger.info(f"👑 挑战者晋级: {old_champion.version_id if old_champion else '?'} "
                     f"→ {self._champion.version_id}")

        bus.publish(Channel.ANOMALY, {
            "type": "champion_promoted",
            "level": "info",
            "new_model": self._champion.version_id,
            "message": f"挑战者模型 {self._champion.version_id} 已晋级为新冠军",
        }, source="champion_challenger")

        self._save_state()

    def rollback(self, reason: str):
        """异常回滚: 恢复到冠军模型"""
        self._stage = ChallengerStage.ROLLED_BACK
        self._traffic_split = {"champion": 1.0, "challenger": 0.0}
        self._consecutive_win_days = 0

        logger.error(f"🔴 挑战者回滚: {reason}")

        bus.publish(Channel.ANOMALY, {
            "type": "challenger_rollback",
            "level": "critical",
            "reason": reason,
            "message": f"挑战者模型已回滚: {reason}",
            "suggested_action": "检查新模型定价策略,确认安全后重新进入影子模式",
        }, source="champion_challenger")

        self._save_state()

    # ============================================================
    # 流量路由
    # ============================================================
    def route_traffic(self) -> str:
        """
        根据当前流量分配返回应该使用的模型

        Returns: "champion" or "challenger"
        """
        ratio = self._traffic_split.get("challenger", 0)
        if ratio <= 0:
            return "champion"
        return "challenger" if np.random.random() < ratio else "champion"

    # ============================================================
    # 业务指标监控(触发回滚)
    # ============================================================
    def check_safety_metrics(
        self,
        conversion_rate: float,
        refund_rate: float,
        baseline_conversion: float,
        baseline_refund: float,
    ) -> bool:
        """
        检查挑战者安全指标

        Returns: True=安全, False=应回滚
        """
        if baseline_conversion > 0:
            conv_drop = (conversion_rate - baseline_conversion) / baseline_conversion
            if conv_drop < self.ROLLBACK_CONVERSION_DROP:
                self.rollback(f"转化率下降{abs(conv_drop)*100:.0f}%")
                return False

        if baseline_refund > 0:
            refund_spike = refund_rate / baseline_refund
            if refund_spike > self.ROLLBACK_REFUND_SPIKE:
                self.rollback(f"退票率飙升{refund_spike:.1f}倍")
                return False

        return True

    # ============================================================
    # 持久化
    # ============================================================
    def _save_state(self):
        state_path = os.path.join(self.storage_dir, "cc_state.json")
        state = {
            "stage": self._stage.value,
            "champion": self._champion.to_dict() if self._champion else None,
            "challenger": self._challenger.to_dict() if self._challenger else None,
            "consecutive_win_days": self._consecutive_win_days,
            "traffic_split": self._traffic_split,
            "stage_updated_at": self._stage_updated_at,
            "daily_comparisons": [
                c.to_dict() for c in self._daily_comparisons[-30:]
            ],
        }
        with open(state_path, "w") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

    def _load_state(self):
        state_path = os.path.join(self.storage_dir, "cc_state.json")
        if not os.path.exists(state_path):
            return
        try:
            with open(state_path) as f:
                data = json.load(f)
            self._stage = ChallengerStage(data.get("stage", "shadowing"))
            self._consecutive_win_days = data.get("consecutive_win_days", 0)
            self._traffic_split = data.get("traffic_split", {"champion": 1.0, "challenger": 0.0})
            if data.get("champion"):
                self._champion = ModelVersion(**data["champion"])
            if data.get("challenger"):
                self._challenger = ModelVersion(**data["challenger"])
            logger.info(f"冠军/挑战者状态已加载: stage={self._stage.value}")
        except Exception as e:
            logger.warning(f"状态加载失败: {e}")

    def get_status(self) -> dict:
        return {
            "stage": self._stage.value,
            "champion_version": self._champion.version_id if self._champion else None,
            "challenger_version": self._challenger.version_id if self._challenger else None,
            "consecutive_win_days": self._consecutive_win_days,
            "traffic_split": self._traffic_split,
            "yesterday_comparison": (
                self._daily_comparisons[-1].to_dict()
                if self._daily_comparisons else None
            ),
        }
