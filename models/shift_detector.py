"""
P1-02: 分布偏移 (Distribution Shift) 检测 + 动态权重降级

背景:
  ML/RL 模型在训练集分布内表现好,但遇到 OOD (out-of-distribution) 输入时
  可能给出离谱价格(如¥80或¥599的极值),直接用会造成严重亏损。

检测策略:
  1. 单变量检测: 每个关键特征是否在训练集的 [μ-3σ, μ+3σ] 范围内
  2. 组合检测:   看某个 (day_type, weather) 组合在训练集中是否出现过
  3. 模型方差:   Ensemble多模型预测差异 > 阈值时,说明模型不自信
  
响应策略:
  无异常  → 正常权重 (rule=0.30, ml=0.45, rl=0.25)
  轻度偏移 → 提升业务规则权重 (rule=0.50, ml=0.35, rl=0.15)
  严重偏移 → 强制降级为规则主导 (rule=0.80, ml=0.15, rl=0.05)
  极端情况 → 100% 业务规则,触发熔断日志告警
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

from utils.logger import get_logger
from config import settings

logger = get_logger("ShiftDetector")


class ShiftLevel(str, Enum):
    NORMAL = "normal"          # 正常,使用原始权重
    LIGHT = "light"            # 轻度偏移,适度提升规则权重
    SEVERE = "severe"          # 严重偏移,规则主导
    CRITICAL = "critical"      # 极端偏移,100%规则


@dataclass
class ShiftDetection:
    level: ShiftLevel
    reasons: List[str]
    feature_z_scores: Dict[str, float]
    unseen_combinations: List[str]
    model_prediction_variance: float
    # 动态调整后的权重
    adjusted_weights: Tuple[float, float, float]  # (rule, ml, rl)
    original_weights: Tuple[float, float, float]
    fallback_triggered: bool

    def to_dict(self):
        d = asdict(self)
        d["level"] = self.level.value
        return d


class DistributionShiftDetector:
    """
    分布偏移检测器
    
    训练时调用 fit(history_df) 学习各特征的分布参数,
    推理时调用 detect(...) 评估输入的偏移程度并给出权重调整建议。
    """

    # 原始权重(可被覆盖)
    DEFAULT_WEIGHTS = (0.30, 0.45, 0.25)  # rule, ml, rl

    # 分级阈值
    LIGHT_Z_THRESHOLD = 2.5
    SEVERE_Z_THRESHOLD = 3.5
    CRITICAL_Z_THRESHOLD = 5.0
    VARIANCE_LIGHT = 0.20    # 多模型预测CV > 20% = 轻度
    VARIANCE_SEVERE = 0.40   # > 40% = 严重

    # 分级对应的权重
    WEIGHTS_BY_LEVEL = {
        ShiftLevel.NORMAL:   (0.30, 0.45, 0.25),
        ShiftLevel.LIGHT:    (0.50, 0.35, 0.15),
        ShiftLevel.SEVERE:   (0.80, 0.15, 0.05),
        ShiftLevel.CRITICAL: (1.00, 0.00, 0.00),
    }

    def __init__(self):
        self.feature_stats: Dict[str, Tuple[float, float]] = {}  # {feat: (mean, std)}
        self.seen_combinations: Set[str] = set()                 # 训练集出现过的 day_type|weather 组合
        self.fitted = False

    def fit(self, history_df: pd.DataFrame):
        """从历史数据学习特征分布"""
        numeric_features = ["temperature", "rainfall", "price", "visitors"]
        for f in numeric_features:
            if f in history_df.columns:
                mean = float(history_df[f].mean())
                std = float(history_df[f].std())
                self.feature_stats[f] = (mean, max(std, 1e-3))

        # 记录所有出现过的 (day_type, weather) 组合
        if "day_type" in history_df.columns and "weather" in history_df.columns:
            combos = history_df[["day_type", "weather"]].drop_duplicates()
            self.seen_combinations = set(
                f"{r['day_type']}|{r['weather']}" for _, r in combos.iterrows()
            )

        self.fitted = True
        logger.info(f"分布学习完成 | {len(self.feature_stats)}个特征 | {len(self.seen_combinations)}个组合")

    def detect(
        self,
        input_features: Dict[str, float],
        day_type: str,
        weather: str,
        model_predictions: Optional[Dict[str, float]] = None,
    ) -> ShiftDetection:
        """
        检测输入是否偏离训练集分布,并给出权重调整建议
        
        model_predictions: 各模型对同一输入的预测(用于计算预测方差)
        """
        if not self.fitted:
            # 未训练检测器,保守起见不触发降级
            return ShiftDetection(
                level=ShiftLevel.NORMAL, reasons=["检测器未训练,跳过"],
                feature_z_scores={}, unseen_combinations=[],
                model_prediction_variance=0.0,
                adjusted_weights=self.DEFAULT_WEIGHTS,
                original_weights=self.DEFAULT_WEIGHTS,
                fallback_triggered=False,
            )

        reasons: List[str] = []
        z_scores: Dict[str, float] = {}
        max_z = 0.0

        # ===== 1. 单变量Z-score =====
        for feat, val in input_features.items():
            if feat in self.feature_stats:
                mean, std = self.feature_stats[feat]
                z = abs((val - mean) / std)
                z_scores[feat] = float(z)
                if z > self.LIGHT_Z_THRESHOLD:
                    reasons.append(f"{feat}={val:.2f} 偏离训练集 (z={z:.2f})")
                max_z = max(max_z, z)

        # ===== 2. 组合未见检测 =====
        unseen = []
        combo_key = f"{day_type}|{weather}"
        if combo_key not in self.seen_combinations:
            unseen.append(combo_key)
            reasons.append(f"训练集未出现的组合: ({day_type}, {weather})")

        # ===== 3. 多模型预测方差 =====
        pred_cv = 0.0  # 变异系数 coefficient of variation
        if model_predictions and len(model_predictions) >= 2:
            preds = np.array(list(model_predictions.values()))
            mean_p = preds.mean()
            if mean_p > 0:
                pred_cv = float(preds.std() / mean_p)
                if pred_cv > self.VARIANCE_LIGHT:
                    reasons.append(f"多模型预测分歧大 (CV={pred_cv:.2%})")

        # ===== 综合判级 =====
        level = ShiftLevel.NORMAL
        if max_z >= self.CRITICAL_Z_THRESHOLD or len(unseen) > 0 and pred_cv > self.VARIANCE_SEVERE:
            level = ShiftLevel.CRITICAL
        elif max_z >= self.SEVERE_Z_THRESHOLD or pred_cv > self.VARIANCE_SEVERE:
            level = ShiftLevel.SEVERE
        elif max_z >= self.LIGHT_Z_THRESHOLD or len(unseen) > 0 or pred_cv > self.VARIANCE_LIGHT:
            level = ShiftLevel.LIGHT

        adjusted = self.WEIGHTS_BY_LEVEL[level]
        fallback = (level == ShiftLevel.CRITICAL)

        if level != ShiftLevel.NORMAL:
            logger.warning(
                f"🚨 分布偏移={level.value} | max_z={max_z:.2f} | CV={pred_cv:.2%} | "
                f"权重调整为 rule={adjusted[0]}, ml={adjusted[1]}, rl={adjusted[2]}"
            )
            for r in reasons:
                logger.warning(f"   · {r}")

        return ShiftDetection(
            level=level,
            reasons=reasons if reasons else ["输入分布正常"],
            feature_z_scores=z_scores,
            unseen_combinations=unseen,
            model_prediction_variance=pred_cv,
            adjusted_weights=adjusted,
            original_weights=self.DEFAULT_WEIGHTS,
            fallback_triggered=fallback,
        )
