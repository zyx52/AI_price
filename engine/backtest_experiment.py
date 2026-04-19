"""
策略回测与A/B测试框架

1. 策略回测(Backtest):
   给定历史数据 + 任一定价策略(函数),在历史时间线上"重演",
   模拟计算:如果当时用了该策略,收入/客流/体验会如何?
   输出:AI策略 vs 实际历史 vs 业务规则 三方对比

2. A/B测试(Experiment):
   将每日定价请求随机分成两组,一组用A策略、一组用B策略,
   持续追踪两组的关键指标(收入、客流、客单价等),
   用t检验判断策略差异是否显著。

这两个是"AI参与闭环"的最后一块拼图,也是打动管理层最有力的武器。
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Callable, Optional, Tuple
import numpy as np
import pandas as pd

from utils.logger import get_logger
from config import settings

logger = get_logger("BacktestExperiment")


# ============================================================
# 策略回测
# ============================================================
@dataclass
class BacktestResult:
    strategy_name: str
    period: str                              # "2024-01-01 ~ 2024-12-31"
    n_days: int
    total_visitors: int
    total_ticket_revenue: float
    total_secondary_revenue: float
    total_revenue: float
    avg_price: float
    avg_load_rate: float
    avg_daily_revenue: float
    daily_df: pd.DataFrame                   # 每日详细数据

    def to_dict(self, include_df=False):
        d = asdict(self)
        if not include_df:
            d.pop("daily_df")
        return d


class StrategyBacktester:
    """历史回测器"""

    def __init__(self, price_elasticity: float = 0.8):
        """
        price_elasticity: 价格弹性系数,用于估算不同价格下的客流
        从历史数据学习: 客流 ≈ 历史客流 × (历史价/新价)^elasticity
        """
        self.elasticity = price_elasticity

    def run(
        self,
        history: pd.DataFrame,
        strategy_fn: Callable[[dict], float],
        strategy_name: str = "AI策略",
    ) -> BacktestResult:
        """
        strategy_fn: 输入当日上下文dict,返回该日的推荐价格
        上下文包含:date, day_type, weather, temperature, rainfall, historical_avg_price等
        """
        df = history.copy().sort_values("date").reset_index(drop=True)

        results = []
        for i, row in df.iterrows():
            # 构造策略上下文
            context = {
                "date": str(row["date"]),
                "day_type": row["day_type"],
                "weather": row["weather"],
                "temperature": row["temperature"],
                "rainfall": row["rainfall"],
                "historical_price": row["price"],
                "historical_visitors": row["visitors"],
                "is_holiday": bool(row["is_holiday"]),
                "is_weekend": bool(row["is_weekend"]),
            }

            # 调用策略
            new_price = float(strategy_fn(context))
            new_price = np.clip(new_price, settings.pricing.min_price, settings.pricing.max_price)

            # 用弹性估算新客流
            # 注意: 这是近似估算,真实场景应该用更复杂的需求模型
            hist_price = row["price"]
            hist_visitors = row["visitors"]
            new_visitors = hist_visitors * (hist_price / new_price) ** self.elasticity
            new_visitors = np.clip(new_visitors, 100, settings.park_capacity)

            ticket_rev = new_price * new_visitors
            secondary_rev = new_visitors * settings.secondary_consumption_ratio * 130
            total_rev = ticket_rev + secondary_rev
            load_rate = new_visitors / settings.park_capacity

            results.append({
                "date": row["date"],
                "price": new_price,
                "visitors": int(new_visitors),
                "ticket_revenue": ticket_rev,
                "secondary_revenue": secondary_rev,
                "total_revenue": total_rev,
                "load_rate": load_rate,
                "day_type": row["day_type"],
                "weather": row["weather"],
            })

        daily_df = pd.DataFrame(results)
        period = f"{df['date'].min().strftime('%Y-%m-%d')} ~ {df['date'].max().strftime('%Y-%m-%d')}"

        return BacktestResult(
            strategy_name=strategy_name,
            period=period,
            n_days=len(daily_df),
            total_visitors=int(daily_df["visitors"].sum()),
            total_ticket_revenue=float(daily_df["ticket_revenue"].sum()),
            total_secondary_revenue=float(daily_df["secondary_revenue"].sum()),
            total_revenue=float(daily_df["total_revenue"].sum()),
            avg_price=float(daily_df["price"].mean()),
            avg_load_rate=float(daily_df["load_rate"].mean()),
            avg_daily_revenue=float(daily_df["total_revenue"].mean()),
            daily_df=daily_df,
        )

    def compare(
        self,
        results: List[BacktestResult],
        baseline_index: int = 0,
    ) -> pd.DataFrame:
        """对多个策略结果做对比"""
        baseline = results[baseline_index]
        rows = []
        for r in results:
            rev_delta = (r.total_revenue - baseline.total_revenue) / baseline.total_revenue * 100
            visitor_delta = (r.total_visitors - baseline.total_visitors) / baseline.total_visitors * 100
            rows.append({
                "策略": r.strategy_name,
                "总收入(万)": round(r.total_revenue / 1e4, 1),
                "vs基准": f"{rev_delta:+.2f}%",
                "总客流": r.total_visitors,
                "客流vs基准": f"{visitor_delta:+.2f}%",
                "均价": f"¥{r.avg_price:.0f}",
                "均负载率": f"{r.avg_load_rate*100:.1f}%",
            })
        return pd.DataFrame(rows)


# ============================================================
# A/B 测试实验框架
# ============================================================
@dataclass
class ExperimentResult:
    experiment_id: str
    start_date: str
    end_date: str
    n_days_a: int
    n_days_b: int
    metric_name: str                       # "total_revenue" | "visitors" | "load_rate"
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    lift_pct: float                        # B 相对 A 的提升百分比
    t_statistic: float
    p_value: float
    significant: bool                      # p < 0.05
    winner: str                            # "A" | "B" | "tie"
    reasoning: str

    def to_dict(self):
        return asdict(self)


class ABTestExperiment:
    """A/B 测试实验"""

    def __init__(self, experiment_id: str, alpha: float = 0.05):
        self.experiment_id = experiment_id
        self.alpha = alpha
        self.group_a: List[dict] = []   # [{date, price, visitors, revenue, ...}]
        self.group_b: List[dict] = []

    def assign(self, date: str, seed_key: Optional[str] = None) -> str:
        """
        将某一天分配到A组或B组
        使用哈希保证同一日期稳定分组(可复现)
        """
        key = seed_key or date
        h = hash(f"{self.experiment_id}|{key}") % 100
        return "A" if h < 50 else "B"

    def record(self, group: str, observation: dict):
        """记录一次观测数据"""
        if group == "A":
            self.group_a.append(observation)
        elif group == "B":
            self.group_b.append(observation)

    def analyze(self, metric: str = "total_revenue") -> ExperimentResult:
        if len(self.group_a) < 5 or len(self.group_b) < 5:
            return ExperimentResult(
                experiment_id=self.experiment_id,
                start_date="", end_date="",
                n_days_a=len(self.group_a), n_days_b=len(self.group_b),
                metric_name=metric,
                mean_a=0, mean_b=0, std_a=0, std_b=0,
                lift_pct=0, t_statistic=0, p_value=1.0,
                significant=False, winner="tie",
                reasoning="样本量不足,至少每组需5天数据",
            )

        a_values = np.array([obs[metric] for obs in self.group_a])
        b_values = np.array([obs[metric] for obs in self.group_b])

        # 双样本t检验
        t_stat, p_value = self._welch_t_test(a_values, b_values)

        mean_a = float(a_values.mean())
        mean_b = float(b_values.mean())
        std_a = float(a_values.std(ddof=1))
        std_b = float(b_values.std(ddof=1))
        lift = (mean_b - mean_a) / mean_a * 100 if mean_a > 0 else 0

        significant = p_value < self.alpha
        if significant:
            winner = "B" if mean_b > mean_a else "A"
        else:
            winner = "tie"

        dates_a = [obs.get("date", "") for obs in self.group_a]
        dates_b = [obs.get("date", "") for obs in self.group_b]
        all_dates = sorted([d for d in dates_a + dates_b if d])

        reasoning = self._build_reasoning(winner, lift, p_value, significant, metric)

        return ExperimentResult(
            experiment_id=self.experiment_id,
            start_date=all_dates[0] if all_dates else "",
            end_date=all_dates[-1] if all_dates else "",
            n_days_a=len(self.group_a), n_days_b=len(self.group_b),
            metric_name=metric,
            mean_a=mean_a, mean_b=mean_b,
            std_a=std_a, std_b=std_b,
            lift_pct=float(lift),
            t_statistic=float(t_stat),
            p_value=float(p_value),
            significant=significant,
            winner=winner,
            reasoning=reasoning,
        )

    # ---------- Welch's t-test(不假设方差相等) ----------
    @staticmethod
    def _welch_t_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
        na, nb = len(a), len(b)
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        if va == 0 and vb == 0:
            return 0.0, 1.0
        t = (mb - ma) / np.sqrt(va / na + vb / nb + 1e-12)
        # Welch-Satterthwaite自由度
        df = (va / na + vb / nb) ** 2 / (
            (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1) + 1e-12
        )
        # 双尾p值(用正态近似,因为通常df足够大)
        from math import erf, sqrt
        z = abs(t)
        p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        return float(t), float(p)

    @staticmethod
    def _build_reasoning(winner, lift, p_value, significant, metric) -> str:
        metric_cn = {
            "total_revenue": "总收入",
            "visitors": "客流",
            "load_rate": "负载率",
            "ticket_revenue": "门票收入",
        }.get(metric, metric)
        if not significant:
            return (f"p={p_value:.3f} > 0.05,统计上不显著。"
                    f"两组{metric_cn}差异{lift:+.2f}%,需要继续观察或扩大样本。")
        if winner == "B":
            return (f"✅ B组策略显著优于A组!{metric_cn}提升{lift:+.2f}% "
                    f"(p={p_value:.4f})。建议将B策略全量上线。")
        return (f"⚠️ A组策略显著优于B组,B组{metric_cn}下降{lift:.2f}% "
                f"(p={p_value:.4f})。建议回滚B策略,排查原因。")


# ============================================================
# 常用策略函数示例
# ============================================================
def strategy_historical(context: dict) -> float:
    """基准策略: 用历史实际价格"""
    return context["historical_price"]


def strategy_business_rule(context: dict) -> float:
    """业务规则策略"""
    base = settings.pricing.base_price
    markup_map = {
        "golden_week": settings.holiday.golden_week_markup,
        "holiday": settings.holiday.holiday_markup,
        "weekend": settings.holiday.weekend_markup,
        "weekday": settings.holiday.weekday_discount,
    }
    price = base * markup_map.get(context["day_type"], 1.0)
    if context["weather"] in ("雨", "暴雨"):
        price *= 0.88
    elif context["weather"] == "酷热":
        price *= 0.94
    return np.clip(price, settings.pricing.min_price, settings.pricing.max_price)


def make_strategy_from_engine(engine, forecaster=None):
    """从 PricingEngine 实例构造策略函数"""
    def fn(context: dict) -> float:
        decision = engine.decide(
            date=context["date"],
            weather=context["weather"],
            temperature=context["temperature"],
            rainfall=context["rainfall"],
            competitor_prices={"A": 310, "B": 280, "C": 350},
            day_type=context["day_type"],
        )
        return decision.recommended_price
    return fn
