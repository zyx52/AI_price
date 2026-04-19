"""
全功能演示 v2 —— 展示所有优化模块

流程:
  1. Ensemble预测(LightGBM + XGBoost + Prophet) + 高级特征工程
  2. RL-V2 细化状态空间 + 价格平滑约束
  3. 分时定价(早鸟/正常/夜场)
  4. 渠道差异化定价(OTA/官网/线下/团体/会员)
  5. 动态容量管理(人员/时间/应急)
  6. 策略回测:AI策略 vs 历史 vs 业务规则
  7. A/B测试演示
"""
from __future__ import annotations
import json
import os
import re
import sys
import unicodedata

_ASCII_OUTPUT_MODE = os.getenv("AI_PRICING_ASCII", "").strip().lower() in {
    "1", "true", "yes", "on",
}
_ASCII_STREAM_PATCHED = False

_ASCII_REPLACEMENTS = str.maketrans({
    "═": "=",
    "─": "-",
    "│": "|",
    "┌": "+",
    "┐": "+",
    "└": "+",
    "┘": "+",
    "•": "-",
    "【": "[",
    "】": "]",
    "（": "(",
    "）": ")",
    "，": ",",
    "。": ".",
    "：": ":",
    "；": ";",
    "～": "~",
    "·": "-",
    "¥": "CNY ",
    "🎢": "",
    "🤖": "",
    "🎯": "",
    "💎": "",
    "📄": "",
    "🔵": "*",
})

_ASCII_PHRASE_MAP = (
    ("AI智能定价票务平台", "AI Pricing Ticketing Platform"),
    ("全功能演示", "Full Demo"),
    ("演示结束", "Demo finished"),
    ("所有8大优化模块已跑通", "All 8 optimization modules completed"),
    ("FeatureCache 使用 Redis 后端", "FeatureCache Redis backend"),
    ("加载历史数据", "load history data"),
    ("历史数据加载完成", "history loaded"),
    ("开始Ensemble训练", "start Ensemble training"),
    ("预热特征缓存", "warm up feature cache"),
    ("特征缓存就绪", "feature cache ready"),
    ("使用 TimeSeriesSplit 滚动CV", "use TimeSeriesSplit rolling CV"),
    ("融合完成", "ensemble complete"),
    ("融合权重", "ensemble weights"),
    ("各模型MAPE", "model MAPE"),
    ("平均MAPE", "avg MAPE"),
    ("各折", "folds"),
    ("需求预测", "demand forecast"),
    ("开始RL-V2训练", "start RL-V2 training"),
    ("训练 RL-V2", "train RL-V2"),
    ("开始训练需求预测模型", "start demand model training"),
    ("训练完成", "training done"),
    ("状态数", "state count"),
    ("样本数", "sample count"),
    ("综合定价决策", "pricing decision"),
    ("推荐票价", "recommended price"),
    ("预计客流", "expected visitors"),
    ("负载率", "load"),
    ("分时定价", "time-slot pricing"),
    ("渠道差异化定价", "channel pricing"),
    ("各渠道策略", "channel strategies"),
    ("附加权益", "benefits"),
    ("动态容量管理", "dynamic capacity"),
    ("负载级别", "load level"),
    ("营业时间", "operating hours"),
    ("人员配置", "staffing"),
    ("成本变化", "cost delta"),
    ("项目容量", "ride capacity"),
    ("应急措施", "emergency actions"),
    ("策略回测", "strategy backtest"),
    ("回测周期", "backtest period"),
    ("历史实际", "historical"),
    ("业务规则", "rule-based"),
    ("AI智能定价", "AI pricing"),
    ("若启用AI策略,预计增收", "AI strategy projected revenue lift"),
    ("实验周期", "experiment period"),
    ("平均日收入", "avg daily revenue"),
    ("提升幅度", "lift"),
    ("统计显著", "significant"),
    ("胜出", "winner"),
    ("结论", "conclusion"),
    ("完整JSON已输出", "JSON output"),
    ("日期", "date"),
    ("晴好", "clear"),
    ("酷热", "hot"),
    ("严寒", "cold"),
    ("暴雨", "storm"),
    ("雨", "rain"),
)


def _replace_ascii_phrases(text):
    for src, dst in _ASCII_PHRASE_MAP:
        text = text.replace(src, dst)
    return text


def _configure_console_output():
    """Avoid UnicodeEncodeError on Windows terminals with legacy encodings."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(errors="replace")


def _to_ascii_text(value):
    text = str(value).translate(_ASCII_REPLACEMENTS)
    text = _replace_ascii_phrases(text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", errors="ignore").decode("ascii")
    return re.sub(r" {2,}", " ", text)


class _AsciiTextStream:
    def __init__(self, stream):
        self._stream = stream
        self._last_was_blank = False

    def write(self, data):
        text = _to_ascii_text(data)
        if text.strip() == "":
            if self._last_was_blank:
                return len(data)
            self._last_was_blank = True
            return self._stream.write("\n")
        self._last_was_blank = False
        return self._stream.write(text)

    def flush(self):
        return self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _configure_ascii_output():
    """Enable pure ASCII output mode via AI_PRICING_ASCII=1."""
    global _ASCII_STREAM_PATCHED
    if not _ASCII_OUTPUT_MODE:
        return
    if _ASCII_STREAM_PATCHED:
        return

    sys.stdout = _AsciiTextStream(sys.stdout)
    sys.stderr = _AsciiTextStream(sys.stderr)
    _ASCII_STREAM_PATCHED = True
    print("[INFO] ASCII output mode enabled via AI_PRICING_ASCII=1")


# Configure output before importing project modules.
_configure_console_output()
_configure_ascii_output()

from utils.logger import get_logger
from utils.date_utils import get_day_type
from data import DataLoader
from models import (
    DemandForecaster, EnsembleDemandForecaster, RLPricerV2, NLPReporter,
)
from engine import (
    PricingEngine, BundleOptimizer,
    TimeSlotPricer, ChannelPricer, CapacityManager,
    StrategyBacktester, ABTestExperiment,
    strategy_historical, strategy_business_rule, make_strategy_from_engine,
)
from monitor import AlertEngine

logger = get_logger("v2")


def print_banner(text):
    bar = "═" * 72
    print(f"\n{bar}\n  {text}\n{bar}")


def run_full_demo(target_date="2026-05-02"):
    print_banner("🎢 AI智能定价票务平台 V2 · 全功能演示")

    # ---------- 数据 ----------
    print("\n【1/7】加载历史数据 + Ensemble训练...")
    loader = DataLoader(source="mock")
    history = loader.load_history("2024-01-01", "2025-12-31")
    external = loader.load_external_signal(target_date)

    # ---------- Ensemble ----------
    ensemble = EnsembleDemandForecaster()
    metrics = ensemble.train(history)
    print(f"\n  🤖 Ensemble 需求预测")
    print(f"     各模型MAPE:")
    for name, m in metrics["individual_metrics"].items():
        print(f"       • {name:10s}: MAPE={m['mape']:.2f}% | MAE={m['mae']:.0f}")
    print(f"     融合权重: {metrics['weights']}")
    print(f"     ┌──────────────────────────────────────┐")
    print(f"     │ 🎯 Ensemble MAPE: {metrics['ensemble_mape']:.2f}%             │")
    print(f"     └──────────────────────────────────────┘")

    # ---------- RL V2 ----------
    print("\n【2/7】训练 RL-V2 (细化状态 + 价格平滑)...")
    rl_v2 = RLPricerV2()
    rl_info = rl_v2.train(history, epochs=5)
    print(f"     状态数: {rl_info['n_states']} (vs V1的7个,大幅细化)")
    print(f"     样本数: {rl_info['n_samples']:,}")

    # ---------- 综合定价决策 ----------
    print("\n【3/7】综合定价决策...")
    # 用老版本的 DemandForecaster 喂给 PricingEngine(避免接口变动)
    fc_old = DemandForecaster(); fc_old.train(history)
    engine = PricingEngine(forecaster=fc_old, rl_pricer=rl_v2)

    weather = external["weather_forecast"]
    rainfall = weather["rainfall_mm"]
    temp_high = weather["temperature_high"]
    weather_label = (
        "暴雨" if rainfall > 20 else "雨" if rainfall > 5 else
        "酷热" if temp_high > 35 else "严寒" if temp_high < 5 else "晴好"
    )
    decision = engine.decide(
        date=target_date, weather=weather_label,
        temperature=temp_high, rainfall=rainfall,
        competitor_prices=external["competitor_prices"],
        day_type=external["day_type"],
    )
    print(f"     日期: {decision.date} ({decision.day_type} / {decision.weather})")
    print(f"     推荐票价: ¥{decision.recommended_price:.0f}")
    print(f"     预计客流: {decision.predicted_visitors:,.0f} | 负载率{decision.load_rate*100:.1f}%")

    # ---------- 分时定价 ----------
    print("\n【4/7】分时定价...")
    ts_pricer = TimeSlotPricer()
    slot_prices = ts_pricer.compute(
        base_price=decision.recommended_price,
        day_type=decision.day_type,
        weather=decision.weather,
        temperature=temp_high,
    )
    print(ts_pricer.format_summary(slot_prices))

    # ---------- 渠道差异化 ----------
    print("\n【5/7】渠道差异化定价...")
    ch_pricer = ChannelPricer()
    channel_strategies = ch_pricer.compute(
        base_price=decision.recommended_price,
        day_type=decision.day_type,
        predicted_load=decision.load_rate,
    )
    print(ch_pricer.format_summary(channel_strategies))

    # ---------- 容量管理 ----------
    print("\n【6/7】动态容量管理...")
    cap_mgr = CapacityManager()
    plan = cap_mgr.plan(
        date=target_date,
        predicted_visitors=int(decision.predicted_visitors),
        day_type=decision.day_type,
        weather=decision.weather,
    )
    print(f"     负载级别: {plan.load_level}")
    print(f"     营业时间: {plan.opening_time} ~ {plan.closing_time} ({plan.total_operating_hours:.1f}小时)")
    print(f"     人员配置: {plan.total_staff}人")
    for role, n in plan.staff_allocation.items():
        print(f"       • {role:15s}: {n}人")
    print(f"     成本变化: ¥{plan.operational_cost_delta:+,.0f}")
    print(f"     项目容量:")
    for ride, adj in plan.ride_capacity_adjustment.items():
        print(f"       • {ride:18s}: {adj}")
    if plan.emergency_actions:
        print(f"     应急措施:")
        for a in plan.emergency_actions[:3]:
            print(f"       • {a}")

    # ---------- 回测 ----------
    print("\n【7/7】策略回测 · AI策略 vs 历史 vs 业务规则...")
    bt = StrategyBacktester(price_elasticity=0.8)

    # 只用最近半年数据加速
    recent = history.tail(180)

    result_hist = bt.run(recent, strategy_historical, "历史实际")
    result_rule = bt.run(recent, strategy_business_rule, "业务规则")
    result_ai = bt.run(recent, make_strategy_from_engine(engine), "AI智能定价")

    comparison = bt.compare([result_hist, result_rule, result_ai], baseline_index=0)
    print(f"\n     回测周期: {result_hist.period} ({result_hist.n_days}天)\n")
    print(comparison.to_string(index=False))

    rev_gain_ai = (result_ai.total_revenue - result_hist.total_revenue) / result_hist.total_revenue * 100
    print(f"\n     💎 若启用AI策略,预计增收 {rev_gain_ai:+.2f}% "
          f"(¥{(result_ai.total_revenue - result_hist.total_revenue)/1e4:+.1f}万)")

    # ---------- A/B测试 ----------
    print("\n【附加】A/B测试演示 (模拟30天实验)...")
    exp = ABTestExperiment(experiment_id="exp_20260501_ai_vs_rule")
    test_df = history.tail(30)

    for i, row in test_df.iterrows():
        date_str = str(row["date"].date())
        group = exp.assign(date_str)

        if group == "A":
            # A组: 业务规则
            price = strategy_business_rule({
                "day_type": row["day_type"], "weather": row["weather"],
                "temperature": row["temperature"], "rainfall": row["rainfall"],
                "historical_price": row["price"],
            })
        else:
            # B组: AI策略
            price = make_strategy_from_engine(engine)({
                "date": str(row["date"].date()),
                "day_type": row["day_type"], "weather": row["weather"],
                "temperature": row["temperature"], "rainfall": row["rainfall"],
                "historical_price": row["price"],
                "historical_visitors": row["visitors"],
                "is_holiday": bool(row["is_holiday"]),
                "is_weekend": bool(row["is_weekend"]),
            })
        # 模拟当日收益(简化)
        hist_visitors = row["visitors"]
        new_visitors = hist_visitors * (row["price"] / price) ** 0.8
        total_rev = price * new_visitors * (1 + 0.45)
        exp.record(group, {
            "date": date_str, "price": price, "visitors": new_visitors,
            "total_revenue": total_rev,
        })

    analysis = exp.analyze(metric="total_revenue")
    print(f"\n     实验周期: {analysis.start_date} ~ {analysis.end_date}")
    print(f"     A组(业务规则) 平均日收入: ¥{analysis.mean_a:,.0f} (n={analysis.n_days_a}天)")
    print(f"     B组(AI策略)  平均日收入: ¥{analysis.mean_b:,.0f} (n={analysis.n_days_b}天)")
    print(f"     提升幅度: {analysis.lift_pct:+.2f}%")
    print(f"     p值: {analysis.p_value:.4f} | 统计显著: {'是' if analysis.significant else '否'}")
    print(f"     胜出: {analysis.winner}")
    print(f"     结论: {analysis.reasoning}")

    # ---------- 导出 ----------
    output = {
        "ensemble_metrics": {k: v for k, v in metrics.items() if k != "individual_metrics"},
        "ensemble_individual": metrics["individual_metrics"],
        "decision": decision.to_dict(),
        "time_slot_prices": [s.to_dict() for s in slot_prices],
        "channel_strategies": [c.to_dict() for c in channel_strategies],
        "capacity_plan": plan.to_dict(),
        "backtest_comparison": comparison.to_dict(orient="records"),
        "ab_test": analysis.to_dict(),
    }
    out = f"pricing_v2_{target_date}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n📄 完整JSON已输出: {out}")
    print_banner("演示结束 · 所有8大优化模块已跑通")
    return output


if __name__ == "__main__":
    run_full_demo("2026-05-02")
