"""
AI智能定价票务平台 —— 主入口

演示完整流程:
  1. 加载历史数据
  2. 训练需求预测模型
  3. 训练RL定价智能体
  4. 对目标日期进行综合定价决策
  5. 生成套餐建议 + 市场预警 + 叙事日报
"""
from __future__ import annotations
import json
from datetime import datetime, timedelta

from utils.logger import get_logger
from data import DataLoader
from models import DemandForecaster, RLPricer, NLPReporter
from engine import PricingEngine, BundleOptimizer
from monitor import AlertEngine

logger = get_logger("main")


def run_pipeline(target_date: str = "2026-05-02"):
    print("\n" + "=" * 70)
    print(" " * 15 + "🎢 AI智能定价票务平台 · 演示流程")
    print("=" * 70)

    # ---------- 1. 数据层 ----------
    logger.info("【1/6】加载数据...")
    loader = DataLoader(source="mock")
    history = loader.load_history("2024-01-01", "2025-12-31")
    external = loader.load_external_signal(target_date)

    print(f"\n  📦 历史数据: {len(history):,} 条")
    print(f"     日期范围: {history['date'].min().date()} ~ {history['date'].max().date()}")
    print(f"     累计门票收入: ¥{history['revenue_ticket'].sum()/1e8:.2f}亿")
    print(f"     累计二消收入: ¥{history['revenue_secondary'].sum()/1e8:.2f}亿")

    # ---------- 2. 训练需求预测 ----------
    logger.info("【2/6】训练需求预测模型...")
    forecaster = DemandForecaster()
    metrics = forecaster.train(history)
    print(f"\n  🤖 需求预测模型训练完成")
    print(f"     验证集 MAE: {metrics['mae']:.0f} 人 | MAPE: {metrics['mape']:.2f}%")
    fi = forecaster.feature_importance()
    print(f"     TOP5特征重要性:")
    for _, r in fi.head(5).iterrows():
        print(f"       • {r['feature']:15s} → {r['importance']:.0f}")

    # ---------- 3. 训练RL定价 ----------
    logger.info("【3/6】训练RL定价智能体...")
    rl_pricer = RLPricer()
    rl_info = rl_pricer.train(history, epochs=3)
    print(f"\n  🧠 RL定价智能体训练完成")
    print(f"     状态数: {rl_info['n_states']} | 样本数: {rl_info['n_samples']:,}")

    # ---------- 4. 综合定价决策 ----------
    logger.info(f"【4/6】对目标日期 {target_date} 进行综合定价...")
    engine = PricingEngine(forecaster=forecaster, rl_pricer=rl_pricer)
    weather_map = {
        (True, False): "暴雨", (True, True): "暴雨",
    }
    weather_fc = external["weather_forecast"]
    rainfall = weather_fc["rainfall_mm"]
    temp_high = weather_fc["temperature_high"]
    weather_label = (
        "暴雨" if rainfall > 20 else
        "雨" if rainfall > 5 else
        "酷热" if temp_high > 35 else
        "严寒" if temp_high < 5 else "晴好"
    )
    decision = engine.decide(
        date=target_date,
        weather=weather_label,
        temperature=temp_high,
        rainfall=rainfall,
        competitor_prices=external["competitor_prices"],
        day_type=external["day_type"],
    )

    print(f"\n  💰 定价决策")
    print(f"     日期:         {decision.date} ({decision.day_type} / {decision.weather})")
    print(f"     业务规则价:   ¥{decision.business_rule_price:.0f}")
    print(f"     ML最优价:     ¥{decision.ml_optimal_price:.0f}")
    print(f"     RL推荐价:     ¥{decision.rl_recommended_price:.0f}")
    print(f"     ┌────────────────────────────────┐")
    print(f"     │  💎 综合推荐: ¥{decision.recommended_price:.0f}            │")
    print(f"     └────────────────────────────────┘")
    print(f"     预计客流:     {decision.predicted_visitors:,.0f} 人 (负载率 {decision.load_rate*100:.1f}%)")
    print(f"     预计总收入:   ¥{decision.predicted_revenue/10000:,.1f} 万元")
    print(f"     决策置信度:   {decision.confidence*100:.0f}%")

    # ---------- 5. 套餐建议 + 预警 ----------
    logger.info("【5/6】生成套餐建议与市场预警...")
    bundle_opt = BundleOptimizer()
    bundles = bundle_opt.suggest(
        day_type=decision.day_type,
        weather=decision.weather,
        temperature=temp_high,
        rainfall=rainfall,
        load_rate=decision.load_rate,
    )
    bundle_text = bundle_opt.format_summary(bundles)

    alert_engine = AlertEngine()
    alerts = alert_engine.check(
        external=external,
        predicted_visitors=decision.predicted_visitors,
        predicted_revenue=decision.predicted_revenue,
        history=history,
    )

    print(f"\n  🎁 套餐建议")
    for line in bundle_text.splitlines():
        print(f"     {line}")
    print(f"\n  🚨 市场预警 ({len(alerts)} 条)")
    for a in alerts:
        print(f"     {a}")
        print(f"        → 建议动作: {a.suggested_action}")

    # ---------- 6. NLP叙事日报 ----------
    logger.info("【6/6】生成NLP叙事日报...")
    reporter = NLPReporter(use_llm=False)  # 无API Key时走模板
    brief = reporter.daily_brief({
        "date": decision.date,
        "recommended_price": decision.recommended_price,
        "predicted_visitors": decision.predicted_visitors,
        "predicted_revenue": decision.predicted_revenue,
        "load_rate": decision.load_rate,
        "weather": decision.weather,
        "day_type": decision.day_type,
        "competitor_avg": sum(external["competitor_prices"].values()) / len(external["competitor_prices"]),
        "bundle_suggestion": bundles[0].name if bundles else "—",
        "alerts": [a.title for a in alerts if a.level in ("critical", "warning")],
    })
    print("\n" + brief)

    # ---------- 7. 导出JSON(供前端/API消费) ----------
    output = {
        "decision": decision.to_dict(),
        "bundles": [b.to_dict() for b in bundles],
        "alerts": [a.to_dict() for a in alerts],
        "external": external,
    }
    out_path = f"/tmp/pricing_{target_date}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  📄 完整JSON已输出到: {out_path}")
    print("=" * 70 + "\n")
    return output


if __name__ == "__main__":
    # 可切换不同日期观察不同策略
    run_pipeline(target_date="2026-05-02")   # 劳动节假期
    # run_pipeline(target_date="2026-10-03") # 国庆黄金周
    # run_pipeline(target_date="2026-03-18") # 工作日
