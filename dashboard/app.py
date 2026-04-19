"""
运营管理看板 (Streamlit)

启动:
  streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from data import DataLoader
from models import DemandForecaster, RLPricer, NLPReporter
from engine import PricingEngine, BundleOptimizer
from monitor import AlertEngine
from config import settings


st.set_page_config(page_title="AI定价票务平台", page_icon="🎢", layout="wide")

# ---------- 缓存 ----------
@st.cache_data(show_spinner="加载历史数据...")
def load_data():
    return DataLoader(source="mock").load_history()

@st.cache_resource(show_spinner="训练AI模型...")
def train_models(history):
    f = DemandForecaster(); f.train(history)
    r = RLPricer(); r.train(history, epochs=2)
    return f, r


# ---------- 标题 ----------
st.title("🎢 AI智能定价票务平台")
st.caption("面向乐园运营管理层 · 动态定价 / 套餐组合 / 市场预警 / 叙事日报")

history = load_data()
forecaster, rl_pricer = train_models(history)


# ============================================================
# 侧边栏: 场景输入
# ============================================================
st.sidebar.header("🎯 定价决策场景")
target_date = st.sidebar.date_input("目标日期", value=pd.to_datetime("2026-05-02"))
weather = st.sidebar.selectbox("天气", ["晴好", "雨", "暴雨", "酷热", "严寒"], index=0)
temperature = st.sidebar.slider("最高气温 (℃)", -10, 42, 24)
rainfall = st.sidebar.slider("降水量 (mm)", 0, 100, 0)
comp_a = st.sidebar.number_input("竞品A票价", 100, 600, 310)
comp_b = st.sidebar.number_input("竞品B票价", 100, 600, 280)
comp_c = st.sidebar.number_input("竞品C票价", 100, 600, 350)

from utils.date_utils import get_day_type
day_type = get_day_type(str(target_date))
st.sidebar.info(f"📅 日期类型: **{day_type}**")


# ============================================================
# 主体: 4 tab
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["💰 定价决策", "📊 历史数据看板", "🎁 套餐+预警", "🧾 AI日报"])

# ---------- 执行定价决策 ----------
engine = PricingEngine(forecaster=forecaster, rl_pricer=rl_pricer)
decision = engine.decide(
    date=str(target_date),
    weather=weather,
    temperature=temperature,
    rainfall=rainfall,
    competitor_prices={"A": comp_a, "B": comp_b, "C": comp_c},
    day_type=day_type,
)


# ========== Tab1: 定价决策 ==========
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("💎 推荐票价", f"¥{decision.recommended_price:.0f}",
                f"{(decision.recommended_price/settings.pricing.base_price-1)*100:+.1f}%")
    col2.metric("预计客流", f"{decision.predicted_visitors:,.0f} 人",
                f"负载率 {decision.load_rate*100:.1f}%")
    col3.metric("预计总收入", f"¥{decision.predicted_revenue/10000:.1f} 万")
    col4.metric("决策置信度", f"{decision.confidence*100:.0f}%")

    st.markdown("#### 三路信号分解")
    signals = pd.DataFrame({
        "信号源": ["业务规则", "ML需求曲线", "RL智能体"],
        "推荐价": [decision.business_rule_price, decision.ml_optimal_price, decision.rl_recommended_price],
        "权重": list(decision.decision_weights.values()),
    })
    fig = go.Figure()
    fig.add_trace(go.Bar(x=signals["信号源"], y=signals["推荐价"],
                         text=[f"¥{p:.0f}" for p in signals["推荐价"]],
                         textposition="auto", name="推荐价",
                         marker_color=["#6366f1", "#10b981", "#f59e0b"]))
    fig.add_hline(y=decision.recommended_price, line_dash="dash",
                  annotation_text=f"综合: ¥{decision.recommended_price:.0f}")
    fig.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"🧠 **决策解释**: {decision.reasoning}")

    # 需求-价格曲线
    st.markdown("#### 需求-价格弹性曲线 (ML模型)")
    import pandas as pd
    d = pd.to_datetime(str(target_date))
    feature_row = {
        "price": settings.pricing.base_price,
        "temperature": temperature,
        "rainfall": rainfall,
        "is_holiday": int(day_type in ("holiday", "golden_week")),
        "is_weekend": int(day_type == "weekend"),
        "day_of_week": int(d.dayofweek),
        "month": int(d.month),
        "season_id": {3:0,4:0,5:0,6:1,7:1,8:1,9:2,10:2,11:2}.get(d.month, 3),
    }
    curve = forecaster.demand_curve(feature_row)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=curve["price"], y=curve["visitors"],
                              name="预计客流", yaxis="y1", line=dict(color="#6366f1")))
    fig2.add_trace(go.Scatter(x=curve["price"], y=curve["revenue"]/10000,
                              name="门票收入(万)", yaxis="y2", line=dict(color="#10b981")))
    fig2.add_vline(x=decision.recommended_price, line_dash="dash",
                   annotation_text=f"推荐 ¥{decision.recommended_price:.0f}")
    fig2.update_layout(
        height=400,
        xaxis=dict(title="票价 (¥)"),
        yaxis=dict(title="客流(人)", side="left"),
        yaxis2=dict(title="收入(万元)", overlaying="y", side="right"),
    )
    st.plotly_chart(fig2, use_container_width=True)


# ========== Tab2: 历史数据看板 ==========
with tab2:
    st.markdown("#### 🗓 历史客流与收入趋势")
    hist_m = history.set_index("date").resample("M").agg({
        "visitors": "sum", "revenue_total": "sum", "price": "mean",
    }).reset_index()
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(hist_m, x="date", y="visitors", title="月度客流")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(hist_m, x="date", y="revenue_total", title="月度总收入")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 📊 不同日期类型客流-票价分布")
    fig = px.box(history, x="day_type", y="visitors", color="day_type",
                 category_orders={"day_type": ["weekday", "weekend", "holiday", "golden_week"]})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔍 特征重要性 (需求预测)")
    fi = forecaster.feature_importance()
    fig = px.bar(fi, x="importance", y="feature", orientation="h")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)


# ========== Tab3: 套餐+预警 ==========
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎁 套餐组合建议")
        opt = BundleOptimizer()
        bundles = opt.suggest(day_type, weather, temperature, rainfall, decision.load_rate)
        for b in bundles:
            with st.container(border=True):
                st.markdown(f"**{b.name}**")
                st.caption(b.description)
                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("价格系数", f"{b.discount*100:.0f}%")
                cc2.metric("目标客群", b.target_segment)
                cc3.metric("预期提振", f"+{b.expected_uplift*100:.0f}%")
                st.info(f"💡 {b.reasoning}")

    with c2:
        st.markdown("### 🚨 市场预警")
        alert_engine = AlertEngine()
        external = {
            "weather_forecast": {
                "rainfall_mm": rainfall,
                "temperature_high": temperature,
                "rain_probability": 0.6 if rainfall > 5 else 0.1,
            },
            "competitor_prices": {"A": comp_a, "B": comp_b, "C": comp_c},
        }
        alerts = alert_engine.check(external, decision.predicted_visitors,
                                    decision.predicted_revenue, history)
        for a in alerts:
            if a.level == "critical":
                st.error(f"🔴 **{a.title}**\n\n{a.message}\n\n→ {a.suggested_action}")
            elif a.level == "warning":
                st.warning(f"🟡 **{a.title}**\n\n{a.message}\n\n→ {a.suggested_action}")
            else:
                st.info(f"🔵 **{a.title}**\n\n{a.message}\n\n→ {a.suggested_action}")


# ========== Tab4: AI日报 ==========
with tab4:
    st.markdown("### 🧾 AI运营叙事日报")
    reporter = NLPReporter(use_llm=False)
    brief = reporter.daily_brief({
        "date": str(target_date),
        "recommended_price": decision.recommended_price,
        "predicted_visitors": decision.predicted_visitors,
        "predicted_revenue": decision.predicted_revenue,
        "load_rate": decision.load_rate,
        "weather": decision.weather,
        "day_type": decision.day_type,
        "competitor_avg": (comp_a+comp_b+comp_c)/3,
        "bundle_suggestion": bundles[0].name if bundles else "—",
        "alerts": [a.title for a in alerts if a.level in ("critical", "warning")],
    })
    st.code(brief, language="text")
    st.caption("💡 接入 Anthropic API 后可生成更自然的叙事(NLPReporter(use_llm=True))")
