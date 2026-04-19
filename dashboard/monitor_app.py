"""
预测偏差监控看板 (Streamlit)

启动:
  streamlit run dashboard/monitor_app.py

功能:
  - 预测 vs 实际对比图
  - 滚动 MAPE 趋势
  - 告警历史
  - AutoEncoder 偏移事件
  - 异常样本池状态
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from monitor import PredictionMonitor

st.set_page_config(page_title="预测监控看板", page_icon="📊", layout="wide")
st.title("📊 AI 预测偏差实时监控看板")
st.caption("预测 vs 实际 · 滚动 MAPE · 告警闭环")


# ============================================================
# 加载监控数据
# ============================================================
@st.cache_resource
def get_monitor():
    return PredictionMonitor(
        mape_threshold=0.12, consecutive_days=3,
        storage_path="./data/monitor",
    )


mon = get_monitor()

# 如果没有数据,生成演示数据
if mon.get_status()["n_records"] == 0:
    st.info("⚠️ 暂无监控数据,使用演示数据展示看板功能")
    import numpy as np
    rng = np.random.default_rng(42)
    for i in range(30):
        date = (datetime.now() - timedelta(days=30 - i)).strftime("%Y-%m-%d")
        actual = float(rng.normal(20000, 3000))
        # 前20天正常误差,后10天逐渐偏大
        err_ratio = 0.05 if i < 20 else 0.05 + (i - 20) * 0.02
        predicted = actual * (1 + rng.normal(0, err_ratio))
        mon.record_prediction(date, predicted)
        mon.record_actual(date, actual)
        spread = max(0.05, abs(rng.normal(0.2 + err_ratio, 0.05)))
        p50 = float(predicted)
        p10 = float(max(100.0, p50 * (1.0 - spread * 0.5)))
        p90 = float(p50 * (1.0 + spread * 0.5))
        shift_level = "critical" if i >= 28 else ("severe" if i >= 24 else "normal")
        mon.record_distribution(
            date=date,
            p10_visitors=p10,
            p50_visitors=p50,
            p90_visitors=p90,
            uncertainty_spread=(p90 - p10) / max(p50, 1.0),
            shift_level=shift_level,
            fallback_mode=(shift_level == "critical"),
        )


# ============================================================
# 顶部KPI
# ============================================================
st.markdown("---")
col1, col2, col3, col4, col5, col6 = st.columns(6)
status = mon.get_status()
col1.metric("监控天数", status["n_records"])
mape_val = status.get("rolling_mape") or 0
col2.metric("滚动 MAPE", f"{mape_val:.2%}",
            delta=f"{(mape_val - status.get('mape_threshold', 0.12))*100:+.1f}pp 相对阈值",
            delta_color="inverse")
col3.metric("告警总数", status["alerts_total"])
col4.metric("状态", "🔴 超标中" if status.get("threshold_breach_now") else "🟢 正常")
col5.metric("OOD触发次数", int(status.get("ood_trigger_count", 0)))
col6.metric("熔断率", f"{float(status.get('fallback_rate', 0.0)):.1%}")


# ============================================================
# 主图: 预测vs实际
# ============================================================
st.markdown("### 📈 预测 vs 实际客流对比")
df = mon.get_dataframe()
if not df.empty:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["predicted"], name="预测",
        line=dict(color="#6366f1", width=2.5),
    ))
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["actual"], name="实际",
        line=dict(color="#10b981", width=2.5),
    ))
    fig.update_layout(
        height=360, xaxis_title="日期", yaxis_title="客流(人)",
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 预测分布水位图 (P10-P90)
# ============================================================
st.markdown("### 🌊 预测分布水位图 (P10-P90)")
dist_df = mon.get_distribution_dataframe()
if not dist_df.empty:
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Scatter(
        x=dist_df["date"], y=dist_df["p10_visitors"],
        mode="lines", line=dict(color="rgba(59,130,246,0.2)", width=1),
        name="P10",
    ))
    fig_dist.add_trace(go.Scatter(
        x=dist_df["date"], y=dist_df["p90_visitors"],
        mode="lines", fill="tonexty", fillcolor="rgba(59,130,246,0.18)",
        line=dict(color="rgba(59,130,246,0.25)", width=1),
        name="P10-P90区间",
    ))
    fig_dist.add_trace(go.Scatter(
        x=dist_df["date"], y=dist_df["p50_visitors"],
        mode="lines", line=dict(color="#1d4ed8", width=2.5),
        name="P50",
    ))
    fig_dist.update_layout(
        height=340,
        xaxis_title="日期",
        yaxis_title="客流(人)",
        hovermode="x unified",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("平均散布度", f"{dist_df['uncertainty_spread'].mean():.2%}")
    c2.metric("高散布(>40%)占比", f"{(dist_df['uncertainty_spread'] > 0.4).mean():.1%}")
else:
    st.info("暂无分位数分布监控数据")


# ============================================================
# 滚动MAPE趋势
# ============================================================
st.markdown("### 📊 滚动 MAPE 趋势")
if not df.empty and len(df) >= 3:
    df["rolling_mape_7d"] = df["abs_pct_error"].rolling(7, min_periods=1).mean()

    fig2 = go.Figure()
    # 柱: 每日误差
    fig2.add_trace(go.Bar(
        x=df["date"], y=df["abs_pct_error"] * 100,
        name="每日 APE(%)",
        marker=dict(color=df["exceed_threshold"].map({True: "#ef4444", False: "#94a3b8"})),
    ))
    # 线: 滚动MAPE
    fig2.add_trace(go.Scatter(
        x=df["date"], y=df["rolling_mape_7d"] * 100,
        name="滚动7天 MAPE(%)",
        line=dict(color="#f59e0b", width=3),
    ))
    # 阈值线
    threshold_pct = status.get("mape_threshold", 0.12) * 100
    fig2.add_hline(
        y=threshold_pct, line_dash="dash", line_color="red",
        annotation_text=f"告警阈值 {threshold_pct:.0f}%",
    )
    fig2.update_layout(
        height=320, xaxis_title="日期", yaxis_title="误差%",
        hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)


# ============================================================
# 告警历史
# ============================================================
st.markdown("### 🚨 告警历史")
alerts = mon.get_alerts()
if alerts:
    for a in reversed(alerts[-5:]):
        ts = datetime.fromtimestamp(a.alert_time).strftime("%Y-%m-%d %H:%M:%S")
        st.error(f"**[{ts}]** {a.message}")
else:
    st.success("✅ 无告警")


# ============================================================
# OOD / 熔断监控
# ============================================================
st.markdown("### 🧯 OOD触发与熔断监控")
if not dist_df.empty:
    level_counts = dist_df["shift_level"].value_counts().rename_axis("shift_level").reset_index(name="count")
    fig_ood = px.bar(
        level_counts,
        x="shift_level",
        y="count",
        color="shift_level",
        color_discrete_map={
            "normal": "#94a3b8",
            "light": "#f59e0b",
            "severe": "#ef4444",
            "critical": "#991b1b",
        },
        title="OOD等级触发次数",
    )
    fig_ood.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig_ood, use_container_width=True)

    fallback_daily = dist_df[["date", "fallback_mode"]].copy()
    fallback_daily["fallback_mode"] = fallback_daily["fallback_mode"].astype(int)
    fallback_daily = fallback_daily.sort_values("date")
    fallback_daily["fallback_rate_7d"] = fallback_daily["fallback_mode"].rolling(7, min_periods=1).mean()
    fig_fb = go.Figure()
    fig_fb.add_trace(go.Scatter(
        x=fallback_daily["date"], y=fallback_daily["fallback_rate_7d"] * 100,
        mode="lines+markers", line=dict(color="#b91c1c", width=2.5),
        name="7日熔断率",
    ))
    fig_fb.update_layout(height=280, yaxis_title="熔断率(%)", xaxis_title="日期")
    st.plotly_chart(fig_fb, use_container_width=True)
else:
    st.info("暂无 OOD / 熔断明细")


# ============================================================
# 详细表格
# ============================================================
with st.expander("🔍 查看详细数据"):
    if not df.empty:
        df_display = df.copy()
        df_display["预测"] = df_display["predicted"].round(0).astype(int)
        df_display["实际"] = df_display["actual"].round(0).astype(int)
        df_display["误差%"] = (df_display["abs_pct_error"] * 100).round(2)
        df_display["超标"] = df_display["exceed_threshold"].map({True: "❌", False: "✓"})
        st.dataframe(
            df_display[["date", "预测", "实际", "误差%", "超标"]],
            use_container_width=True, hide_index=True,
        )


st.markdown("---")
st.caption(f"🔄 数据每日更新 · 阈值: MAPE>{status.get('mape_threshold', 0.12):.0%}连续"
           f"{status.get('rolling_window', 14)}天的"
           f"{status.get('rolling_window',14)}天滚动窗口")
