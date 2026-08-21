
from pathlib import Path
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="AI Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent  # expected: AI-Financial-Risk-Prediction-main/
RESULTS = PROJECT / "results"
LSTM = PROJECT / "lstm_gold_crypto"
MODELS = PROJECT / "models"

# ---------- Theme ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid rgba(128,128,128,.25);
        border-radius: 14px;
        padding: 18px;
        background: rgba(128,128,128,.06);
    }
    .risk-high {color:#d62728;font-weight:700;}
    .risk-medium {color:#ff8c00;font-weight:700;}
    .risk-low {color:#2ca02c;font-weight:700;}
</style>
""", unsafe_allow_html=True)

# ---------- Data helpers ----------
@st.cache_data
def load_csv(path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_data
def load_json(path):
    return json.loads(path.read_text()) if path.exists() else {}

@st.cache_data
def load_xlsx(path, sheet_name=None):
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=sheet_name)

@st.cache_data
def load_asset_prediction(asset):
    path = LSTM / "results" / f"{asset}_prediction_output.csv"
    return load_csv(path)

def pct(x):
    return f"{x:.2f}%"

def risk_from_components(fraud, market, sentiment, volatility):
    score = 0.30*fraud + 0.30*market + 0.20*sentiment + 0.20*volatility
    level = "HIGH" if score >= 67 else "MEDIUM" if score >= 34 else "LOW"
    return float(score), level

def sentiment_risk(df):
    if df.empty or "Risk Score" not in df.columns:
        return 0
    # Normalize observed risk score to a 0-100 indicator.
    mx = max(float(df["Risk Score"].max()), 1.0)
    return float(np.clip(df["Risk Score"].mean() / mx * 100, 0, 100))

# ---------- Load existing project outputs ----------
fraud_eval = load_csv(RESULTS / "fraud_model_evaluation.csv")
fraud_results = load_csv(RESULTS / "final_fraud_detection_results.csv")
sentiment = load_xlsx(RESULTS / "combined_sentiment_report.xlsx", "Sentiment Results")
topic_summary = load_xlsx(RESULTS / "combined_sentiment_report.xlsx", "Topic Summary")
gold_metrics = load_json(LSTM / "results" / "gold_metrics.json")
btc_metrics = load_json(LSTM / "results" / "bitcoin_metrics.json")
backtest = load_csv(RESULTS / "backtesting" / "combined_backtest_summary.csv")

# ---------- Sidebar ----------
st.sidebar.title("AI Financial Risk")
st.sidebar.caption("Gold • Crypto • Fraud • Sentiment")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Fraud Detection",
        "Market Prediction",
        "Sentiment Intelligence",
        "Risk Intelligence",
        "Simulation",
        "Explainable AI",
    ],
)

# ---------- Risk calculations ----------
fraud_risk = 0
if not fraud_results.empty and "Fraud_Risk_Score" in fraud_results:
    fraud_risk = float(np.clip(fraud_results["Fraud_Risk_Score"].mean(), 0, 100))

market_risk = 0
for m in [gold_metrics, btc_metrics]:
    if m:
        direction = 50 if m.get("latest_trend") == "UP" else 65
        vol = float(np.clip(m.get("latest_predicted_future_volatility", 0) * 1000, 0, 100))
        market_risk = max(market_risk, 0.55*direction + 0.45*vol)

sent_risk = sentiment_risk(sentiment)

vol_risks = []
for m in [gold_metrics, btc_metrics]:
    if m:
        vol_risks.append(float(np.clip(m.get("latest_predicted_future_volatility", 0)*1000, 0, 100)))
volatility_risk = max(vol_risks) if vol_risks else 0

overall_score, overall_level = risk_from_components(
    fraud_risk, market_risk, sent_risk, volatility_risk
)

# ---------- Overview ----------
if page == "Overview":
    st.title("AI Financial Risk Intelligence")
    st.caption("Integrated dashboard for fraud detection, market forecasting, sentiment and risk analysis")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Risk", f"{overall_score:.0f}/100", overall_level)
    c2.metric("Fraud Risk", f"{fraud_risk:.0f}/100")
    c3.metric("Market Risk", f"{market_risk:.0f}/100")
    c4.metric("Sentiment Risk", f"{sent_risk:.0f}/100")

    st.divider()
    a, b = st.columns(2)

    with a:
        st.subheader("Gold Forecast")
        if gold_metrics:
            st.metric("Latest Actual", f"{gold_metrics['latest_actual_price']:,.2f}")
            st.metric("Predicted Price", f"{gold_metrics['latest_predicted_price']:,.2f}",
                      f"{gold_metrics['latest_predicted_return_percent']:.2f}%")
            st.info(f"Model trend: **{gold_metrics['latest_trend']}**")
            df = load_asset_prediction("gold")
            if not df.empty:
                st.line_chart(df.select_dtypes(include="number").iloc[:, :2])

    with b:
        st.subheader("Bitcoin Forecast")
        if btc_metrics:
            st.metric("Latest Actual", f"{btc_metrics['latest_actual_price']:,.2f}")
            st.metric("Predicted Price", f"{btc_metrics['latest_predicted_price']:,.2f}",
                      f"{btc_metrics['latest_predicted_return_percent']:.2f}%")
            st.info(f"Model trend: **{btc_metrics['latest_trend']}**")
            df = load_asset_prediction("bitcoin")
            if not df.empty:
                st.line_chart(df.select_dtypes(include="number").iloc[:, :2])

    st.subheader("Risk Components")
    risk_df = pd.DataFrame({
        "Component": ["Fraud", "Market", "Sentiment", "Volatility"],
        "Risk": [fraud_risk, market_risk, sent_risk, volatility_risk],
    })
    fig = px.bar(risk_df, x="Component", y="Risk", range_y=[0,100], text_auto=".0f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent High-Risk Events")
    if not sentiment.empty:
        high = sentiment[sentiment.get("Risk Level", "").astype(str).str.upper().eq("HIGH")]
        cols = [c for c in ["Timestamp", "Source Type", "Sentiment", "Topics", "Risk Level", "Risk Score", "Text"] if c in high.columns]
        st.dataframe(high[cols].head(10), use_container_width=True, hide_index=True)
    else:
        st.info("Sentiment report not found.")

# ---------- Fraud ----------
elif page == "Fraud Detection":
    st.title("🚨 Fraud Detection")
    st.caption("Isolation Forest results from the existing project outputs")

    if fraud_eval.empty:
        st.warning("Fraud evaluation file not found.")
    else:
        metrics = {r["Metric"]: r["Value"] for _, r in fraud_eval.iterrows()}
        cols = st.columns(4)
        for col, name in zip(cols, ["Accuracy", "Precision", "Recall", "F1 Score"]):
            col.metric(name, f"{metrics.get(name, 0)*100:.2f}%")

    if not fraud_results.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric("Transactions", f"{len(fraud_results):,}")
        c2.metric("Predicted Fraud", f"{int(fraud_results['Fraud_Prediction'].sum()):,}")
        high_count = int((fraud_results["Fraud_Risk_Level"].astype(str).str.upper()=="HIGH").sum())
        c3.metric("High-Risk", f"{high_count:,}")

        fig = px.histogram(
            fraud_results, x="Fraud_Risk_Score", color="Fraud_Risk_Level",
            nbins=40, title="Fraud Risk Score Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Risk-Level Breakdown")
        breakdown = fraud_results["Fraud_Risk_Level"].value_counts().reset_index()
        breakdown.columns = ["Risk Level", "Count"]
        fig = px.pie(breakdown, names="Risk Level", values="Count", hole=.45)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Transaction Results")
        st.dataframe(fraud_results.head(100), use_container_width=True, hide_index=True)

# ---------- Market ----------
elif page == "Market Prediction":
    st.title("📈 Market Prediction")
    asset = st.selectbox("Asset", ["Gold", "Bitcoin"])
    key = "gold" if asset == "Gold" else "bitcoin"
    metrics = gold_metrics if key == "gold" else btc_metrics
    df = load_asset_prediction(key)

    if metrics:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Actual Price", f"{metrics['latest_actual_price']:,.2f}")
        c2.metric("Predicted Price", f"{metrics['latest_predicted_price']:,.2f}")
        c3.metric("Trend", metrics["latest_trend"])
        c4.metric("Directional Accuracy", f"{metrics['directional_accuracy']*100:.2f}%")

        st.subheader("Model Performance")
        p1,p2,p3 = st.columns(3)
        p1.metric("RMSE", f"{metrics['rmse']:,.2f}")
        p2.metric("MAE", f"{metrics['mae']:,.2f}")
        p3.metric("MAPE", f"{metrics['mape']:.2f}%")

        if not df.empty:
            st.subheader("Prediction Output")
            numeric = df.select_dtypes(include="number")
            if numeric.shape[1] >= 2:
                st.line_chart(numeric.iloc[:, :2])
            st.dataframe(df.tail(100), use_container_width=True, hide_index=True)

        st.subheader("Volatility")
        v = pd.DataFrame({
            "Metric": ["Historical Volatility (30)", "Actual Future Volatility", "Predicted Future Volatility"],
            "Value": [
                metrics["latest_historical_volatility_30"],
                metrics["latest_actual_future_volatility"],
                metrics["latest_predicted_future_volatility"],
            ],
        })
        fig = px.bar(v, x="Metric", y="Value", text_auto=".4f")
        st.plotly_chart(fig, use_container_width=True)

# ---------- Sentiment ----------
elif page == "Sentiment Intelligence":
    st.title("📰 Sentiment Intelligence")
    if sentiment.empty:
        st.warning("Combined sentiment report not found.")
    else:
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Items", f"{len(sentiment):,}")
        neg = (sentiment["Sentiment"].astype(str).str.lower()=="negative").mean()*100
        pos = (sentiment["Sentiment"].astype(str).str.lower()=="positive").mean()*100
        c2.metric("Negative", f"{neg:.1f}%")
        c3.metric("Positive", f"{pos:.1f}%")
        c4.metric("High Risk", f"{(sentiment['Risk Level'].astype(str).str.upper()=='HIGH').mean()*100:.1f}%")

        dist = sentiment["Sentiment"].value_counts().reset_index()
        dist.columns = ["Sentiment", "Count"]
        fig = px.pie(dist, names="Sentiment", values="Count", hole=.4)
        st.plotly_chart(fig, use_container_width=True)

        if not topic_summary.empty:
            st.subheader("Topic Risk")
            fig = px.bar(topic_summary, x="Topic", y="Total Risk Score", text_auto=True)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("High-Risk News / Social Signals")
        high = sentiment[sentiment["Risk Level"].astype(str).str.upper()=="HIGH"]
        cols = [c for c in ["Timestamp","Source Type","Sentiment","Score","Topics","Risk Level","Risk Score","Text"] if c in high.columns]
        st.dataframe(high[cols].head(100), use_container_width=True, hide_index=True)

# ---------- Risk ----------
elif page == "Risk Intelligence":
    st.title("🎯 Unified Risk Intelligence")
    st.caption("Weighted aggregation layer for the dashboard decision view")

    c = st.columns(4)
    for col, label, value in zip(
        c,
        ["Fraud Risk", "Market Risk", "Sentiment Risk", "Volatility Risk"],
        [fraud_risk, market_risk, sent_risk, volatility_risk]
    ):
        col.metric(label, f"{value:.0f}/100")

    st.divider()
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=overall_score,
        title={"text": f"Overall Risk — {overall_level}"},
        gauge={"axis":{"range":[0,100]}, "bar":{"thickness":0.35}}
    ))
    st.plotly_chart(gauge, use_container_width=True)

    st.subheader("Risk Formula")
    st.code(
        "Overall Risk = 0.30 × Fraud + 0.30 × Market + "
        "0.20 × Sentiment + 0.20 × Volatility",
        language="text"
    )
    st.warning("The weights above are a dashboard prototype. Validate and document the final research-paper formula before final submission.")

# ---------- Simulation ----------
elif page == "Simulation":
    st.title("🧪 Scenario Simulation")
    st.caption("Hypothetical scenario mode requested in the project roadmap")

    gold_change = st.slider("Gold change (%)", -20.0, 20.0, 0.0, 0.5)
    crypto_change = st.slider("Crypto change (%)", -30.0, 30.0, 0.0, 0.5)
    sentiment_scenario = st.selectbox("Sentiment", ["Positive", "Neutral", "Negative"])
    fraud_scenario = st.selectbox("Fraud Risk", ["Low", "Medium", "High"])

    if st.button("Run Simulation", type="primary"):
        fraud_map = {"Low":20, "Medium":55, "High":90}
        sent_map = {"Positive":15, "Neutral":45, "Negative":85}
        market = float(np.clip(50 + abs(gold_change)*1.5 + abs(crypto_change)*1.2, 0, 100))
        vol = float(np.clip(30 + abs(crypto_change)*1.8, 0, 100))
        sim, level = risk_from_components(
            fraud_map[fraud_scenario], market, sent_map[sentiment_scenario], vol
        )
        st.metric("Simulated Overall Risk", f"{sim:.0f}/100", level)
        st.progress(sim/100)
        st.write("This scenario combines the selected hypothetical inputs into the same dashboard risk framework.")

# ---------- Explainability ----------
elif page == "Explainable AI":
    st.title("🔍 Explainable AI")
    tab1, tab2, tab3 = st.tabs(["Fraud", "Gold", "Bitcoin"])
    shap_dir = RESULTS / "shap"

    with tab1:
        p = shap_dir / "fraud_feature_importance.csv"
        df = load_csv(p)
        if not df.empty:
            fig = px.bar(df.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.head(20), use_container_width=True, hide_index=True)

    with tab2:
        p = shap_dir / "gold_return_shap_importance.csv"
        df = load_csv(p)
        if not df.empty:
            fig = px.bar(df.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        p = shap_dir / "bitcoin_return_shap_importance.csv"
        df = load_csv(p)
        if not df.empty:
            fig = px.bar(df.head(15).sort_values("mean_abs_shap"), x="mean_abs_shap", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Backtesting ----------
elif page == "Backtesting":
    st.title("📊 Backtesting")
    if backtest.empty:
        st.warning("Backtesting summary not found.")
    else:
        st.dataframe(backtest, use_container_width=True, hide_index=True)
        asset = st.selectbox("Asset", backtest["Asset"].tolist())
        row = backtest[backtest["Asset"]==asset].iloc[0]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Strategy Return", row["Strategy Total Return"])
        c2.metric("Buy & Hold", row["B&H Total Return"])
        c3.metric("Strategy Sharpe", row["Strategy Sharpe"])
        c4.metric("Max Drawdown", row["Strategy Max DD"])

        comparison = pd.DataFrame({
            "Metric": ["Total Return", "CAGR", "Sharpe", "Max Drawdown"],
            "Strategy": [
                row["Strategy Total Return"], row["Strategy CAGR"],
                row["Strategy Sharpe"], row["Strategy Max DD"]
            ],
            "Buy & Hold": [
                row["B&H Total Return"], row["B&H CAGR"],
                row["B&H Sharpe"], row["B&H Max DD"]
            ]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

st.sidebar.divider()
st.sidebar.caption("Prototype dashboard • Built from the project's existing result files")
