# AI-Financial-Risk-Prediction
# AI-Based Price Trend, Volatility & Risk Prediction System for Gold & Cryptocurrencies with Fraud Detection

## Team Members

- P Shreeya (1CR23AI077)
- Raghavi R (1CR23AI092)
- Sanjanna Rameshh (1CR23AI108)
- Varshini Papa Reddy (1CR23AI104)


**Guide:** Dr. Shalma  
**Department:** Artificial Intelligence and Machine Learning  
**College:** CMR Institute of Technology, Bengaluru

---

## Project Objective

The objective of this project is to develop an AI-powered financial analytics platform that predicts gold and cryptocurrency price trends, analyzes market sentiment, detects fraudulent financial transactions, and generates a unified financial risk score. The system helps investors and financial institutions make informed decisions through intelligent data analysis. 

---

## Features

- Fraud Detection using Isolation Forest
- Gold Price Prediction using LSTM
- Cryptocurrency Price Trend Prediction
- Market Volatility Analysis
- Financial News Sentiment Analysis
- Unified Financial Risk Score
- Explainable AI using SHAP
- Interactive Dashboard
- Historical Backtesting
- Simulation Mode

---

## Technologies Used

### Programming Language
- Python

### Libraries & Frameworks
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn
- SHAP
- Streamlit
- yFinance
- NLTK
- TF-IDF

### Machine Learning Models
- Isolation Forest
- LSTM (Long Short-Term Memory)
- Logistic Regression

---

## Dataset

The project uses multiple datasets:

- Historical Gold Price Data
- Cryptocurrency Price Data
- European Credit Card Fraud Detection Dataset
- Financial News Dataset
- Social Media Sentiment Data
- Yahoo Finance API for market data 

---

The dashboard provides:

- Fraud Detection Results
- Gold Price Prediction
- Cryptocurrency Trend Prediction
- Sentiment Analysis
- Risk Score Visualization
- Historical Graphs
- Financial Insights

---

## Future Enhancements

- Real-time cryptocurrency streaming
- Advanced Deep Learning models
- Portfolio recommendation system
- Cloud deployment
- Mobile application support
- Multi-language support

---

## License

This project is developed for academic purposes as part of the Major Project for the Bachelor of Engineering (Artificial Intelligence and Machine Learning) at CMR Institute of Technology.

---

## Acknowledgements

We sincerely thank our project guide **Dr. Shalma**, the Department of Artificial Intelligence and Machine Learning, CMR Institute of Technology, and all faculty members for their valuable guidance and support throughout the project. 

---

## Week 3 — SHAP Explainability and Backtesting

### How to Run

```bash
# Run the complete Week 3 pipeline (SHAP + Backtesting)
python src/run_week3.py

# Run only SHAP explainability
python src/run_week3.py --shap-only

# Run only backtesting
python src/run_week3.py --bt-only
```

---

### SHAP Explainability

**What is SHAP?**

SHAP (SHapley Additive exPlanations) is a mathematically rigorous method for explaining individual predictions made by any machine learning model. It assigns an "importance score" to each input feature for every prediction, based on Shapley values from cooperative game theory. A positive SHAP value means the feature pushed the prediction higher; a negative value means it pushed it lower.

**Why SHAP?**

In financial risk prediction, it is not enough to know *that* a model flags a transaction as fraudulent or predicts a price drop — you need to know *why*. SHAP provides:
- Global feature importance (which features matter most across all predictions)
- Individual prediction explanations (why did this specific transaction get flagged?)
- Directional insight (does high RSI increase or decrease the predicted return?)

**Which Models are Explained**

| Model | SHAP Explainer | Features |
|-------|---------------|----------|
| Isolation Forest (Fraud) | `shap.TreeExplainer` | Time, V1-V28, Amount (30 features) |
| Gold LSTM (Return head) | `shap.GradientExplainer` | 16 technical indicators, aggregated over 60-day window |
| Gold LSTM (Volatility head) | `shap.GradientExplainer` | Same 16 features |
| Bitcoin LSTM (Return head) | `shap.GradientExplainer` | Same 16 features |
| Bitcoin LSTM (Volatility head) | `shap.GradientExplainer` | Same 16 features |

**What the SHAP Plots Mean**

- **Bar plot** (`fraud_shap_summary.png`, `gold_return_shap.png`, etc.): Horizontal bars showing mean |SHAP| per feature. Longer bar = more influential feature. Red = positive average SHAP; Blue = negative average SHAP.
- **Beeswarm plot** (`fraud_shap_beeswarm.png`): Each dot is one data point. Color represents the feature value (red = high, blue = low). Position on x-axis shows SHAP value (right = increases prediction, left = decreases).
- **For LSTMs**: SHAP values are aggregated across the 60-day window (mean absolute value) so each feature gets one importance score per prediction.

**Where Outputs are Saved**

```
results/shap/
  fraud_shap_summary.png              # Global importance bar chart (Isolation Forest)
  fraud_shap_beeswarm.png             # Per-transaction beeswarm plot
  fraud_feature_importance.csv        # Ranked features with mean SHAP values
  gold_return_shap.png                # Gold LSTM: return prediction importance
  gold_volatility_shap.png            # Gold LSTM: volatility prediction importance
  gold_return_shap_importance.csv
  gold_volatility_shap_importance.csv
  bitcoin_return_shap.png
  bitcoin_volatility_shap.png
  bitcoin_return_shap_importance.csv
  bitcoin_volatility_shap_importance.csv
```

---

### Backtesting Engine v1

**Strategy Logic**

The backtesting engine uses the LSTM model's predicted log-return to generate a trading signal:

```
If predicted_return > +threshold  -->  LONG  (buy +1)
If predicted_return < -threshold  -->  SHORT (sell -1)
Otherwise                         -->  FLAT  (hold cash = 0)
```

**Signal Threshold**

The threshold is auto-computed as the **20th percentile of |predicted_return|** across the test period. This means approximately 60% of days are FLAT and 40% generate a directional signal — a conservative approach appropriate for a model with ~48% directional accuracy.

**How Look-Ahead Bias is Avoided**

The prediction CSV stores:
- `predicted_return[T]` = model output generated from the 60-day window **ending at T-1** (before day T opens)
- `actual_return[T]` = log-return that actually happened on day T

The strategy uses:
```
position[T] = signal generated from predicted_return[T]  (shift by 1 day)
strategy_return[T] = position[T] x actual_return[T]
```

This is valid because `predicted_return[T]` was computed from information available before day T. No future data is used.

**Position Logic**

| Signal | Position | Meaning |
|--------|----------|---------|
| LONG | +1 | Full allocation to asset |
| SHORT | -1 | Full short position |
| FLAT | 0 | All cash (no exposure) |

**Transaction Costs**

A cost of 0.1% (configurable via `week3_config.py`) is applied whenever the position changes. This represents realistic costs for gold futures and cryptocurrency exchanges.

**Performance Metrics Calculated**

1. Total Return
2. Annualized Return (CAGR)
3. Sharpe Ratio (annualized, with 5% risk-free rate)
4. Maximum Drawdown
5. Win Rate
6. Number of Trades
7. Annualized Volatility

Annualization: 252 days/year for Gold (futures trading days), 365 days/year for Bitcoin (24/7 market).

**Buy-and-Hold Comparison**

All metrics are reported for both the LSTM strategy and a simple buy-and-hold benchmark.

**Where Outputs are Saved**

```
results/backtesting/
  gold_backtest.csv                   # Full daily transaction log for Gold
  bitcoin_backtest.csv                # Full daily transaction log for Bitcoin
  gold_performance_comparison.csv     # Strategy vs B&H comparison table
  bitcoin_performance_comparison.csv  # Strategy vs B&H comparison table
  combined_backtest_summary.csv       # Both assets side-by-side
  gold_equity_curve.png               # Equity curve (cumulative return + portfolio value)
  bitcoin_equity_curve.png            # Same for Bitcoin
```

**Configuration**

All key parameters are in `src/week3_config.py`:

```python
INITIAL_CAPITAL = 100_000       # Starting portfolio value in USD
TRANSACTION_COST = 0.001        # 0.1% per position change
RISK_FREE_RATE = 0.05           # Annual risk-free rate for Sharpe calculation
SIGNAL_THRESHOLD = None         # Auto-computed from data (or set a float)
ALLOW_SHORT = True              # Allow short positions
SHAP_LSTM_BACKGROUND_SEQUENCES = 30
SHAP_LSTM_EXPLAIN_SEQUENCES = 50
```

---

### New Files (Week 3)

| File | Description |
|------|-------------|
| `src/week3_config.py` | Central configuration for all Week 3 parameters |
| `src/shap_explainability.py` | SHAP module for Fraud + LSTM models |
| `src/backtesting.py` | Backtesting engine with strategy, metrics, and equity curves |
| `src/run_week3.py` | Single script to run the complete Week 3 pipeline |
