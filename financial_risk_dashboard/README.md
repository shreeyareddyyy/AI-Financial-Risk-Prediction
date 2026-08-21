# AI Financial Risk Dashboard

This is the first dashboard layer for the uploaded AI Financial Risk Prediction project.

## Expected folder layout

Place this `dashboard` folder inside the project root:

AI-Financial-Risk-Prediction-main/
  dashboard/
    app.py
    requirements-dashboard.txt

The app reads the existing `results/` and `lstm_gold_crypto/results/` outputs.

## Run

```bash
pip install -r dashboard/requirements-dashboard.txt
streamlit run dashboard/app.py
```

## Included pages

- Overview
- Fraud Detection
- Market Prediction
- Sentiment Intelligence
- Risk Intelligence
- Simulation
- Explainable AI
- Backtesting

## Important

The unified risk weights in this prototype are placeholders for the dashboard. Your roadmap calls for a weighted aggregation, but the exact validated final weights should be confirmed before using them in the research paper.
