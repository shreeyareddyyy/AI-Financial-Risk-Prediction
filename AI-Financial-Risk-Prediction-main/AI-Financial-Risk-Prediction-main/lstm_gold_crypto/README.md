# LSTM Price Trend & Volatility Module

Major-project module for **Gold and Bitcoin price forecasting** using daily Yahoo Finance data.

## Assets
- Gold: `GC=F` (Gold Futures)
- Bitcoin: `BTC-USD` (Bitcoin USD)

## Model
A separate multivariate LSTM is trained for each asset. The model uses a 60-day lookback and the following daily features:

`Open, High, Low, Close, Volume, Return, SMA_7, SMA_30, EMA_14, RSI_14, MACD, MACD_Signal, Volatility_7, Volatility_30, BB_Upper, BB_Lower`

The target is the **next-day closing price**. The pipeline uses chronological train/validation/test splits, training-only scaling, early stopping, learning-rate reduction, dropout and gradient clipping.

## Outputs
After training, each asset produces:
- `models/<asset>_lstm.pt`
- `results/<asset>_prediction_output.csv`
- `results/<asset>_metrics.json`
- actual-vs-predicted plot
- training/validation loss plot
- rolling-volatility plot

The CSV/JSON outputs are intended for integration with the team's risk-score and Streamlit dashboard.

## Run in VS Code
```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
python src/train.py --asset gold
python src/train.py --asset bitcoin
```

For a latest dashboard-style prediction after models are trained:
```powershell
python src/predict.py --asset gold
python src/predict.py --asset bitcoin
```

## Notes
This is an LSTM forecasting module, not financial advice. Evaluation metrics must be reported from the actual training run; do not invent values.
