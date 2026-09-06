"""Load a saved LSTM and produce live price and volatility predictions."""

import argparse
import pathlib
import json

import numpy as np
import pandas as pd
import torch
import yfinance as yf
import shap

from model import PriceVolatilityLSTM
from preprocessing import (
    add_features,
    update_features_for_prediction,
    FEATURES
)

TICKERS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD"
}
class ReturnOnlyModel(torch.nn.Module):
    """
    Wrapper so SHAP explains only the price-return output.
    """

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        pred_return, _ = self.base_model(x)
        return pred_return
# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def generate_recursive_forecast(model, window, feature_scaler, return_scaler, df, frequency):
    working = df.copy()
    forecast = {}
    horizons = {1, 3, 7, 30}
    current_price = float(working["Close"].iloc[-1])

    for day in range(1, 31):

        features = feature_scaler.transform(working[FEATURES])

        X = features[-window:].astype(np.float32)[None, :, :]

        with torch.no_grad():
            pred_return_scaled, _ = model(torch.tensor(X))

        pred_log_return = (
            return_scaler.inverse_transform(
                pred_return_scaled.numpy().reshape(-1, 1)
            ).ravel()[0]
        )

        next_close = working["Close"].iloc[-1] * np.exp(pred_log_return)

        new_row = working.iloc[-1].copy()
        new_row["Open"] = next_close
        new_row["High"] = next_close
        new_row["Low"] = next_close
        new_row["Close"] = next_close

        next_date = working.index[-1] + pd.Timedelta(days=1)
        working.loc[next_date] = new_row

        working = update_features_for_prediction(working)

        if day in horizons:
            change = round((next_close / current_price - 1) * 100, 2)

            if abs(change) < 0.01:
                change = 0.0

            if frequency == "daily":
                label = f"{day}_day"
            elif frequency == "hourly":
                label = f"{day}_hour" if day == 1 else f"{day}_hours"
            else:
                label = f"{day}_minute" if day == 1 else f"{day}_minutes"

            forecast[label] = {
                "price": round(float(next_close), 2),
                "change_percent": f"{change}%"
            }

    return forecast


def calculate_confidence_interval(predicted_price, predicted_volatility):
    margin = predicted_price * predicted_volatility * 1.96

    return {
        "lower": round(predicted_price - margin, 2),
        "upper": round(predicted_price + margin, 2),
        "interval_level": "95%"
    }


def get_reliability(price_mape, directional_accuracy=None):

    if price_mape <= 2:
        level = "HIGH"
    elif price_mape <= 5:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "level": level,
        "price_mape": f"{round(float(price_mape), 2)}%",
        "directional_accuracy": (
            f"{round(directional_accuracy * 100, 2)}%"
            if directional_accuracy is not None
            else None
        )
    }


def get_volatility_regime(volatility):

    if volatility < 0.01:
        return "LOW"
    elif volatility < 0.02:
        return "MODERATE"
    return "HIGH"


def get_recommendation(expected_return, future_volatility, reliability_level):

    if expected_return > 2 and future_volatility < 0.02:
        return "BUY"

    if expected_return > 0.5 and reliability_level == "HIGH":
        return "CONSIDER BUY"

    if expected_return < -0.5:
        return "AVOID"

    return "HOLD"


def explain_prediction(df):

    latest = df.iloc[-1]
    reasons = []

    if latest["RSI_14"] > 60:
        reasons.append("RSI indicates bullish momentum.")
    elif latest["RSI_14"] < 40:
        reasons.append("RSI indicates bearish pressure.")

    if latest["MACD"] > latest["MACD_Signal"]:
        reasons.append("MACD is above its signal line.")
    else:
        reasons.append("MACD remains below its signal line.")

    if latest["Close"] > latest["SMA_30"]:
        reasons.append("Price is trading above the 30-day moving average.")
    else:
        reasons.append("Price is below the 30-day moving average.")

    if latest["Volatility_30"] > 0.02:
        reasons.append("Market volatility is elevated.")

    return reasons


def feature_importance(df):

    latest = df.iloc[-1]

    scores = {
        "RSI": min(abs(latest["RSI_14"] - 50) / 50, 1),
        "MACD": min(abs(latest["MACD"] - latest["MACD_Signal"]) * 25, 1),
        "SMA30": min(abs(latest["Close"] - latest["SMA_30"]) / latest["Close"] * 20, 1),
        "Volatility": min(latest["Volatility_30"] / 0.03, 1)
    }

    total = sum(scores.values())

    return {
        k: f"{round(v / total * 100, 1)}%"
        for k, v in scores.items()
    }

def local_shap_explanation(model, X_scaled, feature_names, window):
    """
    Generate local SHAP values for the latest prediction.
    Works with the 3D input required by the LSTM.
    """

    wrapped_model = ReturnOnlyModel(model)
    wrapped_model.eval()

    # Create rolling sequences exactly like the LSTM training data
    sequences = []

    for i in range(window, len(X_scaled)):
        sequences.append(X_scaled[i-window:i])

    sequences = np.asarray(sequences, dtype=np.float32)

    # Need enough history
    if len(sequences) < 20:
        return {}

    # Recent background samples
    background = torch.tensor(sequences[-120:-20])

    # Latest prediction window
    sample = torch.tensor(sequences[-1:])

    explainer = shap.GradientExplainer(
        wrapped_model,
        background
    )

    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    # Average importance across time steps
    values = np.abs(shap_values[0]).mean(axis=0)
    values = np.asarray(values).reshape(-1)

    # Convert to percentages
    total = values.sum()
    if total > 0:
        values = values / total * 100

    top = np.argsort(values)[::-1][:5]

    return {
        feature_names[int(i)]: f"{round(float(values[int(i)]), 1)}%"
        for i in top
    }

def save_prediction_history(root, output):

    history_file = root / "results" / "prediction_history.csv"

    row = pd.DataFrame([{
        "timestamp": output["last_updated"],
        "asset": output["asset"],
        "mode": output["prediction_mode"],
        "current_price": round(output["current_price"], 2),
        "predicted_price": round(output["predicted_next_interval_price"], 2),
        "predicted_return_percent": output["predicted_return_percent"],
        "confidence_lower": output["confidence_interval"]["lower"],
        "confidence_upper": output["confidence_interval"]["upper"]
    }])

    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, row], ignore_index=True)

        # Keep only the latest entry for the same asset, mode and timestamp
        history = history.drop_duplicates(
            subset=["timestamp", "asset", "mode"],
            keep="last"
        )

        history.to_csv(history_file, index=False)

    else:
        row.to_csv(history_file, index=False)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main(asset, frequency="daily"):

    root = pathlib.Path(__file__).resolve().parents[1]

    model_path = root / f"models/{asset}_{frequency}_lstm.pt"

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    if frequency == "daily":
        period = "1y"
        interval = "1d"

    elif frequency == "hourly":
        period = "60d"
        interval = "1h"

    else:  # minute
        period = "7d"
        interval = "1m"

    data_file = root / f"data/{asset}_{frequency}_raw.csv"

    try:
        # Try fetching fresh market data first
        raw = yf.download(
            TICKERS[asset],
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False
        )

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if raw.empty:
            raise RuntimeError("Yahoo Finance returned empty data.")

        # Cache the latest successful download
        raw.to_csv(data_file)
        data_source = "Live Yahoo Finance"

    except Exception:
        if data_file.exists():
            raw = pd.read_csv(
                data_file,
                index_col=0,
                parse_dates=True
            )
            data_source = "Cached local data"
        else:
            raise RuntimeError(
                "No live market data available and no cached data found."
            )

    df = add_features(raw)


    # Convert latest market timestamp to IST
    ts = pd.Timestamp(df.index[-1])

    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")

    last_updated = ts.tz_convert("Asia/Kolkata").strftime("%Y-%m-%d %H:%M:%S IST")

    feature_scaler = ckpt["feature_scaler"]
    return_scaler = ckpt["return_scaler"]
    volatility_scaler = ckpt["volatility_scaler"]

    window = int(ckpt["window_size"])

    X_scaled = feature_scaler.transform(df[FEATURES])
    X = X_scaled[-window:].astype(np.float32)[None, :, :]

    model = PriceVolatilityLSTM(
        input_size=len(FEATURES),
        hidden1=96,
        hidden2=48,
        dropout=0.2
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        pred_return_scaled, pred_vol_scaled = model(torch.tensor(X))

    predicted_log_return = float(
        return_scaler.inverse_transform(
            pred_return_scaled.numpy().reshape(-1, 1)
        ).ravel()[0]
    )

    predicted_return_percent = float(np.expm1(predicted_log_return) * 100)

    predicted_future_volatility = float(
        np.exp(
            volatility_scaler.inverse_transform(
                pred_vol_scaled.numpy().reshape(-1, 1)
            ).ravel()[0]
        )
    )

    current_price = float(df["Close"].iloc[-1])

    predicted_next_interval_price = float(
        current_price * np.exp(predicted_log_return)
    )

    confidence_interval = calculate_confidence_interval(
        predicted_next_interval_price,
        predicted_future_volatility
    )

    forecast = generate_recursive_forecast(
        model,
        window,
        feature_scaler,
        return_scaler,
        df,
        frequency
    )

    reliability = get_reliability(
        ckpt.get("mape", 999),
        ckpt.get("directional_accuracy")
    )

    explanation = explain_prediction(df)
    feature_scores = feature_importance(df)

    shap_values = local_shap_explanation(
        model,
        X_scaled,
        FEATURES,
        window
    )

    trend = (
        "UP"
        if predicted_log_return > 0
        else "DOWN"
        if predicted_log_return < 0
        else "NEUTRAL"
    )
    if frequency == "daily":
        prediction_horizon = "Next day"
    elif frequency == "hourly":
        prediction_horizon = "Next hour"
    else:
        prediction_horizon = "Next minute"
    
    if frequency == "daily":
        return_target = "next_day_log_return"
    elif frequency == "hourly":
        return_target = "next_hour_log_return"
    else:
        return_target = "next_minute_log_return"

    if frequency == "daily":
        seven_key = "7_day"
    elif frequency == "hourly":
        seven_key = "7_hours"
    else:
        seven_key = "7_minutes"

    output = {
        "asset": asset,
        "ticker": TICKERS[asset],
        "data_source": data_source,
        "last_updated": last_updated,
        "current_price": round(current_price, 2),
        "predicted_next_interval_price": round(predicted_next_interval_price, 2),
        "prediction_horizon": prediction_horizon,
        "confidence_interval": confidence_interval,
        "forecast": forecast,
        "prediction_mode": frequency,
        "candle_interval": interval,
        "predicted_log_return": round(predicted_log_return, 6),
        "predicted_return_percent": f"{round(predicted_return_percent, 2)}%",
        "trend": trend,
        "current_30_day_historical_volatility": round(
            float(df["Volatility_30"].iloc[-1]), 6
        ),
        "predicted_next_5_day_volatility": round(predicted_future_volatility, 6),
        "volatility_regime": get_volatility_regime(
            float(df["Volatility_30"].iloc[-1])
        ),
        "reliability": reliability,
        "investment_recommendation": get_recommendation(
            float(forecast[seven_key]["change_percent"].replace("%", "")),
            predicted_future_volatility,
            reliability["level"]
        ),
        "prediction_explanation": explanation,
        "local_shap_explanation": shap_values,
        "feature_importance": feature_scores,
        "model": "Multivariate LSTM",
        "targets": [
            return_target,
            "next_5_day_future_volatility"
        ],
        "window_size": window,
        "features": list(FEATURES),
        "model_metrics": {
            "price_rmse": round(float(ckpt.get("rmse")), 2),
            "price_mae": round(float(ckpt.get("mae")), 2),
            "price_mape": f"{round(float(ckpt.get('mape')), 2)}%",
            "directional_accuracy": f"{round(float(ckpt.get('directional_accuracy', 0)) * 100, 2)}%",
            "volatility_rmse": round(float(ckpt.get("volatility_rmse")), 6),
            "volatility_mae": round(float(ckpt.get("volatility_mae")), 6),
            "volatility_mape": f"{round(float(ckpt.get('volatility_mape')), 2)}%"
        }
    }

    print(json.dumps(output, indent=2))
    save_prediction_history(root, output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--asset",
        choices=TICKERS,
        required=True
    )

    parser.add_argument(
        "--frequency",
        choices=["daily", "hourly", "minute"],
        default="daily"
    )

    args = parser.parse_args()

    main(args.asset, args.frequency)