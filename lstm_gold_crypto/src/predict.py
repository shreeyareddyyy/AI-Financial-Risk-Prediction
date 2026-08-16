"""Load a saved LSTM and produce the latest price and volatility prediction."""

import argparse
import pathlib
import json

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from model import PriceVolatilityLSTM
from preprocessing import add_features, FEATURES


TICKERS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD",
}


def main(asset):

    root = pathlib.Path(__file__).resolve().parents[1]

    model_path = root / f"models/{asset}_lstm.pt"

    # ---------------------------------------------------------
    # LOAD CHECKPOINT
    # ---------------------------------------------------------

    ckpt = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False
    )

    # ---------------------------------------------------------
    # DOWNLOAD LATEST DATA
    # ---------------------------------------------------------

    raw = yf.download(
        TICKERS[asset],
        period="1y",
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # ---------------------------------------------------------
    # FEATURES
    # ---------------------------------------------------------

    df = add_features(raw)

    # ---------------------------------------------------------
    # USE EXACT TRAINING SCALERS
    # ---------------------------------------------------------

    feature_scaler = ckpt["feature_scaler"]
    return_scaler = ckpt["return_scaler"]
    volatility_scaler = ckpt["volatility_scaler"]

    # ---------------------------------------------------------
    # WINDOW
    # ---------------------------------------------------------

    window = int(ckpt["window_size"])

    if len(df) < window:
        raise ValueError(
            f"Not enough data for {window}-day window. "
            f"Only {len(df)} rows available."
        )

    # ---------------------------------------------------------
    # SCALE FEATURES
    # ---------------------------------------------------------

    X_scaled = feature_scaler.transform(
        df[FEATURES]
    )

    X = (
        X_scaled[-window:]
        [None, :, :]
        .astype(np.float32)
    )

    # ---------------------------------------------------------
    # LOAD MODEL
    # ---------------------------------------------------------

    model = PriceVolatilityLSTM(
        input_size=len(FEATURES),
        hidden1=96,
        hidden2=48,
        dropout=0.2,
    )

    model.load_state_dict(
        ckpt["model_state_dict"]
    )

    model.eval()

    # ---------------------------------------------------------
    # MODEL PREDICTION
    # ---------------------------------------------------------

    with torch.no_grad():

        pred_return_scaled, pred_vol_scaled = model(
            torch.tensor(X)
        )

    pred_return_scaled = (
        pred_return_scaled
        .cpu()
        .numpy()
        .ravel()
    )

    pred_vol_scaled = (
        pred_vol_scaled
        .cpu()
        .numpy()
        .ravel()
    )

    # ---------------------------------------------------------
    # NEXT-DAY LOG RETURN
    # ---------------------------------------------------------

    predicted_log_return = float(
        return_scaler
        .inverse_transform(
            pred_return_scaled.reshape(-1, 1)
        )
        .ravel()[0]
    )

    predicted_return_percent = float(
        np.expm1(predicted_log_return) * 100
    )

    # ---------------------------------------------------------
    # FUTURE VOLATILITY
    # ---------------------------------------------------------
    #
    # Training target is log(Future_Volatility).
    #
    # Therefore:
    #
    # scaled prediction
    #       ↓
    # inverse StandardScaler
    #       ↓
    # log volatility
    #       ↓
    # exp()
    #       ↓
    # actual volatility
    #
    # ---------------------------------------------------------

    predicted_log_volatility = float(
        volatility_scaler
        .inverse_transform(
            pred_vol_scaled.reshape(-1, 1)
        )
        .ravel()[0]
    )

    predicted_future_volatility = float(
        np.exp(predicted_log_volatility)
    )

    # Safety check
    predicted_future_volatility = max(
        predicted_future_volatility,
        0.0
    )

    # ---------------------------------------------------------
    # CURRENT PRICE
    # ---------------------------------------------------------

    current_price = float(
        df["Close"].iloc[-1]
    )

    # ---------------------------------------------------------
    # PREDICTED NEXT-DAY PRICE
    # ---------------------------------------------------------

    predicted_next_day_price = float(
        current_price *
        np.exp(predicted_log_return)
    )

    # ---------------------------------------------------------
    # TREND
    # ---------------------------------------------------------

    if predicted_log_return > 0:
        trend = "UP"
    elif predicted_log_return < 0:
        trend = "DOWN"
    else:
        trend = "NEUTRAL"

    # ---------------------------------------------------------
    # CURRENT HISTORICAL VOLATILITY
    # ---------------------------------------------------------

    current_30_day_volatility = float(
        df["Volatility_30"].iloc[-1]
    )

    # ---------------------------------------------------------
    # OUTPUT
    # ---------------------------------------------------------

    output = {

        "asset": asset,

        "ticker": TICKERS[asset],

        "date":
            str(df.index[-1].date()),

        # ---------------- PRICE ----------------

        "current_price":
            current_price,

        "predicted_next_day_price":
            predicted_next_day_price,

        "predicted_log_return":
            predicted_log_return,

        "predicted_return_percent":
            predicted_return_percent,

        "trend":
            trend,

        # ---------------- VOLATILITY ----------------

        "current_30_day_historical_volatility":
            current_30_day_volatility,

        "predicted_next_5_day_volatility":
            predicted_future_volatility,

        # ---------------- MODEL ----------------

        "model":
            "Multivariate LSTM",

        "targets": [
            "next_day_log_return",
            "next_5_day_future_volatility"
        ],

        "window_size":
            window,

        "features":
            list(FEATURES),

        # ---------------- METRICS ----------------

        "model_metrics": {

            "price_rmse":
                ckpt.get("rmse"),

            "price_mae":
                ckpt.get("mae"),

            "price_mape":
                ckpt.get("mape"),

            "directional_accuracy":
                ckpt.get("directional_accuracy"),

            "volatility_rmse":
                ckpt.get("volatility_rmse"),

            "volatility_mae":
                ckpt.get("volatility_mae"),

            "volatility_mape":
                ckpt.get("volatility_mape"),
        },
    }

    print(
        json.dumps(
            output,
            indent=2
        )
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--asset",
        choices=TICKERS,
        required=True
    )

    args = parser.parse_args()

    main(args.asset)