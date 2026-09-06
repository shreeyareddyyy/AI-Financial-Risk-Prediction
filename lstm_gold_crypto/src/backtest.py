
import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from model import PriceVolatilityLSTM
from preprocessing import FEATURES, add_features

TICKERS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD"
}


def get_market_settings(freq):

    if freq == "daily":
        return "1y", "1d", 252

    elif freq == "hourly":
        return "60d", "1h", 24 * 365

    else:
        raise ValueError("Use daily or hourly.")


def calculate_metrics(equity):

    returns = equity.pct_change().dropna()

    total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100

    years = len(equity) / 252

    cagr = (
        ((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1) * 100
        if years > 0
        else 0
    )

    sharpe = (
        (returns.mean() / returns.std()) * np.sqrt(252)
        if returns.std() > 0
        else 0
    )

    rolling = equity.cummax()

    drawdown = (equity - rolling) / rolling

    max_drawdown = drawdown.min() * 100

    return {
        "total_return": round(float(total_return), 2),
        "cagr": round(float(cagr), 2),
        "sharpe_ratio": round(float(sharpe), 2),
        "max_drawdown": round(float(max_drawdown), 2)
    }


def main(asset, frequency):

    root = pathlib.Path(__file__).resolve().parents[1]

    model_path = root / f"models/{asset}_{frequency}_lstm.pt"

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    period, interval, annual_factor = get_market_settings(frequency)

    raw = yf.download(
        TICKERS[asset],
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = add_features(raw)

    feature_scaler = ckpt["feature_scaler"]
    return_scaler = ckpt["return_scaler"]

    window = int(ckpt["window_size"])

    X_scaled = feature_scaler.transform(df[FEATURES])

    model = PriceVolatilityLSTM(
        input_size=len(FEATURES),
        hidden1=96,
        hidden2=48,
        dropout=0.2
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    predictions = []
    actual_returns = []
    dates = []

    for i in range(window, len(df) - 1):

        seq = X_scaled[i-window:i]

        X = torch.tensor(seq.astype(np.float32)[None])

        with torch.no_grad():
            pred_scaled, _ = model(X)

        pred_return = return_scaler.inverse_transform(
            pred_scaled.numpy().reshape(-1, 1)
        ).ravel()[0]

        actual_return = np.log(
            df["Close"].iloc[i+1] / df["Close"].iloc[i]
        )

        predictions.append(pred_return)
        actual_returns.append(actual_return)
        dates.append(df.index[i+1])

    bt = pd.DataFrame({
        "date": dates,
        "predicted": predictions,
        "actual": actual_returns
    })

    # Use the sign of the predicted return instead of a large threshold
    bt["signal"] = np.where(
        bt["predicted"] > 0,
        "BUY",
        np.where(bt["predicted"] < 0, "AVOID", "HOLD")
)

    bt["position"] = np.where(bt["signal"] == "BUY", 1, 0)

    bt["strategy_return"] = bt["position"] * bt["actual"]
    bt["buy_hold_return"] = bt["actual"]

    bt["strategy_equity"] = np.exp(bt["strategy_return"].cumsum())
    bt["buy_hold_equity"] = np.exp(bt["buy_hold_return"].cumsum())

    strategy_metrics = calculate_metrics(bt["strategy_equity"])
    buyhold_metrics = calculate_metrics(bt["buy_hold_equity"])

    trades = int((bt["position"].diff().fillna(0) != 0).sum())
    win_rate = (bt["strategy_return"] > 0).mean() * 100

    output = {
        "asset": asset,
        "frequency": frequency,
        "ai_strategy": strategy_metrics,
        "buy_and_hold": buyhold_metrics,
        "win_rate": round(float(win_rate), 2),
        "number_of_trades": trades
    }

    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    bt.to_csv(
        results_dir / f"{asset}_{frequency}_backtest.csv",
        index=False
    )

    with open(
        results_dir / f"{asset}_{frequency}_backtest_summary.json",
        "w"
    ) as f:
        json.dump(output, f, indent=2)

    print(json.dumps(output, indent=2))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--asset", choices=TICKERS, required=True)

    parser.add_argument(
        "--frequency",
        choices=["daily", "hourly"],
        default="daily"
    )

    args = parser.parse_args()

    main(args.asset, args.frequency)