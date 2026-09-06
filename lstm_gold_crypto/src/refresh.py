"""
Refresh workflow for the LSTM forecasting system.

Steps:
1. Download latest Yahoo Finance data.
2. Detect whether a new candle exists.
3. Retrain the selected model.
4. Generate a fresh prediction.
5. Save it into prediction_history.csv.
"""

import argparse
import pathlib
import subprocess
import sys

import yfinance as yf


TICKERS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD"
}


def latest_market_timestamp(asset, frequency):
    ticker = TICKERS[asset]

    if frequency == "daily":
        period = "5d"
        interval = "1d"
    elif frequency == "hourly":
        period = "5d"
        interval = "1h"
    else:
        period = "5d"
        interval = "1m"

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False
    )

    if df.empty:
        raise RuntimeError("Could not download market data.")

    return str(df.index[-1])


def refresh_model(asset, frequency):
    root = pathlib.Path(__file__).resolve().parents[1]

    stamp_file = root / f"data/{asset}_{frequency}_last_update.txt"

    latest = latest_market_timestamp(asset, frequency)

    previous = None

    if stamp_file.exists():
        previous = stamp_file.read_text().strip()

    if previous == latest:
        print("\nNo new candle detected.")
        print("Using latest trained model.\n")

    else:
        print("\nNew market data detected.")
        print("Retraining model...\n")

        subprocess.run(
            [
                sys.executable,
                str(root / "src" / "train.py"),
                "--asset",
                asset,
                "--frequency",
                frequency
            ],
            check=True
        )

        stamp_file.write_text(latest)

    print("\nGenerating updated prediction...\n")

    subprocess.run(
        [
            sys.executable,
            str(root / "src" / "predict.py"),
            "--asset",
            asset,
            "--frequency",
            frequency
        ],
        check=True
    )


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

    refresh_model(args.asset, args.frequency)