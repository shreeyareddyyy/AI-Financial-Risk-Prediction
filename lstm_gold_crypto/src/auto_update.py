import pathlib
import subprocess
import pandas as pd
import yfinance as yf
import sys

TICKERS = {
    "gold": "GC=F",
    "bitcoin": "BTC-USD"
}

root = pathlib.Path(__file__).resolve().parents[1]

for asset, ticker in TICKERS.items():

    csv_path = root / f"data/{asset}_raw.csv"

    latest = yf.download(
        ticker,
        period="5d",
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    latest_date = pd.to_datetime(latest.index[-1]).date()

    if csv_path.exists():

        old = pd.read_csv(csv_path, index_col=0)

        old_date = pd.to_datetime(old.index[-1]).date()

        if latest_date == old_date:
            print(f"{asset}: already updated.")
            continue

    print(f"{asset}: new data found. Retraining...")

    subprocess.run(
    [
        sys.executable,
        str(root / "src" / "train.py"),
        "--asset",
        asset
    ],
    cwd=root,
    check=True
)

print("Update complete.")