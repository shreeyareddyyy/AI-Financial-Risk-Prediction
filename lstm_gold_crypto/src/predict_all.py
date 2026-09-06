import json
import subprocess
import sys
import pathlib
from refresh import refresh_model

ROOT = pathlib.Path(__file__).resolve().parents[1]

assets = ["bitcoin", "gold"]
frequencies = ["daily", "hourly", "minute"]

results = {}

# Refresh all models first
for asset in assets:
    for freq in frequencies:
        try:
            refresh_model(asset, freq)
        except Exception as e:
            print(f"Skipping refresh for {asset} {freq}: {e}")

# Generate fresh predictions
for asset in assets:
    results[asset] = {}

    for freq in frequencies:
        cmd = [
            sys.executable,
            str(ROOT / "src" / "predict.py"),
            "--asset", asset,
            "--frequency", freq,
        ]

        try:
            output = subprocess.check_output(cmd, text=True)
            results[asset][freq] = json.loads(output)
        except Exception as e:
            results[asset][freq] = {"error": str(e)}

print(json.dumps(results, indent=2))