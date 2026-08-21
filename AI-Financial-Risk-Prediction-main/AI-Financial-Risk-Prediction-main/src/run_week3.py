"""
run_week3.py
============
Week 3 Execution Script — AI Financial Risk Prediction
SHAP Explainability + Backtesting Engine v1

Usage
-----
    python src/run_week3.py              # Run full pipeline
    python src/run_week3.py --shap-only  # Run only SHAP
    python src/run_week3.py --bt-only    # Run only backtesting

What this script does
---------------------
1. SHAP Explainability
   a. Fraud Detection (Isolation Forest) — TreeExplainer
   b. Gold LSTM — GradientExplainer (return + volatility heads)
   c. Bitcoin LSTM — GradientExplainer (return + volatility heads)

2. Backtesting Engine v1
   a. Gold strategy backtest using existing prediction CSV
   b. Bitcoin strategy backtest using existing prediction CSV
   c. Performance metrics + equity curves + comparison CSVs

All outputs are saved under:
    results/shap/          — SHAP plots and CSVs
    results/backtesting/   — Backtest results, equity curves, metrics CSVs

Outputs Summary
---------------
results/shap/
    fraud_shap_summary.png              Global fraud feature importance (bar)
    fraud_shap_beeswarm.png             SHAP beeswarm/dot plot for fraud
    fraud_feature_importance.csv        Ranked fraud features + SHAP values
    gold_return_shap.png                Gold LSTM: SHAP for return prediction
    gold_volatility_shap.png            Gold LSTM: SHAP for volatility prediction
    gold_return_shap_importance.csv     Gold return feature importance
    gold_volatility_shap_importance.csv Gold volatility feature importance
    bitcoin_return_shap.png             Bitcoin LSTM: SHAP for return prediction
    bitcoin_volatility_shap.png         Bitcoin LSTM: SHAP for volatility prediction
    bitcoin_return_shap_importance.csv  Bitcoin return feature importance
    bitcoin_volatility_shap_importance.csv Bitcoin volatility feature importance

results/backtesting/
    gold_backtest.csv                   Daily transaction log for Gold
    bitcoin_backtest.csv                Daily transaction log for Bitcoin
    gold_performance_comparison.csv     Gold strategy vs buy-and-hold
    bitcoin_performance_comparison.csv  Bitcoin strategy vs buy-and-hold
    combined_backtest_summary.csv       Both assets side-by-side
    gold_equity_curve.png               Gold equity curve plot
    bitcoin_equity_curve.png            Bitcoin equity curve plot
"""

import sys
import os
import pathlib
import argparse
import time
import traceback

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

_LSTM_SRC = _SRC_DIR.parent / "lstm_gold_crypto" / "src"
if str(_LSTM_SRC) not in sys.path:
    sys.path.insert(0, str(_LSTM_SRC))


# ---------------------------------------------------------------------------
# BANNER
# ---------------------------------------------------------------------------

BANNER = """
================================================================
      AI Financial Risk Prediction - Week 3 Pipeline
      SHAP Explainability + Backtesting Engine v1
================================================================
"""


def print_section(title: str):
    print(f"\n{'-'*62}")
    print(f"  {title}")
    print(f"{'-'*62}")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------

def run_shap_pipeline(verbose: bool = True) -> dict:
    """Run all SHAP explanations."""
    print_section("STEP 1/2: SHAP EXPLAINABILITY")

    from shap_explainability import run_all_shap
    results = run_all_shap()
    return results


def run_backtest_pipeline(verbose: bool = True) -> dict:
    """Run all backtests."""
    print_section("STEP 2/2: BACKTESTING ENGINE v1")

    from backtesting import run_all_backtests
    results = run_all_backtests()
    return results


def print_final_summary(shap_results: dict, bt_results: dict):
    """Print a clean final summary of all results."""

    from week3_config import SHAP_OUTPUT_DIR, BACKTEST_OUTPUT_DIR

    print(f"\n{'='*62}")
    print("  WEEK 3 COMPLETE - FINAL SUMMARY")
    print(f"{'='*62}")

    # SHAP Summary
    print("\n  [SHAP OUTPUTS] (results/shap/):")
    if SHAP_OUTPUT_DIR.exists():
        shap_files = sorted(SHAP_OUTPUT_DIR.iterdir())
        for f in shap_files:
            size_kb = f.stat().st_size / 1024
            print(f"     * {f.name:<50} ({size_kb:.1f} KB)")
    else:
        print("     (No SHAP outputs generated)")

    # SHAP top features summary
    print("\n  [TOP FEATURES BY MODEL]:")
    import pandas as pd

    # Fraud
    fraud_csv = SHAP_OUTPUT_DIR / "fraud_feature_importance.csv"
    if fraud_csv.exists():
        df = pd.read_csv(fraud_csv)
        top3 = df.head(3)["feature"].tolist()
        print(f"     Fraud Detection : {', '.join(top3)}")

    # Gold Return
    gold_ret_csv = SHAP_OUTPUT_DIR / "gold_return_shap_importance.csv"
    if gold_ret_csv.exists():
        df = pd.read_csv(gold_ret_csv)
        top3 = df.head(3)["feature"].tolist()
        print(f"     Gold Return     : {', '.join(top3)}")

    # Gold Volatility
    gold_vol_csv = SHAP_OUTPUT_DIR / "gold_volatility_shap_importance.csv"
    if gold_vol_csv.exists():
        df = pd.read_csv(gold_vol_csv)
        top3 = df.head(3)["feature"].tolist()
        print(f"     Gold Volatility : {', '.join(top3)}")

    # Bitcoin Return
    btc_ret_csv = SHAP_OUTPUT_DIR / "bitcoin_return_shap_importance.csv"
    if btc_ret_csv.exists():
        df = pd.read_csv(btc_ret_csv)
        top3 = df.head(3)["feature"].tolist()
        print(f"     Bitcoin Return  : {', '.join(top3)}")

    # Bitcoin Volatility
    btc_vol_csv = SHAP_OUTPUT_DIR / "bitcoin_volatility_shap_importance.csv"
    if btc_vol_csv.exists():
        df = pd.read_csv(btc_vol_csv)
        top3 = df.head(3)["feature"].tolist()
        print(f"     Bitcoin Volatility: {', '.join(top3)}")

    # Backtesting Summary
    print("\n  [BACKTESTING RESULTS]:")
    for asset in ["gold", "bitcoin"]:
        comp_csv = BACKTEST_OUTPUT_DIR / f"{asset}_performance_comparison.csv"
        if comp_csv.exists():
            df = pd.read_csv(comp_csv)
            df = df.set_index("Metric")
            print(f"\n     {asset.upper()}:")
            for metric in ["Total Return", "CAGR (Annualized Return)", "Sharpe Ratio",
                           "Max Drawdown", "Win Rate", "Number of Trades"]:
                if metric in df.index:
                    strat = df.loc[metric, "Strategy"]
                    bh = df.loc[metric, "Buy & Hold"]
                    if bh == "N/A":
                        print(f"       {metric:<28}: Strategy={strat}")
                    else:
                        print(f"       {metric:<28}: Strategy={strat} | B&H={bh}")

    # Output paths
    print(f"\n  [OUTPUT DIRECTORIES]:")
    print(f"     SHAP      : {SHAP_OUTPUT_DIR}")
    print(f"     Backtest  : {BACKTEST_OUTPUT_DIR}")

    print(f"\n{'='*62}")
    print("  Done! All Week 3 outputs saved successfully.")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Week 3 Pipeline: SHAP Explainability + Backtesting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--shap-only",
        action="store_true",
        help="Run only SHAP explainability (skip backtesting)",
    )
    parser.add_argument(
        "--bt-only",
        action="store_true",
        help="Run only backtesting (skip SHAP)",
    )
    args = parser.parse_args()

    print(BANNER)
    start_time = time.time()

    shap_results = {}
    bt_results = {}

    # SHAP
    if not args.bt_only:
        try:
            shap_results = run_shap_pipeline()
        except Exception as e:
            print(f"\n  [ERROR] SHAP pipeline failed: {e}")
            traceback.print_exc()
            print("  Continuing to backtesting...")

    # Backtesting
    if not args.shap_only:
        try:
            bt_results = run_backtest_pipeline()
        except Exception as e:
            print(f"\n  [ERROR] Backtesting pipeline failed: {e}")
            traceback.print_exc()

    # Final summary
    try:
        print_final_summary(shap_results, bt_results)
    except Exception as e:
        print(f"\n  [WARNING] Could not print full summary: {e}")

    elapsed = time.time() - start_time
    print(f"  Total runtime: {elapsed:.1f} seconds")


if __name__ == "__main__":
    main()
