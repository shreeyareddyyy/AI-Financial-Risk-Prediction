"""
backtesting.py
==============
Backtesting Engine v1 — AI Financial Risk Prediction (Week 3)

Strategy Overview
-----------------
This engine uses the LSTM model's predicted log-return to generate a
directional trading signal. The signal is applied to the *next* period's
actual return, ensuring no look-ahead bias.

Signal Logic
------------
    predicted_return > +threshold  →  LONG  (+1)
    predicted_return < -threshold  →  SHORT (-1)   [if ALLOW_SHORT=True]
    otherwise                      →  FLAT   (0)

Threshold is auto-computed as the 20th percentile of |predicted_return|
unless overridden in week3_config.py.

Look-Ahead Bias Prevention
--------------------------
The existing prediction CSVs produced by lstm_gold_crypto/src/train.py
store:
    - predicted_return[T] = model output generated from the 60-day window
                            ending at day T-1 (before T's return was known)
    - actual_return[T]    = log-return that actually occurred on day T

Therefore:
    signal[T]          = f(predicted_return[T])     -- uses pre-T info only
    strategy_return[T] = signal[T-1] * actual_return[T]

Wait—because predicted_return[T] was already computed before day T opens,
using signal[T] × actual_return[T] is valid (no lookahead).

The standard "shift-by-one" implementation:
    position[T] = signal[T-1]   (position is SET at T-1, executed at T open)
    strategy_return[T] = position[T] * actual_return[T]

This is documented below and verified step-by-step.

Transaction Costs
-----------------
Applied whenever the position changes (|position[T] - position[T-1]| > 0).
Cost = TRANSACTION_COST * |actual_price[T]| (applied as return drag).
In return terms: cost_return[T] = -TRANSACTION_COST * |position_change[T]|

Performance Metrics
-------------------
1. Total Return
2. Annualized Return (CAGR)
3. Sharpe Ratio (annualized, using risk-free rate)
4. Maximum Drawdown
5. Win Rate (fraction of non-flat trades where strategy_return > 0)
6. Number of Trades (position changes)
7. Annualized Volatility of strategy returns

Usage
-----
    from backtesting import run_backtest
    results = run_backtest("gold")
    results = run_backtest("bitcoin")

Or via run_week3.py:
    python src/run_week3.py
"""

import os
import sys
import pathlib
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------

# Allow imports from the same directory (src/)
_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from week3_config import (
    GOLD_PREDICTION_CSV,
    BITCOIN_PREDICTION_CSV,
    BACKTEST_OUTPUT_DIR,
    INITIAL_CAPITAL,
    TRANSACTION_COST,
    RISK_FREE_RATE,
    GOLD_ANNUALIZATION_FACTOR,
    BITCOIN_ANNUALIZATION_FACTOR,
    SIGNAL_THRESHOLD,
    ALLOW_SHORT,
    PLOT_DPI,
    PLOT_FIGSIZE_WIDE,
    PLOT_STYLE,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

ASSET_CONFIG = {
    "gold": {
        "csv": GOLD_PREDICTION_CSV,
        "annual_factor": GOLD_ANNUALIZATION_FACTOR,
        "color_strategy": "#f5a623",
        "color_bh": "#4a90d9",
        "label": "Gold",
    },
    "bitcoin": {
        "csv": BITCOIN_PREDICTION_CSV,
        "annual_factor": BITCOIN_ANNUALIZATION_FACTOR,
        "color_strategy": "#f7931a",
        "color_bh": "#7b68ee",
        "label": "Bitcoin",
    },
}


def _compute_threshold(predicted_returns: pd.Series, override) -> float:
    """
    Compute the signal threshold.

    If override is None: use the 20th percentile of absolute predicted returns.
    This means ~60% of days will be FLAT (within threshold), and 40% will
    generate a directional signal — a reasonable balance for a noisy predictor.

    If override is a float: use that value directly.
    """
    if override is not None:
        return float(override)

    abs_pred = predicted_returns.abs()
    threshold = float(abs_pred.quantile(0.20))

    # Guard against near-zero threshold (model with very small predictions)
    if threshold < 1e-8:
        threshold = float(abs_pred.median())
        if threshold < 1e-8:
            threshold = 0.0  # effectively always-in

    return threshold


def _compute_metrics(
    returns: pd.Series,
    annual_factor: int,
    rf_rate: float,
) -> dict:
    """
    Compute performance metrics for a return series.

    Parameters
    ----------
    returns : daily return series (NOT cumulative)
    annual_factor : 252 for gold, 365 for bitcoin
    rf_rate : annual risk-free rate

    Returns
    -------
    dict with Total Return, CAGR, Sharpe, Max Drawdown, Annualized Volatility
    """
    n = len(returns)
    if n == 0:
        return {}

    cum_return = (1 + returns).cumprod()
    total_return = float(cum_return.iloc[-1] - 1)

    years = n / annual_factor
    cagr = float((1 + total_return) ** (1 / max(years, 1e-6)) - 1) if total_return > -1 else -1.0

    ann_vol = float(returns.std() * np.sqrt(annual_factor))

    daily_rf = rf_rate / annual_factor
    excess = returns - daily_rf
    sharpe = float(excess.mean() / excess.std() * np.sqrt(annual_factor)) if excess.std() > 0 else 0.0

    # Maximum Drawdown
    running_max = cum_return.cummax()
    drawdown = (cum_return - running_max) / running_max
    max_drawdown = float(drawdown.min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def _plot_equity_curve(
    df: pd.DataFrame,
    asset: str,
    output_dir: pathlib.Path,
    annual_factor: int,
) -> pathlib.Path:
    """
    Plot and save an equity curve comparing strategy vs buy-and-hold.

    Parameters
    ----------
    df : backtest DataFrame (must include 'cum_strategy', 'cum_buy_and_hold',
         'portfolio_value', 'date' columns)
    asset : "gold" or "bitcoin"
    output_dir : where to save the PNG
    annual_factor : for display purposes

    Returns
    -------
    Path to the saved PNG
    """
    cfg = ASSET_CONFIG[asset]
    label = cfg["label"]

    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        plt.style.use("default")

    fig, axes = plt.subplots(2, 1, figsize=PLOT_FIGSIZE_WIDE, sharex=True)
    fig.suptitle(
        f"{label} — Strategy vs Buy & Hold",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    dates = pd.to_datetime(df["date"])

    # --- Top panel: Cumulative Return ---
    ax1 = axes[0]
    ax1.plot(
        dates,
        (df["cum_strategy"] - 1) * 100,
        label="LSTM Strategy",
        color=cfg["color_strategy"],
        linewidth=2,
    )
    ax1.plot(
        dates,
        (df["cum_buy_and_hold"] - 1) * 100,
        label="Buy & Hold",
        color=cfg["color_bh"],
        linewidth=2,
        linestyle="--",
    )
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cumulative Return (%)", fontsize=11)
    ax1.legend(fontsize=10)
    ax1.set_title("Cumulative Return", fontsize=12)

    # --- Bottom panel: Portfolio Value ---
    ax2 = axes[1]
    ax2.fill_between(
        dates,
        df["portfolio_value"],
        alpha=0.4,
        color=cfg["color_strategy"],
        label="Portfolio Value (Strategy)",
    )
    ax2.plot(dates, df["portfolio_value"], color=cfg["color_strategy"], linewidth=1.5)
    ax2.set_ylabel("Portfolio Value (USD)", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.set_title("Portfolio Value", fontsize=12)

    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout()

    out_path = output_dir / f"{asset}_equity_curve.png"
    fig.savefig(out_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)

    return out_path


# ---------------------------------------------------------------------------
# MAIN BACKTEST FUNCTION
# ---------------------------------------------------------------------------

def run_backtest(asset: str) -> dict:
    """
    Run the backtesting engine for a given asset.

    Parameters
    ----------
    asset : "gold" or "bitcoin"

    Returns
    -------
    dict with:
        "transaction_df"   : pd.DataFrame with full transaction log
        "strategy_metrics" : dict of performance metrics for the LSTM strategy
        "bh_metrics"       : dict of performance metrics for Buy & Hold
        "threshold"        : float, the signal threshold used
        "n_trades"         : int, number of position changes
        "win_rate"         : float
        "equity_curve_path": pathlib.Path
        "csv_path"         : pathlib.Path
    """
    if asset not in ASSET_CONFIG:
        raise ValueError(f"Unknown asset '{asset}'. Choose from: {list(ASSET_CONFIG.keys())}")

    cfg = ASSET_CONFIG[asset]
    csv_path = cfg["csv"]
    annual_factor = cfg["annual_factor"]

    # ------------------------------------------------------------------
    # 1. LOAD PREDICTION CSV
    # ------------------------------------------------------------------
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prediction CSV not found: {csv_path}\n"
            "Run lstm_gold_crypto/src/train.py first to generate predictions."
        )

    df = pd.read_csv(csv_path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Required columns
    required_cols = ["date", "actual_return", "predicted_return", "actual_price"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {csv_path.name}: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"\n{'='*60}")
    print(f"BACKTESTING: {asset.upper()}")
    print(f"{'='*60}")
    print(f"  Prediction CSV  : {csv_path.name}")
    print(f"  Rows            : {len(df)}")
    print(f"  Date range      : {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Annualization   : {annual_factor} days/year")

    # ------------------------------------------------------------------
    # 2. SIGNAL GENERATION
    # ------------------------------------------------------------------
    threshold = _compute_threshold(df["predicted_return"], SIGNAL_THRESHOLD)
    print(f"  Signal threshold: {threshold:.6f} (auto={'yes' if SIGNAL_THRESHOLD is None else 'no'})")
    print(f"  Allow short     : {ALLOW_SHORT}")
    print(f"  Transaction cost: {TRANSACTION_COST:.4%}")
    print(f"  Initial capital : ${INITIAL_CAPITAL:,.0f}")

    # Raw signal at time T (based on predicted_return[T])
    # predicted_return[T] was computed using window ending at T-1 → no lookahead
    if ALLOW_SHORT:
        df["signal"] = np.where(
            df["predicted_return"] > threshold, 1,
            np.where(df["predicted_return"] < -threshold, -1, 0)
        )
    else:
        df["signal"] = np.where(df["predicted_return"] > threshold, 1, 0)

    # ------------------------------------------------------------------
    # 3. POSITION (shift signal by 1 day to avoid using T's info at T)
    #
    # At the start of day T, we hold the position decided at T-1.
    # strategy_return[T] = position[T] × actual_return[T]
    #
    # position[T] = signal[T-1]   (shifted by 1)
    # This means: on day 0 we have no position (first day has no prior signal)
    # ------------------------------------------------------------------
    df["position"] = df["signal"].shift(1).fillna(0)

    # ------------------------------------------------------------------
    # 4. STRATEGY RETURN
    # ------------------------------------------------------------------

    # Raw strategy return (before transaction costs)
    df["strategy_return_gross"] = df["position"] * df["actual_return"]

    # Transaction cost: applied when position changes
    df["position_change"] = df["position"].diff().abs().fillna(0)
    df["transaction_cost_return"] = -TRANSACTION_COST * df["position_change"]

    # Net strategy return
    df["strategy_return"] = (
        df["strategy_return_gross"] + df["transaction_cost_return"]
    )

    # Buy and Hold return (simply hold the asset throughout)
    df["buy_and_hold_return"] = df["actual_return"]

    # ------------------------------------------------------------------
    # 5. CUMULATIVE RETURNS AND PORTFOLIO VALUE
    # ------------------------------------------------------------------
    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_buy_and_hold"] = (1 + df["buy_and_hold_return"]).cumprod()
    df["portfolio_value"] = INITIAL_CAPITAL * df["cum_strategy"]

    # ------------------------------------------------------------------
    # 6. PERFORMANCE METRICS
    # ------------------------------------------------------------------
    strategy_metrics = _compute_metrics(df["strategy_return"], annual_factor, RISK_FREE_RATE)
    bh_metrics = _compute_metrics(df["buy_and_hold_return"], annual_factor, RISK_FREE_RATE)

    # Win rate: fraction of non-flat days where strategy return > 0
    active_days = df[df["position"] != 0]
    n_active = len(active_days)
    win_rate = float((active_days["strategy_return"] > 0).sum() / max(n_active, 1))

    # Number of trades (position changes)
    n_trades = int((df["position_change"] > 0).sum())

    # Position breakdown
    n_long = int((df["position"] == 1).sum())
    n_short = int((df["position"] == -1).sum())
    n_flat = int((df["position"] == 0).sum())

    print(f"\n  Position breakdown:")
    print(f"    LONG : {n_long} days ({n_long/len(df):.1%})")
    print(f"    SHORT: {n_short} days ({n_short/len(df):.1%})")
    print(f"    FLAT : {n_flat} days ({n_flat/len(df):.1%})")
    print(f"  Trades: {n_trades}")
    print(f"  Win rate: {win_rate:.2%}")

    print(f"\n  --- Strategy Performance ---")
    print(f"    Total Return     : {strategy_metrics['total_return']:.2%}")
    print(f"    CAGR             : {strategy_metrics['cagr']:.2%}")
    print(f"    Sharpe Ratio     : {strategy_metrics['sharpe_ratio']:.3f}")
    print(f"    Max Drawdown     : {strategy_metrics['max_drawdown']:.2%}")
    print(f"    Annual Volatility: {strategy_metrics['annualized_volatility']:.2%}")

    print(f"\n  --- Buy & Hold Performance ---")
    print(f"    Total Return     : {bh_metrics['total_return']:.2%}")
    print(f"    CAGR             : {bh_metrics['cagr']:.2%}")
    print(f"    Sharpe Ratio     : {bh_metrics['sharpe_ratio']:.3f}")
    print(f"    Max Drawdown     : {bh_metrics['max_drawdown']:.2%}")
    print(f"    Annual Volatility: {bh_metrics['annualized_volatility']:.2%}")

    # ------------------------------------------------------------------
    # 7. SAVE OUTPUTS
    # ------------------------------------------------------------------
    BACKTEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Transaction log CSV
    output_cols = [
        "date", "actual_price", "predicted_return", "actual_return",
        "signal", "position", "position_change",
        "strategy_return_gross", "transaction_cost_return", "strategy_return",
        "buy_and_hold_return", "cum_strategy", "cum_buy_and_hold", "portfolio_value",
    ]
    # Add predicted_price and predicted_volatility if available
    for col in ["predicted_price", "actual_volatility", "predicted_volatility", "trend"]:
        if col in df.columns:
            output_cols.append(col)

    out_df = df[[c for c in output_cols if c in df.columns]].copy()

    # Round floats for readability
    float_cols = out_df.select_dtypes(include=[float]).columns
    out_df[float_cols] = out_df[float_cols].round(8)

    csv_out = BACKTEST_OUTPUT_DIR / f"{asset}_backtest.csv"
    out_df.to_csv(csv_out, index=False)
    print(f"\n  Saved transaction log: {csv_out}")

    # Performance comparison CSV
    comparison = pd.DataFrame({
        "Metric": [
            "Total Return",
            "CAGR (Annualized Return)",
            "Sharpe Ratio",
            "Max Drawdown",
            "Annualized Volatility",
            "Win Rate",
            "Number of Trades",
            "Signal Threshold",
            "Transaction Cost",
            "Initial Capital",
            "Final Portfolio Value",
            "Data Points",
            "Date Range Start",
            "Date Range End",
        ],
        "Strategy": [
            f"{strategy_metrics['total_return']:.4%}",
            f"{strategy_metrics['cagr']:.4%}",
            f"{strategy_metrics['sharpe_ratio']:.4f}",
            f"{strategy_metrics['max_drawdown']:.4%}",
            f"{strategy_metrics['annualized_volatility']:.4%}",
            f"{win_rate:.4%}",
            str(n_trades),
            f"{threshold:.6f}",
            f"{TRANSACTION_COST:.4%}",
            f"${INITIAL_CAPITAL:,.0f}",
            f"${df['portfolio_value'].iloc[-1]:,.2f}",
            str(len(df)),
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
        "Buy & Hold": [
            f"{bh_metrics['total_return']:.4%}",
            f"{bh_metrics['cagr']:.4%}",
            f"{bh_metrics['sharpe_ratio']:.4f}",
            f"{bh_metrics['max_drawdown']:.4%}",
            f"{bh_metrics['annualized_volatility']:.4%}",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            f"${INITIAL_CAPITAL:,.0f}",
            f"${INITIAL_CAPITAL * df['cum_buy_and_hold'].iloc[-1]:,.2f}",
            str(len(df)),
            str(df["date"].min().date()),
            str(df["date"].max().date()),
        ],
    })

    comp_csv = BACKTEST_OUTPUT_DIR / f"{asset}_performance_comparison.csv"
    comparison.to_csv(comp_csv, index=False)
    print(f"  Saved performance comparison: {comp_csv}")

    # Equity curve
    equity_path = _plot_equity_curve(df, asset, BACKTEST_OUTPUT_DIR, annual_factor)
    print(f"  Saved equity curve: {equity_path}")

    return {
        "asset": asset,
        "transaction_df": out_df,
        "strategy_metrics": strategy_metrics,
        "bh_metrics": bh_metrics,
        "threshold": threshold,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "n_long": n_long,
        "n_short": n_short,
        "n_flat": n_flat,
        "equity_curve_path": equity_path,
        "csv_path": csv_out,
        "comparison_csv_path": comp_csv,
    }


def run_all_backtests() -> dict:
    """Run backtests for both Gold and Bitcoin and save a combined summary."""
    results = {}

    for asset in ["gold", "bitcoin"]:
        try:
            results[asset] = run_backtest(asset)
        except Exception as e:
            print(f"\nERROR running {asset} backtest: {e}")
            import traceback
            traceback.print_exc()

    # Combined comparison CSV
    if results:
        rows = []
        for asset, r in results.items():
            sm = r["strategy_metrics"]
            bh = r["bh_metrics"]
            rows.append({
                "Asset": asset.capitalize(),
                "Strategy Total Return": f"{sm['total_return']:.4%}",
                "B&H Total Return": f"{bh['total_return']:.4%}",
                "Strategy CAGR": f"{sm['cagr']:.4%}",
                "B&H CAGR": f"{bh['cagr']:.4%}",
                "Strategy Sharpe": f"{sm['sharpe_ratio']:.4f}",
                "B&H Sharpe": f"{bh['sharpe_ratio']:.4f}",
                "Strategy Max DD": f"{sm['max_drawdown']:.4%}",
                "B&H Max DD": f"{bh['max_drawdown']:.4%}",
                "Strategy Ann. Vol": f"{sm['annualized_volatility']:.4%}",
                "Trades": r["n_trades"],
                "Win Rate": f"{r['win_rate']:.4%}",
                "Signal Threshold": f"{r['threshold']:.6f}",
            })
        combined = pd.DataFrame(rows)
        combined_path = BACKTEST_OUTPUT_DIR / "combined_backtest_summary.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n  Saved combined summary: {combined_path}")
        results["_combined_path"] = combined_path

    return results


if __name__ == "__main__":
    run_all_backtests()
