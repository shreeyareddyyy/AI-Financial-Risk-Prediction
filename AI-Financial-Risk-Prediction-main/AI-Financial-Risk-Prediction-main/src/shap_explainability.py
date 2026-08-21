"""
shap_explainability.py
======================
SHAP (SHapley Additive exPlanations) Explainability Layer — Week 3

This module generates SHAP-based explanations for:
  1. Fraud Detection  — Isolation Forest (TreeExplainer)
  2. Gold LSTM        — PriceVolatilityLSTM (GradientExplainer)
  3. Bitcoin LSTM     — PriceVolatilityLSTM (GradientExplainer)

Why Different Explainers?
--------------------------
  - Isolation Forest is a tree-based ensemble → TreeExplainer is exact, fast,
    and native (no approximations needed).
  - LSTM is a PyTorch neural network → GradientExplainer uses backpropagation
    through the network to estimate SHAP values efficiently. It is the
    recommended approach for sequential neural networks in PyTorch.

LSTM SHAP Notes
---------------
The LSTM takes inputs of shape (batch, window=60, features=16).
SHAP values will therefore be shape (n_samples, 60, 16).

To produce interpretable feature-level importance:
  - Aggregate SHAP values across the time dimension (mean absolute value over
    the 60-day window) → shape (n_samples, 16) → mean across samples → (16,)
  - This gives a per-feature "average influence on prediction over the window"

The model has two output heads:
  - return_head    → predicts next-day log return
  - volatility_head → predicts log of 5-day future volatility

We wrap the model in two thin PyTorch wrappers (ReturnWrapper,
VolatilityWrapper) that output only one head each, and run SHAP separately
for both heads. This gives 4 explanation sets:
  - Gold return, Gold volatility, Bitcoin return, Bitcoin volatility

Fraud SHAP Notes
----------------
The Isolation Forest model was trained on:
  Time, V1–V28, Amount  (30 features total)

The raw creditcard.csv dataset may not be present in this repository
(it is ~150 MB and typically excluded via .gitignore). We therefore:
  1. Attempt to load creditcard.csv from known paths.
  2. If not found: generate a synthetic background dataset that mirrors
     the statistical properties expected by the model (standard normal
     V1-V28, uniform Time, and log-normal Amount — matching the preprocessing
     applied in src/preprocessing.py: StandardScaler on Time and Amount).

The SHAP values represent the contribution of each feature to moving a
transaction's anomaly score away from the baseline (average normal score).
Positive SHAP → makes the transaction look MORE anomalous.
Negative SHAP → makes it look LESS anomalous.

Usage
-----
    from shap_explainability import (
        run_fraud_shap,
        run_lstm_shap,
        run_all_shap,
    )

    run_fraud_shap()
    run_lstm_shap("gold")
    run_lstm_shap("bitcoin")
    run_all_shap()   # runs all three

Or via run_week3.py:
    python src/run_week3.py
"""

import sys
import os
import pathlib
import warnings
import traceback

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for saving PNGs
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------------

_SRC_DIR = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _SRC_DIR.parent
_LSTM_SRC = _REPO_ROOT / "lstm_gold_crypto" / "src"

for _p in [str(_SRC_DIR), str(_LSTM_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# IMPORTS (with clear error messages)
# ---------------------------------------------------------------------------

try:
    import shap
except ImportError:
    raise ImportError(
        "SHAP is not installed. Run: pip install shap\n"
        "Then re-run this script."
    )

try:
    import joblib
except ImportError:
    raise ImportError("joblib is required. Run: pip install joblib")

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is not installed. Run: pip install torch\n"
        "LSTM SHAP explanations require PyTorch."
    )

# LSTM model (from lstm_gold_crypto/src/model.py)
try:
    from model import PriceVolatilityLSTM
except ImportError:
    raise ImportError(
        f"Cannot import PriceVolatilityLSTM from {_LSTM_SRC}.\n"
        "Ensure lstm_gold_crypto/src/model.py exists."
    )

# LSTM preprocessing (for feature names and data loading)
try:
    from preprocessing import add_features, FEATURES as LSTM_FEATURES
except ImportError:
    raise ImportError(
        f"Cannot import from preprocessing in {_LSTM_SRC}."
    )

# Week 3 config
from week3_config import (
    FRAUD_MODEL_PATH,
    FRAUD_RISK_SCALER_PATH,
    FRAUD_RESULTS_CSV,
    GOLD_MODEL_PATH,
    BITCOIN_MODEL_PATH,
    GOLD_PREDICTION_CSV,
    BITCOIN_PREDICTION_CSV,
    SHAP_OUTPUT_DIR,
    SHAP_FRAUD_BACKGROUND_SAMPLES,
    SHAP_FRAUD_EXPLAIN_SAMPLES,
    SHAP_LSTM_BACKGROUND_SEQUENCES,
    SHAP_LSTM_EXPLAIN_SEQUENCES,
    SHAP_RANDOM_SEED,
    PLOT_DPI,
    PLOT_STYLE,
)

# Fraud feature names (same as model.feature_names_in_)
FRAUD_FEATURES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ============================================================
# PLOTTING HELPERS
# ============================================================

def _apply_style():
    try:
        plt.style.use(PLOT_STYLE)
    except Exception:
        plt.style.use("default")


def _save_shap_bar_plot(
    shap_values: np.ndarray,
    feature_names: list,
    title: str,
    output_path: pathlib.Path,
    top_n: int = 15,
):
    """
    Save a horizontal bar chart of mean |SHAP| values (global feature importance).

    Parameters
    ----------
    shap_values : 2D array (n_samples, n_features)
    feature_names : list of feature name strings
    title : plot title
    output_path : where to save
    top_n : how many top features to show
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_val = np.mean(shap_values, axis=0)

    order = np.argsort(mean_abs)[::-1][:top_n]
    ordered_features = [feature_names[i] for i in order]
    ordered_abs = mean_abs[order]

    _apply_style()
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))

    colors = ["#e74c3c" if mean_val[order[i]] >= 0 else "#3498db"
              for i in range(len(order))]

    bars = ax.barh(
        range(len(ordered_features)),
        ordered_abs[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )
    ax.set_yticks(range(len(ordered_features)))
    ax.set_yticklabels(ordered_features[::-1], fontsize=10)
    ax.set_xlabel("Mean |SHAP value|", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", label="Avg SHAP > 0 (increases risk/prediction)"),
        Patch(facecolor="#3498db", label="Avg SHAP < 0 (decreases risk/prediction)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close(fig)


def _save_feature_importance_csv(
    shap_values: np.ndarray,
    feature_names: list,
    output_path: pathlib.Path,
):
    """
    Save a CSV: feature, mean_abs_shap, mean_shap, rank
    """
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    mean_shap = np.mean(shap_values, axis=0)

    order = np.argsort(mean_abs)[::-1]
    df = pd.DataFrame({
        "feature": [feature_names[i] for i in order],
        "mean_abs_shap": mean_abs[order],
        "mean_shap": mean_shap[order],
        "rank": range(1, len(feature_names) + 1),
    })
    df.to_csv(output_path, index=False)
    return df


# ============================================================
# FRAUD SHAP
# ============================================================

def _get_fraud_background(n_samples: int, rng: np.random.Generator) -> pd.DataFrame:
    """
    Build a background dataset for the Isolation Forest SHAP explainer.

    Priority:
    1. Use real data from creditcard.csv if it exists.
    2. Use the fraud results CSV (which only has prediction outputs, not features).
    3. Fall back to synthetic data that mirrors the expected feature distribution.

    For the IsolationForest trained on this dataset:
    - Time: scaled by StandardScaler (so roughly N(0,1))
    - V1–V28: PCA components → roughly N(0, sigma_i) with varying sigmas
    - Amount: scaled by StandardScaler (so roughly N(0,1))

    We generate synthetic data from N(0,1) for all features as a fallback.
    This is an approximation but allows TreeExplainer to produce valid marginal
    SHAP values through feature perturbation.
    """
    # Try to find creditcard.csv
    possible_paths = [
        _REPO_ROOT / "data" / "creditcard.csv",
        _REPO_ROOT / "creditcard.csv",
        pathlib.Path("data") / "creditcard.csv",
        pathlib.Path("creditcard.csv"),
    ]
    for p in possible_paths:
        if p.exists():
            print(f"  Loading real fraud data from: {p}")
            df = pd.read_csv(p, usecols=FRAUD_FEATURES, nrows=n_samples * 5)
            # Apply StandardScaler to Time and Amount (as done in preprocessing.py)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
            # Sample
            df = df.sample(n=min(n_samples, len(df)), random_state=SHAP_RANDOM_SEED)
            return df[FRAUD_FEATURES].reset_index(drop=True)

    # Fallback: synthetic background
    print("  [INFO] creditcard.csv not found — using synthetic background dataset.")
    print("         SHAP values will be approximate (marginal/interventional).")
    print("         For exact SHAP, place creditcard.csv in data/creditcard.csv")

    n = n_samples
    data = {}

    # Time: StandardScaler output → N(0,1)
    data["Time"] = rng.standard_normal(n)

    # V1–V28: PCA components → N(0,1) is a reasonable approximation
    for i in range(1, 29):
        data[f"V{i}"] = rng.standard_normal(n)

    # Amount: StandardScaler output → N(0,1)
    data["Amount"] = rng.standard_normal(n)

    return pd.DataFrame(data, columns=FRAUD_FEATURES)


def run_fraud_shap() -> dict:
    """
    Generate SHAP explanations for the Isolation Forest fraud model.

    Returns dict with paths to saved outputs.
    """
    print(f"\n{'='*60}")
    print("SHAP EXPLAINABILITY: FRAUD DETECTION (Isolation Forest)")
    print(f"{'='*60}")

    # --- Load model ---
    if not FRAUD_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Fraud model not found: {FRAUD_MODEL_PATH}\n"
            "Ensure models/final_isolation_forest_model.pkl exists."
        )

    print(f"  Loading model: {FRAUD_MODEL_PATH.name}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = joblib.load(FRAUD_MODEL_PATH)

    print(f"  Model: IsolationForest, n_estimators={model.n_estimators}, "
          f"contamination={model.contamination}")
    print(f"  Features ({model.n_features_in_}): {list(model.feature_names_in_)}")

    # Verify feature names match
    model_features = list(model.feature_names_in_)
    if model_features != FRAUD_FEATURES:
        print(f"  WARNING: Feature mismatch!")
        print(f"    Expected: {FRAUD_FEATURES}")
        print(f"    Model has: {model_features}")
        # Use model's actual features
        fraud_features = model_features
    else:
        fraud_features = FRAUD_FEATURES

    # --- Build background dataset ---
    rng = np.random.default_rng(SHAP_RANDOM_SEED)
    print(f"  Building background dataset ({SHAP_FRAUD_BACKGROUND_SAMPLES} samples)...")
    background_df = _get_fraud_background(SHAP_FRAUD_BACKGROUND_SAMPLES, rng)

    # Ensure column order matches model
    background_df = background_df[fraud_features]

    # Build explain dataset (same source, different samples)
    explain_df = _get_fraud_background(
        SHAP_FRAUD_EXPLAIN_SAMPLES + SHAP_FRAUD_BACKGROUND_SAMPLES, rng
    )
    explain_df = explain_df[fraud_features].iloc[SHAP_FRAUD_BACKGROUND_SAMPLES:].reset_index(drop=True)

    # --- SHAP TreeExplainer ---
    print(f"  Creating TreeExplainer...")
    explainer = shap.TreeExplainer(
        model,
        data=background_df,
        feature_perturbation="interventional",
    )

    print(f"  Computing SHAP values for {len(explain_df)} samples...")
    # For IsolationForest, decision_function is the score to explain
    # shap_values shape: (n_samples, n_features)
    shap_vals = explainer.shap_values(explain_df)

    # If shap_vals is a list (multi-output), take first
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]

    print(f"  SHAP values shape: {shap_vals.shape}")

    # --- Outputs ---
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Feature importance CSV
    fi_csv_path = SHAP_OUTPUT_DIR / "fraud_feature_importance.csv"
    fi_df = _save_feature_importance_csv(shap_vals, fraud_features, fi_csv_path)
    print(f"  Saved feature importance CSV: {fi_csv_path.name}")
    print(f"  Top 5 features:")
    for _, row in fi_df.head(5).iterrows():
        direction = "(increases anomaly)" if row["mean_shap"] > 0 else "(decreases anomaly)"
        print(f"    {int(row['rank'])}. {row['feature']}: {row['mean_abs_shap']:.6f} {direction}")

    # 2. SHAP bar plot (global feature importance)
    bar_path = SHAP_OUTPUT_DIR / "fraud_shap_summary.png"
    _save_shap_bar_plot(
        shap_vals,
        fraud_features,
        title="Fraud Detection — SHAP Feature Importance (Isolation Forest)\n"
              "Mean |SHAP value| across all explained samples",
        output_path=bar_path,
        top_n=20,
    )
    print(f"  Saved bar plot: {bar_path.name}")

    # 3. SHAP beeswarm/dot summary plot
    summary_path = SHAP_OUTPUT_DIR / "fraud_shap_beeswarm.png"
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        explain_df,
        feature_names=fraud_features,
        show=False,
        max_display=20,
        plot_type="dot",
    )
    plt.title(
        "Fraud Detection — SHAP Beeswarm Plot (Isolation Forest)\n"
        "Each dot = one transaction; color = feature value",
        fontsize=11,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(summary_path, dpi=PLOT_DPI, bbox_inches="tight")
    plt.close("all")
    print(f"  Saved beeswarm plot: {summary_path.name}")

    return {
        "shap_values": shap_vals,
        "feature_names": fraud_features,
        "feature_importance_csv": fi_csv_path,
        "bar_plot": bar_path,
        "summary_plot": summary_path,
    }


# ============================================================
# LSTM SHAP WRAPPERS
# ============================================================

class _ReturnWrapper(nn.Module):
    """Wraps PriceVolatilityLSTM to output only the return head."""
    def __init__(self, model: PriceVolatilityLSTM):
        super().__init__()
        self._model = model

    def forward(self, x):
        ret, _ = self._model(x)
        return ret


class _VolatilityWrapper(nn.Module):
    """Wraps PriceVolatilityLSTM to output only the volatility head."""
    def __init__(self, model: PriceVolatilityLSTM):
        super().__init__()
        self._model = model

    def forward(self, x):
        _, vol = self._model(x)
        return vol


# ============================================================
# LSTM SHAP
# ============================================================

def run_lstm_shap(asset: str) -> dict:
    """
    Generate SHAP explanations for Gold or Bitcoin LSTM.

    Uses GradientExplainer on the PyTorch PriceVolatilityLSTM model.
    Produces separate explanations for:
      - Return prediction
      - Volatility prediction

    Parameters
    ----------
    asset : "gold" or "bitcoin"

    Returns
    -------
    dict with paths to all saved outputs and SHAP arrays
    """
    assert asset in ("gold", "bitcoin"), f"Unknown asset: {asset}"

    print(f"\n{'='*60}")
    print(f"SHAP EXPLAINABILITY: {asset.upper()} LSTM")
    print(f"{'='*60}")

    # --- Load checkpoint ---
    model_path = GOLD_MODEL_PATH if asset == "gold" else BITCOIN_MODEL_PATH
    if not model_path.exists():
        raise FileNotFoundError(
            f"LSTM checkpoint not found: {model_path}\n"
            "Run lstm_gold_crypto/src/train.py first."
        )

    print(f"  Loading checkpoint: {model_path.name}")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    feature_scaler = ckpt["feature_scaler"]
    return_scaler = ckpt["return_scaler"]
    volatility_scaler = ckpt["volatility_scaler"]
    window = int(ckpt["window_size"])

    print(f"  Window size: {window}")
    print(f"  Input features ({len(LSTM_FEATURES)}): {LSTM_FEATURES}")

    # --- Load raw data ---
    raw_data_path = _REPO_ROOT / "lstm_gold_crypto" / "data" / f"{asset}_raw.csv"
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data not found: {raw_data_path}\n"
            "Run lstm_gold_crypto/src/train.py to download and save raw data."
        )

    print(f"  Loading raw data: {raw_data_path.name}")
    raw_df = pd.read_csv(raw_data_path, index_col=0, parse_dates=True)

    # Compute features
    df_feat = add_features(raw_df)
    df_feat = df_feat.dropna()

    print(f"  Data rows after feature engineering: {len(df_feat)}")

    # --- Scale features ---
    X_scaled = feature_scaler.transform(df_feat[LSTM_FEATURES]).astype(np.float32)

    # --- Create sequences ---
    # We use the LAST portion of data for SHAP (test-like period)
    n_seqs_needed = SHAP_LSTM_BACKGROUND_SEQUENCES + SHAP_LSTM_EXPLAIN_SEQUENCES

    if len(X_scaled) < window + n_seqs_needed:
        n_seqs_needed = max(20, len(X_scaled) - window - 10)
        bg = max(5, n_seqs_needed // 3)
        ex = n_seqs_needed - bg
        print(f"  WARNING: Not enough data for full SHAP. Using bg={bg}, explain={ex}")
    else:
        bg = SHAP_LSTM_BACKGROUND_SEQUENCES
        ex = SHAP_LSTM_EXPLAIN_SEQUENCES

    # Build sequences: X[i] = X_scaled[i : i+window]
    sequences = np.array(
        [X_scaled[i: i + window] for i in range(len(X_scaled) - window)]
    )
    # Use the last n sequences
    sequences = sequences[-(bg + ex):]

    background_seqs = sequences[:bg]     # shape: (bg, window, n_features)
    explain_seqs = sequences[bg:]        # shape: (ex, window, n_features)

    print(f"  Background sequences: {background_seqs.shape}")
    print(f"  Explain sequences   : {explain_seqs.shape}")

    # --- Build model ---
    model_net = PriceVolatilityLSTM(
        input_size=len(LSTM_FEATURES),
        hidden1=96,
        hidden2=48,
        dropout=0.2,
    )
    model_net.load_state_dict(ckpt["model_state_dict"])
    model_net.eval()

    return_wrapper = _ReturnWrapper(model_net)
    vol_wrapper = _VolatilityWrapper(model_net)

    # Convert to tensors
    bg_tensor = torch.tensor(background_seqs)   # (bg, window, features)
    ex_tensor = torch.tensor(explain_seqs)       # (ex, window, features)

    results = {}
    SHAP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- SHAP for each head ---
    for head_name, wrapper in [("return", return_wrapper), ("volatility", vol_wrapper)]:

        print(f"\n  Computing SHAP for {head_name} head...")

        try:
            explainer = shap.GradientExplainer(wrapper, bg_tensor)
            shap_vals = explainer.shap_values(ex_tensor)

            # shap_vals may be a list or a single array
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

            shap_arr = np.array(shap_vals)
            print(f"  Raw SHAP shape: {shap_arr.shape}")

            # GradientExplainer with a single-output model returns shape
            # (n_samples, window, n_features) or (n_samples, window, n_features, 1).
            # Squeeze any trailing output dimension.
            if shap_arr.ndim == 4:
                shap_arr = shap_arr.squeeze(-1)   # (n_samples, window, n_features)

            # Aggregate over time dimension (axis=1) -> (n_samples, n_features)
            # Use mean absolute SHAP across the 60-day window
            shap_2d = np.mean(np.abs(shap_arr), axis=1)   # (n_samples, n_features)

            # For signed direction: use mean (not abs)
            shap_2d_signed = np.mean(shap_arr, axis=1)    # (n_samples, n_features)

            print(f"  Aggregated SHAP shape: {shap_2d.shape}")

            # Feature importance CSV
            fi_csv = SHAP_OUTPUT_DIR / f"{asset}_{head_name}_shap_importance.csv"
            mean_abs = np.mean(shap_2d, axis=0)       # (n_features,)
            mean_signed = np.mean(shap_2d_signed, axis=0)  # (n_features,)
            order = np.argsort(mean_abs)[::-1]        # integer array
            fi_df = pd.DataFrame({
                "feature": [LSTM_FEATURES[int(i)] for i in order],
                "mean_abs_shap": [float(mean_abs[int(i)]) for i in order],
                "mean_shap": [float(mean_signed[int(i)]) for i in order],
                "rank": range(1, len(LSTM_FEATURES) + 1),
            })
            fi_df.to_csv(fi_csv, index=False)
            print(f"  Saved: {fi_csv.name}")

            print(f"  Top 5 features for {head_name}:")
            for _, row in fi_df.head(5).iterrows():
                direction = "(up)" if row["mean_shap"] > 0 else "(down)"
                print(f"    {int(row['rank'])}. {row['feature']}: {row['mean_abs_shap']:.6f} {direction}")

            # Bar plot
            bar_path = SHAP_OUTPUT_DIR / f"{asset}_{head_name}_shap.png"
            _save_shap_bar_plot(
                shap_values=shap_2d_signed,
                feature_names=LSTM_FEATURES,
                title=(
                    f"{asset.capitalize()} LSTM — SHAP Feature Importance\n"
                    f"Target: {'Next-Day Return' if head_name == 'return' else '5-Day Future Volatility'}\n"
                    "Aggregated over 60-day window"
                ),
                output_path=bar_path,
                top_n=16,
            )
            print(f"  Saved bar plot: {bar_path.name}")

            results[head_name] = {
                "shap_values_raw": shap_arr,
                "shap_2d": shap_2d,
                "shap_2d_signed": shap_2d_signed,
                "feature_importance_csv": fi_csv,
                "bar_plot": bar_path,
            }

        except Exception as e:
            print(f"  ERROR during {head_name} SHAP: {e}")
            traceback.print_exc()
            print(f"  Skipping {head_name} SHAP for {asset}.")
            results[head_name] = {"error": str(e)}

    return {
        "asset": asset,
        "heads": results,
        "features": LSTM_FEATURES,
        "window": window,
    }


# ============================================================
# RUN ALL SHAP
# ============================================================

def run_all_shap() -> dict:
    """
    Run SHAP explanations for all models:
      1. Fraud Detection (Isolation Forest)
      2. Gold LSTM (return + volatility)
      3. Bitcoin LSTM (return + volatility)

    Returns dict with all results.
    """
    all_results = {}

    # 1. Fraud
    try:
        all_results["fraud"] = run_fraud_shap()
    except Exception as e:
        print(f"\nERROR in fraud SHAP: {e}")
        traceback.print_exc()
        all_results["fraud"] = {"error": str(e)}

    # 2. Gold LSTM
    try:
        all_results["gold"] = run_lstm_shap("gold")
    except Exception as e:
        print(f"\nERROR in gold LSTM SHAP: {e}")
        traceback.print_exc()
        all_results["gold"] = {"error": str(e)}

    # 3. Bitcoin LSTM
    try:
        all_results["bitcoin"] = run_lstm_shap("bitcoin")
    except Exception as e:
        print(f"\nERROR in bitcoin LSTM SHAP: {e}")
        traceback.print_exc()
        all_results["bitcoin"] = {"error": str(e)}

    print(f"\n{'='*60}")
    print("SHAP RESULTS SAVED TO:", SHAP_OUTPUT_DIR)
    print(f"{'='*60}")
    if SHAP_OUTPUT_DIR.exists():
        for f in sorted(SHAP_OUTPUT_DIR.iterdir()):
            print(f"  {f.name}")

    return all_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_all_shap()
