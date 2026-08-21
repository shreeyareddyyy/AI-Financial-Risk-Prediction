"""
week3_config.py
===============
Central configuration for Week 3: SHAP Explainability + Backtesting Engine v1.

All parameters are documented. Change values here rather than editing source code.
"""

import os
import pathlib

# ============================================================
# ROOT PATHS
# ============================================================

# Root of the repository (two levels up from this file: src/ -> project root)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# Root of the LSTM sub-module
LSTM_ROOT = REPO_ROOT / "lstm_gold_crypto"

# ============================================================
# MODEL PATHS
# ============================================================

FRAUD_MODEL_PATH = REPO_ROOT / "models" / "final_isolation_forest_model.pkl"
FRAUD_RISK_SCALER_PATH = REPO_ROOT / "models" / "final_fraud_risk_scaler.pkl"

GOLD_MODEL_PATH = LSTM_ROOT / "models" / "gold_lstm.pt"
BITCOIN_MODEL_PATH = LSTM_ROOT / "models" / "bitcoin_lstm.pt"

# ============================================================
# EXISTING PREDICTION CSVs (output of the LSTM training script)
# ============================================================

GOLD_PREDICTION_CSV = LSTM_ROOT / "results" / "gold_prediction_output.csv"
BITCOIN_PREDICTION_CSV = LSTM_ROOT / "results" / "bitcoin_prediction_output.csv"

# Existing fraud results (produced by the full pipeline)
FRAUD_RESULTS_CSV = REPO_ROOT / "results" / "final_fraud_detection_results.csv"

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

SHAP_OUTPUT_DIR = REPO_ROOT / "results" / "shap"
BACKTEST_OUTPUT_DIR = REPO_ROOT / "results" / "backtesting"

# ============================================================
# SHAP PARAMETERS
# ============================================================

# Number of background samples for SHAP (higher = more accurate, slower)
SHAP_FRAUD_BACKGROUND_SAMPLES = 200   # samples from fraud dataset for TreeExplainer background
SHAP_FRAUD_EXPLAIN_SAMPLES = 200      # samples to generate SHAP values for (for plots)

# For LSTM GradientExplainer:
SHAP_LSTM_BACKGROUND_SEQUENCES = 30   # number of 60-day sequences as background
SHAP_LSTM_EXPLAIN_SEQUENCES = 50      # sequences to explain

# Random seed for reproducibility
SHAP_RANDOM_SEED = 42

# ============================================================
# BACKTESTING PARAMETERS
# ============================================================

# Initial portfolio capital in USD
INITIAL_CAPITAL = 100_000.0

# Transaction cost as a fraction of trade value (round-trip not doubled; applied per trade)
# 0.001 = 0.1% per position change — conservative for liquid futures/crypto
TRANSACTION_COST = 0.001

# Risk-free rate for Sharpe Ratio calculation (annual)
# Using US 3-month T-bill approximate rate
RISK_FREE_RATE = 0.05

# Annualization factor
# Both Gold and Bitcoin are daily data (252 trading days/year for Gold futures,
# 365 for Bitcoin which trades 24/7; we use 252 for Gold and 365 for Bitcoin)
GOLD_ANNUALIZATION_FACTOR = 252
BITCOIN_ANNUALIZATION_FACTOR = 365

# Signal threshold: predicted_return > SIGNAL_THRESHOLD → LONG
#                  predicted_return < -SIGNAL_THRESHOLD → SHORT
#                  otherwise → FLAT
#
# Set to None to auto-compute from data (20th percentile of |predicted_return|)
# Set to a float to override (e.g., 0.001 = 0.1% predicted log-return)
SIGNAL_THRESHOLD = None  # auto-computed from data distribution

# Allow SHORT positions?
# True  = LONG / FLAT / SHORT strategy
# False = LONG / FLAT only (no shorting)
ALLOW_SHORT = True

# ============================================================
# POSITION SIZING
# ============================================================
# Version 1 uses simple binary position sizing:
#   LONG  = +1 (full allocation)
#   SHORT = -1 (full short)
#   FLAT  =  0 (cash)
# No fractional Kelly or volatility-scaling in V1.

# ============================================================
# PLOTTING STYLE
# ============================================================

PLOT_DPI = 150
PLOT_FIGSIZE_WIDE = (14, 6)
PLOT_FIGSIZE_SQUARE = (10, 8)
PLOT_STYLE = "seaborn-v0_8-darkgrid"
