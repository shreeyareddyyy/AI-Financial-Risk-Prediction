import os
import joblib
import pandas as pd
import numpy as np

from preprocessing import load_and_preprocess_data


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_isolation_forest_model.pkl"
)

RISK_SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_fraud_risk_scaler.pkl"
)


# =========================================================
# LOAD TRAINED MODEL AND SCALERS
# =========================================================

model = joblib.load(MODEL_PATH)

risk_scaler = joblib.load(
    RISK_SCALER_PATH
)

_, feature_scaler = load_and_preprocess_data()


# =========================================================
# FEATURES USED BY THE MODEL
# =========================================================

MODEL_FEATURES = [
    "Time",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13",
    "V14", "V15", "V16", "V17", "V18", "V19",
    "V20", "V21", "V22", "V23", "V24", "V25",
    "V26", "V27", "V28",
    "Amount"
]


# =========================================================
# PREDICT TRANSACTION
# =========================================================

def predict_transaction(transaction):

    # Accept a dictionary or DataFrame
    if isinstance(transaction, dict):

        transaction = pd.DataFrame(
            [transaction]
        )

    elif isinstance(transaction, pd.DataFrame):

        transaction = transaction.copy()

    else:

        raise ValueError(
            "Transaction must be a dictionary or DataFrame."
        )


    # =====================================================
    # CHECK REQUIRED FEATURES
    # =====================================================

    missing_features = [
        feature
        for feature in MODEL_FEATURES
        if feature not in transaction.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            + ", ".join(missing_features)
        )


    # Keep only model features
    transaction = transaction[
        MODEL_FEATURES
    ].copy()


    # =====================================================
    # SCALE FEATURES
    # =====================================================

    transaction[
        ["Time", "Amount"]
    ] = feature_scaler.transform(
        transaction[
            ["Time", "Amount"]
        ]
    )


    # =====================================================
    # ISOLATION FOREST PREDICTION
    # =====================================================

    prediction = model.predict(
        transaction
    )


    # Isolation Forest:
    # -1 = anomaly
    #  1 = normal

    fraud_prediction = np.where(
        prediction == -1,
        1,
        0
    )


    # =====================================================
    # ANOMALY SCORE
    # =====================================================

    anomaly_score = model.decision_function(
        transaction
    )


    # =====================================================
    # FRAUD RISK SCORE
    # =====================================================

    risk_input = (
        -anomaly_score
    ).reshape(-1, 1)

    fraud_risk_score = risk_scaler.transform(
        risk_input
    ).flatten()

    fraud_risk_score = np.clip(
        fraud_risk_score,
        0,
        100
    )


    # =====================================================
    # FRAUD RISK LEVEL
    # =====================================================

    risk_levels = []

    for score in fraud_risk_score:

        if score < 30:

            risk = "Low"

        elif score < 70:

            risk = "Medium"

        else:

            risk = "High"

        risk_levels.append(risk)


    # =====================================================
    # FINAL RESULT
    # =====================================================

    results = pd.DataFrame({

        "Fraud_Prediction":
            fraud_prediction,

        "Anomaly_Score":
            anomaly_score,

        "Fraud_Risk_Score":
            fraud_risk_score,

        "Fraud_Risk_Level":
            risk_levels
    })


    return results