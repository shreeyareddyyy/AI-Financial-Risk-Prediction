import joblib
import pandas as pd
import numpy as np

from preprocessing import load_and_preprocess_data


MODEL_PATH = "models/final_isolation_forest_model.pkl"
RISK_SCALER_PATH = "models/final_fraud_risk_scaler.pkl"


# Load trained model
model = joblib.load(MODEL_PATH)

# Load fraud risk scaler
risk_scaler = joblib.load(RISK_SCALER_PATH)

# Load feature scaler used for Time and Amount
_, feature_scaler = load_and_preprocess_data()


def predict_transaction(transaction):

    # Convert input into DataFrame
    transaction = pd.DataFrame(transaction)

    # Features expected by the trained model
    expected_features = [
        "Time",
        "V1", "V2", "V3", "V4", "V5", "V6", "V7",
        "V8", "V9", "V10", "V11", "V12", "V13",
        "V14", "V15", "V16", "V17", "V18", "V19",
        "V20", "V21", "V22", "V23", "V24", "V25",
        "V26", "V27", "V28",
        "Amount"
    ]

    # Check whether the transaction has the correct features
    if list(transaction.columns) != expected_features:
        raise ValueError(
            "Transaction must contain exactly these features: "
            "Time, V1-V28, and Amount"
        )

    # Apply the same preprocessing to Time and Amount
    transaction[["Time", "Amount"]] = feature_scaler.transform(
        transaction[["Time", "Amount"]]
    )

    # Predict using Isolation Forest
    prediction = model.predict(transaction)

    # Convert Isolation Forest output:
    # -1 = anomaly/fraud
    #  1 = normal
    fraud_prediction = np.where(
        prediction == -1,
        1,
        0
    )

    # Calculate anomaly score
    anomaly_score = model.decision_function(transaction)

    # Convert anomaly score into fraud risk score
    risk_input = (-anomaly_score).reshape(-1, 1)

    fraud_risk_score = risk_scaler.transform(
        risk_input
    ).flatten()

    # Keep risk score between 0 and 100
    fraud_risk_score = np.clip(
        fraud_risk_score,
        0,
        100
    )

    # Assign risk levels
    risk_levels = []

    for score in fraud_risk_score:

        if score < 30:
            risk = "Low"

        elif score < 70:
            risk = "Medium"

        else:
            risk = "High"

        risk_levels.append(risk)

    # Create final results
    results = pd.DataFrame({
        "Fraud_Prediction": fraud_prediction,
        "Anomaly_Score": anomaly_score,
        "Fraud_Risk_Score": fraud_risk_score,
        "Fraud_Risk_Level": risk_levels
    })

    return results


if __name__ == "__main__":

    print("========================================")
    print("RAGHAVI FRAUD DETECTION MODULE")
    print("========================================")

    print("\nModel loaded successfully!")
    print("n_estimators:", model.n_estimators)
    print("contamination:", model.contamination)

    print("\nFraud detection module is ready!")