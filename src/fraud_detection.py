import joblib
import pandas as pd
import numpy as np



MODEL_PATH = "models/final_isolation_forest_model.pkl"
RISK_SCALER_PATH = "models/final_fraud_risk_scaler.pkl"


model = joblib.load(MODEL_PATH)
risk_scaler = joblib.load(RISK_SCALER_PATH)




def predict_transaction(transaction):

    
    transaction = pd.DataFrame(transaction)

   
    prediction = model.predict(transaction)

    
    fraud_prediction = np.where(
        prediction == -1,
        1,
        0
    )

    anomaly_score = model.decision_function(transaction)

    
    risk_input = (-anomaly_score).reshape(-1, 1)

    fraud_risk_score = risk_scaler.transform(
        risk_input
    ).flatten()

    
    fraud_risk_score = np.clip(
        fraud_risk_score,
        0,
        100
    )

    risk_levels = []

    for score in fraud_risk_score:

        if score < 30:
            risk = "Low"

        elif score < 70:
            risk = "Medium"

        else:
            risk = "High"

        risk_levels.append(risk)

    
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