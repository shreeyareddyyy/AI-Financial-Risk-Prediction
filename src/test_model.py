import joblib

model_path = "models/final_isolation_forest_model.pkl"
scaler_path = "models/final_fraud_risk_scaler.pkl"

model = joblib.load(model_path)
risk_scaler = joblib.load(scaler_path)

print("Model loaded successfully!")
print("Risk scaler loaded successfully!")

print("\nModel parameters:")
print("n_estimators:", model.n_estimators)
print("contamination:", model.contamination)