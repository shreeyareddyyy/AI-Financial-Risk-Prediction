import joblib
import pandas as pd
from preprocessing import load_and_preprocess_data



model = joblib.load(
    "models/final_isolation_forest_model.pkl"
)

risk_scaler = joblib.load(
    "models/final_fraud_risk_scaler.pkl"
)



df, feature_scaler = load_and_preprocess_data()

X = df.drop("Class", axis=1)
y = df["Class"]



predictions = model.predict(X)

anomaly_scores = model.decision_function(X)



fraud_predictions = (predictions == -1).astype(int)




risk_input = (-anomaly_scores).reshape(-1, 1)

fraud_risk_scores = risk_scaler.transform(
    risk_input
).flatten()

fraud_risk_scores = fraud_risk_scores.clip(0, 100)



risk_levels = []

for score in fraud_risk_scores:

    if score < 30:
        risk_levels.append("Low")

    elif score < 70:
        risk_levels.append("Medium")

    else:
        risk_levels.append("High")




results = pd.DataFrame({
    "Actual_Class": y.values,
    "Fraud_Prediction": fraud_predictions,
    "Anomaly_Score": anomaly_scores,
    "Fraud_Risk_Score": fraud_risk_scores,
    "Fraud_Risk_Level": risk_levels
})



print("========================================")
print("RAGHAVI FRAUD DETECTION RESULTS")
print("========================================")

print("\nTotal transactions:")
print(len(results))

print("\nFraud predictions:")
print(results["Fraud_Prediction"].value_counts())

print("\nRisk level distribution:")
print(results["Fraud_Risk_Level"].value_counts())

print("\nFirst 10 results:")
print(results.head(10).to_string(index=False))




results.to_csv(
    "results/vs_code_fraud_detection_results.csv",
    index=False
)

print("\nResults saved successfully!")
print("File: results/vs_code_fraud_detection_results.csv")