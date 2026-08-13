import joblib
import pandas as pd

from preprocessing import load_and_preprocess_data




print("========================================")
print("RAGHAVI MODULE INTEGRATION TEST")
print("========================================")




model = joblib.load(
    "models/final_isolation_forest_model.pkl"
)

risk_scaler = joblib.load(
    "models/final_fraud_risk_scaler.pkl"
)

print("\n[1] Model loading: PASS")




df, feature_scaler = load_and_preprocess_data()

X = df.drop("Class", axis=1)


X_test = X.iloc[:100]

print("[2] Preprocessing: PASS")




predictions = model.predict(X_test)

print("[3] Fraud prediction: PASS")




anomaly_scores = model.decision_function(X_test)

print("[4] Anomaly scoring: PASS")





risk_input = (-anomaly_scores).reshape(-1, 1)

fraud_risk_scores = risk_scaler.transform(
    risk_input
).flatten()

fraud_risk_scores = fraud_risk_scores.clip(0, 100)

print("[5] Fraud risk scoring: PASS")



risk_levels = []

for score in fraud_risk_scores:

    if score < 30:
        risk_levels.append("Low")

    elif score < 70:
        risk_levels.append("Medium")

    else:
        risk_levels.append("High")

print("[6] Risk-level generation: PASS")




integration_results = pd.DataFrame({
    "Fraud_Prediction": (predictions == -1).astype(int),
    "Anomaly_Score": anomaly_scores,
    "Fraud_Risk_Score": fraud_risk_scores,
    "Fraud_Risk_Level": risk_levels
})



required_columns = [
    "Fraud_Prediction",
    "Anomaly_Score",
    "Fraud_Risk_Score",
    "Fraud_Risk_Level"
]

columns_ok = all(
    column in integration_results.columns
    for column in required_columns
)

scores_ok = (
    integration_results["Fraud_Risk_Score"].between(0, 100).all()
)

risk_levels_ok = integration_results[
    "Fraud_Risk_Level"
].isin(
    ["Low", "Medium", "High"]
).all()


if columns_ok and scores_ok and risk_levels_ok:

    print("[7] Output validation: PASS")

else:

    print("[7] Output validation: FAIL")




print("\nSample integration output:")
print(
    integration_results.head(10).to_string(index=False)
)



if (
    columns_ok
    and scores_ok
    and risk_levels_ok
):

    print("\n========================================")
    print("INTEGRATION TEST PASSED")
    print("========================================")

else:

    print("\n========================================")
    print("INTEGRATION TEST FAILED")
    print("========================================")