import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocessing import load_and_preprocess_data




model = joblib.load(
    "models/final_isolation_forest_model.pkl"
)




df, feature_scaler = load_and_preprocess_data()

X = df.drop("Class", axis=1)
y = df["Class"]




predictions = model.predict(X)



fraud_predictions = (predictions == -1).astype(int)




accuracy = accuracy_score(
    y,
    fraud_predictions
)

precision = precision_score(
    y,
    fraud_predictions,
    zero_division=0
)

recall = recall_score(
    y,
    fraud_predictions,
    zero_division=0
)

f1 = f1_score(
    y,
    fraud_predictions,
    zero_division=0
)




print("========================================")
print("RAGHAVI MODEL EVALUATION")
print("========================================")

print("\nAccuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))



cm = confusion_matrix(
    y,
    fraud_predictions
)

print("\nConfusion Matrix:")
print(cm)




print("\nClassification Report:")
print(
    classification_report(
        y,
        fraud_predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    )
)



evaluation = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        f1
    ]
})

evaluation.to_csv(
    "results/fraud_model_evaluation.csv",
    index=False
)

print("\nEvaluation results saved!")
print("File: results/fraud_model_evaluation.csv")