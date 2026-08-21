import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from preprocessing import load_and_preprocess_data




model = joblib.load(
    "models/final_isolation_forest_model.pkl"
)



df, feature_scaler = load_and_preprocess_data()

X = df.drop("Class", axis=1)
y = df["Class"]




predictions = model.predict(X)

fraud_predictions = (predictions == -1).astype(int)




cm = confusion_matrix(
    y,
    fraud_predictions
)




display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Fraud"]
)

display.plot()

plt.title("Raghavi - Fraud Detection Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "results/confusion_matrix.png",
    dpi=300
)

plt.show()

print("Confusion matrix saved successfully!")
print("File: results/confusion_matrix.png")