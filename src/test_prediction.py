import pandas as pd

from preprocessing import load_and_preprocess_data
from fraud_detection import predict_transaction


# Load the original dataset
df = pd.read_csv("data/creditcard.csv")

# Take one transaction
transaction = df.drop("Class", axis=1).head(1)

# Predict the transaction
result = predict_transaction(transaction)

print("========================================")
print("RAGHAVI FRAUD DETECTION TEST")
print("========================================")

print("\nInput transaction:")
print(transaction.to_string(index=False))

print("\nPrediction result:")
print(result.to_string(index=False))