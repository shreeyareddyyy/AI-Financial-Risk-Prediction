import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data():

    # Load original dataset
    df = pd.read_csv("data/creditcard.csv")

    # Remove duplicates
    df = df.drop_duplicates()

    # Create feature scaler
    feature_scaler = StandardScaler()

    # Scale Time and Amount
    df[["Time", "Amount"]] = feature_scaler.fit_transform(
        df[["Time", "Amount"]]
    )

    return df, feature_scaler