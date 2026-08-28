import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "creditcard.csv"
)


def load_and_preprocess_data():

    df = pd.read_csv(DATASET_PATH)

    df = df.drop_duplicates()

    feature_scaler = StandardScaler()

    df[["Time", "Amount"]] = feature_scaler.fit_transform(
        df[["Time", "Amount"]]
    )

    return df, feature_scaler