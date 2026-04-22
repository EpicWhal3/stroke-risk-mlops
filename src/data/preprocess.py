import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


def build_preprocessor():
    num_features = ["age", "avg_glucose_level", "bmi"]
    cat_features = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), num_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                cat_features,
            ),
        ]
    )
    return preprocessor


def prepare_data(df: pd.DataFrame, test_size=0.2, seed=42):
    X = df.drop(columns=["id", "stroke"])
    y = df["stroke"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.joblib")

    return X_train_proc, X_test_proc, y_train.values, y_test.values


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    print("📥 Loading raw data...")
    df = pd.read_csv("data/raw/stroke_dataset.csv")

    X_train, X_test, y_train, y_test = prepare_data(df)

    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    print("Preprocessing complete. Files saved to data/processed/")
