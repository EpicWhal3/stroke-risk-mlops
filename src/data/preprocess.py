import os
import json

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_config(path: str = "configs/data.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_preprocessor():
    num_features = [
        "age",
        "avg_glucose_level",
        "bmi",
        "hypertension",
        "heart_disease",
    ]

    cat_features = [
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
        ]
    )

    return preprocessor, num_features, cat_features


def prepare_data(df: pd.DataFrame):
    config = load_config()
    test_size = config["split"]["test_size"]
    random_state = config["split"]["random_state"]
    target_col = config["dataset"]["target"]

    required_columns = {
        "id",
        "age",
        "avg_glucose_level",
        "bmi",
        "gender",
        "ever_married",
        "work_type",
        "Residence_type",
        "smoking_status",
        "hypertension",
        "heart_disease",
        target_col,
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df.drop(columns=["id", target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, num_features, cat_features = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save("data/processed/X_train.npy", X_train_proc)
    np.save("data/processed/X_test.npy", X_test_proc)
    np.save("data/processed/y_train.npy", y_train.to_numpy())
    np.save("data/processed/y_test.npy", y_test.to_numpy())

    joblib.dump(preprocessor, "models/preprocessor.joblib")

    meta = {
        "num_features": num_features,
        "cat_features": cat_features,
        "train_shape": list(X_train_proc.shape),
        "test_shape": list(X_test_proc.shape),
    }

    with open("data/processed/dataset_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("Preprocessing complete.")
    print(f"Train shape: {X_train_proc.shape}")
    print(f"Test shape: {X_test_proc.shape}")


if __name__ == "__main__":
    config = load_config()
    raw_path = os.path.join(
        config["dataset"]["raw_dir"],
        config["dataset"]["raw_filename"],
    )

    df = pd.read_csv(raw_path)
    prepare_data(df)
