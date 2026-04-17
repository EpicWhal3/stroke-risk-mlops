import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib


def build_preprocessor():
    num_features = ["age", "avg_glucose", "bmi"]
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
    Y = df["stroke"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed, stratify=Y
    )

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump(preprocessor, "models/preprocessor.joblib")
    return X_train_proc, X_test_proc, Y_train, Y_test
