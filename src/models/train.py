import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import json
import os


def main():
    print("Loading processed data...")
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    mlflow.set_experiment("stroke-risk-mlops")

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "class_weight": "balanced",
        "random_state": 42,
        "min_samples_split": 5,
    }

    print("Training Random Forest...")
    with mlflow.start_run(run_name="rf_baseline"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": float((y_pred == y_test).mean()),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print("Training complete!")
        print(f"Metrics: {metrics}")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
