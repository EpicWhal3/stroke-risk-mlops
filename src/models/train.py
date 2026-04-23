import json
import os

import joblib
import mlflow
import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def load_config(path: str = "configs/model.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    experiment_name = config["experiment"]["name"]
    params = config["random_forest"]

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="random_forest_baseline"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

        report = classification_report(y_test, y_pred, output_dict=True)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

        os.makedirs("models", exist_ok=True)

        joblib.dump(model, "models/model.pkl")

        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        with open("reports/classification_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        mlflow.log_artifact("metrics.json")
        mlflow.log_artifact("models/model.pkl")
        mlflow.log_artifact("reports/classification_report.json")

        print("Training complete.")
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    os.makedirs("reports", exist_ok=True)
    main()
