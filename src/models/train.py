import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, roc_auc_score


def train_and_log(X_train, X_test, Y_train, Y_test, params=None):
    if params is None:
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "class_weight": "balanced",
            "random_state": 42,
        }

    with mlflow.start_run(run_name="rf_stroke_v1"):
        model = RandomForestClassifier(**params)
        model.fit(X_train, Y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(Y_test, probs),
            "f1_score": f1_score(Y_test, preds),
            "precision": precision_score(Y_test, preds),
        }

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("model_type", "RandomForest")

        print(f"Metrics: {metrics}")
        return model
