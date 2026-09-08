"""
Trains a fraud-detection classifier on the synthetic dataset and
registers it in MLflow's model registry under the name given by
MODEL_NAME (default "fraud-detector").

Run without --as-challenger to register the first ("champion") model.
Run again later with --as-challenger to register a second version and
mark it as the "challenger" — this is what lets the A/B routing and
retraining-trigger pieces have two real, different models to compare.

Usage:
    python -m training.train
    python -m training.train --as-challenger
"""

import argparse
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data.prepare_dataset import FEATURE_NAMES, generate_dataset

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-detector")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")


def load_or_generate_data() -> pd.DataFrame:
    csv_path = "data/training_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return generate_dataset()


def train_model(as_challenger: bool) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud-detection")

    df = load_or_generate_data()
    X = df[FEATURE_NAMES]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Champion is deliberately simpler (LogisticRegression); challenger uses
    # a RandomForest — different enough that A/B comparison is meaningful,
    # not just the same model retrained on the same data.
    if as_challenger:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
        ])
        model_type = "random_forest_challenger"
    else:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        model_type = "logistic_regression_champion"

    with mlflow.start_run(run_name=model_type):
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        mlflow.log_param("model_type", model_type)
        mlflow.log_param("training_rows", len(X_train))
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=MODEL_NAME
        )

        print(f"Trained {model_type}")
        print(f"Metrics: {metrics}")
        print(f"Registered as {MODEL_NAME}, version info: {model_info.model_uri}")

        _assign_alias(model_info, as_challenger)


def _assign_alias(model_info, as_challenger: bool) -> None:
    client = MlflowClient()
    # model_info.model_uri looks like "models:/fraud-detector/<version>"
    version = model_info.model_uri.split("/")[-1]
    alias = "challenger" if as_challenger else "champion"
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    print(f"Set alias '{alias}' -> version {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--as-challenger", action="store_true",
                         help="Register this run as the challenger model instead of champion")
    args = parser.parse_args()

    train_model(as_challenger=args.as_challenger)
