"""
Runs a prediction through the selected model variant and logs the result.
This is the piece the API route actually calls - it hides the details of
which model was used, feature ordering, and logging behind one method.
"""

import uuid

import pandas as pd
from sqlalchemy.orm import Session

from app.db.models import PredictionLog
from app.registry.model_registry import ModelRegistry
from app.serving.ab_router import ABRouter
from data.prepare_dataset import FEATURE_NAMES


class Predictor:

    def __init__(self, registry: ModelRegistry, router: ABRouter):
        self.registry = registry
        self.router = router

    def predict(self, features: dict[str, float], db: Session) -> dict:
        request_id = str(uuid.uuid4())
        alias = self.router.choose_alias(request_id)

        model = self.registry.get_model(alias)
        version = self.registry.get_version(alias)

        feature_row = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

        # mlflow.pyfunc models expose .predict(); for a sklearn Pipeline
        # ending in a classifier, this returns the class prediction. We
        # separately need predict_proba, which pyfunc doesn't expose
        # directly - so we unwrap to the underlying sklearn model for that.
        raw_prediction = model.predict(feature_row)
        prediction = int(raw_prediction[0])

        probability = self._get_fraud_probability(model, feature_row)

        log_entry = PredictionLog(
            request_id=request_id,
            model_alias=alias,
            model_version=version,
            features=features,
            prediction=prediction,
            probability=probability,
        )
        db.add(log_entry)
        db.commit()

        return {
            "request_id": request_id,
            "model_alias": alias,
            "model_version": version,
            "prediction": prediction,
            "probability": probability,
        }

    @staticmethod
    def _get_fraud_probability(model, feature_row: pd.DataFrame) -> float:
        underlying = getattr(model, "_model_impl", None)
        sklearn_model = getattr(underlying, "sklearn_model", None) if underlying else None

        if sklearn_model is not None and hasattr(sklearn_model, "predict_proba"):
            return float(sklearn_model.predict_proba(feature_row)[0][1])

        # Fallback: if probability isn't available for some model type,
        # treat the class prediction itself as a 0.0/1.0 probability
        # rather than failing the request.
        return float(model.predict(feature_row)[0])
