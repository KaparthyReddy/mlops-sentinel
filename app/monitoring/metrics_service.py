"""
Reads recent prediction logs and computes the aggregate stats a
monitoring dashboard/endpoint would show: prediction volume, average
confidence, per-model-alias breakdown, and current drift scores per
feature. Separated from DriftDetector itself since this is about
reporting/aggregation, not the PSI math.
"""

from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.db.models import PredictionLog, DriftScore
from app.monitoring.drift_detector import DriftDetector
from data.prepare_dataset import FEATURE_NAMES


class MetricsService:

    def __init__(self, drift_detector: DriftDetector):
        self.drift_detector = drift_detector

    def get_prediction_summary(self, db: Session, window_minutes: int = 60) -> dict:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

        rows = db.query(PredictionLog).filter(PredictionLog.created_at >= cutoff).all()

        if not rows:
            return {
                "window_minutes": window_minutes,
                "total_predictions": 0,
                "by_model_alias": {},
                "average_probability": None,
            }

        by_alias: dict[str, int] = {}
        for row in rows:
            by_alias[row.model_alias] = by_alias.get(row.model_alias, 0) + 1

        avg_probability = sum(r.probability for r in rows) / len(rows)

        return {
            "window_minutes": window_minutes,
            "total_predictions": len(rows),
            "by_model_alias": by_alias,
            "average_probability": round(avg_probability, 4),
        }

    def compute_and_store_drift(self, db: Session, window_minutes: int = 60, min_samples: int = 30) -> dict:
        cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
        rows = db.query(PredictionLog).filter(PredictionLog.created_at >= cutoff).all()

        if len(rows) < min_samples:
            return {
                "status": "INSUFFICIENT_DATA",
                "sample_size": len(rows),
                "min_required": min_samples,
            }

        production_df = pd.DataFrame([r.features for r in rows])
        psi_scores = self.drift_detector.compute_all_features(production_df)

        for feature_name, psi in psi_scores.items():
            db.add(DriftScore(feature_name=feature_name, psi_score=psi, sample_size=len(rows)))
        db.commit()

        classified = {
            feature: {"psi": round(psi, 4), "status": self.drift_detector.classify_psi(psi)}
            for feature, psi in psi_scores.items()
        }

        return {
            "status": "COMPUTED",
            "sample_size": len(rows),
            "features": classified,
            "max_psi": round(max(psi_scores.values()), 4) if psi_scores else 0.0,
        }
