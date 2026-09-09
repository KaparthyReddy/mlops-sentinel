"""
Decides whether a retraining event should be triggered, based on the most
recent drift computation. This class deliberately only *decides and
logs* the trigger — it does not itself run training/train.py. Actually
kicking off retraining is a separate, more consequential action (costs
compute, produces a new model version) that in a real system would go
through a review/approval step or at minimum a distinct, explicitly
invoked job - conflating "detected a problem" with "took automated
action" is how you get an incident, not a feature.

A background scheduler (wired in main.py) calls check_and_log() on an
interval. Retraining events are recorded as PENDING; a human (or a
separate, explicitly-run job) decides whether and when to actually act
on them.
"""

from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import RetrainingEvent


class RetrainTrigger:

    def __init__(self, psi_threshold: float | None = None):
        self.psi_threshold = psi_threshold if psi_threshold is not None else settings.drift_psi_threshold

    def check_and_log(self, db: Session, drift_result: dict) -> RetrainingEvent | None:
        if drift_result.get("status") != "COMPUTED":
            return None

        max_psi = drift_result.get("max_psi", 0.0)
        if max_psi < self.psi_threshold:
            return None

        breached_features = [
            name for name, info in drift_result.get("features", {}).items()
            if info["psi"] >= self.psi_threshold
        ]

        event = RetrainingEvent(
            trigger_reason=f"PSI threshold ({self.psi_threshold}) exceeded for: {', '.join(breached_features)}",
            max_psi_score=max_psi,
            status="PENDING",
        )
        db.add(event)
        db.commit()
        db.refresh(event)
        return event
