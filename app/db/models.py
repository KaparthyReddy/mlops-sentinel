from sqlalchemy import Column, BigInteger, String, Integer, Float, DateTime, JSON
from sqlalchemy.sql import func

from app.db.database import Base


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(BigInteger, primary_key=True)
    request_id = Column(String(64), unique=True, nullable=False)
    model_alias = Column(String(20), nullable=False)
    model_version = Column(String(20), nullable=False)
    features = Column(JSON, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class FeedbackLabel(Base):
    __tablename__ = "feedback_labels"

    id = Column(BigInteger, primary_key=True)
    request_id = Column(String(64), nullable=False)
    actual_label = Column(Integer, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


class DriftScore(Base):
    __tablename__ = "drift_scores"

    id = Column(BigInteger, primary_key=True)
    feature_name = Column(String(100), nullable=False)
    psi_score = Column(Float, nullable=False)
    sample_size = Column(Integer, nullable=False)
    computed_at = Column(DateTime, server_default=func.now())


class RetrainingEvent(Base):
    __tablename__ = "retraining_events"

    id = Column(BigInteger, primary_key=True)
    trigger_reason = Column(String(200), nullable=False)
    max_psi_score = Column(Float, nullable=True)
    triggered_at = Column(DateTime, server_default=func.now())
    status = Column(String(20), default="PENDING")
