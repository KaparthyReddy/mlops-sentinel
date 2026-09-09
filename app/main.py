"""
FastAPI entrypoint. Wires dependencies once at startup and stores them on
app.state - same pattern as TrialMatch Agents, kept consistent across
projects deliberately.

A background APScheduler job periodically runs drift detection against
recent traffic, so drift/retraining checks happen automatically rather
than only when someone happens to call /monitoring/drift manually.
"""

from contextlib import asynccontextmanager

from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI

from app.config import settings
from app.db.database import SessionLocal
from app.registry.model_registry import ModelRegistry
from app.serving.ab_router import ABRouter
from app.serving.predictor import Predictor
from app.monitoring.drift_detector import DriftDetector
from app.monitoring.metrics_service import MetricsService
from app.retraining.retrain_trigger import RetrainTrigger
from app.api.routes import router

scheduler = BackgroundScheduler()


def _scheduled_drift_check(metrics_service: MetricsService, retrain_trigger: RetrainTrigger):
    db = SessionLocal()
    try:
        result = metrics_service.compute_and_store_drift(db)
        if result.get("status") == "COMPUTED":
            retrain_trigger.check_and_log(db, result)
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    registry = ModelRegistry()
    router_component = ABRouter(registry)
    predictor = Predictor(registry, router_component)
    drift_detector = DriftDetector()
    metrics_service = MetricsService(drift_detector)
    retrain_trigger = RetrainTrigger()

    app.state.registry = registry
    app.state.predictor = predictor
    app.state.metrics_service = metrics_service
    app.state.retrain_trigger = retrain_trigger

    scheduler.add_job(
        _scheduled_drift_check,
        "interval",
        minutes=settings.retraining_check_interval_minutes,
        args=[metrics_service, retrain_trigger],
        id="drift_check",
    )
    scheduler.start()

    yield

    scheduler.shutdown(wait=False)


app = FastAPI(
    title="MLOps Sentinel",
    description="Production MLOps platform: model registry, A/B serving, drift detection, retraining triggers",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
