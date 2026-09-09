from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import FeedbackLabel, PredictionLog
from app.schemas import (
    PredictRequest, PredictResponse, FeedbackRequest,
    HealthResponse, DriftResponse, PredictionSummaryResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    registry = request.app.state.registry
    try:
        champion_version = registry.get_version("champion")
        mlflow_reachable = True
    except Exception:
        champion_version = None
        mlflow_reachable = False

    challenger_version = None
    if registry.has_challenger():
        try:
            challenger_version = registry.get_version("challenger")
        except Exception:
            pass

    return HealthResponse(
        status="UP" if mlflow_reachable else "DEGRADED",
        mlflow_reachable=mlflow_reachable,
        champion_version=champion_version,
        challenger_version=challenger_version,
    )


@router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, body: PredictRequest, db: Session = Depends(get_db)):
    predictor = request.app.state.predictor
    result = predictor.predict(body.to_dict(), db)
    return PredictResponse(**result)


@router.post("/feedback")
async def feedback(body: FeedbackRequest, db: Session = Depends(get_db)):
    log_entry = db.query(PredictionLog).filter(PredictionLog.request_id == body.request_id).first()
    if not log_entry:
        raise HTTPException(status_code=404, detail=f"No prediction found for request_id {body.request_id}")

    label = FeedbackLabel(request_id=body.request_id, actual_label=body.actual_label)
    db.add(label)
    db.commit()
    return {"status": "recorded"}


@router.get("/monitoring/summary", response_model=PredictionSummaryResponse)
async def prediction_summary(request: Request, window_minutes: int = 60, db: Session = Depends(get_db)):
    metrics_service = request.app.state.metrics_service
    result = metrics_service.get_prediction_summary(db, window_minutes=window_minutes)
    return PredictionSummaryResponse(**result)


@router.get("/monitoring/drift", response_model=DriftResponse)
async def drift(request: Request, window_minutes: int = 60, db: Session = Depends(get_db)):
    metrics_service = request.app.state.metrics_service
    result = metrics_service.compute_and_store_drift(db, window_minutes=window_minutes)

    if result["status"] == "INSUFFICIENT_DATA":
        return DriftResponse(status=result["status"], sample_size=result["sample_size"], features={}, max_psi=None)

    retrain_trigger = request.app.state.retrain_trigger
    retrain_trigger.check_and_log(db, result)

    return DriftResponse(**result)


@router.post("/registry/promote-challenger")
async def promote_challenger(request: Request):
    registry = request.app.state.registry
    if not registry.has_challenger():
        raise HTTPException(status_code=400, detail="No challenger model is currently registered")

    registry.promote_challenger_to_champion()
    return {"status": "promoted", "new_champion_version": registry.get_version("champion")}
