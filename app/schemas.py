from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    feature_0: float
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float

    def to_dict(self) -> dict[str, float]:
        return self.model_dump()


class PredictResponse(BaseModel):
    request_id: str
    model_alias: str
    model_version: str
    prediction: int
    probability: float


class FeedbackRequest(BaseModel):
    request_id: str
    actual_label: int = Field(..., ge=0, le=1)


class HealthResponse(BaseModel):
    status: str
    mlflow_reachable: bool
    champion_version: str | None = None
    challenger_version: str | None = None


class DriftFeatureResult(BaseModel):
    psi: float
    status: str


class DriftResponse(BaseModel):
    status: str
    sample_size: int
    features: dict[str, DriftFeatureResult] = Field(default_factory=dict)
    max_psi: float | None = None


class PredictionSummaryResponse(BaseModel):
    window_minutes: int
    total_predictions: int
    by_model_alias: dict[str, int]
    average_probability: float | None
