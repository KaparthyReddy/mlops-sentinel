"""
Wraps MLflow's model registry so the rest of the app never talks to
MLflow directly. Loads models by *alias* ("champion" / "challenger")
rather than by hardcoded version number — this is the piece that makes
promoting a challenger to champion (after a successful A/B test) a
one-line alias reassignment in MLflow, not a code deploy.

Models are cached in-process after first load, since re-fetching from
MLflow on every single prediction request would add unnecessary latency.
Call `refresh()` after a retraining event or alias change to bust the
cache.
"""

import mlflow.pyfunc
from mlflow.tracking import MlflowClient

from app.config import settings


class ModelRegistry:

    def __init__(self, model_name: str | None = None, tracking_uri: str | None = None):
        self.model_name = model_name or settings.model_name
        mlflow.set_tracking_uri(tracking_uri or settings.mlflow_tracking_uri)
        self.client = MlflowClient()
        self._cache: dict[str, mlflow.pyfunc.PyFuncModel] = {}
        self._version_cache: dict[str, str] = {}

    def get_model(self, alias: str) -> mlflow.pyfunc.PyFuncModel:
        if alias not in self._cache:
            model_uri = f"models:/{self.model_name}@{alias}"
            self._cache[alias] = mlflow.pyfunc.load_model(model_uri)
            version = self.client.get_model_version_by_alias(self.model_name, alias)
            self._version_cache[alias] = version.version
        return self._cache[alias]

    def get_version(self, alias: str) -> str:
        if alias not in self._version_cache:
            self.get_model(alias)  # populates version cache as a side effect
        return self._version_cache[alias]

    def has_challenger(self) -> bool:
        try:
            self.client.get_model_version_by_alias(self.model_name, "challenger")
            return True
        except Exception:
            return False

    def promote_challenger_to_champion(self) -> None:
        """Called after a successful A/B test — reassigns the 'champion'
        alias to whatever version currently holds 'challenger'."""
        challenger_version = self.client.get_model_version_by_alias(self.model_name, "challenger")
        self.client.set_registered_model_alias(self.model_name, "champion", challenger_version.version)
        self.refresh()

    def refresh(self) -> None:
        self._cache.clear()
        self._version_cache.clear()
