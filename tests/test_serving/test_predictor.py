from unittest.mock import MagicMock

from app.serving.predictor import Predictor
from data.prepare_dataset import FEATURE_NAMES


def test_predict_logs_and_returns_result():
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]

    registry = MagicMock()
    registry.get_model.return_value = mock_model
    registry.get_version.return_value = "3"

    router = MagicMock()
    router.choose_alias.return_value = "champion"

    predictor = Predictor(registry, router)
    features = {name: 0.5 for name in FEATURE_NAMES}

    mock_db = MagicMock()
    result = predictor.predict(features, mock_db)

    assert result["model_alias"] == "champion"
    assert result["model_version"] == "3"
    assert result["prediction"] == 1
    assert "request_id" in result

    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()


def test_probability_fallback_when_predict_proba_unavailable():
    mock_model = MagicMock(spec=["predict"])  # no predict_proba, no _model_impl
    mock_model.predict.return_value = [0]

    registry = MagicMock()
    registry.get_model.return_value = mock_model
    registry.get_version.return_value = "1"

    router = MagicMock()
    router.choose_alias.return_value = "champion"

    predictor = Predictor(registry, router)
    features = {name: 0.1 for name in FEATURE_NAMES}

    result = predictor.predict(features, MagicMock())
    assert result["probability"] == 0.0
