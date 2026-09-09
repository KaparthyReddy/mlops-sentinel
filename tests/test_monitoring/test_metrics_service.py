from unittest.mock import MagicMock

from app.monitoring.metrics_service import MetricsService


def test_prediction_summary_empty_window():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = []

    service = MetricsService(drift_detector=MagicMock())
    result = service.get_prediction_summary(mock_db, window_minutes=60)

    assert result["total_predictions"] == 0
    assert result["by_model_alias"] == {}
    assert result["average_probability"] is None


def test_prediction_summary_aggregates_by_alias():
    mock_row_a = MagicMock(model_alias="champion", probability=0.2)
    mock_row_b = MagicMock(model_alias="champion", probability=0.4)
    mock_row_c = MagicMock(model_alias="challenger", probability=0.9)

    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [mock_row_a, mock_row_b, mock_row_c]

    service = MetricsService(drift_detector=MagicMock())
    result = service.get_prediction_summary(mock_db, window_minutes=60)

    assert result["total_predictions"] == 3
    assert result["by_model_alias"] == {"champion": 2, "challenger": 1}
    assert abs(result["average_probability"] - 0.5) < 0.001


def test_compute_drift_insufficient_data():
    mock_db = MagicMock()
    mock_db.query.return_value.filter.return_value.all.return_value = [MagicMock()] * 5

    service = MetricsService(drift_detector=MagicMock())
    result = service.compute_and_store_drift(mock_db, min_samples=30)

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["sample_size"] == 5
