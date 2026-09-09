from unittest.mock import MagicMock

from app.retraining.retrain_trigger import RetrainTrigger


def test_no_event_when_status_not_computed():
    trigger = RetrainTrigger(psi_threshold=0.2)
    mock_db = MagicMock()

    result = trigger.check_and_log(mock_db, {"status": "INSUFFICIENT_DATA"})

    assert result is None
    mock_db.add.assert_not_called()


def test_no_event_when_max_psi_below_threshold():
    trigger = RetrainTrigger(psi_threshold=0.2)
    mock_db = MagicMock()
    drift_result = {
        "status": "COMPUTED",
        "max_psi": 0.05,
        "features": {"feature_0": {"psi": 0.05, "status": "STABLE"}},
    }

    result = trigger.check_and_log(mock_db, drift_result)

    assert result is None
    mock_db.add.assert_not_called()


def test_event_logged_when_threshold_exceeded():
    trigger = RetrainTrigger(psi_threshold=0.2)
    mock_db = MagicMock()
    drift_result = {
        "status": "COMPUTED",
        "max_psi": 0.35,
        "features": {
            "feature_0": {"psi": 0.35, "status": "SIGNIFICANT_SHIFT"},
            "feature_1": {"psi": 0.05, "status": "STABLE"},
        },
    }

    trigger.check_and_log(mock_db, drift_result)

    mock_db.add.assert_called_once()
    logged_event = mock_db.add.call_args[0][0]
    assert logged_event.max_psi_score == 0.35
    assert "feature_0" in logged_event.trigger_reason
    assert "feature_1" not in logged_event.trigger_reason
    assert logged_event.status == "PENDING"
    mock_db.commit.assert_called_once()


def test_custom_threshold_respected():
    trigger = RetrainTrigger(psi_threshold=0.5)
    mock_db = MagicMock()
    drift_result = {
        "status": "COMPUTED",
        "max_psi": 0.3,
        "features": {"feature_0": {"psi": 0.3, "status": "MODERATE_SHIFT"}},
    }

    result = trigger.check_and_log(mock_db, drift_result)

    assert result is None
