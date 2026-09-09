from unittest.mock import MagicMock, patch

from app.registry.model_registry import ModelRegistry


def _make_registry_with_mocked_client():
    with patch("app.registry.model_registry.mlflow.set_tracking_uri"), \
         patch("app.registry.model_registry.MlflowClient") as mock_client_cls:
        registry = ModelRegistry(model_name="test-model", tracking_uri="http://fake:5000")
        return registry, mock_client_cls.return_value


@patch("app.registry.model_registry.mlflow.pyfunc.load_model")
def test_get_model_caches_after_first_load(mock_load_model):
    registry, mock_client = _make_registry_with_mocked_client()
    mock_client.get_model_version_by_alias.return_value = MagicMock(version="5")
    mock_load_model.return_value = MagicMock()

    first = registry.get_model("champion")
    second = registry.get_model("champion")

    assert first is second
    mock_load_model.assert_called_once()


@patch("app.registry.model_registry.mlflow.pyfunc.load_model")
def test_get_version_returns_correct_version(mock_load_model):
    registry, mock_client = _make_registry_with_mocked_client()
    mock_client.get_model_version_by_alias.return_value = MagicMock(version="7")
    mock_load_model.return_value = MagicMock()

    version = registry.get_version("champion")
    assert version == "7"


def test_has_challenger_true_when_alias_exists():
    registry, mock_client = _make_registry_with_mocked_client()
    mock_client.get_model_version_by_alias.return_value = MagicMock(version="2")

    assert registry.has_challenger() is True


def test_has_challenger_false_when_alias_missing():
    registry, mock_client = _make_registry_with_mocked_client()
    mock_client.get_model_version_by_alias.side_effect = Exception("not found")

    assert registry.has_challenger() is False


@patch("app.registry.model_registry.mlflow.pyfunc.load_model")
def test_promote_challenger_reassigns_champion_alias(mock_load_model):
    registry, mock_client = _make_registry_with_mocked_client()
    mock_client.get_model_version_by_alias.return_value = MagicMock(version="9")
    mock_load_model.return_value = MagicMock()

    registry.promote_challenger_to_champion()

    mock_client.set_registered_model_alias.assert_called_once_with("test-model", "champion", "9")


@patch("app.registry.model_registry.mlflow.pyfunc.load_model")
def test_refresh_clears_cache(mock_load_model):
    registry, mock_client = _make_registry_with_mocked_client()
    mock_client.get_model_version_by_alias.return_value = MagicMock(version="1")
    mock_load_model.return_value = MagicMock()

    registry.get_model("champion")
    assert "champion" in registry._cache

    registry.refresh()
    assert registry._cache == {}
    assert registry._version_cache == {}
