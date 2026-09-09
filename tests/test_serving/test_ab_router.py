from unittest.mock import MagicMock

from app.serving.ab_router import ABRouter


def test_routes_to_champion_when_no_challenger():
    registry = MagicMock()
    registry.has_challenger.return_value = False

    router = ABRouter(registry, challenger_traffic_percent=50)
    assert router.choose_alias("any-request-id") == "champion"


def test_same_request_id_always_routes_same_way():
    registry = MagicMock()
    registry.has_challenger.return_value = True

    router = ABRouter(registry, challenger_traffic_percent=50)
    first = router.choose_alias("consistent-id-123")
    second = router.choose_alias("consistent-id-123")

    assert first == second


def test_zero_percent_traffic_always_champion():
    registry = MagicMock()
    registry.has_challenger.return_value = True

    router = ABRouter(registry, challenger_traffic_percent=0)
    results = {router.choose_alias(f"id-{i}") for i in range(50)}

    assert results == {"champion"}


def test_hundred_percent_traffic_always_challenger():
    registry = MagicMock()
    registry.has_challenger.return_value = True

    router = ABRouter(registry, challenger_traffic_percent=100)
    results = {router.choose_alias(f"id-{i}") for i in range(50)}

    assert results == {"challenger"}
