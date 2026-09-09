"""
Consistent-hash-based A/B routing: the same request_id always routes to
the same model variant, which matters for reproducibility (re-running a
request for debugging shouldn't silently hit a different model) and for
clean analysis later (you can group prediction_logs by model_alias and
trust the split held steady, rather than routing being re-randomized
per-request).

Falls back to always routing to "champion" if no challenger is currently
registered — the A/B split only activates once training/train.py has
been run with --as-challenger at least once.
"""

import hashlib

from app.config import settings
from app.registry.model_registry import ModelRegistry


class ABRouter:

    def __init__(self, registry: ModelRegistry, challenger_traffic_percent: int | None = None):
        self.registry = registry
        self.challenger_traffic_percent = (
            challenger_traffic_percent
            if challenger_traffic_percent is not None
            else settings.ab_test_challenger_traffic_percent
        )

    def choose_alias(self, request_id: str) -> str:
        if not self.registry.has_challenger():
            return "champion"

        bucket = self._hash_to_bucket(request_id)
        return "challenger" if bucket < self.challenger_traffic_percent else "champion"

    @staticmethod
    def _hash_to_bucket(request_id: str) -> int:
        """Maps a request_id deterministically to a 0-99 bucket."""
        digest = hashlib.sha256(request_id.encode()).hexdigest()
        return int(digest, 16) % 100
