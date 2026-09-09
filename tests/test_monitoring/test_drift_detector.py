import numpy as np
import pytest

from app.monitoring.drift_detector import DriftDetector


@pytest.fixture
def baseline():
    # Simple baseline: feature_0 uniformly distributed 0-10 across 5 bins
    return {
        "feature_0": {
            "bin_edges": [0, 2, 4, 6, 8, 10],
            "proportions": [0.2, 0.2, 0.2, 0.2, 0.2],
        }
    }


def test_identical_distribution_gives_near_zero_psi(baseline):
    detector = DriftDetector(baseline=baseline)
    # Same shape as baseline: roughly even spread across bins
    production_values = np.array([1, 3, 5, 7, 9] * 20)

    psi = detector.compute_psi_for_feature("feature_0", production_values)
    assert psi < 0.05


def test_shifted_distribution_gives_high_psi(baseline):
    detector = DriftDetector(baseline=baseline)
    # All values crammed into the top bin - a strong distribution shift
    production_values = np.array([9.5] * 100)

    psi = detector.compute_psi_for_feature("feature_0", production_values)
    assert psi > 0.2


def test_classify_psi_bands():
    assert DriftDetector.classify_psi(0.05) == "STABLE"
    assert DriftDetector.classify_psi(0.15) == "MODERATE_SHIFT"
    assert DriftDetector.classify_psi(0.35) == "SIGNIFICANT_SHIFT"


def test_compute_all_features_skips_missing_columns(baseline):
    import pandas as pd
    detector = DriftDetector(baseline=baseline)
    df = pd.DataFrame({"feature_0": [1, 2, 3, 4, 5], "unrelated_column": [1, 2, 3, 4, 5]})

    result = detector.compute_all_features(df)
    assert list(result.keys()) == ["feature_0"]
