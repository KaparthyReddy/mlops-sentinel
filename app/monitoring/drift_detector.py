"""
Computes PSI (Population Stability Index) between the training-time
feature baseline and a window of recent production predictions. PSI is
the standard metric for this in industry - interpretation bands are
well established:

    PSI < 0.1  -> no significant shift
    0.1 - 0.2  -> moderate shift, worth watching
    PSI > 0.2  -> significant shift, model likely degrading

Rather than only reporting a raw score, this class also classifies each
score against those bands, since a raw float means little without that
context - especially in a monitoring endpoint someone might check without
having memorized PSI theory.
"""

import numpy as np
import pandas as pd

from data.prepare_dataset import FEATURE_NAMES
from training.feature_baseline import load_baseline

EPSILON = 1e-6  # avoid division by / log of zero in PSI formula


class DriftDetector:

    def __init__(self, baseline: dict | None = None):
        self.baseline = baseline or load_baseline()

    def compute_psi_for_feature(self, feature_name: str, production_values: np.ndarray) -> float:
        baseline_info = self.baseline[feature_name]
        bin_edges = np.array(baseline_info["bin_edges"])
        baseline_proportions = np.array(baseline_info["proportions"])

        prod_counts, _ = np.histogram(production_values, bins=bin_edges)
        prod_proportions = prod_counts / max(prod_counts.sum(), 1)

        # Avoid zero-division/log(0) for empty bins
        baseline_safe = np.where(baseline_proportions == 0, EPSILON, baseline_proportions)
        prod_safe = np.where(prod_proportions == 0, EPSILON, prod_proportions)

        psi = np.sum((prod_safe - baseline_safe) * np.log(prod_safe / baseline_safe))
        return float(psi)

    def compute_all_features(self, production_df: pd.DataFrame) -> dict[str, float]:
        return {
            feature: self.compute_psi_for_feature(feature, production_df[feature].values)
            for feature in FEATURE_NAMES
            if feature in production_df.columns
        }

    @staticmethod
    def classify_psi(psi: float) -> str:
        if psi < 0.1:
            return "STABLE"
        elif psi < 0.2:
            return "MODERATE_SHIFT"
        else:
            return "SIGNIFICANT_SHIFT"
