"""
Computes and saves baseline feature distributions from the training data,
used later by DriftDetector to calculate PSI (Population Stability Index)
against live production traffic. This is deliberately a plain JSON file
rather than an MLflow artifact — the drift detector needs to load it
quickly and often, and doesn't need MLflow's versioning for this.

Run this once after training the champion model:
    python -m training.feature_baseline
"""

import json
import os

import numpy as np
import pandas as pd

from data.prepare_dataset import FEATURE_NAMES

N_BINS = 10
BASELINE_PATH = "data/feature_baseline.json"


def compute_baseline(df: pd.DataFrame, n_bins: int = N_BINS) -> dict:
    """
    For each feature, computes bin edges and the proportion of training
    data falling into each bin. PSI later compares production data's
    bin proportions against these baseline proportions.
    """
    baseline = {}

    for feature in FEATURE_NAMES:
        values = df[feature].values
        bin_edges = np.histogram_bin_edges(values, bins=n_bins)
        counts, _ = np.histogram(values, bins=bin_edges)
        proportions = (counts / counts.sum()).tolist()

        baseline[feature] = {
            "bin_edges": bin_edges.tolist(),
            "proportions": proportions,
        }

    return baseline


def save_baseline(baseline: dict, path: str = BASELINE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(baseline, f, indent=2)
    print(f"Baseline saved to {path}")


def load_baseline(path: str = BASELINE_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    csv_path = "data/training_data.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found — run `python -m data.prepare_dataset` first, "
            "or run training/train.py which generates it if missing."
        )

    df = pd.read_csv(csv_path)
    baseline = compute_baseline(df)
    save_baseline(baseline)
