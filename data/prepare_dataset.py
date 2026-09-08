"""
Generates a synthetic, deliberately imbalanced binary classification
dataset shaped like real fraud data (rare positive class, overlapping
feature distributions) — no external download or API key required.

The point of this project is the MLOps infrastructure, not the dataset
or model sophistication, so a real fraud dataset (e.g. Kaggle's credit
card fraud set) would add setup friction without adding to the actual
learning goal here.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_NAMES = [f"feature_{i}" for i in range(10)]


def generate_dataset(n_samples: int = 20000, fraud_ratio: float = 0.03, random_state: int = 42) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=2,
        weights=[1 - fraud_ratio, fraud_ratio],
        flip_y=0.01,
        class_sep=0.8,
        random_state=random_state,
    )

    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["is_fraud"] = y
    return df


def generate_drifted_dataset(n_samples: int = 2000, fraud_ratio: float = 0.03,
                              drift_magnitude: float = 1.5, random_state: int = 99) -> pd.DataFrame:
    """
    Generates a dataset with a shifted feature distribution, simulating
    real-world drift (e.g. fraud patterns evolving over time). Used to
    exercise the drift-detection pipeline realistically.
    """
    df = generate_dataset(n_samples=n_samples, fraud_ratio=fraud_ratio, random_state=random_state)
    shift = np.random.RandomState(random_state).normal(loc=drift_magnitude, scale=0.3, size=len(FEATURE_NAMES))
    for i, col in enumerate(FEATURE_NAMES):
        df[col] = df[col] + shift[i]
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("data/training_data.csv", index=False)
    print(f"Generated {len(df)} rows, fraud rate: {df['is_fraud'].mean():.3%}")
