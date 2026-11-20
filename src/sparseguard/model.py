"""Gradient-boosted hourly sepsis classifier (CPU only)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .config import SEED


def make_model(seed: int = SEED) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=400,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=25,
        random_state=seed,
    )


def fit(X: np.ndarray, y: np.ndarray, seed: int = SEED) -> HistGradientBoostingClassifier:
    return make_model(seed).fit(X, y)


def predict(model, X: np.ndarray, batch: int = 250_000) -> np.ndarray:
    return np.concatenate([model.predict_proba(X[i : i + batch])[:, 1] for i in range(0, len(X), batch)]).astype(
        np.float32
    )
