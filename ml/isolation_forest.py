"""
ml/isolation_forest.py

Isolation Forest for point anomaly detection.
Detects when a single snapshot's feature vector is unusual
compared to the healthy baseline it was trained on.

Why Isolation Forest: it's unsupervised (no labels needed),
fast at inference, and naturally handles high-dimensional
feature spaces. It's the first line of defence.
"""

import os
import pickle
import logging
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "isolation_forest.pkl")


class IsolationForestDetector:
    """
    Trains on healthy baseline feature vectors.
    Scores new vectors — higher score = more anomalous.
    Score is normalised to [0, 1] for easy combination with LSTM score.
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self._model: IsolationForest = None
        self._scaler: StandardScaler = StandardScaler()
        self._trained = False
        self._score_min = -1.0
        self._score_max =  0.0

    # ── training ───────────────────────────────────────────────────────────

    def fit(self, feature_vectors: list) -> "IsolationForestDetector":
        """
        Train on a list of FeatureVector objects collected during
        healthy baseline operation.
        """
        if len(feature_vectors) < 20:
            raise ValueError(f"Need at least 20 samples to train, got {len(feature_vectors)}")

        X = np.array([fv.to_list() for fv in feature_vectors])
        X = self._scaler.fit_transform(X)

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self._model.fit(X)

        # Calibrate score range on training data for normalisation
        raw_scores = self._model.score_samples(X)
        self._score_min = float(raw_scores.min())
        self._score_max = float(raw_scores.max())
        self._trained = True

        log.info(
            f"Isolation Forest trained on {len(feature_vectors)} samples "
            f"(score range [{self._score_min:.3f}, {self._score_max:.3f}])"
        )
        return self

    # ── inference ──────────────────────────────────────────────────────────

    def score(self, feature_vector) -> float:
        """
        Returns anomaly score in [0, 1].
        0.0 = perfectly normal, 1.0 = maximally anomalous.
        """
        if not self._trained:
            return 0.0

        X = np.array([feature_vector.to_list()])
        X = self._scaler.transform(X)

        raw = float(self._model.score_samples(X)[0])

        # Normalise: lower raw score = more anomalous → invert to [0,1]
        span = self._score_max - self._score_min
        if span == 0:
            return 0.0
        normalised = (raw - self._score_min) / span
        return float(np.clip(1.0 - normalised, 0.0, 1.0))

    def predict(self, feature_vector) -> bool:
        """Returns True if the model considers this point anomalous."""
        if not self._trained:
            return False
        X = np.array([feature_vector.to_list()])
        X = self._scaler.transform(X)
        return int(self._model.predict(X)[0]) == -1

    # ── persistence ────────────────────────────────────────────────────────

    def save(self, path: str = MODEL_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Isolation Forest saved to {path}")

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "IsolationForestDetector":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        log.info(f"Isolation Forest loaded from {path}")
        return obj

    @property
    def is_trained(self) -> bool:
        return self._trained