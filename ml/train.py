"""
ml/train.py

Training script for both models.
Run this once to collect a healthy baseline and train IF + LSTM.
Saves models to ml/models/ for the scorer to load.

Usage:
    uv run python3 -m ml.train
"""

import time
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

from dotenv import load_dotenv
load_dotenv()

from nodes import Fleet
from telemetry.pipeline import TelemetryPipeline
from telemetry.features import FeatureStore
from ml.isolation_forest import IsolationForestDetector
from ml.lstm_autoencoder import LSTMDetector


COLLECT_SECONDS = 180   # 3 minutes of healthy baseline
NODE_COUNT      = 6


def collect_baseline(seconds: int = COLLECT_SECONDS) -> list:
    """
    Run the fleet with NO chaos injected and collect feature vectors.
    This is the healthy baseline both models train on.
    """
    log.info(f"Collecting {seconds}s of healthy baseline from {NODE_COUNT} nodes...")

    fleet    = Fleet(node_count=NODE_COUNT)
    store    = FeatureStore()
    pipeline = TelemetryPipeline(
        fleet.metrics_queue,
        datadog_enabled=False,   # don't need Datadog during training
    )

    fleet.start()
    pipeline.start()

    feature_vectors = []
    deadline = time.time() + seconds

    while time.time() < deadline:
        remaining = int(deadline - time.time())
        if remaining % 20 == 0 and remaining > 0:
            log.info(f"Collecting... {remaining}s remaining ({len(feature_vectors)} vectors so far)")

        try:
            fv = pipeline.feature_queue.get(timeout=1.0)
            feature_vectors.append(fv)
        except Exception:
            pass

    pipeline.stop()
    fleet.stop()

    log.info(f"Collected {len(feature_vectors)} healthy feature vectors")
    return feature_vectors


def train(feature_vectors: list):
    """Train both models and save to disk."""

    log.info("Training Isolation Forest...")
    if_detector = IsolationForestDetector(contamination=0.05, n_estimators=200)
    if_detector.fit(feature_vectors)
    if_detector.save()

    log.info("Training LSTM Autoencoder...")
    lstm_detector = LSTMDetector(seq_len=15, hidden_size=64, num_layers=2, epochs=30)
    lstm_detector.fit(feature_vectors)
    lstm_detector.save()

    log.info("Both models trained and saved to ml/models/")
    return if_detector, lstm_detector


def verify(if_detector, lstm_detector, feature_vectors: list):
    """Quick sanity check — score the last 10 healthy vectors."""
    log.info("Verifying models on healthy data (scores should be low)...")
    for fv in feature_vectors[-10:]:
        if_score   = if_detector.score(fv)
        lstm_score = lstm_detector.score(fv)
        log.info(f"  {fv.node_id} IF={if_score:.3f} LSTM={lstm_score:.3f}")


if __name__ == "__main__":
    vectors = collect_baseline(COLLECT_SECONDS)

    if len(vectors) < 50:
        log.error(f"Only got {len(vectors)} vectors — need at least 50. Run longer.")
        sys.exit(1)

    if_det, lstm_det = train(vectors)
    verify(if_det, lstm_det, vectors)

    log.info("Training complete. Run main.py to start the full system.")