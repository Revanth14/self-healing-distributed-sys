"""
telemetry/pipeline.py

Telemetry pipeline — the bridge between Layer 1 (nodes) and Layer 3 (ML).

Consumes NodeMetrics from the fleet queue in a background thread.
For each snapshot it:
  1. Ships raw metrics to Datadog
  2. Runs feature engineering (rolling stats, deltas, lags)
  3. Puts the FeatureVector into an output queue for the ML scorer

This is intentionally stateless and scale-out: you can add more nodes
to the fleet without touching this file at all.
"""

import time
import threading
import logging
from queue import Queue, Empty
from typing import Optional, Callable

from telemetry.features import FeatureStore, FeatureVector
from telemetry.datadog_client import send_metrics

log = logging.getLogger(__name__)


class TelemetryPipeline:
    """
    Reads from fleet.metrics_queue, engineers features,
    ships to Datadog, and puts FeatureVectors into feature_queue
    for the ML layer to consume.
    """

    BATCH_INTERVAL_S = 2.0   # how often to flush to Datadog

    def __init__(
        self,
        metrics_queue: Queue,
        feature_queue: Optional[Queue] = None,
        anomaly_score_fn: Optional[Callable] = None,
        datadog_enabled: bool = True,
    ):
        self.metrics_queue = metrics_queue
        self.feature_queue = feature_queue or Queue(maxsize=500)
        self.anomaly_score_fn = anomaly_score_fn   # plugged in by ML layer
        self.datadog_enabled = datadog_enabled

        self._store = FeatureStore()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed = 0
        self._dropped = 0

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run,
            name="telemetry-pipeline",
            daemon=True,
        )
        self._thread.start()
        log.info("Telemetry pipeline started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        log.info(f"Pipeline stopped — processed={self._processed} dropped={self._dropped}")

    # ── stats ──────────────────────────────────────────────────────────────

    @property
    def stats(self) -> dict:
        return {
            "processed": self._processed,
            "dropped":   self._dropped,
            "nodes_seen": len(self._store.node_ids()),
        }

    # ── internals ─────────────────────────────────────────────────────────

    def _run(self):
        while self._running:
            try:
                metrics = self.metrics_queue.get(timeout=1.0)
            except Empty:
                continue

            try:
                self._process(metrics)
                self._processed += 1
            except Exception as e:
                log.error(f"Pipeline error for {metrics.node_id}: {e}")
                self._dropped += 1

    def _process(self, metrics):
        # 1. Get anomaly score if ML layer is wired in
        anomaly_score = 0.0
        if self.anomaly_score_fn:
            try:
                anomaly_score = self.anomaly_score_fn(metrics)
            except Exception as e:
                log.warning(f"Anomaly score fn error: {e}")

        # 2. Ship raw metrics + anomaly score to Datadog
        if self.datadog_enabled:
            send_metrics(metrics, anomaly_score=anomaly_score)

        # 3. Feature engineering
        feature_vector = self._store.push(metrics)

        # 4. Put FeatureVector into output queue for ML layer
        if feature_vector is not None:
            try:
                self.feature_queue.put_nowait(feature_vector)
            except Exception:
                self._dropped += 1
                log.debug(f"Feature queue full — dropping {metrics.node_id}")