"""
ml/scorer.py

AnomalyScorer: combines Isolation Forest + LSTM scores into a single
confidence score per node per cycle. Fires Datadog anomaly events
and notifies the Java control plane when score crosses threshold.

The dual-model approach is the centrepiece of this project:
- Isolation Forest catches point anomalies fast
- LSTM catches gradual sequence drift that IF misses
- Combined score reduces false positives from either model alone
"""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Weights for combining the two model scores
IF_WEIGHT   = 0.3
LSTM_WEIGHT = 0.7

# Score above this triggers an anomaly event
ANOMALY_THRESHOLD = 0.65


@dataclass
class ScorerResult:
    node_id: str
    ts: float
    if_score: float        # Isolation Forest score [0,1]
    lstm_score: float      # LSTM autoencoder score [0,1]
    combined_score: float  # weighted combination [0,1]
    is_anomaly: bool
    failure_mode: str


class AnomalyScorer:
    """
    Combines IF + LSTM scores and decides whether to fire an alert.
    Plugs directly into TelemetryPipeline as anomaly_score_fn.
    """

    def __init__(
        self,
        if_detector=None,
        lstm_detector=None,
        threshold: float = ANOMALY_THRESHOLD,
        datadog_enabled: bool = True,
    ):
        self.if_detector     = if_detector
        self.lstm_detector   = lstm_detector
        self.threshold       = threshold
        self.datadog_enabled = datadog_enabled

        self._last_scores: dict[str, ScorerResult] = {}
        self._anomaly_count = 0

    # ── main entry points ──────────────────────────────────────────────────

    def score_metrics(self, metrics) -> float:
        """
        Called by TelemetryPipeline for every raw NodeMetrics snapshot.
        Returns a heuristic anomaly score [0,1] before ML models kick in.
        """
        score = 0.0
        if metrics.cpu_pct > 80:
            score = max(score, (metrics.cpu_pct - 80) / 20)
        if metrics.latency_p99_ms > 200:
            score = max(score, min((metrics.latency_p99_ms - 200) / 300, 1.0))
        if metrics.error_rate > 0.05:
            score = max(score, min(metrics.error_rate / 0.2, 1.0))
        return round(score, 4)

    def score_features(self, feature_vector, metrics=None) -> ScorerResult:
        """
        Full scoring using both models on an engineered FeatureVector.
        feature_vector carries failure_mode so metrics param is optional.
        """
        if_score   = self.if_detector.score(feature_vector)   if self.if_detector   else 0.0
        lstm_score = self.lstm_detector.score(feature_vector) if self.lstm_detector else 0.0

        combined   = round(IF_WEIGHT * if_score + LSTM_WEIGHT * lstm_score, 4)
        is_anomaly = combined >= self.threshold

        result = ScorerResult(
            node_id=feature_vector.node_id,
            ts=feature_vector.ts,
            if_score=round(if_score, 4),
            lstm_score=round(lstm_score, 4),
            combined_score=combined,
            is_anomaly=is_anomaly,
            failure_mode=getattr(feature_vector, "failure_mode", "unknown"),
        )

        self._last_scores[feature_vector.node_id] = result

        if is_anomaly:
            self._anomaly_count += 1
            log.warning(
                f"ANOMALY {feature_vector.node_id} | "
                f"combined={combined:.3f} "
                f"if={if_score:.3f} lstm={lstm_score:.3f} "
                f"mode={result.failure_mode}"
            )
            # Fire Datadog event if raw metrics available
            if self.datadog_enabled and metrics:
                self._fire_datadog_event(result, metrics)

            # Always notify control plane — use feature_vector for metric values
            self._fire_control_plane(result, feature_vector)

        return result

    # ── integrations ───────────────────────────────────────────────────────

    def _fire_datadog_event(self, result: ScorerResult, metrics):
        try:
            from telemetry.datadog_client import send_anomaly_event
            send_anomaly_event(
                node_id=result.node_id,
                score=result.combined_score,
                failure_mode=result.failure_mode,
                metrics=metrics,
            )
        except Exception as e:
            log.warning(f"Failed to fire Datadog event: {e}")

    def _fire_control_plane(self, result: ScorerResult, source):
        """
        POST anomaly to Java control plane.
        source can be raw NodeMetrics or FeatureVector — both have
        cpu_pct, latency_p99_ms, error_rate, mem_used_pct fields.
        """
        import requests
        try:
            r = requests.post(
                "http://localhost:8080/api/anomaly",
                json={
                    "node_id":        result.node_id,
                    "combined_score": result.combined_score,
                    "if_score":       result.if_score,
                    "lstm_score":     result.lstm_score,
                    "failure_mode":   result.failure_mode,
                    "cpu_pct":        getattr(source, "cpu_pct", 0.0),
                    "latency_p99_ms": getattr(source, "latency_p99_ms", 0.0),
                    "error_rate":     getattr(source, "error_rate", 0.0),
                    "mem_used_pct":   getattr(source, "mem_used_pct", 0.0),
                },
                timeout=2,
            )
            if r.status_code == 200:
                log.info(f"Control plane notified: {result.node_id} → {r.json().get('action')}")
            else:
                log.warning(f"Control plane returned {r.status_code}")
        except Exception as e:
            log.warning(f"Control plane unreachable: {e}")

    # ── status ────────────────────────────────────────────────────────────

    def latest_scores(self) -> dict[str, ScorerResult]:
        return dict(self._last_scores)

    def anomaly_count(self) -> int:
        return self._anomaly_count

    @property
    def models_ready(self) -> bool:
        return (
            self.if_detector is not None and self.if_detector.is_trained
            and self.lstm_detector is not None and self.lstm_detector.is_trained
        )