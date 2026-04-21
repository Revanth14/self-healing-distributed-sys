"""
telemetry/features.py

Feature engineering for the ML anomaly detection layer.
Converts raw NodeMetrics snapshots into ML-ready feature vectors
using rolling statistics, lag features, and rate-of-change signals.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional
import statistics


WINDOW_SIZE = 30


@dataclass
class FeatureVector:
    node_id: str
    ts: float

    # Raw values
    cpu_pct: float
    mem_used_pct: float
    latency_p99_ms: float
    error_rate: float
    disk_io_util_pct: float
    net_retransmits: float
    gc_pause_ms: float

    # Rolling means
    cpu_mean: float
    latency_mean: float
    error_mean: float
    mem_mean: float

    # Rolling stds
    cpu_std: float
    latency_std: float
    error_std: float

    # Deltas
    cpu_delta: float
    latency_delta: float
    error_delta: float
    mem_delta: float

    # Lag features
    cpu_lag1: float
    latency_lag1: float
    error_lag1: float

    # Failure mode — carried from raw NodeMetrics, default last so dataclass is valid
    failure_mode: str = "none"

    def to_list(self) -> list[float]:
        return [
            self.cpu_pct, self.mem_used_pct, self.latency_p99_ms,
            self.error_rate, self.disk_io_util_pct, self.net_retransmits,
            self.gc_pause_ms,
            self.cpu_mean, self.latency_mean, self.error_mean, self.mem_mean,
            self.cpu_std, self.latency_std, self.error_std,
            self.cpu_delta, self.latency_delta, self.error_delta, self.mem_delta,
            self.cpu_lag1, self.latency_lag1, self.error_lag1,
        ]

    @property
    def feature_names(self) -> list[str]:
        return [
            "cpu_pct", "mem_used_pct", "latency_p99_ms",
            "error_rate", "disk_io_util_pct", "net_retransmits",
            "gc_pause_ms",
            "cpu_mean", "latency_mean", "error_mean", "mem_mean",
            "cpu_std", "latency_std", "error_std",
            "cpu_delta", "latency_delta", "error_delta", "mem_delta",
            "cpu_lag1", "latency_lag1", "error_lag1",
        ]


class NodeFeatureBuffer:
    def __init__(self, node_id: str, window: int = WINDOW_SIZE):
        self.node_id = node_id
        self.window  = window
        self._buf: deque = deque(maxlen=window)

    def push(self, metrics) -> Optional[FeatureVector]:
        self._buf.append(metrics)
        if len(self._buf) < 3:
            return None
        return self._compute()

    def _vals(self, key: str) -> list[float]:
        return [getattr(m, key) for m in self._buf]

    def _mean(self, key: str) -> float:
        v = self._vals(key)
        return statistics.mean(v) if v else 0.0

    def _std(self, key: str) -> float:
        v = self._vals(key)
        return statistics.stdev(v) if len(v) >= 2 else 0.0

    def _delta(self, key: str) -> float:
        if len(self._buf) < 2:
            return 0.0
        return getattr(self._buf[-1], key) - getattr(self._buf[-2], key)

    def _lag(self, key: str, n: int = 1) -> float:
        idx = -(n + 1)
        if abs(idx) > len(self._buf):
            return getattr(self._buf[0], key)
        return getattr(self._buf[idx], key)

    def _compute(self) -> FeatureVector:
        latest = self._buf[-1]
        return FeatureVector(
            node_id=self.node_id,
            ts=latest.ts,
            failure_mode=latest.failure_mode,
            cpu_pct=latest.cpu_pct,
            mem_used_pct=latest.mem_used_pct,
            latency_p99_ms=latest.latency_p99_ms,
            error_rate=latest.error_rate,
            disk_io_util_pct=latest.disk_io_util_pct,
            net_retransmits=latest.net_retransmits,
            gc_pause_ms=latest.gc_pause_ms,
            cpu_mean=self._mean("cpu_pct"),
            latency_mean=self._mean("latency_p99_ms"),
            error_mean=self._mean("error_rate"),
            mem_mean=self._mean("mem_used_pct"),
            cpu_std=self._std("cpu_pct"),
            latency_std=self._std("latency_p99_ms"),
            error_std=self._std("error_rate"),
            cpu_delta=self._delta("cpu_pct"),
            latency_delta=self._delta("latency_p99_ms"),
            error_delta=self._delta("error_rate"),
            mem_delta=self._delta("mem_used_pct"),
            cpu_lag1=self._lag("cpu_pct", 1),
            latency_lag1=self._lag("latency_p99_ms", 1),
            error_lag1=self._lag("error_rate", 1),
        )


class FeatureStore:
    def __init__(self):
        self._buffers: dict[str, NodeFeatureBuffer] = {}

    def push(self, metrics) -> Optional[FeatureVector]:
        node_id = metrics.node_id
        if node_id not in self._buffers:
            self._buffers[node_id] = NodeFeatureBuffer(node_id)
        return self._buffers[node_id].push(metrics)

    def node_ids(self) -> list[str]:
        return list(self._buffers.keys())