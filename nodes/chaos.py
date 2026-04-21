"""
nodes/chaos.py

Failure mode definitions, chaos configuration, and the NodeMetrics dataclass.
Everything the rest of the system needs to understand what a node is doing.
"""

import time
import random
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class FailureMode(Enum):
    NONE              = "none"
    CPU_SPIKE         = "cpu_spike"        # CPU saturates, latency climbs
    MEMORY_LEAK       = "memory_leak"      # Memory grows until OOM
    LATENCY_BLOWOUT   = "latency_blowout"  # Latency spikes, errors follow
    DISK_SATURATION   = "disk_saturation"  # Disk I/O maxes out
    SILENT_DEATH      = "silent_death"     # Node stops emitting entirely
    NETWORK_FLAP      = "network_flap"     # Intermittent packet loss
    CASCADING         = "cascading"        # CPU → latency → errors in sequence


@dataclass
class ChaosConfig:
    mode: FailureMode       = FailureMode.NONE
    severity: float         = 1.0    # 0.0–1.0 multiplier on failure intensity
    duration_s: float       = 60.0
    started_at: float       = field(default_factory=time.time)

    def elapsed(self) -> float:
        return time.time() - self.started_at

    def is_active(self) -> bool:
        return self.mode != FailureMode.NONE and self.elapsed() < self.duration_s

    def progress(self) -> float:
        """0.0 = just started, 1.0 = fully developed failure."""
        return min(self.elapsed() / max(self.duration_s * 0.3, 1.0), 1.0)


def apply_chaos(base: dict, chaos: ChaosConfig) -> dict:
    """
    Overlay failure-mode signals onto a healthy baseline metric dict.
    Failures ramp up gradually via progress() — critical for LSTM to learn
    the temporal degradation pattern, not just the end state.
    """
    m = dict(base)
    p = chaos.progress()
    s = chaos.severity

    if chaos.mode == FailureMode.CPU_SPIKE:
        m["cpu_pct"]        = min(98.0, base["cpu_pct"] + p * s * 65.0 + random.gauss(0, 4))
        m["latency_p99_ms"] = base["latency_p99_ms"] + p * s * 280.0 + random.gauss(0, 20)
        m["latency_p95_ms"] = base["latency_p95_ms"] + p * s * 150.0
        m["error_rate"]     = base["error_rate"] + max(0, p - 0.6) * s * 0.15

    elif chaos.mode == FailureMode.MEMORY_LEAK:
        frac = min(chaos.elapsed() / chaos.duration_s, 1.0)
        m["mem_used_pct"]   = min(99.0, base["mem_used_pct"] + frac * s * 52.0)
        m["gc_pause_ms"]    = base["gc_pause_ms"] + frac * s * 180.0
        m["latency_p99_ms"] = base["latency_p99_ms"] + frac * s * 120.0

    elif chaos.mode == FailureMode.LATENCY_BLOWOUT:
        m["latency_p50_ms"]     = base["latency_p50_ms"] + p * s * 80.0 + random.gauss(0, 8)
        m["latency_p95_ms"]     = base["latency_p95_ms"] + p * s * 200.0 + random.gauss(0, 15)
        m["latency_p99_ms"]     = base["latency_p99_ms"] + p * s * 400.0 + random.gauss(0, 30)
        m["error_rate"]         = base["error_rate"] + p * s * 0.18 + random.gauss(0, 0.01)
        m["active_connections"] = int(base["active_connections"] * (1 + p * s * 1.5))

    elif chaos.mode == FailureMode.DISK_SATURATION:
        m["disk_io_util_pct"] = min(100.0, base["disk_io_util_pct"] + p * s * 80.0)
        m["latency_p99_ms"]   = base["latency_p99_ms"] + p * s * 150.0
        m["open_fds"]         = int(base["open_fds"] + p * s * 800)

    elif chaos.mode == FailureMode.NETWORK_FLAP:
        if random.random() < p * s * 0.4:
            m["net_retransmits"] = int(base["net_retransmits"] + random.gauss(50, 15))
            m["error_rate"]      = base["error_rate"] + random.gauss(0.05, 0.02)
            m["latency_p99_ms"]  = base["latency_p99_ms"] + random.gauss(120, 30)

    elif chaos.mode == FailureMode.CASCADING:
        if p < 0.33:
            pp = p / 0.33
            m["cpu_pct"] = min(98.0, base["cpu_pct"] + pp * s * 55.0)
        elif p < 0.66:
            pp = (p - 0.33) / 0.33
            m["cpu_pct"]        = min(98.0, base["cpu_pct"] + s * 55.0)
            m["latency_p99_ms"] = base["latency_p99_ms"] + pp * s * 320.0
        else:
            pp = (p - 0.66) / 0.34
            m["cpu_pct"]        = min(98.0, base["cpu_pct"] + s * 55.0)
            m["latency_p99_ms"] = base["latency_p99_ms"] + s * 320.0
            m["error_rate"]     = base["error_rate"] + pp * s * 0.20

    # Real degradation is never clean — add noise on top
    for key in ["cpu_pct", "latency_p99_ms", "error_rate", "mem_used_pct"]:
        if key in m:
            m[key] = max(0.0, m[key] + random.gauss(0, abs(m[key]) * 0.03))

    return m


@dataclass
class NodeMetrics:
    node_id: str
    ts: float

    # System-level
    cpu_pct: float            = 0.0
    mem_used_pct: float       = 0.0
    disk_io_util_pct: float   = 0.0
    net_bytes_out: float      = 0.0
    net_retransmits: int      = 0
    open_fds: int             = 0
    ctx_switches_per_sec: float = 0.0

    # Process-level
    latency_p50_ms: float     = 0.0
    latency_p95_ms: float     = 0.0
    latency_p99_ms: float     = 0.0
    requests_per_sec: float   = 0.0
    error_rate: float         = 0.0
    active_connections: int   = 0
    gc_pause_ms: float        = 0.0

    # Health signals
    uptime_s: float           = 0.0
    is_healthy: bool          = True
    last_remediation: Optional[str] = None
    anomaly_score: float      = 0.0   # filled in by ML layer
    failure_mode: str         = "none"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_datadog_tags(self) -> list[str]:
        return [
            f"node_id:{self.node_id}",
            f"failure_mode:{self.failure_mode}",
            f"healthy:{str(self.is_healthy).lower()}",
        ]