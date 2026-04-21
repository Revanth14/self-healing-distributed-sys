"""
nodes/host_node.py

HostNode: simulates a single S3 fleet host.
LocalWatchdog: in-process safety net, independent of the control plane.
"""

import time
import random
import threading
import logging
from typing import Optional
from queue import Queue, Full

from nodes.chaos import (
    FailureMode,
    ChaosConfig,
    NodeMetrics,
    apply_chaos,
)

log = logging.getLogger(__name__)


class LocalWatchdog:
    """
    Lightweight in-process watchdog.
    Restarts the node locally after N consecutive failures — without waiting
    for the control plane. This is the answer to "what if the control plane
    goes down?" — nodes can still self-heal at a basic level.
    """

    CPU_HARD_LIMIT                 = 98.0
    ERROR_RATE_HARD_LIMIT          = 0.5
    CONSECUTIVE_FAILURES_THRESHOLD = 5

    def __init__(self, node: "HostNode"):
        self.node = node
        self.consecutive_failures = 0
        self.log = logging.getLogger(f"watchdog.{node.node_id}")

    def check(self, metrics: NodeMetrics) -> bool:
        """Returns True if a local restart was triggered."""
        bad = (
            metrics.cpu_pct    > self.CPU_HARD_LIMIT
            or metrics.error_rate > self.ERROR_RATE_HARD_LIMIT
            or not metrics.is_healthy
        )

        if bad:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0

        if self.consecutive_failures >= self.CONSECUTIVE_FAILURES_THRESHOLD:
            self.log.warning(
                f"Local restart after {self.consecutive_failures} failures "
                f"(cpu={metrics.cpu_pct:.1f}% err={metrics.error_rate:.3f})"
            )
            self.node.local_restart()
            self.consecutive_failures = 0
            return True

        return False


class HostNode:
    """
    Simulates a single S3 fleet host node.

    Emits NodeMetrics every emit_interval_s seconds into a shared queue.
    Baseline has Gaussian noise so the ML model learns real variance.
    Chaos is injected externally via inject_failure().
    """

    EMIT_INTERVAL_S = 2.0

    # Healthy baseline: (mean, std)
    BASELINE = {
        "cpu_pct":            (28.0,  4.0),
        "mem_used_pct":       (45.0,  3.0),
        "disk_io_util_pct":   (15.0,  5.0),
        "net_bytes_out":      (5e6,   1e6),
        "net_retransmits":    (2.0,   1.0),
        "open_fds":           (120,   10),
        "ctx_switches":       (800,   80),
        "latency_p50_ms":     (8.0,   1.5),
        "latency_p95_ms":     (18.0,  3.0),
        "latency_p99_ms":     (30.0,  5.0),
        "requests_per_sec":   (500.0, 40.0),
        "error_rate":         (0.002, 0.001),
        "active_connections": (80,    10),
        "gc_pause_ms":        (5.0,   2.0),
    }

    def __init__(
        self,
        node_id: str,
        metrics_queue: Queue,
        emit_interval_s: float = EMIT_INTERVAL_S,
    ):
        self.node_id = node_id
        self.metrics_queue = metrics_queue
        self.emit_interval_s = emit_interval_s

        self._started_at = time.time()
        self._chaos = ChaosConfig()
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._restart_count = 0
        self._last_remediation: Optional[str] = None

        self.watchdog = LocalWatchdog(self)
        self.log = logging.getLogger(f"node.{node_id}")

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._emit_loop,
            name=f"node-{self.node_id}",
            daemon=True,
        )
        self._thread.start()
        self.log.info("Started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        self.log.info("Stopped")

    def local_restart(self):
        """Called by LocalWatchdog or control plane for process restart."""
        self._restart_count += 1
        self._last_remediation = f"restart#{self._restart_count}"
        with self._lock:
            self._chaos = ChaosConfig()   # clear any active failure
        self._started_at = time.time()
        self.log.info(f"Restarted (#{self._restart_count})")

    def apply_remediation(self, action: str):
        """Called by the control plane."""
        self._last_remediation = action
        self.log.info(f"Remediation: {action}")
        if action in ("restart_process", "isolate_node"):
            self.local_restart()

    # ── chaos ─────────────────────────────────────────────────────────────

    def inject_failure(
        self,
        mode: FailureMode,
        severity: float = 1.0,
        duration_s: float = 60.0,
    ):
        with self._lock:
            self._chaos = ChaosConfig(mode=mode, severity=severity, duration_s=duration_s)
        self.log.warning(f"Chaos: {mode.value} severity={severity:.1f} duration={duration_s}s")

    def clear_failure(self):
        with self._lock:
            self._chaos = ChaosConfig()
        self.log.info("Chaos cleared")

    @property
    def current_failure_mode(self) -> FailureMode:
        with self._lock:
            return self._chaos.mode if self._chaos.is_active() else FailureMode.NONE

    # ── internals ─────────────────────────────────────────────────────────

    def _sample(self, key: str) -> float:
        mean, std = self.BASELINE[key]
        return max(0.0, random.gauss(mean, std))

    def _build_metrics(self) -> NodeMetrics:
        with self._lock:
            chaos = self._chaos

        base = {
            "cpu_pct":            self._sample("cpu_pct"),
            "mem_used_pct":       self._sample("mem_used_pct"),
            "disk_io_util_pct":   self._sample("disk_io_util_pct"),
            "net_bytes_out":      self._sample("net_bytes_out"),
            "net_retransmits":    int(self._sample("net_retransmits")),
            "open_fds":           int(self._sample("open_fds")),
            "ctx_switches":       self._sample("ctx_switches"),
            "latency_p50_ms":     self._sample("latency_p50_ms"),
            "latency_p95_ms":     self._sample("latency_p95_ms"),
            "latency_p99_ms":     self._sample("latency_p99_ms"),
            "requests_per_sec":   self._sample("requests_per_sec"),
            "error_rate":         max(0.0, self._sample("error_rate")),
            "active_connections": int(self._sample("active_connections")),
            "gc_pause_ms":        self._sample("gc_pause_ms"),
        }

        if chaos.is_active():
            base = apply_chaos(base, chaos)
        else:
            with self._lock:
                self._chaos = ChaosConfig()   # auto-expire

        is_healthy = (
            base["cpu_pct"]        < 90.0
            and base["error_rate"]     < 0.10
            and base["latency_p99_ms"] < 500.0
            and base["mem_used_pct"]   < 95.0
        )

        return NodeMetrics(
            node_id=self.node_id,
            ts=time.time(),
            cpu_pct=round(base["cpu_pct"], 2),
            mem_used_pct=round(base["mem_used_pct"], 2),
            disk_io_util_pct=round(base["disk_io_util_pct"], 2),
            net_bytes_out=round(base["net_bytes_out"], 0),
            net_retransmits=base["net_retransmits"],
            open_fds=base["open_fds"],
            ctx_switches_per_sec=round(base["ctx_switches"], 1),
            latency_p50_ms=round(base["latency_p50_ms"], 2),
            latency_p95_ms=round(base["latency_p95_ms"], 2),
            latency_p99_ms=round(base["latency_p99_ms"], 2),
            requests_per_sec=round(base["requests_per_sec"], 1),
            error_rate=round(base["error_rate"], 5),
            active_connections=base["active_connections"],
            gc_pause_ms=round(base["gc_pause_ms"], 2),
            uptime_s=round(time.time() - self._started_at, 1),
            is_healthy=is_healthy,
            last_remediation=self._last_remediation,
            failure_mode=chaos.mode.value if chaos.is_active() else "none",
        )

    def _emit_loop(self):
        while self._running:
            try:
                with self._lock:
                    silent = (
                        self._chaos.is_active()
                        and self._chaos.mode == FailureMode.SILENT_DEATH
                    )
                if not silent:
                    metrics = self._build_metrics()
                    self.watchdog.check(metrics)
                    try:
                        self.metrics_queue.put_nowait(metrics)
                    except Full:
                        self.log.warning("Queue full — dropping snapshot")
            except Exception as e:
                self.log.error(f"Emit error: {e}")

            time.sleep(self.emit_interval_s)