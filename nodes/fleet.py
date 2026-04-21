"""
nodes/fleet.py

Fleet: manages the full collection of HostNodes.
Entry point for starting/stopping the simulated S3 host fleet,
injecting chaos, and exposing the shared metrics queue to Layer 2.
"""

import random
import logging
from queue import Queue
from typing import Optional

from nodes.chaos import FailureMode
from nodes.host_node import HostNode

log = logging.getLogger(__name__)


class Fleet:
    """
    Manages a collection of HostNodes as a simulated S3 fleet.

    All nodes share a single metrics_queue — Layer 2 (telemetry pipeline)
    reads from this queue and fans out to Datadog + the ML layer.
    Adding more nodes requires zero changes anywhere else in the system.
    That's the scale-out story.
    """

    def __init__(self, node_count: int = 6, queue_maxsize: int = 1000):
        self.metrics_queue: Queue = Queue(maxsize=queue_maxsize)
        self.nodes: dict[str, HostNode] = {
            f"node-{i:02d}": HostNode(f"node-{i:02d}", self.metrics_queue)
            for i in range(node_count)
        }
        log.info(f"Fleet initialised — {node_count} nodes")

    # ── lifecycle ──────────────────────────────────────────────────────────

    def start(self):
        for node in self.nodes.values():
            node.start()
        log.info(f"Fleet started — {len(self.nodes)} nodes running")

    def stop(self):
        for node in self.nodes.values():
            node.stop()
        log.info("Fleet stopped")

    # ── chaos ─────────────────────────────────────────────────────────────

    def inject(
        self,
        node_id: str,
        mode: FailureMode,
        severity: float = 1.0,
        duration_s: float = 60.0,
    ):
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")
        self.nodes[node_id].inject_failure(mode, severity, duration_s)

    def inject_random(self, mode: Optional[FailureMode] = None) -> tuple[str, FailureMode]:
        """Inject a random (or specified) failure into a random node."""
        node_id = random.choice(list(self.nodes.keys()))
        if mode is None:
            mode = random.choice([m for m in FailureMode if m != FailureMode.NONE])
        self.inject(node_id, mode)
        return node_id, mode

    def clear(self, node_id: str):
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")
        self.nodes[node_id].clear_failure()

    def clear_all(self):
        for node in self.nodes.values():
            node.clear_failure()

    # ── remediation (called by control plane) ──────────────────────────────

    def apply_remediation(self, node_id: str, action: str):
        if node_id not in self.nodes:
            raise ValueError(f"Unknown node: {node_id}")
        self.nodes[node_id].apply_remediation(action)

    # ── status ────────────────────────────────────────────────────────────

    def status(self) -> dict[str, str]:
        return {
            nid: node.current_failure_mode.value
            for nid, node in self.nodes.items()
        }

    def healthy_count(self) -> int:
        return sum(
            1 for node in self.nodes.values()
            if node.current_failure_mode == FailureMode.NONE
        )