"""
main.py

Single entry point for the S3 Fleet Management self-healing system.
Starts all Python layers and runs until Ctrl+C.

The Java control plane (Layer 4) must be started separately:
    cd control-plane && mvn spring-boot:run

Usage:
    uv run python3 main.py                    # normal run
    uv run python3 main.py --chaos            # inject random failures every 60s
    uv run python3 main.py --chaos --node node-02 --mode cascading
"""

import os
import sys
import time
import signal
import logging
import argparse
import threading

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")


def parse_args():
    parser = argparse.ArgumentParser(description="S3 Fleet Management System")
    parser.add_argument("--nodes",  type=int, default=6,    help="Number of simulated nodes")
    parser.add_argument("--chaos",  action="store_true",    help="Inject random failures automatically")
    parser.add_argument("--node",   type=str, default=None, help="Specific node to inject failure into")
    parser.add_argument("--mode",   type=str, default=None, help="Failure mode (cascading, cpu_spike, etc)")
    parser.add_argument("--interval", type=int, default=60, help="Chaos injection interval in seconds")
    parser.add_argument("--no-datadog", action="store_true", help="Disable Datadog (useful for testing)")
    return parser.parse_args()


def chaos_loop(fleet, interval: int, node: str = None, mode_str: str = None):
    """Injects random failures in the background if --chaos flag is set."""
    from nodes.chaos import FailureMode
    import random

    time.sleep(30)   # let baseline settle first

    while True:
        try:
            if node and mode_str:
                mode = FailureMode[mode_str.upper()]
                fleet.inject(node, mode, severity=1.0, duration_s=40)
                log.warning(f"Chaos injected: {mode_str} → {node}")
            else:
                node_id, mode = fleet.inject_random()
                log.warning(f"Random chaos injected: {mode.value} → {node_id}")
        except Exception as e:
            log.error(f"Chaos injection error: {e}")

        time.sleep(interval)


def scoring_loop(pipeline, scorer):
    """Consumes FeatureVectors and runs full ML scoring in a background thread."""
    from queue import Empty
    while True:
        try:
            fv = pipeline.feature_queue.get(timeout=1.0)
            scorer.score_features(fv)
        except Empty:
            continue
        except Exception as e:
            log.error(f"Scoring error: {e}")


def print_status(fleet, scorer, pipeline):
    """Print a live status line every 10 seconds."""
    while True:
        time.sleep(10)
        status = fleet.status()
        degraded = [n for n, m in status.items() if m != "none"]
        healthy  = len(status) - len(degraded)
        print(
            f"[{time.strftime('%H:%M:%S')}] "
            f"Fleet: {healthy}/{len(status)} healthy | "
            f"Anomalies detected: {scorer.anomaly_count()} | "
            f"Pipeline: processed={pipeline.stats['processed']} dropped={pipeline.stats['dropped']}"
            + (f" | DEGRADED: {degraded}" if degraded else "")
        )


def main():
    args = parse_args()

    print("=" * 60)
    print("S3 Fleet Management — Self-Healing System")
    print("=" * 60)
    print(f"Nodes: {args.nodes} | Chaos: {args.chaos} | Datadog: {not args.no_datadog}")
    print("Control plane: http://localhost:8080/api/status")
    print("Ctrl+C to stop")
    print("=" * 60)

    # ── initialise Datadog ─────────────────────────────────────────────────
    datadog_enabled = not args.no_datadog
    if datadog_enabled:
        try:
            from telemetry.datadog_client import init as datadog_init
            datadog_init()
            print("Datadog: connected")
        except Exception as e:
            log.warning(f"Datadog init failed: {e} — continuing without Datadog")
            datadog_enabled = False

    # ── load ML models ─────────────────────────────────────────────────────
    from ml.isolation_forest import IsolationForestDetector
    from ml.lstm_autoencoder import LSTMDetector
    from ml.scorer import AnomalyScorer

    try:
        if_det   = IsolationForestDetector.load()
        lstm_det = LSTMDetector.load()
        print("ML models: loaded")
    except Exception as e:
        print(f"ML models not found: {e}")
        print("Run 'uv run python3 -m ml.train' first")
        sys.exit(1)

    scorer = AnomalyScorer(
        if_det, lstm_det,
        threshold=0.65,
        datadog_enabled=datadog_enabled,
    )

    # ── start fleet ────────────────────────────────────────────────────────
    from nodes.fleet import Fleet
    fleet = Fleet(node_count=args.nodes)

    # ── start telemetry pipeline ───────────────────────────────────────────
    from telemetry.pipeline import TelemetryPipeline
    pipeline = TelemetryPipeline(
        fleet.metrics_queue,
        datadog_enabled=datadog_enabled,
        anomaly_score_fn=scorer.score_metrics,
    )

    fleet.start()
    pipeline.start()
    print(f"Fleet: {args.nodes} nodes running")

    # ── start background threads ───────────────────────────────────────────
    threads = []

    scorer_thread = threading.Thread(
        target=scoring_loop, args=(pipeline, scorer), daemon=True
    )
    scorer_thread.start()
    threads.append(scorer_thread)

    status_thread = threading.Thread(
        target=print_status, args=(fleet, scorer, pipeline), daemon=True
    )
    status_thread.start()
    threads.append(status_thread)

    if args.chaos:
        chaos_thread = threading.Thread(
            target=chaos_loop,
            args=(fleet, args.interval, args.node, args.mode),
            daemon=True,
        )
        chaos_thread.start()
        threads.append(chaos_thread)
        print(f"Chaos mode: ON (every {args.interval}s)")

    print("\nSystem running — watching for anomalies...\n")

    # ── graceful shutdown ──────────────────────────────────────────────────
    def shutdown(sig, frame):
        print("\nShutting down...")
        pipeline.stop()
        fleet.stop()
        print(f"Final stats: anomalies={scorer.anomaly_count()} {pipeline.stats}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()