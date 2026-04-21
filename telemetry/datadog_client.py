"""
telemetry/datadog_client.py

Sends metrics and events directly to Datadog via HTTP API.
No Datadog Agent required — works out of the box in any dev environment.
"""

import os
import time
import logging
import requests

log = logging.getLogger(__name__)

_api_key = None
_app_key = None
_site    = "datadoghq.com"
_base    = None


def init():
    global _api_key, _app_key, _site, _base

    _api_key = os.getenv("DD_API_KEY")
    _app_key = os.getenv("DD_APP_KEY")
    _site    = os.getenv("DD_SITE", "datadoghq.com")
    _base    = f"https://api.{_site}"

    if not _api_key:
        raise RuntimeError("DD_API_KEY not set — check your .env file")

    log.info(f"Datadog HTTP client initialised (site={_site})")


def _headers() -> dict:
    return {
        "DD-API-KEY":          _api_key,
        "DD-APPLICATION-KEY":  _app_key,
        "Content-Type":        "application/json",
    }


def send_metrics(metrics, anomaly_score: float = 0.0):
    tags = metrics.to_datadog_tags()
    now  = int(metrics.ts)

    gauges = {
        "fleet.node.cpu_pct":            metrics.cpu_pct,
        "fleet.node.mem_used_pct":       metrics.mem_used_pct,
        "fleet.node.disk_io_util_pct":   metrics.disk_io_util_pct,
        "fleet.node.latency_p50_ms":     metrics.latency_p50_ms,
        "fleet.node.latency_p95_ms":     metrics.latency_p95_ms,
        "fleet.node.latency_p99_ms":     metrics.latency_p99_ms,
        "fleet.node.error_rate":         metrics.error_rate,
        "fleet.node.requests_per_sec":   metrics.requests_per_sec,
        "fleet.node.active_connections": metrics.active_connections,
        "fleet.node.gc_pause_ms":        metrics.gc_pause_ms,
        "fleet.node.net_retransmits":    metrics.net_retransmits,
        "fleet.node.open_fds":           metrics.open_fds,
        "fleet.node.uptime_s":           metrics.uptime_s,
        "fleet.node.anomaly_score":      anomaly_score,
        "fleet.node.is_healthy":         int(metrics.is_healthy),
    }

    series = [
        {
            "metric": name,
            "type":   3,
            "points": [{"timestamp": now, "value": value}],
            "tags":   tags,
        }
        for name, value in gauges.items()
    ]

    try:
        r = requests.post(
            f"{_base}/api/v2/series",
            headers=_headers(),
            json={"series": series},
            timeout=5,
        )
        if r.status_code != 202:
            log.warning(f"Datadog metrics HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        log.warning(f"Failed to send metrics: {e}")


def send_anomaly_event(node_id: str, score: float, failure_mode: str, metrics):
    try:
        r = requests.post(
            f"{_base}/api/v1/events",
            headers=_headers(),
            json={
                "title":      f"Anomaly detected: {node_id}",
                "text":       (
                    f"%%% \n"
                    f"**Node**: {node_id}  \n"
                    f"**Anomaly score**: {score:.3f}  \n"
                    f"**Failure mode**: {failure_mode}  \n"
                    f"**CPU**: {metrics.cpu_pct:.1f}%  "
                    f"**Latency p99**: {metrics.latency_p99_ms:.1f}ms  "
                    f"**Error rate**: {metrics.error_rate:.4f}  \n"
                    f"%%%"
                ),
                "alert_type": "error",
                "tags": [f"node_id:{node_id}", f"failure_mode:{failure_mode}", "source:s3-fleet-ml"],
            },
            timeout=5,
        )
        if r.status_code == 202:
            log.info(f"Anomaly event sent for {node_id} (score={score:.3f})")
        else:
            log.warning(f"Anomaly event HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        log.warning(f"Failed to send anomaly event: {e}")


def send_remediation_event(node_id: str, action: str, anomaly_score: float):
    try:
        r = requests.post(
            f"{_base}/api/v1/events",
            headers=_headers(),
            json={
                "title":      f"Remediation fired: {action} on {node_id}",
                "text":       (
                    f"%%% \n"
                    f"**Node**: {node_id}  \n"
                    f"**Action**: {action}  \n"
                    f"**Anomaly score at trigger**: {anomaly_score:.3f}  \n"
                    f"%%%"
                ),
                "alert_type": "warning",
                "tags": [f"node_id:{node_id}", f"action:{action}", "source:s3-fleet-control-plane"],
            },
            timeout=5,
        )
        if r.status_code == 202:
            log.info(f"Remediation event sent: {action} on {node_id}")
        else:
            log.warning(f"Remediation event HTTP {r.status_code}: {r.text[:120]}")
    except Exception as e:
        log.warning(f"Failed to send remediation event: {e}")