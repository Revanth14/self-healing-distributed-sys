package com.s3fleet.controller;

import com.s3fleet.model.AnomalyEvent;
import com.s3fleet.service.RemediationService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.Map;

/**
 * AnomalyController: REST API for the control plane.
 *
 * The Python ML scorer POSTs anomaly events here.
 * The dashboard GETs status and remediation history from here.
 *
 * Endpoints:
 * POST /api/anomaly — receive anomaly from ML layer
 * GET /api/events — recent remediation log
 * GET /api/events/{nodeId} — events for a specific node
 * GET /api/status — overall fleet status
 * GET /api/health — control plane health check
 */
@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class AnomalyController {

    private static final Logger log = LoggerFactory.getLogger(AnomalyController.class);

    private final RemediationService remediationService;

    public AnomalyController(RemediationService remediationService) {
        this.remediationService = remediationService;
    }

    /**
     * Main endpoint — called by Python scorer when anomaly detected.
     * Triggers decision engine + remediation + persistence.
     */
    @PostMapping("/anomaly")
    public ResponseEntity<Map<String, Object>> receiveAnomaly(
            @RequestBody Map<String, Object> payload) {
        try {
            String nodeId = (String) payload.get("node_id");
            double score = toDouble(payload.get("combined_score"));
            double ifScore = toDouble(payload.get("if_score"));
            double lstmScore = toDouble(payload.get("lstm_score"));
            String failureMode = (String) payload.getOrDefault("failure_mode", "unknown");
            double cpuPct = toDouble(payload.getOrDefault("cpu_pct", 0.0));
            double latencyP99 = toDouble(payload.getOrDefault("latency_p99_ms", 0.0));
            double errorRate = toDouble(payload.getOrDefault("error_rate", 0.0));
            double memPct = toDouble(payload.getOrDefault("mem_used_pct", 0.0));

            if (nodeId == null || nodeId.isBlank()) {
                return ResponseEntity.badRequest()
                        .body(Map.of("error", "node_id is required"));
            }

            AnomalyEvent event = remediationService.process(
                    nodeId, score, ifScore, lstmScore, failureMode,
                    cpuPct, latencyP99, errorRate, memPct);

            return ResponseEntity.ok(Map.of(
                    "status", "processed",
                    "event_id", event.getId(),
                    "action", event.getRemediationAction(),
                    "outcome", event.getStatus()));

        } catch (Exception e) {
            log.error("Error processing anomaly: {}", e.getMessage(), e);
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", e.getMessage()));
        }
    }

    /**
     * Returns the 50 most recent anomaly + remediation events.
     * Used by the dashboard remediation log widget.
     */
    @GetMapping("/events")
    public ResponseEntity<List<AnomalyEvent>> getRecentEvents() {
        return ResponseEntity.ok(remediationService.getRecentEvents());
    }

    /**
     * Returns all events for a specific node.
     * Useful for investigating a flaky node.
     */
    @GetMapping("/events/{nodeId}")
    public ResponseEntity<List<AnomalyEvent>> getNodeEvents(
            @PathVariable String nodeId) {
        return ResponseEntity.ok(remediationService.getEventsByNode(nodeId));
    }

    /**
     * Fleet status summary.
     */
    @GetMapping("/status")
    public ResponseEntity<Map<String, Object>> getStatus() {
        List<AnomalyEvent> recent = remediationService.getRecentEvents();
        List<AnomalyEvent> pending = remediationService.getPendingEvents();

        long applied = recent.stream()
                .filter(e -> "APPLIED".equals(e.getStatus()))
                .count();

        long failed = recent.stream()
                .filter(e -> "FAILED".equals(e.getStatus()))
                .count();

        return ResponseEntity.ok(Map.of(
                "total_events", recent.size(),
                "applied", applied,
                "pending", pending.size(),
                "failed", failed,
                "control_plane", "healthy"));
    }

    /**
     * Simple health check — used by load balancer and monitoring.
     */
    @GetMapping("/health")
    public ResponseEntity<Map<String, String>> health() {
        return ResponseEntity.ok(Map.of(
                "status", "UP",
                "service", "s3-fleet-control-plane"));
    }

    private double toDouble(Object val) {
        if (val == null)
            return 0.0;
        if (val instanceof Number n)
            return n.doubleValue();
        return Double.parseDouble(val.toString());
    }
}