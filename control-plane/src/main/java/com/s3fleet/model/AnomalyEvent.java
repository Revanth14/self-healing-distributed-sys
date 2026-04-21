package com.s3fleet.model;

import jakarta.persistence.*;
import java.time.Instant;

/**
 * Persisted record of every anomaly event received from the ML layer.
 * Stored in H2 (dev) or PostgreSQL (prod).
 * This is the answer to "what if the control plane crashes?" —
 * it replays from this table on restart.
 */
@Entity
@Table(name = "anomaly_events")
public class AnomalyEvent {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false)
    private String nodeId;

    @Column(nullable = false)
    private Double anomalyScore;

    @Column(nullable = false)
    private Double ifScore;

    @Column(nullable = false)
    private Double lstmScore;

    @Column(nullable = false)
    private String failureMode;

    private Double cpuPct;
    private Double latencyP99Ms;
    private Double errorRate;
    private Double memUsedPct;

    @Column(nullable = false)
    private String remediationAction;

    @Column(nullable = false)
    private String status; // PENDING, APPLIED, FAILED

    @Column(nullable = false)
    private Instant receivedAt;

    private Instant resolvedAt;

    // ── constructors ──────────────────────────────────────────────────────

    public AnomalyEvent() {
    }

    public AnomalyEvent(String nodeId, Double anomalyScore, Double ifScore,
            Double lstmScore, String failureMode,
            Double cpuPct, Double latencyP99Ms,
            Double errorRate, Double memUsedPct,
            String remediationAction) {
        this.nodeId = nodeId;
        this.anomalyScore = anomalyScore;
        this.ifScore = ifScore;
        this.lstmScore = lstmScore;
        this.failureMode = failureMode;
        this.cpuPct = cpuPct;
        this.latencyP99Ms = latencyP99Ms;
        this.errorRate = errorRate;
        this.memUsedPct = memUsedPct;
        this.remediationAction = remediationAction;
        this.status = "PENDING";
        this.receivedAt = Instant.now();
    }

    // ── getters / setters ─────────────────────────────────────────────────

    public Long getId() {
        return id;
    }

    public String getNodeId() {
        return nodeId;
    }

    public Double getAnomalyScore() {
        return anomalyScore;
    }

    public Double getIfScore() {
        return ifScore;
    }

    public Double getLstmScore() {
        return lstmScore;
    }

    public String getFailureMode() {
        return failureMode;
    }

    public Double getCpuPct() {
        return cpuPct;
    }

    public Double getLatencyP99Ms() {
        return latencyP99Ms;
    }

    public Double getErrorRate() {
        return errorRate;
    }

    public Double getMemUsedPct() {
        return memUsedPct;
    }

    public String getRemediationAction() {
        return remediationAction;
    }

    public String getStatus() {
        return status;
    }

    public Instant getReceivedAt() {
        return receivedAt;
    }

    public Instant getResolvedAt() {
        return resolvedAt;
    }

    public void setStatus(String status) {
        this.status = status;
    }

    public void setResolvedAt(Instant resolvedAt) {
        this.resolvedAt = resolvedAt;
    }

    public void setRemediationAction(String action) {
        this.remediationAction = action;
    }
}