package com.s3fleet.engine;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

/**
 * DecisionEngine: maps anomaly signals to remediation actions.
 *
 * This is intentionally rule-based — simple, auditable, and explainable.
 * In production you'd layer ML-based action selection on top of this,
 * but a rule engine is the right foundation because every decision
 * needs to be defensible and traceable.
 *
 * Rule priority (highest first):
 * 1. Silent death → isolate node (it's gone, drain it)
 * 2. High error rate → restart process (fastest recovery)
 * 3. Cascading failure → reroute traffic first, then restart
 * 4. CPU spike → reroute traffic (buy time)
 * 5. Memory leak → restart process (only fix)
 * 6. Latency blowout → reroute traffic
 * 7. Disk saturation → alert + escalate (needs human)
 * 8. High combined score → restart process (safe default)
 * 9. Moderate score → alert only
 */
@Component
public class DecisionEngine {

    private static final Logger log = LoggerFactory.getLogger(DecisionEngine.class);

    // Thresholds
    private static final double HIGH_SCORE = 0.80;
    private static final double MODERATE_SCORE = 0.55;
    private static final double HIGH_ERROR = 0.10;
    private static final double HIGH_CPU = 85.0;
    private static final double HIGH_LATENCY = 300.0;
    private static final double HIGH_MEM = 90.0;

    // Max remediations per node before escalating to human
    private static final int MAX_AUTO_REMEDIATIONS = 3;

    public record Decision(String action, String reason, int priority) {
    }

    /**
     * Core decision logic. Called by RemediationService for every anomaly event.
     *
     * @param nodeId      the affected node
     * @param score       combined anomaly score [0,1]
     * @param failureMode failure mode string from Python layer
     * @param cpuPct      current CPU %
     * @param latencyP99  current p99 latency ms
     * @param errorRate   current error rate
     * @param memPct      current memory %
     * @param priorCount  how many times this node has been remediated before
     * @return Decision with action, reason, and priority
     */
    public Decision decide(
            String nodeId,
            double score,
            String failureMode,
            double cpuPct,
            double latencyP99,
            double errorRate,
            double memPct,
            long priorCount) {
        log.info("DecisionEngine: node={} score={} mode={} cpu={} lat={} err={}",
                nodeId, score, failureMode, cpuPct, latencyP99, errorRate);

        // Circuit breaker — too many auto-remediations, escalate to human
        if (priorCount >= MAX_AUTO_REMEDIATIONS) {
            return new Decision(
                    "escalate",
                    String.format("Node %s has been remediated %d times — escalating to human", nodeId, priorCount),
                    1);
        }

        // Rule 1: Silent death — node stopped emitting
        if ("silent_death".equals(failureMode)) {
            return new Decision("isolate_node",
                    "Node appears dead — isolating and draining traffic", 1);
        }

        // Rule 2: High error rate — restart is the fastest fix
        if (errorRate > HIGH_ERROR && score > HIGH_SCORE) {
            return new Decision("restart_process",
                    String.format("Error rate %.3f exceeds threshold — restarting process", errorRate), 2);
        }

        // Rule 3: Cascading failure — reroute first to protect users
        if ("cascading".equals(failureMode) && score > MODERATE_SCORE) {
            return new Decision("reroute_traffic",
                    "Cascading failure detected — rerouting traffic to healthy nodes", 2);
        }

        // Rule 4: CPU spike — reroute to buy time
        if (cpuPct > HIGH_CPU && score > MODERATE_SCORE) {
            return new Decision("reroute_traffic",
                    String.format("CPU %.1f%% critical — rerouting traffic", cpuPct), 3);
        }

        // Rule 5: Memory leak — only a restart fixes this
        if ("memory_leak".equals(failureMode) && memPct > HIGH_MEM) {
            return new Decision("restart_process",
                    String.format("Memory %.1f%% — restarting to clear leak", memPct), 3);
        }

        // Rule 6: Latency blowout — reroute
        if (latencyP99 > HIGH_LATENCY && score > MODERATE_SCORE) {
            return new Decision("reroute_traffic",
                    String.format("Latency p99 %.1fms — rerouting traffic", latencyP99), 3);
        }

        // Rule 7: Disk saturation — needs human
        if ("disk_saturation".equals(failureMode)) {
            return new Decision("escalate",
                    "Disk saturation requires manual intervention", 4);
        }

        // Rule 8: High combined score with no specific mode — safe default
        if (score > HIGH_SCORE) {
            return new Decision("restart_process",
                    String.format("High anomaly score %.3f — restarting as safe default", score), 5);
        }

        // Rule 9: Moderate score — alert only, watch and wait
        if (score > MODERATE_SCORE) {
            return new Decision("alert",
                    String.format("Moderate anomaly score %.3f — alerting, no action yet", score), 6);
        }

        // Below threshold — no action
        return new Decision("none",
                String.format("Score %.3f below threshold — no action required", score), 99);
    }
}