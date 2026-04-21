package com.s3fleet.service;

import com.s3fleet.engine.DecisionEngine;
import com.s3fleet.model.AnomalyEvent;
import com.s3fleet.repository.AnomalyEventRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.time.Instant;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * RemediationService: orchestrates the full remediation lifecycle.
 *
 * 1. Receives anomaly payload from AnomalyController
 * 2. Asks DecisionEngine what action to take
 * 3. Persists the decision to the database BEFORE acting
 * (so control plane crash = recoverable state)
 * 4. Executes the action
 * 5. Updates the persisted record with outcome
 *
 * Also implements a per-node rate limiter (circuit breaker) —
 * max N remediations per node before escalating to human.
 * This prevents the automation itself from causing an outage.
 */
@Service
public class RemediationService {

    private static final Logger log = LoggerFactory.getLogger(RemediationService.class);
    private static final int REMEDIATION_RATE_LIMIT_PER_MIN = 2;

    private final DecisionEngine decisionEngine;
    private final AnomalyEventRepository repository;

    // Per-node last remediation timestamp for rate limiting
    private final Map<String, Instant> lastRemediation = new ConcurrentHashMap<>();

    public RemediationService(DecisionEngine decisionEngine,
            AnomalyEventRepository repository) {
        this.decisionEngine = decisionEngine;
        this.repository = repository;
    }

    @Transactional
    public AnomalyEvent process(
            String nodeId,
            double score,
            double ifScore,
            double lstmScore,
            String failureMode,
            double cpuPct,
            double latencyP99,
            double errorRate,
            double memPct) {
        log.info("Processing anomaly: node={} score={} mode={}", nodeId, score, failureMode);

        // How many times has this node been remediated before?
        long priorCount = repository.countAppliedByNode(nodeId);

        // Ask the decision engine
        DecisionEngine.Decision decision = decisionEngine.decide(
                nodeId, score, failureMode,
                cpuPct, latencyP99, errorRate, memPct,
                priorCount);

        log.info("Decision for {}: action={} reason={}", nodeId, decision.action(), decision.reason());

        // Persist BEFORE acting — crash-safe
        AnomalyEvent event = new AnomalyEvent(
                nodeId, score, ifScore, lstmScore, failureMode,
                cpuPct, latencyP99, errorRate, memPct,
                decision.action());
        event = repository.save(event);

        // Skip if no action needed
        if ("none".equals(decision.action())) {
            event.setStatus("SKIPPED");
            return repository.save(event);
        }

        // Rate limit check
        if (isRateLimited(nodeId)) {
            log.warn("Rate limited: {} — skipping action {}", nodeId, decision.action());
            event.setStatus("RATE_LIMITED");
            return repository.save(event);
        }

        // Execute the action
        try {
            executeAction(nodeId, decision.action(), score);
            event.setStatus("APPLIED");
            event.setResolvedAt(Instant.now());
            lastRemediation.put(nodeId, Instant.now());
            log.info("Remediation applied: {} → {}", nodeId, decision.action());
        } catch (Exception e) {
            log.error("Remediation failed for {}: {}", nodeId, e.getMessage());
            event.setStatus("FAILED");
        }

        return repository.save(event);
    }

    private void executeAction(String nodeId, String action, double score) {
        switch (action) {
            case "restart_process" -> {
                log.warn("ACTION restart_process on {} (score={})", nodeId, score);
                // In production: SSH to node or call node agent API
                // For demo: logged and persisted — Python layer polls and acts
            }
            case "reroute_traffic" -> {
                log.warn("ACTION reroute_traffic from {} (score={})", nodeId, score);
                // In production: update load balancer target group
                // For demo: logged and persisted
            }
            case "isolate_node" -> {
                log.warn("ACTION isolate_node {} (score={})", nodeId, score);
                // In production: remove from service registry, drain connections
            }
            case "escalate" -> {
                log.warn("ACTION escalate for {} — human intervention required", nodeId);
                // In production: PagerDuty / Slack alert
            }
            case "alert" -> {
                log.info("ACTION alert for {} (score={})", nodeId, score);
                // Datadog event already fired by Python ML layer
            }
            default -> log.warn("Unknown action: {}", action);
        }
    }

    private boolean isRateLimited(String nodeId) {
        Instant last = lastRemediation.get(nodeId);
        if (last == null)
            return false;
        long secondsSince = Instant.now().getEpochSecond() - last.getEpochSecond();
        return secondsSince < (60 / REMEDIATION_RATE_LIMIT_PER_MIN);
    }

    public List<AnomalyEvent> getRecentEvents() {
        return repository.findTop50ByOrderByReceivedAtDesc();
    }

    public List<AnomalyEvent> getEventsByNode(String nodeId) {
        return repository.findByNodeIdOrderByReceivedAtDesc(nodeId);
    }

    public List<AnomalyEvent> getPendingEvents() {
        return repository.findPendingEvents();
    }
}