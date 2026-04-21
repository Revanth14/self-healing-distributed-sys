package com.s3fleet.repository;

import com.s3fleet.model.AnomalyEvent;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AnomalyEventRepository extends JpaRepository<AnomalyEvent, Long> {

    List<AnomalyEvent> findByNodeIdOrderByReceivedAtDesc(String nodeId);

    List<AnomalyEvent> findByStatusOrderByReceivedAtDesc(String status);

    List<AnomalyEvent> findTop50ByOrderByReceivedAtDesc();

    @Query("SELECT e FROM AnomalyEvent e WHERE e.status = 'PENDING' ORDER BY e.receivedAt ASC")
    List<AnomalyEvent> findPendingEvents();

    @Query("SELECT COUNT(e) FROM AnomalyEvent e WHERE e.nodeId = :nodeId AND e.status = 'APPLIED'")
    Long countAppliedByNode(String nodeId);
}