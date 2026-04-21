package com.s3fleet;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * S3 Fleet Management Control Plane
 *
 * Spring Boot REST service that receives anomaly scores from the Python ML
 * layer,
 * decides which remediation action to take, persists every decision to
 * H2/PostgreSQL,
 * and exposes a status API for the observability dashboard.
 *
 * Runs on port 8080 by default.
 * The Python scorer POSTs anomaly events here via HTTP.
 */
@SpringBootApplication
public class ControlPlaneApplication {
    public static void main(String[] args) {
        SpringApplication.run(ControlPlaneApplication.class, args);
    }
}