# Monitoring and Observability

Comprehensive monitoring and observability setup for photon-mlir-bridge, providing visibility into compiler performance, hardware status, and system health.

## Overview

The monitoring stack includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and management
- **Loki**: Log aggregation (optional)
- **Jaeger**: Distributed tracing (optional)

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │───▶│   Prometheus    │───▶│     Grafana     │
│                 │    │                 │    │                 │
│ • Compiler      │    │ • Metrics       │    │ • Dashboards    │
│ • Runtime       │    │ • Rules         │    │ • Alerts        │
│ • Devices       │    │ • Storage       │    │ • Visualization │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │              ┌─────────────────┐              │
         │              │  Alertmanager   │              │
         └──────────────┤                 │──────────────┘
                        │ • Routing       │
                        │ • Grouping      │
                        │ • Silencing     │
                        └─────────────────┘
```

## Key Metrics

### Compiler Metrics

- `photon_compilations_total`: Total number of model compilations
- `photon_compilation_duration_seconds`: Time spent compiling models
- `photon_compilation_errors_total`: Number of compilation failures
- `photon_cache_hits_total`: Compilation cache hit count
- `photon_memory_usage_bytes`: Memory usage during compilation

### Hardware Metrics

- `photon_device_status`: Status of photonic devices (0=down, 1=up)
- `photon_optical_power_mw`: Current optical power consumption
- `photon_device_temperature_celsius`: Device temperature
- `photon_calibration_drift`: Calibration drift measurements
- `photon_inference_latency_seconds`: Hardware inference latency

### Runtime Metrics

- `photon_active_sessions`: Number of active inference sessions
- `photon_throughput_ops_per_second`: Inference throughput
- `photon_queue_depth`: Inference request queue depth
- `photon_runtime_errors_total`: Runtime error count

## Setup Instructions

### 1. Local Development Setup

```bash
# Start monitoring stack with Docker Compose
docker-compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d

# Access Grafana at http://localhost:3000
# Default credentials: admin/admin
```

### 2. Production Deployment

#### Kubernetes Deployment

```bash
# Deploy monitoring namespace
kubectl create namespace monitoring

# Deploy Prometheus
kubectl apply -f monitoring/k8s/prometheus/

# Deploy Grafana
kubectl apply -f monitoring/k8s/grafana/

# Deploy Alertmanager
kubectl apply -f monitoring/k8s/alertmanager/
```

#### Configure Data Sources

```bash
# Apply Grafana datasource configuration
kubectl apply -f monitoring/grafana/datasources/
```

### 3. Dashboard Import

Import provided dashboards:

- **System Overview**: `monitoring/grafana/dashboards/photon-mlir-overview.json`
- **Hardware Dashboard**: `monitoring/grafana/dashboards/hardware-metrics.json`
- **Performance Dashboard**: `monitoring/grafana/dashboards/performance-metrics.json`

## Alerting

### Alert Rules

Key alerts are configured for:

- **Compiler Down**: Compilation service unavailable
- **High Error Rate**: Compilation error rate > 5%
- **Device Failure**: Photonic device offline
- **High Memory Usage**: Memory usage > 80%
- **Thermal Issues**: Device temperature > threshold

### Alert Channels

Configure notification channels:

```yaml
# Slack notifications
slack:
  api_url: 'https://hooks.slack.com/services/...'
  channel: '#photon-mlir-alerts'
  title: 'PhotonMLIR Alert'

# PagerDuty for critical alerts
pagerduty:
  routing_key: 'YOUR_PAGERDUTY_KEY'
  severity: 'critical'

# Email notifications
email:
  to: 'ops-team@example.com'
  subject: 'PhotonMLIR Alert: {{ .GroupLabels.alertname }}'
```

## Custom Metrics

### Adding Application Metrics

```python
# Example: Adding custom metrics in Python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
compilation_counter = Counter(
    'photon_compilations_total',
    'Total number of model compilations',
    ['target', 'status', 'model_type']
)

compilation_duration = Histogram(
    'photon_compilation_duration_seconds',
    'Time spent compiling models',
    ['target', 'model_size'],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0]
)

active_devices = Gauge(
    'photon_active_devices',
    'Number of active photonic devices',
    ['device_type', 'location']
)

# Start metrics server
start_http_server(8080)

# Use metrics in code
def compile_model(model, target):
    start_time = time.time()
    try:
        result = compile_model_impl(model, target)
        compilation_counter.labels(
            target=target, 
            status='success',
            model_type=model.type
        ).inc()
    except Exception as e:
        compilation_counter.labels(
            target=target,
            status='error', 
            model_type=model.type
        ).inc()
        raise
    finally:
        compilation_duration.labels(
            target=target,
            model_size='large' if model.size > 100_000 else 'small'
        ).observe(time.time() - start_time)
```

### C++ Metrics Integration

```cpp
// Example: C++ metrics using prometheus-cpp
#include <prometheus/counter.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

class PhotonMetrics {
private:
    std::shared_ptr<prometheus::Registry> registry_;
    prometheus::Family<prometheus::Counter>* compilation_counter_;
    prometheus::Family<prometheus::Histogram>* compilation_duration_;

public:
    PhotonMetrics() {
        registry_ = std::make_shared<prometheus::Registry>();
        
        compilation_counter_ = &prometheus::BuildCounter()
            .Name("photon_compilations_total")
            .Help("Total number of model compilations")
            .Register(*registry_);
            
        compilation_duration_ = &prometheus::BuildHistogram()
            .Name("photon_compilation_duration_seconds")
            .Help("Time spent compiling models")
            .Register(*registry_);
    }
    
    void recordCompilation(const std::string& target, bool success) {
        compilation_counter_->Add({{"target", target}, 
                                  {"status", success ? "success" : "error"}})
                          .Increment();
    }
    
    void recordDuration(const std::string& target, double duration) {
        compilation_duration_->Add({{"target", target}})
                             .Observe(duration);
    }
};
```

## Log Management

### Structured Logging

```python
import structlog
import logging

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage in application
logger.info("Compilation started", 
           model_id="resnet50",
           target="lightmatter_envise",
           user_id="user123")

logger.error("Compilation failed",
            model_id="resnet50", 
            target="lightmatter_envise",
            error="Invalid model format",
            duration_ms=1234)
```

### Log Aggregation with Loki

```yaml
# promtail configuration for log collection
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: photon-mlir-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: photon-mlir
          __path__: /var/log/photon-mlir/*.log
    
    pipeline_stages:
      - json:
          expressions:
            timestamp: timestamp
            level: level
            logger: logger
            message: message
            model_id: model_id
            target: target
      
      - timestamp:
          source: timestamp
          format: RFC3339
      
      - labels:
          level:
          logger:
          model_id:
          target:
```

## Performance Optimization

### Query Optimization

```promql
# Efficient queries for high-cardinality metrics

# Instead of: rate(photon_compilations_total[5m])
# Use: rate(photon_compilations_total{instance=~"compiler-.*"}[5m])

# Use recording rules for complex queries
groups:
  - name: photon_mlir_recording_rules
    rules:
      - record: photon:compilation_rate_5m
        expr: rate(photon_compilations_total[5m])
      
      - record: photon:error_rate_5m
        expr: |
          rate(photon_compilations_total{status="error"}[5m]) /
          rate(photon_compilations_total[5m])
```

### Resource Management

```yaml
# Prometheus resource limits
resources:
  requests:
    memory: 2Gi
    cpu: 1
  limits:
    memory: 8Gi
    cpu: 4

# Retention configuration
prometheus.yml:
  storage.tsdb.retention.time: 30d
  storage.tsdb.retention.size: 100GB
```

## Troubleshooting

### Common Issues

1. **High Memory Usage**
   - Reduce metric cardinality
   - Implement proper retention policies
   - Use recording rules for complex queries

2. **Missing Metrics**
   - Check service discovery configuration
   - Verify network connectivity
   - Review Prometheus targets

3. **Dashboard Not Loading** 
   - Verify datasource configuration
   - Check Grafana logs
   - Validate dashboard JSON

### Debug Commands

```bash
# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Query specific metrics
curl 'http://prometheus:9090/api/v1/query?query=photon_compilations_total'

# Check Grafana health
curl http://grafana:3000/api/health

# View logs
kubectl logs -f deployment/prometheus -n monitoring
kubectl logs -f deployment/grafana -n monitoring
```

This comprehensive monitoring setup provides full visibility into the photon-mlir-bridge system performance and health.