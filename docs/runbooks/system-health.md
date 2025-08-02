# System Health Runbook

Operational procedures for maintaining photon-mlir-bridge system health and responding to common incidents.

## Quick Reference

### Emergency Contacts

- **On-call Engineer**: +1-555-0123 (PagerDuty)
- **Hardware Team**: hardware-team@example.com
- **Infrastructure Team**: infra-team@example.com
- **Escalation Manager**: manager@example.com

### Critical System URLs

- **Grafana Dashboard**: https://grafana.photon-mlir.example.com
- **Prometheus**: https://prometheus.photon-mlir.example.com
- **Alertmanager**: https://alertmanager.photon-mlir.example.com
- **Kubernetes Dashboard**: https://k8s.photon-mlir.example.com

## Alert Response Procedures

### 1. Compiler Service Down

**Alert**: `PhotonCompilerDown`

**Symptoms**:
- HTTP 503 errors from compilation endpoint
- Zero compilation rate metrics
- Health check failures

**Immediate Actions**:
```bash
# 1. Check service status
kubectl get pods -n photon-mlir -l app=photon-compiler

# 2. Check recent logs
kubectl logs -n photon-mlir -l app=photon-compiler --tail=100

# 3. Check resource usage
kubectl top pods -n photon-mlir -l app=photon-compiler

# 4. Restart failed pods if necessary
kubectl delete pod -n photon-mlir <failing-pod-name>
```

**Root Cause Investigation**:
- Check for OOM kills: `kubectl describe pod <pod-name>`
- Review resource limits and requests
- Check for dependency failures (MLIR libraries, device drivers)
- Verify configuration changes

**Resolution**:
- Scale deployment if load-related: `kubectl scale deployment photon-compiler --replicas=5`
- Update resource limits if needed
- Rollback recent deployments if problematic

### 2. High Error Rate

**Alert**: `PhotonHighErrorRate`

**Symptoms**:
- Compilation error rate > 5%
- User complaints about failed compilations
- Spike in error logs

**Immediate Actions**:
```bash
# 1. Check error distribution by target
curl -s 'http://prometheus:9090/api/v1/query?query=rate(photon_compilations_total{status="error"}[5m]) by (target)'

# 2. Check recent error logs
kubectl logs -n photon-mlir -l app=photon-compiler | grep -i error | tail -20

# 3. Check model types causing errors
curl -s 'http://prometheus:9090/api/v1/query?query=rate(photon_compilations_total{status="error"}[5m]) by (model_type)'
```

**Common Causes & Solutions**:

| Error Pattern | Likely Cause | Resolution |
|--------------|--------------|------------|
| MLIR dialect errors | Invalid model format | Update model validation |
| Timeout errors | Large model compilation | Increase timeout limits |
| Memory errors | Insufficient resources | Scale resources |
| Hardware errors | Device unavailable | Check device connectivity |

**Escalation**: If error rate doesn't decrease within 15 minutes, escalate to hardware team.

### 3. Photonic Device Failure

**Alert**: `PhotonDeviceOffline`

**Symptoms**:
- Device status = 0 in metrics
- Hardware compilation targets failing
- Temperature/power anomalies

**Immediate Actions**:
```bash
# 1. Check device status
photon-debug --list-devices

# 2. Attempt device reconnection
photon-debug --reconnect-device <device-id>

# 3. Check device logs
tail -f /var/log/photon-mlir/device-<device-id>.log

# 4. Verify physical connections
ping <device-ip-address>
```

**Device-Specific Procedures**:

#### Lightmatter Envise
```bash
# Check Lightmatter SDK status
lightmatter-cli status

# Reset device if unresponsive
lightmatter-cli reset --device <device-id>

# Verify calibration
lightmatter-cli calibrate --device <device-id> --verify
```

#### MIT Silicon Photonics
```bash
# Check FPGA connection
fpga-util list

# Reset photonic array
mit-photonic-cli reset --array <array-id>

# Run diagnostic
mit-photonic-cli diagnose --verbose
```

**Escalation**: Hardware failures require immediate escalation to hardware vendor.

### 4. Memory/CPU Issues

**Alert**: `PhotonHighResourceUsage`

**Symptoms**:
- Memory usage > 80%
- CPU usage sustained > 90%
- Slow compilation times

**Immediate Actions**:
```bash
# 1. Check resource usage
kubectl top pods -n photon-mlir --sort-by=memory
kubectl top pods -n photon-mlir --sort-by=cpu

# 2. Check for resource leaks
kubectl exec -it <pod-name> -- ps aux | head -20

# 3. Check compilation queue
curl -s 'http://prometheus:9090/api/v1/query?query=photon_queue_depth'
```

**Mitigation**:
```bash
# Temporary scale up
kubectl scale deployment photon-compiler --replicas=8

# Restart high-memory pods
kubectl delete pod -n photon-mlir <high-memory-pod>

# Enable resource limits if not set
kubectl patch deployment photon-compiler -p '{"spec":{"template":{"spec":{"containers":[{"name":"compiler","resources":{"limits":{"memory":"4Gi","cpu":"2"}}}]}}}}'
```

### 5. Cache Performance Issues

**Alert**: `PhotonLowCacheHitRate`

**Symptoms**:
- Cache hit rate < 60%
- Increased compilation times
- High CPU usage

**Investigation**:
```bash
# 1. Check cache statistics
curl -s 'http://prometheus:9090/api/v1/query?query=photon_cache_hit_rate'

# 2. Check cache size and usage
du -sh /var/cache/photon-mlir/
df -h /var/cache/photon-mlir/

# 3. Check cache configuration
cat /etc/photon-mlir/config.yaml | grep -A5 cache
```

**Resolution**:
```bash
# Clear corrupted cache entries
find /var/cache/photon-mlir -type f -mtime +7 -delete

# Increase cache size if storage allows
kubectl patch configmap photon-mlir-config -p '{"data":{"cache_size_gb":"20"}}'

# Restart services to pick up new configuration
kubectl rollout restart deployment photon-compiler
```

## Health Check Procedures

### Daily Health Checks

```bash
#!/bin/bash
# daily-health-check.sh

echo "=== PhotonMLIR Daily Health Check ==="
echo "Date: $(date)"

# 1. Service Status
echo -e "\n1. Service Status:"
kubectl get pods -n photon-mlir -o wide

# 2. Compilation Metrics (last 24h)
echo -e "\n2. Compilation Statistics:"
COMPILATIONS=$(curl -s 'http://prometheus:9090/api/v1/query?query=increase(photon_compilations_total[24h])' | jq '.data.result[0].value[1]' -r)
ERRORS=$(curl -s 'http://prometheus:9090/api/v1/query?query=increase(photon_compilations_total{status="error"}[24h])' | jq '.data.result[0].value[1]' -r)

echo "Total Compilations: $COMPILATIONS"
echo "Errors: $ERRORS"
echo "Error Rate: $(echo "scale=2; $ERRORS/$COMPILATIONS*100" | bc)%"

# 3. Device Status
echo -e "\n3. Device Status:"
curl -s 'http://prometheus:9090/api/v1/query?query=photon_device_status' | jq '.data.result[] | {device: .metric.device_id, status: .value[1]}'

# 4. Resource Usage
echo -e "\n4. Resource Usage:"
kubectl top pods -n photon-mlir --no-headers | awk '{print $1 ": CPU=" $2 " Memory=" $3}'

# 5. Storage Usage
echo -e "\n5. Storage Usage:"
kubectl exec -n photon-mlir deployment/photon-compiler -- df -h | grep -E "(cache|tmp)"

echo -e "\n=== Health Check Complete ===\n"
```

### Weekly Deep Health Check

```bash
#!/bin/bash
# weekly-deep-check.sh

echo "=== PhotonMLIR Weekly Deep Health Check ==="

# 1. Performance Trends (7 days)
echo -e "\n1. Performance Trends:"
curl -s 'http://prometheus:9090/api/v1/query_range?query=photon:compilation_rate_5m&start='$(date -d '7 days ago' +%s)'&end='$(date +%s)'&step=3600' | \
  jq '.data.result[0].values[] | .[1]' | awk '{sum+=$1; count++} END {print "Avg Compilation Rate: " sum/count " ops/s"}'

# 2. Error Analysis
echo -e "\n2. Top Error Categories:"
curl -s 'http://prometheus:9090/api/v1/query?query=topk(5, sum by (error_type) (increase(photon_compilation_errors_total[7d])))' | \
  jq '.data.result[] | {error_type: .metric.error_type, count: .value[1]}'

# 3. Device Health History
echo -e "\n3. Device Uptime (7 days):"
curl -s 'http://prometheus:9090/api/v1/query?query=avg_over_time(photon_device_status[7d]) by (device_id)' | \
  jq '.data.result[] | {device: .metric.device_id, uptime_pct: (.value[1] | tonumber * 100)}'

# 4. Cache Efficiency
echo -e "\n4. Cache Performance:"
CACHE_HITS=$(curl -s 'http://prometheus:9090/api/v1/query?query=increase(photon_cache_hits_total[7d])' | jq '.data.result[0].value[1]' -r)
CACHE_MISSES=$(curl -s 'http://prometheus:9090/api/v1/query?query=increase(photon_cache_misses_total[7d])' | jq '.data.result[0].value[1]' -r)
echo "Cache Hit Rate: $(echo "scale=2; $CACHE_HITS/($CACHE_HITS+$CACHE_MISSES)*100" | bc)%"

# 5. Security Status
echo -e "\n5. Security Status:"
# Check for security updates
kubectl get pods -n photon-mlir -o jsonpath='{.items[*].spec.containers[*].image}' | tr ' ' '\n' | sort -u | while read image; do
  echo "Checking $image for vulnerabilities..."
  # This would integrate with your vulnerability scanner
done
```

## Maintenance Procedures

### Routine Maintenance

#### Weekly Tasks
- [ ] Review and acknowledge resolved alerts
- [ ] Check disk usage and clean old logs
- [ ] Verify backup integrity
- [ ] Update security patches (if needed)
- [ ] Review performance trends

#### Monthly Tasks
- [ ] Update monitoring dashboards
- [ ] Review and update alert thresholds
- [ ] Test disaster recovery procedures
- [ ] Update documentation
- [ ] Performance optimization review

#### Quarterly Tasks
- [ ] Full security audit
- [ ] Capacity planning review  
- [ ] Hardware maintenance coordination
- [ ] Update monitoring stack versions
- [ ] Review and update runbooks

### Planned Maintenance Windows

#### Standard Maintenance Window
**Schedule**: Sundays 2:00-4:00 AM UTC
**Duration**: 2 hours
**Notification**: 48 hours advance notice

```bash
# Pre-maintenance checklist
- [ ] Notify users via status page
- [ ] Scale down non-critical services
- [ ] Backup current configuration
- [ ] Prepare rollback plan
- [ ] Coordinate with hardware vendors

# During maintenance
- [ ] Apply updates systematically
- [ ] Test each component after update
- [ ] Monitor metrics for anomalies
- [ ] Document any issues

# Post-maintenance
- [ ] Verify all services operational
- [ ] Update status page
- [ ] Send completion notification
- [ ] Update change log
```

## Escalation Procedures

### Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|------------------|
| P0 (Critical) | 15 minutes | On-call → Manager → Director |
| P1 (High) | 1 hour | On-call → Team Lead |
| P2 (Medium) | 4 hours | On-call → Team |
| P3 (Low) | Next business day | Team |

### Incident Communication

```bash
# Incident communication template
Subject: [P0 INCIDENT] PhotonMLIR Compiler Outage

Status: INVESTIGATING/IDENTIFIED/MONITORING/RESOLVED
Start Time: YYYY-MM-DD HH:MM UTC
Impact: Description of user impact
Root Cause: TBD/Description

Current Actions:
- Action 1
- Action 2

Next Update: In X minutes

---
For real-time updates: https://status.photon-mlir.example.com
Incident Commander: Name <email>
```

## Recovery Procedures

### Service Recovery

```bash
# Quick service recovery script
#!/bin/bash
SERVICE_NAME=${1:-"photon-compiler"}
NAMESPACE=${2:-"photon-mlir"}

echo "Initiating recovery for $SERVICE_NAME in $NAMESPACE"

# 1. Check current state
kubectl get deployment $SERVICE_NAME -n $NAMESPACE

# 2. Scale to 0 and back up
kubectl scale deployment $SERVICE_NAME --replicas=0 -n $NAMESPACE
sleep 10
kubectl scale deployment $SERVICE_NAME --replicas=3 -n $NAMESPACE

# 3. Wait for readiness
kubectl wait --for=condition=available deployment/$SERVICE_NAME -n $NAMESPACE --timeout=300s

# 4. Verify health
kubectl get pods -n $NAMESPACE -l app=$SERVICE_NAME
```

### Data Recovery

```bash
# Cache recovery from backup
#!/bin/bash
BACKUP_DATE=${1:-$(date -d yesterday +%Y%m%d)}

echo "Recovering cache from backup: $BACKUP_DATE"

# 1. Stop services using cache
kubectl scale deployment photon-compiler --replicas=0

# 2. Restore from backup
aws s3 sync s3://photon-mlir-backups/$BACKUP_DATE/cache/ /var/cache/photon-mlir/

# 3. Set proper permissions
chown -R photon:photon /var/cache/photon-mlir/

# 4. Restart services
kubectl scale deployment photon-compiler --replicas=3

echo "Cache recovery complete"
```

This runbook provides comprehensive procedures for maintaining system health and responding to incidents efficiently.