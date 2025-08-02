# Incident Response Runbook

Comprehensive incident response procedures for photon-mlir-bridge system outages and critical issues.

## Emergency Response Overview

### Incident Severity Levels

| Level | Description | Response Time | Examples |
|-------|-------------|---------------|----------|
| **P0 - Critical** | Complete system outage, data loss risk | 15 minutes | Service completely down, hardware failure |
| **P1 - High** | Major feature broken, significant impact | 1 hour | Compilation failures, device offline |
| **P2 - Medium** | Minor feature broken, workaround exists | 4 hours | Performance degradation, cache issues |
| **P3 - Low** | Cosmetic issues, minimal impact | Next business day | UI glitches, documentation errors |

### Incident Commander Responsibilities

- Coordinate response efforts
- Communicate with stakeholders
- Make critical decisions
- Ensure proper documentation
- Conduct post-incident review

## P0 Critical Incident Response

### Initial Response (0-15 minutes)

1. **Acknowledge Alert**
   ```bash
   # Acknowledge PagerDuty alert
   pd ack --incident-key <incident-key>
   ```

2. **Form Response Team**
   - Incident Commander (IC)
   - Subject Matter Expert (SME) 
   - Communications Lead
   - Executive Sponsor (if needed)

3. **Create Incident Channel**
   ```bash
   # Create Slack incident channel
   /slack create-channel incident-$(date +%Y%m%d-%H%M)
   /slack invite @photon-team @ops-team
   ```

4. **Initial Assessment**
   ```bash
   # Quick system status check
   kubectl get pods -n photon-mlir --show-labels
   kubectl get services -n photon-mlir
   curl -f https://photon-compiler.example.com/health || echo "Service DOWN"
   ```

### Investigation Phase (15-30 minutes)

1. **Gather Information**
   ```bash
   # Check recent deployments
   kubectl rollout history deployment/photon-compiler -n photon-mlir
   
   # Check resource usage
   kubectl top pods -n photon-mlir
   kubectl describe nodes | grep -A5 "Non-terminated Pods"
   
   # Check logs
   kubectl logs -n photon-mlir -l app=photon-compiler --tail=50
   kubectl logs -n photon-mlir -l app=photon-compiler --previous
   ```

2. **Check Monitoring**
   - Review Grafana dashboards
   - Check Prometheus alerts
   - Analyze error patterns

3. **Identify Root Cause**
   - Recent changes/deployments
   - Infrastructure issues
   - Hardware failures
   - External dependencies

### Mitigation Actions

#### Service Recovery
```bash
# Emergency service restart
kubectl rollout restart deployment/photon-compiler -n photon-mlir

# Scale up for redundancy
kubectl scale deployment photon-compiler --replicas=5 -n photon-mlir

# Rollback to previous version if needed
kubectl rollout undo deployment/photon-compiler -n photon-mlir
```

#### Hardware Issues
```bash
# Switch to backup hardware
photon-config set-primary-device backup-device-001

# Disable failed device
photon-admin disable-device failed-device-id

# Route traffic to healthy instances
kubectl patch service photon-compiler -p '{"spec":{"selector":{"health":"ready"}}}'
```

#### Database Recovery
```bash
# Switch to read replica if master fails
kubectl patch configmap db-config -p '{"data":{"db_host":"photon-db-replica.example.com"}}'

# Restart dependent services
kubectl rollout restart deployment/photon-compiler -n photon-mlir
```

### Communication Templates

#### Initial Notification
```
🚨 INCIDENT: PhotonMLIR P0 - Service Outage

Status: INVESTIGATING
Started: $(date -u)
Impact: Complete service unavailability
ETA: TBD

IC: @your-name
SME: @sme-name

Updates every 15 minutes in #incident-channel
Status page: https://status.photon-mlir.example.com
```

#### Progress Updates
```
📊 INCIDENT UPDATE - $(date -u)

Status: INVESTIGATING → IDENTIFIED
Root Cause: Kubernetes cluster resource exhaustion
Actions Taken:
- Scaled cluster nodes from 3 to 6
- Redeployed photon-compiler with resource limits
- Verified service health

Current Status: Service restored, monitoring for stability
Next Update: 15 minutes
```

#### Resolution Notification
```
✅ INCIDENT RESOLVED - $(date -u)

Duration: X hours Y minutes
Root Cause: Brief description
Resolution: Brief description

Impact: Detailed impact description
Prevention: Steps taken to prevent recurrence

Post-mortem scheduled for: Date/Time
Document: Link to incident report
```

## P1 High Priority Response

### Response Actions (1 hour)

1. **Quick Triage**
   ```bash
   # Check service health
   photon-health-check --verbose
   
   # Check error rates
   curl -s 'http://prometheus:9090/api/v1/query?query=rate(photon_compilation_errors_total[5m])'
   
   # Check affected components
   kubectl get pods -n photon-mlir -o wide | grep -v Running
   ```

2. **Containment**
   - Isolate affected components
   - Implement workarounds
   - Scale unaffected services

3. **Resolution**
   - Apply targeted fixes
   - Monitor recovery metrics
   - Validate full functionality

### Common P1 Scenarios

#### High Error Rate
```bash
# Check error distribution
kubectl logs -n photon-mlir -l app=photon-compiler | grep ERROR | tail -20

# Identify problematic models
curl -s 'http://prometheus:9090/api/v1/query?query=rate(photon_compilation_errors_total[5m]) by (model_type)'

# Implement error handling
kubectl patch configmap photon-config -p '{"data":{"error_threshold":"10"}}'
```

#### Device Performance Issues
```bash
# Check device metrics
curl -s 'http://prometheus:9090/api/v1/query?query=photon_device_temperature_celsius' | jq .

# Reset thermal calibration
photon-admin recalibrate-thermal --device-id all

# Adjust power limits
photon-admin set-power-limit --device-id <id> --limit 80
```

## Recovery Procedures

### Automated Recovery

```bash
#!/bin/bash
# auto-recovery.sh - Automated recovery script

NAMESPACE="photon-mlir"
SERVICE="photon-compiler"

# Function to check service health
check_health() {
    kubectl get pods -n $NAMESPACE -l app=$SERVICE | grep Running | wc -l
}

# Function to restart unhealthy pods
restart_unhealthy() {
    kubectl get pods -n $NAMESPACE -l app=$SERVICE --field-selector=status.phase!=Running -o name | \
        xargs -I {} kubectl delete {} -n $NAMESPACE
}

# Function to scale service
scale_service() {
    local replicas=$1
    kubectl scale deployment $SERVICE --replicas=$replicas -n $NAMESPACE
}

# Main recovery logic
echo "Starting automated recovery..."

HEALTHY_PODS=$(check_health)
DESIRED_REPLICAS=3

if [ $HEALTHY_PODS -lt $DESIRED_REPLICAS ]; then
    echo "Only $HEALTHY_PODS/$DESIRED_REPLICAS pods healthy. Initiating recovery..."
    
    # Restart unhealthy pods
    restart_unhealthy
    
    # Wait for pods to start
    sleep 30
    
    # Check again
    HEALTHY_PODS=$(check_health)
    
    if [ $HEALTHY_PODS -lt $DESIRED_REPLICAS ]; then
        echo "Still unhealthy. Scaling up..."
        scale_service 5
        
        # Wait for scale up
        sleep 60
        
        # Scale back down
        scale_service $DESIRED_REPLICAS
    fi
fi

echo "Recovery complete. Healthy pods: $(check_health)"
```

### Manual Recovery Steps

#### Service Won't Start
```bash
# 1. Check resource constraints
kubectl describe pod <pod-name> -n photon-mlir

# 2. Check node capacity
kubectl describe nodes | grep -A5 "Allocated resources"

# 3. Clear resource constraints temporarily
kubectl patch deployment photon-compiler -p '{"spec":{"template":{"spec":{"containers":[{"name":"compiler","resources":{}}]}}}}'

# 4. Force pod scheduling
kubectl patch deployment photon-compiler -p '{"spec":{"template":{"spec":{"nodeSelector":{}}}}}'
```

#### Database Connection Issues
```bash
# 1. Test connectivity
kubectl exec -it deployment/photon-compiler -- nc -zv postgres-service 5432

# 2. Check credentials
kubectl get secret db-credentials -o yaml

# 3. Reset connection pool
kubectl exec -it deployment/photon-compiler -- photon-admin reset-db-pool

# 4. Restart with fresh connections
kubectl rollout restart deployment/photon-compiler
```

#### Hardware Recovery
```bash
# 1. Check device status
photon-admin list-devices --status

# 2. Attempt soft reset
photon-admin reset-device --device-id <id> --soft

# 3. Hard reset if necessary
photon-admin reset-device --device-id <id> --hard

# 4. Re-run calibration
photon-admin calibrate --device-id <id> --full

# 5. Verify functionality
photon-admin test-device --device-id <id> --test-suite basic
```

## Post-Incident Procedures

### Immediate Actions (Within 1 hour of resolution)

1. **Update Status Page**
   ```bash
   # Mark incident as resolved
   statuspage-cli update --incident-id <id> --status resolved
   ```

2. **Notify Stakeholders**
   - Send resolution notification
   - Update incident channel
   - Inform customer support

3. **Collect Evidence**
   ```bash
   # Save logs
   kubectl logs -n photon-mlir -l app=photon-compiler --since=2h > incident-logs.txt
   
   # Export metrics
   curl 'http://prometheus:9090/api/v1/query_range?query=up&start=...' > incident-metrics.json
   
   # Take system snapshot
   kubectl get all -n photon-mlir -o yaml > system-state.yaml
   ```

### Post-Mortem Process

#### Timeline Creation
```bash
# Generate incident timeline
echo "Incident Timeline - $(date)" > post-mortem.md
echo "======================" >> post-mortem.md
echo "" >> post-mortem.md

# Extract key events from logs
kubectl logs -n photon-mlir -l app=photon-compiler --since=3h --timestamps | \
    grep -E "(ERROR|FATAL|Started|Stopped)" >> post-mortem.md
```

#### Root Cause Analysis

1. **Technical Root Cause**
   - What exactly failed?
   - Why did it fail?
   - Why wasn't it detected earlier?

2. **Process Root Cause**
   - Were procedures followed?
   - Were tools adequate?
   - Was communication effective?

3. **Contributing Factors**
   - Recent changes
   - Environmental factors
   - Human factors

#### Action Items Template

```markdown
## Action Items

### Immediate (Complete within 24 hours)
- [ ] Fix identified bug/issue
- [ ] Update monitoring alerts
- [ ] Document workaround

### Short-term (Complete within 1 week)
- [ ] Implement better error handling
- [ ] Add automated recovery
- [ ] Update runbooks

### Long-term (Complete within 1 month)
- [ ] Architectural improvements
- [ ] Process improvements
- [ ] Training updates

### Responsible Parties
- Technical fixes: @engineering-team
- Process improvements: @ops-team
- Documentation: @docs-team
```

## Learning and Improvement

### Incident Metrics

Track these metrics for continuous improvement:

- Mean Time To Detection (MTTD)
- Mean Time To Resolution (MTTR)
- Incident frequency by category
- Customer impact duration
- Repeat incident rate

### Review Process

#### Weekly Incident Review
- Review all incidents from past week
- Identify patterns and trends
- Update procedures and documentation
- Share learnings with team

#### Monthly Deep Dive
- Analyze incident metrics
- Review effectiveness of changes
- Plan improvements for next month
- Update training materials

### Prevention Strategies

1. **Improved Monitoring**
   - Add missing alerts
   - Reduce false positives
   - Implement predictive alerts

2. **Automation**
   - Automate common fixes
   - Implement circuit breakers
   - Add self-healing capabilities

3. **Testing**
   - Chaos engineering
   - Disaster recovery drills
   - Load testing

4. **Documentation**
   - Keep runbooks updated
   - Document all procedures
   - Create decision trees

This incident response runbook ensures rapid, coordinated response to system issues and continuous improvement of our incident management capabilities.