# THERMAL-AWARE QUANTUM OPTIMIZATION DEPLOYMENT GUIDE

## 🎯 PRODUCTION DEPLOYMENT OVERVIEW

This guide provides comprehensive instructions for deploying the **thermal-aware quantum optimization** system for silicon photonic neural network compilation in production environments.

---

## 📋 SYSTEM REQUIREMENTS

### **Hardware Requirements**
- **CPU**: 8+ cores recommended (16+ for large-scale optimization)
- **Memory**: 16GB+ RAM (32GB+ for complex workloads)
- **Storage**: 10GB+ free space for caching and logging
- **Network**: Low latency for distributed optimization

### **Software Requirements**
- **Python**: 3.9+ (3.11 recommended)
- **Operating System**: Linux (Ubuntu 20.04+), macOS 12+, Windows 11
- **Dependencies**: See `pyproject.toml` for complete list

### **Optional Hardware**
- **Silicon Photonic Accelerators**: Lightmatter Envise, MIT Photonic Processor, Custom Research Chips
- **GPU**: For ML-based thermal modeling (CUDA 11.0+)

---

## 🔧 INSTALLATION & SETUP

### **1. Standard Installation**
```bash
# Clone repository
git clone https://github.com/yourusername/photon-mlir-bridge.git
cd photon-mlir-bridge

# Install dependencies
pip install -e ".[dev,thermal]"

# Verify installation
python -c "import photon_mlir; print('Installation successful')"
```

### **2. Container Deployment**
```bash
# Build thermal-optimized container
docker build -t photon-mlir:thermal -f Dockerfile.thermal .

# Run with thermal optimization enabled
docker run -d --name photon-thermal \
  -e PHOTON_ENABLE_THERMAL=true \
  -e PHOTON_THERMAL_MODEL=arrhenius_based \
  -e PHOTON_COOLING_STRATEGY=adaptive \
  photon-mlir:thermal
```

### **3. Kubernetes Deployment**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photon-thermal-scheduler
spec:
  replicas: 3
  selector:
    matchLabels:
      app: photon-thermal
  template:
    metadata:
      labels:
        app: photon-thermal
    spec:
      containers:
      - name: thermal-scheduler
        image: photon-mlir:thermal
        env:
        - name: PHOTON_ENABLE_THERMAL
          value: "true"
        - name: PHOTON_MAX_WORKERS
          value: "8"
        - name: PHOTON_CACHE_STRATEGY
          value: "hybrid"
        resources:
          requests:
            cpu: "4"
            memory: "8Gi"
          limits:
            cpu: "8" 
            memory: "16Gi"
```

---

## ⚙️ CONFIGURATION

### **Basic Configuration**
```python
from photon_mlir import (
    RobustThermalScheduler, ThermalModel, CoolingStrategy,
    ThermalConstraints, PhotonicDevice
)

# Configure thermal constraints
constraints = ThermalConstraints(
    max_device_temperature=85.0,  # °C
    max_thermal_gradient=5.0,     # °C/mm
    max_power_density=100.0,      # mW/mm²
    thermal_time_constant=10.0    # ms
)

# Configure photonic device
device = PhotonicDevice(
    device_id="production_chip",
    device_type="mzi_mesh",
    area_mm2=25.0,
    thermal_resistance=50.0,      # K/W
    num_phase_shifters=64,
    wavelength_channels=8,
    max_optical_power=100.0       # mW
)

# Initialize robust scheduler
scheduler = RobustThermalScheduler(
    thermal_model=ThermalModel.ARRHENIUS_BASED,
    cooling_strategy=CoolingStrategy.ADAPTIVE,
    constraints=constraints,
    device=device,
    max_retries=3,
    timeout_seconds=600.0,
    enable_monitoring=True,
    enable_circuit_breaker=True
)
```

### **Advanced Configuration**
```python
# Environment-based configuration
import os
from photon_mlir import GlobalConfig

config = GlobalConfig()
config.thermal_model = os.getenv('PHOTON_THERMAL_MODEL', 'arrhenius_based')
config.cooling_strategy = os.getenv('PHOTON_COOLING_STRATEGY', 'adaptive')
config.max_workers = int(os.getenv('PHOTON_MAX_WORKERS', '8'))
config.cache_strategy = os.getenv('PHOTON_CACHE_STRATEGY', 'hybrid')
config.enable_monitoring = os.getenv('PHOTON_ENABLE_MONITORING', 'true').lower() == 'true'
config.log_level = os.getenv('PHOTON_LOG_LEVEL', 'INFO')

# Apply configuration
photon_mlir.configure(config)
```

---

## 🎯 USAGE PATTERNS

### **1. Basic Thermal-Aware Scheduling**
```python
from photon_mlir import (
    QuantumTaskPlanner, RobustThermalScheduler, ValidationLevel
)

# Create compilation plan
planner = QuantumTaskPlanner()
tasks = planner.create_compilation_plan({
    "model_type": "neural_network",
    "layers": 12,
    "precision": "fp16",
    "target_device": "lightmatter_envise"
})

# Initialize scheduler
scheduler = RobustThermalScheduler()

# Perform thermal-aware scheduling
result = scheduler.schedule_tasks_robust(
    tasks, 
    validation_level=ValidationLevel.STRICT
)

# Access results
print(f"Optimized makespan: {result.makespan:.2f}s")
print(f"Thermal efficiency: {result.thermal_efficiency:.2%}")
print(f"Max device temperature: {result.max_device_temperature:.1f}°C")
```

### **2. Comparative Benchmarking**
```python
from photon_mlir import ThermalAwareBenchmark

# Initialize benchmark suite
benchmark = ThermalAwareBenchmark()

# Create diverse test workloads
workloads = [
    create_workload("resnet50"),
    create_workload("bert_base"), 
    create_workload("gpt2_small")
]

# Run comparative study
results = benchmark.run_comparative_study(
    task_sets=workloads,
    iterations=10  # For statistical significance
)

# Analyze results
if results["research_contribution"]["statistical_validation"]:
    print("✅ Statistically significant improvements achieved")
    
for metric, data in results["detailed_comparison"].items():
    improvement = data["improvement_percent"]
    print(f"{metric}: {improvement:+.1f}% improvement")
```

### **3. Production Monitoring**
```python
import time
from photon_mlir import RobustThermalScheduler

scheduler = RobustThermalScheduler(enable_monitoring=True)

# Add health monitoring callback
def health_alert_handler(metrics):
    if metrics.health_score < 0.7:
        print(f"⚠️  Health Alert: Score {metrics.health_score:.2f}")
        # Integrate with alerting system (PagerDuty, Slack, etc.)

scheduler.health_monitor.add_alert_callback(health_alert_handler)

# Production scheduling loop
while True:
    try:
        tasks = get_next_compilation_tasks()  # Your task source
        result = scheduler.schedule_tasks_robust(tasks)
        
        # Process result
        deploy_optimized_schedule(result)
        
        # Get system status
        status = scheduler.get_system_status()
        log_performance_metrics(status)
        
    except Exception as e:
        logger.error(f"Scheduling failed: {e}")
    
    time.sleep(60)  # 1 minute intervals
```

---

## 📊 MONITORING & OBSERVABILITY

### **Key Metrics to Monitor**

#### **Performance Metrics**
- **Makespan**: Total compilation time
- **Resource Utilization**: CPU, memory, GPU usage efficiency  
- **Throughput**: Tasks processed per hour
- **Success Rate**: Percentage of successful optimizations

#### **Thermal Metrics**
- **Peak Device Temperature**: Maximum temperature reached
- **Thermal Efficiency**: How well thermal constraints are maintained
- **Thermal Hotspots**: Number of temperature violations
- **Phase Stability**: Thermal impact on phase shifters

#### **System Health Metrics**
- **Error Rate**: Errors per hour
- **Circuit Breaker State**: Open/closed/half-open
- **Cache Hit Rate**: Optimization caching effectiveness
- **Process Success Rate**: Distributed optimization success

### **Prometheus Integration**
```yaml
# prometheus-config.yml
global:
  scrape_interval: 30s

scrape_configs:
  - job_name: 'photon-thermal-scheduler'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### **Grafana Dashboard Queries**
```promql
# Thermal efficiency trend
avg_over_time(photon_thermal_efficiency[5m])

# Error rate
rate(photon_errors_total[1m])

# Performance trend  
avg_over_time(photon_makespan_seconds[10m])

# Resource utilization
avg_over_time(photon_resource_utilization[5m])
```

---

## 🚨 ALERTING & INCIDENT RESPONSE

### **Critical Alerts**
- **Thermal Violations**: Device temperature > 80°C
- **High Error Rate**: > 5% errors in 10 minutes
- **Circuit Breaker Open**: Optimization failures
- **Low Cache Hit Rate**: < 30% cache effectiveness

### **Alert Configuration**
```yaml
# alertmanager-rules.yml
groups:
- name: photon_thermal_alerts
  rules:
  - alert: ThermalViolation
    expr: photon_max_device_temperature > 80
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "Device temperature critical: {{ $value }}°C"
      
  - alert: HighErrorRate
    expr: rate(photon_errors_total[10m]) > 0.05
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High error rate: {{ $value | humanizePercentage }}"
      
  - alert: CircuitBreakerOpen
    expr: photon_circuit_breaker_state == 2  # open = 2
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Circuit breaker open - optimization failing"
```

### **Incident Response Playbook**

#### **Thermal Violations**
1. **Immediate**: Reduce workload or enable aggressive cooling
2. **Short-term**: Analyze thermal timeline for hotspots
3. **Long-term**: Optimize task distribution or upgrade cooling

#### **High Error Rate**
1. **Immediate**: Check system resources and dependencies
2. **Short-term**: Review error logs and recovery strategies
3. **Long-term**: Investigate root cause and improve error handling

#### **Performance Degradation**
1. **Immediate**: Check cache performance and resource utilization
2. **Short-term**: Analyze workload complexity and auto-scaling
3. **Long-term**: Optimize algorithms or scale infrastructure

---

## 🔧 PERFORMANCE TUNING

### **Optimization Levels**
```python
from photon_mlir import OptimizationLevel

# For development/testing
scheduler_fast = RobustThermalScheduler(
    optimization_level=OptimizationLevel.FAST
)

# For production balance
scheduler_balanced = RobustThermalScheduler(
    optimization_level=OptimizationLevel.BALANCED
)

# For research/quality focus
scheduler_quality = RobustThermalScheduler(
    optimization_level=OptimizationLevel.QUALITY
)
```

### **Caching Strategies**
```python
from photon_mlir import CacheStrategy

# Memory-only for speed
cache_memory = QuantumCache(CacheStrategy.MEMORY_ONLY, max_memory_entries=2000)

# Hybrid for balance  
cache_hybrid = QuantumCache(CacheStrategy.HYBRID, max_memory_entries=1000, max_disk_size_mb=1000)

# Distributed for scale
cache_distributed = QuantumCache(CacheStrategy.DISTRIBUTED)
```

### **Auto-Scaling Tuning**
```python
# Get auto-scaling recommendations
scheduler = RobustThermalScheduler()

# After running several optimizations
stats = scheduler.get_optimization_stats()
recommendations = stats["auto_scaling_recommendations"]

if "suggested_max_workers" in recommendations:
    new_workers = recommendations["suggested_max_workers"]
    scheduler.max_workers = new_workers
    print(f"Auto-scaled to {new_workers} workers")
```

---

## 🔒 SECURITY & COMPLIANCE

### **Security Features**
- **Input Validation**: Comprehensive task and parameter validation
- **Resource Limits**: Configurable limits to prevent resource exhaustion
- **Circuit Breaker**: Protection against cascading failures
- **Audit Logging**: Complete operation audit trail

### **Compliance Considerations**
- **Data Privacy**: No model data stored in optimization cache
- **Resource Isolation**: Process-based isolation for multi-tenancy
- **Monitoring**: Complete observability for compliance auditing
- **Error Handling**: Graceful degradation without data loss

---

## 📈 SCALING PATTERNS

### **Horizontal Scaling**
```python
# Multi-instance deployment with load balancing
class DistributedThermalScheduler:
    def __init__(self, instances: List[str]):
        self.instances = [
            RobustThermalScheduler(endpoint=instance) 
            for instance in instances
        ]
        self.load_balancer = RoundRobinBalancer()
    
    def schedule_distributed(self, tasks):
        instance = self.load_balancer.get_next_instance()
        return instance.schedule_tasks_robust(tasks)
```

### **Vertical Scaling**
```python
# Resource-aware scaling
import psutil

def get_optimal_workers():
    cpu_count = psutil.cpu_count()
    memory_gb = psutil.virtual_memory().total // (1024**3)
    
    # Scale based on available resources
    if memory_gb >= 64:
        return min(cpu_count, 16)  # High memory: more workers
    elif memory_gb >= 32:
        return min(cpu_count, 12)  # Medium memory
    else:
        return min(cpu_count, 8)   # Conservative scaling

scheduler = RobustThermalScheduler(max_workers=get_optimal_workers())
```

---

## 🧪 TESTING & VALIDATION

### **Unit Testing**
```bash
# Run comprehensive test suite
pytest tests/unit/python/test_thermal_optimization.py -v

# Run with coverage
pytest tests/unit/python/test_thermal_optimization.py --cov=photon_mlir --cov-report=html
```

### **Integration Testing**
```bash
# End-to-end thermal optimization test
pytest tests/integration/end_to_end/test_thermal_integration.py -v

# Performance benchmark tests
pytest tests/benchmarks/performance/test_thermal_benchmarks.py -v
```

### **Production Validation**
```python
# Health check endpoint
def health_check():
    scheduler = RobustThermalScheduler()
    status = scheduler.get_system_status()
    
    if status["performance"]["success_rate"] > 0.95:
        return {"status": "healthy", "details": status}
    else:
        return {"status": "degraded", "details": status}
```

---

## 📚 TROUBLESHOOTING

### **Common Issues**

#### **High Memory Usage**
```python
# Solution: Adjust cache settings
cache = QuantumCache(
    strategy=CacheStrategy.DISK_ONLY,  # Use disk instead of memory
    max_memory_entries=500,            # Reduce memory cache
    max_disk_size_mb=2000             # Increase disk cache
)
```

#### **Slow Optimization**
```python
# Solution: Reduce complexity or increase workers
scheduler = RobustThermalScheduler(
    optimization_level=OptimizationLevel.FAST,  # Faster optimization
    max_workers=get_optimal_workers(),          # More parallelism
    timeout_seconds=300                         # Shorter timeout
)
```

#### **Thermal Violations**
```python
# Solution: Stricter thermal constraints
constraints = ThermalConstraints(
    max_device_temperature=75.0,  # Lower limit
    max_thermal_gradient=3.0,     # Stricter gradient
    thermal_time_constant=15.0    # More conservative
)
```

### **Debug Mode**
```python
import logging
logging.basicConfig(level=logging.DEBUG)

scheduler = RobustThermalScheduler(enable_monitoring=True)
# Enable verbose logging for troubleshooting
```

---

## 🚀 DEPLOYMENT CHECKLIST

### **Pre-Deployment**
- [ ] Dependencies installed and verified
- [ ] Configuration validated
- [ ] Test suite passes (unit + integration)
- [ ] Performance benchmarks meet requirements
- [ ] Monitoring/alerting configured
- [ ] Security review completed

### **Deployment**
- [ ] Rolling deployment with health checks
- [ ] Database migrations (if applicable)
- [ ] Cache warming completed
- [ ] Load balancer configuration updated
- [ ] Monitoring dashboards active

### **Post-Deployment**
- [ ] Health checks passing
- [ ] Performance metrics within expected ranges
- [ ] Error rates below thresholds
- [ ] User acceptance testing completed
- [ ] Documentation updated
- [ ] Team training completed

---

## 📞 SUPPORT & MAINTENANCE

### **Maintenance Tasks**
- **Daily**: Monitor health metrics and error rates
- **Weekly**: Review performance trends and optimization effectiveness
- **Monthly**: Update dependencies and security patches
- **Quarterly**: Performance optimization and capacity planning

### **Support Contacts**
- **Technical Issues**: [engineering-team@yourcompany.com]
- **Performance Issues**: [performance-team@yourcompany.com]
- **Research Questions**: [research-team@yourcompany.com]

### **Documentation**
- **API Documentation**: `/docs/api/thermal_optimization.html`
- **Research Papers**: `/docs/research/quantum_thermal_optimization.pdf`
- **Troubleshooting Guide**: `/docs/troubleshooting/thermal_scheduler.md`

---

This deployment guide provides comprehensive instructions for successfully deploying and operating the thermal-aware quantum optimization system in production environments, ensuring both performance and reliability for silicon photonic neural network compilation.