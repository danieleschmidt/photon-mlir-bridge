# Deployment Guide

Comprehensive deployment guide for photon-mlir-bridge in various environments.

## Overview

This guide covers deployment strategies for the photon-mlir-bridge compiler and runtime across different environments, from development to production photonic hardware deployments.

## Deployment Environments

### 1. Development Environment

#### Local Development Setup

```bash
# Clone repository
git clone --recursive https://github.com/terragon/photon-mlir-bridge.git
cd photon-mlir-bridge

# Setup development environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[dev,test,docs]"

# Install pre-commit hooks
pre-commit install

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DMLIR_DIR=/usr/local/lib/cmake/mlir
make -j$(nproc)

# Run tests to verify setup
pytest tests/
ctest --verbose
```

#### Development Container

```bash
# Using provided devcontainer
code .  # VSCode will prompt to reopen in container

# Or using Docker directly
docker-compose -f docker-compose.dev.yml up -d
docker-compose exec photon-dev bash
```

### 2. CI/CD Environment

#### GitHub Actions Deployment

The repository includes comprehensive CI/CD workflows:

- **Continuous Integration**: Automated testing and building
- **Security Scanning**: Daily vulnerability assessments  
- **Performance Monitoring**: Regression detection
- **Documentation**: Automated doc generation and deployment

#### Required Secrets

Configure these secrets in GitHub repository settings:

```bash
# PyPI publishing
PYPI_API_TOKEN=pypi-xxxxxxxxxxxx

# Container registry (optional)
DOCKER_USERNAME=your-username
DOCKER_PASSWORD=your-password

# Code coverage (optional)
CODECOV_TOKEN=xxxxxxxxxxxx
```

### 3. Production Environment

#### System Requirements

**Minimum Requirements:**
- CPU: 4 cores, 2.4GHz
- Memory: 8GB RAM
- Storage: 20GB available space
- OS: Ubuntu 20.04+ or equivalent

**Recommended for Large Models:**
- CPU: 16+ cores, 3.2GHz+
- Memory: 32GB+ RAM
- Storage: 100GB+ SSD
- GPU: Optional, for hybrid photonic-electronic workflows

#### Production Installation

```bash
# Install from PyPI
pip install photon-mlir

# Or install from wheel
pip install photon_mlir-0.1.0-py3-none-any.whl

# Verify installation
photon-compile --version
python -c "import photon_mlir; print('Installation successful')"
```

#### Configuration Management

```yaml
# /etc/photon-mlir/config.yaml
compiler:
  default_target: "lightmatter_envise"
  optimization_level: 3
  cache_directory: "/var/cache/photon-mlir"
  
logging:
  level: "INFO"
  file: "/var/log/photon-mlir/compiler.log"
  max_size_mb: 100
  backup_count: 5

hardware:
  discovery_timeout: 30
  connection_retry: 3
  calibration_interval: 3600  # seconds

security:
  enable_signing: true
  trusted_keys_path: "/etc/photon-mlir/trusted-keys.gpg"
```

## Container Deployment

### Docker Image

```dockerfile
# Dockerfile (production optimized)
FROM ubuntu:22.04 as builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake ninja-build \
    python3 python3-pip python3-dev \
    libmlir-17-dev llvm-17-dev

COPY . /src
WORKDIR /src

# Build C++ components
RUN mkdir build && cd build && \
    cmake .. -GNinja -DCMAKE_BUILD_TYPE=Release && \
    ninja

# Build Python wheel
RUN pip install build && python -m build

FROM ubuntu:22.04 as runtime

# Runtime dependencies only
RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    libmlir-17 llvm-17 \
    && rm -rf /var/lib/apt/lists/*

# Install photon-mlir
COPY --from=builder /src/dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm /tmp/*.whl

# Create non-root user
RUN useradd -m -u 1000 photon && \
    mkdir -p /var/cache/photon-mlir && \
    chown photon:photon /var/cache/photon-mlir

USER photon
WORKDIR /home/photon

ENTRYPOINT ["photon-compile"]
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: photon-mlir-compiler
  labels:
    app: photon-mlir
spec:
  replicas: 3
  selector:
    matchLabels:
      app: photon-mlir
  template:
    metadata:
      labels:
        app: photon-mlir
    spec:
      containers:
      - name: compiler
        image: ghcr.io/terragon/photon-mlir:latest
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi" 
            cpu: "4"
        ports:
        - containerPort: 8080
        env:
        - name: PHOTON_CONFIG_PATH
          value: "/etc/photon-mlir/config.yaml"
        volumeMounts:
        - name: config
          mountPath: /etc/photon-mlir
        - name: cache
          mountPath: /var/cache/photon-mlir
      volumes:
      - name: config
        configMap:
          name: photon-mlir-config
      - name: cache
        emptyDir:
          sizeLimit: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: photon-mlir-service
spec:
  selector:
    app: photon-mlir
  ports:
  - port: 80
    targetPort: 8080
  type: LoadBalancer
```

## Hardware Integration

### Photonic Device Drivers

#### Lightmatter Envise Integration

```bash
# Install Lightmatter SDK
wget https://releases.lightmatter.co/envise-sdk-v2.1.tar.gz
tar xzf envise-sdk-v2.1.tar.gz
cd envise-sdk && ./install.sh

# Configure photon-mlir for Lightmatter
export LIGHTMATTER_SDK_PATH=/opt/lightmatter
export LD_LIBRARY_PATH=$LIGHTMATTER_SDK_PATH/lib:$LD_LIBRARY_PATH

# Test hardware connection
photon-debug --test-hardware lightmatter_envise
```

#### Custom Hardware Integration

```cpp
// Example: Custom photonic device driver
#include "photon/hardware/DeviceDriver.h"

class CustomPhotonicDevice : public photon::DeviceDriver {
public:
  bool initialize() override {
    // Initialize custom hardware
    return device_->connect();
  }
  
  bool upload(const CompiledModel& model) override {
    // Upload compiled photonic circuit
    return device_->loadCircuit(model.getPhotonicAssembly());
  }
  
  Tensor infer(const Tensor& input) override {
    // Execute inference on photonic hardware
    return device_->execute(input);
  }
};

REGISTER_DEVICE_DRIVER("custom_photonic", CustomPhotonicDevice);
```

### Hardware Monitoring

```yaml
# monitoring/photonic-device-monitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: photonic-devices
spec:
  selector:
    matchLabels:
      app: photon-mlir-hardware
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

## Monitoring and Observability

### Prometheus Metrics

```python
# Export key performance metrics
from prometheus_client import Counter, Histogram, Gauge

compilation_counter = Counter('photon_compilations_total',
                            'Total number of model compilations',
                            ['target', 'status'])

compilation_duration = Histogram('photon_compilation_duration_seconds',
                                'Time spent compiling models',
                                ['target', 'model_size'])

active_devices = Gauge('photon_active_devices',
                      'Number of active photonic devices',
                      ['device_type'])

photonic_power = Gauge('photon_optical_power_mw',
                      'Current optical power consumption',
                      ['device_id'])
```

### Logging Configuration

```yaml
# logging-config.yaml
version: 1
formatters:
  detailed:
    format: '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
  json:
    format: '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

handlers:
  console:
    class: logging.StreamHandler
    level: INFO
    formatter: detailed
    stream: ext://sys.stdout
    
  file:
    class: logging.handlers.RotatingFileHandler
    level: DEBUG
    formatter: json
    filename: /var/log/photon-mlir/compiler.log
    maxBytes: 104857600  # 100MB
    backupCount: 5

loggers:
  photon_mlir:
    level: DEBUG
    handlers: [console, file]
    propagate: false
    
  photon_mlir.hardware:
    level: INFO
    handlers: [console, file]
    propagate: false

root:
  level: WARNING
  handlers: [console]
```

### Health Checks

```python
# health_check.py
from fastapi import FastAPI
from photon_mlir.hardware import DeviceManager

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        # Check compiler health
        from photon_mlir import compile_model
        test_model = create_test_model()
        compile_model(test_model)
        
        # Check hardware connectivity
        device_manager = DeviceManager()
        devices = device_manager.list_devices()
        
        return {
            "status": "healthy",
            "compiler": "operational",
            "devices": len(devices),
            "timestamp": datetime.utcnow()
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "error": str(e),
            "timestamp": datetime.utcnow()
        }

@app.get("/metrics")
async def metrics():
    # Prometheus metrics endpoint
    from prometheus_client import generate_latest
    return Response(generate_latest(), media_type="text/plain")
```

## Scaling and Performance

### Horizontal Scaling

```yaml
# autoscaling.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: photon-mlir-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: photon-mlir-compiler
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Load Balancing

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: photon-mlir-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/load-balance: "round_robin"
    nginx.ingress.kubernetes.io/upstream-hash-by: "$request_uri"
spec:
  rules:
  - host: photon-compiler.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: photon-mlir-service
            port:
              number: 80
```

## Security Considerations

### Network Security

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: photon-mlir-netpol
spec:
  podSelector:
    matchLabels:
      app: photon-mlir
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: photon-clients
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS
    - protocol: TCP  
      port: 53   # DNS
    - protocol: UDP
      port: 53   # DNS
```

### Secrets Management

```bash
# Using Kubernetes secrets
kubectl create secret generic photon-mlir-secrets \
  --from-literal=hardware-key="your-hardware-key" \
  --from-literal=api-token="your-api-token"

# Using external secret management
kubectl apply -f - <<EOF
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: photon-mlir-external-secret
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: photon-mlir-secrets
  data:
  - secretKey: hardware-key
    remoteRef:
      key: photon-mlir/hardware-key
  - secretKey: api-token
    remoteRef:
      key: photon-mlir/api-token
EOF
```

## Backup and Recovery

### Data Backup Strategy

```bash
# Backup compiled models and cache
#!/bin/bash
BACKUP_DIR="/backup/photon-mlir/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup compiled models
tar czf "$BACKUP_DIR/compiled-models.tar.gz" /var/cache/photon-mlir/models/

# Backup configuration
cp -r /etc/photon-mlir "$BACKUP_DIR/config"

# Upload to cloud storage
aws s3 sync "$BACKUP_DIR" s3://photon-mlir-backups/$(date +%Y%m%d)/
```

### Disaster Recovery

```yaml
# disaster-recovery-plan.yaml
recovery_procedures:
  hardware_failure:
    - Switch to backup photonic devices
    - Redirect traffic to healthy nodes
    - Notify hardware vendor
    
  software_corruption:
    - Rollback to previous container image
    - Restore from backup
    - Validate system functionality
    
  data_loss:
    - Restore from cloud backup
    - Rebuild cache from source models
    - Verify data integrity

rto: 30 minutes  # Recovery Time Objective
rpo: 1 hour      # Recovery Point Objective
```

This comprehensive deployment guide ensures reliable, secure, and scalable deployment of photon-mlir-bridge across all environments.