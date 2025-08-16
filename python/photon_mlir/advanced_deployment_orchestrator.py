"""
Advanced Deployment Orchestrator for Photonic Computing
Generation 2: Kubernetes-native deployment with auto-scaling and blue-green deployments
"""

import os
import time
import json
import yaml
import subprocess
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import tempfile
import shutil
import uuid
from datetime import datetime, timedelta

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from .logging_config import get_global_logger
from .validation import ValidationResult
from .enterprise_monitoring_system import create_monitoring_system


class DeploymentStrategy(Enum):
    """Deployment strategies."""
    ROLLING_UPDATE = "rolling_update"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    RECREATE = "recreate"
    A_B_TEST = "a_b_test"


class DeploymentStatus(Enum):
    """Deployment status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    TERMINATED = "terminated"


class EnvironmentType(Enum):
    """Environment types."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"
    CANARY = "canary"


@dataclass
class DeploymentConfig:
    """Configuration for deployment."""
    name: str
    version: str
    image: str
    strategy: DeploymentStrategy
    environment: EnvironmentType
    replicas: int = 3
    resource_requests: Dict[str, str] = None
    resource_limits: Dict[str, str] = None
    env_vars: Dict[str, str] = None
    labels: Dict[str, str] = None
    annotations: Dict[str, str] = None
    health_check_path: str = "/health"
    readiness_probe_path: str = "/ready"
    port: int = 8080
    namespace: str = "default"
    
    def __post_init__(self):
        if self.resource_requests is None:
            self.resource_requests = {"cpu": "100m", "memory": "256Mi"}
        if self.resource_limits is None:
            self.resource_limits = {"cpu": "500m", "memory": "1Gi"}
        if self.env_vars is None:
            self.env_vars = {}
        if self.labels is None:
            self.labels = {}
        if self.annotations is None:
            self.annotations = {}
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DeploymentResult:
    """Result of a deployment operation."""
    deployment_id: str
    status: DeploymentStatus
    message: str
    start_time: float
    end_time: Optional[float] = None
    rollback_version: Optional[str] = None
    health_checks_passed: bool = False
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def duration_seconds(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class KubernetesDeployer:
    """Kubernetes-native deployment manager."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.kubeconfig_path = kubeconfig_path or os.environ.get('KUBECONFIG')
        self.logger = get_global_logger()
        self.kubectl_cmd = self._find_kubectl()
        self.deployments = {}
        self._validate_kubernetes_connection()
    
    def _find_kubectl(self) -> str:
        """Find kubectl command."""
        kubectl_paths = ['kubectl', '/usr/local/bin/kubectl', '/usr/bin/kubectl']
        
        for path in kubectl_paths:
            try:
                result = subprocess.run([path, 'version', '--client'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.logger.info(f"Found kubectl at: {path}")
                    return path
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        
        self.logger.warning("kubectl not found, some features may not work")
        return 'kubectl'  # Fallback
    
    def _validate_kubernetes_connection(self):
        """Validate connection to Kubernetes cluster."""
        try:
            result = self._run_kubectl(['cluster-info'], timeout=10)
            if result.returncode == 0:
                self.logger.info("Successfully connected to Kubernetes cluster")
            else:
                self.logger.warning(f"Failed to connect to Kubernetes: {result.stderr}")
        except Exception as e:
            self.logger.warning(f"Could not validate Kubernetes connection: {e}")
    
    def _run_kubectl(self, args: List[str], timeout: int = 60, 
                    input_data: str = None) -> subprocess.CompletedProcess:
        """Run kubectl command with proper error handling."""
        cmd = [self.kubectl_cmd] + args
        
        if self.kubeconfig_path:
            cmd.extend(['--kubeconfig', self.kubeconfig_path])
        
        self.logger.debug(f"Running kubectl command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                input=input_data
            )
            
            if result.stdout:
                self.logger.debug(f"kubectl stdout: {result.stdout}")
            if result.stderr and result.returncode != 0:
                self.logger.error(f"kubectl stderr: {result.stderr}")
            
            return result
            
        except subprocess.TimeoutExpired as e:
            self.logger.error(f"kubectl command timed out after {timeout}s: {' '.join(cmd)}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to run kubectl command: {e}")
            raise
    
    def generate_kubernetes_manifests(self, config: DeploymentConfig) -> Dict[str, str]:
        """Generate Kubernetes manifests for deployment."""
        manifests = {}
        
        # Deployment manifest
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': config.name,
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'version': config.version,
                    'environment': config.environment.value,
                    **config.labels
                },
                'annotations': {
                    'deployment.kubernetes.io/revision': '1',
                    'photonic.mlir/deployed-at': datetime.utcnow().isoformat(),
                    **config.annotations
                }
            },
            'spec': {
                'replicas': config.replicas,
                'strategy': self._get_deployment_strategy(config.strategy),
                'selector': {
                    'matchLabels': {
                        'app': config.name,
                        'version': config.version
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': config.name,
                            'version': config.version,
                            'environment': config.environment.value,
                            **config.labels
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': config.name,
                            'image': config.image,
                            'ports': [{
                                'containerPort': config.port,
                                'name': 'http'
                            }],
                            'env': [{'name': k, 'value': v} for k, v in config.env_vars.items()],
                            'resources': {
                                'requests': config.resource_requests,
                                'limits': config.resource_limits
                            },
                            'livenessProbe': {
                                'httpGet': {
                                    'path': config.health_check_path,
                                    'port': 'http'
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': config.readiness_probe_path,
                                    'port': 'http'
                                },
                                'initialDelaySeconds': 5,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'failureThreshold': 3
                            }
                        }],
                        'restartPolicy': 'Always'
                    }
                }
            }
        }
        
        manifests['deployment'] = yaml.dump(deployment_manifest, default_flow_style=False)
        
        # Service manifest
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': f"{config.name}-service",
                'namespace': config.namespace,
                'labels': {
                    'app': config.name,
                    'version': config.version,
                    **config.labels
                }
            },
            'spec': {
                'selector': {
                    'app': config.name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': config.port,
                    'protocol': 'TCP',
                    'name': 'http'
                }],
                'type': 'ClusterIP'
            }
        }
        
        manifests['service'] = yaml.dump(service_manifest, default_flow_style=False)
        
        # HorizontalPodAutoscaler manifest
        if config.environment in [EnvironmentType.PRODUCTION, EnvironmentType.STAGING]:
            hpa_manifest = {
                'apiVersion': 'autoscaling/v2',
                'kind': 'HorizontalPodAutoscaler',
                'metadata': {
                    'name': f"{config.name}-hpa",
                    'namespace': config.namespace
                },
                'spec': {
                    'scaleTargetRef': {
                        'apiVersion': 'apps/v1',
                        'kind': 'Deployment',
                        'name': config.name
                    },
                    'minReplicas': max(1, config.replicas // 2),
                    'maxReplicas': config.replicas * 3,
                    'metrics': [
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'cpu',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 70
                                }
                            }
                        },
                        {
                            'type': 'Resource',
                            'resource': {
                                'name': 'memory',
                                'target': {
                                    'type': 'Utilization',
                                    'averageUtilization': 80
                                }
                            }
                        }
                    ]
                }
            }
            
            manifests['hpa'] = yaml.dump(hpa_manifest, default_flow_style=False)
        
        return manifests
    
    def _get_deployment_strategy(self, strategy: DeploymentStrategy) -> Dict[str, Any]:
        """Get Kubernetes deployment strategy configuration."""
        if strategy == DeploymentStrategy.ROLLING_UPDATE:
            return {
                'type': 'RollingUpdate',
                'rollingUpdate': {
                    'maxUnavailable': '25%',
                    'maxSurge': '25%'
                }
            }
        elif strategy == DeploymentStrategy.RECREATE:
            return {'type': 'Recreate'}
        else:
            # Default to rolling update
            return {
                'type': 'RollingUpdate',
                'rollingUpdate': {
                    'maxUnavailable': '25%',
                    'maxSurge': '25%'
                }
            }
    
    def deploy(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy application using specified strategy."""
        deployment_id = str(uuid.uuid4())
        start_time = time.time()
        
        self.logger.info(f"Starting deployment {deployment_id} for {config.name} v{config.version}")
        
        try:
            # Generate manifests
            manifests = self.generate_kubernetes_manifests(config)
            
            # Create namespace if it doesn't exist
            self._ensure_namespace(config.namespace)
            
            # Apply deployment strategy
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                result = self._deploy_blue_green(config, manifests, deployment_id, start_time)
            elif config.strategy == DeploymentStrategy.CANARY:
                result = self._deploy_canary(config, manifests, deployment_id, start_time)
            else:
                result = self._deploy_standard(config, manifests, deployment_id, start_time)
            
            self.deployments[deployment_id] = result
            return result
            
        except Exception as e:
            error_msg = f"Deployment failed: {str(e)}"
            self.logger.error(error_msg)
            
            result = DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.FAILED,
                message=error_msg,
                start_time=start_time,
                end_time=time.time()
            )
            
            self.deployments[deployment_id] = result
            return result
    
    def _ensure_namespace(self, namespace: str):
        """Ensure namespace exists."""
        if namespace == 'default':
            return
        
        # Check if namespace exists
        result = self._run_kubectl(['get', 'namespace', namespace], timeout=10)
        
        if result.returncode != 0:
            # Create namespace
            namespace_manifest = {
                'apiVersion': 'v1',
                'kind': 'Namespace',
                'metadata': {
                    'name': namespace,
                    'labels': {
                        'created-by': 'photonic-deployer'
                    }
                }
            }
            
            manifest_yaml = yaml.dump(namespace_manifest, default_flow_style=False)
            result = self._run_kubectl(['apply', '-f', '-'], input_data=manifest_yaml)
            
            if result.returncode == 0:
                self.logger.info(f"Created namespace: {namespace}")
            else:
                raise Exception(f"Failed to create namespace {namespace}: {result.stderr}")
    
    def _deploy_standard(self, config: DeploymentConfig, manifests: Dict[str, str],
                        deployment_id: str, start_time: float) -> DeploymentResult:
        """Standard deployment (rolling update or recreate)."""
        try:
            # Apply service first
            if 'service' in manifests:
                result = self._run_kubectl(['apply', '-f', '-'], 
                                         input_data=manifests['service'])
                if result.returncode != 0:
                    raise Exception(f"Failed to apply service: {result.stderr}")
            
            # Apply deployment
            result = self._run_kubectl(['apply', '-f', '-'], 
                                     input_data=manifests['deployment'])
            if result.returncode != 0:
                raise Exception(f"Failed to apply deployment: {result.stderr}")
            
            # Apply HPA if present
            if 'hpa' in manifests:
                result = self._run_kubectl(['apply', '-f', '-'], 
                                         input_data=manifests['hpa'])
                if result.returncode != 0:
                    self.logger.warning(f"Failed to apply HPA: {result.stderr}")
            
            # Wait for deployment to be ready
            success = self._wait_for_deployment(config.name, config.namespace, timeout=300)
            
            if success:
                # Perform health checks
                health_checks_passed = self._perform_health_checks(config)
                
                return DeploymentResult(
                    deployment_id=deployment_id,
                    status=DeploymentStatus.DEPLOYED if health_checks_passed else DeploymentStatus.FAILED,
                    message="Deployment completed successfully" if health_checks_passed else "Health checks failed",
                    start_time=start_time,
                    end_time=time.time(),
                    health_checks_passed=health_checks_passed
                )
            else:
                return DeploymentResult(
                    deployment_id=deployment_id,
                    status=DeploymentStatus.FAILED,
                    message="Deployment timed out",
                    start_time=start_time,
                    end_time=time.time()
                )
                
        except Exception as e:
            raise Exception(f"Standard deployment failed: {str(e)}")
    
    def _deploy_blue_green(self, config: DeploymentConfig, manifests: Dict[str, str],
                          deployment_id: str, start_time: float) -> DeploymentResult:
        """Blue-green deployment strategy."""
        try:
            # Create green deployment (new version)
            green_config = DeploymentConfig(
                name=f"{config.name}-green",
                version=config.version,
                image=config.image,
                strategy=DeploymentStrategy.RECREATE,
                environment=config.environment,
                replicas=config.replicas,
                resource_requests=config.resource_requests,
                resource_limits=config.resource_limits,
                env_vars=config.env_vars,
                labels={**config.labels, 'deployment-type': 'green'},
                annotations=config.annotations,
                health_check_path=config.health_check_path,
                readiness_probe_path=config.readiness_probe_path,
                port=config.port,
                namespace=config.namespace
            )
            
            green_manifests = self.generate_kubernetes_manifests(green_config)
            
            # Deploy green version
            result = self._run_kubectl(['apply', '-f', '-'], 
                                     input_data=green_manifests['deployment'])
            if result.returncode != 0:
                raise Exception(f"Failed to deploy green version: {result.stderr}")
            
            # Wait for green deployment
            success = self._wait_for_deployment(green_config.name, config.namespace, timeout=300)
            if not success:
                raise Exception("Green deployment failed to become ready")
            
            # Perform health checks on green
            health_checks_passed = self._perform_health_checks(green_config)
            if not health_checks_passed:
                raise Exception("Health checks failed on green deployment")
            
            # Switch traffic to green (update service selector)
            service_patch = {
                'spec': {
                    'selector': {
                        'app': green_config.name
                    }
                }
            }
            
            patch_json = json.dumps(service_patch)
            result = self._run_kubectl([
                'patch', 'service', f"{config.name}-service",
                '-n', config.namespace,
                '--type', 'merge',
                '-p', patch_json
            ])
            
            if result.returncode != 0:
                raise Exception(f"Failed to switch traffic to green: {result.stderr}")
            
            # Clean up old blue deployment after successful switch
            self._cleanup_old_deployment(config.name, config.namespace)
            
            # Rename green to main
            result = self._run_kubectl([
                'patch', 'deployment', green_config.name,
                '-n', config.namespace,
                '--type', 'json',
                '-p', f'[{{"op": "replace", "path": "/metadata/name", "value": "{config.name}"}}]'
            ])
            
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.DEPLOYED,
                message="Blue-green deployment completed successfully",
                start_time=start_time,
                end_time=time.time(),
                health_checks_passed=True
            )
            
        except Exception as e:
            # Rollback on failure
            self._cleanup_old_deployment(f"{config.name}-green", config.namespace)
            raise Exception(f"Blue-green deployment failed: {str(e)}")
    
    def _deploy_canary(self, config: DeploymentConfig, manifests: Dict[str, str],
                      deployment_id: str, start_time: float) -> DeploymentResult:
        """Canary deployment strategy."""
        try:
            # Create canary deployment with reduced replicas
            canary_replicas = max(1, config.replicas // 4)  # 25% of traffic
            
            canary_config = DeploymentConfig(
                name=f"{config.name}-canary",
                version=config.version,
                image=config.image,
                strategy=DeploymentStrategy.RECREATE,
                environment=config.environment,
                replicas=canary_replicas,
                resource_requests=config.resource_requests,
                resource_limits=config.resource_limits,
                env_vars=config.env_vars,
                labels={**config.labels, 'deployment-type': 'canary'},
                annotations=config.annotations,
                health_check_path=config.health_check_path,
                readiness_probe_path=config.readiness_probe_path,
                port=config.port,
                namespace=config.namespace
            )
            
            canary_manifests = self.generate_kubernetes_manifests(canary_config)
            
            # Deploy canary
            result = self._run_kubectl(['apply', '-f', '-'], 
                                     input_data=canary_manifests['deployment'])
            if result.returncode != 0:
                raise Exception(f"Failed to deploy canary: {result.stderr}")
            
            # Wait for canary deployment
            success = self._wait_for_deployment(canary_config.name, config.namespace, timeout=300)
            if not success:
                raise Exception("Canary deployment failed to become ready")
            
            # Monitor canary for a period
            canary_monitoring_duration = 300  # 5 minutes
            self.logger.info(f"Monitoring canary for {canary_monitoring_duration} seconds")
            
            time.sleep(canary_monitoring_duration)
            
            # Check canary health and metrics
            health_checks_passed = self._perform_health_checks(canary_config)
            if not health_checks_passed:
                raise Exception("Canary health checks failed")
            
            # If canary is healthy, proceed with full deployment
            # Scale down main deployment
            main_replicas = config.replicas - canary_replicas
            self._scale_deployment(config.name, config.namespace, main_replicas)
            
            # Gradually shift traffic (this is simplified - in practice would use ingress controllers)
            # For now, we'll just complete the rollout
            result = self._run_kubectl([
                'patch', 'deployment', canary_config.name,
                '-n', config.namespace,
                '--type', 'json',
                '-p', f'[{{"op": "replace", "path": "/spec/replicas", "value": {config.replicas}}}]'
            ])
            
            # Clean up old deployment
            self._cleanup_old_deployment(config.name, config.namespace)
            
            return DeploymentResult(
                deployment_id=deployment_id,
                status=DeploymentStatus.DEPLOYED,
                message="Canary deployment completed successfully",
                start_time=start_time,
                end_time=time.time(),
                health_checks_passed=True,
                metrics={'canary_monitoring_duration': canary_monitoring_duration}
            )
            
        except Exception as e:
            # Rollback canary on failure
            self._cleanup_old_deployment(f"{config.name}-canary", config.namespace)
            raise Exception(f"Canary deployment failed: {str(e)}")
    
    def _wait_for_deployment(self, deployment_name: str, namespace: str, 
                           timeout: int = 300) -> bool:
        """Wait for deployment to be ready."""
        self.logger.info(f"Waiting for deployment {deployment_name} to be ready...")
        
        result = self._run_kubectl([
            'rollout', 'status', 'deployment', deployment_name,
            '-n', namespace,
            f'--timeout={timeout}s'
        ], timeout=timeout + 10)
        
        if result.returncode == 0:
            self.logger.info(f"Deployment {deployment_name} is ready")
            return True
        else:
            self.logger.error(f"Deployment {deployment_name} failed to become ready: {result.stderr}")
            return False
    
    def _perform_health_checks(self, config: DeploymentConfig) -> bool:
        """Perform health checks on deployed application."""
        self.logger.info(f"Performing health checks for {config.name}")
        
        try:
            # Port-forward for health check
            port_forward_proc = None
            
            if _REQUESTS_AVAILABLE:
                # Start port forwarding in background
                local_port = 8888
                port_forward_proc = subprocess.Popen([
                    self.kubectl_cmd, 'port-forward',
                    f'deployment/{config.name}',
                    f'{local_port}:{config.port}',
                    '-n', config.namespace
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Wait a moment for port forward to establish
                time.sleep(5)
                
                try:
                    # Health check
                    health_url = f'http://localhost:{local_port}{config.health_check_path}'
                    response = requests.get(health_url, timeout=10)
                    
                    if response.status_code == 200:
                        self.logger.info("Health check passed")
                        return True
                    else:
                        self.logger.warning(f"Health check failed: {response.status_code}")
                        return False
                        
                except requests.RequestException as e:
                    self.logger.warning(f"Health check request failed: {e}")
                    return False
                finally:
                    if port_forward_proc:
                        port_forward_proc.terminate()
                        port_forward_proc.wait(timeout=5)
            else:
                # Fallback: check if pods are running
                result = self._run_kubectl([
                    'get', 'pods',
                    '-l', f'app={config.name}',
                    '-n', config.namespace,
                    '-o', 'jsonpath={.items[*].status.phase}'
                ])
                
                if result.returncode == 0 and 'Running' in result.stdout:
                    self.logger.info("Pod health check passed")
                    return True
                else:
                    self.logger.warning("Pod health check failed")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Health check failed with exception: {e}")
            return False
    
    def _scale_deployment(self, deployment_name: str, namespace: str, replicas: int):
        """Scale deployment to specified replicas."""
        result = self._run_kubectl([
            'scale', 'deployment', deployment_name,
            '-n', namespace,
            f'--replicas={replicas}'
        ])
        
        if result.returncode != 0:
            raise Exception(f"Failed to scale deployment {deployment_name}: {result.stderr}")
    
    def _cleanup_old_deployment(self, deployment_name: str, namespace: str):
        """Clean up old deployment."""
        try:
            result = self._run_kubectl([
                'delete', 'deployment', deployment_name,
                '-n', namespace,
                '--ignore-not-found=true'
            ])
            
            if result.returncode == 0:
                self.logger.info(f"Cleaned up old deployment: {deployment_name}")
            else:
                self.logger.warning(f"Failed to clean up deployment {deployment_name}: {result.stderr}")
                
        except Exception as e:
            self.logger.warning(f"Exception during cleanup of {deployment_name}: {e}")
    
    def rollback_deployment(self, deployment_id: str) -> DeploymentResult:
        """Rollback a deployment."""
        if deployment_id not in self.deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.deployments[deployment_id]
        
        # Extract deployment info from original deployment
        # This is simplified - in practice would need to track previous versions
        self.logger.info(f"Rolling back deployment {deployment_id}")
        
        try:
            # Perform rollback using kubectl
            # This assumes the deployment exists and has rollout history
            config_parts = deployment_id.split('-')
            deployment_name = config_parts[0] if config_parts else 'unknown'
            
            result = self._run_kubectl([
                'rollout', 'undo', f'deployment/{deployment_name}',
                '-n', 'default'  # Default namespace for this example
            ])
            
            if result.returncode == 0:
                deployment.status = DeploymentStatus.ROLLED_BACK
                deployment.message = "Deployment rolled back successfully"
                deployment.end_time = time.time()
                
                self.logger.info(f"Successfully rolled back deployment {deployment_id}")
                return deployment
            else:
                raise Exception(f"Rollback failed: {result.stderr}")
                
        except Exception as e:
            error_msg = f"Rollback failed: {str(e)}"
            self.logger.error(error_msg)
            
            deployment.status = DeploymentStatus.FAILED
            deployment.message = error_msg
            deployment.end_time = time.time()
            
            return deployment
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentResult]:
        """Get deployment status."""
        return self.deployments.get(deployment_id)
    
    def list_deployments(self, namespace: str = None) -> List[Dict[str, Any]]:
        """List all deployments."""
        try:
            cmd = ['get', 'deployments', '-o', 'json']
            if namespace:
                cmd.extend(['-n', namespace])
            else:
                cmd.append('--all-namespaces')
            
            result = self._run_kubectl(cmd)
            
            if result.returncode == 0:
                deployments_data = json.loads(result.stdout)
                return deployments_data.get('items', [])
            else:
                self.logger.error(f"Failed to list deployments: {result.stderr}")
                return []
                
        except Exception as e:
            self.logger.error(f"Exception listing deployments: {e}")
            return []


class AdvancedDeploymentOrchestrator:
    """High-level deployment orchestrator with advanced features."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        self.deployer = KubernetesDeployer(kubeconfig_path)
        self.logger = get_global_logger()
        self.monitoring_system = None
        self.deployment_queue = []
        self.active_deployments = {}
        
        # Initialize monitoring
        try:
            self.monitoring_system = create_monitoring_system({
                'system_interval': 10.0,
                'photonic_interval': 5.0,
                'performance_interval': 2.0
            })
            self.monitoring_system.start()
            self.logger.info("Deployment monitoring system started")
        except Exception as e:
            self.logger.warning(f"Failed to start monitoring system: {e}")
    
    def deploy_application(self, config: DeploymentConfig, 
                          wait_for_completion: bool = True) -> DeploymentResult:
        """Deploy application with advanced orchestration."""
        self.logger.info(f"Orchestrating deployment of {config.name} v{config.version}")
        
        # Validate configuration
        validation_result = self._validate_deployment_config(config)
        if not validation_result.is_valid:
            raise ValueError(f"Invalid deployment configuration: {validation_result.errors}")
        
        # Pre-deployment checks
        self._perform_pre_deployment_checks(config)
        
        # Execute deployment
        result = self.deployer.deploy(config)
        
        # Store active deployment
        self.active_deployments[result.deployment_id] = {
            'config': config,
            'result': result,
            'start_time': time.time()
        }
        
        if wait_for_completion:
            # Monitor deployment progress
            self._monitor_deployment_progress(result.deployment_id, config)
        
        return result
    
    def _validate_deployment_config(self, config: DeploymentConfig) -> ValidationResult:
        """Validate deployment configuration."""
        errors = []
        warnings = []
        
        # Basic validation
        if not config.name:
            errors.append("Deployment name is required")
        
        if not config.image:
            errors.append("Container image is required")
        
        if config.replicas < 1:
            errors.append("Replicas must be at least 1")
        
        if config.port < 1 or config.port > 65535:
            errors.append("Port must be between 1 and 65535")
        
        # Environment-specific validation
        if config.environment == EnvironmentType.PRODUCTION:
            if config.replicas < 2:
                warnings.append("Production deployments should have at least 2 replicas")
            
            if not config.resource_limits:
                warnings.append("Production deployments should have resource limits")
        
        # Strategy validation
        if (config.strategy in [DeploymentStrategy.BLUE_GREEN, DeploymentStrategy.CANARY] and 
            config.replicas < 2):
            warnings.append(f"{config.strategy.value} strategy works best with multiple replicas")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics={'validation_time': time.time()}
        )
    
    def _perform_pre_deployment_checks(self, config: DeploymentConfig):
        """Perform pre-deployment checks."""
        self.logger.info("Performing pre-deployment checks")
        
        # Check if namespace exists
        try:
            result = self.deployer._run_kubectl(['get', 'namespace', config.namespace])
            if result.returncode != 0 and config.namespace != 'default':
                self.logger.warning(f"Namespace {config.namespace} does not exist, will be created")
        except Exception as e:
            self.logger.warning(f"Could not check namespace: {e}")
        
        # Check resource availability (simplified)
        try:
            # Get node resources
            result = self.deployer._run_kubectl([
                'top', 'nodes', '--no-headers'
            ])
            
            if result.returncode == 0:
                self.logger.info("Node resource check completed")
            else:
                self.logger.warning("Could not check node resources - metrics server may not be available")
        except Exception as e:
            self.logger.warning(f"Resource check failed: {e}")
    
    def _monitor_deployment_progress(self, deployment_id: str, config: DeploymentConfig):
        """Monitor deployment progress."""
        self.logger.info(f"Monitoring deployment progress for {deployment_id}")
        
        # This would integrate with the monitoring system to track deployment metrics
        if self.monitoring_system:
            health_status = self.monitoring_system.get_system_health()
            self.logger.info(f"System health during deployment: {health_status['status']}")
    
    def rollback_deployment(self, deployment_id: str) -> DeploymentResult:
        """Rollback deployment with advanced error handling."""
        self.logger.info(f"Initiating rollback for deployment {deployment_id}")
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found in active deployments")
        
        # Perform rollback
        result = self.deployer.rollback_deployment(deployment_id)
        
        # Update active deployments
        if deployment_id in self.active_deployments:
            self.active_deployments[deployment_id]['result'] = result
        
        return result
    
    def get_deployment_metrics(self, deployment_id: str) -> Dict[str, Any]:
        """Get deployment metrics."""
        if deployment_id not in self.active_deployments:
            return {}
        
        deployment_info = self.active_deployments[deployment_id]
        config = deployment_info['config']
        
        # Get basic metrics
        metrics = {
            'deployment_id': deployment_id,
            'application': config.name,
            'version': config.version,
            'strategy': config.strategy.value,
            'environment': config.environment.value,
            'start_time': deployment_info['start_time'],
            'duration': time.time() - deployment_info['start_time']
        }
        
        # Add monitoring system metrics if available
        if self.monitoring_system:
            health_status = self.monitoring_system.get_system_health()
            metrics['system_health'] = health_status
        
        return metrics
    
    def cleanup_deployments(self, older_than_hours: int = 24):
        """Clean up old deployment records."""
        cutoff_time = time.time() - (older_than_hours * 3600)
        
        deployments_to_remove = []
        for deployment_id, info in self.active_deployments.items():
            if info['start_time'] < cutoff_time:
                deployments_to_remove.append(deployment_id)
        
        for deployment_id in deployments_to_remove:
            del self.active_deployments[deployment_id]
        
        if deployments_to_remove:
            self.logger.info(f"Cleaned up {len(deployments_to_remove)} old deployment records")
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get overall cluster status."""
        try:
            # Get nodes
            nodes_result = self.deployer._run_kubectl(['get', 'nodes', '-o', 'json'])
            
            # Get deployments
            deployments_result = self.deployer._run_kubectl(['get', 'deployments', '--all-namespaces', '-o', 'json'])
            
            cluster_status = {
                'timestamp': time.time(),
                'nodes_ready': 0,
                'total_nodes': 0,
                'total_deployments': 0,
                'healthy_deployments': 0
            }
            
            if nodes_result.returncode == 0:
                nodes_data = json.loads(nodes_result.stdout)
                cluster_status['total_nodes'] = len(nodes_data.get('items', []))
                
                # Count ready nodes
                for node in nodes_data.get('items', []):
                    conditions = node.get('status', {}).get('conditions', [])
                    for condition in conditions:
                        if condition.get('type') == 'Ready' and condition.get('status') == 'True':
                            cluster_status['nodes_ready'] += 1
                            break
            
            if deployments_result.returncode == 0:
                deployments_data = json.loads(deployments_result.stdout)
                cluster_status['total_deployments'] = len(deployments_data.get('items', []))
                
                # Count healthy deployments
                for deployment in deployments_data.get('items', []):
                    status = deployment.get('status', {})
                    ready_replicas = status.get('readyReplicas', 0)
                    replicas = status.get('replicas', 0)
                    
                    if ready_replicas == replicas and replicas > 0:
                        cluster_status['healthy_deployments'] += 1
            
            return cluster_status
            
        except Exception as e:
            self.logger.error(f"Failed to get cluster status: {e}")
            return {'error': str(e)}


# Convenience functions
def create_deployment_config(name: str, image: str, version: str = "latest", 
                           **kwargs) -> DeploymentConfig:
    """Create deployment configuration with defaults."""
    return DeploymentConfig(
        name=name,
        image=image,
        version=version,
        strategy=DeploymentStrategy.ROLLING_UPDATE,
        environment=EnvironmentType.DEVELOPMENT,
        **kwargs
    )


def deploy_photonic_application(name: str, image: str, version: str = "latest",
                               strategy: str = "rolling_update",
                               environment: str = "development",
                               **kwargs) -> DeploymentResult:
    """Deploy photonic application with simplified interface."""
    strategy_map = {
        'rolling_update': DeploymentStrategy.ROLLING_UPDATE,
        'blue_green': DeploymentStrategy.BLUE_GREEN,
        'canary': DeploymentStrategy.CANARY,
        'recreate': DeploymentStrategy.RECREATE
    }
    
    env_map = {
        'development': EnvironmentType.DEVELOPMENT,
        'staging': EnvironmentType.STAGING,
        'production': EnvironmentType.PRODUCTION,
        'testing': EnvironmentType.TESTING
    }
    
    config = DeploymentConfig(
        name=name,
        image=image,
        version=version,
        strategy=strategy_map.get(strategy, DeploymentStrategy.ROLLING_UPDATE),
        environment=env_map.get(environment, EnvironmentType.DEVELOPMENT),
        **kwargs
    )
    
    orchestrator = AdvancedDeploymentOrchestrator()
    return orchestrator.deploy_application(config)
