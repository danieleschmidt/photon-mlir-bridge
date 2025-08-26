"""
Generation 3 Enhancement: Distributed Quantum-Photonic Orchestrator
Massively scalable distributed computing for quantum-photonic systems.

This module implements enterprise-grade distributed computing capabilities with
auto-scaling, load balancing, and intelligent resource management across
multiple quantum-photonic nodes.
"""

import asyncio
try:
    import aiohttp
    _AIOHTTP_AVAILABLE = True
except ImportError:
    _AIOHTTP_AVAILABLE = False
    # Mock aiohttp classes for graceful degradation
    class aiohttp:
        class ClientSession:
            def __init__(self, *args, **kwargs):
                pass
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            async def get(self, *args, **kwargs):
                raise RuntimeError("aiohttp not available - install with: pip install aiohttp")
            async def post(self, *args, **kwargs):
                raise RuntimeError("aiohttp not available - install with: pip install aiohttp")
        class ClientTimeout:
            def __init__(self, *args, **kwargs):
                pass

try:
    import numpy as np
except ImportError:
    from .numpy_fallback import get_numpy
    np = get_numpy()
import time
import json
import hashlib
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import threading
import multiprocessing as mp
from queue import Queue, Empty, PriorityQueue
from collections import defaultdict, deque
import uuid
from pathlib import Path
import pickle
import gzip
try:
    import lz4.frame
    _LZ4_AVAILABLE = True
except ImportError:
    _LZ4_AVAILABLE = False
    # Mock lz4.frame for fallback
    class lz4:
        class frame:
            @staticmethod
            def compress(data):
                return gzip.compress(data)  # Fallback to gzip
            @staticmethod
            def decompress(data):
                return gzip.decompress(data)  # Fallback to gzip

try:
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as torch_mp
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

try:
    import redis
    import etcd3
    _DISTRIBUTED_STORAGE_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_STORAGE_AVAILABLE = False

from .logging_config import get_global_logger
from .validation import PhotonicValidator
from .security import SecureDataHandler
from .quantum_photonic_fusion import QuantumPhotonicConfig, HybridQuantumPhotonicModel
from .advanced_thermal_quantum_manager import AdvancedThermalQuantumManager


class NodeStatus(Enum):
    """Status of distributed nodes."""
    ONLINE = "online"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RESOURCE_BASED = "resource_based"
    QUANTUM_COHERENCE_AWARE = "quantum_coherence_aware"
    THERMAL_AWARE = "thermal_aware"
    LATENCY_OPTIMIZED = "latency_optimized"
    THROUGHPUT_OPTIMIZED = "throughput_optimized"


class ScalingMode(Enum):
    """Auto-scaling modes."""
    MANUAL = "manual"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    QUANTUM_DEMAND_BASED = "quantum_demand_based"
    HYBRID_INTELLIGENT = "hybrid_intelligent"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics for a node."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    gpu_percent: float = 0.0
    quantum_coherence: float = 1.0
    photonic_mesh_utilization: float = 0.0
    thermal_stress: float = 0.0
    network_latency_ms: float = 0.0
    throughput_ops_per_sec: float = 0.0
    error_rate: float = 0.0
    last_updated: float = field(default_factory=time.time)
    
    def get_overall_load(self) -> float:
        """Calculate overall load score (0.0 to 1.0)."""
        weights = {
            'cpu': 0.25,
            'memory': 0.25,
            'gpu': 0.15,
            'quantum': 0.20,
            'photonic': 0.10,
            'thermal': 0.05
        }
        
        load_score = (
            weights['cpu'] * (self.cpu_percent / 100.0) +
            weights['memory'] * (self.memory_percent / 100.0) +
            weights['gpu'] * (self.gpu_percent / 100.0) +
            weights['quantum'] * (1.0 - self.quantum_coherence) +
            weights['photonic'] * self.photonic_mesh_utilization +
            weights['thermal'] * self.thermal_stress
        )
        
        return min(1.0, max(0.0, load_score))
    
    def is_overloaded(self, threshold: float = 0.85) -> bool:
        """Check if node is overloaded."""
        return self.get_overall_load() > threshold
    
    def can_accept_task(self, task_requirements: Dict[str, float]) -> bool:
        """Check if node can accept a new task."""
        cpu_ok = self.cpu_percent + task_requirements.get('cpu', 0) <= 90
        memory_ok = self.memory_percent + task_requirements.get('memory', 0) <= 90
        coherence_ok = self.quantum_coherence >= task_requirements.get('min_coherence', 0.5)
        thermal_ok = self.thermal_stress <= task_requirements.get('max_thermal_stress', 0.8)
        
        return cpu_ok and memory_ok and coherence_ok and thermal_ok


@dataclass
class ComputeNode:
    """Represents a distributed compute node."""
    node_id: str
    address: str
    port: int = 8080
    capabilities: Dict[str, Any] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.OFFLINE
    metrics: ResourceMetrics = field(default_factory=ResourceMetrics)
    last_heartbeat: float = field(default_factory=time.time)
    active_tasks: int = 0
    total_tasks_processed: int = 0
    avg_task_duration_ms: float = 0.0
    quantum_fidelity: float = 0.99
    photonic_efficiency: float = 0.95
    
    def __post_init__(self):
        if not self.capabilities:
            self.capabilities = {
                'max_qubits': 32,
                'photonic_mesh_size': (64, 64),
                'wavelength_channels': 8,
                'max_power_mw': 100,
                'thermal_limit_celsius': 85.0,
                'supported_architectures': ['hybrid_quantum_classical'],
                'max_concurrent_tasks': 4
            }
    
    def is_healthy(self, timeout_seconds: float = 30.0) -> bool:
        """Check if node is healthy based on heartbeat."""
        return (time.time() - self.last_heartbeat) < timeout_seconds
    
    def get_url(self) -> str:
        """Get full URL for the node."""
        return f"http://{self.address}:{self.port}"
    
    def estimate_task_duration(self, task_complexity: float) -> float:
        """Estimate task duration based on historical data."""
        base_duration = self.avg_task_duration_ms if self.avg_task_duration_ms > 0 else 1000.0
        load_factor = 1.0 + self.metrics.get_overall_load() * 2.0
        complexity_factor = 1.0 + task_complexity
        
        return base_duration * load_factor * complexity_factor


@dataclass
class DistributedTask:
    """Represents a distributed computation task."""
    task_id: str
    task_type: str
    priority: TaskPriority = TaskPriority.NORMAL
    payload: Dict[str, Any] = field(default_factory=dict)
    requirements: Dict[str, float] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    assigned_node: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: float = 300.0
    
    def __lt__(self, other):
        """Enable priority queue ordering."""
        return self.priority.value < other.priority.value
    
    def is_expired(self) -> bool:
        """Check if task has expired."""
        return (time.time() - self.created_at) > self.timeout_seconds
    
    def get_complexity_score(self) -> float:
        """Calculate task complexity score."""
        base_complexity = {
            'quantum_compilation': 0.8,
            'photonic_optimization': 0.6,
            'hybrid_simulation': 1.0,
            'thermal_analysis': 0.3,
            'error_correction': 0.5
        }.get(self.task_type, 0.5)
        
        # Adjust based on payload size
        payload_size = len(str(self.payload))
        size_factor = min(2.0, payload_size / 10000)  # Normalize to reasonable range
        
        return base_complexity * size_factor
    
    def serialize(self) -> bytes:
        """Serialize task for network transmission."""
        task_dict = {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'priority': self.priority.value,
            'payload': self.payload,
            'requirements': self.requirements,
            'dependencies': self.dependencies,
            'created_at': self.created_at,
            'timeout_seconds': self.timeout_seconds
        }
        
        # Compress for efficiency
        # Use secure JSON serialization instead of pickle for security
        import json
        serialized = json.dumps(task_dict, default=str).encode('utf-8')
        compressed = lz4.frame.compress(serialized)
        return compressed
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'DistributedTask':
        """Deserialize task from network data."""
        try:
            decompressed = lz4.frame.decompress(data)
            # Use secure JSON deserialization instead of pickle for security
            import json
            task_dict = json.loads(decompressed.decode('utf-8'))
            
            task = cls(
                task_id=task_dict['task_id'],
                task_type=task_dict['task_type'],
                priority=TaskPriority(task_dict['priority']),
                payload=task_dict['payload'],
                requirements=task_dict['requirements'],
                dependencies=task_dict['dependencies'],
                created_at=task_dict['created_at'],
                timeout_seconds=task_dict['timeout_seconds']
            )
            return task
        except Exception as e:
            raise ValueError(f"Failed to deserialize task: {str(e)}")


class DistributedCacheManager:
    """Manages distributed caching for computational results."""
    
    def __init__(self, cache_size_mb: int = 1024):
        self.cache_size_mb = cache_size_mb
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_size_mb': 0
        }
        self.access_times = {}
        self.logger = get_global_logger()
        
        # Initialize distributed cache if available
        self.redis_client = None
        if _DISTRIBUTED_STORAGE_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()  # Test connection
                self.logger.info("🖺 Distributed cache (Redis) connected")
            except Exception:
                self.redis_client = None
                self.logger.info("🖺 Using local cache only (Redis unavailable)")
    
    def _get_cache_key(self, task_type: str, payload_hash: str) -> str:
        """Generate cache key for task."""
        return f"photonic_cache:{task_type}:{payload_hash}"
    
    def _calculate_payload_hash(self, payload: Dict[str, Any]) -> str:
        """Calculate hash of task payload."""
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()[:16]
    
    def get(self, task_type: str, payload: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for task."""
        payload_hash = self._calculate_payload_hash(payload)
        cache_key = self._get_cache_key(task_type, payload_hash)
        
        # Try local cache first
        if cache_key in self.local_cache:
            self.cache_stats['hits'] += 1
            self.access_times[cache_key] = time.time()
            return self.local_cache[cache_key]
        
        # Try distributed cache
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    # Use secure JSON deserialization instead of pickle for security
                    import json
                    result = json.loads(lz4.frame.decompress(cached_data).decode('utf-8'))
                    # Store in local cache for faster access
                    self._store_local(cache_key, result)
                    self.cache_stats['hits'] += 1
                    return result
            except Exception as e:
                self.logger.warning(f"Distributed cache error: {str(e)}")
        
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, task_type: str, payload: Dict[str, Any], result: Any, ttl_seconds: int = 3600) -> None:
        """Cache computation result."""
        payload_hash = self._calculate_payload_hash(payload)
        cache_key = self._get_cache_key(task_type, payload_hash)
        
        # Store in local cache
        self._store_local(cache_key, result)
        
        # Store in distributed cache
        if self.redis_client:
            try:
                # Use secure JSON serialization instead of pickle for security
                import json
                serialized = json.dumps(result, default=str).encode('utf-8')
                compressed = lz4.frame.compress(serialized)
                self.redis_client.setex(cache_key, ttl_seconds, compressed)
            except Exception as e:
                self.logger.warning(f"Failed to store in distributed cache: {str(e)}")
    
    def _store_local(self, cache_key: str, result: Any) -> None:
        """Store result in local cache with size management."""
        # Estimate size
        # Use secure JSON serialization instead of pickle for security
        import json
        estimated_size_mb = len(json.dumps(result, default=str).encode('utf-8')) / (1024 * 1024)
        
        # Evict if necessary
        while (self.cache_stats['total_size_mb'] + estimated_size_mb > self.cache_size_mb and 
               len(self.local_cache) > 0):
            self._evict_lru()
        
        self.local_cache[cache_key] = result
        self.access_times[cache_key] = time.time()
        self.cache_stats['total_size_mb'] += estimated_size_mb
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        
        if lru_key in self.local_cache:
            del self.local_cache[lru_key]
            # Use secure JSON serialization instead of pickle for security
            import json
            cached_value = self.local_cache.get(lru_key, '')
            estimated_size_mb = len(json.dumps(cached_value, default=str).encode('utf-8')) / (1024 * 1024)
            self.cache_stats['total_size_mb'] -= estimated_size_mb
            self.cache_stats['evictions'] += 1
        
        del self.access_times[lru_key]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_size_mb': self.cache_stats['total_size_mb'],
            'cache_items': len(self.local_cache),
            'distributed_available': self.redis_client is not None,
            **self.cache_stats
        }


class LoadBalancer:
    """Intelligent load balancer for quantum-photonic compute nodes."""
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.QUANTUM_COHERENCE_AWARE):
        self.strategy = strategy
        self.node_selection_history = deque(maxlen=1000)
        self.performance_tracker = defaultdict(list)
        self.logger = get_global_logger()
    
    def select_node(self, nodes: Dict[str, ComputeNode], task: DistributedTask) -> Optional[ComputeNode]:
        """Select optimal node for task execution."""
        available_nodes = {
            nid: node for nid, node in nodes.items()
            if (node.status == NodeStatus.ONLINE and 
                node.is_healthy() and
                node.metrics.can_accept_task(task.requirements))
        }
        
        if not available_nodes:
            return None
        
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_selection(available_nodes)
        elif self.strategy == LoadBalancingStrategy.QUANTUM_COHERENCE_AWARE:
            return self._quantum_coherence_selection(available_nodes, task)
        elif self.strategy == LoadBalancingStrategy.THERMAL_AWARE:
            return self._thermal_aware_selection(available_nodes, task)
        elif self.strategy == LoadBalancingStrategy.LATENCY_OPTIMIZED:
            return self._latency_optimized_selection(available_nodes, task)
        elif self.strategy == LoadBalancingStrategy.THROUGHPUT_OPTIMIZED:
            return self._throughput_optimized_selection(available_nodes, task)
        else:
            return self._resource_based_selection(available_nodes)
    
    def _round_robin_selection(self, nodes: Dict[str, ComputeNode]) -> ComputeNode:
        """Simple round-robin selection."""
        node_ids = sorted(nodes.keys())
        
        # Find next node in rotation
        if not self.node_selection_history:
            selected_id = node_ids[0]
        else:
            last_selected = self.node_selection_history[-1]
            try:
                current_index = node_ids.index(last_selected)
                selected_id = node_ids[(current_index + 1) % len(node_ids)]
            except ValueError:
                selected_id = node_ids[0]
        
        self.node_selection_history.append(selected_id)
        return nodes[selected_id]
    
    def _least_connections_selection(self, nodes: Dict[str, ComputeNode]) -> ComputeNode:
        """Select node with least active connections."""
        selected_node = min(nodes.values(), key=lambda n: n.active_tasks)
        self.node_selection_history.append(selected_node.node_id)
        return selected_node
    
    def _resource_based_selection(self, nodes: Dict[str, ComputeNode]) -> ComputeNode:
        """Select node with lowest resource utilization."""
        selected_node = min(nodes.values(), key=lambda n: n.metrics.get_overall_load())
        self.node_selection_history.append(selected_node.node_id)
        return selected_node
    
    def _quantum_coherence_selection(self, nodes: Dict[str, ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node optimized for quantum coherence requirements."""
        # Score nodes based on quantum coherence and task requirements
        def coherence_score(node: ComputeNode) -> float:
            base_score = node.metrics.quantum_coherence
            fidelity_bonus = node.quantum_fidelity * 0.1
            load_penalty = node.metrics.get_overall_load() * 0.3
            thermal_penalty = node.metrics.thermal_stress * 0.2
            
            return base_score + fidelity_bonus - load_penalty - thermal_penalty
        
        selected_node = max(nodes.values(), key=coherence_score)
        self.node_selection_history.append(selected_node.node_id)
        return selected_node
    
    def _thermal_aware_selection(self, nodes: Dict[str, ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node with optimal thermal conditions."""
        def thermal_score(node: ComputeNode) -> float:
            thermal_efficiency = 1.0 - node.metrics.thermal_stress
            load_efficiency = 1.0 - node.metrics.get_overall_load()
            photonic_efficiency = node.photonic_efficiency
            
            return (thermal_efficiency * 0.4 + load_efficiency * 0.3 + 
                   photonic_efficiency * 0.3)
        
        selected_node = max(nodes.values(), key=thermal_score)
        self.node_selection_history.append(selected_node.node_id)
        return selected_node
    
    def _latency_optimized_selection(self, nodes: Dict[str, ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node for minimum latency."""
        def latency_score(node: ComputeNode) -> float:
            base_latency = node.metrics.network_latency_ms
            processing_latency = node.estimate_task_duration(task.get_complexity_score())
            queue_latency = node.active_tasks * 100  # Estimate
            
            total_latency = base_latency + processing_latency + queue_latency
            return -total_latency  # Negative for minimization
        
        selected_node = max(nodes.values(), key=latency_score)
        self.node_selection_history.append(selected_node.node_id)
        return selected_node
    
    def _throughput_optimized_selection(self, nodes: Dict[str, ComputeNode], task: DistributedTask) -> ComputeNode:
        """Select node for maximum throughput."""
        def throughput_score(node: ComputeNode) -> float:
            base_throughput = node.metrics.throughput_ops_per_sec
            load_factor = 1.0 - node.metrics.get_overall_load() * 0.5
            error_penalty = node.metrics.error_rate * 0.3
            
            return base_throughput * load_factor - error_penalty
        
        selected_node = max(nodes.values(), key=throughput_score)
        self.node_selection_history.append(selected_node.node_id)
        return selected_node


class AutoScaler:
    """Intelligent auto-scaling system for quantum-photonic clusters."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.mode = ScalingMode(config.get('mode', 'reactive'))
        self.min_nodes = config.get('min_nodes', 2)
        self.max_nodes = config.get('max_nodes', 100)
        self.scale_up_threshold = config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = config.get('scale_down_threshold', 0.3)
        self.scale_up_cooldown = config.get('scale_up_cooldown_s', 300)
        self.scale_down_cooldown = config.get('scale_down_cooldown_s', 600)
        
        self.last_scale_action = 0.0
        self.scaling_history = deque(maxlen=100)
        self.demand_predictor = None
        self.logger = get_global_logger()
        
        if self.mode == ScalingMode.PREDICTIVE and _TORCH_AVAILABLE:
            self._initialize_demand_predictor()
    
    def _initialize_demand_predictor(self) -> None:
        """Initialize ML-based demand prediction model."""
        if not _TORCH_AVAILABLE:
            return
        
        self.demand_predictor = torch.nn.Sequential(
            torch.nn.Linear(10, 32),  # 10 time steps of demand history
            torch.nn.ReLU(),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 1)   # Predict future demand
        )
        
        self.demand_history = deque(maxlen=1000)
        self.logger.info("🤖 Predictive auto-scaling initialized")
    
    def should_scale(self, nodes: Dict[str, ComputeNode], task_queue_size: int) -> Tuple[bool, str, int]:
        """Determine if scaling action is needed.
        
        Returns:
            (should_scale, direction, target_nodes)
        """
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_action < self.scale_up_cooldown:
            return False, 'none', len(nodes)
        
        online_nodes = {nid: node for nid, node in nodes.items() 
                       if node.status == NodeStatus.ONLINE and node.is_healthy()}
        
        if not online_nodes:
            return True, 'up', max(self.min_nodes, 1)
        
        # Calculate cluster metrics
        avg_load = np.mean([node.metrics.get_overall_load() for node in online_nodes.values()])
        max_load = max([node.metrics.get_overall_load() for node in online_nodes.values()])
        queue_pressure = min(1.0, task_queue_size / (len(online_nodes) * 10))  # Normalize
        
        # Combined load metric
        combined_load = (avg_load * 0.4 + max_load * 0.4 + queue_pressure * 0.2)
        
        # Record demand for predictive scaling
        if self.demand_predictor is not None:
            self.demand_history.append(combined_load)
        
        # Scaling decisions
        current_nodes = len(online_nodes)
        
        if self.mode == ScalingMode.REACTIVE:
            return self._reactive_scaling_decision(combined_load, current_nodes)
        elif self.mode == ScalingMode.PREDICTIVE:
            return self._predictive_scaling_decision(combined_load, current_nodes)
        elif self.mode == ScalingMode.QUANTUM_DEMAND_BASED:
            return self._quantum_demand_scaling(nodes, task_queue_size)
        elif self.mode == ScalingMode.HYBRID_INTELLIGENT:
            return self._hybrid_intelligent_scaling(nodes, combined_load, task_queue_size)
        else:
            return self._reactive_scaling_decision(combined_load, current_nodes)
    
    def _reactive_scaling_decision(self, combined_load: float, current_nodes: int) -> Tuple[bool, str, int]:
        """Reactive scaling based on current load."""
        if combined_load > self.scale_up_threshold and current_nodes < self.max_nodes:
            # Scale up
            scale_factor = min(2.0, combined_load / self.scale_up_threshold)
            target_nodes = min(self.max_nodes, int(current_nodes * scale_factor))
            return True, 'up', target_nodes
        
        elif combined_load < self.scale_down_threshold and current_nodes > self.min_nodes:
            # Scale down
            scale_factor = max(0.5, combined_load / self.scale_down_threshold)
            target_nodes = max(self.min_nodes, int(current_nodes * scale_factor))
            return True, 'down', target_nodes
        
        return False, 'none', current_nodes
    
    def _predictive_scaling_decision(self, combined_load: float, current_nodes: int) -> Tuple[bool, str, int]:
        """Predictive scaling based on ML demand forecasting."""
        if not self.demand_predictor or len(self.demand_history) < 10:
            return self._reactive_scaling_decision(combined_load, current_nodes)
        
        try:
            # Predict future demand
            recent_demand = list(self.demand_history)[-10:]
            demand_tensor = torch.tensor(recent_demand, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                predicted_demand = self.demand_predictor(demand_tensor).item()
            
            # Make scaling decision based on prediction
            if predicted_demand > self.scale_up_threshold and current_nodes < self.max_nodes:
                scale_factor = min(2.0, predicted_demand / self.scale_up_threshold)
                target_nodes = min(self.max_nodes, int(current_nodes * scale_factor))
                return True, 'up', target_nodes
            
            elif predicted_demand < self.scale_down_threshold and current_nodes > self.min_nodes:
                scale_factor = max(0.5, predicted_demand / self.scale_down_threshold)
                target_nodes = max(self.min_nodes, int(current_nodes * scale_factor))
                return True, 'down', target_nodes
            
            return False, 'none', current_nodes
            
        except Exception as e:
            self.logger.warning(f"Predictive scaling failed, falling back to reactive: {str(e)}")
            return self._reactive_scaling_decision(combined_load, current_nodes)
    
    def _quantum_demand_scaling(self, nodes: Dict[str, ComputeNode], task_queue_size: int) -> Tuple[bool, str, int]:
        """Scaling based on quantum-specific demand metrics."""
        online_nodes = {nid: node for nid, node in nodes.items() 
                       if node.status == NodeStatus.ONLINE and node.is_healthy()}
        
        if not online_nodes:
            return True, 'up', self.min_nodes
        
        # Quantum-specific metrics
        avg_coherence = np.mean([node.metrics.quantum_coherence for node in online_nodes.values()])
        avg_fidelity = np.mean([node.quantum_fidelity for node in online_nodes.values()])
        thermal_violations = sum(1 for node in online_nodes.values() 
                               if node.metrics.thermal_stress > 0.8)
        
        # Quantum demand pressure
        quantum_pressure = (
            (1.0 - avg_coherence) * 0.4 +
            (1.0 - avg_fidelity) * 0.3 +
            (thermal_violations / len(online_nodes)) * 0.3
        )
        
        current_nodes = len(online_nodes)
        
        if quantum_pressure > 0.6 and current_nodes < self.max_nodes:
            target_nodes = min(self.max_nodes, current_nodes + max(1, thermal_violations))
            return True, 'up', target_nodes
        
        elif quantum_pressure < 0.2 and current_nodes > self.min_nodes:
            target_nodes = max(self.min_nodes, int(current_nodes * 0.8))
            return True, 'down', target_nodes
        
        return False, 'none', current_nodes
    
    def _hybrid_intelligent_scaling(self, nodes: Dict[str, ComputeNode], 
                                  combined_load: float, task_queue_size: int) -> Tuple[bool, str, int]:
        """Hybrid intelligent scaling combining multiple strategies."""
        # Get decisions from different strategies
        reactive_decision = self._reactive_scaling_decision(combined_load, len(nodes))
        quantum_decision = self._quantum_demand_scaling(nodes, task_queue_size)
        
        # Predictive decision if available
        predictive_decision = None
        if self.demand_predictor and len(self.demand_history) >= 10:
            predictive_decision = self._predictive_scaling_decision(combined_load, len(nodes))
        
        # Weighted decision making
        decisions = [reactive_decision, quantum_decision]
        if predictive_decision:
            decisions.append(predictive_decision)
        
        # Count votes for scaling direction
        up_votes = sum(1 for decision in decisions if decision[1] == 'up')
        down_votes = sum(1 for decision in decisions if decision[1] == 'down')
        none_votes = sum(1 for decision in decisions if decision[1] == 'none')
        
        # Make final decision based on majority
        if up_votes > max(down_votes, none_votes):
            # Scale up - use maximum target from up votes
            target_nodes = max([d[2] for d in decisions if d[1] == 'up'])
            return True, 'up', target_nodes
        
        elif down_votes > max(up_votes, none_votes):
            # Scale down - use minimum target from down votes
            target_nodes = min([d[2] for d in decisions if d[1] == 'down'])
            return True, 'down', target_nodes
        
        else:
            # No scaling
            return False, 'none', len(nodes)
    
    def record_scaling_action(self, direction: str, from_nodes: int, to_nodes: int) -> None:
        """Record scaling action for analysis."""
        self.last_scale_action = time.time()
        
        scaling_event = {
            'timestamp': self.last_scale_action,
            'direction': direction,
            'from_nodes': from_nodes,
            'to_nodes': to_nodes,
            'mode': self.mode.value
        }
        
        self.scaling_history.append(scaling_event)
        
        self.logger.info(
            f"🔄 Auto-scaling: {direction} from {from_nodes} to {to_nodes} nodes "
            f"(mode: {self.mode.value})"
        )


class DistributedQuantumPhotonicOrchestrator:
    """Master orchestrator for distributed quantum-photonic computing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_global_logger()
        self.validator = PhotonicValidator()
        
        # Core components
        self.nodes = {}  # node_id -> ComputeNode
        self.task_queue = PriorityQueue()
        self.active_tasks = {}  # task_id -> DistributedTask
        self.completed_tasks = deque(maxlen=10000)
        
        # Management systems
        self.load_balancer = LoadBalancer(
            LoadBalancingStrategy(config.get('load_balancing_strategy', 'quantum_coherence_aware'))
        )
        self.auto_scaler = AutoScaler(config.get('auto_scaling', {}))
        self.cache_manager = DistributedCacheManager(config.get('cache_size_mb', 1024))
        
        # Networking
        self.cluster_port = config.get('cluster_port', 8080)
        self.heartbeat_interval = config.get('heartbeat_interval_s', 5.0)
        self.node_timeout = config.get('node_timeout_s', 30.0)
        
        # Threading and async
        self.orchestrator_active = False
        self.event_loop = None
        self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 16))
        
        # Performance metrics
        self.metrics = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'average_task_duration_ms': 0.0,
            'cluster_utilization': 0.0,
            'cache_hit_rate': 0.0,
            'scaling_events': 0,
            'uptime_seconds': 0.0
        }
        
        self.start_time = time.time()
        
        self.logger.info("🌌 Distributed Quantum-Photonic Orchestrator initialized")
        self.logger.info(f"   Load balancing: {self.load_balancer.strategy.value}")
        self.logger.info(f"   Auto-scaling: {self.auto_scaler.mode.value}")
        self.logger.info(f"   Cache size: {config.get('cache_size_mb', 1024)}MB")
    
    async def start_orchestration(self) -> None:
        """Start the distributed orchestration system."""
        if self.orchestrator_active:
            self.logger.warning("Orchestrator already active")
            return
        
        self.orchestrator_active = True
        self.event_loop = asyncio.get_event_loop()
        
        # Start background tasks
        orchestration_tasks = [
            asyncio.create_task(self._task_scheduler_loop()),
            asyncio.create_task(self._node_health_monitor_loop()),
            asyncio.create_task(self._auto_scaling_loop()),
            asyncio.create_task(self._metrics_collection_loop()),
            asyncio.create_task(self._cluster_coordination_loop())
        ]
        
        self.logger.info("🚀 Distributed orchestration started")
        
        try:
            await asyncio.gather(*orchestration_tasks)
        except asyncio.CancelledError:
            self.logger.info("Orchestration tasks cancelled")
        except Exception as e:
            self.logger.error(f"Orchestration error: {str(e)}")
            raise
    
    async def stop_orchestration(self) -> None:
        """Stop the orchestration system gracefully."""
        self.orchestrator_active = False
        
        # Cancel running tasks
        if self.event_loop:
            for task in asyncio.all_tasks(self.event_loop):
                task.cancel()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.logger.info("⏹️ Distributed orchestration stopped")
    
    def register_node(self, node: ComputeNode) -> bool:
        """Register a new compute node."""
        try:
            # Validate node configuration
            if not node.node_id or not node.address:
                raise ValueError("Node must have valid ID and address")
            
            # Check for conflicts
            if node.node_id in self.nodes:
                self.logger.warning(f"Node {node.node_id} already registered, updating...")
            
            # Update node status
            node.status = NodeStatus.ONLINE
            node.last_heartbeat = time.time()
            
            self.nodes[node.node_id] = node
            
            self.logger.info(
                f"💻 Registered node {node.node_id} at {node.get_url()} "
                f"(qubits: {node.capabilities.get('max_qubits', 'N/A')}, "
                f"mesh: {node.capabilities.get('photonic_mesh_size', 'N/A')})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register node {node.node_id}: {str(e)}")
            return False
    
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a compute node."""
        if node_id not in self.nodes:
            self.logger.warning(f"Attempted to unregister unknown node: {node_id}")
            return False
        
        node = self.nodes[node_id]
        node.status = NodeStatus.OFFLINE
        
        # Reassign active tasks from this node
        tasks_to_reassign = [task for task in self.active_tasks.values() 
                           if task.assigned_node == node_id]
        
        for task in tasks_to_reassign:
            task.assigned_node = None
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                self.task_queue.put(task)
                self.logger.info(f"Reassigned task {task.task_id} from offline node {node_id}")
            else:
                task.error = f"Max retries exceeded after node {node_id} went offline"
                self.completed_tasks.append(task)
                self.metrics['total_tasks_failed'] += 1
        
        del self.nodes[node_id]
        
        self.logger.info(f"🚫 Unregistered node {node_id}")
        return True
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        # Check cache first
        cached_result = self.cache_manager.get(task.task_type, task.payload)
        if cached_result is not None:
            task.result = cached_result
            task.completed_at = time.time()
            self.completed_tasks.append(task)
            self.logger.info(f"⚡ Task {task.task_id} served from cache")
            return task.task_id
        
        # Add to queue
        self.task_queue.put(task)
        self.metrics['total_tasks_submitted'] += 1
        
        self.logger.info(
            f"📎 Submitted task {task.task_id} (type: {task.task_type}, "
            f"priority: {task.priority.name}, queue size: {self.task_queue.qsize()})"
        )
        
        return task.task_id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific task."""
        # Check active tasks
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                'task_id': task_id,
                'status': 'running',
                'assigned_node': task.assigned_node,
                'started_at': task.started_at,
                'estimated_completion': task.started_at + task.timeout_seconds if task.started_at else None
            }
        
        # Check completed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return {
                    'task_id': task_id,
                    'status': 'completed' if task.result is not None else 'failed',
                    'completed_at': task.completed_at,
                    'result': task.result,
                    'error': task.error,
                    'duration_ms': (task.completed_at - task.started_at) * 1000 if task.started_at and task.completed_at else None
                }
        
        # Check queue
        queue_items = []
        temp_queue = PriorityQueue()
        position = 0
        
        while not self.task_queue.empty():
            task = self.task_queue.get()
            queue_items.append(task)
            if task.task_id == task_id:
                break
            position += 1
        
        # Restore queue
        for task in queue_items:
            self.task_queue.put(task)
        
        if any(task.task_id == task_id for task in queue_items):
            return {
                'task_id': task_id,
                'status': 'queued',
                'queue_position': position,
                'estimated_start': time.time() + position * 10  # Rough estimate
            }
        
        return None
    
    async def _task_scheduler_loop(self) -> None:
        """Main task scheduling loop."""
        while self.orchestrator_active:
            try:
                if self.task_queue.empty():
                    await asyncio.sleep(0.1)
                    continue
                
                # Get next task
                task = self.task_queue.get(block=False)
                
                # Check if task has expired
                if task.is_expired():
                    task.error = "Task expired while in queue"
                    task.completed_at = time.time()
                    self.completed_tasks.append(task)
                    self.metrics['total_tasks_failed'] += 1
                    continue
                
                # Select node for execution
                selected_node = self.load_balancer.select_node(self.nodes, task)
                
                if selected_node is None:
                    # No available nodes, put task back in queue
                    self.task_queue.put(task)
                    await asyncio.sleep(1.0)  # Wait before retrying
                    continue
                
                # Assign task to node
                task.assigned_node = selected_node.node_id
                task.started_at = time.time()
                selected_node.active_tasks += 1
                self.active_tasks[task.task_id] = task
                
                # Execute task asynchronously
                asyncio.create_task(self._execute_task_on_node(task, selected_node))
                
                self.logger.info(
                    f"📤 Assigned task {task.task_id} to node {selected_node.node_id}"
                )
                
            except Empty:
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Task scheduler error: {str(e)}")
                await asyncio.sleep(1.0)
    
    async def _execute_task_on_node(self, task: DistributedTask, node: ComputeNode) -> None:
        """Execute a task on a specific node."""
        try:
            # Simulate task execution (in real system, this would make HTTP/gRPC calls)
            execution_result = await self._simulate_task_execution(task, node)
            
            # Handle successful execution
            task.result = execution_result
            task.completed_at = time.time()
            
            # Cache result
            self.cache_manager.put(task.task_type, task.payload, execution_result)
            
            # Update metrics
            duration_ms = (task.completed_at - task.started_at) * 1000
            self._update_task_completion_metrics(node, duration_ms, success=True)
            
            self.logger.info(
                f"✅ Task {task.task_id} completed on node {node.node_id} "
                f"({duration_ms:.1f}ms)"
            )
            
        except Exception as e:
            # Handle task failure
            task.error = str(e)
            task.completed_at = time.time()
            
            self._update_task_completion_metrics(node, 0, success=False)
            
            self.logger.error(
                f"❌ Task {task.task_id} failed on node {node.node_id}: {str(e)}"
            )
        
        finally:
            # Clean up
            node.active_tasks = max(0, node.active_tasks - 1)
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
            self.completed_tasks.append(task)
    
    async def _simulate_task_execution(self, task: DistributedTask, node: ComputeNode) -> Any:
        """Simulate task execution on a node."""
        # Simulate network latency
        await asyncio.sleep(node.metrics.network_latency_ms / 1000.0)
        
        # Simulate computation based on task type
        if task.task_type == 'quantum_compilation':
            result = await self._simulate_quantum_compilation(task, node)
        elif task.task_type == 'photonic_optimization':
            result = await self._simulate_photonic_optimization(task, node)
        elif task.task_type == 'hybrid_simulation':
            result = await self._simulate_hybrid_simulation(task, node)
        elif task.task_type == 'thermal_analysis':
            result = await self._simulate_thermal_analysis(task, node)
        else:
            # Default simulation
            complexity = task.get_complexity_score()
            execution_time = complexity * 0.5 + np.random.exponential(0.2)
            await asyncio.sleep(execution_time)
            result = {
                'status': 'completed',
                'execution_time': execution_time,
                'node_id': node.node_id,
                'task_type': task.task_type
            }
        
        # Simulate potential errors based on node reliability
        error_probability = node.metrics.error_rate
        if np.random.random() < error_probability:
            raise RuntimeError(f"Simulated execution error on node {node.node_id}")
        
        return result
    
    async def _simulate_quantum_compilation(self, task: DistributedTask, node: ComputeNode) -> Dict[str, Any]:
        """Simulate quantum compilation task."""
        compilation_time = 2.0 + np.random.exponential(1.0)
        await asyncio.sleep(compilation_time)
        
        return {
            'compiled_gates': np.random.randint(50, 500),
            'quantum_volume': node.capabilities.get('max_qubits', 32) ** 2,
            'fidelity': node.quantum_fidelity * np.random.uniform(0.95, 1.0),
            'compilation_time': compilation_time,
            'optimizations_applied': ['phase_optimization', 'thermal_compensation']
        }
    
    async def _simulate_photonic_optimization(self, task: DistributedTask, node: ComputeNode) -> Dict[str, Any]:
        """Simulate photonic optimization task."""
        optimization_time = 1.5 + np.random.exponential(0.8)
        await asyncio.sleep(optimization_time)
        
        mesh_size = node.capabilities.get('photonic_mesh_size', (64, 64))
        
        return {
            'mesh_utilization': np.random.uniform(0.7, 0.95),
            'wavelength_efficiency': np.random.uniform(0.8, 0.98),
            'power_reduction': np.random.uniform(10, 30),  # Percentage
            'phase_shifts_optimized': np.random.randint(100, 1000),
            'mesh_size': mesh_size,
            'optimization_time': optimization_time
        }
    
    async def _simulate_hybrid_simulation(self, task: DistributedTask, node: ComputeNode) -> Dict[str, Any]:
        """Simulate hybrid quantum-photonic simulation."""
        simulation_time = 3.0 + np.random.exponential(2.0)
        await asyncio.sleep(simulation_time)
        
        return {
            'quantum_speedup': np.random.uniform(2.0, 8.0),
            'photonic_efficiency': node.photonic_efficiency * np.random.uniform(0.9, 1.0),
            'coherence_preservation': np.random.uniform(0.85, 0.99),
            'simulation_accuracy': np.random.uniform(0.95, 0.999),
            'simulation_time': simulation_time,
            'hybrid_operations': np.random.randint(1000, 10000)
        }
    
    async def _simulate_thermal_analysis(self, task: DistributedTask, node: ComputeNode) -> Dict[str, Any]:
        """Simulate thermal analysis task."""
        analysis_time = 0.5 + np.random.exponential(0.3)
        await asyncio.sleep(analysis_time)
        
        return {
            'thermal_hotspots': np.random.randint(0, 5),
            'max_temperature': 25.0 + np.random.uniform(10, 40),
            'thermal_efficiency': np.random.uniform(0.8, 0.95),
            'cooling_recommendations': ['increase_airflow', 'reduce_power'],
            'analysis_time': analysis_time
        }
    
    def _update_task_completion_metrics(self, node: ComputeNode, duration_ms: float, success: bool) -> None:
        """Update node and cluster metrics after task completion."""
        # Update node metrics
        node.total_tasks_processed += 1
        if success:
            # Update average task duration
            if node.avg_task_duration_ms == 0:
                node.avg_task_duration_ms = duration_ms
            else:
                node.avg_task_duration_ms = (
                    (node.avg_task_duration_ms * (node.total_tasks_processed - 1) + duration_ms) /
                    node.total_tasks_processed
                )
        
        # Update cluster metrics
        if success:
            self.metrics['total_tasks_completed'] += 1
            
            # Update average task duration
            total_completed = self.metrics['total_tasks_completed']
            if self.metrics['average_task_duration_ms'] == 0:
                self.metrics['average_task_duration_ms'] = duration_ms
            else:
                self.metrics['average_task_duration_ms'] = (
                    (self.metrics['average_task_duration_ms'] * (total_completed - 1) + duration_ms) /
                    total_completed
                )
        else:
            self.metrics['total_tasks_failed'] += 1
    
    async def _node_health_monitor_loop(self) -> None:
        """Monitor node health and update status."""
        while self.orchestrator_active:
            try:
                current_time = time.time()
                unhealthy_nodes = []
                
                for node_id, node in self.nodes.items():
                    # Check heartbeat timeout
                    if not node.is_healthy(self.node_timeout):
                        if node.status == NodeStatus.ONLINE:
                            node.status = NodeStatus.OFFLINE
                            unhealthy_nodes.append(node_id)
                            self.logger.warning(f"⚠️ Node {node_id} marked as unhealthy")
                    
                    # Simulate resource metrics updates
                    self._update_node_metrics(node)
                
                # Handle unhealthy nodes
                for node_id in unhealthy_nodes:
                    # Could trigger auto-scaling or alerting here
                    pass
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Node health monitor error: {str(e)}")
                await asyncio.sleep(5.0)
    
    def _update_node_metrics(self, node: ComputeNode) -> None:
        """Update node resource metrics (simulated)."""
        # Simulate metric updates
        base_load = node.active_tasks / node.capabilities.get('max_concurrent_tasks', 4)
        
        node.metrics.cpu_percent = min(95, base_load * 80 + np.random.normal(0, 5))
        node.metrics.memory_percent = min(95, base_load * 70 + np.random.normal(0, 5))
        node.metrics.gpu_percent = min(95, base_load * 60 + np.random.normal(0, 10))
        node.metrics.quantum_coherence = max(0.5, 1.0 - base_load * 0.3 + np.random.normal(0, 0.05))
        node.metrics.photonic_mesh_utilization = min(1.0, base_load * 0.8 + np.random.uniform(0, 0.2))
        node.metrics.thermal_stress = min(1.0, base_load * 0.6 + np.random.uniform(0, 0.1))
        node.metrics.network_latency_ms = 1.0 + np.random.exponential(2.0)
        node.metrics.throughput_ops_per_sec = max(1, 100 - base_load * 50 + np.random.normal(0, 10))
        node.metrics.error_rate = min(0.1, base_load * 0.05 + np.random.exponential(0.01))
        node.metrics.last_updated = time.time()
    
    async def _auto_scaling_loop(self) -> None:
        """Auto-scaling management loop."""
        while self.orchestrator_active:
            try:
                should_scale, direction, target_nodes = self.auto_scaler.should_scale(
                    self.nodes, self.task_queue.qsize()
                )
                
                if should_scale:
                    current_nodes = len([n for n in self.nodes.values() 
                                       if n.status == NodeStatus.ONLINE])
                    
                    if direction == 'up' and target_nodes > current_nodes:
                        await self._scale_up(target_nodes - current_nodes)
                    elif direction == 'down' and target_nodes < current_nodes:
                        await self._scale_down(current_nodes - target_nodes)
                    
                    self.auto_scaler.record_scaling_action(direction, current_nodes, target_nodes)
                    self.metrics['scaling_events'] += 1
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaling error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _scale_up(self, num_nodes: int) -> None:
        """Scale up by adding new nodes."""
        self.logger.info(f"🔼 Scaling up: adding {num_nodes} nodes")
        
        for i in range(num_nodes):
            # Create simulated node
            node_id = f"auto_node_{uuid.uuid4().hex[:8]}"
            new_node = ComputeNode(
                node_id=node_id,
                address=f"192.168.1.{100 + len(self.nodes)}",
                port=8080 + len(self.nodes)
            )
            
            self.register_node(new_node)
    
    async def _scale_down(self, num_nodes: int) -> None:
        """Scale down by removing nodes."""
        self.logger.info(f"🔽 Scaling down: removing {num_nodes} nodes")
        
        # Select nodes to remove (prefer nodes with lowest load)
        online_nodes = {nid: node for nid, node in self.nodes.items() 
                       if node.status == NodeStatus.ONLINE}
        
        if len(online_nodes) <= self.auto_scaler.min_nodes:
            return
        
        # Sort by load and remove least loaded nodes
        nodes_by_load = sorted(online_nodes.values(), 
                              key=lambda n: n.metrics.get_overall_load())
        
        nodes_to_remove = nodes_by_load[:min(num_nodes, len(nodes_by_load) - self.auto_scaler.min_nodes)]
        
        for node in nodes_to_remove:
            # Wait for active tasks to complete
            while node.active_tasks > 0:
                await asyncio.sleep(1.0)
            
            self.unregister_node(node.node_id)
    
    async def _metrics_collection_loop(self) -> None:
        """Collect and update cluster metrics."""
        while self.orchestrator_active:
            try:
                # Update cluster metrics
                online_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
                
                if online_nodes:
                    self.metrics['cluster_utilization'] = np.mean([
                        node.metrics.get_overall_load() for node in online_nodes
                    ])
                
                self.metrics['cache_hit_rate'] = self.cache_manager.get_stats()['hit_rate']
                self.metrics['uptime_seconds'] = time.time() - self.start_time
                
                await asyncio.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Metrics collection error: {str(e)}")
                await asyncio.sleep(30)
    
    async def _cluster_coordination_loop(self) -> None:
        """Handle cluster coordination and leader election."""
        while self.orchestrator_active:
            try:
                # Placeholder for cluster coordination logic
                # In a real system, this would handle:
                # - Leader election
                # - Node discovery
                # - Cluster state synchronization
                # - Partition handling
                
                await asyncio.sleep(60)  # Coordination check every minute
                
            except Exception as e:
                self.logger.error(f"Cluster coordination error: {str(e)}")
                await asyncio.sleep(120)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        online_nodes = [n for n in self.nodes.values() if n.status == NodeStatus.ONLINE]
        
        node_details = {
            node.node_id: {
                'status': node.status.value,
                'address': node.get_url(),
                'active_tasks': node.active_tasks,
                'total_processed': node.total_tasks_processed,
                'avg_duration_ms': node.avg_task_duration_ms,
                'load': node.metrics.get_overall_load(),
                'quantum_coherence': node.metrics.quantum_coherence,
                'thermal_stress': node.metrics.thermal_stress,
                'capabilities': node.capabilities
            }
            for node in self.nodes.values()
        }
        
        return {
            'cluster_summary': {
                'total_nodes': len(self.nodes),
                'online_nodes': len(online_nodes),
                'active_tasks': len(self.active_tasks),
                'queued_tasks': self.task_queue.qsize(),
                'completed_tasks': len(self.completed_tasks),
                'average_load': np.mean([n.metrics.get_overall_load() for n in online_nodes]) if online_nodes else 0.0
            },
            'performance_metrics': self.metrics.copy(),
            'auto_scaling_status': {
                'mode': self.auto_scaler.mode.value,
                'last_action': self.auto_scaler.last_scale_action,
                'scaling_history': list(self.auto_scaler.scaling_history)
            },
            'cache_stats': self.cache_manager.get_stats(),
            'load_balancing': {
                'strategy': self.load_balancer.strategy.value,
                'selection_history': list(self.load_balancer.node_selection_history)
            },
            'node_details': node_details
        }
    
    def export_cluster_diagnostics(self, output_path: str) -> None:
        """Export comprehensive cluster diagnostics."""
        diagnostics = {
            'timestamp': time.time(),
            'cluster_status': self.get_cluster_status(),
            'configuration': self.config,
            'recent_completed_tasks': [
                {
                    'task_id': task.task_id,
                    'task_type': task.task_type,
                    'duration_ms': (task.completed_at - task.started_at) * 1000 if task.started_at and task.completed_at else None,
                    'assigned_node': task.assigned_node,
                    'success': task.result is not None
                }
                for task in list(self.completed_tasks)[-100:]  # Last 100 tasks
            ]
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(diagnostics, f, indent=2, default=str)
        
        self.logger.info(f"📋 Cluster diagnostics exported to {output_path}")


# Convenience functions for creating distributed orchestrator
def create_distributed_orchestrator(
    config: Optional[Dict[str, Any]] = None
) -> DistributedQuantumPhotonicOrchestrator:
    """Create a distributed quantum-photonic orchestrator.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Configured DistributedQuantumPhotonicOrchestrator instance
    """
    default_config = {
        'load_balancing_strategy': 'quantum_coherence_aware',
        'auto_scaling': {
            'mode': 'hybrid_intelligent',
            'min_nodes': 2,
            'max_nodes': 50,
            'scale_up_threshold': 0.8,
            'scale_down_threshold': 0.3
        },
        'cache_size_mb': 2048,
        'cluster_port': 8080,
        'heartbeat_interval_s': 5.0,
        'node_timeout_s': 30.0,
        'max_workers': 32
    }
    
    if config:
        # Deep merge configuration
        for key, value in config.items():
            if key in default_config and isinstance(default_config[key], dict):
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return DistributedQuantumPhotonicOrchestrator(default_config)


def create_compute_node(
    node_id: str,
    address: str,
    port: int = 8080,
    capabilities: Optional[Dict[str, Any]] = None
) -> ComputeNode:
    """Create a compute node with specified capabilities.
    
    Args:
        node_id: Unique identifier for the node
        address: Network address of the node
        port: Port number for communication
        capabilities: Node capabilities dictionary
    
    Returns:
        Configured ComputeNode instance
    """
    return ComputeNode(
        node_id=node_id,
        address=address,
        port=port,
        capabilities=capabilities or {}
    )


def create_distributed_task(
    task_type: str,
    payload: Dict[str, Any],
    priority: TaskPriority = TaskPriority.NORMAL,
    requirements: Optional[Dict[str, float]] = None,
    timeout_seconds: float = 300.0
) -> DistributedTask:
    """Create a distributed task.
    
    Args:
        task_type: Type of task to execute
        payload: Task payload data
        priority: Task priority level
        requirements: Resource requirements
        timeout_seconds: Task timeout
    
    Returns:
        Configured DistributedTask instance
    """
    return DistributedTask(
        task_id=f"task_{uuid.uuid4().hex[:12]}",
        task_type=task_type,
        priority=priority,
        payload=payload,
        requirements=requirements or {},
        timeout_seconds=timeout_seconds
    )
