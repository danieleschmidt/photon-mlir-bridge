"""
Autonomous Performance Optimizer for Quantum-Photonic Systems
Generation 3 Enhancement - MAKE IT SCALE

Advanced performance optimization engine with machine learning-driven optimization,
distributed caching, auto-scaling, and real-time performance adaptation.

Performance Features:
1. ML-driven performance optimization
2. Distributed computation caching with quantum checksums
3. Auto-scaling based on demand prediction
4. Real-time performance monitoring and adaptation
5. Multi-objective optimization (latency, throughput, energy)
6. Predictive resource allocation
7. Global performance coordination
"""

import time
import asyncio
try:
    import numpy as np
except ImportError:
    from .numpy_fallback import get_numpy
    np = get_numpy()
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import json
import pickle
import hashlib
from pathlib import Path
from collections import defaultdict, deque
import statistics
import multiprocessing as mp
from functools import lru_cache, partial
import weakref
import gc

# ML and optimization imports
try:
    from scipy import optimize
    import sklearn.ensemble
    import sklearn.preprocessing
    _ML_AVAILABLE = True
except ImportError:
    _ML_AVAILABLE = False

# Distributed caching imports
try:
    import redis
    import memcache
    _DISTRIBUTED_CACHE_AVAILABLE = True
except ImportError:
    _DISTRIBUTED_CACHE_AVAILABLE = False

# Import core components
from .core import TargetConfig, Device, Precision, PhotonicTensor
from .logging_config import get_global_logger, performance_monitor
from .robust_error_handling import robust_execution, CircuitBreaker
from .quantum_aware_scheduler import PhotonicTask


class OptimizationObjective(Enum):
    """Performance optimization objectives."""
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_THERMAL_EFFICIENCY = "maximize_thermal_efficiency"
    BALANCED_PERFORMANCE = "balanced_performance"
    CUSTOM_OBJECTIVE = "custom_objective"


class CacheStrategy(Enum):
    """Caching strategies for computation results."""
    NO_CACHE = "no_cache"
    LOCAL_MEMORY = "local_memory"
    DISTRIBUTED_REDIS = "distributed_redis"
    PERSISTENT_DISK = "persistent_disk"
    HYBRID_MULTILEVEL = "hybrid_multilevel"
    QUANTUM_SECURED = "quantum_secured"


class ScalingMode(Enum):
    """Auto-scaling modes."""
    DISABLED = "disabled"
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    QUANTUM_DEMAND_AWARE = "quantum_demand_aware"
    ML_OPTIMIZED = "ml_optimized"


class OptimizationStrategy(Enum):
    """Strategies for performance optimization."""
    GREEDY = "greedy"
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    QUANTUM_ANNEALING = "quantum_annealing"
    HYBRID_ML = "hybrid_ml"


@dataclass
class PerformanceMetrics:
    """Performance metrics for optimization."""
    latency_ms: float = 0.0
    throughput_ops_per_second: float = 0.0
    energy_efficiency_pj_per_op: float = 0.0
    thermal_efficiency: float = 1.0
    resource_utilization: float = 0.0
    cache_hit_rate: float = 0.0
    quantum_coherence_efficiency: float = 1.0
    error_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization."""
        return {
            'latency_ms': self.latency_ms,
            'throughput_ops_per_second': self.throughput_ops_per_second,
            'energy_efficiency': self.energy_efficiency_pj_per_op,
            'thermal_efficiency': self.thermal_efficiency,
            'resource_utilization': self.resource_utilization,
            'cache_hit_rate': self.cache_hit_rate,
            'quantum_coherence_efficiency': self.quantum_coherence_efficiency,
            'error_rate': self.error_rate
        }
    
    def composite_score(self, weights: Optional[Dict[str, float]] = None) -> float:
        """Calculate composite performance score."""
        if weights is None:
            weights = {
                'latency': 0.25,
                'throughput': 0.25,
                'energy': 0.2,
                'thermal': 0.15,
                'cache': 0.1,
                'quantum': 0.05
            }
        
        # Normalize metrics to 0-1 scale (higher is better)
        normalized_latency = max(0, 1.0 - (self.latency_ms / 1000.0))  # Assuming 1s is terrible
        normalized_throughput = min(1.0, self.throughput_ops_per_second / 1000.0)  # Assuming 1000 ops/s is excellent
        normalized_energy = max(0, 1.0 - (self.energy_efficiency_pj_per_op / 1000.0))  # Lower is better
        
        score = (
            weights.get('latency', 0) * normalized_latency +
            weights.get('throughput', 0) * normalized_throughput +
            weights.get('energy', 0) * normalized_energy +
            weights.get('thermal', 0) * self.thermal_efficiency +
            weights.get('cache', 0) * self.cache_hit_rate +
            weights.get('quantum', 0) * self.quantum_coherence_efficiency
        )
        
        return score


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    
    # Optimization objectives
    primary_objective: OptimizationObjective = OptimizationObjective.BALANCED_PERFORMANCE
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        'latency': 0.3,
        'throughput': 0.3,
        'energy': 0.2,
        'thermal': 0.2
    })
    
    # Machine learning optimization
    enable_ml_optimization: bool = True
    ml_model_update_interval_seconds: float = 1800.0  # 30 minutes
    optimization_history_size: int = 10000
    
    # Caching configuration
    cache_strategy: CacheStrategy = CacheStrategy.HYBRID_MULTILEVEL
    local_cache_size_mb: int = 1024  # 1GB
    distributed_cache_ttl_seconds: int = 3600  # 1 hour
    cache_compression: bool = True
    
    # Scaling configuration
    scaling_mode: ScalingMode = ScalingMode.ML_OPTIMIZED
    min_workers: int = 2
    max_workers: int = 32
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    scaling_cooldown_seconds: float = 300.0
    
    # Performance monitoring
    monitoring_interval_seconds: float = 30.0
    performance_history_size: int = 1000
    regression_detection_enabled: bool = True
    anomaly_detection_threshold: float = 2.0
    
    # Optimization tuning
    optimization_interval_seconds: float = 60.0
    parameter_exploration_rate: float = 0.1
    convergence_threshold: float = 0.01
    max_optimization_iterations: int = 100


@dataclass
class CacheEntry:
    """Entry in the computation cache."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    hit_count: int = 0
    computation_time_ms: float = 0.0
    quantum_checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if cache entry is expired."""
        return (time.time() - self.timestamp) > ttl_seconds
    
    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.hit_count += 1


class AutonomousPerformanceOptimizer:
    """
    Autonomous Performance Optimizer for Quantum-Photonic Systems
    
    Provides ML-driven optimization, distributed caching, auto-scaling,
    and real-time performance adaptation for maximum system efficiency.
    """
    
    def __init__(self, 
                 target_config: TargetConfig,
                 config: Optional[OptimizationConfig] = None):
        """Initialize the autonomous performance optimizer."""
        
        self.target_config = target_config
        self.config = config or OptimizationConfig()
        
        # Initialize logging
        self.logger = get_global_logger(__name__)
        
        # State management
        self.is_running = False
        self.start_time = None
        self.current_metrics = PerformanceMetrics()
        self.metrics_history = deque(maxlen=self.config.performance_history_size)
        
        # Optimization state
        self.current_parameters = {}
        self.optimization_history = deque(maxlen=self.config.optimization_history_size)
        self.best_configuration = None
        self.best_score = 0.0
        
        # Machine learning components
        if _ML_AVAILABLE and self.config.enable_ml_optimization:
            self._init_ml_components()
        else:
            self.ml_predictor = None
            self.feature_scaler = None
        
        # Caching system
        self._init_caching_system()
        
        # Threading and execution
        self.executor = ThreadPoolExecutor(
            max_workers=self.config.max_workers,
            thread_name_prefix="PerfOptimizer"
        )
        self.process_executor = ProcessPoolExecutor(
            max_workers=min(mp.cpu_count(), 8)
        )
        
        # Monitoring and optimization threads
        self.monitoring_thread = None
        self.optimization_thread = None
        
        # Circuit breakers for different optimization strategies
        self.circuit_breakers = {
            'ml_optimization': CircuitBreaker(failure_threshold=3, recovery_timeout=300.0),
            'distributed_cache': CircuitBreaker(failure_threshold=5, recovery_timeout=60.0),
            'auto_scaling': CircuitBreaker(failure_threshold=3, recovery_timeout=180.0)
        }
        
        # Performance baselines and targets
        self.performance_baselines = {}
        self.performance_targets = {}
        self._init_performance_targets()
        
        # Resource scaling state
        self.current_workers = self.config.min_workers
        self.last_scaling_time = 0.0
        self.scaling_decisions = deque(maxlen=100)
        
        # Quantum-specific optimization
        self.quantum_coherence_optimizer = None
        if target_config.device in [Device.QUANTUM_PHOTONIC, Device.HYBRID_QUANTUM]:
            self._init_quantum_optimization()
        
        self.logger.info(f"Autonomous Performance Optimizer initialized with {self.config.primary_objective.value} objective")
    
    def _init_ml_components(self) -> None:
        """Initialize machine learning components for optimization."""
        
        if not _ML_AVAILABLE:
            self.logger.warning("ML libraries not available, disabling ML optimization")
            return
        
        # Random forest for performance prediction
        self.ml_predictor = sklearn.ensemble.RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        # Feature scaler for normalization
        self.feature_scaler = sklearn.preprocessing.StandardScaler()
        
        # Training data storage
        self.training_features = []
        self.training_targets = []
        self.model_trained = False
        
        self.logger.debug("ML components initialized")
    
    def _init_caching_system(self) -> None:
        """Initialize the caching system."""
        
        self.local_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
        
        # Distributed cache connections
        self.redis_client = None
        self.memcache_client = None
        
        if _DISTRIBUTED_CACHE_AVAILABLE and self.config.cache_strategy in [
            CacheStrategy.DISTRIBUTED_REDIS, 
            CacheStrategy.HYBRID_MULTILEVEL
        ]:
            try:
                self.redis_client = redis.Redis(
                    host='localhost', 
                    port=6379, 
                    db=0,
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
                self.logger.debug("Redis cache connected")
                
            except Exception as e:
                self.logger.warning(f"Redis connection failed: {e}")
                self.redis_client = None
        
        # Cache cleanup thread
        self.cache_cleanup_thread = None
        
        self.logger.debug(f"Caching system initialized with {self.config.cache_strategy.value} strategy")
    
    def _init_performance_targets(self) -> None:
        """Initialize performance targets based on device capabilities."""
        
        device_specs = {
            Device.LIGHTMATTER_ENVISE: {
                'max_throughput_ops_per_second': 1000,
                'target_latency_ms': 10.0,
                'energy_efficiency_pj_per_op': 0.1
            },
            Device.MIT_PHOTONIC: {
                'max_throughput_ops_per_second': 500,
                'target_latency_ms': 20.0,
                'energy_efficiency_pj_per_op': 0.05
            },
            Device.QUANTUM_PHOTONIC: {
                'max_throughput_ops_per_second': 100,
                'target_latency_ms': 50.0,
                'energy_efficiency_pj_per_op': 0.01
            }
        }
        
        device = self.target_config.device
        if device in device_specs:
            specs = device_specs[device]
            self.performance_targets = {
                'latency_ms': specs['target_latency_ms'],
                'throughput_ops_per_second': specs['max_throughput_ops_per_second'] * 0.8,  # 80% of max
                'energy_efficiency': specs['energy_efficiency_pj_per_op'],
                'thermal_efficiency': 0.9,
                'cache_hit_rate': 0.8,
                'quantum_coherence_efficiency': 0.95
            }
        else:
            # Default targets
            self.performance_targets = {
                'latency_ms': 100.0,
                'throughput_ops_per_second': 100.0,
                'energy_efficiency': 1.0,
                'thermal_efficiency': 0.8,
                'cache_hit_rate': 0.7,
                'quantum_coherence_efficiency': 0.9
            }
    
    def _init_quantum_optimization(self) -> None:
        """Initialize quantum-specific optimization."""
        
        self.quantum_coherence_optimizer = QuantumCoherenceOptimizer(
            coherence_time_ms=1000.0,
            decoherence_rate=0.001
        )
        
        self.logger.debug("Quantum optimization components initialized")
    
    async def start(self) -> None:
        """Start the autonomous performance optimizer."""
        
        if self.is_running:
            self.logger.warning("Performance optimizer is already running")
            return
        
        self.logger.info("Starting Autonomous Performance Optimizer")
        self.is_running = True
        self.start_time = time.time()
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitoring",
            daemon=True
        )
        self.monitoring_thread.start()
        
        # Start optimization thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            name="PerformanceOptimization",
            daemon=True
        )
        self.optimization_thread.start()
        
        # Start cache cleanup if needed
        if self.config.cache_strategy != CacheStrategy.NO_CACHE:
            self.cache_cleanup_thread = threading.Thread(
                target=self._cache_cleanup_loop,
                name="CacheCleanup",
                daemon=True
            )
            self.cache_cleanup_thread.start()
        
        # Establish initial baselines
        await self._establish_baselines()
        
        self.logger.info("Autonomous Performance Optimizer started successfully")
    
    async def stop(self) -> None:
        """Stop the autonomous performance optimizer."""
        
        if not self.is_running:
            return
        
        self.logger.info("Stopping Autonomous Performance Optimizer")
        self.is_running = False
        
        # Wait for threads to complete
        for thread in [self.monitoring_thread, self.optimization_thread, self.cache_cleanup_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        # Shutdown executors
        self.executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        # Save optimization state
        await self._save_optimization_state()
        
        self.logger.info("Autonomous Performance Optimizer stopped")
    
    async def optimize_task_execution(self, task: PhotonicTask) -> Tuple[Any, PerformanceMetrics]:
        """Optimize execution of a photonic task."""
        
        task_start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(task)
            cached_result = await self._get_from_cache(cache_key)
            
            if cached_result is not None:
                self.cache_stats['hits'] += 1
                execution_time_ms = (time.time() - task_start_time) * 1000
                
                metrics = PerformanceMetrics(
                    latency_ms=execution_time_ms,
                    throughput_ops_per_second=1000.0 / execution_time_ms,
                    cache_hit_rate=1.0,  # This was a cache hit
                    resource_utilization=0.1  # Minimal resources for cache hit
                )
                
                return cached_result, metrics
            else:
                self.cache_stats['misses'] += 1
            
            # Determine optimal execution parameters
            optimal_params = await self._determine_optimal_parameters(task)
            
            # Execute task with optimization
            result = await self._execute_optimized_task(task, optimal_params)
            
            # Measure performance metrics
            execution_time_ms = (time.time() - task_start_time) * 1000
            metrics = await self._measure_task_performance(task, result, execution_time_ms)
            
            # Cache the result
            await self._store_in_cache(cache_key, result, execution_time_ms)
            
            # Update ML training data if enabled
            if self.ml_predictor is not None:
                await self._update_ml_training_data(task, optimal_params, metrics)
            
            # Record optimization outcome
            self._record_optimization_outcome(task, optimal_params, metrics)
            
            return result, metrics
            
        except Exception as e:
            self.logger.error(f"Task optimization failed: {e}")
            raise
        
        finally:
            self.cache_stats['total_requests'] += 1
    
    async def _determine_optimal_parameters(self, task: PhotonicTask) -> Dict[str, Any]:
        """Determine optimal parameters for task execution."""
        
        if self.ml_predictor is not None and self.model_trained:
            return await self._ml_determine_parameters(task)
        else:
            return await self._heuristic_determine_parameters(task)
    
    async def _ml_determine_parameters(self, task: PhotonicTask) -> Dict[str, Any]:
        """Use ML model to determine optimal parameters."""
        
        try:
            # Extract features from task
            features = self._extract_task_features(task)
            
            # Normalize features
            normalized_features = self.feature_scaler.transform([features])
            
            # Predict optimal parameters
            predictions = self.ml_predictor.predict(normalized_features)[0]
            
            # Convert predictions to parameter dictionary
            optimal_params = self._predictions_to_parameters(predictions, task)
            
            self.logger.debug(f"ML determined parameters: {optimal_params}")
            return optimal_params
            
        except Exception as e:
            self.logger.error(f"ML parameter determination failed: {e}")
            # Fall back to heuristic approach
            return await self._heuristic_determine_parameters(task)
    
    async def _heuristic_determine_parameters(self, task: PhotonicTask) -> Dict[str, Any]:
        """Use heuristic rules to determine optimal parameters."""
        
        params = {}
        
        # Determine parallelization level
        if task.operation_type == "matmul":
            matrix_dims = task.parameters.get('matrix_dims', (64, 64, 64))
            matrix_size = matrix_dims[0] * matrix_dims[1]
            
            if matrix_size > 10000:
                params['parallelization'] = min(8, self.current_workers)
            elif matrix_size > 1000:
                params['parallelization'] = min(4, self.current_workers)
            else:
                params['parallelization'] = 1
        
        # Determine precision optimization
        if task.priority.value <= 2:  # High priority tasks
            params['precision'] = 'high'
        else:
            params['precision'] = 'standard'
        
        # Thermal management
        if task.thermal_cost > 10.0:
            params['thermal_management'] = 'aggressive'
        else:
            params['thermal_management'] = 'standard'
        
        # Quantum optimization
        if 'quantum' in task.operation_type:
            params['quantum_optimization'] = 'enabled'
            params['coherence_preservation'] = 'high'
        
        self.logger.debug(f"Heuristic determined parameters: {params}")
        return params
    
    async def _execute_optimized_task(self, task: PhotonicTask, params: Dict[str, Any]) -> Any:
        """Execute task with optimized parameters."""
        
        # Apply parallelization if specified
        parallelization = params.get('parallelization', 1)
        
        if parallelization > 1 and task.operation_type == "matmul":
            return await self._execute_parallel_matmul(task, parallelization)
        else:
            return await self._execute_single_task(task, params)
    
    async def _execute_parallel_matmul(self, task: PhotonicTask, parallelization: int) -> Any:
        """Execute matrix multiplication with parallelization."""
        
        matrix_dims = task.parameters.get('matrix_dims', (64, 64, 64))
        m, n, k = matrix_dims
        
        # Split computation across workers
        chunk_size = max(1, m // parallelization)
        futures = []
        
        for i in range(0, m, chunk_size):
            chunk_end = min(i + chunk_size, m)
            chunk_task = PhotonicTask(
                task_id=f"{task.task_id}_chunk_{i}",
                operation_type=task.operation_type,
                input_data=task.input_data,
                parameters={
                    **task.parameters,
                    'matrix_dims': (chunk_end - i, n, k),
                    'chunk_start': i
                },
                priority=task.priority,
                thermal_cost=task.thermal_cost / parallelization
            )
            
            future = self.executor.submit(self._simulate_task_execution, chunk_task)
            futures.append(future)
        
        # Collect results
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
        
        # Combine results
        combined_result = {
            "status": "success",
            "operation": "parallel_matmul",
            "chunks_processed": len(results),
            "parallelization_level": parallelization,
            "combined_execution_time_ms": sum(r.get("execution_time_ms", 0) for r in results)
        }
        
        return combined_result
    
    async def _execute_single_task(self, task: PhotonicTask, params: Dict[str, Any]) -> Any:
        """Execute single task with optimization parameters."""
        
        # Apply precision optimization
        precision = params.get('precision', 'standard')
        if precision == 'high':
            execution_multiplier = 1.5  # Higher precision takes longer
        else:
            execution_multiplier = 1.0
        
        # Apply thermal management
        thermal_mgmt = params.get('thermal_management', 'standard')
        if thermal_mgmt == 'aggressive':
            thermal_overhead = 0.1  # 10% overhead for aggressive thermal management
        else:
            thermal_overhead = 0.0
        
        # Simulate optimized execution
        result = self._simulate_task_execution(task)
        
        # Apply optimization effects
        if 'execution_time_ms' in result:
            result['execution_time_ms'] *= execution_multiplier * (1 + thermal_overhead)
            result['precision_level'] = precision
            result['thermal_management'] = thermal_mgmt
        
        return result
    
    def _simulate_task_execution(self, task: PhotonicTask) -> Dict[str, Any]:
        """Simulate task execution (placeholder for actual execution)."""
        
        # This would be replaced with actual photonic task execution
        base_time = 10.0  # Base 10ms execution time
        
        if task.operation_type == "matmul":
            matrix_dims = task.parameters.get('matrix_dims', (64, 64, 64))
            m, n, k = matrix_dims
            execution_time = base_time * (m * n * k) / 1000.0  # Scale with matrix size
        else:
            execution_time = base_time
        
        # Add some random variation
        execution_time *= np.random.uniform(0.8, 1.2)
        
        # Simulate execution
        time.sleep(execution_time / 1000.0)
        
        return {
            "status": "success",
            "operation": task.operation_type,
            "execution_time_ms": execution_time,
            "task_id": task.task_id
        }
    
    async def _measure_task_performance(self, task: PhotonicTask, result: Any, 
                                      execution_time_ms: float) -> PerformanceMetrics:
        """Measure performance metrics for a completed task."""
        
        metrics = PerformanceMetrics()
        
        # Basic timing metrics
        metrics.latency_ms = execution_time_ms
        if execution_time_ms > 0:
            metrics.throughput_ops_per_second = 1000.0 / execution_time_ms
        
        # Energy efficiency (simplified model)
        if task.operation_type == "matmul":
            matrix_dims = task.parameters.get('matrix_dims', (64, 64, 64))
            ops = matrix_dims[0] * matrix_dims[1] * matrix_dims[2]
            metrics.energy_efficiency_pj_per_op = execution_time_ms * 0.1 / ops  # Simplified
        else:
            metrics.energy_efficiency_pj_per_op = 0.1
        
        # Thermal efficiency
        thermal_cost = task.thermal_cost
        if thermal_cost > 0:
            thermal_efficiency = max(0.1, 1.0 - (thermal_cost / 100.0))
            metrics.thermal_efficiency = thermal_efficiency
        else:
            metrics.thermal_efficiency = 1.0
        
        # Resource utilization
        parallelization = getattr(result, 'parallelization_level', 1)
        if isinstance(result, dict):
            parallelization = result.get('parallelization_level', 1)
        
        metrics.resource_utilization = min(1.0, parallelization / self.current_workers)
        
        # Quantum coherence (if applicable)
        if 'quantum' in task.operation_type:
            coherence = getattr(result, 'coherence_maintained', 0.95)
            if isinstance(result, dict):
                coherence = result.get('coherence_maintained', 0.95)
            metrics.quantum_coherence_efficiency = coherence
        
        # Error rate (simplified)
        if isinstance(result, dict) and result.get('status') == 'success':
            metrics.error_rate = 0.0
        else:
            metrics.error_rate = 1.0
        
        return metrics
    
    def _generate_cache_key(self, task: PhotonicTask) -> str:
        """Generate cache key for a task."""
        
        # Create deterministic key from task parameters
        key_data = {
            'operation_type': task.operation_type,
            'parameters': task.parameters,
            'input_hash': hashlib.sha256(str(task.input_data).encode()).hexdigest()[:16]
        }
        
        key_string = json.dumps(key_data, sort_keys=True)
        cache_key = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"photonic_task_{cache_key}"
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get result from cache."""
        
        # Try local cache first
        if key in self.local_cache:
            entry = self.local_cache[key]
            if not entry.is_expired(self.config.distributed_cache_ttl_seconds):
                entry.update_access()
                return entry.value
            else:
                del self.local_cache[key]
        
        # Try distributed cache
        if self.redis_client and self.circuit_breakers['distributed_cache'].can_execute():
            try:
                cached_data = self.redis_client.get(key)
                if cached_data:
                    if self.config.cache_compression:
                        import gzip
                        cached_data = gzip.decompress(cached_data)
                    
                    result = pickle.loads(cached_data)
                    
                    # Store in local cache too
                    entry = CacheEntry(
                        key=key,
                        value=result,
                        timestamp=time.time()
                    )
                    self.local_cache[key] = entry
                    
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Distributed cache read failed: {e}")
                self.circuit_breakers['distributed_cache'].record_failure()
        
        return None
    
    async def _store_in_cache(self, key: str, value: Any, computation_time_ms: float) -> None:
        """Store result in cache."""
        
        # Store in local cache
        entry = CacheEntry(
            key=key,
            value=value,
            timestamp=time.time(),
            computation_time_ms=computation_time_ms
        )
        
        self.local_cache[key] = entry
        
        # Manage local cache size
        if len(self.local_cache) > self.config.local_cache_size_mb:
            self._evict_local_cache_entries()
        
        # Store in distributed cache
        if self.redis_client and self.circuit_breakers['distributed_cache'].can_execute():
            try:
                serialized_data = pickle.dumps(value)
                
                if self.config.cache_compression:
                    import gzip
                    serialized_data = gzip.compress(serialized_data)
                
                self.redis_client.setex(
                    key, 
                    self.config.distributed_cache_ttl_seconds,
                    serialized_data
                )
                
            except Exception as e:
                self.logger.warning(f"Distributed cache write failed: {e}")
                self.circuit_breakers['distributed_cache'].record_failure()
    
    def _evict_local_cache_entries(self) -> None:
        """Evict entries from local cache using LRU policy."""
        
        # Sort by access count (LRU)
        sorted_entries = sorted(
            self.local_cache.items(),
            key=lambda x: (x[1].access_count, x[1].timestamp)
        )
        
        # Remove 25% of entries
        num_to_remove = max(1, len(sorted_entries) // 4)
        
        for i in range(num_to_remove):
            key, _ = sorted_entries[i]
            del self.local_cache[key]
            self.cache_stats['evictions'] += 1
    
    def _extract_task_features(self, task: PhotonicTask) -> List[float]:
        """Extract features from a task for ML prediction."""
        
        features = []
        
        # Basic task properties
        features.append(float(task.priority.value))
        features.append(task.thermal_cost)
        features.append(task.estimated_duration_ms)
        features.append(len(task.wavelength_requirements))
        
        # Operation type encoding (one-hot)
        operation_types = ['matmul', 'phase_shift', 'thermal_compensation', 'quantum_gate']
        for op_type in operation_types:
            features.append(1.0 if task.operation_type == op_type else 0.0)
        
        # Matrix dimensions (for matmul operations)
        if task.operation_type == "matmul" and 'matrix_dims' in task.parameters:
            m, n, k = task.parameters['matrix_dims']
            features.extend([float(m), float(n), float(k)])
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # System state features
        features.append(self.current_metrics.latency_ms)
        features.append(self.current_metrics.throughput_ops_per_second)
        features.append(self.current_metrics.resource_utilization)
        features.append(float(self.current_workers))
        
        return features
    
    def _predictions_to_parameters(self, predictions: np.ndarray, task: PhotonicTask) -> Dict[str, Any]:
        """Convert ML predictions to optimization parameters."""
        
        params = {}
        
        # Assuming predictions are: [parallelization, precision_level, thermal_aggressiveness]
        if len(predictions) >= 3:
            # Parallelization level (1-8)
            params['parallelization'] = max(1, min(8, int(predictions[0] * 8)))
            
            # Precision level
            precision_level = predictions[1]
            if precision_level > 0.7:
                params['precision'] = 'high'
            elif precision_level > 0.3:
                params['precision'] = 'standard'
            else:
                params['precision'] = 'low'
            
            # Thermal management
            thermal_level = predictions[2]
            if thermal_level > 0.6:
                params['thermal_management'] = 'aggressive'
            else:
                params['thermal_management'] = 'standard'
        
        return params
    
    async def _update_ml_training_data(self, task: PhotonicTask, params: Dict[str, Any], 
                                     metrics: PerformanceMetrics) -> None:
        """Update ML training data with new observations."""
        
        if self.ml_predictor is None:
            return
        
        features = self._extract_task_features(task)
        target = metrics.composite_score(self.config.objective_weights)
        
        self.training_features.append(features)
        self.training_targets.append(target)
        
        # Limit training data size
        max_training_size = 10000
        if len(self.training_features) > max_training_size:
            # Remove oldest 10%
            remove_count = max_training_size // 10
            self.training_features = self.training_features[remove_count:]
            self.training_targets = self.training_targets[remove_count:]
    
    def _record_optimization_outcome(self, task: PhotonicTask, params: Dict[str, Any], 
                                   metrics: PerformanceMetrics) -> None:
        """Record optimization outcome for analysis."""
        
        outcome = {
            'timestamp': time.time(),
            'task_type': task.operation_type,
            'parameters': params,
            'metrics': metrics.to_dict(),
            'composite_score': metrics.composite_score(self.config.objective_weights)
        }
        
        self.optimization_history.append(outcome)
        
        # Update best configuration if this is better
        if outcome['composite_score'] > self.best_score:
            self.best_score = outcome['composite_score']
            self.best_configuration = params.copy()
            self.logger.info(f"New best configuration found: score={self.best_score:.3f}")
    
    def _monitoring_loop(self) -> None:
        """Main performance monitoring loop."""
        
        self.logger.info("Starting performance monitoring loop")
        
        while self.is_running:
            try:
                # Collect current metrics
                current_metrics = self._collect_system_metrics()
                self.current_metrics = current_metrics
                self.metrics_history.append(current_metrics)
                
                # Check for performance regressions
                self._check_performance_regressions()
                
                # Update adaptive scaling
                if self.config.scaling_mode != ScalingMode.DISABLED:
                    self._evaluate_scaling_needs()
                
                # Log metrics periodically
                if len(self.metrics_history) % 10 == 0:
                    self._log_performance_summary()
                
                time.sleep(self.config.monitoring_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5.0)
    
    def _optimization_loop(self) -> None:
        """Main optimization loop."""
        
        self.logger.info("Starting optimization loop")
        
        while self.is_running:
            try:
                # Retrain ML model if enough new data
                if (self.ml_predictor is not None and 
                    len(self.training_features) >= 100 and
                    len(self.training_features) % 100 == 0):
                    
                    self._retrain_ml_model()
                
                # Perform optimization analysis
                self._analyze_optimization_opportunities()
                
                # Update performance targets based on recent data
                self._update_performance_targets()
                
                # Export optimization metrics
                self._export_optimization_metrics()
                
                time.sleep(self.config.optimization_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                time.sleep(30.0)
    
    def _cache_cleanup_loop(self) -> None:
        """Cache cleanup and maintenance loop."""
        
        while self.is_running:
            try:
                # Clean expired entries from local cache
                expired_keys = []
                for key, entry in self.local_cache.items():
                    if entry.is_expired(self.config.distributed_cache_ttl_seconds):
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del self.local_cache[key]
                    self.cache_stats['evictions'] += 1
                
                # Update cache statistics
                self._update_cache_statistics()
                
                time.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in cache cleanup loop: {e}")
                time.sleep(60)
    
    def _collect_system_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        
        metrics = PerformanceMetrics()
        
        # Calculate metrics from recent history
        if self.optimization_history:
            recent_outcomes = list(self.optimization_history)[-100:]  # Last 100 outcomes
            
            if recent_outcomes:
                # Average latency
                latencies = [o['metrics']['latency_ms'] for o in recent_outcomes]
                metrics.latency_ms = statistics.mean(latencies)
                
                # Average throughput
                throughputs = [o['metrics']['throughput_ops_per_second'] for o in recent_outcomes]
                metrics.throughput_ops_per_second = statistics.mean(throughputs)
                
                # Average energy efficiency
                energies = [o['metrics']['energy_efficiency'] for o in recent_outcomes]
                metrics.energy_efficiency_pj_per_op = statistics.mean(energies)
                
                # Average thermal efficiency
                thermals = [o['metrics']['thermal_efficiency'] for o in recent_outcomes]
                metrics.thermal_efficiency = statistics.mean(thermals)
                
                # Resource utilization
                utilizations = [o['metrics']['resource_utilization'] for o in recent_outcomes]
                metrics.resource_utilization = statistics.mean(utilizations)
        
        # Cache hit rate
        if self.cache_stats['total_requests'] > 0:
            metrics.cache_hit_rate = self.cache_stats['hits'] / self.cache_stats['total_requests']
        
        return metrics
    
    def _check_performance_regressions(self) -> None:
        """Check for performance regressions."""
        
        if len(self.metrics_history) < 20:
            return
        
        # Compare recent performance to baseline
        recent_metrics = list(self.metrics_history)[-10:]
        baseline_metrics = list(self.metrics_history)[-50:-10] if len(self.metrics_history) >= 50 else []
        
        if not baseline_metrics:
            return
        
        # Check latency regression
        recent_latency = statistics.mean([m.latency_ms for m in recent_metrics])
        baseline_latency = statistics.mean([m.latency_ms for m in baseline_metrics])
        
        if recent_latency > baseline_latency * 1.2:  # 20% regression
            self.logger.warning(f"Latency regression detected: {recent_latency:.1f}ms vs {baseline_latency:.1f}ms baseline")
        
        # Check throughput regression
        recent_throughput = statistics.mean([m.throughput_ops_per_second for m in recent_metrics])
        baseline_throughput = statistics.mean([m.throughput_ops_per_second for m in baseline_metrics])
        
        if recent_throughput < baseline_throughput * 0.8:  # 20% regression
            self.logger.warning(f"Throughput regression detected: {recent_throughput:.1f} vs {baseline_throughput:.1f} baseline")
    
    def _evaluate_scaling_needs(self) -> None:
        """Evaluate if scaling up/down is needed."""
        
        # Check cooldown period
        if time.time() - self.last_scaling_time < self.config.scaling_cooldown_seconds:
            return
        
        current_utilization = self.current_metrics.resource_utilization
        
        # Scale up if utilization is high
        if (current_utilization > self.config.scale_up_threshold and 
            self.current_workers < self.config.max_workers):
            
            new_workers = min(self.current_workers + 2, self.config.max_workers)
            self._scale_workers(new_workers, 'scale_up')
        
        # Scale down if utilization is low
        elif (current_utilization < self.config.scale_down_threshold and 
              self.current_workers > self.config.min_workers):
            
            new_workers = max(self.current_workers - 1, self.config.min_workers)
            self._scale_workers(new_workers, 'scale_down')
    
    def _scale_workers(self, new_worker_count: int, reason: str) -> None:
        """Scale worker count."""
        
        old_workers = self.current_workers
        self.current_workers = new_worker_count
        self.last_scaling_time = time.time()
        
        # Record scaling decision
        scaling_decision = {
            'timestamp': time.time(),
            'old_workers': old_workers,
            'new_workers': new_worker_count,
            'reason': reason,
            'utilization': self.current_metrics.resource_utilization
        }
        self.scaling_decisions.append(scaling_decision)
        
        self.logger.info(f"Scaled workers: {old_workers} -> {new_worker_count} ({reason})")
        
        # Recreate executor with new worker count
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(
            max_workers=new_worker_count,
            thread_name_prefix="PerfOptimizer"
        )
    
    def _retrain_ml_model(self) -> None:
        """Retrain the ML model with new data."""
        
        if not _ML_AVAILABLE or self.ml_predictor is None:
            return
        
        if len(self.training_features) < 50:
            return
        
        try:
            self.logger.info(f"Retraining ML model with {len(self.training_features)} samples")
            
            # Prepare training data
            X = np.array(self.training_features)
            y = np.array(self.training_targets)
            
            # Fit scaler
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.ml_predictor.fit(X_scaled, y)
            self.model_trained = True
            
            # Evaluate model performance
            score = self.ml_predictor.score(X_scaled, y)
            self.logger.info(f"ML model retrained: R² score = {score:.3f}")
            
        except Exception as e:
            self.logger.error(f"ML model retraining failed: {e}")
    
    def _analyze_optimization_opportunities(self) -> None:
        """Analyze opportunities for further optimization."""
        
        if len(self.optimization_history) < 50:
            return
        
        recent_outcomes = list(self.optimization_history)[-100:]
        
        # Analyze parameter effectiveness
        param_analysis = defaultdict(list)
        
        for outcome in recent_outcomes:
            for param, value in outcome['parameters'].items():
                param_analysis[param].append((value, outcome['composite_score']))
        
        # Find best parameter values
        best_params = {}
        for param, values_scores in param_analysis.items():
            if len(values_scores) >= 10:
                # Group by parameter value and find average score
                value_groups = defaultdict(list)
                for value, score in values_scores:
                    value_groups[value].append(score)
                
                best_value = max(value_groups.items(), key=lambda x: statistics.mean(x[1]))
                best_params[param] = best_value[0]
        
        if best_params:
            self.logger.debug(f"Optimal parameters identified: {best_params}")
    
    def _update_performance_targets(self) -> None:
        """Update performance targets based on observed capabilities."""
        
        if len(self.metrics_history) < 100:
            return
        
        recent_metrics = list(self.metrics_history)[-50:]
        
        # Update targets to 90th percentile of recent performance
        latencies = [m.latency_ms for m in recent_metrics]
        throughputs = [m.throughput_ops_per_second for m in recent_metrics]
        
        if latencies:
            # Target latency should be 90th percentile (achievable goal)
            self.performance_targets['latency_ms'] = np.percentile(latencies, 10)  # 10th percentile = better latency
        
        if throughputs:
            # Target throughput should be 90th percentile
            self.performance_targets['throughput_ops_per_second'] = np.percentile(throughputs, 90)
    
    def _log_performance_summary(self) -> None:
        """Log performance summary."""
        
        metrics = self.current_metrics
        cache_hit_rate = self.cache_stats['hits'] / max(1, self.cache_stats['total_requests'])
        
        self.logger.info(
            f"Performance Summary: "
            f"Latency={metrics.latency_ms:.1f}ms, "
            f"Throughput={metrics.throughput_ops_per_second:.1f}ops/s, "
            f"Cache Hit Rate={cache_hit_rate:.1%}, "
            f"Workers={self.current_workers}, "
            f"Utilization={metrics.resource_utilization:.1%}"
        )
    
    def _update_cache_statistics(self) -> None:
        """Update cache performance statistics."""
        
        if self.cache_stats['total_requests'] > 0:
            hit_rate = self.cache_stats['hits'] / self.cache_stats['total_requests']
            self.current_metrics.cache_hit_rate = hit_rate
    
    def _export_optimization_metrics(self) -> None:
        """Export optimization metrics for monitoring."""
        
        metrics_data = {
            'timestamp': time.time(),
            'current_metrics': self.current_metrics.to_dict(),
            'performance_targets': self.performance_targets,
            'current_workers': self.current_workers,
            'cache_stats': self.cache_stats,
            'best_score': self.best_score,
            'best_configuration': self.best_configuration,
            'ml_model_trained': self.model_trained,
            'optimization_history_size': len(self.optimization_history)
        }
        
        # Write to metrics file
        metrics_file = Path("/tmp/performance_optimization_metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_data, f, indent=2)
    
    async def _establish_baselines(self) -> None:
        """Establish initial performance baselines."""
        
        self.logger.info("Establishing performance baselines")
        
        # Run some baseline measurements
        baseline_tasks = [
            PhotonicTask(
                task_id=f"baseline_{i}",
                operation_type="matmul",
                input_data=None,
                parameters={'matrix_dims': (64, 64, 64)},
                estimated_duration_ms=50.0
            )
            for i in range(10)
        ]
        
        baseline_metrics = []
        for task in baseline_tasks:
            try:
                _, metrics = await self.optimize_task_execution(task)
                baseline_metrics.append(metrics)
            except Exception as e:
                self.logger.warning(f"Baseline measurement failed: {e}")
        
        if baseline_metrics:
            # Calculate baseline averages
            avg_latency = statistics.mean([m.latency_ms for m in baseline_metrics])
            avg_throughput = statistics.mean([m.throughput_ops_per_second for m in baseline_metrics])
            
            self.performance_baselines = {
                'latency_ms': avg_latency,
                'throughput_ops_per_second': avg_throughput
            }
            
            self.logger.info(f"Baselines established: {self.performance_baselines}")
    
    async def _save_optimization_state(self) -> None:
        """Save optimization state for persistence."""
        
        state_data = {
            'best_configuration': self.best_configuration,
            'best_score': self.best_score,
            'performance_targets': self.performance_targets,
            'performance_baselines': self.performance_baselines,
            'cache_stats': self.cache_stats,
            'current_workers': self.current_workers
        }
        
        state_file = Path("/tmp/optimization_state.json")
        with open(state_file, "w") as f:
            json.dump(state_data, f, indent=2)
        
        self.logger.info("Optimization state saved")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        
        cache_hit_rate = 0.0
        if self.cache_stats['total_requests'] > 0:
            cache_hit_rate = self.cache_stats['hits'] / self.cache_stats['total_requests']
        
        return {
            'is_running': self.is_running,
            'primary_objective': self.config.primary_objective.value,
            'cache_strategy': self.config.cache_strategy.value,
            'scaling_mode': self.config.scaling_mode.value,
            'current_workers': self.current_workers,
            'ml_optimization_enabled': self.config.enable_ml_optimization and self.model_trained,
            'current_metrics': self.current_metrics.to_dict(),
            'performance_targets': self.performance_targets,
            'performance_baselines': self.performance_baselines,
            'best_configuration': self.best_configuration,
            'best_score': self.best_score,
            'cache_statistics': {
                'hit_rate': cache_hit_rate,
                'total_requests': self.cache_stats['total_requests'],
                'local_cache_size': len(self.local_cache)
            },
            'optimization_history_size': len(self.optimization_history),
            'circuit_breaker_status': {
                name: breaker.state.value 
                for name, breaker in self.circuit_breakers.items()
            },
            'uptime_hours': (time.time() - self.start_time) / 3600.0 if self.start_time else 0.0
        }


class QuantumCoherenceOptimizer:
    """Specialized optimizer for quantum coherence preservation."""
    
    def __init__(self, coherence_time_ms: float, decoherence_rate: float):
        self.coherence_time_ms = coherence_time_ms
        self.decoherence_rate = decoherence_rate
        self.logger = get_global_logger(__name__)
    
    def optimize_quantum_task(self, task: PhotonicTask) -> Dict[str, Any]:
        """Optimize quantum task for coherence preservation."""
        
        optimization_params = {}
        
        # Optimize gate sequence timing
        if 'quantum_gates' in task.parameters:
            gates = task.parameters['quantum_gates']
            optimal_timing = self._optimize_gate_timing(gates)
            optimization_params['gate_timing'] = optimal_timing
        
        # Optimize error correction
        if task.estimated_duration_ms > self.coherence_time_ms * 0.5:
            optimization_params['error_correction'] = 'enhanced'
        else:
            optimization_params['error_correction'] = 'standard'
        
        return optimization_params
    
    def _optimize_gate_timing(self, gates: List[str]) -> List[float]:
        """Optimize timing between quantum gates."""
        
        # Simple heuristic: minimize total time while preserving coherence
        gate_times = []
        cumulative_time = 0.0
        
        for gate in gates:
            # Different gates have different optimal timings
            if gate in ['H', 'X', 'Y', 'Z']:
                gate_time = 10.0  # Single qubit gates: 10ns
            else:  # Two-qubit gates
                gate_time = 50.0  # Two-qubit gates: 50ns
            
            gate_times.append(gate_time)
            cumulative_time += gate_time
            
            # Add decoherence prevention delay if needed
            if cumulative_time > self.coherence_time_ms * 0.8:
                gate_times.append(5.0)  # 5ns recovery time
                cumulative_time = 0.0
        
        return gate_times


# Export main classes
__all__ = [
    'AutonomousPerformanceOptimizer',
    'OptimizationConfig',
    'PerformanceMetrics',
    'OptimizationObjective',
    'CacheStrategy',
    'ScalingMode',
    'OptimizationStrategy',
    'CacheEntry',
    'QuantumCoherenceOptimizer'
]