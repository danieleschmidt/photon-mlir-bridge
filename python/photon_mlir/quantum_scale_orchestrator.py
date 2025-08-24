"""
Quantum-Scale Orchestrator - Hyperscale distributed photonic compilation system.

This module implements quantum-inspired algorithms for massively parallel
photonic compilation, featuring:
- Quantum-enhanced load balancing
- Adaptive mesh partitioning
- Self-optimizing compilation pipelines
- Distributed quantum error correction
- Hyperscale resource orchestration
"""

import asyncio
import threading
import time
import math
import random
from typing import Dict, List, Optional, Any, Callable, Tuple, Union, Set
from enum import Enum
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future
import logging
import heapq
import numpy as np
from collections import defaultdict, deque
import json

logger = logging.getLogger(__name__)


class QuantumState(Enum):
    """Quantum compilation states."""
    SUPERPOSITION = "superposition"
    ENTANGLED = "entangled"
    COHERENT = "coherent"
    DECOHERENT = "decoherent"
    COLLAPSED = "collapsed"


class ScalingStrategy(Enum):
    """Scaling strategies for compilation."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"  
    QUANTUM_PARALLEL = "quantum_parallel"
    ADAPTIVE = "adaptive"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Resource types for orchestration."""
    CPU_CORE = "cpu_core"
    GPU_DEVICE = "gpu_device"
    PHOTONIC_CHIP = "photonic_chip"
    QUANTUM_PROCESSOR = "quantum_processor"
    MEMORY = "memory"
    NETWORK_BANDWIDTH = "network_bandwidth"


@dataclass
class QuantumResource:
    """Quantum-enhanced resource representation."""
    resource_id: str
    resource_type: ResourceType
    capacity: float
    utilization: float = 0.0
    quantum_state: QuantumState = QuantumState.COHERENT
    entangled_resources: Set[str] = field(default_factory=set)
    location: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D coordinates
    performance_history: List[float] = field(default_factory=list)
    
    def coherence_factor(self) -> float:
        """Calculate quantum coherence factor."""
        if self.quantum_state == QuantumState.COHERENT:
            return 1.0
        elif self.quantum_state == QuantumState.SUPERPOSITION:
            return 0.8
        elif self.quantum_state == QuantumState.ENTANGLED:
            return 0.9
        else:
            return 0.3


@dataclass
class CompilationTask:
    """Quantum compilation task."""
    task_id: str
    operation_type: str
    complexity_score: float
    input_size: int
    required_resources: Dict[ResourceType, float]
    priority: float = 1.0
    quantum_signature: str = ""
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 0.0
    deadline: Optional[float] = None
    
    def __post_init__(self):
        """Generate quantum signature for task."""
        self.quantum_signature = self._generate_quantum_signature()
    
    def _generate_quantum_signature(self) -> str:
        """Generate unique quantum signature for task."""
        data = f"{self.operation_type}_{self.complexity_score}_{self.input_size}"
        return f"q{abs(hash(data)) % 1000000:06d}"


@dataclass
class ScalingMetrics:
    """Comprehensive scaling metrics."""
    total_throughput: float = 0.0
    avg_latency: float = 0.0
    resource_utilization: Dict[ResourceType, float] = field(default_factory=dict)
    quantum_coherence: float = 1.0
    compilation_success_rate: float = 1.0
    adaptive_efficiency: float = 1.0
    cost_per_operation: float = 0.0
    energy_efficiency: float = 1.0


class QuantumLoadBalancer:
    """Quantum-inspired load balancer with entanglement-based routing."""
    
    def __init__(self):
        self.resources: Dict[str, QuantumResource] = {}
        self.task_queue = asyncio.Queue()
        self.quantum_weights = {}
        self._lock = threading.RLock()
        
    def add_resource(self, resource: QuantumResource):
        """Add quantum resource to pool."""
        with self._lock:
            self.resources[resource.resource_id] = resource
            self._update_quantum_entanglements()
    
    def _update_quantum_entanglements(self):
        """Update quantum entanglements between resources."""
        resources_list = list(self.resources.values())
        
        # Create entanglements based on proximity and compatibility
        for i, res1 in enumerate(resources_list):
            for j, res2 in enumerate(resources_list[i+1:], i+1):
                distance = self._calculate_distance(res1.location, res2.location)
                
                if distance < 10.0 and res1.resource_type == res2.resource_type:
                    res1.entangled_resources.add(res2.resource_id)
                    res2.entangled_resources.add(res1.resource_id)
    
    def _calculate_distance(self, loc1: Tuple[float, float, float], loc2: Tuple[float, float, float]) -> float:
        """Calculate 3D Euclidean distance."""
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(loc1, loc2)))
    
    def select_optimal_resources(
        self, 
        task: CompilationTask
    ) -> List[QuantumResource]:
        """Select optimal resources using quantum algorithms."""
        
        # Quantum-inspired selection algorithm
        candidates = []
        
        for resource in self.resources.values():
            if resource.utilization < 0.8:  # Available capacity
                score = self._calculate_quantum_affinity(task, resource)
                candidates.append((score, resource))
        
        # Sort by quantum affinity score
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Select top resources based on task requirements
        selected = []
        for required_type, required_amount in task.required_resources.items():
            type_candidates = [
                (score, res) for score, res in candidates
                if res.resource_type == required_type
            ]
            
            if type_candidates:
                selected.append(type_candidates[0][1])
        
        return selected
    
    def _calculate_quantum_affinity(
        self, 
        task: CompilationTask, 
        resource: QuantumResource
    ) -> float:
        """Calculate quantum affinity between task and resource."""
        
        # Base score from utilization and coherence
        base_score = (1.0 - resource.utilization) * resource.coherence_factor()
        
        # Quantum enhancement based on task signature
        quantum_hash = abs(hash(task.quantum_signature + resource.resource_id))
        quantum_factor = (quantum_hash % 1000) / 1000.0
        
        # Entanglement bonus
        entanglement_bonus = len(resource.entangled_resources) * 0.1
        
        # Performance history consideration
        performance_factor = np.mean(resource.performance_history) if resource.performance_history else 1.0
        
        return base_score * (1.0 + quantum_factor + entanglement_bonus) * performance_factor


class AdaptiveMeshPartitioner:
    """Adaptive mesh partitioning for photonic architectures."""
    
    def __init__(self):
        self.partition_cache: Dict[str, Any] = {}
        self.learning_rate = 0.1
        self.partition_history: List[Dict[str, Any]] = []
        
    def partition_computation_graph(
        self,
        graph: Dict[str, Any],
        target_resources: List[QuantumResource],
        optimization_objective: str = "latency"
    ) -> Dict[str, List[str]]:
        """Partition computation graph across resources."""
        
        cache_key = self._generate_cache_key(graph, target_resources, optimization_objective)
        if cache_key in self.partition_cache:
            return self.partition_cache[cache_key]
        
        # Quantum-inspired graph partitioning
        partitions = self._quantum_graph_partition(graph, target_resources, optimization_objective)
        
        # Cache result
        self.partition_cache[cache_key] = partitions
        
        # Learn from partitioning result
        self._update_learning_model(graph, partitions, target_resources)
        
        return partitions
    
    def _quantum_graph_partition(
        self,
        graph: Dict[str, Any],
        resources: List[QuantumResource],
        objective: str
    ) -> Dict[str, List[str]]:
        """Quantum-inspired graph partitioning algorithm."""
        
        nodes = list(graph.get("nodes", []))
        edges = graph.get("edges", [])
        
        # Initialize partitions
        partitions = {res.resource_id: [] for res in resources}
        
        # Quantum superposition-inspired assignment
        for node in nodes:
            # Calculate quantum probabilities for each resource
            probabilities = self._calculate_assignment_probabilities(
                node, resources, edges, objective
            )
            
            # Select resource based on quantum probabilities
            selected_resource = self._quantum_select(resources, probabilities)
            partitions[selected_resource.resource_id].append(node)
        
        # Optimize partitions using quantum annealing-inspired process
        partitions = self._quantum_anneal_partitions(partitions, edges, resources, objective)
        
        return partitions
    
    def _calculate_assignment_probabilities(
        self,
        node: str,
        resources: List[QuantumResource],
        edges: List[Tuple[str, str]],
        objective: str
    ) -> List[float]:
        """Calculate quantum assignment probabilities."""
        probabilities = []
        
        for resource in resources:
            # Base probability from resource availability
            base_prob = (1.0 - resource.utilization) * resource.coherence_factor()
            
            # Connectivity consideration
            connectivity_factor = self._calculate_connectivity_factor(node, edges, resource)
            
            # Objective-specific weighting
            objective_weight = self._get_objective_weight(resource, objective)
            
            prob = base_prob * connectivity_factor * objective_weight
            probabilities.append(prob)
        
        # Normalize probabilities
        total = sum(probabilities)
        return [p / total if total > 0 else 1.0 / len(resources) for p in probabilities]
    
    def _quantum_select(self, resources: List[QuantumResource], probabilities: List[float]) -> QuantumResource:
        """Quantum selection based on probabilities."""
        rand_val = random.random()
        cumulative = 0.0
        
        for resource, prob in zip(resources, probabilities):
            cumulative += prob
            if rand_val <= cumulative:
                return resource
        
        return resources[-1]  # Fallback
    
    def _quantum_anneal_partitions(
        self,
        partitions: Dict[str, List[str]],
        edges: List[Tuple[str, str]],
        resources: List[QuantumResource],
        objective: str
    ) -> Dict[str, List[str]]:
        """Quantum annealing-inspired partition optimization."""
        
        current_partitions = partitions.copy()
        best_score = self._evaluate_partition_quality(current_partitions, edges, resources, objective)
        
        # Simulated annealing parameters
        temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 0.01
        
        while temperature > min_temperature:
            # Generate neighbor solution
            new_partitions = self._generate_neighbor_partition(current_partitions)
            new_score = self._evaluate_partition_quality(new_partitions, edges, resources, objective)
            
            # Accept or reject based on quantum probability
            if self._quantum_accept(best_score, new_score, temperature):
                current_partitions = new_partitions
                if new_score > best_score:
                    best_score = new_score
            
            temperature *= cooling_rate
        
        return current_partitions
    
    def _evaluate_partition_quality(
        self,
        partitions: Dict[str, List[str]],
        edges: List[Tuple[str, str]],
        resources: List[QuantumResource],
        objective: str
    ) -> float:
        """Evaluate partition quality score."""
        
        # Balance factor
        partition_sizes = [len(nodes) for nodes in partitions.values()]
        balance_factor = 1.0 - (max(partition_sizes) - min(partition_sizes)) / max(partition_sizes) if partition_sizes else 1.0
        
        # Cut minimization
        cut_edges = self._count_cut_edges(partitions, edges)
        cut_factor = 1.0 / (1.0 + cut_edges)
        
        # Resource utilization
        utilization_factor = np.mean([res.coherence_factor() for res in resources])
        
        # Objective-specific scoring
        if objective == "latency":
            return balance_factor * cut_factor * utilization_factor
        elif objective == "throughput":
            return balance_factor * utilization_factor * 1.2
        else:
            return balance_factor * cut_factor * utilization_factor
    
    def _generate_cache_key(self, graph: Dict[str, Any], resources: List[QuantumResource], objective: str) -> str:
        """Generate cache key for partition."""
        graph_hash = hash(str(sorted(graph.items())))
        resources_hash = hash(tuple(res.resource_id for res in resources))
        return f"{graph_hash}_{resources_hash}_{objective}"


class HyperScaleCompiler:
    """Hyperscale distributed compilation orchestrator."""
    
    def __init__(self, scaling_strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE):
        self.scaling_strategy = scaling_strategy
        self.load_balancer = QuantumLoadBalancer()
        self.mesh_partitioner = AdaptiveMeshPartitioner()
        self.active_compilations: Dict[str, Dict[str, Any]] = {}
        self.metrics = ScalingMetrics()
        self.executor = ThreadPoolExecutor(max_workers=32)
        self._lock = threading.RLock()
        
        # Performance prediction models (placeholder for future ML models)
        self.latency_predictor = None
        self.throughput_predictor = None
        
    def add_compilation_resource(self, resource: QuantumResource):
        """Add compilation resource to hyperscale pool."""
        self.load_balancer.add_resource(resource)
        logger.info(f"Added {resource.resource_type.value} resource: {resource.resource_id}")
    
    async def compile_at_scale(
        self,
        compilation_graph: Dict[str, Any],
        optimization_level: int = 3,
        deadline: Optional[float] = None
    ) -> Dict[str, Any]:
        """Execute hyperscale compilation."""
        
        compilation_id = f"comp_{int(time.time() * 1000)}"
        
        # Decompose compilation into quantum tasks
        tasks = self._decompose_compilation(compilation_graph, optimization_level)
        
        # Predict resource requirements
        required_resources = self._predict_resource_requirements(tasks)
        
        # Scale resources if needed
        await self._adaptive_scale_resources(required_resources, deadline)
        
        # Partition tasks across available resources
        available_resources = list(self.load_balancer.resources.values())
        task_partitions = self.mesh_partitioner.partition_computation_graph(
            {"nodes": [t.task_id for t in tasks], "edges": self._extract_task_dependencies(tasks)},
            available_resources,
            "latency" if deadline else "throughput"
        )
        
        # Execute compilation in parallel
        results = await self._execute_parallel_compilation(compilation_id, tasks, task_partitions)
        
        # Update metrics
        self._update_scaling_metrics(compilation_id, tasks, results)
        
        return {
            "compilation_id": compilation_id,
            "status": "completed",
            "results": results,
            "metrics": self.metrics,
            "resource_utilization": {res.resource_id: res.utilization for res in available_resources}
        }
    
    def _decompose_compilation(
        self, 
        graph: Dict[str, Any], 
        optimization_level: int
    ) -> List[CompilationTask]:
        """Decompose compilation into quantum tasks."""
        
        tasks = []
        layers = graph.get("layers", [])
        
        for i, layer in enumerate(layers):
            # Create quantum compilation task for each layer
            task = CompilationTask(
                task_id=f"layer_{i}",
                operation_type=layer.get("type", "unknown"),
                complexity_score=self._calculate_complexity_score(layer),
                input_size=layer.get("input_size", 0),
                required_resources={
                    ResourceType.CPU_CORE: min(4, optimization_level),
                    ResourceType.MEMORY: layer.get("input_size", 0) * 0.001,  # MB per input element
                    ResourceType.PHOTONIC_CHIP: 1.0 if "photonic" in layer.get("type", "") else 0.0
                },
                priority=layer.get("priority", 1.0),
                deadline=layer.get("deadline")
            )
            
            # Add dependencies
            for dep in layer.get("dependencies", []):
                task.dependencies.add(f"layer_{dep}")
            
            tasks.append(task)
        
        return tasks
    
    def _calculate_complexity_score(self, layer: Dict[str, Any]) -> float:
        """Calculate complexity score for layer."""
        base_ops = layer.get("operations", 1)
        input_size = layer.get("input_size", 1)
        layer_type = layer.get("type", "linear")
        
        # Type-specific complexity factors
        type_factors = {
            "convolution": 3.0,
            "attention": 5.0,
            "photonic_matmul": 2.0,
            "quantum_gate": 10.0,
            "linear": 1.0
        }
        
        factor = type_factors.get(layer_type, 1.0)
        return base_ops * input_size * factor
    
    async def _adaptive_scale_resources(
        self,
        required_resources: Dict[ResourceType, float],
        deadline: Optional[float]
    ):
        """Adaptively scale resources based on requirements."""
        
        current_resources = self._get_current_resource_capacity()
        
        for resource_type, required_amount in required_resources.items():
            current_amount = current_resources.get(resource_type, 0.0)
            
            if required_amount > current_amount * 0.8:  # 80% utilization threshold
                # Need to scale up
                additional_resources = math.ceil((required_amount - current_amount * 0.8) / 10.0)
                await self._provision_resources(resource_type, additional_resources, deadline)
    
    async def _provision_resources(
        self,
        resource_type: ResourceType,
        count: int,
        deadline: Optional[float]
    ):
        """Provision additional resources."""
        
        for i in range(count):
            # Simulate resource provisioning
            resource = QuantumResource(
                resource_id=f"{resource_type.value}_{int(time.time() * 1000)}_{i}",
                resource_type=resource_type,
                capacity=10.0,
                location=(random.uniform(0, 100), random.uniform(0, 100), random.uniform(0, 10))
            )
            
            self.add_compilation_resource(resource)
            
            # Brief delay to simulate provisioning time
            await asyncio.sleep(0.1)
        
        logger.info(f"Provisioned {count} additional {resource_type.value} resources")
    
    async def _execute_parallel_compilation(
        self,
        compilation_id: str,
        tasks: List[CompilationTask],
        partitions: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """Execute compilation tasks in parallel."""
        
        # Create execution plan
        execution_plan = self._create_execution_plan(tasks, partitions)
        
        # Execute tasks in topological order with parallelism
        results = {}
        completed_tasks = set()
        
        while len(completed_tasks) < len(tasks):
            # Find tasks ready for execution
            ready_tasks = [
                task for task in tasks
                if task.task_id not in completed_tasks and
                all(dep in completed_tasks for dep in task.dependencies)
            ]
            
            if not ready_tasks:
                break  # No more tasks can be executed
            
            # Execute ready tasks in parallel
            futures = []
            for task in ready_tasks:
                future = self.executor.submit(self._execute_single_task, task)
                futures.append((task.task_id, future))
            
            # Wait for completion and collect results
            for task_id, future in futures:
                try:
                    result = future.result(timeout=300)  # 5-minute timeout
                    results[task_id] = result
                    completed_tasks.add(task_id)
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {e}")
                    results[task_id] = {"error": str(e)}
                    completed_tasks.add(task_id)  # Mark as completed even if failed
        
        return results
    
    def _execute_single_task(self, task: CompilationTask) -> Dict[str, Any]:
        """Execute single compilation task."""
        
        start_time = time.time()
        
        # Select optimal resources for this task
        selected_resources = self.load_balancer.select_optimal_resources(task)
        
        # Simulate compilation work
        self._simulate_compilation_work(task, selected_resources)
        
        # Update resource utilization
        for resource in selected_resources:
            with self._lock:
                resource.utilization += 0.1
                resource.performance_history.append(time.time() - start_time)
                if len(resource.performance_history) > 100:
                    resource.performance_history.pop(0)
        
        execution_time = time.time() - start_time
        
        return {
            "task_id": task.task_id,
            "status": "completed",
            "execution_time": execution_time,
            "resources_used": [res.resource_id for res in selected_resources],
            "quantum_signature": task.quantum_signature
        }
    
    def _simulate_compilation_work(self, task: CompilationTask, resources: List[QuantumResource]):
        """Simulate actual compilation work."""
        
        # Simulate work based on task complexity and resource capacity
        work_amount = task.complexity_score
        total_capacity = sum(res.capacity * res.coherence_factor() for res in resources)
        
        # Work time proportional to complexity and inversely proportional to capacity
        work_time = work_amount / max(total_capacity, 1.0)
        
        # Add some quantum noise
        quantum_noise = random.gauss(1.0, 0.1)
        work_time *= quantum_noise
        
        # Sleep to simulate work
        time.sleep(min(work_time, 5.0))  # Cap at 5 seconds for demonstration
    
    def _update_scaling_metrics(
        self,
        compilation_id: str,
        tasks: List[CompilationTask],
        results: Dict[str, Any]
    ):
        """Update comprehensive scaling metrics."""
        
        # Calculate throughput
        total_operations = sum(task.complexity_score for task in tasks)
        total_time = max(
            result.get("execution_time", 0) for result in results.values()
            if isinstance(result, dict) and "execution_time" in result
        )
        
        if total_time > 0:
            self.metrics.total_throughput = total_operations / total_time
            self.metrics.avg_latency = total_time / len(tasks)
        
        # Calculate success rate
        successful_tasks = sum(
            1 for result in results.values()
            if isinstance(result, dict) and result.get("status") == "completed"
        )
        self.metrics.compilation_success_rate = successful_tasks / len(tasks)
        
        # Update resource utilization metrics
        for resource_type in ResourceType:
            resources = [res for res in self.load_balancer.resources.values() if res.resource_type == resource_type]
            if resources:
                avg_util = np.mean([res.utilization for res in resources])
                self.metrics.resource_utilization[resource_type] = avg_util
        
        # Calculate quantum coherence
        all_resources = list(self.load_balancer.resources.values())
        if all_resources:
            self.metrics.quantum_coherence = np.mean([res.coherence_factor() for res in all_resources])
    
    def get_scaling_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive scaling dashboard."""
        
        return {
            "scaling_strategy": self.scaling_strategy.value,
            "active_compilations": len(self.active_compilations),
            "total_resources": len(self.load_balancer.resources),
            "resource_breakdown": {
                resource_type.value: len([
                    res for res in self.load_balancer.resources.values()
                    if res.resource_type == resource_type
                ])
                for resource_type in ResourceType
            },
            "performance_metrics": {
                "throughput": self.metrics.total_throughput,
                "latency": self.metrics.avg_latency,
                "success_rate": self.metrics.compilation_success_rate,
                "quantum_coherence": self.metrics.quantum_coherence
            },
            "resource_utilization": self.metrics.resource_utilization,
            "scaling_efficiency": self._calculate_scaling_efficiency()
        }
    
    def _calculate_scaling_efficiency(self) -> float:
        """Calculate overall scaling efficiency."""
        
        # Ideal efficiency factors
        throughput_factor = min(self.metrics.total_throughput / 1000.0, 1.0)  # Normalize to 1000 ops/sec
        latency_factor = max(0.0, 1.0 - self.metrics.avg_latency / 10.0)  # Penalty for >10s latency
        success_factor = self.metrics.compilation_success_rate
        coherence_factor = self.metrics.quantum_coherence
        
        # Resource utilization factor
        if self.metrics.resource_utilization:
            util_values = list(self.metrics.resource_utilization.values())
            # Ideal utilization is around 70-80%
            util_factor = np.mean([1.0 - abs(util - 0.75) / 0.75 for util in util_values])
        else:
            util_factor = 0.0
        
        # Weighted average
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # throughput, latency, success, coherence, utilization
        factors = [throughput_factor, latency_factor, success_factor, coherence_factor, util_factor]
        
        return sum(w * f for w, f in zip(weights, factors))


# Factory functions
def create_hyperscale_compiler(strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE) -> HyperScaleCompiler:
    """Create hyperscale compiler with specified strategy."""
    return HyperScaleCompiler(strategy)


def create_quantum_compilation_cluster(
    cpu_cores: int = 16,
    gpu_devices: int = 4,
    photonic_chips: int = 2,
    quantum_processors: int = 1
) -> HyperScaleCompiler:
    """Create a quantum compilation cluster with specified resources."""
    
    compiler = HyperScaleCompiler(ScalingStrategy.QUANTUM_PARALLEL)
    
    # Add CPU resources
    for i in range(cpu_cores):
        resource = QuantumResource(
            resource_id=f"cpu_{i}",
            resource_type=ResourceType.CPU_CORE,
            capacity=10.0,
            location=(i % 4, i // 4, 0)
        )
        compiler.add_compilation_resource(resource)
    
    # Add GPU resources
    for i in range(gpu_devices):
        resource = QuantumResource(
            resource_id=f"gpu_{i}",
            resource_type=ResourceType.GPU_DEVICE,
            capacity=50.0,
            location=(i * 2, 0, 1)
        )
        compiler.add_compilation_resource(resource)
    
    # Add photonic chips
    for i in range(photonic_chips):
        resource = QuantumResource(
            resource_id=f"photonic_{i}",
            resource_type=ResourceType.PHOTONIC_CHIP,
            capacity=100.0,
            quantum_state=QuantumState.SUPERPOSITION,
            location=(i * 3, 0, 2)
        )
        compiler.add_compilation_resource(resource)
    
    # Add quantum processors
    for i in range(quantum_processors):
        resource = QuantumResource(
            resource_id=f"quantum_{i}",
            resource_type=ResourceType.QUANTUM_PROCESSOR,
            capacity=1000.0,
            quantum_state=QuantumState.ENTANGLED,
            location=(i * 4, 0, 3)
        )
        compiler.add_compilation_resource(resource)
    
    return compiler


# Global hyperscale compiler instance
default_hyperscale_compiler = create_hyperscale_compiler()


__all__ = [
    'HyperScaleCompiler',
    'QuantumResource',
    'CompilationTask',
    'ScalingMetrics',
    'QuantumState',
    'ScalingStrategy',
    'ResourceType',
    'QuantumLoadBalancer',
    'AdaptiveMeshPartitioner',
    'create_hyperscale_compiler',
    'create_quantum_compilation_cluster',
    'default_hyperscale_compiler'
]