"""
Quantum-inspired task scheduler with thermal awareness for photonic computing.
Implements advanced scheduling algorithms for optimal photonic resource utilization.
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from concurrent.futures import ThreadPoolExecutor, Future
import logging

from .core import TargetConfig, Device
from .thermal_optimization import ThermalModel, ThermalOptimizer


class TaskPriority(Enum):
    """Task priority levels for quantum-aware scheduling."""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


class SchedulingStrategy(Enum):
    """Scheduling strategies for photonic tasks."""
    QUANTUM_ANNEALING = "quantum_annealing"
    THERMAL_AWARE_FIFO = "thermal_aware_fifo"
    ENERGY_OPTIMAL = "energy_optimal"
    LATENCY_MINIMAL = "latency_minimal"
    ADAPTIVE_HYBRID = "adaptive_hybrid"


@dataclass
class PhotonicTask:
    """Represents a photonic computation task."""
    task_id: str
    operation_type: str  # matmul, phase_shift, thermal_compensation, etc.
    input_data: Any
    parameters: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    thermal_cost: float = 0.0  # Estimated thermal impact
    wavelength_requirements: List[int] = field(default_factory=list)
    mesh_region: Optional[Tuple[int, int, int, int]] = None  # (x1, y1, x2, y2)
    dependencies: List[str] = field(default_factory=list)
    estimated_duration_ms: float = 0.0
    created_time: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.wavelength_requirements:
            self.wavelength_requirements = [1550]  # Default C-band wavelength


@dataclass
class SchedulingResult:
    """Result of task scheduling operation."""
    scheduled_tasks: List[PhotonicTask]
    execution_plan: List[Tuple[float, str]]  # (start_time, task_id)
    total_execution_time_ms: float
    thermal_efficiency: float
    resource_utilization: float
    energy_estimate_mj: float


class QuantumAwareScheduler:
    """
    Advanced quantum-inspired scheduler for photonic computing tasks.
    
    Features:
    - Quantum annealing-inspired optimization
    - Thermal-aware task placement
    - Wavelength division multiplexing optimization
    - Dynamic load balancing
    - Real-time adaptation
    """
    
    def __init__(self, target_config: TargetConfig, 
                 strategy: SchedulingStrategy = SchedulingStrategy.ADAPTIVE_HYBRID,
                 max_workers: int = 4):
        self.config = target_config
        self.strategy = strategy
        self.max_workers = max_workers
        
        # Initialize thermal model
        self.thermal_model = ThermalModel(target_config)
        self.thermal_optimizer = ThermalOptimizer(self.thermal_model)
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, PhotonicTask] = {}
        self.completed_tasks: Dict[str, PhotonicTask] = {}
        self.task_results: Dict[str, Any] = {}
        
        # Resource tracking
        self.mesh_utilization = np.zeros(target_config.array_size, dtype=float)
        self.wavelength_allocation: Dict[int, List[str]] = {}  # wavelength -> task_ids
        self.thermal_state = np.zeros(target_config.array_size, dtype=float)
        
        # Scheduling state
        self.current_time = 0.0
        self.total_energy_consumed = 0.0
        self.scheduling_history: List[SchedulingResult] = []
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.scheduler_lock = threading.RLock()
        self.running = False
        
        self.logger = logging.getLogger(__name__)
        
    def submit_task(self, task: PhotonicTask) -> str:
        """Submit a task for scheduling."""
        with self.scheduler_lock:
            # Estimate thermal cost and duration
            task.thermal_cost = self._estimate_thermal_cost(task)
            task.estimated_duration_ms = self._estimate_duration(task)
            
            # Add to queue with priority-based ordering
            priority_value = (task.priority.value, task.created_time)
            self.task_queue.put((priority_value, task))
            
            self.logger.info(f"Task {task.task_id} submitted with priority {task.priority.name}")
            return task.task_id
    
    def schedule_batch(self, tasks: List[PhotonicTask], 
                      optimize_for: str = "balanced") -> SchedulingResult:
        """
        Schedule a batch of tasks using quantum-inspired optimization.
        
        Args:
            tasks: List of tasks to schedule
            optimize_for: "thermal", "latency", "energy", "balanced"
        
        Returns:
            SchedulingResult with optimized execution plan
        """
        with self.scheduler_lock:
            if not tasks:
                return SchedulingResult([], [], 0.0, 1.0, 0.0, 0.0)
            
            self.logger.info(f"Scheduling batch of {len(tasks)} tasks")
            
            # Apply quantum-inspired optimization
            if self.strategy == SchedulingStrategy.QUANTUM_ANNEALING:
                result = self._quantum_annealing_schedule(tasks, optimize_for)
            elif self.strategy == SchedulingStrategy.THERMAL_AWARE_FIFO:
                result = self._thermal_aware_fifo_schedule(tasks)
            elif self.strategy == SchedulingStrategy.ENERGY_OPTIMAL:
                result = self._energy_optimal_schedule(tasks)
            elif self.strategy == SchedulingStrategy.LATENCY_MINIMAL:
                result = self._latency_minimal_schedule(tasks)
            else:  # ADAPTIVE_HYBRID
                result = self._adaptive_hybrid_schedule(tasks, optimize_for)
            
            # Update scheduling history
            self.scheduling_history.append(result)
            
            self.logger.info(f"Batch scheduled: {result.total_execution_time_ms:.2f}ms total, "
                           f"thermal efficiency: {result.thermal_efficiency:.3f}")
            
            return result
    
    def _quantum_annealing_schedule(self, tasks: List[PhotonicTask], 
                                  optimize_for: str) -> SchedulingResult:
        """
        Quantum annealing-inspired scheduling using simulated annealing.
        """
        n_tasks = len(tasks)
        if n_tasks == 1:
            return self._simple_schedule(tasks)
        
        # Initialize random solution
        current_order = list(range(n_tasks))
        np.random.shuffle(current_order)
        current_cost = self._evaluate_schedule_cost(tasks, current_order, optimize_for)
        
        best_order = current_order.copy()
        best_cost = current_cost
        
        # Simulated annealing parameters
        initial_temp = 1000.0
        final_temp = 1.0
        cooling_rate = 0.95
        max_iterations = min(1000, n_tasks * 50)
        
        temp = initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor solution by swapping two random tasks
            new_order = current_order.copy()
            i, j = np.random.choice(n_tasks, 2, replace=False)
            new_order[i], new_order[j] = new_order[j], new_order[i]
            
            new_cost = self._evaluate_schedule_cost(tasks, new_order, optimize_for)
            
            # Accept or reject based on probability
            delta_cost = new_cost - current_cost
            if delta_cost < 0 or np.random.random() < np.exp(-delta_cost / temp):
                current_order = new_order
                current_cost = new_cost
                
                if current_cost < best_cost:
                    best_order = current_order.copy()
                    best_cost = current_cost
            
            # Cool down
            temp *= cooling_rate
            if temp < final_temp:
                break
        
        # Build execution plan from best solution
        return self._build_execution_plan([tasks[i] for i in best_order], optimize_for)
    
    def _thermal_aware_fifo_schedule(self, tasks: List[PhotonicTask]) -> SchedulingResult:
        """Schedule tasks FIFO with thermal awareness."""
        # Sort by creation time
        sorted_tasks = sorted(tasks, key=lambda t: t.created_time)
        
        # Apply thermal constraints
        thermal_constrained_tasks = []
        current_thermal_load = 0.0
        thermal_threshold = 0.8  # 80% of max thermal capacity
        
        for task in sorted_tasks:
            if current_thermal_load + task.thermal_cost <= thermal_threshold:
                thermal_constrained_tasks.append(task)
                current_thermal_load += task.thermal_cost
            else:
                # Insert cooling delay
                cooling_task = self._create_cooling_task()
                thermal_constrained_tasks.append(cooling_task)
                thermal_constrained_tasks.append(task)
                current_thermal_load = task.thermal_cost
        
        return self._build_execution_plan(thermal_constrained_tasks, "thermal")
    
    def _energy_optimal_schedule(self, tasks: List[PhotonicTask]) -> SchedulingResult:
        """Schedule for minimal energy consumption."""
        # Group tasks by operation type for efficient batching
        operation_groups: Dict[str, List[PhotonicTask]] = {}
        for task in tasks:
            op_type = task.operation_type
            if op_type not in operation_groups:
                operation_groups[op_type] = []
            operation_groups[op_type].append(task)
        
        # Schedule each group optimally
        scheduled_tasks = []
        for op_type, group_tasks in operation_groups.items():
            # Sort by thermal cost (ascending) for energy efficiency
            group_tasks.sort(key=lambda t: t.thermal_cost)
            scheduled_tasks.extend(group_tasks)
        
        return self._build_execution_plan(scheduled_tasks, "energy")
    
    def _latency_minimal_schedule(self, tasks: List[PhotonicTask]) -> SchedulingResult:
        """Schedule for minimal latency."""
        # Topological sort based on dependencies
        dependency_graph = self._build_dependency_graph(tasks)
        sorted_tasks = self._topological_sort(tasks, dependency_graph)
        
        # Apply parallelization where possible
        parallel_groups = self._identify_parallel_groups(sorted_tasks, dependency_graph)
        optimized_tasks = self._optimize_parallel_execution(parallel_groups)
        
        return self._build_execution_plan(optimized_tasks, "latency")
    
    def _adaptive_hybrid_schedule(self, tasks: List[PhotonicTask], 
                                optimize_for: str) -> SchedulingResult:
        """
        Adaptive hybrid scheduling that combines multiple strategies.
        """
        # Analyze task characteristics
        task_analysis = self._analyze_task_batch(tasks)
        
        # Choose strategy based on analysis and current system state
        if task_analysis["thermal_intensive_ratio"] > 0.6:
            return self._thermal_aware_fifo_schedule(tasks)
        elif task_analysis["dependency_ratio"] > 0.4:
            return self._latency_minimal_schedule(tasks)
        elif task_analysis["energy_critical_ratio"] > 0.3:
            return self._energy_optimal_schedule(tasks)
        else:
            # Use quantum annealing for complex optimization
            return self._quantum_annealing_schedule(tasks, optimize_for)
    
    def _evaluate_schedule_cost(self, tasks: List[PhotonicTask], 
                               order: List[int], optimize_for: str) -> float:
        """Evaluate the cost of a given task ordering."""
        ordered_tasks = [tasks[i] for i in order]
        
        # Simulate execution
        current_time = 0.0
        thermal_load = 0.0
        energy_cost = 0.0
        total_latency = 0.0
        
        for task in ordered_tasks:
            # Add thermal constraints
            if thermal_load + task.thermal_cost > 1.0:
                # Need cooling delay
                cooling_time = self._calculate_cooling_time(thermal_load)
                current_time += cooling_time
                thermal_load *= 0.5  # Cooling reduces thermal load
            
            # Execute task
            task_latency = task.estimated_duration_ms
            task_energy = task.thermal_cost * task_latency * 0.001  # mJ
            
            current_time += task_latency
            thermal_load += task.thermal_cost
            energy_cost += task_energy
            total_latency = current_time
        
        # Calculate composite cost based on optimization target
        if optimize_for == "thermal":
            return thermal_load * 1000 + total_latency * 0.1
        elif optimize_for == "energy":
            return energy_cost * 1000 + total_latency * 0.1
        elif optimize_for == "latency":
            return total_latency + energy_cost * 0.01
        else:  # balanced
            return total_latency + thermal_load * 100 + energy_cost * 10
    
    def _build_execution_plan(self, tasks: List[PhotonicTask], 
                            optimize_for: str) -> SchedulingResult:
        """Build detailed execution plan from ordered tasks."""
        execution_plan = []
        current_time = 0.0
        total_energy = 0.0
        max_thermal_load = 0.0
        current_thermal_load = 0.0
        
        for task in tasks:
            # Check thermal constraints
            if current_thermal_load + task.thermal_cost > 1.0:
                cooling_time = self._calculate_cooling_time(current_thermal_load)
                current_time += cooling_time
                current_thermal_load *= 0.6  # Partial cooling
            
            # Schedule task
            execution_plan.append((current_time, task.task_id))
            
            # Update metrics
            task_energy = task.thermal_cost * task.estimated_duration_ms * 0.001
            total_energy += task_energy
            current_thermal_load += task.thermal_cost
            max_thermal_load = max(max_thermal_load, current_thermal_load)
            current_time += task.estimated_duration_ms
        
        # Calculate efficiency metrics
        thermal_efficiency = 1.0 - (max_thermal_load / len(tasks))
        resource_utilization = len(tasks) / (current_time / 1000.0) if current_time > 0 else 0.0
        
        return SchedulingResult(
            scheduled_tasks=tasks,
            execution_plan=execution_plan,
            total_execution_time_ms=current_time,
            thermal_efficiency=thermal_efficiency,
            resource_utilization=resource_utilization,
            energy_estimate_mj=total_energy
        )
    
    def _estimate_thermal_cost(self, task: PhotonicTask) -> float:
        """Estimate thermal cost of a task."""
        base_costs = {
            "matmul": 0.3,
            "phase_shift": 0.1,
            "thermal_compensation": 0.05,
            "quantum_phase_gate": 0.2,
            "wavelength_multiplex": 0.15,
            "mesh_calibration": 0.4,
            "power_balancing": 0.1
        }
        
        base_cost = base_costs.get(task.operation_type, 0.2)
        
        # Adjust for task complexity
        if hasattr(task.input_data, 'size') and task.input_data.size > 1000000:
            base_cost *= 1.5
        
        # Adjust for wavelength count
        if len(task.wavelength_requirements) > 1:
            base_cost *= (1 + 0.1 * len(task.wavelength_requirements))
        
        return min(base_cost, 1.0)
    
    def _estimate_duration(self, task: PhotonicTask) -> float:
        """Estimate task execution duration in milliseconds."""
        base_durations = {
            "matmul": 10.0,
            "phase_shift": 2.0,
            "thermal_compensation": 50.0,
            "quantum_phase_gate": 5.0,
            "wavelength_multiplex": 8.0,
            "mesh_calibration": 100.0,
            "power_balancing": 15.0
        }
        
        base_duration = base_durations.get(task.operation_type, 10.0)
        
        # Adjust for data size
        if hasattr(task.input_data, 'size'):
            size_factor = max(1.0, np.log10(task.input_data.size) / 3.0)
            base_duration *= size_factor
        
        return base_duration
    
    def _calculate_cooling_time(self, thermal_load: float) -> float:
        """Calculate required cooling time for given thermal load."""
        return max(0.0, thermal_load * 100.0)  # ms
    
    def _analyze_task_batch(self, tasks: List[PhotonicTask]) -> Dict[str, float]:
        """Analyze characteristics of a task batch."""
        if not tasks:
            return {"thermal_intensive_ratio": 0.0, "dependency_ratio": 0.0, "energy_critical_ratio": 0.0}
        
        thermal_intensive = sum(1 for t in tasks if t.thermal_cost > 0.5)
        has_dependencies = sum(1 for t in tasks if t.dependencies)
        energy_critical = sum(1 for t in tasks if t.priority in [TaskPriority.CRITICAL, TaskPriority.HIGH])
        
        total = len(tasks)
        return {
            "thermal_intensive_ratio": thermal_intensive / total,
            "dependency_ratio": has_dependencies / total,
            "energy_critical_ratio": energy_critical / total
        }
    
    def _create_cooling_task(self) -> PhotonicTask:
        """Create a cooling/idle task for thermal management."""
        return PhotonicTask(
            task_id=f"cooling_{int(time.time() * 1000000)}",
            operation_type="cooling",
            input_data=None,
            parameters={},
            priority=TaskPriority.NORMAL,
            thermal_cost=-0.3,  # Negative for cooling
            estimated_duration_ms=50.0
        )
    
    def _build_dependency_graph(self, tasks: List[PhotonicTask]) -> Dict[str, List[str]]:
        """Build dependency graph from tasks."""
        graph = {}
        for task in tasks:
            graph[task.task_id] = task.dependencies.copy()
        return graph
    
    def _topological_sort(self, tasks: List[PhotonicTask], 
                         graph: Dict[str, List[str]]) -> List[PhotonicTask]:
        """Perform topological sort on tasks based on dependencies."""
        # Simple implementation - in production would use more efficient algorithm
        task_map = {task.task_id: task for task in tasks}
        sorted_tasks = []
        remaining = set(task.task_id for task in tasks)
        
        while remaining:
            # Find tasks with no unresolved dependencies
            ready = [tid for tid in remaining 
                    if all(dep not in remaining for dep in graph.get(tid, []))]
            
            if not ready:
                # Break cycles by selecting highest priority task
                ready = [min(remaining, key=lambda tid: task_map[tid].priority.value)]
            
            for task_id in ready:
                sorted_tasks.append(task_map[task_id])
                remaining.remove(task_id)
        
        return sorted_tasks
    
    def _identify_parallel_groups(self, tasks: List[PhotonicTask], 
                                graph: Dict[str, List[str]]) -> List[List[PhotonicTask]]:
        """Identify tasks that can run in parallel."""
        groups = []
        remaining = tasks.copy()
        
        while remaining:
            # Find tasks with no dependencies in remaining set
            parallel_group = []
            for task in remaining[:]:
                deps_satisfied = all(dep not in [t.task_id for t in remaining] 
                                   for dep in graph.get(task.task_id, []))
                if deps_satisfied:
                    parallel_group.append(task)
                    remaining.remove(task)
            
            if parallel_group:
                groups.append(parallel_group)
            else:
                # Force progress by taking first task
                groups.append([remaining.pop(0)])
        
        return groups
    
    def _optimize_parallel_execution(self, parallel_groups: List[List[PhotonicTask]]) -> List[PhotonicTask]:
        """Optimize execution within parallel groups."""
        optimized = []
        for group in parallel_groups:
            # Sort group by priority and thermal cost
            group.sort(key=lambda t: (t.priority.value, t.thermal_cost))
            optimized.extend(group)
        return optimized
    
    def _simple_schedule(self, tasks: List[PhotonicTask]) -> SchedulingResult:
        """Simple scheduling for single task or fallback."""
        return self._build_execution_plan(tasks, "balanced")
    
    def get_scheduling_stats(self) -> Dict[str, Any]:
        """Get comprehensive scheduling statistics."""
        if not self.scheduling_history:
            return {"total_batches": 0, "avg_efficiency": 0.0}
        
        total_batches = len(self.scheduling_history)
        avg_thermal_efficiency = np.mean([r.thermal_efficiency for r in self.scheduling_history])
        avg_resource_utilization = np.mean([r.resource_utilization for r in self.scheduling_history])
        total_energy = sum(r.energy_estimate_mj for r in self.scheduling_history)
        avg_execution_time = np.mean([r.total_execution_time_ms for r in self.scheduling_history])
        
        return {
            "total_batches": total_batches,
            "avg_thermal_efficiency": avg_thermal_efficiency,
            "avg_resource_utilization": avg_resource_utilization,
            "total_energy_consumption_mj": total_energy,
            "avg_execution_time_ms": avg_execution_time,
            "strategy": self.strategy.value
        }
    
    def reset_scheduler(self):
        """Reset scheduler state."""
        with self.scheduler_lock:
            self.task_queue = queue.PriorityQueue()
            self.active_tasks.clear()
            self.completed_tasks.clear()
            self.task_results.clear()
            self.mesh_utilization.fill(0.0)
            self.wavelength_allocation.clear()
            self.thermal_state.fill(0.0)
            self.current_time = 0.0
            self.total_energy_consumed = 0.0
            self.scheduling_history.clear()
    
    def __enter__(self):
        self.running = True
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        self.executor.shutdown(wait=True)


# Convenience functions
def create_quantum_scheduler(target_config: TargetConfig, 
                           strategy: str = "adaptive_hybrid") -> QuantumAwareScheduler:
    """Create a quantum-aware scheduler with specified strategy."""
    strategy_map = {
        "quantum_annealing": SchedulingStrategy.QUANTUM_ANNEALING,
        "thermal_aware": SchedulingStrategy.THERMAL_AWARE_FIFO,
        "energy_optimal": SchedulingStrategy.ENERGY_OPTIMAL,
        "latency_minimal": SchedulingStrategy.LATENCY_MINIMAL,
        "adaptive_hybrid": SchedulingStrategy.ADAPTIVE_HYBRID
    }
    
    return QuantumAwareScheduler(
        target_config=target_config,
        strategy=strategy_map.get(strategy, SchedulingStrategy.ADAPTIVE_HYBRID)
    )


def create_photonic_task(task_id: str, operation: str, data: Any, 
                        priority: str = "normal", **kwargs) -> PhotonicTask:
    """Create a photonic task with specified parameters."""
    priority_map = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "normal": TaskPriority.NORMAL,
        "low": TaskPriority.LOW
    }
    
    return PhotonicTask(
        task_id=task_id,
        operation_type=operation,
        input_data=data,
        parameters=kwargs,
        priority=priority_map.get(priority, TaskPriority.NORMAL)
    )