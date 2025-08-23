"""
Quantum-inspired task scheduling and optimization for photonic compilation.

This module implements quantum computing principles for optimizing task scheduling
in photonic neural network compilation, including superposition of scheduling states,
quantum annealing for optimization, and entanglement-inspired dependency resolution.
"""

try:
    import numpy as np
except ImportError:
    from .numpy_fallback import get_numpy
    np = get_numpy()
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of compilation tasks in the photonic pipeline."""
    GRAPH_LOWERING = "graph_lowering"
    PHOTONIC_OPTIMIZATION = "photonic_optimization"
    THERMAL_COMPENSATION = "thermal_compensation"
    MESH_MAPPING = "mesh_mapping"
    PHASE_OPTIMIZATION = "phase_optimization"
    POWER_BALANCING = "power_balancing"
    WAVELENGTH_ALLOCATION = "wavelength_allocation"  # New: Wavelength resource management
    CROSSTALK_MINIMIZATION = "crosstalk_minimization"  # New: Optical crosstalk reduction
    CALIBRATION_INJECTION = "calibration_injection"  # New: Runtime calibration insertion
    CODE_GENERATION = "code_generation"


class QuantumState(Enum):
    """Quantum-inspired states for task scheduling."""
    SUPERPOSITION = "superposition"  # Task can be in multiple scheduling positions
    ENTANGLED = "entangled"  # Task is correlated with other tasks
    COLLAPSED = "collapsed"  # Task has been assigned to specific schedule slot


@dataclass
class CompilationTask:
    """Represents a compilation task with quantum-inspired properties."""
    id: str
    task_type: TaskType
    dependencies: Set[str] = field(default_factory=set)
    estimated_duration: float = 1.0  # seconds
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    priority: float = 1.0
    quantum_state: QuantumState = QuantumState.SUPERPOSITION
    superposition_weights: Dict[int, float] = field(default_factory=dict)
    entangled_tasks: Set[str] = field(default_factory=set)
    
    # Photonic-specific properties
    thermal_load: float = 0.0  # Thermal energy generated (mW)
    phase_shifts_required: int = 0  # Number of phase shifters needed
    wavelength_channels: Set[int] = field(default_factory=set)  # Required wavelength channels
    optical_power_budget: float = 1.0  # Optical power consumption (mW)
    crosstalk_sensitivity: float = 0.1  # Sensitivity to optical crosstalk (0-1)
    calibration_frequency: float = 0.0  # Required calibration rate (Hz)
    
    def __post_init__(self):
        """Initialize quantum and photonic properties."""
        if not self.resource_requirements:
            self.resource_requirements = {
                "cpu": 1.0,
                "memory": 512.0,  # MB
                "gpu": 0.0
            }
        
        # Initialize photonic-specific properties based on task type
        self._initialize_photonic_properties()
    
    def _initialize_photonic_properties(self):
        """Initialize photonic-specific properties based on task type."""
        photonic_profiles = {
            TaskType.PHASE_OPTIMIZATION: {
                "thermal_load": 15.0,  # High thermal load from phase shifters
                "phase_shifts_required": 64,
                "optical_power_budget": 2.5,
                "crosstalk_sensitivity": 0.3
            },
            TaskType.THERMAL_COMPENSATION: {
                "thermal_load": 5.0,  # Low thermal load
                "calibration_frequency": 10.0,  # 10 Hz calibration
                "optical_power_budget": 0.5
            },
            TaskType.WAVELENGTH_ALLOCATION: {
                "wavelength_channels": {1550, 1551, 1552},  # C-band channels
                "optical_power_budget": 3.0,
                "crosstalk_sensitivity": 0.8
            },
            TaskType.CROSSTALK_MINIMIZATION: {
                "thermal_load": 2.0,
                "crosstalk_sensitivity": 0.05,  # Very low after optimization
                "optical_power_budget": 1.0
            }
        }
        
        if self.task_type in photonic_profiles:
            profile = photonic_profiles[self.task_type]
            for key, value in profile.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def get_thermal_footprint(self) -> float:
        """Calculate thermal footprint considering task duration and load."""
        return self.thermal_load * self.estimated_duration
    
    def get_optical_complexity(self) -> float:
        """Calculate optical complexity score."""
        complexity = (
            self.phase_shifts_required * 0.1 +
            len(self.wavelength_channels) * 0.5 +
            self.optical_power_budget * 0.2 +
            self.crosstalk_sensitivity * 10
        )
        return complexity


@dataclass 
class SchedulingState:
    """Represents a quantum-inspired scheduling state with photonic awareness."""
    tasks: List[CompilationTask]
    schedule: Dict[int, List[str]]  # timeslot -> task_ids
    total_energy: float = float('inf')
    makespan: float = float('inf')
    resource_utilization: float = 0.0
    
    # Photonic-specific metrics
    peak_thermal_load: float = 0.0  # Peak simultaneous thermal load (mW)
    total_phase_shifts: int = 0  # Total phase shifts required
    wavelength_utilization: float = 0.0  # Wavelength channel efficiency
    optical_power_efficiency: float = 0.0  # Optical power usage efficiency
    thermal_hotspots: List[Tuple[int, float]] = field(default_factory=list)  # (timeslot, thermal_load)
    crosstalk_violations: int = 0  # Number of potential crosstalk issues
    
    def calculate_fitness(self) -> float:
        """Calculate fitness score using quantum-inspired energy function."""
        # Minimize makespan and maximize resource utilization
        if self.makespan == 0:
            return float('inf')
        
        utilization_bonus = self.resource_utilization * 10
        dependency_penalty = self._calculate_dependency_violations() * 100
        
        return self.makespan - utilization_bonus + dependency_penalty
    
    def _calculate_dependency_violations(self) -> int:
        """Count dependency violations in current schedule."""
        violations = 0
        task_slots = {}
        
        # Build task -> slot mapping
        for slot, task_ids in self.schedule.items():
            for task_id in task_ids:
                task_slots[task_id] = slot
        
        # Check dependencies
        for task in self.tasks:
            if task.id in task_slots:
                task_slot = task_slots[task.id]
                for dep_id in task.dependencies:
                    if dep_id in task_slots:
                        dep_slot = task_slots[dep_id]
                        if dep_slot >= task_slot:
                            violations += 1
        
        return violations


class QuantumInspiredScheduler:
    """Quantum-inspired task scheduler using superposition and annealing."""
    
    def __init__(self, 
                 population_size: int = 50,
                 max_iterations: int = 1000,
                 initial_temperature: float = 100.0,
                 cooling_rate: float = 0.95,
                 mutation_rate: float = 0.1,
                 enable_validation: bool = True,
                 enable_monitoring: bool = True):
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.mutation_rate = mutation_rate
        self.population: List[SchedulingState] = []
        
        # Robust features
        self.enable_validation = enable_validation
        self.enable_monitoring = enable_monitoring
        self.convergence_threshold = 0.001
        self.stagnation_limit = 100  # Iterations without improvement
        self.early_stopping = True
        
        # Error handling
        self.error_recovery_attempts = 3
        self.fallback_strategies = ["random_restart", "population_diversification", "greedy_fallback"]
        
        # Performance tracking
        self.iteration_times: List[float] = []
        self.convergence_history: List[float] = []
        self.best_energy_history: List[float] = []
        
    def schedule_tasks(self, tasks: List[CompilationTask]) -> SchedulingState:
        """
        Schedule tasks using quantum-inspired optimization with robust error handling.
        
        Args:
            tasks: List of compilation tasks to schedule
            
        Returns:
            Optimal scheduling state
            
        Raises:
            ValueError: If tasks are invalid
            RuntimeError: If scheduling fails after all recovery attempts
        """
        if not tasks:
            raise ValueError("No tasks provided for scheduling")
        
        logger.info(f"Starting quantum-inspired scheduling for {len(tasks)} tasks")
        start_time = time.time()
        
        # Input validation
        if self.enable_validation:
            from .quantum_validation import QuantumValidator, ValidationLevel
            validator = QuantumValidator(ValidationLevel.STRICT)
            validation_result = validator.validate_tasks(tasks)
            
            if not validation_result.is_valid:
                error_msg = f"Task validation failed: {'; '.join(validation_result.errors)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            if validation_result.warnings:
                logger.warning(f"Task validation warnings: {'; '.join(validation_result.warnings)}")
        
        best_state = None
        attempts = 0
        
        while attempts < self.error_recovery_attempts:
            try:
                attempts += 1
                logger.debug(f"Scheduling attempt {attempts}/{self.error_recovery_attempts}")
                
                # Clear previous state
                self.iteration_times.clear()
                self.convergence_history.clear()
                self.best_energy_history.clear()
                
                # Phase 1: Initialize superposition states
                self._initialize_superposition(tasks)
                
                # Phase 2: Quantum annealing optimization
                best_state = self._quantum_annealing_robust(tasks)
                
                # Phase 3: Collapse quantum states to final schedule
                self._collapse_states(best_state)
                
                # Validate result
                if self.enable_validation:
                    schedule_validation = validator.validate_schedule(best_state)
                    if not schedule_validation.is_valid:
                        raise RuntimeError(f"Generated schedule is invalid: {'; '.join(schedule_validation.errors)}")
                
                # Success
                break
                
            except Exception as e:
                logger.warning(f"Scheduling attempt {attempts} failed: {e}")
                
                if attempts < self.error_recovery_attempts:
                    self._apply_recovery_strategy(attempts - 1)
                else:
                    logger.error("All scheduling attempts failed")
                    raise RuntimeError(f"Scheduling failed after {self.error_recovery_attempts} attempts: {e}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        logger.info(f"Scheduling complete in {total_time:.3f}s after {attempts} attempts. "
                   f"Makespan: {best_state.makespan:.2f}s, "
                   f"Utilization: {best_state.resource_utilization:.2%}")
        
        return best_state
    
    def _initialize_superposition(self, tasks: List[CompilationTask]):
        """Initialize tasks in quantum superposition of scheduling possibilities."""
        logger.debug("Initializing quantum superposition states")
        
        # Estimate total scheduling horizon
        total_duration = sum(task.estimated_duration for task in tasks)
        max_slots = int(total_duration * 2)  # Allow for parallelization
        
        # Initialize each task in superposition across possible time slots
        for task in tasks:
            task.quantum_state = QuantumState.SUPERPOSITION
            task.superposition_weights = {}
            
            # Create probability distribution over time slots
            for slot in range(max_slots):
                # Higher probability for earlier slots (weighted by priority)
                weight = math.exp(-slot * 0.1) * task.priority
                # Reduce probability if dependencies not satisfied
                if task.dependencies:
                    weight *= 0.5  # Prefer later slots for dependent tasks
                task.superposition_weights[slot] = weight
            
            # Normalize weights
            total_weight = sum(task.superposition_weights.values())
            for slot in task.superposition_weights:
                task.superposition_weights[slot] /= total_weight
    
    def _quantum_annealing(self, tasks: List[CompilationTask]) -> SchedulingState:
        """Use quantum annealing to find optimal schedule."""
        logger.debug("Starting quantum annealing optimization")
        
        # Initialize population of scheduling states
        self.population = [self._generate_random_state(tasks) 
                          for _ in range(self.population_size)]
        
        # Evaluate initial population
        for state in self.population:
            state.total_energy = state.calculate_fitness()
        
        best_state = min(self.population, key=lambda s: s.total_energy)
        temperature = self.initial_temperature
        
        for iteration in range(self.max_iterations):
            # Quantum tunneling: allow exploration of distant states
            if random.random() < 0.1:  # 10% chance of quantum tunneling
                self._quantum_tunnel(tasks)
            
            # Standard annealing step
            new_population = []
            for state in self.population:
                neighbor = self._generate_neighbor(state, tasks)
                neighbor.total_energy = neighbor.calculate_fitness()
                
                # Acceptance probability with temperature
                if neighbor.total_energy < state.total_energy:
                    new_population.append(neighbor)
                else:
                    delta = neighbor.total_energy - state.total_energy
                    probability = math.exp(-delta / temperature)
                    if random.random() < probability:
                        new_population.append(neighbor)
                    else:
                        new_population.append(state)
            
            self.population = new_population
            current_best = min(self.population, key=lambda s: s.total_energy)
            
            if current_best.total_energy < best_state.total_energy:
                best_state = current_best
                logger.debug(f"Iteration {iteration}: New best energy: {best_state.total_energy:.2f}")
            
            # Cool the system
            temperature *= self.cooling_rate
            
            # Early termination if converged
            if temperature < 0.001:
                break
        
        logger.info(f"Annealing completed in {iteration + 1} iterations")
        return best_state
    
    def _generate_random_state(self, tasks: List[CompilationTask]) -> SchedulingState:
        """Generate a random scheduling state respecting dependencies."""
        schedule = {}
        task_slots = {}
        
        # Topological sort to respect dependencies
        sorted_tasks = self._topological_sort(tasks)
        
        for task in sorted_tasks:
            # Find earliest available slot considering dependencies
            min_slot = 0
            for dep_id in task.dependencies:
                if dep_id in task_slots:
                    # Must start after dependency completes
                    dep_slot = task_slots[dep_id]
                    dep_task = next(t for t in tasks if t.id == dep_id)
                    completion_slot = dep_slot + int(dep_task.estimated_duration)
                    min_slot = max(min_slot, completion_slot)
            
            # Add some randomness while respecting constraints
            slot = min_slot + random.randint(0, 3)
            
            if slot not in schedule:
                schedule[slot] = []
            schedule[slot].append(task.id)
            task_slots[task.id] = slot
        
        state = SchedulingState(tasks=tasks, schedule=schedule)
        self._calculate_schedule_metrics(state)
        return state
    
    def _generate_neighbor(self, state: SchedulingState, tasks: List[CompilationTask]) -> SchedulingState:
        """Generate a neighboring state using quantum-inspired mutations."""
        new_schedule = {k: list(v) for k, v in state.schedule.items()}
        
        # Quantum mutation: move task to different superposition state
        if random.random() < self.mutation_rate:
            # Select random task
            all_tasks = [task_id for task_list in new_schedule.values() 
                        for task_id in task_list]
            if all_tasks:
                task_id = random.choice(all_tasks)
                task = next(t for t in tasks if t.id == task_id)
                
                # Remove from current slot
                for slot, task_list in new_schedule.items():
                    if task_id in task_list:
                        task_list.remove(task_id)
                        if not task_list:
                            del new_schedule[slot]
                        break
                
                # Find new slot using superposition weights
                if task.superposition_weights:
                    weights = list(task.superposition_weights.values())
                    slots = list(task.superposition_weights.keys())
                    new_slot = np.random.choice(slots, p=weights)
                else:
                    new_slot = random.randint(0, max(new_schedule.keys()) + 3)
                
                # Validate dependencies
                if self._validate_dependencies(task, new_slot, new_schedule, tasks):
                    if new_slot not in new_schedule:
                        new_schedule[new_slot] = []
                    new_schedule[new_slot].append(task_id)
        
        new_state = SchedulingState(tasks=tasks, schedule=new_schedule)
        self._calculate_schedule_metrics(new_state)
        return new_state
    
    def _quantum_tunnel(self, tasks: List[CompilationTask]):
        """Implement quantum tunneling for exploration of distant states."""
        # Replace worst states with completely random new states
        self.population.sort(key=lambda s: s.total_energy)
        num_replace = self.population_size // 4  # Replace worst 25%
        
        for i in range(-num_replace, 0):
            self.population[i] = self._generate_random_state(tasks)
            self.population[i].total_energy = self.population[i].calculate_fitness()
    
    def _topological_sort(self, tasks: List[CompilationTask]) -> List[CompilationTask]:
        """Perform topological sort of tasks based on dependencies."""
        in_degree = {task.id: 0 for task in tasks}
        task_map = {task.id: task for task in tasks}
        
        # Calculate in-degrees
        for task in tasks:
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[task.id] += 1
        
        # Kahn's algorithm
        queue = [task_id for task_id, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current_id = queue.pop(0)
            result.append(task_map[current_id])
            
            # Update dependencies
            for task in tasks:
                if current_id in task.dependencies:
                    in_degree[task.id] -= 1
                    if in_degree[task.id] == 0:
                        queue.append(task.id)
        
        return result
    
    def _validate_dependencies(self, task: CompilationTask, slot: int,
                              schedule: Dict[int, List[str]], tasks: List[CompilationTask]) -> bool:
        """Validate that dependencies are satisfied for a task at a given slot."""
        task_map = {t.id: t for t in tasks}
        
        for dep_id in task.dependencies:
            dep_task = task_map.get(dep_id)
            if not dep_task:
                continue
                
            # Find dependency's slot
            dep_slot = None
            for s, task_list in schedule.items():
                if dep_id in task_list:
                    dep_slot = s
                    break
            
            if dep_slot is not None:
                # Dependency must complete before task starts
                completion_slot = dep_slot + int(dep_task.estimated_duration)
                if completion_slot > slot:
                    return False
        
        return True
    
    def _calculate_schedule_metrics(self, state: SchedulingState):
        """Calculate scheduling metrics for a state."""
        if not state.schedule:
            state.makespan = 0
            state.resource_utilization = 0
            return
        
        # Calculate makespan
        max_completion = 0
        task_map = {t.id: t for t in state.tasks}
        
        for slot, task_ids in state.schedule.items():
            for task_id in task_ids:
                task = task_map[task_id]
                completion_time = slot + task.estimated_duration
                max_completion = max(max_completion, completion_time)
        
        state.makespan = max_completion
        
        # Calculate resource utilization
        total_task_time = sum(task.estimated_duration for task in state.tasks)
        if state.makespan > 0:
            state.resource_utilization = total_task_time / state.makespan
        else:
            state.resource_utilization = 0
    
    def _collapse_states(self, state: SchedulingState):
        """Collapse quantum superposition states to final schedule."""
        logger.debug("Collapsing quantum states to final schedule")
        
        for task in state.tasks:
            task.quantum_state = QuantumState.COLLAPSED
            task.superposition_weights.clear()
    
    def _quantum_annealing_robust(self, tasks: List[CompilationTask]) -> SchedulingState:
        """Robust quantum annealing with error handling and monitoring."""
        logger.debug("Starting robust quantum annealing optimization")
        
        try:
            # Initialize population with error checking
            self.population = []
            for i in range(self.population_size):
                try:
                    state = self._generate_random_state(tasks)
                    self.population.append(state)
                except Exception as e:
                    logger.warning(f"Failed to generate initial state {i}: {e}")
                    # Use fallback simple state
                    state = self._generate_fallback_state(tasks)
                    self.population.append(state)
            
            if not self.population:
                raise RuntimeError("Failed to initialize any valid states")
            
            # Evaluate initial population
            for state in self.population:
                try:
                    state.total_energy = state.calculate_fitness()
                except Exception as e:
                    logger.warning(f"Failed to calculate fitness, using penalty: {e}")
                    state.total_energy = float('inf')
            
            best_state = min(self.population, key=lambda s: s.total_energy)
            if best_state.total_energy == float('inf'):
                raise RuntimeError("No valid initial states found")
            
            temperature = self.initial_temperature
            stagnation_count = 0
            last_best_energy = best_state.total_energy
            
            for iteration in range(self.max_iterations):
                iteration_start = time.time()
                
                try:
                    # Quantum tunneling with error recovery
                    if random.random() < 0.1:
                        self._quantum_tunnel_safe(tasks)
                    
                    # Standard annealing step
                    new_population = []
                    for state in self.population:
                        try:
                            neighbor = self._generate_neighbor(state, tasks)
                            neighbor.total_energy = neighbor.calculate_fitness()
                            
                            # Acceptance probability with temperature
                            if neighbor.total_energy < state.total_energy:
                                new_population.append(neighbor)
                            else:
                                delta = neighbor.total_energy - state.total_energy
                                if delta < 1000:  # Prevent overflow
                                    probability = math.exp(-delta / max(temperature, 0.001))
                                    if random.random() < probability:
                                        new_population.append(neighbor)
                                    else:
                                        new_population.append(state)
                                else:
                                    new_population.append(state)
                        
                        except Exception as e:
                            logger.debug(f"Neighbor generation failed: {e}")
                            new_population.append(state)  # Keep original state
                    
                    if new_population:
                        self.population = new_population
                    
                    current_best = min(self.population, key=lambda s: s.total_energy)
                    
                    # Update best state
                    if current_best.total_energy < best_state.total_energy:
                        best_state = current_best
                        stagnation_count = 0
                        logger.debug(f"Iteration {iteration}: New best energy: {best_state.total_energy:.2f}")
                    else:
                        stagnation_count += 1
                    
                    # Track performance
                    iteration_time = time.time() - iteration_start
                    self.iteration_times.append(iteration_time)
                    self.best_energy_history.append(best_state.total_energy)
                    
                    convergence_rate = abs(last_best_energy - best_state.total_energy) / max(last_best_energy, 0.001)
                    self.convergence_history.append(convergence_rate)
                    last_best_energy = best_state.total_energy
                    
                    # Cool the system
                    temperature *= self.cooling_rate
                    
                    # Early termination conditions
                    if self.early_stopping:
                        if temperature < self.convergence_threshold:
                            logger.debug(f"Converged at iteration {iteration} (temperature: {temperature:.6f})")
                            break
                        
                        if stagnation_count >= self.stagnation_limit:
                            logger.debug(f"Stagnated at iteration {iteration} (no improvement for {stagnation_count} iterations)")
                            break
                
                except Exception as e:
                    logger.warning(f"Iteration {iteration} failed: {e}")
                    # Try to continue with existing population
                    continue
            
            logger.info(f"Annealing completed in {iteration + 1} iterations")
            return best_state
        
        except Exception as e:
            logger.error(f"Quantum annealing failed: {e}")
            # Return fallback schedule
            return self._generate_fallback_state(tasks)
    
    def _generate_fallback_state(self, tasks: List[CompilationTask]) -> SchedulingState:
        """Generate a simple fallback schedule."""
        logger.debug("Generating fallback schedule")
        
        schedule = {}
        current_slot = 0
        
        # Simple sequential schedule
        sorted_tasks = self._topological_sort(tasks)
        
        for task in sorted_tasks:
            schedule[current_slot] = [task.id]
            current_slot += int(task.estimated_duration) + 1
        
        state = SchedulingState(tasks=tasks, schedule=schedule)
        self._calculate_schedule_metrics(state)
        return state
    
    def _quantum_tunnel_safe(self, tasks: List[CompilationTask]):
        """Safe quantum tunneling with error handling."""
        try:
            # Replace worst states with completely random new states
            self.population.sort(key=lambda s: s.total_energy)
            num_replace = max(1, self.population_size // 4)  # Replace worst 25%
            
            for i in range(-num_replace, 0):
                try:
                    self.population[i] = self._generate_random_state(tasks)
                    self.population[i].total_energy = self.population[i].calculate_fitness()
                except Exception as e:
                    logger.debug(f"Failed to replace state {i}: {e}")
                    # Keep the original state
        except Exception as e:
            logger.warning(f"Quantum tunneling failed: {e}")
    
    def _apply_recovery_strategy(self, attempt: int):
        """Apply recovery strategy based on attempt number."""
        strategy = self.fallback_strategies[min(attempt, len(self.fallback_strategies) - 1)]
        
        logger.info(f"Applying recovery strategy: {strategy}")
        
        if strategy == "random_restart":
            # Clear population for fresh start
            self.population.clear()
            # Increase population size for better exploration
            self.population_size = min(self.population_size * 2, 200)
        
        elif strategy == "population_diversification":
            # Increase mutation rate and temperature
            self.mutation_rate = min(self.mutation_rate * 2, 0.5)
            self.initial_temperature *= 2
        
        elif strategy == "greedy_fallback":
            # Use simpler, more conservative parameters
            self.population_size = max(self.population_size // 2, 10)
            self.max_iterations = max(self.max_iterations // 2, 100)
            self.mutation_rate = 0.05  # Very conservative
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        if not self.iteration_times:
            return {"status": "no_data"}
        
        return {
            "total_iterations": len(self.iteration_times),
            "average_iteration_time": sum(self.iteration_times) / len(self.iteration_times),
            "total_optimization_time": sum(self.iteration_times),
            "convergence_rate": self.convergence_history[-1] if self.convergence_history else 0,
            "best_energy": self.best_energy_history[-1] if self.best_energy_history else float('inf'),
            "energy_improvement": (
                (self.best_energy_history[0] - self.best_energy_history[-1]) / self.best_energy_history[0]
                if len(self.best_energy_history) > 1 and self.best_energy_history[0] != 0 else 0
            )
        }


class QuantumTaskPlanner:
    """Main interface for quantum-inspired task planning in photonic compilation."""
    
    def __init__(self):
        self.scheduler = QuantumInspiredScheduler()
        self.task_counter = 0
    
    def create_compilation_plan(self, model_config: Dict[str, Any]) -> List[CompilationTask]:
        """
        Create a compilation plan for a photonic model.
        
        Args:
            model_config: Configuration for the model to be compiled
            
        Returns:
            List of compilation tasks
        """
        tasks = []
        
        # Graph lowering tasks
        graph_task = self._create_task(
            TaskType.GRAPH_LOWERING,
            estimated_duration=2.0,
            resource_requirements={"cpu": 2.0, "memory": 1024.0}
        )
        tasks.append(graph_task)
        
        # Photonic optimization (depends on graph lowering)
        photonic_task = self._create_task(
            TaskType.PHOTONIC_OPTIMIZATION,
            dependencies={graph_task.id},
            estimated_duration=5.0,
            resource_requirements={"cpu": 4.0, "memory": 2048.0}
        )
        tasks.append(photonic_task)
        
        # Parallel optimization tasks with photonic-specific tasks
        mesh_task = self._create_task(
            TaskType.MESH_MAPPING,
            dependencies={photonic_task.id},
            estimated_duration=3.0,
            resource_requirements={"cpu": 2.0, "memory": 1024.0}
        )
        
        phase_task = self._create_task(
            TaskType.PHASE_OPTIMIZATION,
            dependencies={photonic_task.id},
            estimated_duration=4.0,
            resource_requirements={"cpu": 3.0, "memory": 1536.0}
        )
        
        power_task = self._create_task(
            TaskType.POWER_BALANCING,
            dependencies={photonic_task.id},
            estimated_duration=2.5,
            resource_requirements={"cpu": 2.0, "memory": 1024.0}
        )
        
        # New photonic-specific tasks
        wavelength_task = self._create_task(
            TaskType.WAVELENGTH_ALLOCATION,
            dependencies={photonic_task.id},
            estimated_duration=2.0,
            resource_requirements={"cpu": 2.0, "memory": 1024.0}
        )
        
        crosstalk_task = self._create_task(
            TaskType.CROSSTALK_MINIMIZATION,
            dependencies={mesh_task.id, wavelength_task.id},
            estimated_duration=3.5,
            resource_requirements={"cpu": 3.0, "memory": 1536.0}
        )
        
        tasks.extend([mesh_task, phase_task, power_task, wavelength_task, crosstalk_task])
        
        # Thermal compensation (depends on mesh, phase, and crosstalk optimization)
        thermal_task = self._create_task(
            TaskType.THERMAL_COMPENSATION,
            dependencies={mesh_task.id, phase_task.id, crosstalk_task.id},
            estimated_duration=3.0,
            resource_requirements={"cpu": 2.0, "memory": 1024.0}
        )
        
        # Calibration injection (depends on thermal compensation)
        calibration_task = self._create_task(
            TaskType.CALIBRATION_INJECTION,
            dependencies={thermal_task.id},
            estimated_duration=1.5,
            resource_requirements={"cpu": 1.0, "memory": 512.0}
        )
        
        tasks.extend([thermal_task, calibration_task])
        
        # Code generation (final step - depends on all optimization tasks)
        codegen_task = self._create_task(
            TaskType.CODE_GENERATION,
            dependencies={mesh_task.id, phase_task.id, power_task.id, 
                         wavelength_task.id, crosstalk_task.id, calibration_task.id},
            estimated_duration=1.5,
            resource_requirements={"cpu": 1.0, "memory": 512.0}
        )
        tasks.append(codegen_task)
        
        return tasks
    
    def optimize_schedule(self, tasks: List[CompilationTask]) -> SchedulingState:
        """
        Optimize task schedule using quantum-inspired algorithms.
        
        Args:
            tasks: List of compilation tasks
            
        Returns:
            Optimized scheduling state
        """
        return self.scheduler.schedule_tasks(tasks)
    
    def _create_task(self, task_type: TaskType, dependencies: Optional[Set[str]] = None,
                    estimated_duration: float = 1.0, 
                    resource_requirements: Optional[Dict[str, float]] = None,
                    priority: float = 1.0) -> CompilationTask:
        """Create a compilation task with unique ID."""
        self.task_counter += 1
        task_id = f"task_{task_type.value}_{self.task_counter:03d}"
        
        return CompilationTask(
            id=task_id,
            task_type=task_type,
            dependencies=dependencies or set(),
            estimated_duration=estimated_duration,
            resource_requirements=resource_requirements or {},
            priority=priority
        )