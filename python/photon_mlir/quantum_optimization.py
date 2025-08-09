"""
Quantum-Inspired Optimization and Scaling Features

Advanced optimization techniques including performance caching, concurrent processing,
load balancing, and adaptive scaling for the quantum-inspired task scheduler.
"""

import logging
import time
import threading
import multiprocessing
import pickle
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from pathlib import Path
import json

from .quantum_scheduler import CompilationTask, SchedulingState, QuantumInspiredScheduler

logger = logging.getLogger(__name__)


class OptimizationLevel(Enum):
    """Optimization levels for quantum scheduling."""
    FAST = "fast"
    BALANCED = "balanced"
    QUALITY = "quality"
    EXTREME = "extreme"


class CacheStrategy(Enum):
    """Caching strategies."""
    NONE = "none"
    MEMORY_ONLY = "memory_only"
    DISK_ONLY = "disk_only"
    HYBRID = "hybrid"
    DISTRIBUTED = "distributed"


@dataclass
class PerformanceProfile:
    """Performance profiling data."""
    task_count: int
    complexity_score: float
    estimated_runtime: float
    memory_requirement: float
    parallelism_factor: float
    cache_hit_rate: float = 0.0
    
    def get_resource_requirements(self) -> Dict[str, float]:
        """Get recommended resource requirements."""
        return {
            "cpu_cores": min(max(self.parallelism_factor, 1), multiprocessing.cpu_count()),
            "memory_mb": self.memory_requirement,
            "threads": min(self.parallelism_factor * 2, 32),
            "timeout_seconds": self.estimated_runtime * 3
        }


@dataclass 
class CacheEntry:
    """Cache entry for scheduling results."""
    key: str
    result: SchedulingState
    timestamp: float
    access_count: int = 0
    computational_cost: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumCache:
    """High-performance caching system for quantum scheduling results."""
    
    def __init__(self, 
                 strategy: CacheStrategy = CacheStrategy.HYBRID,
                 max_memory_entries: int = 1000,
                 max_disk_size_mb: int = 500,
                 cache_dir: Optional[Path] = None):
        
        self.strategy = strategy
        self.max_memory_entries = max_memory_entries
        self.max_disk_size_mb = max_disk_size_mb
        self.cache_dir = cache_dir or Path.home() / ".photon_mlir" / "cache"
        
        # Memory cache
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # LRU tracking
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "disk_reads": 0,
            "disk_writes": 0
        }
        
        # Initialize cache directory
        if self.strategy in [CacheStrategy.DISK_ONLY, CacheStrategy.HYBRID]:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized QuantumCache with strategy: {strategy.value}")
    
    def get(self, tasks: List[CompilationTask]) -> Optional[SchedulingState]:
        """Get cached scheduling result."""
        cache_key = self._generate_cache_key(tasks)
        
        with self._lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                entry = self.memory_cache[cache_key]
                entry.access_count += 1
                self._update_lru_order(cache_key)
                self.stats["hits"] += 1
                logger.debug(f"Cache hit (memory): {cache_key[:16]}...")
                return entry.result
            
            # Check disk cache
            if self.strategy in [CacheStrategy.DISK_ONLY, CacheStrategy.HYBRID]:
                disk_result = self._load_from_disk(cache_key)
                if disk_result:
                    # Promote to memory cache
                    self._store_in_memory(cache_key, disk_result)
                    self.stats["hits"] += 1
                    self.stats["disk_reads"] += 1
                    logger.debug(f"Cache hit (disk): {cache_key[:16]}...")
                    return disk_result.result
            
            # Cache miss
            self.stats["misses"] += 1
            logger.debug(f"Cache miss: {cache_key[:16]}...")
            return None
    
    def put(self, tasks: List[CompilationTask], result: SchedulingState, 
            computational_cost: float = 0.0):
        """Store scheduling result in cache."""
        cache_key = self._generate_cache_key(tasks)
        
        entry = CacheEntry(
            key=cache_key,
            result=result,
            timestamp=time.time(),
            computational_cost=computational_cost,
            metadata={
                "task_count": len(tasks),
                "makespan": result.makespan,
                "utilization": result.resource_utilization
            }
        )
        
        with self._lock:
            # Store in memory
            if self.strategy in [CacheStrategy.MEMORY_ONLY, CacheStrategy.HYBRID]:
                self._store_in_memory(cache_key, entry)
            
            # Store on disk
            if self.strategy in [CacheStrategy.DISK_ONLY, CacheStrategy.HYBRID]:
                self._save_to_disk(cache_key, entry)
        
        logger.debug(f"Cached result: {cache_key[:16]}... (cost: {computational_cost:.2f}s)")
    
    def _generate_cache_key(self, tasks: List[CompilationTask]) -> str:
        """Generate deterministic cache key from tasks."""
        # Create stable representation
        task_repr = []
        
        for task in sorted(tasks, key=lambda t: t.id):
            task_data = {
                "id": task.id,
                "type": task.task_type.value,
                "duration": task.estimated_duration,
                "deps": sorted(list(task.dependencies)),
                "resources": dict(sorted(task.resource_requirements.items()))
            }
            task_repr.append(json.dumps(task_data, sort_keys=True))
        
        full_repr = json.dumps(task_repr, sort_keys=True)
        return hashlib.sha256(full_repr.encode()).hexdigest()
    
    def _store_in_memory(self, key: str, entry: CacheEntry):
        """Store entry in memory cache with LRU eviction."""
        if key in self.memory_cache:
            # Update existing entry
            self.memory_cache[key] = entry
            self._update_lru_order(key)
        else:
            # Add new entry
            self.memory_cache[key] = entry
            self.access_order.append(key)
            
            # Evict if over capacity
            while len(self.memory_cache) > self.max_memory_entries:
                lru_key = self.access_order.pop(0)
                del self.memory_cache[lru_key]
                self.stats["evictions"] += 1
    
    def _update_lru_order(self, key: str):
        """Update LRU order for accessed key."""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def _save_to_disk(self, key: str, entry: CacheEntry):
        """Save entry to disk."""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(entry, f)
            self.stats["disk_writes"] += 1
        except Exception as e:
            logger.warning(f"Failed to save cache entry to disk: {e}")
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load entry from disk."""
        try:
            file_path = self.cache_dir / f"{key}.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache entry from disk: {e}")
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
            
            return {
                **self.stats,
                "memory_entries": len(self.memory_cache),
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def clear(self):
        """Clear all cached entries."""
        with self._lock:
            self.memory_cache.clear()
            self.access_order.clear()
            
            # Clear disk cache
            if self.cache_dir.exists():
                for cache_file in self.cache_dir.glob("*.pkl"):
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to delete cache file {cache_file}: {e}")
        
        logger.info("Cache cleared")


class ParallelQuantumScheduler:
    """Parallel quantum scheduler with advanced optimization features."""
    
    def __init__(self,
                 optimization_level: OptimizationLevel = OptimizationLevel.BALANCED,
                 cache_strategy: CacheStrategy = CacheStrategy.HYBRID,
                 max_workers: Optional[int] = None,
                 enable_profiling: bool = True):
        
        self.optimization_level = optimization_level
        self.max_workers = max_workers or min(multiprocessing.cpu_count(), 8)
        self.enable_profiling = enable_profiling
        
        # Initialize cache
        self.cache = QuantumCache(cache_strategy)
        
        # Get optimization parameters
        self.scheduler_params = self._get_optimization_params()
        
        # Performance tracking
        self.performance_history: List[PerformanceProfile] = []
        self.adaptive_params = {}
        
        logger.info(f"Initialized ParallelQuantumScheduler with optimization level: {optimization_level.value}")
    
    def schedule_tasks_optimized(self, tasks: List[CompilationTask]) -> SchedulingState:
        """
        Optimized task scheduling with caching, profiling, and adaptive optimization.
        
        Args:
            tasks: List of compilation tasks
            
        Returns:
            Optimized scheduling state
        """
        start_time = time.time()
        
        # Check cache first
        cached_result = self.cache.get(tasks)
        if cached_result:
            logger.info(f"Using cached scheduling result for {len(tasks)} tasks")
            return cached_result
        
        # Profile the problem
        profile = self._create_performance_profile(tasks)
        self.performance_history.append(profile)
        
        # Adaptive parameter tuning
        adapted_params = self._adapt_parameters(profile)
        
        # Select execution strategy
        if profile.task_count < 10:
            # Small problems: use single-threaded for speed
            result = self._schedule_sequential(tasks, adapted_params)
        elif profile.complexity_score < 100:
            # Medium problems: use thread-based parallelism
            result = self._schedule_threaded(tasks, adapted_params)
        else:
            # Large problems: use process-based parallelism
            result = self._schedule_distributed(tasks, adapted_params)
        
        # Cache the result
        computation_time = time.time() - start_time
        self.cache.put(tasks, result, computation_time)
        
        logger.info(f"Optimized scheduling completed in {computation_time:.3f}s "
                   f"(makespan: {result.makespan:.2f}s, utilization: {result.resource_utilization:.2%})")
        
        return result
    
    def _get_optimization_params(self) -> Dict[str, Any]:
        """Get parameters based on optimization level."""
        base_params = {
            "population_size": 50,
            "max_iterations": 1000,
            "initial_temperature": 100.0,
            "cooling_rate": 0.95,
            "mutation_rate": 0.1
        }
        
        if self.optimization_level == OptimizationLevel.FAST:
            return {
                **base_params,
                "population_size": 20,
                "max_iterations": 200,
                "mutation_rate": 0.2
            }
        elif self.optimization_level == OptimizationLevel.QUALITY:
            return {
                **base_params,
                "population_size": 100,
                "max_iterations": 2000,
                "mutation_rate": 0.05
            }
        elif self.optimization_level == OptimizationLevel.EXTREME:
            return {
                **base_params,
                "population_size": 200,
                "max_iterations": 5000,
                "initial_temperature": 200.0,
                "mutation_rate": 0.02
            }
        else:  # BALANCED
            return base_params
    
    def _create_performance_profile(self, tasks: List[CompilationTask]) -> PerformanceProfile:
        """Create performance profile for the scheduling problem."""
        task_count = len(tasks)
        
        # Calculate complexity score
        dependency_complexity = sum(len(task.dependencies) for task in tasks)
        resource_complexity = sum(len(task.resource_requirements) for task in tasks)
        duration_variance = self._calculate_duration_variance(tasks)
        
        complexity_score = (
            task_count * 10 +
            dependency_complexity * 5 +
            resource_complexity * 2 +
            duration_variance * 100
        )
        
        # Estimate runtime
        base_runtime = task_count * 0.01  # 10ms per task base cost
        complexity_multiplier = 1 + (complexity_score / 1000)
        estimated_runtime = base_runtime * complexity_multiplier
        
        # Memory requirement
        memory_requirement = task_count * 0.5 + complexity_score * 0.1  # MB
        
        # Parallelism factor
        max_parallel_tasks = self._calculate_max_parallelism(tasks)
        parallelism_factor = min(max_parallel_tasks, self.max_workers)
        
        return PerformanceProfile(
            task_count=task_count,
            complexity_score=complexity_score,
            estimated_runtime=estimated_runtime,
            memory_requirement=memory_requirement,
            parallelism_factor=parallelism_factor
        )
    
    def _calculate_duration_variance(self, tasks: List[CompilationTask]) -> float:
        """Calculate variance in task durations."""
        if not tasks:
            return 0.0
        
        durations = [task.estimated_duration for task in tasks]
        mean_duration = sum(durations) / len(durations)
        variance = sum((d - mean_duration) ** 2 for d in durations) / len(durations)
        
        return variance
    
    def _calculate_max_parallelism(self, tasks: List[CompilationTask]) -> int:
        """Calculate maximum theoretical parallelism."""
        # Build dependency graph
        in_degree = {task.id: len(task.dependencies) for task in tasks}
        
        # Count tasks that can run in parallel at each level
        max_parallel = 0
        remaining_tasks = set(task.id for task in tasks)
        
        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [tid for tid in remaining_tasks if in_degree[tid] == 0]
            max_parallel = max(max_parallel, len(ready_tasks))
            
            # Remove ready tasks and update dependencies
            for ready_task in ready_tasks:
                remaining_tasks.remove(ready_task)
                
                # Update in-degrees
                for task in tasks:
                    if ready_task in task.dependencies:
                        in_degree[task.id] -= 1
        
        return max_parallel
    
    def _adapt_parameters(self, profile: PerformanceProfile) -> Dict[str, Any]:
        """Adapt scheduling parameters based on performance profile."""
        params = self.scheduler_params.copy()
        
        # Adaptive population sizing
        if profile.task_count > 100:
            params["population_size"] = min(params["population_size"] * 2, 500)
        elif profile.task_count < 20:
            params["population_size"] = max(params["population_size"] // 2, 10)
        
        # Adaptive iteration count
        if profile.complexity_score > 500:
            params["max_iterations"] = min(params["max_iterations"] * 2, 10000)
        elif profile.complexity_score < 50:
            params["max_iterations"] = max(params["max_iterations"] // 2, 50)
        
        # Adaptive cooling rate
        if profile.complexity_score > 1000:
            params["cooling_rate"] = 0.99  # Slower cooling for complex problems
        elif profile.complexity_score < 100:
            params["cooling_rate"] = 0.9   # Faster cooling for simple problems
        
        logger.debug(f"Adapted parameters: {params}")
        return params
    
    def _schedule_sequential(self, tasks: List[CompilationTask], params: Dict[str, Any]) -> SchedulingState:
        """Sequential scheduling for small problems."""
        scheduler = QuantumInspiredScheduler(**params)
        return scheduler.schedule_tasks(tasks)
    
    def _schedule_threaded(self, tasks: List[CompilationTask], params: Dict[str, Any]) -> SchedulingState:
        """Thread-based parallel scheduling."""
        # Run multiple schedulers in parallel and select best result
        num_runs = min(4, self.max_workers)
        
        with ThreadPoolExecutor(max_workers=num_runs) as executor:
            futures = []
            
            for i in range(num_runs):
                # Vary parameters slightly for diversity
                run_params = params.copy()
                run_params["population_size"] = max(10, params["population_size"] + (i - num_runs//2) * 10)
                
                scheduler = QuantumInspiredScheduler(**run_params)
                future = executor.submit(scheduler.schedule_tasks, tasks)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=600)  # 10 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Threaded scheduling run failed: {e}")
            
            if not results:
                raise RuntimeError("All threaded scheduling runs failed")
            
            # Return best result
            return min(results, key=lambda s: s.total_energy)
    
    def _schedule_distributed(self, tasks: List[CompilationTask], params: Dict[str, Any]) -> SchedulingState:
        """Process-based distributed scheduling for large problems with auto-scaling."""
        # For very large problems, partition tasks and solve subproblems
        if len(tasks) > 200:
            return self._schedule_hierarchical(tasks, params)
        
        # Auto-scale number of processes based on task complexity and available resources
        optimal_processes = self._calculate_optimal_process_count(tasks)
        
        with ProcessPoolExecutor(max_workers=optimal_processes) as executor:
            futures = []
            
            for i in range(optimal_processes):
                # Create diverse parameter variations with adaptive scaling
                run_params = self._create_adaptive_params(params, i, optimal_processes, tasks)
                
                future = executor.submit(self._run_quantum_scheduler_with_monitoring, tasks, run_params)
                futures.append(future)
            
            # Collect results with adaptive timeout
            adaptive_timeout = self._calculate_adaptive_timeout(tasks)
            results = []
            
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=adaptive_timeout)
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Distributed scheduling process failed: {e}")
            
            if not results:
                raise RuntimeError("All distributed scheduling processes failed")
            
            # Return best result with performance tracking
            best_result = min(results, key=lambda s: s.total_energy)
            self._track_distributed_performance(len(results), optimal_processes, best_result)
            
            return best_result
    
    def _schedule_hierarchical(self, tasks: List[CompilationTask], params: Dict[str, Any]) -> SchedulingState:
        """Hierarchical scheduling for very large problems."""
        logger.info(f"Using hierarchical scheduling for {len(tasks)} tasks")
        
        # Group tasks by type and dependencies
        task_groups = self._partition_tasks(tasks)
        
        # Schedule each group independently
        group_schedules = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            
            for group_tasks in task_groups:
                future = executor.submit(self._run_quantum_scheduler, group_tasks, params)
                futures.append(future)
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    group_schedules.append(result)
                except Exception as e:
                    logger.error(f"Group scheduling failed: {e}")
                    raise
        
        # Merge group schedules
        return self._merge_schedules(group_schedules, tasks)
    
    def _partition_tasks(self, tasks: List[CompilationTask]) -> List[List[CompilationTask]]:
        """Partition tasks into groups for hierarchical scheduling."""
        # Simple partitioning by task type and dependencies
        task_by_type = {}
        
        for task in tasks:
            task_type = task.task_type
            if task_type not in task_by_type:
                task_by_type[task_type] = []
            task_by_type[task_type].append(task)
        
        # Create groups ensuring dependency constraints
        groups = list(task_by_type.values())
        
        # Ensure no group is too large
        max_group_size = 100
        final_groups = []
        
        for group in groups:
            if len(group) <= max_group_size:
                final_groups.append(group)
            else:
                # Split large groups
                for i in range(0, len(group), max_group_size):
                    final_groups.append(group[i:i + max_group_size])
        
        return final_groups
    
    def _merge_schedules(self, group_schedules: List[SchedulingState], all_tasks: List[CompilationTask]) -> SchedulingState:
        """Merge multiple group schedules into a single schedule."""
        merged_schedule = {}
        current_offset = 0
        
        for schedule in group_schedules:
            for slot, task_ids in schedule.schedule.items():
                merged_slot = slot + current_offset
                merged_schedule[merged_slot] = task_ids
            
            # Update offset based on makespan of this group
            current_offset += int(schedule.makespan) + 1
        
        # Create final scheduling state
        final_state = SchedulingState(tasks=all_tasks, schedule=merged_schedule)
        final_state._calculate_schedule_metrics = lambda: None  # Skip recalculation
        
        # Calculate metrics manually
        max_completion = 0
        for slot, task_ids in merged_schedule.items():
            for task_id in task_ids:
                task = next((t for t in all_tasks if t.id == task_id), None)
                if task:
                    completion_time = slot + task.estimated_duration
                    max_completion = max(max_completion, completion_time)
        
        final_state.makespan = max_completion
        
        total_task_time = sum(task.estimated_duration for task in all_tasks)
        final_state.resource_utilization = total_task_time / max_completion if max_completion > 0 else 0
        
        return final_state
    
    @staticmethod
    def _run_quantum_scheduler(tasks: List[CompilationTask], params: Dict[str, Any]) -> SchedulingState:
        """Static method to run quantum scheduler (for multiprocessing)."""
        scheduler = QuantumInspiredScheduler(**params)
        return scheduler.schedule_tasks(tasks)
    
    @staticmethod
    def _run_quantum_scheduler_with_monitoring(tasks: List[CompilationTask], params: Dict[str, Any]) -> SchedulingState:
        """Static method to run quantum scheduler with performance monitoring."""
        import time
        start_time = time.time()
        
        try:
            scheduler = QuantumInspiredScheduler(**params)
            result = scheduler.schedule_tasks(tasks)
            
            # Add performance metadata
            optimization_time = time.time() - start_time
            result.optimization_metadata = {
                "optimization_time": optimization_time,
                "parameters_used": params,
                "convergence_achieved": True
            }
            
            return result
            
        except Exception as e:
            # Create fallback result with error metadata
            logger.error(f"Quantum scheduler failed: {e}")
            
            # Simple sequential fallback
            fallback_result = SchedulingState(
                tasks=tasks,
                schedule={i: [task.id] for i, task in enumerate(tasks)},
                makespan=sum(task.estimated_duration for task in tasks),
                resource_utilization=0.1
            )
            
            fallback_result.optimization_metadata = {
                "optimization_time": time.time() - start_time,
                "parameters_used": params,
                "convergence_achieved": False,
                "fallback_used": True,
                "error": str(e)
            }
            
            return fallback_result
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics with performance insights."""
        cache_stats = self.cache.get_stats()
        
        # Advanced performance analytics
        performance_analytics = self._calculate_performance_analytics()
        
        return {
            "cache_stats": cache_stats,
            "optimization_level": self.optimization_level.value,
            "max_workers": self.max_workers,
            "performance_profiles": len(self.performance_history),
            "average_task_count": (
                sum(p.task_count for p in self.performance_history) / len(self.performance_history)
                if self.performance_history else 0
            ),
            "average_complexity": (
                sum(p.complexity_score for p in self.performance_history) / len(self.performance_history)
                if self.performance_history else 0
            ),
            "performance_analytics": performance_analytics,
            "auto_scaling_recommendations": self._get_auto_scaling_recommendations(),
            "resource_efficiency": self._calculate_resource_efficiency()
        }
    
    def _calculate_optimal_process_count(self, tasks: List[CompilationTask]) -> int:
        """Calculate optimal number of processes based on workload analysis."""
        # Base calculation on task complexity and system resources
        complexity_factor = self._calculate_workload_complexity(tasks)
        
        # Start with available CPU cores
        base_processes = min(self.max_workers, multiprocessing.cpu_count())
        
        # Scale based on task complexity
        if complexity_factor > 100:  # High complexity
            optimal = min(base_processes, 8)
        elif complexity_factor > 50:  # Medium complexity
            optimal = min(base_processes, 6) 
        else:  # Low complexity
            optimal = min(base_processes, 4)
        
        # Adjust for task count
        if len(tasks) < 10:
            optimal = min(optimal, 2)  # Don't over-parallelize small problems
        elif len(tasks) > 100:
            optimal = max(optimal, 6)  # Ensure sufficient parallelism for large problems
        
        return max(1, optimal)
    
    def _calculate_workload_complexity(self, tasks: List[CompilationTask]) -> float:
        """Calculate workload complexity score for optimization planning."""
        if not tasks:
            return 0
        
        # Factors contributing to complexity
        task_count_factor = len(tasks) * 2
        dependency_factor = sum(len(task.dependencies) for task in tasks) * 3
        duration_variance = self._calculate_duration_variance(tasks) * 10
        resource_diversity = len(set(
            tuple(task.resource_requirements.items()) for task in tasks
        )) * 2
        
        # Photonic-specific complexity factors
        thermal_complexity = sum(
            getattr(task, 'thermal_load', 0) for task in tasks
        ) * 0.1
        
        phase_complexity = sum(
            getattr(task, 'phase_shifts_required', 0) for task in tasks
        ) * 0.05
        
        total_complexity = (
            task_count_factor + dependency_factor + duration_variance +
            resource_diversity + thermal_complexity + phase_complexity
        )
        
        return total_complexity
    
    def _create_adaptive_params(self, base_params: Dict[str, Any], process_index: int, 
                              total_processes: int, tasks: List[CompilationTask]) -> Dict[str, Any]:
        """Create adaptive parameters for each optimization process."""
        params = base_params.copy()
        
        # Diversify parameters across processes
        diversity_factor = process_index / max(1, total_processes - 1)
        
        # Adaptive population size
        complexity = self._calculate_workload_complexity(tasks)
        if complexity > 100:
            params["population_size"] = int(base_params["population_size"] * (1 + diversity_factor * 0.5))
        else:
            params["population_size"] = max(10, int(base_params["population_size"] * (0.7 + diversity_factor * 0.6)))
        
        # Adaptive temperature and cooling
        params["initial_temperature"] = base_params["initial_temperature"] * (0.8 + diversity_factor * 0.4)
        params["cooling_rate"] = base_params["cooling_rate"] + (diversity_factor - 0.5) * 0.05
        
        # Adaptive iterations based on task complexity
        if complexity > 150:
            params["max_iterations"] = int(base_params["max_iterations"] * 1.2)
        elif complexity < 50:
            params["max_iterations"] = int(base_params["max_iterations"] * 0.8)
        
        # Mutation rate adaptation
        params["mutation_rate"] = max(0.01, min(0.3, 
            base_params["mutation_rate"] + (diversity_factor - 0.5) * 0.1
        ))
        
        return params
    
    def _calculate_adaptive_timeout(self, tasks: List[CompilationTask]) -> float:
        """Calculate adaptive timeout based on task complexity."""
        base_timeout = 1800.0  # 30 minutes base
        
        # Scale based on task count and complexity
        task_factor = len(tasks) / 50.0  # Scale factor per 50 tasks
        complexity = self._calculate_workload_complexity(tasks)
        complexity_factor = complexity / 100.0  # Scale factor per 100 complexity points
        
        # Thermal optimization may need more time
        has_thermal_tasks = any(
            getattr(task, 'thermal_load', 0) > 0 for task in tasks
        )
        thermal_factor = 1.3 if has_thermal_tasks else 1.0
        
        adaptive_timeout = base_timeout * (1 + task_factor) * (1 + complexity_factor) * thermal_factor
        
        # Reasonable bounds
        return max(300, min(3600, adaptive_timeout))  # 5 minutes to 1 hour
    
    def _track_distributed_performance(self, successful_processes: int, 
                                     total_processes: int, best_result: SchedulingState):
        """Track performance metrics from distributed scheduling."""
        success_rate = successful_processes / total_processes
        
        # Store performance metadata
        if not hasattr(self, 'distributed_performance'):
            self.distributed_performance = []
        
        perf_record = {
            "timestamp": time.time(),
            "successful_processes": successful_processes,
            "total_processes": total_processes,
            "success_rate": success_rate,
            "best_makespan": best_result.makespan,
            "best_utilization": best_result.resource_utilization
        }
        
        if hasattr(best_result, 'optimization_metadata'):
            perf_record.update(best_result.optimization_metadata)
        
        self.distributed_performance.append(perf_record)
        
        # Keep only recent records
        if len(self.distributed_performance) > 100:
            self.distributed_performance = self.distributed_performance[-50:]
        
        logger.info(f"Distributed scheduling: {successful_processes}/{total_processes} processes succeeded, "
                   f"best makespan: {best_result.makespan:.2f}s")
    
    def _calculate_performance_analytics(self) -> Dict[str, Any]:
        """Calculate advanced performance analytics."""
        if not hasattr(self, 'distributed_performance') or not self.distributed_performance:
            return {"status": "no_distributed_data"}
        
        recent_records = self.distributed_performance[-10:]  # Last 10 runs
        
        analytics = {
            "avg_success_rate": sum(r["success_rate"] for r in recent_records) / len(recent_records),
            "avg_makespan": sum(r["best_makespan"] for r in recent_records) / len(recent_records),
            "avg_utilization": sum(r["best_utilization"] for r in recent_records) / len(recent_records),
            "convergence_rate": sum(
                1 for r in recent_records 
                if r.get("convergence_achieved", False)
            ) / len(recent_records),
            "fallback_rate": sum(
                1 for r in recent_records 
                if r.get("fallback_used", False)
            ) / len(recent_records)
        }
        
        # Performance trend analysis
        if len(recent_records) >= 5:
            first_half = recent_records[:len(recent_records)//2]
            second_half = recent_records[len(recent_records)//2:]
            
            first_avg = sum(r["best_makespan"] for r in first_half) / len(first_half)
            second_avg = sum(r["best_makespan"] for r in second_half) / len(second_half)
            
            analytics["performance_trend"] = (first_avg - second_avg) / first_avg  # Positive = improving
        
        return analytics
    
    def _get_auto_scaling_recommendations(self) -> Dict[str, Any]:
        """Generate auto-scaling recommendations based on performance data."""
        recommendations = {"status": "analysis_needed"}
        
        if not hasattr(self, 'distributed_performance') or len(self.distributed_performance) < 5:
            recommendations["message"] = "Insufficient data for recommendations"
            return recommendations
        
        recent_data = self.distributed_performance[-10:]
        avg_success_rate = sum(r["success_rate"] for r in recent_data) / len(recent_data)
        avg_utilization = sum(r["best_utilization"] for r in recent_data) / len(recent_data)
        
        # Generate recommendations
        recs = []
        
        if avg_success_rate < 0.8:
            recs.append("Consider reducing max_workers due to low process success rate")
            recommendations["suggested_max_workers"] = max(1, self.max_workers - 2)
        elif avg_success_rate > 0.95 and avg_utilization > 0.8:
            recs.append("Consider increasing max_workers for better performance")
            recommendations["suggested_max_workers"] = min(multiprocessing.cpu_count(), self.max_workers + 2)
        
        if avg_utilization < 0.5:
            recs.append("Low resource utilization - consider optimizing task scheduling")
        
        # Cache performance recommendations
        cache_stats = self.cache.get_stats()
        if cache_stats.get("hit_rate", 0) < 0.3:
            recs.append("Low cache hit rate - consider increasing cache size or improving task similarity")
        
        recommendations["recommendations"] = recs
        recommendations["confidence"] = min(1.0, len(recent_data) / 20.0)  # Higher confidence with more data
        
        return recommendations
    
    def _calculate_resource_efficiency(self) -> Dict[str, float]:
        """Calculate resource efficiency metrics."""
        if not self.performance_history:
            return {"status": "no_data"}
        
        recent_profiles = self.performance_history[-10:]
        
        # CPU efficiency (tasks per CPU core per second)
        total_cpu_time = sum(p.estimated_runtime for p in recent_profiles)
        total_tasks = sum(p.task_count for p in recent_profiles)
        cpu_efficiency = total_tasks / (total_cpu_time * self.max_workers) if total_cpu_time > 0 else 0
        
        # Memory efficiency (tasks per GB of memory usage)
        avg_memory = sum(p.memory_requirement for p in recent_profiles) / len(recent_profiles)
        memory_efficiency = (total_tasks / len(recent_profiles)) / (avg_memory / 1024) if avg_memory > 0 else 0
        
        # Parallelism efficiency
        avg_parallelism = sum(p.parallelism_factor for p in recent_profiles) / len(recent_profiles)
        parallelism_efficiency = avg_parallelism / self.max_workers
        
        return {
            "cpu_efficiency": cpu_efficiency,
            "memory_efficiency": memory_efficiency,
            "parallelism_efficiency": parallelism_efficiency,
            "overall_efficiency": (cpu_efficiency + memory_efficiency + parallelism_efficiency) / 3
        }