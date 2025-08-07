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
        """Process-based distributed scheduling for large problems."""
        # For very large problems, partition tasks and solve subproblems
        if len(tasks) > 200:
            return self._schedule_hierarchical(tasks, params)
        
        # Run multiple independent optimization processes
        num_processes = min(self.max_workers, 8)
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = []
            
            for i in range(num_processes):
                # Create diverse parameter variations
                run_params = params.copy()
                run_params["population_size"] = params["population_size"] + i * 20
                run_params["initial_temperature"] = params["initial_temperature"] * (1 + i * 0.2)
                
                future = executor.submit(self._run_quantum_scheduler, tasks, run_params)
                futures.append(future)
            
            # Collect results
            results = []
            for future in as_completed(futures):
                try:
                    result = future.result(timeout=1800)  # 30 minute timeout
                    results.append(result)
                except Exception as e:
                    logger.warning(f"Distributed scheduling process failed: {e}")
            
            if not results:
                raise RuntimeError("All distributed scheduling processes failed")
            
            # Return best result
            return min(results, key=lambda s: s.total_energy)
    
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
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        cache_stats = self.cache.get_stats()
        
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
            )
        }