"""
Comprehensive unit tests for quantum-inspired task scheduling system.

Tests all components of the quantum scheduling system including basic functionality,
error handling, performance optimization, and edge cases.
"""

import pytest
import logging
import time
from typing import List
import tempfile
from pathlib import Path

from photon_mlir.quantum_scheduler import (
    QuantumTaskPlanner, QuantumInspiredScheduler, CompilationTask, 
    TaskType, QuantumState, SchedulingState
)
from photon_mlir.quantum_optimization import (
    ParallelQuantumScheduler, OptimizationLevel, CacheStrategy
)
from photon_mlir.quantum_validation import (
    QuantumValidator, ValidationLevel, QuantumMonitor
)


class TestCompilationTask:
    """Test CompilationTask class."""
    
    def test_basic_task_creation(self):
        """Test basic task creation."""
        task = CompilationTask(
            id="test_task_001",
            task_type=TaskType.GRAPH_LOWERING,
            estimated_duration=2.5,
            priority=1.5
        )
        
        assert task.id == "test_task_001"
        assert task.task_type == TaskType.GRAPH_LOWERING
        assert task.estimated_duration == 2.5
        assert task.priority == 1.5
        assert task.quantum_state == QuantumState.SUPERPOSITION
        assert len(task.dependencies) == 0
        assert task.resource_requirements["cpu"] == 1.0
        assert task.resource_requirements["memory"] == 512.0
    
    def test_task_with_dependencies(self):
        """Test task creation with dependencies."""
        task = CompilationTask(
            id="dependent_task",
            task_type=TaskType.PHOTONIC_OPTIMIZATION,
            dependencies={"task1", "task2"},
            resource_requirements={"cpu": 4.0, "memory": 2048.0}
        )
        
        assert len(task.dependencies) == 2
        assert "task1" in task.dependencies
        assert "task2" in task.dependencies
        assert task.resource_requirements["cpu"] == 4.0
    
    def test_task_post_init(self):
        """Test task post-initialization."""
        task = CompilationTask(
            id="test_task",
            task_type=TaskType.MESH_MAPPING
        )
        
        # Should have default resource requirements
        assert "cpu" in task.resource_requirements
        assert "memory" in task.resource_requirements
        assert "gpu" in task.resource_requirements


class TestQuantumTaskPlanner:
    """Test QuantumTaskPlanner class."""
    
    def test_planner_initialization(self):
        """Test planner initialization."""
        planner = QuantumTaskPlanner()
        
        assert planner.scheduler is not None
        assert planner.task_counter == 0
    
    def test_create_compilation_plan(self):
        """Test compilation plan creation."""
        planner = QuantumTaskPlanner()
        config = {
            "model_type": "transformer",
            "layers": 6,
            "hidden_size": 512
        }
        
        tasks = planner.create_compilation_plan(config)
        
        assert len(tasks) == 7  # Expected number of tasks
        assert all(isinstance(task, CompilationTask) for task in tasks)
        
        # Check task types are present
        task_types = {task.task_type for task in tasks}
        expected_types = {
            TaskType.GRAPH_LOWERING,
            TaskType.PHOTONIC_OPTIMIZATION,
            TaskType.MESH_MAPPING,
            TaskType.PHASE_OPTIMIZATION,
            TaskType.POWER_BALANCING,
            TaskType.THERMAL_COMPENSATION,
            TaskType.CODE_GENERATION
        }
        assert task_types == expected_types
        
        # Check dependencies are properly set
        graph_task = next(t for t in tasks if t.task_type == TaskType.GRAPH_LOWERING)
        assert len(graph_task.dependencies) == 0
        
        photonic_task = next(t for t in tasks if t.task_type == TaskType.PHOTONIC_OPTIMIZATION)
        assert graph_task.id in photonic_task.dependencies
    
    def test_optimize_schedule_basic(self):
        """Test basic schedule optimization."""
        planner = QuantumTaskPlanner()
        
        # Create simple tasks
        tasks = [
            CompilationTask(id="task1", task_type=TaskType.GRAPH_LOWERING, estimated_duration=1.0),
            CompilationTask(id="task2", task_type=TaskType.PHOTONIC_OPTIMIZATION, 
                          dependencies={"task1"}, estimated_duration=2.0),
            CompilationTask(id="task3", task_type=TaskType.CODE_GENERATION, 
                          dependencies={"task2"}, estimated_duration=1.5)
        ]
        
        result = planner.optimize_schedule(tasks)
        
        assert isinstance(result, SchedulingState)
        assert result.makespan > 0
        assert 0 <= result.resource_utilization <= 1
        assert len(result.schedule) > 0


class TestQuantumInspiredScheduler:
    """Test QuantumInspiredScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test scheduler initialization."""
        scheduler = QuantumInspiredScheduler(
            population_size=20,
            max_iterations=100,
            enable_validation=True
        )
        
        assert scheduler.population_size == 20
        assert scheduler.max_iterations == 100
        assert scheduler.enable_validation == True
        assert len(scheduler.population) == 0
    
    def test_empty_task_list(self):
        """Test scheduling with empty task list."""
        scheduler = QuantumInspiredScheduler()
        
        with pytest.raises(ValueError, match="No tasks provided"):
            scheduler.schedule_tasks([])
    
    def test_single_task_scheduling(self):
        """Test scheduling with single task."""
        scheduler = QuantumInspiredScheduler(max_iterations=10)
        
        task = CompilationTask(
            id="single_task",
            task_type=TaskType.GRAPH_LOWERING,
            estimated_duration=1.0
        )
        
        result = scheduler.schedule_tasks([task])
        
        assert result.makespan >= 1.0
        assert len(result.schedule) >= 1
        assert "single_task" in [tid for task_ids in result.schedule.values() for tid in task_ids]
    
    def test_dependency_handling(self):
        """Test proper dependency handling."""
        scheduler = QuantumInspiredScheduler(max_iterations=50)
        
        tasks = [
            CompilationTask(id="task_a", task_type=TaskType.GRAPH_LOWERING, estimated_duration=1.0),
            CompilationTask(id="task_b", task_type=TaskType.PHOTONIC_OPTIMIZATION, 
                          dependencies={"task_a"}, estimated_duration=2.0),
            CompilationTask(id="task_c", task_type=TaskType.CODE_GENERATION, 
                          dependencies={"task_b"}, estimated_duration=1.0)
        ]
        
        result = scheduler.schedule_tasks(tasks)
        
        # Verify dependency order is respected
        task_slots = {}
        for slot, task_ids in result.schedule.items():
            for task_id in task_ids:
                task_slots[task_id] = slot
        
        assert task_slots["task_a"] < task_slots["task_b"]
        assert task_slots["task_b"] < task_slots["task_c"]
    
    def test_parallel_task_scheduling(self):
        """Test scheduling of independent parallel tasks."""
        scheduler = QuantumInspiredScheduler(max_iterations=50)
        
        # Create tasks that can run in parallel
        tasks = [
            CompilationTask(id="root", task_type=TaskType.GRAPH_LOWERING, estimated_duration=1.0),
            CompilationTask(id="parallel_1", task_type=TaskType.MESH_MAPPING, 
                          dependencies={"root"}, estimated_duration=2.0),
            CompilationTask(id="parallel_2", task_type=TaskType.PHASE_OPTIMIZATION, 
                          dependencies={"root"}, estimated_duration=2.0),
            CompilationTask(id="parallel_3", task_type=TaskType.POWER_BALANCING, 
                          dependencies={"root"}, estimated_duration=2.0),
        ]
        
        result = scheduler.schedule_tasks(tasks)
        
        # Should be able to achieve better utilization than sequential
        sequential_time = sum(task.estimated_duration for task in tasks)
        assert result.makespan < sequential_time
        assert result.resource_utilization > 0.5
    
    def test_error_recovery(self):
        """Test error recovery mechanisms."""
        # Create scheduler with aggressive parameters that might cause issues
        scheduler = QuantumInspiredScheduler(
            population_size=5,
            max_iterations=10,
            error_recovery_attempts=2
        )
        
        # Create complex task set
        tasks = []
        for i in range(10):
            task = CompilationTask(
                id=f"task_{i}",
                task_type=TaskType.PHOTONIC_OPTIMIZATION,
                estimated_duration=1.0 + i * 0.1
            )
            if i > 0:
                task.dependencies.add(f"task_{i-1}")
            tasks.append(task)
        
        # Should complete without raising exceptions
        result = scheduler.schedule_tasks(tasks)
        assert result is not None
        assert result.makespan > 0


class TestQuantumValidator:
    """Test QuantumValidator class."""
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = QuantumValidator(ValidationLevel.STRICT)
        
        assert validator.validation_level == ValidationLevel.STRICT
        assert validator.max_task_count == 10000
        assert len(validator.blacklisted_patterns) > 0
    
    def test_valid_task_validation(self):
        """Test validation of valid tasks."""
        validator = QuantumValidator()
        
        tasks = [
            CompilationTask(id="valid_task_1", task_type=TaskType.GRAPH_LOWERING),
            CompilationTask(id="valid_task_2", task_type=TaskType.PHOTONIC_OPTIMIZATION,
                          dependencies={"valid_task_1"})
        ]
        
        result = validator.validate_tasks(tasks)
        
        assert result.is_valid == True
        assert len(result.errors) == 0
    
    def test_invalid_task_validation(self):
        """Test validation of invalid tasks."""
        validator = QuantumValidator()
        
        # Create invalid tasks
        tasks = [
            CompilationTask(id="task1", task_type=TaskType.GRAPH_LOWERING),
            CompilationTask(id="task1", task_type=TaskType.PHOTONIC_OPTIMIZATION),  # Duplicate ID
        ]
        
        result = validator.validate_tasks(tasks)
        
        assert result.is_valid == False
        assert len(result.errors) > 0
        assert any("Duplicate task ID" in error for error in result.errors)
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection."""
        validator = QuantumValidator()
        
        # Create circular dependency
        tasks = [
            CompilationTask(id="task_a", task_type=TaskType.GRAPH_LOWERING, dependencies={"task_b"}),
            CompilationTask(id="task_b", task_type=TaskType.PHOTONIC_OPTIMIZATION, dependencies={"task_a"})
        ]
        
        result = validator.validate_tasks(tasks)
        
        assert result.is_valid == False
        assert any("Circular dependencies" in error for error in result.errors)
    
    def test_resource_validation(self):
        """Test resource requirement validation."""
        validator = QuantumValidator()
        
        # Create task with excessive resource requirements
        task = CompilationTask(
            id="resource_hog",
            task_type=TaskType.PHOTONIC_OPTIMIZATION,
            resource_requirements={"cpu": 1000.0, "memory": 100000.0}
        )
        
        result = validator.validate_tasks([task])
        
        assert result.is_valid == False
        assert any("Excessive" in error for error in result.errors)
    
    def test_schedule_validation(self):
        """Test schedule validation."""
        validator = QuantumValidator()
        
        tasks = [
            CompilationTask(id="task1", task_type=TaskType.GRAPH_LOWERING, estimated_duration=1.0),
            CompilationTask(id="task2", task_type=TaskType.PHOTONIC_OPTIMIZATION, 
                          dependencies={"task1"}, estimated_duration=2.0)
        ]
        
        # Create valid schedule
        valid_schedule = SchedulingState(
            tasks=tasks,
            schedule={0: ["task1"], 2: ["task2"]}
        )
        valid_schedule.makespan = 4.0
        valid_schedule.resource_utilization = 0.75
        
        result = validator.validate_schedule(valid_schedule)
        assert result.is_valid == True
        
        # Create invalid schedule (dependency violation)
        invalid_schedule = SchedulingState(
            tasks=tasks,
            schedule={0: ["task2"], 1: ["task1"]}  # task2 starts before task1
        )
        invalid_schedule.makespan = 3.0
        invalid_schedule.resource_utilization = 1.0
        
        result = validator.validate_schedule(invalid_schedule)
        assert result.is_valid == False


class TestParallelQuantumScheduler:
    """Test ParallelQuantumScheduler class."""
    
    def test_scheduler_initialization(self):
        """Test parallel scheduler initialization."""
        scheduler = ParallelQuantumScheduler(
            optimization_level=OptimizationLevel.FAST,
            cache_strategy=CacheStrategy.MEMORY_ONLY
        )
        
        assert scheduler.optimization_level == OptimizationLevel.FAST
        assert scheduler.cache is not None
        assert scheduler.max_workers > 0
    
    def test_small_problem_optimization(self):
        """Test optimization for small problems."""
        scheduler = ParallelQuantumScheduler(optimization_level=OptimizationLevel.FAST)
        
        # Create small problem (< 10 tasks)
        tasks = [
            CompilationTask(id=f"task_{i}", task_type=TaskType.GRAPH_LOWERING, estimated_duration=1.0)
            for i in range(5)
        ]
        
        start_time = time.time()
        result = scheduler.schedule_tasks_optimized(tasks)
        end_time = time.time()
        
        assert isinstance(result, SchedulingState)
        assert result.makespan > 0
        assert end_time - start_time < 10  # Should be fast for small problems
    
    def test_caching_functionality(self):
        """Test caching functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_dir = Path(temp_dir) / "test_cache"
            
            scheduler = ParallelQuantumScheduler(cache_strategy=CacheStrategy.HYBRID)
            scheduler.cache.cache_dir = cache_dir
            
            tasks = [
                CompilationTask(id="cached_task_1", task_type=TaskType.GRAPH_LOWERING),
                CompilationTask(id="cached_task_2", task_type=TaskType.PHOTONIC_OPTIMIZATION,
                              dependencies={"cached_task_1"})
            ]
            
            # First run - should compute and cache
            start_time = time.time()
            result1 = scheduler.schedule_tasks_optimized(tasks)
            first_time = time.time() - start_time
            
            # Second run - should use cache
            start_time = time.time()
            result2 = scheduler.schedule_tasks_optimized(tasks)
            second_time = time.time() - start_time
            
            # Results should be identical
            assert result1.makespan == result2.makespan
            assert result1.resource_utilization == result2.resource_utilization
            
            # Second run should be much faster (cache hit)
            assert second_time < first_time * 0.5
            
            # Check cache stats
            stats = scheduler.cache.get_stats()
            assert stats["hits"] >= 1


class TestQuantumMonitor:
    """Test QuantumMonitor class."""
    
    def test_monitor_initialization(self):
        """Test monitor initialization."""
        monitor = QuantumMonitor()
        
        assert monitor.is_monitoring == False
        assert len(monitor.metrics_history) == 0
        assert len(monitor.performance_alerts) == 0
    
    def test_metrics_recording(self):
        """Test metrics recording."""
        from photon_mlir.quantum_validation import QuantumMetrics
        
        monitor = QuantumMonitor()
        
        metrics = QuantumMetrics(
            convergence_rate=0.95,
            quantum_coherence=0.88,
            entanglement_efficiency=0.92,
            superposition_utilization=0.75,
            annealing_temperature=1.0,
            state_transitions=100
        )
        
        monitor.record_metrics(metrics)
        
        assert len(monitor.metrics_history) == 1
        assert monitor.metrics_history[0].convergence_rate == 0.95
    
    def test_performance_summary(self):
        """Test performance summary generation."""
        from photon_mlir.quantum_validation import QuantumMetrics
        
        monitor = QuantumMonitor()
        
        # Record some metrics
        for i in range(5):
            metrics = QuantumMetrics(
                convergence_rate=0.9 + i * 0.01,
                quantum_coherence=0.8 + i * 0.02,
                entanglement_efficiency=0.85 + i * 0.01,
                superposition_utilization=0.7 + i * 0.02,
                annealing_temperature=1.0 - i * 0.1,
                state_transitions=100 + i * 10
            )
            monitor.record_metrics(metrics)
        
        summary = monitor.get_performance_summary()
        
        assert summary["total_measurements"] == 5
        assert "recent_convergence_rate" in summary
        assert "average_coherence" in summary
        assert summary["latest_metrics"] is not None


class TestIntegration:
    """Integration tests for the complete quantum scheduling system."""
    
    def test_end_to_end_scheduling(self):
        """Test complete end-to-end scheduling workflow."""
        # Create planner
        planner = QuantumTaskPlanner()
        
        # Create model configuration
        config = {
            "model_type": "cnn",
            "layers": 5,
            "channels": 64,
            "precision": "int8"
        }
        
        # Generate tasks
        tasks = planner.create_compilation_plan(config)
        
        # Validate tasks
        validator = QuantumValidator(ValidationLevel.STRICT)
        validation_result = validator.validate_tasks(tasks)
        assert validation_result.is_valid
        
        # Optimize schedule
        parallel_scheduler = ParallelQuantumScheduler(
            optimization_level=OptimizationLevel.BALANCED
        )
        
        result = parallel_scheduler.schedule_tasks_optimized(tasks)
        
        # Validate result
        schedule_validation = validator.validate_schedule(result)
        assert schedule_validation.is_valid
        
        # Check performance
        assert result.makespan > 0
        assert 0 < result.resource_utilization <= 1
        
        # Get optimization stats
        stats = parallel_scheduler.get_optimization_stats()
        assert "cache_stats" in stats
        assert "optimization_level" in stats
    
    def test_large_scale_scheduling(self):
        """Test scheduling with larger problem sizes."""
        scheduler = ParallelQuantumScheduler(
            optimization_level=OptimizationLevel.FAST  # Use fast mode for testing
        )
        
        # Create larger problem (50 tasks with complex dependencies)
        tasks = []
        for i in range(50):
            task_type = [TaskType.GRAPH_LOWERING, TaskType.PHOTONIC_OPTIMIZATION, 
                        TaskType.MESH_MAPPING, TaskType.PHASE_OPTIMIZATION][i % 4]
            
            task = CompilationTask(
                id=f"large_task_{i:03d}",
                task_type=task_type,
                estimated_duration=1.0 + (i % 5) * 0.5,
                priority=1.0 + (i % 3) * 0.5
            )
            
            # Add some dependencies
            if i > 0 and i % 4 != 0:
                task.dependencies.add(f"large_task_{i-1:03d}")
            if i > 5 and i % 7 == 0:
                task.dependencies.add(f"large_task_{i-5:03d}")
            
            tasks.append(task)
        
        start_time = time.time()
        result = scheduler.schedule_tasks_optimized(tasks)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert end_time - start_time < 30  # 30 seconds max for 50 tasks
        
        # Should achieve decent parallelization
        sequential_time = sum(task.estimated_duration for task in tasks)
        assert result.makespan < sequential_time * 0.8  # At least 20% improvement
        
        assert result.resource_utilization > 0.3  # Reasonable utilization
    
    def test_error_handling_integration(self):
        """Test error handling across integrated system."""
        planner = QuantumTaskPlanner()
        
        # Create tasks with potential issues
        tasks = [
            CompilationTask(id="normal_task", task_type=TaskType.GRAPH_LOWERING),
            CompilationTask(
                id="problematic_task",
                task_type=TaskType.PHOTONIC_OPTIMIZATION,
                dependencies={"nonexistent_task"},  # Bad dependency
                estimated_duration=-1.0  # Invalid duration
            )
        ]
        
        # Should catch validation errors
        validator = QuantumValidator()
        validation_result = validator.validate_tasks(tasks)
        assert not validation_result.is_valid
        
        # Scheduler should handle validation gracefully
        scheduler = ParallelQuantumScheduler(cache_strategy=CacheStrategy.NONE)
        
        with pytest.raises((ValueError, RuntimeError)):
            scheduler.schedule_tasks_optimized(tasks)


# Performance and stress tests
class TestPerformance:
    """Performance and stress tests."""
    
    @pytest.mark.slow
    def test_scheduler_performance_scaling(self):
        """Test scheduler performance with increasing problem size."""
        sizes = [10, 25, 50, 100]
        times = []
        
        for size in sizes:
            scheduler = ParallelQuantumScheduler(
                optimization_level=OptimizationLevel.FAST
            )
            
            # Generate tasks
            tasks = [
                CompilationTask(
                    id=f"perf_task_{i}",
                    task_type=list(TaskType)[i % len(TaskType)],
                    estimated_duration=1.0
                )
                for i in range(size)
            ]
            
            # Add some dependencies for realism
            for i in range(1, size):
                if i % 5 == 0:
                    tasks[i].dependencies.add(tasks[i-1].id)
            
            # Time the scheduling
            start_time = time.time()
            result = scheduler.schedule_tasks_optimized(tasks)
            elapsed = time.time() - start_time
            
            times.append(elapsed)
            
            # Verify result quality
            assert result.makespan > 0
            assert 0 < result.resource_utilization <= 1
        
        # Performance should not degrade exponentially
        # (allowing for some variation in test environment)
        for i in range(1, len(times)):
            scaling_factor = (sizes[i] / sizes[i-1])
            time_ratio = times[i] / times[i-1]
            
            # Time should not grow worse than O(n^2)
            assert time_ratio < scaling_factor ** 2 * 2  # With 2x buffer
    
    @pytest.mark.slow  
    def test_cache_performance(self):
        """Test cache performance with repeated scheduling."""
        scheduler = ParallelQuantumScheduler(
            cache_strategy=CacheStrategy.MEMORY_ONLY
        )
        
        # Create consistent task set
        tasks = [
            CompilationTask(
                id=f"cache_task_{i}",
                task_type=TaskType.PHOTONIC_OPTIMIZATION,
                estimated_duration=2.0
            )
            for i in range(20)
        ]
        
        # First run - populate cache
        time1 = time.time()
        result1 = scheduler.schedule_tasks_optimized(tasks)
        elapsed1 = time.time() - time1
        
        # Subsequent runs - should hit cache
        cache_times = []
        for _ in range(5):
            time_start = time.time()
            result = scheduler.schedule_tasks_optimized(tasks)
            cache_times.append(time.time() - time_start)
            
            # Results should be identical
            assert result.makespan == result1.makespan
        
        # Cache hits should be much faster
        avg_cache_time = sum(cache_times) / len(cache_times)
        assert avg_cache_time < elapsed1 * 0.1  # 10x speedup minimum
        
        # Verify cache stats
        stats = scheduler.get_optimization_stats()
        assert stats["cache_stats"]["hits"] >= 5
        assert stats["cache_stats"]["hit_rate"] > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])