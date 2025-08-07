"""
End-to-end integration tests for quantum-inspired task planning system.

Tests the complete workflow from high-level model compilation through
quantum-inspired task scheduling to final execution plan generation.
"""

import pytest
import logging
import time
import tempfile
from pathlib import Path
from typing import Dict, Any

import photon_mlir as pm


class TestQuantumIntegrationWorkflow:
    """Test complete quantum-inspired scheduling integration."""
    
    def test_complete_workflow_basic_model(self):
        """Test complete workflow with basic model."""
        # Step 1: Create model configuration
        config = {
            "model_type": "linear",
            "layers": 3,
            "hidden_size": 256,
            "precision": "int8",
            "target_device": "lightmatter_envise"
        }
        
        # Step 2: Generate quantum-optimized compilation plan
        planner = pm.QuantumTaskPlanner()
        tasks = planner.create_compilation_plan(config)
        
        assert len(tasks) > 0
        assert all(isinstance(task, pm.CompilationTask) for task in tasks)
        
        # Step 3: Validate tasks
        validator = pm.QuantumValidator(pm.ValidationLevel.STRICT)
        validation_result = validator.validate_tasks(tasks)
        
        if not validation_result.is_valid:
            pytest.fail(f"Task validation failed: {validation_result.errors}")
        
        # Step 4: Optimize schedule with quantum algorithms
        scheduler = pm.ParallelQuantumScheduler(
            optimization_level=pm.OptimizationLevel.BALANCED,
            cache_strategy=pm.CacheStrategy.HYBRID
        )
        
        optimized_schedule = scheduler.schedule_tasks_optimized(tasks)
        
        assert optimized_schedule.makespan > 0
        assert 0 < optimized_schedule.resource_utilization <= 1
        
        # Step 5: Validate final schedule
        schedule_validation = validator.validate_schedule(optimized_schedule)
        
        if not schedule_validation.is_valid:
            pytest.fail(f"Schedule validation failed: {schedule_validation.errors}")
        
        # Step 6: Verify optimization quality
        sequential_time = sum(task.estimated_duration for task in tasks)
        speedup = sequential_time / optimized_schedule.makespan
        
        assert speedup >= 1.0  # Should at least not be slower
        logging.info(f"Achieved {speedup:.2f}x speedup with quantum scheduling")
    
    def test_workflow_with_complex_model(self):
        """Test workflow with complex transformer model."""
        config = {
            "model_type": "transformer",
            "layers": 12,
            "hidden_size": 768,
            "attention_heads": 12,
            "sequence_length": 512,
            "precision": "int8",
            "enable_thermal_compensation": True,
            "photonic_array_size": (64, 64)
        }
        
        # Generate and validate tasks
        planner = pm.QuantumTaskPlanner()
        tasks = planner.create_compilation_plan(config)
        
        validator = pm.QuantumValidator()
        validation_result = validator.validate_tasks(tasks)
        assert validation_result.is_valid
        
        # Use quality optimization for complex models
        scheduler = pm.ParallelQuantumScheduler(
            optimization_level=pm.OptimizationLevel.QUALITY,
            cache_strategy=pm.CacheStrategy.HYBRID
        )
        
        start_time = time.time()
        result = scheduler.schedule_tasks_optimized(tasks)
        optimization_time = time.time() - start_time
        
        # Verify results
        assert result.makespan > 0
        assert result.resource_utilization > 0.3  # Should achieve decent parallelism
        
        # Should complete in reasonable time even for quality optimization
        assert optimization_time < 60  # 1 minute max
        
        # Get detailed stats
        stats = scheduler.get_optimization_stats()
        assert stats["optimization_level"] == "quality"
        
        logging.info(f"Complex model optimized in {optimization_time:.2f}s, "
                    f"achieved {result.resource_utilization:.1%} utilization")
    
    def test_workflow_with_caching(self):
        """Test workflow with caching enabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create scheduler with disk caching
            scheduler = pm.ParallelQuantumScheduler(
                cache_strategy=pm.CacheStrategy.HYBRID
            )
            scheduler.cache.cache_dir = Path(temp_dir)
            
            config = {
                "model_type": "cnn",
                "layers": 8,
                "channels": 128,
                "precision": "fp16"
            }
            
            planner = pm.QuantumTaskPlanner()
            tasks = planner.create_compilation_plan(config)
            
            # First optimization - should populate cache
            time1 = time.time()
            result1 = scheduler.schedule_tasks_optimized(tasks)
            elapsed1 = time.time() - time1
            
            # Second optimization - should hit cache
            time2 = time.time()
            result2 = scheduler.schedule_tasks_optimized(tasks)
            elapsed2 = time.time() - time2
            
            # Results should be identical
            assert result1.makespan == result2.makespan
            assert result1.resource_utilization == result2.resource_utilization
            
            # Second run should be much faster
            assert elapsed2 < elapsed1 * 0.2  # 5x speedup minimum
            
            # Verify cache was used
            cache_stats = scheduler.cache.get_stats()
            assert cache_stats["hits"] >= 1
            assert cache_stats["hit_rate"] > 0
            
            logging.info(f"Cache speedup: {elapsed1/elapsed2:.1f}x "
                        f"(first: {elapsed1:.3f}s, cached: {elapsed2:.3f}s)")
    
    def test_workflow_with_monitoring(self):
        """Test workflow with performance monitoring."""
        # Start monitoring
        monitor = pm.QuantumMonitor()
        monitor.start_monitoring(interval=0.1)
        
        try:
            config = {
                "model_type": "resnet",
                "layers": 18,
                "channels": 64
            }
            
            planner = pm.QuantumTaskPlanner()
            tasks = planner.create_compilation_plan(config)
            
            scheduler = pm.ParallelQuantumScheduler(
                optimization_level=pm.OptimizationLevel.BALANCED
            )
            
            # Record initial metrics
            from photon_mlir.quantum_validation import QuantumMetrics
            initial_metrics = QuantumMetrics(
                convergence_rate=0.0,
                quantum_coherence=1.0,
                entanglement_efficiency=0.5,
                superposition_utilization=1.0,
                annealing_temperature=100.0,
                state_transitions=0
            )
            monitor.record_metrics(initial_metrics)
            
            # Run optimization
            result = scheduler.schedule_tasks_optimized(tasks)
            
            # Record final metrics
            final_metrics = QuantumMetrics(
                convergence_rate=0.95,
                quantum_coherence=0.88,
                entanglement_efficiency=0.92,
                superposition_utilization=0.2,
                annealing_temperature=1.0,
                state_transitions=500
            )
            monitor.record_metrics(final_metrics)
            
            # Give monitor time to collect data
            time.sleep(0.2)
            
            # Verify monitoring data
            summary = monitor.get_performance_summary()
            assert summary["total_measurements"] >= 2
            assert summary["latest_metrics"] is not None
            
            logging.info(f"Monitoring captured {summary['total_measurements']} measurements")
            
        finally:
            monitor.stop_monitoring()
    
    def test_workflow_error_scenarios(self):
        """Test workflow with various error scenarios."""
        planner = pm.QuantumTaskPlanner()
        validator = pm.QuantumValidator()
        
        # Scenario 1: Invalid task configuration
        invalid_tasks = [
            pm.CompilationTask(
                id="invalid_task",
                task_type=pm.quantum_scheduler.TaskType.GRAPH_LOWERING,
                estimated_duration=-1.0  # Invalid duration
            )
        ]
        
        validation_result = validator.validate_tasks(invalid_tasks)
        assert not validation_result.is_valid
        assert len(validation_result.errors) > 0
        
        # Scenario 2: Circular dependencies
        circular_tasks = [
            pm.CompilationTask(
                id="task_a",
                task_type=pm.quantum_scheduler.TaskType.GRAPH_LOWERING,
                dependencies={"task_b"}
            ),
            pm.CompilationTask(
                id="task_b", 
                task_type=pm.quantum_scheduler.TaskType.PHOTONIC_OPTIMIZATION,
                dependencies={"task_a"}
            )
        ]
        
        validation_result = validator.validate_tasks(circular_tasks)
        assert not validation_result.is_valid
        assert any("Circular dependencies" in error for error in validation_result.errors)
        
        # Scenario 3: Scheduler should handle validation errors gracefully
        scheduler = pm.ParallelQuantumScheduler()
        
        with pytest.raises((ValueError, RuntimeError)):
            scheduler.schedule_tasks_optimized(invalid_tasks)
    
    def test_workflow_performance_optimization_levels(self):
        """Test workflow with different optimization levels."""
        config = {
            "model_type": "bert",
            "layers": 6,
            "hidden_size": 512,
            "attention_heads": 8
        }
        
        planner = pm.QuantumTaskPlanner()
        tasks = planner.create_compilation_plan(config)
        
        optimization_levels = [
            pm.OptimizationLevel.FAST,
            pm.OptimizationLevel.BALANCED,
            pm.OptimizationLevel.QUALITY
        ]
        
        results = {}
        
        for level in optimization_levels:
            scheduler = pm.ParallelQuantumScheduler(
                optimization_level=level,
                cache_strategy=pm.CacheStrategy.NONE  # Disable cache to ensure fresh runs
            )
            
            start_time = time.time()
            result = scheduler.schedule_tasks_optimized(tasks)
            elapsed = time.time() - start_time
            
            results[level] = {
                "makespan": result.makespan,
                "utilization": result.resource_utilization,
                "time": elapsed
            }
        
        # Verify optimization level trade-offs
        # Fast should be quickest
        assert results[pm.OptimizationLevel.FAST]["time"] <= results[pm.OptimizationLevel.BALANCED]["time"]
        assert results[pm.OptimizationLevel.BALANCED]["time"] <= results[pm.OptimizationLevel.QUALITY]["time"]
        
        # Quality should potentially achieve better makespan (though not guaranteed due to randomness)
        # At minimum, all should produce valid results
        for level, result in results.items():
            assert result["makespan"] > 0
            assert 0 < result["utilization"] <= 1
            
            logging.info(f"{level.value}: makespan={result['makespan']:.2f}s, "
                        f"utilization={result['utilization']:.1%}, "
                        f"time={result['time']:.3f}s")
    
    def test_workflow_large_scale_model(self):
        """Test workflow with large-scale model compilation."""
        # Large transformer model
        config = {
            "model_type": "gpt",
            "layers": 24,
            "hidden_size": 1024,
            "attention_heads": 16,
            "sequence_length": 2048,
            "precision": "fp16",
            "enable_thermal_compensation": True,
            "optimization_level": 3
        }
        
        planner = pm.QuantumTaskPlanner()
        tasks = planner.create_compilation_plan(config)
        
        # Should generate substantial number of tasks for large model
        assert len(tasks) >= 7  # At least the standard pipeline
        
        # Use extreme optimization for large models
        scheduler = pm.ParallelQuantumScheduler(
            optimization_level=pm.OptimizationLevel.EXTREME,
            max_workers=min(8, scheduler.max_workers)  # Limit for test stability
        )
        
        start_time = time.time()
        result = scheduler.schedule_tasks_optimized(tasks)
        optimization_time = time.time() - start_time
        
        # Large models should still complete in reasonable time
        assert optimization_time < 120  # 2 minutes max
        
        # Should achieve good parallelization for large models
        sequential_time = sum(task.estimated_duration for task in tasks)
        speedup = sequential_time / result.makespan
        assert speedup > 1.5  # At least 1.5x speedup
        
        logging.info(f"Large model: {len(tasks)} tasks, "
                    f"{speedup:.2f}x speedup, "
                    f"optimized in {optimization_time:.1f}s")
    
    def test_workflow_memory_management(self):
        """Test workflow memory management and resource cleanup."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        config = {
            "model_type": "densenet",
            "layers": 121,
            "growth_rate": 32
        }
        
        # Run multiple optimization cycles
        for iteration in range(3):
            planner = pm.QuantumTaskPlanner()
            tasks = planner.create_compilation_plan(config)
            
            scheduler = pm.ParallelQuantumScheduler(
                cache_strategy=pm.CacheStrategy.MEMORY_ONLY
            )
            
            result = scheduler.schedule_tasks_optimized(tasks)
            assert result.makespan > 0
            
            # Force cleanup
            del planner
            del scheduler
            del tasks
            del result
            gc.collect()
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory usage should not grow excessively
        assert memory_increase < 100  # Less than 100MB increase
        
        logging.info(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB "
                    f"(+{memory_increase:.1f}MB)")


class TestQuantumAdvancedFeatures:
    """Test advanced quantum-inspired features."""
    
    def test_superposition_state_management(self):
        """Test quantum superposition state management."""
        planner = pm.QuantumTaskPlanner()
        
        # Create tasks that will be in superposition
        tasks = planner.create_compilation_plan({"model_type": "simple", "layers": 3})
        
        # Initially, all tasks should be in superposition
        for task in tasks:
            assert task.quantum_state == pm.quantum_scheduler.QuantumState.SUPERPOSITION
        
        # After scheduling, states should be collapsed
        scheduler = pm.QuantumInspiredScheduler(max_iterations=10)
        result = scheduler.schedule_tasks(tasks)
        
        for task in tasks:
            assert task.quantum_state == pm.quantum_scheduler.QuantumState.COLLAPSED
        
        assert result.makespan > 0
    
    def test_entanglement_simulation(self):
        """Test quantum entanglement simulation for correlated tasks."""
        # Create entangled tasks manually
        task1 = pm.CompilationTask(
            id="entangled_task_1",
            task_type=pm.quantum_scheduler.TaskType.MESH_MAPPING,
            estimated_duration=3.0
        )
        task2 = pm.CompilationTask(
            id="entangled_task_2", 
            task_type=pm.quantum_scheduler.TaskType.PHASE_OPTIMIZATION,
            estimated_duration=3.0
        )
        
        # Entangle the tasks
        task1.entangled_tasks.add(task2.id)
        task2.entangled_tasks.add(task1.id)
        
        tasks = [task1, task2]
        
        scheduler = pm.QuantumInspiredScheduler(max_iterations=50)
        result = scheduler.schedule_tasks(tasks)
        
        # Entangled tasks should be scheduled considering their correlation
        assert result.makespan > 0
        assert len(result.schedule) > 0
        
        # Verify entanglement is preserved in task objects
        assert task2.id in task1.entangled_tasks
        assert task1.id in task2.entangled_tasks
    
    def test_quantum_annealing_convergence(self):
        """Test quantum annealing convergence behavior."""
        planner = pm.QuantumTaskPlanner()
        tasks = planner.create_compilation_plan({"model_type": "medium", "layers": 8})
        
        # Test different annealing parameters
        scheduler_fast = pm.QuantumInspiredScheduler(
            population_size=10,
            max_iterations=50,
            cooling_rate=0.8  # Fast cooling
        )
        
        scheduler_slow = pm.QuantumInspiredScheduler(
            population_size=10,
            max_iterations=50,
            cooling_rate=0.99  # Slow cooling
        )
        
        result_fast = scheduler_fast.schedule_tasks(tasks)
        result_slow = scheduler_slow.schedule_tasks(tasks)
        
        # Both should produce valid results
        assert result_fast.makespan > 0
        assert result_slow.makespan > 0
        
        # Get performance metrics
        metrics_fast = scheduler_fast.get_performance_metrics()
        metrics_slow = scheduler_slow.get_performance_metrics()
        
        assert "convergence_rate" in metrics_fast
        assert "convergence_rate" in metrics_slow
    
    def test_adaptive_parameter_tuning(self):
        """Test adaptive parameter tuning based on problem characteristics."""
        scheduler = pm.ParallelQuantumScheduler(
            optimization_level=pm.OptimizationLevel.BALANCED
        )
        
        # Test with different problem sizes
        small_tasks = [
            pm.CompilationTask(id=f"small_{i}", task_type=pm.quantum_scheduler.TaskType.GRAPH_LOWERING)
            for i in range(5)
        ]
        
        large_tasks = [
            pm.CompilationTask(id=f"large_{i}", task_type=pm.quantum_scheduler.TaskType.PHOTONIC_OPTIMIZATION)
            for i in range(50)
        ]
        
        # Create profiles for different problem sizes
        small_profile = scheduler._create_performance_profile(small_tasks)
        large_profile = scheduler._create_performance_profile(large_tasks)
        
        # Verify different profiles are generated
        assert small_profile.task_count < large_profile.task_count
        assert small_profile.complexity_score < large_profile.complexity_score
        
        # Adaptive parameters should be different
        small_params = scheduler._adapt_parameters(small_profile)
        large_params = scheduler._adapt_parameters(large_profile)
        
        # Large problems should get more iterations or larger population
        assert (large_params["max_iterations"] >= small_params["max_iterations"] or
                large_params["population_size"] >= small_params["population_size"])
    
    def test_hierarchical_scheduling(self):
        """Test hierarchical scheduling for very large problems."""
        scheduler = pm.ParallelQuantumScheduler(
            optimization_level=pm.OptimizationLevel.BALANCED
        )
        
        # Create very large task set that triggers hierarchical scheduling
        large_tasks = []
        for i in range(250):  # Large enough to trigger hierarchical mode
            task = pm.CompilationTask(
                id=f"hierarchical_task_{i:03d}",
                task_type=list(pm.quantum_scheduler.TaskType)[i % len(pm.quantum_scheduler.TaskType)],
                estimated_duration=1.0 + (i % 5) * 0.2
            )
            large_tasks.append(task)
        
        start_time = time.time()
        result = scheduler.schedule_tasks_optimized(large_tasks)
        optimization_time = time.time() - start_time
        
        # Should complete and produce valid result
        assert result.makespan > 0
        assert result.resource_utilization > 0
        
        # Should handle large problem in reasonable time
        assert optimization_time < 180  # 3 minutes max
        
        # Should achieve some parallelization
        sequential_time = sum(task.estimated_duration for task in large_tasks)
        assert result.makespan < sequential_time * 0.9  # At least 10% improvement
        
        logging.info(f"Hierarchical scheduling: {len(large_tasks)} tasks, "
                    f"makespan={result.makespan:.1f}s, "
                    f"time={optimization_time:.1f}s")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pytest.main([__file__, "-v", "--tb=short"])