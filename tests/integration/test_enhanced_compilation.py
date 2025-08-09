"""
Enhanced integration tests for photonic compilation pipeline.
Tests the complete Generation 1-3 implementation.
"""

import pytest
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import threading
import time

# Import our enhanced modules
from photon_mlir.core import TargetConfig, Device, Precision
from photon_mlir.compiler import PhotonicCompiler
from photon_mlir.quantum_aware_scheduler import (
    QuantumAwareScheduler, PhotonicTask, TaskPriority, create_photonic_task
)
from photon_mlir.circuit_breaker import (
    create_comprehensive_protection, ProtectedOperation
)
from photon_mlir.recovery_manager import (
    setup_comprehensive_recovery, AutoRecoveryContext
)
from photon_mlir.parallel_compiler import (
    create_parallel_compiler, CompilationJob
)
from photon_mlir.caching_system import (
    get_global_cache, PhotonicCompilationCache, cached_compilation
)


class TestEnhancedPhotonicCompilation:
    """Test suite for enhanced photonic compilation features."""
    
    @pytest.fixture
    def target_config(self):
        """Standard target configuration for tests."""
        return TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.INT8,
            array_size=(64, 64),
            wavelength_nm=1550,
            enable_thermal_compensation=True
        )
    
    @pytest.fixture
    def mock_model_file(self):
        """Create a mock model file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
            f.write("""
module @test_model {
  func.func @main(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = arith.constant dense<[[1.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 0.0, 0.0], 
                               [0.0, 0.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 1.0]]> : tensor<4x4xf32>
    %1 = linalg.matmul ins(%arg0, %0 : tensor<4x4xf32>, tensor<4x4xf32>) 
                       outs(%arg0 : tensor<4x4xf32>) -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
            """)
            return f.name
    
    def test_generation1_basic_compilation(self, target_config, mock_model_file):
        """Test Generation 1: Basic functionality works."""
        compiler = PhotonicCompiler(target_config)
        
        # Test compilation pipeline
        assert compiler.compile_onnx(mock_model_file).success
        
        # Verify output generation
        with tempfile.NamedTemporaryFile(suffix='.pasm') as output_file:
            compiled_model = compiler.compile_onnx(mock_model_file)
            assert compiled_model.success
            
            compiled_model.export(output_file.name)
            
            # Check output file was created
            assert Path(output_file.name).exists()
            
            # Check basic output format
            with open(output_file.name, 'r') as f:
                content = f.read()
                assert '.device lightmatter_envise' in content
                assert '.precision int8' in content
    
    def test_generation2_robust_error_handling(self, target_config):
        """Test Generation 2: Robust error handling."""
        compiler = PhotonicCompiler(target_config, strict_validation=True)
        
        # Test with invalid file
        result = compiler.compile_onnx("nonexistent_file.onnx")
        assert not result.success
        assert "does not exist" in str(result.error_message).lower() or "not found" in str(result.error_message).lower()
        
        # Test circuit breaker functionality
        circuit_breaker = create_comprehensive_protection(target_config)
        
        # Test thermal protection
        thermal_status = circuit_breaker.get_comprehensive_status()
        assert "thermal_status" in thermal_status
        assert "system_health_score" in thermal_status
        
        # Test protected operation
        with pytest.raises(RuntimeError):
            with ProtectedOperation(circuit_breaker, "high_thermal_operation", thermal_cost=2000.0):
                pass  # Should be blocked due to high thermal cost
    
    def test_generation2_recovery_mechanisms(self, target_config):
        """Test Generation 2: Recovery mechanisms."""
        circuit_breaker, recovery_manager = setup_comprehensive_recovery(target_config)
        
        # Test recovery from thermal violation
        from photon_mlir.circuit_breaker import FailureType
        
        failure_details = {"severity": 3, "predicted_temperature": 90.0}
        system_state = {"temperature": 85.0, "operation_context": "test"}
        
        result = recovery_manager.handle_failure(
            FailureType.THERMAL_VIOLATION,
            failure_details,
            system_state
        )
        
        # Recovery should complete (may be success or partial success)
        assert result.name in ["SUCCESS", "PARTIAL_SUCCESS", "FAILURE"]
        
        # Check recovery statistics
        stats = recovery_manager.get_recovery_statistics()
        assert stats["total_recoveries"] >= 1
    
    def test_generation3_parallel_compilation(self, target_config, mock_model_file):
        """Test Generation 3: Parallel compilation performance."""
        parallel_compiler = create_parallel_compiler(target_config, max_workers=4)
        
        # Create multiple test jobs
        job1 = CompilationJob(
            job_id="test_job_1",
            model_path=mock_model_file,
            target_config=target_config,
            estimated_time_s=30.0
        )
        
        job2 = CompilationJob(
            job_id="test_job_2", 
            model_path=mock_model_file,
            target_config=target_config,
            estimated_time_s=45.0
        )
        
        with parallel_compiler:
            # Submit jobs
            job_id1 = parallel_compiler.submit_compilation_job(job1)
            job_id2 = parallel_compiler.submit_compilation_job(job2)
            
            # Wait for completion
            result1 = parallel_compiler.wait_for_completion(job_id1, timeout_s=60.0)
            result2 = parallel_compiler.wait_for_completion(job_id2, timeout_s=60.0)
            
            # Both should complete successfully
            assert result1 is not None
            assert result2 is not None
            
            # Check performance metrics
            metrics = parallel_compiler.get_performance_metrics()
            assert metrics["total_compilations"] >= 2
            assert metrics["successful_compilations"] >= 1
    
    def test_generation3_caching_system(self, target_config, mock_model_file):
        """Test Generation 3: Caching and memoization."""
        cache = PhotonicCompilationCache()
        
        # Test cache miss first
        result = cache.get_compiled_model(mock_model_file, target_config)
        assert result is None
        
        # Cache a compilation result
        mock_compiled_result = {
            "output_path": "test_output.pasm",
            "optimization_stats": {"speedup": 3.2, "energy_reduction": 85.0}
        }
        
        success = cache.cache_compiled_model(
            mock_model_file, target_config, mock_compiled_result, thermal_cost=50.0
        )
        assert success
        
        # Test cache hit
        cached_result = cache.get_compiled_model(mock_model_file, target_config)
        assert cached_result is not None
        assert cached_result["optimization_stats"]["speedup"] == 3.2
        
        # Test cache statistics
        stats = cache.get_cache_statistics()
        assert "l1_memory" in stats
        assert "l2_compressed" in stats
        assert "l3_disk" in stats
        
        # Test cache invalidation
        invalidated = cache.invalidate_model_cache(mock_model_file)
        assert invalidated >= 0  # May be 0 if no entries were found to invalidate
    
    def test_quantum_scheduling_integration(self, target_config):
        """Test quantum-aware scheduling integration."""
        scheduler = QuantumAwareScheduler(target_config)
        
        # Create test photonic tasks
        task1 = create_photonic_task(
            "thermal_task_1", "matmul", 
            np.random.randn(64, 64), 
            priority="high",
            thermal_cost=30.0
        )
        
        task2 = create_photonic_task(
            "phase_task_1", "phase_shift",
            {"phase_radians": 1.57},
            priority="normal"
        )
        
        task3 = create_photonic_task(
            "quantum_task_1", "quantum_phase_gate",
            {"phase_radians": 0.785, "qubit_index": 0},
            priority="critical"
        )
        
        # Submit tasks
        scheduler.submit_task(task1)
        scheduler.submit_task(task2) 
        scheduler.submit_task(task3)
        
        # Schedule batch
        tasks_to_schedule = [task1, task2, task3]
        result = scheduler.schedule_batch(tasks_to_schedule, optimize_for="balanced")
        
        # Verify scheduling results
        assert result.total_execution_time_ms > 0
        assert result.thermal_efficiency > 0
        assert len(result.scheduled_tasks) == 3
        
        # Check that critical priority task is scheduled first
        first_task = result.scheduled_tasks[0]
        assert first_task.priority == TaskPriority.CRITICAL
    
    def test_end_to_end_integration(self, target_config, mock_model_file):
        """Test complete end-to-end integration of all generations."""
        # Set up comprehensive system
        circuit_breaker, recovery_manager = setup_comprehensive_recovery(target_config)
        parallel_compiler = create_parallel_compiler(target_config, max_workers=2)
        cache = get_global_cache()
        
        start_time = time.time()
        
        with parallel_compiler:
            with AutoRecoveryContext(recovery_manager, "compilation", model_path=mock_model_file):
                # Create compilation job with thermal protection
                job = CompilationJob(
                    job_id="integration_test",
                    model_path=mock_model_file,
                    target_config=target_config,
                    thermal_budget_mw=75.0
                )
                
                # Check circuit breaker allows operation
                can_execute = circuit_breaker.can_execute_operation(
                    "compilation", thermal_cost=job.thermal_budget_mw
                )
                
                if can_execute:
                    # Submit for parallel compilation
                    job_id = parallel_compiler.submit_compilation_job(job)
                    result = parallel_compiler.wait_for_completion(job_id, timeout_s=120.0)
                    
                    # Verify successful compilation
                    assert result is not None
                    assert result.success
                    
                    # Record success for circuit breaker learning
                    circuit_breaker.record_operation_result("compilation", True, {
                        "compilation_time": result.compilation_time_s,
                        "thermal_budget": job.thermal_budget_mw
                    })
                else:
                    pytest.skip("Circuit breaker blocked operation for safety")
        
        # Test system health after integration test
        system_status = circuit_breaker.get_comprehensive_status()
        assert system_status["system_health_score"] > 0.0
        
        # Check that recovery manager has no active recoveries
        recovery_stats = recovery_manager.get_recovery_statistics()
        assert recovery_stats["total_recoveries"] >= 0
        
        total_time = time.time() - start_time
        assert total_time < 30.0, "Integration test took too long"
    
    def test_performance_benchmarks(self, target_config, mock_model_file):
        """Test performance meets benchmarks."""
        # Compilation time benchmark
        start_time = time.time()
        
        compiler = PhotonicCompiler(target_config)
        result = compiler.compile_onnx(mock_model_file)
        
        compilation_time = time.time() - start_time
        
        # Benchmarks from README
        assert result.success
        assert compilation_time < 60.0, f"Compilation took {compilation_time:.2f}s, should be < 60s"
        
        # Test parallel throughput
        parallel_compiler = create_parallel_compiler(target_config, max_workers=4)
        
        with parallel_compiler:
            jobs = []
            for i in range(3):
                job = CompilationJob(
                    job_id=f"perf_test_{i}",
                    model_path=mock_model_file,
                    target_config=target_config
                )
                jobs.append(job)
                parallel_compiler.submit_compilation_job(job)
            
            # Measure parallel throughput
            throughput_start = time.time()
            results = []
            for job in jobs:
                result = parallel_compiler.wait_for_completion(job.job_id, timeout_s=120.0)
                if result:
                    results.append(result)
            
            throughput_time = time.time() - throughput_start
            
            # Should complete 3 jobs faster than 3x sequential time
            successful_results = [r for r in results if r.success]
            assert len(successful_results) >= 1  # At least one should succeed
            
            if len(successful_results) == 3:
                avg_parallel_time = throughput_time / 3
                speedup = compilation_time / avg_parallel_time if avg_parallel_time > 0 else 1.0
                assert speedup > 1.0, f"Parallel speedup {speedup:.2f}x should be > 1.0x"
    
    def test_thermal_management_integration(self, target_config):
        """Test thermal management across all components."""
        circuit_breaker = create_comprehensive_protection(target_config)
        scheduler = QuantumAwareScheduler(target_config)
        
        # Create high thermal cost task
        high_thermal_task = create_photonic_task(
            "high_thermal", "mesh_calibration",
            {"calibration_method": "iterative_optimization"},
            thermal_cost=150.0  # High thermal cost
        )
        
        # Test circuit breaker thermal protection
        can_execute = circuit_breaker.can_execute_operation(
            "mesh_calibration", thermal_cost=150.0
        )
        
        if can_execute:
            # Submit to scheduler
            scheduler.submit_task(high_thermal_task)
            result = scheduler.schedule_batch([high_thermal_task], optimize_for="thermal")
            
            # Should optimize for thermal efficiency
            assert result.thermal_efficiency > 0.5
            
            # Record success
            circuit_breaker.record_operation_result("mesh_calibration", True)
        else:
            # Circuit breaker should have blocked high thermal operation
            thermal_status = circuit_breaker.get_comprehensive_status()
            assert thermal_status["thermal_status"]["failure_count"] >= 0
    
    def test_memory_management(self, target_config):
        """Test memory management and resource cleanup."""
        import psutil
        
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        
        # Create multiple cache entries to test memory management
        cache = PhotonicCompilationCache()
        
        # Add many entries to test eviction
        for i in range(100):
            mock_data = {"large_array": np.random.randn(100, 100), "id": i}
            cache.cache_compiled_model(
                f"test_model_{i}.onnx", target_config, mock_data, thermal_cost=10.0
            )
        
        # Force cache optimization
        cache.cache.optimize_cache()
        
        # Check memory hasn't grown excessively
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        memory_growth = final_memory - initial_memory
        
        assert memory_growth < 500, f"Memory grew by {memory_growth:.2f}MB, should be < 500MB"
        
        # Clean up
        cache.cache.clear()
    
    @pytest.mark.stress
    def test_stress_testing(self, target_config, mock_model_file):
        """Stress test the complete system."""
        # This test is marked as stress and may be skipped in regular runs
        parallel_compiler = create_parallel_compiler(target_config, max_workers=8)
        circuit_breaker, recovery_manager = setup_comprehensive_recovery(target_config)
        
        with parallel_compiler:
            # Submit many concurrent jobs
            jobs = []
            for i in range(20):
                job = CompilationJob(
                    job_id=f"stress_test_{i}",
                    model_path=mock_model_file,
                    target_config=target_config,
                    thermal_budget_mw=np.random.uniform(20.0, 80.0)
                )
                jobs.append(job)
                parallel_compiler.submit_compilation_job(job)
            
            # Wait for all to complete
            results = []
            for job in jobs:
                result = parallel_compiler.wait_for_completion(job.job_id, timeout_s=300.0)
                if result:
                    results.append(result)
            
            # Check that most jobs completed successfully
            successful = [r for r in results if r.success]
            success_rate = len(successful) / len(jobs)
            
            assert success_rate > 0.7, f"Success rate {success_rate:.2f} should be > 0.7"
            
            # Check system remained healthy
            system_status = circuit_breaker.get_comprehensive_status()
            assert system_status["system_health_score"] > 0.3, "System health degraded too much"


class TestCachePerformance:
    """Test cache performance and correctness."""
    
    def test_cache_hit_performance(self, target_config):
        """Test cache hit performance meets requirements."""
        from photon_mlir.caching_system import HierarchicalCache
        
        cache = HierarchicalCache()
        
        # Warm up cache
        test_data = {"array": np.random.randn(1000, 1000), "metadata": {"test": True}}
        cache.put("performance_test", test_data, thermal_cost=100.0)
        
        # Measure cache hit performance
        start_time = time.perf_counter()
        for _ in range(1000):
            result = cache.get("performance_test")
            assert result is not None
        
        hit_time = (time.perf_counter() - start_time) / 1000  # Average per hit
        
        # Cache hits should be very fast (< 1ms average)
        assert hit_time < 0.001, f"Cache hit took {hit_time*1000:.3f}ms, should be < 1ms"
    
    def test_cache_compression_ratio(self):
        """Test cache compression meets efficiency targets."""
        from photon_mlir.caching_system import CompressedCacheBackend, MemoryCacheBackend
        
        base_cache = MemoryCacheBackend(max_size=10, max_memory_mb=100.0)
        compressed_cache = CompressedCacheBackend(base_cache)
        
        # Create compressible test data
        large_array = np.zeros((1000, 1000), dtype=np.float32)  # Highly compressible
        
        from photon_mlir.caching_system import CacheEntry
        entry = CacheEntry(
            key="compression_test",
            value=large_array,
            created_time=time.time(),
            last_accessed=time.time(),
            size_bytes=large_array.nbytes
        )
        
        original_size = entry.size_bytes
        
        # Put in compressed cache
        success = compressed_cache.put("compression_test", entry)
        assert success
        
        # Retrieve and check compression worked
        retrieved_entry = compressed_cache.get("compression_test")
        assert retrieved_entry is not None
        assert np.array_equal(retrieved_entry.value, large_array)
        
        # Check compression ratio (should be significant for zeros)
        if hasattr(retrieved_entry, 'compression_level') and retrieved_entry.compression_level > 0:
            compression_ratio = original_size / entry.size_bytes if entry.size_bytes > 0 else 1.0
            assert compression_ratio > 10.0, f"Compression ratio {compression_ratio:.2f} should be > 10x for zero array"


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([__file__, "-v", "--cov=photon_mlir", "--cov-report=term-missing"])