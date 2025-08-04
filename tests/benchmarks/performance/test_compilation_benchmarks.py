"""
Performance benchmarks for photonic compilation.
"""

import pytest
import time
import statistics
import tempfile
import os
from typing import List, Dict, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from photon_mlir.compiler import PhotonicCompiler, compile
from photon_mlir.core import TargetConfig, Device, Precision
from photon_mlir.optimization import get_profiler, profile_performance


@pytest.mark.benchmark
class TestCompilationPerformance:
    """Benchmark compilation performance."""
    
    def measure_compilation_time(self, model_size: str, num_runs: int = 5) -> Dict[str, float]:
        """Measure compilation time for different model sizes."""
        times = []
        
        for _ in range(num_runs):
            # Create dummy model file
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
                # Simulate different model sizes with different file sizes
                if model_size == "small":
                    f.write(b"small model data" * 100)
                elif model_size == "medium":
                    f.write(b"medium model data" * 1000)
                else:  # large
                    f.write(b"large model data" * 10000)
                model_path = f.name
            
            try:
                start_time = time.perf_counter()
                
                # Compile model
                compiler = PhotonicCompiler()
                compiled_model = compiler.compile_onnx(model_path)
                
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
            finally:
                os.unlink(model_path)
        
        return {
            "mean": statistics.mean(times),
            "median": statistics.median(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "runs": num_runs
        }
    
    def test_small_model_compilation_time(self):
        """Benchmark small model compilation."""
        results = self.measure_compilation_time("small", num_runs=10)
        
        # Performance expectations (adjust based on actual performance)
        assert results["mean"] < 5.0, f"Small model compilation too slow: {results['mean']:.3f}s"
        assert results["max"] < 10.0, f"Worst case too slow: {results['max']:.3f}s"
        
        print(f"Small model compilation: {results['mean']:.3f}s ± {results['stdev']:.3f}s")
    
    def test_medium_model_compilation_time(self):
        """Benchmark medium model compilation."""
        results = self.measure_compilation_time("medium", num_runs=5)
        
        # Allow more time for medium models
        assert results["mean"] < 15.0, f"Medium model compilation too slow: {results['mean']:.3f}s"
        assert results["max"] < 30.0, f"Worst case too slow: {results['max']:.3f}s"
        
        print(f"Medium model compilation: {results['mean']:.3f}s ± {results['stdev']:.3f}s")
    
    @pytest.mark.slow
    def test_large_model_compilation_time(self):
        """Benchmark large model compilation."""
        results = self.measure_compilation_time("large", num_runs=3)
        
        # Allow significant time for large models
        assert results["mean"] < 60.0, f"Large model compilation too slow: {results['mean']:.3f}s"
        assert results["max"] < 120.0, f"Worst case too slow: {results['max']:.3f}s"
        
        print(f"Large model compilation: {results['mean']:.3f}s ± {results['stdev']:.3f}s")
    
    def test_compilation_scalability(self):
        """Test compilation time scaling with model complexity."""
        model_sizes = ["small", "medium", "large"]
        results = {}
        
        for size in model_sizes:
            results[size] = self.measure_compilation_time(size, num_runs=3)
        
        # Verify reasonable scaling (shouldn't be exponential)
        small_time = results["small"]["mean"]
        medium_time = results["medium"]["mean"]
        large_time = results["large"]["mean"]
        
        # Medium should be at most 5x slower than small
        assert medium_time <= small_time * 5, "Medium model scaling too poor"
        
        # Large should be at most 10x slower than medium
        assert large_time <= medium_time * 10, "Large model scaling too poor"
        
        print(f"Scaling: small={small_time:.3f}s, medium={medium_time:.3f}s, large={large_time:.3f}s")


@pytest.mark.benchmark
class TestTargetConfigurationPerformance:
    """Benchmark performance with different target configurations."""
    
    def test_device_compilation_performance(self):
        """Compare compilation performance across different devices."""
        devices = [Device.LIGHTMATTER_ENVISE, Device.MIT_PHOTONIC_PROCESSOR, Device.CUSTOM_RESEARCH_CHIP]
        results = {}
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"test model data" * 500)
            model_path = f.name
        
        try:
            for device in devices:
                config = TargetConfig(device=device)
                
                times = []
                for _ in range(3):
                    start_time = time.perf_counter()
                    
                    compiler = PhotonicCompiler(config)
                    compiled_model = compiler.compile_onnx(model_path)
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                results[device.value] = {
                    "mean": statistics.mean(times),
                    "stdev": statistics.stdev(times) if len(times) > 1 else 0
                }
            
            # Verify all devices perform reasonably
            for device_name, stats in results.items():
                assert stats["mean"] < 10.0, f"Device {device_name} too slow: {stats['mean']:.3f}s"
                print(f"Device {device_name}: {stats['mean']:.3f}s ± {stats['stdev']:.3f}s")
            
        finally:
            os.unlink(model_path)
    
    def test_precision_compilation_performance(self):
        """Compare compilation performance across different precisions."""
        precisions = [Precision.INT8, Precision.INT16, Precision.FP16, Precision.FP32]
        results = {}
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"test model data" * 500)
            model_path = f.name
        
        try:
            for precision in precisions:
                config = TargetConfig(precision=precision)
                
                times = []
                for _ in range(3):
                    start_time = time.perf_counter()
                    
                    compiler = PhotonicCompiler(config)
                    compiled_model = compiler.compile_onnx(model_path)
                    
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                results[precision.value] = statistics.mean(times)
            
            # Print results
            for precision_name, mean_time in results.items():
                print(f"Precision {precision_name}: {mean_time:.3f}s")
            
            # Verify reasonable performance
            for mean_time in results.values():
                assert mean_time < 10.0, f"Precision compilation too slow: {mean_time:.3f}s"
            
        finally:
            os.unlink(model_path)


@pytest.mark.benchmark
class TestSimulationPerformance:
    """Benchmark simulation performance."""
    
    def test_simulation_throughput(self):
        """Benchmark simulation throughput with different input sizes."""
        from photon_mlir.simulator import PhotonicSimulator
        
        # Create compiled model
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model data")
            model_path = f.name
        
        try:
            compiled_model = compile(model_path)
            simulator = PhotonicSimulator(noise_model="realistic")
            
            # Test different input sizes
            input_sizes = [
                (1, 10),      # Small
                (10, 100),    # Medium  
                (100, 1000),  # Large
            ]
            
            results = {}
            
            for batch_size, features in input_sizes:
                input_data = np.random.randn(batch_size, features).astype(np.float32)
                
                # Measure simulation time
                times = []
                for _ in range(5):
                    start_time = time.perf_counter()
                    output = simulator.run(compiled_model, input_data)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                mean_time = statistics.mean(times)
                throughput = (batch_size * features) / mean_time  # Elements per second
                
                results[f"{batch_size}x{features}"] = {
                    "time": mean_time,
                    "throughput": throughput
                }
                
                print(f"Input {batch_size}x{features}: {mean_time:.3f}s, {throughput:.0f} elem/s")
            
            # Verify reasonable performance
            for size, stats in results.items():
                assert stats["time"] < 5.0, f"Simulation too slow for {size}: {stats['time']:.3f}s"
                assert stats["throughput"] > 100, f"Throughput too low for {size}: {stats['throughput']:.0f}"
            
        finally:
            os.unlink(model_path)
    
    def test_noise_model_performance(self):
        """Compare performance of different noise models."""
        from photon_mlir.simulator import PhotonicSimulator
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model")
            model_path = f.name
        
        try:
            compiled_model = compile(model_path)
            input_data = np.random.randn(50, 100).astype(np.float32)
            
            noise_models = ["ideal", "realistic", "pessimistic"]
            results = {}
            
            for noise_model in noise_models:
                simulator = PhotonicSimulator(noise_model=noise_model)
                
                times = []
                for _ in range(3):
                    start_time = time.perf_counter()
                    output = simulator.run(compiled_model, input_data)
                    end_time = time.perf_counter()
                    times.append(end_time - start_time)
                
                mean_time = statistics.mean(times)
                results[noise_model] = mean_time
                
                print(f"Noise model {noise_model}: {mean_time:.3f}s")
            
            # Verify all models perform reasonably
            for model, time_taken in results.items():
                assert time_taken < 5.0, f"Noise model {model} too slow: {time_taken:.3f}s"
            
            # Ideal should be fastest (no noise computation)
            assert results["ideal"] <= results["realistic"], "Ideal model should be fastest"
            
        finally:
            os.unlink(model_path)


@pytest.mark.benchmark
class TestCachePerformance:
    """Benchmark caching performance."""
    
    def test_cache_hit_performance(self):
        """Benchmark cache hit vs miss performance."""
        from photon_mlir.optimization import get_compilation_cache
        
        cache = get_compilation_cache()
        
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model data" * 1000)
            model_path = f.name
        
        try:
            target_config = TargetConfig().to_dict()
            
            # Measure cache miss (first compilation)
            start_time = time.perf_counter()
            compiled_model = compile(model_path)
            cache.put_compiled_model(model_path, target_config, compiled_model)
            miss_time = time.perf_counter() - start_time
            
            # Measure cache hit
            hit_times = []
            for _ in range(10):
                start_time = time.perf_counter()
                cached_model = cache.get_compiled_model(model_path, target_config)
                hit_time = time.perf_counter() - start_time
                hit_times.append(hit_time)
                
                assert cached_model is not None, "Cache should hit"
            
            mean_hit_time = statistics.mean(hit_times)
            
            print(f"Cache miss: {miss_time:.3f}s")
            print(f"Cache hit: {mean_hit_time:.6f}s")
            
            # Cache hit should be much faster
            assert mean_hit_time < miss_time / 10, "Cache hit not significantly faster"
            assert mean_hit_time < 0.01, "Cache hit too slow"
            
        finally:
            os.unlink(model_path)
    
    def test_cache_memory_performance(self):
        """Test cache performance under memory pressure."""
        from photon_mlir.optimization import CompilationCache
        
        # Create cache with small memory limit
        cache = CompilationCache(
            memory_cache_size=5,  # Only 5 items
            memory_cache_mb=1.0   # 1MB limit
        )
        
        # Create multiple models to exceed cache size
        model_paths = []
        compiled_models = []
        
        for i in range(10):  # More than cache size
            with tempfile.NamedTemporaryFile(suffix=f'_model_{i}.onnx', delete=False) as f:
                f.write(f"model {i} data".encode() * 100)
                model_paths.append(f.name)
        
        try:
            target_config = TargetConfig().to_dict()
            
            # Fill cache beyond capacity
            for i, model_path in enumerate(model_paths):
                compiled_model = compile(model_path)
                cache.put_compiled_model(model_path, target_config, compiled_model)
                compiled_models.append(compiled_model)
            
            # Test cache performance with eviction
            hit_times = []
            miss_times = []
            
            for model_path in model_paths:
                start_time = time.perf_counter()
                cached_model = cache.get_compiled_model(model_path, target_config)
                lookup_time = time.perf_counter() - start_time
                
                if cached_model is not None:
                    hit_times.append(lookup_time)
                else:
                    miss_times.append(lookup_time)
            
            # Should have both hits and misses due to eviction
            print(f"Cache hits: {len(hit_times)}, misses: {len(miss_times)}")
            
            if hit_times:
                print(f"Average hit time: {statistics.mean(hit_times):.6f}s")
            if miss_times:
                print(f"Average miss time: {statistics.mean(miss_times):.6f}s")
            
            # Performance should still be reasonable
            all_times = hit_times + miss_times
            assert statistics.mean(all_times) < 0.01, "Cache lookup too slow under pressure"
            
        finally:
            for path in model_paths:
                os.unlink(path)


@pytest.mark.benchmark
class TestConcurrencyPerformance:
    """Benchmark concurrent processing performance."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_parallel_compilation_performance(self):
        """Benchmark parallel vs sequential compilation."""
        from photon_mlir.concurrency import get_worker_pool
        
        # Create multiple models
        model_paths = []
        for i in range(5):
            model = torch.nn.Linear(100, 50)
            with tempfile.NamedTemporaryFile(suffix=f'_model_{i}.pt', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                model_paths.append(f.name)
        
        try:
            target_configs = [TargetConfig().to_dict() for _ in model_paths]
            
            # Sequential compilation
            start_time = time.perf_counter()
            sequential_results = []
            for model_path, config in zip(model_paths, target_configs):
                compiler = PhotonicCompiler(TargetConfig(**config))
                # Note: This would normally call compile_pytorch, but we're using the file path
                # In real implementation, this would be properly handled
                compiled_model = compile(model_path)
                sequential_results.append(compiled_model)
            sequential_time = time.perf_counter() - start_time
            
            # Parallel compilation
            worker_pool = get_worker_pool()
            
            start_time = time.perf_counter()
            futures = worker_pool.compile_batch_async(model_paths, target_configs, max_concurrent=3)
            parallel_results, failed = worker_pool.wait_for_completion(futures, timeout=30.0)
            parallel_time = time.perf_counter() - start_time
            
            assert len(failed) == 0, "Some parallel compilations failed"
            assert len(parallel_results) == len(model_paths), "Not all models compiled"
            
            print(f"Sequential: {sequential_time:.3f}s")
            print(f"Parallel: {parallel_time:.3f}s")
            print(f"Speedup: {sequential_time / parallel_time:.2f}x")
            
            # Parallel should be faster for multiple models
            assert parallel_time < sequential_time, "Parallel compilation should be faster"
            
            # Should achieve reasonable speedup
            speedup = sequential_time / parallel_time
            assert speedup > 1.5, f"Insufficient speedup: {speedup:.2f}x"
            
        finally:
            for path in model_paths:
                os.unlink(path)
    
    def test_resource_pool_performance(self):
        """Benchmark resource pool acquisition performance."""
        from photon_mlir.concurrency import ResourcePool
        
        # Create resource pool
        def create_resource():
            return {"data": "resource", "id": id({})}
        
        def destroy_resource(resource):
            pass  # Nothing to clean up for dict
        
        pool = ResourcePool(
            factory=create_resource,
            destroyer=destroy_resource,
            min_size=2,
            max_size=10
        )
        
        # Benchmark resource acquisition
        acquisition_times = []
        
        for _ in range(100):
            start_time = time.perf_counter()
            with pool.acquire() as resource:
                assert resource is not None
                # Simulate some work
                time.sleep(0.001)  # 1ms
            acquisition_time = time.perf_counter() - start_time
            acquisition_times.append(acquisition_time)
        
        mean_time = statistics.mean(acquisition_times)
        max_time = max(acquisition_times)
        
        print(f"Resource acquisition: {mean_time:.6f}s ± {statistics.stdev(acquisition_times):.6f}s")
        print(f"Max acquisition time: {max_time:.6f}s")
        
        # Resource acquisition should be fast
        assert mean_time < 0.01, f"Resource acquisition too slow: {mean_time:.6f}s"
        assert max_time < 0.05, f"Worst case too slow: {max_time:.6f}s"
        
        pool.shutdown()


@pytest.mark.benchmark
class TestRegressionDetection:
    """Benchmark regression detection tests."""
    
    def test_performance_regression_baseline(self):
        """Establish performance baselines for regression detection."""
        baselines = {}
        
        # Small model compilation baseline
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"small model" * 50)
            model_path = f.name
        
        try:
            times = []
            for _ in range(5):
                start_time = time.perf_counter()
                compiled_model = compile(model_path)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            baselines['small_model_compilation'] = {
                'mean': statistics.mean(times),
                'p95': sorted(times)[int(len(times) * 0.95)],
                'samples': len(times)
            }
            
        finally:
            os.unlink(model_path)
        
        # Simulation baseline
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model")
            model_path = f.name
        
        try:
            compiled_model = compile(model_path)
            input_data = np.random.randn(10, 50).astype(np.float32)
            
            times = []
            for _ in range(10):
                start_time = time.perf_counter()
                output = compiled_model.simulate(input_data)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            baselines['simulation_10x50'] = {
                'mean': statistics.mean(times),
                'p95': sorted(times)[int(len(times) * 0.95)],
                'samples': len(times)
            }
            
        finally:
            os.unlink(model_path)
        
        # Print baselines for future reference
        print("Performance Baselines:")
        for test_name, stats in baselines.items():
            print(f"  {test_name}: {stats['mean']:.6f}s mean, {stats['p95']:.6f}s p95")
        
        # Store baselines (in real implementation, would save to file)
        self._baselines = baselines
        
        # All baselines should be reasonable
        assert baselines['small_model_compilation']['mean'] < 5.0
        assert baselines['simulation_10x50']['mean'] < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])