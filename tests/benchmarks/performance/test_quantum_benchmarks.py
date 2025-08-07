"""
Performance benchmarks for quantum-inspired task scheduling system.

Comprehensive benchmarking suite to measure and track performance
characteristics of the quantum scheduling algorithms.
"""

import pytest
import time
import statistics
import logging
from typing import List, Dict, Any
import concurrent.futures
from pathlib import Path
import json

import photon_mlir as pm


class QuantumSchedulerBenchmark:
    """Benchmark suite for quantum scheduler performance."""
    
    def __init__(self):
        self.results = {}
        self.baseline_times = {}
        
    def benchmark_scheduling_scalability(self):
        """Benchmark scheduling performance across different problem sizes."""
        print("\n" + "="*60)
        print("QUANTUM SCHEDULER SCALABILITY BENCHMARK")
        print("="*60)
        
        task_counts = [10, 25, 50, 100, 200]
        optimization_levels = [
            pm.OptimizationLevel.FAST,
            pm.OptimizationLevel.BALANCED,
            pm.OptimizationLevel.QUALITY
        ]
        
        results = {}
        
        for level in optimization_levels:
            level_results = {}
            
            for task_count in task_counts:
                print(f"\nTesting {level.value} optimization with {task_count} tasks...")
                
                # Generate tasks
                tasks = self._generate_benchmark_tasks(task_count)
                
                # Run benchmark
                scheduler = pm.ParallelQuantumScheduler(
                    optimization_level=level,
                    cache_strategy=pm.CacheStrategy.NONE
                )
                
                times = []
                makespans = []
                utilizations = []
                
                # Multiple runs for statistical significance
                runs = 3 if task_count <= 100 else 1
                
                for run in range(runs):
                    start_time = time.perf_counter()
                    result = scheduler.schedule_tasks_optimized(tasks)
                    elapsed = time.perf_counter() - start_time
                    
                    times.append(elapsed)
                    makespans.append(result.makespan)
                    utilizations.append(result.resource_utilization)
                
                # Calculate statistics
                level_results[task_count] = {
                    "avg_time": statistics.mean(times),
                    "std_time": statistics.stdev(times) if len(times) > 1 else 0,
                    "avg_makespan": statistics.mean(makespans),
                    "avg_utilization": statistics.mean(utilizations),
                    "times": times
                }
                
                print(f"  Average time: {level_results[task_count]['avg_time']:.3f}s ± "
                      f"{level_results[task_count]['std_time']:.3f}s")
                print(f"  Makespan: {level_results[task_count]['avg_makespan']:.2f}s")
                print(f"  Utilization: {level_results[task_count]['avg_utilization']:.1%}")
            
            results[level.value] = level_results
        
        # Print scaling analysis
        print(f"\n{'Level':<12} {'Size':<6} {'Time (s)':<12} {'Scaling':<10}")
        print("-" * 50)
        
        for level_name, level_data in results.items():
            prev_time = None
            prev_size = None
            
            for size, data in level_data.items():
                avg_time = data['avg_time']
                
                if prev_time is not None:
                    scaling = (avg_time / prev_time) / (size / prev_size)
                    scaling_str = f"{scaling:.2f}x"
                else:
                    scaling_str = "baseline"
                
                print(f"{level_name:<12} {size:<6} {avg_time:<12.3f} {scaling_str:<10}")
                
                prev_time = avg_time
                prev_size = size
        
        self.results["scalability"] = results
        return results
    
    def benchmark_caching_performance(self):
        """Benchmark caching system performance."""
        print("\n" + "="*60)
        print("QUANTUM SCHEDULER CACHING BENCHMARK")
        print("="*60)
        
        cache_strategies = [
            pm.CacheStrategy.NONE,
            pm.CacheStrategy.MEMORY_ONLY,
            pm.CacheStrategy.HYBRID
        ]
        
        task_count = 50
        tasks = self._generate_benchmark_tasks(task_count)
        
        results = {}
        
        for strategy in cache_strategies:
            print(f"\nTesting cache strategy: {strategy.value}")
            
            scheduler = pm.ParallelQuantumScheduler(
                optimization_level=pm.OptimizationLevel.BALANCED,
                cache_strategy=strategy
            )
            
            # First run - populate cache
            start_time = time.perf_counter()
            result1 = scheduler.schedule_tasks_optimized(tasks)
            first_run_time = time.perf_counter() - start_time
            
            cache_times = []
            
            # Subsequent runs - should hit cache (if enabled)
            for run in range(5):
                start_time = time.perf_counter()
                result = scheduler.schedule_tasks_optimized(tasks)
                cache_time = time.perf_counter() - start_time
                cache_times.append(cache_time)
                
                # Verify results are identical (for cached strategies)
                if strategy != pm.CacheStrategy.NONE:
                    assert abs(result.makespan - result1.makespan) < 0.001
            
            avg_cache_time = statistics.mean(cache_times)
            
            cache_stats = scheduler.cache.get_stats() if strategy != pm.CacheStrategy.NONE else {}
            
            results[strategy.value] = {
                "first_run_time": first_run_time,
                "avg_cache_time": avg_cache_time,
                "speedup": first_run_time / avg_cache_time,
                "cache_stats": cache_stats
            }
            
            print(f"  First run: {first_run_time:.3f}s")
            print(f"  Cached runs: {avg_cache_time:.3f}s (avg)")
            print(f"  Speedup: {first_run_time / avg_cache_time:.1f}x")
            
            if cache_stats:
                print(f"  Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
        
        self.results["caching"] = results
        return results
    
    def benchmark_parallel_efficiency(self):
        """Benchmark parallel processing efficiency."""
        print("\n" + "="*60)
        print("QUANTUM SCHEDULER PARALLELIZATION BENCHMARK")
        print("="*60)
        
        task_count = 100
        tasks = self._generate_benchmark_tasks(task_count, complex_dependencies=True)
        
        max_workers_settings = [1, 2, 4, 8]
        results = {}
        
        baseline_time = None
        
        for max_workers in max_workers_settings:
            print(f"\nTesting with {max_workers} workers...")
            
            scheduler = pm.ParallelQuantumScheduler(
                optimization_level=pm.OptimizationLevel.BALANCED,
                max_workers=max_workers,
                cache_strategy=pm.CacheStrategy.NONE
            )
            
            start_time = time.perf_counter()
            result = scheduler.schedule_tasks_optimized(tasks)
            elapsed = time.perf_counter() - start_time
            
            if baseline_time is None:
                baseline_time = elapsed
            
            parallel_efficiency = (baseline_time / elapsed) / max_workers
            speedup = baseline_time / elapsed
            
            results[max_workers] = {
                "time": elapsed,
                "speedup": speedup,
                "efficiency": parallel_efficiency,
                "makespan": result.makespan,
                "utilization": result.resource_utilization
            }
            
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Speedup: {speedup:.2f}x")
            print(f"  Efficiency: {parallel_efficiency:.1%}")
            print(f"  Makespan: {result.makespan:.2f}s")
        
        self.results["parallelization"] = results
        return results
    
    def benchmark_algorithm_convergence(self):
        """Benchmark algorithm convergence characteristics."""
        print("\n" + "="*60)
        print("QUANTUM ALGORITHM CONVERGENCE BENCHMARK")  
        print("="*60)
        
        task_count = 75
        tasks = self._generate_benchmark_tasks(task_count)
        
        # Test different annealing parameters
        configurations = [
            {"name": "aggressive", "cooling_rate": 0.8, "max_iterations": 200},
            {"name": "balanced", "cooling_rate": 0.95, "max_iterations": 500},
            {"name": "conservative", "cooling_rate": 0.99, "max_iterations": 1000}
        ]
        
        results = {}
        
        for config in configurations:
            print(f"\nTesting {config['name']} configuration...")
            
            scheduler = pm.QuantumInspiredScheduler(
                population_size=30,
                cooling_rate=config["cooling_rate"],
                max_iterations=config["max_iterations"],
                enable_validation=False  # For pure performance measurement
            )
            
            start_time = time.perf_counter()
            result = scheduler.schedule_tasks(tasks)
            elapsed = time.perf_counter() - start_time
            
            # Get performance metrics
            metrics = scheduler.get_performance_metrics()
            
            results[config["name"]] = {
                "time": elapsed,
                "makespan": result.makespan,
                "utilization": result.resource_utilization,
                "iterations": metrics.get("total_iterations", 0),
                "convergence_rate": metrics.get("convergence_rate", 0),
                "energy_improvement": metrics.get("energy_improvement", 0)
            }
            
            print(f"  Time: {elapsed:.3f}s")
            print(f"  Iterations: {results[config['name']]['iterations']}")
            print(f"  Makespan: {result.makespan:.2f}s")
            print(f"  Energy improvement: {results[config['name']]['energy_improvement']:.1%}")
        
        self.results["convergence"] = results
        return results
    
    def benchmark_memory_usage(self):
        """Benchmark memory usage characteristics."""
        print("\n" + "="*60)
        print("QUANTUM SCHEDULER MEMORY BENCHMARK")
        print("="*60)
        
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
        except ImportError:
            print("psutil not available, skipping memory benchmark")
            return {}
        
        task_counts = [50, 100, 200, 500]
        results = {}
        
        for task_count in task_counts:
            print(f"\nTesting memory usage with {task_count} tasks...")
            
            # Measure initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Generate tasks and run scheduler
            tasks = self._generate_benchmark_tasks(task_count)
            
            scheduler = pm.ParallelQuantumScheduler(
                optimization_level=pm.OptimizationLevel.BALANCED,
                cache_strategy=pm.CacheStrategy.MEMORY_ONLY
            )
            
            peak_memory = initial_memory
            
            def memory_monitor():
                nonlocal peak_memory
                while True:
                    current_memory = process.memory_info().rss / 1024 / 1024
                    peak_memory = max(peak_memory, current_memory)
                    time.sleep(0.01)
            
            # Start memory monitoring
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                monitor_future = executor.submit(memory_monitor)
                
                try:
                    # Run scheduling
                    start_time = time.perf_counter()
                    result = scheduler.schedule_tasks_optimized(tasks)
                    elapsed = time.perf_counter() - start_time
                    
                    # Stop monitoring
                    monitor_future.cancel()
                except Exception as e:
                    monitor_future.cancel()
                    raise e
            
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            results[task_count] = {
                "initial_memory": initial_memory,
                "peak_memory": peak_memory,
                "final_memory": final_memory,
                "memory_increase": final_memory - initial_memory,
                "peak_increase": peak_memory - initial_memory,
                "time": elapsed,
                "memory_per_task": (peak_memory - initial_memory) / task_count
            }
            
            print(f"  Initial: {initial_memory:.1f}MB")
            print(f"  Peak: {peak_memory:.1f}MB (+{peak_memory - initial_memory:.1f}MB)")
            print(f"  Final: {final_memory:.1f}MB")
            print(f"  Memory per task: {results[task_count]['memory_per_task']:.3f}MB")
        
        self.results["memory"] = results
        return results
    
    def _generate_benchmark_tasks(self, count: int, complex_dependencies: bool = False) -> List[pm.CompilationTask]:
        """Generate tasks for benchmarking."""
        tasks = []
        task_types = list(pm.quantum_scheduler.TaskType)
        
        for i in range(count):
            task_type = task_types[i % len(task_types)]
            
            task = pm.CompilationTask(
                id=f"bench_task_{i:04d}",
                task_type=task_type,
                estimated_duration=1.0 + (i % 10) * 0.2,
                priority=1.0 + (i % 5) * 0.2,
                resource_requirements={
                    "cpu": 1.0 + (i % 4),
                    "memory": 512 + (i % 8) * 256,
                    "gpu": (i % 3) * 0.5
                }
            )
            
            # Add dependencies for complexity
            if complex_dependencies and i > 0:
                # Linear dependencies
                if i % 4 != 0:
                    task.dependencies.add(f"bench_task_{i-1:04d}")
                
                # Some cross-dependencies
                if i > 5 and i % 7 == 0:
                    task.dependencies.add(f"bench_task_{i-5:04d}")
                
                # Fan-out dependencies
                if i > 10 and i % 10 == 0:
                    task.dependencies.add(f"bench_task_{i-10:04d}")
            
            tasks.append(task)
        
        return tasks
    
    def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """Run all benchmarks and return comprehensive results."""
        print("QUANTUM-INSPIRED TASK SCHEDULER - COMPREHENSIVE BENCHMARK SUITE")
        print("="*80)
        
        start_time = time.time()
        
        # Run all benchmarks
        self.benchmark_scheduling_scalability()
        self.benchmark_caching_performance()
        self.benchmark_parallel_efficiency()
        self.benchmark_algorithm_convergence()
        self.benchmark_memory_usage()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUITE COMPLETED in {total_time:.1f}s")
        print(f"{'='*80}")
        
        # Generate summary report
        self._generate_summary_report()
        
        return self.results
    
    def _generate_summary_report(self):
        """Generate summary report of benchmark results."""
        print("\nBENCHMARK SUMMARY REPORT")
        print("-" * 40)
        
        # Scalability summary
        if "scalability" in self.results:
            print("\n📊 SCALABILITY:")
            for level, data in self.results["scalability"].items():
                sizes = list(data.keys())
                times = [data[size]["avg_time"] for size in sizes]
                print(f"  {level}: {times[0]:.3f}s → {times[-1]:.3f}s "
                      f"({sizes[0]} → {sizes[-1]} tasks)")
        
        # Caching summary
        if "caching" in self.results:
            print("\n⚡ CACHING:")
            for strategy, data in self.results["caching"].items():
                if "speedup" in data:
                    print(f"  {strategy}: {data['speedup']:.1f}x speedup")
        
        # Parallelization summary
        if "parallelization" in self.results:
            print("\n🔄 PARALLELIZATION:")
            max_workers = max(self.results["parallelization"].keys())
            max_speedup = self.results["parallelization"][max_workers]["speedup"]
            max_efficiency = self.results["parallelization"][max_workers]["efficiency"]
            print(f"  Best speedup: {max_speedup:.2f}x with {max_workers} workers")
            print(f"  Efficiency: {max_efficiency:.1%}")
        
        # Memory summary
        if "memory" in self.results:
            print("\n💾 MEMORY:")
            sizes = list(self.results["memory"].keys())
            max_size = max(sizes)
            memory_data = self.results["memory"][max_size]
            print(f"  Peak usage: {memory_data['peak_memory']:.1f}MB "
                  f"({memory_data['memory_per_task']:.3f}MB/task)")
        
        print()


class TestQuantumBenchmarks:
    """Test class for running benchmarks in pytest."""
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_run_scalability_benchmark(self):
        """Run scalability benchmark."""
        benchmark = QuantumSchedulerBenchmark()
        results = benchmark.benchmark_scheduling_scalability()
        
        # Verify results are reasonable
        assert len(results) > 0
        
        # Check that larger problems don't scale exponentially
        for level_data in results.values():
            sizes = sorted(level_data.keys())
            for i in range(1, len(sizes)):
                prev_size, curr_size = sizes[i-1], sizes[i]
                prev_time = level_data[prev_size]["avg_time"]
                curr_time = level_data[curr_size]["avg_time"]
                
                size_ratio = curr_size / prev_size
                time_ratio = curr_time / prev_time
                
                # Time should not grow worse than quadratically
                assert time_ratio < size_ratio ** 2 * 2  # 2x buffer for test variance
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_run_caching_benchmark(self):
        """Run caching benchmark."""
        benchmark = QuantumSchedulerBenchmark()
        results = benchmark.benchmark_caching_performance()
        
        # Verify caching provides speedup
        assert "memory_only" in results or "hybrid" in results
        
        for strategy, data in results.items():
            if strategy != "none":
                assert data["speedup"] > 2.0  # At least 2x speedup from caching
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_run_parallel_benchmark(self):
        """Run parallel efficiency benchmark."""
        benchmark = QuantumSchedulerBenchmark()
        results = benchmark.benchmark_parallel_efficiency()
        
        # Verify parallelization provides some benefit
        baseline_time = results[1]["time"]
        
        for workers, data in results.items():
            if workers > 1:
                # Should show some speedup (not necessarily linear due to coordination overhead)
                assert data["time"] <= baseline_time * 1.5  # Allow for some overhead
    
    @pytest.mark.benchmark
    def test_run_convergence_benchmark(self):
        """Run algorithm convergence benchmark."""
        benchmark = QuantumSchedulerBenchmark()
        results = benchmark.benchmark_algorithm_convergence()
        
        # Verify all configurations produce valid results
        for config, data in results.items():
            assert data["makespan"] > 0
            assert 0 < data["utilization"] <= 1
            assert data["iterations"] > 0
    
    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_comprehensive_benchmark_suite(self):
        """Run the complete benchmark suite."""
        benchmark = QuantumSchedulerBenchmark()
        results = benchmark.run_comprehensive_benchmark()
        
        # Verify all benchmark categories ran
        expected_categories = ["scalability", "caching", "parallelization", "convergence"]
        for category in expected_categories:
            assert category in results
        
        # Save results for analysis
        results_file = Path("benchmark_results.json")
        with open(results_file, "w") as f:
            # Convert to JSON-serializable format
            json_results = self._convert_to_json_serializable(results)
            json.dump(json_results, f, indent=2)
        
        print(f"\nBenchmark results saved to: {results_file}")
    
    def _convert_to_json_serializable(self, obj):
        """Convert benchmark results to JSON-serializable format."""
        if isinstance(obj, dict):
            return {str(k): self._convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_json_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return self._convert_to_json_serializable(obj.__dict__)
        else:
            return obj


if __name__ == "__main__":
    # Run benchmarks directly
    benchmark = QuantumSchedulerBenchmark()
    results = benchmark.run_comprehensive_benchmark()
    
    # Save results
    with open("quantum_benchmark_results.json", "w") as f:
        import json
        json.dump(results, f, indent=2, default=str)
    
    print("\nBenchmark results saved to quantum_benchmark_results.json")