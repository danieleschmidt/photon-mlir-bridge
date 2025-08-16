"""
Enterprise Performance Benchmarks for Photonic Computing Platform
Generation 3: Comprehensive performance testing with scalability analysis
"""

import pytest
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Any, Tuple
import tempfile
import shutil
from pathlib import Path
import json
import statistics
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Import components for benchmarking
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from photon_mlir.core import TargetConfig, Device, Precision, PhotonicTensor
from photon_mlir.compiler import PhotonicCompiler, CompiledPhotonicModel
from photon_mlir.high_performance_distributed_compiler import (
    create_distributed_compiler, benchmark_distributed_compilation,
    ComputeBackend, ScalingPolicy, OptimizationLevel
)
from photon_mlir.enterprise_monitoring_system import (
    create_monitoring_system, MetricType
)
from photon_mlir.advanced_quantum_error_correction import (
    benchmark_error_correction, create_quantum_error_corrector
)
from photon_mlir.logging_config import get_global_logger


@dataclass
class BenchmarkResult:
    """Represents a benchmark test result."""
    test_name: str
    duration_seconds: float
    throughput: float
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_rate: float
    additional_metrics: Dict[str, Any]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'duration_seconds': self.duration_seconds,
            'throughput': self.throughput,
            'memory_usage_mb': self.memory_usage_mb,
            'cpu_usage_percent': self.cpu_usage_percent,
            'success_rate': self.success_rate,
            'error_rate': self.error_rate,
            'additional_metrics': self.additional_metrics,
            'timestamp': self.timestamp
        }


class PerformanceBenchmarkSuite:
    """Comprehensive performance benchmark suite."""
    
    def __init__(self):
        self.logger = get_global_logger()
        self.temp_dir = None
        self.benchmark_results = []
        
    def setup(self):
        """Set up benchmark environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.logger.info(f"Benchmark environment set up in {self.temp_dir}")
        
    def teardown(self):
        """Clean up benchmark environment."""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            self.logger.info("Benchmark environment cleaned up")
    
    def create_test_models(self, count: int, size_variation: bool = True) -> List[Path]:
        """Create test models for benchmarking."""
        model_paths = []
        
        for i in range(count):
            if size_variation:
                # Vary model sizes for realistic testing
                base_size = 100 + (i * 50)
                content_size = base_size + np.random.randint(-20, 50)
            else:
                content_size = 100
            
            model_path = self.temp_dir / f"benchmark_model_{i}.onnx"
            
            # Create mock model content
            content = f"# Benchmark model {i}\n"
            content += "# Model metadata and structure\n"
            content += "# " + "X" * max(content_size, 50) + "\n"  # Variable content
            
            with open(model_path, 'w') as f:
                f.write(content)
            
            model_paths.append(model_path)
        
        return model_paths
    
    def measure_resource_usage(self) -> Dict[str, float]:
        """Measure current resource usage."""
        try:
            import psutil
            process = psutil.Process()
            
            return {
                'cpu_percent': process.cpu_percent(),
                'memory_mb': process.memory_info().rss / (1024 * 1024),
                'memory_percent': process.memory_percent()
            }
        except ImportError:
            return {
                'cpu_percent': 0.0,
                'memory_mb': 0.0,
                'memory_percent': 0.0
            }
    
    def benchmark_basic_compilation(self, model_count: int = 10) -> BenchmarkResult:
        """Benchmark basic compilation performance."""
        self.logger.info(f"Running basic compilation benchmark with {model_count} models")
        
        start_time = time.time()
        start_resources = self.measure_resource_usage()
        
        # Create test models
        model_paths = self.create_test_models(model_count)
        target_config = TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.INT8,
            array_size=(32, 32)
        )
        
        # Benchmark compilation
        compiler = PhotonicCompiler(target_config=target_config)
        
        successful_compilations = 0
        failed_compilations = 0
        compilation_times = []
        
        for model_path in model_paths:
            compilation_start = time.time()
            
            try:
                compiled_model = compiler.compile_onnx(str(model_path))
                compilation_end = time.time()
                compilation_times.append(compilation_end - compilation_start)
                successful_compilations += 1
                
                # Test simulation for successful compilations
                if hasattr(compiled_model, 'simulate'):
                    test_input = PhotonicTensor(np.random.randn(10, 10))
                    result = compiled_model.simulate(test_input)
                    assert isinstance(result, PhotonicTensor)
                
            except Exception as e:
                compilation_end = time.time()
                compilation_times.append(compilation_end - compilation_start)
                failed_compilations += 1
                # Expected for mock models
        
        end_time = time.time()
        end_resources = self.measure_resource_usage()
        
        # Calculate metrics
        total_duration = end_time - start_time
        throughput = model_count / total_duration  # models per second
        success_rate = successful_compilations / model_count
        error_rate = failed_compilations / model_count
        avg_memory_usage = (start_resources['memory_mb'] + end_resources['memory_mb']) / 2
        avg_cpu_usage = (start_resources['cpu_percent'] + end_resources['cpu_percent']) / 2
        
        result = BenchmarkResult(
            test_name="basic_compilation",
            duration_seconds=total_duration,
            throughput=throughput,
            memory_usage_mb=avg_memory_usage,
            cpu_usage_percent=avg_cpu_usage,
            success_rate=success_rate,
            error_rate=error_rate,
            additional_metrics={
                'model_count': model_count,
                'avg_compilation_time': statistics.mean(compilation_times) if compilation_times else 0.0,
                'min_compilation_time': min(compilation_times) if compilation_times else 0.0,
                'max_compilation_time': max(compilation_times) if compilation_times else 0.0,
                'successful_compilations': successful_compilations,
                'failed_compilations': failed_compilations
            },
            timestamp=datetime.now().isoformat()
        )
        
        self.benchmark_results.append(result)
        self.logger.info(f"Basic compilation benchmark completed: {throughput:.2f} models/sec")
        
        return result
    
    def benchmark_distributed_compilation(self, model_count: int = 20, 
                                        worker_counts: List[int] = None) -> Dict[str, BenchmarkResult]:
        """Benchmark distributed compilation with different worker counts."""
        if worker_counts is None:
            worker_counts = [1, 2, min(4, mp.cpu_count())]
        
        self.logger.info(f"Running distributed compilation benchmark with {model_count} models")
        
        results = {}
        
        # Create test models and configurations
        model_paths = [str(p) for p in self.create_test_models(model_count)]
        target_configs = [
            TargetConfig(
                device=Device.LIGHTMATTER_ENVISE,
                precision=Precision.INT8,
                array_size=(16, 16)
            ) for _ in range(model_count)
        ]
        
        for worker_count in worker_counts:
            self.logger.info(f"Testing with {worker_count} workers")
            
            start_time = time.time()
            start_resources = self.measure_resource_usage()
            
            # Create distributed compiler
            compiler = create_distributed_compiler(
                backend="cpu",
                workers=worker_count,
                scaling="fixed"  # Fixed scaling for consistent benchmarking
            )
            
            compiler.start()
            
            try:
                # Submit batch compilation
                job_ids = compiler.compile_batch(
                    model_paths,
                    target_configs,
                    optimization_level=OptimizationLevel.O1,
                    wait_for_completion=True
                )
                
                # Collect results
                successful_jobs = 0
                failed_jobs = 0
                
                for job_id in job_ids:
                    result = compiler.get_job_result(job_id)
                    if result:
                        if result.error:
                            failed_jobs += 1
                        else:
                            successful_jobs += 1
                
                end_time = time.time()
                end_resources = self.measure_resource_usage()
                
                # Get cluster statistics
                cluster_status = compiler.get_cluster_status()
                
                # Calculate metrics
                total_duration = end_time - start_time
                throughput = model_count / total_duration
                success_rate = successful_jobs / len(job_ids) if job_ids else 0.0
                error_rate = failed_jobs / len(job_ids) if job_ids else 0.0
                avg_memory_usage = (start_resources['memory_mb'] + end_resources['memory_mb']) / 2
                avg_cpu_usage = (start_resources['cpu_percent'] + end_resources['cpu_percent']) / 2
                
                result = BenchmarkResult(
                    test_name=f"distributed_compilation_{worker_count}_workers",
                    duration_seconds=total_duration,
                    throughput=throughput,
                    memory_usage_mb=avg_memory_usage,
                    cpu_usage_percent=avg_cpu_usage,
                    success_rate=success_rate,
                    error_rate=error_rate,
                    additional_metrics={
                        'worker_count': worker_count,
                        'model_count': model_count,
                        'successful_jobs': successful_jobs,
                        'failed_jobs': failed_jobs,
                        'cluster_metrics': cluster_status.get('cluster_metrics', {}),
                        'final_worker_count': cluster_status.get('total_workers', 0),
                        'total_queue_size': cluster_status.get('total_queue_size', 0)
                    },
                    timestamp=datetime.now().isoformat()
                )
                
                results[f"{worker_count}_workers"] = result
                self.benchmark_results.append(result)
                
                self.logger.info(f"Distributed compilation ({worker_count} workers): {throughput:.2f} models/sec")
                
            finally:
                compiler.stop()
        
        return results
    
    def benchmark_quantum_error_correction(self, state_count: int = 50,
                                         state_sizes: List[int] = None) -> Dict[str, BenchmarkResult]:
        """Benchmark quantum error correction performance."""
        if state_sizes is None:
            state_sizes = [8, 16, 32]  # Different quantum state sizes
        
        self.logger.info(f"Running quantum error correction benchmark with {state_count} states")
        
        results = {}
        strategies = ['surface_code', 'ml_adaptive']
        
        for state_size in state_sizes:
            # Create quantum states for testing
            quantum_states = [
                np.random.randn(state_size, state_size).astype(np.complex64)
                for _ in range(state_count)
            ]
            
            for strategy in strategies:
                test_name = f"error_correction_{strategy}_{state_size}x{state_size}"
                self.logger.info(f"Testing {test_name}")
                
                start_time = time.time()
                start_resources = self.measure_resource_usage()
                
                corrector = create_quantum_error_corrector(strategy)
                
                successful_corrections = 0
                failed_corrections = 0
                correction_times = []
                fidelities = []
                
                for state in quantum_states:
                    correction_start = time.time()
                    
                    try:
                        corrected_state, validation = corrector.correct_quantum_state(state)
                        correction_end = time.time()
                        
                        correction_times.append(correction_end - correction_start)
                        
                        if validation.is_valid:
                            successful_corrections += 1
                            fidelity = validation.metrics.get('current_fidelity', 0.0)
                            fidelities.append(fidelity)
                        else:
                            failed_corrections += 1
                            
                    except Exception as e:
                        correction_end = time.time()
                        correction_times.append(correction_end - correction_start)
                        failed_corrections += 1
                
                end_time = time.time()
                end_resources = self.measure_resource_usage()
                
                # Calculate metrics
                total_duration = end_time - start_time
                throughput = state_count / total_duration  # states per second
                success_rate = successful_corrections / state_count
                error_rate = failed_corrections / state_count
                avg_memory_usage = (start_resources['memory_mb'] + end_resources['memory_mb']) / 2
                avg_cpu_usage = (start_resources['cpu_percent'] + end_resources['cpu_percent']) / 2
                
                result = BenchmarkResult(
                    test_name=test_name,
                    duration_seconds=total_duration,
                    throughput=throughput,
                    memory_usage_mb=avg_memory_usage,
                    cpu_usage_percent=avg_cpu_usage,
                    success_rate=success_rate,
                    error_rate=error_rate,
                    additional_metrics={
                        'strategy': strategy,
                        'state_size': state_size,
                        'state_count': state_count,
                        'avg_correction_time': statistics.mean(correction_times) if correction_times else 0.0,
                        'avg_fidelity': statistics.mean(fidelities) if fidelities else 0.0,
                        'min_fidelity': min(fidelities) if fidelities else 0.0,
                        'max_fidelity': max(fidelities) if fidelities else 0.0,
                        'successful_corrections': successful_corrections,
                        'failed_corrections': failed_corrections
                    },
                    timestamp=datetime.now().isoformat()
                )
                
                results[test_name] = result
                self.benchmark_results.append(result)
                
                self.logger.info(f"{test_name}: {throughput:.2f} states/sec, {success_rate:.1%} success rate")
        
        return results
    
    def benchmark_monitoring_system(self, duration_seconds: int = 10) -> BenchmarkResult:
        """Benchmark monitoring system performance."""
        self.logger.info(f"Running monitoring system benchmark for {duration_seconds} seconds")
        
        start_time = time.time()
        start_resources = self.measure_resource_usage()
        
        # Create monitoring system with fast intervals for intensive testing
        monitoring = create_monitoring_system({
            'system_interval': 0.1,
            'photonic_interval': 0.05,
            'performance_interval': 0.05,
            'anomaly_sensitivity': 1.0
        })
        
        monitoring.start()
        
        # Let monitoring run
        time.sleep(duration_seconds)
        
        # Collect metrics
        all_metrics = monitoring.get_metrics(limit=10000)
        health_checks = []
        
        # Perform multiple health checks during benchmark
        for _ in range(10):
            health = monitoring.get_system_health()
            health_checks.append(health)
            time.sleep(0.1)
        
        monitoring.stop()
        
        end_time = time.time()
        end_resources = self.measure_resource_usage()
        
        # Calculate metrics
        total_duration = end_time - start_time
        metrics_collected = len(all_metrics)
        throughput = metrics_collected / total_duration  # metrics per second
        
        # Analyze metric types
        metric_types = {}
        for metric in all_metrics:
            metric_type = metric.metric_type.value
            metric_types[metric_type] = metric_types.get(metric_type, 0) + 1
        
        # Check health check consistency
        healthy_checks = sum(1 for h in health_checks if h.get('status') in ['HEALTHY', 'WARNING'])
        health_success_rate = healthy_checks / len(health_checks) if health_checks else 0.0
        
        avg_memory_usage = (start_resources['memory_mb'] + end_resources['memory_mb']) / 2
        avg_cpu_usage = (start_resources['cpu_percent'] + end_resources['cpu_percent']) / 2
        
        result = BenchmarkResult(
            test_name="monitoring_system",
            duration_seconds=total_duration,
            throughput=throughput,
            memory_usage_mb=avg_memory_usage,
            cpu_usage_percent=avg_cpu_usage,
            success_rate=health_success_rate,
            error_rate=1.0 - health_success_rate,
            additional_metrics={
                'metrics_collected': metrics_collected,
                'metric_types': metric_types,
                'health_checks_performed': len(health_checks),
                'healthy_checks': healthy_checks,
                'benchmark_duration': duration_seconds
            },
            timestamp=datetime.now().isoformat()
        )
        
        self.benchmark_results.append(result)
        self.logger.info(f"Monitoring system benchmark: {throughput:.2f} metrics/sec, {metrics_collected} total metrics")
        
        return result
    
    def benchmark_concurrent_operations(self, operation_count: int = 50) -> BenchmarkResult:
        """Benchmark concurrent operations across all systems."""
        self.logger.info(f"Running concurrent operations benchmark with {operation_count} operations")
        
        start_time = time.time()
        start_resources = self.measure_resource_usage()
        
        # Set up systems
        monitoring = create_monitoring_system({'system_interval': 0.5})
        monitoring.start()
        
        compiler = create_distributed_compiler(backend="cpu", workers=2)
        compiler.start()
        
        error_corrector = create_quantum_error_corrector("ml_adaptive")
        
        # Create test data
        model_paths = [str(p) for p in self.create_test_models(10)]
        target_configs = [
            TargetConfig(device=Device.LIGHTMATTER_ENVISE, precision=Precision.INT8, array_size=(8, 8))
            for _ in range(10)
        ]
        quantum_states = [
            np.random.randn(4, 4).astype(np.complex64) for _ in range(20)
        ]
        
        successful_operations = 0
        failed_operations = 0
        operation_results = []
        
        def run_compilation_job(model_path, config):
            try:
                job_id = compiler.submit_compilation_job(model_path, config)
                return ('compilation', job_id, None)
            except Exception as e:
                return ('compilation', None, str(e))
        
        def run_error_correction(state):
            try:
                corrected, validation = error_corrector.correct_quantum_state(state)
                return ('error_correction', corrected is not None, None)
            except Exception as e:
                return ('error_correction', False, str(e))
        
        def check_monitoring():
            try:
                health = monitoring.get_system_health()
                return ('monitoring', health is not None, None)
            except Exception as e:
                return ('monitoring', False, str(e))
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            
            # Submit compilation jobs
            for i in range(min(operation_count // 3, len(model_paths))):
                future = executor.submit(run_compilation_job, 
                                       model_paths[i % len(model_paths)], 
                                       target_configs[i % len(target_configs)])
                futures.append(future)
            
            # Submit error correction tasks
            for i in range(min(operation_count // 3, len(quantum_states))):
                future = executor.submit(run_error_correction, 
                                       quantum_states[i % len(quantum_states)])
                futures.append(future)
            
            # Submit monitoring checks
            for i in range(operation_count - len(futures)):
                future = executor.submit(check_monitoring)
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result(timeout=30.0)
                    operation_results.append(result)
                    
                    if result[2] is None:  # No error
                        successful_operations += 1
                    else:
                        failed_operations += 1
                        
                except Exception as e:
                    operation_results.append(('unknown', False, str(e)))
                    failed_operations += 1
        
        # Clean up
        monitoring.stop()
        compiler.stop()
        
        end_time = time.time()
        end_resources = self.measure_resource_usage()
        
        # Calculate metrics
        total_duration = end_time - start_time
        throughput = len(operation_results) / total_duration  # operations per second
        success_rate = successful_operations / len(operation_results) if operation_results else 0.0
        error_rate = failed_operations / len(operation_results) if operation_results else 0.0
        avg_memory_usage = (start_resources['memory_mb'] + end_resources['memory_mb']) / 2
        avg_cpu_usage = (start_resources['cpu_percent'] + end_resources['cpu_percent']) / 2
        
        # Analyze operation types
        operation_types = {}
        for op_type, success, error in operation_results:
            if op_type not in operation_types:
                operation_types[op_type] = {'total': 0, 'successful': 0, 'failed': 0}
            
            operation_types[op_type]['total'] += 1
            if error is None:
                operation_types[op_type]['successful'] += 1
            else:
                operation_types[op_type]['failed'] += 1
        
        result = BenchmarkResult(
            test_name="concurrent_operations",
            duration_seconds=total_duration,
            throughput=throughput,
            memory_usage_mb=avg_memory_usage,
            cpu_usage_percent=avg_cpu_usage,
            success_rate=success_rate,
            error_rate=error_rate,
            additional_metrics={
                'total_operations': len(operation_results),
                'successful_operations': successful_operations,
                'failed_operations': failed_operations,
                'operation_types': operation_types,
                'requested_operations': operation_count
            },
            timestamp=datetime.now().isoformat()
        )
        
        self.benchmark_results.append(result)
        self.logger.info(f"Concurrent operations benchmark: {throughput:.2f} ops/sec, {success_rate:.1%} success rate")
        
        return result
    
    def run_comprehensive_benchmark_suite(self) -> Dict[str, Any]:
        """Run all benchmarks and generate comprehensive report."""
        self.logger.info("Starting comprehensive benchmark suite")
        suite_start_time = time.time()
        
        try:
            # Run all benchmarks
            basic_result = self.benchmark_basic_compilation(model_count=5)
            distributed_results = self.benchmark_distributed_compilation(model_count=10, worker_counts=[1, 2])
            quantum_results = self.benchmark_quantum_error_correction(state_count=20, state_sizes=[8, 16])
            monitoring_result = self.benchmark_monitoring_system(duration_seconds=5)
            concurrent_result = self.benchmark_concurrent_operations(operation_count=20)
            
            suite_end_time = time.time()
            suite_duration = suite_end_time - suite_start_time
            
            # Generate comprehensive report
            report = {
                'benchmark_suite': {
                    'start_time': datetime.fromtimestamp(suite_start_time).isoformat(),
                    'end_time': datetime.fromtimestamp(suite_end_time).isoformat(),
                    'total_duration_seconds': suite_duration,
                    'total_tests_run': len(self.benchmark_results)
                },
                'results': {
                    'basic_compilation': basic_result.to_dict(),
                    'distributed_compilation': {k: v.to_dict() for k, v in distributed_results.items()},
                    'quantum_error_correction': {k: v.to_dict() for k, v in quantum_results.items()},
                    'monitoring_system': monitoring_result.to_dict(),
                    'concurrent_operations': concurrent_result.to_dict()
                },
                'summary': {
                    'avg_throughput': statistics.mean([r.throughput for r in self.benchmark_results]),
                    'avg_success_rate': statistics.mean([r.success_rate for r in self.benchmark_results]),
                    'avg_error_rate': statistics.mean([r.error_rate for r in self.benchmark_results]),
                    'avg_memory_usage_mb': statistics.mean([r.memory_usage_mb for r in self.benchmark_results]),
                    'avg_cpu_usage_percent': statistics.mean([r.cpu_usage_percent for r in self.benchmark_results]),
                    'total_benchmark_time': suite_duration
                },
                'performance_analysis': self._analyze_performance_trends(),
                'recommendations': self._generate_performance_recommendations()
            }
            
            self.logger.info(f"Comprehensive benchmark suite completed in {suite_duration:.2f} seconds")
            return report
            
        except Exception as e:
            self.logger.error(f"Benchmark suite failed: {e}")
            return {
                'error': str(e),
                'partial_results': [r.to_dict() for r in self.benchmark_results],
                'timestamp': datetime.now().isoformat()
            }
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends across benchmarks."""
        if not self.benchmark_results:
            return {}
        
        # Throughput analysis
        throughputs = [r.throughput for r in self.benchmark_results if r.throughput > 0]
        memory_usages = [r.memory_usage_mb for r in self.benchmark_results]
        cpu_usages = [r.cpu_usage_percent for r in self.benchmark_results]
        success_rates = [r.success_rate for r in self.benchmark_results]
        
        return {
            'throughput_stats': {
                'min': min(throughputs) if throughputs else 0,
                'max': max(throughputs) if throughputs else 0,
                'mean': statistics.mean(throughputs) if throughputs else 0,
                'median': statistics.median(throughputs) if throughputs else 0,
                'stdev': statistics.stdev(throughputs) if len(throughputs) > 1 else 0
            },
            'memory_usage_stats': {
                'min': min(memory_usages) if memory_usages else 0,
                'max': max(memory_usages) if memory_usages else 0,
                'mean': statistics.mean(memory_usages) if memory_usages else 0
            },
            'cpu_usage_stats': {
                'min': min(cpu_usages) if cpu_usages else 0,
                'max': max(cpu_usages) if cpu_usages else 0,
                'mean': statistics.mean(cpu_usages) if cpu_usages else 0
            },
            'reliability_stats': {
                'min_success_rate': min(success_rates) if success_rates else 0,
                'max_success_rate': max(success_rates) if success_rates else 0,
                'avg_success_rate': statistics.mean(success_rates) if success_rates else 0
            }
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        if not self.benchmark_results:
            return ["No benchmark data available for recommendations"]
        
        # Analyze success rates
        avg_success_rate = statistics.mean([r.success_rate for r in self.benchmark_results])
        if avg_success_rate < 0.8:
            recommendations.append(
                f"Low success rate ({avg_success_rate:.1%}). Consider improving error handling and validation."
            )
        
        # Analyze memory usage
        avg_memory = statistics.mean([r.memory_usage_mb for r in self.benchmark_results])
        if avg_memory > 1000:  # > 1GB
            recommendations.append(
                f"High memory usage ({avg_memory:.0f}MB). Consider implementing memory optimization strategies."
            )
        
        # Analyze CPU usage
        avg_cpu = statistics.mean([r.cpu_usage_percent for r in self.benchmark_results])
        if avg_cpu > 80:
            recommendations.append(
                f"High CPU usage ({avg_cpu:.1f}%). Consider optimizing computational algorithms or adding more workers."
            )
        
        # Analyze throughput variance
        throughputs = [r.throughput for r in self.benchmark_results if r.throughput > 0]
        if len(throughputs) > 1:
            throughput_cv = statistics.stdev(throughputs) / statistics.mean(throughputs)
            if throughput_cv > 0.5:  # High coefficient of variation
                recommendations.append(
                    "High throughput variance detected. Consider load balancing improvements."
                )
        
        # Check for distributed vs single-threaded performance
        distributed_results = [r for r in self.benchmark_results if 'distributed' in r.test_name]
        single_results = [r for r in self.benchmark_results if 'distributed' not in r.test_name]
        
        if distributed_results and single_results:
            avg_distributed_throughput = statistics.mean([r.throughput for r in distributed_results])
            avg_single_throughput = statistics.mean([r.throughput for r in single_results])
            
            if avg_distributed_throughput < avg_single_throughput * 1.5:
                recommendations.append(
                    "Distributed processing shows limited scaling benefits. Review task distribution and worker efficiency."
                )
        
        if not recommendations:
            recommendations.append("Performance benchmarks show good results across all tested scenarios.")
        
        return recommendations


class TestEnterprisePerformanceBenchmarks:
    """Test class for enterprise performance benchmarks."""
    
    @pytest.fixture(autouse=True)
    def setup_benchmark_suite(self):
        """Set up benchmark suite for testing."""
        self.benchmark_suite = PerformanceBenchmarkSuite()
        self.benchmark_suite.setup()
        
        yield
        
        self.benchmark_suite.teardown()
    
    def test_basic_compilation_benchmark(self):
        """Test basic compilation benchmark."""
        result = self.benchmark_suite.benchmark_basic_compilation(model_count=3)
        
        assert isinstance(result, BenchmarkResult)
        assert result.test_name == "basic_compilation"
        assert result.duration_seconds > 0
        assert result.throughput >= 0
        assert 0 <= result.success_rate <= 1
        assert 0 <= result.error_rate <= 1
        assert result.success_rate + result.error_rate == 1.0  # Should sum to 1
        
        # Check additional metrics
        assert 'model_count' in result.additional_metrics
        assert result.additional_metrics['model_count'] == 3
    
    def test_distributed_compilation_benchmark(self):
        """Test distributed compilation benchmark."""
        results = self.benchmark_suite.benchmark_distributed_compilation(
            model_count=4, worker_counts=[1, 2]
        )
        
        assert len(results) == 2
        assert "1_workers" in results
        assert "2_workers" in results
        
        for worker_count, result in results.items():
            assert isinstance(result, BenchmarkResult)
            assert "distributed_compilation" in result.test_name
            assert result.duration_seconds > 0
            assert result.throughput >= 0
            
            # Check scaling metrics
            metrics = result.additional_metrics
            assert 'worker_count' in metrics
            assert 'successful_jobs' in metrics
            assert 'failed_jobs' in metrics
    
    def test_quantum_error_correction_benchmark(self):
        """Test quantum error correction benchmark."""
        results = self.benchmark_suite.benchmark_quantum_error_correction(
            state_count=10, state_sizes=[4, 8]
        )
        
        assert len(results) >= 2  # At least 2 strategies × 2 sizes
        
        for test_name, result in results.items():
            assert isinstance(result, BenchmarkResult)
            assert "error_correction" in result.test_name
            assert result.duration_seconds > 0
            assert result.throughput >= 0
            
            # Check quantum-specific metrics
            metrics = result.additional_metrics
            assert 'strategy' in metrics
            assert 'state_size' in metrics
            assert 'avg_fidelity' in metrics
            assert 0 <= metrics['avg_fidelity'] <= 1
    
    def test_monitoring_system_benchmark(self):
        """Test monitoring system benchmark."""
        result = self.benchmark_suite.benchmark_monitoring_system(duration_seconds=3)
        
        assert isinstance(result, BenchmarkResult)
        assert result.test_name == "monitoring_system"
        assert result.duration_seconds >= 3  # Should be at least the requested duration
        assert result.throughput > 0  # Should collect metrics
        
        # Check monitoring-specific metrics
        metrics = result.additional_metrics
        assert 'metrics_collected' in metrics
        assert 'metric_types' in metrics
        assert metrics['metrics_collected'] > 0
    
    def test_concurrent_operations_benchmark(self):
        """Test concurrent operations benchmark."""
        result = self.benchmark_suite.benchmark_concurrent_operations(operation_count=12)
        
        assert isinstance(result, BenchmarkResult)
        assert result.test_name == "concurrent_operations"
        assert result.duration_seconds > 0
        assert result.throughput > 0
        
        # Check concurrent operation metrics
        metrics = result.additional_metrics
        assert 'total_operations' in metrics
        assert 'operation_types' in metrics
        assert metrics['total_operations'] > 0
        
        # Verify operation types were tracked
        op_types = metrics['operation_types']
        assert isinstance(op_types, dict)
    
    def test_comprehensive_benchmark_suite(self):
        """Test comprehensive benchmark suite."""
        report = self.benchmark_suite.run_comprehensive_benchmark_suite()
        
        assert 'benchmark_suite' in report
        assert 'results' in report
        assert 'summary' in report
        assert 'performance_analysis' in report
        assert 'recommendations' in report
        
        # Check benchmark suite metadata
        suite_info = report['benchmark_suite']
        assert 'start_time' in suite_info
        assert 'end_time' in suite_info
        assert 'total_duration_seconds' in suite_info
        assert suite_info['total_duration_seconds'] > 0
        
        # Check results structure
        results = report['results']
        assert 'basic_compilation' in results
        assert 'distributed_compilation' in results
        assert 'quantum_error_correction' in results
        assert 'monitoring_system' in results
        assert 'concurrent_operations' in results
        
        # Check summary statistics
        summary = report['summary']
        assert 'avg_throughput' in summary
        assert 'avg_success_rate' in summary
        assert summary['avg_success_rate'] >= 0
        assert summary['avg_success_rate'] <= 1
        
        # Check recommendations
        recommendations = report['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
    
    def test_performance_analysis(self):
        """Test performance analysis functionality."""
        # Run a few benchmarks to generate data
        self.benchmark_suite.benchmark_basic_compilation(model_count=2)
        self.benchmark_suite.benchmark_monitoring_system(duration_seconds=2)
        
        analysis = self.benchmark_suite._analyze_performance_trends()
        
        assert 'throughput_stats' in analysis
        assert 'memory_usage_stats' in analysis
        assert 'cpu_usage_stats' in analysis
        assert 'reliability_stats' in analysis
        
        # Check throughput statistics
        throughput_stats = analysis['throughput_stats']
        assert 'min' in throughput_stats
        assert 'max' in throughput_stats
        assert 'mean' in throughput_stats
        assert throughput_stats['min'] >= 0
        assert throughput_stats['max'] >= throughput_stats['min']
    
    def test_performance_recommendations(self):
        """Test performance recommendation generation."""
        # Run benchmarks to generate data
        self.benchmark_suite.benchmark_basic_compilation(model_count=2)
        
        recommendations = self.benchmark_suite._generate_performance_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # All recommendations should be strings
        for recommendation in recommendations:
            assert isinstance(recommendation, str)
            assert len(recommendation) > 0


if __name__ == "__main__":
    # Run performance benchmarks
    pytest.main([__file__, "-v", "-s", "--tb=short"])
