"""
Unit tests for thermal-aware quantum optimization functionality.

Tests cover:
- Thermal-aware optimization algorithms
- Photonic constraint handling
- Error recovery and validation
- Performance benchmarking
- Statistical validation
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

import sys
from pathlib import Path

# Add the python directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "python"))

from photon_mlir.quantum_scheduler import (
    CompilationTask, TaskType, SchedulingState, QuantumTaskPlanner
)
from photon_mlir.thermal_optimization import (
    ThermalAwareOptimizer, ThermalAwareBenchmark, ThermalModel, 
    CoolingStrategy, ThermalConstraints, PhotonicDevice
)
from photon_mlir.robust_thermal_scheduler import (
    RobustThermalScheduler, ErrorSeverity, RecoveryStrategy, 
    ErrorHandler, HealthMonitor, CircuitBreaker
)


class TestThermalAwareOptimizer:
    """Test cases for thermal-aware optimization."""
    
    def test_optimizer_initialization(self):
        """Test optimizer initialization with different configurations."""
        # Test default initialization
        optimizer = ThermalAwareOptimizer()
        assert optimizer.thermal_model == ThermalModel.ARRHENIUS_BASED
        assert optimizer.cooling_strategy == CoolingStrategy.ADAPTIVE
        assert isinstance(optimizer.constraints, ThermalConstraints)
        assert isinstance(optimizer.device, PhotonicDevice)
    
    def test_custom_constraints(self):
        """Test optimizer with custom thermal constraints."""
        custom_constraints = ThermalConstraints(
            max_device_temperature=70.0,
            max_thermal_gradient=3.0,
            thermal_time_constant=15.0
        )
        
        optimizer = ThermalAwareOptimizer(constraints=custom_constraints)
        assert optimizer.constraints.max_device_temperature == 70.0
        assert optimizer.constraints.max_thermal_gradient == 3.0
        assert optimizer.constraints.thermal_time_constant == 15.0
    
    def test_custom_device(self):
        """Test optimizer with custom photonic device."""
        custom_device = PhotonicDevice(
            device_id="test_chip",
            area_mm2=16.0,
            num_phase_shifters=32,
            wavelength_channels=4
        )
        
        optimizer = ThermalAwareOptimizer(device=custom_device)
        assert optimizer.device.device_id == "test_chip"
        assert optimizer.device.area_mm2 == 16.0
        assert optimizer.device.num_phase_shifters == 32
    
    def test_thermal_optimization_basic(self):
        """Test basic thermal optimization functionality."""
        # Create test tasks
        planner = QuantumTaskPlanner()
        tasks = planner.create_compilation_plan({"test": "config"})
        
        # Create initial schedule
        from photon_mlir.quantum_scheduler import QuantumInspiredScheduler
        scheduler = QuantumInspiredScheduler(population_size=10, max_iterations=50)
        initial_schedule = scheduler.schedule_tasks(tasks)
        
        # Apply thermal optimization
        optimizer = ThermalAwareOptimizer()
        optimized_schedule = optimizer.optimize_thermal_schedule(
            initial_schedule, max_iterations=100
        )
        
        # Verify optimization completed
        assert optimized_schedule is not None
        assert hasattr(optimized_schedule, 'makespan')
        assert hasattr(optimized_schedule, 'thermal_efficiency')
        assert optimized_schedule.makespan > 0
    
    def test_thermal_metrics_calculation(self):
        """Test thermal metrics calculation."""
        optimizer = ThermalAwareOptimizer()
        
        # Create test tasks with thermal properties
        tasks = [
            CompilationTask(
                id="test_task_1",
                task_type=TaskType.PHASE_OPTIMIZATION,
                estimated_duration=2.0,
                thermal_load=15.0,
                phase_shifts_required=32
            ),
            CompilationTask(
                id="test_task_2",
                task_type=TaskType.THERMAL_COMPENSATION,
                estimated_duration=1.5,
                thermal_load=5.0,
                calibration_frequency=10.0
            )
        ]
        
        # Create simple schedule
        schedule = SchedulingState(
            tasks=tasks,
            schedule={0: ["test_task_1"], 1: ["test_task_2"]},
            makespan=3.5,
            resource_utilization=0.8
        )
        
        # Calculate thermal metrics
        optimizer._calculate_thermal_metrics(schedule)
        
        # Verify metrics were calculated
        assert hasattr(schedule, 'thermal_efficiency')
        assert hasattr(schedule, 'max_device_temperature')
        assert hasattr(schedule, 'phase_stability')
        assert schedule.thermal_efficiency >= 0
        assert schedule.max_device_temperature > 0
    
    def test_performance_report(self):
        """Test thermal performance report generation."""
        optimizer = ThermalAwareOptimizer()
        
        # Simulate some optimization runs
        optimizer.temperature_history = [45.0, 48.0, 46.0, 47.0]
        optimizer.thermal_efficiency_scores = [0.8, 0.85, 0.82, 0.87]
        
        report = optimizer.get_thermal_performance_report()
        
        assert "max_temperature" in report
        assert "avg_temperature" in report
        assert "avg_thermal_efficiency" in report
        assert "quantum_thermal_integration_score" in report
        assert "photonic_awareness_factor" in report
        
        assert report["max_temperature"] == 48.0
        assert abs(report["avg_temperature"] - 46.5) < 0.1
        assert report["avg_thermal_efficiency"] > 0.8


class TestThermalAwareBenchmark:
    """Test cases for thermal-aware benchmarking."""
    
    def test_benchmark_initialization(self):
        """Test benchmark initialization."""
        benchmark = ThermalAwareBenchmark()
        assert isinstance(benchmark.baseline_results, list)
        assert isinstance(benchmark.thermal_aware_results, list)
        assert len(benchmark.baseline_results) == 0
        assert len(benchmark.thermal_aware_results) == 0
    
    def test_baseline_optimization(self):
        """Test baseline optimization run."""
        benchmark = ThermalAwareBenchmark()
        
        # Create test tasks
        planner = QuantumTaskPlanner()
        tasks = planner.create_compilation_plan({"test": "small"})[:5]  # Small subset
        
        # Run baseline optimization
        result = benchmark._run_baseline_optimization(tasks)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "makespan" in result
        assert "resource_utilization" in result
        assert "optimization_time" in result
        assert "thermal_efficiency" in result
        
        assert result["makespan"] > 0
        assert 0 <= result["resource_utilization"] <= 1
        assert result["optimization_time"] > 0
    
    def test_thermal_optimization(self):
        """Test thermal-aware optimization run."""
        benchmark = ThermalAwareBenchmark()
        
        # Create test tasks
        planner = QuantumTaskPlanner()
        tasks = planner.create_compilation_plan({"test": "small"})[:5]  # Small subset
        
        # Run thermal optimization
        result = benchmark._run_thermal_optimization(tasks)
        
        # Verify result structure
        assert isinstance(result, dict)
        assert "makespan" in result
        assert "thermal_efficiency" in result
        assert "max_temperature" in result
        
        assert result["makespan"] > 0
        assert 0 <= result["thermal_efficiency"] <= 1
        assert result["max_temperature"] > 0
    
    def test_comparative_study_small(self):
        """Test small comparative study."""
        benchmark = ThermalAwareBenchmark()
        
        # Create small test dataset
        planner = QuantumTaskPlanner()
        task_sets = [
            planner.create_compilation_plan({"complexity": "small"})[:3]
            for _ in range(2)  # 2 task sets
        ]
        
        # Run comparative study with minimal iterations
        results = benchmark.run_comparative_study(task_sets, iterations=2)
        
        # Verify results structure
        assert "experiment_summary" in results
        assert "detailed_comparison" in results
        assert "research_contribution" in results
        
        summary = results["experiment_summary"]
        assert summary["total_runs"] == 4  # 2 task sets * 2 iterations
        assert summary["task_sets_tested"] == 2
    
    def test_statistical_significance(self):
        """Test statistical significance calculation."""
        benchmark = ThermalAwareBenchmark()
        
        # Test data
        baseline_values = [1.0, 1.1, 1.05, 1.2, 1.15]
        thermal_values = [0.9, 0.95, 0.92, 1.0, 0.98]
        
        p_value = benchmark._calculate_significance(baseline_values, thermal_values)
        
        assert 0 <= p_value <= 1
        assert isinstance(p_value, float)


class TestRobustThermalScheduler:
    """Test cases for robust thermal scheduler."""
    
    def test_scheduler_initialization(self):
        """Test robust scheduler initialization."""
        scheduler = RobustThermalScheduler()
        
        assert scheduler.thermal_model == ThermalModel.ARRHENIUS_BASED
        assert scheduler.cooling_strategy == CoolingStrategy.ADAPTIVE
        assert scheduler.max_retries == 3
        assert scheduler.timeout_seconds == 300.0
        assert scheduler.health_monitor is not None
        assert scheduler.circuit_breaker is not None
    
    def test_scheduler_with_custom_config(self):
        """Test scheduler with custom configuration."""
        scheduler = RobustThermalScheduler(
            thermal_model=ThermalModel.SIMPLE_LINEAR,
            cooling_strategy=CoolingStrategy.PASSIVE,
            max_retries=5,
            timeout_seconds=600.0,
            enable_monitoring=False,
            enable_circuit_breaker=False
        )
        
        assert scheduler.thermal_model == ThermalModel.SIMPLE_LINEAR
        assert scheduler.cooling_strategy == CoolingStrategy.PASSIVE
        assert scheduler.max_retries == 5
        assert scheduler.timeout_seconds == 600.0
        assert scheduler.health_monitor is None
        assert scheduler.circuit_breaker is None
    
    def test_robust_scheduling_success(self):
        """Test successful robust scheduling."""
        scheduler = RobustThermalScheduler(
            timeout_seconds=120.0,  # Shorter timeout for tests
            enable_monitoring=False,  # Disable for simpler testing
            enable_circuit_breaker=False
        )
        
        # Create test tasks
        planner = QuantumTaskPlanner()
        tasks = planner.create_compilation_plan({"test": "small"})[:5]
        
        # Schedule tasks
        result = scheduler.schedule_tasks_robust(tasks)
        
        # Verify result
        assert result is not None
        assert hasattr(result, 'makespan')
        assert hasattr(result, 'schedule')
        assert result.makespan > 0
        assert len(result.schedule) > 0
        assert scheduler.success_count == 1
        assert scheduler.failure_count == 0
    
    def test_input_validation_error(self):
        """Test handling of input validation errors."""
        scheduler = RobustThermalScheduler(enable_monitoring=False, enable_circuit_breaker=False)
        
        # Test with invalid tasks (empty list)
        with pytest.raises(RuntimeError):
            scheduler.schedule_tasks_robust([])
    
    def test_failsafe_schedule_creation(self):
        """Test failsafe schedule creation."""
        scheduler = RobustThermalScheduler()
        
        # Create test tasks
        tasks = [
            CompilationTask("task1", TaskType.GRAPH_LOWERING, set(), 1.0),
            CompilationTask("task2", TaskType.PHOTONIC_OPTIMIZATION, {"task1"}, 2.0),
            CompilationTask("task3", TaskType.CODE_GENERATION, {"task2"}, 1.0)
        ]
        
        # Create failsafe schedule
        failsafe = scheduler._create_failsafe_schedule(tasks)
        
        # Verify failsafe schedule
        assert failsafe is not None
        assert len(failsafe.schedule) > 0
        assert failsafe.makespan > 0
        
        # Verify all tasks are scheduled
        scheduled_tasks = set()
        for task_list in failsafe.schedule.values():
            scheduled_tasks.update(task_list)
        
        assert len(scheduled_tasks) == 3
        assert "task1" in scheduled_tasks
        assert "task2" in scheduled_tasks
        assert "task3" in scheduled_tasks
    
    def test_system_status(self):
        """Test system status reporting."""
        scheduler = RobustThermalScheduler()
        
        # Get initial status
        status = scheduler.get_system_status()
        
        # Verify status structure
        assert "scheduler_config" in status
        assert "performance" in status
        assert "error_statistics" in status
        assert "circuit_breaker" in status
        
        config = status["scheduler_config"]
        assert config["thermal_model"] == "arrhenius_based"
        assert config["cooling_strategy"] == "adaptive"
        assert config["max_retries"] == 3
        
        perf = status["performance"]
        assert "success_count" in perf
        assert "failure_count" in perf
        assert "success_rate" in perf


class TestErrorHandling:
    """Test cases for error handling components."""
    
    def test_error_handler_initialization(self):
        """Test error handler initialization."""
        handler = ErrorHandler()
        
        assert handler.max_error_history == 1000
        assert len(handler.error_history) == 0
        assert isinstance(handler.recovery_strategies, dict)
        assert isinstance(handler.error_count_by_type, dict)
    
    def test_error_handling(self):
        """Test error handling and context creation."""
        handler = ErrorHandler()
        
        # Simulate an error
        try:
            raise ValueError("Test error message")
        except Exception as e:
            error_context = handler.handle_error(
                e, "test_component", "test_operation", 
                {"test_data": "test_value"}
            )
        
        # Verify error context
        assert error_context.error_message == "Test error message"
        assert error_context.component == "test_component"
        assert error_context.operation == "test_operation"
        assert error_context.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        assert error_context.context_data["test_data"] == "test_value"
        
        # Verify error was recorded
        assert len(handler.error_history) == 1
        assert "ValueError" in handler.error_count_by_type
        assert handler.error_count_by_type["ValueError"] == 1
    
    def test_recovery_strategy_selection(self):
        """Test recovery strategy selection."""
        handler = ErrorHandler()
        
        # Test different severity levels
        from photon_mlir.robust_thermal_scheduler import ErrorContext
        
        critical_error = ErrorContext(
            error_id="test1",
            timestamp=time.time(),
            severity=ErrorSeverity.CRITICAL,
            component="test",
            operation="test",
            error_message="Critical error"
        )
        
        high_error = ErrorContext(
            error_id="test2",
            timestamp=time.time(),
            severity=ErrorSeverity.HIGH,
            component="test",
            operation="test",
            error_message="High error"
        )
        
        medium_error = ErrorContext(
            error_id="test3",
            timestamp=time.time(),
            severity=ErrorSeverity.MEDIUM,
            component="test",
            operation="test",
            error_message="Medium error"
        )
        
        # Test strategy selection
        assert handler.get_recovery_strategy(critical_error) == RecoveryStrategy.FAIL_SAFE
        assert handler.get_recovery_strategy(high_error) == RecoveryStrategy.FALLBACK
        assert handler.get_recovery_strategy(medium_error) == RecoveryStrategy.RETRY
    
    def test_error_statistics(self):
        """Test error statistics generation."""
        handler = ErrorHandler()
        
        # Generate some test errors
        for i in range(5):
            try:
                if i % 2 == 0:
                    raise ValueError(f"Test error {i}")
                else:
                    raise RuntimeError(f"Test error {i}")
            except Exception as e:
                handler.handle_error(e, "test", "test")
        
        stats = handler.get_error_statistics()
        
        assert stats["total_errors"] == 5
        assert stats["errors_by_type"]["ValueError"] == 3
        assert stats["errors_by_type"]["RuntimeError"] == 2
        assert stats["most_common_error"] == "ValueError"


class TestCircuitBreaker:
    """Test cases for circuit breaker functionality."""
    
    def test_circuit_breaker_initialization(self):
        """Test circuit breaker initialization."""
        cb = CircuitBreaker(failure_threshold=3, reset_timeout=30.0)
        
        assert cb.failure_threshold == 3
        assert cb.reset_timeout == 30.0
        assert cb.failure_count == 0
        assert cb.state == "closed"
    
    def test_circuit_breaker_success(self):
        """Test circuit breaker with successful operations."""
        cb = CircuitBreaker()
        
        def successful_operation():
            return "success"
        
        # Multiple successful calls should work
        for _ in range(10):
            result = cb.call(successful_operation)
            assert result == "success"
        
        assert cb.state == "closed"
        assert cb.failure_count == 0
    
    def test_circuit_breaker_failure(self):
        """Test circuit breaker with failing operations."""
        cb = CircuitBreaker(failure_threshold=3)
        
        def failing_operation():
            raise RuntimeError("Operation failed")
        
        # First few failures should pass through
        for i in range(2):
            with pytest.raises(RuntimeError):
                cb.call(failing_operation)
            assert cb.state == "closed"
        
        # Threshold failure should open circuit
        with pytest.raises(RuntimeError):
            cb.call(failing_operation)
        
        assert cb.state == "open"
        
        # Subsequent calls should fail immediately
        with pytest.raises(RuntimeError, match="Circuit breaker is open"):
            cb.call(failing_operation)


class TestHealthMonitoring:
    """Test cases for health monitoring."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor(monitoring_interval=1.0)
        
        assert monitor.monitoring_interval == 1.0
        assert not monitor.is_monitoring
        assert monitor.monitor_thread is None
        assert len(monitor.alert_callbacks) == 0
    
    def test_health_metrics_update(self):
        """Test health metrics updates."""
        monitor = HealthMonitor()
        
        # Update metrics
        monitor.update_metrics(
            cpu_usage=50.0,
            memory_usage=30.0,
            error_rate=0.1,
            success_rate=0.9
        )
        
        # Verify metrics
        metrics = monitor.health_metrics
        assert metrics.cpu_usage == 50.0
        assert metrics.memory_usage == 30.0
        assert metrics.error_rate == 0.1
        assert metrics.success_rate == 0.9
        assert 0 <= metrics.health_score <= 1
    
    def test_health_status(self):
        """Test health status reporting."""
        monitor = HealthMonitor()
        
        # Update with good metrics
        monitor.update_metrics(
            cpu_usage=20.0,
            memory_usage=30.0,
            error_rate=0.01,
            success_rate=0.99
        )
        
        status = monitor.get_health_status()
        
        assert "health_score" in status
        assert "cpu_usage" in status
        assert "memory_usage" in status
        assert "status" in status
        
        assert status["cpu_usage"] == 20.0
        assert status["memory_usage"] == 30.0
        assert status["status"] in ["excellent", "good", "fair", "poor", "critical"]
    
    def test_alert_callback(self):
        """Test health alert callbacks."""
        monitor = HealthMonitor()
        alerts_received = []
        
        def alert_callback(metrics):
            alerts_received.append(metrics.health_score)
        
        monitor.add_alert_callback(alert_callback)
        
        # Update with poor health metrics (should trigger alert)
        monitor.update_metrics(
            cpu_usage=95.0,
            memory_usage=90.0,
            error_rate=0.5,
            success_rate=0.5
        )
        
        # Should have triggered alert callback
        assert len(alerts_received) > 0
        assert alerts_received[0] < 0.7  # Health threshold


# Integration test
class TestThermalOptimizationIntegration:
    """Integration tests for thermal optimization system."""
    
    def test_end_to_end_optimization(self):
        """Test complete end-to-end thermal optimization."""
        # Create realistic workload
        planner = QuantumTaskPlanner()
        tasks = planner.create_compilation_plan({"layers": 4, "complexity": "medium"})
        
        # Test with robust scheduler
        scheduler = RobustThermalScheduler(
            thermal_model=ThermalModel.ARRHENIUS_BASED,
            cooling_strategy=CoolingStrategy.ADAPTIVE,
            max_retries=2,
            timeout_seconds=60.0,
            enable_monitoring=False,  # Disable for test stability
            enable_circuit_breaker=False
        )
        
        # Run optimization
        result = scheduler.schedule_tasks_robust(tasks)
        
        # Verify result quality
        assert result is not None
        assert result.makespan > 0
        assert 0 <= result.resource_utilization <= 1
        
        # Check thermal-specific metrics if present
        if hasattr(result, 'thermal_efficiency'):
            assert 0 <= result.thermal_efficiency <= 1
        
        if hasattr(result, 'max_device_temperature'):
            assert result.max_device_temperature > 0
            assert result.max_device_temperature < 150  # Reasonable upper bound
        
        # Verify system status
        status = scheduler.get_system_status()
        assert status["performance"]["success_count"] == 1
        assert status["performance"]["failure_count"] == 0
    
    def test_benchmark_integration(self):
        """Test benchmark integration."""
        # Create test task sets
        planner = QuantumTaskPlanner()
        task_sets = [
            planner.create_compilation_plan({"test": f"set_{i}"})[:4]  # Small sets for speed
            for i in range(2)
        ]
        
        # Run benchmark
        benchmark = ThermalAwareBenchmark()
        results = benchmark.run_comparative_study(task_sets, iterations=1)
        
        # Verify benchmark results
        assert "experiment_summary" in results
        assert "detailed_comparison" in results
        
        summary = results["experiment_summary"]
        assert summary["total_runs"] == 2
        
        comparison = results["detailed_comparison"]
        for metric_name, metric_data in comparison.items():
            assert "baseline_avg" in metric_data
            assert "thermal_avg" in metric_data
            assert "improvement_percent" in metric_data


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])