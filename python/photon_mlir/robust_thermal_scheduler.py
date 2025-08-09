"""
Robust Thermal-Aware Scheduler with Enterprise-Grade Error Handling

This module provides production-ready thermal-aware quantum scheduling with
comprehensive error handling, monitoring, recovery mechanisms, and validation.
It builds on the research implementation with enterprise reliability features.
"""

import logging
import time
import traceback
import threading
import queue
import multiprocessing
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from .quantum_scheduler import CompilationTask, SchedulingState
from .thermal_optimization import ThermalAwareOptimizer, ThermalModel, CoolingStrategy
from .quantum_validation import QuantumValidator, ValidationLevel, ValidationResult

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for different failure modes."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    FAIL_SAFE = "fail_safe"
    RESTART = "restart"


@dataclass
class ErrorContext:
    """Context information for errors."""
    error_id: str
    timestamp: float
    severity: ErrorSeverity
    component: str
    operation: str
    error_message: str
    stack_trace: Optional[str] = None
    recovery_attempts: int = 0
    context_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "component": self.component,
            "operation": self.operation,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "recovery_attempts": self.recovery_attempts,
            "context_data": self.context_data
        }


@dataclass
class HealthMetrics:
    """System health metrics."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    success_rate: float = 1.0
    avg_response_time: float = 0.0
    active_operations: int = 0
    total_operations: int = 0
    last_error_time: Optional[float] = None
    health_score: float = 1.0  # 0-1 scale
    
    def update_health_score(self):
        """Update overall health score based on metrics."""
        factors = [
            max(0, 1.0 - self.cpu_usage / 100.0),     # CPU factor
            max(0, 1.0 - self.memory_usage / 100.0),  # Memory factor
            self.success_rate,                         # Success rate factor
            max(0, 1.0 - min(self.error_rate, 1.0))   # Error rate factor
        ]
        
        self.health_score = sum(factors) / len(factors)


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function with circuit breaker protection."""
        with self._lock:
            if self.state == "open":
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = "half_open"
                    self.failure_count = 0
                else:
                    raise RuntimeError("Circuit breaker is open")
            
            try:
                result = func(*args, **kwargs)
                
                if self.state == "half_open":
                    self.state = "closed"
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = "open"
                
                raise e


class HealthMonitor:
    """Real-time health monitoring system."""
    
    def __init__(self, monitoring_interval: float = 5.0):
        self.monitoring_interval = monitoring_interval
        self.health_metrics = HealthMetrics()
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.alert_callbacks: List[Callable[[HealthMetrics], None]] = []
        self._lock = threading.Lock()
    
    def start_monitoring(self):
        """Start health monitoring."""
        if self.is_monitoring:
            logger.warning("Health monitoring already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        logger.info("Health monitoring stopped")
    
    def add_alert_callback(self, callback: Callable[[HealthMetrics], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def update_metrics(self, **updates):
        """Update health metrics."""
        with self._lock:
            for key, value in updates.items():
                if hasattr(self.health_metrics, key):
                    setattr(self.health_metrics, key, value)
            
            self.health_metrics.update_health_score()
            
            # Check for alerts
            if self.health_metrics.health_score < 0.7:  # Health threshold
                for callback in self.alert_callbacks:
                    try:
                        callback(self.health_metrics)
                    except Exception as e:
                        logger.error(f"Health alert callback failed: {e}")
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self._lock:
            return {
                "health_score": self.health_metrics.health_score,
                "cpu_usage": self.health_metrics.cpu_usage,
                "memory_usage": self.health_metrics.memory_usage,
                "error_rate": self.health_metrics.error_rate,
                "success_rate": self.health_metrics.success_rate,
                "avg_response_time": self.health_metrics.avg_response_time,
                "active_operations": self.health_metrics.active_operations,
                "total_operations": self.health_metrics.total_operations,
                "status": self._get_health_status_text()
            }
    
    def _get_health_status_text(self) -> str:
        """Get textual health status."""
        score = self.health_metrics.health_score
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.7:
            return "fair"
        elif score >= 0.5:
            return "poor"
        else:
            return "critical"
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics (simplified for demo)
                import psutil
                
                with self._lock:
                    self.health_metrics.cpu_usage = psutil.cpu_percent()
                    self.health_metrics.memory_usage = psutil.virtual_memory().percent
                    self.health_metrics.update_health_score()
                
            except ImportError:
                # psutil not available, use placeholder values
                pass
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            time.sleep(self.monitoring_interval)


class ErrorHandler:
    """Comprehensive error handling and recovery system."""
    
    def __init__(self, max_error_history: int = 1000):
        self.max_error_history = max_error_history
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.error_count_by_type: Dict[str, int] = {}
        self._lock = threading.Lock()
        
        # Default recovery strategies
        self._setup_default_strategies()
    
    def handle_error(self, error: Exception, component: str, operation: str, 
                    context_data: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Handle an error with appropriate recovery strategy."""
        error_id = self._generate_error_id(error, component, operation)
        
        error_context = ErrorContext(
            error_id=error_id,
            timestamp=time.time(),
            severity=self._determine_severity(error),
            component=component,
            operation=operation,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context_data=context_data or {}
        )
        
        with self._lock:
            self.error_history.append(error_context)
            
            # Trim history if too long
            if len(self.error_history) > self.max_error_history:
                self.error_history = self.error_history[-self.max_error_history//2:]
            
            # Update error count
            error_type = type(error).__name__
            self.error_count_by_type[error_type] = self.error_count_by_type.get(error_type, 0) + 1
        
        # Log the error
        log_level = self._get_log_level(error_context.severity)
        logger.log(log_level, f"Error in {component}.{operation}: {error_context.error_message}")
        
        return error_context
    
    def get_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Get appropriate recovery strategy for error."""
        key = f"{error_context.component}.{error_context.operation}"
        
        # Use specific strategy if defined
        if key in self.recovery_strategies:
            return self.recovery_strategies[key]
        
        # Use general strategy based on severity
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAIL_SAFE
        elif error_context.severity == ErrorSeverity.HIGH:
            return RecoveryStrategy.FALLBACK
        elif error_context.severity == ErrorSeverity.MEDIUM:
            return RecoveryStrategy.RETRY
        else:
            return RecoveryStrategy.GRACEFUL_DEGRADATION
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics."""
        with self._lock:
            total_errors = len(self.error_history)
            
            if total_errors == 0:
                return {"total_errors": 0, "error_rate": 0.0}
            
            # Recent errors (last hour)
            hour_ago = time.time() - 3600
            recent_errors = [e for e in self.error_history if e.timestamp > hour_ago]
            
            # Errors by severity
            severity_counts = {}
            for error in self.error_history:
                severity_counts[error.severity.value] = severity_counts.get(error.severity.value, 0) + 1
            
            return {
                "total_errors": total_errors,
                "recent_errors": len(recent_errors),
                "error_rate": len(recent_errors) / 3600,  # errors per second
                "errors_by_severity": severity_counts,
                "errors_by_type": dict(self.error_count_by_type),
                "most_common_error": max(self.error_count_by_type.items(), 
                                       key=lambda x: x[1])[0] if self.error_count_by_type else None
            }
    
    def _setup_default_strategies(self):
        """Setup default recovery strategies."""
        self.recovery_strategies = {
            "thermal_optimizer.optimize_thermal_schedule": RecoveryStrategy.RETRY,
            "quantum_scheduler.schedule_tasks": RecoveryStrategy.FALLBACK,
            "validator.validate_tasks": RecoveryStrategy.GRACEFUL_DEGRADATION,
            "thermal_optimizer.calculate_thermal_metrics": RecoveryStrategy.RETRY
        }
    
    def _generate_error_id(self, error: Exception, component: str, operation: str) -> str:
        """Generate unique error ID."""
        content = f"{type(error).__name__}:{component}:{operation}:{int(time.time())}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity."""
        error_type = type(error).__name__
        
        critical_errors = ["SystemExit", "KeyboardInterrupt", "MemoryError"]
        high_errors = ["RuntimeError", "ValueError", "TypeError"]
        medium_errors = ["AttributeError", "KeyError", "IndexError"]
        
        if error_type in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_type in high_errors:
            return ErrorSeverity.HIGH
        elif error_type in medium_errors:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW
    
    def _get_log_level(self, severity: ErrorSeverity) -> int:
        """Get appropriate log level for severity."""
        mapping = {
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.LOW: logging.INFO
        }
        return mapping.get(severity, logging.INFO)


class RobustThermalScheduler:
    """
    Production-ready thermal-aware quantum scheduler with comprehensive
    error handling, monitoring, and recovery capabilities.
    """
    
    def __init__(self,
                 thermal_model: ThermalModel = ThermalModel.ARRHENIUS_BASED,
                 cooling_strategy: CoolingStrategy = CoolingStrategy.ADAPTIVE,
                 max_retries: int = 3,
                 timeout_seconds: float = 300.0,
                 enable_monitoring: bool = True,
                 enable_circuit_breaker: bool = True):
        
        self.thermal_model = thermal_model
        self.cooling_strategy = cooling_strategy
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        
        # Core components
        self.thermal_optimizer = ThermalAwareOptimizer(thermal_model, cooling_strategy)
        self.validator = QuantumValidator(ValidationLevel.STRICT)
        self.error_handler = ErrorHandler()
        
        # Monitoring and reliability
        self.health_monitor = HealthMonitor() if enable_monitoring else None
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # Performance tracking
        self.operation_times: List[float] = []
        self.success_count = 0
        self.failure_count = 0
        
        # Setup monitoring
        if self.health_monitor:
            self.health_monitor.add_alert_callback(self._handle_health_alert)
            self.health_monitor.start_monitoring()
        
        logger.info(f"RobustThermalScheduler initialized with {thermal_model.value} model")
    
    def schedule_tasks_robust(self, tasks: List[CompilationTask], 
                            validation_level: ValidationLevel = ValidationLevel.STRICT) -> SchedulingState:
        """
        Robust task scheduling with comprehensive error handling and recovery.
        
        Args:
            tasks: List of compilation tasks
            validation_level: Validation strictness level
            
        Returns:
            Optimized scheduling state
            
        Raises:
            RuntimeError: If scheduling fails after all recovery attempts
        """
        operation_start = time.time()
        
        # Update monitoring
        if self.health_monitor:
            self.health_monitor.update_metrics(active_operations=1)
        
        try:
            # Validate inputs
            validation_result = self._validate_inputs_robust(tasks, validation_level)
            if not validation_result.is_valid:
                raise ValueError(f"Input validation failed: {validation_result.errors}")
            
            # Perform scheduling with circuit breaker protection
            if self.circuit_breaker:
                result = self.circuit_breaker.call(self._schedule_with_recovery, tasks)
            else:
                result = self._schedule_with_recovery(tasks)
            
            # Validate output
            output_validation = self.validator.validate_schedule(result)
            if not output_validation.is_valid:
                logger.warning(f"Output validation issues: {output_validation.errors}")
            
            # Update success metrics
            self.success_count += 1
            operation_time = time.time() - operation_start
            self.operation_times.append(operation_time)
            
            if self.health_monitor:
                self.health_monitor.update_metrics(
                    active_operations=0,
                    total_operations=self.success_count + self.failure_count,
                    success_rate=self.success_count / (self.success_count + self.failure_count),
                    avg_response_time=sum(self.operation_times) / len(self.operation_times)
                )
            
            logger.info(f"Robust scheduling completed successfully in {operation_time:.3f}s")
            return result
            
        except Exception as e:
            # Handle error
            error_context = self.error_handler.handle_error(
                e, "robust_scheduler", "schedule_tasks_robust",
                {"task_count": len(tasks), "validation_level": validation_level.value}
            )
            
            self.failure_count += 1
            
            if self.health_monitor:
                self.health_monitor.update_metrics(
                    active_operations=0,
                    error_rate=self.failure_count / (self.success_count + self.failure_count),
                    last_error_time=time.time()
                )
            
            # Determine recovery strategy
            recovery_strategy = self.error_handler.get_recovery_strategy(error_context)
            
            if recovery_strategy == RecoveryStrategy.FAIL_SAFE:
                # Return minimal safe schedule
                return self._create_failsafe_schedule(tasks)
            else:
                # Re-raise for calling code to handle
                raise RuntimeError(f"Scheduling failed: {error_context.error_message}")
    
    def _validate_inputs_robust(self, tasks: List[CompilationTask], 
                              validation_level: ValidationLevel) -> ValidationResult:
        """Robust input validation with error handling."""
        try:
            # Update validator level
            self.validator.validation_level = validation_level
            
            # Perform validation with timeout
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.validator.validate_tasks, tasks)
                return future.result(timeout=30.0)  # 30 second timeout for validation
                
        except TimeoutError:
            error_result = ValidationResult(
                is_valid=False,
                errors=["Validation timed out"],
                warnings=[],
                security_threats=[],
                performance_issues=["Validation timeout indicates performance issues"],
                suggestions=["Consider reducing task complexity"]
            )
            return error_result
        except Exception as e:
            self.error_handler.handle_error(e, "validator", "validate_tasks")
            
            # Return basic validation result
            error_result = ValidationResult(
                is_valid=len(tasks) > 0,  # Basic check
                errors=[f"Validation error: {str(e)}"],
                warnings=[],
                security_threats=[],
                performance_issues=[],
                suggestions=[]
            )
            return error_result
    
    def _schedule_with_recovery(self, tasks: List[CompilationTask]) -> SchedulingState:
        """Schedule tasks with built-in recovery mechanisms."""
        attempt = 0
        last_error = None
        
        while attempt < self.max_retries:
            attempt += 1
            
            try:
                logger.debug(f"Scheduling attempt {attempt}/{self.max_retries}")
                
                # Get initial schedule from quantum scheduler
                from .quantum_scheduler import QuantumInspiredScheduler
                
                base_scheduler = QuantumInspiredScheduler(
                    population_size=max(20, 50 - attempt * 10),  # Reduce complexity on retries
                    max_iterations=max(100, 500 - attempt * 100),
                    enable_validation=True
                )
                
                # Execute with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(base_scheduler.schedule_tasks, tasks)
                    initial_schedule = future.result(timeout=self.timeout_seconds / 2)
                
                # Apply thermal optimization
                future = executor.submit(
                    self.thermal_optimizer.optimize_thermal_schedule,
                    initial_schedule,
                    max(200, 500 - attempt * 100)  # Reduce iterations on retries
                )
                
                optimized_schedule = future.result(timeout=self.timeout_seconds / 2)
                
                logger.debug(f"Scheduling successful on attempt {attempt}")
                return optimized_schedule
                
            except TimeoutError as e:
                last_error = e
                logger.warning(f"Scheduling attempt {attempt} timed out")
                
                # Exponential backoff
                if attempt < self.max_retries:
                    delay = min(2 ** attempt, 30)  # Max 30 seconds
                    time.sleep(delay)
                    
            except Exception as e:
                last_error = e
                error_context = self.error_handler.handle_error(
                    e, "thermal_scheduler", "schedule_with_recovery",
                    {"attempt": attempt, "max_retries": self.max_retries}
                )
                
                # Decide whether to retry based on error type
                if error_context.severity == ErrorSeverity.CRITICAL:
                    break  # Don't retry critical errors
                
                logger.warning(f"Scheduling attempt {attempt} failed: {str(e)}")
                
                if attempt < self.max_retries:
                    delay = min(2 ** attempt, 30)
                    time.sleep(delay)
        
        # All attempts failed
        raise RuntimeError(f"All {self.max_retries} scheduling attempts failed. Last error: {last_error}")
    
    def _create_failsafe_schedule(self, tasks: List[CompilationTask]) -> SchedulingState:
        """Create a minimal safe schedule as fallback."""
        logger.warning("Creating failsafe schedule")
        
        # Simple sequential schedule
        schedule = {}
        current_slot = 0
        
        try:
            # Sort tasks by dependencies (simple topological sort)
            sorted_tasks = self._simple_topological_sort(tasks)
            
            for task in sorted_tasks:
                schedule[current_slot] = [task.id]
                current_slot += max(1, int(task.estimated_duration))
            
            failsafe_state = SchedulingState(
                tasks=tasks,
                schedule=schedule,
                makespan=current_slot,
                resource_utilization=0.5  # Conservative estimate
            )
            
            logger.info(f"Failsafe schedule created with makespan {current_slot}")
            return failsafe_state
            
        except Exception as e:
            logger.error(f"Failsafe schedule creation failed: {e}")
            
            # Ultra-minimal schedule
            schedule = {i: [task.id] for i, task in enumerate(tasks)}
            
            return SchedulingState(
                tasks=tasks,
                schedule=schedule,
                makespan=len(tasks),
                resource_utilization=0.1
            )
    
    def _simple_topological_sort(self, tasks: List[CompilationTask]) -> List[CompilationTask]:
        """Simple topological sort with error handling."""
        try:
            # Kahn's algorithm (simplified)
            in_degree = {task.id: 0 for task in tasks}
            task_map = {task.id: task for task in tasks}
            
            # Calculate in-degrees
            for task in tasks:
                for dep_id in task.dependencies:
                    if dep_id in in_degree:
                        in_degree[task.id] += 1
            
            # Process tasks with no dependencies first
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
            
            if len(result) == len(tasks):
                return result
            else:
                # Circular dependencies detected, fall back to original order
                logger.warning("Circular dependencies detected in topological sort")
                return tasks
                
        except Exception as e:
            logger.error(f"Topological sort failed: {e}")
            return tasks
    
    def _handle_health_alert(self, metrics: HealthMetrics):
        """Handle health monitoring alerts."""
        logger.warning(f"Health alert: score={metrics.health_score:.2f}, "
                      f"cpu={metrics.cpu_usage:.1f}%, "
                      f"memory={metrics.memory_usage:.1f}%, "
                      f"error_rate={metrics.error_rate:.3f}")
        
        # Take corrective actions based on health metrics
        if metrics.health_score < 0.5:
            logger.critical("System health critical - consider reducing load")
            
        if metrics.cpu_usage > 90:
            logger.warning("High CPU usage - scheduling operations may be slower")
            
        if metrics.memory_usage > 90:
            logger.warning("High memory usage - risk of memory exhaustion")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "scheduler_config": {
                "thermal_model": self.thermal_model.value,
                "cooling_strategy": self.cooling_strategy.value,
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds
            },
            "performance": {
                "success_count": self.success_count,
                "failure_count": self.failure_count,
                "success_rate": (
                    self.success_count / (self.success_count + self.failure_count)
                    if (self.success_count + self.failure_count) > 0 else 0
                ),
                "avg_operation_time": (
                    sum(self.operation_times) / len(self.operation_times)
                    if self.operation_times else 0
                )
            },
            "error_statistics": self.error_handler.get_error_statistics(),
            "circuit_breaker": {
                "state": self.circuit_breaker.state if self.circuit_breaker else "disabled",
                "failure_count": self.circuit_breaker.failure_count if self.circuit_breaker else 0
            }
        }
        
        if self.health_monitor:
            status["health"] = self.health_monitor.get_health_status()
        
        return status
    
    def shutdown(self):
        """Graceful shutdown of the scheduler."""
        logger.info("Shutting down RobustThermalScheduler")
        
        if self.health_monitor:
            self.health_monitor.stop_monitoring()
        
        # Save error statistics for analysis
        try:
            error_stats = self.error_handler.get_error_statistics()
            logger.info(f"Final error statistics: {error_stats}")
        except Exception as e:
            logger.error(f"Failed to log final statistics: {e}")
        
        logger.info("RobustThermalScheduler shutdown complete")