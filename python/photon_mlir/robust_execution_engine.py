"""
Generation 2: Robust Execution Engine
Advanced error handling, recovery mechanisms, and reliability features.
"""

import asyncio
import logging
import time
import traceback
import json
import pickle
import threading
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, TimeoutError
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps, lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Union, Tuple, Generator
import sqlite3
import uuid
import psutil
import hashlib
from collections import defaultdict, deque

try:
    from .logging_config import get_global_logger, performance_monitor
    from .validation import PhotonicValidator, ValidationResult
    from .circuit_breaker import CircuitBreaker, CircuitState
    from .caching_system import SecureCacheManager
    from .recovery_manager import RecoveryManager
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False
    get_global_logger = performance_monitor = None
    PhotonicValidator = ValidationResult = None
    CircuitBreaker = CircuitState = None
    SecureCacheManager = RecoveryManager = None


class ExecutionState(Enum):
    """Execution states for robust tracking."""
    PENDING = auto()
    INITIALIZING = auto()
    RUNNING = auto()
    PAUSED = auto()
    RECOVERING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()
    TIMEOUT = auto()


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4
    CATASTROPHIC = 5


class RecoveryStrategy(Enum):
    """Recovery strategies for different error types."""
    RETRY = auto()
    FALLBACK = auto()
    CIRCUIT_BREAK = auto()
    RESTART = auto()
    ESCALATE = auto()
    IGNORE = auto()


@dataclass
class ExecutionContext:
    """Comprehensive execution context for robust operations."""
    execution_id: str
    operation_name: str
    start_time: datetime
    timeout_seconds: float
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    circuit_breaker_enabled: bool = True
    checkpoint_enabled: bool = True
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY
    
    # State tracking
    current_retry: int = 0
    state: ExecutionState = ExecutionState.PENDING
    last_checkpoint: Optional[Dict[str, Any]] = None
    error_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Resource monitoring
    max_memory_mb: float = 1024.0
    max_cpu_percent: float = 80.0
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def elapsed_time(self) -> timedelta:
        return datetime.now() - self.start_time
    
    @property
    def is_timeout(self) -> bool:
        return self.elapsed_time.total_seconds() > self.timeout_seconds
    
    @property
    def should_retry(self) -> bool:
        return (self.current_retry < self.max_retries and 
                self.state in [ExecutionState.FAILED, ExecutionState.TIMEOUT])
    
    def add_error(self, error: Exception, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Add error to history."""
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity.name,
            'traceback': traceback.format_exc(),
            'retry_attempt': self.current_retry
        }
        self.error_history.append(error_info)


@dataclass
class RobustExecutionResult:
    """Result of robust execution with comprehensive information."""
    success: bool
    execution_id: str
    operation_name: str
    result: Optional[Any] = None
    error: Optional[Exception] = None
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    total_duration_ms: float = 0.0
    
    # Execution statistics
    retry_count: int = 0
    checkpoint_count: int = 0
    recovery_actions: List[str] = field(default_factory=list)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    reliability_score: float = 1.0
    performance_score: float = 1.0
    error_severity: ErrorSeverity = ErrorSeverity.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'success': self.success,
            'execution_id': self.execution_id,
            'operation_name': self.operation_name,
            'total_duration_ms': self.total_duration_ms,
            'retry_count': self.retry_count,
            'checkpoint_count': self.checkpoint_count,
            'recovery_actions': self.recovery_actions,
            'resource_usage': self.resource_usage,
            'reliability_score': self.reliability_score,
            'performance_score': self.performance_score,
            'error_severity': self.error_severity.name,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None
        }


class ResourceMonitor:
    """Real-time resource monitoring for execution safety."""
    
    def __init__(self, check_interval_seconds: float = 1.0):
        self.check_interval_seconds = check_interval_seconds
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.resource_history: deque = deque(maxlen=1000)
        self.alerts: List[Dict[str, Any]] = []
        
    def start_monitoring(self, execution_context: ExecutionContext):
        """Start resource monitoring for execution context."""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.execution_context = execution_context
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get current resource usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
                
                resource_snapshot = {
                    'timestamp': datetime.now(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_percent,
                    'execution_id': self.execution_context.execution_id
                }
                
                self.resource_history.append(resource_snapshot)
                
                # Check thresholds
                if memory_mb > self.execution_context.max_memory_mb:
                    alert = {
                        'type': 'memory_threshold_exceeded',
                        'current_mb': memory_mb,
                        'threshold_mb': self.execution_context.max_memory_mb,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                
                if cpu_percent > self.execution_context.max_cpu_percent:
                    alert = {
                        'type': 'cpu_threshold_exceeded',
                        'current_percent': cpu_percent,
                        'threshold_percent': self.execution_context.max_cpu_percent,
                        'timestamp': datetime.now().isoformat()
                    }
                    self.alerts.append(alert)
                
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                # Monitoring errors shouldn't break execution
                pass
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            process = psutil.Process()
            return {
                'memory_mb': process.memory_info().rss / 1024 / 1024,
                'cpu_percent': process.cpu_percent(),
                'open_files': len(process.open_files()),
                'num_threads': process.num_threads()
            }
        except Exception:
            return {}


class CheckpointManager:
    """Manages execution checkpoints for recovery."""
    
    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = checkpoint_dir or Path.home() / '.photon_checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints: Dict[str, Dict[str, Any]] = {}
        
    def create_checkpoint(self, execution_id: str, state: Dict[str, Any]) -> str:
        """Create a checkpoint for execution state."""
        checkpoint_id = f"{execution_id}_{int(time.time() * 1000)}"
        
        checkpoint_data = {
            'checkpoint_id': checkpoint_id,
            'execution_id': execution_id,
            'timestamp': datetime.now().isoformat(),
            'state': state,
            'state_hash': hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()
        }
        
        # Save to memory
        self.checkpoints[checkpoint_id] = checkpoint_data
        
        # Persist to disk
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
        except Exception:
            # Non-critical error, continue without disk persistence
            pass
            
        return checkpoint_id
    
    def restore_checkpoint(self, checkpoint_id: str) -> Optional[Dict[str, Any]]:
        """Restore state from checkpoint."""
        # Try memory first
        if checkpoint_id in self.checkpoints:
            return self.checkpoints[checkpoint_id]['state']
        
        # Try disk
        try:
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.pkl"
            if checkpoint_file.exists():
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                    return checkpoint_data['state']
        except Exception:
            pass
            
        return None
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24):
        """Clean up old checkpoints."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        # Clean memory
        to_remove = []
        for checkpoint_id, data in self.checkpoints.items():
            checkpoint_time = datetime.fromisoformat(data['timestamp'])
            if checkpoint_time < cutoff_time:
                to_remove.append(checkpoint_id)
        
        for checkpoint_id in to_remove:
            del self.checkpoints[checkpoint_id]
        
        # Clean disk
        try:
            for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
                if checkpoint_file.stat().st_mtime < cutoff_time.timestamp():
                    checkpoint_file.unlink()
        except Exception:
            pass


class RobustExecutionEngine:
    """Generation 2: Robust execution engine with comprehensive error handling and recovery."""
    
    def __init__(self, max_workers: int = 4, enable_monitoring: bool = True,
                 enable_checkpoints: bool = True, logger: Optional[logging.Logger] = None):
        self.max_workers = max_workers
        self.enable_monitoring = enable_monitoring
        self.enable_checkpoints = enable_checkpoints
        self.logger = logger or (get_global_logger() if DEPENDENCIES_AVAILABLE else logging.getLogger(__name__))
        
        # Core components
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RobustExec")
        self.process_pool = ProcessPoolExecutor(max_workers=max(1, max_workers // 2))
        
        # Robust components
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.resource_monitor = ResourceMonitor() if enable_monitoring else None
        self.checkpoint_manager = CheckpointManager() if enable_checkpoints else None
        
        if DEPENDENCIES_AVAILABLE:
            self.cache_manager = SecureCacheManager()
            self.recovery_manager = RecoveryManager()
            self.validator = PhotonicValidator()
        else:
            self.cache_manager = None
            self.recovery_manager = None
            self.validator = None
        
        # Execution tracking
        self.active_executions: Dict[str, ExecutionContext] = {}
        self.execution_history: List[RobustExecutionResult] = []
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'avg_execution_time_ms': 0.0,
            'recovery_actions_taken': 0,
            'circuit_breaker_trips': 0
        }
        
        # Synchronization
        self.execution_lock = threading.RLock()
        self.stats_lock = threading.Lock()
        
        self.logger.info(f"Robust Execution Engine initialized (workers={max_workers}, monitoring={enable_monitoring})")
    
    @contextmanager
    def robust_execution(self, operation_name: str, timeout_seconds: float = 300.0,
                        max_retries: int = 3, **kwargs) -> Generator[ExecutionContext, None, None]:
        """Context manager for robust execution with comprehensive error handling."""
        
        execution_id = str(uuid.uuid4())
        context = ExecutionContext(
            execution_id=execution_id,
            operation_name=operation_name,
            start_time=datetime.now(),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            **kwargs
        )
        
        self.logger.info(f"🔧 Starting robust execution: {operation_name} (ID: {execution_id})")
        
        try:
            # Register execution
            with self.execution_lock:
                self.active_executions[execution_id] = context
            
            # Start resource monitoring
            if self.resource_monitor and self.enable_monitoring:
                self.resource_monitor.start_monitoring(context)
            
            context.state = ExecutionState.RUNNING
            yield context
            
            # Success
            context.state = ExecutionState.COMPLETED
            self.logger.info(f"✅ Robust execution completed: {operation_name}")
            
        except Exception as e:
            context.add_error(e, self._classify_error_severity(e))
            context.state = ExecutionState.FAILED
            
            self.logger.error(f"❌ Robust execution failed: {operation_name} - {e}")
            
            # Attempt recovery if configured
            if context.recovery_strategy != RecoveryStrategy.IGNORE:
                recovery_success = self._attempt_recovery(context, e)
                if recovery_success:
                    context.state = ExecutionState.COMPLETED
                    self.logger.info(f"🔄 Recovery successful for: {operation_name}")
            
            raise
            
        finally:
            # Cleanup
            if self.resource_monitor and self.enable_monitoring:
                self.resource_monitor.stop_monitoring()
            
            with self.execution_lock:
                self.active_executions.pop(execution_id, None)
            
            # Record execution result
            self._record_execution_result(context)
    
    async def execute_with_resilience(self, operation: Callable, *args, **kwargs) -> RobustExecutionResult:
        """Execute operation with full resilience features."""
        
        operation_name = getattr(operation, '__name__', 'unknown_operation')
        timeout = kwargs.pop('timeout_seconds', 300.0)
        max_retries = kwargs.pop('max_retries', 3)
        
        start_time = datetime.now()
        execution_id = str(uuid.uuid4())
        
        result = RobustExecutionResult(
            success=False,
            execution_id=execution_id,
            operation_name=operation_name,
            start_time=start_time
        )
        
        try:
            with self.robust_execution(operation_name, timeout, max_retries) as context:
                # Check circuit breaker
                circuit_breaker = self._get_circuit_breaker(operation_name)
                
                if circuit_breaker.state == CircuitState.OPEN:
                    raise RuntimeError(f"Circuit breaker is OPEN for {operation_name}")
                
                # Execute with timeout
                try:
                    if asyncio.iscoroutinefunction(operation):
                        operation_result = await asyncio.wait_for(
                            operation(*args, **kwargs),
                            timeout=timeout
                        )
                    else:
                        # Run sync operation in thread pool
                        operation_result = await asyncio.get_event_loop().run_in_executor(
                            self.thread_pool,
                            operation,
                            *args,
                            **{k: v for k, v in kwargs.items() if k not in ['timeout_seconds', 'max_retries']}
                        )
                    
                    result.result = operation_result
                    result.success = True
                    
                    # Update circuit breaker on success
                    circuit_breaker.record_success()
                    
                except asyncio.TimeoutError:
                    context.state = ExecutionState.TIMEOUT
                    raise TimeoutError(f"Operation {operation_name} timed out after {timeout} seconds")
                
        except Exception as e:
            result.error = e
            result.success = False
            
            # Update circuit breaker on failure
            circuit_breaker = self._get_circuit_breaker(operation_name)
            circuit_breaker.record_failure()
            
            # Classify error severity
            result.error_severity = self._classify_error_severity(e)
            
            self.logger.error(f"Resilient execution failed: {operation_name} - {e}")
        
        finally:
            result.end_time = datetime.now()
            result.total_duration_ms = (result.end_time - result.start_time).total_seconds() * 1000
            
            # Add resource usage if monitoring enabled
            if self.resource_monitor:
                result.resource_usage = self.resource_monitor.get_current_usage()
            
            # Update statistics
            self._update_execution_stats(result)
        
        return result
    
    def _get_circuit_breaker(self, operation_name: str) -> 'CircuitBreaker':
        """Get or create circuit breaker for operation."""
        if operation_name not in self.circuit_breakers:
            if DEPENDENCIES_AVAILABLE:
                self.circuit_breakers[operation_name] = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout_seconds=30,
                    expected_exception=Exception
                )
            else:
                # Mock circuit breaker
                class MockCircuitBreaker:
                    def __init__(self):
                        self.state = 'CLOSED'
                    def record_success(self): pass
                    def record_failure(self): pass
                
                self.circuit_breakers[operation_name] = MockCircuitBreaker()
        
        return self.circuit_breakers[operation_name]
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity for appropriate response."""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CATASTROPHIC
        
        # High severity
        if isinstance(error, (OSError, IOError, PermissionError)):
            return ErrorSeverity.HIGH
        
        if 'timeout' in error_message or 'deadlock' in error_message:
            return ErrorSeverity.HIGH
        
        # Medium severity  
        if isinstance(error, (ValueError, TypeError, RuntimeError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity for most other errors
        return ErrorSeverity.LOW
    
    def _attempt_recovery(self, context: ExecutionContext, error: Exception) -> bool:
        """Attempt to recover from error based on strategy."""
        
        if context.recovery_strategy == RecoveryStrategy.IGNORE:
            return False
        
        self.logger.info(f"🔄 Attempting recovery for {context.operation_name} using {context.recovery_strategy.name}")
        
        try:
            if context.recovery_strategy == RecoveryStrategy.RETRY:
                if context.should_retry:
                    context.current_retry += 1
                    time.sleep(context.retry_delay_seconds * context.current_retry)  # Exponential backoff
                    return True
                    
            elif context.recovery_strategy == RecoveryStrategy.FALLBACK:
                # Implement fallback logic
                self.logger.info("Using fallback recovery mechanism")
                return True
                
            elif context.recovery_strategy == RecoveryStrategy.CIRCUIT_BREAK:
                circuit_breaker = self._get_circuit_breaker(context.operation_name)
                circuit_breaker.record_failure()
                return False
                
            elif context.recovery_strategy == RecoveryStrategy.RESTART:
                # Clean restart logic
                self.logger.info("Performing clean restart")
                if self.checkpoint_manager and context.last_checkpoint:
                    restored_state = self.checkpoint_manager.restore_checkpoint(context.last_checkpoint)
                    if restored_state:
                        return True
                        
            elif context.recovery_strategy == RecoveryStrategy.ESCALATE:
                # Escalate to human intervention
                self.logger.critical(f"Escalating {context.operation_name} for manual intervention")
                return False
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery attempt failed: {recovery_error}")
            
        return False
    
    def _record_execution_result(self, context: ExecutionContext):
        """Record execution result for analysis."""
        end_time = datetime.now()
        duration_ms = (end_time - context.start_time).total_seconds() * 1000
        
        result = RobustExecutionResult(
            success=(context.state == ExecutionState.COMPLETED),
            execution_id=context.execution_id,
            operation_name=context.operation_name,
            start_time=context.start_time,
            end_time=end_time,
            total_duration_ms=duration_ms,
            retry_count=context.current_retry
        )
        
        # Add resource usage
        if self.resource_monitor:
            result.resource_usage = self.resource_monitor.get_current_usage()
        
        # Add error information
        if context.error_history:
            last_error = context.error_history[-1]
            result.error_severity = ErrorSeverity[last_error['severity']]
        
        # Calculate reliability and performance scores
        result.reliability_score = self._calculate_reliability_score(context)
        result.performance_score = self._calculate_performance_score(duration_ms, context.timeout_seconds)
        
        with self.execution_lock:
            self.execution_history.append(result)
            
            # Limit history size
            if len(self.execution_history) > 10000:
                self.execution_history = self.execution_history[-5000:]
    
    def _calculate_reliability_score(self, context: ExecutionContext) -> float:
        """Calculate reliability score based on execution characteristics."""
        base_score = 1.0
        
        # Penalty for retries
        if context.current_retry > 0:
            base_score -= context.current_retry * 0.1
        
        # Penalty for errors
        if context.error_history:
            error_penalty = len(context.error_history) * 0.05
            base_score -= error_penalty
        
        # Penalty for timeouts
        if context.is_timeout:
            base_score -= 0.3
        
        return max(0.0, min(1.0, base_score))
    
    def _calculate_performance_score(self, duration_ms: float, timeout_seconds: float) -> float:
        """Calculate performance score based on execution time."""
        timeout_ms = timeout_seconds * 1000
        
        # Perfect score for executions under 10% of timeout
        if duration_ms < timeout_ms * 0.1:
            return 1.0
        
        # Linear scaling from 1.0 to 0.1 as we approach timeout
        performance_ratio = 1.0 - ((duration_ms - timeout_ms * 0.1) / (timeout_ms * 0.9))
        return max(0.1, min(1.0, performance_ratio))
    
    def _update_execution_stats(self, result: RobustExecutionResult):
        """Update global execution statistics."""
        with self.stats_lock:
            self.execution_stats['total_executions'] += 1
            
            if result.success:
                self.execution_stats['successful_executions'] += 1
            else:
                self.execution_stats['failed_executions'] += 1
            
            # Update average execution time (exponential moving average)
            alpha = 0.1
            current_avg = self.execution_stats['avg_execution_time_ms']
            self.execution_stats['avg_execution_time_ms'] = (
                alpha * result.total_duration_ms + (1 - alpha) * current_avg
            )
            
            if result.retry_count > 0:
                self.execution_stats['recovery_actions_taken'] += result.retry_count
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Get comprehensive execution report."""
        with self.stats_lock:
            success_rate = (self.execution_stats['successful_executions'] / 
                           max(1, self.execution_stats['total_executions']))
            
            report = {
                'execution_statistics': self.execution_stats.copy(),
                'success_rate': success_rate,
                'active_executions': len(self.active_executions),
                'circuit_breakers': {
                    name: {'state': str(cb.state) if hasattr(cb, 'state') else 'UNKNOWN'}
                    for name, cb in self.circuit_breakers.items()
                },
                'features': {
                    'monitoring_enabled': self.enable_monitoring,
                    'checkpoints_enabled': self.enable_checkpoints,
                    'max_workers': self.max_workers
                },
                'timestamp': datetime.now().isoformat()
            }
            
            # Add recent execution results
            recent_executions = self.execution_history[-10:]
            report['recent_executions'] = [result.to_dict() for result in recent_executions]
            
            # Add resource alerts if monitoring enabled
            if self.resource_monitor and self.resource_monitor.alerts:
                report['resource_alerts'] = self.resource_monitor.alerts[-5:]  # Last 5 alerts
            
            return report
    
    def shutdown(self):
        """Shutdown the robust execution engine."""
        self.logger.info("Shutting down Robust Execution Engine...")
        
        # Wait for active executions
        active_count = len(self.active_executions)
        if active_count > 0:
            self.logger.info(f"Waiting for {active_count} active executions...")
            timeout = 30
            start_time = time.time()
            
            while self.active_executions and (time.time() - start_time) < timeout:
                time.sleep(1)
        
        # Stop resource monitoring
        if self.resource_monitor:
            self.resource_monitor.stop_monitoring()
        
        # Clean up checkpoints
        if self.checkpoint_manager:
            self.checkpoint_manager.cleanup_old_checkpoints()
        
        # Shutdown thread pools
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)
        
        self.logger.info("✅ Robust Execution Engine shutdown complete")


# Decorators for easy robust execution
def robust_execution_decorator(timeout_seconds: float = 300.0, max_retries: int = 3,
                              recovery_strategy: RecoveryStrategy = RecoveryStrategy.RETRY):
    """Decorator for making functions robust."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            engine = RobustExecutionEngine()
            try:
                with engine.robust_execution(
                    func.__name__, timeout_seconds, max_retries, 
                    recovery_strategy=recovery_strategy
                ) as context:
                    return func(*args, **kwargs)
            finally:
                engine.shutdown()
        return wrapper
    return decorator


# Context managers for specific use cases
@contextmanager  
def fault_tolerant_operation(operation_name: str, max_failures: int = 3):
    """Context manager for fault-tolerant operations."""
    engine = RobustExecutionEngine()
    try:
        with engine.robust_execution(
            operation_name, 
            recovery_strategy=RecoveryStrategy.RETRY,
            max_retries=max_failures
        ) as context:
            yield context
    finally:
        engine.shutdown()


@asynccontextmanager
async def resilient_async_operation(operation_name: str, timeout_seconds: float = 300.0):
    """Async context manager for resilient operations."""
    engine = RobustExecutionEngine()
    try:
        with engine.robust_execution(operation_name, timeout_seconds) as context:
            yield context
    finally:
        engine.shutdown()


# Factory functions
def create_robust_executor(**kwargs) -> RobustExecutionEngine:
    """Create a robust execution engine with default settings."""
    return RobustExecutionEngine(**kwargs)


async def execute_robustly(operation: Callable, *args, **kwargs) -> RobustExecutionResult:
    """Execute a single operation robustly."""
    engine = create_robust_executor()
    try:
        return await engine.execute_with_resilience(operation, *args, **kwargs)
    finally:
        engine.shutdown()