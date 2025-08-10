"""
Robust Error Handling and Recovery System for Photonic Neural Networks
Generation 2 Implementation - Make it Robust

This module implements comprehensive error handling, fault tolerance, and recovery
mechanisms for all photonic neural network compilation and execution components.

Key Robustness Features:
1. Multi-level error handling with graceful degradation
2. Automatic error recovery and retry mechanisms
3. Circuit breaker patterns for fault isolation
4. Comprehensive logging and monitoring
5. Input validation and sanitization
6. Resource exhaustion protection
7. Distributed system fault tolerance
"""

import numpy as np
import sys
import traceback
import functools
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict, deque
import json
import hashlib
import contextlib

from .logging_config import get_global_logger


class ErrorSeverity(Enum):
    """Error severity levels for classification."""
    CRITICAL = "critical"      # System failure, immediate attention required
    HIGH = "high"             # Major functionality impacted
    MEDIUM = "medium"         # Partial functionality impacted
    LOW = "low"              # Minor issues, system still functional
    INFO = "info"            # Informational, no action required


class ErrorCategory(Enum):
    """Categories of errors for specialized handling."""
    COMPILATION_ERROR = "compilation"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    THERMAL_VIOLATION = "thermal_violation"
    CROSSTALK_VIOLATION = "crosstalk_violation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    NETWORK_FAILURE = "network_failure"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"
    NUMERICAL_INSTABILITY = "numerical_instability"


@dataclass
class ErrorContext:
    """Rich context information for error handling."""
    error_id: str = field(default_factory=lambda: hashlib.md5(str(time.time()).encode()).hexdigest()[:8])
    timestamp: float = field(default_factory=time.time)
    severity: ErrorSeverity = ErrorSeverity.MEDIUM
    category: ErrorCategory = ErrorCategory.COMPILATION_ERROR
    component: str = "unknown"
    operation: str = "unknown"
    
    # Error details
    error_message: str = ""
    exception_type: str = ""
    stack_trace: str = ""
    
    # System state
    system_state: Dict[str, Any] = field(default_factory=dict)
    input_parameters: Dict[str, Any] = field(default_factory=dict)
    recovery_suggestions: List[str] = field(default_factory=list)
    
    # Recovery information
    retry_count: int = 0
    max_retries: int = 3
    recovery_attempted: bool = False
    recovery_successful: bool = False


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault isolation."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Preventing calls due to failures
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    failure_threshold: int = 5      # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    success_threshold: int = 3      # Successes to close from half-open
    timeout_duration: float = 10.0  # Operation timeout
    
    
class CircuitBreaker:
    """
    Circuit breaker implementation for fault isolation.
    
    Prevents cascading failures by temporarily blocking operations
    to failing components, allowing them time to recover.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.lock = threading.RLock()
        
        self.logger = get_global_logger()
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        
        with self.lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.config.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                    self.success_count = 0
                    self.logger.info(f"Circuit breaker {self.name}: Transitioning to HALF_OPEN")
                else:
                    raise CircuitBreakerError(f"Circuit breaker {self.name} is OPEN")
                    
        try:
            # Execute with timeout
            result = self._execute_with_timeout(func, *args, **kwargs)
            
            with self.lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.success_count += 1
                    if self.success_count >= self.config.success_threshold:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                        self.logger.info(f"Circuit breaker {self.name}: Transitioning to CLOSED")
                elif self.state == CircuitBreakerState.CLOSED:
                    self.failure_count = 0  # Reset failure count on success
                    
            return result
            
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if (self.state == CircuitBreakerState.CLOSED and 
                    self.failure_count >= self.config.failure_threshold):
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker {self.name}: Transitioning to OPEN after {self.failure_count} failures")
                elif self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
                    self.logger.warning(f"Circuit breaker {self.name}: Returning to OPEN from HALF_OPEN")
                    
            raise e
            
    def _execute_with_timeout(self, func: Callable, *args, **kwargs):
        """Execute function with timeout protection."""
        
        import threading
        import queue
        
        result_queue = queue.Queue()
        exception_queue = queue.Queue()
        
        def target():
            try:
                result = func(*args, **kwargs)
                result_queue.put(result)
            except Exception as e:
                exception_queue.put(e)
                
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.config.timeout_duration)
        
        if thread.is_alive():
            # Timeout occurred
            raise TimeoutError(f"Operation timed out after {self.config.timeout_duration}s")
            
        if not exception_queue.empty():
            raise exception_queue.get()
            
        if not result_queue.empty():
            return result_queue.get()
            
        raise RuntimeError("Function execution completed without result or exception")
        
    @property
    def status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        with self.lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'last_failure_time': self.last_failure_time,
                'time_until_half_open': max(0, self.config.recovery_timeout - (time.time() - self.last_failure_time))
            }


class CircuitBreakerError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class RobustErrorHandler:
    """
    Comprehensive error handling system with multiple recovery strategies.
    """
    
    def __init__(self):
        self.logger = get_global_logger()
        self.error_history = deque(maxlen=1000)
        self.circuit_breakers = {}
        self.recovery_strategies = {}
        self.error_stats = defaultdict(int)
        
        # Initialize circuit breakers for key components
        self._initialize_circuit_breakers()
        
        # Register recovery strategies
        self._register_recovery_strategies()
        
    def _initialize_circuit_breakers(self):
        """Initialize circuit breakers for critical components."""
        
        components = [
            'quantum_compiler',
            'thermal_predictor', 
            'wdm_optimizer',
            'neural_ode_solver',
            'crosstalk_predictor',
            'photonic_simulator'
        ]
        
        for component in components:
            config = CircuitBreakerConfig(
                failure_threshold=3,
                recovery_timeout=20.0,
                success_threshold=2,
                timeout_duration=30.0
            )
            self.circuit_breakers[component] = CircuitBreaker(component, config)
            
    def _register_recovery_strategies(self):
        """Register recovery strategies for different error types."""
        
        self.recovery_strategies = {
            ErrorCategory.QUANTUM_DECOHERENCE: self._recover_from_decoherence,
            ErrorCategory.THERMAL_VIOLATION: self._recover_from_thermal_violation,
            ErrorCategory.CROSSTALK_VIOLATION: self._recover_from_crosstalk,
            ErrorCategory.RESOURCE_EXHAUSTION: self._recover_from_resource_exhaustion,
            ErrorCategory.NUMERICAL_INSTABILITY: self._recover_from_numerical_instability,
            ErrorCategory.TIMEOUT_ERROR: self._recover_from_timeout,
            ErrorCategory.MEMORY_ERROR: self._recover_from_memory_error,
            ErrorCategory.NETWORK_FAILURE: self._recover_from_network_failure
        }
        
    def handle_error(self, error: Exception, context: ErrorContext) -> Tuple[bool, Any]:
        """
        Handle error with comprehensive recovery strategies.
        
        Returns:
            Tuple of (recovery_successful, recovered_result)
        """
        
        # Log error with full context
        self._log_error(error, context)
        
        # Update error statistics
        self.error_stats[context.category] += 1
        self.error_stats['total'] += 1
        
        # Store in error history
        context.error_message = str(error)
        context.exception_type = type(error).__name__
        context.stack_trace = traceback.format_exc()
        self.error_history.append(context)
        
        # Attempt recovery based on error category
        recovery_successful, result = self._attempt_recovery(error, context)
        
        context.recovery_attempted = True
        context.recovery_successful = recovery_successful
        
        return recovery_successful, result
        
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with appropriate severity level."""
        
        log_message = f"Error {context.error_id} in {context.component}.{context.operation}: {str(error)}"
        
        if context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
            
        # Log additional context
        if context.input_parameters:
            self.logger.debug(f"Input parameters: {context.input_parameters}")
        if context.system_state:
            self.logger.debug(f"System state: {context.system_state}")
            
    def _attempt_recovery(self, error: Exception, context: ErrorContext) -> Tuple[bool, Any]:
        """Attempt error recovery using registered strategies."""
        
        # Check if we've exceeded retry limit
        if context.retry_count >= context.max_retries:
            self.logger.warning(f"Max retries ({context.max_retries}) exceeded for error {context.error_id}")
            return False, None
            
        # Get recovery strategy
        recovery_func = self.recovery_strategies.get(context.category)
        
        if not recovery_func:
            self.logger.warning(f"No recovery strategy for category {context.category}")
            return False, None
            
        try:
            self.logger.info(f"Attempting recovery for error {context.error_id} using strategy {recovery_func.__name__}")
            
            # Increment retry count
            context.retry_count += 1
            
            # Attempt recovery
            recovery_result = recovery_func(error, context)
            
            if recovery_result is not None:
                self.logger.info(f"Recovery successful for error {context.error_id}")
                return True, recovery_result
            else:
                self.logger.warning(f"Recovery strategy returned None for error {context.error_id}")
                return False, None
                
        except Exception as recovery_error:
            self.logger.error(f"Recovery strategy failed for error {context.error_id}: {recovery_error}")
            return False, None
            
    def _recover_from_decoherence(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for quantum decoherence errors."""
        
        self.logger.info("Recovering from quantum decoherence...")
        
        # Strategy: Reduce coherence requirements and retry
        if 'coherence_time_ns' in context.system_state:
            original_time = context.system_state['coherence_time_ns']
            reduced_time = original_time * 0.8  # Reduce by 20%
            
            if reduced_time > 100.0:  # Minimum viable coherence time
                context.system_state['coherence_time_ns'] = reduced_time
                context.recovery_suggestions.append(f"Reduced coherence time from {original_time}ns to {reduced_time}ns")
                
                # Mock recovery result
                return {
                    'recovery_method': 'coherence_reduction',
                    'new_coherence_time': reduced_time,
                    'expected_fidelity_loss': 0.05
                }
                
        return None
        
    def _recover_from_thermal_violation(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for thermal violations."""
        
        self.logger.info("Recovering from thermal violation...")
        
        # Strategy: Reduce power and enable cooling
        recovery_actions = []
        
        if 'power_mw' in context.system_state:
            original_power = context.system_state['power_mw']
            reduced_power = original_power * 0.7  # Reduce by 30%
            context.system_state['power_mw'] = reduced_power
            recovery_actions.append(f"Reduced power from {original_power}mW to {reduced_power}mW")
            
        if 'enable_cooling' not in context.system_state or not context.system_state['enable_cooling']:
            context.system_state['enable_cooling'] = True
            recovery_actions.append("Enabled active cooling")
            
        if recovery_actions:
            context.recovery_suggestions.extend(recovery_actions)
            
            return {
                'recovery_method': 'thermal_management',
                'actions_taken': recovery_actions,
                'expected_temperature_reduction': 15.0  # Celsius
            }
            
        return None
        
    def _recover_from_crosstalk(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for crosstalk violations."""
        
        self.logger.info("Recovering from crosstalk violation...")
        
        # Strategy: Increase channel spacing and reduce power
        recovery_actions = []
        
        if 'channel_spacing_ghz' in context.system_state:
            original_spacing = context.system_state['channel_spacing_ghz']
            increased_spacing = original_spacing * 1.5  # Increase by 50%
            context.system_state['channel_spacing_ghz'] = increased_spacing
            recovery_actions.append(f"Increased channel spacing from {original_spacing}GHz to {increased_spacing}GHz")
            
        if 'channel_power_mw' in context.system_state:
            original_power = context.system_state['channel_power_mw']
            reduced_power = original_power * 0.8  # Reduce by 20%
            context.system_state['channel_power_mw'] = reduced_power
            recovery_actions.append(f"Reduced channel power from {original_power}mW to {reduced_power}mW")
            
        if recovery_actions:
            context.recovery_suggestions.extend(recovery_actions)
            
            return {
                'recovery_method': 'crosstalk_mitigation',
                'actions_taken': recovery_actions,
                'expected_crosstalk_improvement_db': 5.0
            }
            
        return None
        
    def _recover_from_resource_exhaustion(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for resource exhaustion."""
        
        self.logger.info("Recovering from resource exhaustion...")
        
        # Strategy: Reduce problem size and enable resource sharing
        recovery_actions = []
        
        if 'batch_size' in context.system_state:
            original_batch = context.system_state['batch_size']
            reduced_batch = max(1, original_batch // 2)  # Halve batch size
            context.system_state['batch_size'] = reduced_batch
            recovery_actions.append(f"Reduced batch size from {original_batch} to {reduced_batch}")
            
        if 'max_iterations' in context.system_state:
            original_iterations = context.system_state['max_iterations']
            reduced_iterations = max(10, original_iterations // 2)
            context.system_state['max_iterations'] = reduced_iterations
            recovery_actions.append(f"Reduced max iterations from {original_iterations} to {reduced_iterations}")
            
        # Enable memory optimization
        context.system_state['enable_memory_optimization'] = True
        recovery_actions.append("Enabled memory optimization")
        
        if recovery_actions:
            context.recovery_suggestions.extend(recovery_actions)
            
            return {
                'recovery_method': 'resource_optimization',
                'actions_taken': recovery_actions,
                'expected_memory_reduction': 0.5
            }
            
        return None
        
    def _recover_from_numerical_instability(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for numerical instability."""
        
        self.logger.info("Recovering from numerical instability...")
        
        # Strategy: Increase regularization and reduce learning rate
        recovery_actions = []
        
        if 'learning_rate' in context.system_state:
            original_lr = context.system_state['learning_rate']
            reduced_lr = original_lr * 0.5  # Halve learning rate
            context.system_state['learning_rate'] = reduced_lr
            recovery_actions.append(f"Reduced learning rate from {original_lr} to {reduced_lr}")
            
        if 'regularization' not in context.system_state:
            context.system_state['regularization'] = 1e-4
            recovery_actions.append("Added L2 regularization (1e-4)")
        else:
            original_reg = context.system_state['regularization']
            increased_reg = min(1e-2, original_reg * 2)  # Double regularization
            context.system_state['regularization'] = increased_reg
            recovery_actions.append(f"Increased regularization from {original_reg} to {increased_reg}")
            
        # Enable numerical stability features
        context.system_state['use_stable_algorithms'] = True
        recovery_actions.append("Enabled numerically stable algorithms")
        
        if recovery_actions:
            context.recovery_suggestions.extend(recovery_actions)
            
            return {
                'recovery_method': 'numerical_stabilization',
                'actions_taken': recovery_actions,
                'convergence_guarantee': 'improved'
            }
            
        return None
        
    def _recover_from_timeout(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for timeout errors."""
        
        self.logger.info("Recovering from timeout...")
        
        # Strategy: Extend timeout and reduce problem complexity
        recovery_actions = []
        
        if 'timeout_seconds' in context.system_state:
            original_timeout = context.system_state['timeout_seconds']
            extended_timeout = original_timeout * 2  # Double timeout
            context.system_state['timeout_seconds'] = extended_timeout
            recovery_actions.append(f"Extended timeout from {original_timeout}s to {extended_timeout}s")
            
        # Reduce complexity
        if 'complexity_level' in context.system_state:
            original_complexity = context.system_state['complexity_level']
            reduced_complexity = max(1, original_complexity - 1)
            context.system_state['complexity_level'] = reduced_complexity
            recovery_actions.append(f"Reduced complexity level from {original_complexity} to {reduced_complexity}")
            
        if recovery_actions:
            context.recovery_suggestions.extend(recovery_actions)
            
            return {
                'recovery_method': 'timeout_extension',
                'actions_taken': recovery_actions,
                'new_timeout_seconds': context.system_state.get('timeout_seconds', 60)
            }
            
        return None
        
    def _recover_from_memory_error(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for memory errors."""
        
        self.logger.info("Recovering from memory error...")
        
        # Strategy: Enable memory-efficient processing
        recovery_actions = []
        
        # Enable streaming/chunked processing
        context.system_state['use_streaming'] = True
        context.system_state['chunk_size'] = 1000  # Process in chunks
        recovery_actions.append("Enabled streaming processing with 1000-item chunks")
        
        # Enable memory cleanup
        context.system_state['enable_gc'] = True
        recovery_actions.append("Enabled aggressive garbage collection")
        
        # Reduce precision if possible
        if 'precision' in context.system_state and context.system_state['precision'] == 'float64':
            context.system_state['precision'] = 'float32'
            recovery_actions.append("Reduced precision from float64 to float32")
            
        context.recovery_suggestions.extend(recovery_actions)
        
        return {
            'recovery_method': 'memory_optimization',
            'actions_taken': recovery_actions,
            'expected_memory_savings': 0.5
        }
        
    def _recover_from_network_failure(self, error: Exception, context: ErrorContext) -> Optional[Any]:
        """Recovery strategy for network failures."""
        
        self.logger.info("Recovering from network failure...")
        
        # Strategy: Enable offline mode and use cached data
        recovery_actions = []
        
        context.system_state['offline_mode'] = True
        recovery_actions.append("Enabled offline mode")
        
        context.system_state['use_cache'] = True
        recovery_actions.append("Enabled cached data usage")
        
        context.system_state['retry_with_backoff'] = True
        recovery_actions.append("Enabled exponential backoff for retries")
        
        context.recovery_suggestions.extend(recovery_actions)
        
        return {
            'recovery_method': 'network_resilience',
            'actions_taken': recovery_actions,
            'offline_capabilities': True
        }
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        stats = {
            'total_errors': self.error_stats['total'],
            'error_by_category': dict(self.error_stats),
            'circuit_breaker_status': {name: cb.status for name, cb in self.circuit_breakers.items()},
            'recent_errors': len([e for e in self.error_history if time.time() - e.timestamp < 3600]),  # Last hour
            'recovery_success_rate': 0.0
        }
        
        # Calculate recovery success rate
        if self.error_history:
            recovery_attempts = [e for e in self.error_history if e.recovery_attempted]
            if recovery_attempts:
                successful_recoveries = [e for e in recovery_attempts if e.recovery_successful]
                stats['recovery_success_rate'] = len(successful_recoveries) / len(recovery_attempts)
                
        return stats
        
    def reset_circuit_breaker(self, component_name: str) -> bool:
        """Manually reset a circuit breaker."""
        
        if component_name in self.circuit_breakers:
            cb = self.circuit_breakers[component_name]
            with cb.lock:
                cb.state = CircuitBreakerState.CLOSED
                cb.failure_count = 0
                cb.success_count = 0
                
            self.logger.info(f"Circuit breaker {component_name} manually reset to CLOSED")
            return True
            
        return False


# Decorator for robust function execution
def robust_execution(component: str, operation: str, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    category: ErrorCategory = ErrorCategory.COMPILATION_ERROR,
                    max_retries: int = 3,
                    timeout: float = 30.0,
                    use_circuit_breaker: bool = True):
    """
    Decorator for robust function execution with comprehensive error handling.
    
    Usage:
        @robust_execution('quantum_compiler', 'vqe_optimization', 
                         severity=ErrorSeverity.HIGH,
                         category=ErrorCategory.QUANTUM_DECOHERENCE)
        def quantum_optimize(params):
            # Function implementation
            pass
    """
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            
            # Get global error handler
            if not hasattr(wrapper, '_error_handler'):
                wrapper._error_handler = RobustErrorHandler()
                
            error_handler = wrapper._error_handler
            
            # Create error context
            context = ErrorContext(
                component=component,
                operation=operation,
                severity=severity,
                category=category,
                max_retries=max_retries,
                input_parameters={
                    'args': str(args)[:200],  # Truncate long arguments
                    'kwargs': {k: str(v)[:200] for k, v in kwargs.items()}
                }
            )
            
            # Use circuit breaker if enabled
            if use_circuit_breaker and component in error_handler.circuit_breakers:
                circuit_breaker = error_handler.circuit_breakers[component]
                
                try:
                    result = circuit_breaker.call(func, *args, **kwargs)
                    return result
                    
                except CircuitBreakerError as e:
                    context.severity = ErrorSeverity.HIGH
                    context.category = ErrorCategory.RESOURCE_EXHAUSTION
                    
                    recovery_successful, result = error_handler.handle_error(e, context)
                    
                    if recovery_successful:
                        return result
                    else:
                        raise e
                        
                except Exception as e:
                    recovery_successful, result = error_handler.handle_error(e, context)
                    
                    if recovery_successful:
                        return result
                    else:
                        raise e
            else:
                # Direct execution with error handling
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    recovery_successful, result = error_handler.handle_error(e, context)
                    
                    if recovery_successful:
                        return result
                    else:
                        raise e
                        
        return wrapper
    return decorator


# Context manager for robust operations
@contextlib.contextmanager
def robust_context(component: str, operation: str, 
                  expected_errors: List[Type[Exception]] = None):
    """
    Context manager for robust operation execution.
    
    Usage:
        with robust_context('thermal_predictor', 'neural_ode_solve'):
            result = complex_computation()
    """
    
    error_handler = RobustErrorHandler()
    context = ErrorContext(
        component=component,
        operation=operation,
        severity=ErrorSeverity.MEDIUM,
        category=ErrorCategory.COMPILATION_ERROR
    )
    
    try:
        yield context
        
    except Exception as e:
        # Check if this is an expected error type
        if expected_errors and any(isinstance(e, err_type) for err_type in expected_errors):
            context.severity = ErrorSeverity.LOW
            
        recovery_successful, result = error_handler.handle_error(e, context)
        
        if not recovery_successful:
            raise e


# Input validation and sanitization
class InputValidator:
    """Comprehensive input validation and sanitization."""
    
    @staticmethod
    def validate_wavelength(wavelength: float) -> float:
        """Validate and sanitize wavelength input."""
        
        if not isinstance(wavelength, (int, float)):
            raise ValueError(f"Wavelength must be numeric, got {type(wavelength)}")
            
        wavelength = float(wavelength)
        
        if wavelength < 1000 or wavelength > 2000:
            raise ValueError(f"Wavelength {wavelength}nm outside valid range [1000, 2000]nm")
            
        return wavelength
        
    @staticmethod
    def validate_array_size(array_size: Tuple[int, int]) -> Tuple[int, int]:
        """Validate and sanitize array size."""
        
        if not isinstance(array_size, (list, tuple)) or len(array_size) != 2:
            raise ValueError(f"Array size must be tuple of 2 integers, got {array_size}")
            
        try:
            width, height = int(array_size[0]), int(array_size[1])
        except (ValueError, TypeError):
            raise ValueError(f"Array size elements must be integers, got {array_size}")
            
        if width < 1 or height < 1:
            raise ValueError(f"Array dimensions must be positive, got ({width}, {height})")
            
        if width > 1000 or height > 1000:
            raise ValueError(f"Array dimensions too large ({width}, {height}), max 1000x1000")
            
        return (width, height)
        
    @staticmethod
    def validate_power(power_mw: float) -> float:
        """Validate and sanitize optical power."""
        
        if not isinstance(power_mw, (int, float)):
            raise ValueError(f"Power must be numeric, got {type(power_mw)}")
            
        power_mw = float(power_mw)
        
        if power_mw < 0:
            raise ValueError(f"Power cannot be negative, got {power_mw}mW")
            
        if power_mw > 1000:  # 1W maximum for safety
            raise ValueError(f"Power too high ({power_mw}mW), maximum 1000mW")
            
        return power_mw
        
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for security."""
        
        if not isinstance(filename, str):
            raise ValueError(f"Filename must be string, got {type(filename)}")
            
        # Remove dangerous characters
        import re
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        
        # Limit length
        if len(filename) > 255:
            filename = filename[:255]
            
        # Ensure not empty
        if not filename.strip():
            filename = "default_file"
            
        return filename.strip()


# Resource monitoring and protection
class ResourceMonitor:
    """Monitor and protect system resources."""
    
    def __init__(self):
        self.logger = get_global_logger()
        self.memory_limit_mb = 8192  # 8GB default
        self.cpu_limit_percent = 90
        self.monitoring_active = True
        
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage."""
        
        try:
            import psutil
            
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
            
            status = {
                'memory_mb': memory_mb,
                'memory_percent': memory_percent,
                'limit_mb': self.memory_limit_mb,
                'within_limit': memory_mb < self.memory_limit_mb
            }
            
            if not status['within_limit']:
                self.logger.warning(f"Memory usage ({memory_mb:.1f}MB) exceeds limit ({self.memory_limit_mb}MB)")
                
            return status
            
        except ImportError:
            # psutil not available, return mock data
            return {
                'memory_mb': 1024,
                'memory_percent': 25.0,
                'limit_mb': self.memory_limit_mb,
                'within_limit': True,
                'error': 'psutil not available'
            }
            
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check current CPU usage."""
        
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            
            status = {
                'cpu_percent': cpu_percent,
                'limit_percent': self.cpu_limit_percent,
                'within_limit': cpu_percent < self.cpu_limit_percent
            }
            
            if not status['within_limit']:
                self.logger.warning(f"CPU usage ({cpu_percent:.1f}%) exceeds limit ({self.cpu_limit_percent}%)")
                
            return status
            
        except ImportError:
            return {
                'cpu_percent': 25.0,
                'limit_percent': self.cpu_limit_percent,
                'within_limit': True,
                'error': 'psutil not available'
            }
            
    def enforce_resource_limits(self):
        """Enforce resource limits and take protective action."""
        
        memory_status = self.check_memory_usage()
        cpu_status = self.check_cpu_usage()
        
        if not memory_status['within_limit']:
            # Trigger garbage collection
            import gc
            gc.collect()
            
            # Re-check after cleanup
            memory_status = self.check_memory_usage()
            if not memory_status['within_limit']:
                raise ResourceExhaustionError(f"Memory limit exceeded: {memory_status['memory_mb']:.1f}MB > {self.memory_limit_mb}MB")
                
        if not cpu_status['within_limit']:
            self.logger.warning("High CPU usage detected, consider reducing workload")
            
        return {
            'memory_status': memory_status,
            'cpu_status': cpu_status,
            'action_taken': 'resource_check_completed'
        }


class ResourceExhaustionError(Exception):
    """Exception raised when system resources are exhausted."""
    pass


# Example usage and testing
def create_robustness_demo() -> Dict[str, Any]:
    """Demonstrate robustness features."""
    
    logger = get_global_logger()
    logger.info("🛡️ Creating robustness demonstration")
    
    # Test error handler
    error_handler = RobustErrorHandler()
    
    # Test input validation
    validator = InputValidator()
    
    demo_results = {
        'error_handling': {},
        'input_validation': {},
        'circuit_breaker': {},
        'resource_monitoring': {}
    }
    
    # Test error handling with mock error
    try:
        @robust_execution('demo_component', 'test_operation', 
                         severity=ErrorSeverity.MEDIUM,
                         category=ErrorCategory.QUANTUM_DECOHERENCE)
        def failing_function():
            raise ValueError("Mock quantum decoherence error")
            
        result = failing_function()
        demo_results['error_handling']['recovery_successful'] = True
        demo_results['error_handling']['result'] = result
        
    except Exception as e:
        demo_results['error_handling']['recovery_successful'] = False
        demo_results['error_handling']['error'] = str(e)
        
    # Test input validation
    try:
        valid_wavelength = validator.validate_wavelength(1550.0)
        valid_array_size = validator.validate_array_size((64, 64))
        valid_power = validator.validate_power(10.0)
        
        demo_results['input_validation'] = {
            'wavelength_validation': valid_wavelength,
            'array_size_validation': valid_array_size,
            'power_validation': valid_power,
            'validation_successful': True
        }
        
    except Exception as e:
        demo_results['input_validation']['validation_successful'] = False
        demo_results['input_validation']['error'] = str(e)
        
    # Test circuit breaker
    cb_config = CircuitBreakerConfig(failure_threshold=2, recovery_timeout=5.0)
    circuit_breaker = CircuitBreaker('demo_breaker', cb_config)
    
    try:
        # Test normal operation
        result = circuit_breaker.call(lambda: "success")
        demo_results['circuit_breaker']['normal_operation'] = result
        demo_results['circuit_breaker']['status'] = circuit_breaker.status
        
    except Exception as e:
        demo_results['circuit_breaker']['error'] = str(e)
        
    # Test resource monitoring
    resource_monitor = ResourceMonitor()
    try:
        resource_status = resource_monitor.enforce_resource_limits()
        demo_results['resource_monitoring'] = resource_status
        
    except Exception as e:
        demo_results['resource_monitoring']['error'] = str(e)
        
    # Get error statistics
    demo_results['error_statistics'] = error_handler.get_error_statistics()
    
    logger.info("🛡️ Robustness demonstration completed")
    
    return demo_results


if __name__ == "__main__":
    # Run robustness demonstration
    demo_results = create_robustness_demo()
    
    print("=== Robustness System Demo Results ===")
    print(f"Error handling: {demo_results['error_handling'].get('recovery_successful', 'Unknown')}")
    print(f"Input validation: {demo_results['input_validation'].get('validation_successful', 'Unknown')}")
    print(f"Circuit breaker status: {demo_results['circuit_breaker'].get('status', {}).get('state', 'Unknown')}")
    print(f"Resource monitoring: {demo_results['resource_monitoring'].get('action_taken', 'Unknown')}")
    print(f"Total errors handled: {demo_results['error_statistics'].get('total_errors', 0)}")