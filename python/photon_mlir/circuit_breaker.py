"""
Circuit breaker patterns for thermal management and system protection.
Generation 2: Robust error handling and automated recovery.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque
import statistics

from .core import TargetConfig
from .thermal_optimization import ThermalModel


class CircuitState(Enum):
    """States of the circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, blocking requests
    HALF_OPEN = "half_open"  # Testing if service is back


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker."""
    THERMAL_VIOLATION = "thermal_violation"
    PHASE_COHERENCE_LOSS = "phase_coherence_loss"
    POWER_OVERLOAD = "power_overload"
    COMPILATION_TIMEOUT = "compilation_timeout"
    HARDWARE_FAULT = "hardware_fault"
    QUANTUM_DECOHERENCE = "quantum_decoherence"
    MESH_CALIBRATION_FAILURE = "mesh_calibration_failure"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker behavior."""
    failure_threshold: int = 5  # Number of failures before opening circuit
    recovery_timeout_s: float = 60.0  # Time to wait before attempting recovery
    half_open_max_calls: int = 3  # Max calls to test during half-open state
    success_threshold: int = 2  # Successes needed to close circuit from half-open
    monitoring_window_s: float = 300.0  # Window for failure rate calculation
    thermal_limit_celsius: float = 85.0  # Thermal protection limit
    phase_error_threshold: float = 0.1  # Phase coherence threshold (radians)
    power_limit_mw: float = 100.0  # Power limit for protection
    enable_adaptive_thresholds: bool = True  # Enable adaptive threshold adjustment


@dataclass
class FailureRecord:
    """Record of a failure event."""
    timestamp: float
    failure_type: FailureType
    severity: int  # 1 (low) to 5 (critical)
    context: Dict[str, Any] = field(default_factory=dict)
    recovery_attempted: bool = False
    recovery_successful: bool = False


class ThermalCircuitBreaker:
    """
    Circuit breaker specifically for thermal management.
    
    Monitors thermal conditions and prevents operations that could
    lead to thermal damage or performance degradation.
    """
    
    def __init__(self, config: CircuitBreakerConfig, thermal_model: ThermalModel):
        self.config = config
        self.thermal_model = thermal_model
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        self.half_open_calls = 0
        self.success_count = 0
        
        self.failure_history: deque[FailureRecord] = deque(maxlen=1000)
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.ThermalCircuitBreaker")
        
        # Thermal monitoring
        self.current_temperature = 25.0  # Celsius
        self.temperature_trend = 0.0  # Rate of change
        self.thermal_budget_mw = 1000.0  # Available thermal budget
        self.cooldown_active = False
        
    def can_execute_operation(self, operation_thermal_cost: float) -> bool:
        """Check if an operation can be executed without thermal violation."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout_s:
                    return False
                else:
                    # Transition to half-open state
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    self.success_count = 0
                    self.logger.info("Circuit breaker transitioning to HALF_OPEN state")
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls >= self.config.half_open_max_calls:
                    return False
            
            # Thermal safety check
            predicted_temperature = self._predict_temperature_rise(operation_thermal_cost)
            if predicted_temperature > self.config.thermal_limit_celsius:
                self._record_thermal_failure(predicted_temperature)
                return False
            
            # Thermal budget check
            if operation_thermal_cost > self.thermal_budget_mw:
                self._record_power_failure(operation_thermal_cost)
                return False
            
            return True
    
    def record_success(self):
        """Record a successful operation."""
        with self.lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    self.logger.info("Circuit breaker closed after successful recovery")
    
    def record_failure(self, failure_type: FailureType, severity: int = 3, 
                      context: Optional[Dict[str, Any]] = None):
        """Record a failure and potentially open the circuit."""
        with self.lock:
            failure = FailureRecord(
                timestamp=time.time(),
                failure_type=failure_type,
                severity=severity,
                context=context or {}
            )
            
            self.failure_history.append(failure)
            self.failure_count += 1
            self.last_failure_time = failure.timestamp
            
            # Check if we should open the circuit
            if self._should_open_circuit():
                self.state = CircuitState.OPEN
                self.logger.warning(f"Circuit breaker opened due to {failure_type.value} "
                                  f"(failures: {self.failure_count})")
                
                # Trigger cooldown for thermal failures
                if failure_type == FailureType.THERMAL_VIOLATION:
                    self._trigger_thermal_cooldown()
    
    def force_cooldown(self, duration_s: float = 300.0):
        """Force thermal cooldown for specified duration."""
        with self.lock:
            self.cooldown_active = True
            self.state = CircuitState.OPEN
            self.logger.info(f"Forced thermal cooldown for {duration_s}s")
            
            # Schedule cooldown completion
            def complete_cooldown():
                time.sleep(duration_s)
                with self.lock:
                    self.cooldown_active = False
                    self.current_temperature *= 0.7  # Simulate cooling
                    self.temperature_trend = -0.1  # Cooling trend
                    self.thermal_budget_mw = min(1000.0, self.thermal_budget_mw * 1.5)
                    self.logger.info("Thermal cooldown completed")
            
            threading.Thread(target=complete_cooldown, daemon=True).start()
    
    def get_thermal_status(self) -> Dict[str, Any]:
        """Get current thermal status."""
        with self.lock:
            recent_failures = [f for f in self.failure_history 
                             if time.time() - f.timestamp < self.config.monitoring_window_s]
            
            thermal_failures = [f for f in recent_failures 
                              if f.failure_type == FailureType.THERMAL_VIOLATION]
            
            return {
                "state": self.state.value,
                "current_temperature_c": self.current_temperature,
                "temperature_trend": self.temperature_trend,
                "thermal_budget_mw": self.thermal_budget_mw,
                "failure_count": self.failure_count,
                "thermal_failures_recent": len(thermal_failures),
                "cooldown_active": self.cooldown_active,
                "time_to_recovery_s": max(0, self.config.recovery_timeout_s - 
                                        (time.time() - self.last_failure_time))
            }
    
    def _predict_temperature_rise(self, thermal_cost: float) -> float:
        """Predict temperature rise from thermal cost."""
        # Simplified thermal model
        base_rise = thermal_cost * 0.01  # 0.01°C per mW
        trend_factor = max(1.0, 1.0 + self.temperature_trend)
        return self.current_temperature + (base_rise * trend_factor)
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened."""
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Adaptive threshold based on recent failure rate
        if self.config.enable_adaptive_thresholds:
            recent_failures = [f for f in self.failure_history 
                             if time.time() - f.timestamp < self.config.monitoring_window_s]
            
            if len(recent_failures) >= self.config.failure_threshold // 2:
                # High failure rate, lower threshold temporarily
                return True
        
        return False
    
    def _record_thermal_failure(self, predicted_temp: float):
        """Record thermal violation failure."""
        context = {
            "predicted_temperature": predicted_temp,
            "current_temperature": self.current_temperature,
            "temperature_limit": self.config.thermal_limit_celsius,
            "thermal_budget": self.thermal_budget_mw
        }
        
        severity = 5 if predicted_temp > self.config.thermal_limit_celsius + 10 else 3
        self.record_failure(FailureType.THERMAL_VIOLATION, severity, context)
    
    def _record_power_failure(self, power_demand: float):
        """Record power overload failure."""
        context = {
            "power_demand_mw": power_demand,
            "thermal_budget_mw": self.thermal_budget_mw,
            "power_limit_mw": self.config.power_limit_mw
        }
        
        self.record_failure(FailureType.POWER_OVERLOAD, 3, context)
    
    def _trigger_thermal_cooldown(self):
        """Trigger thermal cooldown sequence."""
        if not self.cooldown_active:
            cooldown_duration = min(300.0, self.failure_count * 30.0)  # Up to 5 min
            self.force_cooldown(cooldown_duration)


class PhaseCoherenceCircuitBreaker:
    """
    Circuit breaker for phase coherence protection.
    
    Monitors phase coherence and prevents operations that could
    lead to quantum decoherence or phase errors.
    """
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0
        
        self.phase_error_history: deque[float] = deque(maxlen=100)
        self.coherence_time_ms = 1000.0  # Quantum coherence time
        self.phase_noise_std = 0.01  # Standard deviation of phase noise
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.PhaseCoherenceCircuitBreaker")
    
    def can_execute_phase_operation(self, phase_change: float, duration_ms: float) -> bool:
        """Check if a phase operation can be executed safely."""
        with self.lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time < self.config.recovery_timeout_s:
                    return False
                else:
                    self.state = CircuitState.HALF_OPEN
            
            # Phase error prediction
            predicted_error = self._predict_phase_error(phase_change, duration_ms)
            
            if predicted_error > self.config.phase_error_threshold:
                context = {
                    "predicted_error": predicted_error,
                    "threshold": self.config.phase_error_threshold,
                    "phase_change": phase_change,
                    "duration_ms": duration_ms
                }
                
                failure = FailureRecord(
                    timestamp=time.time(),
                    failure_type=FailureType.PHASE_COHERENCE_LOSS,
                    severity=4,
                    context=context
                )
                
                self._record_failure(failure)
                return False
            
            # Coherence time check
            if duration_ms > self.coherence_time_ms:
                context = {
                    "duration_ms": duration_ms,
                    "coherence_time_ms": self.coherence_time_ms,
                    "coherence_violation": True
                }
                
                failure = FailureRecord(
                    timestamp=time.time(),
                    failure_type=FailureType.QUANTUM_DECOHERENCE,
                    severity=3,
                    context=context
                )
                
                self._record_failure(failure)
                return False
            
            return True
    
    def record_phase_measurement(self, measured_phase: float, expected_phase: float):
        """Record phase measurement for error tracking."""
        phase_error = abs(measured_phase - expected_phase)
        self.phase_error_history.append(phase_error)
        
        # Update phase noise statistics
        if len(self.phase_error_history) > 10:
            self.phase_noise_std = statistics.stdev(self.phase_error_history)
    
    def _predict_phase_error(self, phase_change: float, duration_ms: float) -> float:
        """Predict phase error based on operation parameters."""
        # Base error from phase noise
        base_error = self.phase_noise_std * (duration_ms / 1000.0) ** 0.5
        
        # Additional error from large phase changes
        change_error = abs(phase_change) * 0.01
        
        # Decoherence contribution
        decoherence_factor = duration_ms / self.coherence_time_ms
        decoherence_error = decoherence_factor * 0.05
        
        return base_error + change_error + decoherence_error
    
    def _record_failure(self, failure: FailureRecord):
        """Record failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = failure.timestamp
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            self.logger.warning(f"Phase coherence circuit breaker opened "
                              f"(failures: {self.failure_count})")


class CompositePhotonicCircuitBreaker:
    """
    Composite circuit breaker that manages multiple protection mechanisms.
    
    Coordinates between thermal, phase, and other circuit breakers to
    provide comprehensive system protection.
    """
    
    def __init__(self, config: CircuitBreakerConfig, target_config: TargetConfig):
        self.config = config
        self.target_config = target_config
        
        # Initialize sub-breakers
        thermal_model = ThermalModel(target_config)
        self.thermal_breaker = ThermalCircuitBreaker(config, thermal_model)
        self.phase_breaker = PhaseCoherenceCircuitBreaker(config)
        
        self.lock = threading.RLock()
        self.logger = logging.getLogger(f"{__name__}.CompositeCircuitBreaker")
        
        # System-wide state
        self.system_health_score = 1.0  # 0.0 (critical) to 1.0 (excellent)
        self.adaptive_protection_enabled = True
        self.emergency_shutdown_active = False
        
        # Metrics
        self.total_operations = 0
        self.blocked_operations = 0
        self.successful_operations = 0
    
    def can_execute_operation(self, operation_type: str, **kwargs) -> bool:
        """
        Check if an operation can be executed across all protection mechanisms.
        
        Args:
            operation_type: Type of operation (matmul, phase_shift, etc.)
            **kwargs: Operation parameters
        
        Returns:
            bool: True if operation can be executed safely
        """
        with self.lock:
            self.total_operations += 1
            
            # Emergency shutdown check
            if self.emergency_shutdown_active:
                self.blocked_operations += 1
                return False
            
            # System health check
            if self.system_health_score < 0.3:
                self.logger.warning("System health critical, blocking operation")
                self.blocked_operations += 1
                return False
            
            # Operation-specific checks
            if operation_type in ["matmul", "wavelength_multiplex", "mesh_calibration"]:
                thermal_cost = kwargs.get("thermal_cost", 50.0)  # mW
                if not self.thermal_breaker.can_execute_operation(thermal_cost):
                    self.blocked_operations += 1
                    return False
            
            if operation_type in ["phase_shift", "quantum_phase_gate"]:
                phase_change = kwargs.get("phase_change", 0.0)
                duration_ms = kwargs.get("duration_ms", 10.0)
                if not self.phase_breaker.can_execute_phase_operation(phase_change, duration_ms):
                    self.blocked_operations += 1
                    return False
            
            # Adaptive protection
            if self.adaptive_protection_enabled:
                if not self._adaptive_safety_check(operation_type, **kwargs):
                    self.blocked_operations += 1
                    return False
            
            self.successful_operations += 1
            return True
    
    def record_operation_result(self, operation_type: str, success: bool, 
                               metrics: Optional[Dict[str, Any]] = None):
        """Record the result of an operation for learning."""
        if success:
            self.thermal_breaker.record_success()
            self._update_system_health(0.01)  # Small positive increment
        else:
            # Determine failure type and record appropriately
            failure_type = self._classify_failure(operation_type, metrics or {})
            self.thermal_breaker.record_failure(failure_type)
            self._update_system_health(-0.05)  # Negative impact on health
    
    def trigger_emergency_shutdown(self, reason: str = "Manual trigger"):
        """Trigger emergency shutdown of all operations."""
        with self.lock:
            self.emergency_shutdown_active = True
            self.thermal_breaker.force_cooldown(600.0)  # 10-minute cooldown
            self.logger.critical(f"Emergency shutdown triggered: {reason}")
    
    def reset_emergency_shutdown(self):
        """Reset emergency shutdown state."""
        with self.lock:
            self.emergency_shutdown_active = False
            self.system_health_score = max(0.5, self.system_health_score)
            self.logger.info("Emergency shutdown reset")
    
    def get_comprehensive_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all circuit breakers."""
        with self.lock:
            return {
                "system_health_score": self.system_health_score,
                "emergency_shutdown_active": self.emergency_shutdown_active,
                "adaptive_protection_enabled": self.adaptive_protection_enabled,
                "thermal_status": self.thermal_breaker.get_thermal_status(),
                "phase_status": {
                    "state": self.phase_breaker.state.value,
                    "failure_count": self.phase_breaker.failure_count,
                    "coherence_time_ms": self.phase_breaker.coherence_time_ms,
                    "phase_noise_std": self.phase_breaker.phase_noise_std
                },
                "operation_metrics": {
                    "total_operations": self.total_operations,
                    "blocked_operations": self.blocked_operations,
                    "successful_operations": self.successful_operations,
                    "success_rate": self.successful_operations / max(1, self.total_operations)
                }
            }
    
    def _adaptive_safety_check(self, operation_type: str, **kwargs) -> bool:
        """Adaptive safety check based on system health and recent history."""
        # More restrictive when system health is low
        health_factor = max(0.1, self.system_health_score)
        
        # Check operation frequency
        if operation_type == "mesh_calibration" and self.system_health_score < 0.7:
            return False  # Skip calibration when health is low
        
        # Thermal budget scaling
        if "thermal_cost" in kwargs:
            scaled_thermal_cost = kwargs["thermal_cost"] / health_factor
            return scaled_thermal_cost <= 100.0  # Scaled thermal limit
        
        return True
    
    def _classify_failure(self, operation_type: str, metrics: Dict[str, Any]) -> FailureType:
        """Classify failure based on operation type and metrics."""
        if "temperature" in metrics and metrics["temperature"] > 80.0:
            return FailureType.THERMAL_VIOLATION
        
        if "phase_error" in metrics and metrics["phase_error"] > 0.1:
            return FailureType.PHASE_COHERENCE_LOSS
        
        if "power" in metrics and metrics["power"] > 100.0:
            return FailureType.POWER_OVERLOAD
        
        if operation_type in ["mesh_calibration"]:
            return FailureType.MESH_CALIBRATION_FAILURE
        
        return FailureType.HARDWARE_FAULT
    
    def _update_system_health(self, delta: float):
        """Update system health score with decay."""
        # Apply health change
        self.system_health_score += delta
        self.system_health_score = max(0.0, min(1.0, self.system_health_score))
        
        # Natural decay towards baseline (0.8)
        baseline = 0.8
        decay_factor = 0.001
        if self.system_health_score > baseline:
            self.system_health_score -= decay_factor
        elif self.system_health_score < baseline:
            self.system_health_score += decay_factor


# Utility functions for easy integration
def create_thermal_protection(target_config: TargetConfig, 
                             thermal_limit: float = 85.0) -> ThermalCircuitBreaker:
    """Create a thermal circuit breaker with standard configuration."""
    config = CircuitBreakerConfig(
        thermal_limit_celsius=thermal_limit,
        failure_threshold=3,
        recovery_timeout_s=60.0
    )
    
    thermal_model = ThermalModel(target_config)
    return ThermalCircuitBreaker(config, thermal_model)


def create_comprehensive_protection(target_config: TargetConfig) -> CompositePhotonicCircuitBreaker:
    """Create comprehensive protection with all circuit breakers."""
    config = CircuitBreakerConfig(
        failure_threshold=5,
        recovery_timeout_s=120.0,
        enable_adaptive_thresholds=True,
        thermal_limit_celsius=85.0,
        phase_error_threshold=0.05,
        power_limit_mw=100.0
    )
    
    return CompositePhotonicCircuitBreaker(config, target_config)


# Context manager for protected operations
class ProtectedOperation:
    """Context manager for executing operations with circuit breaker protection."""
    
    def __init__(self, circuit_breaker: CompositePhotonicCircuitBreaker, 
                 operation_type: str, **operation_params):
        self.breaker = circuit_breaker
        self.operation_type = operation_type
        self.operation_params = operation_params
        self.allowed = False
        self.start_time = 0.0
    
    def __enter__(self):
        self.allowed = self.breaker.can_execute_operation(
            self.operation_type, **self.operation_params
        )
        
        if not self.allowed:
            raise RuntimeError(f"Operation {self.operation_type} blocked by circuit breaker")
        
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        execution_time = time.time() - self.start_time
        
        metrics = {
            "execution_time_ms": execution_time * 1000,
            "exception_type": str(exc_type) if exc_type else None
        }
        
        self.breaker.record_operation_result(
            self.operation_type, success, metrics
        )