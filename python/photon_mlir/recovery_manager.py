"""
Automated recovery mechanisms for photonic compiler failures.
Generation 2: Self-healing and adaptive recovery strategies.
"""

import time
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from abc import ABC, abstractmethod
import json
import pickle
from pathlib import Path

from .core import TargetConfig
from .circuit_breaker import FailureType, CircuitBreakerConfig, CompositePhotonicCircuitBreaker


class RecoveryStrategy(Enum):
    """Available recovery strategies."""
    RETRY_WITH_BACKOFF = "retry_with_backoff"
    THERMAL_COOLDOWN = "thermal_cooldown"
    CONFIGURATION_ROLLBACK = "configuration_rollback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CHECKPOINT_RESTORE = "checkpoint_restore"
    ADAPTIVE_RECONFIGURATION = "adaptive_reconfiguration"
    EMERGENCY_SAFE_MODE = "emergency_safe_mode"


class RecoveryResult(Enum):
    """Results of recovery attempts."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    RETRY_NEEDED = "retry_needed"
    ESCALATION_REQUIRED = "escalation_required"


@dataclass
class RecoveryContext:
    """Context information for recovery operations."""
    failure_type: FailureType
    failure_timestamp: float
    failure_details: Dict[str, Any]
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    escalation_threshold: int = 2
    current_strategy: Optional[RecoveryStrategy] = None
    checkpoint_data: Optional[Dict[str, Any]] = None
    system_state: Dict[str, Any] = field(default_factory=dict)


class RecoveryAction(ABC):
    """Abstract base class for recovery actions."""
    
    @abstractmethod
    def can_handle(self, failure_type: FailureType, context: RecoveryContext) -> bool:
        """Check if this action can handle the given failure type."""
        pass
    
    @abstractmethod
    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute the recovery action."""
        pass
    
    @abstractmethod
    def estimate_recovery_time(self, context: RecoveryContext) -> float:
        """Estimate recovery time in seconds."""
        pass


class ThermalRecoveryAction(RecoveryAction):
    """Recovery action for thermal violations."""
    
    def __init__(self, cooldown_duration: float = 300.0):
        self.cooldown_duration = cooldown_duration
        self.logger = logging.getLogger(f"{__name__}.ThermalRecoveryAction")
    
    def can_handle(self, failure_type: FailureType, context: RecoveryContext) -> bool:
        return failure_type in [FailureType.THERMAL_VIOLATION, FailureType.POWER_OVERLOAD]
    
    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute thermal recovery by cooling down the system."""
        try:
            self.logger.info("Starting thermal recovery sequence")
            
            # Calculate adaptive cooldown duration
            severity = context.failure_details.get("severity", 3)
            adaptive_duration = self.cooldown_duration * (severity / 3.0)
            
            # Simulate cooling process
            current_temp = context.system_state.get("temperature", 85.0)
            target_temp = 40.0  # Target temperature after cooldown
            
            cooling_steps = 10
            step_duration = adaptive_duration / cooling_steps
            
            for i in range(cooling_steps):
                # Exponential cooling model
                progress = (i + 1) / cooling_steps
                new_temp = target_temp + (current_temp - target_temp) * (1 - progress) ** 2
                
                context.system_state["temperature"] = new_temp
                context.system_state["cooling_progress"] = progress
                
                self.logger.debug(f"Cooling step {i+1}/{cooling_steps}: {new_temp:.1f}°C")
                time.sleep(step_duration)
                
                # Check for early termination if temperature is acceptable
                if new_temp < 50.0 and i > cooling_steps // 2:
                    self.logger.info(f"Early cooldown completion at {new_temp:.1f}°C")
                    break
            
            # Verify recovery success
            final_temp = context.system_state.get("temperature", target_temp)
            if final_temp < 60.0:
                self.logger.info(f"Thermal recovery successful: {final_temp:.1f}°C")
                return RecoveryResult.SUCCESS
            else:
                self.logger.warning(f"Thermal recovery incomplete: {final_temp:.1f}°C")
                return RecoveryResult.PARTIAL_SUCCESS
                
        except Exception as e:
            self.logger.error(f"Thermal recovery failed: {e}")
            return RecoveryResult.FAILURE
    
    def estimate_recovery_time(self, context: RecoveryContext) -> float:
        severity = context.failure_details.get("severity", 3)
        return self.cooldown_duration * (severity / 3.0)


class PhaseRecoveryAction(RecoveryAction):
    """Recovery action for phase coherence issues."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PhaseRecoveryAction")
    
    def can_handle(self, failure_type: FailureType, context: RecoveryContext) -> bool:
        return failure_type in [FailureType.PHASE_COHERENCE_LOSS, FailureType.QUANTUM_DECOHERENCE]
    
    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute phase recovery by recalibrating phase shifters."""
        try:
            self.logger.info("Starting phase coherence recovery")
            
            # Phase correction algorithm
            phase_error = context.failure_details.get("predicted_error", 0.1)
            correction_factor = min(0.5, phase_error * 2.0)
            
            # Simulate phase recalibration
            calibration_points = [
                {"phase": 0.0, "correction": 0.0},
                {"phase": 1.57, "correction": correction_factor * 0.3},
                {"phase": 3.14, "correction": correction_factor * 0.6},
                {"phase": 4.71, "correction": correction_factor * 0.2}
            ]
            
            self.logger.debug(f"Applying phase corrections with factor {correction_factor}")
            
            # Apply corrections and verify
            total_correction = sum(point["correction"] for point in calibration_points)
            
            if total_correction < phase_error * 0.5:
                context.system_state["phase_coherence_restored"] = True
                context.system_state["residual_phase_error"] = phase_error - total_correction
                self.logger.info("Phase coherence recovery successful")
                return RecoveryResult.SUCCESS
            else:
                self.logger.warning("Phase recovery incomplete, may need hardware recalibration")
                return RecoveryResult.PARTIAL_SUCCESS
                
        except Exception as e:
            self.logger.error(f"Phase recovery failed: {e}")
            return RecoveryResult.FAILURE
    
    def estimate_recovery_time(self, context: RecoveryContext) -> float:
        return 30.0  # Phase recalibration typically takes 30 seconds


class ConfigurationRollbackAction(RecoveryAction):
    """Recovery action that rolls back to a known good configuration."""
    
    def __init__(self, checkpoint_manager):
        self.checkpoint_manager = checkpoint_manager
        self.logger = logging.getLogger(f"{__name__}.ConfigurationRollbackAction")
    
    def can_handle(self, failure_type: FailureType, context: RecoveryContext) -> bool:
        return (context.checkpoint_data is not None and 
                failure_type in [FailureType.COMPILATION_TIMEOUT, FailureType.HARDWARE_FAULT])
    
    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute configuration rollback to last known good state."""
        try:
            self.logger.info("Starting configuration rollback")
            
            if not context.checkpoint_data:
                self.logger.error("No checkpoint data available for rollback")
                return RecoveryResult.FAILURE
            
            # Restore configuration from checkpoint
            restored_config = self.checkpoint_manager.restore_configuration(
                context.checkpoint_data
            )
            
            if restored_config:
                context.system_state["configuration_restored"] = True
                context.system_state["rollback_timestamp"] = time.time()
                self.logger.info("Configuration rollback successful")
                return RecoveryResult.SUCCESS
            else:
                self.logger.error("Failed to restore configuration from checkpoint")
                return RecoveryResult.FAILURE
                
        except Exception as e:
            self.logger.error(f"Configuration rollback failed: {e}")
            return RecoveryResult.FAILURE
    
    def estimate_recovery_time(self, context: RecoveryContext) -> float:
        return 10.0  # Configuration rollback is usually quick


class AdaptiveReconfigurationAction(RecoveryAction):
    """Recovery action that adaptively reconfigures system parameters."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.AdaptiveReconfigurationAction")
    
    def can_handle(self, failure_type: FailureType, context: RecoveryContext) -> bool:
        return context.recovery_attempts > 0  # Try after initial recovery attempts
    
    def execute(self, context: RecoveryContext) -> RecoveryResult:
        """Execute adaptive reconfiguration based on failure pattern."""
        try:
            self.logger.info("Starting adaptive reconfiguration")
            
            reconfig_changes = {}
            
            # Adapt based on failure type
            if context.failure_type == FailureType.THERMAL_VIOLATION:
                # Reduce power limits and increase cooling intervals
                reconfig_changes.update({
                    "max_power_mw": context.system_state.get("max_power_mw", 100) * 0.8,
                    "thermal_limit": context.system_state.get("thermal_limit", 85) - 5,
                    "cooling_interval": context.system_state.get("cooling_interval", 60) * 1.5
                })
                
            elif context.failure_type == FailureType.PHASE_COHERENCE_LOSS:
                # Tighten phase tolerances and increase calibration frequency
                reconfig_changes.update({
                    "phase_tolerance": context.system_state.get("phase_tolerance", 0.1) * 0.7,
                    "calibration_frequency": context.system_state.get("calibration_frequency", 10) * 1.3
                })
                
            elif context.failure_type == FailureType.COMPILATION_TIMEOUT:
                # Reduce complexity limits
                reconfig_changes.update({
                    "max_matrix_size": context.system_state.get("max_matrix_size", 1024) // 2,
                    "optimization_level": max(1, context.system_state.get("optimization_level", 3) - 1)
                })
            
            # Apply changes
            for param, value in reconfig_changes.items():
                context.system_state[param] = value
                self.logger.debug(f"Reconfigured {param} to {value}")
            
            context.system_state["reconfiguration_applied"] = True
            context.system_state["reconfig_changes"] = reconfig_changes
            
            self.logger.info(f"Adaptive reconfiguration applied {len(reconfig_changes)} changes")
            return RecoveryResult.SUCCESS
            
        except Exception as e:
            self.logger.error(f"Adaptive reconfiguration failed: {e}")
            return RecoveryResult.FAILURE
    
    def estimate_recovery_time(self, context: RecoveryContext) -> float:
        return 5.0  # Reconfiguration is fast


class CheckpointManager:
    """Manages system checkpoints for recovery operations."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.CheckpointManager")
        
        self.max_checkpoints = 10
        self.checkpoint_interval = 300.0  # 5 minutes
        self.last_checkpoint_time = 0.0
    
    def create_checkpoint(self, system_state: Dict[str, Any], 
                         config: Dict[str, Any]) -> str:
        """Create a system checkpoint."""
        try:
            checkpoint_id = f"checkpoint_{int(time.time())}"
            checkpoint_data = {
                "id": checkpoint_id,
                "timestamp": time.time(),
                "system_state": system_state.copy(),
                "configuration": config.copy(),
                "version": "1.0"
            }
            
            checkpoint_file = self.checkpoint_dir / f"{checkpoint_id}.json"
            
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            # Clean old checkpoints
            self._cleanup_old_checkpoints()
            
            self.last_checkpoint_time = time.time()
            self.logger.info(f"Created checkpoint: {checkpoint_id}")
            return checkpoint_id
            
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
            return ""
    
    def restore_configuration(self, checkpoint_data: Dict[str, Any]) -> bool:
        """Restore configuration from checkpoint data."""
        try:
            if "configuration" not in checkpoint_data:
                return False
            
            # In a real implementation, this would restore actual system configuration
            self.logger.info(f"Restored configuration from checkpoint {checkpoint_data.get('id', 'unknown')}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore configuration: {e}")
            return False
    
    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Get the most recent checkpoint."""
        try:
            checkpoint_files = list(self.checkpoint_dir.glob("checkpoint_*.json"))
            if not checkpoint_files:
                return None
            
            latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Failed to load latest checkpoint: {e}")
            return None
    
    def should_create_checkpoint(self) -> bool:
        """Check if it's time to create a new checkpoint."""
        return time.time() - self.last_checkpoint_time > self.checkpoint_interval
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints to save disk space."""
        try:
            checkpoint_files = sorted(
                self.checkpoint_dir.glob("checkpoint_*.json"),
                key=lambda f: f.stat().st_mtime,
                reverse=True
            )
            
            # Keep only the most recent checkpoints
            for old_file in checkpoint_files[self.max_checkpoints:]:
                old_file.unlink()
                self.logger.debug(f"Removed old checkpoint: {old_file.name}")
                
        except Exception as e:
            self.logger.error(f"Failed to cleanup checkpoints: {e}")


class PhotonicRecoveryManager:
    """
    Main recovery manager that coordinates recovery strategies.
    
    Provides intelligent recovery orchestration with learning capabilities
    and escalation paths for complex failures.
    """
    
    def __init__(self, target_config: TargetConfig, 
                 circuit_breaker: CompositePhotonicCircuitBreaker):
        self.target_config = target_config
        self.circuit_breaker = circuit_breaker
        self.logger = logging.getLogger(f"{__name__}.PhotonicRecoveryManager")
        
        # Recovery components
        self.checkpoint_manager = CheckpointManager()
        self.recovery_actions: List[RecoveryAction] = [
            ThermalRecoveryAction(),
            PhaseRecoveryAction(),
            ConfigurationRollbackAction(self.checkpoint_manager),
            AdaptiveReconfigurationAction()
        ]
        
        # Recovery state
        self.recovery_history: List[Dict[str, Any]] = []
        self.active_recoveries: Dict[str, RecoveryContext] = {}
        self.recovery_lock = threading.RLock()
        
        # Learning and adaptation
        self.success_patterns: Dict[str, float] = {}  # Pattern -> Success rate
        self.recovery_timing: Dict[str, List[float]] = {}  # Strategy -> Times
        
        # Escalation
        self.escalation_callbacks: List[Callable] = []
    
    def handle_failure(self, failure_type: FailureType, 
                      failure_details: Dict[str, Any],
                      system_state: Dict[str, Any]) -> RecoveryResult:
        """
        Handle a system failure with appropriate recovery strategy.
        
        Args:
            failure_type: Type of failure that occurred
            failure_details: Detailed information about the failure
            system_state: Current system state
            
        Returns:
            RecoveryResult indicating success/failure of recovery
        """
        with self.recovery_lock:
            # Create recovery context
            context = RecoveryContext(
                failure_type=failure_type,
                failure_timestamp=time.time(),
                failure_details=failure_details,
                system_state=system_state.copy()
            )
            
            # Add checkpoint data if available
            latest_checkpoint = self.checkpoint_manager.get_latest_checkpoint()
            if latest_checkpoint:
                context.checkpoint_data = latest_checkpoint
            
            failure_id = f"failure_{int(context.failure_timestamp)}"
            self.active_recoveries[failure_id] = context
            
            self.logger.info(f"Starting recovery for {failure_type.value}")
            
            try:
                # Execute recovery strategy
                result = self._execute_recovery_strategy(context)
                
                # Record recovery attempt
                self._record_recovery_attempt(context, result)
                
                # Handle result
                if result == RecoveryResult.SUCCESS:
                    self.logger.info("Recovery completed successfully")
                    self._on_recovery_success(context)
                elif result == RecoveryResult.RETRY_NEEDED:
                    self.logger.info("Recovery requires retry")
                    result = self._retry_recovery(context)
                elif result == RecoveryResult.ESCALATION_REQUIRED:
                    self.logger.warning("Recovery requires escalation")
                    self._escalate_recovery(context)
                else:
                    self.logger.error(f"Recovery failed with result: {result.value}")
                
                return result
                
            except Exception as e:
                self.logger.error(f"Recovery execution failed: {e}")
                self._escalate_recovery(context)
                return RecoveryResult.FAILURE
            
            finally:
                # Cleanup
                if failure_id in self.active_recoveries:
                    del self.active_recoveries[failure_id]
    
    def add_escalation_callback(self, callback: Callable[[RecoveryContext], None]):
        """Add a callback for escalation scenarios."""
        self.escalation_callbacks.append(callback)
    
    def create_system_checkpoint(self, system_state: Dict[str, Any], 
                               config: Dict[str, Any]) -> str:
        """Create a system checkpoint for recovery purposes."""
        return self.checkpoint_manager.create_checkpoint(system_state, config)
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get comprehensive recovery statistics."""
        with self.recovery_lock:
            total_recoveries = len(self.recovery_history)
            
            if total_recoveries == 0:
                return {"total_recoveries": 0}
            
            successful_recoveries = sum(1 for r in self.recovery_history 
                                      if r.get("result") == RecoveryResult.SUCCESS.value)
            
            # Calculate statistics by failure type
            failure_type_stats = {}
            for record in self.recovery_history:
                ft = record.get("failure_type")
                if ft:
                    if ft not in failure_type_stats:
                        failure_type_stats[ft] = {"attempts": 0, "successes": 0}
                    failure_type_stats[ft]["attempts"] += 1
                    if record.get("result") == RecoveryResult.SUCCESS.value:
                        failure_type_stats[ft]["successes"] += 1
            
            # Average recovery times by strategy
            avg_recovery_times = {}
            for strategy, times in self.recovery_timing.items():
                if times:
                    avg_recovery_times[strategy] = sum(times) / len(times)
            
            return {
                "total_recoveries": total_recoveries,
                "successful_recoveries": successful_recoveries,
                "success_rate": successful_recoveries / total_recoveries,
                "failure_type_statistics": failure_type_stats,
                "average_recovery_times": avg_recovery_times,
                "active_recoveries": len(self.active_recoveries),
                "checkpoint_count": len(list(self.checkpoint_manager.checkpoint_dir.glob("*.json")))
            }
    
    def _execute_recovery_strategy(self, context: RecoveryContext) -> RecoveryResult:
        """Execute the most appropriate recovery strategy."""
        # Select recovery actions that can handle this failure type
        suitable_actions = [action for action in self.recovery_actions 
                          if action.can_handle(context.failure_type, context)]
        
        if not suitable_actions:
            self.logger.error(f"No suitable recovery actions for {context.failure_type.value}")
            return RecoveryResult.ESCALATION_REQUIRED
        
        # Sort by estimated recovery time (try fastest first)
        suitable_actions.sort(key=lambda a: a.estimate_recovery_time(context))
        
        # Try recovery actions in sequence
        for action in suitable_actions:
            self.logger.info(f"Trying recovery action: {action.__class__.__name__}")
            
            start_time = time.time()
            result = action.execute(context)
            execution_time = time.time() - start_time
            
            # Record timing
            action_name = action.__class__.__name__
            if action_name not in self.recovery_timing:
                self.recovery_timing[action_name] = []
            self.recovery_timing[action_name].append(execution_time)
            
            if result in [RecoveryResult.SUCCESS, RecoveryResult.PARTIAL_SUCCESS]:
                return result
            
            self.logger.warning(f"Recovery action {action_name} failed: {result.value}")
        
        return RecoveryResult.FAILURE
    
    def _retry_recovery(self, context: RecoveryContext) -> RecoveryResult:
        """Retry recovery with exponential backoff."""
        if context.recovery_attempts >= context.max_recovery_attempts:
            return RecoveryResult.ESCALATION_REQUIRED
        
        context.recovery_attempts += 1
        
        # Exponential backoff
        backoff_time = min(60.0, 2 ** context.recovery_attempts)
        self.logger.info(f"Retrying recovery after {backoff_time}s backoff")
        time.sleep(backoff_time)
        
        return self._execute_recovery_strategy(context)
    
    def _escalate_recovery(self, context: RecoveryContext):
        """Escalate recovery to higher-level handlers."""
        self.logger.critical(f"Escalating recovery for {context.failure_type.value}")
        
        # Trigger emergency shutdown if critical
        if context.failure_type in [FailureType.THERMAL_VIOLATION, FailureType.HARDWARE_FAULT]:
            self.circuit_breaker.trigger_emergency_shutdown(\n                f\"Recovery escalation: {context.failure_type.value}\"\n            )\n        \n        # Call escalation callbacks\n        for callback in self.escalation_callbacks:\n            try:\n                callback(context)\n            except Exception as e:\n                self.logger.error(f\"Escalation callback failed: {e}\")\n    \n    def _record_recovery_attempt(self, context: RecoveryContext, result: RecoveryResult):\n        \"\"\"Record recovery attempt for learning.\"\"\"\n        record = {\n            \"timestamp\": context.failure_timestamp,\n            \"failure_type\": context.failure_type.value,\n            \"result\": result.value,\n            \"recovery_attempts\": context.recovery_attempts,\n            \"system_state_keys\": list(context.system_state.keys()),\n            \"strategy_used\": context.current_strategy.value if context.current_strategy else None\n        }\n        \n        self.recovery_history.append(record)\n        \n        # Keep only recent history\n        if len(self.recovery_history) > 1000:\n            self.recovery_history = self.recovery_history[-500:]\n    \n    def _on_recovery_success(self, context: RecoveryContext):\n        \"\"\"Handle successful recovery.\"\"\"\n        # Update success patterns\n        pattern_key = f\"{context.failure_type.value}_{context.current_strategy.value if context.current_strategy else 'unknown'}\"\n        \n        if pattern_key not in self.success_patterns:\n            self.success_patterns[pattern_key] = 0.5\n        \n        # Update success rate with exponential moving average\n        self.success_patterns[pattern_key] = (\n            0.8 * self.success_patterns[pattern_key] + 0.2 * 1.0\n        )\n        \n        # Create checkpoint after successful recovery\n        if self.checkpoint_manager.should_create_checkpoint():\n            self.checkpoint_manager.create_checkpoint(\n                context.system_state, \n                {\"recovery_timestamp\": time.time()}\n            )\n\n\n# Utility functions\ndef create_recovery_manager(target_config: TargetConfig, \n                          circuit_breaker: CompositePhotonicCircuitBreaker) -> PhotonicRecoveryManager:\n    \"\"\"Create a recovery manager with standard configuration.\"\"\"\n    return PhotonicRecoveryManager(target_config, circuit_breaker)\n\n\ndef setup_comprehensive_recovery(target_config: TargetConfig) -> tuple:\n    \"\"\"Set up comprehensive recovery system with circuit breakers.\"\"\"\n    from .circuit_breaker import create_comprehensive_protection\n    \n    # Create circuit breaker\n    circuit_breaker = create_comprehensive_protection(target_config)\n    \n    # Create recovery manager\n    recovery_manager = create_recovery_manager(target_config, circuit_breaker)\n    \n    # Add default escalation callback\n    def default_escalation_handler(context: RecoveryContext):\n        logging.getLogger(__name__).critical(\n            f\"ESCALATION: {context.failure_type.value} recovery failed after \"\n            f\"{context.recovery_attempts} attempts\"\n        )\n    \n    recovery_manager.add_escalation_callback(default_escalation_handler)\n    \n    return circuit_breaker, recovery_manager\n\n\n# Context manager for automatic recovery\nclass AutoRecoveryContext:\n    \"\"\"Context manager that automatically handles failures with recovery.\"\"\"\n    \n    def __init__(self, recovery_manager: PhotonicRecoveryManager,\n                 operation_type: str, **operation_params):\n        self.recovery_manager = recovery_manager\n        self.operation_type = operation_type\n        self.operation_params = operation_params\n        self.start_time = 0.0\n        \n    def __enter__(self):\n        self.start_time = time.time()\n        return self\n        \n    def __exit__(self, exc_type, exc_val, exc_tb):\n        if exc_type is not None:\n            # Determine failure type from exception\n            failure_type = self._classify_exception(exc_type, exc_val)\n            \n            failure_details = {\n                \"exception_type\": str(exc_type),\n                \"exception_message\": str(exc_val),\n                \"operation_type\": self.operation_type,\n                \"operation_params\": self.operation_params,\n                \"execution_time_s\": time.time() - self.start_time\n            }\n            \n            system_state = {\n                \"timestamp\": time.time(),\n                \"operation_context\": self.operation_type\n            }\n            \n            # Attempt recovery\n            result = self.recovery_manager.handle_failure(\n                failure_type, failure_details, system_state\n            )\n            \n            # Suppress exception if recovery was successful\n            if result == RecoveryResult.SUCCESS:\n                return True  # Suppress the exception\n        \n        return False  # Let exception propagate\n    \n    def _classify_exception(self, exc_type, exc_val) -> FailureType:\n        \"\"\"Classify exception type to determine recovery strategy.\"\"\"\n        exc_name = exc_type.__name__.lower()\n        exc_msg = str(exc_val).lower()\n        \n        if \"thermal\" in exc_msg or \"temperature\" in exc_msg:\n            return FailureType.THERMAL_VIOLATION\n        elif \"phase\" in exc_msg or \"coherence\" in exc_msg:\n            return FailureType.PHASE_COHERENCE_LOSS\n        elif \"timeout\" in exc_msg or \"timeout\" in exc_name:\n            return FailureType.COMPILATION_TIMEOUT\n        elif \"power\" in exc_msg:\n            return FailureType.POWER_OVERLOAD\n        else:\n            return FailureType.HARDWARE_FAULT"