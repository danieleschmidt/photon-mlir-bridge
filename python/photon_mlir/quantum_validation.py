"""
Quantum-Inspired Task Planning Validation and Error Handling

Comprehensive validation, error handling, and security measures for the
quantum-inspired task scheduling system in photonic compilation.
"""

import logging
import time
import threading
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

from .quantum_scheduler import CompilationTask, SchedulingState, TaskType, QuantumState

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityThreat(Enum):
    """Security threat types."""
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DEPENDENCY_INJECTION = "dependency_injection"
    INFINITE_LOOP = "infinite_loop"
    MALICIOUS_CONFIG = "malicious_config"


@dataclass
class ValidationResult:
    """Result of task validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_threats: List[SecurityThreat]
    performance_issues: List[str]
    suggestions: List[str]


@dataclass
class QuantumMetrics:
    """Metrics for quantum scheduling performance."""
    convergence_rate: float
    quantum_coherence: float
    entanglement_efficiency: float
    superposition_utilization: float
    annealing_temperature: float
    state_transitions: int
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "convergence_rate": self.convergence_rate,
            "quantum_coherence": self.quantum_coherence,
            "entanglement_efficiency": self.entanglement_efficiency,
            "superposition_utilization": self.superposition_utilization,
            "annealing_temperature": self.annealing_temperature,
            "state_transitions": self.state_transitions
        }


class QuantumValidator:
    """Comprehensive validator for quantum-inspired scheduling."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STRICT):
        self.validation_level = validation_level
        self.known_safe_configs = set()
        self.blacklisted_patterns = set()
        self.max_task_count = 10000
        self.max_dependency_depth = 100
        self.max_resource_requirement = {"cpu": 64.0, "memory": 32768.0, "gpu": 8.0}
        
        # Load security patterns
        self._load_security_patterns()
        
    def validate_tasks(self, tasks: List[CompilationTask]) -> ValidationResult:
        """
        Comprehensive validation of compilation tasks.
        
        Args:
            tasks: List of tasks to validate
            
        Returns:
            Detailed validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            security_threats=[],
            performance_issues=[],
            suggestions=[]
        )
        
        try:
            # Basic validation
            self._validate_basic_structure(tasks, result)
            
            if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
                # Advanced validation
                self._validate_dependencies(tasks, result)
                self._validate_resource_requirements(tasks, result)
                self._validate_security(tasks, result)
                
            if self.validation_level == ValidationLevel.PARANOID:
                # Paranoid validation
                self._validate_quantum_constraints(tasks, result)
                self._validate_performance_characteristics(tasks, result)
            
            # Set overall validity
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            result.is_valid = False
            result.errors.append(f"Validation exception: {str(e)}")
        
        return result
    
    def validate_schedule(self, schedule: SchedulingState) -> ValidationResult:
        """
        Validate a scheduling state for correctness and security.
        
        Args:
            schedule: Scheduling state to validate
            
        Returns:
            Validation result
        """
        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=[],
            security_threats=[],
            performance_issues=[],
            suggestions=[]
        )
        
        try:
            # Validate schedule structure
            self._validate_schedule_structure(schedule, result)
            
            # Validate dependency satisfaction
            self._validate_schedule_dependencies(schedule, result)
            
            # Validate resource constraints
            self._validate_schedule_resources(schedule, result)
            
            # Performance validation
            self._validate_schedule_performance(schedule, result)
            
            result.is_valid = len(result.errors) == 0
            
        except Exception as e:
            logger.error(f"Schedule validation failed: {e}")
            result.is_valid = False
            result.errors.append(f"Schedule validation exception: {str(e)}")
        
        return result
    
    def _validate_basic_structure(self, tasks: List[CompilationTask], result: ValidationResult):
        """Validate basic task structure."""
        if not tasks:
            result.errors.append("No tasks provided")
            return
        
        if len(tasks) > self.max_task_count:
            result.errors.append(f"Too many tasks: {len(tasks)} > {self.max_task_count}")
        
        task_ids = set()
        for task in tasks:
            # Check for duplicate IDs
            if task.id in task_ids:
                result.errors.append(f"Duplicate task ID: {task.id}")
            task_ids.add(task.id)
            
            # Validate task ID format
            if not task.id or len(task.id) > 100:
                result.errors.append(f"Invalid task ID: {task.id}")
            
            # Validate duration
            if task.estimated_duration <= 0 or task.estimated_duration > 86400:  # 24 hours max
                result.errors.append(f"Invalid duration for task {task.id}: {task.estimated_duration}")
            
            # Validate priority
            if task.priority < 0 or task.priority > 10:
                result.warnings.append(f"Unusual priority for task {task.id}: {task.priority}")
    
    def _validate_dependencies(self, tasks: List[CompilationTask], result: ValidationResult):
        """Validate task dependencies."""
        task_ids = {task.id for task in tasks}
        
        for task in tasks:
            # Check for missing dependencies
            for dep_id in task.dependencies:
                if dep_id not in task_ids:
                    result.errors.append(f"Task {task.id} depends on non-existent task: {dep_id}")
            
            # Check for self-dependencies
            if task.id in task.dependencies:
                result.errors.append(f"Task {task.id} depends on itself")
        
        # Check for circular dependencies
        if self._has_circular_dependencies(tasks):
            result.errors.append("Circular dependencies detected")
        
        # Check dependency depth
        max_depth = self._calculate_max_dependency_depth(tasks)
        if max_depth > self.max_dependency_depth:
            result.errors.append(f"Dependency depth too high: {max_depth} > {self.max_dependency_depth}")
    
    def _validate_resource_requirements(self, tasks: List[CompilationTask], result: ValidationResult):
        """Validate resource requirements."""
        for task in tasks:
            for resource, requirement in task.resource_requirements.items():
                if requirement < 0:
                    result.errors.append(f"Negative resource requirement for {task.id}: {resource}={requirement}")
                
                max_allowed = self.max_resource_requirement.get(resource, float('inf'))
                if requirement > max_allowed:
                    result.errors.append(f"Excessive {resource} requirement for {task.id}: {requirement} > {max_allowed}")
    
    def _validate_security(self, tasks: List[CompilationTask], result: ValidationResult):
        """Validate security aspects."""
        total_resources = {"cpu": 0, "memory": 0, "gpu": 0}
        
        for task in tasks:
            # Check for resource exhaustion attacks
            for resource, requirement in task.resource_requirements.items():
                total_resources[resource] = total_resources.get(resource, 0) + requirement
            
            # Check for suspicious task patterns
            if self._is_suspicious_task(task):
                result.security_threats.append(SecurityThreat.DEPENDENCY_INJECTION)
                result.warnings.append(f"Suspicious task pattern detected: {task.id}")
        
        # Check total resource usage
        for resource, total in total_resources.items():
            max_total = self.max_resource_requirement.get(resource, float('inf')) * 10
            if total > max_total:
                result.security_threats.append(SecurityThreat.RESOURCE_EXHAUSTION)
                result.errors.append(f"Total {resource} usage exceeds safe limits: {total} > {max_total}")
    
    def _validate_quantum_constraints(self, tasks: List[CompilationTask], result: ValidationResult):
        """Validate quantum-specific constraints."""
        superposition_tasks = [t for t in tasks if t.quantum_state == QuantumState.SUPERPOSITION]
        
        if len(superposition_tasks) > len(tasks) * 0.8:  # More than 80% in superposition
            result.warnings.append("High superposition ratio may indicate optimization issues")
        
        # Validate entanglement patterns
        entangled_pairs = 0
        for task in tasks:
            entangled_pairs += len(task.entangled_tasks)
        
        if entangled_pairs > len(tasks):
            result.warnings.append("High entanglement density may cause scheduling complexity")
    
    def _validate_performance_characteristics(self, tasks: List[CompilationTask], result: ValidationResult):
        """Validate performance characteristics."""
        durations = [task.estimated_duration for task in tasks]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            max_duration = max(durations)
            
            # Check for outliers
            if max_duration > avg_duration * 10:
                result.performance_issues.append("Task duration outliers detected")
                result.suggestions.append("Consider breaking down long-running tasks")
            
            # Check for too many short tasks
            short_tasks = sum(1 for d in durations if d < 0.1)
            if short_tasks > len(tasks) * 0.5:
                result.performance_issues.append("Too many very short tasks")
                result.suggestions.append("Consider task consolidation for better efficiency")
    
    def _validate_schedule_structure(self, schedule: SchedulingState, result: ValidationResult):
        """Validate schedule structure."""
        if not schedule.schedule:
            result.errors.append("Empty schedule")
            return
        
        # Check for negative time slots
        for slot in schedule.schedule.keys():
            if slot < 0:
                result.errors.append(f"Negative time slot: {slot}")
        
        # Check for empty slots
        empty_slots = [slot for slot, tasks in schedule.schedule.items() if not tasks]
        if empty_slots:
            result.warnings.append(f"Empty time slots: {empty_slots}")
    
    def _validate_schedule_dependencies(self, schedule: SchedulingState, result: ValidationResult):
        """Validate dependency satisfaction in schedule."""
        task_slots = {}
        task_map = {t.id: t for t in schedule.tasks}
        
        # Build task -> slot mapping
        for slot, task_ids in schedule.schedule.items():
            for task_id in task_ids:
                task_slots[task_id] = slot
        
        # Check dependencies
        for task in schedule.tasks:
            if task.id in task_slots:
                task_slot = task_slots[task.id]
                for dep_id in task.dependencies:
                    if dep_id in task_slots:
                        dep_task = task_map.get(dep_id)
                        if dep_task:
                            dep_slot = task_slots[dep_id]
                            dep_completion = dep_slot + dep_task.estimated_duration
                            if dep_completion > task_slot:
                                result.errors.append(f"Dependency violation: {task.id} starts before {dep_id} completes")
    
    def _validate_schedule_resources(self, schedule: SchedulingState, result: ValidationResult):
        """Validate resource constraints in schedule."""
        resource_usage = {}
        task_map = {t.id: t for t in schedule.tasks}
        
        for slot, task_ids in schedule.schedule.items():
            slot_resources = {"cpu": 0, "memory": 0, "gpu": 0}
            
            for task_id in task_ids:
                task = task_map.get(task_id)
                if task:
                    for resource, requirement in task.resource_requirements.items():
                        slot_resources[resource] = slot_resources.get(resource, 0) + requirement
            
            resource_usage[slot] = slot_resources
            
            # Check for resource over-subscription
            for resource, usage in slot_resources.items():
                max_allowed = self.max_resource_requirement.get(resource, float('inf'))
                if usage > max_allowed:
                    result.errors.append(f"Resource over-subscription at slot {slot}: {resource}={usage} > {max_allowed}")
    
    def _validate_schedule_performance(self, schedule: SchedulingState, result: ValidationResult):
        """Validate schedule performance characteristics."""
        if schedule.makespan <= 0:
            result.errors.append("Invalid makespan")
        
        if schedule.resource_utilization < 0 or schedule.resource_utilization > 1:
            result.warnings.append(f"Unusual resource utilization: {schedule.resource_utilization:.2%}")
        
        if schedule.resource_utilization < 0.3:
            result.performance_issues.append("Low resource utilization")
            result.suggestions.append("Consider increasing parallelism or task consolidation")
    
    def _has_circular_dependencies(self, tasks: List[CompilationTask]) -> bool:
        """Check for circular dependencies using DFS."""
        task_map = {task.id: task for task in tasks}
        visited = set()
        rec_stack = set()
        
        def dfs(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            task = task_map.get(task_id)
            if task:
                for dep_id in task.dependencies:
                    if dfs(dep_id):
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in tasks:
            if dfs(task.id):
                return True
        
        return False
    
    def _calculate_max_dependency_depth(self, tasks: List[CompilationTask]) -> int:
        """Calculate maximum dependency depth."""
        task_map = {task.id: task for task in tasks}
        depth_cache = {}
        
        def calculate_depth(task_id: str) -> int:
            if task_id in depth_cache:
                return depth_cache[task_id]
            
            task = task_map.get(task_id)
            if not task or not task.dependencies:
                depth_cache[task_id] = 0
                return 0
            
            max_dep_depth = max(calculate_depth(dep_id) for dep_id in task.dependencies)
            depth = max_dep_depth + 1
            depth_cache[task_id] = depth
            return depth
        
        if not tasks:
            return 0
        
        return max(calculate_depth(task.id) for task in tasks)
    
    def _is_suspicious_task(self, task: CompilationTask) -> bool:
        """Check if task exhibits suspicious patterns."""
        # Check for unusual resource patterns
        if task.resource_requirements:
            cpu = task.resource_requirements.get("cpu", 0)
            memory = task.resource_requirements.get("memory", 0)
            
            # Suspicious if requesting enormous resources
            if cpu > 32 or memory > 16384:
                return True
        
        # Check for suspicious ID patterns
        if any(pattern in task.id.lower() for pattern in self.blacklisted_patterns):
            return True
        
        return False
    
    def _load_security_patterns(self):
        """Load known security patterns."""
        self.blacklisted_patterns = {
            "exploit", "attack", "malicious", "backdoor", 
            "inject", "overflow", "shell", "exec"
        }


class QuantumMonitor:
    """Monitor quantum scheduling performance and health."""
    
    def __init__(self):
        self.metrics_history: List[QuantumMetrics] = []
        self.performance_alerts: List[str] = []
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def start_monitoring(self, interval: float = 1.0):
        """Start performance monitoring."""
        if self.is_monitoring:
            logger.warning("Monitor already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Quantum performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Quantum performance monitoring stopped")
    
    def record_metrics(self, metrics: QuantumMetrics):
        """Record quantum metrics."""
        with self._lock:
            self.metrics_history.append(metrics)
            
            # Keep only recent metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-500:]
            
            # Check for performance alerts
            self._check_performance_alerts(metrics)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics_history:
                return {"status": "no_data"}
            
            recent_metrics = self.metrics_history[-10:]  # Last 10 measurements
            
            return {
                "total_measurements": len(self.metrics_history),
                "recent_convergence_rate": sum(m.convergence_rate for m in recent_metrics) / len(recent_metrics),
                "average_coherence": sum(m.quantum_coherence for m in recent_metrics) / len(recent_metrics),
                "entanglement_efficiency": sum(m.entanglement_efficiency for m in recent_metrics) / len(recent_metrics),
                "active_alerts": len(self.performance_alerts),
                "latest_metrics": recent_metrics[-1].to_dict() if recent_metrics else None
            }
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect system metrics (simplified)
                metrics = QuantumMetrics(
                    convergence_rate=0.95,  # Placeholder
                    quantum_coherence=0.88,
                    entanglement_efficiency=0.92,
                    superposition_utilization=0.75,
                    annealing_temperature=1.0,
                    state_transitions=100
                )
                
                self.record_metrics(metrics)
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _check_performance_alerts(self, metrics: QuantumMetrics):
        """Check for performance alerts."""
        alerts = []
        
        if metrics.convergence_rate < 0.8:
            alerts.append(f"Low convergence rate: {metrics.convergence_rate:.2%}")
        
        if metrics.quantum_coherence < 0.7:
            alerts.append(f"Low quantum coherence: {metrics.quantum_coherence:.2%}")
        
        if metrics.entanglement_efficiency < 0.5:
            alerts.append(f"Poor entanglement efficiency: {metrics.entanglement_efficiency:.2%}")
        
        # Update alert list
        current_time = time.time()
        for alert in alerts:
            alert_entry = f"[{current_time:.0f}] {alert}"
            if alert_entry not in self.performance_alerts:
                self.performance_alerts.append(alert_entry)
                logger.warning(f"Performance alert: {alert}")
        
        # Clean old alerts (older than 1 hour)
        hour_ago = current_time - 3600
        self.performance_alerts = [
            alert for alert in self.performance_alerts 
            if float(alert.split(']')[0][1:]) > hour_ago
        ]


class QuantumSecurityManager:
    """Security manager for quantum scheduling operations."""
    
    def __init__(self):
        self.security_log: List[Dict[str, Any]] = []
        self.blocked_operations: Set[str] = set()
        self.audit_trail: List[Dict[str, Any]] = []
        
    def authenticate_operation(self, operation: str, context: Dict[str, Any]) -> bool:
        """Authenticate scheduling operation."""
        # Simple authentication (production would use proper auth)
        operation_hash = hashlib.sha256(f"{operation}:{json.dumps(context, sort_keys=True)}".encode()).hexdigest()
        
        if operation_hash in self.blocked_operations:
            self._log_security_event("blocked_operation", {
                "operation": operation,
                "hash": operation_hash,
                "context": context
            })
            return False
        
        self._log_audit_event("operation_authenticated", {
            "operation": operation,
            "hash": operation_hash,
            "timestamp": time.time()
        })
        
        return True
    
    def validate_input_safety(self, data: Any) -> Tuple[bool, List[str]]:
        """Validate input data for safety."""
        issues = []
        
        # Check data size
        data_str = str(data)
        if len(data_str) > 1_000_000:  # 1MB limit
            issues.append("Input data too large")
        
        # Check for suspicious patterns
        suspicious_patterns = ["eval(", "exec(", "__import__", "subprocess"]
        for pattern in suspicious_patterns:
            if pattern in data_str:
                issues.append(f"Suspicious pattern detected: {pattern}")
        
        is_safe = len(issues) == 0
        
        if not is_safe:
            self._log_security_event("unsafe_input", {
                "issues": issues,
                "data_length": len(data_str)
            })
        
        return is_safe, issues
    
    def _log_security_event(self, event_type: str, details: Dict[str, Any]):
        """Log security event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        self.security_log.append(event)
        logger.warning(f"Security event: {event_type} - {details}")
    
    def _log_audit_event(self, event_type: str, details: Dict[str, Any]):
        """Log audit event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "details": details
        }
        self.audit_trail.append(event)
        logger.debug(f"Audit event: {event_type}")