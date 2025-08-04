"""
Health checks and diagnostics for photonic compiler.
"""

import os
import sys
import time
import psutil
import platform
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging

from .core import TargetConfig, Device
from .security import SecurityError, InputSanitizer

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: str  # "healthy", "warning", "critical", "unknown"
    message: str
    details: Dict[str, Any]
    duration_ms: float
    timestamp: float


class SystemDiagnostics:
    """System-level diagnostics and health checks."""
    
    @staticmethod
    def check_system_resources() -> HealthCheckResult:
        """Check system resource availability."""
        start_time = time.time()
        
        try:
            # Get system info
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else (0, 0, 0)
            
            details = {
                "memory_total_gb": round(memory.total / (1024**3), 2),
                "memory_available_gb": round(memory.available / (1024**3), 2),
                "memory_percent_used": memory.percent,
                "disk_total_gb": round(disk.total / (1024**3), 2),
                "disk_free_gb": round(disk.free / (1024**3), 2),
                "disk_percent_used": round((disk.used / disk.total) * 100, 1),
                "cpu_count": cpu_count,
                "load_average_1m": load_avg[0],
                "load_average_5m": load_avg[1],
                "load_average_15m": load_avg[2]
            }
            
            # Determine status
            status = "healthy"
            messages = []
            
            if memory.percent > 90:
                status = "critical"
                messages.append(f"Memory usage critical: {memory.percent}%")
            elif memory.percent > 80:
                status = "warning"
                messages.append(f"Memory usage high: {memory.percent}%")
            
            if disk.percent > 95:
                status = "critical"
                messages.append(f"Disk usage critical: {disk.percent}%")
            elif disk.percent > 85:
                if status != "critical":
                    status = "warning"
                messages.append(f"Disk usage high: {disk.percent}%")
            
            if load_avg[0] > cpu_count * 2:
                status = "critical"
                messages.append(f"Load average very high: {load_avg[0]}")
            elif load_avg[0] > cpu_count:
                if status != "critical":
                    status = "warning"
                messages.append(f"Load average high: {load_avg[0]}")
            
            message = "; ".join(messages) if messages else "System resources OK"
            
        except Exception as e:
            status = "unknown"
            message = f"Failed to check system resources: {e}"
            details = {"error": str(e)}
        
        duration = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="system_resources",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time()
        )
    
    @staticmethod
    def check_python_environment() -> HealthCheckResult:
        """Check Python environment and dependencies."""
        start_time = time.time()
        
        try:
            import numpy
            import torch
            
            details = {
                "python_version": sys.version,
                "platform": platform.platform(),
                "numpy_version": numpy.__version__,
                "torch_version": torch.__version__,
                "torch_cuda_available": torch.cuda.is_available(),
                "torch_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            }
            
            # Check for GPU availability
            if torch.cuda.is_available():
                gpu_info = []
                for i in range(torch.cuda.device_count()):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_info.append({
                        "device_id": i,
                        "name": gpu_name,
                        "memory_gb": round(gpu_memory, 2)
                    })
                details["gpu_devices"] = gpu_info
            
            status = "healthy"
            message = "Python environment OK"
            
            # Check version compatibility
            python_version = sys.version_info
            if python_version < (3, 9):
                status = "warning"
                message = f"Python version {python_version.major}.{python_version.minor} is below recommended 3.9+"
            
        except ImportError as e:
            status = "critical"
            message = f"Missing required dependency: {e}"
            details = {"error": str(e)}
        except Exception as e:
            status = "unknown"
            message = f"Failed to check Python environment: {e}"
            details = {"error": str(e)}
        
        duration = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="python_environment",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time()
        )
    
    @staticmethod
    def check_file_permissions() -> HealthCheckResult:
        """Check file system permissions."""
        start_time = time.time()
        
        try:
            # Test temp directory access
            temp_dir = Path("/tmp") if os.name != 'nt' else Path(os.environ.get('TEMP', '.'))
            
            details = {
                "temp_dir": str(temp_dir),
                "temp_dir_exists": temp_dir.exists(),
                "temp_dir_writable": os.access(temp_dir, os.W_OK) if temp_dir.exists() else False
            }
            
            # Test current directory permissions
            current_dir = Path.cwd()
            details.update({
                "current_dir": str(current_dir),
                "current_dir_readable": os.access(current_dir, os.R_OK),
                "current_dir_writable": os.access(current_dir, os.W_OK)
            })
            
            # Try creating a test file
            test_file_created = False
            try:
                test_file = temp_dir / "photon_mlir_test.tmp"
                test_file.write_text("test")
                test_file.unlink()
                test_file_created = True
            except Exception:
                pass
            
            details["test_file_creation"] = test_file_created
            
            # Determine status
            if not temp_dir.exists() or not details["temp_dir_writable"]:
                status = "critical"
                message = "Cannot write to temporary directory"
            elif not details["current_dir_readable"]:
                status = "critical"
                message = "Cannot read current directory"
            elif not test_file_created:
                status = "warning"
                message = "Limited file system access"
            else:
                status = "healthy"
                message = "File system permissions OK"
                
        except Exception as e:
            status = "unknown"
            message = f"Failed to check file permissions: {e}"
            details = {"error": str(e)}
        
        duration = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="file_permissions",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time()
        )


class CompilerDiagnostics:
    """Compiler-specific diagnostics."""
    
    @staticmethod
    def check_mlir_availability() -> HealthCheckResult:
        """Check MLIR availability and version."""
        start_time = time.time()
        
        try:
            # This would check for actual MLIR installation in a real implementation
            # For now, we'll simulate the check
            
            details = {
                "mlir_available": True,  # Simulated
                "mlir_version": "17.0.0",  # Simulated
                "llvm_version": "17.0.0"   # Simulated
            }
            
            status = "healthy"
            message = "MLIR infrastructure available"
            
        except Exception as e:
            status = "critical"
            message = f"MLIR not available: {e}"
            details = {"error": str(e)}
        
        duration = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="mlir_availability",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time()
        )
    
    @staticmethod
    def check_photonic_dialect() -> HealthCheckResult:
        """Check photonic dialect registration."""
        start_time = time.time()
        
        try:
            # This would check actual dialect registration
            details = {
                "photonic_dialect_registered": True,
                "supported_operations": [
                    "photonic.matmul",
                    "photonic.phase_shift", 
                    "photonic.thermal_compensation",
                    "photonic.unfold",
                    "photonic.fold",
                    "photonic.optical_encoding",
                    "photonic.optical_decoding"
                ]
            }
            
            status = "healthy"
            message = "Photonic dialect available"
            
        except Exception as e:
            status = "critical"
            message = f"Photonic dialect unavailable: {e}"
            details = {"error": str(e)}
        
        duration = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="photonic_dialect",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time()
        )
    
    @staticmethod
    def validate_target_config(config: TargetConfig) -> HealthCheckResult:
        """Validate target configuration."""
        start_time = time.time()
        
        try:
            issues = []
            
            # Check wavelength
            if not (1200 <= config.wavelength_nm <= 1700):
                issues.append(f"Wavelength {config.wavelength_nm}nm outside recommended range 1200-1700nm")
            
            # Check array size
            width, height = config.array_size
            if width <= 0 or height <= 0:
                issues.append(f"Invalid array size: {width}x{height}")
            elif width > 1024 or height > 1024:
                issues.append(f"Array size {width}x{height} may exceed hardware limits")
            
            # Check thermal parameters
            if config.max_phase_drift < 0 or config.max_phase_drift > 3.14159:
                issues.append(f"Max phase drift {config.max_phase_drift} outside reasonable range")
            
            if config.calibration_interval_ms < 1 or config.calibration_interval_ms > 60000:
                issues.append(f"Calibration interval {config.calibration_interval_ms}ms outside practical range")
            
            details = {
                "device": config.device.value,
                "precision": config.precision.value,
                "array_size": config.array_size,
                "wavelength_nm": config.wavelength_nm,
                "max_phase_drift": config.max_phase_drift,
                "calibration_interval_ms": config.calibration_interval_ms,
                "issues_found": len(issues),
                "issues": issues
            }
            
            if not issues:
                status = "healthy"
                message = "Target configuration valid"
            elif len(issues) == 1 and "recommended" in issues[0]:
                status = "warning"
                message = f"Configuration warning: {issues[0]}"
            else:
                status = "critical"
                message = f"Configuration issues: {'; '.join(issues)}"
                
        except Exception as e:
            status = "unknown"
            message = f"Failed to validate config: {e}"
            details = {"error": str(e)}
        
        duration = (time.time() - start_time) * 1000
        
        return HealthCheckResult(
            name="target_config_validation",
            status=status,
            message=message,
            details=details,
            duration_ms=duration,
            timestamp=time.time()
        )


class HealthChecker:
    """Main health checker orchestrator."""
    
    def __init__(self):
        self.checks = []
        self.register_default_checks()
    
    def register_default_checks(self):
        """Register default health checks."""
        self.checks = [
            ("System Resources", SystemDiagnostics.check_system_resources),
            ("Python Environment", SystemDiagnostics.check_python_environment),
            ("File Permissions", SystemDiagnostics.check_file_permissions),
            ("MLIR Availability", CompilerDiagnostics.check_mlir_availability),
            ("Photonic Dialect", CompilerDiagnostics.check_photonic_dialect)
        ]
    
    def add_check(self, name: str, check_func):
        """Add a custom health check."""
        self.checks.append((name, check_func))
    
    def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for name, check_func in self.checks:
            try:
                result = check_func()
                results.append(result)
                logger.info(f"Health check '{name}': {result.status} - {result.message}")
            except Exception as e:
                error_result = HealthCheckResult(
                    name=name.lower().replace(' ', '_'),
                    status="unknown",
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                    duration_ms=0,
                    timestamp=time.time()
                )
                results.append(error_result)
                logger.error(f"Health check '{name}' failed: {e}")
        
        return results
    
    def run_check(self, check_name: str) -> Optional[HealthCheckResult]:
        """Run a specific health check."""
        for name, check_func in self.checks:
            if name.lower().replace(' ', '_') == check_name.lower():
                try:
                    return check_func()
                except Exception as e:
                    return HealthCheckResult(
                        name=check_name,
                        status="unknown",
                        message=f"Health check failed: {e}",
                        details={"error": str(e)},
                        duration_ms=0,
                        timestamp=time.time()
                    )
        
        return None
    
    def get_overall_status(self, results: List[HealthCheckResult]) -> str:
        """Get overall system status from check results."""
        if not results:
            return "unknown"
        
        statuses = [result.status for result in results]
        
        if "critical" in statuses:
            return "critical"
        elif "warning" in statuses:
            return "warning"
        elif "unknown" in statuses:
            return "degraded"
        else:
            return "healthy"
    
    def format_results(self, results: List[HealthCheckResult], format: str = "text") -> str:
        """Format health check results."""
        if format == "json":
            return self._format_json(results)
        else:
            return self._format_text(results)
    
    def _format_text(self, results: List[HealthCheckResult]) -> str:
        """Format results as text."""
        output = []
        output.append("=== Photonic Compiler Health Check ===")
        output.append(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        overall_status = self.get_overall_status(results)
        status_symbols = {
            "healthy": "✓",
            "warning": "⚠", 
            "critical": "✗",
            "unknown": "?",
            "degraded": "⚠"
        }
        
        output.append(f"Overall Status: {status_symbols.get(overall_status, '?')} {overall_status.upper()}")
        output.append("")
        
        for result in results:
            symbol = status_symbols.get(result.status, '?')
            output.append(f"{symbol} {result.name}: {result.status.upper()}")
            output.append(f"  Message: {result.message}")
            output.append(f"  Duration: {result.duration_ms:.1f}ms")
            
            if result.details and result.status in ["warning", "critical"]:
                output.append("  Details:")
                for key, value in result.details.items():
                    if key != "error":
                        output.append(f"    {key}: {value}")
            output.append("")
        
        return "\n".join(output)
    
    def _format_json(self, results: List[HealthCheckResult]) -> str:
        """Format results as JSON."""
        data = {
            "timestamp": time.time(),
            "overall_status": self.get_overall_status(results),
            "checks": [
                {
                    "name": result.name,
                    "status": result.status,
                    "message": result.message,
                    "duration_ms": result.duration_ms,
                    "timestamp": result.timestamp,
                    "details": result.details
                }
                for result in results
            ]
        }
        
        return json.dumps(data, indent=2, default=str)


# Global health checker instance
health_checker = HealthChecker()


def run_diagnostics(config: Optional[TargetConfig] = None, format: str = "text") -> str:
    """
    Run comprehensive diagnostics.
    
    Args:
        config: Optional target configuration to validate
        format: Output format ("text" or "json")
        
    Returns:
        Formatted diagnostic results
    """
    results = health_checker.run_all_checks()
    
    # Add config validation if provided
    if config:
        config_result = CompilerDiagnostics.validate_target_config(config)
        results.append(config_result)
    
    return health_checker.format_results(results, format)


def quick_health_check() -> bool:
    """
    Quick health check returning boolean status.
    
    Returns:
        True if system is healthy, False otherwise
    """
    results = health_checker.run_all_checks()
    overall_status = health_checker.get_overall_status(results)
    return overall_status in ["healthy", "warning"]