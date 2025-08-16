"""
Photon-MLIR: MLIR-based compiler for silicon photonic neural network accelerators.

This package provides Python bindings for compiling neural networks to photonic hardware.
"""

# Core components (lightweight, minimal dependencies)
from .core import Device, Precision, TargetConfig
from .logging_config import get_global_logger
from .validation import PhotonicValidator, ValidationResult

# Optional imports with graceful fallbacks
try:
    from .compiler import PhotonicCompiler, compile, compile_onnx, compile_pytorch
    _COMPILER_AVAILABLE = True
except ImportError:
    _COMPILER_AVAILABLE = False
    PhotonicCompiler = None

try:
    from .simulator import PhotonicSimulator
    _SIMULATOR_AVAILABLE = True
except ImportError:
    _SIMULATOR_AVAILABLE = False
    PhotonicSimulator = None

try:
    from .transforms import optimize_for_photonics
    _TRANSFORMS_AVAILABLE = True
except ImportError:
    _TRANSFORMS_AVAILABLE = False
    optimize_for_photonics = None

try:
    from .visualizer import MeshVisualizer, OptimizationDashboard
    _VISUALIZER_AVAILABLE = True
except ImportError:
    _VISUALIZER_AVAILABLE = False
    MeshVisualizer = None
    OptimizationDashboard = None
# Optional advanced imports
try:
    from .quantum_scheduler import QuantumTaskPlanner, QuantumInspiredScheduler, CompilationTask, TaskType
    _QUANTUM_SCHEDULER_AVAILABLE = True
except ImportError:
    _QUANTUM_SCHEDULER_AVAILABLE = False
    QuantumTaskPlanner = QuantumInspiredScheduler = CompilationTask = TaskType = None

try:
    from .quantum_optimization import ParallelQuantumScheduler, OptimizationLevel, CacheStrategy
    _QUANTUM_OPT_AVAILABLE = True
except ImportError:
    _QUANTUM_OPT_AVAILABLE = False
    ParallelQuantumScheduler = OptimizationLevel = CacheStrategy = None

try:
    from .quantum_validation import QuantumValidator, QuantumMonitor, ValidationLevel
    _QUANTUM_VAL_AVAILABLE = True
except ImportError:
    _QUANTUM_VAL_AVAILABLE = False
    QuantumValidator = QuantumMonitor = ValidationLevel = None

try:
    from .thermal_optimization import (
        ThermalAwareOptimizer, ThermalAwareBenchmark, ThermalModel, 
        CoolingStrategy, ThermalConstraints, PhotonicDevice
    )
    _THERMAL_AVAILABLE = True
except ImportError:
    _THERMAL_AVAILABLE = False
    ThermalAwareOptimizer = ThermalAwareBenchmark = ThermalModel = None
    CoolingStrategy = ThermalConstraints = PhotonicDevice = None

try:
    from .i18n import GlobalizationManager, SupportedLanguage, setup_i18n, t
    _I18N_AVAILABLE = True
except ImportError:
    _I18N_AVAILABLE = False
    GlobalizationManager = SupportedLanguage = setup_i18n = t = None

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"
__email__ = "daniel@terragon.dev"

__all__ = [
    "PhotonicCompiler",
    "compile", 
    "compile_onnx",
    "compile_pytorch",
    "Device", 
    "Precision",
    "TargetConfig",
    "PhotonicSimulator",
    "optimize_for_photonics",
    "MeshVisualizer",
    "OptimizationDashboard",
    "QuantumTaskPlanner",
    "QuantumInspiredScheduler", 
    "CompilationTask",
    "TaskType",
    "ParallelQuantumScheduler",
    "OptimizationLevel",
    "CacheStrategy",
    "QuantumValidator",
    "QuantumMonitor",
    "ValidationLevel",
    "ThermalAwareOptimizer",
    "ThermalAwareBenchmark",
    "ThermalModel",
    "CoolingStrategy",
    "ThermalConstraints",
    "PhotonicDevice",
    "GlobalizationManager",
    "SupportedLanguage", 
    "setup_i18n",
    "t"
]