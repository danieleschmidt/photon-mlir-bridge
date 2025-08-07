"""
Photon-MLIR: MLIR-based compiler for silicon photonic neural network accelerators.

This package provides Python bindings for compiling neural networks to photonic hardware.
"""

from .compiler import PhotonicCompiler, compile, compile_onnx, compile_pytorch
from .core import Device, Precision, TargetConfig
from .simulator import PhotonicSimulator
from .transforms import optimize_for_photonics
from .visualizer import MeshVisualizer, OptimizationDashboard
from .quantum_scheduler import QuantumTaskPlanner, QuantumInspiredScheduler, CompilationTask
from .quantum_optimization import ParallelQuantumScheduler, OptimizationLevel, CacheStrategy
from .quantum_validation import QuantumValidator, QuantumMonitor, ValidationLevel
from .i18n import GlobalizationManager, SupportedLanguage, setup_i18n, t

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
    "ParallelQuantumScheduler",
    "OptimizationLevel",
    "CacheStrategy",
    "QuantumValidator",
    "QuantumMonitor",
    "ValidationLevel",
    "GlobalizationManager",
    "SupportedLanguage", 
    "setup_i18n",
    "t"
]