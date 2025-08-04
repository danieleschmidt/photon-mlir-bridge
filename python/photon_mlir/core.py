"""
Core photonic computing types and configuration.
"""

from enum import Enum
from typing import Tuple
from dataclasses import dataclass


class Device(Enum):
    """Supported photonic hardware devices."""
    LIGHTMATTER_ENVISE = "lightmatter_envise"
    MIT_PHOTONIC_PROCESSOR = "mit_photonic_processor" 
    CUSTOM_RESEARCH_CHIP = "custom_research_chip"


class Precision(Enum):
    """Supported precision modes."""
    INT8 = "int8"
    INT16 = "int16"
    FP16 = "fp16"
    FP32 = "fp32"


@dataclass
class TargetConfig:
    """Configuration for photonic compilation target."""
    device: Device = Device.LIGHTMATTER_ENVISE
    precision: Precision = Precision.INT8
    array_size: Tuple[int, int] = (64, 64)
    wavelength_nm: int = 1550
    max_phase_drift: float = 0.1  # radians
    calibration_interval_ms: int = 100
    enable_thermal_compensation: bool = True
    mesh_topology: str = "butterfly"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for C++ interop."""
        return {
            "device": self.device.value,
            "precision": self.precision.value,
            "array_size": self.array_size,
            "wavelength_nm": self.wavelength_nm,
            "max_phase_drift": self.max_phase_drift,
            "calibration_interval_ms": self.calibration_interval_ms,
            "enable_thermal_compensation": self.enable_thermal_compensation,
            "mesh_topology": self.mesh_topology
        }


class PhotonicTensor:
    """Represents a tensor in the photonic computing domain."""
    
    def __init__(self, data, wavelength: int = 1550, power_mw: float = 1.0):
        self.data = data
        self.wavelength = wavelength
        self.power_mw = power_mw
        
    @property
    def shape(self):
        """Get tensor shape."""
        if hasattr(self.data, 'shape'):
            return self.data.shape
        return None
    
    def __repr__(self):
        return f"PhotonicTensor(shape={self.shape}, wavelength={self.wavelength}nm, power={self.power_mw}mW)"