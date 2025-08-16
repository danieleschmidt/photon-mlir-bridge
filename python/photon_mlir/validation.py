"""
Comprehensive input validation and error handling for photonic compiler.
Generation 2: Make it Robust - Validation and Error Handling
"""

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None

from typing import Union, List, Tuple, Optional, Any, Dict
from dataclasses import dataclass
from enum import Enum
import os
import warnings

from .core import TargetConfig, Device, Precision


class ValidationError(Exception):
    """Base exception for validation errors."""
    pass


class ConfigurationError(ValidationError):
    """Configuration-related validation errors."""
    pass


class ModelError(ValidationError):
    """Model-related validation errors."""
    pass


class HardwareError(ValidationError):
    """Hardware-related validation errors."""
    pass


class DataError(ValidationError):
    """Data-related validation errors."""
    pass


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def __post_init__(self):
        """Ensure lists are initialized."""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.recommendations is None:
            self.recommendations = []
    
    def add_error(self, message: str):
        """Add validation error."""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """Add validation warning."""
        self.warnings.append(message)
    
    def add_recommendation(self, message: str):
        """Add optimization recommendation."""
        self.recommendations.append(message)
    
    def raise_if_invalid(self):
        """Raise ValidationError if validation failed."""
        if not self.is_valid:
            error_msg = "Validation failed:\n" + "\n".join(f"  • {err}" for err in self.errors)
            raise ValidationError(error_msg)
    
    def print_summary(self, verbose: bool = True):
        """Print validation summary."""
        if self.is_valid:
            print("✅ Validation passed")
        else:
            print("❌ Validation failed")
            
        if self.errors:
            print(f"🔴 Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"   • {error}")
        
        if self.warnings and verbose:
            print(f"🟡 Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        if self.recommendations and verbose:
            print(f"💡 Recommendations ({len(self.recommendations)}):")
            for rec in self.recommendations:
                print(f"   • {rec}")


class PhotonicValidator:
    """Comprehensive validator for photonic compiler inputs and configurations."""
    
    def __init__(self, strict_mode: bool = False):
        """Initialize validator.
        
        Args:
            strict_mode: If True, treat warnings as errors
        """
        self.strict_mode = strict_mode
        
        # Hardware constraints
        self.hardware_limits = {
            Device.LIGHTMATTER_ENVISE: {
                'max_array_size': (128, 128),
                'min_array_size': (8, 8),
                'supported_wavelengths': [1310, 1550, 1570],
                'max_power_mw': 100.0,
                'max_thermal_drift': 2.0,
                'supported_precisions': [Precision.INT8, Precision.INT16, Precision.FP16]
            },
            Device.MIT_PHOTONIC_PROCESSOR: {
                'max_array_size': (64, 64),
                'min_array_size': (4, 4),
                'supported_wavelengths': [1550],
                'max_power_mw': 50.0,
                'max_thermal_drift': 1.0,
                'supported_precisions': [Precision.INT8, Precision.FP16, Precision.FP32]
            },
            Device.CUSTOM_RESEARCH_CHIP: {
                'max_array_size': (32, 32),
                'min_array_size': (2, 2),
                'supported_wavelengths': [1310, 1550],
                'max_power_mw': 200.0,
                'max_thermal_drift': 5.0,
                'supported_precisions': [Precision.INT8, Precision.INT16, Precision.FP16, Precision.FP32]
            }
        }
    
    def validate_target_config(self, config: TargetConfig) -> ValidationResult:
        """Validate target configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], recommendations=[])
        
        try:
            # Validate device
            if not isinstance(config.device, Device):
                result.add_error(f"Invalid device type: {type(config.device)}. Must be Device enum.")
                return result
            
            limits = self.hardware_limits.get(config.device)
            if not limits:
                result.add_error(f"Unsupported device: {config.device}")
                return result
            
            # Validate array size
            if not isinstance(config.array_size, (tuple, list)) or len(config.array_size) != 2:
                result.add_error("Array size must be a tuple/list of 2 integers (rows, cols)")
            else:
                rows, cols = config.array_size
                if not isinstance(rows, int) or not isinstance(cols, int):
                    result.add_error("Array size dimensions must be integers")
                elif rows <= 0 or cols <= 0:
                    result.add_error("Array size dimensions must be positive")
                else:
                    max_rows, max_cols = limits['max_array_size']
                    min_rows, min_cols = limits['min_array_size']
                    
                    if rows > max_rows or cols > max_cols:
                        result.add_error(f"Array size {config.array_size} exceeds maximum {limits['max_array_size']} for {config.device.value}")
                    elif rows < min_rows or cols < min_cols:
                        result.add_error(f"Array size {config.array_size} below minimum {limits['min_array_size']} for {config.device.value}")
                    
                    # Recommendations for optimal size
                    if rows != cols:
                        result.add_warning("Non-square arrays may have reduced performance")
                    
                    if not (rows & (rows - 1) == 0) or not (cols & (cols - 1) == 0):
                        result.add_recommendation("Power-of-2 array sizes often provide better performance")
            
            # Validate precision
            if not isinstance(config.precision, Precision):
                result.add_error(f"Invalid precision type: {type(config.precision)}. Must be Precision enum.")
            elif config.precision not in limits['supported_precisions']:
                result.add_error(f"Precision {config.precision.value} not supported by {config.device.value}")
            
            # Validate wavelength
            if not isinstance(config.wavelength_nm, int):
                result.add_error("Wavelength must be an integer")
            elif config.wavelength_nm not in limits['supported_wavelengths']:
                supported = ', '.join(map(str, limits['supported_wavelengths']))
                result.add_error(f"Wavelength {config.wavelength_nm}nm not supported. Supported: {supported}nm")
            
            # Validate thermal parameters
            if not isinstance(config.max_phase_drift, (int, float)):
                result.add_error("Max phase drift must be a number")
            elif config.max_phase_drift < 0:
                result.add_error("Max phase drift must be non-negative")
            elif config.max_phase_drift > limits['max_thermal_drift']:
                result.add_warning(f"Phase drift {config.max_phase_drift} exceeds recommended maximum {limits['max_thermal_drift']} radians")
            
            if not isinstance(config.calibration_interval_ms, int):
                result.add_error("Calibration interval must be an integer")
            elif config.calibration_interval_ms <= 0:
                result.add_error("Calibration interval must be positive")
            elif config.calibration_interval_ms < 10:
                result.add_warning("Very short calibration intervals may impact performance")
            elif config.calibration_interval_ms > 10000:
                result.add_warning("Long calibration intervals may reduce accuracy")
            
            # Validate mesh topology
            valid_topologies = ['butterfly', 'crossbar', 'mesh_of_trees', 'ring', 'torus']
            if not isinstance(config.mesh_topology, str):
                result.add_error("Mesh topology must be a string")
            elif config.mesh_topology not in valid_topologies:
                result.add_error(f"Invalid mesh topology '{config.mesh_topology}'. Valid options: {', '.join(valid_topologies)}")
            
            # Performance recommendations
            if config.precision == Precision.FP32 and config.device != Device.MIT_PHOTONIC_PROCESSOR:
                result.add_recommendation("FP32 precision may be slower than INT8/INT16 for this device")
            
            if config.array_size[0] * config.array_size[1] > 1024:
                result.add_recommendation("Large arrays may require increased power and thermal management")
            
        except Exception as e:
            result.add_error(f"Unexpected error during config validation: {str(e)}")
        
        return result
    
    def validate_input_data(self, data: Union['np.ndarray', List, Tuple], 
                          expected_shape: Optional[Tuple[int, ...]] = None,
                          data_name: str = "input") -> ValidationResult:
        """Validate input data."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], recommendations=[])
        
        try:
            # Convert to numpy array if needed
            if _NUMPY_AVAILABLE and not isinstance(data, np.ndarray):
                try:
                    data = np.array(data)
                except Exception as e:
                    result.add_error(f"Cannot convert {data_name} to numpy array: {str(e)}")
                    return result
            
            # Check data type
            if not np.issubdtype(data.dtype, np.number):
                result.add_error(f"{data_name} must contain numeric data, got {data.dtype}")
            
            # Check for invalid values
            if np.any(np.isnan(data)):
                result.add_error(f"{data_name} contains NaN values")
            
            if np.any(np.isinf(data)):
                result.add_error(f"{data_name} contains infinite values")
            
            # Check data range
            data_min, data_max = np.min(data), np.max(data)
            if data_min < -1000 or data_max > 1000:
                result.add_warning(f"{data_name} has extreme values (min: {data_min:.2f}, max: {data_max:.2f})")
                result.add_recommendation("Consider normalizing input data to reasonable range")
            
            # Check shape
            if expected_shape and data.shape != expected_shape:
                result.add_error(f"{data_name} shape {data.shape} doesn't match expected {expected_shape}")
            
            # Performance recommendations
            if data.size > 1_000_000:
                result.add_warning(f"Large {data_name} size ({data.size} elements) may impact performance")
            
            if len(data.shape) > 4:
                result.add_warning(f"High-dimensional {data_name} ({len(data.shape)}D) may require special handling")
            
        except Exception as e:
            result.add_error(f"Unexpected error during {data_name} validation: {str(e)}")
        
        return result
    
    def validate_model_path(self, path: str) -> ValidationResult:
        """Validate model file path."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], recommendations=[])
        
        try:
            if not isinstance(path, str):
                result.add_error(f"Model path must be a string, got {type(path)}")
                return result
            
            if not path:
                result.add_error("Model path cannot be empty")
                return result
            
            if not os.path.exists(path):
                result.add_error(f"Model file not found: {path}")
                return result
            
            if not os.path.isfile(path):
                result.add_error(f"Path is not a file: {path}")
                return result
            
            # Check file extension
            _, ext = os.path.splitext(path)
            supported_extensions = ['.onnx', '.pt', '.pth', '.pb', '.tflite']
            if ext.lower() not in supported_extensions:
                result.add_warning(f"Unsupported file extension '{ext}'. Supported: {', '.join(supported_extensions)}")
            
            # Check file size
            file_size = os.path.getsize(path)
            if file_size == 0:
                result.add_error("Model file is empty")
            elif file_size > 1_000_000_000:  # 1GB
                result.add_warning(f"Large model file ({file_size / 1_000_000:.1f} MB) may take time to process")
            
            # Check file permissions
            if not os.access(path, os.R_OK):
                result.add_error(f"Cannot read model file: {path}")
            
        except Exception as e:
            result.add_error(f"Unexpected error during model path validation: {str(e)}")
        
        return result
    
    def validate_compilation_compatibility(self, config: TargetConfig, 
                                         model_info: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Validate compatibility between configuration and model."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], recommendations=[])
        
        try:
            if not model_info:
                result.add_warning("No model information provided for compatibility check")
                return result
            
            # Check model size vs array capacity
            if 'parameters' in model_info:
                param_count = model_info['parameters']
                array_capacity = config.array_size[0] * config.array_size[1]
                
                if param_count > array_capacity * 1000:  # Rough heuristic
                    result.add_warning(f"Model has {param_count:,} parameters, may exceed array capacity")
                    result.add_recommendation("Consider using a larger array size or model pruning")
            
            # Check precision requirements
            if 'requires_fp32' in model_info and model_info['requires_fp32']:
                if config.precision in [Precision.INT8, Precision.INT16]:
                    result.add_warning("Model may require FP32 precision for accuracy")
            
            # Check operation types
            if 'unsupported_ops' in model_info:
                unsupported = model_info['unsupported_ops']
                if unsupported:
                    result.add_error(f"Model contains unsupported operations: {', '.join(unsupported)}")
            
        except Exception as e:
            result.add_error(f"Unexpected error during compatibility validation: {str(e)}")
        
        return result
    
    def validate_simulation_params(self, noise_model: str, precision: str, 
                                 crosstalk_db: float) -> ValidationResult:
        """Validate simulation parameters."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], recommendations=[])
        
        try:
            # Validate noise model
            valid_noise_models = ['ideal', 'realistic', 'pessimistic']
            if noise_model not in valid_noise_models:
                result.add_error(f"Invalid noise model '{noise_model}'. Valid options: {', '.join(valid_noise_models)}")
            
            # Validate precision
            valid_precisions = ['8bit', '16bit', 'fp16', 'fp32']
            if precision not in valid_precisions:
                result.add_error(f"Invalid precision '{precision}'. Valid options: {', '.join(valid_precisions)}")
            
            # Validate crosstalk
            if not isinstance(crosstalk_db, (int, float)):
                result.add_error("Crosstalk must be a number")
            elif crosstalk_db > 0:
                result.add_error("Crosstalk must be negative (dB)")
            elif crosstalk_db < -60:
                result.add_warning(f"Very low crosstalk ({crosstalk_db} dB) may be unrealistic")
            elif crosstalk_db > -10:
                result.add_warning(f"High crosstalk ({crosstalk_db} dB) will significantly impact accuracy")
            
        except Exception as e:
            result.add_error(f"Unexpected error during simulation parameter validation: {str(e)}")
        
        return result
    
    def comprehensive_validation(self, config: TargetConfig, 
                               model_path: Optional[str] = None,
                               input_data: Optional['np.ndarray'] = None,
                               simulation_params: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """Perform comprehensive validation of all components."""
        print("🔍 Running comprehensive validation...")
        
        overall_result = ValidationResult(is_valid=True, errors=[], warnings=[], recommendations=[])
        
        # Validate configuration
        print("   • Validating target configuration...")
        config_result = self.validate_target_config(config)
        overall_result.errors.extend(config_result.errors)
        overall_result.warnings.extend(config_result.warnings)
        overall_result.recommendations.extend(config_result.recommendations)
        if not config_result.is_valid:
            overall_result.is_valid = False
        
        # Validate model path if provided
        if model_path:
            print("   • Validating model path...")
            model_result = self.validate_model_path(model_path)
            overall_result.errors.extend(model_result.errors)
            overall_result.warnings.extend(model_result.warnings)
            overall_result.recommendations.extend(model_result.recommendations)
            if not model_result.is_valid:
                overall_result.is_valid = False
        
        # Validate input data if provided
        if input_data is not None:
            print("   • Validating input data...")
            data_result = self.validate_input_data(input_data)
            overall_result.errors.extend(data_result.errors)
            overall_result.warnings.extend(data_result.warnings)
            overall_result.recommendations.extend(data_result.recommendations)
            if not data_result.is_valid:
                overall_result.is_valid = False
        
        # Validate simulation parameters if provided
        if simulation_params:
            print("   • Validating simulation parameters...")
            sim_result = self.validate_simulation_params(**simulation_params)
            overall_result.errors.extend(sim_result.errors)
            overall_result.warnings.extend(sim_result.warnings)
            overall_result.recommendations.extend(sim_result.recommendations)
            if not sim_result.is_valid:
                overall_result.is_valid = False
        
        # Apply strict mode
        if self.strict_mode and overall_result.warnings:
            for warning in overall_result.warnings:
                overall_result.add_error(f"[STRICT MODE] {warning}")
        
        return overall_result


def validate_and_raise(validator: PhotonicValidator, *args, **kwargs) -> ValidationResult:
    """Convenience function to validate and raise on error."""
    result = validator.comprehensive_validation(*args, **kwargs)
    result.raise_if_invalid()
    return result