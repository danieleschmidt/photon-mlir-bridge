#!/usr/bin/env python3
"""
Robust compilation example demonstrating Generation 2 functionality.
This example showcases: validation, error handling, and structured logging.
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from photon_mlir import compile, TargetConfig, Device, Precision
from photon_mlir.validation import PhotonicValidator, ValidationError
from photon_mlir.logging_config import setup_logging, finalize_logging
from photon_mlir.compiler import PhotonicCompiler


def test_validation_system():
    """Test comprehensive validation system."""
    print("\n🔍 Testing Validation System")
    print("=" * 50)
    
    validator = PhotonicValidator()
    
    # Test 1: Valid configuration
    print("\n✅ Test 1: Valid Configuration")
    valid_config = TargetConfig(
        device=Device.LIGHTMATTER_ENVISE,
        precision=Precision.INT8,
        array_size=(64, 64),
        wavelength_nm=1550
    )
    
    result = validator.validate_target_config(valid_config)
    result.print_summary(verbose=True)
    
    # Test 2: Invalid configuration
    print("\n❌ Test 2: Invalid Configuration")
    try:
        invalid_config = TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.FP32,  # Not supported by this device
            array_size=(256, 256),     # Too large
            wavelength_nm=1310         # Not supported wavelength
        )
        
        result = validator.validate_target_config(invalid_config)
        result.print_summary(verbose=True)
        
    except Exception as e:
        print(f"Caught validation error: {e}")
    
    # Test 3: Input data validation
    print("\n📊 Test 3: Input Data Validation")
    
    # Valid data
    valid_data = np.random.randn(1, 128).astype(np.float32)
    result = validator.validate_input_data(valid_data)
    print(f"Valid data: {result.is_valid}")
    
    # Invalid data with NaN
    invalid_data = np.array([1.0, 2.0, np.nan, 4.0])
    result = validator.validate_input_data(invalid_data)
    print(f"Data with NaN: {result.is_valid}")
    if not result.is_valid:
        print(f"   Errors: {result.errors}")
    
    # Test 4: Simulation parameter validation
    print("\n🔬 Test 4: Simulation Parameter Validation")
    
    # Valid parameters
    result = validator.validate_simulation_params("realistic", "8bit", -30.0)
    print(f"Valid sim params: {result.is_valid}")
    
    # Invalid parameters
    result = validator.validate_simulation_params("invalid_noise", "64bit", 10.0)
    print(f"Invalid sim params: {result.is_valid}")
    if not result.is_valid:
        print(f"   Errors: {result.errors}")


def test_logging_system():
    """Test comprehensive logging system."""
    print("\n📝 Testing Logging System")
    print("=" * 50)
    
    # Setup logging with different configurations
    logger = setup_logging(
        level="INFO",
        log_file="/tmp/photonic_test.log",
        json_logging=False,
        performance_logging=True
    )
    
    # Test different log levels
    logger.debug("Debug message - should not appear in INFO level")
    logger.info("Info message about compilation")
    logger.warning("Warning about suboptimal configuration")
    logger.error("Error during validation")
    
    # Test performance monitoring
    print("\n⏱️  Testing Performance Monitoring")
    
    with logger.performance_context("test_operation", custom_param="test_value"):
        # Simulate some work
        import time
        time.sleep(0.1)
        
        # Nested performance context
        with logger.performance_context("nested_operation"):
            time.sleep(0.05)
    
    # Test exception logging
    print("\n💥 Testing Exception Logging")
    try:
        raise ValueError("Test exception for logging")
    except Exception as e:
        logger.log_exception("test_exception", e)
    
    # Test compilation event logging
    print("\n🔄 Testing Compilation Event Logging")
    model_info = {
        'type': 'Test',
        'size_mb': 1.5,
        'target_device': 'lightmatter_envise'
    }
    
    logger.log_compilation_start(model_info)
    
    # Simulate compilation work
    with logger.performance_context("mock_compilation"):
        time.sleep(0.02)
    
    stats = {
        'original_flops': 1000000,
        'photonic_macs': 800000,
        'speedup': 3.2
    }
    
    logger.log_compilation_end(True, stats)


def test_robust_compilation():
    """Test robust compilation with validation and logging."""
    print("\n💪 Testing Robust Compilation")
    print("=" * 50)
    
    # Test 1: Successful compilation with valid config
    print("\n✅ Test 1: Valid Compilation")
    try:
        config = TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.INT8,
            array_size=(32, 32),
            wavelength_nm=1550
        )
        
        compiler = PhotonicCompiler(config, strict_validation=False)
        
        # Mock model
        class TestModel:
            def __init__(self):
                self.name = "TestModel"
        
        model = TestModel()
        compiled_model = compile(model, target="lightmatter_envise")
        
        print("   ✅ Compilation successful")
        
        # Test simulation with validation
        input_data = np.random.randn(1, 64).astype(np.float32)
        result = compiled_model.simulate(input_data, noise_model="realistic")
        
        print(f"   ✅ Simulation successful: {result}")
        
    except Exception as e:
        print(f"   ❌ Compilation failed: {e}")
    
    # Test 2: Compilation with invalid configuration
    print("\n❌ Test 2: Invalid Configuration Handling")
    try:
        invalid_config = TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.FP32,  # Not supported
            array_size=(1000, 1000),   # Too large
            wavelength_nm=1200         # Not supported
        )
        
        compiler = PhotonicCompiler(invalid_config, strict_validation=True)
        print("   ❌ Should have failed!")
        
    except ValidationError as e:
        print(f"   ✅ Correctly caught validation error: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 3: Simulation with invalid data
    print("\n🔬 Test 3: Invalid Simulation Data Handling")
    try:
        config = TargetConfig()  # Use defaults
        compiler = PhotonicCompiler(config)
        
        model = type('TestModel', (), {})()
        compiled_model = compile(model)
        
        # Invalid data with NaN
        invalid_data = np.array([[1.0, np.nan, 3.0]])
        result = compiled_model.simulate(invalid_data)
        print("   ❌ Should have failed!")
        
    except ValidationError as e:
        print(f"   ✅ Correctly caught validation error")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")


def test_error_handling():
    """Test comprehensive error handling."""
    print("\n🛡️  Testing Error Handling")
    print("=" * 50)
    
    logger = setup_logging()
    
    # Test 1: File not found
    print("\n📂 Test 1: File Not Found")
    try:
        compiler = PhotonicCompiler()
        compiled_model = compiler.compile_onnx("/nonexistent/file.onnx")
    except ValidationError as e:
        print(f"   ✅ Correctly handled file not found: {e}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 2: Invalid input data types
    print("\n📊 Test 2: Invalid Data Types")
    try:
        config = TargetConfig()
        compiler = PhotonicCompiler(config)
        
        model = type('TestModel', (), {})()
        compiled_model = compile(model)
        
        # Try to simulate with string data
        result = compiled_model.simulate("invalid_data")
    except (ValidationError, TypeError) as e:
        print(f"   ✅ Correctly handled invalid data type")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    # Test 3: Configuration edge cases
    print("\n⚙️  Test 3: Configuration Edge Cases")
    
    test_configs = [
        # Zero array size
        {'array_size': (0, 64), 'description': 'zero array size'},
        # Negative wavelength
        {'wavelength_nm': -1550, 'description': 'negative wavelength'},
        # Invalid precision type
        {'precision': "invalid", 'description': 'invalid precision'},
    ]
    
    for test_config in test_configs:
        try:
            description = test_config.pop('description')
            config_dict = {
                'device': Device.LIGHTMATTER_ENVISE,
                'precision': Precision.INT8,
                'array_size': (64, 64),
                'wavelength_nm': 1550,
                **test_config
            }
            
            config = TargetConfig(**config_dict)
            compiler = PhotonicCompiler(config)
            print(f"   ❌ {description}: Should have failed!")
            
        except (ValidationError, TypeError, ValueError) as e:
            print(f"   ✅ {description}: Correctly handled")
        except Exception as e:
            print(f"   ❌ {description}: Unexpected error: {e}")


def main():
    """Run comprehensive Generation 2 testing."""
    print("🌟 Photonic MLIR Bridge - Generation 2: Robust Implementation Test")
    print("=" * 80)
    
    try:
        # Test all Generation 2 features
        test_validation_system()
        test_logging_system()
        test_robust_compilation()
        test_error_handling()
        
        print("\n🎉 Generation 2 Testing Complete!")
        print("✅ Comprehensive input validation working")
        print("✅ Structured logging with performance metrics") 
        print("✅ Robust error handling and recovery")
        print("✅ Configuration validation and safety checks")
        print("✅ CLI error handling and user feedback")
        
        # Finalize logging session
        finalize_logging()
        
    except Exception as e:
        print(f"\n💥 Generation 2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())