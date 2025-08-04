"""
Unit tests for photon_mlir.core module.
"""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from photon_mlir.core import (
    Device, Precision, TargetConfig, PhotonicTensor
)


class TestDevice:
    """Test Device enum."""
    
    def test_device_values(self):
        """Test device enum values."""
        assert Device.LIGHTMATTER_ENVISE.value == "lightmatter_envise"
        assert Device.MIT_PHOTONIC_PROCESSOR.value == "mit_photonic_processor"
        assert Device.CUSTOM_RESEARCH_CHIP.value == "custom_research_chip"
    
    def test_device_from_string(self):
        """Test creating device from string."""
        device = Device("lightmatter_envise")
        assert device == Device.LIGHTMATTER_ENVISE


class TestPrecision:
    """Test Precision enum."""
    
    def test_precision_values(self):
        """Test precision enum values."""
        assert Precision.INT8.value == "int8"
        assert Precision.INT16.value == "int16"
        assert Precision.FP16.value == "fp16"
        assert Precision.FP32.value == "fp32"
    
    def test_precision_from_string(self):
        """Test creating precision from string."""
        precision = Precision("int8")
        assert precision == Precision.INT8


class TestTargetConfig:
    """Test TargetConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TargetConfig()
        
        assert config.device == Device.LIGHTMATTER_ENVISE
        assert config.precision == Precision.INT8
        assert config.array_size == (64, 64)
        assert config.wavelength_nm == 1550
        assert config.max_phase_drift == 0.1
        assert config.calibration_interval_ms == 100
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TargetConfig(
            device=Device.MIT_PHOTONIC_PROCESSOR,
            precision=Precision.FP32,
            array_size=(32, 32),
            wavelength_nm=1310,
            max_phase_drift=0.05,
            calibration_interval_ms=50
        )
        
        assert config.device == Device.MIT_PHOTONIC_PROCESSOR
        assert config.precision == Precision.FP32
        assert config.array_size == (32, 32)
        assert config.wavelength_nm == 1310
        assert config.max_phase_drift == 0.05
        assert config.calibration_interval_ms == 50
    
    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TargetConfig()
        config_dict = config.to_dict()
        
        expected_keys = {
            'device', 'precision', 'array_size', 
            'wavelength_nm', 'max_phase_drift', 'calibration_interval_ms'
        }
        assert set(config_dict.keys()) == expected_keys
        assert config_dict['device'] == "lightmatter_envise"
        assert config_dict['precision'] == "int8"


class TestPhotonicTensor:
    """Test PhotonicTensor class."""
    
    def test_creation_with_numpy(self):
        """Test creating tensor with numpy array."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        tensor = PhotonicTensor(data, wavelength=1550, power_mw=10.0)
        
        assert np.array_equal(tensor.data, data)
        assert tensor.wavelength == 1550
        assert tensor.power_mw == 10.0
        assert tensor.shape == (2, 3)
    
    def test_creation_with_list(self):
        """Test creating tensor with list."""
        data = [[1, 2], [3, 4]]
        tensor = PhotonicTensor(data)
        
        assert tensor.data == data
        assert tensor.wavelength == 1550  # Default
        assert tensor.power_mw == 1.0     # Default
        assert tensor.shape is None       # Lists don't have shape attribute
    
    def test_default_values(self):
        """Test default wavelength and power values."""
        tensor = PhotonicTensor([1, 2, 3])
        
        assert tensor.wavelength == 1550
        assert tensor.power_mw == 1.0
    
    def test_repr(self):
        """Test string representation."""
        data = np.array([1, 2, 3])
        tensor = PhotonicTensor(data, wavelength=1310, power_mw=5.0)
        
        repr_str = repr(tensor)
        assert "PhotonicTensor" in repr_str
        assert "shape=(3,)" in repr_str
        assert "wavelength=1310nm" in repr_str
        assert "power=5.0mW" in repr_str
    
    @pytest.mark.parametrize("wavelength,power", [
        (1200, 0.5),
        (1550, 10.0),
        (1700, 100.0)
    ])
    def test_various_parameters(self, wavelength, power):
        """Test tensor with various wavelength and power values."""
        data = np.zeros((5, 5))
        tensor = PhotonicTensor(data, wavelength=wavelength, power_mw=power)
        
        assert tensor.wavelength == wavelength
        assert tensor.power_mw == power
        assert tensor.shape == (5, 5)


class TestTargetConfigValidation:
    """Test target configuration validation."""
    
    def test_valid_wavelengths(self):
        """Test valid wavelength values."""
        valid_wavelengths = [1200, 1310, 1550, 1700]
        
        for wavelength in valid_wavelengths:
            config = TargetConfig(wavelength_nm=wavelength)
            assert config.wavelength_nm == wavelength
    
    def test_valid_array_sizes(self):
        """Test valid array size values."""
        valid_sizes = [(1, 1), (32, 32), (64, 64), (128, 128)]
        
        for size in valid_sizes:
            config = TargetConfig(array_size=size)
            assert config.array_size == size
    
    def test_valid_phase_drift(self):
        """Test valid phase drift values."""
        valid_drifts = [0.0, 0.05, 0.1, 0.2, 3.14159]
        
        for drift in valid_drifts:
            config = TargetConfig(max_phase_drift=drift)
            assert config.max_phase_drift == drift
    
    def test_valid_calibration_intervals(self):
        """Test valid calibration interval values."""
        valid_intervals = [1, 50, 100, 1000, 10000]
        
        for interval in valid_intervals:
            config = TargetConfig(calibration_interval_ms=interval) 
            assert config.calibration_interval_ms == interval


@pytest.fixture
def sample_config():
    """Fixture providing a sample target configuration."""
    return TargetConfig(
        device=Device.LIGHTMATTER_ENVISE,
        precision=Precision.INT8,
        array_size=(64, 64),
        wavelength_nm=1550,
        max_phase_drift=0.1,
        calibration_interval_ms=100
    )


@pytest.fixture
def sample_tensor():
    """Fixture providing a sample photonic tensor."""
    data = np.random.randn(10, 10)
    return PhotonicTensor(data, wavelength=1550, power_mw=5.0)


class TestIntegrationScenarios:
    """Test integration scenarios with core components."""
    
    def test_config_tensor_compatibility(self, sample_config, sample_tensor):
        """Test compatibility between config and tensor."""
        # Both should use same wavelength
        assert sample_config.wavelength_nm == sample_tensor.wavelength
    
    def test_multiple_configs(self):
        """Test creating multiple different configurations."""
        configs = [
            TargetConfig(device=Device.LIGHTMATTER_ENVISE, precision=Precision.INT8),
            TargetConfig(device=Device.MIT_PHOTONIC_PROCESSOR, precision=Precision.FP16),
            TargetConfig(device=Device.CUSTOM_RESEARCH_CHIP, precision=Precision.FP32)
        ]
        
        assert len(configs) == 3
        assert configs[0].device != configs[1].device
        assert configs[1].precision != configs[2].precision
    
    def test_tensor_operations(self, sample_tensor):
        """Test basic tensor operations."""
        # Test that we can access the underlying data
        assert hasattr(sample_tensor.data, 'shape')
        assert sample_tensor.data.shape == (10, 10)
        
        # Test that tensor maintains metadata
        assert sample_tensor.wavelength == 1550
        assert sample_tensor.power_mw == 5.0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_tensor_with_none_data(self):
        """Test tensor with None data."""
        tensor = PhotonicTensor(None)
        assert tensor.data is None
        assert tensor.shape is None
    
    def test_tensor_with_empty_array(self):
        """Test tensor with empty numpy array."""
        data = np.array([])
        tensor = PhotonicTensor(data)
        assert tensor.shape == (0,)
    
    def test_config_extreme_values(self):
        """Test configuration with extreme but valid values."""
        config = TargetConfig(
            array_size=(1, 1),
            wavelength_nm=1200,
            max_phase_drift=0.0,
            calibration_interval_ms=1
        )
        
        assert config.array_size == (1, 1)
        assert config.wavelength_nm == 1200
        assert config.max_phase_drift == 0.0
        assert config.calibration_interval_ms == 1


if __name__ == "__main__":
    pytest.main([__file__])