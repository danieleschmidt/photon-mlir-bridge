# pytest configuration and fixtures for photon-mlir tests

import os
import sys
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any
import pytest
# Graceful imports for optional dependencies
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Add src to Python path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Test configuration
pytest_plugins = ["pytest_benchmark"]


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provides path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Provides a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def simple_linear_model():
    """Creates a simple linear model for testing."""
    if not TORCH_AVAILABLE:
        pytest.skip("PyTorch not available")
    return torch.nn.Linear(10, 5)


@pytest.fixture
def conv_model() -> torch.nn.Module:
    """Creates a simple convolutional model for testing."""
    return torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10)
    )


@pytest.fixture
def transformer_block() -> torch.nn.Module:
    """Creates a simple transformer block for testing."""
    return torch.nn.TransformerEncoderLayer(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        dropout=0.1,
        batch_first=True
    )


@pytest.fixture
def sample_input_data() -> Dict[str, torch.Tensor]:
    """Provides sample input data for different model types."""
    return {
        "linear": torch.randn(1, 10),
        "conv": torch.randn(1, 3, 32, 32),
        "transformer": torch.randn(1, 100, 512),
        "batch_linear": torch.randn(32, 10),
        "batch_conv": torch.randn(32, 3, 32, 32)
    }


@pytest.fixture
def compiler_configs() -> Dict[str, Dict[str, Any]]:
    """Provides various compiler configurations for testing."""
    return {
        "simulation": {
            "target": "simulation",
            "optimization_level": 1,
            "enable_debug": True
        },
        "lightmatter": {
            "target": "lightmatter_envise",
            "optimization_level": 2,
            "array_size": (64, 64),
            "wavelength": 1550
        },
        "optimized": {
            "target": "simulation",
            "optimization_level": 3,
            "enable_thermal_compensation": True,
            "enable_phase_optimization": True
        },
        "minimal": {
            "target": "simulation",
            "optimization_level": 0,
            "enable_debug": True,
            "preserve_intermediate": True
        }
    }


@pytest.fixture
def hardware_available() -> bool:
    """Checks if actual photonic hardware is available for testing."""
    # This would check for actual hardware availability
    # For now, assume no hardware in CI environment
    return os.getenv("PHOTON_HARDWARE_AVAILABLE", "false").lower() == "true"


@pytest.fixture
def mock_hardware_config() -> Dict[str, Any]:
    """Provides mock hardware configuration for testing."""
    return {
        "device_type": "lightmatter_envise",
        "array_size": (64, 64),
        "wavelength": 1550,
        "max_power": 100.0,  # mW
        "thermal_range": (20.0, 80.0),  # Celsius
        "phase_precision": 0.01,  # radians
        "crosstalk": -30.0,  # dB
        "insertion_loss": 0.1  # dB
    }


@pytest.fixture(autouse=True)
def setup_test_environment(monkeypatch):
    """Sets up test environment variables."""
    monkeypatch.setenv("PHOTON_TEST_MODE", "1")
    monkeypatch.setenv("PHOTON_LOG_LEVEL", "WARNING")
    
    # Disable hardware detection in tests
    monkeypatch.setenv("PHOTON_FORCE_SIMULATION", "1")


class TestModelFactory:
    """Factory for creating test models with various configurations."""
    
    @staticmethod
    def create_linear_stack(num_layers: int, input_size: int, hidden_size: int, output_size: int) -> torch.nn.Module:
        """Create a stack of linear layers."""
        layers = [torch.nn.Linear(input_size, hidden_size), torch.nn.ReLU()]
        
        for _ in range(num_layers - 2):
            layers.extend([
                torch.nn.Linear(hidden_size, hidden_size),
                torch.nn.ReLU()
            ])
        
        layers.append(torch.nn.Linear(hidden_size, output_size))
        return torch.nn.Sequential(*layers)
    
    @staticmethod
    def create_resnet_block(channels: int) -> torch.nn.Module:
        """Create a simple ResNet-style block."""
        return torch.nn.Sequential(
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels, 3, padding=1),
            torch.nn.BatchNorm2d(channels)
        )
    
    @staticmethod
    def create_attention_model(d_model: int, num_heads: int) -> torch.nn.Module:
        """Create a multi-head attention model."""
        return torch.nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True
        )


@pytest.fixture
def model_factory() -> TestModelFactory:
    """Provides model factory for creating test models."""
    return TestModelFactory()


# Custom markers for test categorization
def pytest_configure(config):
    """Register custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require actual hardware"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks unit tests"
    )
    config.addinivalue_line(
        "markers", "benchmark: marks performance benchmark tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add default markers."""
    for item in items:
        # Add unit marker by default to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        
        # Add integration marker to tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Skip hardware tests by default unless explicitly enabled
        if item.get_closest_marker("hardware"):
            if not config.getoption("--run-hardware"):
                item.add_marker(pytest.mark.skip(reason="Hardware tests disabled"))


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-hardware",
        action="store_true",
        default=False,
        help="Run tests that require actual hardware"
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests"
    )
    parser.addoption(
        "--benchmark-only",
        action="store_true",
        default=False,
        help="Run only benchmark tests"
    )


# Test utilities
class TestUtils:
    """Utility functions for tests."""
    
    @staticmethod
    def assert_tensor_close(actual: torch.Tensor, expected: torch.Tensor, 
                          rtol: float = 1e-5, atol: float = 1e-8) -> None:
        """Assert that two tensors are close within tolerance."""
        torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> int:
        """Count the number of parameters in a model."""
        return sum(p.numel() for p in model.parameters())
    
    @staticmethod
    def model_size_mb(model: torch.nn.Module) -> float:
        """Calculate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)
    
    @staticmethod
    def generate_random_input(shape: tuple, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Generate random input tensor with given shape."""
        return torch.randn(shape, dtype=dtype)


@pytest.fixture
def test_utils() -> TestUtils:
    """Provides test utility functions."""
    return TestUtils()


# Logging configuration for tests
import logging

@pytest.fixture(autouse=True)
def configure_logging():
    """Configure logging for tests."""
    logging.basicConfig(
        level=logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Suppress verbose logging from external libraries
    logging.getLogger('torch').setLevel(logging.WARNING)
    logging.getLogger('numpy').setLevel(logging.WARNING)


# Performance tracking
@pytest.fixture
def performance_tracker():
    """Track performance metrics during tests."""
    import time
    import psutil
    
    class PerformanceTracker:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
        
        def stop(self):
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss
            
            return {
                'execution_time': end_time - self.start_time,
                'memory_delta': end_memory - self.start_memory,
                'peak_memory': end_memory
            }
    
    return PerformanceTracker()


# Error injection for robustness testing
@pytest.fixture
def error_injector():
    """Utility for injecting errors to test robustness."""
    
    class ErrorInjector:
        def __init__(self):
            self.injected_errors = []
        
        def inject_compilation_error(self, error_type: str = "syntax"):
            """Inject a compilation error."""
            self.injected_errors.append(("compilation", error_type))
        
        def inject_runtime_error(self, error_type: str = "memory"):
            """Inject a runtime error."""
            self.injected_errors.append(("runtime", error_type))
        
        def clear(self):
            """Clear all injected errors."""
            self.injected_errors.clear()
    
    return ErrorInjector()


# Cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup resources after each test."""
    yield
    
    # Clear CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()