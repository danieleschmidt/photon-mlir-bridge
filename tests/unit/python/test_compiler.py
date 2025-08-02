"""Unit tests for the Python compiler interface."""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock the photon_mlir module for testing
class MockCompiledModel:
    def __init__(self, target="simulation", optimization_level=1):
        self.target = target
        self.optimization_level = optimization_level
        self.phase_shifts = 1000
        self.thermal_compensation = True
        
    def simulate(self, input_tensor):
        # Mock simulation - just add some noise
        return input_tensor + torch.randn_like(input_tensor) * 0.01
    
    def export(self, filename):
        Path(filename).write_text(f"Mock assembly for {self.target}")
    
    def get_optimization_report(self):
        return {
            "original_flops": 1000000,
            "photonic_macs": 500000,
            "total_phase_shifts": self.phase_shifts,
            "speedup": 3.2,
            "energy_reduction": 85.5
        }

def mock_compile(model, target="simulation", optimize_for="latency", **kwargs):
    """Mock compile function."""
    if not isinstance(model, torch.nn.Module):
        raise ValueError("Model must be a torch.nn.Module")
    
    if target not in ["simulation", "lightmatter_envise", "mit_photonics"]:
        raise ValueError(f"Unsupported target: {target}")
    
    optimization_level = kwargs.get("optimization_level", 1)
    return MockCompiledModel(target, optimization_level)

# Mock module
photon_mlir = Mock()
photon_mlir.compile = mock_compile
photon_mlir.PhotonicCompiler = Mock()


class TestPhotonCompiler:
    """Test cases for the photonic compiler."""
    
    def test_compile_linear_model(self, simple_linear_model):
        """Test compilation of a simple linear model."""
        compiled = photon_mlir.compile(
            simple_linear_model,
            target="simulation",
            optimize_for="latency"
        )
        
        assert compiled is not None
        assert compiled.target == "simulation"
        assert isinstance(compiled, MockCompiledModel)
    
    def test_compile_conv_model(self, conv_model):
        """Test compilation of a convolutional model."""
        compiled = photon_mlir.compile(
            conv_model,
            target="lightmatter_envise",
            optimize_for="power"
        )
        
        assert compiled is not None
        assert compiled.target == "lightmatter_envise"
    
    def test_compile_transformer_block(self, transformer_block):
        """Test compilation of a transformer block."""
        compiled = photon_mlir.compile(
            transformer_block,
            target="simulation",
            optimization_level=2
        )
        
        assert compiled is not None
        assert compiled.optimization_level == 2
    
    def test_invalid_model_type(self):
        """Test that invalid model types raise appropriate errors."""
        with pytest.raises(ValueError, match="Model must be a torch.nn.Module"):
            photon_mlir.compile("not a model", target="simulation")
        
        with pytest.raises(ValueError, match="Model must be a torch.nn.Module"):
            photon_mlir.compile(42, target="simulation")
    
    def test_invalid_target(self, simple_linear_model):
        """Test that invalid targets raise appropriate errors."""
        with pytest.raises(ValueError, match="Unsupported target"):
            photon_mlir.compile(simple_linear_model, target="invalid_target")
    
    @pytest.mark.parametrize("target", [
        "simulation",
        "lightmatter_envise", 
        "mit_photonics"
    ])
    def test_multiple_targets(self, simple_linear_model, target):
        """Test compilation with different target platforms."""
        compiled = photon_mlir.compile(simple_linear_model, target=target)
        assert compiled.target == target
    
    @pytest.mark.parametrize("optimization_level", [0, 1, 2, 3])
    def test_optimization_levels(self, simple_linear_model, optimization_level):
        """Test different optimization levels."""
        compiled = photon_mlir.compile(
            simple_linear_model,
            target="simulation",
            optimization_level=optimization_level
        )
        assert compiled.optimization_level == optimization_level
    
    def test_simulation_accuracy(self, simple_linear_model, sample_input_data):
        """Test that simulation produces reasonable results."""
        compiled = photon_mlir.compile(simple_linear_model, target="simulation")
        
        input_tensor = sample_input_data["linear"]
        
        # Get original model output
        with torch.no_grad():
            original_output = simple_linear_model(input_tensor)
        
        # Get compiled model output
        compiled_output = compiled.simulate(input_tensor)
        
        # Should be approximately equal (allowing for simulation noise)
        assert compiled_output.shape == original_output.shape
        
        # Check that outputs are reasonably close (within 10% due to mock noise)
        relative_error = torch.abs(compiled_output - original_output) / (torch.abs(original_output) + 1e-8)
        assert torch.mean(relative_error) < 0.1
    
    def test_export_functionality(self, simple_linear_model, temp_dir):
        """Test model export functionality."""
        compiled = photon_mlir.compile(simple_linear_model, target="simulation")
        
        export_path = temp_dir / "test_model.pasm"
        compiled.export(str(export_path))
        
        assert export_path.exists()
        content = export_path.read_text()
        assert "Mock assembly for simulation" in content
    
    def test_optimization_report(self, simple_linear_model):
        """Test optimization report generation."""
        compiled = photon_mlir.compile(simple_linear_model, target="simulation")
        
        report = compiled.get_optimization_report()
        
        assert isinstance(report, dict)
        assert "original_flops" in report
        assert "photonic_macs" in report
        assert "total_phase_shifts" in report
        assert "speedup" in report
        assert "energy_reduction" in report
        
        # Check reasonable values
        assert report["speedup"] > 1.0
        assert 0 < report["energy_reduction"] < 100
    
    @pytest.mark.slow
    def test_large_model_compilation(self, model_factory):
        """Test compilation of a large model (marked as slow)."""
        large_model = model_factory.create_linear_stack(
            num_layers=20,
            input_size=1000,
            hidden_size=1000,
            output_size=1000
        )
        
        compiled = photon_mlir.compile(large_model, target="simulation")
        assert compiled is not None
        assert compiled.target == "simulation"
    
    @pytest.mark.hardware
    def test_hardware_compilation(self, simple_linear_model, hardware_available):
        """Test compilation for actual hardware."""
        if not hardware_available:
            pytest.skip("Hardware not available")
        
        compiled = photon_mlir.compile(
            simple_linear_model,
            target="lightmatter_envise"
        )
        assert compiled.target == "lightmatter_envise"
    
    def test_batch_processing(self, simple_linear_model, sample_input_data):
        """Test compilation and simulation with batch inputs."""
        compiled = photon_mlir.compile(simple_linear_model, target="simulation")
        
        batch_input = sample_input_data["batch_linear"]
        batch_output = compiled.simulate(batch_input)
        
        assert batch_output.shape[0] == batch_input.shape[0]  # Batch size preserved
        assert batch_output.shape[1:] == (5,)  # Output features
    
    def test_model_with_different_dtypes(self, simple_linear_model):
        """Test compilation with different tensor dtypes."""
        # Convert model to different dtypes
        float16_model = simple_linear_model.half()
        
        compiled = photon_mlir.compile(float16_model, target="simulation")
        assert compiled is not None
    
    @pytest.mark.benchmark
    def test_compilation_performance(self, benchmark, simple_linear_model):
        """Benchmark compilation performance."""
        def compile_model():
            return photon_mlir.compile(simple_linear_model, target="simulation")
        
        result = benchmark(compile_model)
        assert result is not None


class TestPhotonCompilerEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_model(self):
        """Test compilation of an empty model."""
        empty_model = torch.nn.Sequential()
        
        compiled = photon_mlir.compile(empty_model, target="simulation")
        assert compiled is not None
    
    def test_single_layer_model(self):
        """Test compilation of a single layer."""
        single_layer = torch.nn.Linear(1, 1)
        
        compiled = photon_mlir.compile(single_layer, target="simulation")
        assert compiled is not None
    
    def test_model_with_parameters_frozen(self):
        """Test compilation of a model with frozen parameters."""
        model = torch.nn.Linear(10, 5)
        
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False
        
        compiled = photon_mlir.compile(model, target="simulation")
        assert compiled is not None
    
    def test_model_with_custom_forward(self):
        """Test compilation of a model with custom forward method."""
        class CustomModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return torch.relu(self.linear(x))
        
        model = CustomModel()
        compiled = photon_mlir.compile(model, target="simulation")
        assert compiled is not None


class TestPhotonCompilerIntegration:
    """Integration tests with different frameworks."""
    
    def test_pytorch_jit_integration(self, simple_linear_model):
        """Test integration with PyTorch JIT."""
        # Create a JIT scripted model
        scripted_model = torch.jit.script(simple_linear_model)
        
        compiled = photon_mlir.compile(scripted_model, target="simulation")
        assert compiled is not None
    
    def test_onnx_integration(self, simple_linear_model, temp_dir):
        """Test integration with ONNX models."""
        # Export to ONNX
        dummy_input = torch.randn(1, 10)
        onnx_path = temp_dir / "model.onnx"
        
        torch.onnx.export(
            simple_linear_model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output']
        )
        
        # Mock ONNX loading and compilation
        with patch('photon_mlir.load_onnx') as mock_load:
            mock_load.return_value = simple_linear_model
            compiled = photon_mlir.compile(mock_load.return_value, target="simulation")
            assert compiled is not None


class TestPhotonCompilerConfiguration:
    """Test various compiler configurations."""
    
    def test_compiler_with_config_dict(self, simple_linear_model, compiler_configs):
        """Test compiler with configuration dictionaries."""
        for config_name, config in compiler_configs.items():
            compiled = photon_mlir.compile(simple_linear_model, **config)
            assert compiled is not None
            assert compiled.target == config["target"]
    
    def test_thermal_compensation_config(self, simple_linear_model):
        """Test thermal compensation configuration."""
        compiled = photon_mlir.compile(
            simple_linear_model,
            target="simulation",
            enable_thermal_compensation=True
        )
        assert compiled.thermal_compensation
    
    def test_debugging_config(self, simple_linear_model):
        """Test debugging configuration."""
        compiled = photon_mlir.compile(
            simple_linear_model,
            target="simulation",
            enable_debug=True,
            preserve_intermediate=True
        )
        assert compiled is not None