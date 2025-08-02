"""End-to-end integration tests for the complete compilation pipeline."""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch


class TestCompilationPipeline:
    """Test the complete compilation pipeline from model to hardware."""
    
    def test_pytorch_to_assembly_pipeline(self, simple_linear_model, temp_dir):
        """Test complete pipeline: PyTorch → MLIR → Assembly → Hardware."""
        
        # Mock the complete pipeline
        with patch('photon_mlir.compile') as mock_compile:
            # Setup mock compiled model
            mock_compiled = Mock()
            mock_compiled.target = "simulation"
            mock_compiled.assembly_code = "PLOAD %weights, @model_weights\nPMUL %result, %input, %weights"
            mock_compiled.phase_shifts = 150
            mock_compiled.thermal_calibration = True
            
            def mock_export(filename):
                Path(filename).write_text(mock_compiled.assembly_code)
            
            mock_compiled.export = mock_export
            mock_compile.return_value = mock_compiled
            
            # Run the pipeline
            compiled = mock_compile(simple_linear_model, target="simulation")
            
            # Export assembly
            assembly_file = temp_dir / "model.pasm"
            compiled.export(str(assembly_file))
            
            # Verify outputs
            assert assembly_file.exists()
            assembly_content = assembly_file.read_text()
            assert "PLOAD" in assembly_content
            assert "PMUL" in assembly_content
            assert compiled.phase_shifts > 0
    
    def test_onnx_to_photonic_pipeline(self, conv_model, temp_dir):
        """Test ONNX model compilation pipeline."""
        
        # Export PyTorch model to ONNX
        dummy_input = torch.randn(1, 3, 32, 32)
        onnx_path = temp_dir / "conv_model.onnx"
        
        torch.onnx.export(
            conv_model,
            dummy_input,
            str(onnx_path),
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}}
        )
        
        # Mock ONNX compilation
        with patch('photon_mlir.load_onnx') as mock_load, \
             patch('photon_mlir.compile') as mock_compile:
            
            mock_load.return_value = conv_model
            mock_compiled = Mock()
            mock_compiled.target = "lightmatter_envise"
            mock_compiled.optimization_report = {
                'original_flops': 5000000,
                'photonic_macs': 2500000,
                'speedup': 4.2,
                'energy_reduction': 87.3
            }
            mock_compile.return_value = mock_compiled
            
            # Load and compile ONNX model
            loaded_model = mock_load(str(onnx_path))
            compiled = mock_compile(loaded_model, target="lightmatter_envise")
            
            assert compiled.target == "lightmatter_envise"
            assert compiled.optimization_report['speedup'] > 1.0
    
    def test_optimization_pipeline_stages(self, transformer_block):
        """Test different optimization stages in the pipeline."""
        
        optimization_levels = [0, 1, 2, 3]
        
        with patch('photon_mlir.compile') as mock_compile:
            results = {}
            
            for level in optimization_levels:
                mock_compiled = Mock()
                mock_compiled.optimization_level = level
                mock_compiled.phase_shifts = max(1000 - level * 200, 200)  # Fewer phase shifts with higher optimization
                mock_compiled.compilation_time = 10 + level * 5  # More time for higher optimization
                mock_compile.return_value = mock_compiled
                
                compiled = mock_compile(
                    transformer_block,
                    target="simulation",
                    optimization_level=level
                )
                
                results[level] = {
                    'phase_shifts': compiled.phase_shifts,
                    'compilation_time': compiled.compilation_time
                }
            
            # Verify optimization progression
            assert results[3]['phase_shifts'] < results[0]['phase_shifts']
            assert results[3]['compilation_time'] > results[0]['compilation_time']
    
    @pytest.mark.slow
    def test_large_model_pipeline(self, model_factory, temp_dir):
        """Test pipeline with a large model."""
        
        # Create a large model
        large_model = model_factory.create_linear_stack(
            num_layers=50,
            input_size=2048,
            hidden_size=2048,
            output_size=1000
        )
        
        with patch('photon_mlir.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compiled.target = "simulation"
            mock_compiled.model_size_mb = 400.5  # Large model
            mock_compiled.partitions = 4  # Multi-chip deployment
            mock_compiled.inter_chip_communication = True
            
            def mock_export_multi_chip(base_filename):
                for i in range(mock_compiled.partitions):
                    chip_file = temp_dir / f"{Path(base_filename).stem}_chip_{i}.pasm"
                    chip_file.write_text(f"# Chip {i} assembly code\nPLOAD %weights_{i}, @partition_{i}")
            
            mock_compiled.export_multi_chip = mock_export_multi_chip
            mock_compile.return_value = mock_compiled
            
            # Compile large model
            compiled = mock_compile(
                large_model,
                target="simulation",
                enable_multi_chip=True
            )
            
            # Export multi-chip deployment
            compiled.export_multi_chip(str(temp_dir / "large_model"))
            
            # Verify multi-chip files were created
            for i in range(compiled.partitions):
                chip_file = temp_dir / f"large_model_chip_{i}.pasm"
                assert chip_file.exists()
                content = chip_file.read_text()
                assert f"partition_{i}" in content
    
    def test_error_handling_in_pipeline(self, simple_linear_model):
        """Test error handling throughout the compilation pipeline."""
        
        # Test compilation error
        with patch('photon_mlir.compile') as mock_compile:
            mock_compile.side_effect = RuntimeError("MLIR compilation failed")
            
            with pytest.raises(RuntimeError, match="MLIR compilation failed"):
                mock_compile(simple_linear_model, target="simulation")
        
        # Test export error
        with patch('photon_mlir.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compiled.export.side_effect = IOError("Failed to write assembly file")
            mock_compile.return_value = mock_compiled
            
            compiled = mock_compile(simple_linear_model, target="simulation")
            
            with pytest.raises(IOError, match="Failed to write assembly file"):
                compiled.export("output.pasm")
    
    def test_pipeline_with_custom_passes(self, conv_model):
        """Test pipeline with custom optimization passes."""
        
        with patch('photon_mlir.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compiled.custom_passes_applied = [
                "CustomPhaseOptimization",
                "CustomThermalCompensation",
                "CustomPowerBalancing"
            ]
            mock_compiled.pass_statistics = {
                "CustomPhaseOptimization": {"time_ms": 150, "phase_reduction": 25},
                "CustomThermalCompensation": {"time_ms": 75, "accuracy_improvement": 12},
                "CustomPowerBalancing": {"time_ms": 50, "power_uniformity": 0.95}
            }
            mock_compile.return_value = mock_compiled
            
            compiled = mock_compile(
                conv_model,
                target="simulation",
                custom_passes=["CustomPhaseOptimization", "CustomThermalCompensation"]
            )
            
            assert "CustomPhaseOptimization" in compiled.custom_passes_applied
            assert compiled.pass_statistics["CustomPhaseOptimization"]["phase_reduction"] > 0
    
    def test_hardware_deployment_pipeline(self, simple_linear_model, mock_hardware_config):
        """Test hardware deployment pipeline."""
        
        with patch('photon_mlir.compile') as mock_compile, \
             patch('photon_mlir.connect_hardware') as mock_connect:
            
            # Mock compiled model
            mock_compiled = Mock()
            mock_compiled.target = "lightmatter_envise"
            mock_compiled.hardware_code = "HARDWARE_SPECIFIC_CODE"
            mock_compile.return_value = mock_compiled
            
            # Mock hardware connection
            mock_device = Mock()
            mock_device.upload = Mock(return_value=True)
            mock_device.calibrate = Mock(return_value={"status": "success"})
            mock_device.test_inference = Mock(return_value=torch.randn(1, 5))
            mock_connect.return_value = mock_device
            
            # Compile model
            compiled = mock_compile(simple_linear_model, target="lightmatter_envise")
            
            # Connect to hardware
            device = mock_connect("lightmatter://192.168.1.100")
            
            # Deploy to hardware
            upload_success = device.upload(compiled.hardware_code)
            calibration_result = device.calibrate()
            test_output = device.test_inference(torch.randn(1, 10))
            
            assert upload_success
            assert calibration_result["status"] == "success"
            assert test_output.shape == (1, 5)
    
    def test_pipeline_performance_tracking(self, simple_linear_model, performance_tracker):
        """Test performance tracking throughout the pipeline."""
        
        with patch('photon_mlir.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compiled.compilation_metrics = {
                "total_time_ms": 2500,
                "memory_peak_mb": 150,
                "optimization_time_ms": 800,
                "lowering_time_ms": 1200,
                "codegen_time_ms": 500
            }
            mock_compile.return_value = mock_compiled
            
            # Track performance
            performance_tracker.start()
            compiled = mock_compile(simple_linear_model, target="simulation")
            metrics = performance_tracker.stop()
            
            # Verify tracking
            assert metrics['execution_time'] > 0
            assert metrics['memory_delta'] is not None
            assert compiled.compilation_metrics["total_time_ms"] > 0
    
    def test_pipeline_with_validation(self, conv_model, sample_input_data):
        """Test pipeline with model validation."""
        
        with patch('photon_mlir.compile') as mock_compile:
            mock_compiled = Mock()
            mock_compiled.target = "simulation"
            
            def mock_validate(input_tensor):
                # Mock validation - check shape and run inference
                with torch.no_grad():
                    original_output = conv_model(input_tensor)
                    simulated_output = original_output + torch.randn_like(original_output) * 0.01
                
                mse = torch.nn.functional.mse_loss(original_output, simulated_output)
                return {
                    "mse": mse.item(),
                    "max_error": torch.max(torch.abs(original_output - simulated_output)).item(),
                    "accuracy_preserved": mse.item() < 0.01
                }
            
            mock_compiled.validate = mock_validate
            mock_compile.return_value = mock_compiled
            
            # Compile with validation
            compiled = mock_compile(conv_model, target="simulation", enable_validation=True)
            
            # Run validation
            validation_input = sample_input_data["conv"]
            validation_result = compiled.validate(validation_input)
            
            assert validation_result["mse"] >= 0
            assert validation_result["accuracy_preserved"]
    
    def test_incremental_compilation_pipeline(self, model_factory):
        """Test incremental compilation for model updates."""
        
        # Create base model
        base_model = model_factory.create_linear_stack(3, 100, 50, 10)
        
        # Create modified model (one additional layer)
        modified_model = model_factory.create_linear_stack(4, 100, 50, 10)
        
        with patch('photon_mlir.compile') as mock_compile:
            # Mock incremental compilation
            mock_base_compiled = Mock()
            mock_base_compiled.compilation_id = "base_v1"
            mock_base_compiled.compilation_time_ms = 1000
            
            mock_incremental_compiled = Mock() 
            mock_incremental_compiled.compilation_id = "incremental_v2"
            mock_incremental_compiled.compilation_time_ms = 300  # Faster incremental
            mock_incremental_compiled.base_compilation = "base_v1"
            mock_incremental_compiled.changes_applied = ["added_layer_3"]
            
            # First compilation
            mock_compile.return_value = mock_base_compiled
            base_compiled = mock_compile(base_model, target="simulation")
            
            # Incremental compilation
            mock_compile.return_value = mock_incremental_compiled
            incremental_compiled = mock_compile(
                modified_model,
                target="simulation",
                base_compilation=base_compiled.compilation_id
            )
            
            # Verify incremental compilation was faster
            assert incremental_compiled.compilation_time_ms < base_compiled.compilation_time_ms
            assert incremental_compiled.base_compilation == "base_v1"
            assert "added_layer_3" in incremental_compiled.changes_applied