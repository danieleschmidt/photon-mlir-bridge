"""
End-to-end integration tests for the photonic compiler pipeline.
"""

import pytest
import tempfile
import os
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from photon_mlir.compiler import PhotonicCompiler, compile
from photon_mlir.core import TargetConfig, Device, Precision
from photon_mlir.simulator import PhotonicSimulator
from photon_mlir.diagnostics import run_diagnostics
from photon_mlir.config import PhotonicConfig, load_config


@pytest.mark.integration
class TestFullCompilationPipeline:
    """Test complete compilation pipeline from model to hardware."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_pytorch_to_photonic_pipeline(self):
        """Test complete PyTorch to photonic compilation pipeline."""
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10),
            torch.nn.Softmax(dim=1)
        )
        
        # Configure target hardware
        config = TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.INT8,
            array_size=(64, 64),
            wavelength_nm=1550
        )
        
        # Compile model
        compiler = PhotonicCompiler(config)
        compiled_model = compiler.compile_pytorch(model)
        
        # Verify compilation
        assert compiled_model is not None
        assert compiled_model.target_config.device == Device.LIGHTMATTER_ENVISE
        
        # Test inference
        dummy_input = torch.randn(1, 784)
        output = compiled_model.simulate(dummy_input)
        
        # Verify output properties
        assert output.data.shape == (1, 10)
        assert output.wavelength == 1550
        assert output.power_mw > 0
        
        # Export model
        with tempfile.NamedTemporaryFile(suffix='.phdl', delete=False) as f:
            export_path = f.name
        
        try:
            compiled_model.export(export_path)
            assert os.path.exists(export_path)
            
            # Verify export content
            with open(export_path, 'r') as f:
                content = f.read()
                assert "lightmatter_envise" in content
                assert "int8" in content
                assert "1550" in content
        finally:
            os.unlink(export_path)
    
    def test_onnx_to_photonic_pipeline(self):
        """Test ONNX to photonic compilation pipeline."""
        # Create dummy ONNX file
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"dummy onnx content")
            onnx_path = f.name
        
        try:
            # Configure for different hardware
            config = TargetConfig(
                device=Device.MIT_PHOTONIC_PROCESSOR,
                precision=Precision.FP16,
                array_size=(32, 32),
                wavelength_nm=1310
            )
            
            # Compile
            compiler = PhotonicCompiler(config)
            compiled_model = compiler.compile_onnx(onnx_path)
            
            # Verify compilation
            assert compiled_model is not None
            assert compiled_model.target_config.device == Device.MIT_PHOTONIC_PROCESSOR
            assert compiled_model.target_config.precision == Precision.FP16
            
            # Test simulation
            dummy_input = np.random.randn(1, 10)
            output = compiled_model.simulate(dummy_input)
            
            assert output.wavelength == 1310
            
        finally:
            os.unlink(onnx_path)
    
    def test_multi_target_compilation(self):
        """Test compiling same model for multiple targets."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model data")
            model_path = f.name
        
        try:
            targets = [
                (Device.LIGHTMATTER_ENVISE, Precision.INT8),
                (Device.MIT_PHOTONIC_PROCESSOR, Precision.FP16),
                (Device.CUSTOM_RESEARCH_CHIP, Precision.FP32)
            ]
            
            compiled_models = []
            
            for device, precision in targets:
                config = TargetConfig(device=device, precision=precision)
                compiler = PhotonicCompiler(config)
                compiled_model = compiler.compile_onnx(model_path)
                compiled_models.append(compiled_model)
            
            # Verify all models compiled successfully
            assert len(compiled_models) == 3
            
            # Verify each has correct configuration
            for i, (device, precision) in enumerate(targets):
                assert compiled_models[i].target_config.device == device
                assert compiled_models[i].target_config.precision == precision
                
        finally:
            os.unlink(model_path)
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_performance_comparison_pipeline(self):
        """Test performance comparison between original and photonic models."""
        # Create model
        model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 20),
            torch.nn.ReLU(), 
            torch.nn.Linear(20, 1)
        )
        
        # Compile for photonic execution
        compiled_model = compile(model, target="lightmatter_envise")
        
        # Test data
        test_input = torch.randn(10, 100)
        
        # Get original output
        with torch.no_grad():
            original_output = model(test_input)
        
        # Get photonic output
        photonic_output = compiled_model.simulate(test_input)
        
        # Compare outputs
        assert original_output.shape == photonic_output.data.shape
        
        # Calculate accuracy (should be reasonably close)
        mse = torch.mean((original_output - photonic_output.data) ** 2)
        assert mse < 1.0  # Reasonable threshold for mock simulation
        
        # Get optimization report
        report = compiled_model.get_optimization_report()
        assert 'estimated_speedup' in report
        assert 'energy_reduction_percent' in report
        assert report['estimated_speedup'] > 1.0


@pytest.mark.integration
class TestSimulationPipeline:
    """Test photonic simulation pipeline."""
    
    def test_noise_model_comparison(self):
        """Test simulation with different noise models."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model")
            model_path = f.name
        
        try:
            # Compile model
            compiled_model = compile(model_path)
            
            # Test with different noise models
            noise_models = ["ideal", "realistic", "pessimistic"]
            simulators = [PhotonicSimulator(noise_model=nm) for nm in noise_models]
            
            # Test input
            test_input = np.random.randn(5, 10)
            
            results = []
            for simulator in simulators:
                result = simulator.run(compiled_model, test_input)
                results.append(result)
            
            # Verify results
            assert len(results) == 3
            for result in results:
                assert hasattr(result, 'data')
                assert result.data.shape == test_input.shape
            
            # Ideal should have least noise, pessimistic most
            # This would require actual noise implementation to test properly
            
        finally:
            os.unlink(model_path)
    
    def test_precision_effects(self):
        """Test effects of different precision modes."""
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            f.write(b"model")
            model_path = f.name
        
        try:
            precisions = [Precision.INT8, Precision.INT16, Precision.FP16, Precision.FP32]
            
            results = []
            for precision in precisions:
                config = TargetConfig(precision=precision)
                compiled_model = compile(model_path, **config.to_dict())
                
                # Simulate
                test_input = np.random.randn(3, 5).astype(np.float32)
                result = compiled_model.simulate(test_input)
                results.append(result)
            
            # All should produce valid results
            assert len(results) == 4
            for result in results:
                assert hasattr(result, 'data')
                assert result.data.shape == (3, 5)
                
        finally:
            os.unlink(model_path)


@pytest.mark.integration 
class TestConfigurationPipeline:
    """Test configuration management pipeline."""
    
    def test_config_file_pipeline(self):
        """Test loading configuration from file and using in compilation."""
        # Create test configuration
        config_data = {
            "target": {
                "device": "mit_photonic_processor",
                "precision": "fp16",
                "array_size": [32, 32],
                "wavelength_nm": 1310
            },
            "compiler": {
                "optimization_level": 3,
                "debug_mode": True
            },
            "simulator": {
                "default_noise_model": "realistic",
                "enable_crosstalk_simulation": True
            }
        }
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            import json
            json.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load configuration
            config = load_config(config_path=config_path)
            
            # Verify configuration loaded correctly
            assert config.target.device == Device.MIT_PHOTONIC_PROCESSOR
            assert config.target.precision == Precision.FP16
            assert config.target.array_size == (32, 32)
            assert config.target.wavelength_nm == 1310
            assert config.compiler.optimization_level == 3
            assert config.compiler.debug_mode == True
            
            # Use configuration in compilation
            with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as model_file:
                model_file.write(b"model")
                model_path = model_file.name
            
            try:
                compiler = PhotonicCompiler(config.target)
                compiled_model = compiler.compile_onnx(model_path)
                
                # Verify model uses correct configuration
                assert compiled_model.target_config.device == Device.MIT_PHOTONIC_PROCESSOR
                assert compiled_model.target_config.precision == Precision.FP16
                
            finally:
                os.unlink(model_path)
                
        finally:
            os.unlink(config_path)


@pytest.mark.integration
class TestDiagnosticsPipeline:
    """Test diagnostics and health checking pipeline."""
    
    def test_system_diagnostics(self):
        """Test running system diagnostics."""
        # Run diagnostics
        results = run_diagnostics(format="json")
        
        # Parse results
        import json
        diagnostic_data = json.loads(results)
        
        # Verify structure
        assert "timestamp" in diagnostic_data
        assert "overall_status" in diagnostic_data
        assert "checks" in diagnostic_data
        
        # Verify checks ran
        checks = diagnostic_data["checks"]
        assert len(checks) > 0
        
        check_names = [check["name"] for check in checks]
        expected_checks = [
            "system_resources",
            "python_environment", 
            "file_permissions",
            "mlir_availability",
            "photonic_dialect"
        ]
        
        for expected in expected_checks:
            assert expected in check_names
    
    def test_config_validation_diagnostics(self):
        """Test configuration validation in diagnostics."""
        # Create problematic configuration
        bad_config = TargetConfig(
            wavelength_nm=2000,  # Out of typical range
            array_size=(2000, 2000),  # Very large
            max_phase_drift=-1.0,  # Invalid
            calibration_interval_ms=100000  # Too long
        )
        
        # Run diagnostics with config
        results = run_diagnostics(config=bad_config, format="json")
        
        # Parse and verify issues detected
        import json
        diagnostic_data = json.loads(results)
        
        # Should detect configuration issues
        config_check = None
        for check in diagnostic_data["checks"]:
            if check["name"] == "target_config_validation":
                config_check = check
                break
        
        assert config_check is not None
        # Should report warnings or errors for bad config
        assert config_check["status"] in ["warning", "critical"]


@pytest.mark.integration
class TestOptimizationPipeline:
    """Test optimization and performance pipeline."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_caching_pipeline(self):
        """Test compilation caching pipeline."""
        from photon_mlir.optimization import get_compilation_cache
        
        # Get cache instance
        cache = get_compilation_cache()
        
        # Create model
        model = torch.nn.Linear(10, 5)
        target_config = TargetConfig().to_dict()
        
        # First compilation (cache miss)
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(model.state_dict(), f.name)
            model_path = f.name
        
        try:
            # Check cache miss
            cached_result = cache.get_compiled_model(model_path, target_config)
            assert cached_result is None
            
            # Compile and cache
            compiled_model = compile(model)
            cache.put_compiled_model(model_path, target_config, compiled_model)
            
            # Check cache hit
            cached_result = cache.get_compiled_model(model_path, target_config)
            assert cached_result is not None
            
            # Verify cache statistics
            stats = cache.get_stats()
            assert 'memory' in stats
            assert 'disk' in stats
            
        finally:
            os.unlink(model_path)
    
    def test_performance_profiling(self):
        """Test performance profiling pipeline."""
        from photon_mlir.optimization import get_profiler, profile_performance
        
        profiler = get_profiler()
        
        # Profile a mock function
        @profile_performance("test_operation")
        def test_function():
            import time
            time.sleep(0.01)  # 10ms
            return "result"
        
        # Run function multiple times
        for _ in range(5):
            result = test_function()
            assert result == "result"
        
        # Check profiling stats
        stats = profiler.get_stats("test_operation")
        assert stats is not None
        assert stats['count'] == 5
        assert stats['mean'] >= 0.01  # At least 10ms
        assert stats['min'] >= 0.01
        assert stats['max'] >= stats['min']


@pytest.mark.integration
@pytest.mark.slow
class TestScalabilityPipeline:
    """Test scalability and concurrent processing."""
    
    @pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")
    def test_concurrent_compilation(self):
        """Test concurrent compilation of multiple models."""
        from photon_mlir.concurrency import get_worker_pool
        
        # Create multiple models
        models = [
            torch.nn.Linear(10, 5),
            torch.nn.Sequential(torch.nn.Linear(20, 10), torch.nn.ReLU()),
            torch.nn.Conv2d(3, 16, 3)
        ]
        
        # Save models to files
        model_paths = []
        for i, model in enumerate(models):
            with tempfile.NamedTemporaryFile(suffix=f'_model_{i}.pt', delete=False) as f:
                torch.save(model.state_dict(), f.name)
                model_paths.append(f.name)
        
        try:
            # Get worker pool
            worker_pool = get_worker_pool()
            
            # Submit compilation tasks
            target_configs = [TargetConfig().to_dict() for _ in models]
            futures = worker_pool.compile_batch_async(model_paths, target_configs)
            
            # Wait for completion
            completed, failed = worker_pool.wait_for_completion(futures, timeout=30.0)
            
            # Verify results
            assert len(failed) == 0, f"Some compilations failed: {failed}"
            assert len(completed) == len(models)
            
            # Check worker pool metrics
            metrics = worker_pool.get_metrics()
            assert 'thread_pool' in metrics
            assert metrics['thread_pool']['completed_tasks'] > 0
            
        finally:
            # Clean up
            for path in model_paths:
                os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])