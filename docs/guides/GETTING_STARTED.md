# Getting Started with Photon-MLIR

This guide will help you get started with the Photon-MLIR compiler for silicon photonic neural network accelerators.

## Installation

### Prerequisites

- Python 3.9 or higher
- PyTorch 2.0 or higher
- MLIR/LLVM 17+ (for C++ components)
- CMake 3.20+ (for building C++ components)

### Quick Installation

```bash
# Install from PyPI (when available)
pip install photon-mlir

# Or install from source
git clone https://github.com/yourusername/photon-mlir-bridge.git
cd photon-mlir-bridge
pip install -e .
```

### Build from Source

```bash
# Clone the repository
git clone --recursive https://github.com/yourusername/photon-mlir-bridge.git
cd photon-mlir-bridge

# Install Python dependencies
pip install -r requirements.txt

# Build C++ components
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package
cd ..
pip install -e .
```

## Your First Photonic Compilation

### Example 1: Simple Linear Model

```python
import torch
import photon_mlir as pm

# Create a simple neural network
model = torch.nn.Sequential(
    torch.nn.Linear(784, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 10)
)

# Compile for photonic hardware
photonic_model = pm.compile(
    model,
    target="lightmatter_envise",
    optimize_for="latency"
)

# Test the compiled model
dummy_input = torch.randn(1, 784)
output = photonic_model.simulate(dummy_input)

print(f"Output shape: {output.data.shape}")
print(f"Optical wavelength: {output.wavelength}nm")
print(f"Optical power: {output.power_mw}mW")
```

### Example 2: Convolutional Neural Network

```python
import torch
import photon_mlir as pm

# Create a CNN
model = torch.nn.Sequential(
    torch.nn.Conv2d(3, 32, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.MaxPool2d(2),
    torch.nn.Conv2d(32, 64, 3, padding=1),
    torch.nn.ReLU(),
    torch.nn.AdaptiveAvgPool2d(1),
    torch.nn.Flatten(),
    torch.nn.Linear(64, 10)
)

# Configure target hardware
config = pm.TargetConfig(
    device=pm.Device.LIGHTMATTER_ENVISE,
    precision=pm.Precision.INT8,
    array_size=(64, 64),
    wavelength_nm=1550
)

# Compile with specific configuration
photonic_model = pm.compile(model, **config.to_dict())

# Test with image data
image_input = torch.randn(1, 3, 32, 32)
output = photonic_model.simulate(image_input)

# Get optimization report
report = photonic_model.get_optimization_report()
print(report)
```

## Understanding Photonic Compilation

### Compilation Process

1. **Model Loading**: Load PyTorch or ONNX models
2. **IR Translation**: Convert to MLIR photonic dialect
3. **Optimization**: Apply photonic-specific optimizations
4. **Code Generation**: Generate photonic assembly
5. **Hardware Mapping**: Map to specific photonic hardware

### Supported Operations

The compiler supports common neural network operations:

- **Linear/Dense layers**: Matrix multiplication using photonic meshes
- **Convolutions**: Converted to matrix operations via im2col
- **Activations**: ReLU, Sigmoid, Tanh (with photonic approximations)
- **Pooling**: Average and max pooling operations
- **Normalization**: Batch normalization and layer normalization

### Hardware Targets

Currently supported photonic hardware:

- **Lightmatter Envise**: 64×64 array, 1 GHz, INT8/FP16
- **MIT Photonic Processor**: 32×32 array, 500 MHz, various precisions
- **Custom Research Chips**: Configurable array sizes and parameters

## Configuration Options

### Target Configuration

```python
config = pm.TargetConfig(
    device=pm.Device.LIGHTMATTER_ENVISE,     # Target hardware
    precision=pm.Precision.INT8,             # Computation precision
    array_size=(64, 64),                     # Photonic array dimensions
    wavelength_nm=1550,                      # Operating wavelength
    max_phase_drift=0.1,                     # Maximum phase drift (radians)
    calibration_interval_ms=100              # Thermal calibration interval
)
```

### Compiler Options

```python
compiler = pm.PhotonicCompiler(target_config=config)

# Advanced options
compiled_model = compiler.compile_pytorch(
    model,
    optimization_level=3,           # 0-3, higher = more optimizations
    enable_thermal_compensation=True, # Thermal drift compensation
    enable_debug=False,             # Debug mode
    cache_compiled_models=True      # Enable compilation caching
)
```

## Simulation and Validation

### Photonic Simulation

```python
# Create simulator with noise modeling
simulator = pm.PhotonicSimulator(
    noise_model="realistic",  # "ideal", "realistic", "pessimistic"
    precision="8bit",
    crosstalk_db=-30.0
)

# Run simulation
result = simulator.run(photonic_model, input_data)

# Compare with ideal computation
comparison = simulator.compare_with_ideal(ideal_output, result)
print(f"MSE: {comparison['mse']:.6f}")
print(f"SNR: {comparison['snr_db']:.1f} dB")
```

### Hardware-in-the-Loop Testing

```python
# Connect to real photonic hardware (when available)
try:
    device = pm.PhotonicDevice.connect("lightmatter://192.168.1.100")
    
    # Upload compiled model
    device.upload(photonic_model)
    
    # Run on hardware
    hardware_output = device.infer(input_data)
    
    print("Successfully ran on hardware!")
    
except pm.HardwareError as e:
    print(f"Hardware not available: {e}")
    # Fall back to simulation
    output = photonic_model.simulate(input_data)
```

## Performance Analysis

### Optimization Reports

```python
# Get detailed optimization report
report = photonic_model.get_optimization_report()
print(f"Original FLOPs: {report['original_flops']:,}")
print(f"Photonic MACs: {report['photonic_macs']:,}")
print(f"Estimated speedup: {report['estimated_speedup']:.1f}x")
print(f"Energy reduction: {report['energy_reduction_percent']:.1f}%")
```

### Visualization

```python
# Visualize mesh utilization
visualizer = pm.MeshVisualizer()
visualizer.plot_temporal_utilization(photonic_model, test_data)
visualizer.export_3d("mesh_usage.html", show_heat_map=True)

# Optimization dashboard
dashboard = pm.OptimizationDashboard()
dashboard.track_compilation(model, ['phase_shifts', 'optical_power'])
dashboard.serve(port=8501)  # View at http://localhost:8501
```

## Command Line Interface

### Basic Usage

```bash
# Compile ONNX model
photon-compile model.onnx -o model.pasm --target lightmatter_envise

# Run simulation
photon-simulate model.pasm --input data.npy --output results.npy

# Profile performance
photon-profile model.pasm --input-shape 1,3,224,224 --runs 100

# Debug compilation
photon-debug model.pasm --visualize mesh,thermal,phase
```

### Advanced Options

```bash
# Custom configuration
photon-compile model.onnx \
    --target lightmatter_envise \
    --precision int8 \
    --array-size 64,64 \
    --wavelength 1550 \
    --optimization-level 3 \
    --show-report

# Batch processing
photon-compile models/*.onnx \
    --output-dir compiled/ \
    --target mit_photonic \
    --parallel 4
```

## Best Practices

### Model Preparation

1. **Quantization**: Pre-quantize models for INT8 targets
2. **Optimization**: Use PyTorch optimizations before compilation
3. **Validation**: Always validate outputs against original model

```python
# Example: Proper model preparation
model = torch.nn.Sequential(...)

# Quantize for INT8 target
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Compile quantized model
photonic_model = pm.compile(
    quantized_model,
    target="lightmatter_envise",
    precision=pm.Precision.INT8
)
```

### Performance Optimization

1. **Batch Processing**: Use larger batch sizes when possible
2. **Caching**: Enable model compilation caching
3. **Profiling**: Profile performance and optimize bottlenecks

```python
# Enable performance optimizations
config = pm.PhotonicConfig()
config.compiler.cache_compiled_models = True
config.compiler.optimization_level = 3

pm.load_config(config_dict=config.to_dict())
```

### Error Handling

```python
try:
    photonic_model = pm.compile(model, target="lightmatter_envise")
except pm.CompilationError as e:
    print(f"Compilation failed: {e}")
    # Handle compilation errors
except pm.HardwareError as e:
    print(f"Hardware issue: {e}")
    # Fall back to simulation
```

## Next Steps

- Read the [Architecture Guide](../ARCHITECTURE.md) for detailed system design
- Explore [Advanced Examples](../examples/) for complex use cases
- Check the [API Reference](../api/) for detailed documentation
- Join the [Community](https://github.com/yourusername/photon-mlir-bridge/discussions) for support

## Troubleshooting

### Common Issues

**Q: Compilation is slow**
A: Enable caching and use appropriate optimization levels. Consider using parallel compilation for multiple models.

**Q: Simulation results don't match original model**
A: Check precision settings and noise model configuration. Use "ideal" noise model for exact comparison.

**Q: Hardware connection fails**
A: Ensure hardware is powered on and network accessible. Check firewall settings and device IP address.

**Q: Out of memory during compilation**
A: Reduce model size, lower optimization level, or increase system memory. Consider model partitioning for very large models.

For more troubleshooting, see the [Troubleshooting Guide](TROUBLESHOOTING.md).