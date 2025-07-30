# Test Framework Structure

This directory contains the test suite for photon-mlir-bridge, organized by test type and component.

## Directory Structure

```
tests/
├── unit/                   # Unit tests
│   ├── core/              # Core functionality tests
│   ├── compiler/          # Compiler tests
│   ├── dialects/          # MLIR dialect tests
│   ├── runtime/           # Runtime tests
│   └── python/            # Python binding tests
├── integration/           # Integration tests
│   ├── end_to_end/       # Full compilation pipeline tests
│   ├── hardware/         # Hardware interface tests
│   └── frameworks/       # Framework integration tests
├── benchmarks/           # Performance benchmarks
│   ├── compilation/      # Compilation time benchmarks
│   ├── runtime/          # Runtime performance benchmarks
│   └── memory/           # Memory usage benchmarks
├── fixtures/            # Test data and fixtures
│   ├── models/          # Test models (ONNX, PyTorch)
│   ├── ir/              # MLIR test files
│   └── data/            # Input/output test data
└── tools/               # Test utilities and helpers
```

## Running Tests

### C++ Tests (using CTest)

```bash
# Configure with testing enabled
mkdir build && cd build
cmake .. -DPHOTON_ENABLE_TESTS=ON -DCMAKE_BUILD_TYPE=Debug

# Build tests
make -j$(nproc)

# Run all tests
ctest --verbose

# Run specific test suite
ctest -R "unit.*" --verbose

# Run with coverage
cmake .. -DPHOTON_ENABLE_COVERAGE=ON
make -j$(nproc)
ctest
make coverage
```

### Python Tests (using pytest)

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all Python tests
pytest

# Run with coverage
pytest --cov=photon_mlir --cov-report=html --cov-report=term

# Run specific test file
pytest tests/python/test_compiler.py

# Run with verbose output
pytest -v -s
```

### Performance Benchmarks

```bash
# Build with benchmarks
cmake .. -DPHOTON_ENABLE_BENCHMARKS=ON -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run benchmarks
./benchmarks/photon_benchmarks

# Generate performance report
./benchmarks/photon_benchmarks --benchmark_out=benchmark_results.json
```

## Test Categories

### Unit Tests

**Purpose**: Test individual components in isolation
**Location**: `tests/unit/`
**Framework**: GoogleTest (C++), pytest (Python)

Examples:
- Individual MLIR passes
- Utility functions
- Data structures
- Algorithm implementations

### Integration Tests

**Purpose**: Test component interactions and workflows
**Location**: `tests/integration/`
**Framework**: GoogleTest (C++), pytest (Python)

Examples:
- Complete compilation pipelines
- Hardware driver integration
- Framework compatibility
- Multi-component interactions

### End-to-End Tests

**Purpose**: Test complete user workflows
**Location**: `tests/integration/end_to_end/`
**Framework**: Custom test harness

Examples:
- Model compilation from ONNX to photonic assembly
- Hardware deployment and execution
- Performance validation against targets

### Performance Benchmarks

**Purpose**: Measure and track performance metrics
**Location**: `tests/benchmarks/`
**Framework**: Google Benchmark (C++), pytest-benchmark (Python)

Examples:
- Compilation time for different model sizes
- Runtime performance on hardware
- Memory usage profiling
- Energy efficiency measurements

## Test Configuration

### CMake Test Configuration

```cmake
# In CMakeLists.txt
option(PHOTON_ENABLE_TESTS "Enable testing" ON)
option(PHOTON_ENABLE_BENCHMARKS "Enable benchmarks" OFF)
option(PHOTON_ENABLE_COVERAGE "Enable coverage reporting" OFF)

if(PHOTON_ENABLE_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    
    # Add test subdirectories
    add_subdirectory(tests/unit)
    add_subdirectory(tests/integration)
    
    if(PHOTON_ENABLE_BENCHMARKS)
        find_package(benchmark REQUIRED)
        add_subdirectory(tests/benchmarks)
    endif()
endif()
```

### Python Test Configuration

```toml
# In pyproject.toml
[tool.pytest.ini_options]
testpaths = ["tests/python"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "--strict-markers",
    "--strict-config",
    "--cov=photon_mlir",
    "--cov-report=term-missing",
    "--cov-report=html",
    "--cov-report=xml",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "hardware: marks tests that require hardware",
    "integration: marks integration tests",
    "unit: marks unit tests"
]
```

## Writing Tests

### C++ Test Example

```cpp
// tests/unit/core/test_photonic_device.cpp
#include <gtest/gtest.h>
#include "photon/core/photonic_device.h"

namespace photon {
namespace test {

class PhotonicDeviceTest : public ::testing::Test {
protected:
    void SetUp() override {
        config_.wavelength = 1550;
        config_.array_size = {64, 64};
        device_ = std::make_unique<PhotonicDevice>(config_);
    }

    DeviceConfig config_;
    std::unique_ptr<PhotonicDevice> device_;
};

TEST_F(PhotonicDeviceTest, SetPhaseShiftUpdatesCorrectly) {
    const double phase = M_PI_2;
    const size_t index = 5;
    
    device_->setPhaseShift(index, phase);
    
    EXPECT_DOUBLE_EQ(device_->getPhaseShift(index), phase);
}

TEST_F(PhotonicDeviceTest, InvalidIndexThrowsException) {
    const size_t invalid_index = 10000;
    
    EXPECT_THROW(
        device_->setPhaseShift(invalid_index, 0.0),
        std::out_of_range
    );
}

}  // namespace test
}  // namespace photon
```

### Python Test Example

```python
# tests/python/test_compiler.py
import pytest
import torch
import photon_mlir as pm

class TestCompiler:
    def test_compile_linear_model(self):
        """Test compilation of a simple linear model."""
        model = torch.nn.Linear(10, 5)
        
        compiled = pm.compile(
            model,
            target="lightmatter_envise",
            optimize_for="latency"
        )
        
        assert compiled is not None
        assert compiled.target == "lightmatter_envise"
    
    @pytest.mark.slow
    def test_compile_large_model(self):
        """Test compilation of a large model (marked as slow).""" 
        model = torch.nn.Sequential(*[
            torch.nn.Linear(1000, 1000) for _ in range(10)
        ])
        
        compiled = pm.compile(model, target="simulation")
        assert compiled is not None
    
    @pytest.mark.hardware
    def test_hardware_execution(self):
        """Test actual hardware execution (requires hardware)."""
        pytest.skip("Hardware not available in CI")
        
        # Hardware-specific test code here
```

### Benchmark Example

```cpp
// tests/benchmarks/compilation_benchmark.cpp
#include <benchmark/benchmark.h>
#include "photon/compiler.h"

static void BM_CompileLinearModel(benchmark::State& state) {
    const int input_size = state.range(0);
    const int output_size = state.range(1);
    
    // Setup model
    auto model = createLinearModel(input_size, output_size);
    photon::CompilerConfig config{
        .target = photon::Target::SIMULATION,
        .optimization_level = 2
    };
    
    for (auto _ : state) {
        auto compiled = photon::compile(model, config);
        benchmark::DoNotOptimize(compiled);
    }
    
    state.SetComplexityN(input_size * output_size);
}

BENCHMARK(BM_CompileLinearModel)
    ->Args({100, 10})
    ->Args({1000, 100})
    ->Args({10000, 1000})
    ->Complexity();

BENCHMARK_MAIN();
```

## Continuous Integration

### Test Automation

Tests are automatically run on:
- Every push to main/develop branches
- Every pull request
- Nightly builds for comprehensive testing

### Test Matrix

- **Compilers**: GCC 11+, Clang 13+
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Operating Systems**: Ubuntu 22.04, macOS 12+
- **Hardware**: Simulation, available photonic hardware

### Coverage Requirements

- **Minimum Coverage**: 80% line coverage
- **Target Coverage**: 90% line coverage for core components
- **Critical Components**: 95+ % coverage required

## Test Data Management

### Fixtures

Test fixtures are stored in `tests/fixtures/` and include:
- Sample neural network models
- Input/output data pairs
- MLIR test files
- Hardware configuration files

### Data Generation

```python
# tests/tools/generate_test_data.py
def generate_test_models():
    """Generate test models for various scenarios."""
    models = {
        'simple_linear': torch.nn.Linear(10, 5),
        'conv_net': create_conv_net(),
        'transformer': create_transformer_block()
    }
    
    for name, model in models.items():
        torch.save(model, f'fixtures/models/{name}.pth')
```

## Hardware Testing

### Simulation Testing

All tests can run in simulation mode without hardware:

```cpp
photon::CompilerConfig config{
    .target = photon::Target::SIMULATION,
    .enable_thermal_simulation = true,
    .enable_noise_simulation = true
};
```

### Hardware-in-the-Loop Testing

For available hardware:

```python
@pytest.mark.hardware
def test_with_real_hardware():
    if not hardware_available():
        pytest.skip("Hardware not available")
    
    # Test with real hardware
    device = pm.connect_hardware("lightmatter://192.168.1.100")
    # ... test code
```

## Debugging Tests

### Debugging Failed Tests

```bash
# Run specific failing test with verbose output
ctest -R "FailingTestName" --verbose

# Run with debugger
gdb --args ./test_executable

# Python debugging
pytest -s --pdb tests/python/test_failing.py
```

### Test Logging

Tests can be configured with different logging levels:

```cpp
// Set log level for tests
photon::setLogLevel(photon::LogLevel::DEBUG);
```

## Contributing Tests

When contributing new code:

1. **Add unit tests** for new functions/classes
2. **Add integration tests** for new features
3. **Update benchmarks** for performance-critical changes
4. **Maintain coverage** above minimum thresholds
5. **Test on multiple platforms** when possible

For detailed testing guidelines, see the [Contributing Guide](../CONTRIBUTING.md).