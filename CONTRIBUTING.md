# Contributing to Photon MLIR Bridge

Thank you for your interest in contributing to the Photon MLIR Bridge project! This document provides guidelines for contributing to this MLIR-based compiler for silicon photonic neural network accelerators.

## 🚀 Getting Started

### Prerequisites

- C++20 compatible compiler (GCC 11+, Clang 13+, MSVC 2022+)
- CMake 3.20+
- LLVM/MLIR 17.0+
- Python 3.9+
- Git

### Development Setup

1. **Clone the repository**:
   ```bash
   git clone --recursive https://github.com/yourusername/photon-mlir-bridge.git
   cd photon-mlir-bridge
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

3. **Setup pre-commit hooks**:
   ```bash
   pre-commit install
   ```

4. **Build the project**:
   ```bash
   mkdir build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Debug
   make -j$(nproc)
   ```

5. **Run tests**:
   ```bash
   ctest --verbose
   ```

## 🎯 How to Contribute

### Reporting Issues

Before creating an issue, please:

1. Check existing issues to avoid duplicates
2. Use our issue templates for bugs and feature requests
3. Provide clear, reproducible examples
4. Include relevant system information

### Submitting Changes

1. **Fork** the repository
2. **Create** a new branch: `git checkout -b feature/your-feature-name`
3. **Make** your changes following our coding standards
4. **Add** tests for new functionality
5. **Run** the test suite: `ctest`
6. **Commit** your changes with clear messages
7. **Push** to your fork: `git push origin feature/your-feature-name`
8. **Submit** a pull request

## 📝 Coding Standards

### C++ Guidelines

- Follow **C++20 standards**
- Use **2-space indentation**
- **RAII** for resource management
- **Smart pointers** instead of raw pointers
- **Const-correctness** throughout
- **Include guards** for headers

```cpp
// Example C++ style
namespace photon {
namespace core {

class PhotonicDevice {
 public:
  explicit PhotonicDevice(const DeviceConfig& config);
  ~PhotonicDevice() = default;

  // Non-copyable, movable
  PhotonicDevice(const PhotonicDevice&) = delete;
  PhotonicDevice& operator=(const PhotonicDevice&) = delete;
  PhotonicDevice(PhotonicDevice&&) = default;
  PhotonicDevice& operator=(PhotonicDevice&&) = default;

  [[nodiscard]] auto getWavelength() const noexcept -> double;
  void setPhaseShift(size_t index, double phase);

 private:
  std::unique_ptr<DeviceImpl> impl_;
};

}  // namespace core
}  // namespace photon
```

### Python Guidelines

- Follow **PEP 8**
- Use **type hints**
- **4-space indentation**
- **Black** for formatting
- **isort** for import organization

```python
from typing import List, Optional, Union
import numpy as np
from photon_mlir.core import PhotonicDevice


def compile_model(
    model: torch.nn.Module,
    target: str,
    optimize_for: str = "latency"
) -> PhotonicModel:
    """Compile PyTorch model for photonic execution.
    
    Args:
        model: PyTorch model to compile
        target: Target photonic device
        optimize_for: Optimization strategy
        
    Returns:
        Compiled photonic model
        
    Raises:
        CompilationError: If compilation fails
    """
    # Implementation here
    pass
```

### MLIR Guidelines

- Use **2-space indentation**
- **Consistent naming** for operations and attributes
- **Comprehensive documentation** for dialects

```mlir
// Example MLIR operation definition
def Photonic_MatMulOp : Photonic_Op<"matmul",
    [Pure, SameOperandsAndResultType]> {
  let summary = "Photonic matrix multiplication operation";
  let description = [{
    Performs matrix multiplication using photonic mesh arrays.
    Supports various decomposition strategies for large matrices.
  }];

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    OptionalAttr<I32Attr>:$wavelength,
    OptionalAttr<StrAttr>:$mesh_config
  );

  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}
```

## 🧪 Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual components
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Benchmark critical paths

### Test Organization

```
test/
├── unit/           # Unit tests
│   ├── core/       # Core functionality tests
│   ├── compiler/   # Compiler tests
│   └── runtime/    # Runtime tests
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── python/         # Python binding tests
```

### Writing Tests

- Use **GoogleTest** for C++ tests
- Use **pytest** for Python tests
- **Descriptive test names**
- **Clear assertions**
- **Proper test isolation**

```cpp
// C++ test example
TEST(PhotonicDeviceTest, SetPhaseShiftUpdatesCorrectly) {
  PhotonicDevice device(DeviceConfig{.wavelength = 1550});
  
  device.setPhaseShift(0, M_PI_2);
  
  EXPECT_DOUBLE_EQ(device.getPhaseShift(0), M_PI_2);
}
```

```python
# Python test example
def test_compile_simple_model():
    """Test compilation of a simple linear model."""
    model = torch.nn.Linear(10, 5)
    
    compiled = photon_mlir.compile(model, target="lightmatter_envise")
    
    assert compiled is not None
    assert compiled.target == "lightmatter_envise"
```

## 🔧 Development Tools

### Pre-commit Hooks

We use pre-commit hooks to ensure code quality:

- **clang-format**: C++ code formatting
- **black**: Python code formatting
- **isort**: Python import sorting
- **flake8**: Python linting
- **mypy**: Python type checking

### Code Coverage

Maintain high code coverage:

- **Target**: >90% line coverage
- **Tools**: gcov for C++, pytest-cov for Python
- **Reports**: Generated in `coverage/` directory

### Static Analysis

Regular static analysis with:

- **clang-tidy**: C++ static analysis
- **cppcheck**: Additional C++ checks
- **mypy**: Python type checking
- **bandit**: Python security analysis

## 🎨 Documentation

### Code Documentation

- **Doxygen** comments for C++ APIs
- **Sphinx** docstrings for Python APIs
- **Inline comments** for complex logic
- **README files** for major components

### User Documentation

Located in `docs/`:

- **guides/**: User guides and tutorials
- **api/**: API reference documentation
- **examples/**: Code examples and samples

## 🏷️ Git Workflow

### Commit Messages

Follow conventional commit format:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style changes
- **refactor**: Code refactoring
- **test**: Test additions/modifications
- **chore**: Maintenance tasks

Example:
```
feat(compiler): add thermal compensation pass

Implement MLIR pass for automatic thermal drift compensation
in photonic devices. Includes runtime calibration insertion
and phase adjustment optimization.

Closes #123
```

### Branch Naming

- **feature/**: New features (`feature/thermal-compensation`)
- **fix/**: Bug fixes (`fix/phase-drift-calculation`)
- **docs/**: Documentation (`docs/api-reference`)
- **refactor/**: Refactoring (`refactor/device-abstraction`)

## 🔍 Review Process

### Pull Request Guidelines

1. **Clear description** of changes
2. **Link to relevant issues**
3. **Screenshots** for UI changes
4. **Performance impact** notes
5. **Breaking change** warnings

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass and coverage maintained
- [ ] Documentation updated
- [ ] No breaking changes (or properly documented)
- [ ] Performance implications considered

## 🌟 Recognition

Contributors are recognized in:

- **AUTHORS.md**: All contributors
- **Release notes**: Major contributions
- **Hall of Fame**: Outstanding contributors

## ❓ Getting Help

- **Discussions**: GitHub Discussions for questions
- **Issues**: Bug reports and feature requests
- **Email**: maintainers@photon-mlir.dev
- **Chat**: Discord server (link in README)

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). 
Please read it before participating.

---

Thank you for contributing to the future of photonic computing! 🚀