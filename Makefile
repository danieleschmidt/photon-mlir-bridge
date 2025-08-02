# Makefile for photon-mlir-bridge
# Provides convenient targets for building, testing, and development

# =============================================================================
# Configuration
# =============================================================================

# Build configuration
BUILD_TYPE ?= Debug
BUILD_DIR ?= build
INSTALL_PREFIX ?= /usr/local
NUM_CORES ?= $(shell nproc)

# Python configuration
PYTHON ?= python3
PIP ?= pip3
PYTEST ?= pytest

# Compiler configuration
CC ?= clang-17
CXX ?= clang++-17

# CMake options
CMAKE_OPTS ?= -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
              -DCMAKE_C_COMPILER=$(CC) \
              -DCMAKE_CXX_COMPILER=$(CXX) \
              -DPHOTON_ENABLE_TESTS=ON \
              -DPHOTON_ENABLE_PYTHON=ON

# Test options
TEST_PATTERN ?= ""
PYTEST_OPTS ?= -v --tb=short
CTEST_OPTS ?= --verbose --parallel $(NUM_CORES)

# =============================================================================
# Main targets
# =============================================================================

.PHONY: all build test clean install help

all: build test ## Build and test everything

help: ## Show this help message
	@echo "photon-mlir-bridge Makefile"
	@echo "============================"
	@echo ""
	@echo "Main targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'
	@echo ""
	@echo "Variables:"
	@echo "  BUILD_TYPE      Build type (Debug, Release, RelWithDebInfo) [$(BUILD_TYPE)]"
	@echo "  BUILD_DIR       Build directory [$(BUILD_DIR)]"
	@echo "  NUM_CORES       Number of parallel jobs [$(NUM_CORES)]"
	@echo "  PYTHON          Python interpreter [$(PYTHON)]"
	@echo ""
	@echo "Examples:"
	@echo "  make build                    # Build with current configuration"
	@echo "  make BUILD_TYPE=Release build # Build optimized version"
	@echo "  make test-python              # Run only Python tests"
	@echo "  make test-cpp                 # Run only C++ tests"
	@echo "  make benchmark                # Run performance benchmarks"

# =============================================================================
# Build targets
# =============================================================================

configure: ## Configure CMake build
	@echo "Configuring build..."
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake $(CMAKE_OPTS) ..

build: configure ## Build the project
	@echo "Building project..."
	cd $(BUILD_DIR) && make -j$(NUM_CORES)

build-release: ## Build optimized release version
	@$(MAKE) BUILD_TYPE=Release build

build-debug: ## Build debug version
	@$(MAKE) BUILD_TYPE=Debug build

rebuild: clean build ## Clean and rebuild

# =============================================================================
# Installation targets
# =============================================================================

install: build ## Install the project
	@echo "Installing project..."
	cd $(BUILD_DIR) && make install

install-python: ## Install Python package in development mode
	@echo "Installing Python package..."
	$(PIP) install -e ".[dev,test,docs]"

uninstall: ## Uninstall the project
	@echo "Uninstalling project..."
	cd $(BUILD_DIR) && make uninstall 2>/dev/null || true
	$(PIP) uninstall -y photon-mlir 2>/dev/null || true

# =============================================================================
# Testing targets
# =============================================================================

test: test-cpp test-python ## Run all tests

test-cpp: build ## Run C++ tests
	@echo "Running C++ tests..."
	cd $(BUILD_DIR) && ctest $(CTEST_OPTS) $(if $(TEST_PATTERN),-R "$(TEST_PATTERN)")

test-python: install-python ## Run Python tests
	@echo "Running Python tests..."
	$(PYTEST) $(PYTEST_OPTS) tests/unit/python tests/integration $(if $(TEST_PATTERN),-k "$(TEST_PATTERN)")

test-unit: ## Run unit tests only
	@echo "Running unit tests..."
	cd $(BUILD_DIR) && ctest $(CTEST_OPTS) -R "unit.*"
	$(PYTEST) $(PYTEST_OPTS) tests/unit/python $(if $(TEST_PATTERN),-k "$(TEST_PATTERN)")

test-integration: ## Run integration tests only
	@echo "Running integration tests..."
	cd $(BUILD_DIR) && ctest $(CTEST_OPTS) -R "integration.*"
	$(PYTEST) $(PYTEST_OPTS) tests/integration $(if $(TEST_PATTERN),-k "$(TEST_PATTERN)")

test-e2e: ## Run end-to-end tests
	@echo "Running end-to-end tests..."
	$(PYTEST) $(PYTEST_OPTS) tests/integration/end_to_end $(if $(TEST_PATTERN),-k "$(TEST_PATTERN)")

test-hardware: ## Run hardware tests (requires actual hardware)
	@echo "Running hardware tests..."
	$(PYTEST) $(PYTEST_OPTS) --run-hardware tests/ $(if $(TEST_PATTERN),-k "$(TEST_PATTERN)")

test-slow: ## Run slow tests
	@echo "Running slow tests..."
	$(PYTEST) $(PYTEST_OPTS) --run-slow tests/ $(if $(TEST_PATTERN),-k "$(TEST_PATTERN)")

test-coverage: ## Run tests with coverage
	@echo "Running tests with coverage..."
	cd $(BUILD_DIR) && cmake -DPHOTON_ENABLE_COVERAGE=ON .. && make -j$(NUM_CORES)
	cd $(BUILD_DIR) && ctest $(CTEST_OPTS)
	cd $(BUILD_DIR) && make coverage
	$(PYTEST) --cov=photon_mlir --cov-report=html --cov-report=term tests/

# =============================================================================
# Benchmarking targets
# =============================================================================

benchmark: build-release ## Run performance benchmarks
	@echo "Running benchmarks..."
	cd $(BUILD_DIR) && cmake -DPHOTON_ENABLE_BENCHMARKS=ON .. && make -j$(NUM_CORES)
	cd $(BUILD_DIR) && ./tests/benchmarks/photon_benchmarks --benchmark_out=benchmark_results.json
	$(PYTEST) -m benchmark tests/

benchmark-compilation: build-release ## Run compilation benchmarks only
	@echo "Running compilation benchmarks..."
	cd $(BUILD_DIR) && ./tests/benchmarks/compilation_benchmark

benchmark-runtime: build-release ## Run runtime benchmarks only
	@echo "Running runtime benchmarks..."
	cd $(BUILD_DIR) && ./tests/benchmarks/runtime_benchmark

# =============================================================================
# Code quality targets
# =============================================================================

format: ## Format code using clang-format and black
	@echo "Formatting code..."
	find src include tests -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | xargs clang-format-17 -i
	find python tests -name "*.py" | xargs black --line-length=88
	find python tests -name "*.py" | xargs isort --profile black

lint: ## Run linting checks
	@echo "Running linting checks..."
	find src include -name "*.cpp" -o -name "*.h" -o -name "*.hpp" | head -20 | xargs clang-tidy-17 --config-file=.clang-tidy
	flake8 python/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	mypy python/ --ignore-missing-imports

check: format lint ## Run all code quality checks

# =============================================================================
# Documentation targets
# =============================================================================

docs: ## Build documentation
	@echo "Building documentation..."
	cd docs && make html

docs-serve: docs ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	cd docs/_build/html && $(PYTHON) -m http.server 8000

docs-clean: ## Clean documentation build
	cd docs && make clean

# =============================================================================
# Development targets
# =============================================================================

dev-setup: ## Set up development environment
	@echo "Setting up development environment..."
	$(PIP) install -e ".[dev,test,docs]"
	pre-commit install --install-hooks
	@echo "Development environment ready!"

dev-clean: ## Clean development environment
	pre-commit uninstall
	$(PIP) uninstall -y photon-mlir

generate-test-data: ## Generate test data and fixtures
	@echo "Generating test data..."
	cd tests/fixtures/data && $(PYTHON) test_models.py

# =============================================================================
# Debugging targets
# =============================================================================

debug-build: ## Build with debugging symbols and run debugger
	@$(MAKE) BUILD_TYPE=Debug build
	@echo "Debug build ready. Run 'gdb $(BUILD_DIR)/photon_compiler' to debug."

debug-test: ## Run tests under debugger
	@$(MAKE) BUILD_TYPE=Debug build
	cd $(BUILD_DIR) && gdb --args ./tests/unit/test_photonic_device

valgrind: build-debug ## Run tests with Valgrind
	@echo "Running tests with Valgrind..."
	cd $(BUILD_DIR) && valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all ctest

sanitize: ## Build with sanitizers
	@echo "Building with sanitizers..."
	cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Debug \
	  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" \
	  -DCMAKE_C_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer" ..
	cd $(BUILD_DIR) && make -j$(NUM_CORES)
	cd $(BUILD_DIR) && ctest

# =============================================================================
# Docker targets
# =============================================================================

docker-build: ## Build Docker development image
	@echo "Building Docker image..."
	docker build -t photon-mlir-dev .

docker-run: ## Run development container
	@echo "Running development container..."
	docker run -it --rm -v $(PWD):/workspace photon-mlir-dev

docker-test: ## Run tests in Docker container
	@echo "Running tests in Docker..."
	docker run --rm -v $(PWD):/workspace photon-mlir-dev make test

# =============================================================================
# Packaging targets
# =============================================================================

package: build-release ## Create distribution packages
	@echo "Creating packages..."
	cd $(BUILD_DIR) && cpack
	$(PYTHON) setup.py sdist bdist_wheel

package-python: ## Create Python package only
	@echo "Creating Python package..."
	$(PYTHON) setup.py sdist bdist_wheel

upload-pypi: package-python ## Upload to PyPI (requires credentials)
	@echo "Uploading to PyPI..."
	twine upload dist/*

# =============================================================================
# Cleanup targets
# =============================================================================

clean: ## Clean build artifacts
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.so" -delete

clean-all: clean docs-clean dev-clean ## Clean everything

# =============================================================================
# Utility targets
# =============================================================================

env: ## Show build environment
	@echo "Build Environment:"
	@echo "=================="
	@echo "BUILD_TYPE      = $(BUILD_TYPE)"
	@echo "BUILD_DIR       = $(BUILD_DIR)"
	@echo "CC              = $(CC)"
	@echo "CXX             = $(CXX)"
	@echo "PYTHON          = $(PYTHON)"
	@echo "NUM_CORES       = $(NUM_CORES)"
	@echo "CMAKE_OPTS      = $(CMAKE_OPTS)"

size: build ## Show binary sizes
	@echo "Binary sizes:"
	@find $(BUILD_DIR) -name "*.so" -o -name "photon_*" -type f -executable | xargs ls -lh

deps: ## Show dependencies
	@echo "System dependencies:"
	@which cmake llvm-config-17 clang-17 clang++-17 python3 pip3 || echo "Missing dependencies"

check-deps: ## Check for required dependencies
	@echo "Checking dependencies..."
	@command -v cmake >/dev/null 2>&1 || { echo "cmake is required but not installed"; exit 1; }
	@command -v llvm-config-17 >/dev/null 2>&1 || { echo "LLVM 17 is required but not installed"; exit 1; }
	@command -v clang-17 >/dev/null 2>&1 || { echo "Clang 17 is required but not installed"; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "Python 3 is required but not installed"; exit 1; }
	@echo "All dependencies found!"

# =============================================================================
# CI/CD targets
# =============================================================================

ci-test: check-deps install-python ## Run CI test suite
	@echo "Running CI test suite..."
	@$(MAKE) build
	@$(MAKE) test-cpp CTEST_OPTS="--output-on-failure"
	@$(MAKE) test-python PYTEST_OPTS="-v --tb=short --maxfail=5"
	@$(MAKE) lint

ci-benchmark: ## Run CI benchmark suite
	@echo "Running CI benchmarks..."
	@$(MAKE) build-release
	@$(MAKE) benchmark

# =============================================================================
# Special targets
# =============================================================================

.DEFAULT_GOAL := help

# Prevent make from deleting intermediate files
.PRECIOUS: $(BUILD_DIR)/Makefile

# Mark phony targets
.PHONY: configure build build-release build-debug rebuild install install-python uninstall \
        test test-cpp test-python test-unit test-integration test-e2e test-hardware test-slow test-coverage \
        benchmark benchmark-compilation benchmark-runtime \
        format lint check \
        docs docs-serve docs-clean \
        dev-setup dev-clean generate-test-data \
        debug-build debug-test valgrind sanitize \
        docker-build docker-run docker-test \
        package package-python upload-pypi \
        clean clean-all \
        env size deps check-deps \
        ci-test ci-benchmark