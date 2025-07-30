# Multi-stage Dockerfile for photon-mlir-bridge development and deployment

# Base image with LLVM/MLIR dependencies
FROM ubuntu:22.04 as base

# Avoid interactive prompts during build
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    curl \
    # LLVM/MLIR dependencies
    llvm-17-dev \
    mlir-17-tools \
    libmlir-17-dev \
    clang-17 \
    clang-tools-17 \
    # Python and development tools
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3-venv \
    # Additional development tools
    gdb \
    valgrind \
    doxygen \
    graphviz \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Create a non-root user for development
RUN useradd -m -s /bin/bash -u 1000 developer && \
    usermod -aG sudo developer && \
    echo "developer ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Development stage
FROM base as development

USER developer
WORKDIR /home/developer

# Set up development environment
ENV PATH="/home/developer/.local/bin:$PATH"
ENV PYTHONPATH="/workspace/python:$PYTHONPATH"
ENV CC=clang-17
ENV CXX=clang++-17

# Install development Python packages
RUN python3.11 -m pip install --user \
    pre-commit \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-benchmark \
    bandit \
    safety

# Set up workspace
WORKDIR /workspace

# Copy source code (in practice, this would be mounted as a volume)
COPY --chown=developer:developer . .

# Install Python dependencies
RUN python3.11 -m pip install --user -e ".[dev,test,docs]"

# Set up pre-commit hooks
RUN pre-commit install --install-hooks

# Build stage
FROM development as build

# Create build directory
RUN mkdir -p /workspace/build

WORKDIR /workspace/build

# Configure CMake with optimizations
RUN cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DPHOTON_ENABLE_TESTS=ON \
    -DPHOTON_ENABLE_PYTHON=ON \
    -DPHOTON_ENABLE_BENCHMARKS=ON

# Build the project
RUN ninja -j$(nproc)

# Run tests to ensure build is working
RUN ctest --output-on-failure

# Production runtime stage
FROM ubuntu:22.04 as runtime

ARG DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y \
    # Runtime libraries
    libmlir-17 \
    llvm-17-runtime \
    python3.11 \
    python3.11-distutils \
    # Networking and security
    ca-certificates \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create runtime user
RUN useradd -m -s /bin/bash -u 1000 photon

# Copy built artifacts from build stage
COPY --from=build --chown=photon:photon /workspace/build/lib /usr/local/lib/photon
COPY --from=build --chown=photon:photon /workspace/build/bin /usr/local/bin
COPY --from=build --chown=photon:photon /workspace/python /usr/local/lib/python3.11/site-packages/photon_mlir

# Set up environment
ENV PATH="/usr/local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/lib/photon:$LD_LIBRARY_PATH"
ENV PYTHONPATH="/usr/local/lib/python3.11/site-packages:$PYTHONPATH"

USER photon
WORKDIR /home/photon

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD photon-compile --version || exit 1

# Default command
CMD ["photon-compile", "--help"]

# Documentation stage
FROM base as docs

USER developer
WORKDIR /workspace

# Install documentation dependencies
RUN python3.11 -m pip install --user \
    sphinx \
    sphinx-rtd-theme \
    myst-parser \
    breathe \
    exhale

# Copy source code
COPY --chown=developer:developer . .

# Build documentation
RUN mkdir -p docs/_build && \
    cd docs && \
    make html

# Expose documentation
EXPOSE 8000
CMD ["python3.11", "-m", "http.server", "8000", "--directory", "docs/_build/html"]

# Testing stage for CI
FROM development as ci

# Override user for CI environments
USER root

# Install additional CI tools
RUN apt-get update && apt-get install -y \
    lcov \
    gcovr \
    && rm -rf /var/lib/apt/lists/*

# Switch back to developer user
USER developer

# Configure for CI
ENV CI=true
ENV COVERAGE=true

# Build with coverage
RUN mkdir -p /workspace/build-ci
WORKDIR /workspace/build-ci

RUN cmake .. \
    -GNinja \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang-17 \
    -DCMAKE_CXX_COMPILER=clang++-17 \
    -DPHOTON_ENABLE_TESTS=ON \
    -DPHOTON_ENABLE_COVERAGE=ON \
    -DPHOTON_ENABLE_BENCHMARKS=ON

RUN ninja -j$(nproc)

# Default CI command
CMD ["ctest", "--output-on-failure", "--verbose"]

# Hardware simulation stage
FROM runtime as simulation

# Install simulation libraries
RUN apt-get update && apt-get install -y \
    # Simulation dependencies
    libopenblas-dev \
    libfftw3-dev \
    # Graphics for visualization
    libgl1-mesa-glx \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Copy simulation models and data
COPY --from=build --chown=photon:photon /workspace/simulation /home/photon/simulation

# Set simulation environment
ENV PHOTON_SIMULATION_MODE=1
ENV PHOTON_HARDWARE_AVAILABLE=0

WORKDIR /home/photon/simulation

# Default simulation command
CMD ["photon-profile", "--model", "examples/resnet50.onnx", "--target", "simulation"]

# Development with GPU support (optional)
FROM development as gpu-dev

USER root

# Install CUDA toolkit (if needed for GPU acceleration)
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && \
    apt-get install -y cuda-toolkit-12-0 && \
    rm -rf /var/lib/apt/lists/* && \
    rm cuda-keyring_1.0-1_all.deb

USER developer

# Set CUDA environment
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"
ENV LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Install GPU-accelerated Python packages
RUN python3.11 -m pip install --user \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu120