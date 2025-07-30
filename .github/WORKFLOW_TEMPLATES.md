# GitHub Actions Workflow Templates

This document provides template workflows for the photon-mlir-bridge project. These workflows should be created in `.github/workflows/` directory.

## Required Workflows

### 1. CI Pipeline (`ci.yml`)

```yaml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-cpp:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        compiler: [gcc-11, clang-13]
    
    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive
    
    - name: Install LLVM/MLIR
      run: |
        sudo apt-get update
        sudo apt-get install -y llvm-17-dev mlir-17-tools
    
    - name: Configure CMake
      run: |
        mkdir build
        cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
    
    - name: Build
      run: |
        cd build
        make -j$(nproc)
    
    - name: Test
      run: |
        cd build
        ctest --verbose

  test-python:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
    
    - name: Run tests
      run: |
        pytest --cov=photon_mlir --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run CodeQL Analysis
      uses: github/codeql-action/init@v2
      with:
        languages: cpp, python
    
    - name: Build for analysis
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Debug
        make -j$(nproc)
    
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2
```

### 2. Release Automation (`release.yml`)

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  create-release:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
      with:
        submodules: recursive
    
    - name: Build Release Artifacts
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release
        make -j$(nproc) package
    
    - name: Create Release
      uses: actions/create-release@v1
      env:
        GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      with:
        tag_name: ${{ github.ref }}
        release_name: Release ${{ github.ref }}
        draft: false
        prerelease: false

  publish-python:
    runs-on: ubuntu-latest
    needs: create-release
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Build Python package
      run: |
        pip install build
        python -m build
    
    - name: Publish to PyPI
      uses: pypa/gh-action-pypi-publish@release/v1
      with:
        password: ${{ secrets.PYPI_API_TOKEN }}
```

### 3. Documentation Build (`docs.yml`)

```yaml
name: Documentation

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Install dependencies
      run: |
        pip install -e ".[docs]"
        sudo apt-get install -y doxygen
    
    - name: Build C++ docs
      run: |
        mkdir build && cd build
        cmake .. -DPHOTON_BUILD_DOCS=ON
        make docs
    
    - name: Build Python docs
      run: |
        cd docs
        make html
    
    - name: Deploy to GitHub Pages
      if: github.ref == 'refs/heads/main'
      uses: peaceiris/actions-gh-pages@v3
      with:
        github_token: ${{ secrets.GITHUB_TOKEN }}
        publish_dir: ./docs/_build/html
```

### 4. Dependency Updates (`dependency-update.yml`)

```yaml
name: Dependency Updates

on:
  schedule:
    - cron: '0 2 * * MON'  # Weekly on Monday
  workflow_dispatch:

jobs:
  update-python-deps:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: "3.11"
    
    - name: Update dependencies
      run: |
        pip install pip-tools
        pip-compile --upgrade pyproject.toml
    
    - name: Create PR
      uses: peter-evans/create-pull-request@v5
      with:
        token: ${{ secrets.GITHUB_TOKEN }}
        commit-message: "chore: update Python dependencies"
        title: "Update Python dependencies"
        branch: update-python-deps

  security-audit:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run security audit
      run: |
        pip install safety
        safety check --json --output safety-report.json || true
    
    - name: Upload security report
      uses: actions/upload-artifact@v3
      with:
        name: security-report
        path: safety-report.json
```

## Integration Requirements

### Environment Secrets

Add these secrets to your GitHub repository:

- `CODECOV_TOKEN`: For code coverage reporting
- `PYPI_API_TOKEN`: For PyPI package publishing
- `DOCKER_HUB_USERNAME` & `DOCKER_HUB_TOKEN`: For container registry

### Branch Protection Rules

Configure these rules for the `main` branch:

- Require status checks: `test-cpp`, `test-python`, `security-scan`
- Require up-to-date branches
- Require linear history
- Include administrators

### Repository Settings

Enable these features:

- Issues and Projects
- Discussions
- Security advisories
- Dependency graph
- Dependabot alerts

## Deployment Integration

### Container Registry Workflow

```yaml
name: Container Build

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build-container:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_TOKEN }}
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: |
          photonmlir/compiler:latest
          photonmlir/compiler:${{ github.sha }}
```

## Performance Monitoring

### Benchmark Workflow

```yaml
name: Performance Benchmarks

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Run benchmarks
      run: |
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release -DPHOTON_ENABLE_BENCHMARKS=ON
        make -j$(nproc)
        ./benchmarks/photon_benchmarks --output=json > benchmark_results.json
    
    - name: Store benchmark results
      uses: benchmark-action/github-action-benchmark@v1
      with:
        tool: 'googlecpp'
        output-file-path: build/benchmark_results.json
        github-token: ${{ secrets.GITHUB_TOKEN }}
        auto-push: true
```

## Workflow Integration Notes

1. **Parallel Execution**: Workflows are designed to run jobs in parallel for faster CI
2. **Matrix Strategy**: Tests across multiple compilers and Python versions
3. **Caching**: Use actions/cache for dependencies to speed up builds
4. **Artifacts**: Store build artifacts and test reports
5. **Security**: Regular security scanning and dependency audits
6. **Documentation**: Automatic documentation builds and deployment

## Manual Setup Instructions

After creating these workflow files:

1. Configure repository secrets
2. Set up branch protection rules
3. Enable required repository features
4. Test workflows with a small PR
5. Monitor workflow performance and adjust as needed

For questions about workflow setup, refer to the [GitHub Actions documentation](https://docs.github.com/en/actions).