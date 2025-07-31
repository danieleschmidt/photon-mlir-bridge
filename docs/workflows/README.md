# GitHub Actions Workflows Documentation

This directory contains documentation for the required GitHub Actions workflows for the photon-mlir-bridge repository.

## Required Workflows

### 1. Continuous Integration (CI) - `ci.yml`

**Purpose**: Automated testing, building, and quality checks for every PR and push to main.

**Triggers**:
- Push to `main` branch
- Pull requests to `main` branch
- Manual dispatch

**Jobs**:
- **matrix-test**: Test across Python 3.9-3.12 and Ubuntu/macOS
- **cpp-build**: Build C++ components with CMake
- **security-scan**: Run Bandit, CodeQL, and dependency scanning
- **performance-test**: Run benchmark tests and regression detection

**Required Secrets**: None (uses GITHUB_TOKEN)

### 2. Release Automation - `release.yml`

**Purpose**: Automated releases when version tags are pushed.

**Triggers**:
- Tags matching `v*.*.*` pattern

**Jobs**:
- **build-artifacts**: Build wheels and source distributions
- **github-release**: Create GitHub release with artifacts
- **pypi-publish**: Upload to PyPI (requires PYPI_API_TOKEN secret)

### 3. Security Scanning - `security.yml`

**Purpose**: Comprehensive security analysis and vulnerability scanning.

**Triggers**:
- Schedule: Daily at 2 AM UTC
- Manual dispatch
- Push to main (security-critical files)

**Jobs**:
- **codeql**: GitHub CodeQL analysis
- **dependency-review**: Analyze dependency vulnerabilities
- **sbom-generation**: Generate Software Bill of Materials
- **container-scan**: Scan Docker images for vulnerabilities

### 4. Documentation - `docs.yml`

**Purpose**: Build and deploy documentation.

**Triggers**:
- Push to `main` (docs changes)
- Pull requests (docs changes)

**Jobs**:
- **build-docs**: Build Sphinx documentation
- **deploy-docs**: Deploy to GitHub Pages (main branch only)

### 5. Performance Monitoring - `performance.yml`

**Purpose**: Track performance metrics and detect regressions.

**Triggers**:
- Schedule: Weekly on Sundays
- Manual dispatch
- PR comments containing `/benchmark`

**Jobs**:
- **benchmark**: Run performance benchmarks
- **regression-check**: Compare against baseline metrics
- **performance-report**: Generate and store performance reports

## Implementation Guide

### Step 1: Create Workflow Files

Create these files in `.github/workflows/`:

1. `ci.yml` - Primary CI pipeline
2. `release.yml` - Release automation
3. `security.yml` - Security scanning
4. `docs.yml` - Documentation building
5. `performance.yml` - Performance monitoring

### Step 2: Configure Secrets

Add these secrets in repository settings:

- `PYPI_API_TOKEN`: For PyPI publishing
- `CODECOV_TOKEN`: For code coverage reporting (optional)

### Step 3: Enable GitHub Pages

1. Go to repository Settings → Pages
2. Set source to "GitHub Actions"
3. Configure custom domain if needed

### Step 4: Configure Branch Protection

Enable branch protection rules for `main`:

- Require status checks to pass
- Require up-to-date branches
- Require signed commits (recommended)
- Include administrators

## Advanced Features

### Matrix Testing Strategy

Test across multiple dimensions:
- Python versions: 3.9, 3.10, 3.11, 3.12
- Operating systems: Ubuntu 22.04, macOS-13, Windows 2022
- Build types: Debug, Release
- MLIR versions: 17.0, 18.0 (when available)

### Caching Strategy

Implement aggressive caching for:
- Python dependencies (`pip` cache)
- CMake build cache
- MLIR build artifacts
- Docker layer cache

### Security Integration

- Automatic dependency updates via Dependabot
- Container vulnerability scanning
- SLSA Level 3 build provenance
- SBOM generation and artifact signing

### Performance Monitoring

- Continuous benchmarking against baseline
- Performance regression detection
- Memory usage profiling
- Compilation time tracking

## Workflow Templates

### Basic CI Template Structure

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.9', '3.10', '3.11', '3.12']
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      # Add specific steps for your project
```

### Security Scanning Template

```yaml
name: Security Scan
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:

jobs:
  security:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    
    steps:
      - uses: actions/checkout@v4
      - name: Run CodeQL Analysis
        uses: github/codeql-action/init@v2
        with:
          languages: python, cpp
      
      # Add specific security scanning steps
```

## Maintenance Guidelines

### Weekly Tasks
- Review workflow run success rates
- Update action versions quarterly
- Monitor performance benchmarks

### Monthly Tasks
- Analyze security scan results
- Update dependency versions
- Review and optimize workflow efficiency

### Best Practices

1. **Fail Fast**: Put quick checks (linting, formatting) first
2. **Parallel Execution**: Use job matrices and parallelization
3. **Caching**: Cache dependencies and build artifacts aggressively
4. **Conditional Execution**: Skip unnecessary jobs on specific changes
5. **Security**: Use minimal required permissions and trusted actions

## Troubleshooting

### Common Issues

1. **Build Failures**: Check CMake configuration and dependencies
2. **Test Timeouts**: Increase timeout or optimize test performance
3. **Cache Misses**: Verify cache key generation and paths
4. **Permission Errors**: Check token permissions and repository settings

### Debug Strategies

1. Enable debug logging: `ACTIONS_RUNNER_DEBUG: true`
2. Use `tmate` action for interactive debugging
3. Check workflow logs and job summaries
4. Validate workflow syntax with act (local testing)

## Migration Path

### From Basic CI to Advanced Workflows

1. **Phase 1**: Implement basic CI with testing
2. **Phase 2**: Add security scanning and quality gates
3. **Phase 3**: Implement release automation
4. **Phase 4**: Add performance monitoring and advanced features

This phased approach ensures stability while gradually improving automation capabilities.