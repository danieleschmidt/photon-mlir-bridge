# GitHub Workflows Setup Guide

This guide provides step-by-step instructions for implementing the GitHub Actions workflows for photon-mlir-bridge.

⚠️ **Important**: Due to GitHub App permission limitations, these workflow files must be created manually by repository maintainers.

## Quick Setup

### 1. Create Workflow Files

Copy the example workflow files from `docs/workflows/examples/` to `.github/workflows/`:

```bash
# Create the workflows directory
mkdir -p .github/workflows

# Copy workflow files
cp docs/workflows/examples/ci.yml .github/workflows/
cp docs/workflows/examples/release.yml .github/workflows/  
cp docs/workflows/examples/security.yml .github/workflows/
cp docs/workflows/examples/docs.yml .github/workflows/
cp docs/workflows/examples/performance.yml .github/workflows/
```

### 2. Configure Repository Secrets

Add the following secrets in **Settings → Secrets and Variables → Actions**:

#### Required Secrets
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `GITHUB_TOKEN`: Automatically provided by GitHub

#### Optional Secrets (for enhanced functionality)
- `CODECOV_TOKEN`: Code coverage reporting
- `SLACK_WEBHOOK_URL`: Slack notifications
- `SLACK_SECURITY_WEBHOOK`: Security-specific Slack channel
- `SNYK_TOKEN`: Snyk vulnerability scanning
- `SEMGREP_APP_TOKEN`: Semgrep security scanning
- `DOCKER_SCOUT_HUB_USER`: Docker Scout scanning
- `DOCKER_SCOUT_HUB_PASSWORD`: Docker Scout password
- `INFLUXDB_TOKEN`: Performance metrics storage
- `INFLUXDB_URL`: InfluxDB instance URL
- `INFLUXDB_ORG`: InfluxDB organization
- `PERFORMANCE_API_URL`: Custom performance API
- `PERFORMANCE_API_TOKEN`: Custom performance API token

### 3. Configure Branch Protection Rules

In **Settings → Branches**, add protection rules for `main`:

```yaml
Branch Protection Rules for 'main':
- Require a pull request before merging
- Require status checks to pass before merging
  ✓ ci-success
  ✓ security-report
  ✓ documentation
- Require branches to be up to date before merging
- Require signed commits (recommended)
- Include administrators
- Allow force pushes: false
- Allow deletions: false
```

### 4. Enable GitHub Pages

In **Settings → Pages**:
- Source: **GitHub Actions**
- Custom domain: (optional)

### 5. Configure Environments

Create deployment environments in **Settings → Environments**:

#### PyPI Environment
- Name: `pypi`
- Deployment protection rules:
  - Required reviewers: (add team members)
  - Wait timer: 0 minutes
- Environment secrets:
  - `PYPI_API_TOKEN`: Production PyPI token

#### GitHub Pages Environment
- Name: `github-pages`
- Deployment protection rules: (optional)

## Detailed Configuration

### CI Workflow (ci.yml)

The continuous integration workflow runs on every push and pull request.

#### Features:
- **Multi-platform testing**: Ubuntu, macOS, Windows
- **Python version matrix**: 3.9, 3.10, 3.11, 3.12
- **C++ compilation**: MLIR/LLVM integration
- **Security scanning**: Bandit, Safety, CodeQL
- **Code coverage**: Codecov integration
- **Performance checks**: Basic benchmarking

#### Configuration Options:

```yaml
# Modify matrix testing in ci.yml
strategy:
  matrix:
    os: [ubuntu-22.04, macos-13, windows-2022]
    python-version: ['3.9', '3.10', '3.11', '3.12']
    # Add/remove OS or Python versions as needed
```

#### Skip CI for Certain Changes:

```yaml
# Add to paths-ignore in ci.yml
paths-ignore:
  - 'docs/**'
  - '*.md'
  - '.gitignore'
  - 'examples/**'  # Add if you want to skip examples
```

### Release Workflow (release.yml)

Automates package building and publishing when version tags are pushed.

#### Triggering a Release:

```bash
# Tag a new version
git tag v1.0.0
git push origin v1.0.0

# Or use GitHub's release interface
```

#### Release Types:
- **Stable release**: `v1.0.0`
- **Pre-release**: `v1.0.0-alpha.1`, `v1.0.0-beta.1`, `v1.0.0-rc.1`

#### Configuration:

```yaml
# Modify platforms in release.yml
strategy:
  matrix:
    os: [ubuntu-22.04, macos-13, windows-2022]
    # Remove platforms you don't want to support
```

### Security Workflow (security.yml)

Comprehensive security scanning runs daily and on security-related changes.

#### Scan Types:
- **SAST**: Static Application Security Testing with CodeQL
- **Dependency scanning**: Vulnerable dependencies detection
- **Container scanning**: Docker image vulnerabilities
- **Secret scanning**: Exposed secrets detection
- **SBOM generation**: Software Bill of Materials

#### Customization:

```yaml
# Modify scan schedule in security.yml
on:
  schedule:
    - cron: '0 2 * * *'  # Change time as needed
```

#### Security Tools Configuration:

```yaml
# Add custom security rules
- name: Run custom security checks
  run: |
    # Add your custom security scripts here
    ./scripts/custom_security_check.sh
```

### Documentation Workflow (docs.yml)

Builds and deploys documentation to GitHub Pages.

#### Features:
- **Sphinx documentation**: Python and C++ API docs
- **Multi-format output**: HTML, PDF, EPUB
- **Accessibility checking**: axe-core and Lighthouse
- **Documentation quality checks**: Link checking, formatting
- **PR documentation diffs**: Compare docs between branches

#### Custom Documentation:

```yaml
# Add custom documentation steps
- name: Build custom docs
  run: |
    # Add custom documentation generation
    python scripts/generate_api_docs.py
```

### Performance Workflow (performance.yml)

Monitors performance and detects regressions.

#### Features:
- **Compilation benchmarks**: Model compilation performance
- **Runtime benchmarks**: Inference performance
- **Memory benchmarks**: Memory usage and leak detection
- **Regression detection**: Automatic performance regression alerts
- **Performance database**: Upload results to time-series database

#### Benchmark Configuration:

```yaml
# Modify benchmark matrix
strategy:
  matrix:
    model_size: [small, medium, large, xl]  # Add xl if needed
    target: [simulation, lightmatter, custom]  # Add custom targets
```

## Advanced Configuration

### Custom Notification Channels

#### Slack Integration:

1. Create a Slack app and webhook
2. Add webhook URL to repository secrets
3. Customize notification format:

```yaml
# In workflow files
- name: Send Slack notification
  env:
    SLACK_WEBHOOK_URL: ${{ secrets.SLACK_WEBHOOK_URL }}
  run: |
    curl -X POST -H 'Content-type: application/json' \
      --data '{
        "text": "Custom notification message",
        "channel": "#photon-mlir-alerts"
      }' \
      $SLACK_WEBHOOK_URL
```

#### Microsoft Teams:

```yaml
- name: Send Teams notification
  env:
    TEAMS_WEBHOOK_URL: ${{ secrets.TEAMS_WEBHOOK_URL }}
  run: |
    curl -X POST -H 'Content-type: application/json' \
      --data '{
        "@type": "MessageCard",
        "summary": "Build Status",
        "text": "Build completed successfully"
      }' \
      $TEAMS_WEBHOOK_URL
```

### Performance Database Integration

#### InfluxDB Setup:

1. Set up InfluxDB instance
2. Create performance bucket
3. Add connection secrets to repository
4. Configure data retention policies

#### Custom Performance API:

```python
# scripts/upload_performance_data.py
import requests
import json

def upload_to_api(results, api_url, token):
    headers = {'Authorization': f'Bearer {token}'}
    response = requests.post(f'{api_url}/benchmarks', 
                           json=results, headers=headers)
    return response.status_code == 200
```

### Security Integration

#### External Security Services:

```yaml
# Add to security.yml
- name: Run external security scan
  env:
    SECURITY_API_KEY: ${{ secrets.SECURITY_API_KEY }}
  run: |
    # Integration with external security service
    security-tool scan --api-key $SECURITY_API_KEY .
```

#### Custom Security Policies:

```yaml
# .github/security-policy.yml
security:
  vulnerability_thresholds:
    critical: 0    # Block on any critical vulnerabilities
    high: 2        # Allow up to 2 high severity issues
    medium: 10     # Allow up to 10 medium severity issues
  
  dependency_policies:
    auto_update: true
    allowed_licenses:
      - MIT
      - Apache-2.0
      - BSD-3-Clause
    blocked_licenses:
      - GPL-3.0
      - AGPL-3.0
```

## Troubleshooting

### Common Issues

#### 1. MLIR/LLVM Installation Failures

**Problem**: MLIR compilation fails on different platforms

**Solution**:
```yaml
# Add to workflow
- name: Debug MLIR installation
  run: |
    llvm-config --version
    mlir-opt --version
    which clang
    ldd $(which mlir-opt) || otool -L $(which mlir-opt)
```

#### 2. Python Package Build Failures

**Problem**: wheel building fails for specific Python versions

**Solution**:
```yaml
# Add debugging steps
- name: Debug Python environment
  run: |
    python --version
    pip list
    python -c "import sys; print(sys.path)"
    pip install --verbose -e .
```

#### 3. Docker Build Timeouts

**Problem**: Docker builds exceed time limits

**Solution**:
```yaml
# Optimize Docker build
- name: Build with cache
  uses: docker/build-push-action@v5
  with:
    cache-from: type=gha
    cache-to: type=gha,mode=max
    build-args: |
      BUILDKIT_INLINE_CACHE=1
```

#### 4. Security Scan False Positives

**Problem**: Security tools report false positives

**Solution**:
```yaml
# Add suppressions
- name: Run security scan with suppressions
  run: |
    bandit -r python/ -x tests/ --skip B101,B601
    safety check --ignore 12345
```

### Performance Optimization

#### Workflow Optimization:

```yaml
# Use larger runners for intensive tasks
runs-on: ubuntu-latest-8-cores  # GitHub hosted
# or
runs-on: self-hosted  # Self-hosted runners
```

#### Parallel Execution:

```yaml
# Increase job parallelism
strategy:
  max-parallel: 10  # Default is usually 5
  matrix:
    # Your matrix configuration
```

#### Caching Optimization:

```yaml
# Optimize cache usage
- name: Cache everything
  uses: actions/cache@v3
  with:
    path: |
      ~/.cache/pip
      ~/.cache/cmake
      build/
      node_modules/
    key: ${{ runner.os }}-all-${{ hashFiles('**/*') }}
    restore-keys: |
      ${{ runner.os }}-all-
      ${{ runner.os }}-
```

## Monitoring and Maintenance

### Weekly Tasks
- [ ] Review workflow success rates
- [ ] Check performance trends
- [ ] Update action versions
- [ ] Review security scan results

### Monthly Tasks
- [ ] Analyze workflow performance
- [ ] Update dependencies
- [ ] Review notification settings
- [ ] Clean up old artifacts

### Quarterly Tasks
- [ ] Update workflow templates
- [ ] Review and update security policies
- [ ] Performance baseline updates
- [ ] Documentation review

This setup guide ensures robust, secure, and efficient CI/CD pipelines for the photon-mlir-bridge project.