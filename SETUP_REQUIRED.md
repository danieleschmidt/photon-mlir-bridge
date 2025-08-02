# Manual Setup Required

This document outlines the manual setup steps required after the automated SDLC implementation due to GitHub App permission limitations.

## Overview

The photon-mlir-bridge repository has been enhanced with a comprehensive Software Development Life Cycle (SDLC) implementation through automated checkpoints. However, some configurations require manual setup due to GitHub App permission restrictions.

## Required Manual Actions

### 1. GitHub Actions Workflows 🔧

**Action Required**: Copy workflow templates to create actual GitHub Actions workflows.

```bash
# Copy workflow templates to active workflows directory
mkdir -p .github/workflows
cp docs/workflows/examples/*.yml .github/workflows/

# Commit the workflow files
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows from templates"
git push origin main
```

**Files to copy**:
- `docs/workflows/examples/ci.yml` → `.github/workflows/ci.yml`
- `docs/workflows/examples/release.yml` → `.github/workflows/release.yml`
- `docs/workflows/examples/security.yml` → `.github/workflows/security.yml`
- `docs/workflows/examples/docs.yml` → `.github/workflows/docs.yml`
- `docs/workflows/examples/performance.yml` → `.github/workflows/performance.yml`

### 2. Repository Secrets Configuration 🔐

**Action Required**: Add required secrets in repository settings.

Navigate to **Settings → Secrets and Variables → Actions** and add:

#### Required Secrets
- `PYPI_API_TOKEN`: Token for PyPI package publishing
  - Get from: https://pypi.org/manage/account/token/
  - Scope: Entire account or specific project

#### Optional Secrets (for enhanced functionality)
- `CODECOV_TOKEN`: Code coverage reporting
- `SLACK_WEBHOOK_URL`: Slack notifications
- `SLACK_SECURITY_WEBHOOK`: Security-specific notifications
- `SNYK_TOKEN`: Snyk vulnerability scanning
- `SEMGREP_APP_TOKEN`: Semgrep security analysis
- `INFLUXDB_TOKEN`: Performance metrics storage
- `INFLUXDB_URL`: InfluxDB instance URL
- `INFLUXDB_ORG`: InfluxDB organization

### 3. Branch Protection Rules 🛡️

**Action Required**: Configure branch protection in repository settings.

Navigate to **Settings → Branches** and add rules for `main`:

```yaml
Branch Protection Settings:
✅ Require a pull request before merging
  ✅ Require approvals (1)
  ✅ Dismiss stale PR approvals when new commits are pushed
  ✅ Require review from code owners

✅ Require status checks to pass before merging
  ✅ ci-success
  ✅ security-report  
  ✅ documentation
  ✅ performance (if applicable)

✅ Require branches to be up to date before merging
✅ Require signed commits (recommended)
✅ Include administrators
❌ Allow force pushes
❌ Allow deletions
```

### 4. GitHub Pages Setup 📄

**Action Required**: Enable GitHub Pages for documentation.

Navigate to **Settings → Pages**:
- **Source**: GitHub Actions
- **Custom domain**: (optional)
- **Enforce HTTPS**: ✅ Enabled

### 5. Repository Environments 🌍

**Action Required**: Create deployment environments.

Navigate to **Settings → Environments** and create:

#### PyPI Environment
- **Name**: `pypi`
- **Deployment protection rules**:
  - Required reviewers: Add team members
  - Wait timer: 0 minutes
- **Environment secrets**:
  - `PYPI_API_TOKEN`: Production PyPI token

#### GitHub Pages Environment  
- **Name**: `github-pages`
- **Deployment protection rules**: (optional)

### 6. Issue and PR Templates 📝

**Action Required**: Review and customize templates.

Templates are provided in:
- `.github/ISSUE_TEMPLATE/`
- `.github/PULL_REQUEST_TEMPLATE.md`

Customize these templates based on your project's specific needs.

### 7. Automation Configuration ⚙️

**Action Required**: Configure automation scripts.

1. **Metrics Collection**: Review `.github/project-metrics.json` and adjust thresholds
2. **Dependency Updates**: Test dependency automation in dry-run mode
3. **Quality Monitoring**: Configure quality thresholds if needed
4. **Repository Maintenance**: Schedule maintenance automation

### 8. Monitoring Setup 📊

**Action Required**: Set up monitoring infrastructure.

If you want to deploy the monitoring stack:

```bash
# Local development
docker-compose -f docker-compose.yml -f monitoring/docker-compose.monitoring.yml up -d

# Kubernetes deployment
kubectl apply -f monitoring/k8s/
```

## Verification Checklist

After completing the manual setup, verify the implementation:

### ✅ Workflows
- [ ] CI workflow runs on PRs and pushes
- [ ] Security scans execute daily
- [ ] Documentation builds and deploys
- [ ] Release workflow triggers on tags
- [ ] Performance monitoring runs on schedule

### ✅ Protection Rules  
- [ ] Main branch is protected
- [ ] Required status checks are enforced
- [ ] PR reviews are required
- [ ] Force pushes are blocked

### ✅ Automation
- [ ] Metrics collection runs successfully
- [ ] Dependency updates create PRs
- [ ] Code quality monitoring creates issues
- [ ] Repository maintenance completes

### ✅ Documentation
- [ ] GitHub Pages deploys successfully
- [ ] API documentation is generated
- [ ] README and guides are accessible

### ✅ Security
- [ ] Secrets are properly configured
- [ ] Security scans run without errors
- [ ] SBOM generation works
- [ ] Vulnerability reporting is functional

## Testing the Implementation

### 1. Test CI Pipeline
```bash
# Create a test branch and PR
git checkout -b test-sdlc-implementation
echo "# Test change" >> README.md
git add README.md
git commit -m "test: verify SDLC implementation"
git push origin test-sdlc-implementation

# Create PR via GitHub UI and verify all checks pass
```

### 2. Test Automation
```bash
# Test metrics collection
python scripts/automation/collect_metrics.py --verbose

# Test dependency updates (dry run)
python scripts/automation/dependency_updates.py --dry-run

# Test quality monitoring
python scripts/automation/code_quality_monitor.py --output-file test-quality.json

# Generate metrics dashboard
python scripts/metrics/generate_metrics_dashboard.py
```

### 3. Test Release Process
```bash
# Create a test release
git tag v0.1.0-test
git push origin v0.1.0-test

# Verify release workflow executes
```

## Support and Troubleshooting

### Common Issues

1. **Workflow permissions**: Ensure `GITHUB_TOKEN` has sufficient permissions
2. **Secret access**: Verify secrets are added to correct repository/environment
3. **Branch protection**: Check if protection rules allow required operations
4. **Dependency issues**: Install missing system dependencies (LLVM, CMake)

### Getting Help

- **Workflow issues**: Check workflow logs in Actions tab
- **Permission errors**: Review repository settings and token permissions  
- **Automation failures**: Check script logs and enable verbose mode
- **Documentation problems**: Verify GitHub Pages configuration

### Documentation References

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/defining-the-mergeability-of-pull-requests/about-protected-branches)
- [Repository Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GitHub Pages](https://docs.github.com/en/pages)

## Implementation Status

This SDLC implementation provides:

✅ **Complete project foundation** with documentation and community files  
✅ **Development environment** with containers and code quality tools  
✅ **Comprehensive testing** infrastructure with multiple test types  
✅ **Build automation** with multi-platform support and containerization  
✅ **Monitoring and observability** with Prometheus, Grafana, and alerting  
✅ **Workflow templates** for CI/CD, security, and performance monitoring  
✅ **Metrics and automation** for code quality, dependency management, and maintenance  
✅ **Integration configuration** with detailed setup instructions  

The implementation follows industry best practices for:
- Software development lifecycle management
- Continuous integration and deployment
- Security and compliance
- Code quality and maintainability
- Community engagement and contribution
- Operational excellence

---

**Next Steps**: Complete the manual setup tasks above, then your repository will have a fully functional, enterprise-grade SDLC implementation! 🚀