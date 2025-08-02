# Automation Scripts

This directory contains automation scripts for the photon-mlir-bridge project that help maintain code quality, manage dependencies, collect metrics, and perform repository maintenance.

## Scripts Overview

### 1. Metrics Collection (`collect_metrics.py`)

Automatically collects various project metrics including code quality, development velocity, build system performance, and community engagement.

```bash
# Basic metrics collection
python scripts/automation/collect_metrics.py

# Generate HTML report
python scripts/automation/collect_metrics.py --output-format html --output-file metrics-report.html

# Verbose output
python scripts/automation/collect_metrics.py --verbose
```

**Features:**
- Code quality metrics (LOC, test coverage, documentation coverage)
- GitHub metrics (stars, forks, contributors, PR activity)
- Build system metrics (success rates, duration)
- Performance metrics (from benchmark results)
- Security metrics (vulnerability scans)
- Automatic configuration updates

**Required Environment Variables:**
- `GITHUB_TOKEN`: For GitHub API access

### 2. Dependency Updates (`dependency_updates.py`)

Automatically checks for outdated dependencies and creates pull requests with updates.

```bash
# Check and update dependencies
python scripts/automation/dependency_updates.py

# Dry run mode
python scripts/automation/dependency_updates.py --dry-run

# Limit number of packages updated at once
python scripts/automation/dependency_updates.py --max-packages 5
```

**Features:**
- Python package updates (pip, pyproject.toml)
- System dependency checking (CMake, LLVM)
- GitHub Actions version updates
- Risk assessment for updates
- Automatic PR creation with detailed descriptions
- Support for major, minor, and patch version updates

**Required Environment Variables:**
- `GITHUB_TOKEN`: For creating pull requests
- `GITHUB_REPOSITORY`: Repository name (usually set by GitHub Actions)

### 3. Code Quality Monitor (`code_quality_monitor.py`)

Monitors code quality metrics and creates GitHub issues for quality improvements.

```bash
# Run quality analysis
python scripts/automation/code_quality_monitor.py

# Create GitHub issue with results
python scripts/automation/code_quality_monitor.py --create-issue

# Save report to file
python scripts/automation/code_quality_monitor.py --output-file quality-report.json

# Use custom thresholds
python scripts/automation/code_quality_monitor.py --config custom-thresholds.json
```

**Features:**
- Test coverage analysis
- Code complexity measurement (cyclomatic complexity, maintainability index)
- Code duplication detection
- Security issue scanning
- Technical debt estimation
- Automated GitHub issue creation/updates
- Customizable quality thresholds

**Dependencies:**
- pytest-cov (for coverage analysis)
- radon (for complexity metrics)
- bandit (for security scanning)
- pylint (for duplication detection)

### 4. Repository Maintenance (`repository_maintenance.py`)

Performs routine repository maintenance tasks to keep the repository clean and organized.

```bash
# Full maintenance run
python scripts/automation/repository_maintenance.py

# Dry run to see what would be done
python scripts/automation/repository_maintenance.py --dry-run

# Only clean up branches
python scripts/automation/repository_maintenance.py --branches-only

# Generate maintenance report
python scripts/automation/repository_maintenance.py --report-file maintenance-report.md
```

**Features:**
- Clean up merged branches
- Archive stale issues
- Git repository optimization
- Update repository topics
- Clean old workflow runs
- Standardize issue labels
- Comprehensive maintenance reporting

**Required Environment Variables:**
- `GITHUB_TOKEN`: For GitHub API operations
- `GITHUB_REPOSITORY`: Repository name

### 5. Metrics Dashboard (`../metrics/generate_metrics_dashboard.py`)

Generates an interactive HTML dashboard displaying project metrics and trends.

```bash
# Generate dashboard
python scripts/metrics/generate_metrics_dashboard.py

# Custom metrics file
python scripts/metrics/generate_metrics_dashboard.py --metrics-file custom-metrics.json

# Custom output location
python scripts/metrics/generate_metrics_dashboard.py --output dashboard/custom-dashboard.html
```

**Features:**
- Interactive HTML dashboard
- Health score calculation
- Category-wise metric visualization
- Trend charts and radar plots
- Responsive design
- Auto-refresh capability
- JSON API endpoint

**Dependencies:**
- matplotlib (for chart generation)
- seaborn (for styling)
- pandas (for data processing)
- jinja2 (for HTML templating)

## Integration with CI/CD

### GitHub Actions Integration

Add these automation scripts to your GitHub Actions workflows:

```yaml
# .github/workflows/automation.yml
name: Automation

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM UTC
  workflow_dispatch:

jobs:
  metrics-collection:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install PyGithub toml packaging matplotlib seaborn pandas jinja2
      
      - name: Collect metrics
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python scripts/automation/collect_metrics.py --verbose
      
      - name: Generate dashboard
        run: python scripts/metrics/generate_metrics_dashboard.py
      
      - name: Upload dashboard
        uses: actions/upload-artifact@v3
        with:
          name: metrics-dashboard
          path: dashboard/

  dependency-updates:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Check for dependency updates
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python scripts/automation/dependency_updates.py --max-packages 5

  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Run quality analysis
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python scripts/automation/code_quality_monitor.py --create-issue

  repository-maintenance:
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Need full history for branch cleanup
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Run maintenance
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: python scripts/automation/repository_maintenance.py
```

### Cron Jobs (Alternative)

For self-hosted environments, you can set up cron jobs:

```bash
# Add to crontab (crontab -e)

# Daily metrics collection at 2 AM
0 2 * * * /usr/bin/python3 /path/to/repo/scripts/automation/collect_metrics.py

# Weekly dependency updates on Sundays at 3 AM
0 3 * * 0 /usr/bin/python3 /path/to/repo/scripts/automation/dependency_updates.py

# Weekly quality analysis on Wednesdays at 4 AM
0 4 * * 3 /usr/bin/python3 /path/to/repo/scripts/automation/code_quality_monitor.py --create-issue

# Monthly repository maintenance on the 1st at 5 AM
0 5 1 * * /usr/bin/python3 /path/to/repo/scripts/automation/repository_maintenance.py
```

## Configuration

### Metrics Configuration

The metrics collection system uses `.github/project-metrics.json` for configuration. Key sections:

```json
{
  "project": {
    "name": "photon-mlir-bridge",
    "repository": "danieleschmidt/photon-mlir-bridge"
  },
  "metrics": {
    "code_quality": {
      "measurements": {
        "test_coverage": {
          "target": 85,
          "warning_threshold": 70,
          "critical_threshold": 60
        }
      }
    }
  }
}
```

### Quality Thresholds

Create a custom quality thresholds file:

```json
{
  "code_coverage": {
    "excellent": 95,
    "good": 85,
    "warning": 75,
    "critical": 65
  },
  "complexity": {
    "excellent": 3,
    "good": 8,
    "warning": 12,
    "critical": 18
  }
}
```

### Environment Variables

Required environment variables:

```bash
# GitHub integration
export GITHUB_TOKEN="ghp_your_token_here"
export GITHUB_REPOSITORY="owner/repository"

# Optional: Custom API endpoints
export INFLUXDB_URL="https://your-influx-instance.com"
export INFLUXDB_TOKEN="your-influx-token"
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/..."
```

## Best Practices

### 1. Gradual Rollout

Start with dry-run mode to understand what the scripts will do:

```bash
python scripts/automation/dependency_updates.py --dry-run
python scripts/automation/repository_maintenance.py --dry-run
```

### 2. Monitor Automation Results

- Review generated pull requests before merging
- Check GitHub issues created by quality monitoring
- Monitor metrics dashboard for trends
- Review maintenance reports

### 3. Customize Thresholds

Adjust quality and metrics thresholds based on your project's needs:

- Start with conservative thresholds
- Gradually tighten as quality improves
- Consider project phase (early development vs. mature)

### 4. Error Handling

All scripts include comprehensive error handling and logging:

- Check logs for issues
- Use `--verbose` flag for detailed output
- Scripts exit with appropriate codes for CI integration

### 5. Security Considerations

- Use fine-grained GitHub tokens with minimal required permissions
- Store sensitive configuration in environment variables
- Review automated changes before merging
- Monitor for unexpected behavior

## Troubleshooting

### Common Issues

1. **GitHub API Rate Limits**
   - Use authentication tokens
   - Implement backoff strategies
   - Monitor rate limit usage

2. **Permission Errors**
   - Ensure GitHub token has required permissions
   - Check repository access levels
   - Verify branch protection rules

3. **Dependency Installation Failures**
   - Check system dependencies (CMake, LLVM)
   - Verify Python environment setup
   - Review package conflicts

4. **Quality Analysis Failures**
   - Install required analysis tools
   - Check file paths and permissions
   - Verify test configuration

### Debug Mode

Enable verbose logging for troubleshooting:

```bash
python scripts/automation/collect_metrics.py --verbose
python scripts/automation/code_quality_monitor.py --verbose
```

### Log Files

Scripts generate log files in the current directory:
- `metrics_collection.log`
- `quality_analysis.log`
- `maintenance.log`

This automation suite provides comprehensive project maintenance capabilities, helping maintain high code quality, up-to-date dependencies, and clean repository organization.