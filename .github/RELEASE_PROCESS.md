# Release Process Documentation

This document outlines the release process for photon-mlir-bridge, including versioning strategy, release automation, and deployment procedures.

## 🏷️ Versioning Strategy

We follow [Semantic Versioning](https://semver.org/) (SemVer):

```
MAJOR.MINOR.PATCH[-PRERELEASE][+BUILD]
```

### Version Components

- **MAJOR** (X.y.z): Incompatible API changes
- **MINOR** (x.Y.z): New functionality (backwards compatible)
- **PATCH** (x.y.Z): Bug fixes (backwards compatible)
- **PRERELEASE**: alpha, beta, rc.N for pre-releases
- **BUILD**: Build metadata (commit hash, build number)

### Examples

- `0.1.0` - Initial release
- `0.2.0` - New features added
- `0.2.1` - Bug fix release
- `0.3.0-alpha.1` - Alpha pre-release
- `0.3.0-beta.2` - Beta pre-release
- `0.3.0-rc.1` - Release candidate
- `1.0.0` - First stable release

## 📋 Release Types

### Major Releases (X.0.0)
- **Frequency**: Annually
- **Content**: Breaking changes, major features
- **Support**: Long-term support (LTS) consideration
- **Planning**: 6-month development cycle

### Minor Releases (x.Y.0)
- **Frequency**: Quarterly
- **Content**: New features, enhancements
- **Support**: Standard support cycle
- **Planning**: 3-month development cycle

### Patch Releases (x.y.Z)
- **Frequency**: As needed
- **Content**: Bug fixes, security patches
- **Support**: Immediate for critical issues
- **Planning**: Reactive, based on issues

### Pre-releases
- **Alpha**: Early development, incomplete features
- **Beta**: Feature complete, testing phase
- **Release Candidate**: Final testing before release

## 🚀 Release Workflow

### 1. Pre-Release Planning

#### Release Planning Meeting
- **When**: 2 weeks before release
- **Attendees**: Core maintainers, key contributors
- **Agenda**: 
  - Review milestone progress
  - Identify release blockers
  - Finalize feature scope
  - Plan release timeline

#### Release Branch Creation
```bash
# Create release branch from develop
git checkout develop
git pull origin develop
git checkout -b release/v0.2.0
git push origin release/v0.2.0
```

### 2. Release Preparation

#### Version Bump
```bash
# Update version in all relevant files
scripts/bump_version.py 0.2.0

# Files to update:
# - pyproject.toml
# - CMakeLists.txt
# - docs/conf.py
# - CHANGELOG.md
# - README.md (if needed)
```

#### CHANGELOG Update
```bash
# Move [Unreleased] changes to new version section
# Update links and dates
# Add new [Unreleased] section
vim CHANGELOG.md
```

#### Documentation Review
- Update API documentation
- Review examples and tutorials
- Check external links
- Validate code samples

#### Quality Assurance
```bash
# Run full test suite
pytest --cov=photon_mlir tests/
ctest --verbose

# Run benchmarks
python scripts/run_benchmarks.py

# Security scanning
bandit -r python/
safety check

# Static analysis
clang-tidy src/**/*.cpp
mypy python/photon_mlir/
```

### 3. Release Candidate

#### Create RC Tag
```bash
git tag -a v0.2.0-rc.1 -m "Release candidate 0.2.0-rc.1"
git push origin v0.2.0-rc.1
```

#### RC Testing
- **Duration**: 1 week minimum
- **Testing**: 
  - Automated CI/CD pipeline
  - Manual hardware validation
  - Community beta testing
  - Performance regression testing

#### RC Communication
- Announce on GitHub Discussions
- Notify hardware partners
- Update documentation preview
- Collect feedback

### 4. Final Release

#### Release Approval
- [ ] All tests passing
- [ ] No critical bugs
- [ ] Documentation complete
- [ ] Hardware validation passed
- [ ] Security review complete
- [ ] Performance benchmarks within targets

#### Create Release Tag
```bash
git checkout release/v0.2.0
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

#### Merge to Main
```bash
git checkout main
git merge --no-ff release/v0.2.0
git push origin main
```

#### Back-merge to Develop
```bash
git checkout develop
git merge --no-ff release/v0.2.0
git push origin develop
```

### 5. Release Deployment

#### GitHub Release
Create release on GitHub with:
- Release notes from CHANGELOG
- Binary artifacts
- Source code archives
- Hardware compatibility matrix
- Migration guide (if needed)

#### Package Distribution
```bash
# Python package to PyPI
python -m build
twine upload dist/*

# Container images
docker build -t photonmlir/compiler:0.2.0 .
docker push photonmlir/compiler:0.2.0
docker tag photonmlir/compiler:0.2.0 photonmlir/compiler:latest
docker push photonmlir/compiler:latest

# Documentation deployment
cd docs
make html
rsync -av _build/html/ user@docs-server:/var/www/photon-mlir/
```

#### Homebrew Formula (if applicable)
```bash
# Update Homebrew formula
brew bump-formula-pr photon-mlir --url=https://github.com/user/repo/archive/v0.2.0.tar.gz
```

### 6. Post-Release

#### Communication
- [ ] Release announcement blog post
- [ ] Social media updates
- [ ] Newsletter to subscribers
- [ ] Academic conference presentations
- [ ] Hardware partner notifications

#### Documentation Updates
- [ ] Update main documentation site
- [ ] Archive previous version docs
- [ ] Update installation instructions
- [ ] Refresh getting started guides

#### Monitoring
- [ ] Monitor CI/CD pipelines
- [ ] Track download metrics
- [ ] Monitor issue reports
- [ ] Watch performance metrics

## 🔄 Automated Release Pipeline

### GitHub Actions Workflow

```yaml
name: Release Pipeline
on:
  push:
    tags:
      - 'v*'

jobs:
  release:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout
        uses: actions/checkout@v4
        
      - name: Build and Test
        run: |
          mkdir build && cd build
          cmake .. -DCMAKE_BUILD_TYPE=Release
          make -j$(nproc)
          ctest
          
      - name: Build Python Package
        run: |
          python -m build
          
      - name: Create GitHub Release
        uses: actions/create-release@v1
        with:
          tag_name: ${{ github.ref }}
          release_name: Release ${{ github.ref }}
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

### Release Automation Scripts

#### Version Bump Script
```python
#!/usr/bin/env python3
# scripts/bump_version.py

import sys
import re
from pathlib import Path

def bump_version(new_version):
    """Update version in all project files."""
    files_to_update = [
        'pyproject.toml',
        'CMakeLists.txt',
        'docs/conf.py',
        'README.md',
    ]
    
    for file_path in files_to_update:
        update_version_in_file(file_path, new_version)

def update_version_in_file(file_path, version):
    """Update version in specific file."""
    # Implementation details...
    pass

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: bump_version.py <new_version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    bump_version(new_version)
```

#### Changelog Generator
```python
#!/usr/bin/env python3
# scripts/generate_changelog.py

import subprocess
import re
from datetime import datetime

def generate_changelog(from_tag, to_tag):
    """Generate changelog from git commits."""
    # Get commits between tags
    cmd = f"git log {from_tag}..{to_tag} --oneline --pretty=format:'%h %s'"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Parse commits and categorize
    commits = result.stdout.strip().split('\n')
    changelog = categorize_commits(commits)
    
    return format_changelog(changelog, to_tag)

def categorize_commits(commits):
    """Categorize commits by type."""
    categories = {
        'Added': [],
        'Changed': [],
        'Fixed': [],
        'Security': [],
        'Performance': [],
    }
    
    for commit in commits:
        # Categorize based on commit message
        # Implementation details...
        pass
    
    return categories
```

## 📊 Release Metrics

### Key Performance Indicators (KPIs)

#### Release Quality
- **Bug Reports**: Number of bugs reported within 30 days
- **Regression Rate**: Percentage of features that regressed
- **Test Coverage**: Code coverage percentage
- **Security Issues**: Number of security vulnerabilities

#### Release Process
- **Lead Time**: Time from feature complete to release
- **Cycle Time**: Time from commit to production
- **Deployment Frequency**: Number of releases per quarter
- **Mean Time to Recovery**: Time to fix critical issues

#### Community Impact
- **Adoption Rate**: Download/installation metrics
- **Community Feedback**: User satisfaction scores
- **Documentation Usage**: Page views and engagement
- **Support Load**: Number of support requests

### Release Dashboard

Track metrics in real-time:
- GitHub releases and downloads
- PyPI package statistics
- Docker Hub pull statistics
- Documentation site analytics
- Community forum activity

## 🛡️ Release Security

### Security Checklist

- [ ] **Dependency Scanning**: All dependencies scanned for vulnerabilities
- [ ] **Code Scanning**: Static analysis completed
- [ ] **Container Scanning**: Docker images scanned
- [ ] **Secrets Scanning**: No credentials in release artifacts
- [ ] **Signing**: All artifacts digitally signed
- [ ] **SBOM**: Software Bill of Materials generated

### Signing and Verification

```bash
# Sign release artifacts
gpg --armor --detach-sig photon-mlir-0.2.0.tar.gz

# Verify signatures
gpg --verify photon-mlir-0.2.0.tar.gz.asc
```

### Supply Chain Security

- Use dependency pinning
- Verify dependency integrity
- Scan for known vulnerabilities
- Monitor security advisories
- Maintain SBOM (Software Bill of Materials)

## 🚨 Hotfix Process

### Criteria for Hotfixes
- Critical security vulnerabilities
- Data corruption bugs
- System crash issues
- Hardware safety concerns

### Hotfix Workflow
```bash
# Create hotfix branch from main
git checkout main
git checkout -b hotfix/v0.1.1

# Make minimal fix
# Test thoroughly
# Update version and changelog

# Tag and release
git tag v0.1.1
git push origin v0.1.1

# Merge back to main and develop
git checkout main
git merge hotfix/v0.1.1
git checkout develop
git merge hotfix/v0.1.1
```

## 📞 Communication Plan

### Internal Communication
- **Team Notifications**: Slack/Discord updates
- **Stakeholder Updates**: Email summaries
- **Partner Briefings**: Direct communication with hardware partners

### External Communication
- **Release Notes**: Detailed GitHub release notes
- **Blog Posts**: Feature highlights and technical deep-dives
- **Social Media**: Twitter, LinkedIn announcements
- **Documentation**: Updated guides and examples

### Emergency Communication
- **Critical Issues**: Immediate notification to all stakeholders
- **Security Patches**: Security advisory publication
- **Service Disruptions**: Status page updates

## 📝 Templates

### Release Notes Template
```markdown
# Release v0.2.0

## Overview
Brief description of the release highlights and key changes.

## 🚀 New Features
- Feature 1: Description and benefits
- Feature 2: Description and benefits

## 🔧 Improvements  
- Enhancement 1: Performance improvements
- Enhancement 2: Usability improvements

## 🐛 Bug Fixes
- Fix 1: Description of issue and resolution
- Fix 2: Description of issue and resolution

## ⚠️ Breaking Changes
- Change 1: What changed and migration path
- Change 2: What changed and migration path

## 📊 Performance
- Compilation time: 15% improvement
- Runtime performance: 8% improvement
- Memory usage: 12% reduction

## 🏗️ Infrastructure
- Updated dependencies
- Enhanced CI/CD pipeline
- Improved documentation

## 🙏 Acknowledgments
Thanks to all contributors who made this release possible.

## 📋 Full Changelog
See the [full changelog](CHANGELOG.md) for complete details.
```

### Hotfix Announcement Template
```markdown
# Security Hotfix v0.1.1

## ⚠️ Critical Security Update

This hotfix addresses a critical security vulnerability that could allow...

## 🚨 Immediate Action Required

All users should upgrade immediately:

```bash
pip install --upgrade photon-mlir==0.1.1
```

## 🔍 Technical Details

- **CVE**: CVE-2025-XXXX
- **Severity**: Critical (CVSS 9.1)
- **Impact**: [Description of impact]
- **Resolution**: [Description of fix]

## 📞 Support

For questions or assistance with this update, contact security@photon-mlir.dev
```

---

For questions about the release process, contact the release team at releases@photon-mlir.dev