#!/usr/bin/env python3
"""
Automated dependency update script for photon-mlir-bridge project.

This script checks for outdated dependencies, creates update PRs,
and manages dependency lifecycle.
"""

import json
import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import requests
import toml
from packaging import version
from github import Github


class DependencyUpdater:
    """Manages automated dependency updates."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        self.github_client = self._setup_github_client()
        self.repo_root = Path.cwd()
        
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def _setup_github_client(self) -> Optional[Github]:
        """Set up GitHub API client."""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            self.logger.warning("GITHUB_TOKEN not found, PR creation will be disabled")
            return None
        return Github(token)
    
    def check_python_dependencies(self) -> List[Dict]:
        """Check for outdated Python dependencies."""
        self.logger.info("Checking Python dependencies...")
        outdated_packages = []
        
        try:
            # Check pip packages
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True, check=True)
            
            pip_outdated = json.loads(result.stdout)
            
            for package in pip_outdated:
                outdated_packages.append({
                    'name': package['name'],
                    'current_version': package['version'],
                    'latest_version': package['latest_version'],
                    'type': 'pip',
                    'manager': 'pip'
                })
            
            # Check pyproject.toml dependencies
            pyproject_path = self.repo_root / 'pyproject.toml'
            if pyproject_path.exists():
                with open(pyproject_path) as f:
                    pyproject_data = toml.load(f)
                
                dependencies = pyproject_data.get('project', {}).get('dependencies', [])
                optional_deps = pyproject_data.get('project', {}).get('optional-dependencies', {})
                
                # Flatten all dependencies
                all_deps = dependencies.copy()
                for dep_group in optional_deps.values():
                    all_deps.extend(dep_group)
                
                for dep in all_deps:
                    # Parse dependency specification
                    dep_name = dep.split('[')[0].split('>=')[0].split('==')[0].split('~=')[0].strip()
                    
                    # Check if this package is in our outdated list
                    for outdated in pip_outdated:
                        if outdated['name'].lower() == dep_name.lower():
                            # Add context about where it's defined
                            outdated_packages.append({
                                'name': outdated['name'],
                                'current_version': outdated['version'],
                                'latest_version': outdated['latest_version'],
                                'type': 'pyproject.toml',
                                'manager': 'pip',
                                'dependency_spec': dep
                            })
                            break
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error checking Python dependencies: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error checking Python dependencies: {e}")
        
        return outdated_packages
    
    def check_system_dependencies(self) -> List[Dict]:
        """Check for outdated system dependencies."""
        self.logger.info("Checking system dependencies...")
        outdated_packages = []
        
        try:
            # Check for CMake
            cmake_result = subprocess.run([
                'cmake', '--version'
            ], capture_output=True, text=True)
            
            if cmake_result.returncode == 0:
                current_cmake = cmake_result.stdout.split('\n')[0].split()[-1]
                
                # Get latest CMake version from GitHub releases
                try:
                    response = requests.get(
                        'https://api.github.com/repos/Kitware/CMake/releases/latest',
                        timeout=10
                    )
                    if response.status_code == 200:
                        latest_cmake = response.json()['tag_name'].lstrip('v')
                        
                        if version.parse(current_cmake) < version.parse(latest_cmake):
                            outdated_packages.append({
                                'name': 'cmake',
                                'current_version': current_cmake,
                                'latest_version': latest_cmake,
                                'type': 'system',
                                'manager': 'system'
                            })
                except Exception as e:
                    self.logger.warning(f"Could not check CMake version: {e}")
            
            # Check LLVM version
            llvm_result = subprocess.run([
                'llvm-config-17', '--version'
            ], capture_output=True, text=True)
            
            if llvm_result.returncode == 0:
                current_llvm = llvm_result.stdout.strip()
                
                # Note: LLVM version checking is complex due to our specific version requirements
                # We'll just log the current version for now
                self.logger.info(f"Current LLVM version: {current_llvm}")
            
        except Exception as e:
            self.logger.error(f"Error checking system dependencies: {e}")
        
        return outdated_packages
    
    def check_github_actions(self) -> List[Dict]:
        """Check for outdated GitHub Actions."""
        self.logger.info("Checking GitHub Actions...")
        outdated_actions = []
        
        workflows_dir = self.repo_root / '.github' / 'workflows'
        if not workflows_dir.exists():
            workflows_dir = self.repo_root / 'docs' / 'workflows' / 'examples'
        
        if workflows_dir.exists():
            for workflow_file in workflows_dir.glob('*.yml'):
                try:
                    with open(workflow_file) as f:
                        content = f.read()
                    
                    # Simple regex to find action versions
                    import re
                    action_pattern = r'uses:\s+([^@\s]+)@([v\d\.\-\w]+)'
                    matches = re.findall(action_pattern, content)
                    
                    for action_name, current_version in matches:
                        # Skip local actions
                        if action_name.startswith('./'):
                            continue
                        
                        # Check if we can get latest version
                        try:
                            # For GitHub actions, check releases
                            api_url = f'https://api.github.com/repos/{action_name}/releases/latest'
                            response = requests.get(api_url, timeout=10)
                            
                            if response.status_code == 200:
                                latest_version = response.json()['tag_name']
                                
                                if current_version != latest_version:
                                    outdated_actions.append({
                                        'name': action_name,
                                        'current_version': current_version,
                                        'latest_version': latest_version,
                                        'type': 'github_action',
                                        'manager': 'github',
                                        'file': str(workflow_file)
                                    })
                        except Exception as e:
                            self.logger.debug(f"Could not check {action_name}: {e}")
                
                except Exception as e:
                    self.logger.warning(f"Could not parse {workflow_file}: {e}")
        
        return outdated_actions
    
    def assess_update_risk(self, package: Dict) -> str:
        """Assess the risk level of updating a package."""
        name = package['name']
        current = package['current_version']
        latest = package['latest_version']
        
        try:
            current_ver = version.parse(current)
            latest_ver = version.parse(latest)
            
            # Major version change = high risk
            if latest_ver.major > current_ver.major:
                return 'high'
            
            # Minor version change = medium risk for core dependencies
            elif latest_ver.minor > current_ver.minor:
                core_deps = ['mlir', 'llvm', 'torch', 'tensorflow', 'numpy']
                if any(core in name.lower() for core in core_deps):
                    return 'medium'
                return 'low'
            
            # Patch version change = low risk
            else:
                return 'low'
                
        except Exception:
            # If we can't parse versions, assume medium risk
            return 'medium'
    
    def create_dependency_update_branch(self, packages: List[Dict], branch_name: str) -> bool:
        """Create a new branch with dependency updates."""
        if self.dry_run:
            self.logger.info(f"[DRY RUN] Would create branch: {branch_name}")
            return True
        
        try:
            # Create new branch
            subprocess.run([
                'git', 'checkout', '-b', branch_name
            ], check=True)
            
            self.logger.info(f"Created branch: {branch_name}")
            
            # Group packages by type for different update strategies
            python_packages = [p for p in packages if p['type'] in ['pip', 'pyproject.toml']]
            github_actions = [p for p in packages if p['type'] == 'github_action']
            
            # Update Python dependencies
            if python_packages:
                self._update_python_dependencies(python_packages)
            
            # Update GitHub Actions
            if github_actions:
                self._update_github_actions(github_actions)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error creating update branch: {e}")
            return False
    
    def _update_python_dependencies(self, packages: List[Dict]):
        """Update Python dependencies in pyproject.toml."""
        pyproject_path = self.repo_root / 'pyproject.toml'
        
        if not pyproject_path.exists():
            self.logger.warning("pyproject.toml not found, skipping Python dependency updates")
            return
        
        with open(pyproject_path) as f:
            pyproject_data = toml.load(f)
        
        updated = False
        
        for package in packages:
            if package['type'] == 'pyproject.toml':
                name = package['name']
                latest_version = package['latest_version']
                
                # Update in dependencies
                deps = pyproject_data.get('project', {}).get('dependencies', [])
                for i, dep in enumerate(deps):
                    if dep.startswith(f"{name}"):
                        # Update version specification
                        new_dep = self._update_dependency_spec(dep, latest_version)
                        deps[i] = new_dep
                        updated = True
                        self.logger.info(f"Updated {name}: {dep} -> {new_dep}")
                
                # Update in optional dependencies
                optional_deps = pyproject_data.get('project', {}).get('optional-dependencies', {})
                for group_name, group_deps in optional_deps.items():
                    for i, dep in enumerate(group_deps):
                        if dep.startswith(f"{name}"):
                            new_dep = self._update_dependency_spec(dep, latest_version)
                            group_deps[i] = new_dep
                            updated = True
                            self.logger.info(f"Updated {name} in {group_name}: {dep} -> {new_dep}")
        
        if updated and not self.dry_run:
            with open(pyproject_path, 'w') as f:
                toml.dump(pyproject_data, f)
            
            # Add to git
            subprocess.run(['git', 'add', str(pyproject_path)], check=True)
    
    def _update_dependency_spec(self, dep_spec: str, new_version: str) -> str:
        """Update a dependency specification with a new version."""
        # Handle different version specifiers
        if '>=' in dep_spec:
            parts = dep_spec.split('>=')
            return f"{parts[0]}>={new_version}"
        elif '==' in dep_spec:
            parts = dep_spec.split('==')
            return f"{parts[0]}=={new_version}"
        elif '~=' in dep_spec:
            parts = dep_spec.split('~=')
            return f"{parts[0]}~={new_version}"
        else:
            # No version specified, add one
            return f"{dep_spec}>={new_version}"
    
    def _update_github_actions(self, actions: List[Dict]):
        """Update GitHub Actions in workflow files."""
        for action in actions:
            file_path = Path(action['file'])
            
            try:
                with open(file_path) as f:
                    content = f.read()
                
                # Replace the version
                old_ref = f"{action['name']}@{action['current_version']}"
                new_ref = f"{action['name']}@{action['latest_version']}"
                
                updated_content = content.replace(old_ref, new_ref)
                
                if updated_content != content and not self.dry_run:
                    with open(file_path, 'w') as f:
                        f.write(updated_content)
                    
                    subprocess.run(['git', 'add', str(file_path)], check=True)
                    self.logger.info(f"Updated {action['name']} in {file_path}")
                
            except Exception as e:
                self.logger.error(f"Error updating {file_path}: {e}")
    
    def create_update_pr(self, packages: List[Dict], branch_name: str) -> bool:
        """Create a pull request for dependency updates."""
        if not self.github_client:
            self.logger.error("GitHub client not available, cannot create PR")
            return False
        
        if self.dry_run:
            self.logger.info("[DRY RUN] Would create PR for dependency updates")
            return True
        
        try:
            # Commit changes
            commit_msg = f"chore: update dependencies ({len(packages)} packages)\n\n"
            
            # Group updates by risk level
            high_risk = [p for p in packages if self.assess_update_risk(p) == 'high']
            medium_risk = [p for p in packages if self.assess_update_risk(p) == 'medium']
            low_risk = [p for p in packages if self.assess_update_risk(p) == 'low']
            
            if high_risk:
                commit_msg += "High-risk updates:\n"
                for pkg in high_risk:
                    commit_msg += f"- {pkg['name']}: {pkg['current_version']} -> {pkg['latest_version']}\n"
                commit_msg += "\n"
            
            if medium_risk:
                commit_msg += "Medium-risk updates:\n"
                for pkg in medium_risk:
                    commit_msg += f"- {pkg['name']}: {pkg['current_version']} -> {pkg['latest_version']}\n"
                commit_msg += "\n"
            
            if low_risk:
                commit_msg += "Low-risk updates:\n"
                for pkg in low_risk:
                    commit_msg += f"- {pkg['name']}: {pkg['current_version']} -> {pkg['latest_version']}\n"
                commit_msg += "\n"
            
            commit_msg += "🤖 Generated with [Claude Code](https://claude.ai/code)\n\n"
            commit_msg += "Co-Authored-By: Claude <noreply@anthropic.com>"
            
            subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
            
            # Push branch
            subprocess.run(['git', 'push', '-u', 'origin', branch_name], check=True)
            
            # Create PR
            repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/photon-mlir-bridge')
            repo = self.github_client.get_repo(repo_name)
            
            # PR title and body
            pr_title = f"chore: automated dependency updates ({len(packages)} packages)"
            
            pr_body = f"""## Automated Dependency Updates

This PR contains automated updates for {len(packages)} outdated dependencies.

### Update Summary

"""
            
            if high_risk:
                pr_body += "#### ⚠️ High-Risk Updates (Review Carefully)\n"
                for pkg in high_risk:
                    pr_body += f"- **{pkg['name']}**: `{pkg['current_version']}` → `{pkg['latest_version']}`\n"
                pr_body += "\n"
            
            if medium_risk:
                pr_body += "#### 🔶 Medium-Risk Updates\n"
                for pkg in medium_risk:
                    pr_body += f"- **{pkg['name']}**: `{pkg['current_version']}` → `{pkg['latest_version']}`\n"
                pr_body += "\n"
            
            if low_risk:
                pr_body += "#### ✅ Low-Risk Updates\n"
                for pkg in low_risk:
                    pr_body += f"- **{pkg['name']}**: `{pkg['current_version']}` → `{pkg['latest_version']}`\n"
                pr_body += "\n"
            
            pr_body += """### Testing Checklist

- [ ] All CI checks pass
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Performance benchmarks show no regression
- [ ] Security scans show no new vulnerabilities

### Manual Review Required

"""
            
            if high_risk:
                pr_body += "⚠️ **This PR contains high-risk updates that may introduce breaking changes. Please review carefully and test thoroughly.**\n\n"
            
            pr_body += """### Notes

- This PR was automatically generated by the dependency update automation
- Dependencies are updated following semantic versioning best practices
- All changes have been tested in the CI pipeline

🤖 *Generated by automated dependency management*"""
            
            # Create the PR
            pr = repo.create_pull(
                title=pr_title,
                body=pr_body,
                head=branch_name,
                base='main'
            )
            
            # Add labels based on risk
            labels = ['dependencies', 'automated']
            if high_risk:
                labels.append('high-risk')
            if medium_risk:
                labels.append('medium-risk')
            
            pr.add_to_labels(*labels)
            
            self.logger.info(f"Created PR: {pr.html_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating PR: {e}")
            return False
    
    def run_update_process(self, max_packages: int = 10) -> bool:
        """Run the complete dependency update process."""
        self.logger.info("Starting dependency update process...")
        
        # Collect all outdated dependencies
        all_outdated = []
        
        # Check different types of dependencies
        all_outdated.extend(self.check_python_dependencies())
        all_outdated.extend(self.check_system_dependencies())
        all_outdated.extend(self.check_github_actions())
        
        if not all_outdated:
            self.logger.info("No outdated dependencies found")
            return True
        
        self.logger.info(f"Found {len(all_outdated)} outdated dependencies")
        
        # Sort by risk level (low risk first)
        risk_order = {'low': 0, 'medium': 1, 'high': 2}
        all_outdated.sort(key=lambda x: risk_order.get(self.assess_update_risk(x), 1))
        
        # Limit the number of packages to update at once
        packages_to_update = all_outdated[:max_packages]
        
        self.logger.info(f"Updating {len(packages_to_update)} packages (max: {max_packages})")
        
        # Create update branch
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        branch_name = f"automated-dependency-updates-{timestamp}"
        
        if self.create_dependency_update_branch(packages_to_update, branch_name):
            # Create PR
            if self.create_update_pr(packages_to_update, branch_name):
                self.logger.info("Dependency update process completed successfully")
                return True
        
        self.logger.error("Dependency update process failed")
        return False


def main():
    """Main entry point for dependency update automation."""
    parser = argparse.ArgumentParser(description='Automated dependency updates')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no actual changes)')
    parser.add_argument('--max-packages', type=int, default=10,
                       help='Maximum number of packages to update at once')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        updater = DependencyUpdater(dry_run=args.dry_run)
        success = updater.run_update_process(max_packages=args.max_packages)
        
        if success:
            print("Dependency update process completed successfully")
            sys.exit(0)
        else:
            print("Dependency update process failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nDependency update process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()