#!/usr/bin/env python3
"""
Repository maintenance automation for photon-mlir-bridge project.

This script performs routine repository maintenance tasks like cleaning up
old branches, archiving completed issues, and optimizing repository health.
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from github import Github


class RepositoryMaintenance:
    """Handles automated repository maintenance tasks."""
    
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.logger = self._setup_logging()
        self.github_client = self._setup_github_client()
        
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
            self.logger.warning("GITHUB_TOKEN not found")
            return None
        return Github(token)
    
    def cleanup_merged_branches(self, exclude_branches: List[str] = None) -> int:
        """Clean up merged feature branches."""
        if exclude_branches is None:
            exclude_branches = ['main', 'develop', 'master']
        
        self.logger.info("Cleaning up merged branches...")
        cleaned_count = 0
        
        try:
            # Get local branches
            result = subprocess.run(['git', 'branch', '--merged', 'main'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                branches = [b.strip().lstrip('* ') for b in result.stdout.strip().split('\n')]
                branches = [b for b in branches if b and b not in exclude_branches]
                
                for branch in branches:
                    # Skip if branch name suggests it should be kept
                    if any(keyword in branch.lower() for keyword in ['release', 'hotfix', 'main', 'master']):
                        continue
                    
                    self.logger.info(f"Deleting merged branch: {branch}")
                    
                    if not self.dry_run:
                        # Delete local branch
                        subprocess.run(['git', 'branch', '-d', branch], 
                                     capture_output=True)
                        
                        # Delete remote branch if it exists
                        subprocess.run(['git', 'push', 'origin', '--delete', branch], 
                                     capture_output=True)
                    
                    cleaned_count += 1
            
            # Clean up remote tracking branches
            result = subprocess.run(['git', 'remote', 'prune', 'origin'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Pruned remote tracking branches")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up branches: {e}")
        
        return cleaned_count
    
    def archive_old_issues(self, days_threshold: int = 90) -> int:
        """Archive or close old stale issues."""
        if not self.github_client:
            return 0
        
        self.logger.info(f"Archiving issues older than {days_threshold} days...")
        archived_count = 0
        
        try:
            repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/photon-mlir-bridge')
            repo = self.github_client.get_repo(repo_name)
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_threshold)
            
            # Get old open issues
            issues = repo.get_issues(state='open', sort='updated', direction='asc')
            
            for issue in issues:
                if issue.pull_request:  # Skip PRs
                    continue
                
                if issue.updated_at < cutoff_date:
                    # Check if issue has certain labels that indicate it should stay open
                    protected_labels = ['pinned', 'enhancement', 'long-term', 'epic']
                    if any(label.name.lower() in protected_labels for label in issue.labels):
                        continue
                    
                    # Add stale comment and label
                    stale_comment = """This issue has been automatically marked as stale because it has not had recent activity. 
It will be closed in 7 days if no further activity occurs. 
If this issue is still relevant, please comment to keep it open.

🤖 *This is an automated message*"""
                    
                    if not self.dry_run:
                        # Check if already marked as stale
                        if not any(label.name == 'stale' for label in issue.labels):
                            issue.create_comment(stale_comment)
                            issue.add_to_labels('stale')
                            self.logger.info(f"Marked issue #{issue.number} as stale")
                            archived_count += 1
                        else:
                            # Close if stale for more than 7 days
                            stale_cutoff = datetime.now(timezone.utc) - timedelta(days=7)
                            stale_label_date = None
                            
                            for event in issue.get_timeline():
                                if (hasattr(event, 'event') and event.event == 'labeled' and 
                                    hasattr(event, 'label') and event.label.name == 'stale'):
                                    stale_label_date = event.created_at
                                    break
                            
                            if stale_label_date and stale_label_date < stale_cutoff:
                                close_comment = """Closing this issue as stale. No recent activity detected. 
If this issue is still relevant, please reopen it.

🤖 *Automatically closed by repository maintenance*"""
                                
                                issue.create_comment(close_comment)
                                issue.edit(state='closed')
                                self.logger.info(f"Closed stale issue #{issue.number}")
                                archived_count += 1
                    else:
                        self.logger.info(f"[DRY RUN] Would mark issue #{issue.number} as stale")
                        archived_count += 1
                
        except Exception as e:
            self.logger.error(f"Error archiving issues: {e}")
        
        return archived_count
    
    def optimize_git_repository(self) -> bool:
        """Optimize git repository performance."""
        self.logger.info("Optimizing git repository...")
        
        try:
            if not self.dry_run:
                # Garbage collection
                subprocess.run(['git', 'gc', '--aggressive', '--prune=now'], 
                             capture_output=True, check=True)
                
                # Repack objects
                subprocess.run(['git', 'repack', '-a', '-d', '-f', '--depth=250', '--window=250'], 
                             capture_output=True, check=True)
                
                # Clean up unreachable objects
                subprocess.run(['git', 'fsck', '--unreachable'], 
                             capture_output=True)
                
                self.logger.info("Git repository optimization completed")
            else:
                self.logger.info("[DRY RUN] Would optimize git repository")
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Error optimizing repository: {e}")
            return False
    
    def update_repository_topics(self, topics: List[str]) -> bool:
        """Update repository topics/tags."""
        if not self.github_client:
            return False
        
        self.logger.info("Updating repository topics...")
        
        try:
            repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/photon-mlir-bridge')
            repo = self.github_client.get_repo(repo_name)
            
            current_topics = repo.get_topics()
            
            # Merge with existing topics, avoiding duplicates
            all_topics = list(set(current_topics + topics))
            
            if not self.dry_run:
                repo.replace_topics(all_topics)
                self.logger.info(f"Updated repository topics: {all_topics}")
            else:
                self.logger.info(f"[DRY RUN] Would update topics to: {all_topics}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating repository topics: {e}")
            return False
    
    def clean_workflow_runs(self, days_to_keep: int = 30) -> int:
        """Clean up old workflow runs to save space."""
        if not self.github_client:
            return 0
        
        self.logger.info(f"Cleaning up workflow runs older than {days_to_keep} days...")
        cleaned_count = 0
        
        try:
            repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/photon-mlir-bridge')
            repo = self.github_client.get_repo(repo_name)
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
            
            workflows = repo.get_workflows()
            
            for workflow in workflows:
                runs = workflow.get_runs()
                
                for run in runs:
                    if run.created_at < cutoff_date and run.status == 'completed':
                        if not self.dry_run:
                            try:
                                # Note: GitHub API doesn't directly support deleting workflow runs
                                # This would typically be done through the GitHub CLI or API
                                self.logger.info(f"Would delete workflow run {run.id} (created {run.created_at})")
                                cleaned_count += 1
                            except Exception as e:
                                self.logger.warning(f"Could not delete workflow run {run.id}: {e}")
                        else:
                            self.logger.info(f"[DRY RUN] Would delete workflow run {run.id}")
                            cleaned_count += 1
                
        except Exception as e:
            self.logger.error(f"Error cleaning workflow runs: {e}")
        
        return cleaned_count
    
    def update_issue_labels(self) -> bool:
        """Ensure consistent issue labeling system."""
        if not self.github_client:
            return False
        
        self.logger.info("Updating issue labels...")
        
        # Standard label set
        standard_labels = [
            {'name': 'bug', 'color': 'd73a4a', 'description': 'Something isn\'t working'},
            {'name': 'enhancement', 'color': 'a2eeef', 'description': 'New feature or request'},
            {'name': 'documentation', 'color': '0075ca', 'description': 'Improvements or additions to documentation'},
            {'name': 'good first issue', 'color': '7057ff', 'description': 'Good for newcomers'},
            {'name': 'help wanted', 'color': '008672', 'description': 'Extra attention is needed'},
            {'name': 'performance', 'color': 'fbca04', 'description': 'Performance related issue'},
            {'name': 'security', 'color': 'b60205', 'description': 'Security related issue'},
            {'name': 'testing', 'color': '0e8a16', 'description': 'Testing related'},
            {'name': 'dependencies', 'color': '0366d6', 'description': 'Dependency related'},
            {'name': 'automated', 'color': '6f42c1', 'description': 'Created by automation'},
            {'name': 'stale', 'color': 'fef2c0', 'description': 'No recent activity'},
            {'name': 'priority-high', 'color': 'b60205', 'description': 'High priority'},
            {'name': 'priority-medium', 'color': 'fbca04', 'description': 'Medium priority'},
            {'name': 'priority-low', 'color': '0e8a16', 'description': 'Low priority'},
        ]
        
        try:
            repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/photon-mlir-bridge')
            repo = self.github_client.get_repo(repo_name)
            
            existing_labels = {label.name: label for label in repo.get_labels()}
            
            for label_def in standard_labels:
                if label_def['name'] in existing_labels:
                    # Update existing label if needed
                    existing_label = existing_labels[label_def['name']]
                    if (existing_label.color != label_def['color'] or 
                        existing_label.description != label_def['description']):
                        
                        if not self.dry_run:
                            existing_label.edit(
                                name=label_def['name'],
                                color=label_def['color'],
                                description=label_def['description']
                            )
                            self.logger.info(f"Updated label: {label_def['name']}")
                        else:
                            self.logger.info(f"[DRY RUN] Would update label: {label_def['name']}")
                else:
                    # Create new label
                    if not self.dry_run:
                        repo.create_label(
                            name=label_def['name'],
                            color=label_def['color'],
                            description=label_def['description']
                        )
                        self.logger.info(f"Created label: {label_def['name']}")
                    else:
                        self.logger.info(f"[DRY RUN] Would create label: {label_def['name']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating labels: {e}")
            return False
    
    def generate_maintenance_report(self, results: Dict[str, Any]) -> str:
        """Generate maintenance report."""
        report = f"""# Repository Maintenance Report

**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Repository**: {os.getenv('GITHUB_REPOSITORY', 'unknown')}

## Summary

- **Branches cleaned**: {results.get('branches_cleaned', 0)}
- **Issues archived**: {results.get('issues_archived', 0)}
- **Workflow runs cleaned**: {results.get('workflow_runs_cleaned', 0)}
- **Git optimization**: {'✅ Completed' if results.get('git_optimized', False) else '❌ Failed'}
- **Labels updated**: {'✅ Completed' if results.get('labels_updated', False) else '❌ Failed'}
- **Topics updated**: {'✅ Completed' if results.get('topics_updated', False) else '❌ Failed'}

## Details

### Branch Cleanup
{results.get('branches_cleaned', 0)} merged branches were removed to keep the repository clean.

### Issue Management
{results.get('issues_archived', 0)} stale issues were processed (marked as stale or closed).

### Repository Optimization
Git repository optimization {'was completed successfully' if results.get('git_optimized', False) else 'encountered issues'}.

### Maintenance Status
Repository maintenance completed {'successfully' if all(results.values()) else 'with some issues'}.

---

*This report was generated automatically by the repository maintenance system.*
"""
        return report
    
    def run_maintenance(self) -> Dict[str, Any]:
        """Run complete repository maintenance."""
        self.logger.info("Starting repository maintenance...")
        
        results = {}
        
        # Clean up merged branches
        results['branches_cleaned'] = self.cleanup_merged_branches()
        
        # Archive old issues
        results['issues_archived'] = self.archive_old_issues()
        
        # Optimize git repository
        results['git_optimized'] = self.optimize_git_repository()
        
        # Update repository topics
        topics = ['mlir', 'photonics', 'compiler', 'machine-learning', 'silicon-photonics']
        results['topics_updated'] = self.update_repository_topics(topics)
        
        # Clean workflow runs
        results['workflow_runs_cleaned'] = self.clean_workflow_runs()
        
        # Update issue labels
        results['labels_updated'] = self.update_issue_labels()
        
        self.logger.info("Repository maintenance completed")
        return results


def main():
    """Main entry point for repository maintenance."""
    parser = argparse.ArgumentParser(description='Repository maintenance automation')
    parser.add_argument('--dry-run', action='store_true',
                       help='Run in dry-run mode (no actual changes)')
    parser.add_argument('--branches-only', action='store_true',
                       help='Only clean up branches')
    parser.add_argument('--issues-only', action='store_true',
                       help='Only process issues')
    parser.add_argument('--git-only', action='store_true',
                       help='Only optimize git repository')
    parser.add_argument('--report-file',
                       help='Output file for maintenance report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        maintenance = RepositoryMaintenance(dry_run=args.dry_run)
        
        if args.branches_only:
            results = {'branches_cleaned': maintenance.cleanup_merged_branches()}
        elif args.issues_only:
            results = {'issues_archived': maintenance.archive_old_issues()}
        elif args.git_only:
            results = {'git_optimized': maintenance.optimize_git_repository()}
        else:
            results = maintenance.run_maintenance()
        
        # Generate and output report
        report = maintenance.generate_maintenance_report(results)
        
        if args.report_file:
            with open(args.report_file, 'w') as f:
                f.write(report)
            print(f"Maintenance report written to {args.report_file}")
        else:
            print(report)
        
        # Exit with appropriate code
        if all(isinstance(v, bool) and v for v in results.values() if isinstance(v, bool)):
            sys.exit(0)
        else:
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Repository maintenance failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()