#!/usr/bin/env python3
"""
Automated metrics collection script for photon-mlir-bridge project.

This script collects various project metrics including code quality, 
development velocity, build system performance, and community engagement.
"""

import json
import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List
import requests
from github import Github


class MetricsCollector:
    """Collects and processes project metrics."""
    
    def __init__(self, config_path: str = ".github/project-metrics.json"):
        self.config_path = Path(config_path)
        self.logger = self._setup_logging()
        self.config = self._load_config()
        self.github_client = self._setup_github_client()
        self.metrics_data = {"timestamp": datetime.now(timezone.utc).isoformat()}
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('metrics_collection.log')
            ]
        )
        return logging.getLogger(__name__)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load metrics configuration."""
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {self.config_path}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in config file: {e}")
            sys.exit(1)
    
    def _setup_github_client(self) -> Optional[Github]:
        """Set up GitHub API client."""
        token = os.getenv('GITHUB_TOKEN')
        if not token:
            self.logger.warning("GITHUB_TOKEN not found, GitHub metrics will be limited")
            return None
        return Github(token)
    
    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        self.logger.info("Collecting code quality metrics...")
        metrics = {}
        
        try:
            # Lines of code
            loc_result = subprocess.run([
                'find', '.', '-name', '*.py', '-o', '-name', '*.cpp', '-o', '-name', '*.h'
            ], capture_output=True, text=True)
            
            if loc_result.returncode == 0:
                files = loc_result.stdout.strip().split('\n')
                total_lines = 0
                for file in files:
                    if file and Path(file).exists():
                        try:
                            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                                total_lines += len(f.readlines())
                        except Exception as e:
                            self.logger.warning(f"Could not read {file}: {e}")
                
                metrics['lines_of_code'] = {
                    'value': total_lines,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            # Test coverage (if pytest-cov is available)
            try:
                coverage_result = subprocess.run([
                    'python', '-m', 'pytest', '--cov=photon_mlir', '--cov-report=json'
                ], capture_output=True, text=True, cwd='.')
                
                if coverage_result.returncode == 0 and Path('coverage.json').exists():
                    with open('coverage.json') as f:
                        coverage_data = json.load(f)
                        metrics['test_coverage'] = {
                            'value': coverage_data.get('totals', {}).get('percent_covered', 0),
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
            except Exception as e:
                self.logger.warning(f"Could not collect test coverage: {e}")
            
            # Documentation coverage
            try:
                doc_files = list(Path('docs').rglob('*.rst')) + list(Path('docs').rglob('*.md'))
                py_files = list(Path('python').rglob('*.py')) if Path('python').exists() else []
                
                if py_files:
                    doc_ratio = (len(doc_files) / len(py_files)) * 100
                    metrics['documentation_coverage'] = {
                        'value': min(doc_ratio, 100),  # Cap at 100%
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
            except Exception as e:
                self.logger.warning(f"Could not calculate documentation coverage: {e}")
                
        except Exception as e:
            self.logger.error(f"Error collecting code quality metrics: {e}")
        
        return metrics
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub-based metrics."""
        self.logger.info("Collecting GitHub metrics...")
        metrics = {}
        
        if not self.github_client:
            return metrics
        
        try:
            repo_name = self.config['project']['repository']
            repo = self.github_client.get_repo(repo_name)
            
            # Basic repository metrics
            metrics['github_stars'] = {
                'value': repo.stargazers_count,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            metrics['github_forks'] = {
                'value': repo.forks_count,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            metrics['open_issues'] = {
                'value': repo.open_issues_count,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Contributors count
            contributors = list(repo.get_contributors())
            metrics['contributors'] = {
                'value': len(contributors),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Recent activity metrics
            pulls = list(repo.get_pulls(state='all', sort='created', direction='desc'))
            recent_pulls = [p for p in pulls[:100] if 
                          (datetime.now(timezone.utc) - p.created_at.replace(tzinfo=timezone.utc)).days <= 7]
            
            metrics['pull_requests_per_week'] = {
                'value': len(recent_pulls),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # PR review times
            if recent_pulls:
                review_times = []
                for pr in recent_pulls[:20]:  # Sample recent PRs
                    if pr.merged_at:
                        review_time = (pr.merged_at - pr.created_at).total_seconds() / 3600
                        review_times.append(review_time)
                
                if review_times:
                    metrics['average_pr_review_time'] = {
                        'value': sum(review_times) / len(review_times),
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }
            
            # Commit frequency
            commits = list(repo.get_commits(since=datetime.now().replace(day=datetime.now().day-7)))
            metrics['commits_per_week'] = {
                'value': len(commits),
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting GitHub metrics: {e}")
        
        return metrics
    
    def collect_build_metrics(self) -> Dict[str, Any]:
        """Collect build system metrics."""
        self.logger.info("Collecting build system metrics...")
        metrics = {}
        
        if not self.github_client:
            return metrics
        
        try:
            repo_name = self.config['project']['repository']
            repo = self.github_client.get_repo(repo_name)
            
            # Get recent workflow runs
            workflows = repo.get_workflows()
            
            total_runs = 0
            successful_runs = 0
            total_duration = 0
            duration_count = 0
            
            for workflow in workflows:
                runs = list(workflow.get_runs()[:50])  # Recent 50 runs
                
                for run in runs:
                    total_runs += 1
                    if run.conclusion == 'success':
                        successful_runs += 1
                    
                    if run.created_at and run.updated_at:
                        duration = (run.updated_at - run.created_at).total_seconds() / 60
                        total_duration += duration
                        duration_count += 1
            
            if total_runs > 0:
                metrics['build_success_rate'] = {
                    'value': (successful_runs / total_runs) * 100,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            if duration_count > 0:
                metrics['average_build_time'] = {
                    'value': total_duration / duration_count,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            self.logger.error(f"Error collecting build metrics: {e}")
        
        return metrics
    
    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics from benchmark results."""
        self.logger.info("Collecting performance metrics...")
        metrics = {}
        
        try:
            # Look for recent benchmark results
            benchmark_files = list(Path('.').glob('**/benchmark_results*.json'))
            
            if benchmark_files:
                # Use the most recent benchmark file
                latest_file = max(benchmark_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file) as f:
                    benchmark_data = json.load(f)
                
                # Extract compilation time metrics
                for benchmark in benchmark_data.get('benchmarks', []):
                    name = benchmark.get('name', '').lower()
                    time_ns = benchmark.get('real_time', 0)
                    time_s = time_ns / 1e9 if time_ns else 0
                    
                    if 'small' in name and 'compilation' in name:
                        metrics['compilation_time_small_models'] = {
                            'value': time_s,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    elif 'large' in name and 'compilation' in name:
                        metrics['compilation_time_large_models'] = {
                            'value': time_s,
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                    elif 'inference' in name:
                        metrics['inference_latency'] = {
                            'value': time_s * 1000,  # Convert to milliseconds
                            'timestamp': datetime.now(timezone.utc).isoformat()
                        }
                        
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
        
        return metrics
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security-related metrics."""
        self.logger.info("Collecting security metrics...")
        metrics = {}
        
        try:
            # Check for vulnerability scan results
            vuln_files = [
                'bandit-results.json',
                'safety-results.json',
                'trivy-results.sarif'
            ]
            
            critical_vulns = 0
            high_vulns = 0
            
            for vuln_file in vuln_files:
                if Path(vuln_file).exists():
                    try:
                        with open(vuln_file) as f:
                            data = json.load(f)
                            
                        # Process based on file type
                        if 'bandit' in vuln_file:
                            for result in data.get('results', []):
                                severity = result.get('issue_severity', '').lower()
                                if severity == 'critical':
                                    critical_vulns += 1
                                elif severity == 'high':
                                    high_vulns += 1
                                    
                    except Exception as e:
                        self.logger.warning(f"Could not parse {vuln_file}: {e}")
            
            metrics['critical_vulnerabilities'] = {
                'value': critical_vulns,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            metrics['high_vulnerabilities'] = {
                'value': high_vulns,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            # Check dependency ages
            if Path('requirements.txt').exists():
                try:
                    result = subprocess.run([
                        'pip', 'list', '--outdated', '--format=json'
                    ], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        outdated = json.loads(result.stdout)
                        if outdated:
                            # Calculate average age of outdated packages
                            metrics['dependency_age'] = {
                                'value': len(outdated),  # Number of outdated packages
                                'timestamp': datetime.now(timezone.utc).isoformat()
                            }
                            
                except Exception as e:
                    self.logger.warning(f"Could not check dependency ages: {e}")
            
        except Exception as e:
            self.logger.error(f"Error collecting security metrics: {e}")
        
        return metrics
    
    def update_config_with_metrics(self, collected_metrics: Dict[str, Dict[str, Any]]):
        """Update the configuration file with collected metrics."""
        self.logger.info("Updating configuration with collected metrics...")
        
        try:
            # Update current values and timestamps
            for category, metrics in collected_metrics.items():
                if category in self.config['metrics']:
                    for metric_name, metric_data in metrics.items():
                        if metric_name in self.config['metrics'][category]['measurements']:
                            self.config['metrics'][category]['measurements'][metric_name]['current'] = metric_data['value']
                            self.config['metrics'][category]['measurements'][metric_name]['last_measured'] = metric_data['timestamp']
                            
                            # Update trend (simplified logic)
                            # In a real implementation, you'd compare with historical data
                            current_value = metric_data['value']
                            target = self.config['metrics'][category]['measurements'][metric_name]['target']
                            
                            if isinstance(current_value, (int, float)) and isinstance(target, (int, float)):
                                if abs(current_value - target) / target < 0.1:  # Within 10% of target
                                    trend = "stable"
                                elif current_value > target:
                                    # Determine if higher is better based on metric type
                                    if metric_name in ['test_coverage', 'github_stars', 'github_forks', 'contributors']:
                                        trend = "improving"
                                    else:
                                        trend = "degrading"
                                else:
                                    if metric_name in ['test_coverage', 'github_stars', 'github_forks', 'contributors']:
                                        trend = "degrading"
                                    else:
                                        trend = "improving"
                                
                                self.config['metrics'][category]['measurements'][metric_name]['trend'] = trend
            
            # Write updated config back to file
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
                
            self.logger.info("Configuration updated successfully")
            
        except Exception as e:
            self.logger.error(f"Error updating configuration: {e}")
    
    def generate_report(self, output_format: str = 'json', output_file: Optional[str] = None) -> str:
        """Generate metrics report in specified format."""
        self.logger.info(f"Generating report in {output_format} format...")
        
        report_data = {
            'project': self.config['project'],
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metrics_summary': {}
        }
        
        # Calculate summary statistics
        for category, category_data in self.config['metrics'].items():
            category_summary = {
                'total_metrics': len(category_data['measurements']),
                'metrics_at_target': 0,
                'metrics_at_warning': 0,
                'metrics_at_critical': 0,
                'metrics': {}
            }
            
            for metric_name, metric_data in category_data['measurements'].items():
                current = metric_data.get('current', 0)
                target = metric_data.get('target', 0)
                warning = metric_data.get('warning_threshold', 0)
                critical = metric_data.get('critical_threshold', 0)
                
                # Determine status
                if isinstance(current, (int, float)) and isinstance(target, (int, float)):
                    if abs(current - target) / max(target, 1) < 0.1:
                        status = 'at_target'
                        category_summary['metrics_at_target'] += 1
                    elif (current >= critical and critical > target) or (current <= critical and critical < target):
                        status = 'critical'
                        category_summary['metrics_at_critical'] += 1
                    elif (current >= warning and warning > target) or (current <= warning and warning < target):
                        status = 'warning'
                        category_summary['metrics_at_warning'] += 1
                    else:
                        status = 'ok'
                else:
                    status = 'unknown'
                
                category_summary['metrics'][metric_name] = {
                    'current': current,
                    'target': target,
                    'status': status,
                    'trend': metric_data.get('trend', 'unknown'),
                    'last_measured': metric_data.get('last_measured')
                }
            
            report_data['metrics_summary'][category] = category_summary
        
        # Format output
        if output_format.lower() == 'json':
            report_content = json.dumps(report_data, indent=2)
        elif output_format.lower() == 'html':
            report_content = self._generate_html_report(report_data)
        else:
            report_content = json.dumps(report_data, indent=2)
        
        # Write to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            self.logger.info(f"Report written to {output_file}")
        
        return report_content
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Project Metrics Report - {project_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f4f4f4; padding: 20px; border-radius: 5px; }}
                .metric-category {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-item {{ margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }}
                .status-at-target {{ color: green; }}
                .status-warning {{ color: orange; }}
                .status-critical {{ color: red; }}
                .status-ok {{ color: blue; }}
                .trend-improving {{ color: green; }}
                .trend-degrading {{ color: red; }}
                .trend-stable {{ color: gray; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Project Metrics Report</h1>
                <h2>{project_name}</h2>
                <p>Generated: {timestamp}</p>
            </div>
            
            {metrics_content}
        </body>
        </html>
        """
        
        metrics_content = ""
        for category, data in report_data['metrics_summary'].items():
            metrics_content += f"""
            <div class="metric-category">
                <h3>{category.replace('_', ' ').title()} Metrics</h3>
                <p>Total: {data['total_metrics']}, At Target: {data['metrics_at_target']}, 
                   Warning: {data['metrics_at_warning']}, Critical: {data['metrics_at_critical']}</p>
            """
            
            for metric_name, metric_data in data['metrics'].items():
                status_class = f"status-{metric_data['status'].replace('_', '-')}"
                trend_class = f"trend-{metric_data['trend']}"
                
                metrics_content += f"""
                <div class="metric-item">
                    <strong>{metric_name.replace('_', ' ').title()}</strong><br>
                    Current: <span class="{status_class}">{metric_data['current']}</span> 
                    (Target: {metric_data['target']}) 
                    <span class="{trend_class}">Trend: {metric_data['trend']}</span><br>
                    Last Measured: {metric_data.get('last_measured', 'Never')}
                </div>
                """
            
            metrics_content += "</div>"
        
        return html_template.format(
            project_name=report_data['project']['name'],
            timestamp=report_data['timestamp'],
            metrics_content=metrics_content
        )
    
    def run_collection(self) -> Dict[str, Any]:
        """Run the complete metrics collection process."""
        self.logger.info("Starting metrics collection...")
        
        all_metrics = {}
        
        # Collect different categories of metrics
        collectors = [
            ('code_quality', self.collect_code_quality_metrics),
            ('development_velocity', self.collect_github_metrics),
            ('build_system', self.collect_build_metrics),
            ('performance', self.collect_performance_metrics),
            ('security', self.collect_security_metrics),
            ('community', self.collect_github_metrics),  # GitHub metrics cover community too
        ]
        
        for category, collector_func in collectors:
            try:
                metrics = collector_func()
                if metrics:
                    all_metrics[category] = metrics
                    self.logger.info(f"Collected {len(metrics)} metrics for {category}")
            except Exception as e:
                self.logger.error(f"Error collecting {category} metrics: {e}")
        
        # Update configuration with collected metrics
        self.update_config_with_metrics(all_metrics)
        
        self.logger.info("Metrics collection completed")
        return all_metrics


def main():
    """Main entry point for the metrics collection script."""
    parser = argparse.ArgumentParser(description='Collect project metrics')
    parser.add_argument('--config', default='.github/project-metrics.json',
                       help='Path to metrics configuration file')
    parser.add_argument('--output-format', choices=['json', 'html'], default='json',
                       help='Output format for the report')
    parser.add_argument('--output-file', help='Output file for the report')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        collector = MetricsCollector(args.config)
        collected_metrics = collector.run_collection()
        
        # Generate report
        report = collector.generate_report(
            output_format=args.output_format,
            output_file=args.output_file
        )
        
        if not args.output_file:
            print(report)
        
        print(f"Successfully collected metrics for {len(collected_metrics)} categories")
        
    except Exception as e:
        logging.error(f"Metrics collection failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()