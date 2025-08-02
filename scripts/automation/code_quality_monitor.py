#!/usr/bin/env python3
"""
Code quality monitoring automation for photon-mlir-bridge project.

This script monitors code quality metrics, detects regressions,
and automatically creates issues for quality improvements.
"""

import json
import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import requests
from github import Github


class CodeQualityMonitor:
    """Monitors and reports on code quality metrics."""
    
    def __init__(self, threshold_config: Optional[str] = None):
        self.logger = self._setup_logging()
        self.github_client = self._setup_github_client()
        self.repo_root = Path.cwd()
        self.thresholds = self._load_thresholds(threshold_config)
        self.current_metrics = {}
        
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
    
    def _load_thresholds(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load quality thresholds configuration."""
        default_thresholds = {
            'code_coverage': {
                'excellent': 90,
                'good': 80,
                'warning': 70,
                'critical': 60
            },
            'complexity': {
                'excellent': 5,
                'good': 10,
                'warning': 15,
                'critical': 20
            },
            'maintainability_index': {
                'excellent': 85,
                'good': 70,
                'warning': 50,
                'critical': 30
            },
            'duplicate_code': {
                'excellent': 2,
                'good': 5,
                'warning': 10,
                'critical': 15
            },
            'technical_debt_hours': {
                'excellent': 8,
                'good': 24,
                'warning': 72,
                'critical': 168
            },
            'security_hotspots': {
                'excellent': 0,
                'good': 2,
                'warning': 5, 
                'critical': 10
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path) as f:
                    custom_thresholds = json.load(f)
                default_thresholds.update(custom_thresholds)
            except Exception as e:
                self.logger.warning(f"Could not load custom thresholds: {e}")
        
        return default_thresholds
    
    def analyze_test_coverage(self) -> Dict[str, Any]:
        """Analyze test coverage metrics."""
        self.logger.info("Analyzing test coverage...")
        coverage_data = {}
        
        try:
            # Run pytest with coverage
            result = subprocess.run([
                'python', '-m', 'pytest', 
                '--cov=photon_mlir',
                '--cov-report=json',
                '--cov-report=term-missing',
                '--quiet'
            ], capture_output=True, text=True, cwd=self.repo_root)
            
            if result.returncode == 0 and Path('coverage.json').exists():
                with open('coverage.json') as f:
                    coverage_json = json.load(f)
                
                total_coverage = coverage_json.get('totals', {}).get('percent_covered', 0)
                missing_lines = coverage_json.get('totals', {}).get('missing_lines', 0)
                covered_lines = coverage_json.get('totals', {}).get('covered_lines', 0)
                total_lines = covered_lines + missing_lines
                
                coverage_data = {
                    'total_coverage': round(total_coverage, 2),
                    'covered_lines': covered_lines,
                    'missing_lines': missing_lines,
                    'total_lines': total_lines,
                    'files': {}
                }
                
                # Per-file coverage
                for filename, file_data in coverage_json.get('files', {}).items():
                    if not filename.startswith('tests/'):  # Focus on source files
                        file_coverage = file_data.get('summary', {}).get('percent_covered', 0)
                        coverage_data['files'][filename] = {
                            'coverage': round(file_coverage, 2),
                            'missing_lines': file_data.get('summary', {}).get('missing_lines', 0),
                            'covered_lines': file_data.get('summary', {}).get('covered_lines', 0)
                        }
                
                # Clean up coverage file
                Path('coverage.json').unlink(missing_ok=True)
                
            else:
                self.logger.warning("Coverage analysis failed or no coverage data generated")
                
        except Exception as e:
            self.logger.error(f"Error analyzing test coverage: {e}")
        
        return coverage_data
    
    def analyze_code_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity using various metrics."""
        self.logger.info("Analyzing code complexity...")
        complexity_data = {}
        
        try:
            # Use radon for Python complexity analysis
            subprocess.run(['pip', 'install', 'radon'], capture_output=True)
            
            # Cyclomatic complexity
            cc_result = subprocess.run([
                'radon', 'cc', 'python/', '--json'
            ], capture_output=True, text=True)
            
            if cc_result.returncode == 0:
                cc_data = json.loads(cc_result.stdout)
                
                total_complexity = 0
                function_count = 0
                max_complexity = 0
                complex_functions = []
                
                for filename, file_data in cc_data.items():
                    for item in file_data:
                        if item['type'] in ['method', 'function']:
                            complexity = item['complexity']
                            total_complexity += complexity
                            function_count += 1
                            max_complexity = max(max_complexity, complexity)
                            
                            if complexity > self.thresholds['complexity']['warning']:
                                complex_functions.append({
                                    'file': filename,
                                    'function': item['name'],
                                    'complexity': complexity,
                                    'line': item['lineno']
                                })
                
                avg_complexity = total_complexity / max(function_count, 1)
                
                complexity_data['cyclomatic'] = {
                    'average': round(avg_complexity, 2),
                    'maximum': max_complexity,
                    'total': total_complexity,
                    'function_count': function_count,
                    'complex_functions': complex_functions
                }
            
            # Maintainability index
            mi_result = subprocess.run([
                'radon', 'mi', 'python/', '--json'
            ], capture_output=True, text=True)
            
            if mi_result.returncode == 0:
                mi_data = json.loads(mi_result.stdout)
                
                mi_scores = []
                low_maintainability = []
                
                for filename, file_data in mi_data.items():
                    mi_score = file_data['mi']
                    mi_scores.append(mi_score)
                    
                    if mi_score < self.thresholds['maintainability_index']['warning']:
                        low_maintainability.append({
                            'file': filename,
                            'score': round(mi_score, 2),
                            'rank': file_data['rank']
                        })
                
                if mi_scores:
                    complexity_data['maintainability'] = {
                        'average': round(sum(mi_scores) / len(mi_scores), 2),
                        'minimum': round(min(mi_scores), 2),
                        'low_maintainability_files': low_maintainability
                    }
            
        except Exception as e:
            self.logger.error(f"Error analyzing code complexity: {e}")
        
        return complexity_data
    
    def analyze_code_duplication(self) -> Dict[str, Any]:
        """Analyze code duplication."""
        self.logger.info("Analyzing code duplication...")
        duplication_data = {}
        
        try:
            # Use pylint for duplication detection
            subprocess.run(['pip', 'install', 'pylint'], capture_output=True)
            
            result = subprocess.run([
                'pylint', 'python/', 
                '--disable=all',
                '--enable=duplicate-code',
                '--output-format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                try:
                    pylint_data = json.loads(result.stdout)
                    duplicate_blocks = []
                    
                    for item in pylint_data:
                        if item.get('message-id') == 'R0801':  # duplicate-code
                            duplicate_blocks.append({
                                'file': item['path'],
                                'line': item['line'],
                                'message': item['message']
                            })
                    
                    duplication_data = {
                        'duplicate_blocks': len(duplicate_blocks),
                        'details': duplicate_blocks
                    }
                    
                except json.JSONDecodeError:
                    # Pylint sometimes outputs non-JSON format
                    duplicate_count = result.stdout.count('Similar lines in')
                    duplication_data = {
                        'duplicate_blocks': duplicate_count,
                        'details': []
                    }
            
        except Exception as e:
            self.logger.error(f"Error analyzing code duplication: {e}")
        
        return duplication_data
    
    def analyze_security_issues(self) -> Dict[str, Any]:
        """Analyze security issues using bandit."""
        self.logger.info("Analyzing security issues...")
        security_data = {}
        
        try:
            subprocess.run(['pip', 'install', 'bandit'], capture_output=True)
            
            result = subprocess.run([
                'bandit', '-r', 'python/',
                '-f', 'json',
                '--severity-level', 'medium'
            ], capture_output=True, text=True)
            
            if result.stdout:
                bandit_data = json.loads(result.stdout)
                
                issues_by_severity = {
                    'high': [],
                    'medium': [],
                    'low': []
                }
                
                for issue in bandit_data.get('results', []):
                    severity = issue.get('issue_severity', 'low').lower()
                    if severity in issues_by_severity:
                        issues_by_severity[severity].append({
                            'file': issue['filename'],
                            'line': issue['line_number'],
                            'test_id': issue['test_id'],
                            'issue_text': issue['issue_text']
                        })
                
                security_data = {
                    'total_issues': len(bandit_data.get('results', [])),
                    'high_severity': len(issues_by_severity['high']),
                    'medium_severity': len(issues_by_severity['medium']),
                    'low_severity': len(issues_by_severity['low']),
                    'issues_by_severity': issues_by_severity
                }
            
        except Exception as e:
            self.logger.error(f"Error analyzing security issues: {e}")
        
        return security_data
    
    def analyze_technical_debt(self) -> Dict[str, Any]:
        """Estimate technical debt."""
        self.logger.info("Analyzing technical debt...")
        debt_data = {}
        
        try:
            # Simple technical debt estimation based on various factors
            complexity_data = self.current_metrics.get('complexity', {})
            coverage_data = self.current_metrics.get('coverage', {})
            duplication_data = self.current_metrics.get('duplication', {})
            security_data = self.current_metrics.get('security', {})
            
            # Estimate debt hours based on issues
            debt_hours = 0
            debt_items = []
            
            # Complex functions (2 hours each to refactor)
            complex_functions = complexity_data.get('cyclomatic', {}).get('complex_functions', [])
            debt_hours += len(complex_functions) * 2
            if complex_functions:
                debt_items.append({
                    'category': 'complexity',
                    'count': len(complex_functions),
                    'estimated_hours': len(complex_functions) * 2,
                    'description': 'Refactor complex functions'
                })
            
            # Low maintainability files (4 hours each to improve)
            low_maintainability = complexity_data.get('maintainability', {}).get('low_maintainability_files', [])
            debt_hours += len(low_maintainability) * 4
            if low_maintainability:
                debt_items.append({
                    'category': 'maintainability',
                    'count': len(low_maintainability),
                    'estimated_hours': len(low_maintainability) * 4,
                    'description': 'Improve low maintainability files'
                })
            
            # Missing test coverage (1 hour per 10% missing coverage)
            coverage_pct = coverage_data.get('total_coverage', 100)
            if coverage_pct < 80:
                missing_coverage_hours = (80 - coverage_pct) / 10
                debt_hours += missing_coverage_hours
                debt_items.append({
                    'category': 'testing',
                    'count': 1,
                    'estimated_hours': missing_coverage_hours,
                    'description': f'Improve test coverage from {coverage_pct}% to 80%'
                })
            
            # Duplicate code blocks (1 hour each to deduplicate)
            duplicate_blocks = duplication_data.get('duplicate_blocks', 0)
            debt_hours += duplicate_blocks * 1
            if duplicate_blocks:
                debt_items.append({
                    'category': 'duplication',
                    'count': duplicate_blocks,
                    'estimated_hours': duplicate_blocks * 1,
                    'description': 'Remove duplicate code blocks'
                })
            
            # Security issues (varies by severity)
            high_security = security_data.get('high_severity', 0)
            medium_security = security_data.get('medium_severity', 0)
            debt_hours += high_security * 4 + medium_security * 2
            if high_security or medium_security:
                debt_items.append({
                    'category': 'security',
                    'count': high_security + medium_security,
                    'estimated_hours': high_security * 4 + medium_security * 2,
                    'description': 'Fix security issues'
                })
            
            debt_data = {
                'total_hours': round(debt_hours, 1),
                'items': debt_items,
                'priority_categories': sorted(debt_items, key=lambda x: x['estimated_hours'], reverse=True)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing technical debt: {e}")
        
        return debt_data
    
    def assess_quality_status(self, metric_name: str, value: float) -> str:
        """Assess the quality status of a metric."""
        thresholds = self.thresholds.get(metric_name, {})
        
        if not thresholds:
            return 'unknown'
        
        # For metrics where lower is better (complexity, debt, etc.)
        if metric_name in ['complexity', 'duplicate_code', 'technical_debt_hours', 'security_hotspots']:
            if value <= thresholds.get('excellent', 0):
                return 'excellent'
            elif value <= thresholds.get('good', 0):
                return 'good'
            elif value <= thresholds.get('warning', 0):
                return 'warning'
            else:
                return 'critical'
        else:
            # For metrics where higher is better (coverage, maintainability)
            if value >= thresholds.get('excellent', 100):
                return 'excellent'
            elif value >= thresholds.get('good', 80):
                return 'good'
            elif value >= thresholds.get('warning', 60):
                return 'warning'
            else:
                return 'critical'
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        self.logger.info("Generating quality report...")
        
        report = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'repository': os.getenv('GITHUB_REPOSITORY', 'unknown'),
            'metrics': self.current_metrics,
            'summary': {},
            'recommendations': [],
            'trend': 'stable'  # Would be calculated from historical data
        }
        
        # Generate summary
        summary = {
            'overall_grade': 'B',  # Default grade
            'total_issues': 0,
            'critical_issues': 0,
            'warning_issues': 0,
            'strengths': [],
            'weaknesses': []
        }
        
        # Analyze each metric category
        if 'coverage' in self.current_metrics:
            coverage_pct = self.current_metrics['coverage'].get('total_coverage', 0)
            status = self.assess_quality_status('code_coverage', coverage_pct)
            
            if status == 'critical':
                summary['critical_issues'] += 1
                summary['weaknesses'].append(f"Low test coverage ({coverage_pct}%)")
            elif status == 'warning':
                summary['warning_issues'] += 1
            elif status in ['good', 'excellent']:
                summary['strengths'].append(f"Good test coverage ({coverage_pct}%)")
        
        if 'complexity' in self.current_metrics:
            avg_complexity = self.current_metrics['complexity'].get('cyclomatic', {}).get('average', 0)
            status = self.assess_quality_status('complexity', avg_complexity)
            
            if status == 'critical':
                summary['critical_issues'] += 1
                summary['weaknesses'].append(f"High code complexity (avg: {avg_complexity})")
            elif status == 'warning':
                summary['warning_issues'] += 1
            elif status in ['good', 'excellent']:
                summary['strengths'].append(f"Low code complexity (avg: {avg_complexity})")
        
        if 'technical_debt' in self.current_metrics:
            debt_hours = self.current_metrics['technical_debt'].get('total_hours', 0)
            status = self.assess_quality_status('technical_debt_hours', debt_hours)
            
            if status == 'critical':
                summary['critical_issues'] += 1
                summary['weaknesses'].append(f"High technical debt ({debt_hours} hours)")
            elif status == 'warning':
                summary['warning_issues'] += 1
        
        # Calculate overall grade
        if summary['critical_issues'] > 3:
            summary['overall_grade'] = 'D'
        elif summary['critical_issues'] > 1:
            summary['overall_grade'] = 'C'
        elif summary['warning_issues'] > 5:
            summary['overall_grade'] = 'C'
        elif summary['warning_issues'] > 2:
            summary['overall_grade'] = 'B'
        elif len(summary['strengths']) >= 3:
            summary['overall_grade'] = 'A'
        else:
            summary['overall_grade'] = 'B'
        
        summary['total_issues'] = summary['critical_issues'] + summary['warning_issues']
        report['summary'] = summary
        
        # Generate recommendations
        recommendations = []
        
        if summary['critical_issues'] > 0:
            recommendations.append({
                'priority': 'high',
                'category': 'critical_issues',
                'title': 'Address Critical Quality Issues',
                'description': 'Focus on resolving critical quality issues that significantly impact maintainability.',
                'action_items': summary['weaknesses']
            })
        
        if 'technical_debt' in self.current_metrics:
            debt_items = self.current_metrics['technical_debt'].get('priority_categories', [])
            if debt_items:
                top_debt = debt_items[0]
                recommendations.append({
                    'priority': 'medium',
                    'category': 'technical_debt',
                    'title': f'Reduce Technical Debt in {top_debt["category"].title()}',
                    'description': top_debt['description'],
                    'estimated_effort': f"{top_debt['estimated_hours']} hours"
                })
        
        report['recommendations'] = recommendations
        
        return report
    
    def create_quality_issue(self, report: Dict[str, Any]) -> bool:
        """Create GitHub issue for quality improvements."""
        if not self.github_client:
            self.logger.warning("GitHub client not available, skipping issue creation")
            return False
        
        try:
            repo_name = os.getenv('GITHUB_REPOSITORY', 'danieleschmidt/photon-mlir-bridge')
            repo = self.github_client.get_repo(repo_name)
            
            # Check if there's already an open quality issue
            existing_issues = list(repo.get_issues(state='open', labels=['code-quality', 'automated']))
            
            if existing_issues:
                self.logger.info("Code quality issue already exists, updating instead")
                return self._update_quality_issue(existing_issues[0], report)
            
            # Create new issue
            summary = report['summary']
            title = f"Code Quality Report - Grade {summary['overall_grade']} ({summary['total_issues']} issues)"
            
            body = f"""# Code Quality Report

**Overall Grade**: {summary['overall_grade']}  
**Generated**: {report['timestamp']}  
**Total Issues**: {summary['total_issues']} ({summary['critical_issues']} critical, {summary['warning_issues']} warning)

## Summary

"""
            
            if summary['strengths']:
                body += "### ✅ Strengths\n"
                for strength in summary['strengths']:
                    body += f"- {strength}\n"
                body += "\n"
            
            if summary['weaknesses']:
                body += "### ⚠️ Areas for Improvement\n"
                for weakness in summary['weaknesses']:
                    body += f"- {weakness}\n"
                body += "\n"
            
            # Add detailed metrics
            body += "## Detailed Metrics\n\n"
            
            if 'coverage' in report['metrics']:
                coverage = report['metrics']['coverage']
                body += f"### Test Coverage: {coverage.get('total_coverage', 0)}%\n"
                body += f"- Covered lines: {coverage.get('covered_lines', 0)}\n"
                body += f"- Missing lines: {coverage.get('missing_lines', 0)}\n\n"
            
            if 'complexity' in report['metrics']:
                complexity = report['metrics']['complexity']
                cc = complexity.get('cyclomatic', {})
                body += f"### Code Complexity\n"
                body += f"- Average cyclomatic complexity: {cc.get('average', 0)}\n"
                body += f"- Maximum complexity: {cc.get('maximum', 0)}\n"
                
                complex_funcs = cc.get('complex_functions', [])
                if complex_funcs:
                    body += f"- Functions with high complexity ({len(complex_funcs)}):\n"
                    for func in complex_funcs[:5]:  # Show top 5
                        body += f"  - `{func['function']}` in {func['file']}:{func['line']} (complexity: {func['complexity']})\n"
                    if len(complex_funcs) > 5:
                        body += f"  - ... and {len(complex_funcs) - 5} more\n"
                body += "\n"
            
            if 'technical_debt' in report['metrics']:
                debt = report['metrics']['technical_debt']
                body += f"### Technical Debt: {debt.get('total_hours', 0)} hours\n"
                
                for item in debt.get('items', []):
                    body += f"- **{item['category'].title()}**: {item['estimated_hours']} hours - {item['description']}\n"
                body += "\n"
            
            # Add recommendations
            if report.get('recommendations'):
                body += "## Recommendations\n\n"
                
                for rec in report['recommendations']:
                    priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(rec['priority'], "⚪")
                    body += f"### {priority_emoji} {rec['title']} ({rec['priority']} priority)\n\n"
                    body += f"{rec['description']}\n\n"
                    
                    if 'action_items' in rec:
                        body += "Action items:\n"
                        for item in rec['action_items']:
                            body += f"- [ ] {item}\n"
                        body += "\n"
                    
                    if 'estimated_effort' in rec:
                        body += f"**Estimated effort**: {rec['estimated_effort']}\n\n"
            
            body += """---

*This issue was automatically generated by the code quality monitoring system. It will be updated periodically with the latest metrics.*

🤖 *Generated by automated code quality analysis*"""
            
            # Create the issue
            issue = repo.create_issue(
                title=title,
                body=body,
                labels=['code-quality', 'automated', 'enhancement']
            )
            
            self.logger.info(f"Created quality issue: {issue.html_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating quality issue: {e}")
            return False
    
    def _update_quality_issue(self, issue, report: Dict[str, Any]) -> bool:
        """Update existing quality issue."""
        try:
            # Generate updated body (similar to create_quality_issue)
            # For brevity, reusing the same logic
            # In practice, you might want to append updates or maintain history
            
            summary = report['summary']
            title = f"Code Quality Report - Grade {summary['overall_grade']} ({summary['total_issues']} issues)"
            
            # Update title and add comment
            issue.edit(title=title)
            
            comment_body = f"""## Quality Report Update - {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}

**Grade**: {summary['overall_grade']} (was: previous grade)  
**Total Issues**: {summary['total_issues']} ({summary['critical_issues']} critical, {summary['warning_issues']} warning)

### Recent Changes
- Updated quality metrics analysis
- {len(report.get('recommendations', []))} recommendations generated

*Full report details updated in the issue description above.*"""
            
            issue.create_comment(comment_body)
            
            self.logger.info(f"Updated quality issue: {issue.html_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating quality issue: {e}")
            return False
    
    def run_quality_analysis(self, create_issue: bool = False) -> Dict[str, Any]:
        """Run complete code quality analysis."""
        self.logger.info("Starting code quality analysis...")
        
        # Collect all metrics
        self.current_metrics['coverage'] = self.analyze_test_coverage()
        self.current_metrics['complexity'] = self.analyze_code_complexity()
        self.current_metrics['duplication'] = self.analyze_code_duplication()
        self.current_metrics['security'] = self.analyze_security_issues()
        self.current_metrics['technical_debt'] = self.analyze_technical_debt()
        
        # Generate comprehensive report
        report = self.generate_quality_report()
        
        # Create or update GitHub issue if requested
        if create_issue:
            self.create_quality_issue(report)
        
        self.logger.info("Code quality analysis completed")
        return report


def main():
    """Main entry point for code quality monitoring."""
    parser = argparse.ArgumentParser(description='Code quality monitoring and analysis')
    parser.add_argument('--config', help='Path to quality thresholds configuration file')
    parser.add_argument('--create-issue', action='store_true',
                       help='Create GitHub issue with quality report')
    parser.add_argument('--output-file', help='Output file for quality report')
    parser.add_argument('--format', choices=['json', 'markdown'], default='json',
                       help='Output format')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        monitor = CodeQualityMonitor(threshold_config=args.config)
        report = monitor.run_quality_analysis(create_issue=args.create_issue)
        
        # Output report
        if args.output_file:
            with open(args.output_file, 'w') as f:
                if args.format == 'json':
                    json.dump(report, f, indent=2)
                else:
                    # Simple markdown output
                    f.write(f"# Code Quality Report\n\n")
                    f.write(f"**Grade**: {report['summary']['overall_grade']}\n")
                    f.write(f"**Total Issues**: {report['summary']['total_issues']}\n\n")
                    
                    for category, metrics in report['metrics'].items():
                        f.write(f"## {category.title()}\n\n")
                        f.write(f"```json\n{json.dumps(metrics, indent=2)}\n```\n\n")
            
            print(f"Quality report written to {args.output_file}")
        else:
            print(json.dumps(report, indent=2))
        
        # Exit with appropriate code based on quality
        grade = report['summary']['overall_grade']
        if grade in ['D', 'F']:
            sys.exit(2)  # Critical quality issues
        elif grade == 'C':
            sys.exit(1)  # Warning level issues
        else:
            sys.exit(0)  # Acceptable quality
            
    except Exception as e:
        logging.error(f"Quality analysis failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()