#!/usr/bin/env python3
"""
Quantum-Inspired Task Scheduler Security Scanner

Comprehensive security scanning and vulnerability assessment for the
quantum-inspired task scheduling system. Implements defensive security
measures including input validation, dependency analysis, and runtime monitoring.
"""

import os
import sys
import logging
import json
import hashlib
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import ast
import re

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from python.photon_mlir.quantum_scheduler import CompilationTask, TaskType
from python.photon_mlir.quantum_validation import QuantumValidator, ValidationLevel


class SecurityThreatLevel(Enum):
    """Security threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"  
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    CODE_INJECTION = "code_injection"
    PATH_TRAVERSAL = "path_traversal"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_EXPOSURE = "data_exposure"
    DEPENDENCY_CONFUSION = "dependency_confusion"
    UNSAFE_DESERIALIZATION = "unsafe_deserialization"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    TIMING_ATTACK = "timing_attack"


@dataclass
class SecurityIssue:
    """Represents a security issue found during scanning."""
    vulnerability_type: VulnerabilityType
    severity: SecurityThreatLevel
    file_path: str
    line_number: Optional[int]
    description: str
    recommendation: str
    cwe_id: Optional[str] = None
    cvss_score: Optional[float] = None


@dataclass
class SecurityReport:
    """Comprehensive security scan report."""
    scan_timestamp: str
    total_files_scanned: int
    issues_found: List[SecurityIssue]
    security_score: float
    passed_checks: List[str]
    failed_checks: List[str]
    recommendations: List[str]


class QuantumSecurityScanner:
    """Security scanner for quantum-inspired scheduling system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.python_files = []
        self.config_files = []
        
        # Security patterns to detect
        self.dangerous_patterns = {
            VulnerabilityType.CODE_INJECTION: [
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__\s*\(',
                r'subprocess\.(call|run|Popen)',
                r'os\.system\s*\(',
                r'os\.popen\s*\(',
            ],
            VulnerabilityType.UNSAFE_DESERIALIZATION: [
                r'pickle\.loads?\s*\(',
                r'marshal\.loads?\s*\(',
                r'yaml\.load\s*\(',
                r'json\.loads?\s*\(',  # When used with user input
            ],
            VulnerabilityType.PATH_TRAVERSAL: [
                r'\.\./',
                r'\.\.\\',
                r'os\.path\.join\s*\([^)]*\.\.[^)]*\)',
            ],
            VulnerabilityType.DATA_EXPOSURE: [
                r'password\s*=',
                r'secret\s*=',
                r'token\s*=',
                r'key\s*=',
                r'api_key\s*=',
            ]
        }
        
        # Safe patterns that indicate proper security practices
        self.secure_patterns = [
            r'hashlib\.(sha256|sha512)',
            r'secrets\.token_',
            r'os\.urandom',
            r'input_validation',
            r'sanitize',
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def scan_project(self) -> SecurityReport:
        """Perform comprehensive security scan of the project."""
        self.logger.info("Starting quantum scheduler security scan...")
        
        # Discover files to scan
        self._discover_files()
        
        issues = []
        passed_checks = []
        failed_checks = []
        
        # Static code analysis
        self.logger.info("Performing static code analysis...")
        static_issues = self._perform_static_analysis()
        issues.extend(static_issues)
        
        # Dependency security scan
        self.logger.info("Scanning dependencies...")
        dep_issues = self._scan_dependencies()
        issues.extend(dep_issues)
        
        # Configuration security
        self.logger.info("Checking configuration security...")
        config_issues = self._check_configuration_security()
        issues.extend(config_issues)
        
        # Permission analysis
        self.logger.info("Analyzing file permissions...")
        perm_issues = self._check_file_permissions()
        issues.extend(perm_issues)
        
        # Quantum scheduler specific security
        self.logger.info("Scanning quantum scheduler security...")
        quantum_issues = self._scan_quantum_security()
        issues.extend(quantum_issues)
        
        # Calculate security score
        security_score = self._calculate_security_score(issues)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(issues)
        
        # Categorize checks
        if not any(issue.severity == SecurityThreatLevel.CRITICAL for issue in issues):
            passed_checks.append("No critical vulnerabilities found")
        else:
            failed_checks.append("Critical vulnerabilities detected")
        
        if not any(issue.vulnerability_type == VulnerabilityType.CODE_INJECTION for issue in issues):
            passed_checks.append("No code injection vulnerabilities")
        else:
            failed_checks.append("Code injection vulnerabilities found")
        
        report = SecurityReport(
            scan_timestamp=str(int(time.time())),
            total_files_scanned=len(self.python_files) + len(self.config_files),
            issues_found=issues,
            security_score=security_score,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            recommendations=recommendations
        )
        
        return report
    
    def _discover_files(self):
        """Discover Python and configuration files to scan."""
        python_extensions = {'.py', '.pyx', '.pyi'}
        config_extensions = {'.yml', '.yaml', '.json', '.toml', '.ini', '.cfg'}
        
        for root, dirs, files in os.walk(self.project_root):
            # Skip hidden directories and common ignore patterns
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in {'__pycache__', 'node_modules'}]
            
            for file in files:
                file_path = Path(root) / file
                
                if file_path.suffix in python_extensions:
                    self.python_files.append(file_path)
                elif file_path.suffix in config_extensions:
                    self.config_files.append(file_path)
    
    def _perform_static_analysis(self) -> List[SecurityIssue]:
        """Perform static code analysis to detect security issues."""
        issues = []
        
        for file_path in self.python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for dangerous patterns
                for vuln_type, patterns in self.dangerous_patterns.items():
                    for pattern in patterns:
                        matches = re.finditer(pattern, content, re.IGNORECASE)
                        for match in matches:
                            line_num = content[:match.start()].count('\n') + 1
                            
                            issue = SecurityIssue(
                                vulnerability_type=vuln_type,
                                severity=self._determine_severity(vuln_type, match.group()),
                                file_path=str(file_path.relative_to(self.project_root)),
                                line_number=line_num,
                                description=f"Potentially dangerous pattern detected: {match.group()}",
                                recommendation=self._get_recommendation(vuln_type),
                                cwe_id=self._get_cwe_id(vuln_type)
                            )
                            issues.append(issue)
                
                # AST-based analysis for more sophisticated checks
                try:
                    tree = ast.parse(content)
                    ast_issues = self._analyze_ast(tree, file_path)
                    issues.extend(ast_issues)
                except SyntaxError:
                    self.logger.warning(f"Could not parse {file_path} for AST analysis")
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        return issues
    
    def _analyze_ast(self, tree: ast.AST, file_path: Path) -> List[SecurityIssue]:
        """Analyze AST for security issues."""
        issues = []
        
        class SecurityVisitor(ast.NodeVisitor):
            def __init__(self, issues_list, file_path):
                self.issues = issues_list
                self.file_path = file_path
            
            def visit_Call(self, node):
                # Check for dangerous function calls
                if isinstance(node.func, ast.Name):
                    func_name = node.func.id
                    
                    if func_name in ['eval', 'exec']:
                        self.issues.append(SecurityIssue(
                            vulnerability_type=VulnerabilityType.CODE_INJECTION,
                            severity=SecurityThreatLevel.CRITICAL,
                            file_path=str(self.file_path.relative_to(project_root)),
                            line_number=node.lineno,
                            description=f"Dynamic code execution using {func_name}()",
                            recommendation="Avoid dynamic code execution. Use safer alternatives.",
                            cwe_id="CWE-94"
                        ))
                
                elif isinstance(node.func, ast.Attribute):
                    # Check for dangerous method calls
                    if hasattr(node.func.value, 'id') and node.func.value.id == 'os':
                        if node.func.attr in ['system', 'popen']:
                            self.issues.append(SecurityIssue(
                                vulnerability_type=VulnerabilityType.CODE_INJECTION,
                                severity=SecurityThreatLevel.HIGH,
                                file_path=str(self.file_path.relative_to(project_root)),
                                line_number=node.lineno,
                                description=f"OS command execution using os.{node.func.attr}()",
                                recommendation="Use subprocess with shell=False for safer command execution.",
                                cwe_id="CWE-78"
                            ))
                
                self.generic_visit(node)
            
            def visit_Import(self, node):
                # Check for potentially dangerous imports
                for alias in node.names:
                    if alias.name in ['pickle', 'marshal', 'eval', 'exec']:
                        self.issues.append(SecurityIssue(
                            vulnerability_type=VulnerabilityType.UNSAFE_DESERIALIZATION,
                            severity=SecurityThreatLevel.MEDIUM,
                            file_path=str(self.file_path.relative_to(project_root)),
                            line_number=node.lineno,
                            description=f"Import of potentially unsafe module: {alias.name}",
                            recommendation="Review usage of this module for security implications.",
                            cwe_id="CWE-502"
                        ))
                
                self.generic_visit(node)
        
        visitor = SecurityVisitor(issues, file_path)
        visitor.visit(tree)
        
        return issues
    
    def _scan_dependencies(self) -> List[SecurityIssue]:
        """Scan dependencies for known vulnerabilities."""
        issues = []
        
        # Check for requirements files
        req_files = ['requirements.txt', 'pyproject.toml', 'setup.py', 'Pipfile']
        
        for req_file in req_files:
            file_path = self.project_root / req_file
            if file_path.exists():
                self.logger.info(f"Checking dependencies in {req_file}")
                
                try:
                    # Run safety check if available
                    result = subprocess.run(
                        ['python', '-m', 'pip', 'list', '--format=json'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    if result.returncode == 0:
                        packages = json.loads(result.stdout)
                        dep_issues = self._check_package_vulnerabilities(packages)
                        issues.extend(dep_issues)
                
                except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError) as e:
                    self.logger.warning(f"Could not scan dependencies: {e}")
        
        return issues
    
    def _check_package_vulnerabilities(self, packages: List[Dict[str, str]]) -> List[SecurityIssue]:
        """Check packages against known vulnerability databases."""
        issues = []
        
        # Known vulnerable patterns (simplified - in production use proper CVE databases)
        vulnerable_patterns = {
            'pillow': {'version': '8.3.1', 'cve': 'CVE-2021-34552'},
            'urllib3': {'version': '1.26.5', 'cve': 'CVE-2021-33503'},
            'jinja2': {'version': '2.11.3', 'cve': 'CVE-2020-28493'},
        }
        
        for package in packages:
            package_name = package['name'].lower()
            package_version = package['version']
            
            if package_name in vulnerable_patterns:
                vuln_info = vulnerable_patterns[package_name]
                
                # Simplified version comparison (use proper version parsing in production)
                if package_version <= vuln_info['version']:
                    issues.append(SecurityIssue(
                        vulnerability_type=VulnerabilityType.DEPENDENCY_CONFUSION,
                        severity=SecurityThreatLevel.HIGH,
                        file_path="requirements",
                        line_number=None,
                        description=f"Vulnerable dependency: {package_name} {package_version}",
                        recommendation=f"Update {package_name} to latest version to fix {vuln_info['cve']}",
                        cwe_id="CWE-1104"
                    ))
        
        return issues
    
    def _check_configuration_security(self) -> List[SecurityIssue]:
        """Check configuration files for security issues."""
        issues = []
        
        for config_file in self.config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for hardcoded secrets
                secret_patterns = [
                    r'password\s*[:=]\s*["\'][^"\']+["\']',
                    r'secret\s*[:=]\s*["\'][^"\']+["\']',
                    r'token\s*[:=]\s*["\'][^"\']+["\']',
                    r'api_key\s*[:=]\s*["\'][^"\']+["\']',
                ]
                
                for pattern in secret_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[:match.start()].count('\n') + 1
                        
                        issues.append(SecurityIssue(
                            vulnerability_type=VulnerabilityType.DATA_EXPOSURE,
                            severity=SecurityThreatLevel.HIGH,
                            file_path=str(config_file.relative_to(self.project_root)),
                            line_number=line_num,
                            description="Potential hardcoded secret in configuration",
                            recommendation="Use environment variables or secure secret management",
                            cwe_id="CWE-798"
                        ))
                        
            except Exception as e:
                self.logger.error(f"Error checking {config_file}: {e}")
        
        return issues
    
    def _check_file_permissions(self) -> List[SecurityIssue]:
        """Check file permissions for security issues."""
        issues = []
        
        for file_path in self.python_files + self.config_files:
            try:
                stat = file_path.stat()
                mode = oct(stat.st_mode)[-3:]  # Last 3 digits of octal mode
                
                # Check for world-writable files
                if mode.endswith('6') or mode.endswith('7'):
                    issues.append(SecurityIssue(
                        vulnerability_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                        severity=SecurityThreatLevel.MEDIUM,
                        file_path=str(file_path.relative_to(self.project_root)),
                        line_number=None,
                        description=f"File has world-writable permissions: {mode}",
                        recommendation="Restrict file permissions to prevent unauthorized access",
                        cwe_id="CWE-732"
                    ))
                        
            except Exception as e:
                self.logger.error(f"Error checking permissions for {file_path}: {e}")
        
        return issues
    
    def _scan_quantum_security(self) -> List[SecurityIssue]:
        """Scan quantum scheduler specific security concerns."""
        issues = []
        
        # Test quantum scheduler security
        try:
            from python.photon_mlir.quantum_scheduler import CompilationTask, TaskType
            from python.photon_mlir.quantum_validation import QuantumValidator, ValidationLevel
            
            # Test input validation
            validator = QuantumValidator(ValidationLevel.PARANOID)
            
            # Test with malicious input
            malicious_tasks = [
                CompilationTask(
                    id="<script>alert('xss')</script>",
                    task_type=TaskType.GRAPH_LOWERING,
                    estimated_duration=1.0
                ),
                CompilationTask(
                    id="../../etc/passwd",
                    task_type=TaskType.PHOTONIC_OPTIMIZATION,
                    estimated_duration=1.0
                ),
                CompilationTask(
                    id="normal_task",
                    task_type=TaskType.CODE_GENERATION,
                    estimated_duration=float('inf')  # Resource exhaustion
                )
            ]
            
            validation_result = validator.validate_tasks(malicious_tasks)
            
            if validation_result.is_valid:
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.CODE_INJECTION,
                    severity=SecurityThreatLevel.HIGH,
                    file_path="quantum_scheduler.py",
                    line_number=None,
                    description="Input validation failed to catch malicious task IDs",
                    recommendation="Strengthen input validation in QuantumValidator",
                    cwe_id="CWE-20"
                ))
            
            # Test for resource exhaustion protection
            large_tasks = [
                CompilationTask(
                    id=f"resource_task_{i}",
                    task_type=TaskType.PHOTONIC_OPTIMIZATION,
                    estimated_duration=86400.0,  # 24 hours
                    resource_requirements={
                        "cpu": 1000.0,
                        "memory": 100000.0  # 100GB
                    }
                )
                for i in range(1000)  # 1000 tasks
            ]
            
            validation_result = validator.validate_tasks(large_tasks)
            
            if validation_result.is_valid:
                issues.append(SecurityIssue(
                    vulnerability_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                    severity=SecurityThreatLevel.HIGH,
                    file_path="quantum_scheduler.py",
                    line_number=None,
                    description="No protection against resource exhaustion attacks",
                    recommendation="Implement resource limits and rate limiting",
                    cwe_id="CWE-770"
                ))
                        
        except Exception as e:
            self.logger.error(f"Error testing quantum security: {e}")
            issues.append(SecurityIssue(
                vulnerability_type=VulnerabilityType.DATA_EXPOSURE,
                severity=SecurityThreatLevel.MEDIUM,
                file_path="quantum_scheduler.py",
                line_number=None,
                description="Could not test quantum scheduler security due to import errors",
                recommendation="Ensure quantum scheduler modules are properly secured",
                cwe_id="CWE-754"
            ))
        
        return issues
    
    def _determine_severity(self, vuln_type: VulnerabilityType, pattern_match: str) -> SecurityThreatLevel:
        """Determine severity based on vulnerability type and context."""
        if vuln_type == VulnerabilityType.CODE_INJECTION:
            if 'eval' in pattern_match or 'exec' in pattern_match:
                return SecurityThreatLevel.CRITICAL
            return SecurityThreatLevel.HIGH
        
        elif vuln_type == VulnerabilityType.UNSAFE_DESERIALIZATION:
            if 'pickle' in pattern_match:
                return SecurityThreatLevel.HIGH
            return SecurityThreatLevel.MEDIUM
        
        elif vuln_type == VulnerabilityType.DATA_EXPOSURE:
            if any(word in pattern_match.lower() for word in ['password', 'secret', 'token']):
                return SecurityThreatLevel.HIGH
            return SecurityThreatLevel.MEDIUM
        
        else:
            return SecurityThreatLevel.MEDIUM
    
    def _get_recommendation(self, vuln_type: VulnerabilityType) -> str:
        """Get security recommendation for vulnerability type."""
        recommendations = {
            VulnerabilityType.CODE_INJECTION: "Avoid dynamic code execution. Use safer alternatives like ast.literal_eval() for data parsing.",
            VulnerabilityType.UNSAFE_DESERIALIZATION: "Use safe serialization formats like JSON. If pickle is needed, validate and sanitize input data.",
            VulnerabilityType.PATH_TRAVERSAL: "Validate and sanitize file paths. Use os.path.abspath() and check against allowed directories.",
            VulnerabilityType.DATA_EXPOSURE: "Use environment variables or secure secret management systems. Never hardcode secrets.",
            VulnerabilityType.RESOURCE_EXHAUSTION: "Implement resource limits, timeouts, and rate limiting.",
            VulnerabilityType.DEPENDENCY_CONFUSION: "Keep dependencies updated and use vulnerability scanners.",
            VulnerabilityType.PRIVILEGE_ESCALATION: "Follow principle of least privilege. Restrict file permissions appropriately."
        }
        return recommendations.get(vuln_type, "Review code for security implications.")
    
    def _get_cwe_id(self, vuln_type: VulnerabilityType) -> str:
        """Get CWE ID for vulnerability type."""
        cwe_mapping = {
            VulnerabilityType.CODE_INJECTION: "CWE-94",
            VulnerabilityType.PATH_TRAVERSAL: "CWE-22",
            VulnerabilityType.RESOURCE_EXHAUSTION: "CWE-770",
            VulnerabilityType.DATA_EXPOSURE: "CWE-200",
            VulnerabilityType.UNSAFE_DESERIALIZATION: "CWE-502",
            VulnerabilityType.DEPENDENCY_CONFUSION: "CWE-1104",
            VulnerabilityType.PRIVILEGE_ESCALATION: "CWE-269"
        }
        return cwe_mapping.get(vuln_type, "CWE-Other")
    
    def _calculate_security_score(self, issues: List[SecurityIssue]) -> float:
        """Calculate overall security score (0-100, higher is better)."""
        if not issues:
            return 100.0
        
        # Weighted scoring based on severity
        severity_weights = {
            SecurityThreatLevel.LOW: 1,
            SecurityThreatLevel.MEDIUM: 3,
            SecurityThreatLevel.HIGH: 7,
            SecurityThreatLevel.CRITICAL: 15
        }
        
        total_weight = sum(severity_weights[issue.severity] for issue in issues)
        
        # Score decreases with number and severity of issues
        base_score = 100.0
        penalty = min(total_weight * 2, 90)  # Cap penalty at 90 points
        
        return max(base_score - penalty, 10.0)  # Minimum score of 10
    
    def _generate_recommendations(self, issues: List[SecurityIssue]) -> List[str]:
        """Generate prioritized security recommendations."""
        recommendations = []
        
        # Critical issues first
        critical_issues = [i for i in issues if i.severity == SecurityThreatLevel.CRITICAL]
        if critical_issues:
            recommendations.append("🚨 CRITICAL: Address all critical security vulnerabilities immediately")
        
        # High severity issues
        high_issues = [i for i in issues if i.severity == SecurityThreatLevel.HIGH]
        if high_issues:
            recommendations.append("⚠️ HIGH: Review and fix high-severity security issues")
        
        # Specific recommendations based on vulnerability types
        vuln_types = {issue.vulnerability_type for issue in issues}
        
        if VulnerabilityType.CODE_INJECTION in vuln_types:
            recommendations.append("🛡️ Implement strict input validation and avoid dynamic code execution")
        
        if VulnerabilityType.DATA_EXPOSURE in vuln_types:
            recommendations.append("🔐 Use secure secret management and avoid hardcoded credentials")
        
        if VulnerabilityType.RESOURCE_EXHAUSTION in vuln_types:
            recommendations.append("⏱️ Implement resource limits and timeout mechanisms")
        
        # General recommendations
        recommendations.extend([
            "🔍 Run security scans regularly as part of CI/CD pipeline",
            "📚 Provide security training for development team",
            "🔄 Keep dependencies updated and monitor for vulnerabilities",
            "🧪 Perform regular penetration testing"
        ])
        
        return recommendations[:10]  # Limit to top 10 recommendations
    
    def generate_report(self, report: SecurityReport, output_file: Optional[Path] = None) -> str:
        """Generate human-readable security report."""
        import time
        
        report_lines = [
            "QUANTUM-INSPIRED TASK SCHEDULER - SECURITY SCAN REPORT",
            "=" * 60,
            f"Scan Date: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Files Scanned: {report.total_files_scanned}",
            f"Security Score: {report.security_score:.1f}/100.0",
            "",
            "SECURITY ISSUES FOUND",
            "-" * 30,
        ]
        
        if not report.issues_found:
            report_lines.append("✅ No security issues found!")
        else:
            # Group issues by severity
            by_severity = {}
            for issue in report.issues_found:
                severity = issue.severity.value
                if severity not in by_severity:
                    by_severity[severity] = []
                by_severity[severity].append(issue)
            
            # Report issues by severity (critical first)
            severity_order = ['critical', 'high', 'medium', 'low']
            severity_icons = {'critical': '🚨', 'high': '⚠️', 'medium': '⚡', 'low': 'ℹ️'}
            
            for severity in severity_order:
                if severity in by_severity:
                    issues = by_severity[severity]
                    report_lines.extend([
                        f"{severity_icons[severity]} {severity.upper()} SEVERITY ({len(issues)} issues)",
                        ""
                    ])
                    
                    for issue in issues:
                        report_lines.extend([
                            f"  • {issue.vulnerability_type.value}: {issue.description}",
                            f"    File: {issue.file_path}:{issue.line_number or 'N/A'}",
                            f"    CWE: {issue.cwe_id or 'N/A'}",
                            f"    Fix: {issue.recommendation}",
                            ""
                        ])
        
        # Passed/Failed checks
        report_lines.extend([
            "SECURITY CHECKS",
            "-" * 20,
        ])
        
        for check in report.passed_checks:
            report_lines.append(f"✅ {check}")
        
        for check in report.failed_checks:
            report_lines.append(f"❌ {check}")
        
        # Recommendations
        if report.recommendations:
            report_lines.extend([
                "",
                "RECOMMENDATIONS",
                "-" * 20,
            ])
            
            for i, rec in enumerate(report.recommendations, 1):
                report_lines.append(f"{i}. {rec}")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Security report saved to: {output_file}")
        
        return report_text


def main():
    """Main entry point for security scanner."""
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Quantum Task Scheduler Security Scanner")
    parser.add_argument("--project-root", type=Path, default=Path.cwd(),
                       help="Project root directory to scan")
    parser.add_argument("--output", type=Path,
                       help="Output file for security report")
    parser.add_argument("--json", action="store_true",
                       help="Output report in JSON format")
    parser.add_argument("--fail-on-high", action="store_true",
                       help="Exit with error code if high/critical issues found")
    
    args = parser.parse_args()
    
    # Initialize scanner
    scanner = QuantumSecurityScanner(args.project_root)
    
    # Run scan
    print("🔍 Starting security scan...")
    start_time = time.time()
    
    report = scanner.scan_project()
    
    scan_time = time.time() - start_time
    print(f"✅ Security scan completed in {scan_time:.2f}s")
    
    # Generate output
    if args.json:
        # JSON output
        import dataclasses
        report_dict = dataclasses.asdict(report)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report_dict, f, indent=2)
        else:
            print(json.dumps(report_dict, indent=2))
    else:
        # Human-readable output
        report_text = scanner.generate_report(report, args.output)
        
        if not args.output:
            print("\n" + report_text)
    
    # Exit with appropriate code
    if args.fail_on_high:
        high_severity_count = len([
            issue for issue in report.issues_found 
            if issue.severity in [SecurityThreatLevel.HIGH, SecurityThreatLevel.CRITICAL]
        ])
        
        if high_severity_count > 0:
            print(f"\n❌ Exiting with error: {high_severity_count} high/critical security issues found")
            sys.exit(1)
    
    print(f"\n🛡️ Security Score: {report.security_score:.1f}/100.0")
    sys.exit(0)


if __name__ == "__main__":
    main()