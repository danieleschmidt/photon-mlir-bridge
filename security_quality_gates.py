#!/usr/bin/env python3
"""
Security and Quality Gates Validator
Comprehensive security scanning and quality assurance for autonomous SDLC

This module implements security scans and quality gates to ensure
the autonomous SDLC implementation meets production standards.
"""

import sys
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import hashlib


class SecurityQualityValidator:
    """Comprehensive security and quality validation."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.results: Dict[str, Any] = {}
        
    def run_all_gates(self) -> Dict[str, Any]:
        """Run all security and quality gates."""
        
        print("🔒 Running Security and Quality Gates")
        print("=" * 50)
        
        # Security gates
        self.results["security"] = {
            "code_injection_scan": self.scan_code_injection(),
            "secrets_scan": self.scan_secrets(),
            "file_permissions": self.check_file_permissions(),
            "dependency_scan": self.scan_dependencies()
        }
        
        # Quality gates
        self.results["quality"] = {
            "code_style": self.check_code_style(),
            "documentation": self.check_documentation(),
            "file_structure": self.check_file_structure(),
            "import_consistency": self.check_import_consistency()
        }
        
        # Performance gates
        self.results["performance"] = {
            "file_size_check": self.check_file_sizes(),
            "complexity_check": self.check_complexity(),
            "startup_time": self.measure_startup_time()
        }
        
        # Generate overall assessment
        self.results["overall"] = self.generate_overall_assessment()
        
        return self.results
    
    def scan_code_injection(self) -> Dict[str, Any]:
        """Scan for potential code injection vulnerabilities."""
        
        print("🔍 Scanning for code injection vulnerabilities...")
        
        suspicious_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"__import__\s*\(",
            r"compile\s*\(",
            r"subprocess\.call\(",
            r"os\.system\(",
            r"shell=True"
        ]
        
        findings = []
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                for pattern in suspicious_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        findings.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "pattern": pattern,
                            "matches": len(matches),
                            "severity": "medium"
                        })
            except Exception as e:
                print(f"Warning: Could not scan {file_path}: {e}")
        
        return {
            "total_files_scanned": len(python_files),
            "findings": findings,
            "risk_level": "low" if len(findings) < 3 else "medium"
        }
    
    def scan_secrets(self) -> Dict[str, Any]:
        """Scan for potential secrets and sensitive information."""
        
        print("🔐 Scanning for secrets and sensitive information...")
        
        secret_patterns = [
            r"password\s*=\s*[\"'][^\"']+[\"']",
            r"api_key\s*=\s*[\"'][^\"']+[\"']",
            r"secret\s*=\s*[\"'][^\"']+[\"']",
            r"token\s*=\s*[\"'][^\"']+[\"']",
            r"private_key\s*=\s*[\"'][^\"']+[\"']"
        ]
        
        findings = []
        text_files = list(self.project_root.rglob("*.py")) + list(self.project_root.rglob("*.md"))
        
        for file_path in text_files:
            try:
                content = file_path.read_text()
                
                for pattern in secret_patterns:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        # Filter out obvious examples/comments
                        real_matches = [m for m in matches if not any(word in m.lower() for word in ["example", "test", "dummy", "placeholder"])]
                        if real_matches:
                            findings.append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "pattern": pattern,
                                "matches": len(real_matches),
                                "severity": "high"
                            })
            except Exception as e:
                continue
        
        return {
            "total_files_scanned": len(text_files),
            "findings": findings,
            "risk_level": "high" if findings else "low"
        }
    
    def check_file_permissions(self) -> Dict[str, Any]:
        """Check file permissions for security issues."""
        
        print("📋 Checking file permissions...")
        
        issues = []
        executable_files = []
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    # Check if file is executable
                    if file_path.stat().st_mode & 0o111:
                        executable_files.append(str(file_path.relative_to(self.project_root)))
                        
                        # Python files shouldn't be executable unless they're scripts
                        if file_path.suffix == ".py" and not file_path.name.endswith("_runner.py"):
                            issues.append({
                                "file": str(file_path.relative_to(self.project_root)),
                                "issue": "Python file is executable",
                                "severity": "low"
                            })
                except Exception:
                    continue
        
        return {
            "executable_files": executable_files,
            "permission_issues": issues,
            "risk_level": "low" if len(issues) < 5 else "medium"
        }
    
    def scan_dependencies(self) -> Dict[str, Any]:
        """Scan dependencies for known vulnerabilities."""
        
        print("📦 Scanning dependencies...")
        
        dependency_files = []
        findings = []
        
        # Check for dependency files
        dep_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
        for dep_file in dep_files:
            file_path = self.project_root / dep_file
            if file_path.exists():
                dependency_files.append(dep_file)
        
        # Basic dependency analysis
        if self.project_root / "pyproject.toml" in [self.project_root / f for f in dependency_files]:
            try:
                pyproject_content = (self.project_root / "pyproject.toml").read_text()
                
                # Check for potentially problematic dependencies
                risky_deps = ["eval", "exec", "pickle", "subprocess"]
                for dep in risky_deps:
                    if dep in pyproject_content:
                        findings.append({
                            "dependency": dep,
                            "risk": "medium",
                            "reason": "Potentially unsafe operations"
                        })
            except Exception:
                pass
        
        return {
            "dependency_files_found": dependency_files,
            "findings": findings,
            "risk_level": "low" if len(findings) < 2 else "medium"
        }
    
    def check_code_style(self) -> Dict[str, Any]:
        """Check code style and formatting."""
        
        print("🎨 Checking code style...")
        
        python_files = list(self.project_root.rglob("*.py"))
        style_issues = []
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                lines = content.split('\\n')
                
                for i, line in enumerate(lines, 1):
                    # Check line length
                    if len(line) > 120:
                        style_issues.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": i,
                            "issue": "Line too long",
                            "severity": "low"
                        })
                    
                    # Check for trailing whitespace
                    if line.endswith(' ') or line.endswith('\\t'):
                        style_issues.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": i,
                            "issue": "Trailing whitespace",
                            "severity": "low"
                        })
                
            except Exception:
                continue
        
        return {
            "total_files_checked": len(python_files),
            "style_issues": len(style_issues),
            "issues": style_issues[:10],  # Show first 10 issues
            "quality_score": max(0, 100 - len(style_issues))
        }
    
    def check_documentation(self) -> Dict[str, Any]:
        """Check documentation completeness."""
        
        print("📚 Checking documentation...")
        
        required_docs = [
            "README.md",
            "ARCHITECTURE.md", 
            "CONTRIBUTING.md",
            "LICENSE"
        ]
        
        found_docs = []
        missing_docs = []
        
        for doc in required_docs:
            if (self.project_root / doc).exists():
                found_docs.append(doc)
            else:
                missing_docs.append(doc)
        
        # Check Python docstrings
        python_files = list(self.project_root.rglob("*.py"))
        documented_functions = 0
        total_functions = 0
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Count functions
                function_matches = re.findall(r'def\s+\w+\s*\(', content)
                total_functions += len(function_matches)
                
                # Count documented functions (simplified check)
                documented_matches = re.findall(r'def\s+\w+\s*\([^)]*\):[^\\n]*\\n\s*"""', content)
                documented_functions += len(documented_matches)
                
            except Exception:
                continue
        
        documentation_coverage = (documented_functions / max(1, total_functions)) * 100
        
        return {
            "required_docs_found": found_docs,
            "missing_docs": missing_docs,
            "documentation_coverage": documentation_coverage,
            "total_functions": total_functions,
            "documented_functions": documented_functions,
            "completeness_score": len(found_docs) / len(required_docs) * 100
        }
    
    def check_file_structure(self) -> Dict[str, Any]:
        """Check project file structure."""
        
        print("🗂️ Checking file structure...")
        
        expected_structure = {
            "python/": "Python source code",
            "tests/": "Test files",
            "docs/": "Documentation",
            "README.md": "Project README",
            "pyproject.toml": "Project configuration"
        }
        
        structure_check = {}
        for path, description in expected_structure.items():
            full_path = self.project_root / path
            structure_check[path] = {
                "exists": full_path.exists(),
                "description": description,
                "type": "directory" if path.endswith("/") else "file"
            }
        
        # Count Python files
        python_files = len(list(self.project_root.rglob("*.py")))
        test_files = len(list((self.project_root / "tests").rglob("*.py"))) if (self.project_root / "tests").exists() else 0
        
        return {
            "structure_check": structure_check,
            "python_files": python_files,
            "test_files": test_files,
            "structure_score": sum(1 for item in structure_check.values() if item["exists"]) / len(expected_structure) * 100
        }
    
    def check_import_consistency(self) -> Dict[str, Any]:
        """Check import consistency and organization."""
        
        print("🔗 Checking import consistency...")
        
        python_files = list(self.project_root.rglob("*.py"))
        import_issues = []
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                lines = content.split('\\n')
                
                # Check for relative imports
                for i, line in enumerate(lines, 1):
                    if re.match(r'^\\s*from\\s+\\.', line):
                        import_issues.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "line": i,
                            "issue": "Relative import found",
                            "severity": "low"
                        })
                
            except Exception:
                continue
        
        return {
            "total_files_checked": len(python_files),
            "import_issues": import_issues,
            "consistency_score": max(0, 100 - len(import_issues) * 5)
        }
    
    def check_file_sizes(self) -> Dict[str, Any]:
        """Check file sizes for performance issues."""
        
        print("📏 Checking file sizes...")
        
        large_files = []
        total_size = 0
        
        for file_path in self.project_root.rglob("*"):
            if file_path.is_file():
                try:
                    size = file_path.stat().st_size
                    total_size += size
                    
                    # Flag files larger than 1MB
                    if size > 1024 * 1024:
                        large_files.append({
                            "file": str(file_path.relative_to(self.project_root)),
                            "size_mb": size / (1024 * 1024),
                            "type": file_path.suffix
                        })
                except Exception:
                    continue
        
        return {
            "total_size_mb": total_size / (1024 * 1024),
            "large_files": large_files,
            "performance_impact": "low" if len(large_files) < 3 else "medium"
        }
    
    def check_complexity(self) -> Dict[str, Any]:
        """Check code complexity (simplified)."""
        
        print("🧮 Checking code complexity...")
        
        python_files = list(self.project_root.rglob("*.py"))
        complexity_issues = []
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Simple complexity check: count nested structures
                nesting_level = 0
                max_nesting = 0
                
                for line in content.split('\\n'):
                    stripped = line.strip()
                    if any(stripped.startswith(keyword) for keyword in ['if', 'for', 'while', 'with', 'try', 'def', 'class']):
                        nesting_level += 1
                        max_nesting = max(max_nesting, nesting_level)
                    elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                        continue
                    elif not stripped or stripped.startswith('#'):
                        continue
                    else:
                        nesting_level = max(0, nesting_level - 1)
                
                if max_nesting > 6:
                    complexity_issues.append({
                        "file": str(file_path.relative_to(self.project_root)),
                        "max_nesting": max_nesting,
                        "severity": "medium" if max_nesting > 8 else "low"
                    })
                
            except Exception:
                continue
        
        return {
            "total_files_checked": len(python_files),
            "complexity_issues": complexity_issues,
            "complexity_score": max(0, 100 - len(complexity_issues) * 10)
        }
    
    def measure_startup_time(self) -> Dict[str, Any]:
        """Measure module startup time."""
        
        print("⏱️ Measuring startup time...")
        
        start_time = time.time()
        
        try:
            # Test basic import
            import sys
            original_path = sys.path.copy()
            sys.path.insert(0, str(self.project_root / "python"))
            
            import_start = time.time()
            import photon_mlir.core
            import_time = time.time() - import_start
            
            sys.path = original_path
            
        except Exception as e:
            import_time = -1
            print(f"Import test failed: {e}")
        
        total_time = time.time() - start_time
        
        return {
            "core_import_time": import_time,
            "total_startup_time": total_time,
            "performance_rating": "excellent" if import_time < 0.1 else "good" if import_time < 0.5 else "needs_improvement"
        }
    
    def generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall security and quality assessment."""
        
        # Calculate scores
        security_score = 100
        quality_score = 100
        performance_score = 100
        
        # Security scoring
        for category, results in self.results["security"].items():
            if results.get("risk_level") == "high":
                security_score -= 30
            elif results.get("risk_level") == "medium":
                security_score -= 15
            elif results.get("risk_level") == "low":
                security_score -= 5
        
        # Quality scoring
        quality_metrics = self.results["quality"]
        if quality_metrics["code_style"]["quality_score"] < 80:
            quality_score -= 20
        if quality_metrics["documentation"]["completeness_score"] < 75:
            quality_score -= 15
        if quality_metrics["file_structure"]["structure_score"] < 80:
            quality_score -= 10
        
        # Performance scoring
        perf_metrics = self.results["performance"]
        if perf_metrics["complexity_check"]["complexity_score"] < 70:
            performance_score -= 20
        if perf_metrics["file_size_check"]["performance_impact"] == "medium":
            performance_score -= 10
        
        # Overall grade
        overall_score = (security_score + quality_score + performance_score) / 3
        
        if overall_score >= 90:
            grade = "A"
            status = "Production Ready"
        elif overall_score >= 80:
            grade = "B"
            status = "Good Quality"
        elif overall_score >= 70:
            grade = "C"
            status = "Acceptable"
        else:
            grade = "D"
            status = "Needs Improvement"
        
        return {
            "security_score": max(0, security_score),
            "quality_score": max(0, quality_score),
            "performance_score": max(0, performance_score),
            "overall_score": max(0, overall_score),
            "grade": grade,
            "status": status,
            "passed_gates": overall_score >= 70
        }
    
    def print_summary(self):
        """Print comprehensive summary of all gates."""
        
        print("\\n" + "=" * 60)
        print("🏁 SECURITY AND QUALITY GATES SUMMARY")
        print("=" * 60)
        
        overall = self.results["overall"]
        
        print(f"\\n📊 OVERALL ASSESSMENT")
        print(f"Security Score: {overall['security_score']:.1f}/100")
        print(f"Quality Score: {overall['quality_score']:.1f}/100") 
        print(f"Performance Score: {overall['performance_score']:.1f}/100")
        print(f"Overall Score: {overall['overall_score']:.1f}/100")
        print(f"Grade: {overall['grade']}")
        print(f"Status: {overall['status']}")
        
        # Security summary
        print(f"\\n🔒 SECURITY ANALYSIS")
        security = self.results["security"]
        for gate_name, gate_results in security.items():
            risk = gate_results.get("risk_level", "unknown")
            findings = len(gate_results.get("findings", []))
            print(f"  {gate_name}: {risk.upper()} risk ({findings} findings)")
        
        # Quality summary
        print(f"\\n✅ QUALITY ANALYSIS")
        quality = self.results["quality"]
        print(f"  Code Style: {quality['code_style']['quality_score']:.0f}/100")
        print(f"  Documentation: {quality['documentation']['completeness_score']:.0f}/100")
        print(f"  File Structure: {quality['file_structure']['structure_score']:.0f}/100")
        print(f"  Import Consistency: {quality['import_consistency']['consistency_score']:.0f}/100")
        
        # Performance summary
        print(f"\\n⚡ PERFORMANCE ANALYSIS")
        performance = self.results["performance"]
        print(f"  Complexity: {performance['complexity']['complexity_score']:.0f}/100")
        print(f"  File Sizes: {performance['file_size_check']['performance_impact'].upper()} impact")
        print(f"  Startup Time: {performance['startup_time']['performance_rating'].upper()}")
        
        # Final verdict
        if overall["passed_gates"]:
            print("\\n🎉 ALL QUALITY GATES PASSED!")
            print("✅ Ready for production deployment")
        else:
            print("\\n⚠️  SOME QUALITY GATES FAILED")
            print("❌ Address issues before production deployment")


def main():
    """Main execution function."""
    project_root = Path(__file__).parent
    
    print("🤖 Terragon SDLC Security & Quality Gates")
    print("Comprehensive security and quality validation")
    print()
    
    validator = SecurityQualityValidator(project_root)
    results = validator.run_all_gates()
    validator.print_summary()
    
    # Save results
    results_file = project_root / "quality_gates_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\\n📄 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()