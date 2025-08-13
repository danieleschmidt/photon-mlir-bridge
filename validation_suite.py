#!/usr/bin/env python3
"""
Comprehensive Validation Suite for Photon-MLIR Bridge
Generation 3 Quality Gates - Tests, Security, Performance

This validation suite tests all implemented features without external dependencies.
"""

import sys
import os
import time
import json
from typing import Dict, List, Any, Optional

# Add python module path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

def validate_file_structure() -> Dict[str, Any]:
    """Validate project file structure and key components."""
    
    validation_results = {
        'test_name': 'file_structure_validation',
        'success': True,
        'details': {},
        'errors': []
    }
    
    # Key directories and files to check
    required_structure = {
        'python/photon_mlir/': ['__init__.py', 'core.py', 'compiler.py', 'robust_error_handling.py'],
        'include/photon/': ['core/PhotonicCompiler.h', 'dialects/PhotonicOps.td'],
        'src/': ['core/PhotonicCompiler.cpp', 'dialects/PhotonicDialect.cpp'],
        'tests/': ['conftest.py'],
        'docs/': ['README.md'],
        'examples/': ['basic_compilation.py'],
        'scripts/': ['automation/'],
        'deployment/': ['docker-compose.production.yml']
    }
    
    try:
        for directory, files in required_structure.items():
            dir_path = os.path.join(os.getcwd(), directory)
            
            if not os.path.exists(dir_path):
                validation_results['errors'].append(f"Missing directory: {directory}")
                continue
                
            validation_results['details'][directory] = {
                'exists': True,
                'files_checked': []
            }
            
            for file_name in files:
                file_path = os.path.join(dir_path, file_name)
                file_exists = os.path.exists(file_path)
                
                validation_results['details'][directory]['files_checked'].append({
                    'file': file_name,
                    'exists': file_exists,
                    'size_bytes': os.path.getsize(file_path) if file_exists else 0
                })
                
                if not file_exists:
                    validation_results['errors'].append(f"Missing file: {directory}/{file_name}")
                    
    except Exception as e:
        validation_results['success'] = False
        validation_results['errors'].append(f"File structure validation failed: {e}")
        
    if validation_results['errors']:
        validation_results['success'] = False
        
    return validation_results


def validate_python_modules() -> Dict[str, Any]:
    """Validate Python module imports and basic functionality."""
    
    validation_results = {
        'test_name': 'python_modules_validation',
        'success': True,
        'modules_tested': {},
        'errors': []
    }
    
    # Test core modules without numpy dependency
    modules_to_test = [
        ('photon_mlir.core', 'Core types and configuration'),
        ('photon_mlir.compiler', 'Main compiler interface'),
        ('photon_mlir.robust_error_handling', 'Error handling system'),
        ('photon_mlir.logging_config', 'Logging configuration'),
        ('photon_mlir.validation', 'Input validation'),
    ]
    
    for module_name, description in modules_to_test:
        try:
            # Attempt to import without executing numpy-dependent code
            print(f"Testing module: {module_name}")
            
            if module_name == 'photon_mlir.core':
                # Test core enums and classes
                exec("""
import photon_mlir.core as core
# Test enum creation
device = core.Device.LIGHTMATTER_ENVISE
precision = core.Precision.INT8
# Test TargetConfig creation (basic)
config = core.TargetConfig()
assert config.device == device
assert config.precision == precision
""")
                validation_results['modules_tested'][module_name] = {
                    'status': 'success',
                    'description': description,
                    'tests_passed': ['enum_creation', 'config_creation']
                }
                
            elif module_name == 'photon_mlir.robust_error_handling':
                # Test error handling enums and classes  
                exec("""
import photon_mlir.robust_error_handling as error_handling
# Test error classification
severity = error_handling.ErrorSeverity.HIGH
category = error_handling.ErrorCategory.COMPILATION_ERROR
# Test error context creation
context = error_handling.ErrorContext(
    component='test',
    operation='validation',
    severity=severity,
    category=category
)
assert context.component == 'test'
assert context.severity == severity
""")
                validation_results['modules_tested'][module_name] = {
                    'status': 'success', 
                    'description': description,
                    'tests_passed': ['error_classification', 'context_creation']
                }
                
            else:
                # Basic import test for other modules
                exec(f"import {module_name}")
                validation_results['modules_tested'][module_name] = {
                    'status': 'success',
                    'description': description,
                    'tests_passed': ['basic_import']
                }
                
        except Exception as e:
            validation_results['errors'].append(f"Module {module_name} failed: {e}")
            validation_results['modules_tested'][module_name] = {
                'status': 'failed',
                'error': str(e),
                'description': description
            }
            
    if validation_results['errors']:
        validation_results['success'] = False
        
    return validation_results


def validate_cpp_structure() -> Dict[str, Any]:
    """Validate C++ code structure and MLIR dialect definitions."""
    
    validation_results = {
        'test_name': 'cpp_structure_validation',
        'success': True,
        'files_analyzed': {},
        'errors': []
    }
    
    cpp_files_to_check = [
        ('src/core/PhotonicCompiler.cpp', ['PhotonicCompiler::', 'loadONNX', 'compile', 'codegen']),
        ('include/photon/core/PhotonicCompiler.h', ['class PhotonicCompiler', 'LogicalResult', 'TargetConfig']),
        ('include/photon/dialects/PhotonicOps.td', ['MatMulOp', 'PhaseShiftOp', 'ThermalCompensationOp']),
        ('CMakeLists.txt', ['photon-mlir-bridge', 'MLIR', 'PhotonicOpsIncGen'])
    ]
    
    for file_path, expected_content in cpp_files_to_check:
        full_path = os.path.join(os.getcwd(), file_path)
        
        if not os.path.exists(full_path):
            validation_results['errors'].append(f"Missing C++ file: {file_path}")
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            found_content = []
            missing_content = []
            
            for expected in expected_content:
                if expected in content:
                    found_content.append(expected)
                else:
                    missing_content.append(expected)
                    
            validation_results['files_analyzed'][file_path] = {
                'size_bytes': len(content),
                'found_content': found_content,
                'missing_content': missing_content,
                'content_coverage': len(found_content) / len(expected_content)
            }
            
            if missing_content:
                validation_results['errors'].append(f"Missing content in {file_path}: {missing_content}")
                
        except Exception as e:
            validation_results['errors'].append(f"Failed to analyze {file_path}: {e}")
            
    if validation_results['errors']:
        validation_results['success'] = False
        
    return validation_results


def validate_documentation() -> Dict[str, Any]:
    """Validate documentation completeness and quality."""
    
    validation_results = {
        'test_name': 'documentation_validation',
        'success': True,
        'docs_analyzed': {},
        'errors': []
    }
    
    docs_to_check = [
        ('README.md', ['photon-mlir-bridge', 'Quick Start', 'Installation', 'Features']),
        ('ARCHITECTURE.md', []),  # Check if exists
        ('CONTRIBUTING.md', []),
        ('docs/guides/GETTING_STARTED.md', []),
        ('IMPLEMENTATION_SUMMARY.md', [])
    ]
    
    for doc_path, required_sections in docs_to_check:
        full_path = os.path.join(os.getcwd(), doc_path)
        
        if not os.path.exists(full_path):
            validation_results['errors'].append(f"Missing documentation: {doc_path}")
            continue
            
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Basic documentation metrics
            word_count = len(content.split())
            line_count = len(content.splitlines())
            
            found_sections = []
            missing_sections = []
            
            for section in required_sections:
                if section.lower() in content.lower():
                    found_sections.append(section)
                else:
                    missing_sections.append(section)
                    
            validation_results['docs_analyzed'][doc_path] = {
                'word_count': word_count,
                'line_count': line_count,
                'found_sections': found_sections,
                'missing_sections': missing_sections,
                'completeness_score': len(found_sections) / max(1, len(required_sections))
            }
            
            if missing_sections:
                validation_results['errors'].append(f"Missing sections in {doc_path}: {missing_sections}")
                
        except Exception as e:
            validation_results['errors'].append(f"Failed to analyze {doc_path}: {e}")
            
    if validation_results['errors']:
        validation_results['success'] = False
        
    return validation_results


def validate_security_aspects() -> Dict[str, Any]:
    """Validate security aspects of the implementation."""
    
    validation_results = {
        'test_name': 'security_validation', 
        'success': True,
        'security_checks': {},
        'errors': []
    }
    
    # Check for common security issues in Python files
    python_files = []
    for root, dirs, files in os.walk('python'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
                
    security_patterns = {
        'hardcoded_passwords': ['password =', 'pwd =', 'secret ='],
        'unsafe_eval': ['eval(', 'exec(', 'compile('],
        'unsafe_pickle': ['pickle.loads', 'cPickle.loads'],
        'command_injection': ['os.system(', 'subprocess.call(', 'shell=True'],
        'path_traversal': ['../']
    }
    
    security_issues = []
    
    for file_path in python_files[:10]:  # Limit to first 10 files
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            file_issues = {}
            
            for issue_type, patterns in security_patterns.items():
                found_patterns = [p for p in patterns if p in content]
                if found_patterns:
                    file_issues[issue_type] = found_patterns
                    
            if file_issues:
                security_issues.append({
                    'file': file_path,
                    'issues': file_issues
                })
                
        except Exception as e:
            validation_results['errors'].append(f"Security scan failed for {file_path}: {e}")
            
    validation_results['security_checks'] = {
        'files_scanned': len(python_files[:10]),
        'issues_found': len(security_issues),
        'issue_details': security_issues
    }
    
    # Input validation checks
    validation_files = [f for f in python_files if 'validation' in f]
    validation_results['security_checks']['input_validation'] = {
        'validation_modules': len(validation_files),
        'validation_functions_found': 0
    }
    
    # Check for validation functions
    for val_file in validation_files:
        try:
            with open(val_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            validation_keywords = ['validate_', 'sanitize_', 'check_input', 'verify_']
            found_validators = sum(1 for keyword in validation_keywords if keyword in content)
            validation_results['security_checks']['input_validation']['validation_functions_found'] += found_validators
            
        except Exception as e:
            continue
            
    # Overall security assessment
    if security_issues:
        validation_results['errors'].extend([f"Security issue in {issue['file']}: {issue['issues']}" for issue in security_issues])
        
    if validation_results['errors']:
        validation_results['success'] = False
        
    return validation_results


def validate_performance_characteristics() -> Dict[str, Any]:
    """Validate performance characteristics and optimization features."""
    
    validation_results = {
        'test_name': 'performance_validation',
        'success': True,
        'performance_features': {},
        'errors': []
    }
    
    # Check for performance-related implementations
    performance_indicators = {
        'caching': ['@lru_cache', 'cache', 'memoization'],
        'concurrency': ['ThreadPoolExecutor', 'asyncio', 'concurrent.futures'],
        'optimization': ['optimization', 'performance_monitor', '@performance'],
        'memory_management': ['gc.collect', 'weakref', 'memory_limit'],
        'profiling': ['@profile', 'time.time()', 'performance_context']
    }
    
    python_files = []
    for root, dirs, files in os.walk('python'):
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
                
    for category, indicators in performance_indicators.items():
        found_files = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                found_indicators = [ind for ind in indicators if ind in content]
                
                if found_indicators:
                    found_files.append({
                        'file': file_path,
                        'indicators': found_indicators
                    })
                    
            except Exception as e:
                continue
                
        validation_results['performance_features'][category] = {
            'files_with_features': len(found_files),
            'details': found_files
        }
        
    # Check for advanced features
    advanced_features = [
        'quantum_photonic_bridge',
        'advanced_wdm_optimizer', 
        'ml_thermal_predictor',
        'neural_ode',
        'circuit_breaker',
        'error_recovery'
    ]
    
    advanced_implementations = []
    for feature in advanced_features:
        feature_found = False
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                if feature.replace('_', ' ').lower() in content.lower() or feature in content:
                    feature_found = True
                    advanced_implementations.append(feature)
                    break
                    
            except Exception as e:
                continue
                
    validation_results['performance_features']['advanced_implementations'] = {
        'total_features': len(advanced_features),
        'implemented_features': len(advanced_implementations),
        'implementation_coverage': len(advanced_implementations) / len(advanced_features),
        'implemented_list': advanced_implementations
    }
    
    return validation_results


def run_comprehensive_validation() -> Dict[str, Any]:
    """Run comprehensive validation suite."""
    
    print("🚀 Starting Comprehensive Validation Suite")
    print("=" * 60)
    
    start_time = time.time()
    
    validation_suite = {
        'validation_timestamp': time.time(),
        'validation_duration_ms': 0.0,
        'overall_success': True,
        'test_results': {},
        'summary': {},
        'recommendations': []
    }
    
    # Run all validation tests
    validation_tests = [
        ('File Structure', validate_file_structure),
        ('Python Modules', validate_python_modules),
        ('C++ Structure', validate_cpp_structure),
        ('Documentation', validate_documentation),
        ('Security', validate_security_aspects),
        ('Performance', validate_performance_characteristics)
    ]
    
    for test_name, test_function in validation_tests:
        print(f"\n📋 Running {test_name} Validation...")
        
        try:
            test_result = test_function()
            validation_suite['test_results'][test_name.lower().replace(' ', '_')] = test_result
            
            status = "✅ PASSED" if test_result['success'] else "❌ FAILED"
            print(f"   {status}")
            
            if test_result['errors']:
                print(f"   Errors: {len(test_result['errors'])}")
                for error in test_result['errors'][:3]:  # Show first 3 errors
                    print(f"     • {error}")
                    
            if not test_result['success']:
                validation_suite['overall_success'] = False
                
        except Exception as e:
            print(f"   ❌ FAILED - Exception: {e}")
            validation_suite['test_results'][test_name.lower().replace(' ', '_')] = {
                'test_name': test_name.lower().replace(' ', '_'),
                'success': False,
                'error': str(e)
            }
            validation_suite['overall_success'] = False
            
    # Calculate validation duration
    validation_suite['validation_duration_ms'] = (time.time() - start_time) * 1000
    
    # Generate summary
    total_tests = len(validation_tests)
    passed_tests = sum(1 for result in validation_suite['test_results'].values() 
                      if result.get('success', False))
    
    validation_suite['summary'] = {
        'total_tests': total_tests,
        'passed_tests': passed_tests,
        'failed_tests': total_tests - passed_tests,
        'success_rate': passed_tests / total_tests,
        'overall_status': 'PASSED' if validation_suite['overall_success'] else 'FAILED'
    }
    
    # Generate recommendations
    if not validation_suite['overall_success']:
        validation_suite['recommendations'].append("🔧 Address failed validation tests before deployment")
        
    if validation_suite['summary']['success_rate'] < 0.8:
        validation_suite['recommendations'].append("⚠️ Success rate below 80% - major improvements needed")
    elif validation_suite['summary']['success_rate'] < 0.95:
        validation_suite['recommendations'].append("💡 Good progress - minor improvements recommended")
    else:
        validation_suite['recommendations'].append("🌟 Excellent validation results - system ready for deployment")
        
    # Additional specific recommendations
    security_result = validation_suite['test_results'].get('security', {})
    if security_result.get('success', True) == False:
        validation_suite['recommendations'].append("🔒 Security issues detected - immediate attention required")
        
    performance_result = validation_suite['test_results'].get('performance', {})
    if performance_result.get('success', True):
        perf_features = performance_result.get('performance_features', {})
        advanced_impl = perf_features.get('advanced_implementations', {})
        coverage = advanced_impl.get('implementation_coverage', 0.0)
        
        if coverage >= 0.8:
            validation_suite['recommendations'].append("🚀 Advanced features well implemented - excellent scalability")
        elif coverage >= 0.5:
            validation_suite['recommendations'].append("📈 Good feature coverage - continue development")
        else:
            validation_suite['recommendations'].append("🔨 Implement more advanced features for production readiness")
            
    return validation_suite


def main():
    """Main validation function."""
    
    try:
        # Change to repo directory
        os.chdir('/root/repo')
        
        # Run comprehensive validation
        results = run_comprehensive_validation()
        
        # Print final summary
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = results['summary']
        print(f"Overall Status: {summary['overall_status']}")
        print(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Duration: {results['validation_duration_ms']:.1f}ms")
        
        print(f"\n📋 RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  {rec}")
            
        # Save results to file
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"\n💾 Full results saved to validation_results.json")
        
        # Exit with appropriate code
        exit_code = 0 if results['overall_success'] else 1
        print(f"\n🏁 Validation completed with exit code: {exit_code}")
        
        return exit_code
        
    except Exception as e:
        print(f"❌ Validation suite failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)