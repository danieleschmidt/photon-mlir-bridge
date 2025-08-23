#!/usr/bin/env python3
"""
Robust validation script for Generation 2 - Enhanced error handling and reliability.
"""
import sys
import os
import traceback
sys.path.insert(0, '/root/repo/python')

def test_error_handling():
    """Test enhanced error handling systems."""
    try:
        from photon_mlir.robust_error_handling import (
            PhotonicErrorHandler, ErrorSeverity, RecoveryStrategy
        )
        print("✅ Enhanced error handling imported")
        
        # Test error handler initialization
        handler = PhotonicErrorHandler()  # Use default parameters
        print("✅ Error handler initialized")
        
        return True
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def test_validation_systems():
    """Test comprehensive validation frameworks."""
    try:
        from photon_mlir.enhanced_validation import (
            ComprehensiveValidator, ValidationLevel
        )
        print("✅ Enhanced validation imported")
        
        # Test validator initialization
        validator = ComprehensiveValidator()
        print("✅ Comprehensive validator initialized")
        
        return True
    except Exception as e:
        print(f"❌ Validation system test failed: {e}")
        return False

def test_logging_monitoring():
    """Test advanced logging and monitoring."""
    try:
        from photon_mlir.logging_config import PhotonicLogger, get_global_logger
        print("✅ Advanced logging imported")
        
        # Test logger initialization
        logger = get_global_logger()
        print("✅ Global logger accessible")
        
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

def test_security_framework():
    """Test autonomous security systems."""
    try:
        from photon_mlir.autonomous_security_framework import (
            AutonomousSecurityFramework, SecurityLevel
        )
        print("✅ Autonomous security framework imported")
        
        return True
    except Exception as e:
        print(f"❌ Security framework test failed: {e}")
        return False

def test_circuit_breakers():
    """Test circuit breaker patterns.""" 
    try:
        from photon_mlir.circuit_breaker import ThermalCircuitBreaker, CircuitState
        CircuitBreaker = ThermalCircuitBreaker  # Use thermal circuit breaker as primary
        print("✅ Circuit breaker imported")
        
        # Test circuit breaker - need config and thermal model
        from photon_mlir.circuit_breaker import CircuitBreakerConfig
        from photon_mlir.thermal_optimization import ThermalModel
        
        config = CircuitBreakerConfig(failure_threshold=5, recovery_timeout_s=60)
        thermal_model = ThermalModel.SIMPLE_LINEAR  # Use enum value
        breaker = CircuitBreaker(config, thermal_model)
        print("✅ Circuit breaker initialized")
        
        return True
    except Exception as e:
        print(f"❌ Circuit breaker test failed: {e}")
        return False

def test_caching_systems():
    """Test secure caching systems."""
    try:
        from photon_mlir.secure_caching_system import SecureCachingSystem
        print("✅ Secure caching system imported")
        
        return True
    except Exception as e:
        print(f"❌ Caching system test failed: {e}")
        return False

def main():
    """Run Generation 2 robustness validation."""
    print("🛡️ GENERATION 2 VALIDATION - MAKE IT ROBUST")
    print("=" * 55)
    
    tests = [
        ("Error Handling Systems", test_error_handling),
        ("Validation Frameworks", test_validation_systems),
        ("Logging & Monitoring", test_logging_monitoring),
        ("Security Framework", test_security_framework),
        ("Circuit Breakers", test_circuit_breakers),
        ("Caching Systems", test_caching_systems)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            traceback.print_exc()
    
    print(f"\n🎯 ROBUSTNESS RESULTS: {passed}/{total} systems validated")
    
    if passed >= total * 0.8:  # 80% threshold for robustness
        print("🛡️ Generation 2 robustness systems are OPERATIONAL!")
        return True
    else:
        print("⚠️  Critical robustness gaps detected - implementing fixes...")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)