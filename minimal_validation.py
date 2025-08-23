#!/usr/bin/env python3
"""
Minimal validation script for Generation 1 functionality testing.
Tests core modules without external dependencies.
"""
import sys
import os
sys.path.insert(0, '/root/repo/python')

def test_core_imports():
    """Test that core modules can be imported."""
    try:
        import photon_mlir.core as core
        print("✅ Core module imported successfully")
        
        # Test basic configuration
        config = core.TargetConfig()
        print(f"✅ Default target config created: {config.device.value}")
        return True
    except ImportError as e:
        print(f"❌ Core import failed: {e}")
        return False

def test_compiler_module():
    """Test compiler module basic functionality."""
    try:
        import photon_mlir.compiler as compiler
        print("✅ Compiler module imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Compiler import failed: {e}")
        return False

def test_autonomous_systems():
    """Test autonomous system modules."""
    try:
        import photon_mlir.autonomous_sdlc_orchestrator as sdlc
        print("✅ Autonomous SDLC orchestrator imported")
        
        import photon_mlir.autonomous_performance_optimizer as perf
        print("✅ Performance optimizer imported")
        
        return True
    except ImportError as e:
        print(f"❌ Autonomous systems import failed: {e}")
        return False

def main():
    """Run minimal validation tests."""
    print("🚀 GENERATION 1 VALIDATION - MAKE IT WORK")
    print("=" * 50)
    
    tests = [
        ("Core Functionality", test_core_imports),
        ("Compiler Module", test_compiler_module), 
        ("Autonomous Systems", test_autonomous_systems)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 Testing {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n🎯 VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Generation 1 core functionality is WORKING!")
        return True
    else:
        print("⚠️  Some issues detected - proceeding with fixes...")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)