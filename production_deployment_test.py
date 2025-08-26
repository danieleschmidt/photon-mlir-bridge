#!/usr/bin/env python3
"""
Production Deployment Test Suite
Tests the complete production deployment pipeline and readiness
"""

import sys
import time
import json
from pathlib import Path

# Add Python path
sys.path.insert(0, str(Path(__file__).parent / "python"))

try:
    import photon_mlir
    from photon_mlir.core import TargetConfig, Device, Precision
    from photon_mlir.autonomous_quantum_execution_engine import AutonomousQuantumExecutionEngine
    from photon_mlir.advanced_quantum_scale_orchestrator import AdvancedQuantumScaleOrchestrator, ScalingConfig
    from photon_mlir.autonomous_performance_optimizer import AutonomousPerformanceOptimizer, OptimizationConfig
    print("✅ All core modules imported successfully")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)


def test_production_components():
    """Test production-ready components."""
    
    print("\n🧪 Testing Production Components")
    print("=" * 50)
    
    try:
        # Test configuration creation
        target_config = TargetConfig(
            device=Device.LIGHTMATTER_ENVISE,
            precision=Precision.FP32,
            array_size=(128, 128),
            wavelength_nm=1550,
            enable_thermal_compensation=True
        )
        print("✅ Target configuration created")
        
        # Test scaling configuration
        scaling_config = ScalingConfig(
            min_nodes=3,
            max_nodes=50,
            target_utilization=0.75,
            enable_predictive_scaling=True,
            enable_cost_optimization=True
        )
        print("✅ Scaling configuration created")
        
        # Test optimization configuration
        opt_config = OptimizationConfig(
            enable_ml_optimization=True,
            cache_strategy='hybrid_multilevel',
            scaling_mode='ml_optimized'
        )
        print("✅ Optimization configuration created")
        
        return True
        
    except Exception as e:
        print(f"❌ Production component test failed: {e}")
        return False


def test_deployment_files():
    """Test deployment file availability."""
    
    print("\n📦 Testing Deployment Files")
    print("=" * 50)
    
    deployment_files = [
        "deployment/docker-compose.production.yml",
        "deployment/kubernetes/quantum-scheduler.yaml",
        "deployment/helm/quantum-scheduler/Chart.yaml",
        "deployment/scripts/entrypoint.sh",
        "deployment/scripts/healthcheck.sh"
    ]
    
    missing_files = []
    for file_path in deployment_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  {len(missing_files)} deployment files missing")
        return False
    else:
        print(f"\n✅ All {len(deployment_files)} deployment files present")
        return True


def test_documentation_completeness():
    """Test documentation completeness."""
    
    print("\n📚 Testing Documentation")
    print("=" * 50)
    
    doc_files = [
        "README.md",
        "ARCHITECTURE.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "docs/guides/GETTING_STARTED.md",
        "docs/operations/deployment-guide.md"
    ]
    
    complete_docs = 0
    for doc_path in doc_files:
        full_path = Path(doc_path)
        if full_path.exists() and full_path.stat().st_size > 1000:  # At least 1KB
            print(f"✅ {doc_path} ({full_path.stat().st_size} bytes)")
            complete_docs += 1
        else:
            print(f"⚠️  {doc_path} (missing or incomplete)")
    
    coverage = complete_docs / len(doc_files)
    print(f"\n📊 Documentation coverage: {coverage:.1%}")
    
    return coverage >= 0.8  # 80% documentation coverage required


def test_monitoring_setup():
    """Test monitoring and observability setup."""
    
    print("\n📊 Testing Monitoring Setup")
    print("=" * 50)
    
    monitoring_files = [
        "monitoring/prometheus.yml",
        "monitoring/grafana/dashboards/photon-mlir-overview.json",
        "deployment/monitoring-stack.yml"
    ]
    
    monitoring_ready = True
    for mon_file in monitoring_files:
        full_path = Path(mon_file)
        if full_path.exists():
            print(f"✅ {mon_file}")
        else:
            print(f"❌ {mon_file}")
            monitoring_ready = False
    
    return monitoring_ready


def test_security_configuration():
    """Test security configuration readiness."""
    
    print("\n🔒 Testing Security Configuration")
    print("=" * 50)
    
    security_checks = [
        ("Security policy", Path("SECURITY.md").exists()),
        ("Docker security", Path("deployment/docker/Dockerfile.production").exists()),
        ("K8s security", "securityContext" in Path("deployment/kubernetes/quantum-scheduler.yaml").read_text() if Path("deployment/kubernetes/quantum-scheduler.yaml").exists() else False),
        ("Secrets management", Path("docs/security/vulnerability-management.md").exists()),
    ]
    
    passed_checks = 0
    for check_name, result in security_checks:
        if result:
            print(f"✅ {check_name}")
            passed_checks += 1
        else:
            print(f"❌ {check_name}")
    
    security_score = passed_checks / len(security_checks)
    print(f"\n🔒 Security readiness: {security_score:.1%}")
    
    return security_score >= 0.75  # 75% security readiness required


def test_scalability_features():
    """Test scalability and performance features."""
    
    print("\n🚀 Testing Scalability Features")
    print("=" * 50)
    
    try:
        # Test that scalability classes can be instantiated
        scaling_config = ScalingConfig(
            min_nodes=2,
            max_nodes=10,
            enable_predictive_scaling=True
        )
        
        target_config = TargetConfig()
        
        # This tests that the class can be created (doesn't need to run)
        orchestrator = AdvancedQuantumScaleOrchestrator(
            target_config=target_config,
            scaling_config=scaling_config
        )
        
        print("✅ Quantum scale orchestrator initialized")
        
        # Test performance optimizer
        opt_config = OptimizationConfig(
            enable_ml_optimization=True,
            min_workers=2,
            max_workers=8
        )
        
        optimizer = AutonomousPerformanceOptimizer(
            target_config=target_config,
            config=opt_config
        )
        
        print("✅ Performance optimizer initialized")
        
        # Test status methods
        orch_status = orchestrator.get_orchestration_status()
        opt_status = optimizer.get_optimization_status()
        
        print("✅ Status reporting functional")
        print(f"✅ Orchestrator supports {orch_status.get('cluster_size', 0)} nodes")
        print(f"✅ Optimizer supports {opt_status.get('current_workers', 0)} workers")
        
        return True
        
    except Exception as e:
        print(f"❌ Scalability test failed: {e}")
        return False


def main():
    """Run production deployment tests."""
    
    print("🚀 PRODUCTION DEPLOYMENT TEST SUITE")
    print("="*60)
    print("Testing production readiness and deployment capabilities")
    print()
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Production Components", test_production_components),
        ("Deployment Files", test_deployment_files),
        ("Documentation", test_documentation_completeness),
        ("Monitoring Setup", test_monitoring_setup),
        ("Security Configuration", test_security_configuration),
        ("Scalability Features", test_scalability_features)
    ]
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("🎯 PRODUCTION DEPLOYMENT TEST SUMMARY")
    print("="*60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print()
    print(f"📊 Overall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
    
    if success_rate >= 0.8:
        print("🎉 PRODUCTION DEPLOYMENT: READY")
        print("✅ System meets production deployment standards")
        exit_code = 0
    elif success_rate >= 0.6:
        print("⚠️  PRODUCTION DEPLOYMENT: NEEDS IMPROVEMENT")
        print("🔧 Some components need attention before deployment")
        exit_code = 1
    else:
        print("❌ PRODUCTION DEPLOYMENT: NOT READY")
        print("🚨 Critical issues must be resolved before deployment")
        exit_code = 2
    
    print()
    print("🏁 Production deployment testing completed")
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)