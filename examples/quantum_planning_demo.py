#!/usr/bin/env python3
"""
Quantum-Inspired Task Planning Demo

Demonstrates the quantum-inspired task scheduling and optimization capabilities
for photonic neural network compilation using superposition states, quantum
annealing, and entanglement-inspired dependency resolution.
"""

import logging
import time
from typing import Dict, Any

import photon_mlir as pm
from photon_mlir.quantum_scheduler import (
    QuantumTaskPlanner, 
    TaskType, 
    CompilationTask
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_complex_model_config() -> Dict[str, Any]:
    """Create a complex model configuration for demonstration."""
    return {
        "model_type": "transformer",
        "layers": 12,
        "hidden_size": 768,
        "attention_heads": 12,
        "sequence_length": 512,
        "precision": "int8",
        "target_device": "lightmatter_envise",
        "optimization_level": 3,
        "enable_thermal_compensation": True,
        "photonic_array_size": (64, 64)
    }


def demonstrate_quantum_planning():
    """Demonstrate quantum-inspired task planning for photonic compilation."""
    logger.info("🚀 Starting Quantum-Inspired Task Planning Demo")
    logger.info("=" * 60)
    
    # Initialize quantum task planner
    planner = QuantumTaskPlanner()
    logger.info("✅ Quantum Task Planner initialized")
    
    # Create model configuration
    model_config = create_complex_model_config()
    logger.info(f"📋 Model Config: {model_config['model_type']} "
               f"({model_config['layers']} layers, {model_config['hidden_size']} hidden)")
    
    # Generate compilation plan
    logger.info("🧠 Generating compilation plan with quantum superposition...")
    start_time = time.time()
    
    tasks = planner.create_compilation_plan(model_config)
    
    planning_time = time.time() - start_time
    logger.info(f"⚡ Plan generation completed in {planning_time:.3f}s")
    logger.info(f"📊 Generated {len(tasks)} compilation tasks")
    
    # Display task information
    print("\n" + "=" * 60)
    print("📋 COMPILATION TASK BREAKDOWN")
    print("=" * 60)
    
    total_estimated_duration = 0
    for i, task in enumerate(tasks, 1):
        deps_str = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
        print(f"{i:2d}. {task.task_type.value.upper()}")
        print(f"    ID: {task.id}")
        print(f"    Duration: {task.estimated_duration:.1f}s")
        print(f"    Resources: CPU={task.resource_requirements.get('cpu', 0):.1f}, "
              f"Memory={task.resource_requirements.get('memory', 0):.0f}MB")
        print(f"    Dependencies: {len(task.dependencies)}{deps_str}")
        print(f"    Quantum State: {task.quantum_state.value}")
        print()
        total_estimated_duration += task.estimated_duration
    
    print(f"📈 Total Sequential Duration: {total_estimated_duration:.1f}s")
    print()
    
    # Optimize schedule using quantum annealing
    logger.info("🔬 Starting quantum annealing optimization...")
    start_time = time.time()
    
    optimized_schedule = planner.optimize_schedule(tasks)
    
    optimization_time = time.time() - start_time
    logger.info(f"⚡ Quantum optimization completed in {optimization_time:.3f}s")
    
    # Display optimization results
    print("=" * 60)
    print("🎯 QUANTUM-OPTIMIZED SCHEDULE")
    print("=" * 60)
    
    print(f"📊 Optimization Metrics:")
    print(f"   • Makespan: {optimized_schedule.makespan:.2f}s")
    print(f"   • Resource Utilization: {optimized_schedule.resource_utilization:.2%}")
    print(f"   • Speedup vs Sequential: {total_estimated_duration/optimized_schedule.makespan:.2f}x")
    print(f"   • Total Energy: {optimized_schedule.total_energy:.2f}")
    print()
    
    print("🗓️  Parallel Execution Schedule:")
    for timeslot in sorted(optimized_schedule.schedule.keys()):
        task_ids = optimized_schedule.schedule[timeslot]
        task_names = []
        for task_id in task_ids:
            task = next(t for t in tasks if t.id == task_id)
            task_names.append(f"{task.task_type.value} ({task.estimated_duration:.1f}s)")
        
        print(f"   T{timeslot:2d}: {' || '.join(task_names)}")
    
    print()
    
    # Quantum state analysis
    print("=" * 60)
    print("🔬 QUANTUM STATE ANALYSIS")
    print("=" * 60)
    
    collapsed_tasks = sum(1 for task in tasks if task.quantum_state.name == "COLLAPSED")
    print(f"📊 Quantum States:")
    print(f"   • Collapsed States: {collapsed_tasks}/{len(tasks)}")
    print(f"   • Final Schedule Slots: {len(optimized_schedule.schedule)}")
    
    # Dependency validation
    dependency_violations = optimized_schedule._calculate_dependency_violations()
    print(f"   • Dependency Violations: {dependency_violations}")
    
    print()
    
    # Performance summary
    print("=" * 60)
    print("⚡ PERFORMANCE SUMMARY")
    print("=" * 60)
    
    efficiency_gain = (total_estimated_duration - optimized_schedule.makespan) / total_estimated_duration
    print(f"🚀 Quantum Planning Results:")
    print(f"   • Planning Time: {planning_time:.3f}s")
    print(f"   • Optimization Time: {optimization_time:.3f}s")
    print(f"   • Total Algorithm Time: {planning_time + optimization_time:.3f}s")
    print(f"   • Compilation Speedup: {total_estimated_duration/optimized_schedule.makespan:.2f}x")
    print(f"   • Efficiency Gain: {efficiency_gain:.1%}")
    print(f"   • Resource Utilization: {optimized_schedule.resource_utilization:.1%}")
    
    print()
    logger.info("✅ Quantum-Inspired Task Planning Demo completed successfully!")
    
    return optimized_schedule


def demonstrate_advanced_features():
    """Demonstrate advanced quantum-inspired features."""
    logger.info("🔬 Demonstrating Advanced Quantum Features")
    print("=" * 60)
    print("🧪 ADVANCED QUANTUM FEATURES")
    print("=" * 60)
    
    # Create custom task with complex dependencies
    planner = QuantumTaskPlanner()
    
    # Manual task creation for advanced demo
    task1 = CompilationTask(
        id="custom_graph_analysis",
        task_type=TaskType.GRAPH_LOWERING,
        estimated_duration=3.0,
        priority=2.0  # High priority
    )
    
    task2 = CompilationTask(
        id="quantum_photonic_opt",
        task_type=TaskType.PHOTONIC_OPTIMIZATION,
        dependencies={task1.id},
        estimated_duration=8.0,
        priority=3.0  # Highest priority
    )
    
    task3 = CompilationTask(
        id="entangled_mesh_phase",
        task_type=TaskType.MESH_MAPPING,
        dependencies={task2.id},
        estimated_duration=4.0,
        priority=1.0
    )
    
    task4 = CompilationTask(
        id="entangled_phase_mesh", 
        task_type=TaskType.PHASE_OPTIMIZATION,
        dependencies={task2.id},
        estimated_duration=5.0,
        priority=1.0
    )
    
    # Mark tasks as entangled (correlated optimization)
    task3.entangled_tasks.add(task4.id)
    task4.entangled_tasks.add(task3.id)
    
    advanced_tasks = [task1, task2, task3, task4]
    
    print("🔗 Created entangled tasks:")
    for task in advanced_tasks:
        entangled_str = f" (entangled with: {', '.join(task.entangled_tasks)})" if task.entangled_tasks else ""
        print(f"   • {task.id}: priority={task.priority}{entangled_str}")
    
    # Optimize with quantum scheduler
    optimized = planner.optimize_schedule(advanced_tasks)
    
    print(f"\n🎯 Advanced Optimization Results:")
    print(f"   • Makespan: {optimized.makespan:.2f}s")
    print(f"   • Utilization: {optimized.resource_utilization:.2%}")
    
    print("\n🗓️  Advanced Schedule:")
    for slot, task_ids in sorted(optimized.schedule.items()):
        print(f"   T{slot:2d}: {', '.join(task_ids)}")
    
    print()


if __name__ == "__main__":
    try:
        # Main demonstration
        schedule = demonstrate_quantum_planning()
        
        # Advanced features
        demonstrate_advanced_features()
        
        print("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise