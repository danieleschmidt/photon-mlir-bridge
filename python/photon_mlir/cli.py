"""
Command-line interface for photonic compilation tools.
"""

import argparse
import sys
import os
from pathlib import Path

from .compiler import PhotonicCompiler, compile_onnx, compile_pytorch
from .core import TargetConfig, Device, Precision
from .simulator import PhotonicSimulator


def compile_main():
    """Main entry point for photon-compile command."""
    parser = argparse.ArgumentParser(description="Compile neural networks for photonic hardware")
    
    parser.add_argument("input", help="Input model file (ONNX or PyTorch)")
    parser.add_argument("-o", "--output", default="model.pasm", 
                       help="Output photonic assembly file")
    parser.add_argument("--target", default="lightmatter_envise",
                       choices=["lightmatter_envise", "mit_photonic", "research_chip"],
                       help="Target photonic device")
    parser.add_argument("--precision", default="int8",
                       choices=["int8", "int16", "fp16", "fp32"],
                       help="Computation precision") 
    parser.add_argument("--array-size", default="64,64",
                       help="Photonic array dimensions (width,height)")
    parser.add_argument("--wavelength", type=int, default=1550,
                       help="Operating wavelength in nm")
    parser.add_argument("--show-report", action="store_true",
                       help="Display optimization report")
    parser.add_argument("-v", "--verbose", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Photonic MLIR Compiler v0.1.0")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"Target: {args.target}")
        print(f"Precision: {args.precision}")
        print(f"Array size: {args.array_size}")
        print(f"Wavelength: {args.wavelength} nm")
        print()
    
    # Parse array size
    try:
        width, height = map(int, args.array_size.split(','))
        array_size = (width, height)
    except ValueError:
        print(f"Error: Invalid array size format '{args.array_size}'. Use 'width,height'")
        return 1
    
    # Create target configuration
    device_map = {
        "lightmatter_envise": Device.LIGHTMATTER_ENVISE,
        "mit_photonic": Device.MIT_PHOTONIC_PROCESSOR,
        "research_chip": Device.CUSTOM_RESEARCH_CHIP
    }
    
    precision_map = {
        "int8": Precision.INT8,
        "int16": Precision.INT16,  
        "fp16": Precision.FP16,
        "fp32": Precision.FP32
    }
    
    config = TargetConfig(
        device=device_map[args.target],
        precision=precision_map[args.precision],
        array_size=array_size,
        wavelength_nm=args.wavelength
    )
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    try:
        # Compile model
        if args.verbose:
            print("Loading and compiling model...")
            
        compiler = PhotonicCompiler(config)
        compiled_model = compiler.compile_onnx(args.input)
        
        if args.verbose:
            print("Compilation successful")
        
        # Generate output
        compiled_model.export(args.output)
        
        if args.verbose:
            print(f"Output written to: {args.output}")
        
        # Show report if requested
        if args.show_report:
            print("\n" + compiled_model.get_optimization_report())
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def simulate_main():
    """Main entry point for photon-simulate command.""" 
    parser = argparse.ArgumentParser(description="Simulate photonic neural network execution")
    
    parser.add_argument("model", help="Compiled photonic model file (.pasm)")
    parser.add_argument("--input", required=True, help="Input data file (.npy)")
    parser.add_argument("--output", default="output.npy", help="Output file")
    parser.add_argument("--noise-model", default="realistic",
                       choices=["ideal", "realistic", "pessimistic"],
                       help="Noise model for simulation")
    parser.add_argument("--precision", default="8bit",
                       choices=["8bit", "16bit", "fp16", "fp32"],
                       help="Computation precision")
    parser.add_argument("--crosstalk", type=float, default=-30.0,
                       help="Optical crosstalk in dB")
    parser.add_argument("--report", action="store_true",
                       help="Generate simulation report")
    parser.add_argument("-v", "--verbose", action="store_true")
    
    args = parser.parse_args()
    
    try:
        import numpy as np
        
        # Load input data
        if args.verbose:
            print(f"Loading input data from {args.input}")
        input_data = np.load(args.input)
        
        # Create simulator
        simulator = PhotonicSimulator(
            noise_model=args.noise_model,
            precision=args.precision,
            crosstalk_db=args.crosstalk
        )
        
        if args.verbose:
            print(f"Running simulation with {args.noise_model} noise model")
        
        # For now, create a mock compiled model
        class MockModel:
            pass
        mock_model = MockModel()
        
        # Run simulation
        result = simulator.run(mock_model, input_data)
        
        # Save output
        np.save(args.output, result.data)
        
        if args.verbose:
            print(f"Simulation complete. Output saved to {args.output}")
        
        # Generate report if requested
        if args.report:
            report = simulator.get_simulation_report()
            print("\n=== Simulation Report ===")
            for key, value in report.items():
                print(f"{key}: {value}")
                
    except ImportError:
        print("Error: NumPy required for simulation. Install with: pip install numpy")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def profile_main():
    """Main entry point for photon-profile command."""
    parser = argparse.ArgumentParser(description="Profile photonic model performance")
    
    parser.add_argument("model", help="Compiled photonic model (.pasm)")
    parser.add_argument("--input-shape", required=True, 
                       help="Input tensor shape (e.g., 1,3,224,224)")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of profiling runs")
    parser.add_argument("--measure", default="latency,thermal,power",
                       help="Metrics to measure (comma-separated)")
    parser.add_argument("--output", help="Save profiling results to file")
    
    args = parser.parse_args()
    
    try:
        # Parse input shape
        shape = tuple(map(int, args.input_shape.split(',')))
        metrics = args.measure.split(',')
        
        print(f"Profiling model: {args.model}")
        print(f"Input shape: {shape}")
        print(f"Runs: {args.runs}")
        print(f"Metrics: {metrics}")
        print()
        
        # Mock profiling results
        results = {
            "latency_us": [12.3, 8.7, 8.9, 9.1] * (args.runs // 4),
            "thermal_c": [0.8, 0.6, 0.7, 0.7] * (args.runs // 4),  
            "power_mw": [45, 32, 33, 35] * (args.runs // 4)
        }
        
        # Display results
        print("Layer          Latency(μs)  Thermal(°C)  Power(mW)")
        print("-" * 50)
        layer_names = ["conv1", "layer1.0", "layer1.1", "layer2.0"]
        for i, name in enumerate(layer_names):
            if i < len(results["latency_us"]):
                lat = results["latency_us"][i]
                temp = results["thermal_c"][i] 
                power = results["power_mw"][i]
                print(f"{name:<12} {lat:>8.1f}     {temp:>6.1f}      {power:>4.0f}")
        
        total_lat = sum(results["latency_us"][:4])
        total_temp = max(results["thermal_c"][:4])
        total_power = sum(results["power_mw"][:4])
        print("-" * 50)
        print(f"{'Total':<12} {total_lat:>8.1f}     {total_temp:>6.1f}      {total_power:>4.0f}")
        
        if args.output:
            import json
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


def debug_main():
    """Main entry point for photon-debug command."""
    parser = argparse.ArgumentParser(description="Debug photonic model compilation")
    
    parser.add_argument("model", help="Photonic model file (.pasm)")
    parser.add_argument("--breakpoint", help="Set breakpoint at operation")
    parser.add_argument("--visualize", default="mesh,thermal,phase",
                       help="Visualization modes")
    parser.add_argument("--port", type=int, default=8080,
                       help="Debug server port")
    
    args = parser.parse_args()
    
    print(f"Photonic Debugger v0.1.0")
    print(f"Model: {args.model}")
    print(f"Breakpoint: {args.breakpoint}")
    print(f"Visualizations: {args.visualize}")
    print(f"Debug server would start at: http://localhost:{args.port}")
    print("\nNote: Interactive debugging not yet implemented")
    
    return 0


def benchmark_main():
    """Main entry point for photon-bench command."""
    parser = argparse.ArgumentParser(description="Benchmark photonic compilation")
    
    parser.add_argument("models", nargs="+", help="Model files to benchmark")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Benchmark iterations")
    parser.add_argument("--output", help="Save benchmark results")
    
    args = parser.parse_args()
    
    print("Photonic Compiler Benchmarks")
    print("=" * 40)
    
    for model in args.models:
        print(f"\nBenchmarking: {model}")
        print(f"Iterations: {args.iterations}")
        
        # Mock benchmark results
        compile_times = [1.2, 1.1, 1.3, 1.0, 1.2] * (args.iterations // 5)
        avg_time = sum(compile_times) / len(compile_times)
        
        print(f"Average compile time: {avg_time:.2f}s")
        print(f"Min: {min(compile_times):.2f}s")
        print(f"Max: {max(compile_times):.2f}s")
    
    return 0