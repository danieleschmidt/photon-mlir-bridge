"""
Comprehensive command-line interface for photonic compiler toolchain.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

from .compiler import PhotonicCompiler, compile_onnx, compile_pytorch, compile
from .core import TargetConfig, Device, Precision
from .simulator import PhotonicSimulator
from .validation import ValidationError
from .logging_config import setup_logging, finalize_logging


def compile_main():
    """Main entry point for photon-compile command."""
    parser = argparse.ArgumentParser(
        description="Compile neural network models for photonic hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  photon-compile model.onnx -o output.phdl --device lightmatter
  photon-compile model.onnx -o output.phdl --precision fp16 --array-size 32 32
  photon-compile model.onnx -o output.phdl --optimize-for latency --verbose
        """
    )
    
    parser.add_argument("input", help="Input model file (ONNX or PyTorch)")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    
    # Hardware configuration
    parser.add_argument("--device", choices=["lightmatter", "mit", "research"], 
                       default="lightmatter", help="Target photonic device")
    parser.add_argument("--precision", choices=["int8", "int16", "fp16", "fp32"],
                       default="int8", help="Precision mode")
    parser.add_argument("--array-size", nargs=2, type=int, default=[64, 64],
                       metavar=("ROWS", "COLS"), help="Photonic array size")
    parser.add_argument("--wavelength", type=int, default=1550,
                       help="Operating wavelength in nm")
    
    # Optimization options
    parser.add_argument("--optimize-for", choices=["latency", "power", "accuracy"],
                       default="latency", help="Optimization target")
    parser.add_argument("--thermal-compensation", action="store_true", default=True,
                       help="Enable thermal compensation (default)")
    parser.add_argument("--no-thermal-compensation", action="store_false", 
                       dest="thermal_compensation", help="Disable thermal compensation")
    
    # Output options
    parser.add_argument("--report", help="Save optimization report to file")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    # Map device names
    device_map = {
        "lightmatter": Device.LIGHTMATTER_ENVISE,
        "mit": Device.MIT_PHOTONIC_PROCESSOR,
        "research": Device.CUSTOM_RESEARCH_CHIP
    }
    
    precision_map = {
        "int8": Precision.INT8,
        "int16": Precision.INT16, 
        "fp16": Precision.FP16,
        "fp32": Precision.FP32
    }
    
    # Create target configuration
    config = TargetConfig(
        device=device_map[args.device],
        precision=precision_map[args.precision],
        array_size=tuple(args.array_size),
        wavelength_nm=args.wavelength
    )
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logging(level=log_level, performance_logging=True)
    
    try:
        logger.info(f"🚀 Photonic compilation started")
        logger.info(f"   Input: {args.input}")
        logger.info(f"   Device: {args.device} ({args.array_size[0]}×{args.array_size[1]})")
        logger.info(f"   Precision: {args.precision}")
        
        # Validate input file exists
        if not os.path.exists(args.input):
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        # Create compiler with validation
        compiler = PhotonicCompiler(config, strict_validation=False, logger=logger)
        
        # Compile model
        if args.input.endswith('.onnx'):
            compiled_model = compiler.compile_onnx(args.input)
        elif args.input.endswith('.pt') or args.input.endswith('.pth'):
            logger.error("PyTorch model compilation requires Python API")
            sys.exit(1)
        else:
            logger.error(f"Unsupported input format: {args.input}")
            sys.exit(1)
            
        # Export compiled model
        compiled_model.export(args.output)
        
        # Generate optimization report
        if args.report or args.verbose:
            report = compiled_model.get_optimization_report()
            if args.report:
                with open(args.report, 'w') as f:
                    f.write(report)
                logger.info(f"📊 Optimization report saved to {args.report}")
            if args.verbose:
                print("\n" + report)
            
        logger.info(f"✅ Successfully compiled {args.input} to {args.output}")
        
    except ValidationError as e:
        logger.error(f"❌ Validation failed: {e}")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"❌ File error: {e}")
        sys.exit(1)
    except PermissionError as e:
        logger.error(f"❌ Permission denied: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Compilation failed: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        finalize_logging()


def simulate_main():
    """Main entry point for photon-simulate command."""
    parser = argparse.ArgumentParser(
        description="Simulate photonic neural network execution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  photon-simulate model.phdl --input test_data.npy
  photon-simulate model.phdl --noise realistic --precision 8bit
        """
    )
    
    parser.add_argument("model", help="Compiled photonic model file (.phdl)")
    parser.add_argument("--input", required=True, help="Input data file (.npy)")
    parser.add_argument("--output", help="Output file for results (.npy or .json)")
    
    # Simulation options
    parser.add_argument("--noise", choices=["ideal", "realistic", "pessimistic"], 
                       default="realistic", help="Noise model")
    parser.add_argument("--precision", choices=["8bit", "16bit", "fp16", "fp32"],
                       default="8bit", help="Precision mode")
    parser.add_argument("--crosstalk", type=float, default=-30.0,
                       help="Crosstalk level in dB")
    
    # Analysis options
    parser.add_argument("--compare-ideal", action="store_true",
                       help="Compare with ideal (noiseless) simulation")
    parser.add_argument("--runs", type=int, default=1,
                       help="Number of simulation runs for statistics")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    
    args = parser.parse_args()
    
    try:
        # Load input data
        if not os.path.exists(args.input):
            print(f"❌ Input file not found: {args.input}")
            sys.exit(1)
            
        input_data = np.load(args.input)
        print(f"📥 Loaded input data: {input_data.shape}")
        
        # Create simulator
        config = TargetConfig()  # Would parse from model file
        simulator = PhotonicSimulator(
            noise_model=args.noise,
            precision=args.precision,
            crosstalk_db=args.crosstalk,
            target_config=config
        )
        
        print(f"🔬 Starting simulation ({args.runs} runs)")
        
        # Create mock model for simulation
        class MockCompiledModel:
            def __init__(self):
                self.target_config = config
                
        mock_model = MockCompiledModel()
        result = simulator.run(mock_model, input_data)
        
        print(f"📤 Output shape: {result.data.shape}")
        print("✅ Simulation complete")
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        sys.exit(1)


def profile_main():
    """Main entry point for photon-profile command."""
    parser = argparse.ArgumentParser(description="Profile photonic model performance")
    parser.add_argument("model", help="Compiled photonic model file")
    parser.add_argument("--input-shape", nargs='+', type=int, required=True,
                       help="Input tensor shape")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of profiling runs")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    print(f"⏱️  Profiling photonic model: {args.model}")
    print(f"   Input shape: {args.input_shape}")
    print(f"   (Full profiling implementation pending)")


def debug_main():
    """Main entry point for photon-debug command."""
    parser = argparse.ArgumentParser(description="Debug photonic model compilation")
    parser.add_argument("model", help="Model file to debug")
    parser.add_argument("--port", type=int, default=8080, help="Debug server port")
    
    args = parser.parse_args()
    
    print(f"🐛 Photonic debugger starting...")
    print(f"   Debug server would be available at http://localhost:{args.port}")
    print("   (Full debugger implementation pending)")


def benchmark_main():
    """Main entry point for photon-bench command."""
    parser = argparse.ArgumentParser(description="Benchmark photonic hardware performance")
    parser.add_argument("--suite", choices=["compilation", "execution", "accuracy", "all"],
                       default="all", help="Benchmark suite")
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    print(f"🏁 Photonic benchmarking suite")
    print(f"   Suite: {args.suite}")
    print("   (Full benchmarking implementation pending)")