# ADR-0002: Photonic Dialect Architecture

**Status**: Accepted  
**Date**: 2024-01-20  
**Deciders**: Daniel Schmidt, Photonic Architecture Team

## Context

The photon-mlir-bridge needs a custom MLIR dialect to represent photonic computing operations and constraints. This dialect must capture:
- Photonic matrix operations with phase constraints
- Optical power distribution requirements  
- Thermal compensation primitives
- Wavelength-dependent operations
- Device-specific capabilities

The dialect design impacts all compiler optimizations and hardware targeting.

## Decision

We have decided to implement a **layered photonic dialect architecture** with three main operation categories:

### 1. High-Level Photonic Operations
```mlir
photonic.matmul %input, %weights {wavelength = 1550 : i32} : 
    tensor<1024x512xf32>, tensor<512x256xf32> -> tensor<1024x256xf32>

photonic.convolution %input, %kernel {mesh_config = "butterfly"} :
    tensor<1x224x224x3xf32>, tensor<64x7x7x3xf32> -> tensor<1x112x112x64xf32>
```

### 2. Low-Level Photonic Primitives
```mlir
photonic.phase_shift %input, %phase : tensor<64x64xcomplex<f32>>, tensor<64x64xf32>
photonic.beam_splitter %input1, %input2 : tensor<32xcomplex<f32>>, tensor<32xcomplex<f32>>
photonic.photodetector %optical : tensor<64xcomplex<f32>> -> tensor<64xf32>
```

### 3. Control and Calibration Operations
```mlir
photonic.thermal_calibrate %sensor : !photonic.sensor -> !photonic.compensation
photonic.power_balance %inputs : !photonic.optical_array -> !photonic.optical_array
```

## Consequences

### Positive
- **Abstraction Levels**: Multiple abstraction levels enable both high-level optimization and low-level control
- **Hardware Flexibility**: Operations can be lowered to different photonic architectures
- **Optimization Opportunities**: Rich semantic information enables photonic-specific optimizations
- **Composability**: Operations can be combined to build complex photonic circuits
- **Type Safety**: Custom types (optical, complex amplitudes) prevent invalid transformations

### Negative
- **Complexity**: Multiple abstraction levels increase compiler complexity
- **Learning Curve**: Domain-specific operations require photonics expertise
- **Maintenance**: Custom dialect requires ongoing maintenance and documentation
- **Debugging**: Harder to debug issues in custom dialect operations

## Alternatives Considered

### 1. Extend Existing Dialects Only
- **Pros**: Reuse existing operations, simpler implementation
- **Cons**: Cannot capture photonic-specific constraints and semantics

### 2. Single-Level Photonic Dialect
- **Pros**: Simpler design, easier to implement
- **Cons**: Either too high-level (loses hardware control) or too low-level (verbose)

### 3. Multiple Separate Dialects
- **Pros**: Clear separation of concerns
- **Cons**: Complex inter-dialect relationships, conversion overhead

## Key Design Principles

1. **Progressive Lowering**: High-level operations lower to primitives, then to hardware
2. **Composability**: Operations can be combined into larger circuits
3. **Hardware Abstraction**: Operations abstract over specific photonic implementations
4. **Constraint Modeling**: Encode photonic constraints (phase limits, power budgets) in types
5. **Optimization Friendly**: Design enables pattern matching and rewriting

## Type System Design

```mlir
// Custom photonic types
!photonic.optical<wavelength=1550, power=1.0>
!photonic.complex_amplitude<precision=f32>
!photonic.phase<range=[-π, π]>
!photonic.mesh<size=64x64, topology="butterfly">
!photonic.sensor<type="thermal", location=[x, y]>
```

## Operation Categories

### Matrix Operations
- `photonic.matmul`: Matrix multiplication with photonic constraints
- `photonic.decompose`: Matrix decomposition for mesh mapping
- `photonic.butterfly`: Butterfly network operation

### Signal Processing
- `photonic.fft`: Photonic Fourier transform
- `photonic.convolution`: Convolution with optical kernels
- `photonic.correlation`: Cross-correlation operation

### Control Operations
- `photonic.calibrate`: Calibration sequence generation
- `photonic.compensate`: Thermal/phase compensation
- `photonic.monitor`: Performance monitoring insertion

## Implementation Phases

1. **Phase 1**: Core matrix operations and basic lowering
2. **Phase 2**: Signal processing operations and optimization passes
3. **Phase 3**: Advanced control operations and multi-device support

## Success Criteria

- Successful compilation of standard ML models (ResNet, BERT)
- Effective optimization of photonic-specific metrics (phase shifts, power consumption)
- Clean lowering to multiple hardware targets
- Maintainable codebase with comprehensive tests

## References

- [MLIR Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)
- [MLIR OpDefinition](https://mlir.llvm.org/docs/OpDefinitions/)
- [Photonic Computing Fundamentals](https://photonic-computing.org/fundamentals/)