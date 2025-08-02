# ADR-0001: MLIR Framework Selection

**Status**: Accepted  
**Date**: 2024-01-15  
**Deciders**: Daniel Schmidt, Photonic Compiler Team

## Context

The photon-mlir-bridge project requires a compiler infrastructure that can handle:
- Multiple input formats (PyTorch, TensorFlow, ONNX)
- Progressive lowering through multiple abstraction levels
- Domain-specific optimizations for photonic computing
- Extensibility for new photonic architectures
- Integration with existing LLVM toolchain

We evaluated several compiler frameworks for this purpose.

## Decision

We have decided to use **LLVM's MLIR (Multi-Level Intermediate Representation)** as the core compiler infrastructure for photon-mlir-bridge.

MLIR provides:
- Flexible dialect system for domain-specific abstractions
- Robust pass infrastructure for optimization
- Strong typing system with SSA form
- Excellent integration with LLVM ecosystem
- Active development and community support

## Consequences

### Positive
- **Extensibility**: Easy to define custom dialects for photonic operations
- **Optimization**: Rich pass infrastructure enables sophisticated optimizations
- **Interoperability**: Seamless integration with existing LLVM tools
- **Community**: Large developer community and extensive documentation
- **Performance**: Proven performance in production compiler systems
- **Type Safety**: Strong type system prevents many compiler bugs

### Negative
- **Learning Curve**: MLIR has a steep learning curve for new contributors
- **Complexity**: Can be overkill for simple compilation tasks
- **Build Dependencies**: Requires LLVM/MLIR as a dependency (~1GB+)
- **API Stability**: MLIR APIs are still evolving (though stabilizing)

## Alternatives Considered

### 1. Custom AST-based Compiler
- **Pros**: Full control, minimal dependencies
- **Cons**: Significant development effort, no reuse of existing infrastructure

### 2. TVM (Tensor Virtual Machine)
- **Pros**: ML-focused, good performance
- **Cons**: Limited photonic-specific features, different design philosophy

### 3. XLA (Accelerated Linear Algebra)
- **Pros**: Battle-tested for ML workloads
- **Cons**: Tightly coupled to TensorFlow, limited extensibility

### 4. LLVM IR Directly
- **Pros**: Mature, well-understood
- **Cons**: Too low-level for high-level ML operations, difficult to maintain

## Implementation Notes

- Start with existing MLIR dialects (tensor, linalg, arith)
- Implement photonic dialect with custom operations
- Use MLIR's progressive lowering approach
- Leverage MLIR's pattern rewriting for optimizations

## Success Criteria

- Support compilation from major ML frameworks (PyTorch, TensorFlow, ONNX)
- Achieve competitive compilation times (<30s for ResNet-50)
- Enable photonic-specific optimizations (phase minimization, thermal compensation)
- Maintain extensibility for new photonic architectures

## References

- [MLIR: Multi-Level Intermediate Representation](https://mlir.llvm.org/)
- [MLIR Rationale](https://mlir.llvm.org/docs/Rationale/)
- [Building a Compiler with MLIR](https://mlir.llvm.org/docs/Tutorials/Toy/)