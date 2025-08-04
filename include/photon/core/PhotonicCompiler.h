//===- PhotonicCompiler.h - Main compiler interface -----------*- C++ -*-===//
//
// This file declares the main PhotonicCompiler class.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_COMPILER_H
#define PHOTONIC_COMPILER_H

#include "photon/dialects/PhotonicDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>

namespace mlir {
namespace photonic {

/// Main photonic compiler class for end-to-end compilation
class PhotonicCompiler {
public:
  PhotonicCompiler();
  ~PhotonicCompiler();

  /// Load ONNX model and convert to MLIR
  LogicalResult loadONNX(llvm::StringRef filename);
  
  /// Load PyTorch JIT model
  LogicalResult loadPyTorch(llvm::StringRef filename);
  
  /// Set target configuration
  void setTargetConfig(const PhotonicTargetConfig& config);
  
  /// Compile to photonic assembly
  LogicalResult compile();
  
  /// Generate device code
  LogicalResult codegen(llvm::StringRef outputFile);
  
  /// Get optimization report
  std::string getOptimizationReport() const;
  
  /// Get compiled module
  ModuleOp getModule() const { return module_; }
  
  /// Get MLIR context
  MLIRContext* getContext() const { return context_.get(); }

private:
  std::unique_ptr<MLIRContext> context_;
  ModuleOp module_;
  PhotonicTargetConfig config_;
  
  /// Build compilation pipeline
  void buildPipeline(PassManager& pm);
  
  /// Load model from various formats
  LogicalResult loadFromFormat(llvm::StringRef filename, llvm::StringRef format);
  
  /// Optimization statistics
  struct OptimizationStats {
    size_t originalFlops = 0;
    size_t photonicMacs = 0;
    size_t totalPhaseShifts = 0;
    double estimatedSpeedup = 1.0;
    double energyReduction = 0.0;
  } stats_;
};

/// Convenience functions for simple compilation
namespace api {

/// Load ONNX model and compile to photonic assembly
LogicalResult compileONNX(llvm::StringRef inputFile, 
                         llvm::StringRef outputFile,
                         const PhotonicTargetConfig& config = {});

/// Load PyTorch model and compile
LogicalResult compilePyTorch(llvm::StringRef inputFile,
                            llvm::StringRef outputFile, 
                            const PhotonicTargetConfig& config = {});

} // namespace api

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_COMPILER_H