//===- PhotonicPasses.cpp - Photonic transformation passes --------------===//
//
// This file implements photonic-specific transformation passes.
//
//===----------------------------------------------------------------------===//

#include "photon/transforms/PhotonicPasses.h"
#include "photon/dialects/PhotonicDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::photonic;

#define DEBUG_TYPE "photonic-passes"

//===----------------------------------------------------------------------===//
// Matrix Decomposition Pass
//===----------------------------------------------------------------------===//

namespace {
struct MatrixDecompositionPass : public PassWrapper<MatrixDecompositionPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    // Walk through all operations and decompose large matrix operations
    func.walk([&](Operation* op) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        decomposeMatmul(matmul);
      }
    });
  }
  
private:
  void decomposeMatmul(linalg::MatmulOp matmul) {
    auto lhsType = matmul.getInputs()[0].getType().cast<ShapedType>();
    auto rhsType = matmul.getInputs()[1].getType().cast<ShapedType>();
    
    // Check if matrices are too large for single photonic mesh (64x64)
    if (lhsType.getDimSize(1) > 64 || rhsType.getDimSize(0) > 64) {
      LLVM_DEBUG(llvm::dbgs() << "Decomposing large matrix multiplication\n");
      
      OpBuilder builder(matmul);
      
      // Create photonic unfold operation
      auto unfoldOp = builder.create<UnfoldOp>(
        matmul.getLoc(),
        matmul.getInputs()[0].getType(),
        matmul.getInputs()[0]
      );
      
      // Create photonic matrix multiply with mesh configuration
      auto photonicMatmul = builder.create<MatMulOp>(
        matmul.getLoc(),
        matmul.getResult(0).getType(),
        unfoldOp.getResult(),
        matmul.getInputs()[1],
        builder.getI32IntegerAttr(1550), // wavelength
        builder.getStringAttr("butterfly") // mesh config
      );
      
      // Create fold operation to reshape result
      auto foldOp = builder.create<FoldOp>(
        matmul.getLoc(),
        matmul.getResult(0).getType(),
        photonicMatmul.getResult()
      );
      
      // Replace original operation
      matmul.getResult(0).replaceAllUsesWith(foldOp.getResult());
      matmul.erase();
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::photonic::createMatrixDecompositionPass() {
  return std::make_unique<MatrixDecompositionPass>();
}

//===----------------------------------------------------------------------===//
// Phase Optimization Pass  
//===----------------------------------------------------------------------===//

namespace {
struct PhaseOptimizationPass : public PassWrapper<PhaseOptimizationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    // Collect all phase shift operations
    SmallVector<PhaseShiftOp, 8> phaseOps;
    func.walk([&](PhaseShiftOp op) {
      phaseOps.push_back(op);
    });
    
    // Optimize phase shifts to minimize total phase changes
    optimizePhaseShifts(phaseOps);
  }
  
private:
  void optimizePhaseShifts(ArrayRef<PhaseShiftOp> phaseOps) {
    LLVM_DEBUG(llvm::dbgs() << "Optimizing " << phaseOps.size() << " phase shifts\n");
    
    // Simple optimization: combine consecutive phase shifts
    for (auto& op : phaseOps) {
      if (auto prevOp = op.getInput().getDefiningOp<PhaseShiftOp>()) {
        // Combine phase shifts: phase1 + phase2
        OpBuilder builder(op);
        auto combinedPhase = prevOp.getPhaseRadians().convertToDouble() + 
                           op.getPhaseRadians().convertToDouble();
        
        auto newOp = builder.create<PhaseShiftOp>(
          op.getLoc(),
          op.getResult().getType(),
          prevOp.getInput(),
          builder.getF32FloatAttr(combinedPhase)
        );
        
        op.getResult().replaceAllUsesWith(newOp.getResult());
        op.erase();
        prevOp.erase();
      }
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::photonic::createPhaseOptimizationPass() {
  return std::make_unique<PhaseOptimizationPass>();
}

//===----------------------------------------------------------------------===//
// Thermal Compensation Pass
//===----------------------------------------------------------------------===//

namespace {
struct ThermalCompensationPass : public PassWrapper<ThermalCompensationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    OpBuilder builder(&func.getBody().front(), func.getBody().front().begin());
    
    // Insert thermal compensation at function entry
    auto thermalOp = builder.create<ThermalCompensationOp>(
      func.getLoc(),
      builder.getNoneType(),
      Value{}, // No input tensor needed
      builder.getF32FloatAttr(0.1), // max_drift
      builder.getI32IntegerAttr(100) // calibration_interval_ms
    );
    
    LLVM_DEBUG(llvm::dbgs() << "Inserted thermal compensation operation\n");
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::photonic::createThermalCompensationPass() {
  return std::make_unique<ThermalCompensationPass>();
}

//===----------------------------------------------------------------------===//
// Power Balancing Pass
//===----------------------------------------------------------------------===//

namespace {
struct PowerBalancingPass : public PassWrapper<PowerBalancingPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    // Find all optical operations and ensure balanced power
    func.walk([&](Operation* op) {
      if (isa<MatMulOp, PhaseShiftOp>(op)) {
        balancePower(op);
      }
    });
  }
  
private:
  void balancePower(Operation* op) {
    LLVM_DEBUG(llvm::dbgs() << "Balancing power for operation: " << op->getName() << "\n");
    
    // Insert power normalization operations as needed
    OpBuilder builder(op);
    
    // Create optical encoding to normalize power
    for (auto operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        if (!isa<OpticalEncodingOp>(definingOp)) {
          auto encodingOp = builder.create<OpticalEncodingOp>(
            op->getLoc(),
            operand.getType(),
            operand,
            builder.getI32IntegerAttr(1550) // wavelength
          );
          op->setOperand(operand.use_begin()->getOperandNumber(), encodingOp.getResult());
        }
      }
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::photonic::createPowerBalancingPass() {
  return std::make_unique<PowerBalancingPass>();
}

//===----------------------------------------------------------------------===//
// Photonic Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct PhotonicLoweringPass : public PassWrapper<PhotonicLoweringPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    // Convert standard operations to photonic dialect
    func.walk([&](Operation* op) {
      if (auto matmul = dyn_cast<linalg::MatmulOp>(op)) {
        lowerMatmul(matmul);
      }
    });
  }
  
private:
  void lowerMatmul(linalg::MatmulOp matmul) {
    OpBuilder builder(matmul);
    
    // Create photonic matrix multiplication
    auto photonicMatmul = builder.create<MatMulOp>(
      matmul.getLoc(),
      matmul.getResult(0).getType(),
      matmul.getInputs()[0],
      matmul.getInputs()[1],
      builder.getI32IntegerAttr(1550), // wavelength
      builder.getStringAttr("butterfly") // mesh config
    );
    
    matmul.getResult(0).replaceAllUsesWith(photonicMatmul.getResult());
    matmul.erase();
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::photonic::createPhotonicLoweringPass() {
  return std::make_unique<PhotonicLoweringPass>();
}

//===----------------------------------------------------------------------===//
// Mesh Optimization Pass
//===----------------------------------------------------------------------===//

namespace {
struct MeshOptimizationPass : public PassWrapper<MeshOptimizationPass, OperationPass<func::FuncOp>> {
  void runOnOperation() override {
    auto func = getOperation();
    
    // Analyze mesh utilization and optimize layout
    analyzeMeshUtilization(func);
    optimizeMeshLayout(func);
  }
  
private:
  void analyzeMeshUtilization(func::FuncOp func) {
    LLVM_DEBUG(llvm::dbgs() << "Analyzing mesh utilization\n");
    
    int totalMatmuls = 0;
    func.walk([&](MatMulOp op) {
      totalMatmuls++;
    });
    
    LLVM_DEBUG(llvm::dbgs() << "Found " << totalMatmuls << " matrix multiplications\n");
  }
  
  void optimizeMeshLayout(func::FuncOp func) {
    // Optimize mesh configuration based on operation patterns
    func.walk([&](MatMulOp op) {
      // Analyze input shapes and optimize mesh configuration
      auto lhsType = op.getLhs().getType().cast<ShapedType>();
      auto rhsType = op.getRhs().getType().cast<ShapedType>();
      
      if (lhsType.hasStaticShape() && rhsType.hasStaticShape()) {
        // Choose optimal mesh topology based on matrix dimensions
        std::string optimalMesh = chooseMeshTopology(
          lhsType.getDimSize(1), 
          rhsType.getDimSize(0)
        );
        
        // Update mesh configuration
        op->setAttr("mesh_config", StringAttr::get(getContext(), optimalMesh));
      }
    });
  }
  
  std::string chooseMeshTopology(int64_t m, int64_t n) {
    // Simple heuristic for mesh topology selection
    if (m <= 32 && n <= 32) {
      return "crossbar";
    } else if (m <= 64 && n <= 64) {
      return "butterfly";
    } else {
      return "mesh_of_trees";
    }
  }
};
} // anonymous namespace

std::unique_ptr<Pass> mlir::photonic::createMeshOptimizationPass() {
  return std::make_unique<MeshOptimizationPass>();
}

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

void mlir::photonic::registerPhotonicPasses() {
  PassRegistration<MatrixDecompositionPass>(
    "photonic-matrix-decomposition",
    "Decompose large matrices for photonic mesh mapping"
  );
  
  PassRegistration<PhaseOptimizationPass>(
    "photonic-phase-optimization", 
    "Optimize phase shift requirements"
  );
  
  PassRegistration<ThermalCompensationPass>(
    "photonic-thermal-compensation",
    "Insert thermal compensation operations"
  );
  
  PassRegistration<PowerBalancingPass>(
    "photonic-power-balancing",
    "Balance optical power across channels"
  );
  
  PassRegistration<PhotonicLoweringPass>(
    "photonic-lowering",
    "Convert standard operations to photonic dialect"
  );
  
  PassRegistration<MeshOptimizationPass>(
    "photonic-mesh-optimization",
    "Optimize mesh utilization and layout"
  );
}