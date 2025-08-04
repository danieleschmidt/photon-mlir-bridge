//===- PhotonicPasses.h - Photonic transformation passes -----*- C++ -*-===//
//
// This file declares photonic-specific transformation passes.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_PASSES_H
#define PHOTONIC_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {
namespace photonic {

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Register all photonic passes
void registerPhotonicPasses();

//===----------------------------------------------------------------------===//
// Pass Declarations
//===----------------------------------------------------------------------===//

/// Creates a pass that decomposes large matrices for photonic mesh mapping
std::unique_ptr<Pass> createMatrixDecompositionPass();

/// Creates a pass that optimizes phase shift requirements
std::unique_ptr<Pass> createPhaseOptimizationPass();

/// Creates a pass that inserts thermal compensation operations
std::unique_ptr<Pass> createThermalCompensationPass();

/// Creates a pass that balances optical power across channels
std::unique_ptr<Pass> createPowerBalancingPass();

/// Creates a pass that converts standard operations to photonic dialect
std::unique_ptr<Pass> createPhotonicLoweringPass();

/// Creates a pass that optimizes mesh utilization
std::unique_ptr<Pass> createMeshOptimizationPass();

//===----------------------------------------------------------------------===//
// Pass Options
//===----------------------------------------------------------------------===//

struct PhotonicPassOptions {
  // Matrix decomposition options
  int max_matrix_size = 64;
  std::string decomposition_strategy = "butterfly";
  
  // Phase optimization options
  double phase_precision = 0.01; // radians
  bool enable_phase_caching = true;
  
  // Thermal compensation options  
  bool enable_thermal_model = true;
  double thermal_sensitivity = 0.1; // rad/°C
  int calibration_points = 100;
  
  // Power balancing options
  double power_tolerance = 0.05; // 5% variation
  bool normalize_power = true;
};

} // namespace photonic
} // namespace mlir

#include "photon/transforms/PhotonicPasses.h.inc"

#endif // PHOTONIC_PASSES_H