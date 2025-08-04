//===- PhotonicDialect.h - Photonic dialect -------------------*- C++ -*-===//
//
// This file declares the Photonic dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_DIALECT_H
#define PHOTONIC_DIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "photon/dialects/PhotonicDialect.h.inc"

#define GET_OP_CLASSES
#include "photon/dialects/PhotonicOps.h.inc"

namespace mlir {
namespace photonic {

/// Photonic device configuration for target-specific compilation
struct PhotonicTargetConfig {
  enum class Device {
    LIGHTMATTER_ENVISE,
    MIT_PHOTONIC_PROCESSOR,
    CUSTOM_RESEARCH_CHIP
  };
  
  enum class Precision {
    INT8,
    INT16,
    FP16,
    FP32
  };
  
  Device device = Device::LIGHTMATTER_ENVISE;
  Precision precision = Precision::INT8;
  std::pair<int, int> array_size = {64, 64};
  int wavelength_nm = 1550;
  double max_phase_drift = 0.1; // radians
  int calibration_interval_ms = 100;
  bool enable_thermal_compensation = true;
  std::string mesh_topology = "butterfly";
};

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_DIALECT_H