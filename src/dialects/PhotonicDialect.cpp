//===- PhotonicDialect.cpp - Photonic dialect ---------------------------===//
//
// This file implements the Photonic dialect.
//
//===----------------------------------------------------------------------===//

#include "photon/dialects/PhotonicDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::photonic;

#include "photon/dialects/PhotonicDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Photonic dialect
//===----------------------------------------------------------------------===//

void PhotonicDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "photon/dialects/PhotonicOps.cpp.inc"
  >();
}

//===----------------------------------------------------------------------===//
// MatMulOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulOp::verify() {
  auto lhsType = getLhs().getType().cast<TensorType>();
  auto rhsType = getRhs().getType().cast<TensorType>();
  auto resultType = getResult().getType().cast<TensorType>();
  
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2) {
    return emitOpError("matrix operands must have rank 2");
  }
  
  auto lhsShape = lhsType.getShape();
  auto rhsShape = rhsType.getShape();
  auto resultShape = resultType.getShape();
  
  if (lhsShape[1] != rhsShape[0]) {
    return emitOpError("matrix inner dimensions must match");
  }
  
  if (resultShape[0] != lhsShape[0] || resultShape[1] != rhsShape[1]) {
    return emitOpError("result shape must match expected output dimensions");
  }
  
  // Validate wavelength is in valid range for silicon photonics
  int wavelength = getWavelength();
  if (wavelength < 1200 || wavelength > 1700) {
    return emitOpError("wavelength must be in range 1200-1700 nm for silicon photonics");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// PhaseShiftOp
//===----------------------------------------------------------------------===//

LogicalResult PhaseShiftOp::verify() {
  float phase = getPhaseRadians().convertToFloat();
  
  // Warn if phase shift is very large (may indicate inefficient encoding)
  if (std::abs(phase) > 4 * M_PI) {
    emitWarning("large phase shift detected, consider normalizing to [0, 2π]");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// ThermalCompensationOp
//===----------------------------------------------------------------------===//

LogicalResult ThermalCompensationOp::verify() {
  float maxDrift = getMaxDrift().convertToFloat();
  int calibrationInterval = getCalibrationIntervalMs();
  
  if (maxDrift < 0 || maxDrift > M_PI) {
    return emitOpError("max_drift must be in range [0, π] radians");
  }
  
  if (calibrationInterval < 1 || calibrationInterval > 10000) {
    return emitOpError("calibration_interval_ms must be in range [1, 10000]");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// OpticalEncodingOp
//===----------------------------------------------------------------------===//

LogicalResult OpticalEncodingOp::verify() {
  int wavelength = getWavelengthNm();
  
  if (wavelength < 1200 || wavelength > 1700) {
    return emitOpError("wavelength must be in range 1200-1700 nm for silicon photonics");
  }
  
  return success();
}

//===----------------------------------------------------------------------===//
// Advanced Quantum-Photonic Operations
//===----------------------------------------------------------------------===//

LogicalResult QuantumPhaseGateOp::verify() {
  float phase = getPhaseRadians().convertToFloat();
  int32_t qubitIndex = getQubitIndex();
  
  if (phase < 0 || phase > 2 * M_PI) {
    return emitOpError("quantum phase gate phase must be in range [0, 2π]");
  }
  
  if (qubitIndex < 0) {
    return emitOpError("qubit index must be non-negative");
  }
  
  return success();
}

LogicalResult ThermalAwareSchedulingOp::verify() {
  int32_t window = getSchedulingWindowMs();
  
  if (window <= 0 || window > 60000) {
    return emitOpError("scheduling window must be in range (0, 60000]ms");
  }
  
  // Validate thermal map dimensions
  auto thermalMap = getThermalMap();
  if (thermalMap.size() == 0) {
    return emitOpError("thermal map cannot be empty");
  }
  
  return success();
}

LogicalResult WavelengthMultiplexOp::verify() {
  auto inputs = getInputs();
  auto wavelengths = getWavelengthsNm();
  
  if (inputs.size() != wavelengths.size()) {
    return emitOpError("number of inputs must match number of wavelengths");
  }
  
  // Check for wavelength conflicts
  std::set<int32_t> uniqueWavelengths;
  for (auto wavelength : wavelengths) {
    auto wavelengthValue = wavelength.cast<IntegerAttr>().getInt();
    if (wavelengthValue < 1200 || wavelengthValue > 1700) {
      return emitOpError("wavelengths must be in range 1200-1700 nm");
    }
    
    if (uniqueWavelengths.count(wavelengthValue)) {
      return emitOpError("duplicate wavelengths not allowed in multiplexing");
    }
    uniqueWavelengths.insert(wavelengthValue);
  }
  
  return success();
}

LogicalResult MeshCalibrationOp::verify() {
  float fidelity = getTargetFidelity().convertToFloat();
  StringRef method = getCalibrationMethod();
  
  if (fidelity <= 0 || fidelity > 1.0) {
    return emitOpError("target fidelity must be in range (0, 1]");
  }
  
  if (method != "self_configuration" && method != "reference_beam" && 
      method != "iterative_optimization") {
    return emitOpError("unsupported calibration method. Supported: "
                      "self_configuration, reference_beam, iterative_optimization");
  }
  
  return success();
}

LogicalResult PowerBalancingOp::verify() {
  float targetPower = getTargetPowerMw().convertToFloat();
  float tolerance = getPowerTolerance().convertToFloat();
  
  if (targetPower <= 0 || targetPower > 100.0) {
    return emitOpError("target power must be in range (0, 100] mW");
  }
  
  if (tolerance <= 0 || tolerance >= 1.0) {
    return emitOpError("power tolerance must be in range (0, 1)");
  }
  
  return success();
}

LogicalResult CrosstalkMitigationOp::verify() {
  float strength = getMitigationStrength().convertToFloat();
  
  if (strength < 0 || strength > 1.0) {
    return emitOpError("mitigation strength must be in range [0, 1]");
  }
  
  auto crosstalkMatrix = getCrosstalkMatrix();
  if (crosstalkMatrix.size() == 0) {
    return emitOpError("crosstalk matrix cannot be empty");
  }
  
  return success();
}