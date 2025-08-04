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