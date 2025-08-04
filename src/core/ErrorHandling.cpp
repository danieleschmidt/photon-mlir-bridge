//===- ErrorHandling.cpp - Error handling implementation ---------------===//
//
// This file implements error handling utilities.
//
//===----------------------------------------------------------------------===//

#include "photon/core/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <sstream>
#include <regex>
#include <cmath>

using namespace mlir;
using namespace mlir::photonic;

thread_local std::string ErrorContext::current_context_ = "";

std::string errorCodeToString(PhotonicErrorCode code) {
  switch (code) {
    case PhotonicErrorCode::SUCCESS: return "Success";
    case PhotonicErrorCode::FILE_NOT_FOUND: return "File not found";
    case PhotonicErrorCode::INVALID_MODEL_FORMAT: return "Invalid model format";
    case PhotonicErrorCode::UNSUPPORTED_OPERATION: return "Unsupported operation";
    case PhotonicErrorCode::HARDWARE_CONSTRAINT_VIOLATION: return "Hardware constraint violation";
    case PhotonicErrorCode::PHASE_RANGE_EXCEEDED: return "Phase range exceeded";
    case PhotonicErrorCode::WAVELENGTH_OUT_OF_RANGE: return "Wavelength out of range";
    case PhotonicErrorCode::ARRAY_SIZE_MISMATCH: return "Array size mismatch";
    case PhotonicErrorCode::THERMAL_CONSTRAINT_VIOLATION: return "Thermal constraint violation";
    case PhotonicErrorCode::COMPILATION_FAILED: return "Compilation failed";
    case PhotonicErrorCode::CODEGEN_FAILED: return "Code generation failed";
    case PhotonicErrorCode::MEMORY_ALLOCATION_FAILED: return "Memory allocation failed";
    case PhotonicErrorCode::VALIDATION_FAILED: return "Validation failed";
    case PhotonicErrorCode::SECURITY_VIOLATION: return "Security violation";
    default: return "Unknown error";
  }
}

PhotonicError::PhotonicError(PhotonicErrorCode code,
                           const std::string& message,
                           const std::string& context,
                           const std::string& suggestion)
    : code_(code), message_(message), context_(context), suggestion_(suggestion) {}

std::string PhotonicError::toString() const {
  std::ostringstream oss;
  oss << "[" << errorCodeToString(code_) << "] " << message_;
  
  if (!context_.empty()) {
    oss << " (in " << context_ << ")";
  }
  
  if (!suggestion_.empty()) {
    oss << "\nSuggestion: " << suggestion_;
  }
  
  return oss.str();
}

void ErrorCollector::addError(const PhotonicError& error) {
  errors_.push_back(error);
}

void ErrorCollector::addError(PhotonicErrorCode code, const std::string& message,
                             const std::string& context, const std::string& suggestion) {
  errors_.emplace_back(code, message, context, suggestion);
}

std::string ErrorCollector::formatErrors() const {
  if (errors_.empty()) {
    return "No errors";
  }
  
  std::ostringstream oss;
  oss << "Found " << errors_.size() << " error(s):\n";
  
  for (size_t i = 0; i < errors_.size(); ++i) {
    oss << (i + 1) << ". " << errors_[i].toString() << "\n";
  }
  
  return oss.str();
}

ErrorContext::ErrorContext(const std::string& context) 
    : previous_context_(current_context_) {
  if (!current_context_.empty()) {
    current_context_ += " -> " + context;
  } else {
    current_context_ = context;
  }
}

ErrorContext::~ErrorContext() {
  current_context_ = previous_context_;
}

std::string ErrorContext::getCurrentContext() {
  return current_context_;
}

//===----------------------------------------------------------------------===//
// Validation utilities
//===----------------------------------------------------------------------===//

namespace mlir {
namespace photonic {
namespace validation {

LogicalResult validateWavelength(int wavelength_nm, ErrorCollector& errors) {
  ErrorContext ctx("wavelength validation");
  
  // Silicon photonics typically operates in 1200-1700nm range
  if (wavelength_nm < 1200 || wavelength_nm > 1700) {
    errors.addError(PhotonicErrorCode::WAVELENGTH_OUT_OF_RANGE,
                   "Wavelength " + std::to_string(wavelength_nm) + "nm is outside supported range",
                   ErrorContext::getCurrentContext(),
                   "Use wavelength between 1200-1700nm for silicon photonics");
    return failure();
  }
  
  // Warn about non-standard wavelengths
  if (wavelength_nm != 1310 && wavelength_nm != 1550) {
    // This would be a warning in a real implementation
    // For now, we'll allow it but could log a warning
  }
  
  return success();
}

LogicalResult validatePhaseShift(double phase_radians, ErrorCollector& errors) {
  ErrorContext ctx("phase shift validation");
  
  // Check for NaN or infinite values
  if (!std::isfinite(phase_radians)) {
    errors.addError(PhotonicErrorCode::PHASE_RANGE_EXCEEDED,
                   "Phase shift must be a finite value",
                   ErrorContext::getCurrentContext(),
                   "Ensure phase calculations don't produce NaN or infinite values");
    return failure();
  }
  
  // Very large phase shifts may indicate inefficient encoding
  if (std::abs(phase_radians) > 10 * M_PI) {
    errors.addError(PhotonicErrorCode::PHASE_RANGE_EXCEEDED,
                   "Phase shift " + std::to_string(phase_radians) + " rad is extremely large",
                   ErrorContext::getCurrentContext(),
                   "Consider normalizing phases to [-π, π] range for efficiency");
    return failure();
  }
  
  return success();
}

LogicalResult validateArraySize(int width, int height, ErrorCollector& errors) {
  ErrorContext ctx("array size validation");
  
  // Check for valid dimensions
  if (width <= 0 || height <= 0) {
    errors.addError(PhotonicErrorCode::ARRAY_SIZE_MISMATCH,
                   "Array dimensions must be positive",
                   ErrorContext::getCurrentContext(),
                   "Specify valid width and height values");
    return failure();
  }
  
  // Check for reasonable size limits (photonic arrays are typically limited)
  if (width > 1024 || height > 1024) {
    errors.addError(PhotonicErrorCode::ARRAY_SIZE_MISMATCH,
                   "Array size " + std::to_string(width) + "x" + std::to_string(height) + 
                   " exceeds practical limits",
                   ErrorContext::getCurrentContext(),
                   "Current photonic hardware typically supports arrays up to 1024x1024");
    return failure();
  }
  
  // Check for power-of-2 dimensions (often preferred for photonic meshes)
  auto isPowerOf2 = [](int n) {
    return n > 0 && (n & (n - 1)) == 0;
  };
  
  if (!isPowerOf2(width) || !isPowerOf2(height)) {
    // This is a warning, not an error, in most cases
    // Real implementation might log a warning here
  }
  
  return success();
}

LogicalResult validateThermalConfig(double max_drift, int interval_ms, ErrorCollector& errors) {
  ErrorContext ctx("thermal configuration validation");
  
  // Validate max drift
  if (!std::isfinite(max_drift) || max_drift < 0) {
    errors.addError(PhotonicErrorCode::THERMAL_CONSTRAINT_VIOLATION,
                   "Maximum phase drift must be a positive finite value",
                   ErrorContext::getCurrentContext(),
                   "Set max_drift to a reasonable value like 0.1 radians");
    return failure();
  }
  
  if (max_drift > M_PI) {
    errors.addError(PhotonicErrorCode::THERMAL_CONSTRAINT_VIOLATION,
                   "Maximum phase drift " + std::to_string(max_drift) + 
                   " rad is too large for practical operation",
                   ErrorContext::getCurrentContext(),
                   "Typical thermal drift should be less than π radians");
    return failure();
  }
  
  // Validate calibration interval
  if (interval_ms <= 0) {
    errors.addError(PhotonicErrorCode::THERMAL_CONSTRAINT_VIOLATION,
                   "Calibration interval must be positive",
                   ErrorContext::getCurrentContext(),
                   "Set a reasonable calibration interval (e.g., 100ms)");
    return failure();
  }
  
  if (interval_ms > 60000) { // More than 1 minute
    errors.addError(PhotonicErrorCode::THERMAL_CONSTRAINT_VIOLATION,
                   "Calibration interval " + std::to_string(interval_ms) + 
                   "ms is too long for stable operation",
                   ErrorContext::getCurrentContext(),
                   "Use shorter calibration intervals for better thermal stability");
    return failure();
  }
  
  return success();
}

LogicalResult validateTensorShapes(llvm::ArrayRef<int64_t> lhsShape,
                                  llvm::ArrayRef<int64_t> rhsShape,
                                  ErrorCollector& errors) {
  ErrorContext ctx("tensor shape validation");
  
  if (lhsShape.empty() || rhsShape.empty()) {
    errors.addError(PhotonicErrorCode::VALIDATION_FAILED,
                   "Tensor shapes cannot be empty",
                   ErrorContext::getCurrentContext(),
                   "Ensure tensors have valid dimensions");
    return failure();
  }
  
  // For matrix multiplication: lhs[..., K] x rhs[K, ...] 
  if (lhsShape.size() < 2 || rhsShape.size() < 2) {
    errors.addError(PhotonicErrorCode::VALIDATION_FAILED,
                   "Matrix operands must have at least 2 dimensions",
                   ErrorContext::getCurrentContext(),
                   "Reshape tensors to have at least 2 dimensions for matrix operations");
    return failure();
  }
  
  // Check inner dimension compatibility
  int64_t lhsInner = lhsShape[lhsShape.size() - 1];
  int64_t rhsInner = rhsShape[rhsShape.size() - 2];
  
  if (lhsInner != rhsInner) {
    errors.addError(PhotonicErrorCode::ARRAY_SIZE_MISMATCH,
                   "Matrix inner dimensions don't match: " + std::to_string(lhsInner) + 
                   " vs " + std::to_string(rhsInner),
                   ErrorContext::getCurrentContext(),
                   "Ensure matrices have compatible dimensions for multiplication");
    return failure();
  }
  
  return success();
}

LogicalResult validateFilePath(const std::string& path, bool mustExist, ErrorCollector& errors) {
  ErrorContext ctx("file path validation");
  
  if (path.empty()) {
    errors.addError(PhotonicErrorCode::FILE_NOT_FOUND,
                   "File path cannot be empty",
                   ErrorContext::getCurrentContext());
    return failure();
  }
  
  // Security check: prevent path traversal attacks
  if (path.find("..") != std::string::npos) {
    errors.addError(PhotonicErrorCode::SECURITY_VIOLATION,
                   "Path traversal detected in file path",
                   ErrorContext::getCurrentContext(),
                   "Use absolute paths or paths without '..' components");
    return failure();
  }
  
  // Check for null bytes (security)
  if (path.find('\0') != std::string::npos) {
    errors.addError(PhotonicErrorCode::SECURITY_VIOLATION,
                   "Null byte detected in file path",
                   ErrorContext::getCurrentContext());
    return failure();
  }
  
  if (mustExist) {
    if (!llvm::sys::fs::exists(path)) {
      errors.addError(PhotonicErrorCode::FILE_NOT_FOUND,
                     "File does not exist: " + path,
                     ErrorContext::getCurrentContext(),
                     "Check that the file path is correct and the file exists");
      return failure();
    }
    
    if (!llvm::sys::fs::is_regular_file(path)) {
      errors.addError(PhotonicErrorCode::FILE_NOT_FOUND,
                     "Path does not point to a regular file: " + path,
                     ErrorContext::getCurrentContext());
      return failure();
    }
  }
  
  return success();
}

LogicalResult validateModelFormat(const std::string& filename, ErrorCollector& errors) {
  ErrorContext ctx("model format validation");
  
  // Check file extension
  std::string ext = llvm::sys::path::extension(filename).str();
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  
  std::vector<std::string> supportedFormats = {".onnx", ".mlir", ".pt", ".pth"};
  
  bool formatSupported = false;
  for (const auto& format : supportedFormats) {
    if (ext == format) {
      formatSupported = true;
      break;
    }
  }
  
  if (!formatSupported) {
    std::ostringstream oss;
    oss << "Unsupported model format: " << ext << ". Supported formats: ";
    for (size_t i = 0; i < supportedFormats.size(); ++i) {
      if (i > 0) oss << ", ";
      oss << supportedFormats[i];
    }
    
    errors.addError(PhotonicErrorCode::INVALID_MODEL_FORMAT,
                   oss.str(),
                   ErrorContext::getCurrentContext(),
                   "Convert your model to a supported format");
    return failure();
  }
  
  // Additional format-specific validation could go here
  if (ext == ".onnx") {
    // ONNX-specific validation
  } else if (ext == ".mlir") {
    // MLIR-specific validation
  }
  
  return success();
}

} // namespace validation
} // namespace photonic
} // namespace mlir