//===- ErrorHandling.h - Comprehensive error handling ---------*- C++ -*-===//
//
// This file defines error handling utilities for the photonic compiler.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_ERROR_HANDLING_H
#define PHOTONIC_ERROR_HANDLING_H

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Support/LogicalResult.h"
#include <string>
#include <vector>
#include <memory>

namespace mlir {
namespace photonic {

/// Error categories for photonic compilation
enum class PhotonicErrorCode {
  SUCCESS = 0,
  FILE_NOT_FOUND,
  INVALID_MODEL_FORMAT,
  UNSUPPORTED_OPERATION,
  HARDWARE_CONSTRAINT_VIOLATION,
  PHASE_RANGE_EXCEEDED,
  WAVELENGTH_OUT_OF_RANGE,
  ARRAY_SIZE_MISMATCH,
  THERMAL_CONSTRAINT_VIOLATION,
  COMPILATION_FAILED,
  CODEGEN_FAILED,
  MEMORY_ALLOCATION_FAILED,
  VALIDATION_FAILED,
  SECURITY_VIOLATION
};

/// Convert error code to human-readable string
std::string errorCodeToString(PhotonicErrorCode code);

/// Detailed error information
class PhotonicError {
public:
  PhotonicError(PhotonicErrorCode code, 
                const std::string& message,
                const std::string& context = "",
                const std::string& suggestion = "");
  
  PhotonicErrorCode getCode() const { return code_; }
  const std::string& getMessage() const { return message_; }
  const std::string& getContext() const { return context_; }
  const std::string& getSuggestion() const { return suggestion_; }
  
  std::string toString() const;
  
private:
  PhotonicErrorCode code_;
  std::string message_;
  std::string context_;
  std::string suggestion_;
};

/// Error collector for multiple validation errors
class ErrorCollector {
public:
  void addError(const PhotonicError& error);
  void addError(PhotonicErrorCode code, const std::string& message,
                const std::string& context = "", const std::string& suggestion = "");
  
  bool hasErrors() const { return !errors_.empty(); }
  size_t getErrorCount() const { return errors_.size(); }
  
  const std::vector<PhotonicError>& getErrors() const { return errors_; }
  
  void clear() { errors_.clear(); }
  
  std::string formatErrors() const;
  
private:
  std::vector<PhotonicError> errors_;
};

/// Input validation utilities
namespace validation {

/// Validate wavelength is in supported range
LogicalResult validateWavelength(int wavelength_nm, ErrorCollector& errors);

/// Validate phase shift is reasonable
LogicalResult validatePhaseShift(double phase_radians, ErrorCollector& errors);

/// Validate array dimensions are supported
LogicalResult validateArraySize(int width, int height, ErrorCollector& errors);

/// Validate thermal parameters
LogicalResult validateThermalConfig(double max_drift, int interval_ms, ErrorCollector& errors);

/// Validate tensor shapes are compatible
LogicalResult validateTensorShapes(llvm::ArrayRef<int64_t> lhsShape,
                                  llvm::ArrayRef<int64_t> rhsShape,
                                  ErrorCollector& errors);

/// Validate file path is safe and accessible
LogicalResult validateFilePath(const std::string& path, bool mustExist, ErrorCollector& errors);

/// Validate model file format
LogicalResult validateModelFormat(const std::string& filename, ErrorCollector& errors);

} // namespace validation

/// RAII wrapper for error context
class ErrorContext {
public:
  ErrorContext(const std::string& context);
  ~ErrorContext();
  
  static std::string getCurrentContext();
  
private:
  std::string previous_context_;
  static thread_local std::string current_context_;
};

/// Macros for convenient error handling
#define PHOTONIC_ERROR(code, message) \
  PhotonicError(code, message, ErrorContext::getCurrentContext())

#define PHOTONIC_VALIDATE(condition, code, message) \
  do { \
    if (!(condition)) { \
      return PhotonicError(code, message, ErrorContext::getCurrentContext()); \
    } \
  } while (0)

#define PHOTONIC_TRY(expr) \
  do { \
    if (auto result = (expr); !result) { \
      return result; \
    } \
  } while (0)

} // namespace photonic
} // namespace mlir

#endif // PHOTONIC_ERROR_HANDLING_H