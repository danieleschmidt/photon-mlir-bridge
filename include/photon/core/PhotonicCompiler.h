//===- PhotonicCompiler.h - Main compiler interface -----------*- C++ -*-===//
//
// This file declares the main PhotonicCompiler class.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_COMPILER_H
#define PHOTONIC_COMPILER_H

#include "photon/dialects/PhotonicDialect.h"
#include "photon/core/ErrorHandling.h"
#include "photon/core/Logging.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Module.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"
#include <memory>
#include <string>
#include <chrono>
#include <stdexcept>

namespace mlir {
namespace photonic {

// Forward declarations
class ErrorHandler;
class Logger;
class ThermalMonitor;

/// Exception types for robust error handling
class CompilationException : public std::runtime_error {
public:
  explicit CompilationException(const std::string& msg) : std::runtime_error(msg) {}
};

class ThermalViolationException : public std::runtime_error {
public:
  explicit ThermalViolationException(const std::string& msg) : std::runtime_error(msg) {}
};

class PhaseCoherenceException : public std::runtime_error {
public:
  explicit PhaseCoherenceException(const std::string& msg) : std::runtime_error(msg) {}
};

class TimeoutException : public std::runtime_error {
public:
  explicit TimeoutException(const std::string& msg) : std::runtime_error(msg) {}
};

/// Compilation context for tracking state and recovery
struct CompilationContext {
  ModuleOp& module;
  const PhotonicTargetConfig& config;
  size_t retry_count = 0;
  bool thermal_override = false;
  double phase_correction_factor = 1.0;
  
  CompilationContext(ModuleOp& mod, const PhotonicTargetConfig& cfg) 
    : module(mod), config(cfg) {}
};

/// Enhanced compiler configuration
struct PhotonicCompilerConfig {
  // Safety limits
  static constexpr size_t MAX_MODEL_SIZE_BYTES = 1024 * 1024 * 1024; // 1GB
  static constexpr int PARSING_TIMEOUT_MS = 30000; // 30 seconds
  static constexpr int COOLDOWN_TIME_MS = 5000; // 5 seconds
  static constexpr double DEFAULT_THERMAL_LIMIT = 85.0; // Celsius
  
  // Compilation parameters
  bool enable_safety_checks = true;
  bool enable_thermal_compensation = true;
  bool enable_quantum_validation = false;
  size_t max_matrix_size = 1024;
  double max_phase_drift = 0.1; // radians
  double target_mesh_fidelity = 0.95;
  double thermal_limit_celsius = DEFAULT_THERMAL_LIMIT;
  double max_optical_power_mw = 100.0;
  int calibration_interval_ms = 100;
};

/// Main photonic compiler class for end-to-end compilation with robust error handling
class PhotonicCompiler {
public:
  PhotonicCompiler();
  ~PhotonicCompiler();

  // Core compilation interface
  LogicalResult loadONNX(llvm::StringRef filename);
  LogicalResult loadPyTorch(llvm::StringRef filename);
  void setTargetConfig(const PhotonicTargetConfig& config);
  LogicalResult compile();
  LogicalResult codegen(llvm::StringRef outputFile);
  
  // Enhanced interface
  LogicalResult compileWithRecovery(int max_retries = 3);
  LogicalResult validateAndCompile();
  void setCompilerConfig(const PhotonicCompilerConfig& config) { compiler_config_ = config; }
  
  // Monitoring and diagnostics
  std::string getOptimizationReport() const;
  std::string getDetailedDiagnostics() const;
  struct CompilationMetrics getMetrics() const { return compilation_metrics_; }
  
  // Access methods
  ModuleOp getModule() const { return module_.get(); }
  MLIRContext* getContext() const { return context_.get(); }
  
  // Thermal management
  double getCurrentTemperature() const;
  LogicalResult performThermalCooldown(int duration_ms = 5000);
  
  // Recovery mechanisms
  LogicalResult recoverFromPhaseError();
  LogicalResult recoverFromThermalViolation();

private:
  // Core components
  std::unique_ptr<MLIRContext> context_;
  std::unique_ptr<ModuleOp> module_;
  PhotonicTargetConfig config_;
  PhotonicCompilerConfig compiler_config_;
  
  // Error handling and logging
  std::unique_ptr<ErrorHandler> error_handler_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<ThermalMonitor> thermal_monitor_;
  
  // Source tracking
  std::string source_file_;
  static constexpr const char* COMPILER_VERSION = "1.0.0";
  
  // Metrics and statistics
  struct CompilationMetrics {
    double total_compilation_time_ms = 0.0;
    size_t successful_compilations = 0;
    size_t failed_compilations = 0;
    size_t thermal_violations = 0;
    size_t phase_errors = 0;
  } compilation_metrics_;
  
  struct OptimizationStats {
    size_t originalFlops = 0;
    size_t photonicMacs = 0;
    size_t totalPhaseShifts = 0;
    double estimatedSpeedup = 1.0;
    double energyReduction = 0.0;
    double compilation_time_ms = 0.0;
  } stats_;
  
  // Pipeline management
  LogicalResult buildPipeline(PassManager& pm);
  LogicalResult buildSafePipeline(PassManager& pm);
  
  // Validation methods
  LogicalResult validateModuleStructure(ModuleOp module);
  LogicalResult performSecurityValidation(ModuleOp module);
  LogicalResult validateCompiledModule(ModuleOp module);
  LogicalResult validatePhaseCoherence(Operation* op);
  
  // Recovery and adaptation
  void adjustCompilationStrategy(CompilationContext& context);
  LogicalResult applyPhaseCorrection(ModuleOp module);
  
  // Code generation helpers
  void generateMatMulInstruction(llvm::raw_ostream& output, MatMulOp op);
  void generatePhaseShiftInstruction(llvm::raw_ostream& output, PhaseShiftOp op);
  void generateThermalCompensationInstructions(llvm::raw_ostream& output, ThermalCompensationOp op);
  void generateQuantumGateInstruction(llvm::raw_ostream& output, QuantumPhaseGateOp op);
  void generateWDMInstruction(llvm::raw_ostream& output, WavelengthMultiplexOp op);
  void generateMeshCalibrationInstructions(llvm::raw_ostream& output, MeshCalibrationOp op);
  void generatePowerBalancingInstruction(llvm::raw_ostream& output, PowerBalancingOp op);
  void generateOpticalEncodingInstruction(llvm::raw_ostream& output, OpticalEncodingOp op);
  void generateOpticalDecodingInstruction(llvm::raw_ostream& output, OpticalDecodingOp op);
  
  // Utility methods
  OptimizationStats calculateOptimizationStatistics(ModuleOp module, double compile_time_ms);
  std::string getCurrentTimestamp() const;
  std::string getPrecisionString(PhotonicTargetConfig::Precision precision) const;
  std::string getMeshTopologyString(const PhotonicTargetConfig& config) const;
  
  // Load model from various formats
  LogicalResult loadFromFormat(llvm::StringRef filename, llvm::StringRef format);
};

/// RAII timeout guard for preventing infinite loops
class TimeoutGuard {
public:
  explicit TimeoutGuard(int timeout_ms);
  ~TimeoutGuard();
private:
  std::chrono::steady_clock::time_point start_time_;
  int timeout_ms_;
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