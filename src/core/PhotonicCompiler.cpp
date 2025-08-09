//===- PhotonicCompiler.cpp - Main compiler implementation --------------===//
//
// This file implements the PhotonicCompiler class.
//
//===----------------------------------------------------------------------===//

#include "photon/core/PhotonicCompiler.h"
#include "photon/dialects/PhotonicDialect.h"
#include "photon/transforms/PhotonicPasses.h"
#include "photon/core/ErrorHandling.h"
#include "photon/core/Logging.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/MemoryBuffer.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <stdexcept>
#include <memory>

using namespace mlir;
using namespace mlir::photonic;

PhotonicCompiler::PhotonicCompiler() : error_handler_(std::make_unique<ErrorHandler>()),
                                        logger_(std::make_unique<Logger>("PhotonicCompiler")) {
  try {
    logger_->info("Initializing Photonic Compiler");
    
    context_ = std::make_unique<MLIRContext>();
    
    // Register required dialects with error checking
    if (!context_->getOrLoadDialect<PhotonicDialect>()) {
      error_handler_->handleCriticalError("Failed to load Photonic dialect");
      return;
    }
    
    if (!context_->getOrLoadDialect<func::FuncDialect>()) {
      error_handler_->handleCriticalError("Failed to load Function dialect");
      return;
    }
    
    if (!context_->getOrLoadDialect<arith::ArithDialect>()) {
      error_handler_->handleCriticalError("Failed to load Arithmetic dialect");
      return;
    }
    
    if (!context_->getOrLoadDialect<linalg::LinalgDialect>()) {
      error_handler_->handleCriticalError("Failed to load Linalg dialect");
      return;
    }
    
    // Register photonic passes
    registerPhotonicPasses();
    
    // Create empty module with error handling
    OpBuilder builder(context_.get());
    module_ = builder.create<ModuleOp>(builder.getUnknownLoc());
    
    if (!module_) {
      error_handler_->handleCriticalError("Failed to create MLIR module");
      return;
    }
    
    // Initialize compilation metrics
    compilation_metrics_ = {
      .total_compilation_time_ms = 0.0,
      .successful_compilations = 0,
      .failed_compilations = 0,
      .thermal_violations = 0,
      .phase_errors = 0
    };
    
    // Initialize thermal monitoring
    thermal_monitor_ = std::make_unique<ThermalMonitor>(config_);
    
    logger_->info("Photonic Compiler initialized successfully");
    
  } catch (const std::exception& e) {
    error_handler_->handleException("PhotonicCompiler constructor", e);
    throw CompilationException("Failed to initialize PhotonicCompiler: " + std::string(e.what()));
  }
}

PhotonicCompiler::~PhotonicCompiler() = default;

LogicalResult PhotonicCompiler::loadONNX(llvm::StringRef filename) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  try {
    logger_->info("Loading ONNX model: " + filename.str());
    
    // Validate file path and existence
    if (filename.empty()) {
      error_handler_->handleError("Empty filename provided to loadONNX");
      return failure();
    }
    
    if (!llvm::sys::fs::exists(filename)) {
      error_handler_->handleError("File does not exist: " + filename.str());
      return failure();
    }
    
    // Check file size constraints (prevent memory exhaustion)
    uint64_t file_size = 0;
    if (llvm::sys::fs::file_size(filename, file_size) || file_size > MAX_MODEL_SIZE_BYTES) {
      error_handler_->handleError("File too large or unreadable: " + filename.str());
      return failure();
    }
    
    // Read file with robust error handling
    auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(filename);
    if (std::error_code error = fileOrErr.getError()) {
      std::string error_msg = "Error reading file: " + error.message();
      error_handler_->handleError(error_msg);
      logger_->error(error_msg);
      return failure();
    }
    
    // Validate file content before parsing
    auto& buffer = **fileOrErr;
    if (buffer.getBufferSize() == 0) {
      error_handler_->handleError("Empty file: " + filename.str());
      return failure();
    }
    
    // Parse with timeout protection
    std::unique_ptr<ModuleOp> module;
    {
      TimeoutGuard timeout_guard(PARSING_TIMEOUT_MS);
      try {
        module = parseSourceFile<ModuleOp>(filename, context_.get());
      } catch (const TimeoutException& e) {
        error_handler_->handleError("Parsing timeout for file: " + filename.str());
        return failure();
      }
    }
    
    if (!module) {
      error_handler_->handleError("Failed to parse MLIR module from: " + filename.str());
      return failure();
    }
    
    // Validate parsed module structure
    if (failed(validateModuleStructure(*module))) {
      error_handler_->handleError("Invalid module structure in: " + filename.str());
      return failure();
    }
    
    // Security validation - prevent malicious models
    if (failed(performSecurityValidation(*module))) {
      error_handler_->handleError("Security validation failed for: " + filename.str());
      return failure();
    }
    
    module_ = std::move(module);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    logger_->info("Successfully loaded ONNX model in " + std::to_string(duration.count()) + "ms");
    
    return success();
    
  } catch (const std::exception& e) {
    error_handler_->handleException("loadONNX", e);
    compilation_metrics_.failed_compilations++;
    return failure();
  }
}

LogicalResult PhotonicCompiler::loadPyTorch(llvm::StringRef filename) {
  // Placeholder implementation - would integrate with PyTorch JIT
  llvm::errs() << "PyTorch loading not yet implemented\n";
  return failure();
}

void PhotonicCompiler::setTargetConfig(const PhotonicTargetConfig& config) {
  config_ = config;
}

LogicalResult PhotonicCompiler::buildPipeline(PassManager& pm) {
  try {
    logger_->info("Building compilation pipeline");
    
    // Enable pass failure recovery
    pm.enableCrashReproducerGeneration("photonic_crash.mlir");
    
    // Stage 1: Pre-processing and validation
    pm.addPass(createPhotonicValidationPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
    
    // Stage 2: Photonic-specific transformations with safety checks
    if (config_.enable_safety_checks) {
      pm.addPass(createPhotonicSafetyPass());
    }
    
    pm.addPass(createPhotonicLoweringPass());
    
    // Matrix decomposition with size limits
    auto matrix_pass = createMatrixDecompositionPass();
    matrix_pass->setMaxMatrixSize(config_.max_matrix_size);
    pm.addPass(std::move(matrix_pass));
    
    // Phase optimization with coherence monitoring
    auto phase_pass = createPhaseOptimizationPass();
    phase_pass->setPhaseToleranceRadians(config_.max_phase_drift);
    pm.addPass(std::move(phase_pass));
    
    // Thermal management passes
    if (config_.enable_thermal_compensation) {
      pm.addPass(createThermalAnalysisPass());
      pm.addPass(createThermalCompensationPass());
      pm.addPass(createThermalValidationPass());
    }
    
    // Stage 3: Hardware-specific optimizations
    pm.addPass(createPowerBalancingPass());
    
    // Mesh optimization with fidelity constraints
    auto mesh_pass = createMeshOptimizationPass();
    mesh_pass->setTargetFidelity(config_.target_mesh_fidelity);
    pm.addPass(std::move(mesh_pass));
    
    // Stage 4: Post-processing and final validation
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createPhotonicFinalValidationPass());
    
    // Add quantum coherence checks if enabled
    if (config_.enable_quantum_validation) {
      pm.addPass(createQuantumCoherencePass());
    }
    
    logger_->info("Pipeline built successfully with " + std::to_string(pm.size()) + " passes");
    return success();
    
  } catch (const std::exception& e) {
    error_handler_->handleException("buildPipeline", e);
    return failure();
  }
}

LogicalResult PhotonicCompiler::compile() {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  try {
    logger_->info("Starting photonic compilation");
    
    // Pre-compilation validation
    if (!module_) {
      error_handler_->handleError("No module loaded for compilation");
      return failure();
    }
    
    // Thermal safety check
    if (thermal_monitor_->getCurrentTemperature() > config_.thermal_limit_celsius) {
      error_handler_->handleError("Thermal limit exceeded, aborting compilation");
      compilation_metrics_.thermal_violations++;
      return failure();
    }
    
    // Create compilation context with monitoring
    CompilationContext comp_context(*module_, config_);
    
    // Build and validate pass pipeline
    PassManager pm(context_.get());
    if (failed(buildPipeline(pm))) {
      error_handler_->handleError("Failed to build compilation pipeline");
      return failure();
    }
    
    // Add pass instrumentation for monitoring
    pm.addInstrumentation(std::make_unique<PhotonicPassInstrumentation>(
      [this](Pass* pass, Operation* op) {
        // Pre-pass thermal check
        if (thermal_monitor_->getCurrentTemperature() > config_.thermal_limit_celsius) {
          throw ThermalViolationException("Thermal limit exceeded during pass: " + std::string(pass->getName()));
        }
      },
      [this](Pass* pass, Operation* op) {
        // Post-pass validation
        if (failed(validatePhaseCoherence(op))) {
          compilation_metrics_.phase_errors++;
        }
      }
    ));
    
    // Run compilation with recovery mechanisms
    LogicalResult result;
    int retry_count = 0;
    const int MAX_RETRIES = 3;
    
    do {
      try {
        // Reset thermal state between retries
        if (retry_count > 0) {
          thermal_monitor_->coolDown(COOLDOWN_TIME_MS);
          logger_->warning("Compilation retry #" + std::to_string(retry_count));
        }
        
        // Run compilation pipeline
        result = pm.run(*module_);
        
        if (succeeded(result)) {
          break;
        } else if (retry_count < MAX_RETRIES - 1) {
          // Analyze failure and adjust strategy
          adjustCompilationStrategy(comp_context);
        }
        
      } catch (const ThermalViolationException& e) {
        logger_->warning("Thermal violation during compilation: " + std::string(e.what()));
        thermal_monitor_->triggerCooldown();
        result = failure();
      } catch (const PhaseCoherenceException& e) {
        logger_->warning("Phase coherence error: " + std::string(e.what()));
        // Apply phase correction
        applyPhaseCorrection(*module_);
        result = failure();
      }
      
      retry_count++;
    } while (failed(result) && retry_count < MAX_RETRIES);
    
    if (failed(result)) {
      error_handler_->handleError("Compilation pipeline failed after " + std::to_string(retry_count) + " attempts");
      compilation_metrics_.failed_compilations++;
      return failure();
    }
    
    // Post-compilation validation
    if (failed(validateCompiledModule(*module_))) {
      error_handler_->handleError("Compiled module validation failed");
      return failure();
    }
    
    // Calculate comprehensive statistics
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    stats_ = calculateOptimizationStatistics(*module_, duration.count());
    compilation_metrics_.total_compilation_time_ms += duration.count();
    compilation_metrics_.successful_compilations++;
    
    logger_->info("Compilation completed successfully in " + std::to_string(duration.count()) + "ms");
    logger_->info("Speedup: " + std::to_string(stats_.estimatedSpeedup) + "x, Energy reduction: " + 
                 std::to_string(stats_.energyReduction) + "%");
    
    return success();
    
  } catch (const std::exception& e) {
    error_handler_->handleException("compile", e);
    compilation_metrics_.failed_compilations++;
    return failure();
  }
}

LogicalResult PhotonicCompiler::codegen(llvm::StringRef outputFile) {
  auto start_time = std::chrono::high_resolution_clock::now();
  
  try {
    logger_->info("Starting code generation for: " + outputFile.str());
    
    // Validate inputs
    if (!module_) {
      error_handler_->handleError("No compiled module available for code generation");
      return failure();
    }
    
    if (outputFile.empty()) {
      error_handler_->handleError("Empty output file path provided");
      return failure();
    }
    
    // Ensure output directory exists
    llvm::StringRef parent_dir = llvm::sys::path::parent_path(outputFile);
    if (!parent_dir.empty() && !llvm::sys::fs::exists(parent_dir)) {
      if (llvm::sys::fs::create_directories(parent_dir)) {
        error_handler_->handleError("Failed to create output directory: " + parent_dir.str());
        return failure();
      }
    }
    
    // Create temporary file for atomic write
    std::string temp_file = outputFile.str() + ".tmp." + std::to_string(time(nullptr));
    
    std::error_code EC;
    llvm::raw_fd_ostream output(temp_file, EC);
    if (EC) {
      error_handler_->handleError("Error opening temporary output file: " + EC.message());
      return failure();
    }
    
    // Generate comprehensive photonic assembly header
    output << "; Photonic Assembly Generated by photon-mlir-bridge v" << COMPILER_VERSION << "\n";
    output << "; Generated: " << getCurrentTimestamp() << "\n";
    output << "; Source: " << (source_file_.empty() ? "<unknown>" : source_file_) << "\n";
    output << ";==================================================\n\n";
    
    // Model metadata
    output << ".model compiled_model\n";
    output << ".version 1.0\n";
    output << ".precision " << getPrecisionString(config_.precision) << "\n";
    output << ".mesh " << getMeshTopologyString(config_) << "\n";
    output << ".wavelength " << config_.wavelength_nm << "\n";
    
    // Thermal configuration
    if (config_.enable_thermal_compensation) {
      output << ".thermal_compensation enabled\n";
      output << ".max_phase_drift " << config_.max_phase_drift << "\n";
      output << ".calibration_interval " << config_.calibration_interval_ms << "\n";
    }
    
    // Resource allocation section
    output << "\n; Resource Allocation\n";
    output << ".array_size " << config_.array_size.first << " " << config_.array_size.second << "\n";
    output << ".max_power_mw " << config_.max_optical_power_mw << "\n";
    
    // Safety limits
    output << "\n; Safety Limits\n";
    output << ".thermal_limit " << config_.thermal_limit_celsius << "\n";
    output << ".phase_tolerance " << config_.max_phase_drift << "\n";
    
    output << "\n; Generated Code\n";
    output << ".code_begin\n\n";
    
    // Generate optimized assembly with error checking
    size_t instruction_count = 0;
    bool has_errors = false;
    
    module_->walk([&](Operation* op) {
      try {
        if (auto matmul = dyn_cast<MatMulOp>(op)) {
          generateMatMulInstruction(output, matmul);
        } else if (auto phaseShift = dyn_cast<PhaseShiftOp>(op)) {
          generatePhaseShiftInstruction(output, phaseShift);
        } else if (auto thermal = dyn_cast<ThermalCompensationOp>(op)) {
          generateThermalCompensationInstructions(output, thermal);
        } else if (auto quantum = dyn_cast<QuantumPhaseGateOp>(op)) {
          generateQuantumGateInstruction(output, quantum);
        } else if (auto wdm = dyn_cast<WavelengthMultiplexOp>(op)) {
          generateWDMInstruction(output, wdm);
        } else if (auto mesh_cal = dyn_cast<MeshCalibrationOp>(op)) {
          generateMeshCalibrationInstructions(output, mesh_cal);
        } else if (auto power = dyn_cast<PowerBalancingOp>(op)) {
          generatePowerBalancingInstruction(output, power);
        } else if (auto encoding = dyn_cast<OpticalEncodingOp>(op)) {
          generateOpticalEncodingInstruction(output, encoding);
        } else if (auto decoding = dyn_cast<OpticalDecodingOp>(op)) {
          generateOpticalDecodingInstruction(output, decoding);
        } else {
          // Handle unknown operations with warning
          output << "; WARNING: Unknown operation: " << op->getName() << "\n";
          logger_->warning("Unknown operation encountered during codegen: " + op->getName().getStringRef().str());
        }
        
        instruction_count++;
        
      } catch (const std::exception& e) {
        output << "; ERROR: Failed to generate instruction for " << op->getName() << ": " << e.what() << "\n";
        error_handler_->handleError("Code generation error for operation: " + std::string(e.what()));
        has_errors = true;
      }
    });
    
    // Generate footer with statistics
    output << "\n.code_end\n";
    output << "\n; Compilation Statistics\n";
    output << "; Instructions generated: " << instruction_count << "\n";
    output << "; Original FLOPs: " << stats_.originalFlops << "\n";
    output << "; Photonic MACs: " << stats_.photonicMacs << "\n";
    output << "; Phase shifts: " << stats_.totalPhaseShifts << "\n";
    output << "; Estimated speedup: " << stats_.estimatedSpeedup << "x\n";
    output << "; Energy reduction: " << stats_.energyReduction << "%\n";
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    output << "; Code generation time: " << duration.count() << "ms\n";
    
    output << "\n; End of generated code\n";
    output.flush();
    output.close();
    
    // Atomically replace output file
    if (llvm::sys::fs::rename(temp_file, outputFile)) {
      error_handler_->handleError("Failed to write final output file: " + outputFile.str());
      llvm::sys::fs::remove(temp_file);  // Clean up temp file
      return failure();
    }
    
    if (has_errors) {
      logger_->warning("Code generation completed with errors. Check output file for details.");
    } else {
      logger_->info("Code generation completed successfully. Generated " + 
                   std::to_string(instruction_count) + " instructions in " + 
                   std::to_string(duration.count()) + "ms");
    }
    
    return success();
    
  } catch (const std::exception& e) {
    error_handler_->handleException("codegen", e);
    return failure();
  }
}

std::string PhotonicCompiler::getOptimizationReport() const {
  std::ostringstream report;
  
  report << "=== Photonic Compilation Report ===\n";
  report << "Original FLOPs: " << stats_.originalFlops << "\n";
  report << "Photonic MACs: " << stats_.photonicMacs << "\n";
  report << "Total Phase Shifts: " << stats_.totalPhaseShifts << "\n";
  report << "Estimated Speedup: " << stats_.estimatedSpeedup << "x\n";
  report << "Energy Reduction: " << stats_.energyReduction << "%\n";
  report << "\nTarget Configuration:\n";
  report << "Device: " << (config_.device == PhotonicTargetConfig::Device::LIGHTMATTER_ENVISE ? "Lightmatter Envise" : "Other") << "\n";
  report << "Array Size: " << config_.array_size.first << "x" << config_.array_size.second << "\n";
  report << "Wavelength: " << config_.wavelength_nm << " nm\n";
  
  return report.str();
}

//===----------------------------------------------------------------------===//
// API Functions
//===----------------------------------------------------------------------===//

LogicalResult api::compileONNX(llvm::StringRef inputFile,
                              llvm::StringRef outputFile,
                              const PhotonicTargetConfig& config) {
  PhotonicCompiler compiler;
  compiler.setTargetConfig(config);
  
  if (failed(compiler.loadONNX(inputFile))) {
    return failure();
  }
  
  if (failed(compiler.compile())) {
    return failure();
  }
  
  return compiler.codegen(outputFile);
}

LogicalResult api::compilePyTorch(llvm::StringRef inputFile,
                                 llvm::StringRef outputFile,
                                 const PhotonicTargetConfig& config) {
  PhotonicCompiler compiler;
  compiler.setTargetConfig(config);
  
  if (failed(compiler.loadPyTorch(inputFile))) {
    return failure();
  }
  
  if (failed(compiler.compile())) {
    return failure();
  }
  
  return compiler.codegen(outputFile);
}