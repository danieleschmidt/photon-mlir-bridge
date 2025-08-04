//===- photon-compile.cpp - Main compiler tool -------------------------===//
//
// Main command-line tool for photonic compilation.
//
//===----------------------------------------------------------------------===//

#include "photon/core/PhotonicCompiler.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::photonic;

// Command line options
static cl::opt<std::string> inputFilename(cl::Positional,
                                         cl::desc("<input file>"),
                                         cl::init("-"));

static cl::opt<std::string> outputFilename("o",
                                          cl::desc("Output filename"),
                                          cl::value_desc("filename"),
                                          cl::init("-"));

static cl::opt<std::string> targetDevice("target",
                                        cl::desc("Target photonic device"),
                                        cl::values(
                                          clEnumVal(lightmatter, "Lightmatter Envise"),
                                          clEnumVal(mit_photonic, "MIT Photonic Processor"),
                                          clEnumVal(research_chip, "Custom Research Chip")),
                                        cl::init("lightmatter"));

static cl::opt<std::string> precision("precision",
                                     cl::desc("Computation precision"),
                                     cl::values(
                                       clEnumVal(int8, "8-bit integer"),
                                       clEnumVal(int16, "16-bit integer"),
                                       clEnumVal(fp16, "16-bit float"),
                                       clEnumVal(fp32, "32-bit float")),
                                     cl::init("int8"));

static cl::opt<int> arrayWidth("array-width",
                              cl::desc("Photonic array width"),
                              cl::init(64));

static cl::opt<int> arrayHeight("array-height", 
                               cl::desc("Photonic array height"),
                               cl::init(64));

static cl::opt<int> wavelength("wavelength",
                              cl::desc("Operating wavelength in nm"),
                              cl::init(1550));

static cl::opt<bool> showReport("show-report",
                               cl::desc("Display optimization report"),
                               cl::init(false));

static cl::opt<bool> verbose("v",
                           cl::desc("Enable verbose output"),
                           cl::init(false));

PhotonicTargetConfig::Device parseDevice(StringRef deviceStr) {
  if (deviceStr == "lightmatter")
    return PhotonicTargetConfig::Device::LIGHTMATTER_ENVISE;
  else if (deviceStr == "mit_photonic")
    return PhotonicTargetConfig::Device::MIT_PHOTONIC_PROCESSOR;
  else if (deviceStr == "research_chip")
    return PhotonicTargetConfig::Device::CUSTOM_RESEARCH_CHIP;
  else
    return PhotonicTargetConfig::Device::LIGHTMATTER_ENVISE;
}

PhotonicTargetConfig::Precision parsePrecision(StringRef precisionStr) {
  if (precisionStr == "int8")
    return PhotonicTargetConfig::Precision::INT8;
  else if (precisionStr == "int16")
    return PhotonicTargetConfig::Precision::INT16;
  else if (precisionStr == "fp16")
    return PhotonicTargetConfig::Precision::FP16;
  else if (precisionStr == "fp32")
    return PhotonicTargetConfig::Precision::FP32;
  else
    return PhotonicTargetConfig::Precision::INT8;
}

int main(int argc, char **argv) {
  InitLLVM y(argc, argv);
  
  cl::ParseCommandLineOptions(argc, argv, "Photonic MLIR Compiler\n");
  
  if (verbose) {
    llvm::outs() << "Photonic MLIR Compiler v0.1.0\n";
    llvm::outs() << "Input file: " << inputFilename << "\n";
    llvm::outs() << "Output file: " << outputFilename << "\n";
    llvm::outs() << "Target device: " << targetDevice << "\n";
    llvm::outs() << "Precision: " << precision << "\n";
    llvm::outs() << "Array size: " << arrayWidth << "x" << arrayHeight << "\n";
    llvm::outs() << "Wavelength: " << wavelength << " nm\n\n";
  }
  
  // Configure target
  PhotonicTargetConfig config;
  config.device = parseDevice(targetDevice);
  config.precision = parsePrecision(precision);
  config.array_size = {arrayWidth, arrayHeight};
  config.wavelength_nm = wavelength;
  
  // Create compiler instance
  PhotonicCompiler compiler;
  compiler.setTargetConfig(config);
  
  // Load input model
  if (failed(compiler.loadONNX(inputFilename))) {
    llvm::errs() << "Error: Failed to load input model\n";
    return 1;
  }
  
  if (verbose) {
    llvm::outs() << "Successfully loaded model\n";
  }
  
  // Compile to photonic representation
  if (failed(compiler.compile())) {
    llvm::errs() << "Error: Compilation failed\n";
    return 1;
  }
  
  if (verbose) {
    llvm::outs() << "Compilation successful\n";
  }
  
  // Generate output code
  if (failed(compiler.codegen(outputFilename))) {
    llvm::errs() << "Error: Code generation failed\n";
    return 1;
  }
  
  if (verbose) {
    llvm::outs() << "Code generation successful\n";
  }
  
  // Show optimization report if requested
  if (showReport) {
    llvm::outs() << "\n" << compiler.getOptimizationReport() << "\n";
  }
  
  if (verbose) {
    llvm::outs() << "Output written to: " << outputFilename << "\n";
  }
  
  return 0;
}