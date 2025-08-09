//===- PhotonicOps.td - Photonic dialect ops -----------*- tablegen -*-===//
//
// This file defines operations for the Photonic dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PHOTONIC_OPS_TD
#define PHOTONIC_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// Photonic Dialect Definition
//===----------------------------------------------------------------------===//

def Photonic_Dialect : Dialect {
  let name = "photonic";
  let summary = "A high-level dialect for photonic computing operations";
  let description = [{
    The Photonic dialect contains operations for photonic neural network
    accelerators, including matrix operations, phase shifts, and thermal
    compensation primitives.
  }];
  let cppNamespace = "::mlir::photonic";
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class Photonic_Op<string mnemonic, list<Trait> traits = []> :
    Op<Photonic_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Core Photonic Operations
//===----------------------------------------------------------------------===//

def MatMulOp : Photonic_Op<"matmul", [Pure]> {
  let summary = "Photonic matrix multiplication";
  let description = [{
    Performs matrix multiplication using a photonic mesh architecture.
    Supports wavelength multiplexing and mesh configuration options.
  }];
  
  let arguments = (ins 
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    I32Attr:$wavelength,
    StrAttr:$mesh_config
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";
}

def PhaseShiftOp : Photonic_Op<"phase_shift", [Pure]> {
  let summary = "Apply phase shift to optical signal";
  let description = [{
    Applies a phase shift to the optical signal using thermal or electro-optic
    phase shifters. Used for implementing photonic weights.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    F32Attr:$phase_radians
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def ThermalCompensationOp : Photonic_Op<"thermal_compensation", []> {
  let summary = "Thermal drift compensation";
  let description = [{
    Inserts thermal compensation calibration points to maintain phase accuracy
    under temperature variations.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    F32Attr:$max_drift,
    I32Attr:$calibration_interval_ms
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def UnfoldOp : Photonic_Op<"unfold", [Pure]> {
  let summary = "Unfold tensor for photonic convolution";
  let description = [{
    Unfolds input tensor to prepare for photonic matrix multiplication
    representation of convolution operations.
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def FoldOp : Photonic_Op<"fold", [Pure]> {
  let summary = "Fold tensor from photonic computation";
  let description = [{
    Reshapes the result of photonic matrix operations back to the expected
    tensor format for subsequent operations.
  }];
  
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def OpticalEncodingOp : Photonic_Op<"optical_encoding", []> {
  let summary = "Convert electronic to optical domain";
  let description = [{
    Converts electronic signals to optical domain using modulators.
    Includes wavelength assignment and power normalization.
  }];
  
  let arguments = (ins 
    AnyTensor:$electronic_input,
    I32Attr:$wavelength_nm
  );
  
  let results = (outs AnyTensor:$optical_output);
  
  let assemblyFormat = "$electronic_input attr-dict `:` type($electronic_input) `->` type($optical_output)";
}

def OpticalDecodingOp : Photonic_Op<"optical_decoding", []> {
  let summary = "Convert optical to electronic domain";
  let description = [{
    Converts optical signals back to electronic domain using photodetectors.
    Includes analog-to-digital conversion and gain normalization.
  }];
  
  let arguments = (ins AnyTensor:$optical_input);
  let results = (outs AnyTensor:$electronic_output);
  
  let assemblyFormat = "$optical_input attr-dict `:` type($optical_input) `->` type($electronic_output)";
}

//===----------------------------------------------------------------------===//
// Advanced Quantum-Photonic Operations
//===----------------------------------------------------------------------===//

def QuantumPhaseGateOp : Photonic_Op<"quantum_phase_gate", [Pure]> {
  let summary = "Quantum phase gate using photonic interferometry";
  let description = [{
    Implements quantum phase gates using Mach-Zehnder interferometer arrays.
    Enables quantum-classical hybrid computing with photonic circuits.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    F32Attr:$phase_radians,
    I32Attr:$qubit_index
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def ThermalAwareSchedulingOp : Photonic_Op<"thermal_scheduling", []> {
  let summary = "Thermal-aware operation scheduling";
  let description = [{
    Schedules photonic operations based on thermal constraints and heat
    distribution patterns to maintain phase coherence.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    F32ArrayAttr:$thermal_map,
    I32Attr:$scheduling_window_ms
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def WavelengthMultiplexOp : Photonic_Op<"wavelength_multiplex", [Pure]> {
  let summary = "Wavelength division multiplexing";
  let description = [{
    Combines multiple optical signals at different wavelengths for parallel
    processing in photonic mesh architectures.
  }];
  
  let arguments = (ins 
    Variadic<AnyTensor>:$inputs,
    I32ArrayAttr:$wavelengths_nm
  );
  
  let results = (outs AnyTensor:$multiplexed_output);
  
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($multiplexed_output)";
}

def MeshCalibrationOp : Photonic_Op<"mesh_calibration", []> {
  let summary = "Photonic mesh calibration";
  let description = [{
    Performs runtime calibration of photonic mesh to compensate for
    fabrication variations and thermal drift.
  }];
  
  let arguments = (ins 
    AnyTensor:$mesh_state,
    StrAttr:$calibration_method,
    F32Attr:$target_fidelity
  );
  
  let results = (outs AnyTensor:$calibrated_mesh);
  
  let assemblyFormat = "$mesh_state attr-dict `:` type($mesh_state) `->` type($calibrated_mesh)";
}

def PowerBalancingOp : Photonic_Op<"power_balancing", []> {
  let summary = "Optical power balancing";
  let description = [{
    Ensures uniform optical power distribution across the photonic mesh
    to maintain consistent operation and prevent thermal hotspots.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    F32Attr:$target_power_mw,
    F32Attr:$power_tolerance
  );
  
  let results = (outs AnyTensor:$balanced_output);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($balanced_output)";
}

def CrosstalkMitigationOp : Photonic_Op<"crosstalk_mitigation", [Pure]> {
  let summary = "Optical crosstalk mitigation";
  let description = [{
    Applies crosstalk compensation techniques to reduce interference
    between adjacent waveguides in dense photonic circuits.
  }];
  
  let arguments = (ins 
    AnyTensor:$input,
    F32ArrayAttr:$crosstalk_matrix,
    F32Attr:$mitigation_strength
  );
  
  let results = (outs AnyTensor:$result);
  
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

#endif // PHOTONIC_OPS_TD