"""
Advanced photonic hardware simulator with quantum effects and thermal modeling.
Generation 1: Enhanced bit-accurate simulation with research-grade features.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Callable
import time
import logging
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
try:
    import scipy.special as sp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .core import PhotonicTensor, TargetConfig, Device


class PhotonicSimulator:
    """Bit-accurate photonic hardware simulator."""
    
    def __init__(self, 
                 noise_model: str = "realistic",
                 precision: str = "8bit", 
                 crosstalk_db: float = -30.0,
                 target_config: Optional[TargetConfig] = None):
        """
        Initialize photonic simulator.
        
        Args:
            noise_model: "ideal", "realistic", or "pessimistic"
            precision: "8bit", "16bit", "fp16", "fp32"
            crosstalk_db: Optical crosstalk in dB (negative value)
            target_config: Hardware target configuration
        """
        self.noise_model = noise_model
        self.precision = precision
        self.crosstalk_db = crosstalk_db
        self.target_config = target_config or TargetConfig()
        
        # Noise parameters based on model
        self._noise_params = self._get_noise_parameters()
        
    def _get_noise_parameters(self) -> Dict[str, float]:
        """Get noise parameters based on selected model."""
        if self.noise_model == "ideal":
            return {
                "shot_noise_factor": 0.0,
                "thermal_noise_factor": 0.0,
                "phase_drift_std": 0.0,
                "insertion_loss_db": 0.0
            }
        elif self.noise_model == "realistic":
            return {
                "shot_noise_factor": 0.005,  # 0.5% shot noise
                "thermal_noise_factor": 0.002,  # 0.2% thermal noise
                "phase_drift_std": 0.05,  # 0.05 radians phase drift
                "insertion_loss_db": -0.5  # 0.5 dB insertion loss
            }
        else:  # pessimistic
            return {
                "shot_noise_factor": 0.02,  # 2% shot noise
                "thermal_noise_factor": 0.01,  # 1% thermal noise  
                "phase_drift_std": 0.15,  # 0.15 radians phase drift
                "insertion_loss_db": -2.0  # 2 dB insertion loss
            }
    
    def run(self, compiled_model, input_data) -> PhotonicTensor:
        """Run simulation on compiled photonic model."""
        print(f"Running photonic simulation with {self.noise_model} noise model")
        
        # Convert input to numpy if needed
        if hasattr(input_data, 'numpy'):
            data = input_data.detach().numpy()
        elif hasattr(input_data, 'data'):
            data = input_data.data
        else:
            data = np.array(input_data, dtype=np.float32)
            
        # Apply precision quantization
        quantized_data = self._apply_quantization(data)
        
        # Simulate optical encoding
        optical_data = self._simulate_optical_encoding(quantized_data)
        
        # Simulate photonic matrix operations
        processed_data = self._simulate_photonic_processing(optical_data)
        
        # Apply noise effects
        noisy_data = self._apply_noise_effects(processed_data)
        
        # Simulate optical decoding
        output_data = self._simulate_optical_decoding(noisy_data)
        
        return PhotonicTensor(
            data=output_data,
            wavelength=self.target_config.wavelength_nm,
            power_mw=self._estimate_output_power(output_data)
        )
    
    def _apply_quantization(self, data: np.ndarray) -> np.ndarray:
        """Apply precision quantization."""
        # Ensure data is a proper numpy array
        data = np.asarray(data, dtype=np.float32)
        
        if self.precision == "8bit":
            # Quantize to 8-bit signed integers
            data_scaled = np.clip(data * 127, -128, 127)
            return np.round(data_scaled) / 127.0
        elif self.precision == "16bit":
            # Quantize to 16-bit signed integers
            data_scaled = np.clip(data * 32767, -32768, 32767)
            return np.round(data_scaled) / 32767.0
        elif self.precision == "fp16":
            # Convert to float16 and back
            return data.astype(np.float16).astype(np.float32)
        else:  # fp32
            return data.astype(np.float32)
    
    def _simulate_optical_encoding(self, data: np.ndarray) -> np.ndarray:
        """Simulate electronic-to-optical conversion."""
        # Apply insertion loss
        loss_factor = 10 ** (self._noise_params["insertion_loss_db"] / 10)
        return data * loss_factor
    
    def _simulate_photonic_processing(self, data: np.ndarray) -> np.ndarray:
        """Simulate photonic matrix operations."""
        # For now, just pass through - real implementation would simulate
        # Mach-Zehnder interferometer networks, phase shifters, etc.
        
        # Simulate some processing (placeholder for actual photonic ops)
        # This would be replaced with actual photonic mesh simulation
        processed = data * 0.95  # Simulate some loss
        
        # Add crosstalk effects
        if self.crosstalk_db < 0:
            crosstalk_factor = 10 ** (self.crosstalk_db / 10)
            crosstalk_noise = np.random.randn(*data.shape) * crosstalk_factor
            processed += crosstalk_noise
            
        return processed
    
    def _apply_noise_effects(self, data: np.ndarray) -> np.ndarray:
        """Apply various noise sources."""
        noisy_data = data.copy()
        
        # Shot noise (Poisson-like, but simplified as Gaussian)
        if self._noise_params["shot_noise_factor"] > 0:
            shot_noise = np.random.randn(*data.shape) * self._noise_params["shot_noise_factor"]
            noisy_data += shot_noise
            
        # Thermal noise
        if self._noise_params["thermal_noise_factor"] > 0:
            thermal_noise = np.random.randn(*data.shape) * self._noise_params["thermal_noise_factor"]
            noisy_data += thermal_noise
            
        # Phase drift (affects magnitude and phase)
        if self._noise_params["phase_drift_std"] > 0:
            phase_drift = np.random.normal(0, self._noise_params["phase_drift_std"], data.shape)
            # Approximate phase drift effect on magnitude
            phase_error_factor = np.cos(phase_drift)  # Simplified model
            noisy_data *= phase_error_factor
            
        return noisy_data
    
    def _simulate_optical_decoding(self, data: np.ndarray) -> np.ndarray:
        """Simulate optical-to-electronic conversion."""
        # Apply photodetector responsivity (simplified)
        responsivity = 0.9  # Typical photodetector responsivity
        return data * responsivity
    
    def _estimate_output_power(self, data: np.ndarray) -> float:
        """Estimate optical output power in mW."""
        # Simplified power estimation based on signal magnitude
        rms_value = np.sqrt(np.mean(data**2))
        return min(rms_value * 10.0, 100.0)  # Cap at 100mW
    
    def get_simulation_report(self) -> Dict[str, Any]:
        """Get detailed simulation parameters and statistics."""
        return {
            "noise_model": self.noise_model,
            "precision": self.precision,
            "crosstalk_db": self.crosstalk_db,
            "noise_parameters": self._noise_params,
            "target_config": self.target_config.to_dict()
        }
    
    def compare_with_ideal(self, ideal_output, photonic_output) -> Dict[str, float]:
        """Compare photonic simulation with ideal computation."""
        if hasattr(ideal_output, 'detach'):
            ideal_data = ideal_output.detach().numpy()
        else:
            ideal_data = np.array(ideal_output)
            
        if hasattr(photonic_output, 'data'):
            photonic_data = photonic_output.data
        else:
            photonic_data = np.array(photonic_output)
            
        # Calculate error metrics
        mse = np.mean((ideal_data - photonic_data) ** 2)
        mae = np.mean(np.abs(ideal_data - photonic_data))
        
        # Signal-to-noise ratio
        signal_power = np.mean(ideal_data ** 2)
        noise_power = np.mean((ideal_data - photonic_data) ** 2)
        snr_db = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
        
        return {
            "mse": float(mse),
            "mae": float(mae), 
            "snr_db": float(snr_db),
            "max_error": float(np.max(np.abs(ideal_data - photonic_data)))
        }


class NoiseModel(Enum):
    """Available noise models for photonic simulation."""
    IDEAL = "ideal"
    REALISTIC = "realistic"
    PESSIMISTIC = "pessimistic"
    QUANTUM_LIMITED = "quantum_limited"
    RESEARCH_GRADE = "research_grade"


@dataclass
class SimulationParams:
    """Parameters for advanced photonic simulation."""
    noise_model: NoiseModel = NoiseModel.REALISTIC
    precision: str = "8bit"
    crosstalk_db: float = -30.0
    temperature_k: float = 300.0  # Room temperature
    wavelength_nm: int = 1550
    quantum_efficiency: float = 0.9
    dark_current_na: float = 1.0  # Dark current in nA
    thermal_coefficient: float = 1e-4  # Thermal refractive index coefficient
    fabrication_tolerance: float = 0.05  # 5% fabrication variations
    enable_quantum_effects: bool = False
    enable_nonlinear_effects: bool = False
    mesh_fidelity: float = 0.95  # Mesh calibration fidelity


class AdvancedPhotonicSimulator:
    """
    Research-grade photonic simulator with advanced physics modeling.
    
    Features:
    - Quantum noise effects (shot noise, thermal noise)
    - Thermal drift and compensation
    - Fabrication variations
    - Nonlinear optical effects
    - Wavelength-dependent loss
    - Crosstalk modeling
    - Multi-wavelength simulation
    - Mach-Zehnder interferometer networks
    """
    
    def __init__(self, params: SimulationParams, target_config: TargetConfig):
        self.params = params
        self.config = target_config
        self.logger = logging.getLogger(__name__)
        
        # Initialize physics models
        self.thermal_model = self._initialize_thermal_model()
        self.fabrication_variations = self._generate_fabrication_variations()
        self.wavelength_response = self._calculate_wavelength_response()
        
        # Performance tracking
        self.simulation_history: List[Dict[str, Any]] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def simulate_photonic_network(self, input_data: np.ndarray, 
                                network_description: Dict[str, Any]) -> PhotonicTensor:
        """
        Simulate complete photonic network with advanced physics.
        
        Args:
            input_data: Input tensor data
            network_description: Description of photonic network topology
            
        Returns:
            PhotonicTensor with simulated output and metadata
        """
        start_time = time.time()
        
        # Validate inputs
        if input_data.size == 0:
            raise ValueError("Input data cannot be empty")
        
        self.logger.info(f"Starting advanced photonic simulation")
        self.logger.debug(f"Input shape: {input_data.shape}, Network: {network_description.get('type', 'unknown')}")
        
        # Stage 1: Electronic-to-optical conversion
        optical_signals = self._electronic_to_optical_conversion(input_data)
        
        # Stage 2: Photonic processing through network
        processed_signals = self._process_through_photonic_network(
            optical_signals, network_description
        )
        
        # Stage 3: Apply advanced physics effects
        physics_affected = self._apply_advanced_physics(processed_signals)
        
        # Stage 4: Optical-to-electronic conversion
        output_data = self._optical_to_electronic_conversion(physics_affected)
        
        # Stage 5: Post-processing and metrics
        simulation_time = time.time() - start_time
        metrics = self._calculate_simulation_metrics(
            input_data, output_data, simulation_time
        )
        
        # Create result tensor
        result = PhotonicTensor(
            data=output_data,
            wavelength=self.params.wavelength_nm,
            power_mw=metrics['output_power_mw']
        )
        
        # Store simulation record
        self.simulation_history.append({
            'timestamp': time.time(),
            'simulation_time_ms': simulation_time * 1000,
            'input_shape': input_data.shape,
            'output_shape': output_data.shape,
            'metrics': metrics,
            'network_type': network_description.get('type', 'unknown')
        })
        
        self.logger.info(f"Simulation completed in {simulation_time*1000:.2f}ms")
        return result
    
    def simulate_mzi_mesh(self, input_matrix: np.ndarray, 
                         weight_matrix: np.ndarray) -> np.ndarray:
        """
        Simulate Mach-Zehnder Interferometer mesh for matrix multiplication.
        
        Args:
            input_matrix: Input optical signals
            weight_matrix: MZI phase configuration
            
        Returns:
            Output optical signals after MZI processing
        """
        mesh_size = self.config.array_size
        
        # Validate matrix dimensions
        if weight_matrix.shape != mesh_size:
            raise ValueError(f"Weight matrix shape {weight_matrix.shape} must match mesh size {mesh_size}")
        
        # Initialize mesh state
        mesh_state = np.zeros(mesh_size, dtype=complex)
        
        # Process through MZI layers
        for layer_idx in range(mesh_size[0]):
            mesh_state = self._process_mzi_layer(
                mesh_state, weight_matrix[layer_idx], layer_idx
            )
        
        # Apply mesh fidelity limitations
        if self.params.mesh_fidelity < 1.0:
            fidelity_noise = np.random.randn(*mesh_state.shape) * (1 - self.params.mesh_fidelity)
            mesh_state += fidelity_noise
        
        # Convert complex amplitudes to intensities
        output_intensities = np.abs(mesh_state) ** 2
        
        return output_intensities
    
    def simulate_quantum_effects(self, optical_signal: np.ndarray) -> np.ndarray:
        """
        Simulate quantum effects in photonic circuits.
        
        Args:
            optical_signal: Input optical signal amplitudes
            
        Returns:
            Signal with quantum noise effects applied
        """
        if not self.params.enable_quantum_effects:
            return optical_signal
        
        # Shot noise (Poisson statistics)
        shot_noise = self._simulate_shot_noise(optical_signal)
        
        # Quantum vacuum fluctuations
        vacuum_noise = self._simulate_vacuum_fluctuations(optical_signal.shape)
        
        # Thermal photons (for finite temperature)
        thermal_photons = self._simulate_thermal_photons(optical_signal.shape)
        
        # Combine quantum effects
        quantum_affected = optical_signal + shot_noise + vacuum_noise + thermal_photons
        
        return quantum_affected
    
    def simulate_thermal_effects(self, signal: np.ndarray, 
                               thermal_map: np.ndarray) -> np.ndarray:
        """
        Simulate thermal effects on photonic devices.
        
        Args:
            signal: Input optical signal
            thermal_map: Temperature distribution across device
            
        Returns:
            Signal affected by thermal variations
        """
        # Thermal phase shifts
        thermal_phase_shift = (thermal_map - self.params.temperature_k) * self.params.thermal_coefficient
        
        # Apply phase modulation
        phase_modulated = signal * np.exp(1j * thermal_phase_shift)
        
        # Thermal noise power spectral density
        thermal_noise_power = self._calculate_thermal_noise_power(thermal_map)
        thermal_noise = np.random.randn(*signal.shape) * np.sqrt(thermal_noise_power)
        
        return np.abs(phase_modulated) + thermal_noise
    
    def _initialize_thermal_model(self) -> Dict[str, Any]:
        """Initialize thermal modeling parameters."""
        return {
            'thermal_time_constant': 1e-3,  # 1ms thermal time constant
            'thermal_conductivity': 150.0,  # Silicon thermal conductivity W/m·K
            'specific_heat': 700.0,  # Silicon specific heat J/kg·K
            'ambient_temperature': self.params.temperature_k
        }
    
    def _generate_fabrication_variations(self) -> np.ndarray:
        """Generate fabrication variation map."""
        mesh_size = self.config.array_size
        variations = np.random.normal(1.0, self.params.fabrication_tolerance, mesh_size)
        return np.clip(variations, 0.5, 1.5)  # Reasonable bounds
    
    def _calculate_wavelength_response(self) -> Callable[[float], float]:
        """Calculate wavelength-dependent response function."""
        def response(wavelength_nm: float) -> float:
            # Simplified wavelength response (would be device-specific)
            center_wl = 1550.0
            bandwidth = 100.0
            return np.exp(-0.5 * ((wavelength_nm - center_wl) / bandwidth) ** 2)
        
        return response
    
    def _electronic_to_optical_conversion(self, electronic_data: np.ndarray) -> np.ndarray:
        """Convert electronic signals to optical domain."""
        # Apply quantization based on precision
        quantized = self._apply_quantization(electronic_data)
        
        # Modulator response (assume Mach-Zehnder modulator)
        modulator_efficiency = 0.85
        optical_amplitude = np.sqrt(quantized * modulator_efficiency)
        
        # Add modulator nonlinearity if enabled
        if self.params.enable_nonlinear_effects:
            optical_amplitude = self._apply_modulator_nonlinearity(optical_amplitude)
        
        # Wavelength response
        wl_response = self.wavelength_response(self.params.wavelength_nm)
        optical_amplitude *= wl_response
        
        return optical_amplitude
    
    def _process_through_photonic_network(self, optical_signals: np.ndarray, 
                                        network_desc: Dict[str, Any]) -> np.ndarray:
        """Process signals through photonic network."""
        network_type = network_desc.get('type', 'linear')
        
        if network_type == 'mzi_mesh':
            # Extract weight configuration from network description
            weights = network_desc.get('weights', np.ones(self.config.array_size))
            return self.simulate_mzi_mesh(optical_signals, weights)
        
        elif network_type == 'wavelength_multiplex':
            # Simulate WDM processing
            return self._simulate_wdm_processing(optical_signals, network_desc)
        
        else:
            # Default linear processing
            processing_matrix = network_desc.get('matrix', np.eye(optical_signals.shape[0]))
            return np.dot(processing_matrix, optical_signals)
    
    def _apply_advanced_physics(self, signals: np.ndarray) -> np.ndarray:
        """Apply advanced physics effects."""
        physics_signals = signals.copy()
        
        # Quantum effects
        if self.params.enable_quantum_effects:
            physics_signals = self.simulate_quantum_effects(physics_signals)
        
        # Thermal effects (simplified uniform temperature)
        thermal_map = np.full(self.config.array_size, self.params.temperature_k)
        physics_signals = self.simulate_thermal_effects(physics_signals, thermal_map)
        
        # Fabrication variations
        physics_signals *= self.fabrication_variations.flatten()[:len(physics_signals)]
        
        # Crosstalk effects
        if self.params.crosstalk_db < 0:
            physics_signals = self._apply_crosstalk(physics_signals)
        
        return physics_signals
    
    def _optical_to_electronic_conversion(self, optical_signals: np.ndarray) -> np.ndarray:
        """Convert optical signals back to electronic domain."""
        # Photodetector quantum efficiency
        detected_photons = optical_signals * self.params.quantum_efficiency
        
        # Shot noise in detection
        if self.params.noise_model != NoiseModel.IDEAL:
            shot_noise = np.random.poisson(detected_photons * 1000) / 1000.0 - detected_photons
            detected_photons += shot_noise * 0.01  # Scale shot noise
        
        # Dark current
        dark_current_electrons = self.params.dark_current_na * 1e-9 / 1.602e-19  # Convert to electrons
        detected_photons += dark_current_electrons
        
        # Transimpedance amplifier (TIA) conversion
        tia_gain = 1000.0  # V/A
        electronic_output = detected_photons * tia_gain * 1e-6  # Simplified conversion
        
        return electronic_output
    
    def _apply_quantization(self, data: np.ndarray) -> np.ndarray:
        """Apply precision quantization."""
        if self.params.precision == "8bit":
            return np.clip(np.round(data * 127) / 127.0, -1.0, 1.0)
        elif self.params.precision == "16bit":
            return np.clip(np.round(data * 32767) / 32767.0, -1.0, 1.0)
        elif self.params.precision == "fp16":
            return data.astype(np.float16).astype(np.float32)
        else:  # fp32
            return data.astype(np.float32)
    
    def _simulate_shot_noise(self, signal: np.ndarray) -> np.ndarray:
        """Simulate quantum shot noise."""
        # Shot noise is proportional to sqrt(signal)
        shot_noise_std = np.sqrt(np.abs(signal) + 1e-10)  # Avoid division by zero
        return np.random.normal(0, shot_noise_std * 0.01, signal.shape)
    
    def _simulate_vacuum_fluctuations(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Simulate quantum vacuum fluctuations."""
        # Vacuum fluctuations are fundamental quantum noise
        return np.random.normal(0, 0.001, shape)
    
    def _simulate_thermal_photons(self, shape: Tuple[int, ...]) -> np.ndarray:
        """Simulate thermal photon noise."""
        # Bose-Einstein distribution for thermal photons
        k_b = 1.381e-23  # Boltzmann constant
        h = 6.626e-34    # Planck constant
        c = 3e8          # Speed of light
        
        frequency = c / (self.params.wavelength_nm * 1e-9)
        thermal_energy = h * frequency / (k_b * self.params.temperature_k)
        
        # Mean number of thermal photons
        n_thermal = 1.0 / (np.exp(thermal_energy) - 1) if thermal_energy < 10 else 0.0
        
        return np.random.poisson(n_thermal * 0.01, shape).astype(float) * 0.001
    
    def _calculate_thermal_noise_power(self, temperature_map: np.ndarray) -> np.ndarray:
        """Calculate thermal noise power spectral density."""
        k_b = 1.381e-23
        return 4 * k_b * temperature_map * 1e12  # Simplified thermal noise power
    
    def _apply_modulator_nonlinearity(self, amplitude: np.ndarray) -> np.ndarray:
        """Apply Mach-Zehnder modulator nonlinearity."""
        # Sinusoidal transfer function for MZ modulator
        return np.sin(np.pi * amplitude / 2)
    
    def _process_mzi_layer(self, state: np.ndarray, weights: np.ndarray, 
                          layer_idx: int) -> np.ndarray:
        """Process signals through a layer of MZI switches."""
        output_state = state.copy()
        
        # Apply MZI transfer function for each switch
        for i in range(0, len(state) - 1, 2):
            if i + 1 < len(state):
                # 2x2 MZI operation
                theta = weights[i // 2] if i // 2 < len(weights) else 0.0
                mzi_matrix = np.array([
                    [np.cos(theta), -1j * np.sin(theta)],
                    [-1j * np.sin(theta), np.cos(theta)]
                ], dtype=complex)
                
                input_pair = np.array([state[i], state[i + 1]])
                output_pair = np.dot(mzi_matrix, input_pair)
                output_state[i] = output_pair[0]
                output_state[i + 1] = output_pair[1]
        
        return output_state
    
    def _simulate_wdm_processing(self, signals: np.ndarray, 
                               network_desc: Dict[str, Any]) -> np.ndarray:
        """Simulate wavelength division multiplexing processing."""
        wavelengths = network_desc.get('wavelengths', [1550])
        
        # Process each wavelength channel
        processed_channels = []
        
        for wl in wavelengths:
            channel_response = self.wavelength_response(wl)
            channel_signal = signals * channel_response
            processed_channels.append(channel_signal)
        
        # Combine wavelength channels
        return np.sum(processed_channels, axis=0)
    
    def _apply_crosstalk(self, signals: np.ndarray) -> np.ndarray:
        """Apply optical crosstalk effects."""
        crosstalk_factor = 10 ** (self.params.crosstalk_db / 10)
        crosstalk_matrix = np.eye(len(signals)) + np.random.randn(len(signals), len(signals)) * crosstalk_factor
        return np.dot(crosstalk_matrix, signals)
    
    def _calculate_simulation_metrics(self, input_data: np.ndarray, 
                                    output_data: np.ndarray, 
                                    sim_time: float) -> Dict[str, float]:
        """Calculate comprehensive simulation metrics."""
        # Power metrics
        input_power = np.mean(input_data ** 2)
        output_power = np.mean(output_data ** 2)
        
        # Signal metrics
        signal_dynamic_range = np.max(output_data) - np.min(output_data)
        
        return {
            'simulation_time_ms': sim_time * 1000,
            'input_power_linear': float(input_power),
            'output_power_mw': float(output_power * 10),  # Convert to mW scale
            'signal_dynamic_range': float(signal_dynamic_range),
            'processing_efficiency': float(output_power / input_power) if input_power > 0 else 0.0,
            'noise_model': self.params.noise_model.value,
            'quantum_effects_enabled': self.params.enable_quantum_effects
        }
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Get comprehensive simulation report."""
        if not self.simulation_history:
            return {"message": "No simulations performed yet"}
        
        # Aggregate statistics
        sim_times = [s['simulation_time_ms'] for s in self.simulation_history]
        total_simulations = len(self.simulation_history)
        
        return {
            'total_simulations': total_simulations,
            'avg_simulation_time_ms': np.mean(sim_times),
            'total_simulation_time_ms': np.sum(sim_times),
            'parameters': {
                'noise_model': self.params.noise_model.value,
                'precision': self.params.precision,
                'temperature_k': self.params.temperature_k,
                'quantum_effects': self.params.enable_quantum_effects,
                'mesh_fidelity': self.params.mesh_fidelity
            },
            'target_config': self.config.to_dict(),
            'recent_simulations': self.simulation_history[-5:]  # Last 5 simulations
        }


# Enhanced PhotonicSimulator with backwards compatibility
class PhotonicSimulator(AdvancedPhotonicSimulator):
    """Enhanced photonic simulator with backwards compatibility."""
    
    def __init__(self, 
                 noise_model: str = "realistic",
                 precision: str = "8bit", 
                 crosstalk_db: float = -30.0,
                 target_config: Optional[TargetConfig] = None,
                 enable_advanced_physics: bool = True):
        """Initialize with backwards-compatible interface."""
        # Convert old noise model strings to enum
        noise_model_map = {
            "ideal": NoiseModel.IDEAL,
            "realistic": NoiseModel.REALISTIC,
            "pessimistic": NoiseModel.PESSIMISTIC
        }
        
        params = SimulationParams(
            noise_model=noise_model_map.get(noise_model, NoiseModel.REALISTIC),
            precision=precision,
            crosstalk_db=crosstalk_db,
            enable_quantum_effects=enable_advanced_physics,
            enable_nonlinear_effects=enable_advanced_physics
        )
        
        config = target_config or TargetConfig()
        super().__init__(params, config)
        
        # Store old interface parameters for compatibility
        self.noise_model = noise_model
        self.precision = precision
        self.crosstalk_db = crosstalk_db
        self.target_config = config
        self._noise_params = self._get_noise_parameters()
    
    def _get_noise_parameters(self) -> Dict[str, float]:
        """Backwards compatibility method."""
        if self.params.noise_model == NoiseModel.IDEAL:
            return {
                "shot_noise_factor": 0.0,
                "thermal_noise_factor": 0.0,
                "phase_drift_std": 0.0,
                "insertion_loss_db": 0.0
            }
        elif self.params.noise_model == NoiseModel.REALISTIC:
            return {
                "shot_noise_factor": 0.005,
                "thermal_noise_factor": 0.002,
                "phase_drift_std": 0.05,
                "insertion_loss_db": -0.5
            }
        else:  # PESSIMISTIC
            return {
                "shot_noise_factor": 0.02,
                "thermal_noise_factor": 0.01,
                "phase_drift_std": 0.15,
                "insertion_loss_db": -2.0
            }
    
    def run(self, compiled_model, input_data) -> PhotonicTensor:
        """Backwards-compatible run method."""
        # Convert input data
        if hasattr(input_data, 'numpy'):
            data = input_data.detach().numpy()
        elif hasattr(input_data, 'data'):
            data = input_data.data
        else:
            data = np.array(input_data, dtype=np.float32)
        
        # Create simple network description for backwards compatibility
        network_desc = {
            'type': 'linear',
            'matrix': np.eye(data.shape[-1]) if data.ndim > 1 else np.array([[1.0]])
        }
        
        return self.simulate_photonic_network(data, network_desc)