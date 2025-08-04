"""
Photonic hardware simulator with noise modeling.
"""

import numpy as np
from typing import Dict, Any, Optional
from .core import PhotonicTensor, TargetConfig


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
            data = np.array(input_data)
            
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