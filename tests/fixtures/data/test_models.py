"""Generate test models and data for the test suite."""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List


def create_test_models() -> Dict[str, nn.Module]:
    """Create a collection of test models for various scenarios."""
    
    models = {}
    
    # Simple linear model
    models['simple_linear'] = nn.Linear(10, 5)
    
    # Multi-layer perceptron
    models['mlp'] = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # Convolutional network
    models['cnn'] = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    )
    
    # ResNet-like block
    class ResidualBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += residual
            return self.relu(out)
    
    models['resnet_block'] = ResidualBlock(64)
    
    # Transformer encoder layer
    models['transformer'] = nn.TransformerEncoderLayer(
        d_model=512, nhead=8, dim_feedforward=2048, dropout=0.1, batch_first=True
    )
    
    # Attention mechanism
    models['attention'] = nn.MultiheadAttention(
        embed_dim=256, num_heads=8, batch_first=True
    )
    
    # LSTM model
    models['lstm'] = nn.LSTM(
        input_size=100, hidden_size=128, num_layers=2, batch_first=True
    )
    
    # Complex model with multiple components
    class ComplexModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(64, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, 10)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.conv(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    models['complex'] = ComplexModel()
    
    # Very small model for quick tests
    models['tiny'] = nn.Linear(2, 1)
    
    # Very large model for stress tests
    models['large'] = nn.Sequential(*[
        nn.Linear(1024, 1024),
        nn.ReLU()
    ] * 10 + [nn.Linear(1024, 100)])
    
    return models


def create_test_inputs() -> Dict[str, torch.Tensor]:
    """Create test input tensors for different model types."""
    
    inputs = {
        'linear_small': torch.randn(1, 10),
        'linear_batch': torch.randn(32, 10),
        'linear_large': torch.randn(1, 1024),
        
        'conv_small': torch.randn(1, 3, 32, 32),
        'conv_batch': torch.randn(16, 3, 32, 32),
        'conv_large': torch.randn(1, 3, 224, 224),
        
        'sequence_short': torch.randn(1, 10, 100),
        'sequence_long': torch.randn(1, 512, 100),
        'sequence_batch': torch.randn(32, 50, 100),
        
        'transformer_input': torch.randn(1, 100, 512),
        'attention_input': torch.randn(1, 50, 256),
        
        'tiny_input': torch.randn(1, 2),
        'stress_input': torch.randn(128, 1024),
    }
    
    return inputs


def create_expected_outputs() -> Dict[str, Dict[str, Any]]:
    """Create expected outputs and metadata for test validation."""
    
    models = create_test_models()
    inputs = create_test_inputs()
    
    expected = {}
    
    with torch.no_grad():
        # Simple linear model
        expected['simple_linear'] = {
            'output_shape': (1, 5),
            'parameter_count': 55,  # 10*5 + 5
            'expected_output': models['simple_linear'](inputs['linear_small'])
        }
        
        # MLP
        expected['mlp'] = {
            'output_shape': (1, 10),
            'parameter_count': 784*256 + 256 + 256*128 + 128 + 128*10 + 10,
            'expected_output': models['mlp'](inputs['linear_large'])  # 784 input
        }
        
        # CNN
        expected['cnn'] = {
            'output_shape': (1, 10),
            'parameter_count': None,  # Complex to calculate
            'expected_output': models['cnn'](inputs['conv_small'])
        }
        
        # Transformer
        expected['transformer'] = {
            'output_shape': (1, 100, 512),
            'parameter_count': None,
            'expected_output': models['transformer'](inputs['transformer_input'])
        }
    
    return expected


def generate_photonic_assembly_examples() -> Dict[str, str]:
    """Generate example photonic assembly code for testing."""
    
    examples = {
        'simple_linear': """
# Simple linear transformation
.model simple_linear
.precision int8
.mesh butterfly_64x64

# Load weights into photonic mesh
PLOAD %weight_matrix, @linear_weights
PCFG %mesh_config, butterfly_decomp

# Input encoding to optical domain
PENC %optical_input, %electronic_input, wavelength=1550

# Photonic matrix multiplication
PMUL %result, %optical_input, %weight_matrix

# Bias addition (electronic domain)
PDEC %electronic_result, %result
EADD %final_result, %electronic_result, %bias

# Output
STORE %final_result, @output
""",
        
        'with_thermal_compensation': """
# Linear transformation with thermal compensation
.model thermal_linear
.precision int8
.mesh butterfly_64x64

# Thermal sensor reading
TSENSE %temperature, sensor_id=0

# Calculate thermal compensation
TCALC %compensation, %temperature, @thermal_model

# Apply compensation to phase array
TCOMP %phase_array, %compensation

# Load compensated weights
PLOAD %weight_matrix, @linear_weights
PCOMP %corrected_weights, %weight_matrix, %phase_array

# Matrix multiplication with compensation
PENC %optical_input, %electronic_input, wavelength=1550
PMUL %result, %optical_input, %corrected_weights
PDEC %electronic_result, %result

STORE %electronic_result, @output
""",
        
        'multi_wavelength': """
# Multi-wavelength operation
.model multi_wl
.precision int8
.mesh butterfly_64x64

# Wavelength 1: 1530nm
PENC %opt_input_1530, %electronic_input, wavelength=1530
PMUL %result_1530, %opt_input_1530, %weights_1530

# Wavelength 2: 1550nm  
PENC %opt_input_1550, %electronic_input, wavelength=1550
PMUL %result_1550, %opt_input_1550, %weights_1550

# Wavelength 3: 1570nm
PENC %opt_input_1570, %electronic_input, wavelength=1570
PMUL %result_1570, %opt_input_1570, %weights_1570

# Combine wavelength results
PWLCOMB %combined, %result_1530, %result_1550, %result_1570
PDEC %final_result, %combined

STORE %final_result, @output
""",
        
        'convolution': """
# Photonic convolution implementation
.model photonic_conv
.precision int8
.mesh butterfly_64x64

# Unfold input for matrix multiplication
PUNFOLD %unfolded_input, %input_tensor, kernel_size=[3,3], stride=[1,1]

# Reshape convolution as matrix multiplication
PRESHAPE %conv_weights, @conv_kernel, shape=[out_channels, kernel_elements]

# Photonic matrix multiplication
PENC %optical_unfolded, %unfolded_input, wavelength=1550
PMUL %conv_result, %optical_unfolded, %conv_weights

# Fold back to spatial dimensions
PFOLD %spatial_result, %conv_result, output_shape=[batch, channels, height, width]
PDEC %final_conv, %spatial_result

STORE %final_conv, @output
"""
    }
    
    return examples


def create_hardware_configs() -> Dict[str, Dict[str, Any]]:
    """Create hardware configuration examples for testing."""
    
    configs = {
        'lightmatter_envise': {
            'device_type': 'lightmatter_envise',
            'array_size': [64, 64],
            'wavelengths': [1530, 1540, 1550, 1560, 1570],  # nm
            'max_power': 100.0,  # mW
            'thermal_range': [20.0, 80.0],  # Celsius
            'phase_precision': 0.01,  # radians
            'insertion_loss': 0.1,  # dB per component
            'crosstalk': -30.0,  # dB
            'bandwidth': 10.0,  # GHz
            'memory_size': 1024,  # MB
            'calibration_interval': 60,  # seconds
        },
        
        'mit_photonics': {
            'device_type': 'mit_silicon_photonics',
            'array_size': [32, 32],
            'wavelengths': [1550],
            'max_power': 50.0,
            'thermal_range': [15.0, 85.0],
            'phase_precision': 0.005,
            'insertion_loss': 0.05,
            'crosstalk': -35.0,
            'bandwidth': 5.0,
            'memory_size': 512,
            'calibration_interval': 30,
        },
        
        'research_chip': {
            'device_type': 'custom_research',
            'array_size': [16, 16],
            'wavelengths': [1310, 1550],
            'max_power': 25.0,
            'thermal_range': [10.0, 90.0],
            'phase_precision': 0.001,
            'insertion_loss': 0.2,
            'crosstalk': -25.0,
            'bandwidth': 20.0,
            'memory_size': 256,
            'calibration_interval': 120,
        },
        
        'simulator': {
            'device_type': 'photonic_simulator',
            'array_size': [128, 128],  # Unlimited in simulation
            'wavelengths': list(range(1530, 1571, 5)),  # Full C-band
            'max_power': float('inf'),
            'thermal_range': [-40.0, 125.0],
            'phase_precision': 1e-6,
            'insertion_loss': 0.0,
            'crosstalk': -60.0,  # Perfect isolation in simulation
            'bandwidth': float('inf'),
            'memory_size': 16384,  # 16GB
            'calibration_interval': 0,  # No calibration needed
            'noise_model': 'gaussian',
            'thermal_model': 'linear',
            'enable_nonlinear_effects': False,
        }
    }
    
    return configs


def save_test_data(output_dir: Path) -> None:
    """Save all test data to files."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    models = create_test_models()
    models_dir = output_dir / 'models'
    models_dir.mkdir(exist_ok=True)
    
    for name, model in models.items():
        torch.save(model.state_dict(), models_dir / f'{name}.pth')
        
        # Also save model architecture info
        model_info = {
            'name': name,
            'type': str(type(model)),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad)
        }
        
        with open(models_dir / f'{name}_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
    
    # Save test inputs
    inputs = create_test_inputs()
    inputs_dir = output_dir / 'inputs'
    inputs_dir.mkdir(exist_ok=True)
    
    for name, tensor in inputs.items():
        torch.save(tensor, inputs_dir / f'{name}.pt')
    
    # Save expected outputs
    expected = create_expected_outputs()
    expected_dir = output_dir / 'expected'
    expected_dir.mkdir(exist_ok=True)
    
    for name, data in expected.items():
        # Save tensors separately
        if 'expected_output' in data:
            torch.save(data['expected_output'], expected_dir / f'{name}_output.pt')
            data = data.copy()
            del data['expected_output']
        
        with open(expected_dir / f'{name}.json', 'w') as f:
            json.dump(data, f, indent=2)
    
    # Save assembly examples
    assembly = generate_photonic_assembly_examples()
    assembly_dir = output_dir / 'assembly'
    assembly_dir.mkdir(exist_ok=True)
    
    for name, code in assembly.items():
        with open(assembly_dir / f'{name}.pasm', 'w') as f:
            f.write(code)
    
    # Save hardware configs
    hw_configs = create_hardware_configs()
    with open(output_dir / 'hardware_configs.json', 'w') as f:
        json.dump(hw_configs, f, indent=2)


if __name__ == '__main__':
    # Generate test data when run as script
    test_data_dir = Path(__file__).parent
    save_test_data(test_data_dir)
    print(f"Test data generated in {test_data_dir}")