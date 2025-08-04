"""
Configuration management for photonic compiler.
"""

import os
import json
import yaml
import toml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, asdict, field
import logging

from .core import TargetConfig, Device, Precision
from .security import InputSanitizer, ConfigurationSecurity, SecurityError

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Optional[str] = None
    max_file_size_mb: int = 100
    backup_count: int = 5
    console_output: bool = True
    json_format: bool = False


@dataclass
class SecurityConfig:
    """Security configuration."""
    max_model_file_size_mb: int = 100
    max_config_file_size_mb: int = 1
    allowed_model_extensions: List[str] = field(default_factory=lambda: ['.onnx', '.mlir', '.pt', '.pth'])
    input_validation_enabled: bool = True
    secure_temp_files: bool = True
    max_compilation_time_seconds: int = 3600
    max_simulation_time_seconds: int = 1800


@dataclass
class CompilerConfig:
    """Compiler configuration."""
    optimization_level: int = 2
    enable_thermal_compensation: bool = True
    enable_phase_optimization: bool = True
    enable_power_balancing: bool = True
    debug_mode: bool = False
    verbose_output: bool = False
    intermediate_output_dir: Optional[str] = None
    cache_compiled_models: bool = True
    cache_dir: Optional[str] = None


@dataclass
class SimulatorConfig:
    """Simulator configuration."""
    default_noise_model: str = "realistic"
    enable_crosstalk_simulation: bool = True
    default_crosstalk_db: float = -30.0
    enable_thermal_noise: bool = True
    enable_shot_noise: bool = True
    max_simulation_memory_gb: int = 8
    parallel_simulation: bool = True
    num_simulation_threads: int = 4


@dataclass
class PhotonicConfig:
    """Main photonic compiler configuration."""
    target: TargetConfig = field(default_factory=TargetConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PhotonicConfig':
        """Create configuration from dictionary."""
        # Handle nested target config
        if 'target' in data and isinstance(data['target'], dict):
            target_data = data['target']
            if 'device' in target_data and isinstance(target_data['device'], str):
                target_data['device'] = Device(target_data['device'])
            if 'precision' in target_data and isinstance(target_data['precision'], str):
                target_data['precision'] = Precision(target_data['precision'])
            data['target'] = TargetConfig(**target_data)
        
        # Handle other nested configs
        for config_name, config_class in [
            ('logging', LoggingConfig),
            ('security', SecurityConfig),
            ('compiler', CompilerConfig),
            ('simulator', SimulatorConfig)
        ]:
            if config_name in data and isinstance(data[config_name], dict):
                data[config_name] = config_class(**data[config_name])
        
        return cls(**data)


class ConfigurationManager:
    """Configuration management with validation and security."""
    
    # Default configuration search paths
    DEFAULT_CONFIG_PATHS = [
        "photon_mlir.json",
        "photon_mlir.yaml",
        "photon_mlir.yml", 
        "photon_mlir.toml",
        "~/.photon_mlir.json",
        "~/.photon_mlir.yaml",
        "~/.config/photon_mlir/config.json",
        "/etc/photon_mlir/config.json"
    ]
    
    def __init__(self):
        self._config: Optional[PhotonicConfig] = None
        self._config_source: Optional[str] = None
    
    def load_config(self, 
                   config_path: Optional[str] = None,
                   config_dict: Optional[Dict[str, Any]] = None,
                   validate: bool = True) -> PhotonicConfig:
        """
        Load configuration from file or dictionary.
        
        Args:
            config_path: Path to configuration file (optional)
            config_dict: Configuration dictionary (optional)
            validate: Whether to validate configuration
            
        Returns:
            Loaded and validated configuration
            
        Raises:
            SecurityError: If configuration is invalid or unsafe
        """
        if config_dict is not None:
            # Load from dictionary
            config_data = config_dict
            self._config_source = "dictionary"
        elif config_path is not None:
            # Load from specific file
            config_data = self._load_config_file(config_path)
            self._config_source = config_path
        else:
            # Search for configuration file
            config_data, config_path = self._find_and_load_config()
            self._config_source = config_path or "defaults"
        
        # Merge with environment variables
        config_data = self._merge_environment_variables(config_data)
        
        # Create configuration object
        try:
            self._config = PhotonicConfig.from_dict(config_data)
        except (TypeError, ValueError) as e:
            raise SecurityError(f"Invalid configuration structure: {e}")
        
        # Validate configuration
        if validate:
            self._validate_configuration(self._config)
        
        logger.info(f"Configuration loaded from: {self._config_source}")
        
        return self._config
    
    def get_config(self) -> PhotonicConfig:
        """Get current configuration."""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def save_config(self, config: PhotonicConfig, file_path: str, format: str = "auto") -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            file_path: Output file path
            format: File format ("json", "yaml", "toml", or "auto")
        """
        # Validate and sanitize file path
        safe_path = InputSanitizer.validate_file_path(file_path, must_exist=False)
        
        # Determine format
        if format == "auto":
            ext = Path(safe_path).suffix.lower()
            if ext == ".json":
                format = "json"
            elif ext in [".yaml", ".yml"]:
                format = "yaml"
            elif ext == ".toml":
                format = "toml"
            else:
                format = "json"  # Default
        
        # Convert to dictionary and redact sensitive values for logging
        config_dict = config.to_dict()
        redacted_config = ConfigurationSecurity.redact_sensitive_config(config_dict)
        
        # Save configuration
        try:
            if format == "json":
                with open(safe_path, 'w') as f:
                    json.dump(config_dict, f, indent=2, default=str)
            elif format == "yaml":
                with open(safe_path, 'w') as f:
                    yaml.dump(config_dict, f, default_flow_style=False)
            elif format == "toml":
                with open(safe_path, 'w') as f:
                    toml.dump(config_dict, f)
            else:
                raise SecurityError(f"Unsupported format: {format}")
            
            logger.info(f"Configuration saved to: {safe_path}")
            
        except (OSError, IOError) as e:
            raise SecurityError(f"Failed to save configuration: {e}")
    
    def _load_config_file(self, file_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        # Validate file path
        safe_path = InputSanitizer.validate_file_path(file_path, must_exist=True)
        
        # Check file size
        InputSanitizer.validate_file_size(safe_path, 
                                        InputSanitizer.MAX_CONFIG_FILE_SIZE)
        
        # Determine file format
        ext = Path(safe_path).suffix.lower()
        
        try:
            with open(safe_path, 'r') as f:
                if ext == ".json":
                    return json.load(f)
                elif ext in [".yaml", ".yml"]:
                    return yaml.safe_load(f) or {}
                elif ext == ".toml":
                    return toml.load(f)
                else:
                    raise SecurityError(f"Unsupported configuration format: {ext}")
                    
        except (json.JSONDecodeError, yaml.YAMLError, toml.TomlDecodeError) as e:
            raise SecurityError(f"Invalid configuration file format: {e}")
        except (OSError, IOError) as e:
            raise SecurityError(f"Cannot read configuration file: {e}")
    
    def _find_and_load_config(self) -> tuple[Dict[str, Any], Optional[str]]:
        """Find and load configuration from default paths."""
        for config_path in self.DEFAULT_CONFIG_PATHS:
            expanded_path = os.path.expanduser(config_path)
            if os.path.exists(expanded_path):
                try:
                    config_data = self._load_config_file(expanded_path)
                    return config_data, expanded_path
                except SecurityError:
                    logger.warning(f"Failed to load configuration from: {expanded_path}")
                    continue
        
        # No configuration file found, return defaults
        logger.info("No configuration file found, using defaults")
        return {}, None
    
    def _merge_environment_variables(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Merge environment variables into configuration.""" 
        env_mappings = {
            "PHOTONIC_LOG_LEVEL": ("logging", "level"),
            "PHOTONIC_LOG_FILE": ("logging", "file"),
            "PHOTONIC_DEBUG": ("compiler", "debug_mode"),
            "PHOTONIC_VERBOSE": ("compiler", "verbose_output"),
            "PHOTONIC_CACHE_DIR": ("compiler", "cache_dir"),
            "PHOTONIC_DEVICE": ("target", "device"),
            "PHOTONIC_PRECISION": ("target", "precision"),
            "PHOTONIC_WAVELENGTH": ("target", "wavelength_nm"),
            "PHOTONIC_ARRAY_SIZE": ("target", "array_size"),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                # Initialize section if not exists
                if section not in config_data:
                    config_data[section] = {}
                
                # Convert value based on expected type
                if key in ["debug_mode", "verbose_output"]:
                    config_data[section][key] = value.lower() in ["true", "1", "yes", "on"]
                elif key == "wavelength_nm":
                    try:
                        config_data[section][key] = int(value)
                    except ValueError:
                        logger.warning(f"Invalid wavelength value: {value}")
                elif key == "array_size":
                    try:
                        width, height = map(int, value.split(','))
                        config_data[section][key] = (width, height)
                    except ValueError:
                        logger.warning(f"Invalid array size value: {value}")
                else:
                    config_data[section][key] = value
                
                logger.debug(f"Applied environment variable: {env_var} = {value}")
        
        return config_data
    
    def _validate_configuration(self, config: PhotonicConfig) -> None:
        """Validate configuration for security and correctness."""
        errors = []
        
        # Validate logging configuration
        if config.logging.level not in ["TRACE", "DEBUG", "INFO", "WARN", "ERROR", "FATAL"]:
            errors.append(f"Invalid logging level: {config.logging.level}")
        
        if config.logging.max_file_size_mb < 1 or config.logging.max_file_size_mb > 1000:
            errors.append(f"Invalid log file size: {config.logging.max_file_size_mb}MB")
        
        # Validate security configuration
        if config.security.max_model_file_size_mb < 1 or config.security.max_model_file_size_mb > 10000:
            errors.append(f"Invalid max model file size: {config.security.max_model_file_size_mb}MB")
        
        if config.security.max_compilation_time_seconds < 1:
            errors.append("Compilation timeout must be positive")
        
        # Validate compiler configuration
        if config.compiler.optimization_level < 0 or config.compiler.optimization_level > 3:
            errors.append(f"Invalid optimization level: {config.compiler.optimization_level}")
        
        # Validate simulator configuration
        if config.simulator.default_noise_model not in ["ideal", "realistic", "pessimistic"]:
            errors.append(f"Invalid noise model: {config.simulator.default_noise_model}")
        
        if config.simulator.num_simulation_threads < 1 or config.simulator.num_simulation_threads > 64:
            errors.append(f"Invalid simulation thread count: {config.simulator.num_simulation_threads}")
        
        # Validate target configuration (reuse existing validation)
        try:
            from .diagnostics import CompilerDiagnostics
            result = CompilerDiagnostics.validate_target_config(config.target)
            if result.status == "critical":
                errors.append(f"Target configuration error: {result.message}")
        except ImportError:
            pass  # Diagnostics not available
        
        # Check for suspicious custom configuration
        if len(str(config.custom)) > 10000:  # 10KB limit for custom config
            errors.append("Custom configuration section too large")
        
        if errors:
            raise SecurityError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def get_config_summary(self) -> str:
        """Get a summary of current configuration."""
        if not self._config:
            return "No configuration loaded"
        
        summary = []
        summary.append(f"Configuration source: {self._config_source}")
        summary.append(f"Target device: {self._config.target.device.value}")
        summary.append(f"Precision: {self._config.target.precision.value}")
        summary.append(f"Array size: {self._config.target.array_size[0]}x{self._config.target.array_size[1]}")
        summary.append(f"Wavelength: {self._config.target.wavelength_nm}nm")
        summary.append(f"Logging level: {self._config.logging.level}")
        summary.append(f"Debug mode: {self._config.compiler.debug_mode}")
        summary.append(f"Optimization level: {self._config.compiler.optimization_level}")
        
        return "\n".join(summary)


# Global configuration manager instance
config_manager = ConfigurationManager()


def get_config() -> PhotonicConfig:
    """Get global configuration instance."""
    return config_manager.get_config()


def load_config(config_path: Optional[str] = None, 
               config_dict: Optional[Dict[str, Any]] = None) -> PhotonicConfig:
    """Load configuration from file or dictionary."""
    return config_manager.load_config(config_path, config_dict)


def save_config(config: PhotonicConfig, file_path: str, format: str = "auto") -> None:
    """Save configuration to file."""
    config_manager.save_config(config, file_path, format)


def create_default_config() -> PhotonicConfig:
    """Create a default configuration with sensible defaults."""
    return PhotonicConfig()


def validate_config(config: PhotonicConfig) -> None:
    """Validate configuration."""
    temp_manager = ConfigurationManager()
    temp_manager._validate_configuration(config)