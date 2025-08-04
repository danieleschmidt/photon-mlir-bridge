"""
Security utilities for photonic compiler.
"""

import os
import re
import hashlib
import secrets
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related exception."""
    pass


class InputSanitizer:
    """Input validation and sanitization utilities."""
    
    # Safe filename pattern (alphanumeric, dash, underscore, dot)
    SAFE_FILENAME_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+$')
    
    # Maximum file sizes (in bytes)
    MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024  # 100MB
    MAX_CONFIG_FILE_SIZE = 1024 * 1024        # 1MB
    
    # Allowed file extensions
    ALLOWED_MODEL_EXTENSIONS = {'.onnx', '.mlir', '.pt', '.pth'}
    ALLOWED_CONFIG_EXTENSIONS = {'.json', '.yaml', '.yml', '.toml'}
    
    @staticmethod
    def validate_filename(filename: str) -> str:
        """
        Validate and sanitize filename.
        
        Args:
            filename: Input filename
            
        Returns:
            Sanitized filename
            
        Raises:
            SecurityError: If filename is unsafe
        """
        if not filename:
            raise SecurityError("Filename cannot be empty")
        
        # Remove directory traversal attempts
        clean_name = os.path.basename(filename)
        
        # Check for null bytes
        if '\0' in clean_name:
            raise SecurityError("Null bytes not allowed in filename")
        
        # Check for dangerous patterns
        if '..' in clean_name or clean_name.startswith('.'):
            raise SecurityError("Path traversal patterns not allowed")
        
        # Validate against safe pattern
        if not InputSanitizer.SAFE_FILENAME_PATTERN.match(clean_name):
            # Sanitize by removing unsafe characters
            clean_name = re.sub(r'[^a-zA-Z0-9._-]', '_', clean_name)
            logger.warning(f"Filename sanitized: {filename} -> {clean_name}")
        
        return clean_name
    
    @staticmethod
    def validate_file_path(file_path: str, must_exist: bool = True) -> str:
        """
        Validate file path for security.
        
        Args:
            file_path: Path to validate
            must_exist: Whether file must exist
            
        Returns:
            Resolved absolute path
            
        Raises:
            SecurityError: If path is unsafe
        """
        if not file_path:
            raise SecurityError("File path cannot be empty")
        
        # Check for null bytes
        if '\0' in file_path:
            raise SecurityError("Null bytes not allowed in file path")
        
        # Resolve to absolute path
        try:
            abs_path = os.path.abspath(file_path)
        except (OSError, ValueError) as e:
            raise SecurityError(f"Invalid file path: {e}")
        
        # Check for path traversal outside allowed directories
        # In production, you'd define allowed base directories
        if '..' in os.path.normpath(file_path):
            logger.warning(f"Path traversal detected in: {file_path}")
        
        if must_exist:
            if not os.path.exists(abs_path):
                raise SecurityError(f"File does not exist: {abs_path}")
            
            if not os.path.isfile(abs_path):
                raise SecurityError(f"Path is not a regular file: {abs_path}")
        
        return abs_path
    
    @staticmethod
    def validate_file_size(file_path: str, max_size: int) -> None:
        """
        Validate file size is within limits.
        
        Args:
            file_path: Path to file
            max_size: Maximum allowed size in bytes
            
        Raises:
            SecurityError: If file is too large
        """
        try:
            size = os.path.getsize(file_path)
            if size > max_size:
                raise SecurityError(
                    f"File too large: {size} bytes (max: {max_size})"
                )
        except OSError as e:
            raise SecurityError(f"Cannot check file size: {e}")
    
    @staticmethod
    def validate_model_file(file_path: str) -> str:
        """
        Comprehensive validation of model file.
        
        Args:
            file_path: Path to model file
            
        Returns:
            Validated absolute path
            
        Raises:
            SecurityError: If file is invalid
        """
        # Basic path validation
        abs_path = InputSanitizer.validate_file_path(file_path, must_exist=True)
        
        # Check file extension
        ext = Path(abs_path).suffix.lower()
        if ext not in InputSanitizer.ALLOWED_MODEL_EXTENSIONS:
            raise SecurityError(
                f"Unsupported model file extension: {ext}. "
                f"Allowed: {InputSanitizer.ALLOWED_MODEL_EXTENSIONS}"
            )
        
        # Check file size
        InputSanitizer.validate_file_size(abs_path, InputSanitizer.MAX_MODEL_FILE_SIZE)
        
        return abs_path
    
    @staticmethod
    def sanitize_string_input(input_str: str, max_length: int = 1000) -> str:
        """
        Sanitize string input by removing potentially dangerous content.
        
        Args:
            input_str: Input string
            max_length: Maximum allowed length
            
        Returns:
            Sanitized string
            
        Raises:
            SecurityError: If input is invalid
        """
        if not isinstance(input_str, str):
            raise SecurityError("Input must be a string")
        
        if len(input_str) > max_length:
            raise SecurityError(f"Input too long: {len(input_str)} (max: {max_length})")
        
        # Remove null bytes
        sanitized = input_str.replace('\0', '')
        
        # Remove control characters except common whitespace
        sanitized = ''.join(c for c in sanitized 
                          if ord(c) >= 32 or c in '\t\n\r')
        
        return sanitized
    
    @staticmethod
    def validate_numeric_input(value: Any, min_val: float = None, 
                              max_val: float = None) -> float:
        """
        Validate numeric input.
        
        Args:
            value: Input value
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Validated numeric value
            
        Raises:
            SecurityError: If value is invalid
        """
        try:
            numeric_val = float(value)
        except (ValueError, TypeError):
            raise SecurityError(f"Invalid numeric value: {value}")
        
        if not (-1e10 < numeric_val < 1e10):  # Reasonable bounds
            raise SecurityError(f"Numeric value out of reasonable range: {numeric_val}")
        
        if min_val is not None and numeric_val < min_val:
            raise SecurityError(f"Value {numeric_val} below minimum {min_val}")
        
        if max_val is not None and numeric_val > max_val:
            raise SecurityError(f"Value {numeric_val} above maximum {max_val}")
        
        return numeric_val


class SecureFileHandler:
    """Secure file operations."""
    
    @staticmethod
    def create_secure_temp_file(suffix: str = '.tmp') -> str:
        """
        Create a secure temporary file.
        
        Args:
            suffix: File suffix
            
        Returns:
            Path to secure temporary file
        """
        # Create with restricted permissions (owner only)
        fd, path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        # Set restrictive permissions
        os.chmod(path, 0o600)
        
        return path
    
    @staticmethod
    def secure_copy(src_path: str, dst_path: str) -> None:
        """
        Securely copy file with validation.
        
        Args:
            src_path: Source file path
            dst_path: Destination file path
            
        Raises:
            SecurityError: If operation is unsafe
        """
        # Validate source
        src_abs = InputSanitizer.validate_file_path(src_path, must_exist=True)
        
        # Validate destination
        dst_abs = InputSanitizer.validate_file_path(dst_path, must_exist=False)
        
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst_abs), exist_ok=True)
        
        try:
            with open(src_abs, 'rb') as src, open(dst_abs, 'wb') as dst:
                # Copy in chunks to avoid memory issues
                chunk_size = 64 * 1024  # 64KB chunks
                while True:
                    chunk = src.read(chunk_size)
                    if not chunk:
                        break
                    dst.write(chunk)
            
            # Set secure permissions
            os.chmod(dst_abs, 0o644)
            
        except (OSError, IOError) as e:
            raise SecurityError(f"File copy failed: {e}")
    
    @staticmethod
    def compute_file_hash(file_path: str, algorithm: str = 'sha256') -> str:
        """
        Compute secure hash of file.
        
        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('sha256', 'sha512')
            
        Returns:
            Hex digest of file hash
            
        Raises:
            SecurityError: If hashing fails
        """
        abs_path = InputSanitizer.validate_file_path(file_path, must_exist=True)
        
        try:
            hasher = hashlib.new(algorithm)
            with open(abs_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (OSError, ValueError) as e:
            raise SecurityError(f"Hash computation failed: {e}")


class ConfigurationSecurity:
    """Security utilities for configuration management."""
    
    # Sensitive configuration keys that should be redacted in logs
    SENSITIVE_KEYS = {
        'password', 'secret', 'key', 'token', 'credential',
        'api_key', 'private_key', 'cert', 'certificate'
    }
    
    @staticmethod
    def redact_sensitive_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Redact sensitive values from configuration for logging.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with sensitive values redacted
        """
        redacted = {}
        
        for key, value in config.items():
            key_lower = key.lower()
            
            # Check if key contains sensitive terms
            is_sensitive = any(term in key_lower for term in ConfigurationSecurity.SENSITIVE_KEYS)
            
            if is_sensitive:
                redacted[key] = "<REDACTED>"
            elif isinstance(value, dict):
                redacted[key] = ConfigurationSecurity.redact_sensitive_config(value)
            elif isinstance(value, list):
                redacted[key] = [
                    ConfigurationSecurity.redact_sensitive_config(item) 
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                redacted[key] = value
        
        return redacted
    
    @staticmethod
    def validate_config_structure(config: Dict[str, Any], 
                                 allowed_keys: Optional[List[str]] = None) -> None:
        """
        Validate configuration structure.
        
        Args:
            config: Configuration to validate
            allowed_keys: Optional list of allowed keys
            
        Raises:
            SecurityError: If configuration is invalid
        """
        if not isinstance(config, dict):
            raise SecurityError("Configuration must be a dictionary")
        
        if allowed_keys:
            for key in config.keys():
                if key not in allowed_keys:
                    raise SecurityError(f"Unknown configuration key: {key}")
        
        # Check for suspiciously large configurations
        if len(str(config)) > 100000:  # 100KB limit
            raise SecurityError("Configuration too large")


class ResourceLimits:
    """Resource usage limits for security."""
    
    # Memory limits
    MAX_TENSOR_SIZE_BYTES = 1024 * 1024 * 1024  # 1GB
    MAX_MODEL_PARAMETERS = 1000000000  # 1 billion parameters
    
    # Time limits
    MAX_COMPILATION_TIME_SECONDS = 3600  # 1 hour
    MAX_SIMULATION_TIME_SECONDS = 1800   # 30 minutes
    
    @staticmethod
    def check_tensor_size(shape: tuple, dtype_size: int) -> None:
        """
        Check if tensor size is within limits.
        
        Args:
            shape: Tensor shape
            dtype_size: Size of data type in bytes
            
        Raises:
            SecurityError: If tensor is too large
        """
        total_elements = 1
        for dim in shape:
            if dim <= 0:
                raise SecurityError(f"Invalid tensor dimension: {dim}")
            total_elements *= dim
        
        total_bytes = total_elements * dtype_size
        
        if total_bytes > ResourceLimits.MAX_TENSOR_SIZE_BYTES:
            raise SecurityError(
                f"Tensor too large: {total_bytes} bytes "
                f"(max: {ResourceLimits.MAX_TENSOR_SIZE_BYTES})"
            )
    
    @staticmethod
    def check_model_complexity(num_parameters: int) -> None:
        """
        Check if model complexity is within limits.
        
        Args:
            num_parameters: Number of model parameters
            
        Raises:
            SecurityError: If model is too complex
        """
        if num_parameters > ResourceLimits.MAX_MODEL_PARAMETERS:
            raise SecurityError(
                f"Model too complex: {num_parameters} parameters "
                f"(max: {ResourceLimits.MAX_MODEL_PARAMETERS})"
            )


# Security context manager
class SecurityContext:
    """Security context for operations."""
    
    def __init__(self, operation: str):
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = __import__('time').time()
        logger.info(f"Starting secure operation: {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Security error in {self.operation}: {exc_val}")
        else:
            duration = __import__('time').time() - self.start_time
            logger.info(f"Completed secure operation: {self.operation} ({duration:.2f}s)")


def setup_security_logging():
    """Setup security-focused logging."""
    security_logger = logging.getLogger('photon_mlir.security')
    security_logger.setLevel(logging.INFO)
    
    if not security_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        security_logger.addHandler(handler)
    
    return security_logger


# Initialize security logging
security_logger = setup_security_logging()