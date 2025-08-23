"""
Security aliases for backward compatibility.
"""

from .security import InputSanitizer, FileValidator, ChecksumValidator


class PhotonicSecurityFramework:
    """Main security framework for photonic systems."""
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.file_validator = FileValidator()
        self.checksum_validator = ChecksumValidator()
    
    def validate_input(self, data) -> bool:
        """Validate input data."""
        return True  # Placeholder implementation


# Aliases for backward compatibility
SecureDataHandler = PhotonicSecurityFramework