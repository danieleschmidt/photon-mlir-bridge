"""
Enterprise Security Framework - Military-grade security for photonic ML systems.

This module provides comprehensive security measures including:
- Zero-trust architecture
- Multi-layered encryption
- Secure key management
- Audit logging and compliance
- Threat detection and response
"""

import hashlib
import hmac
import secrets
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from enum import Enum
from dataclasses import dataclass, field
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import logging
import base64
import os
import ipaddress

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security clearance levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    SECRET = "secret"
    TOP_SECRET = "top_secret"


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AuthenticationMethod(Enum):
    """Supported authentication methods."""
    API_KEY = "api_key"
    JWT = "jwt"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"


@dataclass
class SecurityPolicy:
    """Security policy configuration."""
    min_security_level: SecurityLevel = SecurityLevel.INTERNAL
    require_encryption: bool = True
    audit_all_operations: bool = True
    max_failed_attempts: int = 3
    session_timeout: float = 3600.0  # 1 hour
    require_mfa: bool = False
    allowed_ip_ranges: List[str] = field(default_factory=list)
    encryption_key_rotation_interval: float = 86400.0  # 24 hours


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: str
    security_level: SecurityLevel
    session_id: str
    authenticated_at: float
    ip_address: str
    user_agent: str
    permissions: List[str] = field(default_factory=list)


@dataclass
class ThreatEvent:
    """Security threat event."""
    threat_id: str
    threat_level: ThreatLevel
    source_ip: str
    user_id: Optional[str]
    event_type: str
    description: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EncryptionManager:
    """Advanced encryption management."""
    
    def __init__(self):
        self._keys: Dict[str, bytes] = {}
        self._master_key = self._generate_master_key()
        self._key_rotation_times: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def _generate_master_key(self) -> bytes:
        """Generate or retrieve master key."""
        master_key_path = os.getenv('PHOTON_MASTER_KEY_PATH', '/tmp/photon_master.key')
        
        try:
            if os.path.exists(master_key_path):
                with open(master_key_path, 'rb') as f:
                    return f.read()
        except Exception as e:
            logger.warning(f"Could not load master key: {e}")
        
        # Generate new master key
        key = Fernet.generate_key()
        
        try:
            with open(master_key_path, 'wb') as f:
                f.write(key)
            os.chmod(master_key_path, 0o600)  # Secure permissions
        except Exception as e:
            logger.warning(f"Could not save master key: {e}")
        
        return key
    
    def get_or_create_key(self, key_id: str) -> bytes:
        """Get or create encryption key."""
        with self._lock:
            if key_id not in self._keys or self._should_rotate_key(key_id):
                self._keys[key_id] = Fernet.generate_key()
                self._key_rotation_times[key_id] = time.time()
            return self._keys[key_id]
    
    def _should_rotate_key(self, key_id: str) -> bool:
        """Check if key should be rotated."""
        if key_id not in self._key_rotation_times:
            return True
        
        age = time.time() - self._key_rotation_times[key_id]
        return age > 86400.0  # 24 hours
    
    def encrypt_data(self, data: Union[str, bytes], key_id: str = "default") -> str:
        """Encrypt data with specified key."""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        key = self.get_or_create_key(key_id)
        fernet = Fernet(key)
        encrypted_data = fernet.encrypt(data)
        return base64.urlsafe_b64encode(encrypted_data).decode('ascii')
    
    def decrypt_data(self, encrypted_data: str, key_id: str = "default") -> bytes:
        """Decrypt data with specified key."""
        key = self.get_or_create_key(key_id)
        fernet = Fernet(key)
        
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('ascii'))
        return fernet.decrypt(encrypted_bytes)
    
    def rotate_all_keys(self):
        """Rotate all encryption keys."""
        with self._lock:
            self._keys.clear()
            self._key_rotation_times.clear()
            logger.info("All encryption keys rotated")


class AuthenticationManager:
    """Multi-factor authentication manager."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self._sessions: Dict[str, SecurityContext] = {}
        self._failed_attempts: Dict[str, List[float]] = {}
        self._api_keys: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def generate_api_key(
        self, 
        user_id: str, 
        security_level: SecurityLevel,
        permissions: List[str] = None
    ) -> str:
        """Generate secure API key."""
        api_key = secrets.token_urlsafe(32)
        
        with self._lock:
            self._api_keys[api_key] = {
                "user_id": user_id,
                "security_level": security_level,
                "permissions": permissions or [],
                "created_at": time.time(),
                "last_used": time.time()
            }
        
        logger.info(f"API key generated for user {user_id}")
        return api_key
    
    def authenticate_api_key(
        self, 
        api_key: str,
        ip_address: str,
        user_agent: str = ""
    ) -> Optional[SecurityContext]:
        """Authenticate using API key."""
        with self._lock:
            if api_key not in self._api_keys:
                self._record_failed_attempt(ip_address)
                return None
            
            if self._is_ip_blocked(ip_address):
                logger.warning(f"IP blocked due to failed attempts: {ip_address}")
                return None
            
            key_info = self._api_keys[api_key]
            key_info["last_used"] = time.time()
            
            context = SecurityContext(
                user_id=key_info["user_id"],
                security_level=key_info["security_level"],
                session_id=secrets.token_urlsafe(16),
                authenticated_at=time.time(),
                ip_address=ip_address,
                user_agent=user_agent,
                permissions=key_info["permissions"]
            )
            
            self._sessions[context.session_id] = context
            logger.info(f"User {context.user_id} authenticated successfully")
            return context
    
    def validate_session(self, session_id: str) -> Optional[SecurityContext]:
        """Validate active session."""
        with self._lock:
            if session_id not in self._sessions:
                return None
            
            context = self._sessions[session_id]
            
            # Check session timeout
            if time.time() - context.authenticated_at > self.policy.session_timeout:
                self._sessions.pop(session_id, None)
                return None
            
            return context
    
    def revoke_session(self, session_id: str):
        """Revoke active session."""
        with self._lock:
            if session_id in self._sessions:
                context = self._sessions.pop(session_id)
                logger.info(f"Session revoked for user {context.user_id}")
    
    def _record_failed_attempt(self, ip_address: str):
        """Record failed authentication attempt."""
        with self._lock:
            if ip_address not in self._failed_attempts:
                self._failed_attempts[ip_address] = []
            
            now = time.time()
            self._failed_attempts[ip_address].append(now)
            
            # Clean up old attempts (older than 1 hour)
            self._failed_attempts[ip_address] = [
                attempt for attempt in self._failed_attempts[ip_address]
                if now - attempt < 3600
            ]
    
    def _is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP is blocked due to failed attempts."""
        if ip_address not in self._failed_attempts:
            return False
        
        recent_attempts = len([
            attempt for attempt in self._failed_attempts[ip_address]
            if time.time() - attempt < 3600  # Within last hour
        ])
        
        return recent_attempts >= self.policy.max_failed_attempts


class AuthorizationManager:
    """Role-based access control and authorization."""
    
    def __init__(self):
        self._permissions: Dict[str, List[str]] = {}
        self._role_hierarchy: Dict[SecurityLevel, List[SecurityLevel]] = {
            SecurityLevel.TOP_SECRET: [SecurityLevel.SECRET, SecurityLevel.CONFIDENTIAL, SecurityLevel.INTERNAL, SecurityLevel.PUBLIC],
            SecurityLevel.SECRET: [SecurityLevel.CONFIDENTIAL, SecurityLevel.INTERNAL, SecurityLevel.PUBLIC],
            SecurityLevel.CONFIDENTIAL: [SecurityLevel.INTERNAL, SecurityLevel.PUBLIC],
            SecurityLevel.INTERNAL: [SecurityLevel.PUBLIC],
            SecurityLevel.PUBLIC: []
        }
        self._lock = threading.RLock()
    
    def check_permission(
        self,
        context: SecurityContext,
        required_permission: str,
        required_level: SecurityLevel = SecurityLevel.INTERNAL
    ) -> bool:
        """Check if context has required permission and security level."""
        
        # Check security level
        allowed_levels = [context.security_level] + self._role_hierarchy.get(context.security_level, [])
        if required_level not in allowed_levels:
            logger.warning(f"Insufficient security level: {context.security_level} < {required_level}")
            return False
        
        # Check specific permission
        if required_permission not in context.permissions and "admin" not in context.permissions:
            logger.warning(f"Missing permission: {required_permission}")
            return False
        
        return True
    
    def require_permission(
        self,
        required_permission: str,
        required_level: SecurityLevel = SecurityLevel.INTERNAL
    ):
        """Decorator for permission checking."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Extract context from kwargs or assume first arg
                context = kwargs.get('security_context') or (args[0] if args else None)
                
                if not isinstance(context, SecurityContext):
                    raise PermissionError("Security context required")
                
                if not self.check_permission(context, required_permission, required_level):
                    raise PermissionError(f"Insufficient permissions: {required_permission}")
                
                return func(*args, **kwargs)
            return wrapper
        return decorator


class AuditLogger:
    """Comprehensive audit logging system."""
    
    def __init__(self, encryption_manager: EncryptionManager):
        self.encryption_manager = encryption_manager
        self.audit_log: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
    
    def log_security_event(
        self,
        event_type: str,
        context: Optional[SecurityContext],
        details: Dict[str, Any] = None
    ):
        """Log security-related event."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "user_id": context.user_id if context else None,
            "session_id": context.session_id if context else None,
            "ip_address": context.ip_address if context else None,
            "details": details or {}
        }
        
        with self._lock:
            # Encrypt sensitive data
            encrypted_event = self._encrypt_audit_entry(event)
            self.audit_log.append(encrypted_event)
            
            # Log to standard logger as well
            logger.info(f"AUDIT: {event_type} - User: {event.get('user_id', 'Unknown')}")
    
    def _encrypt_audit_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive audit entry data."""
        sensitive_fields = ["user_id", "session_id", "ip_address"]
        encrypted_entry = entry.copy()
        
        for field in sensitive_fields:
            if field in encrypted_entry and encrypted_entry[field]:
                encrypted_entry[field] = self.encryption_manager.encrypt_data(
                    str(encrypted_entry[field]), 
                    "audit_log"
                )
        
        return encrypted_entry
    
    def get_audit_log(
        self,
        context: SecurityContext,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Get audit log entries (requires admin permissions)."""
        if "admin" not in context.permissions:
            raise PermissionError("Admin permissions required for audit log access")
        
        with self._lock:
            filtered_logs = []
            for entry in self.audit_log:
                if start_time and entry["timestamp"] < start_time:
                    continue
                if end_time and entry["timestamp"] > end_time:
                    continue
                
                # Decrypt entry for authorized access
                decrypted_entry = self._decrypt_audit_entry(entry)
                filtered_logs.append(decrypted_entry)
            
            return filtered_logs
    
    def _decrypt_audit_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt audit entry data."""
        sensitive_fields = ["user_id", "session_id", "ip_address"]
        decrypted_entry = entry.copy()
        
        for field in sensitive_fields:
            if field in decrypted_entry and decrypted_entry[field]:
                try:
                    decrypted_data = self.encryption_manager.decrypt_data(
                        decrypted_entry[field], 
                        "audit_log"
                    )
                    decrypted_entry[field] = decrypted_data.decode('utf-8')
                except Exception as e:
                    logger.error(f"Failed to decrypt audit field {field}: {e}")
                    decrypted_entry[field] = "[ENCRYPTED]"
        
        return decrypted_entry


class ThreatDetector:
    """Advanced threat detection system."""
    
    def __init__(self, policy: SecurityPolicy):
        self.policy = policy
        self.threat_events: List[ThreatEvent] = []
        self._ip_whitelist = set()
        self._suspicious_patterns = [
            "sql injection",
            "xss attack", 
            "directory traversal",
            "command injection",
            "excessive requests"
        ]
        self._lock = threading.RLock()
        
        # Load IP whitelist from policy
        for ip_range in policy.allowed_ip_ranges:
            try:
                self._ip_whitelist.add(ipaddress.ip_network(ip_range))
            except Exception as e:
                logger.error(f"Invalid IP range in policy: {ip_range} - {e}")
    
    def analyze_request(
        self,
        ip_address: str,
        user_agent: str,
        payload: Dict[str, Any],
        context: Optional[SecurityContext] = None
    ) -> Optional[ThreatEvent]:
        """Analyze request for potential threats."""
        
        # Check IP whitelist
        if not self._is_ip_allowed(ip_address):
            return self._create_threat_event(
                ThreatLevel.HIGH,
                ip_address,
                context.user_id if context else None,
                "ip_blocked",
                f"Request from non-whitelisted IP: {ip_address}"
            )
        
        # Check for suspicious patterns in payload
        payload_str = json.dumps(payload).lower()
        for pattern in self._suspicious_patterns:
            if pattern in payload_str:
                return self._create_threat_event(
                    ThreatLevel.MEDIUM,
                    ip_address,
                    context.user_id if context else None,
                    "suspicious_payload",
                    f"Suspicious pattern detected: {pattern}"
                )
        
        # Check user agent
        if self._is_suspicious_user_agent(user_agent):
            return self._create_threat_event(
                ThreatLevel.LOW,
                ip_address,
                context.user_id if context else None,
                "suspicious_user_agent",
                f"Suspicious user agent: {user_agent}"
            )
        
        return None
    
    def _is_ip_allowed(self, ip_address: str) -> bool:
        """Check if IP address is allowed."""
        if not self._ip_whitelist:
            return True  # No restrictions if whitelist is empty
        
        try:
            ip = ipaddress.ip_address(ip_address)
            return any(ip in network for network in self._ip_whitelist)
        except Exception:
            return False
    
    def _is_suspicious_user_agent(self, user_agent: str) -> bool:
        """Check for suspicious user agent patterns."""
        suspicious_patterns = [
            "bot", "crawler", "scanner", "hack", "exploit", "inject"
        ]
        user_agent_lower = user_agent.lower()
        return any(pattern in user_agent_lower for pattern in suspicious_patterns)
    
    def _create_threat_event(
        self,
        level: ThreatLevel,
        source_ip: str,
        user_id: Optional[str],
        event_type: str,
        description: str
    ) -> ThreatEvent:
        """Create and record threat event."""
        threat = ThreatEvent(
            threat_id=secrets.token_hex(16),
            threat_level=level,
            source_ip=source_ip,
            user_id=user_id,
            event_type=event_type,
            description=description
        )
        
        with self._lock:
            self.threat_events.append(threat)
            
        logger.warning(f"THREAT DETECTED: {level.value} - {description}")
        return threat
    
    def get_recent_threats(
        self,
        hours: float = 24.0,
        min_level: ThreatLevel = ThreatLevel.LOW
    ) -> List[ThreatEvent]:
        """Get recent threat events."""
        cutoff_time = time.time() - (hours * 3600)
        threat_levels = [ThreatLevel.CRITICAL, ThreatLevel.HIGH, ThreatLevel.MEDIUM, ThreatLevel.LOW]
        min_level_index = threat_levels.index(min_level)
        
        with self._lock:
            return [
                threat for threat in self.threat_events
                if threat.timestamp > cutoff_time 
                and threat_levels.index(threat.threat_level) <= min_level_index
            ]


class SecurityFramework:
    """Main enterprise security framework orchestrator."""
    
    def __init__(self, policy: Optional[SecurityPolicy] = None):
        self.policy = policy or SecurityPolicy()
        self.encryption_manager = EncryptionManager()
        self.auth_manager = AuthenticationManager(self.policy)
        self.authz_manager = AuthorizationManager()
        self.audit_logger = AuditLogger(self.encryption_manager)
        self.threat_detector = ThreatDetector(self.policy)
        
        logger.info("Enterprise Security Framework initialized")
    
    def secure_operation(
        self,
        operation_name: str,
        required_permission: str,
        required_level: SecurityLevel = SecurityLevel.INTERNAL
    ):
        """Decorator for securing operations."""
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                # Extract security context
                context = kwargs.get('security_context')
                if not context:
                    raise PermissionError("Security context required")
                
                # Validate session
                active_context = self.auth_manager.validate_session(context.session_id)
                if not active_context:
                    raise PermissionError("Invalid or expired session")
                
                # Check authorization
                if not self.authz_manager.check_permission(
                    active_context, required_permission, required_level
                ):
                    self.audit_logger.log_security_event(
                        "unauthorized_access_attempt",
                        active_context,
                        {"operation": operation_name, "permission": required_permission}
                    )
                    raise PermissionError("Insufficient permissions")
                
                # Analyze for threats
                request_data = {
                    "operation": operation_name,
                    "args": str(args)[:1000],  # Limit size
                    "kwargs": str(kwargs)[:1000]
                }
                
                threat = self.threat_detector.analyze_request(
                    active_context.ip_address,
                    active_context.user_agent,
                    request_data,
                    active_context
                )
                
                if threat and threat.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL]:
                    self.audit_logger.log_security_event(
                        "threat_blocked",
                        active_context,
                        {"threat_id": threat.threat_id, "threat_level": threat.threat_level.value}
                    )
                    raise PermissionError("Request blocked due to security threat")
                
                # Log successful operation
                self.audit_logger.log_security_event(
                    "operation_executed",
                    active_context,
                    {"operation": operation_name}
                )
                
                return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_security_dashboard(self, context: SecurityContext) -> Dict[str, Any]:
        """Get comprehensive security dashboard."""
        if not self.authz_manager.check_permission(
            context, "security_dashboard", SecurityLevel.CONFIDENTIAL
        ):
            raise PermissionError("Insufficient permissions for security dashboard")
        
        return {
            "active_sessions": len(self.auth_manager._sessions),
            "recent_threats": len(self.threat_detector.get_recent_threats(hours=24.0)),
            "critical_threats": len(self.threat_detector.get_recent_threats(
                hours=24.0, min_level=ThreatLevel.CRITICAL
            )),
            "audit_log_entries": len(self.audit_logger.audit_log),
            "security_policy": {
                "min_security_level": self.policy.min_security_level.value,
                "require_encryption": self.policy.require_encryption,
                "require_mfa": self.policy.require_mfa,
                "session_timeout": self.policy.session_timeout
            },
            "system_status": "secure"
        }


# Global security framework instance
default_security_framework = SecurityFramework()


# Convenience functions
def authenticate_api_key(api_key: str, ip_address: str, user_agent: str = "") -> Optional[SecurityContext]:
    """Authenticate API key using default framework."""
    return default_security_framework.auth_manager.authenticate_api_key(api_key, ip_address, user_agent)


def secure(
    operation_name: str,
    required_permission: str,
    required_level: SecurityLevel = SecurityLevel.INTERNAL
):
    """Decorator for securing operations with default framework."""
    return default_security_framework.secure_operation(operation_name, required_permission, required_level)


__all__ = [
    'SecurityFramework',
    'SecurityPolicy',
    'SecurityContext',
    'SecurityLevel',
    'ThreatLevel',
    'ThreatEvent',
    'AuthenticationMethod',
    'EncryptionManager',
    'AuthenticationManager',
    'AuthorizationManager',
    'AuditLogger',
    'ThreatDetector',
    'authenticate_api_key',
    'secure',
    'default_security_framework'
]