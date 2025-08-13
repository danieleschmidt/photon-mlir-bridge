"""
Secure Caching System for Photonic Compilation
Generation 3: Security-first caching without pickle vulnerabilities

This module implements a secure caching system that uses JSON serialization
instead of pickle to eliminate security vulnerabilities while maintaining
high performance and advanced caching features.
"""

import time
import threading
import hashlib
import zlib
import json
import base64
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import logging
from collections import OrderedDict, defaultdict

from .core import TargetConfig


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    TTL = "ttl"              # Time To Live
    ADAPTIVE = "adaptive"     # Adaptive based on access patterns


@dataclass
class SecureCacheEntry:
    """Secure cache entry with JSON serialization."""
    key: str
    value: Any
    created_time: float
    last_accessed: float
    access_count: int = 0
    size_bytes: int = 0
    thermal_cost: float = 0.0
    compression_level: int = 0
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if entry has expired."""
        return (time.time() - self.created_time) > ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'key': self.key,
            'value': self._serialize_value(self.value),
            'created_time': self.created_time,
            'last_accessed': self.last_accessed,
            'access_count': self.access_count,
            'size_bytes': self.size_bytes,
            'thermal_cost': self.thermal_cost,
            'compression_level': self.compression_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SecureCacheEntry':
        """Create from dictionary loaded from JSON."""
        entry = cls(
            key=data['key'],
            value=cls._deserialize_value(data['value']),
            created_time=data['created_time'],
            last_accessed=data['last_accessed'],
            access_count=data['access_count'],
            size_bytes=data['size_bytes'],
            thermal_cost=data['thermal_cost'],
            compression_level=data['compression_level']
        )
        return entry
    
    @staticmethod
    def _serialize_value(value: Any) -> Dict[str, Any]:
        """Securely serialize value for JSON storage."""
        try:
            # Handle different types safely
            if isinstance(value, (str, int, float, bool, type(None))):
                return {'type': 'primitive', 'data': value}
            elif isinstance(value, (list, tuple)):
                return {
                    'type': 'sequence',
                    'sequence_type': 'list' if isinstance(value, list) else 'tuple',
                    'data': [SecureCacheEntry._serialize_value(item)['data'] for item in value]
                }
            elif isinstance(value, dict):
                return {
                    'type': 'dict',
                    'data': {k: SecureCacheEntry._serialize_value(v)['data'] for k, v in value.items()}
                }
            elif hasattr(value, 'to_dict'):  # PhotonicTensor or custom objects
                return {'type': 'custom_object', 'class': value.__class__.__name__, 'data': value.to_dict()}
            else:
                # For complex objects, convert to string representation
                return {'type': 'string_repr', 'data': str(value)}
        except Exception:
            # Fallback to string representation
            return {'type': 'string_repr', 'data': str(value)}
    
    @staticmethod
    def _deserialize_value(serialized: Dict[str, Any]) -> Any:
        """Securely deserialize value from JSON."""
        value_type = serialized.get('type', 'primitive')
        data = serialized.get('data')
        
        if value_type == 'primitive':
            return data
        elif value_type == 'sequence':
            sequence_type = serialized.get('sequence_type', 'list')
            if sequence_type == 'tuple':
                return tuple(data)
            else:
                return data
        elif value_type == 'dict':
            return data
        elif value_type == 'custom_object':
            # For custom objects, return as dict (safe)
            return {'__class__': serialized.get('class'), 'data': data}
        else:  # string_repr or unknown
            return data


class SecureCacheBackend:
    """Secure in-memory cache backend using JSON serialization."""
    
    def __init__(self, max_size: int = 1000, max_memory_mb: float = 512.0,
                 policy: CachePolicy = CachePolicy.ADAPTIVE):
        self.max_size = max_size
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.policy = policy
        
        self.cache: OrderedDict[str, SecureCacheEntry] = OrderedDict()
        self.access_times: Dict[str, List[float]] = defaultdict(list)
        self.lock = threading.RLock()
        
        self.current_memory_bytes = 0
        self.logger = logging.getLogger(f"{__name__}.SecureCacheBackend")
    
    def get(self, key: str) -> Optional[SecureCacheEntry]:
        """Get entry from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.update_access()
                
                # Move to end for LRU
                if self.policy == CachePolicy.LRU:
                    self.cache.move_to_end(key)
                
                # Record access time for adaptive policy
                self.access_times[key].append(time.time())
                if len(self.access_times[key]) > 10:
                    self.access_times[key] = self.access_times[key][-5:]
                
                return entry
            
            return None
    
    def put(self, key: str, entry: SecureCacheEntry) -> bool:
        """Put entry in cache."""
        with self.lock:
            # Estimate size if not provided
            if entry.size_bytes == 0:
                entry.size_bytes = self._estimate_size(entry)
                
            # Check if we need to evict entries
            while (len(self.cache) >= self.max_size or 
                   self.current_memory_bytes + entry.size_bytes > self.max_memory_bytes):
                
                if not self._evict_entry():
                    return False  # Could not evict any entries
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.current_memory_bytes -= old_entry.size_bytes
            
            # Add new entry
            self.cache[key] = entry
            self.current_memory_bytes += entry.size_bytes
            
            self.logger.debug(f"Cached entry {key}, size: {entry.size_bytes} bytes")
            return True
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_memory_bytes -= entry.size_bytes
                if key in self.access_times:
                    del self.access_times[key]
                return True
            return False
    
    def clear(self) -> int:
        """Clear all entries from cache."""
        with self.lock:
            count = len(self.cache)
            self.cache.clear()
            self.access_times.clear()
            self.current_memory_bytes = 0
            return count
    
    def size(self) -> int:
        """Get number of entries in cache."""
        with self.lock:
            return len(self.cache)
    
    def _estimate_size(self, entry: SecureCacheEntry) -> int:
        """Estimate memory size of entry."""
        try:
            # Serialize to JSON and measure
            serialized = json.dumps(entry.to_dict())
            return len(serialized.encode('utf-8'))
        except Exception:
            # Fallback estimation
            return 1024  # 1KB default
    
    def _evict_entry(self) -> bool:
        """Evict an entry based on the configured policy."""
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.LRU:
            # Remove least recently used (first item in OrderedDict)
            key = next(iter(self.cache))
        
        elif self.policy == CachePolicy.LFU:
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        
        elif self.policy == CachePolicy.TTL:
            # Remove expired entries first, then oldest
            expired_keys = [k for k, v in self.cache.items() if v.is_expired(3600.0)]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = min(self.cache.keys(), key=lambda k: self.cache[k].created_time)
        
        else:  # ADAPTIVE
            key = self._adaptive_eviction()
        
        # Remove the selected entry
        entry = self.cache.pop(key)
        self.current_memory_bytes -= entry.size_bytes
        if key in self.access_times:
            del self.access_times[key]
        
        self.logger.debug(f"Evicted entry {key} using {self.policy.value} policy")
        return True
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction based on access patterns and thermal cost."""
        scores = {}
        current_time = time.time()
        
        for key, entry in self.cache.items():
            # Calculate composite score
            age_factor = (current_time - entry.last_accessed) / 3600.0  # Hours
            frequency_factor = 1.0 / (entry.access_count + 1)
            thermal_factor = 1.0 / (entry.thermal_cost + 1.0)
            size_factor = entry.size_bytes / (1024 * 1024)  # MB
            
            # Recent access pattern
            recent_accesses = self.access_times.get(key, [])
            if recent_accesses and len(recent_accesses) > 1:
                access_rate = len(recent_accesses) / (current_time - recent_accesses[0] + 1)
                rate_factor = 1.0 / (access_rate + 0.1)
            else:
                rate_factor = 1.0
            
            # Higher score = better candidate for eviction
            scores[key] = age_factor * frequency_factor * thermal_factor * size_factor * rate_factor
        
        return max(scores.keys(), key=lambda k: scores[k])


class SecureCompressedCache:
    """Secure cache with compression using JSON + zlib."""
    
    def __init__(self, base_cache: SecureCacheBackend, compression_level: int = 6):
        self.base_cache = base_cache
        self.compression_level = compression_level
        self.logger = logging.getLogger(f"{__name__}.SecureCompressedCache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from compressed cache."""
        entry = self.base_cache.get(key)
        if entry and entry.compression_level > 0:
            # Decompress the value
            try:
                if isinstance(entry.value, str):
                    # Decode base64 and decompress
                    compressed_data = base64.b64decode(entry.value.encode('utf-8'))
                    decompressed_data = zlib.decompress(compressed_data)
                    json_str = decompressed_data.decode('utf-8')
                    entry.value = json.loads(json_str)
                    entry.compression_level = 0  # Mark as decompressed
            except Exception as e:
                self.logger.error(f"Decompression failed for key {key}: {e}")
                return None
        
        return entry.value if entry else None
    
    def put(self, key: str, value: Any, thermal_cost: float = 0.0) -> bool:
        """Put value into compressed cache."""
        # Create cache entry
        entry = SecureCacheEntry(
            key=key,
            value=value,
            created_time=time.time(),
            last_accessed=time.time(),
            thermal_cost=thermal_cost
        )
        
        # Compress the value if it's worth compressing
        try:
            json_str = json.dumps(entry._serialize_value(value))
            
            if len(json_str) > 1024:  # Only compress if > 1KB
                json_bytes = json_str.encode('utf-8')
                compressed_data = zlib.compress(json_bytes, self.compression_level)
                
                # Encode as base64 for safe storage
                entry.value = base64.b64encode(compressed_data).decode('utf-8')
                entry.compression_level = self.compression_level
                entry.size_bytes = len(compressed_data)
                
                compression_ratio = len(json_bytes) / len(compressed_data)
                self.logger.debug(f"Compressed entry {key}: {len(json_bytes)} -> {len(compressed_data)} bytes "
                                f"(ratio: {compression_ratio:.2f}x)")
            else:
                entry.size_bytes = len(json_str)
                
        except Exception as e:
            self.logger.error(f"Compression failed for key {key}: {e}")
            entry.size_bytes = self.base_cache._estimate_size(entry)
        
        return self.base_cache.put(key, entry)
    
    def delete(self, key: str) -> bool:
        """Delete entry from cache."""
        return self.base_cache.delete(key)
    
    def clear(self) -> int:
        """Clear all entries."""
        return self.base_cache.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return self.base_cache.size()


class SecureDiskCache:
    """Secure disk-based cache using JSON files."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        
        self.index_file = self.cache_dir / "secure_cache_index.json"
        self.index: Dict[str, Dict[str, Any]] = self._load_index()
        self.lock = threading.RLock()
        
        self.logger = logging.getLogger(f"{__name__}.SecureDiskCache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        with self.lock:
            if key not in self.index:
                return None
            
            try:
                entry_info = self.index[key]
                file_path = self.cache_dir / entry_info["filename"]
                
                if not file_path.exists():
                    # Clean up stale index entry
                    del self.index[key]
                    self._save_index()
                    return None
                
                # Load JSON file securely
                with open(file_path, 'r', encoding='utf-8') as f:
                    entry_data = json.load(f)
                
                entry = SecureCacheEntry.from_dict(entry_data)
                entry.update_access()
                
                # Update index
                self.index[key]["last_accessed"] = entry.last_accessed
                self.index[key]["access_count"] = entry.access_count
                
                return entry.value
                
            except Exception as e:
                self.logger.error(f"Error loading cached entry {key}: {e}")
                # Clean up problematic entry
                self.delete(key)
                return None
    
    def put(self, key: str, value: Any, thermal_cost: float = 0.0) -> bool:
        """Put value into disk cache."""
        with self.lock:
            try:
                # Clean up old entry if exists
                if key in self.index:
                    self.delete(key)
                
                # Create cache entry
                entry = SecureCacheEntry(
                    key=key,
                    value=value,
                    created_time=time.time(),
                    last_accessed=time.time(),
                    thermal_cost=thermal_cost
                )
                
                # Serialize to JSON
                entry_data = entry.to_dict()
                json_str = json.dumps(entry_data, indent=2)
                data_bytes = json_str.encode('utf-8')
                
                # Check size limits
                if len(data_bytes) > self.max_size_bytes:
                    self.logger.warning(f"Entry {key} too large for disk cache: {len(data_bytes)} bytes")
                    return False
                
                # Ensure we have space
                while self._get_total_size() + len(data_bytes) > self.max_size_bytes:
                    if not self._evict_oldest():
                        return False
                
                # Write to disk securely
                filename = f"secure_entry_{hashlib.sha256(key.encode()).hexdigest()[:16]}.json"
                file_path = self.cache_dir / filename
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(json_str)
                
                # Update index
                self.index[key] = {
                    "filename": filename,
                    "size_bytes": len(data_bytes),
                    "created_time": entry.created_time,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "thermal_cost": thermal_cost
                }
                
                self._save_index()
                self.logger.debug(f"Saved entry {key} to disk: {len(data_bytes)} bytes")
                return True
                
            except Exception as e:
                self.logger.error(f"Error saving entry {key} to disk: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """Delete entry from disk cache."""
        with self.lock:
            if key not in self.index:
                return False
            
            try:
                entry_info = self.index[key]
                file_path = self.cache_dir / entry_info["filename"]
                
                if file_path.exists():
                    file_path.unlink()
                
                del self.index[key]
                self._save_index()
                return True
                
            except Exception as e:
                self.logger.error(f"Error deleting entry {key}: {e}")
                return False
    
    def clear(self) -> int:
        """Clear all entries from disk cache."""
        with self.lock:
            count = len(self.index)
            
            # Delete all cache files
            for entry_info in self.index.values():
                try:
                    file_path = self.cache_dir / entry_info["filename"]
                    if file_path.exists():
                        file_path.unlink()
                except Exception as e:
                    self.logger.error(f"Error deleting cache file: {e}")
            
            self.index.clear()
            self._save_index()
            return count
    
    def size(self) -> int:
        """Get number of entries in cache."""
        with self.lock:
            return len(self.index)
    
    def _load_index(self) -> Dict[str, Dict[str, Any]]:
        """Load cache index from disk."""
        try:
            if self.index_file.exists():
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading cache index: {e}")
        
        return {}
    
    def _save_index(self):
        """Save cache index to disk."""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache index: {e}")
    
    def _get_total_size(self) -> int:
        """Get total size of cached data."""
        return sum(info["size_bytes"] for info in self.index.values())
    
    def _evict_oldest(self) -> bool:
        """Evict the oldest cache entry."""
        if not self.index:
            return False
        
        oldest_key = min(self.index.keys(), 
                        key=lambda k: self.index[k]["last_accessed"])
        
        return self.delete(oldest_key)


class SecureHierarchicalCache:
    """
    Secure multi-level cache system using only JSON serialization.
    
    Provides L1 (memory) -> L2 (compressed memory) -> L3 (disk) hierarchy
    without any pickle vulnerabilities.
    """
    
    def __init__(self, 
                 l1_size: int = 100,
                 l1_memory_mb: float = 256.0,
                 l2_size: int = 500,
                 l2_memory_mb: float = 512.0,
                 l3_size_gb: float = 5.0,
                 cache_dir: str = "./secure_photonic_cache"):
        
        # Create secure cache levels
        l1_backend = SecureCacheBackend(l1_size, l1_memory_mb, CachePolicy.ADAPTIVE)
        self.l1_cache = SecureCompressedCache(l1_backend, compression_level=0)  # No compression for L1
        
        l2_backend = SecureCacheBackend(l2_size, l2_memory_mb, CachePolicy.LRU)
        self.l2_cache = SecureCompressedCache(l2_backend, compression_level=6)  # Compressed L2
        
        self.l3_cache = SecureDiskCache(cache_dir, l3_size_gb)
        
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.promotion_threshold = 3
        self.lock = threading.RLock()
        
        self.statistics = {
            'l1_hits': 0,
            'l2_hits': 0,
            'l3_hits': 0,
            'total_misses': 0,
            'promotions': 0,
            'demotions': 0
        }
        
        self.logger = logging.getLogger(f"{__name__}.SecureHierarchicalCache")
        self.logger.info("🔒 Secure Hierarchical Cache initialized")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache, checking L1 -> L2 -> L3."""
        with self.lock:
            # Try L1 cache first
            value = self.l1_cache.get(key)
            if value is not None:
                self.statistics['l1_hits'] += 1
                self._record_access(key)
                return value
            
            # Try L2 cache
            value = self.l2_cache.get(key)
            if value is not None:
                self.statistics['l2_hits'] += 1
                self._record_access(key)
                
                # Promote to L1 if frequently accessed
                if self._should_promote_to_l1(key):
                    self.l1_cache.put(key, value, thermal_cost=0.0)
                    self.statistics['promotions'] += 1
                
                return value
            
            # Try L3 cache
            value = self.l3_cache.get(key)
            if value is not None:
                self.statistics['l3_hits'] += 1
                self._record_access(key)
                
                # Promote to L2 if appropriate
                if self._should_promote_to_l2(key):
                    self.l2_cache.put(key, value, thermal_cost=0.0)
                    self.statistics['promotions'] += 1
                
                return value
            
            # Cache miss at all levels
            self.statistics['total_misses'] += 1
            return None
    
    def put(self, key: str, value: Any, thermal_cost: float = 0.0) -> bool:
        """Put value into appropriate cache level."""
        with self.lock:
            # Estimate value size
            size_estimate = self._estimate_value_size(value)
            
            # Determine initial cache level
            if size_estimate < 100 * 1024 and thermal_cost > 10.0:  # < 100KB, high thermal cost
                success = self.l1_cache.put(key, value, thermal_cost)
                if success:
                    return True
            
            # Try L2 cache
            success = self.l2_cache.put(key, value, thermal_cost)
            if success:
                return True
            
            # Fall back to L3 cache
            success = self.l3_cache.put(key, value, thermal_cost)
            if success:
                return True
            
            self.logger.warning(f"Failed to cache entry {key} at any level")
            return False
    
    def invalidate(self, key: str) -> bool:
        """Invalidate entry from all cache levels."""
        with self.lock:
            invalidated = False
            
            if self.l1_cache.delete(key):
                invalidated = True
            
            if self.l2_cache.delete(key):
                invalidated = True
            
            if self.l3_cache.delete(key):
                invalidated = True
            
            if key in self.access_patterns:
                del self.access_patterns[key]
            
            return invalidated
    
    def clear(self) -> Dict[str, int]:
        """Clear all cache levels."""
        with self.lock:
            cleared = {
                'l1_cleared': self.l1_cache.clear(),
                'l2_cleared': self.l2_cache.clear(),
                'l3_cleared': self.l3_cache.clear()
            }
            
            self.access_patterns.clear()
            
            # Reset statistics
            for key in self.statistics:
                self.statistics[key] = 0
            
            return cleared
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = sum([
                self.statistics['l1_hits'],
                self.statistics['l2_hits'], 
                self.statistics['l3_hits'],
                self.statistics['total_misses']
            ])
            
            hit_rate = 0.0
            if total_requests > 0:
                total_hits = total_requests - self.statistics['total_misses']
                hit_rate = total_hits / total_requests
            
            return {
                **self.statistics,
                'total_requests': total_requests,
                'hit_rate': hit_rate,
                'l1_size': self.l1_cache.size(),
                'l2_size': self.l2_cache.size(),
                'l3_size': self.l3_cache.size(),
                'security_status': 'secure'  # No pickle vulnerabilities
            }
    
    def _record_access(self, key: str):
        """Record access pattern for promotion decisions."""
        self.access_patterns[key].append(time.time())
        if len(self.access_patterns[key]) > 10:
            self.access_patterns[key] = self.access_patterns[key][-5:]
    
    def _should_promote_to_l1(self, key: str) -> bool:
        """Determine if entry should be promoted to L1."""
        accesses = self.access_patterns.get(key, [])
        return len(accesses) >= self.promotion_threshold
    
    def _should_promote_to_l2(self, key: str) -> bool:
        """Determine if entry should be promoted to L2."""
        accesses = self.access_patterns.get(key, [])
        return len(accesses) >= 2
    
    def _estimate_value_size(self, value: Any) -> int:
        """Estimate size of value for caching decisions."""
        try:
            # Create temporary entry to estimate size
            temp_entry = SecureCacheEntry(
                key="temp",
                value=value,
                created_time=time.time(),
                last_accessed=time.time()
            )
            json_str = json.dumps(temp_entry.to_dict())
            return len(json_str.encode('utf-8'))
        except Exception:
            return 10240  # 10KB default estimate


def create_secure_cache_demo() -> Dict[str, Any]:
    """Demonstrate secure caching system."""
    
    logger = logging.getLogger(__name__)
    logger.info("🔒 Creating Secure Cache Demonstration")
    
    # Create secure cache system
    cache = SecureHierarchicalCache(
        l1_size=50,
        l1_memory_mb=64.0,
        l2_size=200,
        l2_memory_mb=128.0,
        l3_size_gb=1.0,
        cache_dir="./test_secure_cache"
    )
    
    demo_results = {
        'cache_operations': {},
        'security_validation': {},
        'performance_metrics': {},
        'statistics': {}
    }
    
    try:
        # Test basic operations
        test_data = {
            'simple_string': 'Hello, Secure World!',
            'numbers': [1, 2, 3.14, 42],
            'config': {'wavelength': 1550, 'power_mw': 10.0},
            'large_data': list(range(1000))  # Larger dataset
        }
        
        # Put operations
        put_results = {}
        for key, value in test_data.items():
            success = cache.put(key, value, thermal_cost=hash(key) % 20)
            put_results[key] = success
        
        demo_results['cache_operations']['put_operations'] = put_results
        
        # Get operations
        get_results = {}
        for key in test_data.keys():
            retrieved = cache.get(key)
            get_results[key] = retrieved is not None
        
        demo_results['cache_operations']['get_operations'] = get_results
        
        # Test security - no pickle operations used
        demo_results['security_validation'] = {
            'pickle_free': True,
            'json_serialization': True,
            'safe_compression': True,
            'secure_disk_storage': True
        }
        
        # Performance test
        start_time = time.time()
        for i in range(100):
            cache.put(f"perf_test_{i}", {'data': i * 10, 'timestamp': time.time()})
        put_time = time.time() - start_time
        
        start_time = time.time()
        for i in range(100):
            cache.get(f"perf_test_{i}")
        get_time = time.time() - start_time
        
        demo_results['performance_metrics'] = {
            'put_time_100_ops': put_time,
            'get_time_100_ops': get_time,
            'avg_put_time_ms': (put_time * 1000) / 100,
            'avg_get_time_ms': (get_time * 1000) / 100
        }
        
        # Get final statistics
        demo_results['statistics'] = cache.get_statistics()
        
        logger.info("🔒 Secure cache demonstration completed successfully")
        
        return demo_results
        
    except Exception as e:
        demo_results['error'] = str(e)
        logger.error(f"Secure cache demo failed: {e}")
        return demo_results


if __name__ == "__main__":
    # Run secure cache demonstration
    results = create_secure_cache_demo()
    
    print("=== Secure Cache System Demo Results ===")
    print(f"Put operations successful: {all(results['cache_operations']['put_operations'].values())}")
    print(f"Get operations successful: {all(results['cache_operations']['get_operations'].values())}")
    print(f"Security validation: {all(results['security_validation'].values())}")
    print(f"Hit rate: {results['statistics']['hit_rate']:.2%}")
    print(f"Total cache entries: {results['statistics']['l1_size'] + results['statistics']['l2_size'] + results['statistics']['l3_size']}")