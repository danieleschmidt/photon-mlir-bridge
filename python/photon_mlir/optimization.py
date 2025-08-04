"""
Performance optimization and caching system for photonic compiler.
"""

import os
import time
import pickle
import hashlib
import threading
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable, TypeVar, Generic
from dataclasses import dataclass, field
from collections import OrderedDict
import weakref
import concurrent.futures
import functools
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    disk_usage_bytes: int = 0
    last_cleanup: float = field(default_factory=time.time)
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def reset(self):
        """Reset statistics."""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_usage_bytes = 0
        self.disk_usage_bytes = 0
        self.last_cleanup = time.time()


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    timestamp: float
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    size_bytes: int = 0
    
    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache(Generic[K, V]):
    """Thread-safe LRU cache with size limits and TTL."""
    
    def __init__(self, 
                 max_size: int = 1000,
                 ttl_seconds: Optional[float] = None,
                 max_memory_mb: Optional[float] = None):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.max_memory_bytes = max_memory_mb * 1024 * 1024 if max_memory_mb else None
        
        self._cache: OrderedDict[K, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self._stats.misses += 1
                return default
            
            entry = self._cache[key]
            
            # Check TTL
            if self.ttl_seconds and time.time() - entry.timestamp > self.ttl_seconds:
                del self._cache[key]
                self._stats.misses += 1
                self._stats.evictions += 1
                return default
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()
            
            self._stats.hits += 1
            return entry.value
    
    def put(self, key: K, value: V, size_hint: Optional[int] = None) -> None:
        """Put value into cache."""
        with self._lock:
            # Estimate size if not provided
            if size_hint is None:
                try:
                    size_hint = len(pickle.dumps(value))
                except:
                    size_hint = 1024  # Default size
            
            # Create cache entry
            entry = CacheEntry(
                value=value,
                timestamp=time.time(),
                size_bytes=size_hint
            )
            
            # Remove existing entry if present
            if key in self._cache:
                old_entry = self._cache[key]
                self._stats.memory_usage_bytes -= old_entry.size_bytes
            
            # Add new entry
            self._cache[key] = entry
            self._cache.move_to_end(key)  # Mark as most recently used
            self._stats.memory_usage_bytes += size_hint
            
            # Evict if necessary
            self._evict_if_needed()
    
    def _evict_if_needed(self):
        """Evict entries if cache limits exceeded."""
        # Size-based eviction
        while len(self._cache) > self.max_size:
            oldest_key = next(iter(self._cache))
            oldest_entry = self._cache.pop(oldest_key)
            self._stats.memory_usage_bytes -= oldest_entry.size_bytes
            self._stats.evictions += 1
        
        # Memory-based eviction
        if self.max_memory_bytes:
            while self._stats.memory_usage_bytes > self.max_memory_bytes and self._cache:
                oldest_key = next(iter(self._cache))
                oldest_entry = self._cache.pop(oldest_key)
                self._stats.memory_usage_bytes -= oldest_entry.size_bytes
                self._stats.evictions += 1
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats.memory_usage_bytes = 0
    
    def size(self) -> int:
        """Get number of cached entries."""
        return len(self._cache)
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return self._stats


class PersistentCache:
    """Persistent disk-based cache for compiled models."""
    
    def __init__(self, cache_dir: str, max_size_gb: float = 10.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024
        
        self._stats = CacheStats()
        self._lock = threading.Lock()
        
        # Initialize disk usage
        self._update_disk_usage()
    
    def _compute_key_hash(self, key: str) -> str:
        """Compute hash for cache key."""
        return hashlib.sha256(key.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key."""
        key_hash = self._compute_key_hash(key)
        return self.cache_dir / f"{key_hash}.pkl"
    
    def _get_metadata_path(self, key: str) -> Path:
        """Get metadata file path for key."""
        key_hash = self._compute_key_hash(key)
        return self.cache_dir / f"{key_hash}.meta"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from persistent cache."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            if not cache_path.exists() or not meta_path.exists():
                self._stats.misses += 1
                return None
            
            try:
                # Load metadata
                with open(meta_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                # Check TTL if specified
                if 'ttl' in metadata and metadata['ttl']:
                    if time.time() - metadata['timestamp'] > metadata['ttl']:
                        self._remove_entry(key)
                        self._stats.misses += 1
                        self._stats.evictions += 1
                        return None
                
                # Load value
                with open(cache_path, 'rb') as f:
                    value = pickle.load(f)
                
                # Update access metadata
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                metadata['last_accessed'] = time.time()
                
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                self._stats.hits += 1
                return value
                
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
                self._remove_entry(key)
                self._stats.misses += 1
                return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Put value into persistent cache."""
        with self._lock:
            cache_path = self._get_cache_path(key)
            meta_path = self._get_metadata_path(key)
            
            try:
                # Save value
                with open(cache_path, 'wb') as f:
                    pickle.dump(value, f)
                
                # Save metadata
                metadata = {
                    'timestamp': time.time(),
                    'ttl': ttl,
                    'access_count': 1,
                    'last_accessed': time.time(),
                    'key': key
                }
                
                with open(meta_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Update disk usage
                self._update_disk_usage()
                
                # Evict if necessary
                self._evict_if_needed()
                
            except Exception as e:
                logger.error(f"Failed to save to cache: {e}")
                # Clean up partial files
                cache_path.unlink(missing_ok=True)
                meta_path.unlink(missing_ok=True)
    
    def _remove_entry(self, key: str):
        """Remove cache entry."""
        cache_path = self._get_cache_path(key)
        meta_path = self._get_metadata_path(key)
        
        cache_path.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    
    def _update_disk_usage(self):
        """Update disk usage statistics."""
        total_size = 0
        for file_path in self.cache_dir.glob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        self._stats.disk_usage_bytes = total_size
    
    def _evict_if_needed(self):
        """Evict entries if disk usage exceeds limit."""
        if self._stats.disk_usage_bytes <= self.max_size_bytes:
            return
        
        # Get all cache entries with metadata
        entries = []
        for meta_file in self.cache_dir.glob('*.meta'):
            try:
                with open(meta_file, 'rb') as f:
                    metadata = pickle.load(f)
                    metadata['meta_file'] = meta_file
                    entries.append(metadata)
            except:
                continue
        
        # Sort by last accessed time (oldest first)
        entries.sort(key=lambda x: x.get('last_accessed', 0))
        
        # Remove oldest entries until under limit
        for metadata in entries:
            if self._stats.disk_usage_bytes <= self.max_size_bytes:
                break
            
            key = metadata['key']
            self._remove_entry(key)
            self._stats.evictions += 1
            
            # Update disk usage
            self._update_disk_usage()
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            for file_path in self.cache_dir.glob('*'):
                if file_path.is_file():
                    file_path.unlink()
            
            self._stats.disk_usage_bytes = 0
    
    def stats(self) -> CacheStats:
        """Get cache statistics."""
        self._update_disk_usage()
        return self._stats


class CompilationCache:
    """High-level compilation cache combining memory and disk caching."""
    
    def __init__(self, 
                 cache_dir: Optional[str] = None,
                 memory_cache_size: int = 100,
                 memory_cache_mb: float = 500.0,
                 disk_cache_gb: float = 10.0,
                 ttl_hours: float = 24.0):
        
        # Memory cache for fast access
        self.memory_cache = LRUCache[str, Any](
            max_size=memory_cache_size,
            max_memory_mb=memory_cache_mb,
            ttl_seconds=ttl_hours * 3600
        )
        
        # Persistent cache for compiled models
        if cache_dir:
            self.disk_cache: Optional[PersistentCache] = PersistentCache(
                cache_dir, disk_cache_gb
            )
        else:
            self.disk_cache = None
        
        self._lock = threading.Lock()
    
    def _compute_cache_key(self, 
                          model_path: str,
                          target_config: Dict[str, Any],
                          optimization_level: int = 2) -> str:
        """Compute cache key for compilation."""
        # Include file modification time and size
        try:
            stat = os.stat(model_path)
            file_info = f"{stat.st_mtime}:{stat.st_size}"
        except OSError:
            file_info = "unknown"
        
        # Create deterministic key
        key_data = {
            'model_path': model_path,
            'file_info': file_info,
            'target_config': sorted(target_config.items()),
            'optimization_level': optimization_level,
            'compiler_version': '0.1.0'  # Should be actual version
        }
        
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_compiled_model(self,
                          model_path: str,
                          target_config: Dict[str, Any],
                          optimization_level: int = 2) -> Optional[Any]:
        """Get compiled model from cache."""
        cache_key = self._compute_cache_key(model_path, target_config, optimization_level)
        
        # Try memory cache first
        result = self.memory_cache.get(cache_key)
        if result is not None:
            logger.debug(f"Memory cache hit for model: {model_path}")
            return result
        
        # Try disk cache
        if self.disk_cache:
            result = self.disk_cache.get(cache_key)
            if result is not None:
                logger.debug(f"Disk cache hit for model: {model_path}")
                # Promote to memory cache
                self.memory_cache.put(cache_key, result)
                return result
        
        logger.debug(f"Cache miss for model: {model_path}")
        return None
    
    def put_compiled_model(self,
                          model_path: str,
                          target_config: Dict[str, Any],
                          compiled_model: Any,
                          optimization_level: int = 2) -> None:
        """Put compiled model into cache."""
        cache_key = self._compute_cache_key(model_path, target_config, optimization_level)
        
        # Add to memory cache
        self.memory_cache.put(cache_key, compiled_model)
        
        # Add to disk cache
        if self.disk_cache:
            self.disk_cache.put(cache_key, compiled_model, ttl=24*3600)  # 24 hours
        
        logger.debug(f"Cached compiled model: {model_path}")
    
    def invalidate_model(self, model_path: str) -> None:
        """Invalidate all cache entries for a model."""
        # This is a simplified implementation
        # In practice, we'd need to track which keys correspond to which models
        logger.info(f"Cache invalidation requested for: {model_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        memory_stats = self.memory_cache.stats()
        disk_stats = self.disk_cache.stats() if self.disk_cache else CacheStats()
        
        return {
            'memory': {
                'hits': memory_stats.hits,
                'misses': memory_stats.misses,
                'hit_rate': memory_stats.hit_rate,
                'size': self.memory_cache.size(),
                'memory_usage_mb': memory_stats.memory_usage_bytes / (1024*1024)
            },
            'disk': {
                'hits': disk_stats.hits,
                'misses': disk_stats.misses,
                'hit_rate': disk_stats.hit_rate,
                'disk_usage_gb': disk_stats.disk_usage_bytes / (1024*1024*1024)
            }
        }


class OptimizationProfiler:
    """Performance profiler for optimization insights."""
    
    def __init__(self):
        self.profiles: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def profile_function(self, func_name: str):
        """Decorator to profile function execution time."""
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @functools.wraps(func)
            def wrapper(*args, **kwargs) -> T:
                start_time = time.perf_counter()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.perf_counter() - start_time
                    self.record_timing(func_name, duration)
            return wrapper
        return decorator
    
    def record_timing(self, operation: str, duration: float):
        """Record timing for an operation."""
        with self._lock:
            if operation not in self.profiles:
                self.profiles[operation] = []
            self.profiles[operation].append(duration)
            
            # Keep only recent timings (last 1000)
            if len(self.profiles[operation]) > 1000:
                self.profiles[operation] = self.profiles[operation][-1000:]
    
    def get_stats(self, operation: str) -> Optional[Dict[str, float]]:
        """Get statistics for an operation."""
        with self._lock:
            if operation not in self.profiles or not self.profiles[operation]:
                return None
            
            timings = self.profiles[operation]
            return {
                'count': len(timings),
                'mean': sum(timings) / len(timings),
                'min': min(timings),
                'max': max(timings),
                'median': sorted(timings)[len(timings) // 2],
                'p95': sorted(timings)[int(len(timings) * 0.95)]
            }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all operations."""
        stats = {}
        for operation in self.profiles:
            operation_stats = self.get_stats(operation)
            if operation_stats:
                stats[operation] = operation_stats
        return stats


class AdaptiveBatchProcessor:
    """Adaptive batch processing for optimal throughput."""
    
    def __init__(self,
                 min_batch_size: int = 1,
                 max_batch_size: int = 32,
                 target_latency_ms: float = 100.0):
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        
        self.current_batch_size = min_batch_size
        self.recent_latencies: List[float] = []
        self.adaptation_history: List[Tuple[int, float]] = []
        
        self._lock = threading.Lock()
    
    def process_batch(self, 
                     items: List[Any],
                     processor: Callable[[List[Any]], List[Any]]) -> List[Any]:
        """Process items in adaptive batches."""
        if not items:
            return []
        
        results = []
        
        for i in range(0, len(items), self.current_batch_size):
            batch = items[i:i + self.current_batch_size]
            
            start_time = time.perf_counter()
            batch_results = processor(batch)
            duration_ms = (time.perf_counter() - start_time) * 1000
            
            results.extend(batch_results)
            
            # Record performance and adapt
            self._record_batch_performance(len(batch), duration_ms)
            self._adapt_batch_size()
        
        return results
    
    def _record_batch_performance(self, batch_size: int, latency_ms: float):
        """Record batch performance metrics."""
        with self._lock:
            # Normalize latency per item
            per_item_latency = latency_ms / batch_size
            self.recent_latencies.append(per_item_latency)
            
            # Keep only recent measurements
            if len(self.recent_latencies) > 50:
                self.recent_latencies = self.recent_latencies[-50:]
            
            # Record adaptation history
            self.adaptation_history.append((batch_size, latency_ms))
            if len(self.adaptation_history) > 100:
                self.adaptation_history = self.adaptation_history[-100:]
    
    def _adapt_batch_size(self):
        """Adapt batch size based on performance."""
        with self._lock:
            if len(self.recent_latencies) < 5:
                return
            
            avg_latency = sum(self.recent_latencies[-5:]) / 5
            total_latency = avg_latency * self.current_batch_size
            
            if total_latency < self.target_latency_ms * 0.8:
                # Too fast, increase batch size
                new_size = min(self.current_batch_size + 1, self.max_batch_size)
            elif total_latency > self.target_latency_ms * 1.2:
                # Too slow, decrease batch size
                new_size = max(self.current_batch_size - 1, self.min_batch_size)
            else:
                # Just right, no change
                new_size = self.current_batch_size
            
            if new_size != self.current_batch_size:
                logger.debug(f"Adapting batch size: {self.current_batch_size} -> {new_size}")
                self.current_batch_size = new_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics."""
        with self._lock:
            if not self.recent_latencies:
                return {}
            
            return {
                'current_batch_size': self.current_batch_size,
                'avg_latency_per_item_ms': sum(self.recent_latencies) / len(self.recent_latencies),
                'min_latency_ms': min(self.recent_latencies),
                'max_latency_ms': max(self.recent_latencies),
                'batches_processed': len(self.adaptation_history)
            }


# Global instances
_compilation_cache: Optional[CompilationCache] = None
_profiler = OptimizationProfiler()


def get_compilation_cache(cache_dir: Optional[str] = None) -> CompilationCache:
    """Get global compilation cache instance."""
    global _compilation_cache
    if _compilation_cache is None:
        _compilation_cache = CompilationCache(cache_dir=cache_dir)
    return _compilation_cache


def get_profiler() -> OptimizationProfiler:
    """Get global profiler instance."""
    return _profiler


def profile_performance(operation_name: str):
    """Decorator for profiling function performance."""
    return _profiler.profile_function(operation_name)


def warm_up_cache(model_paths: List[str], target_configs: List[Dict[str, Any]]):
    """Warm up cache with commonly used models."""
    cache = get_compilation_cache()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for model_path in model_paths:
            for config in target_configs:
                # This would trigger actual compilation in a real implementation
                future = executor.submit(
                    lambda: logger.info(f"Would warm up cache for {model_path}")
                )
                futures.append(future)
        
        # Wait for completion
        concurrent.futures.wait(futures)
    
    logger.info(f"Cache warm-up completed for {len(model_paths)} models")