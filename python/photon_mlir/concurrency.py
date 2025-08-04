"""
Concurrent processing and resource pooling for photonic compiler.
"""

import os
import time
import threading
import multiprocessing
import queue
import weakref
from typing import Dict, Any, Optional, List, Callable, TypeVar, Generic, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, Future, as_completed
from contextlib import contextmanager
import psutil
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class ResourceMetrics:
    """Resource usage metrics."""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    active_threads: int = 0
    active_processes: int = 0
    queue_size: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    avg_task_duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class WorkerConfig:
    """Configuration for worker pools."""
    min_workers: int = 1
    max_workers: int = 8
    idle_timeout_seconds: float = 60.0
    task_timeout_seconds: float = 300.0
    queue_maxsize: int = 1000
    enable_auto_scaling: bool = True
    cpu_threshold: float = 80.0
    memory_threshold: float = 80.0


class AdaptiveThreadPool:
    """Thread pool with adaptive scaling based on system load."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.current_workers = config.min_workers
        self.executor: Optional[ThreadPoolExecutor] = None
        self.metrics = ResourceMetrics()
        
        self._lock = threading.Lock()
        self._task_times: List[float] = []
        self._last_scale_time = time.time()
        self._scale_cooldown = 30.0  # seconds
        
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the thread pool."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.executor = ThreadPoolExecutor(
            max_workers=self.current_workers,
            thread_name_prefix="photonic_worker"
        )
        
        logger.info(f"Initialized thread pool with {self.current_workers} workers")
    
    def submit(self, func: Callable[..., T], *args, **kwargs) -> Future[T]:
        """Submit task to thread pool."""
        if not self.executor:
            raise RuntimeError("Thread pool not initialized")
        
        # Monitor system resources before submitting
        self._update_metrics()
        
        # Auto-scale if enabled
        if self.config.enable_auto_scaling:
            self._maybe_scale_pool()
        
        # Wrap function to track execution time
        def timed_func(*args, **kwargs):
            start_time = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                self.metrics.completed_tasks += 1
                return result
            except Exception as e:
                self.metrics.failed_tasks += 1
                raise
            finally:
                duration = (time.perf_counter() - start_time) * 1000
                self._record_task_time(duration)
        
        return self.executor.submit(timed_func, *args, **kwargs)
    
    def map(self, func: Callable[[T], R], iterable: Iterator[T], 
           chunk_size: Optional[int] = None) -> Iterator[R]:
        """Map function over iterable using thread pool."""
        if not self.executor:
            raise RuntimeError("Thread pool not initialized")
        
        # Submit all tasks
        futures = [self.submit(func, item) for item in iterable]
        
        # Yield results as they complete
        for future in as_completed(futures):
            yield future.result()
    
    def _update_metrics(self):
        """Update resource usage metrics."""
        try:
            process = psutil.Process()
            
            self.metrics.cpu_percent = process.cpu_percent()
            self.metrics.memory_percent = process.memory_percent()
            self.metrics.memory_mb = process.memory_info().rss / (1024 * 1024)
            self.metrics.active_threads = process.num_threads()
            self.metrics.timestamp = time.time()
            
            # Update average task duration
            if self._task_times:
                self.metrics.avg_task_duration_ms = sum(self._task_times) / len(self._task_times)
        
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    
    def _record_task_time(self, duration_ms: float):
        """Record task execution time."""
        with self._lock:
            self._task_times.append(duration_ms)
            # Keep only recent measurements
            if len(self._task_times) > 100:
                self._task_times = self._task_times[-100:]
    
    def _maybe_scale_pool(self):
        """Scale thread pool based on system load."""
        now = time.time()
        if now - self._last_scale_time < self._scale_cooldown:
            return  # Still in cooldown period
        
        cpu_high = self.metrics.cpu_percent > self.config.cpu_threshold
        memory_high = self.metrics.memory_percent > self.config.memory_threshold
        
        # Scale down if resource usage is high
        if (cpu_high or memory_high) and self.current_workers > self.config.min_workers:
            new_workers = max(
                self.config.min_workers,
                self.current_workers - 1
            )
            self._scale_to(new_workers)
            self._last_scale_time = now
            
        # Scale up if resources are available and tasks are queuing
        elif (not cpu_high and not memory_high and 
              self.current_workers < self.config.max_workers and
              self.metrics.avg_task_duration_ms > 100):  # Tasks taking long
            
            new_workers = min(
                self.config.max_workers,
                self.current_workers + 1
            )
            self._scale_to(new_workers)
            self._last_scale_time = now
    
    def _scale_to(self, new_worker_count: int):
        """Scale thread pool to new worker count."""
        if new_worker_count == self.current_workers:
            return
        
        logger.info(f"Scaling thread pool: {self.current_workers} -> {new_worker_count}")
        
        self.current_workers = new_worker_count
        self._initialize_pool()
    
    def shutdown(self, wait: bool = True):
        """Shutdown thread pool."""
        if self.executor:
            self.executor.shutdown(wait=wait)
    
    def get_metrics(self) -> ResourceMetrics:
        """Get current resource metrics."""
        self._update_metrics()
        return self.metrics


class CompilationWorkerPool:
    """Specialized worker pool for photonic compilation tasks."""
    
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.thread_pool = AdaptiveThreadPool(config)
        
        # Process pool for CPU-intensive tasks
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self._init_process_pool()
        
        self._active_compilations: Dict[str, Future] = {}
        self._compilation_cache: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
    
    def _init_process_pool(self):
        """Initialize process pool for CPU-intensive tasks."""
        cpu_count = multiprocessing.cpu_count()
        max_workers = min(cpu_count, self.config.max_workers)
        
        self.process_pool = ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context('spawn')
        )
        
        logger.info(f"Initialized process pool with {max_workers} workers")
    
    def compile_model_async(self,
                           model_path: str,
                           target_config: Dict[str, Any],
                           use_processes: bool = False) -> Future[Any]:
        """Compile model asynchronously."""
        # Create task identifier
        task_id = f"{model_path}:{hash(str(sorted(target_config.items())))}"
        
        # Check if already compiling
        if task_id in self._active_compilations:
            logger.debug(f"Compilation already in progress: {model_path}")
            return self._active_compilations[task_id]
        
        # Check cache
        if task_id in self._compilation_cache:
            logger.debug(f"Returning cached compilation: {model_path}")
            # Return completed future with cached result
            future = Future()
            future.set_result(self._compilation_cache[task_id])
            return future
        
        # Submit compilation task
        if use_processes and self.process_pool:
            future = self.process_pool.submit(
                self._compile_model_worker,
                model_path,
                target_config
            )
        else:
            future = self.thread_pool.submit(
                self._compile_model_worker,
                model_path,
                target_config
            )
        
        # Track active compilation
        self._active_compilations[task_id] = future
        
        # Clean up when done
        def cleanup(fut):
            self._active_compilations.pop(task_id, None)
            if not fut.exception():
                self._compilation_cache[task_id] = fut.result()
        
        future.add_done_callback(cleanup)
        
        return future
    
    def _compile_model_worker(self, model_path: str, target_config: Dict[str, Any]) -> Any:
        """Worker function for model compilation."""
        logger.info(f"Starting compilation: {model_path}")
        start_time = time.time()
        
        try:
            # Import here to avoid import issues in multiprocessing
            from .compiler import PhotonicCompiler
            from .core import TargetConfig
            
            # Create compiler
            config = TargetConfig(**target_config)
            compiler = PhotonicCompiler(config)
            
            # Compile model
            compiled_model = compiler.compile_onnx(model_path)
            
            duration = time.time() - start_time
            logger.info(f"Compilation completed: {model_path} ({duration:.2f}s)")
            
            return compiled_model
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Compilation failed: {model_path} ({duration:.2f}s): {e}")
            raise
    
    def compile_batch_async(self,
                           model_paths: List[str],
                           target_configs: List[Dict[str, Any]],
                           max_concurrent: Optional[int] = None) -> List[Future[Any]]:
        """Compile multiple models concurrently."""
        if max_concurrent is None:
            max_concurrent = self.config.max_workers
        
        futures = []
        semaphore = threading.Semaphore(max_concurrent)
        
        def limited_compile(model_path, config):
            with semaphore:
                return self._compile_model_worker(model_path, config)
        
        for model_path, config in zip(model_paths, target_configs):
            future = self.thread_pool.submit(limited_compile, model_path, config)
            futures.append(future)
        
        return futures
    
    def wait_for_completion(self, futures: List[Future], timeout: Optional[float] = None):
        """Wait for all futures to complete."""
        completed = []
        failed = []
        
        for future in as_completed(futures, timeout=timeout):
            try:
                result = future.result()
                completed.append(result)
            except Exception as e:
                failed.append(e)
                logger.error(f"Task failed: {e}")
        
        return completed, failed
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get worker pool metrics."""
        thread_metrics = self.thread_pool.get_metrics()
        
        return {
            'thread_pool': {
                'current_workers': self.thread_pool.current_workers,
                'cpu_percent': thread_metrics.cpu_percent,
                'memory_percent': thread_metrics.memory_percent,
                'memory_mb': thread_metrics.memory_mb,
                'active_threads': thread_metrics.active_threads,
                'completed_tasks': thread_metrics.completed_tasks,
                'failed_tasks': thread_metrics.failed_tasks,
                'avg_task_duration_ms': thread_metrics.avg_task_duration_ms
            },
            'process_pool': {
                'max_workers': self.process_pool._max_workers if self.process_pool else 0
            },
            'active_compilations': len(self._active_compilations),
            'cached_compilations': len(self._compilation_cache)
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown worker pools."""
        self.thread_pool.shutdown(wait=wait)
        if self.process_pool:
            self.process_pool.shutdown(wait=wait)


class ResourcePool(Generic[T]):
    """Generic resource pool with lifecycle management."""
    
    def __init__(self,
                 factory: Callable[[], T],
                 destroyer: Optional[Callable[[T], None]] = None,
                 min_size: int = 1,
                 max_size: int = 10,
                 idle_timeout: float = 300.0):
        
        self.factory = factory
        self.destroyer = destroyer
        self.min_size = min_size
        self.max_size = max_size
        self.idle_timeout = idle_timeout
        
        self._pool: queue.Queue[T] = queue.Queue(maxsize=max_size)
        self._in_use: set[T] = set()
        self._created_count = 0
        self._last_access: Dict[T, float] = {}
        self._lock = threading.Lock()
        
        # Pre-populate pool
        self._fill_pool()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _fill_pool(self):
        """Fill pool to minimum size."""
        while self._pool.qsize() < self.min_size and self._created_count < self.max_size:
            try:
                resource = self.factory()
                self._pool.put_nowait(resource)
                self._created_count += 1
                self._last_access[resource] = time.time()
            except queue.Full:
                break
    
    @contextmanager
    def acquire(self, timeout: Optional[float] = None):
        """Acquire resource from pool."""
        resource = None
        try:
            # Try to get existing resource
            try:
                resource = self._pool.get(timeout=timeout or 1.0)
            except queue.Empty:
                # Create new resource if under limit
                with self._lock:
                    if self._created_count < self.max_size:
                        resource = self.factory()
                        self._created_count += 1
                    else:
                        # Wait for available resource
                        resource = self._pool.get(timeout=timeout)
            
            if resource is None:
                raise RuntimeError("Could not acquire resource")
            
            # Mark as in use
            with self._lock:
                self._in_use.add(resource)
                self._last_access[resource] = time.time()
            
            yield resource
            
        finally:
            # Return resource to pool
            if resource is not None:
                with self._lock:
                    self._in_use.discard(resource)
                    self._last_access[resource] = time.time()
                
                try:
                    self._pool.put_nowait(resource)
                except queue.Full:
                    # Pool is full, destroy excess resource
                    if self.destroyer:
                        self.destroyer(resource)
                    with self._lock:
                        self._created_count -= 1
                        self._last_access.pop(resource, None)
    
    def _cleanup_worker(self):
        """Background thread to clean up idle resources."""
        while True:
            try:
                time.sleep(60)  # Check every minute
                self._cleanup_idle_resources()
            except Exception as e:
                logger.error(f"Resource pool cleanup error: {e}")
    
    def _cleanup_idle_resources(self):
        """Clean up idle resources."""
        now = time.time()
        resources_to_remove = []
        
        with self._lock:
            # Find idle resources
            for resource, last_access in self._last_access.items():
                if (resource not in self._in_use and 
                    now - last_access > self.idle_timeout and
                    self._created_count > self.min_size):
                    resources_to_remove.append(resource)
        
        # Remove idle resources
        for resource in resources_to_remove:
            try:
                # Try to remove from queue
                temp_resources = []
                found = False
                
                while not self._pool.empty():
                    try:
                        pooled_resource = self._pool.get_nowait()
                        if pooled_resource == resource:
                            found = True
                            break
                        else:
                            temp_resources.append(pooled_resource)
                    except queue.Empty:
                        break
                
                # Put back non-matching resources
                for temp_resource in temp_resources:
                    try:
                        self._pool.put_nowait(temp_resource)
                    except queue.Full:
                        pass
                
                if found:
                    # Destroy the resource
                    if self.destroyer:
                        self.destroyer(resource)
                    
                    with self._lock:
                        self._created_count -= 1
                        self._last_access.pop(resource, None)
                    
                    logger.debug(f"Cleaned up idle resource (pool size: {self._created_count})")
            
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")
    
    def size(self) -> Tuple[int, int, int]:
        """Get pool statistics: (total, in_use, available)."""
        with self._lock:
            total = self._created_count
            in_use = len(self._in_use)
            available = self._pool.qsize()
            return total, in_use, available
    
    def shutdown(self):
        """Shutdown resource pool."""
        # Clear queue and destroy resources
        while not self._pool.empty():
            try:
                resource = self._pool.get_nowait()
                if self.destroyer:
                    self.destroyer(resource)
            except queue.Empty:
                break
        
        # Destroy in-use resources (they will be cleaned up when returned)
        with self._lock:
            self._created_count = 0
            self._last_access.clear()


# Global worker pool instance
_worker_pool: Optional[CompilationWorkerPool] = None


def get_worker_pool(config: Optional[WorkerConfig] = None) -> CompilationWorkerPool:
    """Get global worker pool instance."""
    global _worker_pool
    if _worker_pool is None:
        if config is None:
            config = WorkerConfig()
        _worker_pool = CompilationWorkerPool(config)
    return _worker_pool


def shutdown_worker_pool():
    """Shutdown global worker pool."""
    global _worker_pool
    if _worker_pool:
        _worker_pool.shutdown()
        _worker_pool = None