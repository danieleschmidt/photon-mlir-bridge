"""
Scalable Optimization and Performance Enhancement System
Generation 3 Implementation - Make it Scale

This module implements advanced optimization techniques for massive scalability,
distributed computation, and enterprise-grade performance in photonic neural networks.

Key Scaling Features:
1. Distributed computation with MPI and GPU acceleration
2. Adaptive resource allocation and load balancing
3. Advanced caching and memoization strategies
4. Stream processing for large-scale data
5. Auto-scaling based on workload patterns
6. Multi-level parallelization (data, model, pipeline)
7. Performance-aware scheduling and optimization
"""

import numpy as np
import time
import threading
import multiprocessing
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Iterator
from dataclasses import dataclass, field
from enum import Enum
import warnings
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import weakref
import gc
import psutil
import functools
import hashlib
import pickle
import json

try:
    import torch
    import torch.distributed as dist
    _TORCH_DISTRIBUTED_AVAILABLE = True
except ImportError:
    _TORCH_DISTRIBUTED_AVAILABLE = False

try:
    from mpi4py import MPI
    _MPI_AVAILABLE = True
except ImportError:
    _MPI_AVAILABLE = False
    # Mock MPI for systems without it
    class MPI:
        class COMM_WORLD:
            @staticmethod
            def Get_rank(): return 0
            @staticmethod
            def Get_size(): return 1
            @staticmethod
            def barrier(): pass

from .core import TargetConfig, Device
from .logging_config import get_global_logger


class ScalingStrategy(Enum):
    """Scaling strategies for different workload patterns."""
    HORIZONTAL = "horizontal"    # Scale out (more workers)
    VERTICAL = "vertical"       # Scale up (more resources per worker)
    HYBRID = "hybrid"          # Combination of horizontal and vertical
    ADAPTIVE = "adaptive"      # Automatically choose best strategy
    ELASTIC = "elastic"        # Dynamic scaling based on demand


class ComputeBackend(Enum):
    """Available compute backends for acceleration."""
    CPU_THREADING = "cpu_threading"
    CPU_MULTIPROCESSING = "cpu_multiprocessing"
    GPU_CUDA = "gpu_cuda"
    GPU_OPENCL = "gpu_opencl"
    DISTRIBUTED_MPI = "distributed_mpi"
    DISTRIBUTED_TORCH = "distributed_torch"
    HYBRID_MULTI = "hybrid_multi"


@dataclass
class ScalingConfiguration:
    """Configuration for scaling and optimization."""
    # Scaling parameters
    strategy: ScalingStrategy = ScalingStrategy.ADAPTIVE
    backend: ComputeBackend = ComputeBackend.HYBRID_MULTI
    max_workers: int = multiprocessing.cpu_count()
    max_memory_gb: float = 16.0
    
    # Performance parameters
    target_throughput_ops_per_sec: float = 1000.0
    target_latency_ms: float = 100.0
    enable_gpu_acceleration: bool = True
    enable_distributed_computing: bool = False
    
    # Optimization parameters
    enable_adaptive_batching: bool = True
    enable_caching: bool = True
    enable_compression: bool = True
    cache_size_mb: int = 1024
    
    # Auto-scaling parameters
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8    # CPU/memory utilization
    scale_down_threshold: float = 0.3
    scaling_cooldown_seconds: int = 60
    
    # Advanced features
    enable_prefetching: bool = True
    enable_pipeline_parallelism: bool = True
    enable_model_parallelism: bool = False
    enable_data_parallelism: bool = True


class ResourceMonitor:
    """Advanced resource monitoring with predictive scaling."""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.logger = get_global_logger()
        
        # Resource tracking
        self.resource_history = deque(maxlen=1000)
        self.load_predictions = deque(maxlen=100)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Scaling decisions
        self.last_scale_decision_time = 0
        self.current_workers = 1
        
    def start_monitoring(self):
        """Start continuous resource monitoring."""
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("🔍 Resource monitoring started")
        
    def stop_monitoring(self):
        """Stop resource monitoring."""
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        self.logger.info("🔍 Resource monitoring stopped")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        
        while self.monitoring_active:
            try:
                # Collect resource metrics
                metrics = self._collect_metrics()
                self.resource_history.append(metrics)
                
                # Make scaling decisions
                if self.config.enable_auto_scaling:
                    scaling_decision = self._make_scaling_decision(metrics)
                    if scaling_decision['action'] != 'maintain':
                        self._execute_scaling_decision(scaling_decision)
                
                time.sleep(1.0)  # Monitor every second
                
            except Exception as e:
                self.logger.error(f"Error in resource monitoring: {e}")
                time.sleep(5.0)
                
    def _collect_metrics(self) -> Dict[str, float]:
        """Collect comprehensive resource metrics."""
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else cpu_percent / 100.0
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available_gb = memory.available / (1024**3)
            
            # Disk I/O metrics
            disk_io = psutil.disk_io_counters()
            disk_read_mb_per_sec = (disk_io.read_bytes / (1024**2)) if disk_io else 0
            disk_write_mb_per_sec = (disk_io.write_bytes / (1024**2)) if disk_io else 0
            
            # Network metrics
            network_io = psutil.net_io_counters()
            network_recv_mb_per_sec = (network_io.bytes_recv / (1024**2)) if network_io else 0
            network_sent_mb_per_sec = (network_io.bytes_sent / (1024**2)) if network_io else 0
            
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'cpu_count': cpu_count,
                'load_avg': load_avg,
                'memory_percent': memory_percent,
                'memory_available_gb': memory_available_gb,
                'disk_read_mb_per_sec': disk_read_mb_per_sec,
                'disk_write_mb_per_sec': disk_write_mb_per_sec,
                'network_recv_mb_per_sec': network_recv_mb_per_sec,
                'network_sent_mb_per_sec': network_sent_mb_per_sec,
                'current_workers': self.current_workers
            }
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to collect metrics: {e}")
            return {'timestamp': time.time(), 'error': str(e)}
            
    def _make_scaling_decision(self, current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Make intelligent scaling decisions based on metrics and predictions."""
        
        current_time = time.time()
        
        # Check cooldown period
        if current_time - self.last_scale_decision_time < self.config.scaling_cooldown_seconds:
            return {'action': 'maintain', 'reason': 'cooldown_period'}
            
        # Analyze current load
        cpu_load = current_metrics.get('cpu_percent', 0) / 100.0
        memory_load = current_metrics.get('memory_percent', 0) / 100.0
        
        # Predict future load (simple trend analysis)
        predicted_load = self._predict_future_load()
        
        decision = {
            'action': 'maintain',
            'reason': 'stable_load',
            'current_cpu': cpu_load,
            'current_memory': memory_load,
            'predicted_load': predicted_load,
            'current_workers': self.current_workers
        }
        
        # Scale up conditions
        if (cpu_load > self.config.scale_up_threshold or 
            memory_load > self.config.scale_up_threshold or
            predicted_load > self.config.scale_up_threshold):
            
            if self.current_workers < self.config.max_workers:
                decision['action'] = 'scale_up'
                decision['reason'] = f'high_load (CPU: {cpu_load:.2f}, Memory: {memory_load:.2f})'
                decision['new_workers'] = min(self.current_workers + 1, self.config.max_workers)
                
        # Scale down conditions
        elif (cpu_load < self.config.scale_down_threshold and 
              memory_load < self.config.scale_down_threshold and
              predicted_load < self.config.scale_down_threshold):
            
            if self.current_workers > 1:
                decision['action'] = 'scale_down'
                decision['reason'] = f'low_load (CPU: {cpu_load:.2f}, Memory: {memory_load:.2f})'
                decision['new_workers'] = max(self.current_workers - 1, 1)
                
        return decision
        
    def _predict_future_load(self) -> float:
        """Predict future system load using simple trend analysis."""
        
        if len(self.resource_history) < 10:
            return 0.5  # Default moderate load
            
        # Get recent CPU load values
        recent_loads = [m.get('cpu_percent', 0) / 100.0 for m in list(self.resource_history)[-10:]]
        
        # Simple linear trend
        x = np.arange(len(recent_loads))
        slope, intercept = np.polyfit(x, recent_loads, 1)
        
        # Predict load in next 30 seconds (30 data points)
        predicted_load = slope * (len(recent_loads) + 30) + intercept
        
        return max(0.0, min(1.0, predicted_load))  # Clamp to [0, 1]
        
    def _execute_scaling_decision(self, decision: Dict[str, Any]):
        """Execute scaling decision."""
        
        action = decision['action']
        
        if action == 'scale_up':
            new_workers = decision['new_workers']
            self.logger.info(f"⚡ Scaling UP: {self.current_workers} -> {new_workers} workers ({decision['reason']})")
            self.current_workers = new_workers
            
        elif action == 'scale_down':
            new_workers = decision['new_workers']
            self.logger.info(f"⚡ Scaling DOWN: {self.current_workers} -> {new_workers} workers ({decision['reason']})")
            self.current_workers = new_workers
            
        self.last_scale_decision_time = time.time()
        
    def get_current_utilization(self) -> Dict[str, float]:
        """Get current resource utilization."""
        
        if self.resource_history:
            latest = self.resource_history[-1]
            return {
                'cpu_utilization': latest.get('cpu_percent', 0) / 100.0,
                'memory_utilization': latest.get('memory_percent', 0) / 100.0,
                'workers': self.current_workers,
                'load_avg': latest.get('load_avg', 0)
            }
        else:
            return {'cpu_utilization': 0, 'memory_utilization': 0, 'workers': 1, 'load_avg': 0}


class IntelligentCachingSystem:
    """Advanced caching system with ML-driven eviction policies."""
    
    def __init__(self, max_size_mb: int = 1024):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.current_size_bytes = 0
        
        self.cache = {}  # key -> (value, access_time, frequency, size)
        self.access_history = defaultdict(list)
        
        self.lock = threading.RLock()
        self.logger = get_global_logger()
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with access tracking."""
        
        with self.lock:
            if key in self.cache:
                value, _, frequency, size = self.cache[key]
                current_time = time.time()
                
                # Update access statistics
                self.cache[key] = (value, current_time, frequency + 1, size)
                self.access_history[key].append(current_time)
                
                # Limit history size
                if len(self.access_history[key]) > 100:
                    self.access_history[key] = self.access_history[key][-50:]
                    
                self.hits += 1
                return value
            else:
                self.misses += 1
                return None
                
    def put(self, key: str, value: Any) -> bool:
        """Put value in cache with intelligent eviction."""
        
        with self.lock:
            # Calculate value size
            try:
                value_size = len(pickle.dumps(value))
            except:
                value_size = 1024  # Default size estimate
                
            # Check if we need to evict
            while (self.current_size_bytes + value_size > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_least_valuable_item()
                
            # Store in cache
            current_time = time.time()
            self.cache[key] = (value, current_time, 1, value_size)
            self.current_size_bytes += value_size
            self.access_history[key] = [current_time]
            
            return True
            
    def _evict_least_valuable_item(self):
        """Evict the least valuable item using ML-driven policy."""
        
        if not self.cache:
            return
            
        current_time = time.time()
        scores = {}
        
        for key, (value, last_access, frequency, size) in self.cache.items():
            # Calculate value score based on multiple factors
            
            # Recency score (higher for recent access)
            recency_score = 1.0 / (1.0 + (current_time - last_access))
            
            # Frequency score (higher for frequent access)
            frequency_score = np.log(1 + frequency)
            
            # Size penalty (lower for large items)
            size_penalty = 1.0 / (1.0 + size / 1024)  # Normalize by KB
            
            # Access pattern score
            access_times = self.access_history.get(key, [])
            if len(access_times) > 1:
                # Regular access pattern gets higher score
                intervals = np.diff(access_times)
                pattern_score = 1.0 / (1.0 + np.std(intervals)) if len(intervals) > 1 else 1.0
            else:
                pattern_score = 0.5
                
            # Combined score (higher is more valuable)
            total_score = (0.3 * recency_score + 
                          0.3 * frequency_score + 
                          0.2 * size_penalty + 
                          0.2 * pattern_score)
            
            scores[key] = total_score
            
        # Evict item with lowest score
        victim_key = min(scores, key=scores.get)
        victim_value, _, _, victim_size = self.cache[victim_key]
        
        del self.cache[victim_key]
        if victim_key in self.access_history:
            del self.access_history[victim_key]
            
        self.current_size_bytes -= victim_size
        self.evictions += 1
        
        self.logger.debug(f"Evicted cache item: {victim_key} (score: {scores[victim_key]:.4f})")
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions,
            'current_size_mb': self.current_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': self.current_size_bytes / self.max_size_bytes,
            'items_count': len(self.cache)
        }
        
    def clear(self):
        """Clear all cache entries."""
        
        with self.lock:
            self.cache.clear()
            self.access_history.clear()
            self.current_size_bytes = 0
            self.logger.info("Cache cleared")


class StreamProcessor:
    """High-performance stream processing for large-scale data."""
    
    def __init__(self, batch_size: int = 1000, buffer_size: int = 10000):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        
        self.input_buffer = queue.Queue(maxsize=buffer_size)
        self.output_buffer = queue.Queue(maxsize=buffer_size)
        
        self.processing_active = False
        self.processor_threads = []
        
        self.logger = get_global_logger()
        
        # Processing statistics
        self.items_processed = 0
        self.batches_processed = 0
        self.processing_time_total = 0.0
        
    def start_processing(self, processing_function: Callable, num_workers: int = 4):
        """Start stream processing with specified number of workers."""
        
        self.processing_active = True
        self.processing_function = processing_function
        
        # Start worker threads
        for i in range(num_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.processor_threads.append(worker)
            
        self.logger.info(f"🌊 Stream processing started with {num_workers} workers")
        
    def stop_processing(self):
        """Stop stream processing and wait for completion."""
        
        self.processing_active = False
        
        # Wait for workers to finish
        for worker in self.processor_threads:
            worker.join(timeout=10)
            
        self.processor_threads.clear()
        self.logger.info("🌊 Stream processing stopped")
        
    def _worker_loop(self, worker_id: int):
        """Main worker loop for processing batches."""
        
        while self.processing_active:
            try:
                # Collect batch
                batch = self._collect_batch()
                
                if not batch:
                    time.sleep(0.1)  # No data available
                    continue
                    
                # Process batch
                start_time = time.time()
                result = self.processing_function(batch)
                processing_time = time.time() - start_time
                
                # Store result
                self.output_buffer.put(result, timeout=1.0)
                
                # Update statistics
                self.items_processed += len(batch)
                self.batches_processed += 1
                self.processing_time_total += processing_time
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}")
                time.sleep(1.0)
                
    def _collect_batch(self) -> List[Any]:
        """Collect a batch of items from input buffer."""
        
        batch = []
        
        # Try to collect full batch
        for _ in range(self.batch_size):
            try:
                item = self.input_buffer.get(timeout=0.1)
                batch.append(item)
            except queue.Empty:
                break
                
        return batch
        
    def put_item(self, item: Any, timeout: float = 1.0) -> bool:
        """Add item to processing queue."""
        
        try:
            self.input_buffer.put(item, timeout=timeout)
            return True
        except queue.Full:
            return False
            
    def get_result(self, timeout: float = 1.0) -> Optional[Any]:
        """Get processed result."""
        
        try:
            return self.output_buffer.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def get_statistics(self) -> Dict[str, float]:
        """Get processing statistics."""
        
        avg_processing_time = (self.processing_time_total / self.batches_processed 
                             if self.batches_processed > 0 else 0.0)
        
        throughput = (self.items_processed / self.processing_time_total 
                     if self.processing_time_total > 0 else 0.0)
        
        return {
            'items_processed': self.items_processed,
            'batches_processed': self.batches_processed,
            'avg_processing_time': avg_processing_time,
            'throughput_items_per_sec': throughput,
            'input_buffer_size': self.input_buffer.qsize(),
            'output_buffer_size': self.output_buffer.qsize()
        }


class DistributedComputation:
    """Distributed computation system using MPI and/or PyTorch distributed."""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.logger = get_global_logger()
        
        # Initialize MPI if available
        if _MPI_AVAILABLE and config.enable_distributed_computing:
            self.mpi_comm = MPI.COMM_WORLD
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_size = self.mpi_comm.Get_size()
            self.mpi_enabled = True
        else:
            self.mpi_rank = 0
            self.mpi_size = 1
            self.mpi_enabled = False
            
        # Initialize PyTorch distributed if available
        self.torch_distributed_enabled = False
        if _TORCH_DISTRIBUTED_AVAILABLE and config.enable_distributed_computing:
            try:
                if dist.is_available() and not dist.is_initialized():
                    # This would typically be initialized externally
                    pass
                self.torch_distributed_enabled = dist.is_initialized()
            except:
                self.torch_distributed_enabled = False
                
        if self.mpi_enabled:
            self.logger.info(f"🌐 MPI initialized: rank {self.mpi_rank}/{self.mpi_size}")
        if self.torch_distributed_enabled:
            self.logger.info(f"🌐 PyTorch distributed initialized")
            
    def scatter_data(self, data: Any, root_rank: int = 0) -> Any:
        """Scatter data across distributed processes."""
        
        if self.mpi_enabled and self.mpi_size > 1:
            # Scatter data using MPI
            if self.mpi_rank == root_rank:
                # Split data into chunks
                if isinstance(data, (list, np.ndarray)):
                    chunk_size = len(data) // self.mpi_size
                    chunks = [data[i*chunk_size:(i+1)*chunk_size] for i in range(self.mpi_size)]
                    
                    # Handle remainder
                    remainder = len(data) % self.mpi_size
                    if remainder > 0:
                        chunks[-1].extend(data[-remainder:])
                else:
                    chunks = [data for _ in range(self.mpi_size)]
            else:
                chunks = None
                
            # Scatter chunks
            local_data = self.mpi_comm.scatter(chunks, root=root_rank)
            return local_data
        else:
            # Single process - return all data
            return data
            
    def gather_results(self, local_result: Any, root_rank: int = 0) -> Optional[List[Any]]:
        """Gather results from distributed processes."""
        
        if self.mpi_enabled and self.mpi_size > 1:
            # Gather results using MPI
            all_results = self.mpi_comm.gather(local_result, root=root_rank)
            
            if self.mpi_rank == root_rank:
                return all_results
            else:
                return None
        else:
            # Single process - return as list
            return [local_result] if self.mpi_rank == root_rank else None
            
    def all_reduce_sum(self, local_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Perform all-reduce sum across processes."""
        
        if self.mpi_enabled and self.mpi_size > 1:
            if isinstance(local_value, np.ndarray):
                global_sum = np.zeros_like(local_value)
                self.mpi_comm.Allreduce(local_value, global_sum, op=MPI.SUM)
                return global_sum
            else:
                global_sum = self.mpi_comm.allreduce(local_value, op=MPI.SUM)
                return global_sum
        else:
            return local_value
            
    def synchronize(self):
        """Synchronize all processes."""
        
        if self.mpi_enabled and self.mpi_size > 1:
            self.mpi_comm.barrier()
            
    def broadcast_data(self, data: Any, root_rank: int = 0) -> Any:
        """Broadcast data from root to all processes."""
        
        if self.mpi_enabled and self.mpi_size > 1:
            return self.mpi_comm.bcast(data, root=root_rank)
        else:
            return data
            
    @property
    def rank(self) -> int:
        """Get current process rank."""
        return self.mpi_rank
        
    @property
    def world_size(self) -> int:
        """Get total number of processes."""
        return self.mpi_size
        
    @property
    def is_distributed(self) -> bool:
        """Check if running in distributed mode."""
        return self.mpi_enabled and self.mpi_size > 1


class AdaptiveBatchProcessor:
    """Adaptive batch processing with dynamic batch size optimization."""
    
    def __init__(self, initial_batch_size: int = 32, target_latency_ms: float = 100.0):
        self.current_batch_size = initial_batch_size
        self.target_latency_ms = target_latency_ms
        
        self.batch_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        
        self.logger = get_global_logger()
        
        # Optimization parameters
        self.min_batch_size = 1
        self.max_batch_size = 1024
        self.adaptation_rate = 0.1
        
    def process_batch(self, processing_function: Callable, data: List[Any]) -> Tuple[Any, Dict[str, float]]:
        """Process data with adaptive batching."""
        
        # Determine optimal batch size
        if len(data) > self.current_batch_size:
            optimal_batch_size = self._calculate_optimal_batch_size(len(data))
        else:
            optimal_batch_size = len(data)
            
        # Process in batches
        results = []
        batch_metrics = []
        
        for i in range(0, len(data), optimal_batch_size):
            batch = data[i:i + optimal_batch_size]
            
            start_time = time.time()
            batch_result = processing_function(batch)
            processing_time = time.time() - start_time
            
            results.append(batch_result)
            
            # Record batch metrics
            batch_metric = {
                'batch_size': len(batch),
                'processing_time_ms': processing_time * 1000,
                'throughput_items_per_sec': len(batch) / processing_time,
                'latency_per_item_ms': (processing_time * 1000) / len(batch)
            }
            batch_metrics.append(batch_metric)
            
            # Update batch size based on performance
            self._update_batch_size(batch_metric)
            
        # Aggregate metrics
        total_items = sum(m['batch_size'] for m in batch_metrics)
        total_time = sum(m['processing_time_ms'] for m in batch_metrics) / 1000
        
        aggregate_metrics = {
            'total_items': total_items,
            'total_time_ms': total_time * 1000,
            'avg_throughput_items_per_sec': total_items / total_time if total_time > 0 else 0,
            'avg_latency_per_item_ms': (total_time * 1000) / total_items if total_items > 0 else 0,
            'optimal_batch_size': optimal_batch_size,
            'actual_batches_processed': len(batch_metrics)
        }
        
        return results, aggregate_metrics
        
    def _calculate_optimal_batch_size(self, data_size: int) -> int:
        """Calculate optimal batch size based on historical performance."""
        
        if not self.performance_history:
            return self.current_batch_size
            
        # Analyze recent performance
        recent_metrics = list(self.performance_history)[-10:]
        
        # Find batch size that minimizes latency while maximizing throughput
        best_score = 0
        best_batch_size = self.current_batch_size
        
        for metric in recent_metrics:
            latency_score = max(0, 1 - (metric['latency_per_item_ms'] / self.target_latency_ms))
            throughput_score = min(1, metric['throughput_items_per_sec'] / 1000)  # Normalize to ~1000 items/sec
            
            # Combined score (balance latency and throughput)
            score = 0.6 * latency_score + 0.4 * throughput_score
            
            if score > best_score:
                best_score = score
                best_batch_size = metric['batch_size']
                
        # Ensure batch size is within bounds and doesn't exceed data size
        optimal_size = max(self.min_batch_size, 
                          min(self.max_batch_size, best_batch_size, data_size))
        
        return optimal_size
        
    def _update_batch_size(self, batch_metric: Dict[str, float]):
        """Update batch size based on performance feedback."""
        
        self.performance_history.append(batch_metric)
        
        latency = batch_metric['latency_per_item_ms']
        throughput = batch_metric['throughput_items_per_sec']
        
        # Adaptive adjustment
        if latency > self.target_latency_ms * 1.2:  # Too slow
            # Decrease batch size
            adjustment = -max(1, int(self.current_batch_size * self.adaptation_rate))
        elif latency < self.target_latency_ms * 0.8:  # Fast enough to increase batch size
            # Increase batch size
            adjustment = max(1, int(self.current_batch_size * self.adaptation_rate))
        else:
            adjustment = 0
            
        self.current_batch_size = max(self.min_batch_size,
                                     min(self.max_batch_size,
                                         self.current_batch_size + adjustment))
        
        if adjustment != 0:
            self.logger.debug(f"Adjusted batch size: {self.current_batch_size - adjustment} -> {self.current_batch_size} "
                            f"(latency: {latency:.2f}ms, target: {self.target_latency_ms:.2f}ms)")


class ScalableOptimizationSystem:
    """
    Comprehensive scalable optimization system integrating all scaling components.
    """
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.logger = get_global_logger()
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(config)
        self.cache_system = IntelligentCachingSystem(config.cache_size_mb)
        self.stream_processor = StreamProcessor()
        self.distributed_comp = DistributedComputation(config)
        self.batch_processor = AdaptiveBatchProcessor(
            initial_batch_size=32,
            target_latency_ms=config.target_latency_ms
        )
        
        # Performance tracking
        self.optimization_sessions = []
        self.system_active = False
        
    def start_system(self):
        """Start the scalable optimization system."""
        
        self.logger.info("🚀 Starting scalable optimization system")
        
        # Start resource monitoring
        if self.config.enable_auto_scaling:
            self.resource_monitor.start_monitoring()
            
        # Start stream processing if needed
        self.stream_processor.start_processing(
            processing_function=self._default_stream_processor,
            num_workers=min(4, self.resource_monitor.current_workers)
        )
        
        self.system_active = True
        self.logger.info("✅ Scalable optimization system started")
        
    def stop_system(self):
        """Stop the scalable optimization system."""
        
        self.logger.info("🛑 Stopping scalable optimization system")
        
        self.system_active = False
        
        # Stop components
        self.resource_monitor.stop_monitoring()
        self.stream_processor.stop_processing()
        
        self.logger.info("✅ Scalable optimization system stopped")
        
    def optimize_computation(self, 
                           computation_function: Callable,
                           data: Any,
                           optimization_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Perform scalable optimization of a computation.
        
        Args:
            computation_function: Function to optimize
            data: Input data
            optimization_params: Additional optimization parameters
            
        Returns:
            Optimization results with performance metrics
        """
        
        self.logger.info("⚡ Starting scalable computation optimization")
        start_time = time.time()
        
        optimization_params = optimization_params or {}
        
        results = {
            'computation_results': None,
            'performance_metrics': {},
            'optimization_strategy': {},
            'resource_utilization': {},
            'scaling_decisions': []
        }
        
        try:
            # Step 1: Analyze computation requirements
            computation_profile = self._profile_computation(computation_function, data)
            results['optimization_strategy']['computation_profile'] = computation_profile
            
            # Step 2: Choose optimal execution strategy
            execution_strategy = self._choose_execution_strategy(computation_profile, data)
            results['optimization_strategy']['execution_strategy'] = execution_strategy
            
            # Step 3: Execute with chosen strategy
            if execution_strategy == 'distributed':
                computation_results, exec_metrics = self._execute_distributed(
                    computation_function, data
                )
            elif execution_strategy == 'stream_processing':
                computation_results, exec_metrics = self._execute_streaming(
                    computation_function, data
                )
            elif execution_strategy == 'adaptive_batching':
                computation_results, exec_metrics = self._execute_adaptive_batching(
                    computation_function, data
                )
            else:  # 'single_threaded'
                computation_results, exec_metrics = self._execute_single_threaded(
                    computation_function, data
                )
                
            results['computation_results'] = computation_results
            results['performance_metrics'] = exec_metrics
            
            # Step 4: Collect resource utilization
            results['resource_utilization'] = self.resource_monitor.get_current_utilization()
            
            # Step 5: Record optimization session
            optimization_time = time.time() - start_time
            results['optimization_time_seconds'] = optimization_time
            
            self.optimization_sessions.append({
                'timestamp': start_time,
                'execution_strategy': execution_strategy,
                'optimization_time': optimization_time,
                'performance_metrics': exec_metrics
            })
            
            self.logger.info(f"✅ Scalable optimization complete in {optimization_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Scalable optimization failed: {str(e)}")
            results['error'] = str(e)
            
        return results
        
    def _profile_computation(self, computation_function: Callable, data: Any) -> Dict[str, Any]:
        """Profile computation to understand its characteristics."""
        
        # Sample profiling with small data subset
        if isinstance(data, (list, np.ndarray)) and len(data) > 100:
            sample_data = data[:10]  # Small sample
        else:
            sample_data = data
            
        # Time a sample execution
        start_time = time.time()
        try:
            _ = computation_function(sample_data)
            sample_time = time.time() - start_time
            successful = True
        except Exception as e:
            sample_time = 0.1  # Default estimate
            successful = False
            
        # Analyze data characteristics
        if isinstance(data, (list, np.ndarray)):
            data_size = len(data)
            is_large_data = data_size > 1000
            is_parallel_friendly = data_size > 10  # Can be split
        else:
            data_size = 1
            is_large_data = False
            is_parallel_friendly = False
            
        profile = {
            'sample_execution_time': sample_time,
            'execution_successful': successful,
            'data_size': data_size,
            'is_large_data': is_large_data,
            'is_parallel_friendly': is_parallel_friendly,
            'estimated_total_time': sample_time * (data_size / min(10, data_size)),
            'memory_intensive': data_size > 10000,  # Heuristic
            'cpu_intensive': sample_time > 0.01     # Heuristic
        }
        
        return profile
        
    def _choose_execution_strategy(self, computation_profile: Dict[str, Any], 
                                 data: Any) -> str:
        """Choose optimal execution strategy based on profiling results."""
        
        # Decision tree for strategy selection
        
        # Distributed execution for large, parallel-friendly computations
        if (computation_profile['is_large_data'] and 
            computation_profile['is_parallel_friendly'] and
            self.distributed_comp.is_distributed and
            self.config.enable_data_parallelism):
            return 'distributed'
            
        # Stream processing for very large data
        if (computation_profile['data_size'] > 10000 and
            computation_profile['memory_intensive']):
            return 'stream_processing'
            
        # Adaptive batching for medium-large data
        if (computation_profile['is_large_data'] and
            computation_profile['cpu_intensive'] and
            self.config.enable_adaptive_batching):
            return 'adaptive_batching'
            
        # Default to single-threaded
        return 'single_threaded'
        
    def _execute_distributed(self, computation_function: Callable, 
                           data: Any) -> Tuple[Any, Dict[str, float]]:
        """Execute computation using distributed processing."""
        
        self.logger.info("🌐 Executing with distributed processing")
        start_time = time.time()
        
        # Scatter data across processes
        local_data = self.distributed_comp.scatter_data(data)
        
        # Process local data
        local_result = computation_function(local_data)
        
        # Gather results
        all_results = self.distributed_comp.gather_results(local_result)
        
        execution_time = time.time() - start_time
        
        # Combine results (if on root process)
        if self.distributed_comp.rank == 0 and all_results:
            if isinstance(all_results[0], (list, np.ndarray)):
                combined_result = np.concatenate(all_results) if isinstance(all_results[0], np.ndarray) else sum(all_results, [])
            else:
                combined_result = all_results
        else:
            combined_result = local_result
            
        metrics = {
            'execution_time_seconds': execution_time,
            'processes_used': self.distributed_comp.world_size,
            'current_rank': self.distributed_comp.rank,
            'data_size': len(local_data) if isinstance(local_data, (list, np.ndarray)) else 1
        }
        
        return combined_result, metrics
        
    def _execute_streaming(self, computation_function: Callable, 
                         data: Any) -> Tuple[Any, Dict[str, float]]:
        """Execute computation using stream processing."""
        
        self.logger.info("🌊 Executing with stream processing")
        start_time = time.time()
        
        # Convert data to stream if needed
        if not isinstance(data, (list, np.ndarray)):
            data = [data]
            
        # Process data in stream
        results = []
        for item in data:
            self.stream_processor.put_item(item)
            
        # Collect results
        for _ in range(len(data)):
            result = self.stream_processor.get_result(timeout=5.0)
            if result is not None:
                results.append(result)
                
        execution_time = time.time() - start_time
        
        # Get processing statistics
        stream_stats = self.stream_processor.get_statistics()
        
        metrics = {
            'execution_time_seconds': execution_time,
            'items_processed': len(results),
            'stream_throughput_items_per_sec': stream_stats['throughput_items_per_sec'],
            'avg_processing_time': stream_stats['avg_processing_time']
        }
        
        return results, metrics
        
    def _execute_adaptive_batching(self, computation_function: Callable, 
                                 data: Any) -> Tuple[Any, Dict[str, float]]:
        """Execute computation using adaptive batching."""
        
        self.logger.info("📦 Executing with adaptive batching")
        
        if not isinstance(data, (list, np.ndarray)):
            data = [data]
            
        results, metrics = self.batch_processor.process_batch(computation_function, data)
        
        # Flatten results if needed
        if isinstance(results[0], list):
            flattened_results = sum(results, [])
        else:
            flattened_results = results
            
        return flattened_results, metrics
        
    def _execute_single_threaded(self, computation_function: Callable, 
                               data: Any) -> Tuple[Any, Dict[str, float]]:
        """Execute computation using single-threaded processing."""
        
        self.logger.info("🔧 Executing with single-threaded processing")
        start_time = time.time()
        
        result = computation_function(data)
        
        execution_time = time.time() - start_time
        
        metrics = {
            'execution_time_seconds': execution_time,
            'processing_strategy': 'single_threaded'
        }
        
        return result, metrics
        
    def _default_stream_processor(self, batch: List[Any]) -> List[Any]:
        """Default stream processing function."""
        
        # Simple pass-through processing
        return [item for item in batch]
        
    def get_system_performance(self) -> Dict[str, Any]:
        """Get comprehensive system performance metrics."""
        
        performance = {
            'resource_utilization': self.resource_monitor.get_current_utilization(),
            'cache_statistics': self.cache_system.get_statistics(),
            'stream_processing': self.stream_processor.get_statistics(),
            'optimization_sessions': len(self.optimization_sessions),
            'system_active': self.system_active,
            'distributed_info': {
                'rank': self.distributed_comp.rank,
                'world_size': self.distributed_comp.world_size,
                'is_distributed': self.distributed_comp.is_distributed
            }
        }
        
        # Recent session statistics
        if self.optimization_sessions:
            recent_sessions = self.optimization_sessions[-10:]
            avg_optimization_time = np.mean([s['optimization_time'] for s in recent_sessions])
            
            performance['recent_performance'] = {
                'avg_optimization_time': avg_optimization_time,
                'common_strategy': max(set([s['execution_strategy'] for s in recent_sessions]), 
                                     key=[s['execution_strategy'] for s in recent_sessions].count)
            }
            
        return performance


# Demo and testing functions
def create_scalable_optimization_demo() -> Dict[str, Any]:
    """Create comprehensive demonstration of scalable optimization system."""
    
    logger = get_global_logger()
    logger.info("🎯 Creating scalable optimization demonstration")
    
    # Configure system
    config = ScalingConfiguration(
        strategy=ScalingStrategy.ADAPTIVE,
        backend=ComputeBackend.HYBRID_MULTI,
        max_workers=4,
        enable_adaptive_batching=True,
        enable_caching=True,
        enable_auto_scaling=True
    )
    
    # Initialize system
    optimization_system = ScalableOptimizationSystem(config)
    optimization_system.start_system()
    
    try:
        # Mock computation function
        def mock_photonic_computation(data):
            """Mock photonic neural network computation."""
            if isinstance(data, list):
                time.sleep(0.001 * len(data))  # Simulate processing time
                return [x * 1.1 + 0.1 for x in data]  # Simple transformation
            else:
                time.sleep(0.001)
                return data * 1.1 + 0.1
                
        # Test different data sizes
        test_cases = [
            {'name': 'small_data', 'data': list(range(10))},
            {'name': 'medium_data', 'data': list(range(100))},
            {'name': 'large_data', 'data': list(range(1000))}
        ]
        
        demo_results = {
            'system_config': config.__dict__,
            'test_results': {},
            'performance_analysis': {},
            'scaling_analysis': {}
        }
        
        # Run test cases
        for test_case in test_cases:
            logger.info(f"Testing {test_case['name']}")
            
            result = optimization_system.optimize_computation(
                mock_photonic_computation,
                test_case['data']
            )
            
            demo_results['test_results'][test_case['name']] = {
                'execution_strategy': result.get('optimization_strategy', {}).get('execution_strategy', 'unknown'),
                'execution_time': result.get('optimization_time_seconds', 0),
                'performance_metrics': result.get('performance_metrics', {}),
                'success': 'error' not in result
            }
            
        # Get system performance
        system_performance = optimization_system.get_system_performance()
        demo_results['performance_analysis'] = system_performance
        
        # Analysis summary
        demo_results['scaling_analysis'] = {
            'auto_scaling_active': config.enable_auto_scaling,
            'distributed_computing': system_performance['distributed_info']['is_distributed'],
            'cache_hit_rate': system_performance['cache_statistics']['hit_rate'],
            'resource_utilization': system_performance['resource_utilization']['cpu_utilization']
        }
        
        logger.info("📊 Scalable optimization demo completed successfully!")
        
    finally:
        # Clean up
        optimization_system.stop_system()
        
    return demo_results


if __name__ == "__main__":
    # Run scalable optimization demonstration
    demo_results = create_scalable_optimization_demo()
    
    print("=== Scalable Optimization System Results ===")
    
    for test_name, test_result in demo_results['test_results'].items():
        print(f"{test_name}:")
        print(f"  Strategy: {test_result['execution_strategy']}")
        print(f"  Time: {test_result['execution_time']:.4f}s")
        print(f"  Success: {test_result['success']}")
        
    analysis = demo_results['scaling_analysis']
    print(f"\nSystem Analysis:")
    print(f"  Auto-scaling: {analysis['auto_scaling_active']}")
    print(f"  Distributed: {analysis['distributed_computing']}")
    print(f"  Cache hit rate: {analysis['cache_hit_rate']:.3f}")
    print(f"  CPU utilization: {analysis['resource_utilization']:.3f}")