"""
Comprehensive logging system for photonic compiler.
Generation 2: Make it Robust - Structured Logging with Performance Metrics
"""

import logging
import logging.handlers
import sys
import time
import json
import os
from typing import Dict, Any, Optional, Union
from contextlib import contextmanager
from functools import wraps
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback


@dataclass
class PerformanceMetrics:
    """Performance metrics for operations."""
    operation: str
    start_time: float
    end_time: float
    duration_ms: float
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    custom_metrics: Optional[Dict[str, Any]] = None
    
    @property
    def duration_seconds(self) -> float:
        """Get duration in seconds."""
        return self.duration_ms / 1000.0


class PhotonicLogger:
    """Advanced logging system for photonic compiler operations."""
    
    def __init__(self, 
                 name: str = "photonic_mlir",
                 level: Union[str, int] = logging.INFO,
                 log_file: Optional[str] = None,
                 json_logging: bool = False,
                 performance_logging: bool = True):
        """Initialize logger with configuration.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            json_logging: Use JSON format for structured logging
            performance_logging: Enable performance metrics logging
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.json_logging = json_logging
        self.performance_logging = performance_logging
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_console_handler()
        if log_file:
            self._setup_file_handler(log_file)
        
        # Performance tracking
        self.performance_metrics: Dict[str, PerformanceMetrics] = {}
        self.operation_stack: list = []
        
        # Session info
        self.session_id = f"session_{int(time.time())}"
        self.session_start = time.time()
        
        self.info(f"🚀 Photonic MLIR Logger initialized (session: {self.session_id})")
    
    def _setup_console_handler(self):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        
        if self.json_logging:
            formatter = JsonFormatter()
        else:
            formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handler(self, log_file: str):
        """Setup file logging handler with rotation."""
        # Create log directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Rotating file handler (10MB max, 5 files)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        if self.json_logging:
            formatter = JsonFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with extra context."""
        extra = {
            'session_id': self.session_id,
            'session_elapsed': time.time() - self.session_start,
            **kwargs
        }
        
        if self.json_logging:
            # For JSON logging, include extra data
            self.logger.log(level, message, extra=extra)
        else:
            # For regular logging, just log the message
            self.logger.log(level, message)
    
    def log_compilation_start(self, model_info: Dict[str, Any]):
        """Log compilation start with model information."""
        self.info(
            f"🔄 Starting compilation",
            operation="compilation_start",
            model_type=model_info.get('type', 'unknown'),
            model_size=model_info.get('size_mb', 0),
            target_device=model_info.get('target_device', 'unknown')
        )
    
    def log_compilation_end(self, success: bool, stats: Dict[str, Any]):
        """Log compilation completion with statistics."""
        if success:
            self.info(
                f"✅ Compilation completed successfully",
                operation="compilation_end",
                success=True,
                **stats
            )
        else:
            self.error(
                f"❌ Compilation failed",
                operation="compilation_end",
                success=False,
                **stats
            )
    
    def log_simulation_start(self, config: Dict[str, Any]):
        """Log simulation start."""
        self.info(
            f"🔬 Starting simulation",
            operation="simulation_start",
            **config
        )
    
    def log_simulation_end(self, success: bool, metrics: Dict[str, Any]):
        """Log simulation completion."""
        if success:
            self.info(
                f"✅ Simulation completed",
                operation="simulation_end",
                success=True,
                **metrics
            )
        else:
            self.error(
                f"❌ Simulation failed",
                operation="simulation_end",
                success=False,
                **metrics
            )
    
    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        if not self.performance_logging:
            return
        
        # Convert to dict and rename operation to avoid conflict
        metrics_dict = asdict(metrics)
        metrics_dict['operation_name'] = metrics_dict.pop('operation')
        
        self.info(
            f"⏱️  {metrics.operation} completed in {metrics.duration_ms:.2f}ms",
            operation="performance_metric",
            **metrics_dict
        )
        
        # Store metrics for analysis
        self.performance_metrics[f"{metrics.operation}_{len(self.performance_metrics)}"] = metrics
    
    def log_validation_result(self, operation: str, is_valid: bool, 
                            errors: list, warnings: list, recommendations: list):
        """Log validation results."""
        if is_valid:
            self.info(
                f"✅ {operation} validation passed",
                operation="validation",
                validation_type=operation,
                is_valid=True,
                warning_count=len(warnings),
                recommendation_count=len(recommendations)
            )
        else:
            self.error(
                f"❌ {operation} validation failed",
                operation="validation",
                validation_type=operation,
                is_valid=False,
                error_count=len(errors),
                warning_count=len(warnings),
                errors=errors[:5]  # Limit to first 5 errors
            )
    
    def log_exception(self, operation: str, exception: Exception):
        """Log exception with full traceback."""
        self.error(
            f"💥 Exception in {operation}: {str(exception)}",
            operation="exception",
            operation_name=operation,
            exception_type=type(exception).__name__,
            exception_message=str(exception),
            traceback=traceback.format_exc()
        )
    
    @contextmanager
    def performance_context(self, operation: str, **custom_metrics):
        """Context manager for performance monitoring."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        self.operation_stack.append(operation)
        
        try:
            self.debug(f"🔄 Starting {operation}")
            yield
            
            # Success path
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            end_memory = self._get_memory_usage()
            
            metrics = PerformanceMetrics(
                operation=operation,
                start_time=start_time,
                end_time=end_time,
                duration_ms=duration_ms,
                memory_usage_mb=end_memory - start_memory if start_memory and end_memory else None,
                custom_metrics=custom_metrics
            )
            
            self.log_performance_metrics(metrics)
            
        except Exception as e:
            # Error path
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            self.error(f"❌ {operation} failed after {duration_ms:.2f}ms: {str(e)}")
            self.log_exception(operation, e)
            raise
        
        finally:
            if self.operation_stack and self.operation_stack[-1] == operation:
                self.operation_stack.pop()
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return None
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get session performance summary."""
        session_duration = time.time() - self.session_start
        
        summary = {
            'session_id': self.session_id,
            'session_duration_seconds': session_duration,
            'operations_count': len(self.performance_metrics),
            'total_operation_time_ms': sum(m.duration_ms for m in self.performance_metrics.values()),
            'average_operation_time_ms': (
                sum(m.duration_ms for m in self.performance_metrics.values()) / 
                len(self.performance_metrics) if self.performance_metrics else 0
            ),
            'slowest_operations': sorted(
                [(m.operation, m.duration_ms) for m in self.performance_metrics.values()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
        
        return summary
    
    def finalize_session(self):
        """Finalize logging session and print summary."""
        summary = self.get_session_summary()
        
        self.info(
            f"📊 Session completed",
            operation="session_end",
            **summary
        )
        
        if summary['operations_count'] > 0:
            self.info(f"   • Total operations: {summary['operations_count']}")
            self.info(f"   • Session duration: {summary['session_duration_seconds']:.2f}s")
            self.info(f"   • Average operation time: {summary['average_operation_time_ms']:.2f}ms")
            
            if summary['slowest_operations']:
                self.info("   • Slowest operations:")
                for op, duration in summary['slowest_operations']:
                    self.info(f"     - {op}: {duration:.2f}ms")


class ColoredFormatter(logging.Formatter):
    """Colored console formatter."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'        # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Color the level name
        record.levelname = f"{log_color}{record.levelname}{reset_color}"
        
        return super().format(record)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'session_id'):
            log_entry['session_id'] = record.session_id
        if hasattr(record, 'operation'):
            log_entry['operation'] = record.operation
        if hasattr(record, 'session_elapsed'):
            log_entry['session_elapsed'] = record.session_elapsed
        
        # Add any additional fields from extra parameter
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'message']:
                if not key.startswith('_'):
                    log_entry[key] = value
        
        return json.dumps(log_entry)


def performance_monitor(operation_name: str = None):
    """Decorator for automatic performance monitoring."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from first argument if it's a class with logger
            logger = None
            if args and hasattr(args[0], 'logger') and isinstance(args[0].logger, PhotonicLogger):
                logger = args[0].logger
            else:
                # Use global logger
                logger = get_global_logger()
            
            with logger.performance_context(operation_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global logger instance
_global_logger: Optional[PhotonicLogger] = None


def get_global_logger() -> PhotonicLogger:
    """Get or create global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = PhotonicLogger()
    return _global_logger


def setup_logging(level: Union[str, int] = logging.INFO,
                 log_file: Optional[str] = None,
                 json_logging: bool = False,
                 performance_logging: bool = True) -> PhotonicLogger:
    """Setup global logging configuration."""
    global _global_logger
    _global_logger = PhotonicLogger(
        level=level,
        log_file=log_file,
        json_logging=json_logging,
        performance_logging=performance_logging
    )
    return _global_logger


def finalize_logging():
    """Finalize global logging session."""
    if _global_logger:
        _global_logger.finalize_session()