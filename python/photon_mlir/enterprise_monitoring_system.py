"""
Enterprise-Grade Monitoring System for Photonic Computing
Generation 2: Real-time monitoring with AI-powered anomaly detection
"""

import time
import threading
import queue
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics
from datetime import datetime, timedelta
import uuid

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

from .logging_config import get_global_logger
from .validation import ValidationResult


class MetricType(Enum):
    """Types of metrics to monitor."""
    PERFORMANCE = "performance"
    THERMAL = "thermal"
    OPTICAL = "optical"
    QUANTUM = "quantum"
    SYSTEM = "system"
    SECURITY = "security"
    ERROR = "error"
    COMPILATION = "compilation"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class Metric:
    """Represents a single metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: float
    labels: Dict[str, str]
    metric_type: MetricType
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'labels': self.labels,
            'metric_type': self.metric_type.value
        }


@dataclass
class Alert:
    """Represents a monitoring alert."""
    id: str
    title: str
    description: str
    severity: AlertSeverity
    status: AlertStatus
    created_at: float
    updated_at: float
    metric_name: str
    threshold_value: float
    current_value: float
    labels: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MetricCollector:
    """Base class for metric collectors."""
    
    def __init__(self, name: str, collection_interval: float = 1.0):
        self.name = name
        self.collection_interval = collection_interval
        self.logger = get_global_logger()
        self._running = False
        self._thread = None
        self._metrics_queue = queue.Queue(maxsize=1000)
        
    def start(self):
        """Start metric collection."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        self.logger.info(f"Started metric collector: {self.name}")
    
    def stop(self):
        """Stop metric collection."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        self.logger.info(f"Stopped metric collector: {self.name}")
    
    def get_metrics(self) -> List[Metric]:
        """Get collected metrics."""
        metrics = []
        try:
            while True:
                metric = self._metrics_queue.get_nowait()
                metrics.append(metric)
        except queue.Empty:
            pass
        return metrics
    
    def _collection_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                metrics = self._collect_metrics()
                for metric in metrics:
                    try:
                        self._metrics_queue.put_nowait(metric)
                    except queue.Full:
                        # Remove oldest metric and add new one
                        try:
                            self._metrics_queue.get_nowait()
                            self._metrics_queue.put_nowait(metric)
                        except queue.Empty:
                            pass
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in metric collection loop for {self.name}: {e}")
                time.sleep(self.collection_interval)
    
    def _collect_metrics(self) -> List[Metric]:
        """Override this method to implement specific metric collection."""
        return []


class SystemMetricsCollector(MetricCollector):
    """Collects system resource metrics."""
    
    def __init__(self, collection_interval: float = 5.0):
        super().__init__("SystemMetrics", collection_interval)
        
    def _collect_metrics(self) -> List[Metric]:
        """Collect system metrics."""
        metrics = []
        timestamp = time.time()
        
        if _PSUTIL_AVAILABLE:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=None)
                metrics.append(Metric(
                    name="system_cpu_usage",
                    value=cpu_percent,
                    unit="percent",
                    timestamp=timestamp,
                    labels={"collector": "system"},
                    metric_type=MetricType.SYSTEM
                ))
                
                # Memory metrics
                memory = psutil.virtual_memory()
                metrics.append(Metric(
                    name="system_memory_usage",
                    value=memory.percent,
                    unit="percent",
                    timestamp=timestamp,
                    labels={"collector": "system"},
                    metric_type=MetricType.SYSTEM
                ))
                
                metrics.append(Metric(
                    name="system_memory_available",
                    value=memory.available / (1024**3),  # GB
                    unit="GB",
                    timestamp=timestamp,
                    labels={"collector": "system"},
                    metric_type=MetricType.SYSTEM
                ))
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                metrics.append(Metric(
                    name="system_disk_usage",
                    value=(disk.used / disk.total) * 100,
                    unit="percent",
                    timestamp=timestamp,
                    labels={"collector": "system", "mount": "/"},
                    metric_type=MetricType.SYSTEM
                ))
                
                # Network metrics
                network = psutil.net_io_counters()
                metrics.append(Metric(
                    name="system_network_bytes_sent",
                    value=network.bytes_sent,
                    unit="bytes",
                    timestamp=timestamp,
                    labels={"collector": "system", "direction": "sent"},
                    metric_type=MetricType.SYSTEM
                ))
                
                metrics.append(Metric(
                    name="system_network_bytes_recv",
                    value=network.bytes_recv,
                    unit="bytes",
                    timestamp=timestamp,
                    labels={"collector": "system", "direction": "received"},
                    metric_type=MetricType.SYSTEM
                ))
                
            except Exception as e:
                self.logger.error(f"Failed to collect system metrics: {e}")
        else:
            # Fallback metrics without psutil
            import os
            try:
                load_avg = os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
                metrics.append(Metric(
                    name="system_load_average",
                    value=load_avg,
                    unit="count",
                    timestamp=timestamp,
                    labels={"collector": "system"},
                    metric_type=MetricType.SYSTEM
                ))
            except:
                pass
        
        return metrics


class PhotonicMetricsCollector(MetricCollector):
    """Collects photonic-specific metrics."""
    
    def __init__(self, collection_interval: float = 2.0):
        super().__init__("PhotonicMetrics", collection_interval)
        self.thermal_readings = deque(maxlen=100)
        self.optical_power_readings = deque(maxlen=100)
        self.phase_drift_readings = deque(maxlen=100)
        
    def _collect_metrics(self) -> List[Metric]:
        """Collect photonic system metrics."""
        metrics = []
        timestamp = time.time()
        
        # Simulate photonic metrics (in real system, these would come from hardware)
        
        # Thermal metrics
        base_temp = 22.0  # Celsius
        thermal_variation = 3.0 * (0.5 - time.time() % 60 / 60.0)  # Simulated variation
        current_temp = base_temp + thermal_variation + (time.time() % 10 - 5) * 0.1
        
        self.thermal_readings.append(current_temp)
        
        metrics.append(Metric(
            name="photonic_temperature",
            value=current_temp,
            unit="celsius",
            timestamp=timestamp,
            labels={"collector": "photonic", "sensor": "main"},
            metric_type=MetricType.THERMAL
        ))
        
        # Optical power metrics
        base_power = 10.0  # mW
        power_variation = 1.0 * np.sin(time.time() * 0.1) if _NUMPY_AVAILABLE else 0.0
        current_power = base_power + power_variation
        
        self.optical_power_readings.append(current_power)
        
        metrics.append(Metric(
            name="photonic_optical_power",
            value=current_power,
            unit="mW",
            timestamp=timestamp,
            labels={"collector": "photonic", "wavelength": "1550nm"},
            metric_type=MetricType.OPTICAL
        ))
        
        # Phase drift metrics
        phase_drift = 0.05 * (time.time() % 30 / 30.0)  # Simulated phase drift
        self.phase_drift_readings.append(phase_drift)
        
        metrics.append(Metric(
            name="photonic_phase_drift",
            value=phase_drift,
            unit="radians",
            timestamp=timestamp,
            labels={"collector": "photonic", "channel": "main"},
            metric_type=MetricType.QUANTUM
        ))
        
        # Compilation metrics
        compilation_rate = len(self.thermal_readings) * 0.1  # Simulated
        metrics.append(Metric(
            name="photonic_compilation_rate",
            value=compilation_rate,
            unit="ops/sec",
            timestamp=timestamp,
            labels={"collector": "photonic", "type": "compilation"},
            metric_type=MetricType.COMPILATION
        ))
        
        # Error rate metrics
        error_rate = max(0, (current_temp - base_temp) / 10.0)  # Errors increase with temperature
        metrics.append(Metric(
            name="photonic_error_rate",
            value=error_rate,
            unit="errors/sec",
            timestamp=timestamp,
            labels={"collector": "photonic", "type": "thermal_errors"},
            metric_type=MetricType.ERROR
        ))
        
        return metrics


class PerformanceMetricsCollector(MetricCollector):
    """Collects performance and latency metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        super().__init__("PerformanceMetrics", collection_interval)
        self.latency_samples = deque(maxlen=1000)
        self.throughput_samples = deque(maxlen=100)
        
    def _collect_metrics(self) -> List[Metric]:
        """Collect performance metrics."""
        metrics = []
        timestamp = time.time()
        
        # Simulate performance metrics
        base_latency = 50.0  # microseconds
        latency_variation = 20.0 * (0.5 - (time.time() % 120) / 120.0)
        current_latency = base_latency + latency_variation + (time.time() % 5) * 2
        
        self.latency_samples.append(current_latency)
        
        metrics.append(Metric(
            name="performance_latency",
            value=current_latency,
            unit="microseconds",
            timestamp=timestamp,
            labels={"collector": "performance", "operation": "inference"},
            metric_type=MetricType.PERFORMANCE
        ))
        
        # Throughput metrics
        base_throughput = 1000.0  # ops/sec
        throughput_variation = 200.0 * np.sin(time.time() * 0.05) if _NUMPY_AVAILABLE else 0.0
        current_throughput = base_throughput + throughput_variation
        
        self.throughput_samples.append(current_throughput)
        
        metrics.append(Metric(
            name="performance_throughput",
            value=current_throughput,
            unit="ops/sec",
            timestamp=timestamp,
            labels={"collector": "performance", "operation": "matrix_multiply"},
            metric_type=MetricType.PERFORMANCE
        ))
        
        # Queue depth metrics
        queue_depth = max(0, int(10 * (current_latency - base_latency) / 20.0))
        metrics.append(Metric(
            name="performance_queue_depth",
            value=queue_depth,
            unit="count",
            timestamp=timestamp,
            labels={"collector": "performance", "queue": "compilation"},
            metric_type=MetricType.PERFORMANCE
        ))
        
        # Calculate statistics if we have enough samples
        if len(self.latency_samples) >= 10:
            avg_latency = statistics.mean(list(self.latency_samples)[-10:])
            p95_latency = np.percentile(list(self.latency_samples)[-100:], 95) if _NUMPY_AVAILABLE else avg_latency * 1.5
            
            metrics.append(Metric(
                name="performance_latency_p95",
                value=p95_latency,
                unit="microseconds",
                timestamp=timestamp,
                labels={"collector": "performance", "percentile": "95"},
                metric_type=MetricType.PERFORMANCE
            ))
        
        return metrics


class AnomalyDetector:
    """AI-powered anomaly detection for metrics."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.metric_histories = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        self.logger = get_global_logger()
        
    def add_metric(self, metric: Metric) -> Optional[Alert]:
        """Add metric and detect anomalies."""
        metric_key = f"{metric.name}_{metric.labels.get('collector', '')}"
        self.metric_histories[metric_key].append(metric.value)
        
        # Need sufficient history for anomaly detection
        if len(self.metric_histories[metric_key]) < 20:
            return None
        
        return self._detect_anomaly(metric, metric_key)
    
    def _detect_anomaly(self, metric: Metric, metric_key: str) -> Optional[Alert]:
        """Detect anomaly in metric using statistical methods."""
        history = list(self.metric_histories[metric_key])
        
        try:
            # Calculate baseline statistics
            mean_value = statistics.mean(history[:-1])  # Exclude current value
            if len(history) > 10:
                std_dev = statistics.stdev(history[:-1])
            else:
                std_dev = abs(mean_value) * 0.1  # Fallback
            
            # Z-score based anomaly detection
            if std_dev > 0:
                z_score = abs(metric.value - mean_value) / std_dev
                
                if z_score > self.sensitivity:
                    # Determine severity based on Z-score
                    if z_score > 4.0:
                        severity = AlertSeverity.CRITICAL
                    elif z_score > 3.0:
                        severity = AlertSeverity.ERROR
                    else:
                        severity = AlertSeverity.WARNING
                    
                    alert = Alert(
                        id=str(uuid.uuid4()),
                        title=f"Anomaly detected in {metric.name}",
                        description=f"Metric {metric.name} value {metric.value:.2f} {metric.unit} "
                                  f"deviates significantly from baseline {mean_value:.2f} "
                                  f"(z-score: {z_score:.2f})",
                        severity=severity,
                        status=AlertStatus.ACTIVE,
                        created_at=metric.timestamp,
                        updated_at=metric.timestamp,
                        metric_name=metric.name,
                        threshold_value=mean_value + self.sensitivity * std_dev,
                        current_value=metric.value,
                        labels=metric.labels.copy()
                    )
                    
                    self.logger.warning(f"Anomaly detected: {alert.title}")
                    return alert
                    
        except Exception as e:
            self.logger.error(f"Error in anomaly detection for {metric_key}: {e}")
        
        return None
    
    def update_baseline(self, metric_key: str):
        """Update baseline for a specific metric."""
        if metric_key in self.metric_histories:
            history = list(self.metric_histories[metric_key])
            if len(history) >= 50:
                self.baselines[metric_key] = {
                    'mean': statistics.mean(history),
                    'std': statistics.stdev(history) if len(history) > 1 else 0.0,
                    'updated_at': time.time()
                }
                self.logger.info(f"Updated baseline for {metric_key}")


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, max_alerts: int = 1000):
        self.max_alerts = max_alerts
        self.alerts = {}
        self.alert_history = deque(maxlen=max_alerts)
        self.alert_handlers = []
        self.logger = get_global_logger()
        self._lock = threading.Lock()
        
    def add_alert(self, alert: Alert):
        """Add new alert."""
        with self._lock:
            self.alerts[alert.id] = alert
            self.alert_history.append(alert)
            
            # Trigger alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert)
                except Exception as e:
                    self.logger.error(f"Alert handler failed: {e}")
            
            self.logger.info(f"Alert created: {alert.title} ({alert.severity.value})")
    
    def acknowledge_alert(self, alert_id: str, user: str = "system") -> bool:
        """Acknowledge an alert."""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.updated_at = time.time()
                alert.labels['acknowledged_by'] = user
                self.logger.info(f"Alert {alert_id} acknowledged by {user}")
                return True
        return False
    
    def resolve_alert(self, alert_id: str, user: str = "system") -> bool:
        """Resolve an alert."""
        with self._lock:
            if alert_id in self.alerts:
                alert = self.alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.updated_at = time.time()
                alert.labels['resolved_by'] = user
                self.logger.info(f"Alert {alert_id} resolved by {user}")
                return True
        return False
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [alert for alert in self.alerts.values() 
                   if alert.status == AlertStatus.ACTIVE]
    
    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """Get alerts by severity level."""
        with self._lock:
            return [alert for alert in self.alerts.values() 
                   if alert.severity == severity]
    
    def add_alert_handler(self, handler: Callable[[Alert], None]):
        """Add alert notification handler."""
        self.alert_handlers.append(handler)
    
    def cleanup_old_alerts(self, max_age_hours: int = 24):
        """Clean up old resolved alerts."""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self._lock:
            alerts_to_remove = []
            for alert_id, alert in self.alerts.items():
                if (alert.status == AlertStatus.RESOLVED and 
                    alert.updated_at < cutoff_time):
                    alerts_to_remove.append(alert_id)
            
            for alert_id in alerts_to_remove:
                del self.alerts[alert_id]
            
            if alerts_to_remove:
                self.logger.info(f"Cleaned up {len(alerts_to_remove)} old alerts")


class EnterpriseMonitoringSystem:
    """Main enterprise monitoring system."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_global_logger()
        
        # Initialize components
        self.collectors = []
        self.anomaly_detector = AnomalyDetector(
            sensitivity=self.config.get('anomaly_sensitivity', 2.0)
        )
        self.alert_manager = AlertManager(
            max_alerts=self.config.get('max_alerts', 1000)
        )
        
        # Metrics storage
        self.metrics_buffer = deque(maxlen=10000)
        self._metrics_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'metrics_collected': 0,
            'alerts_generated': 0,
            'anomalies_detected': 0,
            'start_time': time.time()
        }
        
        # Set up default alert handlers
        self.alert_manager.add_alert_handler(self._default_alert_handler)
        
        # Background processing
        self._running = False
        self._processing_thread = None
        
    def start(self):
        """Start the monitoring system."""
        if self._running:
            return
        
        self.logger.info("Starting Enterprise Monitoring System")
        
        # Start default collectors
        self._start_default_collectors()
        
        # Start background processing
        self._running = True
        self._processing_thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._processing_thread.start()
        
        self.logger.info("Enterprise Monitoring System started")
    
    def stop(self):
        """Stop the monitoring system."""
        self.logger.info("Stopping Enterprise Monitoring System")
        
        # Stop collectors
        for collector in self.collectors:
            collector.stop()
        
        # Stop background processing
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=10.0)
        
        self.logger.info("Enterprise Monitoring System stopped")
    
    def _start_default_collectors(self):
        """Start default metric collectors."""
        # System metrics
        system_collector = SystemMetricsCollector(
            collection_interval=self.config.get('system_interval', 5.0)
        )
        self.add_collector(system_collector)
        
        # Photonic metrics
        photonic_collector = PhotonicMetricsCollector(
            collection_interval=self.config.get('photonic_interval', 2.0)
        )
        self.add_collector(photonic_collector)
        
        # Performance metrics
        performance_collector = PerformanceMetricsCollector(
            collection_interval=self.config.get('performance_interval', 1.0)
        )
        self.add_collector(performance_collector)
    
    def add_collector(self, collector: MetricCollector):
        """Add a metric collector."""
        self.collectors.append(collector)
        collector.start()
        self.logger.info(f"Added metric collector: {collector.name}")
    
    def _processing_loop(self):
        """Main processing loop for metrics and alerts."""
        while self._running:
            try:
                # Collect metrics from all collectors
                all_metrics = []
                for collector in self.collectors:
                    try:
                        metrics = collector.get_metrics()
                        all_metrics.extend(metrics)
                    except Exception as e:
                        self.logger.error(f"Error collecting metrics from {collector.name}: {e}")
                
                # Process metrics
                for metric in all_metrics:
                    self._process_metric(metric)
                
                # Clean up old alerts periodically
                if int(time.time()) % 3600 == 0:  # Every hour
                    self.alert_manager.cleanup_old_alerts()
                
                time.sleep(1.0)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Error in monitoring processing loop: {e}")
                time.sleep(5.0)
    
    def _process_metric(self, metric: Metric):
        """Process a single metric."""
        # Store metric
        with self._metrics_lock:
            self.metrics_buffer.append(metric)
        
        self.stats['metrics_collected'] += 1
        
        # Check for anomalies
        alert = self.anomaly_detector.add_metric(metric)
        if alert:
            self.alert_manager.add_alert(alert)
            self.stats['alerts_generated'] += 1
            self.stats['anomalies_detected'] += 1
    
    def _default_alert_handler(self, alert: Alert):
        """Default alert handler that logs alerts."""
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.INFO)
        
        self.logger.log(log_level, f"ALERT [{alert.severity.value.upper()}]: {alert.title}")
        self.logger.log(log_level, f"       {alert.description}")
    
    def get_metrics(self, metric_name: str = None, 
                   since: float = None, limit: int = 100) -> List[Metric]:
        """Get stored metrics with optional filtering."""
        with self._metrics_lock:
            metrics = list(self.metrics_buffer)
        
        # Filter by name
        if metric_name:
            metrics = [m for m in metrics if m.name == metric_name]
        
        # Filter by time
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        # Sort by timestamp (newest first)
        metrics.sort(key=lambda m: m.timestamp, reverse=True)
        
        return metrics[:limit]
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        active_alerts = self.alert_manager.get_active_alerts()
        critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
        error_alerts = [a for a in active_alerts if a.severity == AlertSeverity.ERROR]
        
        # Determine overall health
        if critical_alerts:
            health_status = "CRITICAL"
        elif error_alerts:
            health_status = "DEGRADED"
        elif active_alerts:
            health_status = "WARNING"
        else:
            health_status = "HEALTHY"
        
        uptime = time.time() - self.stats['start_time']
        
        return {
            'status': health_status,
            'uptime_seconds': uptime,
            'active_alerts': len(active_alerts),
            'critical_alerts': len(critical_alerts),
            'error_alerts': len(error_alerts),
            'metrics_collected': self.stats['metrics_collected'],
            'anomalies_detected': self.stats['anomalies_detected'],
            'collectors_running': len([c for c in self.collectors if c._running]),
            'total_collectors': len(self.collectors)
        }
    
    def export_metrics(self, format: str = "json", 
                      since: float = None) -> str:
        """Export metrics in specified format."""
        metrics = self.get_metrics(since=since, limit=10000)
        
        if format.lower() == "json":
            return json.dumps([m.to_dict() for m in metrics], indent=2)
        elif format.lower() == "prometheus":
            return self._export_prometheus_format(metrics)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_prometheus_format(self, metrics: List[Metric]) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for metric in metrics:
            metrics_by_name[metric.name].append(metric)
        
        for metric_name, metric_list in metrics_by_name.items():
            # Add metric help and type
            lines.append(f"# HELP {metric_name} {metric_name} metric")
            lines.append(f"# TYPE {metric_name} gauge")
            
            # Add metric values
            for metric in metric_list[-1:]:  # Only latest value
                labels = ",".join([f'{k}="{v}"' for k, v in metric.labels.items()])
                if labels:
                    lines.append(f"{metric_name}{{{labels}}} {metric.value} {int(metric.timestamp * 1000)}")
                else:
                    lines.append(f"{metric_name} {metric.value} {int(metric.timestamp * 1000)}")
            
            lines.append("")  # Empty line between metrics
        
        return "\n".join(lines)


# Convenience functions
def create_monitoring_system(config: Dict[str, Any] = None) -> EnterpriseMonitoringSystem:
    """Create and configure enterprise monitoring system."""
    return EnterpriseMonitoringSystem(config)


def start_monitoring(config: Dict[str, Any] = None) -> EnterpriseMonitoringSystem:
    """Start monitoring system with default configuration."""
    monitoring = create_monitoring_system(config)
    monitoring.start()
    return monitoring
