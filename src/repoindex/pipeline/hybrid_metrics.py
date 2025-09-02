"""
Comprehensive Performance Metrics and Monitoring for Hybrid Pipeline.

This module provides detailed monitoring, metrics collection, and performance
analysis for the hybrid Mimir-Lens pipeline operations.
"""

import asyncio
import psutil
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from enum import Enum
import json
from pathlib import Path
import threading

from ..util.log import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMING = "timing"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'value': self.value,
            'labels': self.labels
        }


@dataclass
class PerformanceSnapshot:
    """Snapshot of system performance at a point in time."""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    process_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_io_read_mb': self.disk_io_read_mb,
            'disk_io_write_mb': self.disk_io_write_mb,
            'network_sent_mb': self.network_sent_mb,
            'network_recv_mb': self.network_recv_mb,
            'process_count': self.process_count
        }


@dataclass
class PipelineMetrics:
    """Comprehensive metrics for pipeline operations."""
    start_time: datetime
    end_time: Optional[datetime] = None
    
    # Stage timings (in milliseconds)
    discovery_time_ms: float = 0.0
    embedding_time_ms: float = 0.0
    bundling_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Throughput metrics
    files_processed: int = 0
    chunks_processed: int = 0
    embeddings_generated: int = 0
    
    # Quality metrics
    lens_success_rate: float = 0.0
    mimir_success_rate: float = 0.0
    cache_hit_rate: float = 0.0
    synthesis_quality_score: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    average_cpu_percent: float = 0.0
    disk_io_total_mb: float = 0.0
    network_io_total_mb: float = 0.0
    
    # Parallel processing metrics
    max_concurrent_tasks: int = 0
    average_task_queue_size: float = 0.0
    task_failure_rate: float = 0.0
    
    # Lens-specific metrics
    lens_requests_sent: int = 0
    lens_average_response_time_ms: float = 0.0
    lens_error_count: int = 0
    lens_circuit_breaker_trips: int = 0
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.start_time:
            self.total_time_ms = (self.end_time - self.start_time).total_seconds() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'discovery_time_ms': self.discovery_time_ms,
            'embedding_time_ms': self.embedding_time_ms,
            'bundling_time_ms': self.bundling_time_ms,
            'total_time_ms': self.total_time_ms,
            'files_processed': self.files_processed,
            'chunks_processed': self.chunks_processed,
            'embeddings_generated': self.embeddings_generated,
            'lens_success_rate': self.lens_success_rate,
            'mimir_success_rate': self.mimir_success_rate,
            'cache_hit_rate': self.cache_hit_rate,
            'synthesis_quality_score': self.synthesis_quality_score,
            'peak_memory_mb': self.peak_memory_mb,
            'average_cpu_percent': self.average_cpu_percent,
            'disk_io_total_mb': self.disk_io_total_mb,
            'network_io_total_mb': self.network_io_total_mb,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'average_task_queue_size': self.average_task_queue_size,
            'task_failure_rate': self.task_failure_rate,
            'lens_requests_sent': self.lens_requests_sent,
            'lens_average_response_time_ms': self.lens_average_response_time_ms,
            'lens_error_count': self.lens_error_count,
            'lens_circuit_breaker_trips': self.lens_circuit_breaker_trips
        }


class MetricsCollector:
    """
    Advanced metrics collection and monitoring system for hybrid pipeline.
    
    Features:
    - Real-time system resource monitoring
    - Pipeline performance tracking
    - Lens integration health monitoring
    - Alert generation based on thresholds
    - Historical metrics storage and analysis
    - Performance trend analysis
    """
    
    def __init__(
        self,
        collection_interval_seconds: float = 1.0,
        history_retention_hours: int = 24,
        enable_alerts: bool = True,
        metrics_storage_path: Optional[Path] = None
    ):
        """Initialize metrics collector."""
        self.collection_interval = collection_interval_seconds
        self.history_retention = timedelta(hours=history_retention_hours)
        self.enable_alerts = enable_alerts
        self.metrics_storage_path = metrics_storage_path
        
        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.performance_snapshots: deque = deque(maxlen=1000)
        self.pipeline_metrics: List[PipelineMetrics] = []
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._lock = threading.Lock()
        
        # Alert thresholds
        self.alert_thresholds = {
            'cpu_percent': {'warning': 80.0, 'critical': 95.0},
            'memory_percent': {'warning': 85.0, 'critical': 95.0},
            'disk_io_mb_per_sec': {'warning': 100.0, 'critical': 500.0},
            'lens_error_rate': {'warning': 0.05, 'critical': 0.15},
            'lens_response_time_ms': {'warning': 5000.0, 'critical': 10000.0},
            'task_failure_rate': {'warning': 0.1, 'critical': 0.25}
        }
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[str, AlertLevel, Dict[str, Any]], None]] = []
        
        logger.info(f"MetricsCollector initialized with {collection_interval_seconds}s interval")
    
    async def start_monitoring(self) -> None:
        """Start background metrics collection."""
        if self._monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self._monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started metrics monitoring")
    
    async def stop_monitoring(self) -> None:
        """Stop background metrics collection."""
        if not self._monitoring_active:
            return
        
        self._monitoring_active = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        # Save metrics to disk if path is configured
        if self.metrics_storage_path:
            await self._save_metrics_to_disk()
        
        logger.info("Stopped metrics monitoring")
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting metrics monitoring loop")
        
        try:
            while self._monitoring_active:
                # Collect system metrics
                snapshot = await self._collect_system_snapshot()
                self.performance_snapshots.append(snapshot)
                
                # Check alert thresholds
                if self.enable_alerts:
                    await self._check_alerts(snapshot)
                
                # Clean old metrics
                await self._cleanup_old_metrics()
                
                # Wait for next collection
                await asyncio.sleep(self.collection_interval)
                
        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Monitoring loop error: {e}")
    
    async def _collect_system_snapshot(self) -> PerformanceSnapshot:
        """Collect current system performance snapshot."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0.0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0.0
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0.0
            network_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0.0
            
            # Process count
            process_count = len(psutil.pids())
            
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=network_sent_mb,
                network_recv_mb=network_recv_mb,
                process_count=process_count
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system snapshot: {e}")
            return PerformanceSnapshot(
                timestamp=datetime.utcnow(),
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_used_mb=0.0,
                disk_io_read_mb=0.0,
                disk_io_write_mb=0.0,
                network_sent_mb=0.0,
                network_recv_mb=0.0,
                process_count=0
            )
    
    def record_metric(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a custom metric."""
        with self._lock:
            point = MetricPoint(
                timestamp=datetime.utcnow(),
                value=value,
                labels=labels or {}
            )
            self.metrics[name].append(point)
    
    def start_pipeline_metrics(self) -> PipelineMetrics:
        """Start collecting metrics for a pipeline run."""
        metrics = PipelineMetrics(start_time=datetime.utcnow())
        return metrics
    
    def finish_pipeline_metrics(self, metrics: PipelineMetrics) -> None:
        """Finish and store pipeline metrics."""
        metrics.finalize()
        self.pipeline_metrics.append(metrics)
        
        # Trim old pipeline metrics
        if len(self.pipeline_metrics) > 100:
            self.pipeline_metrics = self.pipeline_metrics[-100:]
        
        logger.info(f"Pipeline completed in {metrics.total_time_ms:.2f}ms")
    
    def record_lens_request(
        self,
        response_time_ms: float,
        success: bool,
        operation: str
    ) -> None:
        """Record Lens request metrics."""
        self.record_metric(
            'lens_response_time_ms',
            response_time_ms,
            MetricType.TIMING,
            {'operation': operation, 'success': str(success)}
        )
        
        self.record_metric(
            'lens_request_count',
            1.0,
            MetricType.COUNTER,
            {'operation': operation, 'success': str(success)}
        )
    
    def record_parallel_task(
        self,
        task_type: str,
        execution_time_ms: float,
        success: bool,
        queue_size: int
    ) -> None:
        """Record parallel task execution metrics."""
        self.record_metric(
            'task_execution_time_ms',
            execution_time_ms,
            MetricType.TIMING,
            {'task_type': task_type, 'success': str(success)}
        )
        
        self.record_metric(
            'task_queue_size',
            float(queue_size),
            MetricType.GAUGE,
            {'task_type': task_type}
        )
    
    async def _check_alerts(self, snapshot: PerformanceSnapshot) -> None:
        """Check metrics against alert thresholds."""
        alerts_to_send = []
        
        # Check CPU usage
        if snapshot.cpu_percent > self.alert_thresholds['cpu_percent']['critical']:
            alerts_to_send.append(('High CPU usage', AlertLevel.CRITICAL, {
                'cpu_percent': snapshot.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']['critical']
            }))
        elif snapshot.cpu_percent > self.alert_thresholds['cpu_percent']['warning']:
            alerts_to_send.append(('Elevated CPU usage', AlertLevel.WARNING, {
                'cpu_percent': snapshot.cpu_percent,
                'threshold': self.alert_thresholds['cpu_percent']['warning']
            }))
        
        # Check memory usage
        if snapshot.memory_percent > self.alert_thresholds['memory_percent']['critical']:
            alerts_to_send.append(('High memory usage', AlertLevel.CRITICAL, {
                'memory_percent': snapshot.memory_percent,
                'memory_used_mb': snapshot.memory_used_mb,
                'threshold': self.alert_thresholds['memory_percent']['critical']
            }))
        elif snapshot.memory_percent > self.alert_thresholds['memory_percent']['warning']:
            alerts_to_send.append(('Elevated memory usage', AlertLevel.WARNING, {
                'memory_percent': snapshot.memory_percent,
                'memory_used_mb': snapshot.memory_used_mb,
                'threshold': self.alert_thresholds['memory_percent']['warning']
            }))
        
        # Check Lens error rates if we have recent data
        lens_error_rate = await self._calculate_lens_error_rate()
        if lens_error_rate is not None:
            if lens_error_rate > self.alert_thresholds['lens_error_rate']['critical']:
                alerts_to_send.append(('High Lens error rate', AlertLevel.CRITICAL, {
                    'error_rate': lens_error_rate,
                    'threshold': self.alert_thresholds['lens_error_rate']['critical']
                }))
            elif lens_error_rate > self.alert_thresholds['lens_error_rate']['warning']:
                alerts_to_send.append(('Elevated Lens error rate', AlertLevel.WARNING, {
                    'error_rate': lens_error_rate,
                    'threshold': self.alert_thresholds['lens_error_rate']['warning']
                }))
        
        # Send alerts
        for message, level, data in alerts_to_send:
            await self._send_alert(message, level, data)
    
    async def _calculate_lens_error_rate(self) -> Optional[float]:
        """Calculate recent Lens error rate."""
        if 'lens_request_count' not in self.metrics:
            return None
        
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=5)  # 5-minute window
        
        total_requests = 0
        error_requests = 0
        
        for point in self.metrics['lens_request_count']:
            if point.timestamp >= window_start:
                total_requests += point.value
                if point.labels.get('success') == 'False':
                    error_requests += point.value
        
        if total_requests == 0:
            return None
        
        return error_requests / total_requests
    
    async def _send_alert(
        self,
        message: str,
        level: AlertLevel,
        data: Dict[str, Any]
    ) -> None:
        """Send alert to registered callbacks."""
        logger.log(
            logger.level if level == AlertLevel.INFO else 
            40 if level in [AlertLevel.ERROR, AlertLevel.CRITICAL] else 30,
            f"ALERT [{level.value.upper()}]: {message} - {data}"
        )
        
        for callback in self.alert_callbacks:
            try:
                callback(message, level, data)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(
        self,
        callback: Callable[[str, AlertLevel, Dict[str, Any]], None]
    ) -> None:
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    async def _cleanup_old_metrics(self) -> None:
        """Remove old metrics beyond retention period."""
        cutoff_time = datetime.utcnow() - self.history_retention
        
        with self._lock:
            for metric_name, points in self.metrics.items():
                # Remove points older than cutoff
                while points and points[0].timestamp < cutoff_time:
                    points.popleft()
            
            # Clean old performance snapshots
            while (self.performance_snapshots and 
                   self.performance_snapshots[0].timestamp < cutoff_time):
                self.performance_snapshots.popleft()
    
    async def _save_metrics_to_disk(self) -> None:
        """Save collected metrics to disk."""
        if not self.metrics_storage_path:
            return
        
        try:
            self.metrics_storage_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            data = {
                'collection_timestamp': datetime.utcnow().isoformat(),
                'metrics': {},
                'performance_snapshots': [
                    snapshot.to_dict() for snapshot in self.performance_snapshots
                ],
                'pipeline_metrics': [
                    metrics.to_dict() for metrics in self.pipeline_metrics
                ]
            }
            
            # Convert metrics to serializable format
            for name, points in self.metrics.items():
                data['metrics'][name] = [point.to_dict() for point in points]
            
            # Write to disk
            with open(self.metrics_storage_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved metrics to {self.metrics_storage_path}")
            
        except Exception as e:
            logger.error(f"Failed to save metrics to disk: {e}")
    
    async def load_metrics_from_disk(self) -> bool:
        """Load previously saved metrics from disk."""
        if not self.metrics_storage_path or not self.metrics_storage_path.exists():
            return False
        
        try:
            with open(self.metrics_storage_path, 'r') as f:
                data = json.load(f)
            
            # Load metrics
            for name, points_data in data.get('metrics', {}).items():
                points = deque(maxlen=10000)
                for point_data in points_data:
                    point = MetricPoint(
                        timestamp=datetime.fromisoformat(point_data['timestamp']),
                        value=point_data['value'],
                        labels=point_data['labels']
                    )
                    points.append(point)
                self.metrics[name] = points
            
            # Load performance snapshots
            for snapshot_data in data.get('performance_snapshots', []):
                snapshot = PerformanceSnapshot(
                    timestamp=datetime.fromisoformat(snapshot_data['timestamp']),
                    cpu_percent=snapshot_data['cpu_percent'],
                    memory_percent=snapshot_data['memory_percent'],
                    memory_used_mb=snapshot_data['memory_used_mb'],
                    disk_io_read_mb=snapshot_data['disk_io_read_mb'],
                    disk_io_write_mb=snapshot_data['disk_io_write_mb'],
                    network_sent_mb=snapshot_data['network_sent_mb'],
                    network_recv_mb=snapshot_data['network_recv_mb'],
                    process_count=snapshot_data['process_count']
                )
                self.performance_snapshots.append(snapshot)
            
            logger.info(f"Loaded metrics from {self.metrics_storage_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load metrics from disk: {e}")
            return False
    
    def get_metric_summary(self, metric_name: str, window_minutes: int = 5) -> Dict[str, float]:
        """Get summary statistics for a metric over a time window."""
        if metric_name not in self.metrics:
            return {}
        
        window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
        values = [
            point.value for point in self.metrics[metric_name]
            if point.timestamp >= window_start
        ]
        
        if not values:
            return {}
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
            'latest': values[-1] if values else 0.0
        }
    
    def get_pipeline_performance_summary(self) -> Dict[str, Any]:
        """Get summary of recent pipeline performance."""
        if not self.pipeline_metrics:
            return {}
        
        recent_metrics = self.pipeline_metrics[-10:]  # Last 10 runs
        
        return {
            'total_runs': len(self.pipeline_metrics),
            'recent_runs': len(recent_metrics),
            'average_total_time_ms': sum(m.total_time_ms for m in recent_metrics) / len(recent_metrics),
            'average_files_processed': sum(m.files_processed for m in recent_metrics) / len(recent_metrics),
            'average_lens_success_rate': sum(m.lens_success_rate for m in recent_metrics) / len(recent_metrics),
            'average_cache_hit_rate': sum(m.cache_hit_rate for m in recent_metrics) / len(recent_metrics),
            'latest_run': recent_metrics[-1].to_dict() if recent_metrics else None
        }
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0.0 to 1.0)."""
        if not self.performance_snapshots:
            return 0.5  # Unknown state
        
        latest = self.performance_snapshots[-1]
        score = 1.0
        
        # CPU penalty
        if latest.cpu_percent > 90:
            score -= 0.3
        elif latest.cpu_percent > 70:
            score -= 0.1
        
        # Memory penalty
        if latest.memory_percent > 90:
            score -= 0.3
        elif latest.memory_percent > 80:
            score -= 0.1
        
        # Lens health penalty
        lens_error_rate = asyncio.run(self._calculate_lens_error_rate())
        if lens_error_rate is not None:
            if lens_error_rate > 0.1:
                score -= 0.2
            elif lens_error_rate > 0.05:
                score -= 0.1
        
        return max(0.0, min(1.0, score))


# Global metrics collector instance
_global_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector."""
    global _global_metrics_collector
    
    if _global_metrics_collector is None:
        _global_metrics_collector = MetricsCollector()
    
    return _global_metrics_collector


async def initialize_metrics_monitoring(
    storage_path: Optional[Path] = None
) -> MetricsCollector:
    """Initialize and start metrics monitoring."""
    collector = get_metrics_collector()
    
    if storage_path:
        collector.metrics_storage_path = storage_path
        await collector.load_metrics_from_disk()
    
    await collector.start_monitoring()
    return collector


async def shutdown_metrics_monitoring():
    """Shutdown global metrics monitoring."""
    global _global_metrics_collector
    
    if _global_metrics_collector is not None:
        await _global_metrics_collector.stop_monitoring()
        _global_metrics_collector = None