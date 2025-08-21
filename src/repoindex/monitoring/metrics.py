"""
Prometheus metrics collection for Mimir Deep Code Research System.

Provides comprehensive application metrics including:
- Pipeline stage performance and success rates
- Vector search latency and accuracy 
- Symbol graph navigation metrics
- Resource utilization and performance
- Business metrics for code research workflows
"""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set
from pathlib import Path
import logging
import threading

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server, push_to_gateway
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class Counter:
        def __init__(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Histogram:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def time(self): return asynccontextmanager(lambda: iter([None]))()
        def labels(self, *args, **kwargs): return self
    
    class Gauge:
        def __init__(self, *args, **kwargs): pass
        def set(self, *args, **kwargs): pass
        def inc(self, *args, **kwargs): pass
        def dec(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Summary:
        def __init__(self, *args, **kwargs): pass
        def observe(self, *args, **kwargs): pass
        def labels(self, *args, **kwargs): return self
    
    class Info:
        def __init__(self, *args, **kwargs): pass
        def info(self, *args, **kwargs): pass

from ..util.logging_config import get_logger


logger = get_logger("monitoring.metrics")


@dataclass
class MetricDefinition:
    """Definition of a custom metric with metadata."""
    name: str
    help_text: str
    metric_type: str  # counter, histogram, gauge, summary, info
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histograms
    quantiles: Optional[Dict[float, float]] = None  # For summaries


class MetricsCollector:
    """
    Comprehensive metrics collector for Mimir system.
    
    Collects and exposes Prometheus metrics for monitoring:
    - Pipeline execution metrics
    - Search performance metrics  
    - Resource utilization metrics
    - Business logic metrics
    """
    
    def __init__(self, registry: Optional = None):
        """Initialize metrics collector with optional custom registry."""
        self.registry = registry if registry and PROMETHEUS_AVAILABLE else None
        self.enabled = PROMETHEUS_AVAILABLE
        self._http_server = None
        self._metrics_cache = {}
        self._custom_metrics = {}
        
        if not self.enabled:
            logger.warning("Prometheus client not available, metrics will be no-op")
            return
            
        self._init_core_metrics()
        logger.info("Metrics collector initialized", enabled=self.enabled)
    
    def _init_core_metrics(self):
        """Initialize core application metrics."""
        if not self.enabled:
            return
            
        registry = self.registry or None
        
        # =========================
        # PIPELINE METRICS
        # =========================
        
        self.pipeline_executions_total = Counter(
            'mimir_pipeline_executions_total',
            'Total number of pipeline executions',
            ['stage', 'status', 'language'],
            registry=registry
        )
        
        self.pipeline_duration_seconds = Histogram(
            'mimir_pipeline_duration_seconds', 
            'Pipeline execution duration in seconds',
            ['stage', 'language'],
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0],
            registry=registry
        )
        
        self.pipeline_files_processed = Histogram(
            'mimir_pipeline_files_processed',
            'Number of files processed by pipeline',
            ['stage', 'language'],
            buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000],
            registry=registry
        )
        
        self.pipeline_errors_total = Counter(
            'mimir_pipeline_errors_total',
            'Total number of pipeline errors',
            ['stage', 'error_type', 'severity'],
            registry=registry
        )
        
        self.pipeline_active_count = Gauge(
            'mimir_pipeline_active_count',
            'Number of currently active pipelines',
            registry=registry
        )
        
        self.pipeline_queue_length = Gauge(
            'mimir_pipeline_queue_length', 
            'Number of pipelines waiting in queue',
            registry=registry
        )
        
        # =========================
        # SEARCH METRICS  
        # =========================
        
        self.search_requests_total = Counter(
            'mimir_search_requests_total',
            'Total number of search requests',
            ['search_type', 'status'],
            registry=registry
        )
        
        self.search_duration_seconds = Histogram(
            'mimir_search_duration_seconds',
            'Search request duration in seconds', 
            ['search_type'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
            registry=registry
        )
        
        self.search_results_count = Histogram(
            'mimir_search_results_count',
            'Number of search results returned',
            ['search_type'],
            buckets=[0, 1, 5, 10, 20, 50, 100, 250, 500, 1000],
            registry=registry
        )
        
        self.vector_similarity_scores = Histogram(
            'mimir_vector_similarity_scores',
            'Distribution of vector similarity scores',
            buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            registry=registry
        )
        
        self.symbol_lookup_duration_seconds = Histogram(
            'mimir_symbol_lookup_duration_seconds',
            'Symbol graph lookup duration in seconds',
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
            registry=registry
        )
        
        # =========================
        # SYSTEM METRICS
        # =========================
        
        self.memory_usage_bytes = Gauge(
            'mimir_memory_usage_bytes',
            'Memory usage in bytes',
            ['type'],  # rss, vms, cache
            registry=registry
        )
        
        self.cpu_usage_percent = Gauge(
            'mimir_cpu_usage_percent',
            'CPU usage percentage',
            registry=registry
        )
        
        self.disk_usage_bytes = Gauge(
            'mimir_disk_usage_bytes',
            'Disk usage in bytes',
            ['path', 'type'],  # total, used, free
            registry=registry
        )
        
        self.file_descriptors_open = Gauge(
            'mimir_file_descriptors_open',
            'Number of open file descriptors',
            registry=registry
        )
        
        self.network_connections_active = Gauge(
            'mimir_network_connections_active',
            'Number of active network connections',
            ['type'],  # tcp, udp
            registry=registry
        )
        
        # =========================
        # MCP SERVER METRICS
        # =========================
        
        self.mcp_requests_total = Counter(
            'mimir_mcp_requests_total',
            'Total number of MCP requests',
            ['method', 'status'],
            registry=registry
        )
        
        self.mcp_request_duration_seconds = Histogram(
            'mimir_mcp_request_duration_seconds',
            'MCP request duration in seconds',
            ['method'],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
            registry=registry
        )
        
        self.mcp_active_connections = Gauge(
            'mimir_mcp_active_connections',
            'Number of active MCP connections',
            registry=registry
        )
        
        # =========================
        # BUSINESS METRICS
        # =========================
        
        self.repositories_indexed_total = Counter(
            'mimir_repositories_indexed_total',
            'Total number of repositories indexed',
            ['language', 'status'],
            registry=registry
        )
        
        self.code_symbols_extracted_total = Counter(
            'mimir_code_symbols_extracted_total',
            'Total number of code symbols extracted',
            ['language', 'symbol_type'],
            registry=registry
        )
        
        self.vector_embeddings_created_total = Counter(
            'mimir_vector_embeddings_created_total', 
            'Total number of vector embeddings created',
            ['model', 'chunk_type'],
            registry=registry
        )
        
        self.research_queries_total = Counter(
            'mimir_research_queries_total',
            'Total number of code research queries',
            ['complexity', 'success'],
            registry=registry
        )
        
        # =========================
        # CACHE METRICS
        # =========================
        
        self.cache_hits_total = Counter(
            'mimir_cache_hits_total',
            'Total number of cache hits',
            ['cache_type'],
            registry=registry
        )
        
        self.cache_misses_total = Counter(
            'mimir_cache_misses_total',
            'Total number of cache misses',
            ['cache_type'],
            registry=registry
        )
        
        self.cache_size_entries = Gauge(
            'mimir_cache_size_entries',
            'Number of entries in cache',
            ['cache_type'],
            registry=registry
        )
        
        # =========================
        # HEALTH METRICS
        # =========================
        
        self.health_check_status = Gauge(
            'mimir_health_check_status',
            'Health check status (1=healthy, 0=unhealthy)',
            ['check_type'],
            registry=registry
        )
        
        self.uptime_seconds = Gauge(
            'mimir_uptime_seconds',
            'Process uptime in seconds',
            registry=registry
        )
        
        # =========================
        # INFO METRICS
        # =========================
        
        self.build_info = Info(
            'mimir_build_info',
            'Mimir build information',
            registry=registry
        )
        
        # Set initial build info
        self.build_info.info({
            'version': '0.1.0',
            'python_version': f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
            'features': 'vector,symbol,graph,pipeline'
        })
    
    # =========================
    # PIPELINE METRICS METHODS
    # =========================
    
    def record_pipeline_start(self, stage: str, language: str = "unknown"):
        """Record pipeline stage start."""
        if not self.enabled:
            return
            
        self.pipeline_active_count.inc()
        logger.debug("Pipeline started", stage=stage, language=language)
    
    def record_pipeline_success(self, stage: str, duration_seconds: float, 
                              language: str = "unknown", files_processed: int = 0):
        """Record successful pipeline completion."""
        if not self.enabled:
            return
            
        self.pipeline_executions_total.labels(
            stage=stage, status="success", language=language
        ).inc()
        
        self.pipeline_duration_seconds.labels(
            stage=stage, language=language
        ).observe(duration_seconds)
        
        if files_processed > 0:
            self.pipeline_files_processed.labels(
                stage=stage, language=language
            ).observe(files_processed)
        
        self.pipeline_active_count.dec()
        
        logger.debug(
            "Pipeline completed successfully",
            stage=stage,
            duration_seconds=duration_seconds,
            files_processed=files_processed
        )
    
    def record_pipeline_error(self, stage: str, error_type: str, 
                            severity: str = "medium", language: str = "unknown"):
        """Record pipeline error."""
        if not self.enabled:
            return
            
        self.pipeline_executions_total.labels(
            stage=stage, status="error", language=language
        ).inc()
        
        self.pipeline_errors_total.labels(
            stage=stage, error_type=error_type, severity=severity
        ).inc()
        
        self.pipeline_active_count.dec()
        
        logger.debug(
            "Pipeline error recorded",
            stage=stage,
            error_type=error_type,
            severity=severity
        )
    
    # =========================
    # SEARCH METRICS METHODS
    # =========================
    
    def record_search_request(self, search_type: str, duration_seconds: float,
                            results_count: int, status: str = "success"):
        """Record search request metrics."""
        if not self.enabled:
            return
            
        self.search_requests_total.labels(
            search_type=search_type, status=status
        ).inc()
        
        self.search_duration_seconds.labels(
            search_type=search_type
        ).observe(duration_seconds)
        
        self.search_results_count.labels(
            search_type=search_type
        ).observe(results_count)
        
        logger.debug(
            "Search request recorded",
            search_type=search_type,
            duration_seconds=duration_seconds,
            results_count=results_count,
            status=status
        )
    
    def record_vector_similarity(self, similarity_score: float):
        """Record vector similarity score."""
        if not self.enabled:
            return
            
        self.vector_similarity_scores.observe(similarity_score)
    
    def record_symbol_lookup(self, duration_seconds: float):
        """Record symbol lookup duration."""
        if not self.enabled:
            return
            
        self.symbol_lookup_duration_seconds.observe(duration_seconds)
    
    # =========================
    # MCP SERVER METRICS METHODS
    # =========================
    
    def record_mcp_request(self, method: str, duration_seconds: float, status: str = "success"):
        """Record MCP request metrics."""
        if not self.enabled:
            return
            
        self.mcp_requests_total.labels(method=method, status=status).inc()
        self.mcp_request_duration_seconds.labels(method=method).observe(duration_seconds)
    
    def set_mcp_active_connections(self, count: int):
        """Set number of active MCP connections."""
        if not self.enabled:
            return
            
        self.mcp_active_connections.set(count)
    
    # =========================
    # SYSTEM METRICS METHODS
    # =========================
    
    def update_system_metrics(self):
        """Update system resource metrics.""" 
        if not self.enabled:
            return
            
        try:
            import psutil
            process = psutil.Process()
            
            # Memory metrics
            memory_info = process.memory_info()
            self.memory_usage_bytes.labels(type="rss").set(memory_info.rss)
            self.memory_usage_bytes.labels(type="vms").set(memory_info.vms)
            
            # CPU metrics
            self.cpu_usage_percent.set(process.cpu_percent())
            
            # File descriptor metrics
            self.file_descriptors_open.set(process.num_fds())
            
            # Network connection metrics
            connections = process.connections()
            tcp_count = sum(1 for conn in connections if conn.type.name == 'SOCK_STREAM')
            udp_count = sum(1 for conn in connections if conn.type.name == 'SOCK_DGRAM')
            
            self.network_connections_active.labels(type="tcp").set(tcp_count)
            self.network_connections_active.labels(type="udp").set(udp_count)
            
        except ImportError:
            logger.warning("psutil not available for system metrics")
        except Exception as e:
            logger.error("Error updating system metrics", error=str(e))
    
    def update_disk_metrics(self, paths: List[str]):
        """Update disk usage metrics for specified paths."""
        if not self.enabled:
            return
            
        try:
            import psutil
            for path in paths:
                try:
                    disk_usage = psutil.disk_usage(path)
                    self.disk_usage_bytes.labels(path=path, type="total").set(disk_usage.total)
                    self.disk_usage_bytes.labels(path=path, type="used").set(disk_usage.used)
                    self.disk_usage_bytes.labels(path=path, type="free").set(disk_usage.free)
                except Exception as e:
                    logger.warning(f"Error getting disk usage for {path}", error=str(e))
        except ImportError:
            logger.warning("psutil not available for disk metrics")
    
    # =========================
    # BUSINESS METRICS METHODS  
    # =========================
    
    def record_repository_indexed(self, language: str, status: str = "success"):
        """Record repository indexing completion."""
        if not self.enabled:
            return
            
        self.repositories_indexed_total.labels(language=language, status=status).inc()
    
    def record_symbols_extracted(self, language: str, symbol_type: str, count: int = 1):
        """Record code symbols extraction."""
        if not self.enabled:
            return
            
        self.code_symbols_extracted_total.labels(
            language=language, symbol_type=symbol_type
        ).inc(count)
    
    def record_embeddings_created(self, model: str, chunk_type: str, count: int = 1):
        """Record vector embeddings creation.""" 
        if not self.enabled:
            return
            
        self.vector_embeddings_created_total.labels(
            model=model, chunk_type=chunk_type
        ).inc(count)
    
    def record_research_query(self, complexity: str, success: bool = True):
        """Record code research query."""
        if not self.enabled:
            return
            
        success_str = "true" if success else "false"
        self.research_queries_total.labels(
            complexity=complexity, success=success_str
        ).inc()
    
    # =========================
    # CACHE METRICS METHODS
    # =========================
    
    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        if not self.enabled:
            return
            
        self.cache_hits_total.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str):
        """Record cache miss."""
        if not self.enabled:
            return
            
        self.cache_misses_total.labels(cache_type=cache_type).inc()
    
    def set_cache_size(self, cache_type: str, size: int):
        """Set cache size."""
        if not self.enabled:
            return
            
        self.cache_size_entries.labels(cache_type=cache_type).set(size)
    
    # =========================
    # HEALTH METRICS METHODS
    # =========================
    
    def set_health_status(self, check_type: str, healthy: bool):
        """Set health check status."""
        if not self.enabled:
            return
            
        status = 1 if healthy else 0
        self.health_check_status.labels(check_type=check_type).set(status)
    
    def update_uptime(self, start_time: float):
        """Update process uptime."""
        if not self.enabled:
            return
            
        uptime = time.time() - start_time
        self.uptime_seconds.set(uptime)
    
    # =========================
    # CUSTOM METRICS
    # =========================
    
    def register_custom_metric(self, definition: MetricDefinition):
        """Register a custom metric."""
        if not self.enabled:
            return
            
        try:
            if definition.metric_type == "counter":
                metric = Counter(
                    definition.name, 
                    definition.help_text,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == "histogram":
                metric = Histogram(
                    definition.name,
                    definition.help_text, 
                    definition.labels,
                    buckets=definition.buckets,
                    registry=self.registry
                )
            elif definition.metric_type == "gauge":
                metric = Gauge(
                    definition.name,
                    definition.help_text,
                    definition.labels, 
                    registry=self.registry
                )
            elif definition.metric_type == "summary":
                metric = Summary(
                    definition.name,
                    definition.help_text,
                    definition.labels,
                    registry=self.registry
                )
            elif definition.metric_type == "info":
                metric = Info(
                    definition.name,
                    definition.help_text,
                    registry=self.registry
                )
            else:
                raise ValueError(f"Unknown metric type: {definition.metric_type}")
            
            self._custom_metrics[definition.name] = metric
            logger.info(f"Registered custom metric: {definition.name}")
            
        except Exception as e:
            logger.error(f"Error registering custom metric {definition.name}", error=str(e))
    
    def get_custom_metric(self, name: str):
        """Get a custom metric by name."""
        return self._custom_metrics.get(name)
    
    # =========================
    # CONTEXT MANAGERS
    # =========================
    
    @asynccontextmanager
    async def time_pipeline_stage(self, stage: str, language: str = "unknown"):
        """Context manager for timing pipeline stages."""
        start_time = time.time()
        self.record_pipeline_start(stage, language)
        
        try:
            yield
            duration = time.time() - start_time
            self.record_pipeline_success(stage, duration, language)
        except Exception as e:
            error_type = type(e).__name__
            self.record_pipeline_error(stage, error_type, "high", language)
            raise
    
    @asynccontextmanager 
    async def time_search_request(self, search_type: str):
        """Context manager for timing search requests."""
        start_time = time.time()
        results_count = 0
        
        try:
            yield lambda count: setattr(self, '_temp_results_count', count)
            duration = time.time() - start_time
            results_count = getattr(self, '_temp_results_count', 0)
            self.record_search_request(search_type, duration, results_count, "success")
        except Exception:
            duration = time.time() - start_time
            self.record_search_request(search_type, duration, 0, "error")
            raise
    
    # =========================
    # EXPORT AND SERVER METHODS
    # =========================
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics in text format."""
        if not self.enabled:
            return "# Prometheus metrics not available\n"
            
        try:
            from prometheus_client import generate_latest
            return generate_latest(self.registry).decode('utf-8')
        except Exception as e:
            logger.error("Error generating metrics", error=str(e))
            return f"# Error generating metrics: {e}\n"
    
    def start_metrics_server(self, port: int = 9100):
        """Start HTTP server for metrics endpoint."""
        if not self.enabled:
            logger.warning("Cannot start metrics server: Prometheus not available")
            return
            
        try:
            self._http_server = start_http_server(port, registry=self.registry)
            logger.info(f"Metrics server started on port {port}")
        except Exception as e:
            logger.error(f"Error starting metrics server on port {port}", error=str(e))
    
    def push_metrics(self, gateway_url: str, job_name: str = "mimir"):
        """Push metrics to Prometheus pushgateway."""
        if not self.enabled:
            return
            
        try:
            push_to_gateway(gateway_url, job=job_name, registry=self.registry)
            logger.debug(f"Pushed metrics to gateway: {gateway_url}")
        except Exception as e:
            logger.error(f"Error pushing metrics to gateway {gateway_url}", error=str(e))


# =========================
# GLOBAL COLLECTOR INSTANCE
# =========================

_metrics_collector: Optional[MetricsCollector] = None
_collector_lock = threading.Lock()


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector instance."""
    global _metrics_collector
    
    if _metrics_collector is None:
        with _collector_lock:
            if _metrics_collector is None:
                _metrics_collector = MetricsCollector()
    
    return _metrics_collector


# =========================
# CONVENIENCE DECORATORS
# =========================

def pipeline_metrics(stage: str, language: str = "unknown"):
    """Decorator for automatic pipeline metrics collection."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                collector = get_metrics_collector()
                async with collector.time_pipeline_stage(stage, language):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                collector = get_metrics_collector()
                start_time = time.time()
                collector.record_pipeline_start(stage, language)
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    collector.record_pipeline_success(stage, duration, language)
                    return result
                except Exception as e:
                    error_type = type(e).__name__
                    collector.record_pipeline_error(stage, error_type, "high", language)
                    raise
                    
            return sync_wrapper
    return decorator


def search_metrics(search_type: str):
    """Decorator for automatic search metrics collection."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                collector = get_metrics_collector()
                async with collector.time_search_request(search_type) as set_count:
                    result = await func(*args, **kwargs)
                    if hasattr(result, '__len__'):
                        set_count(len(result))
                    elif hasattr(result, 'total_count'):
                        set_count(result.total_count)
                    return result
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                collector = get_metrics_collector()
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    results_count = 0
                    
                    if hasattr(result, '__len__'):
                        results_count = len(result)
                    elif hasattr(result, 'total_count'):
                        results_count = result.total_count
                    
                    collector.record_search_request(search_type, duration, results_count, "success")
                    return result
                except Exception:
                    duration = time.time() - start_time
                    collector.record_search_request(search_type, duration, 0, "error")
                    raise
                    
            return sync_wrapper
    return decorator


def server_metrics(method: str):
    """Decorator for automatic server metrics collection."""
    def decorator(func):
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                collector = get_metrics_collector()
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    duration = time.time() - start_time
                    collector.record_mcp_request(method, duration, "success")
                    return result
                except Exception:
                    duration = time.time() - start_time
                    collector.record_mcp_request(method, duration, "error")
                    raise
            return async_wrapper
        else:
            def sync_wrapper(*args, **kwargs):
                collector = get_metrics_collector()
                start_time = time.time()
                
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    collector.record_mcp_request(method, duration, "success")
                    return result
                except Exception:
                    duration = time.time() - start_time
                    collector.record_mcp_request(method, duration, "error")
                    raise
                    
            return sync_wrapper
    return decorator


def custom_metrics(metric_name: str, metric_type: str = "counter", labels: Dict[str, str] = None):
    """Decorator for custom metrics collection."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            collector = get_metrics_collector()
            custom_metric = collector.get_custom_metric(metric_name)
            
            if custom_metric:
                try:
                    if metric_type == "counter":
                        if labels:
                            custom_metric.labels(**labels).inc()
                        else:
                            custom_metric.inc()
                except Exception as e:
                    logger.warning(f"Error updating custom metric {metric_name}", error=str(e))
            
            return func(*args, **kwargs)
        return wrapper
    return decorator