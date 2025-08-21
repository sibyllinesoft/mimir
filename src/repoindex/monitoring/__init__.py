"""
Monitoring and observability module for Mimir Deep Code Research System.

Provides comprehensive metrics collection, distributed tracing, structured logging,
and performance monitoring for production environments.
"""

from .alerts import AlertManager, check_alert_conditions, get_alert_manager, register_alert_rule
from .dashboard import (
    DashboardGenerator,
    generate_error_dashboard,
    generate_performance_dashboard,
    generate_pipeline_dashboard,
)
from .metrics import (
    MetricsCollector,
    custom_metrics,
    get_metrics_collector,
    pipeline_metrics,
    search_metrics,
    server_metrics,
)
from .tracing import (
    TraceManager,
    create_span,
    get_trace_manager,
    trace_operation,
    trace_pipeline_stage,
)

__all__ = [
    # Metrics
    "MetricsCollector",
    "get_metrics_collector",
    "pipeline_metrics",
    "search_metrics",
    "server_metrics",
    "custom_metrics",
    # Tracing
    "TraceManager",
    "get_trace_manager",
    "trace_pipeline_stage",
    "trace_operation",
    "create_span",
    # Alerting
    "AlertManager",
    "get_alert_manager",
    "register_alert_rule",
    "check_alert_conditions",
    # Dashboards
    "DashboardGenerator",
    "generate_pipeline_dashboard",
    "generate_performance_dashboard",
    "generate_error_dashboard",
]
