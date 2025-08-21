"""
Monitoring and observability module for Mimir Deep Code Research System.

Provides comprehensive metrics collection, distributed tracing, structured logging,
and performance monitoring for production environments.
"""

from .metrics import (
    MetricsCollector,
    get_metrics_collector,
    pipeline_metrics,
    search_metrics,
    server_metrics,
    custom_metrics
)
from .tracing import (
    TraceManager,
    get_trace_manager,
    trace_pipeline_stage,
    trace_operation,
    create_span
)
from .alerts import (
    AlertManager,
    get_alert_manager,
    register_alert_rule,
    check_alert_conditions
)
from .dashboard import (
    DashboardGenerator,
    generate_pipeline_dashboard,
    generate_performance_dashboard,
    generate_error_dashboard
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
    "generate_error_dashboard"
]