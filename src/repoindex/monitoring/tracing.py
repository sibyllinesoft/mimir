"""
Distributed tracing for Mimir Deep Code Research System.

Provides comprehensive tracing of requests through the 6-stage pipeline with
OpenTelemetry integration for performance monitoring and debugging.
"""

import threading
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from opentelemetry import trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.propagate import extract, inject
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.trace import Status, StatusCode
    from opentelemetry.util.http import get_scheme_host_port

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

    # Dummy classes for graceful degradation
    class DummySpan:
        def set_attribute(self, key, value):
            pass

        def set_status(self, status):
            pass

        def record_exception(self, exception):
            pass

        def add_event(self, name, attributes=None):
            pass

        def end(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class DummyTracer:
        def start_span(self, name, **kwargs):
            return DummySpan()

        def start_as_current_span(self, name, **kwargs):
            return contextmanager(lambda: iter([DummySpan()]))()


from ..util.logging_config import get_logger

logger = get_logger("monitoring.tracing")

# Context variable for current trace context
_current_trace_context: ContextVar[dict[str, Any] | None] = ContextVar(
    "trace_context", default=None
)


@dataclass
class SpanContext:
    """Span context with correlation information."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    baggage: dict[str, str] = field(default_factory=dict)


@dataclass
class TraceMetrics:
    """Metrics collected during tracing."""

    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    duration_ms: float | None = None
    memory_usage_mb: float | None = None
    cpu_percent: float | None = None
    error_occurred: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)


class TraceManager:
    """
    Comprehensive trace manager for Mimir system.

    Provides distributed tracing capabilities with:
    - Pipeline stage tracing
    - Cross-service correlation
    - Performance metrics collection
    - Error tracking and debugging
    """

    def __init__(
        self,
        service_name: str = "mimir-repoindex",
        jaeger_endpoint: str | None = None,
        otlp_endpoint: str | None = None,
        console_export: bool = False,
        sample_rate: float = 1.0,
    ):
        """Initialize trace manager with exporters."""
        self.service_name = service_name
        self.enabled = OPENTELEMETRY_AVAILABLE
        self.sample_rate = sample_rate
        self._spans_cache = {}
        self._active_spans = {}

        if not self.enabled:
            logger.warning("OpenTelemetry not available, tracing will be no-op")
            self.tracer = DummyTracer()
            return

        # Configure tracing
        self._setup_tracing(jaeger_endpoint, otlp_endpoint, console_export)

        logger.info(
            "Trace manager initialized",
            service_name=service_name,
            enabled=self.enabled,
            sample_rate=sample_rate,
        )

    def _setup_tracing(
        self, jaeger_endpoint: str | None, otlp_endpoint: str | None, console_export: bool
    ):
        """Setup OpenTelemetry tracing with configured exporters."""
        if not self.enabled:
            return

        # Create resource
        resource = Resource.create(
            {
                "service.name": self.service_name,
                "service.version": "0.1.0",
                "deployment.environment": "development",  # Could be configured
            }
        )

        # Create tracer provider
        tracer_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(tracer_provider)

        # Configure exporters
        exporters = []

        if jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name="localhost",
                    agent_port=6831,
                    collector_endpoint=jaeger_endpoint,
                )
                exporters.append(jaeger_exporter)
                logger.info(f"Jaeger exporter configured: {jaeger_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to configure Jaeger exporter: {e}")

        if otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                exporters.append(otlp_exporter)
                logger.info(f"OTLP exporter configured: {otlp_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to configure OTLP exporter: {e}")

        if console_export:
            console_exporter = ConsoleSpanExporter()
            exporters.append(console_exporter)
            logger.info("Console exporter configured")

        # Add span processors
        for exporter in exporters:
            span_processor = BatchSpanProcessor(exporter)
            tracer_provider.add_span_processor(span_processor)

        # Get tracer
        self.tracer = trace.get_tracer(self.service_name)

    def create_trace_context(
        self, operation: str, pipeline_id: str | None = None, **attributes
    ) -> SpanContext:
        """Create new trace context for operation."""
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())

        context = SpanContext(
            trace_id=trace_id,
            span_id=span_id,
            baggage={
                "operation": operation,
                "pipeline_id": pipeline_id or "",
                **{k: str(v) for k, v in attributes.items()},
            },
        )

        _current_trace_context.set(
            {
                "trace_id": trace_id,
                "span_id": span_id,
                "operation": operation,
                "pipeline_id": pipeline_id,
                **attributes,
            }
        )

        return context

    def get_current_context(self) -> dict[str, Any] | None:
        """Get current trace context."""
        return _current_trace_context.get()

    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **attributes):
        """Context manager for tracing operations."""
        if not self.enabled:
            yield DummySpan()
            return

        span = self.tracer.start_span(operation_name)

        try:
            # Set standard attributes
            span.set_attribute("operation.name", operation_name)
            span.set_attribute("service.name", self.service_name)

            # Set custom attributes
            for key, value in attributes.items():
                span.set_attribute(f"custom.{key}", str(value))

            # Add context from current trace
            current_context = self.get_current_context()
            if current_context:
                span.set_attribute("trace.parent_operation", current_context.get("operation", ""))
                if "pipeline_id" in current_context:
                    span.set_attribute("pipeline.id", current_context["pipeline_id"])

            # Track span
            span_id = str(uuid.uuid4())
            self._active_spans[span_id] = span

            # Add events
            span.add_event(
                "operation.started", {"timestamp": time.time(), "operation": operation_name}
            )

            yield span

            # Success
            span.set_status(Status(StatusCode.OK))
            span.add_event("operation.completed", {"timestamp": time.time(), "status": "success"})

        except Exception as e:
            # Error handling
            span.set_status(Status(StatusCode.ERROR, str(e)))
            span.record_exception(e)
            span.add_event(
                "operation.failed",
                {"timestamp": time.time(), "error_type": type(e).__name__, "error_message": str(e)},
            )
            raise

        finally:
            span.end()
            if span_id in self._active_spans:
                del self._active_spans[span_id]

    @asynccontextmanager
    async def trace_pipeline_stage(self, stage: str, pipeline_id: str, **attributes):
        """Context manager for tracing pipeline stages."""
        stage_name = f"pipeline.{stage}"

        async with self.trace_operation(
            stage_name, pipeline_id=pipeline_id, stage=stage, **attributes
        ) as span:
            # Pipeline-specific attributes
            span.set_attribute("pipeline.stage", stage)
            span.set_attribute("pipeline.id", pipeline_id)

            # System metrics at start
            try:
                import psutil

                process = psutil.Process()
                memory_info = process.memory_info()

                span.set_attribute("system.memory_rss_mb", memory_info.rss / 1024 / 1024)
                span.set_attribute("system.cpu_percent", process.cpu_percent())
                span.set_attribute("system.open_files", process.num_fds())
            except (ImportError, Exception):
                pass

            yield span

    @asynccontextmanager
    async def trace_search_request(self, search_type: str, query: str, **attributes):
        """Context manager for tracing search requests."""
        async with self.trace_operation(
            f"search.{search_type}", search_type=search_type, query_length=len(query), **attributes
        ) as span:
            # Search-specific attributes
            span.set_attribute("search.type", search_type)
            span.set_attribute("search.query_length", len(query))
            span.set_attribute("search.query_hash", str(hash(query)))

            yield span

    @asynccontextmanager
    async def trace_mcp_request(self, method: str, request_id: str | None = None, **attributes):
        """Context manager for tracing MCP requests."""
        async with self.trace_operation(
            f"mcp.{method}", method=method, request_id=request_id or str(uuid.uuid4()), **attributes
        ) as span:
            # MCP-specific attributes
            span.set_attribute("mcp.method", method)
            if request_id:
                span.set_attribute("mcp.request_id", request_id)

            yield span

    def create_span(self, operation_name: str, parent_span: Optional = None, **attributes):
        """Create a new span with optional parent."""
        if not self.enabled:
            return DummySpan()

        span = self.tracer.start_span(operation_name, context=parent_span)

        # Set attributes
        span.set_attribute("operation.name", operation_name)
        for key, value in attributes.items():
            span.set_attribute(key, str(value))

        return span

    def inject_context(self, headers: dict[str, str]) -> dict[str, str]:
        """Inject trace context into headers for cross-service propagation."""
        if not self.enabled:
            return headers

        try:
            inject(headers)
            return headers
        except Exception as e:
            logger.warning(f"Failed to inject trace context: {e}")
            return headers

    def extract_context(self, headers: dict[str, str]) -> Any | None:
        """Extract trace context from headers."""
        if not self.enabled:
            return None

        try:
            return extract(headers)
        except Exception as e:
            logger.warning(f"Failed to extract trace context: {e}")
            return None

    def get_trace_id(self) -> str | None:
        """Get current trace ID."""
        current_context = self.get_current_context()
        return current_context.get("trace_id") if current_context else None

    def add_baggage(self, key: str, value: str):
        """Add baggage to current trace context."""
        current_context = self.get_current_context()
        if current_context:
            current_context[f"baggage.{key}"] = value
            _current_trace_context.set(current_context)

    def get_baggage(self, key: str) -> str | None:
        """Get baggage from current trace context."""
        current_context = self.get_current_context()
        return current_context.get(f"baggage.{key}") if current_context else None

    def flush_traces(self):
        """Flush pending traces to exporters."""
        if not self.enabled:
            return

        try:
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "_active_span_processor"):
                tracer_provider._active_span_processor.force_flush()
        except Exception as e:
            logger.warning(f"Failed to flush traces: {e}")

    def shutdown(self):
        """Shutdown trace manager and flush remaining traces."""
        if not self.enabled:
            return

        try:
            self.flush_traces()
            tracer_provider = trace.get_tracer_provider()
            if hasattr(tracer_provider, "shutdown"):
                tracer_provider.shutdown()
            logger.info("Trace manager shutdown completed")
        except Exception as e:
            logger.error(f"Error during trace manager shutdown: {e}")


# =========================
# GLOBAL TRACE MANAGER
# =========================

_trace_manager: TraceManager | None = None
_tracer_lock = threading.Lock()


def get_trace_manager() -> TraceManager:
    """Get or create global trace manager instance."""
    global _trace_manager

    if _trace_manager is None:
        with _tracer_lock:
            if _trace_manager is None:
                # Configure from environment
                import os

                # Try to get configuration from centralized config first, fall back to environment
                try:
                    from ..config import get_monitoring_config
                    from ..config_migration import migration_tracker
                    
                    config = get_monitoring_config()
                    service_name = config.service_name
                    jaeger_endpoint = config.jaeger_endpoint
                    otlp_endpoint = config.otlp_endpoint
                    console_export = config.trace_console
                    sample_rate = config.trace_sample_rate
                    
                    # Mark this file as migrated
                    migration_tracker.mark_migrated(__file__, ["monitoring"])
                    
                except ImportError:
                    # Centralized config not available, fall back to environment variables
                    service_name = os.getenv("MIMIR_SERVICE_NAME", "mimir-repoindex")
                    jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
                    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
                    console_export = os.getenv("MIMIR_TRACE_CONSOLE", "false").lower() == "true"
                    sample_rate = float(os.getenv("MIMIR_TRACE_SAMPLE_RATE", "1.0"))

                _trace_manager = TraceManager(
                    service_name=service_name,
                    jaeger_endpoint=jaeger_endpoint,
                    otlp_endpoint=otlp_endpoint,
                    console_export=console_export,
                    sample_rate=sample_rate,
                )

    return _trace_manager


# =========================
# CONVENIENCE FUNCTIONS
# =========================


async def trace_pipeline_stage(stage: str, pipeline_id: str, **attributes):
    """Convenience function for tracing pipeline stages."""
    trace_manager = get_trace_manager()
    return trace_manager.trace_pipeline_stage(stage, pipeline_id, **attributes)


async def trace_operation(operation_name: str, **attributes):
    """Convenience function for tracing operations."""
    trace_manager = get_trace_manager()
    return trace_manager.trace_operation(operation_name, **attributes)


def create_span(operation_name: str, parent_span=None, **attributes):
    """Convenience function for creating spans."""
    trace_manager = get_trace_manager()
    return trace_manager.create_span(operation_name, parent_span, **attributes)


def get_current_trace_id() -> str | None:
    """Get current trace ID."""
    trace_manager = get_trace_manager()
    return trace_manager.get_trace_id()


def inject_trace_context(headers: dict[str, str]) -> dict[str, str]:
    """Inject trace context into headers."""
    trace_manager = get_trace_manager()
    return trace_manager.inject_context(headers)


def extract_trace_context(headers: dict[str, str]):
    """Extract trace context from headers."""
    trace_manager = get_trace_manager()
    return trace_manager.extract_context(headers)


# =========================
# TRACING DECORATORS
# =========================


def trace_async(operation_name: str = None, **trace_attributes):
    """Decorator for tracing async functions."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            trace_manager = get_trace_manager()

            async with trace_manager.trace_operation(op_name, **trace_attributes) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                return await func(*args, **kwargs)

        return wrapper

    return decorator


def trace_sync(operation_name: str = None, **trace_attributes):
    """Decorator for tracing sync functions."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            if not OPENTELEMETRY_AVAILABLE:
                return func(*args, **kwargs)

            tracer = trace.get_tracer(__name__)
            with tracer.start_as_current_span(op_name) as span:
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)

                # Add custom attributes
                for key, value in trace_attributes.items():
                    span.set_attribute(key, str(value))

                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise

        return wrapper

    return decorator


# =========================
# TRACE CONTEXT UTILITIES
# =========================


class TraceContextManager:
    """Context manager for setting trace context."""

    def __init__(self, **context):
        self.context = context
        self.previous_context = None

    def __enter__(self):
        self.previous_context = _current_trace_context.get()
        new_context = (self.previous_context or {}).copy()
        new_context.update(self.context)
        _current_trace_context.set(new_context)
        return new_context

    def __exit__(self, exc_type, exc_val, exc_tb):
        _current_trace_context.set(self.previous_context)


def set_trace_context(**context):
    """Set trace context for current execution."""
    return TraceContextManager(**context)


def clear_trace_context():
    """Clear current trace context."""
    _current_trace_context.set(None)
