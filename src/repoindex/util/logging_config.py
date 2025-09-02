"""
Production-grade logging configuration and utilities.

Provides structured logging with JSON output, performance tracking,
error correlation, and observability integration for the Mimir system.
"""

import json
import logging
import logging.config
import os
import sys
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Context variables for request/session tracking
REQUEST_ID: ContextVar[str | None] = ContextVar("request_id", default=None)
SESSION_ID: ContextVar[str | None] = ContextVar("session_id", default=None)
USER_ID: ContextVar[str | None] = ContextVar("user_id", default=None)
PIPELINE_ID: ContextVar[str | None] = ContextVar("pipeline_id", default=None)


@dataclass
class LoggingMetrics:
    """Performance and operational metrics for logging."""

    start_time: float = field(default_factory=time.time)
    memory_usage_mb: float | None = None
    cpu_percent: float | None = None
    disk_io_read: int | None = None
    disk_io_write: int | None = None
    network_bytes_sent: int | None = None
    network_bytes_recv: int | None = None

    def duration_ms(self) -> float:
        """Get duration since start time in milliseconds."""
        return (time.time() - self.start_time) * 1000


class StructuredFormatter(logging.Formatter):
    """
    JSON formatter for structured logging with rich context.

    Includes request correlation, performance metrics, and
    error context for production observability.
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""

        # Base log structure
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add correlation IDs from context
        if REQUEST_ID.get():
            log_entry["request_id"] = REQUEST_ID.get()
        if SESSION_ID.get():
            log_entry["session_id"] = SESSION_ID.get()
        if USER_ID.get():
            log_entry["user_id"] = USER_ID.get()
        if PIPELINE_ID.get():
            log_entry["pipeline_id"] = PIPELINE_ID.get()

        # Add custom fields from record
        if hasattr(record, "component"):
            log_entry["component"] = record.component
        if hasattr(record, "operation"):
            log_entry["operation"] = record.operation
        if hasattr(record, "metrics"):
            log_entry["metrics"] = record.metrics
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": self.formatException(record.exc_info),
            }

        # Add error context from MimirError if available
        if hasattr(record, "error_context"):
            log_entry["error"] = record.error_context

        return json.dumps(log_entry, default=str, separators=(",", ":"))


class PerformanceLogger:
    """
    Context manager for performance tracking and automatic logging.

    Tracks operation duration, resource usage, and automatically
    logs performance metrics with structured context.
    """

    def __init__(
        self,
        logger: logging.Logger,
        operation: str,
        component: str = None,
        level: int = logging.INFO,
        include_system_metrics: bool = True,
    ):
        self.logger = logger
        self.operation = operation
        self.component = component or "unknown"
        self.level = level
        self.include_system_metrics = include_system_metrics
        self.metrics = LoggingMetrics()
        self.start_time = None

    def __enter__(self):
        """Start performance tracking."""
        self.start_time = time.time()
        self.metrics.start_time = self.start_time

        # Capture initial system metrics
        if self.include_system_metrics:
            try:
                import psutil

                process = psutil.Process()
                self.initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                self.initial_cpu_times = process.cpu_times()
                self.initial_io = process.io_counters()
                self.initial_net = psutil.net_io_counters()
            except ImportError:
                # psutil not available, skip system metrics
                self.include_system_metrics = False

        # Log operation start
        self.logger.log(
            self.level,
            f"Starting {self.operation}",
            extra={"component": self.component, "operation": self.operation, "event": "start"},
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End performance tracking and log results."""
        end_time = time.time()
        duration_ms = (end_time - self.start_time) * 1000

        # Calculate system metrics delta
        if self.include_system_metrics:
            try:
                import psutil

                process = psutil.Process()
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                process.cpu_times()
                final_io = process.io_counters()
                final_net = psutil.net_io_counters()

                self.metrics.memory_usage_mb = final_memory - self.initial_memory
                self.metrics.cpu_percent = process.cpu_percent()
                self.metrics.disk_io_read = final_io.read_bytes - self.initial_io.read_bytes
                self.metrics.disk_io_write = final_io.write_bytes - self.initial_io.write_bytes
                self.metrics.network_bytes_sent = final_net.bytes_sent - self.initial_net.bytes_sent
                self.metrics.network_bytes_recv = final_net.bytes_recv - self.initial_net.bytes_recv
            except Exception:
                pass  # Ignore metrics collection errors

        # Determine log level based on duration and errors
        log_level = self.level
        if exc_type:
            log_level = logging.ERROR
        elif duration_ms > 10000:  # > 10 seconds
            log_level = logging.WARNING

        # Log operation completion
        self.logger.log(
            log_level,
            f"Completed {self.operation} in {duration_ms:.2f}ms",
            extra={
                "component": self.component,
                "operation": self.operation,
                "event": "complete",
                "metrics": {
                    "duration_ms": duration_ms,
                    "memory_delta_mb": self.metrics.memory_usage_mb,
                    "cpu_percent": self.metrics.cpu_percent,
                    "disk_read_bytes": self.metrics.disk_io_read,
                    "disk_write_bytes": self.metrics.disk_io_write,
                    "network_sent_bytes": self.metrics.network_bytes_sent,
                    "network_recv_bytes": self.metrics.network_bytes_recv,
                },
                "success": exc_type is None,
            },
        )


class MimirLoggerAdapter(logging.LoggerAdapter):
    """
    Enhanced logger adapter with automatic context injection.

    Provides convenience methods for structured logging with
    component/operation context and error handling integration.
    """

    def __init__(self, logger: logging.Logger, component: str):
        self.component = component
        super().__init__(logger, {"component": component})

    def process(self, msg, kwargs):
        """Add component context to all log records."""
        if "extra" not in kwargs:
            kwargs["extra"] = {}
        kwargs["extra"]["component"] = self.component
        return msg, kwargs

    def operation_start(self, operation: str, **context):
        """Log operation start with context."""
        self.info(
            f"Starting {operation}",
            extra={"operation": operation, "event": "start", "extra_fields": context},
        )

    def operation_success(self, operation: str, duration_ms: float = None, **context):
        """Log successful operation completion."""
        msg = f"Completed {operation}"
        if duration_ms is not None:
            msg += f" in {duration_ms:.2f}ms"

        self.info(
            msg,
            extra={
                "operation": operation,
                "event": "success",
                "metrics": {"duration_ms": duration_ms} if duration_ms else {},
                "extra_fields": context,
            },
        )

    def operation_error(self, operation: str, error: Exception, **context):
        """Log operation error with structured context."""
        # Import here to avoid circular dependency
        from .errors import MimirError

        extra = {"operation": operation, "event": "error", "extra_fields": context}

        if isinstance(error, MimirError):
            extra["error_context"] = error.to_dict()

        self.error(f"Error in {operation}: {str(error)}", exc_info=True, extra=extra)

    def performance_track(
        self, operation: str, level: int = logging.INFO, include_system_metrics: bool = True
    ) -> PerformanceLogger:
        """Create performance tracking context manager."""
        return PerformanceLogger(
            self.logger, operation, self.component, level, include_system_metrics
        )


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "structured",  # "structured" or "human"
    log_file: str | Path | None = None,
    enable_console: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 5,
) -> None:
    """
    Configure production logging with structured output.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Output format ("structured" for JSON, "human" for readable)
        log_file: Optional file path for log output
        enable_console: Whether to enable console output
        max_bytes: Maximum log file size before rotation
        backup_count: Number of backup files to keep
    """

    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Create formatters
    if log_format == "structured":
        formatter = StructuredFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        from logging.handlers import RotatingFileHandler

        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = RotatingFileHandler(log_path, maxBytes=max_bytes, backupCount=backup_count)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure third-party loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(component: str) -> MimirLoggerAdapter:
    """
    Get a component-specific logger with automatic context injection.

    Args:
        component: Component name for automatic context inclusion

    Returns:
        Enhanced logger adapter with structured logging methods
    """
    base_logger = logging.getLogger(f"mimir.{component}")
    return MimirLoggerAdapter(base_logger, component)


def set_request_context(
    request_id: str = None, session_id: str = None, user_id: str = None, pipeline_id: str = None
) -> None:
    """
    Set request context for correlation across log entries.

    Args:
        request_id: Unique request identifier
        session_id: Session identifier for grouping related requests
        user_id: User identifier for user-specific operations
        pipeline_id: Pipeline execution identifier
    """
    if request_id:
        REQUEST_ID.set(request_id)
    if session_id:
        SESSION_ID.set(session_id)
    if user_id:
        USER_ID.set(user_id)
    if pipeline_id:
        PIPELINE_ID.set(pipeline_id)


def clear_request_context() -> None:
    """Clear all request context variables."""
    REQUEST_ID.set(None)
    SESSION_ID.set(None)
    USER_ID.set(None)
    PIPELINE_ID.set(None)


def generate_request_id() -> str:
    """Generate a unique request ID using UUID."""
    return str(uuid.uuid4())


# Configure default logging on import
def _configure_default_logging():
    """Configure basic logging if not already configured."""
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        # Try to get configuration from centralized config first, fall back to environment
        try:
            from ..config import get_logging_config
            from ..config_migration import migration_tracker
            
            config = get_logging_config()
            setup_logging(
                log_level=config.log_level,
                log_format=config.log_format,
                log_file=config.log_file
            )
            
            # Mark this file as migrated
            migration_tracker.mark_migrated(__file__, ["logging"])
            
        except ImportError:
            # Centralized config not available, fall back to environment variables
            log_level = os.getenv("MIMIR_LOG_LEVEL", "INFO")
            log_format = os.getenv("MIMIR_LOG_FORMAT", "human")
            log_file = os.getenv("MIMIR_LOG_FILE")

            setup_logging(log_level=log_level, log_format=log_format, log_file=log_file)


# Initialize default logging
_configure_default_logging()


# Convenience functions for common logging patterns


def log_pipeline_start(pipeline_id: str, config: dict[str, Any]) -> None:
    """Log pipeline execution start with configuration."""
    logger = get_logger("pipeline")
    set_request_context(pipeline_id=pipeline_id)

    logger.info(
        "Pipeline execution started",
        extra={
            "operation": "pipeline_start",
            "extra_fields": {"pipeline_id": pipeline_id, "config": config},
        },
    )


def log_pipeline_complete(pipeline_id: str, duration_ms: float, stats: dict[str, Any]) -> None:
    """Log successful pipeline completion with stats."""
    logger = get_logger("pipeline")

    logger.info(
        f"Pipeline execution completed in {duration_ms:.2f}ms",
        extra={
            "operation": "pipeline_complete",
            "metrics": {"duration_ms": duration_ms},
            "extra_fields": {"pipeline_id": pipeline_id, "stats": stats},
        },
    )


def log_pipeline_error(pipeline_id: str, error: Exception, stage: str = None) -> None:
    """Log pipeline execution error."""
    logger = get_logger("pipeline")

    extra_fields = {"pipeline_id": pipeline_id}
    if stage:
        extra_fields["failed_stage"] = stage

    logger.operation_error("pipeline_execution", error, **extra_fields)


def log_external_tool_call(
    tool: str,
    command: list,
    duration_ms: float,
    exit_code: int,
    stdout_lines: int = 0,
    stderr_lines: int = 0,
) -> None:
    """Log external tool execution with metrics."""
    logger = get_logger("external_tools")

    level = logging.INFO if exit_code == 0 else logging.ERROR

    logger.log(
        level,
        f"External tool {tool} completed with exit code {exit_code}",
        extra={
            "operation": "external_tool_call",
            "metrics": {
                "duration_ms": duration_ms,
                "exit_code": exit_code,
                "stdout_lines": stdout_lines,
                "stderr_lines": stderr_lines,
            },
            "extra_fields": {
                "tool": tool,
                "command": (
                    command[:3] + ["..."] if len(command) > 3 else command
                ),  # Truncate long commands
                "success": exit_code == 0,
            },
        },
    )
