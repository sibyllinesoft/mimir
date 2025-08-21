"""
Comprehensive error handling and recovery system.

Provides structured error types, recovery strategies, and error context
preservation for production-grade error handling across the pipeline.
"""

import traceback
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ErrorSeverity(str, Enum):
    """Error severity levels for categorization and response."""

    CRITICAL = "critical"  # System cannot continue, immediate intervention required
    HIGH = "high"  # Major functionality impacted, urgent fix needed
    MEDIUM = "medium"  # Functionality degraded, fix needed soon
    LOW = "low"  # Minor issues, can be addressed later
    INFO = "info"  # Informational, no action required


class ErrorCategory(str, Enum):
    """Error categories for systematic handling."""

    VALIDATION = "validation"  # Input validation failures
    EXTERNAL_TOOL = "external_tool"  # External tool failures (repomapper, serena, etc.)
    FILESYSTEM = "filesystem"  # File system operations
    NETWORK = "network"  # Network connectivity issues
    PERMISSION = "permission"  # Permission/access control
    CONFIGURATION = "configuration"  # Configuration errors
    RESOURCE = "resource"  # Resource exhaustion (memory, disk, etc.)
    LOGIC = "logic"  # Internal logic errors
    INTEGRATION = "integration"  # Integration failures between components
    TIMEOUT = "timeout"  # Operation timeouts


class RecoveryStrategy(str, Enum):
    """Recovery strategies for different error types."""

    RETRY = "retry"  # Retry the operation
    FALLBACK = "fallback"  # Use alternative approach
    SKIP = "skip"  # Skip and continue
    ABORT = "abort"  # Stop processing
    ESCALATE = "escalate"  # Require human intervention
    IGNORE = "ignore"  # Log but continue


@dataclass
class ErrorContext:
    """Rich context information for error analysis."""

    component: str
    operation: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    file_path: str | None = None
    line_number: int | None = None
    function_name: str | None = None

    # System context
    python_version: str | None = None
    platform: str | None = None
    memory_usage: int | None = None
    disk_usage: int | None = None

    # Operation context
    parameters: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)


class MimirError(Exception):
    """
    Base exception class with structured error handling.

    Provides rich error context, recovery suggestions, and structured
    logging integration for production error handling.
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        category: ErrorCategory = ErrorCategory.LOGIC,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.ABORT,
        context: ErrorContext | None = None,
        cause: Exception | None = None,
        details: dict[str, Any] | None = None,
        suggestions: list[str] | None = None,
    ):
        super().__init__(message)
        self.message = message
        self.severity = severity
        self.category = category
        self.recovery_strategy = recovery_strategy
        self.context = context or ErrorContext(component="unknown", operation="unknown")
        self.cause = cause
        self.details = details or {}
        self.suggestions = suggestions or []
        self.timestamp = datetime.now(UTC)
        self.traceback_str = traceback.format_exc()

    def to_dict(self) -> dict[str, Any]:
        """Convert error to structured dictionary for logging."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recovery_strategy": self.recovery_strategy.value,
            "timestamp": self.timestamp.isoformat(),
            "context": {
                "component": self.context.component,
                "operation": self.context.operation,
                "timestamp": self.context.timestamp.isoformat(),
                "file_path": self.context.file_path,
                "line_number": self.context.line_number,
                "function_name": self.context.function_name,
                "parameters": self.context.parameters,
                "state": self.context.state,
                "metrics": self.context.metrics,
            },
            "details": self.details,
            "suggestions": self.suggestions,
            "cause": str(self.cause) if self.cause else None,
            "traceback": self.traceback_str,
        }

    def __str__(self) -> str:
        """Human-readable error representation."""
        parts = [
            f"[{self.severity.value.upper()}] {self.category.value}: {self.message}",
            f"Component: {self.context.component}",
            f"Operation: {self.context.operation}",
        ]

        if self.context.file_path:
            parts.append(f"File: {self.context.file_path}")

        if self.suggestions:
            parts.append("Suggestions:")
            for suggestion in self.suggestions:
                parts.append(f"  - {suggestion}")

        return "\n".join(parts)


# Specific error types for different components


class ValidationError(MimirError):
    """Input validation errors."""

    def __init__(self, message: str, field: str = None, value: Any = None, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs,
        )
        if field:
            self.details["field"] = field
        if value is not None:
            self.details["value"] = str(value)


class ExternalToolError(MimirError):
    """External tool execution errors."""

    def __init__(self, tool: str, message: str, exit_code: int = None, **kwargs):
        super().__init__(
            f"{tool}: {message}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.EXTERNAL_TOOL,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs,
        )
        self.details["tool"] = tool
        if exit_code is not None:
            self.details["exit_code"] = exit_code


class FileSystemError(MimirError):
    """File system operation errors."""

    def __init__(self, path: str | Path, operation: str, message: str, **kwargs):
        super().__init__(
            f"File operation '{operation}' failed on '{path}': {message}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.FILESYSTEM,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs,
        )
        self.details["path"] = str(path)
        self.details["operation"] = operation


class ConfigurationError(MimirError):
    """Configuration and setup errors."""

    def __init__(self, config_key: str, message: str, **kwargs):
        super().__init__(
            f"Configuration error for '{config_key}': {message}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.CONFIGURATION,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs,
        )
        self.details["config_key"] = config_key


class ResourceError(MimirError):
    """Resource exhaustion errors."""

    def __init__(self, resource: str, current: str, limit: str = None, **kwargs):
        message = f"Resource '{resource}' exhausted: {current}"
        if limit:
            message += f" (limit: {limit})"

        super().__init__(
            message,
            severity=ErrorSeverity.CRITICAL,
            category=ErrorCategory.RESOURCE,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            **kwargs,
        )
        self.details["resource"] = resource
        self.details["current"] = current
        if limit:
            self.details["limit"] = limit


class TimeoutError(MimirError):
    """Operation timeout errors."""

    def __init__(self, operation: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"Operation '{operation}' timed out after {timeout_seconds} seconds",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.TIMEOUT,
            recovery_strategy=RecoveryStrategy.RETRY,
            **kwargs,
        )
        self.details["operation"] = operation
        self.details["timeout_seconds"] = timeout_seconds


class IntegrationError(MimirError):
    """Component integration errors."""

    def __init__(self, source: str, target: str, message: str, **kwargs):
        super().__init__(
            f"Integration error between '{source}' and '{target}': {message}",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.INTEGRATION,
            recovery_strategy=RecoveryStrategy.FALLBACK,
            **kwargs,
        )
        self.details["source"] = source
        self.details["target"] = target


class SecurityError(MimirError):
    """Base class for security-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.VALIDATION,
            recovery_strategy=RecoveryStrategy.ABORT,
            **kwargs,
        )


class AuthenticationError(SecurityError):
    """Authentication failures and errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(f"Authentication failed: {message}", **kwargs)


# Error aggregation for batch operations


class ErrorCollector:
    """Collects and aggregates errors from batch operations."""

    def __init__(self):
        self.errors: list[MimirError] = []
        self.warnings: list[MimirError] = []

    def add_error(self, error: MimirError) -> None:
        """Add an error to the collection."""
        if error.severity in (ErrorSeverity.CRITICAL, ErrorSeverity.HIGH):
            self.errors.append(error)
        else:
            self.warnings.append(error)

    def add_exception(self, exc: Exception, context: ErrorContext) -> None:
        """Convert exception to MimirError and add to collection."""
        if isinstance(exc, MimirError):
            self.add_error(exc)
        else:
            error = MimirError(message=str(exc), context=context, cause=exc)
            self.add_error(error)

    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if any warnings were collected."""
        return len(self.warnings) > 0

    def get_summary(self) -> dict[str, Any]:
        """Get summary of collected errors and warnings."""
        return {
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "errors": [error.to_dict() for error in self.errors],
            "warnings": [warning.to_dict() for warning in self.warnings],
            "severities": {
                severity.value: len(
                    [e for e in self.errors + self.warnings if e.severity == severity]
                )
                for severity in ErrorSeverity
            },
            "categories": {
                category.value: len(
                    [e for e in self.errors + self.warnings if e.category == category]
                )
                for category in ErrorCategory
            },
        }

    def create_aggregate_error(self) -> MimirError | None:
        """Create a single error summarizing all collected errors."""
        if not self.has_errors():
            return None

        if len(self.errors) == 1:
            return self.errors[0]

        summary = self.get_summary()
        message = f"Multiple errors occurred: {summary['error_count']} errors, {summary['warning_count']} warnings"

        return MimirError(
            message=message,
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.LOGIC,
            recovery_strategy=RecoveryStrategy.ESCALATE,
            details=summary,
        )


# Utility functions for error handling


def create_error_context(component: str, operation: str, **kwargs) -> ErrorContext:
    """Create error context with automatic stack frame detection."""
    import inspect
    import platform
    import sys

    import psutil

    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_frame = frame.f_back
        context = ErrorContext(
            component=component,
            operation=operation,
            file_path=caller_frame.f_code.co_filename,
            line_number=caller_frame.f_lineno,
            function_name=caller_frame.f_code.co_name,
            python_version=sys.version,
            platform=platform.platform(),
            **kwargs,
        )

        # Add system metrics if available
        try:
            process = psutil.Process()
            context.memory_usage = process.memory_info().rss
            context.disk_usage = psutil.disk_usage("/").used
        except:
            pass  # Ignore if psutil not available or permission denied

        return context

    return ErrorContext(component=component, operation=operation, **kwargs)


def handle_external_tool_error(
    tool: str, command: list[str], exit_code: int, stdout: str, stderr: str, context: ErrorContext
) -> ExternalToolError:
    """Create structured error for external tool failures."""
    suggestions = []

    if exit_code == 127:
        suggestions.append(f"Check if {tool} is installed and in PATH")
    elif exit_code == 126:
        suggestions.append(f"Check if {tool} has execute permissions")
    elif "permission denied" in stderr.lower():
        suggestions.append("Check file permissions and access rights")
    elif "no such file" in stderr.lower():
        suggestions.append("Check if input files exist")
    elif "timeout" in stderr.lower():
        suggestions.append("Increase timeout or check system load")

    return ExternalToolError(
        tool=tool,
        message=f"Command failed with exit code {exit_code}",
        exit_code=exit_code,
        context=context,
        details={
            "command": " ".join(command),
            "stdout": stdout[:1000] if stdout else "",  # Truncate for logging
            "stderr": stderr[:1000] if stderr else "",
        },
        suggestions=suggestions,
    )


def with_error_handling(component: str, operation: str, **context_kwargs):
    """Decorator for automatic error handling and context creation."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            context = create_error_context(component, operation, **context_kwargs)
            try:
                return func(*args, **kwargs)
            except MimirError:
                raise  # Re-raise structured errors as-is
            except Exception as e:
                # Convert unhandled exceptions to structured errors
                raise MimirError(
                    message=f"Unexpected error in {operation}: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    category=ErrorCategory.LOGIC,
                    context=context,
                    cause=e,
                )

        return wrapper

    return decorator
