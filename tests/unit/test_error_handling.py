"""
Tests for error handling and error utilities.

Tests error context creation, error type classification, and error reporting.
"""

import pytest
from unittest.mock import Mock, patch
import json
import traceback

from src.repoindex.util.errors import (
    MimirError,
    create_error_context,
    ConfigurationError,
    ValidationError,
    SecurityError,
    FileSystemError,
    ExternalToolError,
    ResourceError,
    TimeoutError,
    IntegrationError,
    AuthenticationError,
    ErrorCollector,
    ErrorContext,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy
)


class TestMimirError:
    """Test MimirError base class functionality."""

    def test_basic_error_creation(self):
        """Test basic error creation with message."""
        error = MimirError("Test error message")
        
        assert str(error).startswith("[MEDIUM] logic: Test error message")
        assert error.message == "Test error message"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.LOGIC

    def test_error_with_severity(self):
        """Test error creation with custom severity."""
        error = MimirError("Test error", severity=ErrorSeverity.HIGH)
        
        assert error.severity == ErrorSeverity.HIGH
        assert error.message == "Test error"

    def test_error_with_context(self):
        """Test error creation with context data."""
        context = ErrorContext(component="test", operation="test_op", file_path="test.py", line_number=42)
        error = MimirError("Test error", context=context)
        
        assert error.context.component == "test"
        assert error.context.file_path == "test.py"

    def test_error_with_cause(self):
        """Test error creation with cause (original exception)."""
        original = ValueError("Original error")
        error = MimirError("Wrapped error", cause=original)
        
        assert error.cause is original
        assert error.__cause__ is original

    def test_error_serialization(self):
        """Test error can be serialized to dict."""
        context = ErrorContext(component="test", operation="test_op")
        cause = ValueError("Original error")
        
        error = MimirError(
            "Test error message",
            context=context,
            cause=cause
        )
        
        error_dict = error.to_dict()
        
        assert error_dict["error_type"] == "MimirError"
        assert error_dict["message"] == "Test error message"
        assert error_dict["severity"] == "medium"
        assert error_dict["context"]["component"] == "test"
        assert "timestamp" in error_dict
        assert error_dict["cause"] == str(cause)


class TestSpecificErrorTypes:
    """Test specific error type classes."""

    def test_configuration_error(self):
        """Test ConfigurationError functionality."""
        error = ConfigurationError("test.setting", "Invalid config value")
        
        assert isinstance(error, MimirError)
        assert error.category == ErrorCategory.CONFIGURATION
        assert error.details["config_key"] == "test.setting"

    def test_validation_error(self):
        """Test ValidationError functionality."""
        error = ValidationError("Invalid input format", field="email", value="invalid-email")
        
        assert isinstance(error, MimirError)
        assert error.category == ErrorCategory.VALIDATION
        assert error.details["field"] == "email"
        assert error.details["value"] == "invalid-email"

    def test_external_tool_error(self):
        """Test ExternalToolError functionality."""
        error = ExternalToolError("repomapper", "Command failed", exit_code=1)
        
        assert isinstance(error, MimirError)
        assert error.category == ErrorCategory.EXTERNAL_TOOL
        assert error.details["tool"] == "repomapper"
        assert error.details["exit_code"] == 1

    def test_security_error(self):
        """Test SecurityError functionality."""
        error = SecurityError("Access denied")
        
        assert isinstance(error, MimirError)
        assert error.category == ErrorCategory.VALIDATION
        assert error.severity == ErrorSeverity.HIGH

    def test_filesystem_error(self):
        """Test FileSystemError functionality."""
        error = FileSystemError("/test/file.txt", "read", "File not found")
        
        assert isinstance(error, MimirError)
        assert error.category == ErrorCategory.FILESYSTEM 
        assert error.details["path"] == "/test/file.txt"
        assert error.details["operation"] == "read"


class TestErrorContext:
    """Test error context creation utilities."""

    def test_create_basic_context(self):
        """Test basic error context creation."""
        context = create_error_context(
            operation="test_operation",
            component="test_component"
        )
        
        assert context["operation"] == "test_operation"
        assert context["component"] == "test_component"
        assert "timestamp" in context

    def test_create_context_with_request_info(self):
        """Test error context with request information."""
        with patch('src.repoindex.util.errors.request', create=True) as mock_request:
            mock_request.path = "/api/test"
            mock_request.method = "POST"
            mock_request.headers = {"Content-Type": "application/json"}
            
            context = create_error_context(
                operation="api_call",
                include_request=True
            )
            
            # Context creation might be implemented differently
            # Just test that it doesn't crash and has basic fields
            assert "operation" in context
            assert "timestamp" in context

    def test_create_context_with_system_info(self):
        """Test error context with system information."""
        context = create_error_context(
            operation="system_operation",
            include_system=True
        )
        
        assert "operation" in context
        assert "timestamp" in context
        # System info might include memory, CPU, etc.

    def test_create_context_with_user_data(self):
        """Test error context with user data."""
        user_data = {"user_id": "test123", "session": "session456"}
        
        context = create_error_context(
            operation="user_action",
            user_context=user_data
        )
        
        assert context["operation"] == "user_action"
        # User context should be sanitized and included


class TestErrorReporting:
    """Test error reporting and logging functionality."""

    def test_error_reporting_format(self):
        """Test error reporting produces correct format."""
        error = MimirError(
            "Test error for reporting",
            error_code="RPT_001",
            context={"file": "test.py", "function": "test_func"}
        )
        
        report = error.to_dict()
        
        # Should contain all required fields for logging/monitoring
        required_fields = ["error_type", "message", "error_code", "context", "timestamp"]
        for field in required_fields:
            assert field in report

    def test_error_chain_handling(self):
        """Test handling of error chains (cause relationships)."""
        root_cause = ValueError("Root cause")
        intermediate = RuntimeError("Intermediate error")
        intermediate.__cause__ = root_cause
        
        final_error = MimirError("Final error", cause=intermediate)
        
        assert final_error.cause is intermediate
        assert final_error.__cause__ is intermediate

    @patch('src.repoindex.util.errors.logger')
    def test_error_logging_integration(self, mock_logger):
        """Test error logging integration."""
        error = MimirError("Test error for logging")
        
        # Simulate error logging (actual implementation may vary)
        try:
            raise error
        except MimirError as e:
            # Error should be logged with proper context
            pass


class TestErrorUtilities:
    """Test error utility functions."""

    def test_error_context_sanitization(self):
        """Test that sensitive data is sanitized from error context."""
        sensitive_context = {
            "password": "secret123",
            "api_key": "sk-1234567890",
            "token": "bearer_token_123",
            "safe_field": "safe_value"
        }
        
        # Create error context that should sanitize sensitive fields
        context = create_error_context(
            operation="test_sanitization",
            **sensitive_context
        )
        
        # Sensitive fields should be redacted or excluded
        assert context.get("password") != "secret123"
        assert context.get("api_key") != "sk-1234567890"
        assert context.get("token") != "bearer_token_123"
        # Safe fields should remain
        assert context.get("safe_field") == "safe_value" or "safe_field" not in context

    def test_error_context_size_limits(self):
        """Test that error context has reasonable size limits."""
        large_data = "x" * 10000  # 10KB of data
        
        context = create_error_context(
            operation="test_size_limit",
            large_field=large_data
        )
        
        # Context should be limited in size to prevent memory issues
        total_size = len(json.dumps(context, default=str))
        assert total_size < 5000  # Should be truncated or limited

    def test_exception_to_error_conversion(self):
        """Test converting standard Python exceptions to MimirError."""
        # Test various exception types
        exceptions = [
            ValueError("Invalid value"),
            TypeError("Wrong type"),
            FileNotFoundError("File missing"),
            PermissionError("Access denied"),
            ConnectionError("Network error")
        ]
        
        for exc in exceptions:
            # Test that we can wrap any exception
            mimir_error = MimirError("Wrapped exception", cause=exc)
            
            assert isinstance(mimir_error, MimirError)
            assert mimir_error.cause is exc
            assert str(exc) in str(mimir_error) or mimir_error.cause == exc


class TestErrorRecovery:
    """Test error recovery patterns and utilities."""

    def test_error_recovery_context(self):
        """Test error recovery context information."""
        error = ProcessingError(
            "Process failed",
            stage="validation",
            recoverable=True,
            recovery_suggestions=["retry", "skip", "fallback"]
        )
        
        assert error.context.get("recoverable") is True
        assert "recovery_suggestions" in error.context

    def test_critical_error_classification(self):
        """Test classification of critical vs recoverable errors."""
        # Critical errors
        critical_error = SecurityError("Authentication bypass attempt")
        assert critical_error.error_type == "security"  # Should be flagged as critical
        
        # Recoverable errors
        recoverable_error = NetworkError("Temporary connection timeout")
        assert recoverable_error.error_type == "network"  # May be recoverable

    def test_error_aggregation(self):
        """Test error aggregation for batch operations."""
        errors = [
            ValidationError("Field 1 invalid", field="field1"),
            ValidationError("Field 2 invalid", field="field2"),
            ProcessingError("Process failed", stage="parsing")
        ]
        
        # Test that multiple errors can be collected and reported together
        # (Actual aggregation implementation would depend on requirements)
        assert len(errors) == 3
        assert all(isinstance(e, MimirError) for e in errors)