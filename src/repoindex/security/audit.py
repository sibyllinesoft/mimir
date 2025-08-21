"""
Security audit logging and event tracking.

Provides comprehensive security event logging, audit trails,
and security monitoring for the Mimir system.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from ..util.log import get_logger

logger = get_logger(__name__)


class SecurityEventType(Enum):
    """Security event types for audit logging."""

    # Authentication events
    AUTH_SUCCESS = "auth_success"
    AUTH_FAILURE = "auth_failure"
    AUTH_TOKEN_GENERATED = "auth_token_generated"
    AUTH_TOKEN_REVOKED = "auth_token_revoked"

    # Authorization events
    AUTHZ_SUCCESS = "authz_success"
    AUTHZ_FAILURE = "authz_failure"
    PERMISSION_GRANTED = "permission_granted"
    PERMISSION_DENIED = "permission_denied"

    # Input validation events
    INPUT_VALIDATION_FAILED = "input_validation_failed"
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    SUSPICIOUS_CONTENT_DETECTED = "suspicious_content_detected"
    MALICIOUS_PATTERN_DETECTED = "malicious_pattern_detected"

    # Rate limiting events
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    IP_BLOCKED = "ip_blocked"
    IP_UNBLOCKED = "ip_unblocked"
    ABUSE_DETECTED = "abuse_detected"

    # System access events
    FILE_ACCESS = "file_access"
    FILE_ACCESS_DENIED = "file_access_denied"
    REPOSITORY_INDEXED = "repository_indexed"
    SEARCH_PERFORMED = "search_performed"

    # Security configuration events
    SECURITY_CONFIG_CHANGED = "security_config_changed"
    SANDBOX_VIOLATION = "sandbox_violation"
    RESOURCE_LIMIT_EXCEEDED = "resource_limit_exceeded"

    # Credential events
    CREDENTIAL_DETECTED = "credential_detected"
    SECRET_ACCESSED = "secret_accessed"
    SECRET_MODIFIED = "secret_modified"

    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    ERROR_OCCURRED = "error_occurred"
    SECURITY_SCAN_COMPLETED = "security_scan_completed"


class SecurityEventSeverity(Enum):
    """Security event severity levels."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Represents a security event for audit logging."""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: SecurityEventType = SecurityEventType.ERROR_OCCURRED
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO
    timestamp: float = field(default_factory=time.time)

    # Event details
    message: str = ""
    description: str = ""

    # Context information
    user_id: str | None = None
    session_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None

    # Request information
    tool_name: str | None = None
    resource_path: str | None = None
    request_id: str | None = None

    # System information
    component: str | None = None
    operation: str | None = None

    # Additional metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    # Risk assessment
    risk_score: int | None = None  # 0-100 scale
    threat_indicators: list[str] = field(default_factory=list)

    # Resolution information
    resolved: bool = False
    resolution_notes: str | None = None
    resolved_at: float | None = None
    resolved_by: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        data = asdict(self)

        # Convert enums to strings
        data["event_type"] = self.event_type.value
        data["severity"] = self.severity.value

        # Convert timestamp to ISO format for readability
        data["timestamp_iso"] = datetime.fromtimestamp(self.timestamp, tz=UTC).isoformat()

        return data

    def mark_resolved(self, notes: str, resolved_by: str) -> None:
        """Mark the event as resolved.

        Args:
            notes: Resolution notes
            resolved_by: Who resolved the event
        """
        self.resolved = True
        self.resolution_notes = notes
        self.resolved_at = time.time()
        self.resolved_by = resolved_by

    def add_threat_indicator(self, indicator: str) -> None:
        """Add a threat indicator to the event.

        Args:
            indicator: Threat indicator description
        """
        if indicator not in self.threat_indicators:
            self.threat_indicators.append(indicator)

    def calculate_risk_score(self) -> int:
        """Calculate risk score based on event characteristics.

        Returns:
            Risk score from 0-100
        """
        base_scores = {
            SecurityEventSeverity.INFO: 10,
            SecurityEventSeverity.LOW: 25,
            SecurityEventSeverity.MEDIUM: 50,
            SecurityEventSeverity.HIGH: 75,
            SecurityEventSeverity.CRITICAL: 90,
        }

        score = base_scores.get(self.severity, 50)

        # Adjust based on event type
        high_risk_events = {
            SecurityEventType.AUTH_FAILURE,
            SecurityEventType.PATH_TRAVERSAL_ATTEMPT,
            SecurityEventType.MALICIOUS_PATTERN_DETECTED,
            SecurityEventType.SANDBOX_VIOLATION,
            SecurityEventType.CREDENTIAL_DETECTED,
        }

        if self.event_type in high_risk_events:
            score += 15

        # Adjust based on threat indicators
        score += min(len(self.threat_indicators) * 5, 20)

        # Cap at 100
        score = min(score, 100)

        self.risk_score = score
        return score


class AuditLogger:
    """Handles audit logging of security events."""

    def __init__(
        self,
        log_file: Path | None = None,
        max_file_size: int = 100 * 1024 * 1024,  # 100MB
        backup_count: int = 5,
        buffer_size: int = 100,
    ):
        """Initialize audit logger.

        Args:
            log_file: Path to audit log file
            max_file_size: Maximum log file size before rotation
            backup_count: Number of backup files to keep
            buffer_size: Number of events to buffer before writing
        """
        self.log_file = log_file
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        self.buffer_size = buffer_size

        # Event buffer for batch writing
        self.event_buffer: list[SecurityEvent] = []

        # Statistics
        self.events_logged = 0
        self.start_time = time.time()

        # Ensure log directory exists
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event: SecurityEvent) -> None:
        """Log a security event.

        Args:
            event: Security event to log
        """
        # Calculate risk score if not set
        if event.risk_score is None:
            event.calculate_risk_score()

        # Add to buffer
        self.event_buffer.append(event)
        self.events_logged += 1

        # Flush buffer if full or high severity event
        if len(self.event_buffer) >= self.buffer_size or event.severity in [
            SecurityEventSeverity.HIGH,
            SecurityEventSeverity.CRITICAL,
        ]:
            self.flush_buffer()

        # Log to standard logger based on severity
        log_data = event.to_dict()

        if event.severity == SecurityEventSeverity.CRITICAL:
            logger.critical("Security event", **log_data)
        elif event.severity == SecurityEventSeverity.HIGH:
            logger.error("Security event", **log_data)
        elif event.severity == SecurityEventSeverity.MEDIUM:
            logger.warning("Security event", **log_data)
        else:
            logger.info("Security event", **log_data)

    def flush_buffer(self) -> None:
        """Flush event buffer to file."""
        if not self.event_buffer or not self.log_file:
            return

        try:
            # Check if rotation is needed
            if self.log_file.exists() and self.log_file.stat().st_size > self.max_file_size:
                self._rotate_logs()

            # Write events to file
            with open(self.log_file, "a", encoding="utf-8") as f:
                for event in self.event_buffer:
                    json.dump(event.to_dict(), f, default=str)
                    f.write("\n")

            logger.debug("Audit buffer flushed", events_count=len(self.event_buffer))

        except Exception as e:
            logger.error("Failed to flush audit buffer", error=str(e))
        finally:
            self.event_buffer.clear()

    def _rotate_logs(self) -> None:
        """Rotate audit log files."""
        try:
            # Move existing backups
            for i in range(self.backup_count - 1, 0, -1):
                old_file = self.log_file.with_suffix(f".{i}")
                new_file = self.log_file.with_suffix(f".{i + 1}")

                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)

            # Move current log to .1
            if self.log_file.exists():
                backup_file = self.log_file.with_suffix(".1")
                self.log_file.rename(backup_file)

            logger.info("Audit logs rotated")

        except Exception as e:
            logger.error("Failed to rotate audit logs", error=str(e))

    def get_statistics(self) -> dict[str, Any]:
        """Get audit logging statistics.

        Returns:
            Dictionary with statistics
        """
        uptime = time.time() - self.start_time

        return {
            "events_logged": self.events_logged,
            "uptime_seconds": uptime,
            "events_per_second": self.events_logged / uptime if uptime > 0 else 0,
            "buffer_size": len(self.event_buffer),
            "log_file": str(self.log_file) if self.log_file else None,
            "log_file_size": (
                self.log_file.stat().st_size if self.log_file and self.log_file.exists() else 0
            ),
        }

    def query_events(
        self,
        event_type: SecurityEventType | None = None,
        severity: SecurityEventSeverity | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query audit events from log file.

        Args:
            event_type: Filter by event type
            severity: Filter by severity
            start_time: Filter events after this timestamp
            end_time: Filter events before this timestamp
            limit: Maximum number of events to return

        Returns:
            List of matching events
        """
        if not self.log_file or not self.log_file.exists():
            return []

        events = []

        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if len(events) >= limit:
                        break

                    try:
                        event_data = json.loads(line.strip())

                        # Apply filters
                        if event_type and event_data.get("event_type") != event_type.value:
                            continue

                        if severity and event_data.get("severity") != severity.value:
                            continue

                        if start_time and event_data.get("timestamp", 0) < start_time:
                            continue

                        if end_time and event_data.get("timestamp", 0) > end_time:
                            continue

                        events.append(event_data)

                    except json.JSONDecodeError:
                        continue

        except Exception as e:
            logger.error("Failed to query audit events", error=str(e))

        return events

    def close(self) -> None:
        """Close audit logger and flush remaining events."""
        self.flush_buffer()
        logger.info("Audit logger closed", total_events=self.events_logged)


class SecurityAuditor:
    """Main security auditor for the system."""

    def __init__(self, audit_log_file: Path | None = None):
        """Initialize security auditor.

        Args:
            audit_log_file: Path to audit log file
        """
        self.audit_logger = AuditLogger(audit_log_file)
        self.active_sessions: dict[str, dict[str, Any]] = {}
        self.threat_intelligence: dict[str, list[str]] = {}

        # Log system startup
        self.log_system_startup()

    def log_system_startup(self) -> None:
        """Log system startup event."""
        event = SecurityEvent(
            event_type=SecurityEventType.SYSTEM_STARTUP,
            severity=SecurityEventSeverity.INFO,
            message="Mimir security system started",
            component="security_auditor",
            operation="startup",
        )
        self.audit_logger.log_event(event)

    def log_authentication_attempt(
        self,
        success: bool,
        user_id: str | None = None,
        client_ip: str | None = None,
        user_agent: str | None = None,
        failure_reason: str | None = None,
    ) -> None:
        """Log authentication attempt.

        Args:
            success: Whether authentication succeeded
            user_id: User identifier
            client_ip: Client IP address
            user_agent: User agent string
            failure_reason: Reason for failure (if applicable)
        """
        event = SecurityEvent(
            event_type=(
                SecurityEventType.AUTH_SUCCESS if success else SecurityEventType.AUTH_FAILURE
            ),
            severity=SecurityEventSeverity.INFO if success else SecurityEventSeverity.MEDIUM,
            message=f"Authentication {'succeeded' if success else 'failed'}",
            user_id=user_id,
            client_ip=client_ip,
            user_agent=user_agent,
            component="authentication",
            operation="login_attempt",
        )

        if not success and failure_reason:
            event.metadata["failure_reason"] = failure_reason
            event.add_threat_indicator(f"auth_failure_{failure_reason}")

        self.audit_logger.log_event(event)

    def log_authorization_check(
        self,
        success: bool,
        user_id: str,
        tool_name: str,
        required_permissions: list[str],
        user_permissions: list[str],
        client_ip: str | None = None,
    ) -> None:
        """Log authorization check.

        Args:
            success: Whether authorization succeeded
            user_id: User identifier
            tool_name: Tool being accessed
            required_permissions: Required permissions
            user_permissions: User's permissions
            client_ip: Client IP address
        """
        event = SecurityEvent(
            event_type=(
                SecurityEventType.AUTHZ_SUCCESS if success else SecurityEventType.AUTHZ_FAILURE
            ),
            severity=SecurityEventSeverity.INFO if success else SecurityEventSeverity.MEDIUM,
            message=f"Authorization {'granted' if success else 'denied'} for {tool_name}",
            user_id=user_id,
            client_ip=client_ip,
            tool_name=tool_name,
            component="authorization",
            operation="permission_check",
            metadata={
                "required_permissions": required_permissions,
                "user_permissions": user_permissions,
            },
        )

        if not success:
            missing_perms = set(required_permissions) - set(user_permissions)
            event.metadata["missing_permissions"] = list(missing_perms)
            event.add_threat_indicator("insufficient_permissions")

        self.audit_logger.log_event(event)

    def log_input_validation_failure(
        self,
        validation_type: str,
        input_value: str,
        violation_details: dict[str, Any],
        client_ip: str | None = None,
        user_id: str | None = None,
    ) -> None:
        """Log input validation failure.

        Args:
            validation_type: Type of validation that failed
            input_value: Input value (truncated for security)
            violation_details: Details of the validation failure
            client_ip: Client IP address
            user_id: User identifier
        """
        severity = (
            SecurityEventSeverity.HIGH
            if "traversal" in validation_type
            else SecurityEventSeverity.MEDIUM
        )

        event = SecurityEvent(
            event_type=SecurityEventType.INPUT_VALIDATION_FAILED,
            severity=severity,
            message=f"Input validation failed: {validation_type}",
            user_id=user_id,
            client_ip=client_ip,
            component="input_validation",
            operation=validation_type,
            metadata={
                "input_value": input_value[:100],  # Truncate for security
                "violation_details": violation_details,
            },
        )

        # Add specific threat indicators
        if "traversal" in validation_type:
            event.event_type = SecurityEventType.PATH_TRAVERSAL_ATTEMPT
            event.add_threat_indicator("path_traversal")

        if "malicious" in validation_type:
            event.event_type = SecurityEventType.MALICIOUS_PATTERN_DETECTED
            event.add_threat_indicator("malicious_pattern")

        self.audit_logger.log_event(event)

    def log_rate_limit_exceeded(
        self,
        client_ip: str,
        limit_type: str,
        limit_details: dict[str, Any],
        user_id: str | None = None,
    ) -> None:
        """Log rate limit exceeded event.

        Args:
            client_ip: Client IP address
            limit_type: Type of rate limit exceeded
            limit_details: Details of the rate limit
            user_id: User identifier
        """
        event = SecurityEvent(
            event_type=SecurityEventType.RATE_LIMIT_EXCEEDED,
            severity=SecurityEventSeverity.MEDIUM,
            message=f"Rate limit exceeded: {limit_type}",
            user_id=user_id,
            client_ip=client_ip,
            component="rate_limiter",
            operation="rate_check",
            metadata=limit_details,
        )

        event.add_threat_indicator("rate_limit_abuse")
        self.audit_logger.log_event(event)

    def log_credential_detected(
        self,
        file_path: str,
        credential_type: str,
        severity: str,
        line_number: int,
        context: list[str],
    ) -> None:
        """Log detected credential in code.

        Args:
            file_path: File containing the credential
            credential_type: Type of credential detected
            severity: Severity of the finding
            line_number: Line number where credential was found
            context: Surrounding code context
        """
        severity_map = {
            "low": SecurityEventSeverity.LOW,
            "medium": SecurityEventSeverity.MEDIUM,
            "high": SecurityEventSeverity.HIGH,
            "critical": SecurityEventSeverity.CRITICAL,
        }

        event = SecurityEvent(
            event_type=SecurityEventType.CREDENTIAL_DETECTED,
            severity=severity_map.get(severity, SecurityEventSeverity.MEDIUM),
            message=f"Credential detected in code: {credential_type}",
            resource_path=file_path,
            component="credential_scanner",
            operation="scan_file",
            metadata={
                "credential_type": credential_type,
                "line_number": line_number,
                "context_lines": len(context),
                "file_extension": Path(file_path).suffix,
            },
        )

        event.add_threat_indicator("credential_exposure")
        self.audit_logger.log_event(event)

    def log_file_access(
        self,
        file_path: str,
        operation: str,
        success: bool,
        user_id: str | None = None,
        client_ip: str | None = None,
        error_message: str | None = None,
    ) -> None:
        """Log file access attempt.

        Args:
            file_path: Path to file accessed
            operation: Operation performed
            success: Whether access succeeded
            user_id: User identifier
            client_ip: Client IP address
            error_message: Error message if access failed
        """
        event = SecurityEvent(
            event_type=(
                SecurityEventType.FILE_ACCESS if success else SecurityEventType.FILE_ACCESS_DENIED
            ),
            severity=SecurityEventSeverity.INFO if success else SecurityEventSeverity.MEDIUM,
            message=f"File {operation} {'succeeded' if success else 'failed'}",
            user_id=user_id,
            client_ip=client_ip,
            resource_path=file_path,
            component="file_system",
            operation=operation,
        )

        if not success and error_message:
            event.metadata["error_message"] = error_message
            event.add_threat_indicator("file_access_denied")

        self.audit_logger.log_event(event)

    def log_sandbox_violation(
        self,
        violation_type: str,
        details: dict[str, Any],
        user_id: str | None = None,
        client_ip: str | None = None,
    ) -> None:
        """Log sandbox violation.

        Args:
            violation_type: Type of sandbox violation
            details: Violation details
            user_id: User identifier
            client_ip: Client IP address
        """
        event = SecurityEvent(
            event_type=SecurityEventType.SANDBOX_VIOLATION,
            severity=SecurityEventSeverity.HIGH,
            message=f"Sandbox violation: {violation_type}",
            user_id=user_id,
            client_ip=client_ip,
            component="sandbox",
            operation="security_check",
            metadata=details,
        )

        event.add_threat_indicator("sandbox_escape_attempt")
        self.audit_logger.log_event(event)

    def generate_security_report(self, hours: int = 24) -> dict[str, Any]:
        """Generate a security report for the specified time period.

        Args:
            hours: Number of hours to include in the report

        Returns:
            Security report dictionary
        """
        end_time = time.time()
        start_time = end_time - (hours * 3600)

        # Query events for the time period
        events = self.audit_logger.query_events(
            start_time=start_time, end_time=end_time, limit=10000
        )

        # Analyze events
        event_counts = {}
        severity_counts = {}
        threat_indicators = {}
        top_ips = {}

        for event_data in events:
            event_type = event_data.get("event_type", "unknown")
            severity = event_data.get("severity", "unknown")
            client_ip = event_data.get("client_ip")
            threats = event_data.get("threat_indicators", [])

            # Count by event type
            event_counts[event_type] = event_counts.get(event_type, 0) + 1

            # Count by severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1

            # Count threat indicators
            for threat in threats:
                threat_indicators[threat] = threat_indicators.get(threat, 0) + 1

            # Count by IP
            if client_ip:
                top_ips[client_ip] = top_ips.get(client_ip, 0) + 1

        # Sort and limit results
        top_event_types = sorted(event_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_threats = sorted(threat_indicators.items(), key=lambda x: x[1], reverse=True)[:10]
        top_client_ips = sorted(top_ips.items(), key=lambda x: x[1], reverse=True)[:10]

        return {
            "report_period": {"start_time": start_time, "end_time": end_time, "hours": hours},
            "summary": {
                "total_events": len(events),
                "unique_ips": len(top_ips),
                "threat_indicators": len(threat_indicators),
            },
            "event_breakdown": {"by_type": dict(top_event_types), "by_severity": severity_counts},
            "top_threats": dict(top_threats),
            "top_client_ips": dict(top_client_ips),
            "recommendations": self._generate_recommendations(events),
        }

    def _generate_recommendations(self, events: list[dict[str, Any]]) -> list[str]:
        """Generate security recommendations based on events.

        Args:
            events: List of security events

        Returns:
            List of recommendations
        """
        recommendations = []

        # Count different types of security events
        auth_failures = sum(1 for e in events if e.get("event_type") == "auth_failure")
        rate_limit_events = sum(1 for e in events if e.get("event_type") == "rate_limit_exceeded")
        credential_detections = sum(
            1 for e in events if e.get("event_type") == "credential_detected"
        )
        input_validation_failures = sum(
            1 for e in events if e.get("event_type") == "input_validation_failed"
        )

        if auth_failures > 10:
            recommendations.append(
                "High number of authentication failures detected. Consider implementing account lockout policies."
            )

        if rate_limit_events > 5:
            recommendations.append(
                "Multiple rate limit violations detected. Review rate limiting configuration."
            )

        if credential_detections > 0:
            recommendations.append(
                "Credentials detected in code. Implement secrets scanning in CI/CD pipeline."
            )

        if input_validation_failures > 20:
            recommendations.append(
                "High number of input validation failures. Review and strengthen input validation."
            )

        # Check for suspicious IPs
        ip_counts = {}
        for event in events:
            ip = event.get("client_ip")
            if ip:
                ip_counts[ip] = ip_counts.get(ip, 0) + 1

        suspicious_ips = [ip for ip, count in ip_counts.items() if count > 50]
        if suspicious_ips:
            recommendations.append(
                f"Suspicious activity from {len(suspicious_ips)} IP addresses. Consider IP blocking."
            )

        if not recommendations:
            recommendations.append("No major security issues detected in the analyzed period.")

        return recommendations

    def close(self) -> None:
        """Close security auditor and flush logs."""
        # Log system shutdown
        event = SecurityEvent(
            event_type=SecurityEventType.SYSTEM_SHUTDOWN,
            severity=SecurityEventSeverity.INFO,
            message="Mimir security system shutting down",
            component="security_auditor",
            operation="shutdown",
        )
        self.audit_logger.log_event(event)

        # Close audit logger
        self.audit_logger.close()


# Global security auditor instance
_security_auditor: SecurityAuditor | None = None


def get_security_auditor() -> SecurityAuditor:
    """Get the global security auditor instance.

    Returns:
        Global security auditor instance
    """
    global _security_auditor
    if _security_auditor is None:
        # Use default audit log location
        audit_log_path = Path.home() / ".cache" / "mimir" / "security" / "audit.log"
        _security_auditor = SecurityAuditor(audit_log_path)
    return _security_auditor


def configure_security_auditor(audit_log_file: Path | None = None) -> None:
    """Configure the global security auditor instance.

    Args:
        audit_log_file: Path to audit log file
    """
    global _security_auditor
    if _security_auditor:
        _security_auditor.close()
    _security_auditor = SecurityAuditor(audit_log_file)
