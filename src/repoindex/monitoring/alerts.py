"""
Alerting system for Mimir Deep Code Research System.

Provides alerting rules and conditions for monitoring critical system health,
performance degradation, and operational issues.
"""

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from ..util.logging_config import get_logger

logger = get_logger("monitoring.alerts")


class AlertSeverity(Enum):
    """Alert severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AlertStatus(Enum):
    """Alert status states."""

    FIRING = "firing"
    RESOLVED = "resolved"
    PENDING = "pending"
    SUPPRESSED = "suppressed"


@dataclass
class AlertRule:
    """Definition of an alerting rule."""

    name: str
    description: str
    severity: AlertSeverity
    condition: Callable[[], bool]
    threshold_value: float | None = None
    comparison_operator: str = ">"  # >, <, >=, <=, ==, !=
    evaluation_interval: int = 60  # seconds
    for_duration: int = 300  # seconds - how long condition must be true
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    enabled: bool = True

    # Runtime state
    last_evaluation: datetime | None = None
    condition_start_time: datetime | None = None
    firing_since: datetime | None = None
    status: AlertStatus = AlertStatus.PENDING


@dataclass
class Alert:
    """Active alert instance."""

    rule_name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    starts_at: datetime | None = None
    ends_at: datetime | None = None
    value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "labels": self.labels,
            "annotations": self.annotations,
            "starts_at": self.starts_at.isoformat() if self.starts_at else None,
            "ends_at": self.ends_at.isoformat() if self.ends_at else None,
            "value": self.value,
        }


class AlertManager:
    """
    Alert manager for monitoring system health and performance.

    Evaluates alerting rules, manages alert lifecycle, and provides
    notification mechanisms for critical system events.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.rules: dict[str, AlertRule] = {}
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: list[Alert] = []
        self.max_history_size = 1000
        self.running = False
        self.evaluation_task: asyncio.Task | None = None
        self._lock = threading.Lock()

        # Notification handlers
        self.notification_handlers: list[Callable[[Alert], None]] = []

        # Initialize default rules
        self._register_default_rules()

        logger.info("Alert manager initialized")

    def _register_default_rules(self):
        """Register default alerting rules for common scenarios."""

        # Pipeline failure rate
        self.register_rule(
            AlertRule(
                name="pipeline_failure_rate_high",
                description="Pipeline failure rate is high",
                severity=AlertSeverity.HIGH,
                condition=lambda: self._get_pipeline_failure_rate() > 0.1,
                threshold_value=0.1,
                evaluation_interval=60,
                for_duration=300,
                labels={"component": "pipeline", "type": "failure_rate"},
                annotations={
                    "summary": "Pipeline failure rate above 10%",
                    "description": "More than 10% of pipelines are failing in the last 10 minutes",
                },
            )
        )

        # Memory usage
        self.register_rule(
            AlertRule(
                name="memory_usage_critical",
                description="Memory usage is critically high",
                severity=AlertSeverity.CRITICAL,
                condition=lambda: self._get_memory_usage_percent() > 90,
                threshold_value=90,
                evaluation_interval=30,
                for_duration=120,
                labels={"component": "system", "type": "memory"},
                annotations={
                    "summary": "Memory usage above 90%",
                    "description": "System memory usage is critically high",
                },
            )
        )

        # CPU usage
        self.register_rule(
            AlertRule(
                name="cpu_usage_high",
                description="CPU usage is high",
                severity=AlertSeverity.MEDIUM,
                condition=lambda: self._get_cpu_usage_percent() > 80,
                threshold_value=80,
                evaluation_interval=60,
                for_duration=300,
                labels={"component": "system", "type": "cpu"},
                annotations={
                    "summary": "CPU usage above 80%",
                    "description": "System CPU usage is high for extended period",
                },
            )
        )

        # Disk space
        self.register_rule(
            AlertRule(
                name="disk_space_low",
                description="Disk space is running low",
                severity=AlertSeverity.HIGH,
                condition=lambda: self._get_disk_free_percent() < 10,
                threshold_value=10,
                comparison_operator="<",
                evaluation_interval=300,
                for_duration=600,
                labels={"component": "system", "type": "disk"},
                annotations={
                    "summary": "Disk space below 10%",
                    "description": "Available disk space is critically low",
                },
            )
        )

        # Search performance
        self.register_rule(
            AlertRule(
                name="search_latency_high",
                description="Search latency is high",
                severity=AlertSeverity.MEDIUM,
                condition=lambda: self._get_search_p95_latency() > 2.0,
                threshold_value=2.0,
                evaluation_interval=120,
                for_duration=300,
                labels={"component": "search", "type": "latency"},
                annotations={
                    "summary": "Search P95 latency above 2 seconds",
                    "description": "Search requests are taking longer than expected",
                },
            )
        )

        # Error rate
        self.register_rule(
            AlertRule(
                name="error_rate_high",
                description="Overall error rate is high",
                severity=AlertSeverity.HIGH,
                condition=lambda: self._get_error_rate() > 0.05,
                threshold_value=0.05,
                evaluation_interval=60,
                for_duration=180,
                labels={"component": "application", "type": "error_rate"},
                annotations={
                    "summary": "Error rate above 5%",
                    "description": "Application error rate is higher than acceptable threshold",
                },
            )
        )

        # MCP connection health
        self.register_rule(
            AlertRule(
                name="mcp_connections_low",
                description="MCP active connections are low",
                severity=AlertSeverity.MEDIUM,
                condition=lambda: self._get_mcp_active_connections() == 0,
                threshold_value=0,
                comparison_operator="==",
                evaluation_interval=120,
                for_duration=300,
                labels={"component": "mcp", "type": "connections"},
                annotations={
                    "summary": "No active MCP connections",
                    "description": "MCP server has no active client connections",
                },
            )
        )

        # Pipeline queue backup
        self.register_rule(
            AlertRule(
                name="pipeline_queue_backup",
                description="Pipeline queue is backing up",
                severity=AlertSeverity.MEDIUM,
                condition=lambda: self._get_pipeline_queue_length() > 10,
                threshold_value=10,
                evaluation_interval=60,
                for_duration=300,
                labels={"component": "pipeline", "type": "queue"},
                annotations={
                    "summary": "Pipeline queue length above 10",
                    "description": "Pipeline processing queue is backing up",
                },
            )
        )

    def register_rule(self, rule: AlertRule):
        """Register a new alerting rule."""
        with self._lock:
            self.rules[rule.name] = rule
            logger.info(f"Registered alert rule: {rule.name} (severity: {rule.severity.value})")

    def unregister_rule(self, rule_name: str) -> bool:
        """Unregister an alerting rule."""
        with self._lock:
            if rule_name in self.rules:
                del self.rules[rule_name]
                # Resolve any active alerts for this rule
                if rule_name in self.active_alerts:
                    self._resolve_alert(rule_name)
                logger.info(f"Unregistered alert rule: {rule_name}")
                return True
            return False

    def enable_rule(self, rule_name: str) -> bool:
        """Enable an alerting rule."""
        with self._lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = True
                logger.info(f"Enabled alert rule: {rule_name}")
                return True
            return False

    def disable_rule(self, rule_name: str) -> bool:
        """Disable an alerting rule."""
        with self._lock:
            if rule_name in self.rules:
                self.rules[rule_name].enabled = False
                # Resolve any active alerts for this rule
                if rule_name in self.active_alerts:
                    self._resolve_alert(rule_name)
                logger.info(f"Disabled alert rule: {rule_name}")
                return True
            return False

    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts."""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")

    def start(self):
        """Start the alert evaluation loop."""
        if self.running:
            return

        self.running = True
        self.evaluation_task = asyncio.create_task(self._evaluation_loop())
        logger.info("Alert manager started")

    async def stop(self):
        """Stop the alert evaluation loop."""
        self.running = False
        if self.evaluation_task:
            self.evaluation_task.cancel()
            try:
                await self.evaluation_task
            except asyncio.CancelledError:
                pass
        logger.info("Alert manager stopped")

    async def _evaluation_loop(self):
        """Main evaluation loop for checking alert conditions."""
        while self.running:
            try:
                await self._evaluate_rules()
                await asyncio.sleep(10)  # Check every 10 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in alert evaluation loop", error=str(e))
                await asyncio.sleep(30)  # Wait longer on error

    async def _evaluate_rules(self):
        """Evaluate all enabled alerting rules."""
        current_time = datetime.now()

        with self._lock:
            rules_to_evaluate = list(self.rules.values())

        for rule in rules_to_evaluate:
            if not rule.enabled:
                continue

            # Check if it's time to evaluate this rule
            if (
                rule.last_evaluation
                and (current_time - rule.last_evaluation).total_seconds() < rule.evaluation_interval
            ):
                continue

            try:
                await self._evaluate_rule(rule, current_time)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}", error=str(e))

    async def _evaluate_rule(self, rule: AlertRule, current_time: datetime):
        """Evaluate a single alerting rule."""
        rule.last_evaluation = current_time

        # Evaluate condition
        try:
            condition_met = rule.condition()
        except Exception as e:
            logger.error(f"Error evaluating condition for rule {rule.name}", error=str(e))
            return

        if condition_met:
            # Condition is true
            if rule.condition_start_time is None:
                rule.condition_start_time = current_time
                rule.status = AlertStatus.PENDING
                logger.debug(f"Alert condition started for rule {rule.name}")

            # Check if condition has been true long enough
            condition_duration = (current_time - rule.condition_start_time).total_seconds()
            if condition_duration >= rule.for_duration:
                if rule.name not in self.active_alerts:
                    # Fire new alert
                    self._fire_alert(rule, current_time)
                else:
                    # Update existing alert
                    self.active_alerts[rule.name].status = AlertStatus.FIRING
        else:
            # Condition is false
            if rule.condition_start_time is not None:
                rule.condition_start_time = None
                rule.status = AlertStatus.PENDING

            # Resolve any active alert
            if rule.name in self.active_alerts:
                self._resolve_alert(rule.name, current_time)

    def _fire_alert(self, rule: AlertRule, current_time: datetime):
        """Fire a new alert."""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            status=AlertStatus.FIRING,
            message=rule.description,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
            starts_at=current_time,
            value=rule.threshold_value,
        )

        self.active_alerts[rule.name] = alert
        rule.firing_since = current_time
        rule.status = AlertStatus.FIRING

        # Add to history
        self.alert_history.append(alert)
        self._trim_history()

        logger.warning(
            f"Alert fired: {rule.name}", severity=rule.severity.value, message=rule.description
        )

        # Send notifications
        self._send_notifications(alert)

    def _resolve_alert(self, rule_name: str, current_time: datetime | None = None):
        """Resolve an active alert."""
        if rule_name not in self.active_alerts:
            return

        resolve_time = current_time or datetime.now()
        alert = self.active_alerts[rule_name]
        alert.status = AlertStatus.RESOLVED
        alert.ends_at = resolve_time

        # Update rule state
        if rule_name in self.rules:
            self.rules[rule_name].firing_since = None
            self.rules[rule_name].status = AlertStatus.RESOLVED

        # Remove from active alerts
        del self.active_alerts[rule_name]

        logger.info(f"Alert resolved: {rule_name}")

        # Send resolution notification
        self._send_notifications(alert)

    def _send_notifications(self, alert: Alert):
        """Send alert notifications to all handlers."""
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error("Error sending alert notification", error=str(e))

    def _trim_history(self):
        """Trim alert history to maximum size."""
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size :]

    def get_active_alerts(self) -> list[Alert]:
        """Get all currently active alerts."""
        with self._lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: int = 100) -> list[Alert]:
        """Get recent alert history."""
        with self._lock:
            return self.alert_history[-limit:]

    def get_rules_status(self) -> dict[str, dict[str, Any]]:
        """Get status of all rules."""
        with self._lock:
            return {
                name: {
                    "enabled": rule.enabled,
                    "status": rule.status.value,
                    "last_evaluation": (
                        rule.last_evaluation.isoformat() if rule.last_evaluation else None
                    ),
                    "firing_since": rule.firing_since.isoformat() if rule.firing_since else None,
                    "severity": rule.severity.value,
                    "description": rule.description,
                }
                for name, rule in self.rules.items()
            }

    def force_evaluation(self, rule_name: str | None = None):
        """Force immediate evaluation of rules."""
        current_time = datetime.now()

        if rule_name:
            if rule_name in self.rules:
                asyncio.create_task(self._evaluate_rule(self.rules[rule_name], current_time))
        else:
            asyncio.create_task(self._evaluate_rules())

    # =========================
    # METRIC COLLECTION METHODS
    # =========================
    # These methods collect metrics for alert evaluation
    # In a real implementation, these would integrate with the metrics collector

    def _get_pipeline_failure_rate(self) -> float:
        """Get pipeline failure rate in the last 10 minutes."""
        # This would integrate with metrics collector
        # For now, return a mock value
        return 0.05  # 5% failure rate

    def _get_memory_usage_percent(self) -> float:
        """Get current memory usage percentage."""
        try:
            import psutil

            return psutil.virtual_memory().percent
        except ImportError:
            return 50.0  # Mock value

    def _get_cpu_usage_percent(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil

            return psutil.cpu_percent(interval=1)
        except ImportError:
            return 30.0  # Mock value

    def _get_disk_free_percent(self) -> float:
        """Get disk free space percentage."""
        try:
            import psutil

            disk_usage = psutil.disk_usage("/")
            return (disk_usage.free / disk_usage.total) * 100
        except ImportError:
            return 50.0  # Mock value

    def _get_search_p95_latency(self) -> float:
        """Get search P95 latency in seconds."""
        # This would integrate with metrics collector
        return 0.5  # Mock value

    def _get_error_rate(self) -> float:
        """Get overall error rate."""
        # This would integrate with metrics collector
        return 0.02  # 2% error rate

    def _get_mcp_active_connections(self) -> int:
        """Get number of active MCP connections."""
        # This would integrate with metrics collector
        return 2  # Mock value

    def _get_pipeline_queue_length(self) -> int:
        """Get pipeline queue length."""
        # This would integrate with metrics collector
        return 3  # Mock value


# =========================
# GLOBAL ALERT MANAGER
# =========================

_alert_manager: AlertManager | None = None
_alert_lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    """Get or create global alert manager instance."""
    global _alert_manager

    if _alert_manager is None:
        with _alert_lock:
            if _alert_manager is None:
                _alert_manager = AlertManager()

    return _alert_manager


# =========================
# CONVENIENCE FUNCTIONS
# =========================


def register_alert_rule(rule: AlertRule):
    """Register an alert rule with the global manager."""
    manager = get_alert_manager()
    manager.register_rule(rule)


def check_alert_conditions():
    """Force evaluation of all alert conditions."""
    manager = get_alert_manager()
    manager.force_evaluation()


def get_active_alerts() -> list[Alert]:
    """Get all currently active alerts."""
    manager = get_alert_manager()
    return manager.get_active_alerts()


# =========================
# NOTIFICATION HANDLERS
# =========================


def log_notification_handler(alert: Alert):
    """Log-based notification handler."""
    if alert.status == AlertStatus.FIRING:
        logger.warning(
            f"ALERT FIRING: {alert.rule_name}",
            severity=alert.severity.value,
            message=alert.message,
            labels=alert.labels,
        )
    elif alert.status == AlertStatus.RESOLVED:
        logger.info(f"ALERT RESOLVED: {alert.rule_name}", message=alert.message)


def webhook_notification_handler(webhook_url: str):
    """Create webhook notification handler."""

    def handler(alert: Alert):
        try:

            import requests

            payload = {"alert": alert.to_dict(), "timestamp": datetime.now().isoformat()}

            response = requests.post(
                webhook_url, json=payload, timeout=10, headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()

        except Exception as e:
            logger.error("Failed to send webhook notification", error=str(e))

    return handler


def email_notification_handler(smtp_config: dict[str, Any]):
    """Create email notification handler."""

    def handler(alert: Alert):
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart()
            msg["From"] = smtp_config["from"]
            msg["To"] = smtp_config["to"]

            if alert.status == AlertStatus.FIRING:
                msg["Subject"] = f"[ALERT] {alert.rule_name} - {alert.severity.value.upper()}"
                body = f"""
Alert: {alert.rule_name}
Severity: {alert.severity.value}
Status: {alert.status.value}
Message: {alert.message}

Started: {alert.starts_at}
Labels: {alert.labels}
Annotations: {alert.annotations}
"""
            else:
                msg["Subject"] = f"[RESOLVED] {alert.rule_name}"
                body = f"""
Alert Resolved: {alert.rule_name}
Message: {alert.message}
Resolved: {alert.ends_at}
"""

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(smtp_config["smtp_server"], smtp_config["smtp_port"]) as server:
                if smtp_config.get("use_tls"):
                    server.starttls()
                if smtp_config.get("username"):
                    server.login(smtp_config["username"], smtp_config["password"])
                server.send_message(msg)

        except Exception as e:
            logger.error("Failed to send email notification", error=str(e))

    return handler


# =========================
# INITIALIZATION
# =========================


def setup_default_notifications():
    """Setup default notification handlers."""
    manager = get_alert_manager()

    # Always add log handler
    manager.add_notification_handler(log_notification_handler)

    # Add webhook handler if configured
    import os

    webhook_url = os.getenv("MIMIR_ALERT_WEBHOOK_URL")
    if webhook_url:
        manager.add_notification_handler(webhook_notification_handler(webhook_url))

    # Add email handler if configured
    smtp_server = os.getenv("MIMIR_ALERT_SMTP_SERVER")
    if smtp_server:
        smtp_config = {
            "smtp_server": smtp_server,
            "smtp_port": int(os.getenv("MIMIR_ALERT_SMTP_PORT", "587")),
            "from": os.getenv("MIMIR_ALERT_EMAIL_FROM"),
            "to": os.getenv("MIMIR_ALERT_EMAIL_TO"),
            "username": os.getenv("MIMIR_ALERT_SMTP_USERNAME"),
            "password": os.getenv("MIMIR_ALERT_SMTP_PASSWORD"),
            "use_tls": os.getenv("MIMIR_ALERT_SMTP_TLS", "true").lower() == "true",
        }
        if smtp_config["from"] and smtp_config["to"]:
            manager.add_notification_handler(email_notification_handler(smtp_config))


# Initialize default notifications
setup_default_notifications()
