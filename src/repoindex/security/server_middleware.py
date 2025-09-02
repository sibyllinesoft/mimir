"""
Security middleware for MCP server operations.

Provides comprehensive security enforcement including authentication,
authorization, input validation, rate limiting, and audit logging.
"""

import time
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent

from ..util.errors import SecurityError
from ..util.log import get_logger
from .audit import SecurityEvent, SecurityEventSeverity, SecurityEventType, get_security_auditor
from .auth import AuthenticationFailed, AuthManager, AuthorizationDenied, RateLimitExceeded
from .sandbox import SandboxViolation, get_sandbox
from .secrets import CredentialScanner
from .validation import InputValidator, SecurityViolation

logger = get_logger(__name__)


class SecurityMiddleware:
    """Security middleware for MCP server operations."""

    def __init__(
        self,
        auth_manager: AuthManager,
        input_validator: InputValidator,
        credential_scanner: CredentialScanner,
        require_auth: bool = True,
        enable_credential_scanning: bool = True,
        allowed_base_paths: list[str] | None = None,
    ):
        """Initialize security middleware.

        Args:
            auth_manager: Authentication manager
            input_validator: Input validator
            credential_scanner: Credential scanner
            require_auth: Whether to require authentication
            enable_credential_scanning: Whether to scan for credentials
            allowed_base_paths: Allowed base paths for operations
        """
        self.auth_manager = auth_manager
        self.input_validator = input_validator
        self.credential_scanner = credential_scanner
        self.require_auth = require_auth
        self.enable_credential_scanning = enable_credential_scanning
        self.allowed_base_paths = allowed_base_paths or []

        # Configure input validator with allowed paths
        if self.allowed_base_paths:
            self.input_validator.path_validator.allowed_base_paths = [
                Path(p).resolve() for p in self.allowed_base_paths
            ]

        self.security_auditor = get_security_auditor()
        self.sandbox = get_sandbox()

        logger.info(
            "Security middleware initialized",
            require_auth=require_auth,
            enable_credential_scanning=enable_credential_scanning,
            allowed_base_paths=len(self.allowed_base_paths),
        )

    def secure_tool(
        self,
        permissions: list[str] | None = None,
        validate_input: bool = True,
        scan_credentials: bool | None = None,
        rate_limit: dict[str, int] | None = None,
    ):
        """Decorator to secure MCP tool functions.

        Args:
            permissions: Required permissions for the tool
            validate_input: Whether to validate input arguments
            scan_credentials: Whether to scan for credentials
            rate_limit: Optional rate limit overrides
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> CallToolResult:
                start_time = time.time()
                tool_name = func.__name__.replace("_", "")

                # Extract arguments and context
                arguments = {}
                client_ip = "127.0.0.1"  # Default for stdio
                auth_header = None

                # Get arguments from function call
                if args and len(args) > 1:
                    if isinstance(args[1], dict):
                        arguments = args[1]

                try:
                    # 1. Authentication and Authorization
                    auth_result = await self._authenticate_request(
                        tool_name=tool_name,
                        client_ip=client_ip,
                        auth_header=auth_header,
                        permissions=permissions,
                    )

                    # 2. Rate Limiting
                    await self._check_rate_limits(
                        tool_name=tool_name,
                        client_ip=client_ip,
                        auth_result=auth_result,
                        rate_limit_overrides=rate_limit,
                    )

                    # 3. Input Validation
                    if validate_input:
                        await self._validate_input(
                            tool_name=tool_name,
                            arguments=arguments,
                            client_ip=client_ip,
                            user_id=auth_result.get("api_key", {}).get("key_id"),
                        )

                    # 4. Credential Scanning (if enabled)
                    if self._should_scan_credentials(scan_credentials) and "path" in arguments:
                        await self._scan_repository_credentials(
                            repo_path=arguments["path"],
                            user_id=auth_result.get("api_key", {}).get("key_id"),
                            client_ip=client_ip,
                        )

                    # 5. Execute the original function with security context
                    logger.info(
                        "Executing secured tool",
                        tool_name=tool_name,
                        user_id=auth_result.get("api_key", {}).get("key_id"),
                        authenticated=auth_result["authenticated"],
                    )

                    # Call original function
                    result = await func(*args, **kwargs)

                    # 6. Log successful operation
                    execution_time = time.time() - start_time

                    self.security_auditor.log_file_access(
                        file_path=arguments.get("path", ""),
                        operation=tool_name,
                        success=True,
                        user_id=auth_result.get("api_key", {}).get("key_id"),
                        client_ip=client_ip,
                    )

                    logger.info(
                        "Tool execution completed successfully",
                        tool_name=tool_name,
                        execution_time=execution_time,
                        user_id=auth_result.get("api_key", {}).get("key_id"),
                    )

                    return result

                except SecurityError as e:
                    # Handle security-specific errors
                    return await self._handle_security_error(
                        e,
                        tool_name,
                        arguments,
                        client_ip,
                        auth_result.get("api_key", {}).get("key_id"),
                    )

                except Exception as e:
                    # Handle general errors
                    logger.error(
                        "Tool execution failed",
                        tool_name=tool_name,
                        error=str(e),
                        arguments=arguments,
                    )

                    # Log security event
                    event = SecurityEvent(
                        event_type=SecurityEventType.ERROR_OCCURRED,
                        severity=SecurityEventSeverity.MEDIUM,
                        message=f"Tool execution failed: {tool_name}",
                        tool_name=tool_name,
                        client_ip=client_ip,
                        component="mcp_server",
                        operation=tool_name,
                        metadata={"error": str(e)},
                    )
                    self.security_auditor.audit_logger.log_event(event)

                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Internal error: {str(e)}")],
                        isError=True,
                    )

            return wrapper

        return decorator

    async def _authenticate_request(
        self,
        tool_name: str,
        client_ip: str,
        auth_header: str | None = None,
        permissions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Authenticate and authorize a request.

        Args:
            tool_name: Name of the tool being called
            client_ip: Client IP address
            auth_header: Authorization header
            permissions: Required permissions

        Returns:
            Authentication result dictionary

        Raises:
            AuthenticationFailed: If authentication fails
            AuthorizationDenied: If authorization is denied
        """
        try:
            auth_result = await self.auth_manager.authenticate_request(
                tool_name=tool_name, client_ip=client_ip, auth_header=auth_header
            )

            # Check specific permissions if provided
            if permissions and auth_result["authenticated"]:
                user_permissions = set(auth_result["permissions"])
                required_permissions_set = set(permissions)

                if not required_permissions_set.issubset(user_permissions):
                    missing_permissions = required_permissions_set - user_permissions

                    self.security_auditor.log_authorization_check(
                        success=False,
                        user_id=auth_result["api_key"]["key_id"],
                        tool_name=tool_name,
                        required_permissions=list(required_permissions_set),
                        user_permissions=list(user_permissions),
                        client_ip=client_ip,
                    )

                    raise AuthorizationDenied(
                        f"Missing permissions for {tool_name}: {list(missing_permissions)}"
                    )

            # Log successful authentication/authorization
            if auth_result["authenticated"]:
                self.security_auditor.log_authentication_attempt(
                    success=True, user_id=auth_result["api_key"]["key_id"], client_ip=client_ip
                )

                self.security_auditor.log_authorization_check(
                    success=True,
                    user_id=auth_result["api_key"]["key_id"],
                    tool_name=tool_name,
                    required_permissions=permissions or [],
                    user_permissions=auth_result["permissions"],
                    client_ip=client_ip,
                )

            return auth_result

        except (AuthenticationFailed, AuthorizationDenied) as e:
            # Log authentication/authorization failure
            if isinstance(e, AuthenticationFailed):
                self.security_auditor.log_authentication_attempt(
                    success=False, client_ip=client_ip, failure_reason=str(e)
                )

            raise

    async def _check_rate_limits(
        self,
        tool_name: str,
        client_ip: str,
        auth_result: dict[str, Any],
        rate_limit_overrides: dict[str, int] | None = None,
    ) -> None:
        """Check rate limits for the request.

        Args:
            tool_name: Name of the tool
            client_ip: Client IP address
            auth_result: Authentication result
            rate_limit_overrides: Optional rate limit overrides

        Raises:
            RateLimitExceeded: If rate limits are exceeded
        """
        try:
            # Apply rate limit overrides if provided
            if rate_limit_overrides:
                # This would require extending the rate limiter to support overrides
                pass

            self.auth_manager.rate_limiter.check_rate_limit(
                client_ip=client_ip, api_key=auth_result.get("api_key_obj"), endpoint=tool_name
            )

        except RateLimitExceeded as e:
            # Log rate limit exceeded
            self.security_auditor.log_rate_limit_exceeded(
                client_ip=client_ip,
                limit_type=e.details.get("limit_type", "unknown"),
                limit_details=e.details,
                user_id=auth_result.get("api_key", {}).get("key_id"),
            )

            raise

    async def _validate_input(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        client_ip: str,
        user_id: str | None = None,
    ) -> None:
        """Validate input arguments.

        Args:
            tool_name: Name of the tool
            arguments: Tool arguments
            client_ip: Client IP address
            user_id: User identifier

        Raises:
            SecurityViolation: If input validation fails
        """
        try:
            validation_result = self.input_validator.validate_mcp_request(
                tool_name=tool_name, arguments=arguments
            )

            # Log warnings if any
            if validation_result.get("warnings"):
                logger.warning(
                    "Input validation warnings",
                    tool_name=tool_name,
                    warnings=validation_result["warnings"],
                    user_id=user_id,
                )

        except SecurityViolation as e:
            # Log input validation failure
            self.security_auditor.log_input_validation_failure(
                validation_type=e.violation_type,
                input_value=str(arguments)[:200],  # Truncate for security
                violation_details=e.details,
                client_ip=client_ip,
                user_id=user_id,
            )

            raise

    async def _scan_repository_credentials(
        self, repo_path: str, user_id: str | None = None, client_ip: str | None = None
    ) -> None:
        """Scan repository for embedded credentials.

        Args:
            repo_path: Repository path to scan
            user_id: User identifier
            client_ip: Client IP address
        """
        if not self.enable_credential_scanning:
            return

        try:
            # Validate path first
            validated_path = self.input_validator.path_validator.validate_repository_path(repo_path)

            # Scan for credentials (limit to prevent DoS)
            max_files = 500  # Limit for performance
            credentials_found = self.credential_scanner.scan_directory(
                directory=validated_path, max_files=max_files
            )

            # Log any credentials found
            if credentials_found:
                for file_path, credentials in credentials_found.items():
                    for credential in credentials:
                        self.security_auditor.log_credential_detected(
                            file_path=file_path,
                            credential_type=credential["pattern_name"],
                            severity=credential["severity"],
                            line_number=credential["line_number"],
                            context=credential.get("context", []),
                        )

                logger.warning(
                    "Credentials detected in repository",
                    repo_path=str(validated_path),
                    files_with_credentials=len(credentials_found),
                    user_id=user_id,
                )

        except Exception as e:
            logger.error(
                "Credential scanning failed", repo_path=repo_path, error=str(e), user_id=user_id
            )

    def _should_scan_credentials(self, scan_override: bool | None = None) -> bool:
        """Determine if credential scanning should be performed.

        Args:
            scan_override: Optional override for scanning

        Returns:
            True if credentials should be scanned
        """
        if scan_override is not None:
            return scan_override
        return self.enable_credential_scanning

    async def _handle_security_error(
        self,
        error: SecurityError,
        tool_name: str,
        arguments: dict[str, Any],
        client_ip: str,
        user_id: str | None = None,
    ) -> CallToolResult:
        """Handle security errors consistently.

        Args:
            error: Security error that occurred
            tool_name: Name of the tool
            arguments: Tool arguments
            client_ip: Client IP address
            user_id: User identifier

        Returns:
            Error result for the client
        """
        error_type = type(error).__name__

        # Determine severity based on error type
        severity_map = {
            "AuthenticationFailed": SecurityEventSeverity.MEDIUM,
            "AuthorizationDenied": SecurityEventSeverity.MEDIUM,
            "RateLimitExceeded": SecurityEventSeverity.MEDIUM,
            "SecurityViolation": SecurityEventSeverity.HIGH,
            "SandboxViolation": SecurityEventSeverity.HIGH,
        }

        severity = severity_map.get(error_type, SecurityEventSeverity.MEDIUM)

        # Create security event
        event = SecurityEvent(
            event_type=SecurityEventType.ERROR_OCCURRED,
            severity=severity,
            message=f"Security error in {tool_name}: {str(error)}",
            user_id=user_id,
            client_ip=client_ip,
            tool_name=tool_name,
            component="security_middleware",
            operation=tool_name,
            metadata={
                "error_type": error_type,
                "error_details": getattr(error, "details", {}),
                "arguments": str(arguments)[:200],  # Truncate for security
            },
        )

        # Add threat indicators
        if isinstance(error, SecurityViolation):
            event.add_threat_indicator("input_validation_failure")
        elif isinstance(error, AuthenticationFailed | AuthorizationDenied):
            event.add_threat_indicator("authentication_failure")
        elif isinstance(error, RateLimitExceeded):
            event.add_threat_indicator("rate_limit_abuse")
        elif isinstance(error, SandboxViolation):
            event.add_threat_indicator("sandbox_escape_attempt")

        self.security_auditor.audit_logger.log_event(event)

        # Record error for rate limiting
        self.auth_manager.rate_limiter.record_error(client_ip, error_type)

        logger.warning(
            "Security error occurred",
            tool_name=tool_name,
            error_type=error_type,
            error_message=str(error),
            user_id=user_id,
            client_ip=client_ip,
        )

        # Return appropriate error response
        if isinstance(error, AuthenticationFailed):
            error_message = "Authentication required"
        elif isinstance(error, AuthorizationDenied):
            error_message = "Access denied"
        elif isinstance(error, RateLimitExceeded):
            error_message = "Rate limit exceeded - please try again later"
        else:
            error_message = "Security validation failed"

        return CallToolResult(content=[TextContent(type="text", text=error_message)], isError=True)


def create_security_middleware(
    require_auth: bool = True,
    enable_credential_scanning: bool = True,
    allowed_base_paths: list[str] | None = None,
    api_keys_file: Path | None = None,
) -> SecurityMiddleware:
    """Create a configured security middleware instance.

    Args:
        require_auth: Whether to require authentication
        enable_credential_scanning: Whether to enable credential scanning
        allowed_base_paths: List of allowed base paths
        api_keys_file: Path to API keys file

    Returns:
        Configured security middleware
    """
    # Create components
    auth_manager = AuthManager(keys_file=api_keys_file, require_auth=require_auth)

    input_validator = InputValidator(allowed_base_paths=allowed_base_paths)
    credential_scanner = CredentialScanner()

    # Create middleware
    middleware = SecurityMiddleware(
        auth_manager=auth_manager,
        input_validator=input_validator,
        credential_scanner=credential_scanner,
        require_auth=require_auth,
        enable_credential_scanning=enable_credential_scanning,
        allowed_base_paths=allowed_base_paths,
    )

    # Generate default API key if none exist and auth is required
    if require_auth and api_keys_file and not auth_manager.api_key_validator.api_keys:
        key_id, raw_key = auth_manager.add_default_key()
        logger.info(
            "Generated default API key for initial setup",
            key_id=key_id,
            note="Save this key securely - it will not be shown again",
        )
        print("\n=== MIMIR SECURITY SETUP ===")
        print("Default API Key Generated:")
        print(f"Key ID: {key_id}")
        print(f"API Key: {raw_key}")
        print("Save this key securely - it will not be shown again!")
        print("===============================\n")

    return middleware
