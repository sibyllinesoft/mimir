"""
Authentication, authorization, and rate limiting for security hardening.

Provides API key validation, rate limiting, and abuse prevention
mechanisms for the MCP server.
"""

import hashlib
import json
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util.errors import AuthenticationError, SecurityError
from ..util.log import get_logger

logger = get_logger(__name__)


class AuthenticationFailed(AuthenticationError):
    """Raised when authentication fails."""

    pass


class RateLimitExceeded(SecurityError):
    """Raised when rate limit is exceeded."""

    pass


class AuthorizationDenied(SecurityError):
    """Raised when authorization is denied."""

    pass


@dataclass
class APIKey:
    """Represents an API key with metadata."""

    key_id: str
    key_hash: str
    name: str
    permissions: list[str]
    created_at: float
    last_used: float | None = None
    usage_count: int = 0
    rate_limit: int | None = None  # Requests per minute
    is_active: bool = True


@dataclass
class RateLimitWindow:
    """Sliding window rate limiter."""

    max_requests: int
    window_seconds: int
    requests: deque = field(default_factory=deque)

    def is_allowed(self) -> bool:
        """Check if a request is allowed within rate limits."""
        now = time.time()

        # Remove expired requests
        while self.requests and self.requests[0] <= now - self.window_seconds:
            self.requests.popleft()

        # Check if we're within limits
        if len(self.requests) >= self.max_requests:
            return False

        # Add current request
        self.requests.append(now)
        return True

    def time_until_reset(self) -> float:
        """Get time until rate limit resets."""
        if not self.requests:
            return 0.0

        oldest_request = self.requests[0]
        reset_time = oldest_request + self.window_seconds
        return max(0.0, reset_time - time.time())

    def remaining_requests(self) -> int:
        """Get number of requests remaining in current window."""
        now = time.time()

        # Remove expired requests
        while self.requests and self.requests[0] <= now - self.window_seconds:
            self.requests.popleft()

        return max(0, self.max_requests - len(self.requests))


class APIKeyValidator:
    """Validates and manages API keys."""

    def __init__(self, keys_file: Path | None = None):
        """Initialize API key validator.

        Args:
            keys_file: Path to API keys configuration file
        """
        self.keys_file = keys_file
        self.api_keys: dict[str, APIKey] = {}
        self.key_id_to_hash: dict[str, str] = {}

        if keys_file and keys_file.exists():
            self._load_keys()

    def _load_keys(self) -> None:
        """Load API keys from configuration file."""
        if not self.keys_file or not self.keys_file.exists():
            return

        try:
            with open(self.keys_file) as f:
                keys_data = json.load(f)

            for key_data in keys_data.get("api_keys", []):
                api_key = APIKey(**key_data)
                self.api_keys[api_key.key_hash] = api_key
                self.key_id_to_hash[api_key.key_id] = api_key.key_hash

            logger.info("API keys loaded", key_count=len(self.api_keys))

        except Exception as e:
            logger.error("Failed to load API keys", error=str(e), keys_file=str(self.keys_file))

    def _save_keys(self) -> None:
        """Save API keys to configuration file."""
        if not self.keys_file:
            return

        try:
            keys_data = {
                "api_keys": [
                    {
                        "key_id": key.key_id,
                        "key_hash": key.key_hash,
                        "name": key.name,
                        "permissions": key.permissions,
                        "created_at": key.created_at,
                        "last_used": key.last_used,
                        "usage_count": key.usage_count,
                        "rate_limit": key.rate_limit,
                        "is_active": key.is_active,
                    }
                    for key in self.api_keys.values()
                ]
            }

            # Ensure directory exists
            self.keys_file.parent.mkdir(parents=True, exist_ok=True)

            # Write atomically
            temp_file = self.keys_file.with_suffix(".tmp")
            with open(temp_file, "w") as f:
                json.dump(keys_data, f, indent=2)

            temp_file.replace(self.keys_file)

            logger.debug("API keys saved", key_count=len(self.api_keys))

        except Exception as e:
            logger.error("Failed to save API keys", error=str(e))

    def hash_key(self, raw_key: str) -> str:
        """Generate a secure hash of an API key.

        Args:
            raw_key: Raw API key string

        Returns:
            Secure hash of the key
        """
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def generate_key(
        self, name: str, permissions: list[str], rate_limit: int | None = None
    ) -> tuple[str, str]:
        """Generate a new API key.

        Args:
            name: Human-readable name for the key
            permissions: List of permissions for the key
            rate_limit: Optional rate limit (requests per minute)

        Returns:
            Tuple of (key_id, raw_key)
        """
        # Generate secure random key
        raw_key = secrets.token_urlsafe(32)
        key_hash = self.hash_key(raw_key)
        key_id = secrets.token_urlsafe(16)

        # Create API key object
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=time.time(),
            rate_limit=rate_limit,
        )

        # Store the key
        self.api_keys[key_hash] = api_key
        self.key_id_to_hash[key_id] = key_hash
        self._save_keys()

        logger.info("API key generated", key_id=key_id, name=name, permissions=permissions)

        return key_id, raw_key

    def validate_key(self, raw_key: str) -> APIKey:
        """Validate an API key and return associated metadata.

        Args:
            raw_key: Raw API key to validate

        Returns:
            API key metadata

        Raises:
            AuthenticationFailed: If key is invalid or inactive
        """
        if not raw_key:
            raise AuthenticationFailed("No API key provided")

        key_hash = self.hash_key(raw_key)

        if key_hash not in self.api_keys:
            raise AuthenticationFailed("Invalid API key")

        api_key = self.api_keys[key_hash]

        if not api_key.is_active:
            raise AuthenticationFailed("API key is inactive")

        # Update usage statistics
        api_key.last_used = time.time()
        api_key.usage_count += 1
        self._save_keys()

        logger.debug("API key validated", key_id=api_key.key_id, name=api_key.name)

        return api_key

    def revoke_key(self, key_id: str) -> bool:
        """Revoke an API key.

        Args:
            key_id: Key ID to revoke

        Returns:
            True if key was revoked, False if not found
        """
        if key_id not in self.key_id_to_hash:
            return False

        key_hash = self.key_id_to_hash[key_id]
        api_key = self.api_keys[key_hash]
        api_key.is_active = False
        self._save_keys()

        logger.info("API key revoked", key_id=key_id, name=api_key.name)
        return True

    def list_keys(self) -> list[dict[str, Any]]:
        """List all API keys (without sensitive data).

        Returns:
            List of API key information
        """
        return [
            {
                "key_id": key.key_id,
                "name": key.name,
                "permissions": key.permissions,
                "created_at": key.created_at,
                "last_used": key.last_used,
                "usage_count": key.usage_count,
                "rate_limit": key.rate_limit,
                "is_active": key.is_active,
            }
            for key in self.api_keys.values()
        ]


class RateLimiter:
    """Rate limiter with multiple strategies and abuse detection."""

    def __init__(self):
        """Initialize rate limiter."""
        # Per-IP rate limiting
        self.ip_limits: dict[str, RateLimitWindow] = defaultdict(
            lambda: RateLimitWindow(max_requests=100, window_seconds=60)
        )

        # Per-API key rate limiting
        self.key_limits: dict[str, RateLimitWindow] = {}

        # Global rate limiting
        self.global_limit = RateLimitWindow(max_requests=1000, window_seconds=60)

        # Abuse detection
        self.suspicious_ips: set[str] = set()
        self.blocked_ips: dict[str, float] = {}  # IP -> unblock_time

        # Error rate tracking
        self.error_rates: dict[str, deque] = defaultdict(lambda: deque())

        logger.info("Rate limiter initialized")

    def check_rate_limit(
        self, client_ip: str, api_key: APIKey | None = None, endpoint: str = "unknown"
    ) -> dict[str, Any]:
        """Check if a request is within rate limits.

        Args:
            client_ip: Client IP address
            api_key: Optional API key information
            endpoint: Endpoint being accessed

        Returns:
            Dictionary with rate limit information

        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        now = time.time()

        # Check if IP is blocked
        if client_ip in self.blocked_ips:
            unblock_time = self.blocked_ips[client_ip]
            if now < unblock_time:
                remaining_time = unblock_time - now
                raise RateLimitExceeded(
                    f"IP blocked for {remaining_time:.0f} more seconds",
                    details={
                        "client_ip": client_ip,
                        "unblock_time": unblock_time,
                        "remaining_seconds": remaining_time,
                        "reason": "abuse_detection",
                    },
                )
            else:
                # Unblock expired IP
                del self.blocked_ips[client_ip]
                if client_ip in self.suspicious_ips:
                    self.suspicious_ips.remove(client_ip)

        # Check global rate limit
        if not self.global_limit.is_allowed():
            raise RateLimitExceeded(
                "Global rate limit exceeded",
                details={
                    "limit_type": "global",
                    "max_requests": self.global_limit.max_requests,
                    "window_seconds": self.global_limit.window_seconds,
                    "reset_time": self.global_limit.time_until_reset(),
                },
            )

        # Check per-IP rate limit
        ip_limit = self.ip_limits[client_ip]
        if not ip_limit.is_allowed():
            # Mark as suspicious if consistently hitting limits
            self._track_suspicious_behavior(client_ip, "rate_limit_exceeded")

            raise RateLimitExceeded(
                "IP rate limit exceeded",
                details={
                    "limit_type": "ip",
                    "client_ip": client_ip,
                    "max_requests": ip_limit.max_requests,
                    "window_seconds": ip_limit.window_seconds,
                    "reset_time": ip_limit.time_until_reset(),
                },
            )

        # Check per-API key rate limit
        if api_key and api_key.rate_limit:
            key_id = api_key.key_id
            if key_id not in self.key_limits:
                self.key_limits[key_id] = RateLimitWindow(
                    max_requests=api_key.rate_limit, window_seconds=60
                )

            key_limit = self.key_limits[key_id]
            if not key_limit.is_allowed():
                raise RateLimitExceeded(
                    "API key rate limit exceeded",
                    details={
                        "limit_type": "api_key",
                        "key_id": key_id,
                        "max_requests": key_limit.max_requests,
                        "window_seconds": key_limit.window_seconds,
                        "reset_time": key_limit.time_until_reset(),
                    },
                )

        # Return rate limit information
        return {
            "allowed": True,
            "global_remaining": self.global_limit.remaining_requests(),
            "ip_remaining": ip_limit.remaining_requests(),
            "key_remaining": (
                self.key_limits.get(api_key.key_id).remaining_requests()
                if api_key and api_key.key_id in self.key_limits
                else None
            ),
            "suspicious": client_ip in self.suspicious_ips,
        }

    def record_error(self, client_ip: str, error_type: str) -> None:
        """Record an error for abuse detection.

        Args:
            client_ip: Client IP address
            error_type: Type of error that occurred
        """
        now = time.time()
        error_window = 300  # 5 minutes

        # Add error to tracking
        errors = self.error_rates[client_ip]
        errors.append((now, error_type))

        # Remove old errors
        while errors and errors[0][0] <= now - error_window:
            errors.popleft()

        # Check for abuse patterns
        if len(errors) >= 20:  # 20 errors in 5 minutes
            self._track_suspicious_behavior(client_ip, f"high_error_rate_{error_type}")

    def _track_suspicious_behavior(self, client_ip: str, reason: str) -> None:
        """Track suspicious behavior and potentially block IP.

        Args:
            client_ip: Client IP address
            reason: Reason for suspicious behavior
        """
        self.suspicious_ips.add(client_ip)

        # Count suspicious events
        now = time.time()

        # For simplicity, block after 3 suspicious events
        # In production, this would be more sophisticated
        if len(self.suspicious_ips) >= 3 or reason == "high_error_rate_authentication":
            block_duration = 3600  # 1 hour
            self.blocked_ips[client_ip] = now + block_duration

            logger.warning(
                "IP blocked due to suspicious behavior",
                client_ip=client_ip,
                reason=reason,
                block_duration=block_duration,
            )

    def get_client_info(self, client_ip: str) -> dict[str, Any]:
        """Get information about a client IP.

        Args:
            client_ip: Client IP address

        Returns:
            Dictionary with client information
        """
        now = time.time()

        return {
            "client_ip": client_ip,
            "is_suspicious": client_ip in self.suspicious_ips,
            "is_blocked": client_ip in self.blocked_ips and self.blocked_ips[client_ip] > now,
            "unblock_time": self.blocked_ips.get(client_ip),
            "error_count": len(self.error_rates.get(client_ip, [])),
            "ip_remaining": self.ip_limits[client_ip].remaining_requests(),
            "ip_reset_time": self.ip_limits[client_ip].time_until_reset(),
        }


class AuthManager:
    """Main authentication and authorization manager."""

    def __init__(self, keys_file: Path | None = None, require_auth: bool = True):
        """Initialize authentication manager.

        Args:
            keys_file: Path to API keys file
            require_auth: Whether to require authentication
        """
        self.require_auth = require_auth
        self.api_key_validator = APIKeyValidator(keys_file)
        self.rate_limiter = RateLimiter()

        # Default permissions for tools
        self.tool_permissions = {
            "ensure_repo_index": ["repo:index"],
            "get_repo_bundle": ["repo:read"],
            "search_repo": ["repo:search"],
            "ask_index": ["repo:ask"],
            "cancel": ["repo:cancel"],
        }

        logger.info(f"Authentication manager initialized (require_auth: {require_auth})")

    async def authenticate_request(
        self, tool_name: str, client_ip: str, auth_header: str | None = None
    ) -> dict[str, Any]:
        """Authenticate and authorize a request.

        Args:
            tool_name: Name of the tool being called
            client_ip: Client IP address
            auth_header: Authorization header value

        Returns:
            Dictionary with authentication results

        Raises:
            AuthenticationFailed: If authentication fails
            AuthorizationDenied: If authorization is denied
            RateLimitExceeded: If rate limits are exceeded
        """
        auth_result = {
            "authenticated": False,
            "api_key": None,
            "permissions": [],
            "rate_limit_info": None,
        }

        # Extract API key from header
        api_key_obj = None
        if auth_header:
            if auth_header.startswith("Bearer "):
                raw_key = auth_header[7:]  # Remove "Bearer " prefix
                try:
                    api_key_obj = self.api_key_validator.validate_key(raw_key)
                    auth_result["authenticated"] = True
                    auth_result["api_key"] = {
                        "key_id": api_key_obj.key_id,
                        "name": api_key_obj.name,
                        "permissions": api_key_obj.permissions,
                    }
                    auth_result["permissions"] = api_key_obj.permissions
                except AuthenticationFailed:
                    self.rate_limiter.record_error(client_ip, "authentication")
                    raise

        # Check if authentication is required
        if self.require_auth and not auth_result["authenticated"]:
            self.rate_limiter.record_error(client_ip, "authentication")
            raise AuthenticationFailed("Authentication required")

        # Check rate limits
        try:
            rate_limit_info = self.rate_limiter.check_rate_limit(
                client_ip=client_ip, api_key=api_key_obj, endpoint=tool_name
            )
            auth_result["rate_limit_info"] = rate_limit_info
        except RateLimitExceeded:
            self.rate_limiter.record_error(client_ip, "rate_limit")
            raise

        # Check tool permissions
        required_permissions = self.tool_permissions.get(tool_name, [])
        if required_permissions and auth_result["authenticated"]:
            user_permissions = set(auth_result["permissions"])
            required_permissions_set = set(required_permissions)

            if not required_permissions_set.issubset(user_permissions):
                missing_permissions = required_permissions_set - user_permissions
                raise AuthorizationDenied(
                    f"Insufficient permissions for {tool_name}",
                    details={
                        "tool_name": tool_name,
                        "required_permissions": list(required_permissions_set),
                        "user_permissions": list(user_permissions),
                        "missing_permissions": list(missing_permissions),
                    },
                )

        logger.debug(
            "Request authenticated",
            tool_name=tool_name,
            client_ip=client_ip,
            authenticated=auth_result["authenticated"],
            api_key_id=api_key_obj.key_id if api_key_obj else None,
        )

        return auth_result

    def add_default_key(self) -> tuple[str, str]:
        """Add a default API key for initial setup.

        Returns:
            Tuple of (key_id, raw_key)
        """
        return self.api_key_validator.generate_key(
            name="default",
            permissions=["repo:index", "repo:read", "repo:search", "repo:ask", "repo:cancel"],
            rate_limit=200,  # 200 requests per minute
        )
