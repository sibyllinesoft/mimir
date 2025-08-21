"""
Security configuration management for Mimir.

Provides centralized security configuration with environment variable
support and secure defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json

from ..util.log import get_logger

logger = get_logger(__name__)


@dataclass
class SecurityConfig:
    """Security configuration for Mimir system."""
    
    # Authentication settings
    require_authentication: bool = True
    api_keys_file: Optional[Path] = None
    auth_token_lifetime: int = 3600  # 1 hour
    
    # Authorization settings
    default_permissions: List[str] = field(default_factory=lambda: [
        "repo:index", "repo:read", "repo:search", "repo:ask", "repo:cancel"
    ])
    
    # Rate limiting settings
    global_rate_limit: int = 1000  # requests per minute
    ip_rate_limit: int = 100  # requests per minute per IP
    api_key_rate_limit: int = 200  # requests per minute per API key
    
    # Input validation settings
    max_path_length: int = 4096
    max_filename_length: int = 255
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    max_query_length: int = 10000
    allowed_base_paths: List[str] = field(default_factory=list)
    
    # Credential scanning settings
    enable_credential_scanning: bool = True
    max_scan_files: int = 1000
    scan_extensions: List[str] = field(default_factory=lambda: [
        '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go',
        '.rs', '.php', '.rb', '.kt', '.swift', '.yml', '.yaml', '.json',
        '.xml', '.properties', '.env', '.config', '.ini', '.conf'
    ])
    
    # Sandbox settings
    enable_sandboxing: bool = True
    max_memory_mb: int = 1024  # 1GB
    max_cpu_time_seconds: int = 300  # 5 minutes
    max_wall_time_seconds: int = 600  # 10 minutes
    max_open_files: int = 1024
    max_processes: int = 32
    
    # Encryption settings
    enable_index_encryption: bool = False
    encrypt_embeddings: bool = True
    encrypt_metadata: bool = True
    master_key_env_var: str = "MIMIR_MASTER_KEY"
    
    # Audit logging settings
    enable_audit_logging: bool = True
    audit_log_file: Optional[Path] = None
    audit_log_max_size_mb: int = 100
    audit_log_backup_count: int = 5
    
    # Security monitoring
    enable_threat_detection: bool = True
    max_auth_failures: int = 5
    max_rate_limit_violations: int = 3
    ip_block_duration_minutes: int = 60
    
    # Abuse prevention
    enable_abuse_prevention: bool = True
    suspicious_pattern_threshold: int = 10
    error_rate_threshold: int = 20  # errors per 5 minutes
    
    @classmethod
    def from_environment(cls) -> "SecurityConfig":
        """Create security config from environment variables.
        
        Returns:
            Security configuration loaded from environment
        """
        config = cls()
        
        # Authentication settings
        config.require_authentication = _get_env_bool("MIMIR_REQUIRE_AUTH", config.require_authentication)
        config.auth_token_lifetime = _get_env_int("MIMIR_AUTH_TOKEN_LIFETIME", config.auth_token_lifetime)
        
        # API keys file
        api_keys_file = os.environ.get("MIMIR_API_KEYS_FILE")
        if api_keys_file:
            config.api_keys_file = Path(api_keys_file)
        
        # Rate limiting
        config.global_rate_limit = _get_env_int("MIMIR_GLOBAL_RATE_LIMIT", config.global_rate_limit)
        config.ip_rate_limit = _get_env_int("MIMIR_IP_RATE_LIMIT", config.ip_rate_limit)
        config.api_key_rate_limit = _get_env_int("MIMIR_API_KEY_RATE_LIMIT", config.api_key_rate_limit)
        
        # Input validation
        config.max_path_length = _get_env_int("MIMIR_MAX_PATH_LENGTH", config.max_path_length)
        config.max_file_size = _get_env_int("MIMIR_MAX_FILE_SIZE", config.max_file_size)
        config.max_query_length = _get_env_int("MIMIR_MAX_QUERY_LENGTH", config.max_query_length)
        
        # Allowed base paths
        allowed_paths = os.environ.get("MIMIR_ALLOWED_BASE_PATHS")
        if allowed_paths:
            config.allowed_base_paths = [p.strip() for p in allowed_paths.split(",")]
        
        # Credential scanning
        config.enable_credential_scanning = _get_env_bool("MIMIR_ENABLE_CREDENTIAL_SCANNING", config.enable_credential_scanning)
        config.max_scan_files = _get_env_int("MIMIR_MAX_SCAN_FILES", config.max_scan_files)
        
        # Sandbox settings
        config.enable_sandboxing = _get_env_bool("MIMIR_ENABLE_SANDBOXING", config.enable_sandboxing)
        config.max_memory_mb = _get_env_int("MIMIR_MAX_MEMORY_MB", config.max_memory_mb)
        config.max_cpu_time_seconds = _get_env_int("MIMIR_MAX_CPU_TIME", config.max_cpu_time_seconds)
        config.max_wall_time_seconds = _get_env_int("MIMIR_MAX_WALL_TIME", config.max_wall_time_seconds)
        
        # Encryption settings
        config.enable_index_encryption = _get_env_bool("MIMIR_ENABLE_ENCRYPTION", config.enable_index_encryption)
        config.encrypt_embeddings = _get_env_bool("MIMIR_ENCRYPT_EMBEDDINGS", config.encrypt_embeddings)
        config.encrypt_metadata = _get_env_bool("MIMIR_ENCRYPT_METADATA", config.encrypt_metadata)
        
        # Audit logging
        config.enable_audit_logging = _get_env_bool("MIMIR_ENABLE_AUDIT_LOGGING", config.enable_audit_logging)
        
        audit_log_file = os.environ.get("MIMIR_AUDIT_LOG_FILE")
        if audit_log_file:
            config.audit_log_file = Path(audit_log_file)
        
        config.audit_log_max_size_mb = _get_env_int("MIMIR_AUDIT_LOG_MAX_SIZE_MB", config.audit_log_max_size_mb)
        config.audit_log_backup_count = _get_env_int("MIMIR_AUDIT_LOG_BACKUP_COUNT", config.audit_log_backup_count)
        
        # Security monitoring
        config.enable_threat_detection = _get_env_bool("MIMIR_ENABLE_THREAT_DETECTION", config.enable_threat_detection)
        config.max_auth_failures = _get_env_int("MIMIR_MAX_AUTH_FAILURES", config.max_auth_failures)
        config.ip_block_duration_minutes = _get_env_int("MIMIR_IP_BLOCK_DURATION", config.ip_block_duration_minutes)
        
        # Abuse prevention
        config.enable_abuse_prevention = _get_env_bool("MIMIR_ENABLE_ABUSE_PREVENTION", config.enable_abuse_prevention)
        config.suspicious_pattern_threshold = _get_env_int("MIMIR_SUSPICIOUS_THRESHOLD", config.suspicious_pattern_threshold)
        config.error_rate_threshold = _get_env_int("MIMIR_ERROR_RATE_THRESHOLD", config.error_rate_threshold)
        
        logger.info("Security configuration loaded from environment")
        return config
    
    @classmethod
    def from_file(cls, config_file: Path) -> "SecurityConfig":
        """Load security config from file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Security configuration
        """
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            config = cls()
            
            # Update config with file data
            for key, value in config_data.items():
                if hasattr(config, key):
                    # Handle Path types
                    if key in ['api_keys_file', 'audit_log_file'] and value:
                        setattr(config, key, Path(value))
                    else:
                        setattr(config, key, value)
            
            logger.info("Security configuration loaded from file", config_file=str(config_file))
            return config
            
        except Exception as e:
            logger.error("Failed to load security config from file", config_file=str(config_file), error=str(e))
            raise
    
    def to_file(self, config_file: Path) -> None:
        """Save security config to file.
        
        Args:
            config_file: Path to save configuration
        """
        try:
            # Convert to serializable format
            config_data = {}
            
            for key, value in self.__dict__.items():
                if isinstance(value, Path):
                    config_data[key] = str(value)
                else:
                    config_data[key] = value
            
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info("Security configuration saved to file", config_file=str(config_file))
            
        except Exception as e:
            logger.error("Failed to save security config to file", config_file=str(config_file), error=str(e))
            raise
    
    def get_default_paths(self) -> Dict[str, Path]:
        """Get default file paths for security components.
        
        Returns:
            Dictionary of default paths
        """
        base_dir = Path.home() / ".cache" / "mimir" / "security"
        
        return {
            "api_keys_file": base_dir / "api_keys.json",
            "audit_log_file": base_dir / "audit.log",
            "secrets_file": base_dir / "secrets.enc",
            "config_file": base_dir / "security_config.json"
        }
    
    def apply_defaults(self) -> None:
        """Apply default file paths if not configured."""
        defaults = self.get_default_paths()
        
        if not self.api_keys_file:
            self.api_keys_file = defaults["api_keys_file"]
        
        if not self.audit_log_file:
            self.audit_log_file = defaults["audit_log_file"]
    
    def validate(self) -> List[str]:
        """Validate security configuration.
        
        Returns:
            List of validation errors
        """
        errors = []
        
        # Validate numeric limits
        if self.max_path_length <= 0:
            errors.append("max_path_length must be positive")
        
        if self.max_file_size <= 0:
            errors.append("max_file_size must be positive")
        
        if self.max_query_length <= 0:
            errors.append("max_query_length must be positive")
        
        # Validate rate limits
        if self.global_rate_limit <= 0:
            errors.append("global_rate_limit must be positive")
        
        if self.ip_rate_limit <= 0:
            errors.append("ip_rate_limit must be positive")
        
        # Validate sandbox limits
        if self.max_memory_mb <= 0:
            errors.append("max_memory_mb must be positive")
        
        if self.max_cpu_time_seconds <= 0:
            errors.append("max_cpu_time_seconds must be positive")
        
        # Validate paths
        if self.allowed_base_paths:
            for path in self.allowed_base_paths:
                if not Path(path).exists():
                    errors.append(f"Allowed base path does not exist: {path}")
        
        # Validate file permissions
        if self.api_keys_file:
            try:
                self.api_keys_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create API keys directory: {e}")
        
        if self.audit_log_file:
            try:
                self.audit_log_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create audit log directory: {e}")
        
        return errors
    
    def get_resource_limits(self) -> Dict[str, int]:
        """Get resource limits for sandbox configuration.
        
        Returns:
            Dictionary of resource limits
        """
        return {
            "max_memory": self.max_memory_mb * 1024 * 1024,  # Convert to bytes
            "max_cpu_time": self.max_cpu_time_seconds,
            "max_wall_time": self.max_wall_time_seconds,
            "max_open_files": self.max_open_files,
            "max_processes": self.max_processes,
            "max_file_size": self.max_file_size
        }


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable.
    
    Args:
        key: Environment variable key
        default: Default value
        
    Returns:
        Boolean value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    return value.lower() in ('true', '1', 'yes', 'on', 'enabled')


def _get_env_int(key: str, default: int) -> int:
    """Get integer value from environment variable.
    
    Args:
        key: Environment variable key
        default: Default value
        
    Returns:
        Integer value
    """
    value = os.environ.get(key)
    if value is None:
        return default
    
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer value for environment variable", key=key, value=value)
        return default


# Global security configuration
_security_config: Optional[SecurityConfig] = None


def get_security_config() -> SecurityConfig:
    """Get the global security configuration.
    
    Returns:
        Global security configuration
    """
    global _security_config
    if _security_config is None:
        _security_config = SecurityConfig.from_environment()
        _security_config.apply_defaults()
        
        # Validate configuration
        errors = _security_config.validate()
        if errors:
            logger.warning("Security configuration validation errors", errors=errors)
    
    return _security_config


def configure_security(config: SecurityConfig) -> None:
    """Configure the global security settings.
    
    Args:
        config: Security configuration to use
    """
    global _security_config
    _security_config = config
    
    # Validate configuration
    errors = config.validate()
    if errors:
        logger.warning("Security configuration validation errors", errors=errors)
    
    logger.info("Security configuration updated")


def load_security_config_file(config_file: Path) -> SecurityConfig:
    """Load security configuration from file and set as global config.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Loaded security configuration
    """
    config = SecurityConfig.from_file(config_file)
    configure_security(config)
    return config