"""
Configuration migration utilities for backward compatibility.

Provides utilities to gradually migrate from scattered environment variable
access to centralized configuration management while maintaining full
backward compatibility.
"""

import os
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar

from .config import get_config
from .util.log import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


class ConfigMigration:
    """Utilities for migrating from direct environment access to centralized config."""
    
    @staticmethod
    def get_env_value(
        key: str, 
        default: T = None,
        config_path: str | None = None,
        deprecation_version: str = "2.0.0"
    ) -> T | None:
        """
        Get environment value with migration warning.
        
        Args:
            key: Environment variable key
            default: Default value if not found
            config_path: Path to new configuration location (for warning)
            deprecation_version: Version when direct env access will be removed
            
        Returns:
            Environment variable value or default
        """
        value = os.environ.get(key, default)
        
        if value != default and config_path:
            warnings.warn(
                f"Direct access to environment variable '{key}' is deprecated. "
                f"Use config.{config_path} instead. "
                f"Direct access will be removed in version {deprecation_version}.",
                DeprecationWarning,
                stacklevel=3
            )
            logger.debug(
                "Deprecated environment variable access",
                key=key,
                config_path=config_path,
                deprecation_version=deprecation_version
            )
        
        return value
    
    @staticmethod
    def migrate_getenv(
        original_func: Callable[..., T]
    ) -> Callable[..., T]:
        """
        Decorator to migrate os.getenv calls to centralized config.
        
        Usage:
            @ConfigMigration.migrate_getenv
            def get_log_level():
                return os.getenv("MIMIR_LOG_LEVEL", "INFO")
        """
        @wraps(original_func)
        def wrapper(*args, **kwargs) -> T:
            # Issue deprecation warning
            warnings.warn(
                f"Function {original_func.__name__} uses deprecated direct environment access. "
                "Consider using centralized configuration instead.",
                DeprecationWarning,
                stacklevel=2
            )
            
            return original_func(*args, **kwargs)
        
        return wrapper


def create_compatibility_getters():
    """
    Create backward-compatible getter functions for commonly used environment variables.
    
    These functions provide the same interface as direct os.getenv calls but issue
    deprecation warnings and guide users to the new configuration system.
    """
    
    def get_log_level(default: str = "INFO") -> str:
        """Get log level with migration support."""
        config = get_config()
        if hasattr(config, 'logging') and hasattr(config.logging, 'log_level'):
            return config.logging.log_level
        return ConfigMigration.get_env_value(
            "MIMIR_LOG_LEVEL", 
            default, 
            "logging.log_level"
        )
    
    def get_log_format(default: str = "json") -> str:
        """Get log format with migration support."""
        config = get_config()
        if hasattr(config, 'logging') and hasattr(config.logging, 'log_format'):
            return config.logging.log_format
        return ConfigMigration.get_env_value(
            "LOG_FORMAT", 
            default, 
            "logging.log_format"
        )
    
    def get_gemini_api_key() -> str | None:
        """Get Gemini API key with migration support."""
        config = get_config()
        if hasattr(config, 'ai') and config.ai.api_key:
            return config.ai.api_key
        
        # Check both environment variables for backward compatibility
        key = (ConfigMigration.get_env_value("GOOGLE_API_KEY", config_path="ai.google_api_key") or
               ConfigMigration.get_env_value("GEMINI_API_KEY", config_path="ai.gemini_api_key"))
        return key
    
    def get_max_workers(default: int = 4) -> int:
        """Get max workers with migration support."""
        config = get_config()
        if hasattr(config, 'server') and hasattr(config.server, 'max_workers'):
            return config.server.max_workers
        
        value = ConfigMigration.get_env_value(
            "MIMIR_MAX_WORKERS", 
            str(default), 
            "server.max_workers"
        )
        return int(value) if value else default
    
    def get_timeout(default: int = 300) -> int:
        """Get request timeout with migration support."""
        config = get_config()
        if hasattr(config, 'server') and hasattr(config.server, 'timeout'):
            return config.server.timeout
        
        value = ConfigMigration.get_env_value(
            "MIMIR_TIMEOUT", 
            str(default), 
            "server.timeout"
        )
        return int(value) if value else default
    
    def get_data_path(default: str = "./data") -> str:
        """Get data path with migration support."""
        config = get_config()
        if hasattr(config, 'storage') and hasattr(config.storage, 'data_path'):
            return str(config.storage.data_path)
        
        return ConfigMigration.get_env_value(
            "MIMIR_DATA_PATH", 
            default, 
            "storage.data_path"
        )
    
    def get_enable_metrics(default: bool = False) -> bool:
        """Get enable metrics flag with migration support."""
        config = get_config()
        if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'enable_metrics'):
            return config.monitoring.enable_metrics
        
        value = ConfigMigration.get_env_value(
            "MIMIR_ENABLE_METRICS", 
            str(default).lower(), 
            "monitoring.enable_metrics"
        )
        return value.lower() in ("true", "1", "yes", "on") if isinstance(value, str) else bool(value)
    
    def get_service_name(default: str = "mimir-repoindex") -> str:
        """Get service name with migration support."""
        config = get_config()
        if hasattr(config, 'monitoring') and hasattr(config.monitoring, 'service_name'):
            return config.monitoring.service_name
        
        return ConfigMigration.get_env_value(
            "MIMIR_SERVICE_NAME", 
            default, 
            "monitoring.service_name"
        )
    
    # Return all compatibility functions
    return {
        'get_log_level': get_log_level,
        'get_log_format': get_log_format,
        'get_gemini_api_key': get_gemini_api_key,
        'get_max_workers': get_max_workers,
        'get_timeout': get_timeout,
        'get_data_path': get_data_path,
        'get_enable_metrics': get_enable_metrics,
        'get_service_name': get_service_name,
    }


# Global compatibility functions for easy import
_compatibility_getters = create_compatibility_getters()

# Export commonly used functions for backward compatibility
get_log_level = _compatibility_getters['get_log_level']
get_log_format = _compatibility_getters['get_log_format']  
get_gemini_api_key = _compatibility_getters['get_gemini_api_key']
get_max_workers = _compatibility_getters['get_max_workers']
get_timeout = _compatibility_getters['get_timeout']
get_data_path = _compatibility_getters['get_data_path']
get_enable_metrics = _compatibility_getters['get_enable_metrics']
get_service_name = _compatibility_getters['get_service_name']


class LegacyConfigAdapter:
    """
    Adapter to make centralized config work with legacy code expecting direct env access.
    """
    
    def __init__(self):
        self.config = get_config()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by environment variable key."""
        
        # Map common environment variables to config paths
        mappings = {
            # Server config
            "MIMIR_UI_HOST": ("server", "host"),
            "MIMIR_UI_PORT": ("server", "port"),
            "MIMIR_TIMEOUT": ("server", "timeout"),
            "MIMIR_MAX_WORKERS": ("server", "max_workers"),
            "MIMIR_MAX_MEMORY_MB": ("server", "max_memory_mb"),
            
            # Storage config  
            "MIMIR_DATA_PATH": ("storage", "data_path"),
            "MIMIR_CACHE_PATH": ("storage", "cache_path"),
            "MIMIR_LOGS_PATH": ("storage", "logs_path"),
            
            # AI config
            "GOOGLE_API_KEY": ("ai", "google_api_key"),
            "GEMINI_API_KEY": ("ai", "gemini_api_key"),
            "GEMINI_MODEL": ("ai", "gemini_model"),
            "GEMINI_MAX_TOKENS": ("ai", "gemini_max_tokens"),
            "MIMIR_ENABLE_GEMINI": ("ai", "enable_gemini"),
            
            # Logging config
            "MIMIR_LOG_LEVEL": ("logging", "log_level"),
            "LOG_FORMAT": ("logging", "log_format"),
            "MIMIR_LOG_FILE": ("logging", "log_file"),
            
            # Monitoring config
            "MIMIR_ENABLE_METRICS": ("monitoring", "enable_metrics"),
            "MIMIR_METRICS_PORT": ("monitoring", "metrics_port"),
            "MIMIR_SERVICE_NAME": ("monitoring", "service_name"),
            "JAEGER_ENDPOINT": ("monitoring", "jaeger_endpoint"),
            
            # Pipeline config
            "PIPELINE_TIMEOUT_ACQUIRE": ("pipeline", "timeout_acquire"),
            "PIPELINE_TIMEOUT_SERENA": ("pipeline", "timeout_serena"),
            "PIPELINE_MAX_FILE_SIZE": ("pipeline", "max_file_size_mb"),
            "GIT_DEFAULT_BRANCH": ("pipeline", "git_default_branch"),
            
            # Performance config
            "ASYNCIO_MAX_WORKERS": ("performance", "asyncio_max_workers"),
            "FILE_READ_BUFFER_SIZE": ("performance", "file_read_buffer_size"),
        }
        
        if key in mappings:
            section_name, attr_name = mappings[key]
            section = getattr(self.config, section_name, None)
            if section and hasattr(section, attr_name):
                value = getattr(section, attr_name)
                
                # Convert to string for environment variable compatibility
                if isinstance(value, (list, tuple)):
                    return ",".join(str(v) for v in value)
                elif isinstance(value, bool):
                    return "true" if value else "false"
                else:
                    return str(value)
        
        # Fall back to actual environment variable
        return os.environ.get(key, default)
    
    def getenv(self, key: str, default: Any = None) -> Any:
        """Alias for get() to match os.getenv interface."""
        return self.get(key, default)


# Global adapter instance (lazy initialization to avoid circular imports)
_legacy_adapter = None


def _get_legacy_adapter():
    """Get legacy adapter instance with lazy initialization."""
    global _legacy_adapter
    if _legacy_adapter is None:
        _legacy_adapter = LegacyConfigAdapter()
    return _legacy_adapter


def get_env_with_config(key: str, default: Any = None) -> Any:
    """
    Drop-in replacement for os.getenv that uses centralized config when available.
    
    Args:
        key: Environment variable key
        default: Default value if not found
        
    Returns:
        Configuration value from centralized config or environment variable
    """
    adapter = _get_legacy_adapter()
    return adapter.get(key, default)


# Monkey-patching utilities (use with caution)
def patch_os_getenv():
    """
    Monkey-patch os.getenv to use centralized configuration.
    
    WARNING: This is a global modification that affects all code.
    Use only for quick testing or migration purposes.
    """
    import os
    
    original_getenv = os.getenv
    
    def patched_getenv(key: str, default: Any = None) -> Any:
        """Patched os.getenv that uses centralized config."""
        try:
            # Try centralized config first
            return get_env_with_config(key, default)
        except Exception:
            # Fall back to original implementation
            return original_getenv(key, default)
    
    os.getenv = patched_getenv
    logger.info("Patched os.getenv to use centralized configuration")


def unpatch_os_getenv():
    """Restore original os.getenv behavior."""
    import os
    
    # This is a simplified approach - in practice you'd want to store the original
    # function reference when patching
    import builtins
    os.getenv = builtins.getenv if hasattr(builtins, 'getenv') else os.environ.get
    logger.info("Restored original os.getenv behavior")


# Migration progress tracking
class MigrationTracker:
    """Track migration progress across the codebase."""
    
    def __init__(self):
        self.migrated_files = set()
        self.pending_migrations = {}
        self.deprecated_calls = {}
    
    def mark_migrated(self, file_path: str, config_sections: list[str]):
        """Mark a file as migrated to centralized config."""
        self.migrated_files.add(file_path)
        logger.info(f"File migrated to centralized config: {file_path} (sections: {config_sections})")
    
    def record_deprecated_call(self, file_path: str, env_var: str, line_number: int):
        """Record a deprecated environment variable access."""
        if file_path not in self.deprecated_calls:
            self.deprecated_calls[file_path] = []
        
        self.deprecated_calls[file_path].append({
            'env_var': env_var,
            'line_number': line_number
        })
    
    def get_migration_report(self) -> dict:
        """Generate a migration progress report."""
        return {
            'migrated_files': len(self.migrated_files),
            'files_with_deprecated_calls': len(self.deprecated_calls),
            'total_deprecated_calls': sum(
                len(calls) for calls in self.deprecated_calls.values()
            ),
            'migrated_file_list': list(self.migrated_files),
            'deprecated_call_details': self.deprecated_calls
        }
    
    def print_migration_status(self):
        """Print migration status to console."""
        report = self.get_migration_report()
        
        print("Configuration Migration Status")
        print("=" * 40)
        print(f"Migrated files: {report['migrated_files']}")
        print(f"Files with deprecated calls: {report['files_with_deprecated_calls']}")
        print(f"Total deprecated calls: {report['total_deprecated_calls']}")
        
        if report['deprecated_call_details']:
            print("\nFiles needing migration:")
            for file_path, calls in report['deprecated_call_details'].items():
                print(f"  {file_path}: {len(calls)} calls")
                for call in calls[:3]:  # Show first 3 calls
                    print(f"    - {call['env_var']} (line {call['line_number']})")
                if len(calls) > 3:
                    print(f"    ... and {len(calls) - 3} more")


# Global migration tracker
migration_tracker = MigrationTracker()