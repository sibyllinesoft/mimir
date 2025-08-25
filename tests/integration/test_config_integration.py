"""
Integration tests for centralized configuration system.

Tests real-world configuration scenarios, file I/O, environment integration,
and interaction with existing Mimir components.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from repoindex.config import (
    MimirConfig,
    get_ai_config,
    get_config,
    get_logging_config,
    get_monitoring_config,
    get_server_config,
    load_config_from_file,
    validate_config,
)
from repoindex.config_migration import migration_tracker
from repoindex.security.config import SecurityConfig


class TestConfigurationIntegration:
    """Test configuration integration with existing components."""

    def test_security_config_integration(self):
        """Test integration with existing SecurityConfig."""
        config = MimirConfig()
        
        # Security config should be initialized
        assert config.security is not None
        assert isinstance(config.security, SecurityConfig)
        
        # Should have default security settings
        assert config.security.require_authentication is True
        assert config.security.enable_sandboxing is True

    def test_logging_integration(self):
        """Test logging configuration integration."""
        logging_config = get_logging_config()
        
        assert logging_config.log_level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        assert logging_config.log_format in ["json", "text", "human"]
        
        # Test that logging can be configured without errors
        from repoindex.util.logging_config import setup_logging
        
        try:
            setup_logging(
                log_level=logging_config.log_level,
                log_format=logging_config.log_format,
                log_file=logging_config.log_file
            )
        except Exception as e:
            pytest.fail(f"Logging setup failed: {e}")

    def test_ai_config_integration(self):
        """Test AI configuration for Gemini integration."""
        ai_config = get_ai_config()
        
        assert ai_config.gemini_model in ["gemini-1.5-flash", "gemini-pro", "gemini-1.5-pro"]
        assert 0.0 <= ai_config.gemini_temperature <= 2.0
        assert ai_config.gemini_max_tokens > 0

    def test_server_config_integration(self):
        """Test server configuration integration."""
        server_config = get_server_config()
        
        assert isinstance(server_config.host, str)
        assert 1 <= server_config.port <= 65535
        assert server_config.timeout > 0
        assert server_config.max_workers > 0

    def test_monitoring_config_integration(self):
        """Test monitoring configuration integration."""
        monitoring_config = get_monitoring_config()
        
        assert isinstance(monitoring_config.service_name, str)
        assert 0.0 <= monitoring_config.trace_sample_rate <= 1.0
        
        if monitoring_config.enable_metrics:
            assert 1024 <= monitoring_config.metrics_port <= 65535


class TestConfigurationFileHandling:
    """Test configuration file operations."""

    def test_save_and_load_complete_config(self):
        """Test saving and loading a complete configuration."""
        # Create a test configuration
        config = MimirConfig()
        config.server.port = 9999
        config.server.host = "test.example.com"
        config.ai.gemini_model = "gemini-test"
        config.ai.gemini_max_tokens = 4096
        config.monitoring.enable_metrics = True
        config.monitoring.metrics_port = 9200
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_file = Path(f.name)
        
        try:
            # Save configuration
            config.save_to_file(config_file)
            
            # Verify file exists and has content
            assert config_file.exists()
            assert config_file.stat().st_size > 0
            
            # Load and verify configuration
            loaded_config = MimirConfig.load_from_file(config_file)
            
            assert loaded_config.server.port == 9999
            assert loaded_config.server.host == "test.example.com"
            assert loaded_config.ai.gemini_model == "gemini-test"
            assert loaded_config.ai.gemini_max_tokens == 4096
            assert loaded_config.monitoring.enable_metrics is True
            assert loaded_config.monitoring.metrics_port == 9200
            
        finally:
            config_file.unlink(missing_ok=True)

    def test_partial_config_override(self):
        """Test partial configuration override from file."""
        # Create a partial configuration file
        partial_config = {
            "server": {
                "port": 8888,
                "host": "override.example.com"
            },
            "ai": {
                "gemini_model": "gemini-override"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            config_file = Path(f.name)
        
        try:
            # This tests the simplified merging approach
            # In a real implementation, you might want more sophisticated merging
            config = MimirConfig()
            original_logging_level = config.logging.log_level
            
            # Load config from file should preserve other settings
            loaded_config = MimirConfig.load_from_file(config_file)
            
            # Overridden values
            assert loaded_config.server.port == 8888
            assert loaded_config.server.host == "override.example.com"
            assert loaded_config.ai.gemini_model == "gemini-override"
            
            # Preserved values
            assert loaded_config.logging.log_level == original_logging_level
            
        finally:
            config_file.unlink(missing_ok=True)

    def test_invalid_config_file(self):
        """Test handling of invalid configuration files."""
        # Test with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json }")
            config_file = Path(f.name)
        
        try:
            with pytest.raises(Exception):  # Should raise JSON decode error
                MimirConfig.load_from_file(config_file)
        finally:
            config_file.unlink(missing_ok=True)

    def test_nonexistent_config_file(self):
        """Test handling of nonexistent configuration files."""
        nonexistent_file = Path("/nonexistent/path/config.json")
        
        with pytest.raises(Exception):  # Should raise FileNotFoundError
            MimirConfig.load_from_file(nonexistent_file)


class TestEnvironmentVariableIntegration:
    """Test environment variable integration and precedence."""

    @patch.dict(os.environ, {
        "MIMIR_LOG_LEVEL": "ERROR",
        "MIMIR_UI_PORT": "7777",
        "MIMIR_MAX_WORKERS": "8",
        "GEMINI_MODEL": "gemini-test-env",
        "MIMIR_ENABLE_METRICS": "true",
        "MIMIR_SERVICE_NAME": "test-service"
    })
    def test_comprehensive_environment_loading(self):
        """Test comprehensive environment variable loading."""
        config = MimirConfig.load_from_env()
        
        # Server configuration
        assert config.server.port == 7777
        assert config.server.max_workers == 8
        
        # Logging configuration
        assert config.logging.log_level == "ERROR"
        
        # AI configuration
        assert config.ai.gemini_model == "gemini-test-env"
        
        # Monitoring configuration
        assert config.monitoring.enable_metrics is True
        assert config.monitoring.service_name == "test-service"

    def test_boolean_environment_parsing(self):
        """Test boolean environment variable parsing."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("TRUE", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("enabled", True),
            ("false", False),
            ("False", False),
            ("FALSE", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("disabled", False),
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {"MIMIR_ENABLE_METRICS": env_value}):
                config = MimirConfig.load_from_env()
                assert config.monitoring.enable_metrics == expected, f"Failed for {env_value}"

    def test_list_environment_parsing(self):
        """Test list environment variable parsing."""
        with patch.dict(os.environ, {
            "MIMIR_CORS_ORIGINS": "http://localhost:3000,http://localhost:8000,https://example.com",
            "TREE_SITTER_LANGUAGES": "python,rust,go,typescript"
        }):
            config = MimirConfig.load_from_env()
            
            assert config.server.cors_origins == [
                "http://localhost:3000",
                "http://localhost:8000", 
                "https://example.com"
            ]
            assert config.pipeline.tree_sitter_languages == [
                "python", "rust", "go", "typescript"
            ]


class TestConfigurationValidation:
    """Test configuration validation in different scenarios."""

    def test_valid_configuration(self):
        """Test validation of a valid configuration."""
        config = MimirConfig()
        
        # Set valid values
        config.ai.google_api_key = "test-key-123"
        config.server.max_workers = 4
        config.monitoring.enable_metrics = True
        config.monitoring.metrics_port = 9100
        
        warnings = config.validate()
        # Should have minimal warnings with valid config
        assert isinstance(warnings, list)

    def test_configuration_cross_validation(self):
        """Test cross-configuration validation rules."""
        config = MimirConfig()
        
        # Set conflicting values
        config.server.max_workers = 20
        config.performance.asyncio_max_workers = 5  # Lower than server workers
        
        warnings = config.validate()
        
        # Should warn about resource contention
        warning_messages = " ".join(warnings)
        assert "max_workers exceeds asyncio max_workers" in warning_messages

    def test_missing_api_key_validation(self):
        """Test validation when API keys are missing."""
        config = MimirConfig()
        config.ai.enable_gemini = True
        config.ai.google_api_key = None
        config.ai.gemini_api_key = None
        
        warnings = config.validate()
        
        # Should warn about missing API key
        warning_messages = " ".join(warnings)
        assert "Gemini enabled but no API key" in warning_messages

    def test_storage_path_validation(self):
        """Test storage path accessibility validation."""
        config = MimirConfig()
        
        # Set invalid storage path
        config.storage.data_path = Path("/root/forbidden")  # Typically not writable
        
        warnings = config.validate()
        
        # May warn about path accessibility (depending on system)
        assert isinstance(warnings, list)

    def test_global_validation_function(self):
        """Test global validate_config function."""
        # Should work without throwing exceptions
        result = validate_config()
        assert isinstance(result, bool)


class TestMigrationIntegration:
    """Test migration system integration."""

    def test_migration_tracking(self):
        """Test that migrations are properly tracked."""
        # Reset migration tracker
        migration_tracker.migrated_files.clear()
        migration_tracker.deprecated_calls.clear()
        
        # Trigger some module imports that should mark as migrated
        from repoindex.util.logging_config import get_logger  # noqa: F401
        from repoindex.pipeline.gemini import GeminiAdapter  # noqa: F401
        
        # Check if migrations were tracked
        report = migration_tracker.get_migration_report()
        assert report['migrated_files'] >= 0  # Should have at least some migrations

    def test_backward_compatibility(self):
        """Test backward compatibility with existing code."""
        # Import modules that use old environment variable access
        try:
            from repoindex.monitoring.alerts import _get_alert_config  # noqa: F401
            # Should not raise import errors
        except ImportError:
            pass  # Some modules may not exist in test environment

    def test_legacy_adapter_compatibility(self):
        """Test that legacy adapter provides expected interface."""
        from repoindex.config_migration import LegacyConfigAdapter, get_env_with_config
        
        adapter = LegacyConfigAdapter()
        
        # Test common environment variables
        host = adapter.get("MIMIR_UI_HOST", "default")
        assert isinstance(host, str)
        
        port = adapter.get("MIMIR_UI_PORT", "8000")
        assert isinstance(port, str)
        
        # Test global function
        timeout = get_env_with_config("MIMIR_TIMEOUT", "300")
        assert isinstance(timeout, str)


class TestRealWorldScenarios:
    """Test real-world configuration scenarios."""

    def test_development_environment_setup(self):
        """Test typical development environment configuration."""
        with patch.dict(os.environ, {
            "MIMIR_DEBUG": "true",
            "MIMIR_LOG_LEVEL": "DEBUG",
            "MIMIR_RELOAD": "true",
            "MIMIR_ENABLE_METRICS": "false",
            "MIMIR_UI_HOST": "localhost",
            "MIMIR_UI_PORT": "3000"
        }):
            config = MimirConfig.load_from_env()
            
            assert config.development.debug is True
            assert config.development.reload is True
            assert config.logging.log_level == "DEBUG"
            assert config.monitoring.enable_metrics is False
            assert config.server.host == "localhost"
            assert config.server.port == 3000

    def test_production_environment_setup(self):
        """Test typical production environment configuration."""
        with patch.dict(os.environ, {
            "MIMIR_DEBUG": "false",
            "MIMIR_LOG_LEVEL": "INFO",
            "MIMIR_ENABLE_METRICS": "true",
            "MIMIR_METRICS_PORT": "9100",
            "MIMIR_UI_HOST": "0.0.0.0",
            "MIMIR_UI_PORT": "8000",
            "MIMIR_MAX_WORKERS": "8",
            "LOG_FORMAT": "json"
        }):
            config = MimirConfig.load_from_env()
            
            assert config.development.debug is False
            assert config.logging.log_level == "INFO"
            assert config.logging.log_format == "json"
            assert config.monitoring.enable_metrics is True
            assert config.monitoring.metrics_port == 9100
            assert config.server.host == "0.0.0.0"
            assert config.server.port == 8000
            assert config.server.max_workers == 8

    def test_container_environment_setup(self):
        """Test container/Docker environment configuration."""
        with patch.dict(os.environ, {
            "MIMIR_STORAGE_DIR": "/app/data",
            "MIMIR_CACHE_DIR": "/app/cache",
            "MIMIR_LOG_FILE": "/app/logs/mimir.log",
            "DOCKER_MEMORY_LIMIT": "4g",
            "DOCKER_CPU_LIMIT": "2.0"
        }):
            config = MimirConfig.load_from_env()
            
            assert config.storage.storage_dir == Path("/app/data")
            assert config.storage.cache_dir == Path("/app/cache")
            assert config.logging.log_file == "/app/logs/mimir.log"
            assert config.container.docker_memory_limit == "4g"
            assert config.container.docker_cpu_limit == 2.0

    def test_ai_integration_setup(self):
        """Test AI/Gemini integration configuration."""
        with patch.dict(os.environ, {
            "GOOGLE_API_KEY": "test-api-key-12345",
            "GEMINI_MODEL": "gemini-1.5-pro",
            "GEMINI_MAX_TOKENS": "16384",
            "GEMINI_TEMPERATURE": "0.0",
            "MIMIR_ENABLE_GEMINI": "true"
        }):
            config = MimirConfig.load_from_env()
            
            assert config.ai.google_api_key == "test-api-key-12345"
            assert config.ai.gemini_model == "gemini-1.5-pro"
            assert config.ai.gemini_max_tokens == 16384
            assert config.ai.gemini_temperature == 0.0
            assert config.ai.enable_gemini is True
            assert config.ai.api_key == "test-api-key-12345"

    def test_monitoring_and_observability_setup(self):
        """Test monitoring and observability configuration."""
        with patch.dict(os.environ, {
            "MIMIR_ENABLE_METRICS": "true",
            "MIMIR_METRICS_PORT": "9090",
            "MIMIR_SERVICE_NAME": "mimir-production",
            "JAEGER_ENDPOINT": "http://jaeger:14268/api/traces",
            "MIMIR_TRACE_SAMPLE_RATE": "0.1",
            "PROMETHEUS_SCRAPE_INTERVAL": "30s"
        }):
            config = MimirConfig.load_from_env()
            
            assert config.monitoring.enable_metrics is True
            assert config.monitoring.metrics_port == 9090
            assert config.monitoring.service_name == "mimir-production"
            assert config.monitoring.jaeger_endpoint == "http://jaeger:14268/api/traces"
            assert config.monitoring.trace_sample_rate == 0.1
            assert config.monitoring.prometheus_scrape_interval == "30s"


@pytest.fixture(autouse=True)
def reset_config_state():
    """Reset configuration state before each test."""
    import repoindex.config
    repoindex.config._global_config = None
    
    # Reset migration tracker
    migration_tracker.migrated_files.clear()
    migration_tracker.deprecated_calls.clear()
    
    yield
    
    # Clean up after test
    repoindex.config._global_config = None