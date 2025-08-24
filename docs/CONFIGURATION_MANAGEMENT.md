# Configuration Management Guide

Mimir uses a centralized, type-safe configuration management system built with Pydantic BaseSettings. This guide covers configuration usage, migration from scattered environment variables, and best practices.

## Overview

The centralized configuration system provides:

- **Type Safety**: Pydantic validation ensures configuration values are valid
- **Environment Integration**: Automatic environment variable loading with sensible defaults
- **Configuration Validation**: Comprehensive validation with clear error messages
- **Backward Compatibility**: Gradual migration path from existing environment variable usage
- **Hot Reloading**: Runtime configuration updates for development
- **Configuration Management CLI**: Tools for validation, export, and management

## Quick Start

### Basic Usage

```python
from repoindex.config import get_config, get_server_config, get_ai_config

# Get complete configuration
config = get_config()
print(f"Server running on {config.server.host}:{config.server.port}")

# Get specific configuration sections
server_config = get_server_config()
ai_config = get_ai_config()

# Configuration is type-safe and validated
assert isinstance(server_config.port, int)
assert server_config.port > 0
```

### Environment Variables

All configuration can be controlled via environment variables:

```bash
# Server configuration
export MIMIR_UI_HOST=localhost
export MIMIR_UI_PORT=8000
export MIMIR_MAX_WORKERS=4

# AI configuration  
export GOOGLE_API_KEY=your-api-key-here
export GEMINI_MODEL=gemini-1.5-flash

# Logging configuration
export MIMIR_LOG_LEVEL=INFO
export LOG_FORMAT=json

# Load and validate
python -c "from repoindex.config import validate_config; validate_config()"
```

## Configuration Structure

The configuration is organized into logical sections:

### ServerConfig
- **Host/Port**: `MIMIR_UI_HOST`, `MIMIR_UI_PORT`
- **Workers**: `MIMIR_MAX_WORKERS`, `MIMIR_MAX_MEMORY_MB` 
- **Timeouts**: `MIMIR_TIMEOUT`
- **CORS**: `MIMIR_CORS_ORIGINS`

### StorageConfig
- **Paths**: `MIMIR_DATA_PATH`, `MIMIR_CACHE_PATH`, `MIMIR_LOGS_PATH`
- **Container Paths**: `MIMIR_STORAGE_DIR`, `MIMIR_CACHE_DIR`

### PipelineConfig
- **Stage Enablement**: `PIPELINE_ENABLE_ACQUIRE`, `PIPELINE_ENABLE_SERENA`, etc.
- **Timeouts**: `PIPELINE_TIMEOUT_ACQUIRE`, `PIPELINE_TIMEOUT_SERENA`, etc.
- **Limits**: `PIPELINE_MAX_FILE_SIZE`, `PIPELINE_MAX_REPO_SIZE`
- **Git Settings**: `GIT_DEFAULT_BRANCH`, `GIT_CLONE_TIMEOUT`

### AIConfig
- **API Keys**: `GOOGLE_API_KEY`, `GEMINI_API_KEY`
- **Model Settings**: `GEMINI_MODEL`, `GEMINI_MAX_TOKENS`, `GEMINI_TEMPERATURE`
- **Feature Flags**: `MIMIR_ENABLE_GEMINI`, `MIMIR_GEMINI_FALLBACK`

### MonitoringConfig
- **Metrics**: `MIMIR_ENABLE_METRICS`, `MIMIR_METRICS_PORT`
- **Tracing**: `MIMIR_SERVICE_NAME`, `JAEGER_ENDPOINT`, `MIMIR_TRACE_SAMPLE_RATE`
- **Health Checks**: `HEALTHCHECK_INTERVAL`, `HEALTHCHECK_TIMEOUT`
- **Alerts**: `MIMIR_ALERT_WEBHOOK_URL`, `MIMIR_ALERT_SMTP_SERVER`

### LoggingConfig
- **Level/Format**: `MIMIR_LOG_LEVEL`, `LOG_FORMAT`, `MIMIR_LOG_FILE`
- **Rotation**: `LOG_MAX_SIZE`, `LOG_MAX_FILES`
- **Options**: `LOG_INCLUDE_TIMESTAMP`, `LOG_INCLUDE_LEVEL`

### SecurityConfig
- **Authentication**: `MIMIR_REQUIRE_AUTH`, `MIMIR_API_KEYS_FILE`
- **Rate Limiting**: `MIMIR_GLOBAL_RATE_LIMIT`, `MIMIR_IP_RATE_LIMIT`
- **Sandboxing**: `MIMIR_ENABLE_SANDBOXING`, `MIMIR_MAX_MEMORY_MB`
- **Encryption**: `MIMIR_ENABLE_ENCRYPTION`, `MIMIR_MASTER_KEY`

## Configuration Management CLI

The `scripts/config_manager.py` tool provides comprehensive configuration management:

### Validation

```bash
# Validate current configuration
python scripts/config_manager.py validate

# Check environment variables
python scripts/config_manager.py check-env
```

### Export/Import

```bash
# Export configuration to JSON
python scripts/config_manager.py export -o config.json --format json

# Export as environment variables
python scripts/config_manager.py export -o .env.production --format env

# Import configuration from file
python scripts/config_manager.py import config.json --validate
```

### Display Configuration

```bash
# Show configuration summary
python scripts/config_manager.py show --format summary

# Show as JSON
python scripts/config_manager.py show --format json

# Show specific section
python scripts/config_manager.py show --format json --section ai
```

### Templates and Comparison

```bash
# Generate environment template
python scripts/config_manager.py generate-template --format env

# Compare configurations
python scripts/config_manager.py diff --file other_config.json

# Show migration status
python scripts/config_manager.py migration-status
```

## Migration from Environment Variables

### Gradual Migration Strategy

The system supports gradual migration from scattered environment variable access:

1. **Phase 1**: Add centralized config alongside existing environment access
2. **Phase 2**: Update modules to use centralized config with backward compatibility
3. **Phase 3**: Remove direct environment variable access

### Migration Utilities

```python
from repoindex.config_migration import (
    ConfigMigration,
    LegacyConfigAdapter, 
    get_env_with_config,
    migration_tracker
)

# Drop-in replacement for os.getenv
value = get_env_with_config("MIMIR_LOG_LEVEL", "INFO")

# Legacy adapter for existing code
adapter = LegacyConfigAdapter()
host = adapter.get("MIMIR_UI_HOST", "localhost")

# Track migration progress
migration_tracker.print_migration_status()
```

### Deprecation Warnings

Modules using the migration utilities will issue deprecation warnings:

```
DeprecationWarning: Direct access to environment variable 'MIMIR_LOG_LEVEL' is deprecated. 
Use config.logging.log_level instead. Direct access will be removed in version 2.0.0.
```

## Configuration File Format

### JSON Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8000,
    "max_workers": 4,
    "timeout": 300
  },
  "ai": {
    "google_api_key": "your-api-key",
    "gemini_model": "gemini-1.5-flash",
    "gemini_max_tokens": 8192,
    "gemini_temperature": 0.1,
    "enable_gemini": true
  },
  "logging": {
    "log_level": "INFO",
    "log_format": "json"
  },
  "monitoring": {
    "enable_metrics": true,
    "metrics_port": 9100,
    "service_name": "mimir-production"
  }
}
```

### Environment File Format

```bash
# Server configuration
MIMIR_UI_HOST=0.0.0.0
MIMIR_UI_PORT=8000
MIMIR_MAX_WORKERS=4

# AI configuration
GOOGLE_API_KEY=your-api-key
GEMINI_MODEL=gemini-1.5-flash
GEMINI_MAX_TOKENS=8192
GEMINI_TEMPERATURE=0.1
MIMIR_ENABLE_GEMINI=true

# Logging configuration
MIMIR_LOG_LEVEL=INFO
LOG_FORMAT=json
```

## Development Workflow

### Local Development

```python
from repoindex.config import get_config, set_config, MimirConfig

# Create development configuration
dev_config = MimirConfig()
dev_config.development.debug = True
dev_config.development.reload = True
dev_config.logging.log_level = "DEBUG"
dev_config.server.host = "localhost"
dev_config.server.port = 3000

# Set as global configuration
set_config(dev_config)

# Or load from environment
config = MimirConfig.load_from_env()
```

### Configuration Validation

```python
from repoindex.config import get_config, validate_config

# Validate configuration
if not validate_config():
    print("Configuration validation failed!")
    
# Get detailed validation results
config = get_config() 
errors = config.validate()
for error in errors:
    print(f"Configuration issue: {error}")
```

### Hot Reloading

```python
from repoindex.config import reload_config

# Reload configuration from environment
new_config = reload_config()
print(f"Reloaded configuration with log level: {new_config.logging.log_level}")
```

## Production Deployment

### Docker Configuration

```dockerfile
# Set configuration via environment variables
ENV MIMIR_LOG_LEVEL=INFO \
    LOG_FORMAT=json \
    MIMIR_UI_HOST=0.0.0.0 \
    MIMIR_UI_PORT=8000 \
    MIMIR_ENABLE_METRICS=true \
    MIMIR_METRICS_PORT=9100

# Or mount configuration file
COPY config/production.json /app/config.json
ENV MIMIR_CONFIG_FILE=/app/config.json
```

### Docker Compose

```yaml
services:
  mimir:
    image: mimir:latest
    environment:
      # Server configuration
      - MIMIR_UI_HOST=0.0.0.0
      - MIMIR_UI_PORT=8000
      - MIMIR_MAX_WORKERS=8
      
      # AI configuration
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - GEMINI_MODEL=gemini-1.5-flash
      
      # Monitoring configuration
      - MIMIR_ENABLE_METRICS=true
      - MIMIR_METRICS_PORT=9100
      - MIMIR_SERVICE_NAME=mimir-production
      - JAEGER_ENDPOINT=http://jaeger:14268/api/traces
      
      # Logging configuration
      - MIMIR_LOG_LEVEL=INFO
      - LOG_FORMAT=json
    
    # Or use env_file
    env_file:
      - .env.production
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mimir-config
data:
  MIMIR_LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  MIMIR_UI_HOST: "0.0.0.0"
  MIMIR_UI_PORT: "8000"
  MIMIR_ENABLE_METRICS: "true"
  MIMIR_METRICS_PORT: "9100"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mimir
spec:
  template:
    spec:
      containers:
      - name: mimir
        image: mimir:latest
        envFrom:
        - configMapRef:
            name: mimir-config
        - secretRef:
            name: mimir-secrets  # For API keys
```

## Security Considerations

### Sensitive Configuration

- **API Keys**: Use environment variables or secret management systems
- **Passwords**: Never commit to version control
- **Certificates**: Mount as files, reference via paths

```bash
# Secure API key handling
export GOOGLE_API_KEY="$(cat /run/secrets/google_api_key)"
export MIMIR_MASTER_KEY="$(cat /run/secrets/master_key)"

# Certificate paths
export SSL_CERT_PATH="/etc/ssl/certs/mimir.crt"
export SSL_KEY_PATH="/etc/ssl/private/mimir.key"
```

### Configuration Validation

Always validate configuration in production:

```python
from repoindex.config import validate_config

# Validate on startup
if not validate_config():
    raise SystemExit("Invalid configuration - aborting startup")
```

### Secret Scanning

The system includes credential scanning by default:

```python
from repoindex.config import get_security_config

security = get_security_config()
if security.enable_credential_scanning:
    # Scan for leaked credentials in indexed repositories
    pass
```

## Troubleshooting

### Common Issues

1. **Configuration Validation Errors**
   ```bash
   python scripts/config_manager.py validate
   ```

2. **Environment Variable Not Loading**
   ```bash
   python scripts/config_manager.py check-env | grep YOUR_VAR
   ```

3. **Migration Status**
   ```bash  
   python scripts/config_manager.py migration-status
   ```

4. **Type Validation Failures**
   ```python
   from repoindex.config import MimirConfig
   try:
       config = MimirConfig()
   except ValidationError as e:
       print(f"Configuration error: {e}")
   ```

### Debug Mode

Enable debug logging to see configuration loading:

```bash
export MIMIR_LOG_LEVEL=DEBUG
python -c "from repoindex.config import get_config; get_config()"
```

### Configuration Comparison

Compare your configuration with defaults:

```bash
python scripts/config_manager.py diff
```

## Best Practices

### Development
- Use `.env` files for local configuration
- Enable debug mode and verbose logging
- Validate configuration on startup
- Use hot reloading for rapid iteration

### Production
- Use environment variables or configuration files
- Enable metrics and monitoring
- Validate configuration before deployment
- Use structured JSON logging
- Implement proper secret management

### Migration
- Update modules gradually using migration utilities
- Monitor deprecation warnings
- Test backward compatibility thoroughly
- Document configuration changes

### Testing
- Use configuration fixtures in tests
- Test with various environment configurations
- Validate configuration schema changes
- Test migration utilities

## API Reference

### Core Functions

```python
from repoindex.config import (
    get_config,              # Get global configuration
    set_config,              # Set global configuration  
    reload_config,           # Reload from environment
    validate_config,         # Validate current config
    load_config_from_file,   # Load from file
)

# Configuration section getters
from repoindex.config import (
    get_server_config,
    get_storage_config, 
    get_pipeline_config,
    get_ai_config,
    get_monitoring_config,
    get_logging_config,
    get_security_config,
    get_performance_config,
    get_database_config,
    get_development_config,
    get_container_config,
)
```

### Migration Utilities

```python
from repoindex.config_migration import (
    ConfigMigration,         # Migration helper class
    LegacyConfigAdapter,     # Backward compatibility adapter
    get_env_with_config,     # Drop-in os.getenv replacement
    migration_tracker,       # Track migration progress
)
```

### Configuration Classes

```python
from repoindex.config import (
    MimirConfig,            # Main configuration
    ServerConfig,           # Server settings
    StorageConfig,          # Storage paths
    PipelineConfig,         # Pipeline settings
    AIConfig,               # AI/LLM settings
    MonitoringConfig,       # Monitoring/metrics
    LoggingConfig,          # Logging settings
    SecurityConfig,         # Security settings (from existing module)
    PerformanceConfig,      # Performance tuning
    DatabaseConfig,         # Database settings
    DevelopmentConfig,      # Development settings
    ContainerConfig,        # Container settings
)
```

This centralized configuration system provides a robust foundation for managing Mimir's extensive configuration requirements while maintaining backward compatibility and enabling smooth migration from the existing scattered environment variable usage.