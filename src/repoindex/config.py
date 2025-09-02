"""
Centralized configuration management for Mimir.

Provides type-safe configuration handling using Pydantic BaseSettings with
comprehensive validation, environment variable support, and backward compatibility.
Implements 12-factor app principles for configuration management.
"""

import os
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
try:
    from pydantic_settings import BaseSettings
except ImportError:
    # Fallback for older pydantic versions
    from pydantic import BaseSettings

from .security.config import SecurityConfig
from .util.log import get_logger

logger = get_logger(__name__)


class ServerConfig(BaseSettings):
    """Server and UI configuration."""
    
    # Core server settings
    host: str = Field(default="0.0.0.0", env="MIMIR_UI_HOST")
    port: int = Field(default=8000, env="MIMIR_UI_PORT")
    
    # Reverse proxy settings
    behind_proxy: bool = Field(default=False, env="MIMIR_UI_BEHIND_PROXY")
    proxy_headers: bool = Field(default=False, env="MIMIR_UI_PROXY_HEADERS")
    
    # Request handling
    timeout: int = Field(default=300, env="MIMIR_TIMEOUT")
    max_workers: int = Field(default=4, env="MIMIR_MAX_WORKERS")
    max_memory_mb: int = Field(default=1024, env="MIMIR_MAX_MEMORY_MB")
    
    # CORS settings
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="MIMIR_CORS_ORIGINS"
    )
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @field_validator("max_workers")
    @classmethod
    def validate_max_workers(cls, v):
        """Validate max workers is reasonable."""
        if v <= 0:
            raise ValueError("max_workers must be positive")
        if v > 32:
            raise ValueError("max_workers should not exceed 32")
        return v
    
    model_config = {"env_prefix": "MIMIR_"}


class StorageConfig(BaseSettings):
    """Storage and data path configuration."""
    
    # Primary data paths
    data_path: Path = Field(default=Path("./data"), env="MIMIR_DATA_PATH")
    cache_path: Path = Field(default=Path("./cache"), env="MIMIR_CACHE_PATH")
    logs_path: Path = Field(default=Path("./logs"), env="MIMIR_LOGS_PATH")
    
    # Container paths (for Docker deployments)
    storage_dir: Path = Field(default=Path("/app/data"), env="MIMIR_STORAGE_DIR")
    cache_dir: Path = Field(default=Path("/app/cache"), env="MIMIR_CACHE_DIR")
    
    @field_validator("data_path", "cache_path", "logs_path", mode="before")
    @classmethod
    def resolve_paths(cls, v):
        """Resolve and expand paths."""
        path = Path(v).expanduser().resolve()
        # Create directories if they don't exist
        try:
            path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directory {path}: {e}")
        return path
    
    model_config = {"env_prefix": "MIMIR_"}


class PipelineConfig(BaseSettings):
    """Pipeline stage configuration."""
    
    # Pipeline stage enablement
    enable_acquire: bool = Field(default=True, env="PIPELINE_ENABLE_ACQUIRE")
    enable_repomapper: bool = Field(default=True, env="PIPELINE_ENABLE_REPOMAPPER")
    enable_serena: bool = Field(default=True, env="PIPELINE_ENABLE_SERENA")
    enable_leann: bool = Field(default=True, env="PIPELINE_ENABLE_LEANN")
    enable_snippets: bool = Field(default=True, env="PIPELINE_ENABLE_SNIPPETS")
    enable_bundle: bool = Field(default=True, env="PIPELINE_ENABLE_BUNDLE")
    
    # Pipeline timeouts (seconds)
    timeout_acquire: int = Field(default=60, env="PIPELINE_TIMEOUT_ACQUIRE")
    timeout_repomapper: int = Field(default=120, env="PIPELINE_TIMEOUT_REPOMAPPER")
    timeout_serena: int = Field(default=180, env="PIPELINE_TIMEOUT_SERENA")
    timeout_leann: int = Field(default=240, env="PIPELINE_TIMEOUT_LEANN")
    timeout_snippets: int = Field(default=60, env="PIPELINE_TIMEOUT_SNIPPETS")
    timeout_bundle: int = Field(default=30, env="PIPELINE_TIMEOUT_BUNDLE")
    
    # File size limits
    max_file_size_mb: int = Field(default=10, env="PIPELINE_MAX_FILE_SIZE")
    max_repo_size_mb: int = Field(default=1000, env="PIPELINE_MAX_REPO_SIZE")
    
    # Tree-sitter configuration
    tree_sitter_languages: list[str] = Field(
        default=["typescript", "javascript", "python", "rust", "go", "java"],
        env="TREE_SITTER_LANGUAGES"
    )
    
    # Git configuration
    git_default_branch: str = Field(default="main", env="GIT_DEFAULT_BRANCH")
    git_clone_timeout: int = Field(default=300, env="GIT_CLONE_TIMEOUT")
    
    @field_validator("tree_sitter_languages", mode="before")
    @classmethod
    def parse_languages(cls, v):
        """Parse comma-separated languages."""
        if isinstance(v, str):
            return [lang.strip() for lang in v.split(",")]
        return v
    
    @field_validator("timeout_acquire", "timeout_repomapper", "timeout_serena", "timeout_leann", "timeout_snippets", "timeout_bundle")
    @classmethod
    def validate_timeouts(cls, v):
        """Validate timeouts are reasonable."""
        if v <= 0:
            raise ValueError("Timeouts must be positive")
        if v > 3600:  # 1 hour max
            raise ValueError("Timeouts should not exceed 1 hour")
        return v
    
    model_config = {"env_prefix": "PIPELINE_"}


class QueryConfig(BaseSettings):
    """Query transformation configuration."""
    
    enable_hyde: bool = Field(default=False, env="QUERY_ENABLE_HYDE")
    transformer_provider: Literal["gemini", "ollama"] = Field(default="ollama", env="QUERY_TRANSFORMER_PROVIDER")
    transformer_model: str | None = Field(default=None, env="QUERY_TRANSFORMER_MODEL")
    
    model_config = {"env_prefix": ""}


class RerankerConfig(BaseSettings):
    """Cross-encoder reranking configuration."""
    
    enabled: bool = Field(default=False, env="QUERY_RERANKER_ENABLED")
    model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", env="QUERY_RERANKER_MODEL")
    top_k: int = Field(default=20, env="QUERY_RERANKER_TOP_K")
    initial_retrieval_k: int = Field(default=100, env="QUERY_INITIAL_K")
    
    model_config = {"env_prefix": ""}


class AIConfig(BaseSettings):
    """AI and LLM integration configuration."""
    
    # Google Gemini configuration
    google_api_key: str | None = Field(default=None, env="GOOGLE_API_KEY")
    gemini_api_key: str | None = Field(default=None, env="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-flash", env="GEMINI_MODEL")
    gemini_max_tokens: int = Field(default=8192, env="GEMINI_MAX_TOKENS")
    gemini_temperature: float = Field(default=0.1, env="GEMINI_TEMPERATURE")
    
    # Ollama configuration  
    ollama_host: str = Field(default="localhost", env="OLLAMA_HOST")
    ollama_port: int = Field(default=11434, env="OLLAMA_PORT")
    ollama_model: str = Field(default="llama3.2:3b", env="OLLAMA_MODEL")
    ollama_max_tokens: int = Field(default=8192, env="OLLAMA_MAX_TOKENS")
    ollama_temperature: float = Field(default=0.1, env="OLLAMA_TEMPERATURE")
    ollama_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")
    
    # Feature flags
    enable_gemini: bool = Field(default=True, env="MIMIR_ENABLE_GEMINI")
    gemini_fallback: bool = Field(default=True, env="MIMIR_GEMINI_FALLBACK")
    enable_ollama: bool = Field(default=True, env="MIMIR_ENABLE_OLLAMA")
    
    # Default LLM provider selection
    default_llm_provider: str = Field(default="ollama", env="MIMIR_DEFAULT_LLM_PROVIDER")
    
    # Vector embeddings configuration
    embedding_service_url: str | None = Field(default=None, env="EMBEDDING_SERVICE_URL")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="EMBEDDING_MODEL"
    )
    
    # Enhanced embedding models for code
    code_embedding_model: str = Field(
        default="microsoft/codebert-base",
        env="CODE_EMBEDDING_MODEL"
    )
    enable_code_embeddings: bool = Field(default=True, env="MIMIR_ENABLE_CODE_EMBEDDINGS")
    
    # RAPTOR hierarchical indexing configuration
    enable_raptor: bool = Field(default=False, env="MIMIR_ENABLE_RAPTOR")
    raptor_cluster_threshold: float = Field(default=0.1, env="RAPTOR_CLUSTER_THRESHOLD")
    raptor_max_clusters: int = Field(default=10, env="RAPTOR_MAX_CLUSTERS")
    raptor_summarization_model: str = Field(default="llama3.2:3b", env="RAPTOR_SUMMARIZATION_MODEL")
    raptor_embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        env="RAPTOR_EMBEDDING_MODEL"
    )
    
    # Query transformation configuration
    query: QueryConfig = Field(default_factory=QueryConfig)
    
    # Reranking configuration  
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    
    # Legacy fields for backward compatibility
    enable_hyde: bool = Field(default=False, env="MIMIR_ENABLE_HYDE")
    hyde_model: str = Field(default="llama3.2:3b", env="HYDE_MODEL")
    hyde_num_hypotheses: int = Field(default=3, env="HYDE_NUM_HYPOTHESES")
    enable_reranking: bool = Field(default=False, env="MIMIR_ENABLE_RERANKING")
    reranking_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        env="RERANKING_MODEL"
    )
    reranking_top_k: int = Field(default=20, env="RERANKING_TOP_K")
    
    @property
    def api_key(self) -> str | None:
        """Get the API key, preferring GOOGLE_API_KEY over GEMINI_API_KEY."""
        return self.google_api_key or self.gemini_api_key
    
    @property 
    def ollama_base_url(self) -> str:
        """Get the full Ollama base URL."""
        return f"http://{self.ollama_host}:{self.ollama_port}"
    
    @field_validator("gemini_temperature", "ollama_temperature", "raptor_cluster_threshold")
    @classmethod
    def validate_temperature(cls, v):
        """Validate temperature is in valid range."""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    @field_validator("gemini_max_tokens", "ollama_max_tokens")
    @classmethod
    def validate_max_tokens(cls, v):
        """Validate max tokens is reasonable."""
        if v <= 0:
            raise ValueError("Max tokens must be positive")
        if v > 32768:  # Reasonable upper limit
            raise ValueError("Max tokens should not exceed 32768")
        return v
    
    @field_validator("default_llm_provider")
    @classmethod
    def validate_llm_provider(cls, v):
        """Validate LLM provider is supported."""
        supported_providers = ["ollama", "gemini", "mock"]
        if v not in supported_providers:
            raise ValueError(f"LLM provider must be one of: {supported_providers}")
        return v
    
    @field_validator("ollama_port")
    @classmethod
    def validate_ollama_port(cls, v):
        """Validate Ollama port is in valid range."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class MonitoringConfig(BaseSettings):
    """Monitoring, metrics, and alerting configuration."""
    
    # Metrics and profiling
    enable_metrics: bool = Field(default=False, env="MIMIR_ENABLE_METRICS")
    metrics_port: int = Field(default=9100, env="MIMIR_METRICS_PORT")
    enable_profiling: bool = Field(default=False, env="MIMIR_ENABLE_PROFILING")
    
    # Prometheus configuration
    prometheus_scrape_interval: str = Field(default="15s", env="PROMETHEUS_SCRAPE_INTERVAL")
    prometheus_evaluation_interval: str = Field(default="15s", env="PROMETHEUS_EVALUATION_INTERVAL")
    
    # Tracing configuration
    service_name: str = Field(default="mimir-repoindex", env="MIMIR_SERVICE_NAME")
    jaeger_endpoint: str | None = Field(default=None, env="JAEGER_ENDPOINT")
    otlp_endpoint: str | None = Field(default=None, env="OTEL_EXPORTER_OTLP_ENDPOINT")
    trace_console: bool = Field(default=False, env="MIMIR_TRACE_CONSOLE")
    trace_sample_rate: float = Field(default=1.0, env="MIMIR_TRACE_SAMPLE_RATE")
    
    # Health check configuration
    healthcheck_interval: str = Field(default="30s", env="HEALTHCHECK_INTERVAL")
    healthcheck_timeout: str = Field(default="10s", env="HEALTHCHECK_TIMEOUT")
    healthcheck_retries: int = Field(default=3, env="HEALTHCHECK_RETRIES")
    
    # Alert configuration
    alert_webhook_url: str | None = Field(default=None, env="MIMIR_ALERT_WEBHOOK_URL")
    alert_smtp_server: str | None = Field(default=None, env="MIMIR_ALERT_SMTP_SERVER")
    alert_smtp_port: int = Field(default=587, env="MIMIR_ALERT_SMTP_PORT")
    alert_email_from: str | None = Field(default=None, env="MIMIR_ALERT_EMAIL_FROM")
    alert_email_to: str | None = Field(default=None, env="MIMIR_ALERT_EMAIL_TO")
    alert_smtp_username: str | None = Field(default=None, env="MIMIR_ALERT_SMTP_USERNAME")
    alert_smtp_password: str | None = Field(default=None, env="MIMIR_ALERT_SMTP_PASSWORD")
    alert_smtp_tls: bool = Field(default=True, env="MIMIR_ALERT_SMTP_TLS")
    
    # Grafana configuration
    grafana_admin_password: str = Field(default="admin", env="GRAFANA_ADMIN_PASSWORD")
    
    @field_validator("trace_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v):
        """Validate sample rate is in valid range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Sample rate must be between 0.0 and 1.0")
        return v
    
    @field_validator("metrics_port")
    @classmethod
    def validate_metrics_port(cls, v):
        """Validate metrics port is reasonable."""
        if not 1024 <= v <= 65535:
            raise ValueError("Metrics port must be between 1024 and 65535")
        return v
    
    model_config = {"env_prefix": "MIMIR_"}


class LoggingConfig(BaseSettings):
    """Logging configuration."""
    
    # Log level and format
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", 
        env="MIMIR_LOG_LEVEL"
    )
    log_format: Literal["json", "text", "human"] = Field(
        default="json", 
        env="LOG_FORMAT"
    )
    log_file: str | None = Field(default=None, env="MIMIR_LOG_FILE")
    
    # Log rotation
    log_max_size: str = Field(default="50m", env="LOG_MAX_SIZE")
    log_max_files: int = Field(default=10, env="LOG_MAX_FILES")
    
    # Structured logging options
    log_include_timestamp: bool = Field(default=True, env="LOG_INCLUDE_TIMESTAMP")
    log_include_level: bool = Field(default=True, env="LOG_INCLUDE_LEVEL")
    log_include_logger: bool = Field(default=True, env="LOG_INCLUDE_LOGGER")
    log_include_thread: bool = Field(default=False, env="LOG_INCLUDE_THREAD")
    
    model_config = {"env_prefix": "MIMIR_"}


class PerformanceConfig(BaseSettings):
    """Performance tuning configuration."""
    
    # Async I/O configuration
    asyncio_max_workers: int = Field(default=10, env="ASYNCIO_MAX_WORKERS")
    asyncio_semaphore_limit: int = Field(default=20, env="ASYNCIO_SEMAPHORE_LIMIT")
    
    # Memory management
    python_gc_threshold: str = Field(default="700,10,10", env="PYTHON_GC_THRESHOLD")
    python_malloc_stats: bool = Field(default=False, env="PYTHON_MALLOC_STATS")
    
    # File I/O buffer sizes
    file_read_buffer_size: int = Field(default=8192, env="FILE_READ_BUFFER_SIZE")
    file_write_buffer_size: int = Field(default=8192, env="FILE_WRITE_BUFFER_SIZE")
    
    @field_validator("asyncio_max_workers", "asyncio_semaphore_limit")
    @classmethod
    def validate_positive(cls, v):
        """Validate value is positive."""
        if v <= 0:
            raise ValueError("Value must be positive")
        return v
    
    @field_validator("file_read_buffer_size", "file_write_buffer_size")
    @classmethod
    def validate_buffer_sizes(cls, v):
        """Validate buffer sizes are reasonable."""
        if not 1024 <= v <= 1024 * 1024:  # 1KB to 1MB
            raise ValueError("Buffer size must be between 1KB and 1MB")
        return v
    
    model_config = {"env_prefix": ""}


class DatabaseConfig(BaseSettings):
    """Database and caching configuration."""
    
    # Redis configuration
    redis_url: str | None = Field(default=None, env="REDIS_URL")
    redis_password: str | None = Field(default=None, env="REDIS_PASSWORD")
    
    # Database configuration
    database_url: str | None = Field(default=None, env="DATABASE_URL")
    
    model_config = {"env_prefix": ""}


class LensConfig(BaseSettings):
    """Lens indexing service integration configuration."""
    
    # Lens service connection
    enabled: bool = Field(default=False, env="LENS_ENABLED")
    base_url: str = Field(default="http://localhost:3001", env="LENS_BASE_URL")
    api_key: str | None = Field(default=None, env="LENS_API_KEY")
    
    # Connection settings
    timeout: int = Field(default=30, env="LENS_TIMEOUT")
    max_retries: int = Field(default=3, env="LENS_MAX_RETRIES")
    retry_delay: float = Field(default=1.0, env="LENS_RETRY_DELAY")
    
    # Health check settings
    health_check_enabled: bool = Field(default=True, env="LENS_HEALTH_CHECK_ENABLED")
    health_check_interval: int = Field(default=60, env="LENS_HEALTH_CHECK_INTERVAL")
    health_check_timeout: int = Field(default=10, env="LENS_HEALTH_CHECK_TIMEOUT")
    
    # Fallback behavior
    fallback_enabled: bool = Field(default=True, env="LENS_FALLBACK_ENABLED")
    fallback_to_local: bool = Field(default=True, env="LENS_FALLBACK_TO_LOCAL")
    
    # Performance settings
    connection_pool_size: int = Field(default=10, env="LENS_CONNECTION_POOL_SIZE")
    keep_alive_timeout: int = Field(default=30, env="LENS_KEEP_ALIVE_TIMEOUT")
    
    # Integration features
    enable_indexing: bool = Field(default=True, env="LENS_ENABLE_INDEXING")
    enable_search: bool = Field(default=True, env="LENS_ENABLE_SEARCH")
    enable_embeddings: bool = Field(default=True, env="LENS_ENABLE_EMBEDDINGS")
    
    # GPU acceleration settings (for future Lens GPU support)
    prefer_gpu: bool = Field(default=False, env="LENS_PREFER_GPU")
    gpu_device_id: int = Field(default=0, env="LENS_GPU_DEVICE_ID") 
    gpu_memory_limit: str = Field(default="auto", env="LENS_GPU_MEMORY_LIMIT")
    gpu_batch_size: int = Field(default=32, env="LENS_GPU_BATCH_SIZE")
    
    # GPU model preferences (when Lens supports GPU)
    gpu_embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="LENS_GPU_EMBEDDING_MODEL")
    cpu_fallback_model: str = Field(default="all-MiniLM-L6-v2", env="LENS_CPU_FALLBACK_MODEL")
    
    @field_validator("timeout", "health_check_timeout")
    @classmethod
    def validate_timeouts(cls, v):
        """Validate timeouts are reasonable."""
        if v <= 0:
            raise ValueError("Timeout must be positive")
        if v > 300:  # 5 minutes max
            raise ValueError("Timeout should not exceed 300 seconds")
        return v
    
    @field_validator("max_retries")
    @classmethod
    def validate_max_retries(cls, v):
        """Validate max retries is reasonable."""
        if v < 0:
            raise ValueError("Max retries cannot be negative")
        if v > 10:
            raise ValueError("Max retries should not exceed 10")
        return v
    
    @field_validator("retry_delay")
    @classmethod
    def validate_retry_delay(cls, v):
        """Validate retry delay is reasonable."""
        if v < 0.1:
            raise ValueError("Retry delay must be at least 0.1 seconds")
        if v > 60.0:
            raise ValueError("Retry delay should not exceed 60 seconds")
        return v
    
    @field_validator("connection_pool_size")
    @classmethod
    def validate_pool_size(cls, v):
        """Validate connection pool size is reasonable."""
        if v <= 0:
            raise ValueError("Connection pool size must be positive")
        if v > 100:
            raise ValueError("Connection pool size should not exceed 100")
        return v
    
    model_config = {"env_prefix": "LENS_"}


class DevelopmentConfig(BaseSettings):
    """Development and testing configuration."""
    
    # Development mode
    debug: bool = Field(default=False, env="MIMIR_DEBUG")
    reload: bool = Field(default=False, env="MIMIR_RELOAD")
    
    # Testing configuration
    pytest_timeout: int = Field(default=60, env="PYTEST_TIMEOUT")
    pytest_asyncio_mode: str = Field(default="auto", env="PYTEST_ASYNCIO_MODE")
    
    model_config = {"env_prefix": "MIMIR_"}


class ContainerConfig(BaseSettings):
    """Container orchestration configuration."""
    
    # Docker resource limits
    docker_memory_limit: str = Field(default="2g", env="DOCKER_MEMORY_LIMIT")
    docker_cpu_limit: float = Field(default=2.0, env="DOCKER_CPU_LIMIT")
    docker_memory_reservation: str = Field(default="512m", env="DOCKER_MEMORY_RESERVATION")
    docker_cpu_reservation: float = Field(default=0.5, env="DOCKER_CPU_RESERVATION")
    
    # Network configuration
    docker_network_subnet: str = Field(default="172.20.0.0/16", env="DOCKER_NETWORK_SUBNET")
    
    # Service ports
    mimir_server_port: int = Field(default=8000, env="MIMIR_SERVER_PORT")
    mimir_ui_service_port: int = Field(default=8080, env="MIMIR_UI_SERVICE_PORT")
    prometheus_port: int = Field(default=9090, env="PROMETHEUS_PORT")
    grafana_port: int = Field(default=3000, env="GRAFANA_PORT")
    loki_port: int = Field(default=3100, env="LOKI_PORT")
    
    model_config = {"env_prefix": ""}


class MimirConfig(BaseSettings):
    """Main Mimir configuration aggregating all subsystems."""
    
    # Subsystem configurations
    server: ServerConfig = Field(default_factory=ServerConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    ai: AIConfig = Field(default_factory=AIConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    lens: LensConfig = Field(default_factory=LensConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    container: ContainerConfig = Field(default_factory=ContainerConfig)
    
    # Security configuration (integrated from existing system)
    security: SecurityConfig | None = Field(default=None)
    
    def __init__(self, **kwargs):
        """Initialize configuration with security config integration."""
        super().__init__(**kwargs)
        
        # Initialize security config if not provided
        if self.security is None:
            self.security = SecurityConfig.from_environment()
            self.security.apply_defaults()
    
    @classmethod
    def load_from_env(cls) -> "MimirConfig":
        """Load configuration from environment variables."""
        try:
            config = cls()
            
            # Validate configuration
            errors = config.validate()
            if errors:
                logger.warning(f"Configuration validation warnings: {errors}")
            
            logger.info("Configuration loaded successfully from environment")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from environment: {e}")
            raise
    
    @classmethod
    def load_from_file(cls, config_file: Path) -> "MimirConfig":
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
            
        Returns:
            Loaded configuration
        """
        try:
            import json
            
            with open(config_file) as f:
                config_data = json.load(f)
            
            # Initialize with environment variables first, then override with file
            config = cls()
            
            # Apply file overrides (this is a simplified approach)
            # In practice, you might want more sophisticated merging
            for section_name, section_data in config_data.items():
                if hasattr(config, section_name):
                    section = getattr(config, section_name)
                    for key, value in section_data.items():
                        if hasattr(section, key):
                            setattr(section, key, value)
            
            logger.info(f"Configuration loaded from file: {config_file}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration from file {config_file}: {e}")
            raise
    
    def validate(self) -> list[str]:
        """Validate the complete configuration.
        
        Returns:
            List of validation warnings/errors
        """
        warnings = []
        
        # Validate security configuration
        if self.security:
            security_errors = self.security.validate()
            if security_errors:
                warnings.extend([f"Security: {err}" for err in security_errors])
        
        # Cross-configuration validation
        if self.server.max_workers > self.performance.asyncio_max_workers:
            warnings.append(
                "Server max_workers exceeds asyncio max_workers - may cause resource contention"
            )
        
        if self.monitoring.enable_metrics and not self.monitoring.metrics_port:
            warnings.append("Metrics enabled but no metrics port configured")
        
        if self.ai.enable_gemini and not self.ai.api_key:
            warnings.append("Gemini enabled but no API key configured")
        
        # Validate Lens configuration
        if self.lens.enabled and not self.lens.base_url:
            warnings.append("Lens integration enabled but no base URL configured")
        
        if self.lens.enabled and not self.lens.health_check_enabled:
            warnings.append("Lens integration enabled but health checks disabled - recommended to enable")
        
        # Validate storage paths are accessible
        try:
            self.storage.data_path.mkdir(parents=True, exist_ok=True)
            self.storage.cache_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            warnings.append(f"Storage paths not accessible: {e}")
        
        return warnings
    
    def to_dict(self) -> dict[str, Any]:
        """Export configuration to dictionary."""
        config_dict = {}
        
        for field_name in self.__fields__:
            if field_name == "security":
                # Handle security config specially
                if self.security:
                    config_dict[field_name] = self.security.__dict__
            else:
                field_value = getattr(self, field_name)
                if hasattr(field_value, "dict"):
                    config_dict[field_name] = field_value.dict()
                else:
                    config_dict[field_name] = field_value
        
        return config_dict
    
    def save_to_file(self, config_file: Path) -> None:
        """Save configuration to JSON file.
        
        Args:
            config_file: Path to save configuration
        """
        try:
            import json
            from datetime import datetime
            from pathlib import Path as PathlibPath
            
            # Convert to serializable format
            config_dict = self.to_dict()
            
            # Handle Path objects
            def convert_paths(obj):
                if isinstance(obj, dict):
                    return {k: convert_paths(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_paths(item) for item in obj]
                elif isinstance(obj, PathlibPath):
                    return str(obj)
                else:
                    return obj
            
            config_dict = convert_paths(config_dict)
            
            # Add metadata
            config_dict["_metadata"] = {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "source": "mimir-config-system"
            }
            
            # Ensure directory exists
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, "w") as f:
                json.dump(config_dict, f, indent=2, sort_keys=True)
            
            logger.info(f"Configuration saved to file: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to save configuration to file {config_file}: {e}")
            raise
    
    model_config = {
        # Allow arbitrary types for SecurityConfig integration
        "arbitrary_types_allowed": True,
        # Use environment variables
        "case_sensitive": False
    }


# Global configuration instance
_global_config: MimirConfig | None = None


def get_config() -> MimirConfig:
    """Get the global configuration instance.
    
    Returns:
        Global configuration instance
    """
    global _global_config
    if _global_config is None:
        _global_config = MimirConfig.load_from_env()
    return _global_config


def set_config(config: MimirConfig) -> None:
    """Set the global configuration instance.
    
    Args:
        config: Configuration to set as global
    """
    global _global_config
    _global_config = config
    logger.info("Global configuration updated")


def reload_config() -> MimirConfig:
    """Reload configuration from environment.
    
    Returns:
        Reloaded configuration
    """
    global _global_config
    _global_config = MimirConfig.load_from_env()
    return _global_config


def load_config_from_file(config_file: Path) -> MimirConfig:
    """Load configuration from file and set as global.
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        Loaded configuration
    """
    config = MimirConfig.load_from_file(config_file)
    set_config(config)
    return config


# Convenience functions for accessing specific configuration sections
def get_server_config() -> ServerConfig:
    """Get server configuration."""
    return get_config().server


def get_storage_config() -> StorageConfig:
    """Get storage configuration."""
    return get_config().storage


def get_pipeline_config() -> PipelineConfig:
    """Get pipeline configuration."""
    return get_config().pipeline


def get_ai_config() -> AIConfig:
    """Get AI configuration."""
    return get_config().ai


def get_monitoring_config() -> MonitoringConfig:
    """Get monitoring configuration."""
    return get_config().monitoring


def get_logging_config() -> LoggingConfig:
    """Get logging configuration."""
    return get_config().logging


def get_security_config() -> SecurityConfig:
    """Get security configuration."""
    config = get_config()
    if config.security is None:
        raise RuntimeError("Security configuration not initialized")
    return config.security


def get_performance_config() -> PerformanceConfig:
    """Get performance configuration."""
    return get_config().performance


def get_database_config() -> DatabaseConfig:
    """Get database configuration."""
    return get_config().database


def get_development_config() -> DevelopmentConfig:
    """Get development configuration."""
    return get_config().development


def get_container_config() -> ContainerConfig:
    """Get container configuration."""
    return get_config().container


def get_lens_config() -> LensConfig:
    """Get Lens integration configuration."""
    return get_config().lens


# Configuration validation and management
def validate_config() -> bool:
    """Validate the current configuration.
    
    Returns:
        True if configuration is valid
    """
    try:
        config = get_config()
        errors = config.validate()
        
        if errors:
            logger.warning(f"Configuration validation found issues: {errors}")
            return False
        
        logger.info("Configuration validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def print_config_summary() -> None:
    """Print a summary of the current configuration."""
    config = get_config()
    
    print("Mimir Configuration Summary")
    print("=" * 40)
    print(f"Server: {config.server.host}:{config.server.port}")
    print(f"Data Path: {config.storage.data_path}")
    print(f"Log Level: {config.logging.log_level}")
    print(f"AI Enabled: {config.ai.enable_gemini}")
    print(f"Metrics Enabled: {config.monitoring.enable_metrics}")
    print(f"Debug Mode: {config.development.debug}")
    
    # Security summary
    if config.security:
        print(f"Auth Required: {config.security.require_authentication}")
        print(f"Sandboxing: {config.security.enable_sandboxing}")
    
    # Validation status
    errors = config.validate()
    if errors:
        print(f"\nValidation Issues: {len(errors)}")
        for error in errors[:5]:  # Show first 5 errors
            print(f"  - {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more")
    else:
        print("\nConfiguration: Valid ✓")