# Mimir Development Guide

## Overview

This guide provides comprehensive instructions for setting up a development environment, contributing to Mimir, and understanding the system architecture from a developer perspective.

## Development Environment Setup

### Prerequisites

#### System Requirements
- **Python 3.11+** (3.12 recommended)
- **Git 2.30+** for repository operations
- **uv** package manager for Python dependencies
- **Docker & Docker Compose** for containerized development
- **4GB+ RAM** (8GB+ recommended)
- **Linux/macOS** (Windows with WSL2)

#### External Tools
Mimir integrates with several external tools that need to be available:

- **RepoMapper**: Tree-sitter based repository analysis
- **Serena**: TypeScript symbol extraction
- **LEANN**: CPU-based vector embeddings

### Quick Setup

```bash
# Clone repository
git clone <repository-url>
cd mimir

# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run setup script
python setup.py

# Activate virtual environment
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Verify installation
uv run pytest tests/unit/ -v
```

### Detailed Setup Steps

#### 1. Python Environment

```bash
# Ensure Python 3.11+
python --version

# Install uv package manager
pip install uv

# Create and activate virtual environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv sync
```

#### 2. Development Dependencies

```bash
# Install development tools
uv add --dev black ruff mypy pytest pytest-asyncio pytest-cov

# Install pre-commit hooks (optional)
uv add --dev pre-commit
pre-commit install
```

#### 3. External Tool Setup

```bash
# Install RepoMapper
pip install repomapper

# Install Serena (requires Node.js)
npm install -g @context-labs/serena

# Install LEANN
pip install leann

# Verify installations
repomapper --version
serena --version  
python -c "import leann; print('LEANN installed')"
```

#### 4. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration
cat > .env << EOF
MIMIR_LOG_LEVEL=DEBUG
MIMIR_CONCURRENCY_IO=4
MIMIR_CONCURRENCY_CPU=2
MIMIR_CACHE_DIR=~/.cache/mimir
MIMIR_ENABLE_METRICS=true
EOF
```

### IDE Setup

#### VS Code Configuration

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "88"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".venv/": true,
        "*.egg-info/": true
    }
}
```

#### PyCharm Configuration

1. Open project in PyCharm
2. Configure Python interpreter: `.venv/bin/python`
3. Set test runner to pytest
4. Configure code style: Black formatter, line length 88
5. Enable type checking with mypy

## Project Structure

### High-Level Architecture

```
src/repoindex/
├── main.py                 # Entry points for CLI and MCP server
├── main_secure.py          # Secure entry point with hardening
├── server.py               # Legacy server (deprecated)
├── health.py               # Health check endpoints
├── 
├── mcp/                    # MCP server implementation
│   ├── server.py           # Main MCP server class
│   └── secure_server.py    # Security-hardened MCP server
│
├── pipeline/               # Core pipeline stages
│   ├── run.py              # Pipeline orchestration and execution
│   ├── secure_run.py       # Security-enhanced pipeline runner
│   ├── discover.py         # Git-aware file discovery
│   ├── repomapper.py       # RepoMapper integration adapter
│   ├── serena.py           # Serena TypeScript analysis adapter
│   ├── leann.py            # LEANN vector embedding adapter
│   ├── snippets.py         # Code snippet extraction
│   ├── bundle.py           # Artifact bundling and compression
│   ├── hybrid_search.py    # Multi-modal search engine
│   └── ask_index.py        # Multi-hop reasoning engine
│
├── data/                   # Data models and schemas
│   ├── schemas.py          # Pydantic data models
│   └── schemas/            # JSON schema definitions
│       ├── manifest.json   # Index manifest schema
│       └── symbol_entry.json # Symbol entry schema
│
├── security/               # Security framework
│   ├── validation.py       # Input validation and sanitization
│   ├── sandbox.py          # Process isolation and resource limits
│   ├── auth.py             # Authentication and authorization
│   ├── secrets.py          # Credential scanning and management
│   ├── audit.py            # Security event logging
│   ├── crypto.py           # Cryptographic operations
│   ├── config.py           # Security configuration management
│   ├── server_middleware.py # MCP server security middleware
│   ├── pipeline_integration.py # Pipeline security integration
│   └── testing.py          # Security testing framework
│
├── monitoring/             # Observability and metrics
│   ├── metrics.py          # Prometheus metrics collection
│   ├── tracing.py          # Distributed tracing with Jaeger
│   ├── alerts.py           # Alerting and notification system
│   └── dashboard.py        # Grafana dashboard configuration
│
├── ui/                     # Web interface (optional)
│   ├── app.py              # FastAPI application
│   └── static/             # Frontend assets
│
└── util/                   # Utilities and helpers
    ├── fs.py               # Filesystem operations
    ├── gitio.py            # Git repository operations
    ├── log.py              # Logging configuration
    ├── errors.py           # Custom exceptions
    └── logging_config.py   # Structured logging setup
```

### Key Design Patterns

#### 1. Adapter Pattern
External tools are wrapped in adapter classes:

```python
class RepoMapperAdapter:
    """Adapter for RepoMapper tool integration."""
    
    async def analyze_repository(self, repo_path: Path) -> RepoMapResult:
        """Execute RepoMapper analysis with error handling."""
        pass

class SerenaAdapter:
    """Adapter for Serena TypeScript analysis."""
    
    async def analyze_project(self, project_path: Path) -> SerenaGraph:
        """Extract TypeScript symbols and relationships."""
        pass
```

#### 2. Pipeline Pattern
Processing stages are composable and independent:

```python
class PipelineStage(ABC):
    """Abstract base for pipeline stages."""
    
    @abstractmethod
    async def execute(self, context: PipelineContext) -> StageResult:
        """Execute stage with given context."""
        pass

class IndexingPipeline:
    """Orchestrates execution of all pipeline stages."""
    
    def __init__(self, storage_dir: Path):
        self.stages = [
            AcquireStage(),
            RepoMapperStage(), 
            SerenaStage(),
            LeannStage(),
            SnippetsStage(),
            BundleStage()
        ]
```

#### 3. Repository Pattern
Data access is abstracted through repository interfaces:

```python
class IndexRepository:
    """Repository for index metadata and artifacts."""
    
    async def save_manifest(self, manifest: IndexManifest) -> None:
        """Save index manifest to storage."""
        pass
    
    async def load_manifest(self, index_id: str) -> IndexManifest:
        """Load manifest from storage."""
        pass
```

## Development Workflow

### Code Standards

#### Python Style Guide
- **Formatter**: Black with 88-character line length
- **Linter**: Ruff with strict configuration
- **Type Hints**: Required for all public functions
- **Docstrings**: Google-style docstrings for all public APIs
- **Import Sorting**: isort with Black compatibility

#### Example Code Style

```python
"""Module for repository indexing operations."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from repoindex.data.schemas import IndexManifest, RepoInfo
from repoindex.util.log import get_logger

logger = get_logger(__name__)


class IndexingService:
    """Service for managing repository indexing operations.
    
    This service orchestrates the complete indexing pipeline,
    from file discovery through vector embedding generation.
    
    Args:
        storage_dir: Directory for storing index artifacts
        concurrency_limit: Maximum concurrent operations
        
    Example:
        >>> service = IndexingService(Path("./storage"))
        >>> manifest = await service.index_repository("/path/to/repo")
        >>> print(f"Indexed {manifest.counts.files_indexed} files")
    """
    
    def __init__(
        self, 
        storage_dir: Path, 
        concurrency_limit: int = 8
    ) -> None:
        self.storage_dir = storage_dir
        self.concurrency_limit = concurrency_limit
        self._semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def index_repository(
        self, 
        repo_path: str, 
        *, 
        language: str = "ts",
        excludes: Optional[List[str]] = None
    ) -> IndexManifest:
        """Index a repository with the complete pipeline.
        
        Args:
            repo_path: Path to repository root
            language: Primary programming language
            excludes: File patterns to exclude from indexing
            
        Returns:
            Index manifest with metadata and artifact paths
            
        Raises:
            ValueError: If repository path is invalid
            IndexingError: If pipeline execution fails
        """
        logger.info(f"Starting indexing for {repo_path}")
        
        # Implementation here
        pass
```

### Git Workflow

#### Branch Naming
- `feature/short-description` - New features
- `fix/bug-description` - Bug fixes
- `refactor/component-name` - Code refactoring
- `docs/topic` - Documentation updates
- `perf/optimization-area` - Performance improvements

#### Commit Messages
Follow conventional commit format:

```
type(scope): short description

Longer description explaining the why and what of the change.
Include any breaking changes or migration notes.

Closes #123
```

Types: `feat`, `fix`, `docs`, `refactor`, `test`, `perf`, `ci`, `build`

#### Pull Request Process

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/vector-search-optimization
   ```

2. **Make Changes with Tests**
   ```bash
   # Make your changes
   git add .
   git commit -m "feat(search): optimize vector similarity computation"
   ```

3. **Run Quality Checks**
   ```bash
   # Format code
   uv run black src/ tests/
   
   # Lint code  
   uv run ruff check src/ tests/
   
   # Type check
   uv run mypy src/
   
   # Run tests
   uv run pytest tests/ -v --cov=src/repoindex
   ```

4. **Create Pull Request**
   - Fill out PR template completely
   - Link to relevant issues
   - Include test coverage information
   - Add screenshots for UI changes

### Testing Strategy

#### Test Organization

```
tests/
├── unit/                   # Fast, isolated unit tests
│   ├── test_mcp_server.py  # MCP server functionality
│   ├── test_schemas.py     # Data model validation
│   ├── test_adapters.py    # External tool adapters
│   └── test_*.py           # Other unit tests
│
├── integration/            # Integration tests with external dependencies
│   ├── test_pipeline_integration.py # Full pipeline execution
│   ├── test_mcp_integration.py     # MCP protocol compliance
│   └── test_*.py           # Other integration tests
│
├── e2e/                    # End-to-end tests (currently minimal)
│   └── test_full_workflow.py
│
├── benchmarks/             # Performance benchmarks
│   └── test_performance.py
│
└── conftest.py             # Shared pytest fixtures
```

#### Test Coverage Requirements

- **Unit Tests**: 85%+ line coverage
- **Integration Tests**: Cover all major user workflows
- **Performance Tests**: Regression detection for key operations
- **Security Tests**: Validate all security controls

#### Writing Tests

```python
"""Tests for vector search functionality."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from repoindex.pipeline.hybrid_search import HybridSearchEngine
from repoindex.data.schemas import SearchResult


class TestHybridSearchEngine:
    """Test cases for hybrid search engine."""
    
    @pytest.fixture
    async def search_engine(self):
        """Create search engine with mocked dependencies."""
        engine = HybridSearchEngine(storage_dir=Path("/tmp/test"))
        engine._vector_index = AsyncMock()
        engine._symbol_index = MagicMock()
        return engine
    
    async def test_vector_search_basic(self, search_engine):
        """Test basic vector search functionality."""
        # Setup
        search_engine._vector_index.search.return_value = [
            SearchResult(path="test.py", span=[0, 100], score=0.85)
        ]
        
        # Execute
        results = await search_engine.vector_search("test query", k=10)
        
        # Verify
        assert len(results) == 1
        assert results[0].score == 0.85
        search_engine._vector_index.search.assert_called_once_with(
            "test query", k=10
        )
    
    async def test_hybrid_search_combines_modalities(self, search_engine):
        """Test that hybrid search combines vector and symbol results."""
        # Setup mock responses
        search_engine._vector_search = AsyncMock(return_value=[
            SearchResult(path="test.py", span=[0, 50], score=0.8)
        ])
        search_engine._symbol_search = AsyncMock(return_value=[
            SearchResult(path="test.py", span=[60, 120], score=0.9)
        ])
        
        # Execute
        results = await search_engine.search(
            query="test function",
            features={"vector": True, "symbol": True, "graph": False}
        )
        
        # Verify combined results
        assert len(results) == 2
        assert any(r.score == 0.8 for r in results)
        assert any(r.score == 0.9 for r in results)
```

#### Running Tests

```bash
# All tests
uv run pytest

# Unit tests only (fast)
uv run pytest tests/unit/ -v

# Integration tests
uv run pytest tests/integration/ -v

# With coverage
uv run pytest --cov=src/repoindex --cov-report=html

# Performance tests
uv run pytest tests/benchmarks/ -v --benchmark-only

# Specific test
uv run pytest tests/unit/test_mcp_server.py::TestMCPServer::test_ensure_repo_index -v

# Skip slow tests
uv run pytest -m "not slow"

# Run in parallel
uv run pytest -n auto
```

### Debugging

#### Development Server

```bash
# Start MCP server with debug logging
MIMIR_LOG_LEVEL=DEBUG uv run python -m repoindex.main mcp

# Start with profiling
uv run python -m cProfile -o profile.prof -m repoindex.main mcp

# Start with memory profiling
uv run python -m memory_profiler -m repoindex.main mcp
```

#### Common Debug Scenarios

```python
# Debug pipeline stage execution
import logging
logging.getLogger("repoindex.pipeline").setLevel(logging.DEBUG)

# Debug MCP communication
logging.getLogger("repoindex.mcp").setLevel(logging.DEBUG)

# Debug external tool integration
logging.getLogger("repoindex.adapters").setLevel(logging.DEBUG)

# Enable asyncio debug mode
import asyncio
asyncio.get_event_loop().set_debug(True)
```

#### Performance Profiling

```python
# Profile specific function
import cProfile
import pstats

def profile_search():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Execute search operation
    result = search_engine.search("query")
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative').print_stats(10)

# Memory profiling
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Function implementation
    pass
```

## Architecture Deep Dive

### Pipeline Architecture

#### Stage Execution Model

```python
@dataclass
class PipelineContext:
    """Context passed between pipeline stages."""
    
    index_id: str
    repo_info: RepoInfo
    config: IndexConfig
    storage_dir: Path
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class PipelineStage(ABC):
    """Abstract base for all pipeline stages."""
    
    @abstractmethod
    async def execute(self, context: PipelineContext) -> StageResult:
        """Execute stage processing."""
        pass
    
    @abstractmethod
    def get_dependencies(self) -> List[str]:
        """Return list of required artifacts from previous stages."""
        pass
```

#### Concurrency and Error Handling

```python
class IndexingPipeline:
    """Pipeline orchestrator with concurrency and error handling."""
    
    async def execute(self, context: PipelineContext) -> IndexManifest:
        """Execute all pipeline stages with error recovery."""
        
        for stage in self.stages:
            try:
                # Check dependencies
                self._validate_dependencies(stage, context)
                
                # Execute stage with timeout
                result = await asyncio.wait_for(
                    stage.execute(context),
                    timeout=self.stage_timeout
                )
                
                # Update context with stage artifacts
                context.artifacts.update(result.artifacts)
                
                # Report progress
                await self._emit_progress(stage.name, result.progress)
                
            except Exception as e:
                # Handle stage failure
                await self._handle_stage_failure(stage, e, context)
                
                if not stage.allow_failure:
                    raise IndexingError(f"Critical stage {stage.name} failed: {e}")
        
        return self._build_manifest(context)
```

### MCP Server Architecture

#### Tool Registration System

```python
class MCPServer:
    """MCP server with dynamic tool registration."""
    
    def __init__(self):
        self.server = Server("mimir-repoindex")
        self.tool_handlers = {}
        self.resource_handlers = {}
        
        # Register built-in tools
        self._register_core_tools()
        self._register_resources()
    
    def register_tool(self, name: str, handler: Callable, schema: dict):
        """Register a new tool with validation schema."""
        self.tool_handlers[name] = handler
        
        @self.server.call_tool()
        async def call_tool(tool_name: str, arguments: dict):
            if tool_name == name:
                # Validate arguments against schema
                validate_json(arguments, schema)
                return await handler(arguments)
            # ... handle other tools
```

#### Resource Management

```python
class ResourceManager:
    """Manages dynamic resource registration and access."""
    
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.active_indexes = {}
    
    async def get_resource_list(self) -> List[Resource]:
        """Generate dynamic resource list based on active indexes."""
        resources = []
        
        # Add static resources
        resources.extend(self._get_static_resources())
        
        # Add dynamic resources for each index
        for index_id in self.active_indexes:
            resources.extend(self._get_index_resources(index_id))
        
        return resources
    
    async def read_resource(self, uri: str) -> ResourceContent:
        """Read resource content with caching and error handling."""
        # Parse URI to determine resource type
        resource_type, identifier = self._parse_uri(uri)
        
        # Check cache
        if cached := self._get_cached_resource(uri):
            return cached
        
        # Load from storage
        content = await self._load_resource(resource_type, identifier)
        
        # Cache for future access
        self._cache_resource(uri, content)
        
        return content
```

### Security Architecture

#### Defense in Depth

```python
class SecurityMiddleware:
    """Multi-layer security middleware for MCP server."""
    
    def __init__(self, config: SecurityConfig):
        self.validators = [
            InputValidator(config.validation_rules),
            AuthValidator(config.auth_config),
            RateLimiter(config.rate_limits),
            AuditLogger(config.audit_config)
        ]
    
    async def process_request(self, request: MCPRequest) -> MCPRequest:
        """Process request through all security layers."""
        
        for validator in self.validators:
            request = await validator.validate(request)
        
        return request
    
    async def process_response(self, response: MCPResponse) -> MCPResponse:
        """Process response through security filters."""
        
        # Sanitize response data
        response = await self._sanitize_response(response)
        
        # Log security events
        await self._log_security_event(response)
        
        return response
```

#### Sandboxing Implementation

```python
class ProcessSandbox:
    """Secure execution environment for external tools."""
    
    def __init__(self, config: SandboxConfig):
        self.max_memory = config.max_memory_mb * 1024 * 1024
        self.max_cpu_time = config.max_cpu_time_seconds
        self.allowed_paths = config.allowed_paths
    
    async def execute(
        self, 
        command: List[str], 
        work_dir: Path,
        timeout: Optional[int] = None
    ) -> ProcessResult:
        """Execute command in sandboxed environment."""
        
        # Set resource limits
        limits = resource.RLIMIT_AS, self.max_memory, self.max_memory
        
        # Create restricted environment
        env = self._create_restricted_env()
        
        # Execute with monitoring
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=work_dir,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, limits)
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout or self.max_cpu_time
            )
            
            return ProcessResult(
                returncode=process.returncode,
                stdout=stdout.decode(),
                stderr=stderr.decode()
            )
            
        except asyncio.TimeoutError:
            process.kill()
            raise SandboxTimeoutError("Process exceeded time limit")
```

## Performance Optimization

### Profiling and Benchmarking

#### Continuous Performance Monitoring

```python
class PerformanceMonitor:
    """Monitor and track performance metrics."""
    
    def __init__(self):
        self.metrics_collector = get_metrics_collector()
        self.baselines = self._load_baselines()
    
    @contextmanager
    def measure_operation(self, operation_name: str):
        """Context manager for measuring operation performance."""
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            memory_delta = self._get_memory_usage() - start_memory
            
            # Record metrics
            self.metrics_collector.record_operation_time(
                operation_name, duration
            )
            self.metrics_collector.record_memory_usage(
                operation_name, memory_delta
            )
            
            # Check for regressions
            self._check_performance_regression(operation_name, duration)

# Usage example
async def search_with_monitoring(query: str):
    monitor = PerformanceMonitor()
    
    with monitor.measure_operation("vector_search"):
        results = await vector_search(query)
    
    return results
```

#### Memory Optimization

```python
class MemoryOptimizedVectorIndex:
    """Memory-efficient vector index with lazy loading."""
    
    def __init__(self, index_path: Path):
        self.index_path = index_path
        self._index = None
        self._cache = LRUCache(maxsize=1000)
    
    async def search(self, query_vector: np.ndarray, k: int) -> List[SearchResult]:
        """Search with memory-efficient loading."""
        
        # Lazy load index
        if self._index is None:
            self._index = await self._load_index_lazy()
        
        # Use cache for recent queries
        cache_key = hash(query_vector.tobytes())
        if cache_key in self._cache:
            return self._cache[cache_key][:k]
        
        # Perform search
        results = await self._search_index(query_vector, k)
        self._cache[cache_key] = results
        
        return results
    
    async def _load_index_lazy(self):
        """Load index in chunks to avoid memory spikes."""
        # Implementation for streaming index loading
        pass
```

### Caching Strategies

```python
class MultiLevelCache:
    """Multi-level caching for search results and embeddings."""
    
    def __init__(self):
        # L1: In-memory LRU cache (fast, small)
        self.l1_cache = LRUCache(maxsize=100)
        
        # L2: Redis cache (medium speed, larger)
        self.l2_cache = redis.Redis(decode_responses=True)
        
        # L3: Disk cache (slow, large)
        self.l3_cache = DiskCache("~/.cache/mimir")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
        
        # Check L1 cache
        if value := self.l1_cache.get(key):
            return value
        
        # Check L2 cache
        if value := await self.l2_cache.get(key):
            self.l1_cache[key] = pickle.loads(value)
            return self.l1_cache[key]
        
        # Check L3 cache
        if value := self.l3_cache.get(key):
            self.l1_cache[key] = value
            await self.l2_cache.set(key, pickle.dumps(value), ex=3600)
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in all cache levels."""
        
        # Set in all levels
        self.l1_cache[key] = value
        await self.l2_cache.set(key, pickle.dumps(value), ex=ttl)
        self.l3_cache.set(key, value)
```

## Monitoring and Observability

### Metrics Collection

```python
# Custom metrics for Mimir
from prometheus_client import Counter, Histogram, Gauge

# Pipeline metrics
PIPELINE_EXECUTIONS = Counter(
    'mimir_pipeline_executions_total',
    'Total pipeline executions',
    ['stage', 'status']
)

PIPELINE_DURATION = Histogram(
    'mimir_pipeline_duration_seconds',
    'Pipeline execution duration',
    ['stage']
)

# Search metrics  
SEARCH_REQUESTS = Counter(
    'mimir_search_requests_total',
    'Total search requests',
    ['type', 'status']
)

SEARCH_DURATION = Histogram(
    'mimir_search_duration_seconds',
    'Search request duration',
    ['type']
)

ACTIVE_INDEXES = Gauge(
    'mimir_active_indexes',
    'Number of active indexes'
)

# Usage in code
class MetricsCollector:
    """Collect and expose Prometheus metrics."""
    
    def record_pipeline_start(self, stage: str):
        """Record pipeline stage start."""
        PIPELINE_EXECUTIONS.labels(stage=stage, status='started').inc()
    
    def record_pipeline_completion(self, stage: str, duration: float):
        """Record pipeline stage completion."""
        PIPELINE_EXECUTIONS.labels(stage=stage, status='completed').inc()
        PIPELINE_DURATION.labels(stage=stage).observe(duration)
    
    def record_search_request(self, search_type: str, duration: float):
        """Record search request metrics."""
        SEARCH_REQUESTS.labels(type=search_type, status='completed').inc()
        SEARCH_DURATION.labels(type=search_type).observe(duration)
```

### Distributed Tracing

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

class TracingManager:
    """Manage distributed tracing for Mimir operations."""
    
    def __init__(self, service_name: str = "mimir"):
        trace.set_tracer_provider(TracerProvider())
        tracer = trace.get_tracer(__name__)
        
        jaeger_exporter = JaegerExporter(
            agent_host_name="localhost",
            agent_port=6831,
        )
        
        span_processor = BatchSpanProcessor(jaeger_exporter)
        trace.get_tracer_provider().add_span_processor(span_processor)
        
        self.tracer = tracer
    
    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Trace an operation with automatic span management."""
        
        with self.tracer.start_as_current_span(operation_name) as span:
            # Add custom attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)
            
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

# Usage example
tracing = TracingManager()

async def search_with_tracing(query: str, index_id: str):
    with tracing.trace_operation(
        "hybrid_search",
        query=query,
        index_id=index_id
    ) as span:
        
        with tracing.trace_operation("vector_search") as vector_span:
            vector_results = await vector_search(query)
            vector_span.set_attribute("results_count", len(vector_results))
        
        with tracing.trace_operation("symbol_search") as symbol_span:
            symbol_results = await symbol_search(query)
            symbol_span.set_attribute("results_count", len(symbol_results))
        
        combined_results = combine_results(vector_results, symbol_results)
        span.set_attribute("total_results", len(combined_results))
        
        return combined_results
```

## Contributing Guidelines

### Code Review Checklist

#### For Authors

- [ ] Code follows style guidelines (Black, Ruff, type hints)
- [ ] All new functions have comprehensive docstrings
- [ ] Tests added for new functionality (unit + integration)
- [ ] Security implications considered and documented
- [ ] Performance impact measured and acceptable
- [ ] Breaking changes documented in PR description
- [ ] Related documentation updated

#### For Reviewers

- [ ] Code is readable and maintainable
- [ ] Test coverage is adequate and tests are meaningful
- [ ] Error handling is comprehensive
- [ ] Security considerations are addressed
- [ ] Performance implications are acceptable
- [ ] API changes are backward compatible (or properly versioned)
- [ ] Documentation is accurate and complete

### Security Guidelines

When contributing to Mimir, always consider:

1. **Input Validation**: All user inputs must be validated
2. **Path Traversal**: File paths must be sanitized
3. **Resource Limits**: Operations must have bounded resource usage
4. **Error Information**: Avoid leaking sensitive information in errors
5. **Dependencies**: Keep dependencies up to date and audit for vulnerabilities

### Performance Guidelines

1. **Async/Await**: Use async patterns consistently
2. **Memory Management**: Avoid memory leaks with proper cleanup
3. **Caching**: Cache expensive operations appropriately
4. **Batching**: Batch I/O operations when possible
5. **Profiling**: Profile critical paths regularly

## Common Development Tasks

### Adding a New Pipeline Stage

```python
# 1. Create stage class
class NewStage(PipelineStage):
    """New pipeline stage for custom processing."""
    
    async def execute(self, context: PipelineContext) -> StageResult:
        """Execute the new stage."""
        
        # Get dependencies from previous stages
        repomap_data = context.artifacts["repomap"]
        
        # Perform stage processing
        result_data = await self._process_data(repomap_data)
        
        # Save artifacts
        artifact_path = context.storage_dir / "new_stage_output.json"
        with open(artifact_path, "w") as f:
            json.dump(result_data, f)
        
        return StageResult(
            artifacts={"new_stage": result_data},
            metadata={"output_path": str(artifact_path)},
            progress=100
        )
    
    def get_dependencies(self) -> List[str]:
        """Return required artifacts from previous stages."""
        return ["repomap"]

# 2. Register stage in pipeline
class IndexingPipeline:
    def __init__(self):
        self.stages = [
            AcquireStage(),
            RepoMapperStage(),
            SerenaStage(),
            NewStage(),  # Add new stage
            LeannStage(),
            SnippetsStage(),
            BundleStage()
        ]

# 3. Add tests
class TestNewStage:
    async def test_execute_with_valid_input(self):
        stage = NewStage()
        context = PipelineContext(
            index_id="test",
            repo_info=RepoInfo(),
            config=IndexConfig(),
            storage_dir=Path("/tmp"),
            artifacts={"repomap": {"files": []}}
        )
        
        result = await stage.execute(context)
        
        assert "new_stage" in result.artifacts
        assert result.progress == 100
```

### Adding a New MCP Tool

```python
# 1. Define request/response schemas
class NewToolRequest(BaseModel):
    """Request schema for new tool."""
    parameter1: str
    parameter2: Optional[int] = None

class NewToolResponse(BaseModel):
    """Response schema for new tool."""
    result: Dict[str, Any]
    status: str

# 2. Implement tool handler
class MCPServer:
    def _register_tools(self):
        # Add to tool list
        tools = [
            # ... existing tools
            Tool(
                name="new_tool",
                description="Description of new tool functionality",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "parameter1": {"type": "string"},
                        "parameter2": {"type": "integer", "default": 42}
                    },
                    "required": ["parameter1"]
                }
            )
        ]
        
        # Add to call handler
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "new_tool":
                return await self._new_tool(arguments)
            # ... handle other tools
    
    async def _new_tool(self, arguments: dict) -> CallToolResult:
        """Handle new tool requests."""
        try:
            request = NewToolRequest(**arguments)
            
            # Implement tool logic
            result_data = await self._process_new_tool(request)
            
            response = NewToolResponse(
                result=result_data,
                status="success"
            )
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(response.model_dump(), indent=2)
                    )
                ]
            )
            
        except Exception as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", 
                        text=f"Error: {str(e)}"
                    )
                ],
                isError=True
            )

# 3. Add tests
class TestNewTool:
    async def test_new_tool_success(self, mcp_server):
        result = await mcp_server._new_tool({
            "parameter1": "test_value",
            "parameter2": 123
        })
        
        assert not result.isError
        response_data = json.loads(result.content[0].text)
        assert response_data["status"] == "success"
```

### Debugging Complex Issues

```python
# Enable comprehensive logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Add performance monitoring
import time
import functools

def monitor_performance(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            duration = time.perf_counter() - start
            logger.info(f"{func.__name__} took {duration:.2f}s")
    return wrapper

# Add memory monitoring  
import tracemalloc

def monitor_memory():
    tracemalloc.start()
    
    # Your code here
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
    print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
    tracemalloc.stop()

# Debug async issues
import asyncio

async def debug_async_tasks():
    # List all running tasks
    tasks = asyncio.all_tasks()
    print(f"Running tasks: {len(tasks)}")
    
    for task in tasks:
        print(f"Task: {task}")
        if not task.done():
            print(f"  Stack: {task.get_stack()}")
```

This comprehensive development guide provides everything needed to contribute effectively to Mimir. For specific questions or advanced scenarios, refer to the code documentation or create an issue for discussion.