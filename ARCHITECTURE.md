# Mimir Deep Code Research System - Detailed Architecture

## System Overview

Mimir is a full-Python, stdio MCP server that provides intelligent code indexing and search capabilities for local repositories. The system follows a clean pipeline architecture with six core stages and exposes both programmatic APIs and an optional management UI.

### Core Design Principles

1. **Zero-prompt indexing**: No user configuration required for basic operation
2. **Defensible citations**: Every result includes precise source location and content hash
3. **CPU-only operation**: No GPU dependencies for maximum compatibility
4. **Agent-friendly**: Clean MCP interface optimized for AI agent consumption
5. **Incremental processing**: Smart caching and delta detection
6. **Robust monorepo support**: Handles large codebases efficiently

## Technical Stack & Dependencies

### Core Runtime
- **Python 3.11+** with uv for dependency management
- **AsyncIO** for concurrent pipeline execution
- **stdio MCP protocol** for agent communication

### Key Libraries
- **Tree-sitter** (via RepoMapper) for AST parsing and analysis
- **TypeScript Server** (via Serena) for symbol resolution
- **LEANN** for CPU-based vector embeddings
- **FastAPI** for optional management UI
- **Git subprocess** integration for repository operations

## Detailed Component Architecture

### 1. MCP Server Layer (`src/repoindex/mcp/`)

#### `server.py` - Main MCP Entry Point
- Implements stdio MCP protocol
- Manages tool dispatch and resource serving
- Handles concurrent indexing operations
- Provides stage transition notifications

#### Key Design Decisions:
- **Single server instance** managing multiple concurrent indexes
- **Resource-based status updates** instead of polling
- **ULID-based index identifiers** for unique operation tracking
- **Graceful degradation** when stages fail

#### Tool Handlers (within server.py)
The MCP tools are implemented as async handlers within the server class:
```python
async def ensure_repo_index(
    path: str,
    rev: Optional[str] = None,
    language: str = "ts",
    index_opts: Optional[dict] = None
) -> dict:
    """Primary indexing tool with full pipeline execution"""

async def get_repo_bundle(index_id: str) -> dict:
    """Retrieve complete index artifacts"""

async def search_repo(
    index_id: str,
    query: str,
    k: int = 20,
    features: dict = {"vector": True, "symbol": True, "graph": True},
    context_lines: int = 5
) -> dict:
    """Hybrid search across all modalities"""

async def ask_index(
    index_id: str,
    question: str,
    context_lines: int = 5
) -> dict:
    """Multi-hop symbol graph reasoning"""
```

#### `resources.py` - Status & Artifacts
- Real-time status JSON with stage progress
- Manifest JSON with complete index metadata
- Human-readable logs in Markdown format
- Compressed bundle artifacts (tar.zst)

### 2. Architectural Separation: Indexing vs Querying

**Major Architectural Improvement (v1.1+)**: Mimir now implements a clean separation between indexing and querying concerns, following the Single Responsibility Principle for improved maintainability and extensibility.

#### Design Philosophy

**Before**: Single `IndexingPipeline` class handled both indexing operations AND querying operations, violating SRP.

**After**: Clean separation into dedicated components:
- **IndexingPipeline** (`pipeline/run.py`) - Pure indexing operations
- **QueryEngine** (`pipeline/query_engine.py`) - Pure querying operations
- **MCP Server** - Orchestrates both components with clean integration

#### `query_engine.py` - Query Operations Engine

```python
class QueryEngine:
    """Dedicated engine for all querying operations against indexed repositories."""
    
    def register_indexed_repository(self, indexed_repo: IndexedRepository):
        """Register completed index for querying operations"""
        
    async def search(
        self, index_id: str, query: str, k: int = 20, 
        features: FeatureConfig = FeatureConfig(), context_lines: int = 5
    ) -> SearchResponse:
        """Execute hybrid search: vector + symbol + graph"""
        
    async def ask(
        self, index_id: str, question: str, context_lines: int = 5
    ) -> AskResponse:
        """Execute multi-hop symbol graph reasoning"""
```

#### `IndexedRepository` - Data Container

```python
class IndexedRepository:
    """Immutable container holding all artifacts from completed indexing pipeline."""
    
    index_id: str
    repo_root: str  
    rev: str
    serena_graph: SerenaGraph | None      # Symbol graph data
    vector_index: Any | None              # LEANN embeddings
    repomap_data: Any | None              # Repository structure
    snippets: Any | None                  # Code snippets
    manifest: IndexManifest | None        # Complete metadata
    
    def is_complete(self) -> bool:
        """Check if repository has required data for querying"""
        
    def validate_for_operation(self, operation: str) -> None:
        """Validate data availability for specific operations"""
```

#### Integration Pattern

```python
# IndexingPipeline completion automatically registers with QueryEngine
class IndexingPipeline:
    def __init__(self, storage_dir: Path, query_engine: QueryEngine | None = None):
        self.query_engine = query_engine
        
    async def _execute_pipeline(self, context: PipelineContext):
        # ... indexing stages ...
        
        # On successful completion:
        if self.query_engine:
            indexed_repo = IndexedRepository(
                index_id=context.index_id,
                repo_root=context.repo_info.root,
                # ... all pipeline artifacts ...
            )
            self.query_engine.register_indexed_repository(indexed_repo)
            
# MCP Server orchestrates both components
class MCPServer:
    def __init__(self, storage_dir: Path, query_engine: QueryEngine | None = None):
        self.query_engine = query_engine or QueryEngine()
        
    async def _ensure_repo_index(self, arguments):
        pipeline = IndexingPipeline(self.indexes_dir, self.query_engine)
        return await pipeline.start_indexing(...)
        
    async def _search_repo(self, arguments):
        return await self.query_engine.search(...)
        
    async def _ask_index(self, arguments):
        return await self.query_engine.ask(...)
```

#### Architectural Benefits

1. **Single Responsibility Principle**: Each class has exactly one reason to change
   - IndexingPipeline: Changes when indexing logic changes
   - QueryEngine: Changes when querying logic changes

2. **Enhanced Testability**: Isolated concerns enable focused unit testing
   - Test indexing without mocking query operations
   - Test querying without running full indexing pipelines

3. **Improved Extensibility**: Easy to add capabilities without cross-cutting changes
   - New indexing stages don't affect query logic
   - New query types don't affect indexing pipeline

4. **Performance Optimization**: Dedicated concurrency controls
   - IndexingPipeline: IO and CPU semaphores for pipeline stages
   - QueryEngine: Query-specific concurrency for search/ask operations

5. **Clean Dependency Injection**: Optional QueryEngine integration
   - IndexingPipeline can operate standalone
   - QueryEngine can be shared across multiple pipelines
   - Easy to mock for testing

6. **Maintainable Codebase**: Clear boundaries reduce cognitive complexity
   - 600+ lines of mixed concerns → 2 focused classes
   - Easier onboarding for new developers
   - Reduced risk of unintended cross-cutting changes

### 3. Pipeline Orchestration (`src/repoindex/pipeline/`) - Indexing Components

#### `run.py` - Indexing Pipeline Controller (Refactored)

**Post-Separation**: IndexingPipeline now focuses exclusively on indexing operations:

```python
class IndexingPipeline:
    """Orchestrates all six indexing pipeline stages with proper error handling"""
    
    def __init__(self, storage_dir: Path, query_engine: QueryEngine | None = None):
        """Optional QueryEngine integration for completed indexes"""
    
    async def start_indexing(self, repo_path: str, ...) -> str:
        """Main indexing workflow - returns index_id"""
        
    async def _execute_pipeline(self, context: PipelineContext):
        """Execute all six pipeline stages with comprehensive error handling"""
        
    # Six indexing stages:
    async def _stage_acquire(self, context: PipelineContext):
    async def _stage_repomapper(self, context: PipelineContext):
    async def _stage_serena(self, context: PipelineContext):
    async def _stage_leann(self, context: PipelineContext):
    async def _stage_snippets(self, context: PipelineContext):
    async def _stage_bundle(self, context: PipelineContext):
    
    # REMOVED: search() and ask() methods moved to QueryEngine
```

#### Key Features:
- **Concurrent stage execution** where dependencies allow
- **Atomic stage completion** with rollback capability
- **Progress tracking** with percentage completion per stage
- **Resource management** with configurable concurrency limits

#### `discover.py` - Git Integration
```python
class GitDiscovery:
    """Git-aware file discovery and change detection"""
    
    def discover_files(self, repo_root: str) -> list[str]:
        """Use git ls-files for tracked file enumeration"""
        
    def compute_overlay(self, repo_root: str) -> dict:
        """Detect uncommitted changes with git hash-object"""
        
    def get_cache_key(self, repo_root: str, config: dict) -> str:
        """Generate cache key from HEAD tree hash + config"""
```

### 4. External Tool Adapters

#### `repomapper.py` - Repository Structure Analysis
- **Adapter pattern** for RepoMapper integration
- **PageRank-based** file importance scoring
- **Tree-sitter AST** analysis for dependency graphs
- **Configurable file type** support

#### `serena.py` - TypeScript Symbol Resolution
```python
class SerenaAdapter:
    """Manages TypeScript Server interaction for symbol analysis"""
    
    async def analyze_project(self, project_root: str) -> SerenaGraph:
        """Full project symbol analysis with first-order type resolution"""
        
    async def resolve_imports(self, files: list[str]) -> dict:
        """Resolve imports to actual source locations"""
        
    def extract_symbols(self, analysis: dict) -> list[Symbol]:
        """Extract definitions, references, and call relationships"""
```

**Key Capabilities:**
- **First-order dependency types** (.d.ts) for entire project
- **Source code analysis** for direct imports only
- **Byte-precise span tracking** for all symbols
- **Import resolution** via package.json exports and sourcemaps

#### `leann.py` - Vector Embedding Generation
```python
class LEANNAdapter:
    """CPU-based vector embedding generation"""
    
    async def build_index(
        self, 
        files: list[str], 
        repomap_order: list[str]
    ) -> VectorIndex:
        """Build vector index respecting RepoMapper file importance"""
        
    def chunk_content(self, content: str) -> list[Chunk]:
        """Function-level chunking with ~400 token fallback"""
```

### 4. Search & Retrieval (`src/repoindex/pipeline/`)

#### `hybrid_search.py` - Multi-Modal Search
```python
class HybridSearchEngine:
    """Combines vector, symbol, and graph-based search"""
    
    def search(
        self,
        index_id: str,
        query: str,
        features: dict,
        k: int
    ) -> list[SearchResult]:
        """Execute hybrid search with configurable feature weights"""
        
    def _vector_search(self, query: str, k: int) -> list[Result]:
        """LEANN vector similarity search"""
        
    def _symbol_search(self, query: str) -> list[Result]:
        """Serena symbol name/signature matching"""
        
    def _graph_expansion(self, candidates: list[Result]) -> list[Result]:
        """RepoMapper-based neighbor expansion"""
```

**Scoring Algorithm:**
```python
def compute_hybrid_score(result: SearchResult) -> float:
    return (
        1.0 * result.vector_score +
        0.9 * result.symbol_score +
        0.3 * result.graph_score
    )
```

#### `ask_index.py` - Multi-Hop Reasoning
```python
class SymbolGraphNavigator:
    """Navigate symbol relationships for complex queries"""
    
    def ask(self, index_id: str, question: str) -> Answer:
        """Multi-hop symbol graph traversal"""
        
    def _parse_intent(self, question: str) -> list[Intent]:
        """Extract symbol, file, import, callgraph intents"""
        
    def _walk_callgraph(
        self, 
        seeds: list[Symbol], 
        max_depth: int = 2
    ) -> list[Symbol]:
        """Breadth-first traversal of call relationships"""
```

### 5. Data Persistence & Caching

#### Storage Architecture
```
var/indexes/{index_id}/
├── status.json              # Current pipeline state
├── manifest.json            # Complete index metadata
├── log.md                   # Human-readable progress log
├── repomap.json             # File importance + dependency graph
├── serena_graph.jsonl       # Symbol definitions, references, calls
├── leann.index              # Vector index (binary format)
├── vectors.bin              # Raw embedding vectors
├── snippets.jsonl           # Source code snippets with context
├── types/                   # First-order dependency .d.ts files
├── vendor_src/              # Direct import source code
└── bundle.tar.zst          # Compressed archive of all artifacts
```

#### Caching Strategy
- **Content-hash based** cache keys for immutable artifacts
- **Incremental overlays** for uncommitted changes
- **Cross-run persistence** with automatic cleanup
- **Size-based sharding** for large repositories (>2GB)

### 6. Management UI (`src/repoindex/ui/`)

#### Backend API (`app.py`)
```python
class UIServer:
    """FastAPI server for local management interface"""
    
    @app.get("/api/runs")
    async def list_runs() -> list[RunSummary]:
        """List all indexing runs"""
        
    @app.get("/api/runs/{run_id}/status")
    async def get_run_status(run_id: str) -> RunStatus:
        """Real-time run status with SSE"""
        
    @app.post("/api/search")
    async def search_endpoint(request: SearchRequest) -> SearchResponse:
        """Interactive search interface"""
```

#### Frontend Architecture
- **Single Page Application** with static file serving
- **Cytoscape.js** for interactive graph visualization
- **Monaco Editor** for syntax-highlighted code viewing
- **TanStack Table** for search results and symbol browsing
- **Real-time updates** via Server-Sent Events

## Data Schemas & Contracts

### Core Index Manifest
```typescript
interface IndexManifest {
  index_id: string;           // ULID
  repo: {
    root: string;             // Absolute path
    rev: string;              // HEAD commit SHA
    worktree_dirty: boolean;  // Uncommitted changes present
  };
  config: {
    languages: string[];      // File extensions to process
    excludes: string[];       // Paths to ignore
    context_lines: number;    // Snippet context size
    features: {
      vector: boolean;
      symbol: boolean;
      graph: boolean;
    };
  };
  counts: {
    files_total: number;
    files_indexed: number;
    symbols_defs: number;
    symbols_refs: number;
    vectors: number;
  };
  paths: Record<string, string>;  // Artifact file paths
  versions: Record<string, string>;  // Tool versions
}
```

### Symbol Graph Format
```typescript
interface SymbolEntry {
  type: "def" | "ref" | "call" | "import";
  symbol?: string;           // Fully qualified symbol name
  path: string;              // Source file path
  span: [number, number];    // Byte offset range
  sig?: string;              // Type signature (for definitions)
  caller?: string;           // Calling symbol (for calls)
  callee?: string;           // Called symbol (for calls)
  from?: string;             // Import source (for imports)
}
```

### Search Result Schema
```typescript
interface SearchResult {
  path: string;
  span: [number, number];
  score: number;
  scores: {
    vector: number;
    symbol: number;
    graph: number;
  };
  content: {
    pre: string;             // Lines before match
    text: string;            // Matched content
    post: string;            // Lines after match
  };
  citation: {
    repo_root: string;
    rev: string;
    path: string;
    span: [number, number];
    content_sha: string;
  };
}
```

## Performance & Scalability

### Concurrency Model
- **I/O Concurrency**: 8 concurrent file operations (configurable)
- **CPU Concurrency**: 2 concurrent processing tasks (configurable)
- **Pipeline Parallelism**: Independent stages run concurrently where possible

### Memory Management
- **Streaming processing** for large files
- **Lazy loading** of vector embeddings
- **LRU caching** for frequently accessed symbols
- **Automatic cleanup** of temporary artifacts

### Size Limits & Handling
- **Warning threshold**: 1.5GB bundle size
- **Sharding threshold**: 2GB bundle size
- **File size limits**: Configurable per file type
- **Timeout management**: Per-stage timeout configuration

## Error Handling & Resilience

### Failure Modes
1. **Graceful degradation**: Continue with partial results when safe
2. **Gap marking**: Document missing data in manifest
3. **Retry logic**: Automatic retry for transient failures
4. **Rollback capability**: Revert to last known good state

### Monitoring & Observability
- **Structured logging** with JSON events
- **Metrics collection** for performance analysis
- **Error categorization** with actionable error messages
- **Progress tracking** with stage-level granularity

## Security & Privacy

### Data Handling
- **Local-only processing**: No external service dependencies
- **No telemetry**: Zero data transmission to external services
- **Content hashing**: Integrity verification for all artifacts
- **Access control**: File system permissions for index storage

### UI Security
- **Local binding only**: UI server binds to 127.0.0.1
- **No external ingress**: No remote access capabilities
- **Optional redaction**: Configurable sensitive data filtering

## Configuration Management

### Environment Variables
```bash
MIMIR_CONCURRENCY_IO=8          # I/O operation concurrency
MIMIR_CONCURRENCY_CPU=2         # CPU-bound task concurrency
MIMIR_MAX_BUNDLE_SIZE=2147483648 # 2GB bundle size limit
MIMIR_CACHE_DIR=~/.cache/mimir  # Cache directory override
MIMIR_LOG_LEVEL=INFO            # Logging verbosity
```

### Runtime Configuration
```python
@dataclass
class IndexConfig:
    features: dict = field(default_factory=lambda: {
        "vector": True,
        "symbol": True,
        "graph": True
    })
    context_lines: int = 5
    max_files_to_embed: Optional[int] = None
    imports_policy: dict = field(default_factory=lambda: {
        "types_for_first_order": True,
        "code_for_direct_imports": True
    })
    watch_mode: bool = False
```

## Implementation Priority & Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
1. Project setup with uv and basic structure
2. MCP server foundation with tool dispatch
3. Basic pipeline orchestration framework
4. Git discovery and file enumeration
5. Data schemas and storage structure

### Phase 2: Pipeline Implementation (Weeks 3-4)
1. RepoMapper adapter integration
2. Serena TypeScript analysis adapter
3. LEANN vector embedding integration
4. Snippet extraction and bundling
5. Basic hybrid search implementation

### Phase 3: Advanced Features (Weeks 5-6)
1. Multi-hop symbol graph navigation
2. Incremental indexing and caching
3. Error handling and resilience
4. Performance optimization
5. Comprehensive testing suite

### Phase 4: Management UI (Weeks 7-8)
1. FastAPI backend with SSE support
2. React SPA with graph visualization
3. Interactive search interface
4. Run comparison and monitoring
5. Context7 integration for external docs

This architecture provides a solid foundation for building a production-ready code research system that can handle large, complex codebases while maintaining excellent performance and usability for AI agents.