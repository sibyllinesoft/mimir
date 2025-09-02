# Mimir-Lens Integration Architecture

## Executive Summary

This document defines the integration architecture between **Mimir** (advanced code analysis and research orchestration) and **Lens** (high-performance document indexing infrastructure). The design preserves each system's core strengths while creating clean interfaces for collaboration.

## System Strengths & Responsibilities

### Lens Core Strengths (General Indexing Infrastructure)
- **High-Performance Async Operations**: FastAPI + async/await patterns for concurrent document processing
- **Modern REST API**: Clean RESTful interface with OpenAPI documentation
- **Vector Embeddings Service**: Production-ready vector search capabilities  
- **Database-Backed Storage**: Persistent, scalable storage with proper indexing
- **Microservices Architecture**: Clean service boundaries and horizontal scalability

### Mimir Unique Value Proposition (Specialized Code Intelligence)
- **Deep Code Research**: AST parsing, semantic analysis, symbol resolution
- **Advanced Pipeline Orchestration**: 6-stage research pipeline with complex dependencies
- **MCP Integration**: Native Claude Code compatibility via stdio MCP protocol
- **Specialized Analysis Tools**: Serena (TypeScript), RAPTOR (hierarchical summaries), RepoMapper
- **Multi-Hop Reasoning**: Symbol graph traversal and complex code relationship analysis

## Integration Architecture

### Core Integration Pattern: Federated Hybrid System

```mermaid
graph TD
    CC[Claude Code] -->|MCP Protocol| M[Mimir MCP Server]
    
    M --> MP[Mimir Pipeline Controller]
    MP --> MA[Mimir Analysis Engines]
    MP --> LH[Lens HTTP Client]
    
    LH --> LA[Lens API Gateway] 
    LA --> LE[Lens Embedding Service]
    LA --> LS[Lens Storage Service]
    LA --> LI[Lens Index Service]
    
    MA --> |Specialized Analysis| MP
    LE --> |Vector Results| LH
    LS --> |Document Storage| LH
    LI --> |Search Results| LH
```

### 1. Interface Layer Design

#### Mimir → Lens Integration Client
```python
# src/repoindex/integrations/lens_client.py
class LensIntegrationClient:
    """High-level client for Lens services integration"""
    
    def __init__(self, lens_base_url: str, timeout: int = 30):
        self.base_url = lens_base_url
        self.session = aiohttp.ClientSession(timeout=timeout)
        
    # Document Storage Operations (High-Volume)
    async def bulk_index_documents(
        self, 
        documents: List[Document], 
        collection: str = "code"
    ) -> IndexResponse:
        """Leverage Lens's high-performance bulk indexing"""
        
    async def get_document_embeddings(
        self, 
        document_ids: List[str]
    ) -> Dict[str, np.ndarray]:
        """Retrieve pre-computed embeddings from Lens"""
        
    # Vector Search Operations (Performance Critical)
    async def vector_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 100,
        filters: Dict[str, Any] = None
    ) -> List[SearchResult]:
        """Use Lens's optimized vector search for initial candidate retrieval"""
        
    # Metadata and Storage
    async def store_analysis_metadata(
        self, 
        index_id: str, 
        metadata: Dict[str, Any]
    ) -> None:
        """Store Mimir analysis results in Lens for persistence"""
```

#### Lens → Mimir Callback Interface  
```python
# For advanced analysis requests that exceed Lens capabilities
class MimirCallbackHandler:
    """Handles complex analysis requests from Lens"""
    
    async def analyze_code_relationships(
        self, 
        file_paths: List[str], 
        analysis_type: str
    ) -> AnalysisResult:
        """Delegate complex code analysis back to Mimir"""
        
    async def multi_hop_symbol_resolution(
        self, 
        symbol_query: str, 
        context: Dict[str, Any]
    ) -> SymbolGraph:
        """Use Mimir's specialized symbol analysis capabilities"""
```

### 2. Data Flow Architecture

#### Primary Indexing Flow: Mimir-Orchestrated with Lens Delegation

```python
# Enhanced Mimir Pipeline with Lens Integration
class HybridIndexingPipeline(IndexingPipeline):
    """Enhanced pipeline that leverages Lens for high-performance operations"""
    
    def __init__(self, storage_dir: Path, lens_client: LensIntegrationClient):
        super().__init__(storage_dir)
        self.lens_client = lens_client
        
    async def _stage_bulk_embedding(self, context: PipelineContext):
        """Delegate bulk embedding to Lens for performance"""
        
        # 1. Prepare documents for Lens
        documents = await self._prepare_documents_for_lens(context)
        
        # 2. Bulk index via Lens (high-performance)
        lens_response = await self.lens_client.bulk_index_documents(
            documents=documents,
            collection=f"mimir_{context.index_id}"
        )
        
        # 3. Store Lens metadata for later retrieval
        context.lens_collection_id = lens_response.collection_id
        
    async def _stage_hybrid_analysis(self, context: PipelineContext):
        """Combine Lens vector search with Mimir specialized analysis"""
        
        # 1. Get vector candidates from Lens (fast, broad search)
        vector_candidates = await self.lens_client.vector_search(
            query_embedding=context.query_embedding,
            k=200,  # Cast wide net
            filters={"collection": f"mimir_{context.index_id}"}
        )
        
        # 2. Apply Mimir's specialized analysis (deep, focused)
        for candidate in vector_candidates:
            # Serena symbol analysis
            symbol_analysis = await self._analyze_symbols(candidate.file_path)
            # RepoMapper dependency analysis  
            dep_analysis = await self._analyze_dependencies(candidate.file_path)
            # RAPTOR hierarchical analysis
            hierarchical_analysis = await self._analyze_hierarchy(candidate.content)
            
            candidate.mimir_analysis = {
                "symbols": symbol_analysis,
                "dependencies": dep_analysis,
                "hierarchy": hierarchical_analysis
            }
            
        return vector_candidates
```

#### Query Processing: Hybrid Approach

```python
# Enhanced Query Engine with Lens Integration
class HybridQueryEngine(QueryEngine):
    """Combines Lens performance with Mimir intelligence"""
    
    async def hybrid_search(
        self, 
        query: str, 
        index_id: str, 
        k: int = 20
    ) -> List[SearchResult]:
        """Multi-stage hybrid search"""
        
        # Stage 1: Lens Vector Search (Performance Layer)
        # - Fast vector similarity for initial candidate pool
        # - Leverages Lens's optimized indexing and search
        vector_candidates = await self.lens_client.vector_search(
            query_embedding=await self._embed_query(query),
            k=k * 5,  # Over-retrieve for downstream filtering
            filters={"index_id": index_id}
        )
        
        # Stage 2: Mimir Semantic Analysis (Intelligence Layer)
        # - Symbol-aware reranking
        # - Code relationship analysis
        # - Multi-hop reasoning
        analyzed_candidates = []
        for candidate in vector_candidates:
            # Apply Mimir's specialized code understanding
            semantic_score = await self._compute_semantic_relevance(
                query, candidate, index_id
            )
            symbol_score = await self._compute_symbol_relevance(
                query, candidate, index_id
            )
            
            candidate.hybrid_score = (
                0.4 * candidate.vector_score +      # Lens strength
                0.4 * semantic_score +               # Mimir strength  
                0.2 * symbol_score                   # Mimir strength
            )
            analyzed_candidates.append(candidate)
            
        # Stage 3: Final Ranking and Context Assembly
        # - Re-rank by hybrid score
        # - Assemble rich context using Mimir's analysis
        return sorted(analyzed_candidates, key=lambda x: x.hybrid_score, reverse=True)[:k]
```

### 3. System Boundaries & Responsibilities

#### Lens Responsibilities (Performance & Scale)
```yaml
Primary:
  - Document ingestion and storage at scale
  - Vector embedding computation and indexing  
  - High-performance vector similarity search
  - Database persistence and backup/recovery
  - API rate limiting and request management
  - Basic text preprocessing and tokenization
  
Secondary:
  - Simple text-based search and filtering
  - Document metadata management
  - Basic analytics and usage metrics
  - User authentication and authorization
  
Explicitly NOT:
  - Code-specific semantic analysis
  - Symbol resolution and relationship mapping
  - Multi-hop reasoning across code relationships
  - AST parsing and syntax understanding
  - Complex pipeline orchestration
```

#### Mimir Responsibilities (Intelligence & Research)
```yaml
Primary:
  - Advanced code analysis (AST, symbols, types)
  - Multi-stage research pipeline orchestration
  - MCP protocol implementation for Claude Code
  - Complex multi-hop reasoning and inference
  - Code relationship graph construction and traversal
  - Specialized tool integration (Serena, RepoMapper, RAPTOR)
  
Secondary:
  - Query understanding and intent parsing
  - Result synthesis and context assembly
  - Research workflow coordination
  - Domain-specific code intelligence
  
Explicitly NOT:
  - High-volume document storage
  - Basic vector embedding computation
  - Simple similarity search
  - Database administration and backup
  - User management and authentication
```

### 4. Integration Patterns

#### Pattern 1: Delegated Heavy Lifting
```python
# Mimir delegates performance-intensive tasks to Lens
async def index_large_repository(self, repo_path: str) -> str:
    """Use Lens for bulk operations, Mimir for intelligence"""
    
    # 1. Discover and preprocess files (Mimir intelligence)
    files_to_index = await self.intelligent_file_discovery(repo_path)
    
    # 2. Bulk index documents (Lens performance) 
    lens_collection = await self.lens_client.bulk_index_documents(
        documents=files_to_index,
        collection=f"repo_{repo_hash}"
    )
    
    # 3. Build specialized analysis (Mimir intelligence)
    analysis_graph = await self.build_analysis_graph(files_to_index)
    
    # 4. Store cross-references
    await self.lens_client.store_analysis_metadata(
        collection_id=lens_collection.id,
        metadata={"mimir_analysis": analysis_graph}
    )
```

#### Pattern 2: Intelligent Filtering  
```python
# Use Lens for broad search, Mimir for intelligent filtering
async def complex_code_search(self, query: str, context: Dict) -> List[Result]:
    """Lens casts wide net, Mimir applies intelligence"""
    
    # 1. Broad vector search via Lens (fast, inclusive)
    candidates = await self.lens_client.vector_search(
        query_embedding=query_embedding,
        k=500,  # Cast wide net
        similarity_threshold=0.3  # Low threshold
    )
    
    # 2. Intelligent filtering via Mimir (precise, context-aware)
    filtered_results = []
    for candidate in candidates:
        if await self.mimir_relevance_filter(candidate, query, context):
            # Add Mimir's specialized analysis
            candidate.code_analysis = await self.analyze_code_context(candidate)
            filtered_results.append(candidate)
            
    return filtered_results[:20]  # Return top results
```

#### Pattern 3: Complementary Analysis
```python
# Each system contributes its strengths to final result
async def comprehensive_analysis(self, target: str) -> AnalysisResult:
    """Combine both systems' capabilities"""
    
    # Parallel execution of complementary analyses
    lens_task = asyncio.create_task(
        self.lens_client.semantic_analysis(target)
    )
    mimir_task = asyncio.create_task(
        self.mimir_deep_analysis(target)  
    )
    
    lens_result, mimir_result = await asyncio.gather(lens_task, mimir_task)
    
    # Synthesis step - combine results intelligently
    return self.synthesize_analysis(lens_result, mimir_result)
```

### 5. Deployment & Configuration

#### Docker Compose Integration
```yaml
# docker-compose.integrated.yml
version: '3.8'

services:
  lens-api:
    image: lens:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://lens:password@lens-db:5432/lens
      - EMBEDDING_SERVICE_URL=http://lens-embeddings:8001
    depends_on:
      - lens-db
      - lens-embeddings
      
  lens-embeddings:
    image: lens-embeddings:latest
    ports:
      - "8001:8001"
    deploy:
      resources:
        limits:
          memory: 4G
          
  lens-db:
    image: postgres:15
    environment:
      - POSTGRES_DB=lens
      - POSTGRES_USER=lens
      - POSTGRES_PASSWORD=password
    volumes:
      - lens_data:/var/lib/postgresql/data
      
  mimir-mcp:
    build:
      context: ./mimir
      dockerfile: Dockerfile
    environment:
      - LENS_API_URL=http://lens-api:8000
      - MIMIR_STORAGE_DIR=/var/mimir/indexes
      - MIMIR_INTEGRATION_MODE=lens_hybrid
    volumes:
      - mimir_indexes:/var/mimir/indexes
    depends_on:
      - lens-api
    stdin_open: true
    tty: true

volumes:
  lens_data:
  mimir_indexes:
```

#### Configuration Management
```python
# src/repoindex/config/integration.py
@dataclass
class LensIntegrationConfig:
    """Configuration for Lens integration"""
    
    enabled: bool = False
    base_url: str = "http://localhost:8000"
    timeout_seconds: int = 30
    max_concurrent_requests: int = 10
    
    # Performance delegation settings
    delegate_bulk_indexing: bool = True
    delegate_vector_search: bool = True  
    delegate_document_storage: bool = True
    
    # Fallback settings
    fallback_on_lens_failure: bool = True
    local_embedding_fallback: bool = True
    
    # Collection management
    collection_prefix: str = "mimir"
    collection_ttl_days: int = 30
```

### 6. API Specifications

#### Mimir MCP Tools (Enhanced for Integration)
```python
# Enhanced MCP tools with Lens integration
TOOLS = [
    {
        "name": "ensure_repo_index",
        "description": "Index repository using hybrid Mimir-Lens pipeline",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "use_lens": {"type": "boolean", "default": True},
                "lens_collection": {"type": "string", "optional": True}
            }
        }
    },
    {
        "name": "hybrid_search",
        "description": "Search using combined Lens performance + Mimir intelligence",
        "inputSchema": {
            "type": "object", 
            "properties": {
                "query": {"type": "string"},
                "index_id": {"type": "string"},
                "lens_weight": {"type": "number", "default": 0.4},
                "mimir_weight": {"type": "number", "default": 0.6}
            }
        }
    },
    {
        "name": "deep_code_analysis", 
        "description": "Advanced code analysis using Mimir's specialized capabilities",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "analysis_depth": {"type": "integer", "default": 2},
                "include_lens_context": {"type": "boolean", "default": True}
            }
        }
    }
]
```

#### Lens HTTP API (Extension Points)
```python
# Lens API extensions for Mimir integration
@app.post("/api/v1/mimir/bulk_index")
async def bulk_index_for_mimir(request: MimirBulkIndexRequest):
    """Specialized bulk indexing endpoint for Mimir"""
    
@app.get("/api/v1/mimir/collections/{collection_id}/embeddings")  
async def get_embeddings_for_mimir(collection_id: str):
    """Retrieve embeddings for Mimir analysis"""
    
@app.post("/api/v1/mimir/analysis_metadata")
async def store_mimir_analysis(request: AnalysisMetadataRequest):
    """Store Mimir analysis results"""
```

### 7. Benefits & Trade-offs

#### Benefits
```yaml
Performance:
  - Lens handles high-volume, performance-critical operations
  - Mimir focuses on complex analysis without performance bottlenecks
  - Better resource utilization and specialization
  
Scalability:
  - Lens provides database-backed persistence and horizontal scaling
  - Mimir can focus on stateless analysis and reasoning
  - Clean separation allows independent scaling
  
Maintainability:
  - Each system maintains its core competencies
  - Clear interfaces reduce coupling and complexity
  - Independent development and deployment cycles
  
Functionality:
  - Combined capabilities exceed either system alone
  - Lens performance + Mimir intelligence = superior results
  - Preserved MCP integration maintains Claude Code compatibility
```

#### Trade-offs
```yaml
Complexity:
  - Additional network communication overhead
  - Two systems to deploy, monitor, and maintain
  - More complex error handling and failure modes
  
Dependencies: 
  - Mimir becomes partially dependent on Lens availability
  - Need robust fallback mechanisms
  - Version compatibility management between systems
  
Latency:
  - Network calls add latency vs pure local processing
  - Need careful optimization of integration points
  - Caching and async patterns required
```

## Implementation Roadmap

### Phase 1: Foundation (Week 1)
- [ ] Create Lens client integration library
- [ ] Implement basic HTTP communication patterns
- [ ] Add configuration management for integration
- [ ] Basic error handling and fallback mechanisms

### Phase 2: Core Integration (Week 2)
- [ ] Implement delegated bulk indexing to Lens
- [ ] Add hybrid search combining both systems
- [ ] Create metadata storage and retrieval patterns
- [ ] Integration testing between systems

### Phase 3: Advanced Features (Week 3)  
- [ ] Multi-stage analysis pipelines
- [ ] Intelligent result synthesis
- [ ] Performance optimization and caching
- [ ] Comprehensive monitoring and observability

### Phase 4: Production Readiness (Week 4)
- [ ] Docker Compose integration
- [ ] Production configuration management
- [ ] Documentation and runbooks
- [ ] Performance benchmarking and optimization

This architecture preserves Mimir's unique MCP integration and specialized code intelligence while leveraging Lens's high-performance infrastructure for scalable document processing. The result is a system that combines the best of both platforms while maintaining clean architectural boundaries.