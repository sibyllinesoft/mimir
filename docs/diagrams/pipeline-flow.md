# Pipeline Flow Diagram

## Overview
This diagram shows the detailed flow through the six-stage indexing pipeline, including stage dependencies, data transformations, and error handling.

```mermaid
graph TD
    %% Input
    subgraph "Input"
        I1[Repository Path]
        I2[Git Revision]
        I3[Index Options]
        I4[Language Config]
    end

    %% Stage 1: Acquire
    subgraph "Stage 1: Acquire"
        A1[Git File Discovery]
        A2[Change Detection]
        A3[Cache Key Generation]
        A4[File List + Overlay]
    end

    %% Stage 2: RepoMapper  
    subgraph "Stage 2: RepoMapper"
        B1[Tree-sitter AST Analysis]
        B2[Dependency Graph Build]
        B3[PageRank Scoring]
        B4[File Importance Map]
    end

    %% Stage 3: Serena
    subgraph "Stage 3: Serena"
        C1[TypeScript Server Init]
        C2[Symbol Extraction]
        C3[Import Resolution]
        C4[Symbol Graph]
    end

    %% Stage 4: LEANN
    subgraph "Stage 4: LEANN"
        D1[Function-level Chunking]
        D2[Vector Embedding Gen]
        D3[Importance-weighted Index]
        D4[Vector Database]
    end

    %% Stage 5: Snippets
    subgraph "Stage 5: Snippets"
        E1[Context Window Extraction]
        E2[Symbol Context Merge]
        E3[Citation Preparation]
        E4[Search-ready Snippets]
    end

    %% Stage 6: Bundle
    subgraph "Stage 6: Bundle"
        F1[Manifest Generation]
        F2[Artifact Compression]
        F3[Bundle Packaging]
        F4[Index Complete]
    end

    %% Error Handling
    subgraph "Error Handling"
        G1[Stage Failure Detection]
        G2[Rollback Mechanism]
        G3[Partial Index Recovery]
        G4[Error Reporting]
    end

    %% Caching System
    subgraph "Caching Layer"
        H1[Stage Result Cache]
        H2[Incremental Updates]
        H3[Cache Invalidation]
        H4[Performance Optimization]
    end

    %% Progress Reporting
    subgraph "Progress Tracking"
        P1[Stage Progress: 0-16%]
        P2[Stage Progress: 17-33%]
        P3[Stage Progress: 34-50%] 
        P4[Stage Progress: 51-66%]
        P5[Stage Progress: 67-83%]
        P6[Stage Progress: 84-100%]
    end

    %% Data Flow
    I1 --> A1
    I2 --> A2
    I3 --> A3
    I4 --> A1
    
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> B1
    
    B1 --> B2
    B2 --> B3
    B3 --> B4
    B4 --> C1
    
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> D1
    
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> E1
    
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> F1
    
    F1 --> F2
    F2 --> F3
    F3 --> F4

    %% Progress Reporting Flow
    A4 --> P1
    B4 --> P2
    C4 --> P3
    D4 --> P4
    E4 --> P5
    F4 --> P6

    %% Error Handling Flow
    A1 -.-> G1
    B1 -.-> G1
    C1 -.-> G1
    D1 -.-> G1
    E1 -.-> G1
    F1 -.-> G1
    
    G1 --> G2
    G2 --> G3
    G3 --> G4

    %% Caching Flow
    A4 --> H1
    B4 --> H1
    C4 --> H1
    D4 --> H1
    E4 --> H1
    F4 --> H1
    
    H1 --> H2
    H2 --> H3
    H3 --> H4

    %% Styling
    classDef input fill:#e3f2fd
    classDef stage fill:#e8f5e8
    classDef error fill:#ffebee
    classDef cache fill:#fff3e0
    classDef progress fill:#f3e5f5

    class I1,I2,I3,I4 input
    class A1,A2,A3,A4,B1,B2,B3,B4,C1,C2,C3,C4,D1,D2,D3,D4,E1,E2,E3,E4,F1,F2,F3,F4 stage
    class G1,G2,G3,G4 error
    class H1,H2,H3,H4 cache
    class P1,P2,P3,P4,P5,P6 progress
```

## Stage Dependencies and Sequencing

### Sequential Dependencies
- **Acquire → RepoMapper**: File list required for AST analysis
- **RepoMapper → Serena**: Dependency graph needed for symbol resolution priority
- **Serena → LEANN**: Symbol information enhances vector embedding quality
- **LEANN → Snippets**: Vector index needed for semantic snippet selection
- **Snippets → Bundle**: All artifacts must be ready for packaging

### Parallel Opportunities
- Within each stage, file processing can be parallelized
- Cache validation can occur concurrently with main processing
- Progress reporting runs independently of stage execution

## Stage Input/Output Details

### Stage 1: Acquire
**Input**: Repository path, git revision, configuration
**Output**: Filtered file list, change overlay, cache key
**Key Operations**:
- `git ls-files` for tracked file enumeration
- `git hash-object` for change detection
- File type filtering based on configuration
- Cache key computation from HEAD + config hash

### Stage 2: RepoMapper
**Input**: File list from Acquire stage
**Output**: Dependency graph, file importance scores
**Key Operations**:
- Tree-sitter AST parsing for each file
- Dependency relationship extraction
- PageRank algorithm for file importance
- Graph serialization for downstream stages

### Stage 3: Serena
**Input**: File list + dependency graph from RepoMapper
**Output**: Symbol graph with definitions, references, types
**Key Operations**:
- TypeScript Language Server initialization
- Symbol extraction and classification
- Import resolution via package.json exports
- First-order type dependency analysis

### Stage 4: LEANN
**Input**: Files + symbol information from Serena
**Output**: Vector embedding index
**Key Operations**:
- Function-level content chunking (~400 tokens)
- CPU-based embedding generation
- Importance-weighted indexing using RepoMapper scores
- Vector database construction

### Stage 5: Snippets
**Input**: All previous stage outputs
**Output**: Search-ready code snippets with context
**Key Operations**:
- Context window extraction around symbols
- Cross-reference merging from symbol graph
- Citation metadata preparation
- Snippet ranking and deduplication

### Stage 6: Bundle
**Input**: All stage artifacts
**Output**: Compressed index bundle + manifest
**Key Operations**:
- Index manifest generation with metadata
- Artifact compression (tar.zst format)
- Bundle integrity verification
- Final status update to MCP clients

## Error Handling Strategy

### Stage Failure Recovery
1. **Detection**: Each stage validates its preconditions and outputs
2. **Rollback**: Failed stages trigger cleanup of partial artifacts
3. **Recovery**: System attempts partial index generation when possible
4. **Reporting**: Detailed error context provided to clients

### Cache Management
- **Validation**: Cache entries validated against current repo state
- **Invalidation**: Smart invalidation based on file changes and config
- **Optimization**: Incremental updates for unchanged components
- **Cleanup**: Automatic cleanup of stale cache entries

## Performance Characteristics

### Typical Processing Times
- **Acquire**: 1-5 seconds (depends on repo size)
- **RepoMapper**: 10-60 seconds (AST parsing intensive)
- **Serena**: 20-120 seconds (TypeScript analysis)
- **LEANN**: 30-180 seconds (CPU vector generation)
- **Snippets**: 5-20 seconds (context extraction)
- **Bundle**: 2-10 seconds (compression)

### Scalability Factors
- **Repository Size**: Linear scaling with number of files
- **Code Complexity**: Quadratic impact on Serena analysis
- **Concurrency**: Configurable parallelism per stage
- **Memory Usage**: Peak during vector generation phase

### Cache Effectiveness
- **Cold Start**: Full pipeline execution required
- **Incremental**: Only changed files re-processed
- **Configuration Changes**: Selective stage re-execution
- **Hit Rate**: Typically 70-90% in active development workflows