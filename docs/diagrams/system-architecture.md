# System Architecture Diagram

## Overview
This diagram shows the high-level architecture of the Mimir Deep Code Research System, including all major components and their interactions.

```mermaid
graph TB
    %% Client Layer
    subgraph "Client Layer"
        A1[AI Agent/Claude]
        A2[Developer Tools]
        A3[IDE Extensions]
    end

    %% MCP Protocol Layer
    subgraph "MCP Protocol Layer"
        B1[MCP Server]
        B2[Tool Dispatcher]
        B3[Resource Manager]
        B4[Status Monitor]
    end

    %% Security Layer
    subgraph "Security Layer"
        S1[Authentication]
        S2[Input Validation]
        S3[Sandbox Execution]
        S4[Audit Logging]
    end

    %% Pipeline Orchestration
    subgraph "Pipeline Orchestration"
        C1[Pipeline Controller]
        C2[Stage Manager]
        C3[Progress Tracker]
        C4[Error Handler]
    end

    %% Core Pipeline Stages
    subgraph "Pipeline Stages"
        D1[Acquire Stage]
        D2[RepoMapper Stage]
        D3[Serena Stage]
        D4[LEANN Stage]
        D5[Snippets Stage]
        D6[Bundle Stage]
    end

    %% External Tools
    subgraph "External Tools"
        E1[Git]
        E2[RepoMapper<br/>Tree-sitter AST]
        E3[Serena<br/>TypeScript Server]
        E4[LEANN<br/>Vector Embeddings]
    end

    %% Storage & Cache
    subgraph "Storage Layer"
        F1[Index Cache]
        F2[Vector Database]
        F3[Symbol Graph]
        F4[Bundle Artifacts]
        F5[Audit Logs]
    end

    %% Search & Query
    subgraph "Search Engine"
        G1[Hybrid Search]
        G2[Vector Search]
        G3[Symbol Search]  
        G4[Graph Expansion]
    end

    %% AI Integration
    subgraph "AI Integration"
        H1[LLM Adapter Interface]
        H2[Gemini Adapter]
        H3[Future: OpenAI Adapter]
        H4[Future: Claude Adapter]
    end

    %% Data Flow Connections
    A1 --> B1
    A2 --> B1
    A3 --> B1
    
    B1 --> B2
    B2 --> B3
    B2 --> B4
    B2 --> S1
    
    S1 --> S2
    S2 --> S3
    S3 --> C1
    
    C1 --> C2
    C2 --> C3
    C2 --> C4
    
    C2 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> D4
    D4 --> D5
    D5 --> D6
    
    D1 --> E1
    D2 --> E2
    D3 --> E3
    D4 --> E4
    
    D2 --> F1
    D3 --> F3
    D4 --> F2
    D6 --> F4
    S4 --> F5
    
    B3 --> G1
    G1 --> G2
    G1 --> G3
    G1 --> G4
    
    G2 --> F2
    G3 --> F3
    G4 --> F3
    
    G1 --> H1
    H1 --> H2
    H1 --> H3
    H1 --> H4

    %% Styling
    classDef client fill:#e1f5fe
    classDef mcp fill:#f3e5f5
    classDef security fill:#ffebee
    classDef pipeline fill:#e8f5e8
    classDef stage fill:#fff3e0
    classDef external fill:#f1f8e9
    classDef storage fill:#fafafa
    classDef search fill:#e3f2fd
    classDef ai fill:#f9fbe7

    class A1,A2,A3 client
    class B1,B2,B3,B4 mcp
    class S1,S2,S3,S4 security
    class C1,C2,C3,C4 pipeline
    class D1,D2,D3,D4,D5,D6 stage
    class E1,E2,E3,E4 external
    class F1,F2,F3,F4,F5 storage
    class G1,G2,G3,G4 search
    class H1,H2,H3,H4 ai
```

## Component Responsibilities

### Client Layer
- **AI Agent/Claude**: Primary consumer of MCP protocol for code analysis
- **Developer Tools**: CLI tools and scripts for direct repository indexing
- **IDE Extensions**: Integration with development environments

### MCP Protocol Layer
- **MCP Server**: Main entry point implementing stdio MCP protocol
- **Tool Dispatcher**: Routes tool calls to appropriate handlers
- **Resource Manager**: Serves status updates, manifests, and artifacts
- **Status Monitor**: Tracks pipeline progress and system health

### Security Layer
- **Authentication**: API key and token validation
- **Input Validation**: Sanitizes and validates all inputs
- **Sandbox Execution**: Isolated execution environment for external tools
- **Audit Logging**: Comprehensive security event logging

### Pipeline Orchestration
- **Pipeline Controller**: Orchestrates the complete indexing workflow
- **Stage Manager**: Manages individual stage execution and dependencies
- **Progress Tracker**: Reports real-time progress to clients
- **Error Handler**: Manages failures and recovery strategies

### Pipeline Stages
- **Acquire**: Git-based file discovery and change detection
- **RepoMapper**: AST analysis and dependency graph construction
- **Serena**: TypeScript symbol resolution and analysis
- **LEANN**: Vector embedding generation
- **Snippets**: Code snippet extraction and context preparation
- **Bundle**: Artifact packaging and manifest generation

### External Tools
- **Git**: Version control integration for file tracking
- **RepoMapper**: Tree-sitter based AST parsing and analysis
- **Serena**: TypeScript language server integration
- **LEANN**: CPU-based vector embedding generation

### Storage Layer
- **Index Cache**: Persistent caching of intermediate results
- **Vector Database**: High-dimensional embedding storage
- **Symbol Graph**: Structured symbol relationships
- **Bundle Artifacts**: Complete index packages for distribution
- **Audit Logs**: Security and operational audit trails

### Search Engine
- **Hybrid Search**: Multi-modal search orchestration
- **Vector Search**: Semantic similarity search using embeddings
- **Symbol Search**: Exact and fuzzy symbol name matching
- **Graph Expansion**: Relationship-based result expansion

### AI Integration
- **LLM Adapter Interface**: Abstract interface for multiple AI providers
- **Gemini Adapter**: Google Gemini AI integration (implemented)
- **OpenAI Adapter**: OpenAI API integration (planned)
- **Claude Adapter**: Anthropic Claude integration (planned)

## Data Flow Summary

1. **Indexing Flow**: Client → MCP Server → Security Layer → Pipeline Controller → 6 Pipeline Stages → External Tools → Storage Layer
2. **Search Flow**: Client → MCP Server → Search Engine → Storage Layer → Results
3. **AI Analysis Flow**: Client → MCP Server → Search Engine → AI Integration → LLM Adapter → Results
4. **Status Flow**: Pipeline → Status Monitor → Resource Manager → Client

## Key Design Principles

- **Zero-configuration**: Minimal setup required for basic operation
- **Incremental processing**: Smart caching and delta detection
- **Concurrent execution**: Async operations where dependencies allow
- **Robust error handling**: Graceful degradation and recovery
- **Extensible architecture**: Plugin-based design for future enhancements