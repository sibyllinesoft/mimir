# Mimir - Deep Code Research System

## Project Purpose
Mimir is a comprehensive repository indexing system that provides intelligent code search and understanding through a multi-stage analysis pipeline. It combines vector embeddings, symbol analysis, and graph relationships to enable deep code research capabilities.

## Tech Stack
- **Language**: Python 3.11+
- **Package Manager**: uv (not pip)
- **Async Framework**: AsyncIO
- **Protocol**: stdio MCP (Model Context Protocol)
- **Web Framework**: FastAPI (for optional UI)
- **External Tools**: 
  - RepoMapper (Tree-sitter AST analysis)
  - Serena (TypeScript Server integration)
  - LEANN (CPU-based vector embeddings)
- **Testing**: pytest with asyncio support
- **Code Quality**: black (formatter), ruff (linter), mypy (type checker)

## Project Structure
```
src/repoindex/
├── data/           # Pydantic schemas and data models
├── mcp/            # MCP server implementation
├── pipeline/       # Core pipeline stages and orchestration
├── ui/             # FastAPI web interface
└── util/           # Filesystem and git utilities

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── e2e/            # End-to-end tests (empty currently)
└── conftest.py     # Pytest configuration and fixtures
```

## Six-Stage Pipeline
1. **Acquire**: Git-aware file discovery with change detection
2. **RepoMapper**: Tree-sitter AST analysis and PageRank scoring
3. **Serena**: TypeScript symbol analysis and dependency graphs
4. **LEANN**: CPU-based vector embeddings for semantic search
5. **Snippets**: Code extraction with context lines
6. **Bundle**: Zstandard compression and manifest generation

## Key Features
- Zero-prompt local indexing
- Defensible citations with exact source locations
- CPU-only operation (no GPU required)
- Hybrid search (vector + symbol + graph)
- Multi-hop reasoning for complex queries
- Real-time UI (optional)
- MCP Server with 5 tools and 4 resources