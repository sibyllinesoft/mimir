# Mimir API Reference

## Overview

Mimir implements the Model Context Protocol (MCP) as a stdio server, providing 5 tools and 4 resource types for intelligent code repository indexing and search. This document provides complete API specifications with examples.

## MCP Server Interface

### Connection

Mimir runs as an MCP stdio server, communicating via JSON-RPC over stdin/stdout:

```bash
# Start the server
python -m repoindex.main mcp

# Or with security enabled
python -m repoindex.main_secure mcp
```

### Capabilities

- **Tools**: 5 core operations for repository indexing and search
- **Resources**: 4 resource types for accessing status, manifests, logs, and bundles
- **Notifications**: Real-time progress updates during pipeline execution

## Tools

### 1. ensure_repo_index

Creates or updates a repository index using the full 6-stage pipeline.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "path": {
      "type": "string",
      "description": "Path to repository root",
      "required": true
    },
    "rev": {
      "type": "string", 
      "description": "Git revision (defaults to HEAD)",
      "default": "HEAD"
    },
    "language": {
      "type": "string",
      "description": "Primary language for analysis",
      "default": "ts",
      "enum": ["ts", "tsx", "js", "jsx", "py", "rs", "go"]
    },
    "index_opts": {
      "type": "object",
      "description": "Additional indexing options",
      "properties": {
        "context_lines": {"type": "integer", "default": 5, "minimum": 0, "maximum": 20},
        "max_files_to_embed": {"type": "integer", "minimum": 1},
        "excludes": {"type": "array", "items": {"type": "string"}},
        "features": {
          "type": "object",
          "properties": {
            "vector": {"type": "boolean", "default": true},
            "symbol": {"type": "boolean", "default": true}, 
            "graph": {"type": "boolean", "default": true}
          }
        }
      }
    }
  },
  "required": ["path"]
}
```

#### Response Format

```json
{
  "index_id": "01JA2B3C4D5E6F7G8H",
  "status_uri": "mimir://indexes/{index_id}/status.json",
  "manifest_uri": "mimir://indexes/{index_id}/manifest.json"
}
```

#### Example Usage

```python
await mcp_client.call_tool("ensure_repo_index", {
    "path": "/path/to/typescript/project",
    "language": "ts",
    "index_opts": {
        "context_lines": 3,
        "max_files_to_embed": 1000,
        "excludes": ["node_modules/", "dist/", "*.test.ts"],
        "features": {
            "vector": true,
            "symbol": true,
            "graph": false
        }
    }
})
```

#### Pipeline Stages

The indexing process executes these stages sequentially:

1. **Acquire**: Git-aware file discovery with change detection
2. **RepoMapper**: AST analysis and PageRank scoring 
3. **Serena**: TypeScript symbol analysis and dependency graphs
4. **LEANN**: CPU-based vector embeddings
5. **Snippets**: Code extraction with context
6. **Bundle**: Zstandard compression and manifest generation

### 2. search_repo

Performs hybrid search using vector similarity, symbol matching, and graph expansion.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "index_id": {
      "type": "string",
      "description": "Index identifier from ensure_repo_index",
      "required": true
    },
    "query": {
      "type": "string", 
      "description": "Search query text",
      "required": true
    },
    "k": {
      "type": "integer",
      "description": "Number of results to return",
      "default": 20,
      "minimum": 1,
      "maximum": 100
    },
    "features": {
      "type": "object",
      "description": "Search modality configuration",
      "properties": {
        "vector": {"type": "boolean", "default": true},
        "symbol": {"type": "boolean", "default": true},
        "graph": {"type": "boolean", "default": true}
      }
    },
    "context_lines": {
      "type": "integer",
      "description": "Lines of context around matches",
      "default": 5,
      "minimum": 0,
      "maximum": 20
    }
  },
  "required": ["index_id", "query"]
}
```

#### Response Format

```json
{
  "results": [
    {
      "path": "src/components/Button.tsx",
      "span": [245, 389],
      "score": 0.87,
      "scores": {
        "vector": 0.82,
        "symbol": 0.95,
        "graph": 0.34
      },
      "content": {
        "pre": "import React from 'react';\n\n",
        "text": "export function Button({ children, onClick }: ButtonProps) {\n  return (\n    <button onClick={onClick}>\n      {children}\n    </button>\n  );\n}",
        "post": "\n\nexport default Button;"
      },
      "citation": {
        "repo_root": "/path/to/project",
        "rev": "a1b2c3d4",
        "path": "src/components/Button.tsx",
        "span": [245, 389],
        "content_sha": "e3b0c44298fc1c14"
      }
    }
  ],
  "total": 23,
  "query_time_ms": 45.2
}
```

#### Search Features

- **Vector Search**: Semantic similarity using LEANN embeddings
- **Symbol Search**: Exact and fuzzy matching of function/class names
- **Graph Expansion**: Related symbols via call graphs and imports

#### Example Usage

```python
# Basic search
await mcp_client.call_tool("search_repo", {
    "index_id": "01JA2B3C4D5E6F7G8H",
    "query": "authentication middleware",
    "k": 10
})

# Vector-only search
await mcp_client.call_tool("search_repo", {
    "index_id": "01JA2B3C4D5E6F7G8H", 
    "query": "handle user login",
    "features": {"vector": true, "symbol": false, "graph": false}
})

# Symbol search with graph expansion
await mcp_client.call_tool("search_repo", {
    "index_id": "01JA2B3C4D5E6F7G8H",
    "query": "validateUser",
    "features": {"vector": false, "symbol": true, "graph": true}
})
```

### 3. ask_index

Performs complex question answering using multi-hop symbol graph reasoning.

#### Input Schema

```json
{
  "type": "object", 
  "properties": {
    "index_id": {
      "type": "string",
      "description": "Index identifier",
      "required": true
    },
    "question": {
      "type": "string",
      "description": "Natural language question about the codebase",
      "required": true
    },
    "context_lines": {
      "type": "integer",
      "description": "Lines of context for evidence",
      "default": 5,
      "minimum": 0,
      "maximum": 20
    }
  },
  "required": ["index_id", "question"]
}
```

#### Response Format

```json
{
  "question": "How does user authentication work?",
  "reasoning_steps": [
    {
      "step": 1,
      "intent": "symbol",
      "query": "authenticate",
      "results_count": 5
    },
    {
      "step": 2, 
      "intent": "callgraph",
      "query": "AuthService.authenticate",
      "results_count": 12
    }
  ],
  "evidence": [
    {
      "path": "src/auth/AuthService.ts",
      "span": [123, 456],
      "relevance": 0.95,
      "content": {
        "pre": "class AuthService {\n",
        "text": "  async authenticate(token: string): Promise<User> {\n    const decoded = jwt.verify(token, this.secret);\n    return this.userService.findById(decoded.userId);\n  }",
        "post": "\n\n  async logout(userId: string) {"
      },
      "citation": {
        "repo_root": "/path/to/project",
        "rev": "a1b2c3d4",
        "path": "src/auth/AuthService.ts",
        "span": [123, 456],
        "content_sha": "a1b2c3d4e5f6"
      }
    }
  ],
  "query_time_ms": 127.3
}
```

#### Question Types

The system recognizes several question intents:

- **Symbol**: Find specific functions, classes, or variables
- **File**: Locate files by name or purpose
- **Import**: Trace dependency relationships
- **Callgraph**: Analyze function call patterns

#### Example Usage

```python
# Architecture questions
await mcp_client.call_tool("ask_index", {
    "index_id": "01JA2B3C4D5E6F7G8H",
    "question": "How does the authentication system work in this codebase?"
})

# Implementation questions
await mcp_client.call_tool("ask_index", {
    "index_id": "01JA2B3C4D5E6F7G8H", 
    "question": "What functions call the validatePassword method?"
})

# Debugging questions
await mcp_client.call_tool("ask_index", {
    "index_id": "01JA2B3C4D5E6F7G8H",
    "question": "Where is the UserNotFoundError exception thrown?"
})
```

### 4. get_repo_bundle

Retrieves the complete index bundle for a repository.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "index_id": {
      "type": "string",
      "description": "Index identifier",
      "required": true
    }
  },
  "required": ["index_id"]
}
```

#### Response Format

```json
{
  "bundle_uri": "mimir://indexes/{index_id}/bundle.tar.zst",
  "manifest_uri": "mimir://indexes/{index_id}/manifest.json"
}
```

#### Bundle Contents

The bundle contains all pipeline artifacts:

- `manifest.json` - Index metadata and configuration
- `repomap.json` - File importance scores and dependency graph
- `serena_graph.jsonl` - Symbol definitions, references, and calls
- `leann.index` - Vector index (binary format)
- `vectors.bin` - Raw embedding vectors
- `snippets.jsonl` - Source code snippets with context
- `types/` - TypeScript declaration files
- `vendor_src/` - Direct import source code

#### Example Usage

```python
# Get bundle URIs
response = await mcp_client.call_tool("get_repo_bundle", {
    "index_id": "01JA2B3C4D5E6F7G8H"
})

# Read bundle resource
bundle_data = await mcp_client.read_resource(response["bundle_uri"])
```

### 5. cancel

Cancels an ongoing indexing operation.

#### Input Schema

```json
{
  "type": "object",
  "properties": {
    "index_id": {
      "type": "string", 
      "description": "Index identifier to cancel",
      "required": true
    }
  },
  "required": ["index_id"]
}
```

#### Response Format

```json
{
  "ok": true,
  "message": "Pipeline cancelled"
}
```

#### Example Usage

```python
await mcp_client.call_tool("cancel", {
    "index_id": "01JA2B3C4D5E6F7G8H"
})
```

## Resources

Resources provide access to index artifacts and status information.

### Resource URI Patterns

- **Global Status**: `repo://status`
- **Templates**: `repo://manifest/{index_id}`, `repo://logs/{index_id}`, `repo://bundle/{index_id}`
- **Specific Resources**: `mimir://indexes/{index_id}/{resource}`

### 1. Status Resources

#### Global Status (`repo://status`)

```json
{
  "server_info": {
    "name": "mimir-repoindex",
    "version": "0.1.0", 
    "uptime": 1642684800.0
  },
  "active_pipelines": ["01JA2B3C4D5E6F7G8H", "01JA2B3C4D5E6F7G8I"],
  "pipeline_count": 2,
  "storage_dir": "/home/user/.cache/mimir"
}
```

#### Pipeline Status (`mimir://indexes/{index_id}/status.json`)

```json
{
  "index_id": "01JA2B3C4D5E6F7G8H",
  "state": "running",
  "stage": "leann",
  "progress": 65,
  "started_at": "2025-01-19T10:30:00Z",
  "estimated_completion": "2025-01-19T10:45:00Z",
  "current_operation": "Generating vector embeddings for src/components/",
  "error": null
}
```

#### Status States

- `queued` - Waiting to start
- `running` - Active processing 
- `completed` - Successfully finished
- `failed` - Error occurred
- `cancelled` - User cancelled

### 2. Manifest Resources

#### Manifest (`mimir://indexes/{index_id}/manifest.json`)

```json
{
  "index_id": "01JA2B3C4D5E6F7G8H",
  "repo": {
    "root": "/path/to/project",
    "rev": "a1b2c3d4e5f6789",
    "worktree_dirty": false
  },
  "config": {
    "languages": ["ts", "tsx", "js"],
    "excludes": ["node_modules/", "dist/"],
    "context_lines": 5,
    "features": {
      "vector": true,
      "symbol": true,
      "graph": true
    }
  },
  "counts": {
    "files_total": 247,
    "files_indexed": 198, 
    "symbols_defs": 1456,
    "symbols_refs": 3247,
    "vectors": 2847
  },
  "paths": {
    "repomap": "repomap.json",
    "serena_graph": "serena_graph.jsonl",
    "leann_index": "leann.index",
    "vectors": "vectors.bin",
    "snippets": "snippets.jsonl",
    "bundle": "bundle.tar.zst"
  },
  "versions": {
    "repomapper": "1.2.3",
    "serena": "2.1.0", 
    "leann": "0.3.1"
  }
}
```

### 3. Log Resources

#### Pipeline Log (`mimir://indexes/{index_id}/log.md`)

```markdown
# Pipeline Log for 01JA2B3C4D5E6F7G8H

## 2025-01-19 10:30:15 - Pipeline Started
- Repository: /path/to/project
- Revision: a1b2c3d4e5f6789
- Language: TypeScript

## 2025-01-19 10:30:16 - Stage: Acquire
- Discovered 247 tracked files
- Detected 12 uncommitted changes
- Generated overlay for dirty worktree

## 2025-01-19 10:32:23 - Stage: RepoMapper  
- Analyzed 198 source files
- Generated dependency graph with 1,247 edges
- Computed PageRank scores

## 2025-01-19 10:35:41 - Stage: Serena
- Found 1,456 symbol definitions
- Resolved 3,247 symbol references
- Traced 892 function calls

## 2025-01-19 10:38:12 - Stage: LEANN
- Generated embeddings for 2,847 code chunks
- Built vector index with 512 dimensions
- Saved 22.4MB vector index

## 2025-01-19 10:41:55 - Stage: Snippets
- Extracted 2,135 code snippets
- Added context lines (5 before/after)
- Total snippet data: 8.7MB

## 2025-01-19 10:43:17 - Stage: Bundle
- Compressed all artifacts to bundle.tar.zst
- Bundle size: 15.2MB (67% compression)
- Pipeline completed successfully
```

### 4. Bundle Resources

#### Bundle Artifact (`mimir://indexes/{index_id}/bundle.tar.zst`)

Binary Zstandard-compressed tar archive containing all index artifacts. Use the MCP blob resource type to access:

```python
# Read bundle as binary data
bundle_resource = await mcp_client.read_resource(
    "mimir://indexes/01JA2B3C4D5E6F7G8H/bundle.tar.zst"
)
bundle_bytes = bundle_resource.contents[0].blob
```

## Error Handling

### Common Error Codes

#### Tool Execution Errors

```json
{
  "content": [
    {
      "type": "text",
      "text": "Error: Repository not found at path /invalid/path"
    }
  ],
  "isError": true
}
```

#### Resource Access Errors

```json
{
  "contents": [
    {
      "type": "text",
      "text": "Error reading resource: Bundle not found for index 01JA2B3C4D5E6F7G8H",
      "mimeType": "text/plain",
      "uri": "mimir://indexes/01JA2B3C4D5E6F7G8H/bundle.tar.zst"
    }
  ]
}
```

### Error Categories

- **Validation Errors**: Invalid input parameters
- **File System Errors**: Missing repositories or artifacts  
- **Pipeline Errors**: Stage execution failures
- **Resource Errors**: Missing or corrupted index data
- **System Errors**: Insufficient memory, disk space, or permissions

## Performance Characteristics

### Indexing Performance

| Repository Size | Files | Time | Memory | Bundle Size |
|----------------|-------|------|---------|-------------|
| Small (< 100 files) | 50 | 30s | 512MB | 2-5MB |
| Medium (100-500 files) | 250 | 2-5min | 1GB | 10-25MB |
| Large (500-2000 files) | 1000 | 10-20min | 2GB | 50-150MB |
| Very Large (> 2000 files) | 5000+ | 30min+ | 4GB+ | 200MB+ |

### Search Performance

- **Vector Search**: 50-200ms for 10K chunks
- **Symbol Search**: < 1ms for exact matches
- **Hybrid Search**: 100-300ms combining all modalities
- **Ask Index**: 200-500ms for multi-hop reasoning

### Optimization Tips

1. **Exclude Large Directories**: Add `node_modules/`, `vendor/`, `dist/` to excludes
2. **Limit File Count**: Use `max_files_to_embed` for large repositories
3. **Disable Unused Features**: Turn off vector, symbol, or graph features if not needed
4. **Use Specific Languages**: Specify exact file extensions instead of broad language categories

## Security Considerations

### Input Validation

All tool inputs are validated against JSON schemas with strict type checking and range validation.

### Path Security

- All file paths are validated to prevent directory traversal
- Only files within the specified repository root are accessible
- Symbolic links are resolved and validated

### Resource Limits

- Maximum file size: 100MB per file
- Memory limits: Configurable per-process limits
- Timeout enforcement: 5-minute default timeout per pipeline stage

### Sandboxing

External tools (RepoMapper, Serena, LEANN) run in isolated processes with restricted capabilities.

## Implementation Examples

### Python Client

```python
import asyncio
from mcp_client import MCPClient

async def index_and_search():
    client = MCPClient("python -m repoindex.main mcp")
    
    # Start indexing
    index_result = await client.call_tool("ensure_repo_index", {
        "path": "/path/to/project",
        "language": "ts"
    })
    
    index_id = index_result["index_id"]
    
    # Wait for completion (poll status)
    while True:
        status = await client.read_resource(
            f"mimir://indexes/{index_id}/status.json"
        )
        status_data = json.loads(status.contents[0].text)
        
        if status_data["state"] in ["completed", "failed"]:
            break
            
        await asyncio.sleep(5)
    
    # Search the index
    search_result = await client.call_tool("search_repo", {
        "index_id": index_id,
        "query": "authentication middleware",
        "k": 10
    })
    
    print(f"Found {len(search_result['results'])} results")
    
    # Ask questions
    answer = await client.call_tool("ask_index", {
        "index_id": index_id,
        "question": "How does user login work?"
    })
    
    print(f"Answer based on {len(answer['evidence'])} pieces of evidence")

if __name__ == "__main__":
    asyncio.run(index_and_search())
```

### TypeScript Client

```typescript
import { MCPClient } from '@mcp/client';

async function indexRepository() {
  const client = new MCPClient({
    command: 'python',
    args: ['-m', 'repoindex.main', 'mcp']
  });
  
  await client.connect();
  
  try {
    // Index repository
    const indexResult = await client.callTool('ensure_repo_index', {
      path: '/path/to/project',
      language: 'ts',
      index_opts: {
        context_lines: 3,
        excludes: ['node_modules/', 'dist/']
      }
    });
    
    const indexId = indexResult.index_id;
    
    // Search
    const searchResult = await client.callTool('search_repo', {
      index_id: indexId,
      query: 'error handling',
      k: 5,
      features: {
        vector: true,
        symbol: true,
        graph: false
      }
    });
    
    console.log(`Found ${searchResult.results.length} results`);
    
    // Get complete bundle
    const bundleResult = await client.callTool('get_repo_bundle', {
      index_id: indexId
    });
    
    const bundleData = await client.readResource(bundleResult.bundle_uri);
    console.log(`Bundle size: ${bundleData.contents[0].blob.length} bytes`);
    
  } finally {
    await client.disconnect();
  }
}
```

## Troubleshooting

### Common Issues

#### Pipeline Fails During RepoMapper Stage
- **Cause**: Missing Tree-sitter dependencies
- **Solution**: Install language-specific Tree-sitter parsers
- **Check**: `tree-sitter --version` and language support

#### Out of Memory During Vector Generation
- **Cause**: Large repository with too many files
- **Solution**: Use `max_files_to_embed` parameter or exclude large directories
- **Monitor**: Pipeline memory usage in status updates

#### Search Returns No Results
- **Cause**: Index may be incomplete or corrupted
- **Solution**: Check manifest counts and re-run indexing
- **Debug**: Review pipeline logs for stage failures

#### Slow Search Performance
- **Cause**: Large vector index or inefficient queries  
- **Solution**: Enable caching, use specific search features
- **Optimize**: Disable unused search modalities

### Diagnostic Commands

```bash
# Check server status
echo '{"jsonrpc":"2.0","method":"resources/list","id":1}' | python -m repoindex.main mcp

# Validate index integrity
python -c "
import json
from pathlib import Path
manifest = json.loads(Path('~/.cache/mimir/indexes/{index_id}/manifest.json').read_text())
print(f'Files: {manifest[\"counts\"][\"files_indexed\"]}')
print(f'Symbols: {manifest[\"counts\"][\"symbols_defs\"]}')
print(f'Vectors: {manifest[\"counts\"][\"vectors\"]}')
"

# Check bundle integrity
python -c "
import tarfile
with tarfile.open('~/.cache/mimir/indexes/{index_id}/bundle.tar.zst', 'r:*') as tar:
    print('Bundle contents:', tar.getnames())
"
```

This API reference provides comprehensive documentation for integrating with Mimir through the MCP protocol. For additional examples and advanced usage patterns, see the main README and development documentation.