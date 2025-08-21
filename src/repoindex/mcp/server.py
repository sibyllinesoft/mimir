"""
Main MCP server implementation for repository indexing.

Provides stdio-based MCP protocol support with tools for repository indexing,
search, and bundle management. Coordinates with the pipeline orchestration
system to provide real-time status updates via MCP resources.
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ReadResourceRequest,
    ReadResourceResult,
    ListResourcesRequest,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    Resource,
    Tool,
    TextContent,
    BlobResourceContents,
    TextResourceContents,
)
from pydantic import ValidationError

from ..data.schemas import (
    EnsureRepoIndexRequest,
    EnsureRepoIndexResponse,
    GetRepoBundleRequest,
    GetRepoBundleResponse,
    SearchRepoRequest,
    SearchResponse,
    AskIndexRequest,
    AskResponse,
    CancelRequest,
    CancelResponse,
    IndexManifest,
    PipelineStatus,
)
from ..monitoring import (
    get_metrics_collector,
    get_trace_manager,
    server_metrics
)
from ..pipeline.run import IndexingPipeline
from ..util.fs import get_index_directory, ensure_directory
from ..util.log import setup_logging


logger = logging.getLogger(__name__)


class MCPServer:
    """
    MCP server for repository indexing operations.
    
    Manages multiple concurrent indexing operations and provides
    real-time status updates through MCP resources.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize MCP server with optional storage directory."""
        self.storage_dir = storage_dir or Path.home() / ".cache" / "mimir"
        self.indexes_dir = self.storage_dir / "indexes"
        self.pipelines: Dict[str, IndexingPipeline] = {}
        self.server = Server("mimir-repoindex")
        
        # Initialize monitoring
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()
        self.start_time = time.time()
        
        # Ensure storage directories exist
        ensure_directory(self.indexes_dir)
        
        # Register MCP handlers
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self) -> None:
        """Register all MCP tools."""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available indexing tools."""
            return [
                Tool(
                    name="ensure_repo_index",
                    description="Ensure repository is indexed with full pipeline execution",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to repository root"
                            },
                            "rev": {
                                "type": "string",
                                "description": "Git revision (defaults to HEAD)"
                            },
                            "language": {
                                "type": "string", 
                                "description": "Primary language (defaults to 'ts')",
                                "default": "ts"
                            },
                            "index_opts": {
                                "type": "object",
                                "description": "Additional indexing options"
                            }
                        },
                        "required": ["path"]
                    }
                ),
                Tool(
                    name="get_repo_bundle",
                    description="Retrieve complete index bundle for a repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {
                                "type": "string",
                                "description": "Index identifier"
                            }
                        },
                        "required": ["index_id"]
                    }
                ),
                Tool(
                    name="search_repo",
                    description="Search repository using hybrid vector + symbol + graph approach",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {
                                "type": "string",
                                "description": "Index identifier"
                            },
                            "query": {
                                "type": "string",
                                "description": "Search query"
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
                                "description": "Feature flags for search modalities",
                                "properties": {
                                    "vector": {"type": "boolean", "default": True},
                                    "symbol": {"type": "boolean", "default": True},
                                    "graph": {"type": "boolean", "default": True}
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
                ),
                Tool(
                    name="ask_index",
                    description="Ask complex questions using multi-hop symbol graph reasoning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {
                                "type": "string",
                                "description": "Index identifier"
                            },
                            "question": {
                                "type": "string",
                                "description": "Question to ask about the codebase"
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
                ),
                Tool(
                    name="cancel",
                    description="Cancel an ongoing indexing operation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {
                                "type": "string",
                                "description": "Index identifier to cancel"
                            }
                        },
                        "required": ["index_id"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
            """Handle tool calls."""
            try:
                if name == "ensure_repo_index":
                    return await self._ensure_repo_index(arguments)
                elif name == "get_repo_bundle":
                    return await self._get_repo_bundle(arguments)
                elif name == "search_repo":
                    return await self._search_repo(arguments)
                elif name == "ask_index":
                    return await self._ask_index(arguments)
                elif name == "cancel":
                    return await self._cancel(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.exception(f"Error in tool {name}: {e}")
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Error: {str(e)}"
                        )
                    ],
                    isError=True
                )
    
    def _register_resources(self) -> None:
        """Register MCP resources for status and artifacts."""
        
        @self.server.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources."""
            resources = []
            
            # Add global status resource
            resources.append(
                Resource(
                    uri="repo://status",
                    name="Global Status",
                    description="Overall system status and active pipelines",
                    mimeType="application/json"
                )
            )
            
            # Add template resources for dynamic index access
            resources.extend([
                Resource(
                    uri="repo://manifest/{index_id}",
                    name="Index Manifest Template",
                    description="Manifest for specific index",
                    mimeType="application/json"
                ),
                Resource(
                    uri="repo://logs/{index_id}",
                    name="Index Logs Template", 
                    description="Logs for specific index",
                    mimeType="text/plain"
                ),
                Resource(
                    uri="repo://bundle/{index_id}",
                    name="Index Bundle Template",
                    description="Bundle for specific index",
                    mimeType="application/zstd"
                )
            ])
            
            # Add resources for each active index
            for index_id in self.pipelines.keys():
                base_uri = f"mimir://indexes/{index_id}"
                resources.extend([
                    Resource(
                        uri=f"{base_uri}/status.json",
                        name=f"Status for {index_id}",
                        description="Real-time pipeline status",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri=f"{base_uri}/manifest.json",
                        name=f"Manifest for {index_id}",
                        description="Complete index metadata",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri=f"{base_uri}/log.md",
                        name=f"Log for {index_id}",
                        description="Human-readable progress log",
                        mimeType="text/markdown"
                    ),
                    Resource(
                        uri=f"{base_uri}/bundle.tar.zst",
                        name=f"Bundle for {index_id}",
                        description="Compressed artifact bundle",
                        mimeType="application/zstd"
                    )
                ])
            
            return resources
        
        @self.server.read_resource()
        async def read_resource(uri: str) -> ReadResourceResult:
            """Read a specific resource."""
            try:
                # Handle repo:// URIs (for global/template resources)
                if uri.startswith("repo://"):
                    return await self._read_repo_resource(uri)
                
                # Handle mimir:// URIs (for specific index resources)
                elif uri.startswith("mimir://"):
                    return await self._read_mimir_resource(uri)
                
                else:
                    raise ValueError(f"Unsupported URI scheme: {uri}")
                    
            except Exception as e:
                logger.exception(f"Error reading resource {uri}: {e}")
                return ReadResourceResult(
                    contents=[
                        TextResourceContents(
                            type="text",
                            text=f"Error reading resource: {str(e)}",
                            mimeType="text/plain",
                            uri=uri
                        )
                    ]
                )
        
        # Store references for testing
        self._list_resources_handler = list_resources
        self._read_resource_handler = read_resource
    
    async def _list_resources(self) -> List[Resource]:
        """Wrapper for testing the list_resources handler."""
        return await self._list_resources_handler()
    
    async def _read_resource(self, uri: str) -> ReadResourceResult:
        """Wrapper for testing the read_resource handler."""
        return await self._read_resource_handler(uri)
    
    async def _read_repo_resource(self, uri: str) -> ReadResourceResult:
        """Handle repo:// URI resources."""
        path = uri.replace("repo://", "")
        
        if path == "status":
            # Global status resource
            status_data = {
                "server_info": {
                    "name": "mimir-repoindex",
                    "version": "0.1.0",
                    "uptime": time.time()
                },
                "active_pipelines": list(self.pipelines.keys()),
                "pipeline_count": len(self.pipelines),
                "storage_dir": str(self.storage_dir)
            }
            
            return ReadResourceResult(
                contents=[
                    TextResourceContents(
                        type="text",
                        text=json.dumps(status_data, indent=2),
                        mimeType="application/json",
                        uri=uri
                    )
                ]
            )
        
        elif path.startswith("manifest/"):
            # Extract index_id from path like "manifest/index_123"
            index_id = path.replace("manifest/", "")
            index_dir = get_index_directory(self.indexes_dir, index_id)
            return await self._get_manifest_resource(index_id, index_dir)
        
        elif path.startswith("logs/"):
            # Extract index_id from path like "logs/index_123"
            index_id = path.replace("logs/", "")
            index_dir = get_index_directory(self.indexes_dir, index_id)
            return await self._get_log_resource(index_id, index_dir)
        
        elif path.startswith("bundle/"):
            # Extract index_id from path like "bundle/index_123"
            index_id = path.replace("bundle/", "")
            index_dir = get_index_directory(self.indexes_dir, index_id)
            return await self._get_bundle_resource(index_id, index_dir)
        
        else:
            raise ValueError(f"Unknown repo resource: {path}")
    
    async def _read_mimir_resource(self, uri: str) -> ReadResourceResult:
        """Handle mimir:// URI resources."""
        # Parse URI: mimir://indexes/{index_id}/{resource}
        parts = uri.replace("mimir://", "").split("/")
        if len(parts) != 3 or parts[0] != "indexes":
            raise ValueError(f"Invalid mimir URI: {uri}")
        
        index_id, resource_name = parts[1], parts[2]
        index_dir = get_index_directory(self.indexes_dir, index_id)
        
        if resource_name == "status.json":
            return await self._get_status_resource(index_id, index_dir)
        elif resource_name == "manifest.json":
            return await self._get_manifest_resource(index_id, index_dir)
        elif resource_name == "log.md":
            return await self._get_log_resource(index_id, index_dir)
        elif resource_name == "bundle.tar.zst":
            return await self._get_bundle_resource(index_id, index_dir)
        else:
            raise ValueError(f"Unknown mimir resource: {resource_name}")
    
    async def _ensure_repo_index(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle ensure_repo_index tool call."""
        try:
            request = EnsureRepoIndexRequest(**arguments)
            
            # Create and start indexing pipeline
            pipeline = IndexingPipeline(storage_dir=self.indexes_dir)
            index_id = await pipeline.start_indexing(
                repo_path=request.path,
                rev=request.rev,
                language=request.language,
                index_opts=request.index_opts or {}
            )
            
            # Store pipeline reference
            self.pipelines[index_id] = pipeline
            
            # Construct resource URIs
            status_uri = f"mimir://indexes/{index_id}/status.json"
            manifest_uri = f"mimir://indexes/{index_id}/manifest.json"
            
            response = EnsureRepoIndexResponse(
                index_id=index_id,
                status_uri=status_uri,
                manifest_uri=manifest_uri
            )
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(response.model_dump(), indent=2)
                    )
                ]
            )
            
        except ValidationError as e:
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Validation error: {e}"
                    )
                ],
                isError=True
            )
        except Exception as e:
            logger.exception(f"Error in ensure_repo_index: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Error: {str(e)}"
                    )
                ],
                isError=True
            )
    
    async def _get_repo_bundle(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle get_repo_bundle tool call."""
        try:
            request = GetRepoBundleRequest(**arguments)
            
            # Check if bundle exists
            index_dir = get_index_directory(self.indexes_dir, request.index_id)
            bundle_path = index_dir / "bundle.tar.zst"
            manifest_path = index_dir / "manifest.json"
            
            if not bundle_path.exists():
                raise FileNotFoundError(f"Bundle not found for index {request.index_id}")
            
            bundle_uri = f"mimir://indexes/{request.index_id}/bundle.tar.zst"
            manifest_uri = f"mimir://indexes/{request.index_id}/manifest.json"
            
            response = GetRepoBundleResponse(
                bundle_uri=bundle_uri,
                manifest_uri=manifest_uri
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
    
    async def _search_repo(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle search_repo tool call."""
        try:
            request = SearchRepoRequest(**arguments)
            
            # Get pipeline for this index
            pipeline = self.pipelines.get(request.index_id)
            if not pipeline:
                raise ValueError(f"No active pipeline for index {request.index_id}")
            
            # Execute search
            response = await pipeline.search(
                query=request.query,
                k=request.k,
                features=request.features,
                context_lines=request.context_lines
            )
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(response.model_dump(), indent=2, default=str)
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
    
    async def _ask_index(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle ask_index tool call."""
        try:
            request = AskIndexRequest(**arguments)
            
            # Get pipeline for this index
            pipeline = self.pipelines.get(request.index_id)
            if not pipeline:
                raise ValueError(f"No active pipeline for index {request.index_id}")
            
            # Execute ask operation
            response = await pipeline.ask(
                question=request.question,
                context_lines=request.context_lines
            )
            
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(response.model_dump(), indent=2, default=str)
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
    
    async def _cancel(self, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle cancel tool call."""
        try:
            request = CancelRequest(**arguments)
            
            # Get and cancel pipeline
            pipeline = self.pipelines.get(request.index_id)
            if not pipeline:
                response = CancelResponse(ok=False, message="Pipeline not found")
            else:
                await pipeline.cancel()
                del self.pipelines[request.index_id]
                response = CancelResponse(ok=True, message="Pipeline cancelled")
            
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
    
    async def _get_status_resource(self, index_id: str, index_dir: Path) -> ReadResourceResult:
        """Get pipeline status resource."""
        status_file = index_dir / "status.json"
        if status_file.exists():
            content = status_file.read_text()
        else:
            # Return default status if file doesn't exist yet
            status = PipelineStatus(index_id=index_id, state="queued", progress=0)
            content = json.dumps(status.model_dump(), indent=2, default=str)
        
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    type="text",
                    text=content,
                    mimeType="application/json",
                    uri=f"mimir://indexes/{index_id}/status.json"
                )
            ]
        )
    
    async def _get_manifest_resource(self, index_id: str, index_dir: Path) -> ReadResourceResult:
        """Get index manifest resource."""
        manifest_file = index_dir / "manifest.json"
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found for index {index_id}")
        
        content = manifest_file.read_text()
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    type="text",
                    text=content,
                    mimeType="application/json",
                    uri=f"mimir://indexes/{index_id}/manifest.json"
                )
            ]
        )
    
    async def _get_log_resource(self, index_id: str, index_dir: Path) -> ReadResourceResult:
        """Get pipeline log resource."""
        log_file = index_dir / "log.md"
        if log_file.exists():
            content = log_file.read_text()
        else:
            content = f"# Pipeline Log for {index_id}\n\nNo log entries yet.\n"
        
        return ReadResourceResult(
            contents=[
                TextResourceContents(
                    type="text",
                    text=content,
                    mimeType="text/markdown",
                    uri=f"mimir://indexes/{index_id}/log.md"
                )
            ]
        )
    
    async def _get_bundle_resource(self, index_id: str, index_dir: Path) -> ReadResourceResult:
        """Get bundle artifact resource."""
        bundle_file = index_dir / "bundle.tar.zst"
        if not bundle_file.exists():
            raise FileNotFoundError(f"Bundle not found for index {index_id}")
        
        blob_data = bundle_file.read_bytes()
        return ReadResourceResult(
            contents=[
                BlobResourceContents(
                    type="blob",
                    blob=blob_data,
                    mimeType="application/zstd",
                    uri=f"mimir://indexes/{index_id}/bundle.tar.zst"
                )
            ]
        )

# Alias for backward compatibility with tests
MimirMCPServer = MCPServer


async def main() -> None:
    """Main entry point for MCP server."""
    setup_logging()
    
    # Create server instance
    mcp_server = MCPServer()
    
    # Run stdio server
    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.server.run(
            read_stream, 
            write_stream,
            InitializationOptions(
                server_name="mimir-repoindex",
                server_version="0.1.0",
                capabilities=mcp_server.server.get_capabilities()
            )
        )


if __name__ == "__main__":
    asyncio.run(main())