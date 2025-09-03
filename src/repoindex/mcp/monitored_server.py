"""
Skald-monitored MCP server implementation for repository indexing.

This module provides a Skald-wrapped version of the MCP server that:
1. Monitors all tool executions with detailed traces
2. Emits agent traces to NATS JetStream for real-time monitoring
3. Provides deep insights into Mimir's research capabilities
4. Maintains full compatibility with the original server interface

The monitoring includes:
- Tool execution timing and performance
- Deep research pipeline traces  
- Agent coordination patterns
- Search query analysis
- Error tracking and diagnostics
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional
import traceback
import uuid

# Skald imports for monitoring
from skald import SurveyingProxy, FeedbackReport, ToolRunMetadata
from skald.schema import ExecutionContext, FeedbackStrength
from skald.monitor import monitor, trace_function

# NATS for trace streaming
try:
    import nats
    from nats.js import JetStreamContext
    HAS_NATS = True
except ImportError:
    HAS_NATS = False

from mcp.server import Server, NotificationOptions, stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import (
    BlobResourceContents,
    ReadResourceResult,
    Resource,
    TextResourceContents,
    Tool,
    TextContent,
)

# Import the base server components
from .server import MCPServer
from ..data.schemas import (
    AskIndexRequest,
    CancelRequest, 
    EnsureRepoIndexRequest,
    GetRepoBundleRequest,
    SearchRepoRequest,
)
from ..util.log import setup_logging

logger = logging.getLogger(__name__)


class NATSTraceEmitter:
    """NATS JetStream emitter for agent traces."""
    
    def __init__(self, nats_url: str = "nats://localhost:4222", stream_name: str = "MIMIR_TRACES"):
        self.nats_url = nats_url
        self.stream_name = stream_name
        self.nc: Optional[nats.NATS] = None
        self.js: Optional[JetStreamContext] = None
        self.connected = False
        
    async def connect(self):
        """Connect to NATS and setup JetStream."""
        if not HAS_NATS:
            logger.warning("NATS not available, trace emission disabled")
            return
            
        try:
            self.nc = await nats.connect(self.nats_url)
            self.js = self.nc.jetstream()
            
            # Create or update the stream
            try:
                await self.js.add_stream(
                    name=self.stream_name,
                    subjects=[f"{self.stream_name.lower()}.*"],
                    max_msgs=100000,  # Keep last 100k traces
                    max_age=7 * 24 * 3600,  # Keep traces for 7 days
                )
            except Exception as e:
                if "stream name already in use" not in str(e):
                    logger.warning(f"Stream setup error (may already exist): {e}")
            
            self.connected = True
            logger.info(f"Connected to NATS at {self.nats_url}, stream: {self.stream_name}")
            
        except Exception as e:
            logger.error(f"Failed to connect to NATS: {e}")
            self.connected = False
    
    async def emit_trace(self, trace_data: Dict[str, Any]):
        """Emit a trace event to NATS JetStream."""
        if not self.connected or not self.js:
            return
            
        try:
            subject = f"{self.stream_name.lower()}.{trace_data.get('tool_name', 'unknown')}"
            payload = json.dumps(trace_data, default=str).encode()
            
            await self.js.publish(subject, payload)
            
        except Exception as e:
            logger.error(f"Failed to emit trace to NATS: {e}")
    
    async def close(self):
        """Close NATS connection."""
        if self.nc:
            await self.nc.close()
            self.connected = False


class MonitoredMCPServer(MCPServer):
    """
    Skald-monitored version of the MCP server.
    
    Wraps all tool executions with comprehensive monitoring and trace emission
    to NATS JetStream for real-time visibility into Mimir's deep research capabilities.
    """
    
    def __init__(self, 
                 storage_dir: Path | None = None, 
                 query_engine=None,
                 nats_url: str = "nats://localhost:4222",
                 enable_nats_traces: bool = True,
                 feedback_strength: str = "FREQUENT"):
        """Initialize monitored MCP server."""
        # Initialize base server
        super().__init__(storage_dir, query_engine)
        
        # Set up monitoring with high reporting
        self.trace_emitter = NATSTraceEmitter(nats_url) if enable_nats_traces else None
        self.session_id = str(uuid.uuid4())
        self.feedback_strength = getattr(FeedbackStrength, feedback_strength, FeedbackStrength.FREQUENT)
        
        # Enhanced monitoring state
        self.tool_call_count = 0
        self.research_sessions = {}  # Track deep research sessions
        self.active_traces = {}  # Track ongoing operations
        
        # Enable detailed Skald monitoring
        self.monitoring_enabled = True
        
        logger.info(f"Monitored MCP server initialized with session_id: {self.session_id}")
        logger.info(f"Skald monitoring enabled with {feedback_strength} reporting level")
    
    def _create_feedback_report(self, tool_name: str, execution_time: float, trace_data: dict) -> FeedbackReport:
        """Create detailed feedback report for Skald monitoring."""
        return FeedbackReport(
            strength=self.feedback_strength,
            metadata={
                "tool_name": tool_name,
                "execution_time": execution_time,
                "session_id": self.session_id,
                "call_sequence": self.tool_call_count,
                "trace_data": trace_data,
                "timestamp": time.time(),
                "monitoring_level": "HIGH"
            }
        )
    
    async def start_monitoring(self):
        """Initialize monitoring connections."""
        if self.trace_emitter:
            await self.trace_emitter.connect()
    
    async def stop_monitoring(self):
        """Clean up monitoring connections."""
        if self.trace_emitter:
            await self.trace_emitter.close()
    
    def _register_tools(self) -> None:
        """Register monitored versions of all MCP tools."""
        tools = self._get_tool_definitions()
        
        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available indexing tools.""" 
            return tools

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]):
            """Handle monitored tool calls."""
            return await self._monitored_tool_call(name, arguments)
    
    @trace_function(name="mcp_tool_execution", invite_feedback=True)
    async def _monitored_tool_call(self, name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """Execute tool call with comprehensive monitoring."""
        self.tool_call_count += 1
        call_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Create trace context
        trace_data = {
            "call_id": call_id,
            "session_id": self.session_id,
            "tool_name": name,
            "arguments": arguments,
            "timestamp": start_time,
            "call_sequence": self.tool_call_count,
            "trace_type": "tool_execution_start"
        }
        
        try:
            # Emit start trace
            if self.trace_emitter:
                await self.trace_emitter.emit_trace(trace_data)
            
            # Execute the actual tool with monitoring
            result = await self._execute_monitored_tool(name, arguments, trace_data)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update trace with success
            trace_data.update({
                "trace_type": "tool_execution_complete",
                "execution_time": execution_time,
                "success": True,
                "result_length": len(str(result)) if result else 0,
                "completion_timestamp": time.time()
            })
            
            # Emit completion trace
            if self.trace_emitter:
                await self.trace_emitter.emit_trace(trace_data)
            
            # Create and log high-detail feedback report
            feedback_report = self._create_feedback_report(name, execution_time, trace_data)
            logger.info(f"Tool {name} executed successfully in {execution_time:.2f}s")
            logger.debug(f"High-detail trace: {trace_data}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update trace with error
            trace_data.update({
                "trace_type": "tool_execution_error",
                "execution_time": execution_time,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc(),
                "completion_timestamp": time.time()
            })
            
            # Emit error trace
            if self.trace_emitter:
                await self.trace_emitter.emit_trace(trace_data)
            
            # Create and log error feedback report
            feedback_report = self._create_feedback_report(name, execution_time, trace_data)
            logger.error(f"Tool {name} failed after {execution_time:.2f}s: {e}")
            logger.debug(f"High-detail error trace: {trace_data}")
            
            # Return error as TextContent
            return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _execute_monitored_tool(self, name: str, arguments: dict[str, Any], trace_data: dict) -> list[TextContent]:
        """Execute individual tool with specific monitoring patterns."""
        
        if name == "ensure_repo_index":
            return await self._monitored_ensure_repo_index(arguments, trace_data)
        elif name == "search_repo":
            return await self._monitored_search_repo(arguments, trace_data)
        elif name == "ask_index":
            return await self._monitored_ask_index(arguments, trace_data)
        elif name == "get_repo_bundle":
            return await self._monitored_get_repo_bundle(arguments, trace_data)
        elif name == "cancel":
            return await self._monitored_cancel(arguments, trace_data)
        else:
            raise ValueError(f"Unknown tool: {name}")
    
    async def _monitored_ensure_repo_index(self, arguments: dict[str, Any], trace_data: dict) -> list[TextContent]:
        """Monitor repository indexing with detailed pipeline traces."""
        request = EnsureRepoIndexRequest(**arguments)
        
        # Enhanced trace data for indexing
        trace_data.update({
            "operation": "repo_indexing",
            "repo_path": request.path,
            "revision": request.rev,
            "language": request.language,
            "index_options": request.index_opts
        })
        
        # Execute with pipeline monitoring
        result = await self._ensure_repo_index(arguments)
        
        # Parse the result to extract index_id for session tracking
        try:
            result_data = json.loads(result)
            index_id = result_data.get("index_id")
            if index_id:
                self.research_sessions[index_id] = {
                    "created_at": time.time(),
                    "repo_path": request.path,
                    "language": request.language,
                    "operations": []
                }
                trace_data["index_id"] = index_id
        except Exception:
            pass  # Continue even if parsing fails
        
        return [TextContent(type="text", text=result)]
    
    async def _monitored_search_repo(self, arguments: dict[str, Any], trace_data: dict) -> list[TextContent]:
        """Monitor repository search with query analysis."""
        request = SearchRepoRequest(**arguments)
        
        # Enhanced trace data for search
        trace_data.update({
            "operation": "repo_search",
            "index_id": request.index_id,
            "query": request.query,
            "k": request.k,
            "features": request.features,
            "context_lines": request.context_lines
        })
        
        # Track search in session
        if request.index_id in self.research_sessions:
            self.research_sessions[request.index_id]["operations"].append({
                "type": "search",
                "query": request.query,
                "timestamp": time.time(),
                "features": request.features
            })
        
        result = await self._search_repo(arguments)
        
        # Analyze result for trace enhancement
        try:
            result_data = json.loads(result)
            results = result_data.get("results", [])
            trace_data.update({
                "results_count": len(results),
                "result_types": list(set(r.get("type", "unknown") for r in results)),
                "has_results": len(results) > 0
            })
        except Exception:
            pass
        
        return [TextContent(type="text", text=result)]
    
    async def _monitored_ask_index(self, arguments: dict[str, Any], trace_data: dict) -> list[TextContent]:
        """Monitor deep research questions with reasoning traces."""
        request = AskIndexRequest(**arguments)
        
        # Enhanced trace data for deep research
        trace_data.update({
            "operation": "deep_research", 
            "index_id": request.index_id,
            "question": request.question,
            "context_lines": request.context_lines,
            "question_length": len(request.question),
            "question_complexity": self._analyze_question_complexity(request.question)
        })
        
        # Track research in session
        if request.index_id in self.research_sessions:
            self.research_sessions[request.index_id]["operations"].append({
                "type": "deep_research",
                "question": request.question,
                "timestamp": time.time(),
                "complexity": self._analyze_question_complexity(request.question)
            })
        
        result = await self._ask_index(arguments)
        
        # Analyze research result for insights
        try:
            result_data = json.loads(result)
            answer = result_data.get("answer", "")
            evidence = result_data.get("evidence", [])
            trace_data.update({
                "answer_length": len(answer),
                "evidence_count": len(evidence),
                "research_depth": len(evidence),
                "answer_confidence": self._analyze_answer_confidence(result_data)
            })
        except Exception:
            pass
        
        return [TextContent(type="text", text=result)]
    
    async def _monitored_get_repo_bundle(self, arguments: dict[str, Any], trace_data: dict) -> list[TextContent]:
        """Monitor bundle retrieval."""
        request = GetRepoBundleRequest(**arguments)
        
        trace_data.update({
            "operation": "bundle_retrieval",
            "index_id": request.index_id
        })
        
        result = await self._get_repo_bundle(arguments)
        return [TextContent(type="text", text=result)]
    
    async def _monitored_cancel(self, arguments: dict[str, Any], trace_data: dict) -> list[TextContent]:
        """Monitor cancellation operations."""
        request = CancelRequest(**arguments)
        
        trace_data.update({
            "operation": "pipeline_cancel",
            "index_id": request.index_id
        })
        
        result = await self._cancel(arguments)
        return [TextContent(type="text", text=result)]
    
    def _analyze_question_complexity(self, question: str) -> str:
        """Analyze research question complexity."""
        question_lower = question.lower()
        
        # Simple heuristics for complexity analysis
        complexity_indicators = {
            "how": 1,
            "why": 2,
            "what": 1,
            "where": 1,
            "when": 1,
            "explain": 2,
            "analyze": 3,
            "compare": 3,
            "relationship": 3,
            "pattern": 3,
            "architecture": 3,
            "design": 2,
            "implement": 2,
            "optimize": 3
        }
        
        total_complexity = 0
        indicators_found = []
        
        for indicator, weight in complexity_indicators.items():
            if indicator in question_lower:
                total_complexity += weight
                indicators_found.append(indicator)
        
        if total_complexity >= 6:
            return "high"
        elif total_complexity >= 3:
            return "medium"
        else:
            return "low"
    
    def _analyze_answer_confidence(self, result_data: dict) -> str:
        """Analyze confidence level of research answer."""
        evidence_count = len(result_data.get("evidence", []))
        answer_length = len(result_data.get("answer", ""))
        
        if evidence_count >= 5 and answer_length > 500:
            return "high"
        elif evidence_count >= 2 and answer_length > 200:
            return "medium"
        else:
            return "low"
    
    async def get_monitoring_stats(self) -> dict:
        """Get comprehensive monitoring statistics."""
        return {
            "session_id": self.session_id,
            "total_tool_calls": self.tool_call_count,
            "active_research_sessions": len(self.research_sessions),
            "research_sessions": {
                k: {
                    "created_at": v["created_at"],
                    "repo_path": v["repo_path"],
                    "language": v["language"],
                    "operation_count": len(v["operations"]),
                    "last_operation": v["operations"][-1] if v["operations"] else None
                }
                for k, v in self.research_sessions.items()
            },
            "nats_connected": self.trace_emitter.connected if self.trace_emitter else False,
            "monitoring_enabled": True
        }


async def async_main(feedback_level: str = "FREQUENT", disable_nats: bool = False) -> None:
    """Async main entry point for monitored MCP server."""
    setup_logging()
    
    # Create monitored server instance with specified feedback level
    mcp_server = MonitoredMCPServer(
        enable_nats_traces=not disable_nats,
        feedback_strength=feedback_level
    )
    
    # Start monitoring systems
    await mcp_server.start_monitoring()
    
    try:
        # Run stdio server
        # Log monitoring status
        logger.info(f"🔍 Skald monitoring: ENABLED with {feedback_level} feedback level")
        logger.info(f"📊 NATS trace streaming: {'ENABLED' if not disable_nats else 'DISABLED'}")
        logger.info(f"🎯 Session ID: {mcp_server.session_id}")
        
        async with stdio_server() as (read_stream, write_stream):
            await mcp_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mimir-repoindex-monitored",
                    server_version="1.0.0",
                    capabilities=mcp_server.server.get_capabilities(
                        notification_options=NotificationOptions(), 
                        experimental_capabilities={}
                    ),
                ),
            )
    finally:
        # Clean up monitoring
        await mcp_server.stop_monitoring()


def main() -> None:
    """Synchronous main entry point for monitored server."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        prog="mimir-server",
        description="Mimir MCP Server - Deep Research with Comprehensive Skald Monitoring (Default)"
    )
    parser.add_argument(
        "--version",
        action="version", 
        version="mimir-server 1.0.0 (with Skald monitoring)"
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        help="Directory for storing indexes and cache (default: ~/.cache/mimir)"
    )
    parser.add_argument(
        "--nats-url",
        type=str,
        default="nats://localhost:4222",
        help="NATS server URL for trace emission"
    )
    parser.add_argument(
        "--disable-nats",
        action="store_true",
        help="Disable NATS trace emission"
    )
    parser.add_argument(
        "--feedback-level",
        choices=["RARE", "OCCASIONAL", "FREQUENT"],
        default="FREQUENT",
        help="Skald feedback reporting level (default: FREQUENT for high detail)"
    )
    
    args = parser.parse_args()
    
    # Set storage directory if provided
    if args.storage_dir:
        import os
        os.environ['MIMIR_DATA_DIR'] = args.storage_dir
    
    try:
        asyncio.run(async_main(
            feedback_level=args.feedback_level,
            disable_nats=args.disable_nats
        ))
    except KeyboardInterrupt:
        print("Mimir server stopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error starting Mimir server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()