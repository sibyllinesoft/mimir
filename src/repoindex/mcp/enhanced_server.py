"""
Enhanced MCP Server with Hybrid Query Engine Integration.

Extends the base MCP server with intelligent search synthesis,
advanced query processing, and Phase 3 hybrid query capabilities.
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from mcp.server import Server, NotificationOptions, stdio_server
from mcp.server.models import InitializationOptions
from mcp.types import (
    BlobResourceContents,
    ReadResourceResult,
    Resource,
    TextResourceContents,
    Tool,
)
from pydantic import ValidationError

from ..data.schemas import (
    AskIndexRequest,
    FeatureConfig,
    SearchRepoRequest,
)
from ..monitoring import get_metrics_collector, get_trace_manager
from ..pipeline.hybrid_query_engine import (
    HybridQueryEngine,
    QueryContext,
    QueryStrategy,
    QueryType,
)
from ..pipeline.advanced_query_processor import (
    AdvancedQueryProcessor,
    create_query_processor,
)
from ..pipeline.intelligent_ranking import RankingStrategy
from ..util.fs import ensure_directory, get_index_directory
from ..util.log import get_logger
from .server import MCPServer  # Import base server

logger = get_logger(__name__)


class EnhancedMCPServer(MCPServer):
    """
    Enhanced MCP server with Phase 3 hybrid query capabilities.
    
    Features:
    - Intelligent query processing and analysis
    - Multiple search strategies with adaptive selection
    - Advanced result synthesis and ranking
    - Performance optimization and caching
    - Real-time query analytics
    """
    
    def __init__(self, storage_dir: Path | None = None):
        """Initialize enhanced MCP server."""
        # Initialize base server without query engine (we'll create our own)
        super().__init__(storage_dir=storage_dir, query_engine=None)
        
        # Initialize Phase 3 components
        self.hybrid_query_engine = HybridQueryEngine()
        self.query_processor = create_query_processor()
        
        # Performance tracking
        self.query_analytics = {
            "total_queries": 0,
            "strategy_usage": {},
            "avg_response_times": {},
            "cache_hit_rate": 0.0
        }
        
        logger.info("Enhanced MCP server initialized with hybrid query capabilities")
    
    async def initialize(self) -> None:
        """Initialize the enhanced server components."""
        # Initialize hybrid query engine
        await self.hybrid_query_engine.initialize()
        logger.info("Hybrid query engine initialized")
    
    def _get_tool_definitions(self) -> List[Tool]:
        """Get enhanced tool definitions with hybrid search capabilities."""
        base_tools = super()._get_tool_definitions()
        
        # Enhanced search tool with intelligent features
        enhanced_search_tool = Tool(
            name="hybrid_search",
            description="Intelligent hybrid search with adaptive strategy selection and result synthesis",
            inputSchema={
                "type": "object",
                "properties": {
                    "index_id": {"type": "string", "description": "Index identifier"},
                    "query": {"type": "string", "description": "Search query with natural language support"},
                    "k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 20,
                        "minimum": 1,
                        "maximum": 100,
                    },
                    "strategy": {
                        "type": "string",
                        "description": "Search strategy (vector_first, semantic_first, parallel_hybrid, adaptive)",
                        "default": "adaptive",
                        "enum": ["vector_first", "semantic_first", "parallel_hybrid", "adaptive"]
                    },
                    "query_type": {
                        "type": "string", 
                        "description": "Query type hint (similarity, exact, fuzzy, semantic, code_structure, multi_modal)",
                        "enum": ["similarity", "exact", "fuzzy", "semantic", "code_structure", "multi_modal"]
                    },
                    "ranking_strategy": {
                        "type": "string",
                        "description": "Result ranking strategy",
                        "default": "context_adaptive",
                        "enum": ["relevance_first", "consensus_boost", "diversity_aware", "context_adaptive"]
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence threshold for results",
                        "default": 0.7,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "enable_expansion": {
                        "type": "boolean",
                        "description": "Enable query expansion",
                        "default": True
                    },
                    "enable_reranking": {
                        "type": "boolean", 
                        "description": "Enable intelligent result reranking",
                        "default": True
                    },
                    "performance_budget_ms": {
                        "type": "integer",
                        "description": "Performance budget in milliseconds",
                        "default": 5000,
                        "minimum": 1000,
                        "maximum": 30000
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context around matches",
                        "default": 5,
                        "minimum": 0,
                        "maximum": 20,
                    },
                },
                "required": ["index_id", "query"],
            },
        )
        
        # Advanced ask tool with query processing
        enhanced_ask_tool = Tool(
            name="intelligent_ask",
            description="Ask complex questions with advanced NLP understanding and multi-hop reasoning",
            inputSchema={
                "type": "object",
                "properties": {
                    "index_id": {"type": "string", "description": "Index identifier"},
                    "question": {
                        "type": "string",
                        "description": "Complex question with natural language processing",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "Lines of context for evidence",
                        "default": 5,
                        "minimum": 0,
                        "maximum": 20,
                    },
                    "enable_semantic_analysis": {
                        "type": "boolean",
                        "description": "Enable deep semantic analysis",
                        "default": True
                    },
                    "max_reasoning_depth": {
                        "type": "integer",
                        "description": "Maximum reasoning depth for multi-hop queries",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 5
                    }
                },
                "required": ["index_id", "question"],
            },
        )
        
        # Query analysis tool
        query_analysis_tool = Tool(
            name="analyze_query",
            description="Analyze query characteristics and suggest optimal search strategies",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Query to analyze"},
                    "include_expansion": {
                        "type": "boolean",
                        "description": "Include query expansion suggestions",
                        "default": True
                    }
                },
                "required": ["query"],
            },
        )
        
        # Performance analytics tool
        analytics_tool = Tool(
            name="get_query_analytics", 
            description="Get query performance analytics and usage statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "reset_stats": {
                        "type": "boolean",
                        "description": "Reset statistics after retrieval",
                        "default": False
                    }
                },
            },
        )
        
        # Add new tools to the base tools
        enhanced_tools = base_tools + [
            enhanced_search_tool,
            enhanced_ask_tool, 
            query_analysis_tool,
            analytics_tool
        ]
        
        return enhanced_tools
    
    def _get_tool_handler(self, name: str):
        """Get handler function for enhanced tools."""
        base_handlers = super()._get_tool_handler(name)
        if base_handlers:
            return base_handlers
        
        # Enhanced tool handlers
        enhanced_handlers = {
            "hybrid_search": self._hybrid_search,
            "intelligent_ask": self._intelligent_ask,
            "analyze_query": self._analyze_query,
            "get_query_analytics": self._get_query_analytics,
        }
        
        return enhanced_handlers.get(name)
    
    async def _hybrid_search(self, arguments: Dict[str, Any]) -> str:
        """Handle hybrid search with intelligent query processing."""
        try:
            start_time = time.time()
            
            # Extract and validate parameters
            index_id = arguments.get("index_id")
            query = arguments.get("query")
            k = arguments.get("k", 20)
            
            if not index_id or not query:
                raise ValueError("index_id and query are required")
            
            # Process query with advanced NLP
            processed_query = self.query_processor.process_query(query)
            logger.debug(f"Query processed - Intent: {processed_query.intent.value}, "
                        f"Complexity: {processed_query.query_complexity:.2f}")
            
            # Create query context
            context = QueryContext(
                strategy=QueryStrategy(arguments.get("strategy", "adaptive")),
                confidence_threshold=arguments.get("confidence_threshold", 0.7),
                max_results=k,
                enable_expansion=arguments.get("enable_expansion", True),
                enable_reranking=arguments.get("enable_reranking", True),
                performance_budget_ms=arguments.get("performance_budget_ms", 5000)
            )
            
            # Set query type if provided
            if "query_type" in arguments:
                context.query_type = QueryType(arguments["query_type"])
            else:
                # Infer from processed query
                context.query_type = self._map_intent_to_query_type(processed_query.intent)
            
            # Get search data from registered repository
            indexed_repo = self.query_engine.get_indexed_repository(index_id)
            if not indexed_repo:
                raise ValueError(f"Repository {index_id} not found or not indexed")
            
            search_data = {
                "vector_index": indexed_repo.vector_index,
                "serena_graph": indexed_repo.serena_graph,
                "repomap": indexed_repo.repomap_data,
                "repo_root": indexed_repo.repo_root,
                "rev": indexed_repo.rev,
                "repo_id": index_id
            }
            
            # Execute hybrid search
            response = await self.hybrid_query_engine.search(
                query=query,
                index_id=index_id,
                context=context,
                **search_data
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            # Update analytics
            self._update_query_analytics(context.strategy, execution_time)
            
            # Enhanced response with query analysis
            enhanced_response = {
                "search_response": response.model_dump(),
                "query_analysis": {
                    "original_query": processed_query.original_query,
                    "normalized_query": processed_query.normalized_query,
                    "intent": processed_query.intent.value,
                    "entities": [
                        {
                            "text": entity.text,
                            "type": entity.entity_type,
                            "confidence": entity.confidence
                        }
                        for entity in processed_query.entities
                    ],
                    "code_patterns": [
                        {
                            "type": pattern.pattern_type,
                            "value": pattern.value,
                            "confidence": pattern.confidence
                        }
                        for pattern in processed_query.code_patterns
                    ],
                    "language_hints": [lang.value for lang in processed_query.language_hints],
                    "complexity": processed_query.query_complexity,
                    "confidence": processed_query.confidence
                },
                "strategy_used": context.strategy.value,
                "execution_time_ms": execution_time
            }
            
            return json.dumps(enhanced_response, indent=2, default=str)
            
        except Exception as e:
            logger.exception(f"Error in hybrid search: {e}")
            raise
    
    async def _intelligent_ask(self, arguments: Dict[str, Any]) -> str:
        """Handle intelligent ask with advanced question processing."""
        try:
            # Extract parameters
            index_id = arguments.get("index_id")
            question = arguments.get("question")
            context_lines = arguments.get("context_lines", 5)
            enable_semantic_analysis = arguments.get("enable_semantic_analysis", True)
            
            if not index_id or not question:
                raise ValueError("index_id and question are required")
            
            # Process question with NLP
            processed_query = self.query_processor.process_query(question)
            
            # Use base ask functionality but with enhanced context
            request = AskIndexRequest(
                index_id=index_id,
                question=question,
                context_lines=context_lines
            )
            
            # Get base response
            base_response = await self.query_engine.ask(
                index_id=request.index_id,
                question=request.question,
                context_lines=request.context_lines,
            )
            
            # Enhanced response with query analysis
            enhanced_response = {
                "ask_response": base_response.model_dump(),
                "question_analysis": {
                    "original_question": processed_query.original_query,
                    "intent": processed_query.intent.value,
                    "complexity": processed_query.query_complexity,
                    "entities": [
                        {
                            "text": entity.text,
                            "type": entity.entity_type,
                            "confidence": entity.confidence
                        }
                        for entity in processed_query.entities
                    ],
                    "semantic_keywords": processed_query.semantic_keywords,
                    "expansion_terms": processed_query.expansion_terms
                },
                "semantic_analysis_enabled": enable_semantic_analysis
            }
            
            return json.dumps(enhanced_response, indent=2, default=str)
            
        except Exception as e:
            logger.exception(f"Error in intelligent ask: {e}")
            raise
    
    async def _analyze_query(self, arguments: Dict[str, Any]) -> str:
        """Handle query analysis for optimization suggestions."""
        try:
            query = arguments.get("query")
            include_expansion = arguments.get("include_expansion", True)
            
            if not query:
                raise ValueError("query is required")
            
            # Process query
            processed_query = self.query_processor.process_query(query)
            
            # Generate optimization suggestions
            suggestions = self._generate_optimization_suggestions(processed_query)
            
            analysis_result = {
                "query_analysis": {
                    "original_query": processed_query.original_query,
                    "normalized_query": processed_query.normalized_query,
                    "intent": processed_query.intent.value,
                    "query_type": self._map_intent_to_query_type(processed_query.intent).value,
                    "complexity": processed_query.query_complexity,
                    "confidence": processed_query.confidence,
                    "entities": [
                        {
                            "text": entity.text,
                            "type": entity.entity_type,
                            "confidence": entity.confidence,
                            "span": entity.span
                        }
                        for entity in processed_query.entities
                    ],
                    "code_patterns": [
                        {
                            "type": pattern.pattern_type,
                            "value": pattern.value,
                            "language_hint": pattern.language_hint.value if pattern.language_hint else None,
                            "confidence": pattern.confidence
                        }
                        for pattern in processed_query.code_patterns
                    ],
                    "language_hints": [lang.value for lang in processed_query.language_hints],
                    "semantic_keywords": processed_query.semantic_keywords,
                },
                "optimization_suggestions": suggestions
            }
            
            if include_expansion:
                analysis_result["expansion_terms"] = processed_query.expansion_terms
            
            return json.dumps(analysis_result, indent=2)
            
        except Exception as e:
            logger.exception(f"Error in query analysis: {e}")
            raise
    
    async def _get_query_analytics(self, arguments: Dict[str, Any]) -> str:
        """Handle query analytics retrieval."""
        try:
            reset_stats = arguments.get("reset_stats", False)
            
            # Get hybrid query engine analytics
            hybrid_analytics = self.hybrid_query_engine.get_analytics()
            
            # Combine with server analytics
            combined_analytics = {
                "server_analytics": self.query_analytics,
                "hybrid_engine_analytics": hybrid_analytics,
                "performance_summary": {
                    "total_queries": hybrid_analytics.get("total_queries", 0),
                    "cache_hit_rate": hybrid_analytics.get("cache_hit_rate", 0.0),
                    "strategy_distribution": hybrid_analytics.get("strategy_usage", {}),
                    "avg_response_times": hybrid_analytics.get("avg_response_times", {})
                },
                "health_status": await self.hybrid_query_engine.health_check()
            }
            
            if reset_stats:
                # Reset analytics
                self.query_analytics = {
                    "total_queries": 0,
                    "strategy_usage": {},
                    "avg_response_times": {},
                    "cache_hit_rate": 0.0
                }
                # Note: Hybrid engine analytics reset would require additional method
            
            return json.dumps(combined_analytics, indent=2, default=str)
            
        except Exception as e:
            logger.exception(f"Error getting analytics: {e}")
            raise
    
    def _map_intent_to_query_type(self, intent) -> QueryType:
        """Map query intent to query type for strategy selection."""
        from ..pipeline.advanced_query_processor import QueryIntent
        
        mapping = {
            QueryIntent.FIND_FUNCTION: QueryType.CODE_STRUCTURE,
            QueryIntent.FIND_CLASS: QueryType.CODE_STRUCTURE,
            QueryIntent.FIND_IMPLEMENTATION: QueryType.SEMANTIC,
            QueryIntent.FIND_USAGE: QueryType.SEMANTIC,
            QueryIntent.UNDERSTAND_CODE: QueryType.SEMANTIC,
            QueryIntent.FIND_PATTERN: QueryType.MULTI_MODAL,
            QueryIntent.FIND_BUG: QueryType.SEMANTIC,
            QueryIntent.FIND_SIMILAR: QueryType.SIMILARITY,
            QueryIntent.GENERAL_SEARCH: QueryType.SIMILARITY,
        }
        
        return mapping.get(intent, QueryType.SIMILARITY)
    
    def _generate_optimization_suggestions(self, processed_query) -> List[Dict[str, str]]:
        """Generate optimization suggestions based on query analysis."""
        suggestions = []
        
        # Strategy suggestions based on query characteristics
        if processed_query.query_complexity > 0.7:
            suggestions.append({
                "type": "strategy",
                "suggestion": "Consider using semantic_first strategy for complex queries",
                "reason": f"Query complexity is {processed_query.query_complexity:.2f}"
            })
        elif len(processed_query.code_patterns) > 2:
            suggestions.append({
                "type": "strategy", 
                "suggestion": "vector_first strategy may be optimal for code pattern searches",
                "reason": f"Found {len(processed_query.code_patterns)} code patterns"
            })
        
        # Query refinement suggestions
        if processed_query.confidence < 0.6:
            suggestions.append({
                "type": "refinement",
                "suggestion": "Consider adding more specific terms or code examples",
                "reason": f"Query confidence is low ({processed_query.confidence:.2f})"
            })
        
        # Performance suggestions
        if len(processed_query.expansion_terms) > 10:
            suggestions.append({
                "type": "performance",
                "suggestion": "Consider disabling expansion for faster results",
                "reason": f"Large expansion set ({len(processed_query.expansion_terms)} terms)"
            })
        
        # Language-specific suggestions
        if processed_query.language_hints:
            primary_lang = processed_query.language_hints[0]
            suggestions.append({
                "type": "optimization",
                "suggestion": f"Query appears to be {primary_lang.value}-specific",
                "reason": "Language-specific optimizations available"
            })
        
        return suggestions
    
    def _update_query_analytics(self, strategy, execution_time_ms: float) -> None:
        """Update query analytics with execution data."""
        self.query_analytics["total_queries"] += 1
        
        strategy_key = strategy.value
        if strategy_key not in self.query_analytics["strategy_usage"]:
            self.query_analytics["strategy_usage"][strategy_key] = 0
        self.query_analytics["strategy_usage"][strategy_key] += 1
        
        if strategy_key not in self.query_analytics["avg_response_times"]:
            self.query_analytics["avg_response_times"][strategy_key] = []
        
        times_list = self.query_analytics["avg_response_times"][strategy_key]
        times_list.append(execution_time_ms)
        
        # Keep only last 100 measurements
        if len(times_list) > 100:
            times_list.pop(0)


async def enhanced_async_main() -> None:
    """Async main entry point for enhanced MCP server."""
    from ..util.log import setup_logging
    setup_logging()

    # Create enhanced server instance
    enhanced_server = EnhancedMCPServer()
    
    # Initialize enhanced components
    await enhanced_server.initialize()

    # Run stdio server
    async with stdio_server() as (read_stream, write_stream):
        await enhanced_server.server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="mimir-enhanced-repoindex",
                server_version="2.0.0",
                capabilities=enhanced_server.server.get_capabilities(
                    notification_options=NotificationOptions(), 
                    experimental_capabilities={}
                ),
            ),
        )


def enhanced_main() -> None:
    """Synchronous main entry point for enhanced server."""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        prog="mimir-enhanced-server",
        description="Mimir Enhanced MCP Server - Phase 3 Hybrid Query Engine"
    )
    parser.add_argument(
        "--version",
        action="version", 
        version="mimir-enhanced-server 2.0.0"
    )
    parser.add_argument(
        "--storage-dir",
        type=str,
        help="Directory for storing indexes and cache (default: ~/.cache/mimir)"
    )
    
    args = parser.parse_args()
    
    # Set storage directory if provided
    if args.storage_dir:
        import os
        os.environ['MIMIR_DATA_DIR'] = args.storage_dir
    
    try:
        asyncio.run(enhanced_async_main())
    except KeyboardInterrupt:
        print("Enhanced Mimir server stopped by user", file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f"Error starting enhanced Mimir server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    enhanced_main()