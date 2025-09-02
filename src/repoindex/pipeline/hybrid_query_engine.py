"""
Hybrid Query Engine for Intelligent Search Synthesis.

Combines Lens vector search speed with Mimir semantic depth to provide
superior search capabilities through intelligent result synthesis.
"""

import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from ..data.schemas import (
    AskResponse,
    Citation,
    CodeSnippet,
    FeatureConfig,
    SearchResponse,
    SearchResult,
    SearchScores,
    SerenaGraph,
    VectorIndex,
)
from ..monitoring import get_metrics_collector, get_trace_manager
from ..util.errors import (
    ErrorCategory,
    ErrorSeverity,
    MimirError,
    RecoveryStrategy,
    create_error_context,
)
from ..util.log import get_logger
from .hybrid_search import HybridSearchEngine
from .lens_client import LensIntegrationClient, LensSearchRequest, init_lens_client

logger = get_logger(__name__)


class QueryStrategy(Enum):
    """Search strategies for combining Lens and Mimir capabilities."""
    VECTOR_FIRST = "vector_first"       # Fast Lens search + Mimir refinement
    SEMANTIC_FIRST = "semantic_first"   # Deep Mimir analysis + Lens expansion  
    PARALLEL_HYBRID = "parallel_hybrid" # Both systems simultaneously with fusion
    ADAPTIVE = "adaptive"               # Strategy selection based on query analysis


class QueryType(Enum):
    """Classification of query types for intelligent routing."""
    SIMILARITY = "similarity"           # Vector similarity search
    EXACT = "exact"                     # Exact string/symbol matching
    FUZZY = "fuzzy"                     # Fuzzy matching with tolerance
    SEMANTIC = "semantic"               # Deep semantic understanding
    CODE_STRUCTURE = "code_structure"   # Code patterns and structure
    MULTI_MODAL = "multi_modal"         # Text + code + semantic combined


@dataclass
class QueryContext:
    """Context information for intelligent query processing."""
    query_type: QueryType = QueryType.SIMILARITY
    strategy: QueryStrategy = QueryStrategy.ADAPTIVE
    complexity_score: float = 0.0
    confidence_threshold: float = 0.7
    max_results: int = 20
    enable_expansion: bool = True
    enable_reranking: bool = True
    performance_budget_ms: int = 5000
    quality_vs_speed_preference: float = 0.5  # 0=speed, 1=quality


@dataclass
class SearchCandidate:
    """Individual search candidate from either system."""
    path: str
    span: Tuple[int, int]
    score: float
    source: str  # "lens", "mimir", "hybrid"
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_preview: Optional[str] = None


@dataclass
class QueryAnalysis:
    """Analysis of query characteristics for strategy selection."""
    is_code_specific: bool = False
    has_semantic_complexity: bool = False
    estimated_result_count: int = 0
    query_tokens: List[str] = field(default_factory=list)
    code_patterns: List[str] = field(default_factory=list)
    suggested_strategy: QueryStrategy = QueryStrategy.ADAPTIVE


class SearchStrategy(ABC):
    """Abstract base class for search strategies."""
    
    @abstractmethod
    async def search(
        self,
        query: str,
        context: QueryContext,
        lens_client: Optional[LensIntegrationClient],
        mimir_engine: HybridSearchEngine,
        **search_data
    ) -> List[SearchCandidate]:
        """Execute search strategy and return candidates."""
        pass

    @abstractmethod
    def get_performance_characteristics(self) -> Dict[str, Any]:
        """Get expected performance characteristics of this strategy."""
        pass


class VectorFirstStrategy(SearchStrategy):
    """Fast Lens vector search followed by Mimir refinement."""
    
    async def search(
        self,
        query: str,
        context: QueryContext,
        lens_client: Optional[LensIntegrationClient],
        mimir_engine: HybridSearchEngine,
        **search_data
    ) -> List[SearchCandidate]:
        """Execute vector-first search strategy."""
        candidates = []
        
        # Phase 1: Fast Lens vector search for broad candidate set
        if lens_client:
            try:
                lens_request = LensSearchRequest(
                    query=query,
                    repository_id=search_data.get("repo_id"),
                    max_results=min(context.max_results * 3, 100),  # Over-fetch for refinement
                    include_embeddings=True
                )
                
                lens_response = await lens_client.search_repository(lens_request)
                
                if lens_response.success and lens_response.data:
                    for result in lens_response.data.get("results", []):
                        candidates.append(SearchCandidate(
                            path=result.get("file_path", ""),
                            span=(result.get("start_line", 0), result.get("end_line", 0)),
                            score=result.get("similarity", 0.0),
                            source="lens",
                            metadata={"lens_metadata": result.get("metadata", {})}
                        ))
            except Exception as e:
                logger.warning(f"Lens search failed in vector-first strategy: {e}")
        
        # Phase 2: Mimir refinement for top candidates
        if candidates and search_data.get("vector_index") and search_data.get("serena_graph"):
            # Use Mimir to refine and re-score top candidates
            try:
                # Take top candidates from Lens for Mimir analysis
                top_lens_candidates = sorted(candidates, key=lambda c: c.score, reverse=True)[:20]
                
                # Create focused query for Mimir based on Lens results
                mimir_features = FeatureConfig(vector=True, symbol=True, graph=False)
                
                mimir_response = await mimir_engine.search(
                    query=query,
                    vector_index=search_data.get("vector_index"),
                    serena_graph=search_data.get("serena_graph"),
                    repomap=search_data.get("repomap"),
                    repo_root=search_data.get("repo_root", ""),
                    rev=search_data.get("rev", ""),
                    features=mimir_features,
                    k=context.max_results,
                    context_lines=5
                )
                
                # Blend Lens and Mimir results
                mimir_candidates = []
                for result in mimir_response.results:
                    mimir_candidates.append(SearchCandidate(
                        path=result.path,
                        span=result.span,
                        score=result.score,
                        source="mimir",
                        metadata={"scores": result.scores.dict()}
                    ))
                
                # Merge and re-rank
                candidates = self._blend_candidates(candidates, mimir_candidates, bias_towards="lens")
                
            except Exception as e:
                logger.warning(f"Mimir refinement failed in vector-first strategy: {e}")
        
        return candidates[:context.max_results]
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        return {
            "avg_latency_ms": 800,
            "max_latency_ms": 2000,
            "quality_score": 0.8,
            "recall_bias": "high",  # Good at finding many relevant results
            "precision_bias": "medium"
        }
    
    def _blend_candidates(
        self,
        lens_candidates: List[SearchCandidate],
        mimir_candidates: List[SearchCandidate],
        bias_towards: str = "lens"
    ) -> List[SearchCandidate]:
        """Intelligently blend candidates from both systems."""
        # Create index of mimir candidates for quick lookup
        mimir_by_location = {(c.path, c.span): c for c in mimir_candidates}
        
        blended = []
        seen_locations = set()
        
        # Start with the preferred source
        primary_candidates = lens_candidates if bias_towards == "lens" else mimir_candidates
        secondary_candidates = mimir_candidates if bias_towards == "lens" else lens_candidates
        secondary_by_location = {(c.path, c.span): c for c in secondary_candidates}
        
        for candidate in primary_candidates:
            location = (candidate.path, candidate.span)
            if location in seen_locations:
                continue
            
            # Check if this location exists in the other system
            if location in secondary_by_location:
                other_candidate = secondary_by_location[location]
                # Blend scores with bias
                blended_score = (candidate.score * 0.7 + other_candidate.score * 0.3) if bias_towards == "lens" else (candidate.score * 0.3 + other_candidate.score * 0.7)
                
                candidate.score = blended_score
                candidate.source = "hybrid"
                candidate.metadata.update(other_candidate.metadata)
            
            blended.append(candidate)
            seen_locations.add(location)
        
        # Add unique results from secondary source
        for candidate in secondary_candidates:
            location = (candidate.path, candidate.span)
            if location not in seen_locations:
                blended.append(candidate)
                seen_locations.add(location)
        
        return sorted(blended, key=lambda c: c.score, reverse=True)


class SemanticFirstStrategy(SearchStrategy):
    """Deep Mimir semantic analysis followed by Lens candidate expansion."""
    
    async def search(
        self,
        query: str,
        context: QueryContext,
        lens_client: Optional[LensIntegrationClient],
        mimir_engine: HybridSearchEngine,
        **search_data
    ) -> List[SearchCandidate]:
        """Execute semantic-first search strategy."""
        candidates = []
        
        # Phase 1: Deep Mimir semantic analysis
        if search_data.get("vector_index") and search_data.get("serena_graph"):
            try:
                # Use all Mimir features for comprehensive analysis
                mimir_features = FeatureConfig(vector=True, symbol=True, graph=True)
                
                mimir_response = await mimir_engine.search(
                    query=query,
                    vector_index=search_data.get("vector_index"),
                    serena_graph=search_data.get("serena_graph"),
                    repomap=search_data.get("repomap"),
                    repo_root=search_data.get("repo_root", ""),
                    rev=search_data.get("rev", ""),
                    features=mimir_features,
                    k=context.max_results,
                    context_lines=5
                )
                
                for result in mimir_response.results:
                    candidates.append(SearchCandidate(
                        path=result.path,
                        span=result.span,
                        score=result.score,
                        source="mimir",
                        metadata={
                            "scores": result.scores.dict(),
                            "semantic_analysis": True
                        }
                    ))
            except Exception as e:
                logger.warning(f"Mimir semantic analysis failed: {e}")
        
        # Phase 2: Lens expansion for broader coverage
        if lens_client and context.enable_expansion:
            try:
                # Expand query based on Mimir results
                expanded_query = await self._expand_query(query, candidates)
                
                lens_request = LensSearchRequest(
                    query=expanded_query,
                    repository_id=search_data.get("repo_id"),
                    max_results=min(context.max_results * 2, 60),
                    include_embeddings=False  # Fast expansion search
                )
                
                lens_response = await lens_client.search_repository(lens_request)
                
                if lens_response.success and lens_response.data:
                    lens_candidates = []
                    for result in lens_response.data.get("results", []):
                        lens_candidates.append(SearchCandidate(
                            path=result.get("file_path", ""),
                            span=(result.get("start_line", 0), result.get("end_line", 0)),
                            score=result.get("similarity", 0.0) * 0.7,  # Lower weight for expansion
                            source="lens",
                            metadata={"expansion": True}
                        ))
                    
                    # Add unique lens results
                    candidates = self._merge_unique_candidates(candidates, lens_candidates)
                
            except Exception as e:
                logger.warning(f"Lens expansion failed in semantic-first strategy: {e}")
        
        return sorted(candidates, key=lambda c: c.score, reverse=True)[:context.max_results]
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        return {
            "avg_latency_ms": 1500,
            "max_latency_ms": 4000,
            "quality_score": 0.95,
            "recall_bias": "medium",
            "precision_bias": "high"  # Excellent at finding highly relevant results
        }
    
    async def _expand_query(self, original_query: str, semantic_results: List[SearchCandidate]) -> str:
        """Expand query based on semantic analysis results."""
        # Simple expansion - in practice could use LLM or more sophisticated NLP
        expansion_terms = set([original_query])
        
        # Extract terms from high-scoring results
        for candidate in semantic_results[:5]:  # Top 5 results
            if candidate.score > 0.8:
                # Extract meaningful terms from path
                path_terms = candidate.path.split("/")[-1].replace(".", " ").replace("_", " ")
                expansion_terms.add(path_terms)
        
        return " ".join(list(expansion_terms))
    
    def _merge_unique_candidates(
        self,
        primary: List[SearchCandidate],
        secondary: List[SearchCandidate]
    ) -> List[SearchCandidate]:
        """Merge candidates, avoiding duplicates."""
        seen_locations = {(c.path, c.span) for c in primary}
        merged = list(primary)
        
        for candidate in secondary:
            location = (candidate.path, candidate.span)
            if location not in seen_locations:
                merged.append(candidate)
                seen_locations.add(location)
        
        return merged


class ParallelHybridStrategy(SearchStrategy):
    """Execute both systems simultaneously and fuse results."""
    
    async def search(
        self,
        query: str,
        context: QueryContext,
        lens_client: Optional[LensIntegrationClient],
        mimir_engine: HybridSearchEngine,
        **search_data
    ) -> List[SearchCandidate]:
        """Execute parallel hybrid search strategy."""
        # Launch both searches simultaneously
        search_tasks = []
        
        # Lens search task
        if lens_client:
            search_tasks.append(self._lens_search(query, context, lens_client, search_data))
        
        # Mimir search task
        if search_data.get("vector_index") and search_data.get("serena_graph"):
            search_tasks.append(self._mimir_search(query, context, mimir_engine, search_data))
        
        if not search_tasks:
            return []
        
        # Execute searches in parallel
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            lens_candidates = []
            mimir_candidates = []
            
            result_idx = 0
            if lens_client:
                if not isinstance(results[result_idx], Exception):
                    lens_candidates = results[result_idx]
                result_idx += 1
            
            if search_data.get("vector_index") and search_data.get("serena_graph"):
                if not isinstance(results[result_idx], Exception):
                    mimir_candidates = results[result_idx]
            
            # Advanced result fusion
            fused_candidates = await self._fuse_results(
                lens_candidates, 
                mimir_candidates,
                query,
                context
            )
            
            return fused_candidates[:context.max_results]
            
        except Exception as e:
            logger.error(f"Parallel hybrid search failed: {e}")
            return []
    
    def get_performance_characteristics(self) -> Dict[str, Any]:
        return {
            "avg_latency_ms": 1200,  # Parallel execution
            "max_latency_ms": 3000,
            "quality_score": 0.92,
            "recall_bias": "high",
            "precision_bias": "high"
        }
    
    async def _lens_search(
        self,
        query: str,
        context: QueryContext,
        lens_client: LensIntegrationClient,
        search_data: Dict[str, Any]
    ) -> List[SearchCandidate]:
        """Execute Lens search component."""
        try:
            lens_request = LensSearchRequest(
                query=query,
                repository_id=search_data.get("repo_id"),
                max_results=context.max_results * 2,
                include_embeddings=True
            )
            
            lens_response = await lens_client.search_repository(lens_request)
            
            if lens_response.success and lens_response.data:
                candidates = []
                for result in lens_response.data.get("results", []):
                    candidates.append(SearchCandidate(
                        path=result.get("file_path", ""),
                        span=(result.get("start_line", 0), result.get("end_line", 0)),
                        score=result.get("similarity", 0.0),
                        source="lens",
                        metadata={"lens_score": result.get("similarity", 0.0)}
                    ))
                return candidates
            
        except Exception as e:
            logger.warning(f"Lens component failed in parallel search: {e}")
        
        return []
    
    async def _mimir_search(
        self,
        query: str,
        context: QueryContext,
        mimir_engine: HybridSearchEngine,
        search_data: Dict[str, Any]
    ) -> List[SearchCandidate]:
        """Execute Mimir search component."""
        try:
            mimir_features = FeatureConfig(vector=True, symbol=True, graph=True)
            
            mimir_response = await mimir_engine.search(
                query=query,
                vector_index=search_data.get("vector_index"),
                serena_graph=search_data.get("serena_graph"),
                repomap=search_data.get("repomap"),
                repo_root=search_data.get("repo_root", ""),
                rev=search_data.get("rev", ""),
                features=mimir_features,
                k=context.max_results * 2,
                context_lines=5
            )
            
            candidates = []
            for result in mimir_response.results:
                candidates.append(SearchCandidate(
                    path=result.path,
                    span=result.span,
                    score=result.score,
                    source="mimir",
                    metadata={"mimir_scores": result.scores.dict()}
                ))
            
            return candidates
            
        except Exception as e:
            logger.warning(f"Mimir component failed in parallel search: {e}")
        
        return []
    
    async def _fuse_results(
        self,
        lens_candidates: List[SearchCandidate],
        mimir_candidates: List[SearchCandidate],
        query: str,
        context: QueryContext
    ) -> List[SearchCandidate]:
        """Advanced result fusion with intelligent ranking."""
        # Create comprehensive candidate map
        candidate_fusion_map = {}
        
        # Process Lens candidates
        for candidate in lens_candidates:
            key = (candidate.path, candidate.span)
            if key not in candidate_fusion_map:
                candidate_fusion_map[key] = {
                    "path": candidate.path,
                    "span": candidate.span,
                    "lens_score": 0.0,
                    "mimir_score": 0.0,
                    "sources": set(),
                    "metadata": {}
                }
            
            candidate_fusion_map[key]["lens_score"] = candidate.score
            candidate_fusion_map[key]["sources"].add("lens")
            candidate_fusion_map[key]["metadata"].update(candidate.metadata)
        
        # Process Mimir candidates
        for candidate in mimir_candidates:
            key = (candidate.path, candidate.span)
            if key not in candidate_fusion_map:
                candidate_fusion_map[key] = {
                    "path": candidate.path,
                    "span": candidate.span,
                    "lens_score": 0.0,
                    "mimir_score": 0.0,
                    "sources": set(),
                    "metadata": {}
                }
            
            candidate_fusion_map[key]["mimir_score"] = candidate.score
            candidate_fusion_map[key]["sources"].add("mimir")
            candidate_fusion_map[key]["metadata"].update(candidate.metadata)
        
        # Calculate fusion scores
        fused_candidates = []
        for key, fusion_data in candidate_fusion_map.items():
            # Adaptive scoring based on agreement between systems
            lens_score = fusion_data["lens_score"]
            mimir_score = fusion_data["mimir_score"]
            sources = fusion_data["sources"]
            
            # Boost score for candidates found by both systems
            if len(sources) == 2:
                # High agreement bonus
                fusion_score = (lens_score + mimir_score) * 0.6 + 0.3  # Consensus bonus
            else:
                # Single system result - weight by system reliability for this query type
                if "lens" in sources:
                    fusion_score = lens_score * 0.8
                else:
                    fusion_score = mimir_score * 0.9  # Mimir slightly more reliable for single results
            
            # Apply context-based adjustments
            fusion_score = self._apply_context_adjustments(
                fusion_score, fusion_data, query, context
            )
            
            fused_candidate = SearchCandidate(
                path=fusion_data["path"],
                span=fusion_data["span"],
                score=fusion_score,
                source="hybrid",
                metadata={
                    "fusion_data": fusion_data,
                    "systems": list(sources),
                    "lens_score": lens_score,
                    "mimir_score": mimir_score
                }
            )
            
            fused_candidates.append(fused_candidate)
        
        return sorted(fused_candidates, key=lambda c: c.score, reverse=True)
    
    def _apply_context_adjustments(
        self,
        base_score: float,
        fusion_data: Dict[str, Any],
        query: str,
        context: QueryContext
    ) -> float:
        """Apply context-specific score adjustments."""
        adjusted_score = base_score
        
        # Boost for code-specific queries
        if context.query_type == QueryType.CODE_STRUCTURE:
            if fusion_data["mimir_score"] > 0:
                adjusted_score *= 1.1  # Mimir better for code structure
        
        # Boost for semantic queries
        elif context.query_type == QueryType.SEMANTIC:
            if "mimir" in fusion_data["sources"]:
                adjusted_score *= 1.15
        
        # Boost for similarity queries
        elif context.query_type == QueryType.SIMILARITY:
            if "lens" in fusion_data["sources"]:
                adjusted_score *= 1.1
        
        return min(adjusted_score, 1.0)  # Cap at 1.0


class HybridQueryEngine:
    """
    Intelligent Query Engine combining Lens and Mimir capabilities.
    
    Features:
    - Multiple search strategies with adaptive selection
    - Query analysis and intelligent routing  
    - Result synthesis and advanced ranking
    - Performance optimization and caching
    - Real-time query analytics
    """
    
    def __init__(self):
        """Initialize hybrid query engine."""
        self.strategies = {
            QueryStrategy.VECTOR_FIRST: VectorFirstStrategy(),
            QueryStrategy.SEMANTIC_FIRST: SemanticFirstStrategy(), 
            QueryStrategy.PARALLEL_HYBRID: ParallelHybridStrategy()
        }
        
        # Performance and caching
        self._query_cache = {}
        self._cache_max_size = 1000
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Analytics
        self._query_analytics = {
            "total_queries": 0,
            "strategy_usage": {strategy.value: 0 for strategy in QueryStrategy},
            "avg_response_times": {},
            "cache_hit_rate": 0.0
        }
        
        # Components
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()
        self.mimir_engine = HybridSearchEngine()
        self._lens_client: Optional[LensIntegrationClient] = None
        
        logger.info("HybridQueryEngine initialized with all strategies")
    
    async def initialize(self) -> None:
        """Initialize the query engine and its components."""
        try:
            # Initialize Lens client
            self._lens_client = await init_lens_client()
            logger.info("Lens client initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize Lens client: {e}")
            self._lens_client = None
    
    async def search(
        self,
        query: str,
        index_id: str,
        context: Optional[QueryContext] = None,
        **search_data
    ) -> SearchResponse:
        """
        Execute intelligent hybrid search with strategy selection.
        
        Args:
            query: Search query string
            index_id: Repository index identifier  
            context: Query context for strategy selection
            **search_data: Additional search data (vector_index, serena_graph, etc.)
            
        Returns:
            SearchResponse with synthesized results
        """
        if context is None:
            context = QueryContext()
        
        start_time = time.time()
        self._query_analytics["total_queries"] += 1
        
        # Check cache first
        cache_key = self._create_cache_key(query, index_id, context)
        cached_result = self._get_cached_result(cache_key)
        if cached_result:
            self._update_cache_hit_rate(True)
            return cached_result
        
        self._update_cache_hit_rate(False)
        
        async with self.trace_manager.trace_operation(
            "hybrid_query_search",
            query=query,
            index_id=index_id,
            strategy=context.strategy.value
        ):
            try:
                # Analyze query if using adaptive strategy
                if context.strategy == QueryStrategy.ADAPTIVE:
                    analysis = await self._analyze_query(query, search_data)
                    context.strategy = analysis.suggested_strategy
                
                # Select and execute strategy
                strategy = self.strategies.get(context.strategy)
                if not strategy:
                    raise MimirError(f"Unknown strategy: {context.strategy}")
                
                # Execute search
                candidates = await strategy.search(
                    query=query,
                    context=context,
                    lens_client=self._lens_client,
                    mimir_engine=self.mimir_engine,
                    **search_data
                )
                
                # Convert candidates to search results
                search_results = await self._candidates_to_search_results(
                    candidates, search_data
                )
                
                # Apply intelligent re-ranking if enabled
                if context.enable_reranking:
                    search_results = await self._rerank_results(
                        search_results, query, context
                    )
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                # Create response
                response = SearchResponse(
                    query=query,
                    results=search_results,
                    total_count=len(search_results),
                    features_used=FeatureConfig(),  # Set based on actual usage
                    execution_time_ms=execution_time_ms,
                    index_id=index_id
                )
                
                # Cache result
                self._cache_result(cache_key, response)
                
                # Update analytics
                self._update_analytics(context.strategy, execution_time_ms)
                
                return response
                
            except Exception as e:
                execution_time_ms = (time.time() - start_time) * 1000
                self._update_analytics(context.strategy, execution_time_ms, error=True)
                
                if not isinstance(e, MimirError):
                    e = MimirError(
                        message=f"Hybrid search failed: {str(e)}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.LOGIC,
                        recovery_strategy=RecoveryStrategy.RETRY,
                        context=create_error_context(
                            component="hybrid_query_engine",
                            operation="search",
                            parameters={"query": query, "index_id": index_id}
                        ),
                        cause=e
                    )
                raise e
    
    async def _analyze_query(self, query: str, search_data: Dict[str, Any]) -> QueryAnalysis:
        """Analyze query characteristics for intelligent strategy selection."""
        analysis = QueryAnalysis()
        
        query_lower = query.lower()
        analysis.query_tokens = query_lower.split()
        
        # Detect code-specific patterns
        code_keywords = [
            "function", "class", "method", "variable", "import", "export",
            "def", "const", "let", "var", "async", "await", "return"
        ]
        analysis.is_code_specific = any(keyword in query_lower for keyword in code_keywords)
        
        # Detect semantic complexity
        semantic_indicators = [
            "how", "why", "what", "when", "where", "explain", "understand",
            "implement", "pattern", "design", "architecture", "relationship"
        ]
        analysis.has_semantic_complexity = any(indicator in query_lower for indicator in semantic_indicators)
        
        # Extract code patterns (simplified)
        import re
        code_patterns = re.findall(r'\b[A-Z][a-z]+(?:[A-Z][a-z]*)*\b', query)  # CamelCase
        code_patterns.extend(re.findall(r'\b[a-z]+_[a-z_]*\b', query))  # snake_case
        analysis.code_patterns = code_patterns
        
        # Suggest strategy based on analysis
        if analysis.has_semantic_complexity:
            analysis.suggested_strategy = QueryStrategy.SEMANTIC_FIRST
        elif analysis.is_code_specific and len(analysis.code_patterns) > 0:
            analysis.suggested_strategy = QueryStrategy.VECTOR_FIRST
        else:
            analysis.suggested_strategy = QueryStrategy.PARALLEL_HYBRID
        
        return analysis
    
    async def _candidates_to_search_results(
        self,
        candidates: List[SearchCandidate],
        search_data: Dict[str, Any]
    ) -> List[SearchResult]:
        """Convert search candidates to SearchResult objects."""
        results = []
        
        for candidate in candidates:
            # Create content snippet (simplified)
            content = CodeSnippet(
                path=candidate.path,
                span=candidate.span,
                hash="",  # Would compute actual hash
                pre="",  # Would extract actual context  
                text=candidate.content_preview or f"Content at {candidate.path}:{candidate.span[0]}-{candidate.span[1]}",
                post="",
                line_start=candidate.span[0],
                line_end=candidate.span[1]
            )
            
            # Create citation
            citation = Citation(
                repo_root=search_data.get("repo_root", ""),
                rev=search_data.get("rev", ""),
                path=candidate.path,
                span=candidate.span,
                content_sha=""  # Would compute actual hash
            )
            
            # Create scores object
            scores = SearchScores()
            if candidate.source == "lens":
                scores.vector = candidate.score
            elif candidate.source == "mimir":
                # Extract individual scores if available
                if "mimir_scores" in candidate.metadata:
                    mimir_scores = candidate.metadata["mimir_scores"]
                    scores.vector = mimir_scores.get("vector", 0.0)
                    scores.symbol = mimir_scores.get("symbol", 0.0)
                    scores.graph = mimir_scores.get("graph", 0.0)
                else:
                    scores.symbol = candidate.score
            elif candidate.source == "hybrid":
                # Use fusion data if available
                if "fusion_data" in candidate.metadata:
                    fusion_data = candidate.metadata["fusion_data"]
                    scores.vector = fusion_data.get("lens_score", 0.0) 
                    scores.symbol = fusion_data.get("mimir_score", 0.0)
            
            result = SearchResult(
                path=candidate.path,
                span=candidate.span,
                score=candidate.score,
                scores=scores,
                content=content,
                citation=citation
            )
            
            results.append(result)
        
        return results
    
    async def _rerank_results(
        self,
        results: List[SearchResult],
        query: str,
        context: QueryContext
    ) -> List[SearchResult]:
        """Apply intelligent re-ranking to improve result quality."""
        # Simplified re-ranking logic
        # In practice, this could use ML models, user behavior, etc.
        
        for result in results:
            # Boost results with high consensus (both systems found it)
            if hasattr(result, 'metadata') and 'systems' in result.citation.__dict__:
                systems = result.citation.__dict__.get('systems', [])
                if len(systems) > 1:
                    result.score *= 1.2
            
            # Boost code structure results for code queries
            if context.query_type == QueryType.CODE_STRUCTURE:
                if result.scores.symbol > 0.5:
                    result.score *= 1.1
            
            # Apply confidence threshold filtering
            if result.score < context.confidence_threshold:
                result.score *= 0.8  # Lower confidence penalty
        
        return sorted(results, key=lambda r: r.score, reverse=True)
    
    def _create_cache_key(self, query: str, index_id: str, context: QueryContext) -> str:
        """Create cache key for query."""
        import hashlib
        key_data = f"{query}:{index_id}:{context.strategy.value}:{context.max_results}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_result(self, cache_key: str) -> Optional[SearchResponse]:
        """Get cached result if available and not expired."""
        if cache_key in self._query_cache:
            cached_item = self._query_cache[cache_key]
            if time.time() - cached_item["timestamp"] < self._cache_ttl_seconds:
                return cached_item["response"]
            else:
                del self._query_cache[cache_key]
        return None
    
    def _cache_result(self, cache_key: str, response: SearchResponse) -> None:
        """Cache query result with TTL."""
        # Simple cache eviction
        if len(self._query_cache) >= self._cache_max_size:
            oldest_key = min(self._query_cache.keys(), 
                           key=lambda k: self._query_cache[k]["timestamp"])
            del self._query_cache[oldest_key]
        
        self._query_cache[cache_key] = {
            "response": response,
            "timestamp": time.time()
        }
    
    def _update_cache_hit_rate(self, cache_hit: bool) -> None:
        """Update cache hit rate statistics."""
        total_queries = self._query_analytics["total_queries"]
        if total_queries == 1:
            self._query_analytics["cache_hit_rate"] = 1.0 if cache_hit else 0.0
        else:
            current_rate = self._query_analytics["cache_hit_rate"]
            new_rate = ((current_rate * (total_queries - 1)) + (1.0 if cache_hit else 0.0)) / total_queries
            self._query_analytics["cache_hit_rate"] = new_rate
    
    def _update_analytics(self, strategy: QueryStrategy, execution_time_ms: float, error: bool = False) -> None:
        """Update query analytics."""
        self._query_analytics["strategy_usage"][strategy.value] += 1
        
        if not error:
            strategy_key = strategy.value
            if strategy_key not in self._query_analytics["avg_response_times"]:
                self._query_analytics["avg_response_times"][strategy_key] = []
            
            times_list = self._query_analytics["avg_response_times"][strategy_key]
            times_list.append(execution_time_ms)
            
            # Keep only last 100 measurements
            if len(times_list) > 100:
                times_list.pop(0)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get query analytics and performance metrics."""
        analytics = dict(self._query_analytics)
        
        # Calculate average response times
        for strategy, times in analytics["avg_response_times"].items():
            if times:
                analytics["avg_response_times"][strategy] = sum(times) / len(times)
            else:
                analytics["avg_response_times"][strategy] = 0.0
        
        return analytics
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the hybrid query engine."""
        health_status = {
            "status": "healthy",
            "components": {
                "mimir_engine": "healthy",
                "lens_client": "unknown",
                "strategies": "healthy"
            },
            "analytics": self.get_analytics(),
            "cache_size": len(self._query_cache)
        }
        
        # Check Lens client health
        if self._lens_client:
            try:
                lens_health = await self._lens_client.check_health()
                health_status["components"]["lens_client"] = lens_health.status.value
            except Exception as e:
                health_status["components"]["lens_client"] = "unhealthy"
                health_status["lens_error"] = str(e)
        else:
            health_status["components"]["lens_client"] = "disabled"
        
        # Determine overall status
        if any(status == "unhealthy" for status in health_status["components"].values()):
            health_status["status"] = "degraded"
        
        return health_status