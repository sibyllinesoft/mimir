"""
Enhanced multi-stage search pipeline with HyDE, code embeddings, and reranking.

Integrates all advanced search components into a cohesive pipeline that
significantly improves search quality through query transformation,
specialized embeddings, and result reranking.
"""

import asyncio
import time
from typing import Dict, List, Optional, Tuple, Any

from ..config import get_ai_config
from ..data.schemas import (
    FeatureConfig,
    SearchResponse, 
    SearchResult,
    VectorChunk,
    VectorIndex,
    SerenaGraph,
    RepoMap,
)
from ..util.errors import MimirError, create_error_context
from ..util.logging_config import get_logger
from ..monitoring import get_metrics_collector, search_metrics
from .hyde import HyDETransformer
from .code_embeddings import CodeEmbeddingAdapter
from .reranking import create_reranker
from .hybrid_search import HybridSearchEngine


logger = get_logger(__name__)


class EnhancedSearchError(MimirError):
    """Errors in enhanced search pipeline."""
    pass


class EnhancedSearchPipeline:
    """
    Multi-stage enhanced search pipeline.
    
    Combines HyDE query transformation, code-specific embeddings,
    hybrid search, and cross-encoder reranking for superior search quality.
    
    Pipeline stages:
    1. Query analysis and preprocessing
    2. HyDE query transformation (optional)
    3. Multi-modal search (vector + symbol + graph) 
    4. Cross-encoder reranking (optional)
    5. Result assembly and response generation
    """
    
    def __init__(self):
        """Initialize enhanced search pipeline."""
        self.config = get_ai_config()
        self.metrics_collector = get_metrics_collector()
        
        # Initialize components
        self.hyde_transformer = HyDETransformer()
        self.code_embedder = CodeEmbeddingAdapter()
        self.reranker = create_reranker()
        self.hybrid_engine = HybridSearchEngine()
        
        # Pipeline configuration
        self.enable_hyde = self.config.query.enable_hyde
        self.enable_reranking = self.config.reranker.enabled
        self.initial_retrieval_k = self.config.reranker.initial_retrieval_k
        self.final_k = self.config.reranker.top_k
        
        # Performance settings
        self.cache_embeddings = True
        self.max_concurrent_operations = 3
        
        logger.info("Initialized EnhancedSearchPipeline")
    
    async def initialize(self) -> None:
        """Initialize all pipeline components."""
        try:
            # Initialize components concurrently
            init_tasks = [
                self.hyde_transformer.initialize(),
                self.code_embedder.initialize(), 
                self.reranker.initialize() if hasattr(self.reranker, 'initialize') else self._noop(),
            ]
            
            await asyncio.gather(*init_tasks, return_exceptions=True)
            logger.info("Enhanced search pipeline initialized")
            
        except Exception as e:
            logger.warning(f"Some pipeline components failed to initialize: {e}")
            # Pipeline can still function with partial initialization
    
    async def _noop(self):
        """No-op coroutine for components without initialize method."""
        pass
    
    @search_metrics("enhanced")
    async def search(
        self,
        query: str,
        vector_index: VectorIndex | None,
        serena_graph: SerenaGraph | None, 
        repomap: RepoMap | None,
        repo_root: str,
        rev: str,
        features: FeatureConfig,
        k: int = 20,
        context_lines: int = 5,
    ) -> SearchResponse:
        """
        Execute enhanced multi-stage search.
        
        Args:
            query: User search query
            vector_index: Vector embeddings index
            serena_graph: Symbol and structure graph
            repomap: Repository structure map
            repo_root: Repository root path
            rev: Git revision
            features: Feature configuration
            k: Final number of results to return
            context_lines: Context lines around matches
            
        Returns:
            Enhanced search response with reranked results
        """
        start_time = time.time()
        
        try:
            # Stage 1: Query Analysis and Preprocessing
            query_info = await self._analyze_query(query)
            logger.debug(f"Query analysis: {query_info['type']}")
            
            # Stage 2: HyDE Query Transformation
            enhanced_query = query
            if self.enable_hyde and query_info['suitable_for_hyde']:
                enhanced_query = await self.hyde_transformer.transform_query(query)
                logger.debug(f"HyDE enhanced query length: {len(enhanced_query)}")
            
            # Stage 3: Multi-modal Search with Enhanced Embeddings
            search_results = await self._enhanced_hybrid_search(
                original_query=query,
                enhanced_query=enhanced_query,
                vector_index=vector_index,
                serena_graph=serena_graph,
                repomap=repomap,
                repo_root=repo_root,
                rev=rev, 
                features=features,
                k=self.initial_retrieval_k,
                context_lines=context_lines,
            )
            
            # Stage 4: Cross-encoder Reranking
            if self.enable_reranking and len(search_results.results) > 1:
                reranked_results = await self.reranker.rerank_search_results(
                    query, search_results.results, top_k=k
                )
                search_results.results = reranked_results
            else:
                search_results.results = search_results.results[:k]
            
            # Stage 5: Final Assembly and Enhancement
            final_response = await self._enhance_search_response(
                search_results, query_info, start_time
            )
            
            execution_time = time.time() - start_time
            
            # Record comprehensive metrics
            self._record_pipeline_metrics(
                query_info, len(final_response.results), execution_time
            )
            
            return final_response
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            # Fallback to basic hybrid search
            return await self.hybrid_engine.search(
                query, vector_index, serena_graph, repomap, 
                repo_root, rev, features, k, context_lines
            )
    
    async def _analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal processing strategy."""
        query_lower = query.lower()
        
        # Detect query characteristics
        is_code_query = any(term in query_lower for term in [
            'function', 'method', 'class', 'implementation', 'algorithm',
            'bug', 'error', 'exception', 'def ', 'return', 'import'
        ])
        
        is_explanation_query = any(term in query_lower for term in [
            'what', 'how', 'why', 'explain', 'describe', 'understand'
        ])
        
        has_specific_symbols = any(term in query for term in [
            '(', ')', '{', '}', '.', '->', '=>', '::'
        ])
        
        # Determine processing strategy
        suitable_for_hyde = (
            len(query.split()) >= 3 and  # Reasonable length
            (is_code_query or is_explanation_query) and  # Appropriate type
            not has_specific_symbols  # Not too specific
        )
        
        query_type = "code" if is_code_query else "explanation" if is_explanation_query else "general"
        
        return {
            'type': query_type,
            'is_code_query': is_code_query,
            'is_explanation_query': is_explanation_query,
            'has_specific_symbols': has_specific_symbols,
            'suitable_for_hyde': suitable_for_hyde,
            'token_count': len(query.split()),
            'char_count': len(query),
        }
    
    async def _enhanced_hybrid_search(
        self,
        original_query: str,
        enhanced_query: str,
        vector_index: VectorIndex | None,
        serena_graph: SerenaGraph | None,
        repomap: RepoMap | None,
        repo_root: str,
        rev: str,
        features: FeatureConfig,
        k: int,
        context_lines: int,
    ) -> SearchResponse:
        """Execute hybrid search with enhanced embeddings."""
        
        # If we have enhanced embeddings capability and vector search is enabled
        if features.vector and vector_index and hasattr(self.code_embedder, 'search_similar_code'):
            try:
                # Use enhanced code embeddings for vector search
                enhanced_vector_results = await self.code_embedder.search_similar_code(
                    enhanced_query, vector_index, k=k*2  # Get more candidates for reranking
                )
                
                # Convert to search results format
                if enhanced_vector_results:
                    logger.debug(f"Enhanced embedding search found {len(enhanced_vector_results)} candidates")
                    
                    # Create temporary vector index with enhanced results
                    enhanced_chunks = [chunk for chunk, score in enhanced_vector_results]
                    temp_index = VectorIndex(chunks=enhanced_chunks, embedding_model=vector_index.embedding_model)
                    
                    # Use hybrid engine with enhanced vector index
                    return await self.hybrid_engine.search(
                        original_query, temp_index, serena_graph, repomap,
                        repo_root, rev, features, k, context_lines
                    )
            except Exception as e:
                logger.warning(f"Enhanced embedding search failed, falling back to standard: {e}")
        
        # Fallback to standard hybrid search
        return await self.hybrid_engine.search(
            enhanced_query, vector_index, serena_graph, repomap,
            repo_root, rev, features, k, context_lines
        )
    
    async def _enhance_search_response(
        self,
        response: SearchResponse,
        query_info: Dict[str, Any], 
        start_time: float
    ) -> SearchResponse:
        """Enhance search response with additional metadata."""
        
        # Add pipeline metadata
        pipeline_metadata = {
            'pipeline_version': '2.0',
            'query_type': query_info['type'],
            'hyde_used': self.enable_hyde and query_info['suitable_for_hyde'],
            'reranking_used': self.enable_reranking,
            'code_embeddings_used': isinstance(self.code_embedder, CodeEmbeddingAdapter),
            'processing_time_ms': (time.time() - start_time) * 1000,
        }
        
        # Update response
        enhanced_response = response.model_copy()
        
        # Add metadata to response if it has metadata field
        if hasattr(enhanced_response, 'metadata'):
            enhanced_response.metadata = enhanced_response.metadata or {}
            enhanced_response.metadata.update(pipeline_metadata)
        
        # Enhanced result scoring information
        for i, result in enumerate(enhanced_response.results):
            if hasattr(result, 'metadata'):
                result.metadata = result.metadata or {}
                result.metadata['enhanced_rank'] = i + 1
                result.metadata['pipeline_processed'] = True
        
        return enhanced_response
    
    def _record_pipeline_metrics(
        self,
        query_info: Dict[str, Any],
        result_count: int,
        execution_time: float
    ) -> None:
        """Record comprehensive pipeline metrics."""
        
        self.metrics_collector.record_search_request(
            "enhanced_pipeline", execution_time, result_count, "success"
        )
        
        # Record component usage
        if self.enable_hyde and query_info['suitable_for_hyde']:
            self.metrics_collector.record_feature_usage("hyde_transformation")
            
        if self.enable_reranking:
            self.metrics_collector.record_feature_usage("cross_encoder_reranking")
        
        # Record query characteristics  
        self.metrics_collector.record_query_analysis(
            query_info['type'],
            query_info['token_count'],
            query_info['suitable_for_hyde']
        )
    
    async def search_with_raptor(
        self,
        query: str,
        raptor_tree: Any,  # RaptorTree
        k: int = 20,
        traversal_strategy: str = "top_down"
    ) -> List[Tuple[VectorChunk, float]]:
        """
        Search using RAPTOR hierarchical structure.
        
        Args:
            query: Search query
            raptor_tree: RAPTOR tree structure
            k: Number of results
            traversal_strategy: Tree traversal strategy
            
        Returns:
            List of (chunk, score) tuples
        """
        try:
            # Transform query if HyDE is enabled
            enhanced_query = query
            if self.enable_hyde:
                enhanced_query = await self.hyde_transformer.transform_query(query)
            
            # Search RAPTOR tree
            from .raptor import RaptorProcessor
            
            processor = RaptorProcessor()
            tree_results = await processor.search_tree(
                enhanced_query, raptor_tree, k=k*2, strategy=traversal_strategy
            )
            
            # Rerank results if enabled
            if self.enable_reranking and len(tree_results) > 1:
                chunks = [chunk for chunk, _ in tree_results]
                reranked = await self.reranker.rerank_chunks(query, chunks, top_k=k)
                return reranked
            
            return tree_results[:k]
            
        except Exception as e:
            logger.error(f"RAPTOR search failed: {e}")
            return []
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get comprehensive pipeline status and capabilities."""
        
        # Get component status
        hyde_status = self.hyde_transformer.get_configuration() if hasattr(self.hyde_transformer, 'get_configuration') else {"enabled": False}
        reranker_status = self.reranker.get_model_info() if hasattr(self.reranker, 'get_model_info') else {"available": False}
        embedder_status = self.code_embedder.get_model_info() if hasattr(self.code_embedder, 'get_model_info') else {"initialized": False}
        
        return {
            'pipeline_version': '2.0',
            'components': {
                'hyde': hyde_status,
                'reranker': reranker_status,
                'code_embeddings': embedder_status,
                'hybrid_search': {'available': True}
            },
            'features': {
                'hyde_enabled': self.enable_hyde,
                'reranking_enabled': self.enable_reranking,
                'code_embeddings_available': hasattr(self.code_embedder, 'search_similar_code'),
            },
            'configuration': {
                'initial_retrieval_k': self.initial_retrieval_k,
                'final_k': self.final_k,
                'cache_embeddings': self.cache_embeddings,
            }
        }
    
    def configure_pipeline(
        self,
        enable_hyde: Optional[bool] = None,
        enable_reranking: Optional[bool] = None,
        initial_k: Optional[int] = None,
        final_k: Optional[int] = None
    ) -> None:
        """
        Configure pipeline settings at runtime.
        
        Args:
            enable_hyde: Enable/disable HyDE transformation
            enable_reranking: Enable/disable cross-encoder reranking
            initial_k: Initial retrieval count
            final_k: Final result count
        """
        if enable_hyde is not None:
            self.enable_hyde = enable_hyde
            
        if enable_reranking is not None:
            self.enable_reranking = enable_reranking
            
        if initial_k is not None:
            self.initial_retrieval_k = initial_k
            
        if final_k is not None:
            self.final_k = final_k
        
        logger.info(f"Pipeline configured: HyDE={self.enable_hyde}, Reranking={self.enable_reranking}")


# Convenience factory function
def create_enhanced_search_pipeline() -> EnhancedSearchPipeline:
    """Create and initialize enhanced search pipeline."""
    return EnhancedSearchPipeline()