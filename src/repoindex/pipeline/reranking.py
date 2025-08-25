"""
Cross-encoder reranking for search result refinement.

Implements cross-encoder models for reranking search results based on
query-document relevance, significantly improving search quality.
"""

import asyncio
import time
from typing import List, Tuple, Optional, Dict, Any

try:
    from sentence_transformers import CrossEncoder
    import torch
    RERANKING_AVAILABLE = True
except ImportError:
    RERANKING_AVAILABLE = False
    CrossEncoder = None
    torch = None

from ..config import get_ai_config
from ..data.schemas import VectorChunk, SearchResult
from ..util.errors import MimirError, create_error_context
from ..util.logging_config import get_logger
from ..monitoring import get_metrics_collector


logger = get_logger(__name__)


class RerankingError(MimirError):
    """Errors related to reranking operations."""
    pass


class CrossEncoderReranker:
    """
    Cross-encoder based reranking for search results.
    
    Uses cross-encoder models that jointly process query and document
    pairs to produce more accurate relevance scores than bi-encoder
    similarity alone.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize cross-encoder reranker.
        
        Args:
            model_name: Model name override, uses config default if None
        """
        if not RERANKING_AVAILABLE:
            logger.warning("Reranking dependencies not available (sentence-transformers, torch)")
            self.available = False
            return
            
        self.config = get_ai_config()
        self.model_name = model_name or self.config.reranker.model
        self.model = None
        self.device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        self.metrics_collector = get_metrics_collector()
        self.available = True
        
        # Reranking parameters
        self.batch_size = 32
        self.max_query_length = 256
        self.max_document_length = 512
        
        logger.info(f"Initializing CrossEncoderReranker with model: {self.model_name} on {self.device}")
    
    async def initialize(self) -> None:
        """Initialize the cross-encoder model asynchronously."""
        if not self.available:
            raise RerankingError("Reranking dependencies not available")
            
        if not self.config.reranker.enabled:
            logger.info("Reranking is disabled in configuration")
            return
            
        if self.model is not None:
            return
            
        try:
            # Load model in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_model
            )
            logger.info(f"Cross-encoder model loaded: {self.model_name}")
            
        except Exception as e:
            raise RerankingError(
                f"Failed to initialize reranking model: {e}",
                context=create_error_context(
                    component="reranking",
                    operation="initialize",
                    parameters={"model_name": self.model_name}
                )
            ) from e
    
    def _load_model(self) -> CrossEncoder:
        """Load the cross-encoder model."""
        model = CrossEncoder(self.model_name, device=self.device)
        return model
    
    async def rerank_chunks(
        self, 
        query: str, 
        chunks: List[VectorChunk], 
        top_k: Optional[int] = None
    ) -> List[Tuple[VectorChunk, float]]:
        """
        Rerank vector chunks using cross-encoder scoring.
        
        Args:
            query: Search query
            chunks: List of chunks to rerank
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List of (chunk, reranking_score) tuples sorted by relevance
        """
        if not self.available or not self.config.reranker.enabled:
            logger.debug("Reranking disabled, returning chunks with dummy scores")
            return [(chunk, 0.5) for chunk in chunks]
        
        await self.initialize()
        
        if not chunks:
            return []
            
        if self.model is None:
            logger.warning("Reranking model not available")
            return [(chunk, 0.5) for chunk in chunks]
        
        top_k = top_k or self.config.reranker.top_k
        start_time = time.time()
        
        try:
            # Prepare query-document pairs
            pairs = self._prepare_pairs(query, chunks)
            
            # Generate reranking scores in batches
            scores = await self._score_pairs_batched(pairs)
            
            # Combine chunks with scores
            chunk_scores = [(chunks[i], scores[i]) for i in range(len(chunks))]
            
            # Sort by reranking score (descending)
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            
            execution_time = time.time() - start_time
            
            # Record metrics
            self.metrics_collector.record_reranking_operation(
                len(chunks),
                top_k,
                execution_time,
                self.model_name
            )
            
            logger.debug(f"Reranked {len(chunks)} chunks in {execution_time:.3f}s")
            
            return chunk_scores[:top_k]
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Fallback to original ordering with dummy scores
            return [(chunk, 0.5) for chunk in chunks[:top_k]]
    
    async def rerank_search_results(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """
        Rerank search results using cross-encoder scoring.
        
        Args:
            query: Search query
            results: List of search results to rerank
            top_k: Number of top results to return
            
        Returns:
            Reranked list of search results with updated scores
        """
        if not self.available or not self.config.reranker.enabled or not results:
            return results[:top_k] if top_k else results
        
        await self.initialize()
        top_k = top_k or self.config.reranker.top_k
        
        try:
            # Extract content for reranking
            contents = [result.content.text for result in results]
            pairs = [(query, content) for content in contents]
            
            # Generate reranking scores
            scores = await self._score_pairs_batched(pairs)
            
            # Update results with new scores
            reranked_results = []
            for i, result in enumerate(results):
                # Create updated result with reranking score
                updated_result = result.model_copy()
                updated_result.score = scores[i]
                # Store original score in metadata if needed
                if hasattr(updated_result, 'metadata'):
                    updated_result.metadata = updated_result.metadata or {}
                    updated_result.metadata['original_score'] = result.score
                reranked_results.append(updated_result)
            
            # Sort by new reranking scores
            reranked_results.sort(key=lambda r: r.score, reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Search result reranking failed: {e}")
            return results[:top_k]
    
    def _prepare_pairs(self, query: str, chunks: List[VectorChunk]) -> List[Tuple[str, str]]:
        """Prepare query-document pairs for cross-encoder scoring."""
        # Truncate query if too long
        if len(query) > self.max_query_length:
            query = query[:self.max_query_length]
        
        pairs = []
        for chunk in chunks:
            # Prepare document content
            content = chunk.content
            if len(content) > self.max_document_length:
                content = content[:self.max_document_length]
            
            pairs.append((query, content))
        
        return pairs
    
    async def _score_pairs_batched(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """Score query-document pairs in batches."""
        if self.model is None:
            return [0.5] * len(pairs)
        
        all_scores = []
        
        # Process in batches to manage memory
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            
            # Run scoring in thread pool
            loop = asyncio.get_event_loop()
            batch_scores = await loop.run_in_executor(
                None, self._score_batch, batch
            )
            
            all_scores.extend(batch_scores)
        
        return all_scores
    
    def _score_batch(self, batch: List[Tuple[str, str]]) -> List[float]:
        """Score a batch of query-document pairs."""
        try:
            scores = self.model.predict(batch)
            
            # Convert to Python float list
            if hasattr(scores, 'tolist'):
                return scores.tolist()
            elif isinstance(scores, (list, tuple)):
                return [float(s) for s in scores]
            else:
                return [float(scores)] if len(batch) == 1 else [0.5] * len(batch)
                
        except Exception as e:
            logger.warning(f"Batch scoring failed: {e}")
            return [0.5] * len(batch)
    
    async def warm_up(self) -> None:
        """Warm up the model with a dummy prediction."""
        if not self.available or self.model is None:
            return
            
        try:
            dummy_pairs = [("test query", "test document content")]
            await self._score_pairs_batched(dummy_pairs)
            logger.debug("Cross-encoder model warmed up")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranking model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "available": self.available,
            "enabled": self.config.reranker.enabled if hasattr(self.config, 'reranker') else False,
            "batch_size": self.batch_size,
            "initialized": self.model is not None,
            "top_k": self.config.reranker.top_k if hasattr(self.config, 'reranker') else None,
        }


class FallbackReranker:
    """
    Fallback reranker for when cross-encoder models are not available.
    
    Uses simple heuristics like term frequency, position, and length
    to provide basic reranking capabilities.
    """
    
    def __init__(self):
        """Initialize fallback reranker."""
        self.metrics_collector = get_metrics_collector()
        logger.info("Initialized FallbackReranker")
    
    async def rerank_chunks(
        self, 
        query: str, 
        chunks: List[VectorChunk], 
        top_k: Optional[int] = None
    ) -> List[Tuple[VectorChunk, float]]:
        """Rerank chunks using simple heuristics."""
        if not chunks:
            return []
        
        config = get_ai_config()
        top_k = top_k or (config.reranker.top_k if hasattr(config, 'reranker') else 20)
        
        start_time = time.time()
        
        # Simple heuristic scoring
        query_tokens = set(query.lower().split())
        scored_chunks = []
        
        for chunk in chunks:
            score = self._calculate_heuristic_score(query_tokens, chunk)
            scored_chunks.append((chunk, score))
        
        # Sort by score
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        execution_time = time.time() - start_time
        
        # Record metrics
        self.metrics_collector.record_reranking_operation(
            len(chunks),
            top_k, 
            execution_time,
            "fallback_heuristic"
        )
        
        return scored_chunks[:top_k]
    
    def _calculate_heuristic_score(self, query_tokens: set, chunk: VectorChunk) -> float:
        """Calculate heuristic relevance score."""
        content_lower = chunk.content.lower()
        content_tokens = set(content_lower.split())
        
        # Term frequency score
        matching_tokens = query_tokens & content_tokens
        tf_score = len(matching_tokens) / max(len(query_tokens), 1)
        
        # Length penalty (prefer moderate length)
        length_score = 1.0
        content_length = len(chunk.content)
        if content_length < 50:
            length_score = 0.7  # Too short
        elif content_length > 2000:
            length_score = 0.8  # Too long
        
        # Position bonus (prefer earlier content)
        position_score = 1.0
        if hasattr(chunk, 'span') and chunk.span:
            # Earlier positions get slight bonus
            position_score = 1.0 - min(chunk.span[0] / 10000, 0.2)
        
        # Combine scores
        final_score = tf_score * 0.7 + length_score * 0.2 + position_score * 0.1
        
        return final_score


def create_reranker(model_name: Optional[str] = None) -> CrossEncoderReranker | FallbackReranker:
    """
    Create appropriate reranker based on availability.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        CrossEncoderReranker if available, otherwise FallbackReranker
    """
    if RERANKING_AVAILABLE:
        return CrossEncoderReranker(model_name)
    else:
        logger.info("Cross-encoder reranking not available, using fallback reranker")
        return FallbackReranker()