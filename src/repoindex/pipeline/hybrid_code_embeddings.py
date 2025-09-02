"""
Hybrid Code Embeddings with Lens Vector Services Integration.

This module enhances the existing CodeEmbeddingAdapter with Lens vector services,
providing high-performance embedding generation and storage while maintaining
Mimir's specialized code understanding capabilities.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .code_embeddings import CodeEmbeddingAdapter, CodeEmbeddingError
from .lens_client import LensIntegrationClient, get_lens_client, LensResponse
from .parallel_processor import ParallelProcessor, get_parallel_processor, TaskPriority
from .stage import AsyncPipelineStage, ProgressCallback
from ..data.schemas import VectorChunk, VectorIndex, PipelineStage
from ..util.log import get_logger
from ..util.errors import MimirError

if TYPE_CHECKING:
    from .run import PipelineContext

logger = get_logger(__name__)


@dataclass
class EmbeddingMetrics:
    """Metrics for hybrid embedding operations."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Chunk metrics
    total_chunks: int = 0
    lens_processed_chunks: int = 0
    mimir_processed_chunks: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Performance metrics
    lens_embedding_time_ms: float = 0.0
    mimir_embedding_time_ms: float = 0.0
    parallel_processing_time_ms: float = 0.0
    average_chunk_time_ms: float = 0.0
    
    # Quality metrics
    embedding_dimension: int = 0
    lens_success_rate: float = 0.0
    mimir_success_rate: float = 1.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    gpu_utilization: float = 0.0
    batch_sizes_used: List[int] = field(default_factory=list)
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.total_chunks > 0:
            total_time = (self.end_time - self.start_time).total_seconds() * 1000
            self.average_chunk_time_ms = total_time / self.total_chunks


@dataclass
class EmbeddingBatch:
    """Batch of chunks for embedding processing."""
    chunks: List[VectorChunk]
    batch_id: str
    priority: TaskPriority = TaskPriority.NORMAL
    use_lens: bool = True
    use_mimir: bool = True
    
    @property
    def size(self) -> int:
        return len(self.chunks)


class HybridCodeEmbeddingStage(AsyncPipelineStage):
    """
    Hybrid code embedding stage with Lens vector services integration.
    
    This stage combines Mimir's specialized code embedding capabilities
    with Lens's high-performance vector services for optimal embedding
    generation and storage.
    
    Features:
    - Parallel Lens and Mimir embedding generation
    - Intelligent caching and deduplication
    - Batch processing optimization
    - Embedding quality validation
    - Performance monitoring and metrics
    """
    
    def __init__(
        self,
        stage_type: PipelineStage = PipelineStage.CODE_EMBEDDINGS,
        concurrency_limit: int = 4,
        lens_client: Optional[LensIntegrationClient] = None,
        mimir_adapter: Optional[CodeEmbeddingAdapter] = None,
        enable_lens_vectors: bool = True,
        batch_size: int = 32,
        max_chunk_length: int = 512,
        embedding_cache_size: int = 10000
    ):
        """Initialize hybrid code embedding stage."""
        super().__init__(stage_type, concurrency_limit)
        
        self.lens_client = lens_client or get_lens_client()
        self.mimir_adapter = mimir_adapter or CodeEmbeddingAdapter()
        self.enable_lens_vectors = enable_lens_vectors
        self.batch_size = batch_size
        self.max_chunk_length = max_chunk_length
        self.embedding_cache_size = embedding_cache_size
        
        # Internal state
        self._parallel_processor: Optional[ParallelProcessor] = None
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._thread_pool: Optional[ThreadPoolExecutor] = None
        
        # Performance optimization
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimal_batch_size = self._calculate_optimal_batch_size()
        
        logger.info(
            f"HybridCodeEmbeddingStage initialized with Lens vectors: {enable_lens_vectors}, "
            f"batch size: {batch_size}, device: {self.device}"
        )
    
    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """Execute hybrid code embedding stage."""
        logger.info("Starting hybrid code embedding execution")
        start_time = time.time()
        
        try:
            # Initialize components
            await self._initialize_components()
            
            # Get chunks from context
            chunks = self._extract_chunks_from_context(context)
            if not chunks:
                logger.warning("No chunks found for embedding")
                return
            
            # Execute hybrid embedding
            embedded_chunks = await self._execute_hybrid_embedding(
                chunks, context, progress_callback
            )
            
            # Store results in context
            self._store_results_in_context(context, embedded_chunks)
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(
                f"Hybrid embedding completed in {execution_time:.2f}ms. "
                f"Processed {len(embedded_chunks)} chunks"
            )
            
        except Exception as e:
            logger.error(f"Hybrid embedding execution failed: {e}")
            raise MimirError(f"Code embedding stage failed: {e}")
    
    async def _initialize_components(self) -> None:
        """Initialize all components for hybrid embedding."""
        # Initialize Mimir adapter
        await self.mimir_adapter.initialize()
        
        # Get parallel processor
        self._parallel_processor = await get_parallel_processor()
        
        # Initialize thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info("All embedding components initialized")
    
    def _extract_chunks_from_context(self, context: "PipelineContext") -> List[VectorChunk]:
        """Extract chunks from pipeline context."""
        # Look for chunks in various context attributes
        chunks = []
        
        if hasattr(context, 'vector_chunks'):
            chunks = context.vector_chunks
        elif hasattr(context, 'chunks'):
            chunks = context.chunks
        elif hasattr(context, 'discovery_result'):
            # Extract chunks from discovery result if available
            discovery = context.discovery_result
            if hasattr(discovery, 'chunks'):
                chunks = discovery.chunks
        
        logger.info(f"Extracted {len(chunks)} chunks from context")
        return chunks
    
    async def _execute_hybrid_embedding(
        self,
        chunks: List[VectorChunk],
        context: "PipelineContext",
        progress_callback: ProgressCallback | None = None
    ) -> List[VectorChunk]:
        """Execute the core hybrid embedding logic."""
        metrics = EmbeddingMetrics()
        metrics.total_chunks = len(chunks)
        
        try:
            # Phase 1: Prepare batches (10% progress)
            batches = await self._prepare_embedding_batches(chunks)
            self._update_progress(10, progress_callback)
            
            # Phase 2: Check cache and filter (20% progress)
            cache_results = await self._check_embedding_cache(chunks)
            chunks_to_process = [
                chunk for chunk, cached in zip(chunks, cache_results) 
                if not cached
            ]
            metrics.cache_hits = len(chunks) - len(chunks_to_process)
            metrics.cache_misses = len(chunks_to_process)
            self._update_progress(20, progress_callback)
            
            # Phase 3: Parallel embedding generation (80% progress)
            if chunks_to_process:
                embedded_results = await self._execute_parallel_embedding(
                    chunks_to_process, batches, progress_callback
                )
                metrics.lens_processed_chunks = embedded_results.get('lens_count', 0)
                metrics.mimir_processed_chunks = embedded_results.get('mimir_count', 0)
                metrics.lens_embedding_time_ms = embedded_results.get('lens_time_ms', 0.0)
                metrics.mimir_embedding_time_ms = embedded_results.get('mimir_time_ms', 0.0)
            else:
                embedded_results = {'processed_chunks': []}
            
            # Phase 4: Combine results and finalize (100% progress)
            final_chunks = await self._combine_embedding_results(
                chunks, cache_results, embedded_results.get('processed_chunks', [])
            )
            
            self._update_progress(100, progress_callback)
            
            # Update metrics
            metrics.finalize()
            if hasattr(context, 'embedding_metrics'):
                context.embedding_metrics = metrics
            
            logger.info(
                f"Hybrid embedding completed: {len(final_chunks)} chunks, "
                f"{metrics.cache_hits} cache hits, {metrics.cache_misses} processed"
            )
            
            return final_chunks
            
        except Exception as e:
            logger.error(f"Hybrid embedding failed: {e}")
            raise CodeEmbeddingError(f"Embedding generation failed: {e}")
    
    async def _prepare_embedding_batches(self, chunks: List[VectorChunk]) -> List[EmbeddingBatch]:
        """Prepare optimized batches for embedding processing."""
        logger.info(f"Preparing embedding batches for {len(chunks)} chunks")
        
        batches = []
        
        # Group chunks by characteristics for optimal batching
        code_chunks = []
        text_chunks = []
        large_chunks = []
        
        for chunk in chunks:
            content_length = len(chunk.content)
            
            if content_length > self.max_chunk_length:
                large_chunks.append(chunk)
            elif self._is_code_chunk(chunk):
                code_chunks.append(chunk)
            else:
                text_chunks.append(chunk)
        
        # Create optimized batches
        batch_id = 0
        
        # Code chunks - use smaller batches for better GPU utilization
        for i in range(0, len(code_chunks), self.batch_size):
            batch_chunks = code_chunks[i:i + self.batch_size]
            batches.append(EmbeddingBatch(
                chunks=batch_chunks,
                batch_id=f"code_batch_{batch_id}",
                priority=TaskPriority.HIGH,
                use_lens=self.enable_lens_vectors,
                use_mimir=True
            ))
            batch_id += 1
        
        # Text chunks - can use larger batches
        text_batch_size = min(self.batch_size * 2, 64)
        for i in range(0, len(text_chunks), text_batch_size):
            batch_chunks = text_chunks[i:i + text_batch_size]
            batches.append(EmbeddingBatch(
                chunks=batch_chunks,
                batch_id=f"text_batch_{batch_id}",
                priority=TaskPriority.NORMAL,
                use_lens=self.enable_lens_vectors,
                use_mimir=True
            ))
            batch_id += 1
        
        # Large chunks - process individually with special handling
        for chunk in large_chunks:
            batches.append(EmbeddingBatch(
                chunks=[chunk],
                batch_id=f"large_chunk_{batch_id}",
                priority=TaskPriority.LOW,
                use_lens=self.enable_lens_vectors,
                use_mimir=True
            ))
            batch_id += 1
        
        logger.info(f"Created {len(batches)} optimized embedding batches")
        return batches
    
    def _is_code_chunk(self, chunk: VectorChunk) -> bool:
        """Determine if chunk contains code content."""
        # Simple heuristics to identify code chunks
        content = chunk.content.lower()
        
        # Check for code-specific patterns
        code_indicators = [
            'function', 'def ', 'class ', 'import ', 'from ',
            '{', '}', ';', '->', '=>', '/*', '//', '#',
            'const ', 'let ', 'var ', 'public ', 'private '
        ]
        
        code_score = sum(1 for indicator in code_indicators if indicator in content)
        return code_score >= 3 or chunk.chunk_type == 'code'
    
    async def _check_embedding_cache(self, chunks: List[VectorChunk]) -> List[bool]:
        """Check which chunks already have cached embeddings."""
        cache_results = []
        
        for chunk in chunks:
            # Generate cache key based on content hash
            cache_key = self._generate_cache_key(chunk)
            
            # Check if embedding is cached
            has_embedding = (
                chunk.embedding is not None and len(chunk.embedding) > 0
            ) or cache_key in self._embedding_cache
            
            cache_results.append(has_embedding)
        
        cache_hits = sum(cache_results)
        logger.info(f"Embedding cache check: {cache_hits} hits, {len(chunks) - cache_hits} misses")
        
        return cache_results
    
    def _generate_cache_key(self, chunk: VectorChunk) -> str:
        """Generate cache key for chunk."""
        import hashlib
        content_hash = hashlib.sha256(chunk.content.encode()).hexdigest()
        return f"embedding_{content_hash}_{self.mimir_adapter.model_name}"
    
    async def _execute_parallel_embedding(
        self,
        chunks: List[VectorChunk],
        batches: List[EmbeddingBatch],
        progress_callback: ProgressCallback | None = None
    ) -> Dict[str, Any]:
        """Execute parallel embedding generation using Lens and Mimir."""
        logger.info(f"Starting parallel embedding for {len(chunks)} chunks in {len(batches)} batches")
        
        if not self._parallel_processor:
            raise MimirError("ParallelProcessor not initialized")
        
        start_time = time.time()
        results = {
            'processed_chunks': [],
            'lens_count': 0,
            'mimir_count': 0,
            'lens_time_ms': 0.0,
            'mimir_time_ms': 0.0
        }
        
        # Submit batch processing tasks
        task_ids = []
        
        for batch in batches:
            if self.enable_lens_vectors and await self._check_lens_availability():
                # Submit Lens embedding task
                lens_task_id = await self._parallel_processor.submit_task(
                    self._process_batch_with_lens,
                    batch,
                    task_id=f"lens_{batch.batch_id}",
                    priority=batch.priority,
                    timeout=120.0
                )
                task_ids.append(('lens', batch, lens_task_id))
            
            # Always submit Mimir embedding task as fallback
            mimir_task_id = await self._parallel_processor.submit_task(
                self._process_batch_with_mimir,
                batch,
                task_id=f"mimir_{batch.batch_id}",
                priority=batch.priority,
                timeout=180.0
            )
            task_ids.append(('mimir', batch, mimir_task_id))
        
        # Process results with progress updates
        completed_tasks = 0
        total_tasks = len(task_ids)
        
        for source, batch, task_id in task_ids:
            try:
                result = await self._parallel_processor.get_result(task_id, timeout=300.0)
                
                if result.get('success'):
                    # Use Lens result if available and successful, otherwise use Mimir
                    if source == 'lens' and not any(
                        chunk.chunk_id in [c.chunk_id for c in results['processed_chunks']]
                        for chunk in batch.chunks
                    ):
                        results['processed_chunks'].extend(result.get('chunks', []))
                        results['lens_count'] += len(result.get('chunks', []))
                        results['lens_time_ms'] += result.get('execution_time_ms', 0.0)
                    
                    elif source == 'mimir':
                        # Add Mimir chunks if not already processed by Lens
                        mimir_chunks = result.get('chunks', [])
                        existing_ids = {c.chunk_id for c in results['processed_chunks']}
                        
                        new_chunks = [
                            c for c in mimir_chunks 
                            if c.chunk_id not in existing_ids
                        ]
                        
                        results['processed_chunks'].extend(new_chunks)
                        results['mimir_count'] += len(new_chunks)
                        results['mimir_time_ms'] += result.get('execution_time_ms', 0.0)
                    
                    logger.debug(f"Task {source}_{batch.batch_id} completed successfully")
                else:
                    logger.warning(f"Task {source}_{batch.batch_id} failed: {result.get('error')}")
                
                completed_tasks += 1
                
                # Update progress
                if progress_callback:
                    base_progress = 20  # Starting after cache check
                    task_progress = int(60 * (completed_tasks / total_tasks))  # 60% for parallel embedding
                    progress_callback(base_progress + task_progress)
                
            except Exception as e:
                logger.error(f"Task {source}_{batch.batch_id} failed: {e}")
                completed_tasks += 1
        
        total_time_ms = (time.time() - start_time) * 1000
        results['total_time_ms'] = total_time_ms
        
        logger.info(
            f"Parallel embedding completed: {results['lens_count']} via Lens, "
            f"{results['mimir_count']} via Mimir, {total_time_ms:.2f}ms total"
        )
        
        return results
    
    async def _check_lens_availability(self) -> bool:
        """Check if Lens vector services are available."""
        try:
            health_check = await self.lens_client.health_check()
            return health_check.status.value in ['healthy', 'degraded']
        except Exception as e:
            logger.warning(f"Lens availability check failed: {e}")
            return False
    
    async def _process_batch_with_lens(self, batch: EmbeddingBatch) -> Dict[str, Any]:
        """Process embedding batch using Lens vector services."""
        logger.debug(f"Processing batch {batch.batch_id} with Lens ({batch.size} chunks)")
        
        start_time = time.time()
        
        try:
            # Prepare documents for Lens
            documents = []
            for chunk in batch.chunks:
                doc = {
                    'id': chunk.chunk_id,
                    'content': chunk.content,
                    'metadata': {
                        'file_path': chunk.file_path,
                        'chunk_type': chunk.chunk_type,
                        'start_line': chunk.start_line,
                        'end_line': chunk.end_line
                    }
                }
                documents.append(doc)
            
            # Send to Lens for embedding
            response = await self.lens_client.generate_embeddings(documents)
            
            if response.success and response.data:
                # Convert Lens response to VectorChunk format
                embedded_chunks = []
                embeddings_data = response.data.get('embeddings', [])
                
                for i, chunk in enumerate(batch.chunks):
                    if i < len(embeddings_data):
                        embedding = embeddings_data[i]
                        
                        # Update chunk with embedding
                        chunk.embedding = np.array(embedding.get('vector', []))
                        chunk.embedding_model = embedding.get('model', 'lens_default')
                        
                        # Cache the embedding
                        cache_key = self._generate_cache_key(chunk)
                        self._embedding_cache[cache_key] = chunk.embedding
                        
                        embedded_chunks.append(chunk)
                
                execution_time_ms = (time.time() - start_time) * 1000
                
                return {
                    'success': True,
                    'chunks': embedded_chunks,
                    'source': 'lens',
                    'execution_time_ms': execution_time_ms,
                    'batch_id': batch.batch_id
                }
            else:
                raise Exception(f"Lens embedding failed: {response.error}")
                
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Lens batch processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'source': 'lens',
                'execution_time_ms': execution_time_ms,
                'batch_id': batch.batch_id
            }
    
    async def _process_batch_with_mimir(self, batch: EmbeddingBatch) -> Dict[str, Any]:
        """Process embedding batch using Mimir's CodeEmbeddingAdapter."""
        logger.debug(f"Processing batch {batch.batch_id} with Mimir ({batch.size} chunks)")
        
        start_time = time.time()
        
        try:
            embedded_chunks = []
            
            # Process chunks sequentially with Mimir adapter
            for chunk in batch.chunks:
                try:
                    embedded_chunk = await self.mimir_adapter.embed_code_chunk(chunk)
                    
                    # Cache the embedding
                    cache_key = self._generate_cache_key(embedded_chunk)
                    if embedded_chunk.embedding is not None:
                        self._embedding_cache[cache_key] = embedded_chunk.embedding
                    
                    embedded_chunks.append(embedded_chunk)
                    
                except Exception as chunk_error:
                    logger.warning(f"Failed to embed chunk {chunk.chunk_id}: {chunk_error}")
                    # Add chunk without embedding to maintain consistency
                    embedded_chunks.append(chunk)
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'chunks': embedded_chunks,
                'source': 'mimir',
                'execution_time_ms': execution_time_ms,
                'batch_id': batch.batch_id
            }
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Mimir batch processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'source': 'mimir',
                'execution_time_ms': execution_time_ms,
                'batch_id': batch.batch_id
            }
    
    async def _combine_embedding_results(
        self,
        original_chunks: List[VectorChunk],
        cache_results: List[bool],
        processed_chunks: List[VectorChunk]
    ) -> List[VectorChunk]:
        """Combine cached and newly processed embeddings."""
        logger.info("Combining embedding results from cache and processing")
        
        final_chunks = []
        processed_index = 0
        
        for i, (chunk, was_cached) in enumerate(zip(original_chunks, cache_results)):
            if was_cached:
                # Use existing embedding from chunk or cache
                if chunk.embedding is None:
                    cache_key = self._generate_cache_key(chunk)
                    if cache_key in self._embedding_cache:
                        chunk.embedding = self._embedding_cache[cache_key]
                
                final_chunks.append(chunk)
            else:
                # Use processed chunk if available
                if processed_index < len(processed_chunks):
                    final_chunks.append(processed_chunks[processed_index])
                    processed_index += 1
                else:
                    # Fallback to original chunk if processing failed
                    final_chunks.append(chunk)
        
        # Validate embedding quality
        embedding_count = sum(1 for chunk in final_chunks if chunk.embedding is not None)
        logger.info(f"Combined results: {embedding_count}/{len(final_chunks)} chunks have embeddings")
        
        return final_chunks
    
    def _store_results_in_context(
        self, 
        context: "PipelineContext", 
        embedded_chunks: List[VectorChunk]
    ) -> None:
        """Store embedding results in pipeline context."""
        # Store in multiple context attributes for compatibility
        context.vector_chunks = embedded_chunks
        context.embedded_chunks = embedded_chunks
        
        # Update vector index if exists
        if hasattr(context, 'vector_index'):
            if context.vector_index is None:
                context.vector_index = VectorIndex()
            context.vector_index.chunks = embedded_chunks
        
        logger.info(f"Stored {len(embedded_chunks)} embedded chunks in context")
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available resources."""
        if self.device == "cuda":
            try:
                # Get GPU memory info
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024 ** 3)
                
                # Estimate batch size based on GPU memory
                if gpu_memory_gb >= 8:
                    return min(self.batch_size * 2, 64)
                elif gpu_memory_gb >= 4:
                    return self.batch_size
                else:
                    return max(self.batch_size // 2, 8)
            except:
                return self.batch_size
        else:
            # CPU-only processing - smaller batches
            return max(self.batch_size // 2, 4)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid embedding stage metrics."""
        base_metrics = super().get_metrics()
        
        hybrid_metrics = {
            'hybrid_embedding': {
                'lens_vectors_enabled': self.enable_lens_vectors,
                'batch_size': self.batch_size,
                'optimal_batch_size': self.optimal_batch_size,
                'max_chunk_length': self.max_chunk_length,
                'cache_size': len(self._embedding_cache),
                'cache_limit': self.embedding_cache_size,
                'device': self.device,
                'mimir_model': self.mimir_adapter.model_name if self.mimir_adapter else None
            }
        }
        
        base_metrics.update(hybrid_metrics)
        return base_metrics
    
    def _get_capabilities(self) -> List[str]:
        """Get hybrid embedding capabilities."""
        capabilities = super()._get_capabilities()
        capabilities.extend([
            'hybrid_embeddings',
            'lens_vector_services',
            'code_specialized_embeddings',
            'batch_processing',
            'embedding_caching',
            'performance_optimization',
            'gpu_acceleration',
            'parallel_processing'
        ])
        return capabilities
    
    async def cleanup(self, context: "PipelineContext") -> None:
        """Clean up hybrid embedding resources."""
        logger.info("Cleaning up hybrid embedding stage")
        
        # Clear embedding cache if too large
        if len(self._embedding_cache) > self.embedding_cache_size:
            # Keep only recent embeddings
            cache_items = list(self._embedding_cache.items())
            self._embedding_cache = dict(cache_items[-self.embedding_cache_size:])
            logger.info(f"Trimmed embedding cache to {len(self._embedding_cache)} items")
        
        # Shutdown thread pool
        if self._thread_pool:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None
        
        # Reset processor reference
        self._parallel_processor = None
        
        await super().cleanup(context)