"""
Advanced code-specific embedding models for enhanced semantic search.

Provides specialized embedding models optimized for code understanding,
including Microsoft CodeBERT, UniXcoder, and GraphCodeBERT variants.
"""

import asyncio
import hashlib
from typing import Any, Dict, List, Optional, Tuple

from sentence_transformers import SentenceTransformer
import torch

from ..config import get_ai_config
from ..data.schemas import VectorChunk, VectorIndex
from ..util.errors import MimirError, create_error_context
from ..util.logging_config import get_logger
from ..monitoring import get_metrics_collector


logger = get_logger(__name__)


class CodeEmbeddingError(MimirError):
    """Errors related to code embedding operations."""
    pass


class CodeEmbeddingAdapter:
    """
    Advanced code embedding adapter using specialized models.
    
    Supports multiple code-specific embedding models with optimized
    preprocessing for programming languages.
    """
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize code embedding adapter.
        
        Args:
            model_name: Override model name, otherwise uses config default
        """
        self.config = get_ai_config()
        self.model_name = model_name or self.config.embedding_model
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.metrics_collector = get_metrics_collector()
        
        # Code-specific preprocessing settings
        self.max_length = 512
        self.code_context_window = 256
        self.overlap_size = 64
        
        logger.info(f"Initializing CodeEmbeddingAdapter with model: {self.model_name} on {self.device}")
    
    async def initialize(self) -> None:
        """Initialize the embedding model asynchronously."""
        if self.model is not None:
            return
            
        try:
            # Run model loading in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, self._load_model
            )
            logger.info(f"Code embedding model loaded: {self.model_name}")
            
        except Exception as e:
            raise CodeEmbeddingError(
                f"Failed to initialize code embedding model: {e}",
                context=create_error_context(
                    component="code_embeddings",
                    operation="initialize",
                    parameters={"model_name": self.model_name}
                )
            ) from e
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        model = SentenceTransformer(self.model_name)
        model.to(self.device)
        return model
    
    async def embed_code_chunk(self, chunk: VectorChunk) -> VectorChunk:
        """
        Generate embeddings for a single code chunk with preprocessing.
        
        Args:
            chunk: Code chunk to embed
            
        Returns:
            Chunk with updated embedding
        """
        await self.initialize()
        
        try:
            # Preprocess code content for better embedding quality
            processed_content = self._preprocess_code(chunk.content)
            
            # Generate embedding in thread pool
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None, self._generate_embedding, processed_content
            )
            
            # Update chunk with new embedding
            updated_chunk = chunk.model_copy()
            updated_chunk.embedding = embedding.tolist()
            
            # Record metrics
            self.metrics_collector.record_embedding_generation(
                len(processed_content), 
                embedding.shape[0],
                self.model_name
            )
            
            return updated_chunk
            
        except Exception as e:
            raise CodeEmbeddingError(
                f"Failed to embed code chunk: {e}",
                context=create_error_context(
                    component="code_embeddings",
                    operation="embed_code_chunk",
                    parameters={"chunk_id": chunk.chunk_id, "path": chunk.path}
                )
            ) from e
    
    async def embed_code_chunks(self, chunks: List[VectorChunk]) -> List[VectorChunk]:
        """
        Generate embeddings for multiple code chunks efficiently.
        
        Args:
            chunks: List of code chunks to embed
            
        Returns:
            List of chunks with updated embeddings
        """
        await self.initialize()
        
        if not chunks:
            return []
        
        try:
            # Process in batches for memory efficiency
            batch_size = 32
            updated_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                batch_results = await self._process_batch(batch)
                updated_chunks.extend(batch_results)
                
                # Log progress for large datasets
                if len(chunks) > 100 and i % (batch_size * 10) == 0:
                    logger.info(f"Processed {i + len(batch)}/{len(chunks)} chunks")
            
            logger.info(f"Generated embeddings for {len(updated_chunks)} code chunks")
            return updated_chunks
            
        except Exception as e:
            raise CodeEmbeddingError(
                f"Failed to embed code chunks: {e}",
                context=create_error_context(
                    component="code_embeddings", 
                    operation="embed_code_chunks",
                    parameters={"chunk_count": len(chunks)}
                )
            ) from e
    
    async def _process_batch(self, batch: List[VectorChunk]) -> List[VectorChunk]:
        """Process a batch of chunks for embedding."""
        # Preprocess all contents
        processed_contents = [self._preprocess_code(chunk.content) for chunk in batch]
        
        # Generate embeddings in thread pool
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self._generate_batch_embeddings, processed_contents
        )
        
        # Update chunks with embeddings
        updated_chunks = []
        for chunk, embedding in zip(batch, embeddings):
            updated_chunk = chunk.model_copy()
            updated_chunk.embedding = embedding.tolist()
            updated_chunks.append(updated_chunk)
        
        return updated_chunks
    
    def _generate_embedding(self, content: str) -> torch.Tensor:
        """Generate single embedding."""
        return self.model.encode(content, convert_to_tensor=True)
    
    def _generate_batch_embeddings(self, contents: List[str]) -> List[torch.Tensor]:
        """Generate embeddings for batch of contents."""
        embeddings = self.model.encode(contents, convert_to_tensor=True)
        return [embeddings[i] for i in range(len(contents))]
    
    def _preprocess_code(self, content: str) -> str:
        """
        Preprocess code content for better embedding quality.
        
        Args:
            content: Raw code content
            
        Returns:
            Preprocessed content optimized for code embeddings
        """
        # Remove excessive whitespace but preserve structure
        lines = content.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Preserve indentation but normalize spaces
            stripped = line.rstrip()
            if stripped:
                # Replace multiple spaces with single space (except leading indentation)
                leading_whitespace = len(line) - len(line.lstrip())
                code_part = stripped[leading_whitespace:]
                normalized_code = ' '.join(code_part.split())
                cleaned_lines.append(line[:leading_whitespace] + normalized_code)
            else:
                cleaned_lines.append('')
        
        processed = '\n'.join(cleaned_lines)
        
        # Truncate if too long, preserving structure
        if len(processed) > self.max_length:
            processed = processed[:self.max_length]
            # Try to end at a reasonable boundary
            last_newline = processed.rfind('\n')
            if last_newline > self.max_length * 0.8:
                processed = processed[:last_newline]
        
        return processed
    
    async def search_similar_code(
        self, 
        query: str, 
        vector_index: VectorIndex, 
        k: int = 20
    ) -> List[Tuple[VectorChunk, float]]:
        """
        Search for semantically similar code using advanced embeddings.
        
        Args:
            query: Search query (can be natural language or code)
            vector_index: Index to search
            k: Number of results to return
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        await self.initialize()
        
        if not vector_index.chunks:
            return []
        
        try:
            # Preprocess query 
            processed_query = self._preprocess_code(query)
            
            # Generate query embedding
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None, self._generate_embedding, processed_query
            )
            
            # Calculate similarities
            similarities = await loop.run_in_executor(
                None, self._calculate_similarities, query_embedding, vector_index.chunks
            )
            
            # Sort and return top k
            results = [(chunk, sim) for chunk, sim in similarities if sim > 0.1]
            results.sort(key=lambda x: x[1], reverse=True)
            
            self.metrics_collector.record_search_request(
                "code_embedding", 
                0.0,  # Time will be recorded by caller
                len(results[:k]), 
                "success"
            )
            
            return results[:k]
            
        except Exception as e:
            raise CodeEmbeddingError(
                f"Failed to search similar code: {e}",
                context=create_error_context(
                    component="code_embeddings",
                    operation="search_similar_code", 
                    parameters={"query": query[:100], "chunk_count": len(vector_index.chunks)}
                )
            ) from e
    
    def _calculate_similarities(
        self, 
        query_embedding: torch.Tensor, 
        chunks: List[VectorChunk]
    ) -> List[Tuple[VectorChunk, float]]:
        """Calculate cosine similarities between query and chunks."""
        similarities = []
        
        for chunk in chunks:
            if not chunk.embedding:
                continue
                
            chunk_embedding = torch.tensor(chunk.embedding)
            
            # Cosine similarity
            similarity = torch.cosine_similarity(
                query_embedding.unsqueeze(0), 
                chunk_embedding.unsqueeze(0)
            ).item()
            
            similarities.append((chunk, similarity))
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from this model."""
        if self.model is None:
            # Common dimensions for code models
            model_dims = {
                "microsoft/codebert-base": 768,
                "microsoft/graphcodebert-base": 768,
                "microsoft/unixcoder-base": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "sentence-transformers/all-MiniLM-L6-v2": 384,
            }
            return model_dims.get(self.model_name, 768)
        
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current embedding model."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "dimension": self.get_embedding_dimension(),
            "max_length": self.max_length,
            "is_code_specific": "code" in self.model_name.lower(),
            "initialized": self.model is not None,
        }