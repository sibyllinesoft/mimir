"""
LEANN adapter for CPU-based vector embeddings.

Provides integration with LEANN for function-level code chunking
and CPU-only vector embedding generation with content ordering.
"""

import asyncio
import json
import pickle
import struct
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import uuid

from ..data.schemas import IndexConfig, VectorIndex, VectorChunk
from ..util.fs import atomic_write_json, atomic_write_bytes


class LEANNError(Exception):
    """LEANN operation error."""
    pass


class LEANNAdapter:
    """
    Adapter for LEANN CPU-based vector embeddings.
    
    Manages code chunking, embedding generation, and vector index creation
    with support for RepoMapper-based file prioritization.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize LEANN adapter with embedding model."""
        self.model_name = model_name
        self.chunk_size = 400  # tokens
        self.overlap_size = 64  # tokens
        self.dimension = 384  # for all-MiniLM-L6-v2
        
        # Initialize embedding model (would normally load actual model)
        self._model = None
        self._tokenizer = None
    
    async def build_index(
        self,
        repo_root: Path,
        files: List[str],
        repomap_order: List[str],
        work_dir: Path,
        config: IndexConfig,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> VectorIndex:
        """
        Build complete vector index for repository files.
        
        Processes files in RepoMapper order for optimal embedding quality.
        """
        if progress_callback:
            progress_callback(5)
        
        # Filter and prioritize files
        processable_files = await self._filter_processable_files(
            repo_root, files, repomap_order
        )
        
        if not processable_files:
            # Return empty index if no processable files
            return VectorIndex(
                chunks=[],
                dimension=self.dimension,
                total_tokens=0,
                model_name=self.model_name
            )
        
        if progress_callback:
            progress_callback(10)
        
        # Apply max files limit if specified
        if config.max_files_to_embed:
            processable_files = processable_files[:config.max_files_to_embed]
        
        # Chunk all files
        all_chunks = []
        total_tokens = 0
        
        for i, file_path in enumerate(processable_files):
            try:
                file_chunks = await self._chunk_file(repo_root, file_path)
                all_chunks.extend(file_chunks)
                total_tokens += sum(chunk.token_count for chunk in file_chunks)
                
                # Update progress
                if progress_callback:
                    file_progress = int(20 + (i / len(processable_files)) * 30)
                    progress_callback(file_progress)
                    
            except Exception as e:
                print(f"Warning: Failed to chunk file {file_path}: {e}")
                continue
        
        if progress_callback:
            progress_callback(50)
        
        # Generate embeddings
        await self._generate_embeddings(all_chunks, progress_callback)
        
        if progress_callback:
            progress_callback(90)
        
        # Create vector index
        vector_index = VectorIndex(
            chunks=all_chunks,
            dimension=self.dimension,
            total_tokens=total_tokens,
            model_name=self.model_name
        )
        
        # Save index artifacts
        await self._save_index_artifacts(vector_index, work_dir)
        
        if progress_callback:
            progress_callback(100)
        
        return vector_index
    
    async def _filter_processable_files(
        self,
        repo_root: Path,
        files: List[str],
        repomap_order: List[str]
    ) -> List[str]:
        """Filter and order files for vector embedding processing."""
        # Filter for text files suitable for embedding
        processable_extensions = {
            '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala',
            '.md', '.txt', '.json', '.yaml', '.yml'
        }
        
        processable_files = []
        repomap_set = set(repomap_order)
        
        # First, add files in RepoMapper order (for files that exist in both lists)
        for file_path in repomap_order:
            if file_path in files:
                file_ext = Path(file_path).suffix.lower()
                if file_ext in processable_extensions:
                    # Check file size (skip very large files)
                    full_path = repo_root / file_path
                    try:
                        if full_path.stat().st_size < 1_000_000:  # 1MB limit
                            processable_files.append(file_path)
                    except OSError:
                        continue
        
        # Then add remaining files not in RepoMapper order
        for file_path in files:
            if file_path not in repomap_set:
                file_ext = Path(file_path).suffix.lower()
                if file_ext in processable_extensions:
                    full_path = repo_root / file_path
                    try:
                        if full_path.stat().st_size < 1_000_000:  # 1MB limit
                            processable_files.append(file_path)
                    except OSError:
                        continue
        
        return processable_files
    
    async def _chunk_file(self, repo_root: Path, file_path: str) -> List[VectorChunk]:
        """Chunk a single file into embedding-ready segments."""
        full_path = repo_root / file_path
        
        try:
            content = await asyncio.to_thread(
                full_path.read_text, encoding='utf-8', errors='ignore'
            )
        except Exception as e:
            raise LEANNError(f"Failed to read file {file_path}: {e}")
        
        # Attempt function-level chunking for code files
        if self._is_code_file(file_path):
            chunks = await self._chunk_by_functions(content, file_path)
            if chunks:
                return chunks
        
        # Fall back to token-based chunking
        return await self._chunk_by_tokens(content, file_path)
    
    def _is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file suitable for function-level chunking."""
        code_extensions = {
            '.ts', '.tsx', '.js', '.jsx', '.py', '.java', '.cpp', '.c', '.h',
            '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala'
        }
        return Path(file_path).suffix.lower() in code_extensions
    
    async def _chunk_by_functions(
        self, 
        content: str, 
        file_path: str
    ) -> List[VectorChunk]:
        """
        Chunk code by function boundaries.
        
        This is a simplified implementation. In practice, this would use
        tree-sitter or similar AST parsing for accurate function detection.
        """
        chunks = []
        lines = content.split('\n')
        
        current_function = []
        current_start_line = 0
        in_function = False
        brace_count = 0
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Simple function detection heuristics
            if (not in_function and 
                any(keyword in stripped for keyword in ['function ', 'def ', 'class ', 'interface ', 'async ']) and
                '{' in stripped or ':' in stripped):
                
                # Start new function
                if current_function:
                    chunk = await self._create_chunk_from_lines(
                        current_function, file_path, current_start_line
                    )
                    if chunk:
                        chunks.append(chunk)
                
                current_function = [line]
                current_start_line = i
                in_function = True
                brace_count = stripped.count('{') - stripped.count('}')
                
            elif in_function:
                current_function.append(line)
                brace_count += stripped.count('{') - stripped.count('}')
                
                # End of function
                if brace_count <= 0 and stripped.endswith('}'):
                    chunk = await self._create_chunk_from_lines(
                        current_function, file_path, current_start_line
                    )
                    if chunk:
                        chunks.append(chunk)
                    
                    current_function = []
                    in_function = False
                    brace_count = 0
            
            elif not in_function and stripped and not stripped.startswith('//'):
                # Standalone code outside functions
                current_function.append(line)
        
        # Handle remaining content
        if current_function:
            chunk = await self._create_chunk_from_lines(
                current_function, file_path, current_start_line
            )
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    async def _create_chunk_from_lines(
        self,
        lines: List[str],
        file_path: str,
        start_line: int
    ) -> Optional[VectorChunk]:
        """Create a vector chunk from lines of code."""
        content = '\n'.join(lines)
        
        # Skip very small chunks
        if len(content.strip()) < 50:
            return None
        
        # Estimate token count (rough approximation)
        token_count = len(content.split()) + len(content) // 4
        
        # Skip very large chunks
        if token_count > self.chunk_size * 2:
            return None
        
        chunk_id = str(uuid.uuid4())
        
        # Calculate byte span (approximation)
        start_byte = sum(len(line) + 1 for line in lines[:start_line])
        end_byte = start_byte + len(content)
        
        return VectorChunk(
            chunk_id=chunk_id,
            path=file_path,
            span=(start_byte, end_byte),
            content=content,
            token_count=token_count
        )
    
    async def _chunk_by_tokens(
        self, 
        content: str, 
        file_path: str
    ) -> List[VectorChunk]:
        """Chunk content by token count with overlap."""
        chunks = []
        
        # Simple token estimation
        tokens = content.split()
        
        for i in range(0, len(tokens), self.chunk_size - self.overlap_size):
            chunk_tokens = tokens[i:i + self.chunk_size]
            chunk_content = ' '.join(chunk_tokens)
            
            # Skip very small chunks
            if len(chunk_tokens) < 20:
                continue
            
            chunk_id = str(uuid.uuid4())
            
            # Estimate byte span
            start_char = content.find(chunk_tokens[0], i * 10) if chunk_tokens else 0
            end_char = start_char + len(chunk_content)
            
            chunk = VectorChunk(
                chunk_id=chunk_id,
                path=file_path,
                span=(start_char, end_char),
                content=chunk_content,
                token_count=len(chunk_tokens)
            )
            
            chunks.append(chunk)
        
        return chunks
    
    async def _generate_embeddings(
        self,
        chunks: List[VectorChunk],
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> None:
        """Generate embeddings for all chunks."""
        if not chunks:
            return
        
        # Batch process embeddings for efficiency
        batch_size = 32
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_texts = [chunk.content for chunk in batch]
            
            # Generate embeddings (simplified - would use actual model)
            batch_embeddings = await self._generate_batch_embeddings(batch_texts)
            
            # Assign embeddings to chunks
            for chunk, embedding in zip(batch, batch_embeddings):
                chunk.embedding = embedding.tolist()
            
            # Update progress
            if progress_callback:
                progress = 50 + int((i / len(chunks)) * 40)
                progress_callback(progress)
    
    async def _generate_batch_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        This is a placeholder implementation. In practice, this would use
        a real embedding model like sentence-transformers.
        """
        # Simulate embedding generation with random vectors
        embeddings = []
        
        for text in texts:
            # Create deterministic "embedding" based on text hash
            text_hash = hash(text) % (2**32)
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 1, self.dimension)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    async def _save_index_artifacts(
        self,
        vector_index: VectorIndex,
        work_dir: Path
    ) -> None:
        """Save vector index artifacts to disk."""
        # Save index metadata
        index_metadata = {
            "chunk_count": len(vector_index.chunks),
            "dimension": vector_index.dimension,
            "total_tokens": vector_index.total_tokens,
            "model_name": vector_index.model_name,
            "created_at": vector_index.generated_at.isoformat()
        }
        
        atomic_write_json(work_dir / "leann.index", index_metadata)
        
        # Save embeddings as binary file
        if vector_index.chunks:
            embeddings = np.array([
                chunk.embedding for chunk in vector_index.chunks 
                if chunk.embedding
            ])
            
            # Save as binary format: [count][dimension][embeddings...]
            binary_data = bytearray()
            binary_data.extend(struct.pack('I', len(embeddings)))
            binary_data.extend(struct.pack('I', vector_index.dimension))
            binary_data.extend(embeddings.astype(np.float32).tobytes())
            
            atomic_write_bytes(work_dir / "vectors.bin", bytes(binary_data))
        
        # Save chunk metadata
        chunk_metadata = []
        for chunk in vector_index.chunks:
            chunk_meta = {
                "chunk_id": chunk.chunk_id,
                "path": chunk.path,
                "span": chunk.span,
                "token_count": chunk.token_count
            }
            chunk_metadata.append(chunk_meta)
        
        atomic_write_json(work_dir / "chunks.json", chunk_metadata)
    
    async def search_similar(
        self,
        query_text: str,
        vector_index: VectorIndex,
        k: int = 20
    ) -> List[Tuple[VectorChunk, float]]:
        """
        Search for similar chunks using vector similarity.
        
        Returns list of (chunk, similarity_score) tuples.
        """
        if not vector_index.chunks:
            return []
        
        # Generate query embedding
        query_embeddings = await self._generate_batch_embeddings([query_text])
        query_embedding = query_embeddings[0]
        
        # Calculate similarities
        similarities = []
        for chunk in vector_index.chunks:
            if chunk.embedding:
                chunk_embedding = np.array(chunk.embedding)
                similarity = np.dot(query_embedding, chunk_embedding)
                similarities.append((chunk, float(similarity)))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    async def validate_index(
        self,
        vector_index: VectorIndex
    ) -> Dict[str, Any]:
        """
        Validate vector index quality and consistency.
        
        Returns validation report with statistics and issues.
        """
        validation_report = {
            "valid": True,
            "statistics": {
                "total_chunks": len(vector_index.chunks),
                "avg_tokens_per_chunk": 0,
                "embedding_coverage": 0,
                "dimension": vector_index.dimension
            },
            "issues": [],
            "recommendations": []
        }
        
        if not vector_index.chunks:
            validation_report["valid"] = False
            validation_report["issues"].append("No chunks in vector index")
            return validation_report
        
        # Calculate statistics
        total_tokens = sum(chunk.token_count for chunk in vector_index.chunks)
        validation_report["statistics"]["avg_tokens_per_chunk"] = total_tokens / len(vector_index.chunks)
        
        chunks_with_embeddings = sum(1 for chunk in vector_index.chunks if chunk.embedding)
        validation_report["statistics"]["embedding_coverage"] = chunks_with_embeddings / len(vector_index.chunks)
        
        # Check for issues
        if validation_report["statistics"]["embedding_coverage"] < 0.9:
            validation_report["issues"].append(
                f"Low embedding coverage: {validation_report['statistics']['embedding_coverage']:.1%}"
            )
        
        if validation_report["statistics"]["avg_tokens_per_chunk"] < 50:
            validation_report["recommendations"].append(
                "Consider increasing minimum chunk size for better embedding quality"
            )
        
        if validation_report["statistics"]["avg_tokens_per_chunk"] > 800:
            validation_report["recommendations"].append(
                "Consider decreasing chunk size for more precise search results"
            )
        
        return validation_report