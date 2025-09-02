"""
Performance and benchmarking tests for Mimir 2.0.

Tests performance characteristics, resource usage, and scalability
of new Mimir 2.0 features under various load conditions.
"""

import asyncio
import time
import psutil
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch
import pytest
import numpy as np

from src.repoindex.config import (
    MimirConfig, 
    AIConfig,
    QueryConfig,
    RerankerConfig,
    PipelineConfig
)
from src.repoindex.data.schemas import VectorChunk, VectorIndex
from src.repoindex.pipeline.enhanced_search import EnhancedSearchPipeline
from src.repoindex.pipeline.ollama import OllamaAdapter


class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time


class MemoryMonitor:
    """Monitor memory usage during operations."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.initial_memory = None
        self.peak_memory = None
        self.final_memory = None
    
    def __enter__(self):
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.initial_memory
        return self
    
    def update_peak(self):
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        if current_memory > self.peak_memory:
            self.peak_memory = current_memory
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.final_memory = self.process.memory_info().rss / 1024 / 1024  # MB


@pytest.fixture
def performance_config():
    """Create configuration optimized for performance testing."""
    from src.repoindex.config import OllamaConfig
    
    return MimirConfig(
        ai=AIConfig(
            provider="ollama",
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                model="llama2:7b",
                timeout=10  # Shorter timeout for performance tests
            )
        ),
        query=QueryConfig(
            max_results=50,
            use_hyde=True,
            hyde_iterations=2,  # Fewer iterations for speed
            enable_reranking=True
        ),
        reranker=RerankerConfig(
            enabled=True,
            top_k=20,
            batch_size=64  # Larger batch size for efficiency
        ),
        pipeline=PipelineConfig(
            max_concurrent_tasks=8,  # More concurrent tasks
            chunk_size=256  # Smaller chunks for faster processing
        )
    )


@pytest.fixture
def small_vector_index():
    """Create small vector index for baseline performance testing."""
    chunks = []
    for i in range(100):
        embedding = np.random.normal(0, 1, 384).tolist()
        chunks.append(VectorChunk(
            path=f"src/small_{i}.py",
            span=(1, 20),
            content=f"def small_function_{i}():\n    return {i} * 2",
            embedding=embedding,
            token_count=15
        ))
    
    return VectorIndex(
        chunks=chunks,
        dimension=384,
        total_tokens=sum(chunk.token_count for chunk in chunks),
        model_name="small-perf-test"
    )


@pytest.fixture
def medium_vector_index():
    """Create medium vector index for scalability testing."""
    chunks = []
    for i in range(1000):
        embedding = np.random.normal(0, 1, 384).tolist()
        chunks.append(VectorChunk(
            path=f"src/medium_{i}.py",
            span=(1, 50),
            content=f"def medium_function_{i}():\n    # Implementation {i}\n    result = {i}\n    return result * 3",
            embedding=embedding,
            token_count=30
        ))
    
    return VectorIndex(
        chunks=chunks,
        dimension=384,
        total_tokens=sum(chunk.token_count for chunk in chunks),
        model_name="medium-perf-test"
    )


@pytest.fixture
def large_vector_index():
    """Create large vector index for stress testing."""
    chunks = []
    for i in range(10000):
        embedding = np.random.normal(0, 1, 384).tolist()
        chunks.append(VectorChunk(
            path=f"src/large_{i}.py",
            span=(1, 100),
            content=f"def large_function_{i}():\n    # Large implementation {i}\n    data = [{{'id': {i}, 'value': {i*2}}}]\n    return process_data(data)",
            embedding=embedding,
            token_count=50
        ))
    
    return VectorIndex(
        chunks=chunks,
        dimension=384,
        total_tokens=sum(chunk.token_count for chunk in chunks),
        model_name="large-perf-test"
    )


class TestOllamaAdapterPerformance:
    """Test Ollama adapter performance characteristics."""

    @pytest.mark.asyncio
    async def test_adapter_initialization_performance(self):
        """Test adapter initialization time."""
        with PerformanceTimer() as timer:
            adapter = OllamaAdapter(base_url="http://localhost:11434", timeout=10)
            
        assert timer.duration < 0.1  # Should initialize very quickly

    @pytest.mark.asyncio
    async def test_connection_check_performance(self):
        """Test connection availability check performance."""
        adapter = OllamaAdapter(base_url="http://localhost:11434", timeout=5)
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            with PerformanceTimer() as timer:
                is_available = await adapter.is_available()
            
            assert is_available is True
            assert timer.duration < 1.0  # Should check availability quickly

    @pytest.mark.asyncio
    async def test_text_generation_performance(self):
        """Test text generation performance with different input sizes."""
        adapter = OllamaAdapter(base_url="http://localhost:11434", timeout=30)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": "Generated response",
                "done": True
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Test different prompt sizes
            prompt_sizes = [50, 200, 500, 1000]  # characters
            
            for size in prompt_sizes:
                prompt = "test " * (size // 5)  # Approximate character count
                
                with PerformanceTimer() as timer:
                    result = await adapter.generate_text(prompt, "llama2:7b")
                
                assert result == "Generated response"
                # Larger prompts should not significantly increase overhead
                # (actual generation time would be longer, but this tests the adapter)
                assert timer.duration < 0.5

    @pytest.mark.asyncio
    async def test_concurrent_requests_performance(self):
        """Test performance with concurrent requests."""
        adapter = OllamaAdapter(base_url="http://localhost:11434", timeout=30)
        
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = {
                "response": "Concurrent response",
                "done": True
            }
            mock_post.return_value.__aenter__.return_value = mock_response
            
            async def generate_task():
                return await adapter.generate_text("concurrent test", "llama2:7b")
            
            # Test concurrent requests
            num_concurrent = 10
            
            with PerformanceTimer() as timer:
                tasks = [generate_task() for _ in range(num_concurrent)]
                results = await asyncio.gather(*tasks)
            
            assert len(results) == num_concurrent
            assert all(result == "Concurrent response" for result in results)
            # Concurrent requests should be faster than sequential
            assert timer.duration < num_concurrent * 0.5


class TestEnhancedSearchPerformance:
    """Test enhanced search pipeline performance."""

    @pytest.mark.asyncio
    async def test_search_pipeline_initialization_performance(self, performance_config):
        """Test search pipeline initialization time."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            with PerformanceTimer() as timer:
                pipeline = EnhancedSearchPipeline(
                    ai_config=performance_config.ai,
                    query_config=performance_config.query,
                    reranker_config=performance_config.reranker,
                    coordinator=mock_coordinator
                )
                await pipeline.initialize()
            
            assert timer.duration < 1.0  # Should initialize quickly

    @pytest.mark.asyncio
    async def test_small_index_search_performance(self, performance_config, small_vector_index):
        """Test search performance on small index."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=performance_config.ai,
                query_config=performance_config.query,
                reranker_config=performance_config.reranker,
                coordinator=mock_coordinator
            )
            await pipeline.initialize()
            
            # Mock all components for consistent performance testing
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["query", "expanded"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                mock_reranker.rerank.return_value = [(small_vector_index.chunks[i], 0.9 - i * 0.1) for i in range(5)]
                mock_reranker_class.return_value = mock_reranker
                
                with PerformanceTimer() as timer, MemoryMonitor() as memory:
                    results = await pipeline.search(
                        query="performance test query",
                        vector_index=small_vector_index,
                        top_k=10
                    )
                    memory.update_peak()
                
                assert len(results) <= 10
                assert timer.duration < 2.0  # Should complete quickly for small index
                assert memory.peak_memory - memory.initial_memory < 50  # Memory usage should be reasonable (MB)

    @pytest.mark.asyncio
    async def test_medium_index_search_performance(self, performance_config, medium_vector_index):
        """Test search performance on medium index."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=performance_config.ai,
                query_config=performance_config.query,
                reranker_config=performance_config.reranker,
                coordinator=mock_coordinator
            )
            await pipeline.initialize()
            
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["query", "expanded"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                mock_reranker.rerank.return_value = [(medium_vector_index.chunks[i], 0.9 - i * 0.01) for i in range(20)]
                mock_reranker_class.return_value = mock_reranker
                
                with PerformanceTimer() as timer, MemoryMonitor() as memory:
                    results = await pipeline.search(
                        query="medium scale test query",
                        vector_index=medium_vector_index,
                        top_k=20
                    )
                    memory.update_peak()
                
                assert len(results) <= 20
                assert timer.duration < 5.0  # Should handle medium index reasonably fast
                assert memory.peak_memory - memory.initial_memory < 100  # Memory usage should scale appropriately

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_index_search_performance(self, performance_config, large_vector_index):
        """Test search performance on large index (stress test)."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=performance_config.ai,
                query_config=performance_config.query,
                reranker_config=performance_config.reranker,
                coordinator=mock_coordinator
            )
            await pipeline.initialize()
            
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["query", "expanded"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                # Simulate realistic reranking time by returning subset
                mock_reranker.rerank.return_value = [(large_vector_index.chunks[i], 0.95 - i * 0.01) for i in range(30)]
                mock_reranker_class.return_value = mock_reranker
                
                with PerformanceTimer() as timer, MemoryMonitor() as memory:
                    results = await pipeline.search(
                        query="large scale stress test query",
                        vector_index=large_vector_index,
                        top_k=30
                    )
                    memory.update_peak()
                
                assert len(results) <= 30
                assert timer.duration < 15.0  # Should handle large index within reasonable time
                assert memory.peak_memory - memory.initial_memory < 500  # Memory usage should not explode

    @pytest.mark.asyncio
    async def test_concurrent_search_performance(self, performance_config, medium_vector_index):
        """Test concurrent search performance."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            # Create multiple pipeline instances for concurrent testing
            pipelines = []
            for i in range(5):
                pipeline = EnhancedSearchPipeline(
                    ai_config=performance_config.ai,
                    query_config=performance_config.query,
                    reranker_config=performance_config.reranker,
                    coordinator=mock_coordinator
                )
                await pipeline.initialize()
                pipelines.append(pipeline)
            
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["concurrent query"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                mock_reranker.rerank.return_value = [(medium_vector_index.chunks[0], 0.9)]
                mock_reranker_class.return_value = mock_reranker
                
                async def search_task(pipeline, query_id):
                    return await pipeline.search(
                        query=f"concurrent test query {query_id}",
                        vector_index=medium_vector_index,
                        top_k=10
                    )
                
                with PerformanceTimer() as timer, MemoryMonitor() as memory:
                    tasks = [search_task(pipelines[i], i) for i in range(5)]
                    results = await asyncio.gather(*tasks)
                    memory.update_peak()
                
                assert len(results) == 5
                assert all(isinstance(result, list) for result in results)
                # Concurrent execution should be faster than sequential
                assert timer.duration < 10.0
                assert memory.peak_memory - memory.initial_memory < 200


class TestRaptorPerformance:
    """Test RAPTOR system performance."""

    @pytest.mark.skipif('not HAS_ML_DEPS', reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_raptor_tree_building_performance(self, performance_config, medium_vector_index):
        """Test RAPTOR tree building performance."""
        from src.repoindex.pipeline.raptor import RaptorProcessor, RaptorConfig
        
        raptor_config = RaptorConfig(
            max_clusters=5,
            min_cluster_size=3,
            tree_depth=2
        )
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.generate_text.side_effect = ["Summary 1", "Summary 2", "Root summary"]
            mock_coordinator_class.return_value = mock_coordinator
            
            processor = RaptorProcessor(
                ai_config=performance_config.ai,
                config=raptor_config,
                coordinator=mock_coordinator
            )
            
            await processor.initialize()
            
            with PerformanceTimer() as timer, MemoryMonitor() as memory:
                tree = await processor.process_embeddings(medium_vector_index)
                memory.update_peak()
            
            assert tree is not None
            assert tree.root is not None
            # Tree building should complete in reasonable time
            assert timer.duration < 30.0
            assert memory.peak_memory - memory.initial_memory < 300

    @pytest.mark.skipif('not HAS_ML_DEPS', reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_raptor_query_performance(self, performance_config, small_vector_index):
        """Test RAPTOR tree querying performance."""
        from src.repoindex.pipeline.raptor import RaptorProcessor, RaptorConfig
        
        raptor_config = RaptorConfig(
            max_clusters=3,
            min_cluster_size=2,
            tree_depth=2
        )
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.generate_text.side_effect = ["Summary 1", "Summary 2", "Root"]
            mock_coordinator_class.return_value = mock_coordinator
            
            processor = RaptorProcessor(
                ai_config=performance_config.ai,
                config=raptor_config,
                coordinator=mock_coordinator
            )
            
            await processor.initialize()
            
            # Build tree first
            tree = await processor.process_embeddings(small_vector_index)
            
            # Test multiple queries for performance
            queries = [
                "test query 1",
                "test query 2", 
                "test query 3"
            ]
            
            with PerformanceTimer() as timer:
                for query in queries:
                    results = await processor.query_tree(tree, query, top_k=5)
                    assert len(results) <= 5
            
            # Multiple queries should complete quickly
            assert timer.duration < 5.0


class TestPerformanceRegressionDetection:
    """Test for performance regression detection."""

    @pytest.mark.asyncio
    async def test_baseline_performance_metrics(self, performance_config, small_vector_index):
        """Establish baseline performance metrics for regression testing."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=performance_config.ai,
                query_config=performance_config.query,
                reranker_config=performance_config.reranker,
                coordinator=mock_coordinator
            )
            await pipeline.initialize()
            
            # Mock components for consistent baseline
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["baseline query"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                mock_reranker.rerank.return_value = [(small_vector_index.chunks[0], 0.9)]
                mock_reranker_class.return_value = mock_reranker
                
                # Run multiple iterations for stable baseline
                times = []
                memory_usage = []
                
                for i in range(10):
                    with PerformanceTimer() as timer, MemoryMonitor() as memory:
                        results = await pipeline.search(
                            query=f"baseline query {i}",
                            vector_index=small_vector_index,
                            top_k=10
                        )
                        memory.update_peak()
                    
                    times.append(timer.duration)
                    memory_usage.append(memory.peak_memory - memory.initial_memory)
                    assert len(results) <= 10
                
                # Calculate baseline metrics
                avg_time = sum(times) / len(times)
                max_time = max(times)
                avg_memory = sum(memory_usage) / len(memory_usage)
                max_memory = max(memory_usage)
                
                # Store baseline metrics for comparison
                baseline_metrics = {
                    "avg_search_time": avg_time,
                    "max_search_time": max_time,
                    "avg_memory_usage": avg_memory,
                    "max_memory_usage": max_memory
                }
                
                # Assert reasonable baseline performance
                assert avg_time < 1.0  # Average search should be under 1 second
                assert max_time < 2.0  # No search should take more than 2 seconds
                assert avg_memory < 50  # Average memory increase should be under 50MB
                assert max_memory < 100  # Peak memory should be under 100MB increase

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, performance_config, small_vector_index):
        """Test for memory leaks during repeated operations."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=performance_config.ai,
                query_config=performance_config.query,
                reranker_config=performance_config.reranker,
                coordinator=mock_coordinator
            )
            await pipeline.initialize()
            
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["leak test query"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                mock_reranker.rerank.return_value = [(small_vector_index.chunks[0], 0.9)]
                mock_reranker_class.return_value = mock_reranker
                
                # Monitor memory over many iterations
                initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_samples = [initial_memory]
                
                # Run many searches to detect memory leaks
                for i in range(50):
                    results = await pipeline.search(
                        query=f"memory leak test {i}",
                        vector_index=small_vector_index,
                        top_k=5
                    )
                    
                    if i % 10 == 0:  # Sample memory every 10 iterations
                        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_samples.append(current_memory)
                    
                    assert len(results) <= 5
                
                # Check for memory leak (significant upward trend)
                memory_growth = memory_samples[-1] - memory_samples[0]
                
                # Memory growth should be minimal (< 20MB over 50 iterations)
                assert memory_growth < 20, f"Potential memory leak detected: {memory_growth}MB growth"
                
                # Memory should stabilize (last few samples shouldn't grow significantly)
                recent_growth = memory_samples[-1] - memory_samples[-3]
                assert recent_growth < 5, f"Memory not stabilizing: {recent_growth}MB recent growth"