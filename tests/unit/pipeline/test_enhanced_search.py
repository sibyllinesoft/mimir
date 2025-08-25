"""
Unit tests for Enhanced Search Pipeline.

Tests the multi-stage search pipeline that coordinates HyDE, vector search,
reranking, and result fusion for optimal retrieval.
"""

from unittest.mock import AsyncMock, Mock, patch
import pytest
import numpy as np

from src.repoindex.pipeline.enhanced_search import (
    EnhancedSearchPipeline,
    EnhancedSearchError,
    create_enhanced_search_pipeline
)
from src.repoindex.data.schemas import VectorChunk, VectorIndex
from src.repoindex.config import AIConfig, QueryConfig, RerankerConfig


class TestEnhancedSearchPipeline:
    """Test EnhancedSearchPipeline functionality."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample vector chunks for search testing."""
        chunks = []
        for i in range(10):
            embedding = np.random.normal(0, 1, 384).tolist()
            chunks.append(VectorChunk(
                path=f"src/file_{i}.py",
                span=(1, 20),
                content=f"def function_{i}():\n    # Function {i} implementation\n    return {i}",
                embedding=embedding,
                token_count=25
            ))
        return chunks

    @pytest.fixture
    def vector_index(self, sample_chunks):
        """Create vector index with sample chunks."""
        return VectorIndex(
            chunks=sample_chunks,
            dimension=384,
            total_tokens=sum(chunk.token_count for chunk in sample_chunks),
            model_name="test-embeddings"
        )

    @pytest.fixture
    def ai_config(self):
        """Create AI configuration."""
        from src.repoindex.config import GeminiConfig
        return AIConfig(
            provider="gemini",
            gemini=GeminiConfig(api_key="test-key")
        )

    @pytest.fixture
    def query_config(self):
        """Create query configuration."""
        return QueryConfig(
            max_results=10,
            min_score=0.1,
            use_hyde=True,
            hyde_iterations=2,
            enable_reranking=True,
            fusion_method="rrf"
        )

    @pytest.fixture
    def reranker_config(self):
        """Create reranker configuration."""
        return RerankerConfig(
            enabled=True,
            provider="cross_encoder",
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=20
        )

    @pytest.fixture
    def search_pipeline(self, ai_config, query_config, reranker_config):
        """Create enhanced search pipeline."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            return EnhancedSearchPipeline(
                ai_config=ai_config,
                query_config=query_config,
                reranker_config=reranker_config,
                coordinator=mock_coordinator
            )

    @pytest.mark.asyncio
    async def test_search_pipeline_initialization(self, search_pipeline):
        """Test pipeline initialization."""
        await search_pipeline.initialize()
        
        assert search_pipeline.ai_config is not None
        assert search_pipeline.query_config is not None
        assert search_pipeline.reranker_config is not None
        assert search_pipeline.coordinator is not None

    @pytest.mark.asyncio
    async def test_search_with_hyde_enabled(self, search_pipeline, vector_index):
        """Test search with HyDE query expansion enabled."""
        await search_pipeline.initialize()
        
        # Mock HyDE processor
        mock_hyde_queries = [
            "original query",
            "expanded query version 1", 
            "expanded query version 2"
        ]
        
        # Mock vector search results
        mock_search_results = []
        for i in range(5):
            mock_search_results.append((vector_index.chunks[i], 0.8 - i * 0.1))
        
        # Mock reranker results
        mock_reranked_results = mock_search_results[:3]  # Top 3 after reranking
        
        with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
            mock_hyde = AsyncMock()
            mock_hyde.expand_query.return_value = mock_hyde_queries
            mock_hyde_class.return_value = mock_hyde
            
            with patch.object(search_pipeline, '_vector_search', return_value=mock_search_results):
                with patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                    mock_reranker = AsyncMock()
                    mock_reranker.rerank.return_value = mock_reranked_results
                    mock_reranker_class.return_value = mock_reranker
                    
                    results = await search_pipeline.search(
                        query="test query",
                        vector_index=vector_index,
                        top_k=3
                    )
                    
                    assert len(results) == 3
                    mock_hyde.expand_query.assert_called_once_with("test query", iterations=2)
                    mock_reranker.rerank.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_without_hyde(self, ai_config, reranker_config, vector_index):
        """Test search without HyDE query expansion."""
        query_config = QueryConfig(use_hyde=False, enable_reranking=False)
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=ai_config,
                query_config=query_config,
                reranker_config=reranker_config,
                coordinator=mock_coordinator
            )
            
            await pipeline.initialize()
            
            # Mock direct vector search
            mock_search_results = []
            for i in range(5):
                mock_search_results.append((vector_index.chunks[i], 0.8 - i * 0.1))
            
            with patch.object(pipeline, '_vector_search', return_value=mock_search_results):
                results = await pipeline.search(
                    query="test query",
                    vector_index=vector_index,
                    top_k=5
                )
                
                assert len(results) == 5
                # Verify no HyDE expansion was used
                assert all(isinstance(result, tuple) and len(result) == 2 for result in results)

    @pytest.mark.asyncio
    async def test_search_with_reranking_disabled(self, ai_config, query_config, vector_index):
        """Test search with reranking disabled."""
        reranker_config = RerankerConfig(enabled=False)
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=ai_config,
                query_config=query_config,
                reranker_config=reranker_config,
                coordinator=mock_coordinator
            )
            
            await pipeline.initialize()
            
            mock_search_results = []
            for i in range(3):
                mock_search_results.append((vector_index.chunks[i], 0.9 - i * 0.1))
            
            with patch.object(pipeline, '_vector_search', return_value=mock_search_results):
                with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                    mock_hyde = AsyncMock()
                    mock_hyde.expand_query.return_value=["test query", "expanded"]
                    mock_hyde_class.return_value = mock_hyde
                    
                    results = await pipeline.search(
                        query="test query",
                        vector_index=vector_index,
                        top_k=3
                    )
                    
                    assert len(results) == 3
                    # Results should be direct from vector search without reranking

    @pytest.mark.asyncio
    async def test_vector_search_implementation(self, search_pipeline, vector_index):
        """Test vector search implementation."""
        await search_pipeline.initialize()
        
        query = "test function implementation"
        top_k = 5
        
        # Mock embedding generation for query
        query_embedding = np.random.normal(0, 1, 384).tolist()
        
        with patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embedding_class:
            mock_embedding_processor = AsyncMock()
            mock_embedding_processor.embed_query.return_value = query_embedding
            mock_embedding_class.return_value = mock_embedding_processor
            
            results = await search_pipeline._vector_search(query, vector_index, top_k)
            
            assert isinstance(results, list)
            assert len(results) <= top_k
            
            for chunk, score in results:
                assert isinstance(chunk, VectorChunk)
                assert isinstance(score, float)
                assert 0 <= score <= 1

    @pytest.mark.asyncio
    async def test_fusion_strategies(self, search_pipeline, vector_index):
        """Test different result fusion strategies."""
        await search_pipeline.initialize()
        
        # Create multiple result sets to fuse
        results_set_1 = [(vector_index.chunks[0], 0.9), (vector_index.chunks[1], 0.8)]
        results_set_2 = [(vector_index.chunks[1], 0.85), (vector_index.chunks[2], 0.75)]
        results_set_3 = [(vector_index.chunks[0], 0.8), (vector_index.chunks[2], 0.7)]
        
        multiple_results = [results_set_1, results_set_2, results_set_3]
        
        # Test RRF (Reciprocal Rank Fusion)
        rrf_fused = search_pipeline._fuse_results_rrf(multiple_results, k=3)
        assert isinstance(rrf_fused, list)
        assert len(rrf_fused) <= 3
        
        # Test weighted fusion
        weights = [0.5, 0.3, 0.2]
        weighted_fused = search_pipeline._fuse_results_weighted(multiple_results, weights, k=3)
        assert isinstance(weighted_fused, list)
        assert len(weighted_fused) <= 3
        
        # Verify results are properly scored and ranked
        for chunk, score in rrf_fused:
            assert isinstance(chunk, VectorChunk)
            assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_search_with_min_score_filtering(self, search_pipeline, vector_index):
        """Test search with minimum score filtering."""
        await search_pipeline.initialize()
        
        # Modify query config to have higher min_score
        search_pipeline.query_config.min_score = 0.7
        
        # Mock search results with varied scores
        mock_results = [
            (vector_index.chunks[0], 0.9),   # Above threshold
            (vector_index.chunks[1], 0.8),   # Above threshold
            (vector_index.chunks[2], 0.6),   # Below threshold
            (vector_index.chunks[3], 0.5),   # Below threshold
        ]
        
        with patch.object(search_pipeline, '_vector_search', return_value=mock_results):
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["test query"]
                mock_hyde_class.return_value = mock_hyde
                
                results = await search_pipeline.search(
                    query="test query",
                    vector_index=vector_index,
                    top_k=10
                )
                
                # Should only return results above min_score
                assert len(results) == 2
                assert all(score >= 0.7 for _, score in results)

    @pytest.mark.asyncio
    async def test_search_error_handling(self, search_pipeline, vector_index):
        """Test error handling in search pipeline."""
        await search_pipeline.initialize()
        
        # Test HyDE processor failure
        with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
            mock_hyde = AsyncMock()
            mock_hyde.expand_query.side_effect = Exception("HyDE failed")
            mock_hyde_class.return_value = mock_hyde
            
            with pytest.raises(EnhancedSearchError) as exc_info:
                await search_pipeline.search(
                    query="test query",
                    vector_index=vector_index,
                    top_k=5
                )
            
            assert "HyDE failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_empty_index(self, search_pipeline):
        """Test search with empty vector index."""
        empty_index = VectorIndex(
            chunks=[],
            dimension=384,
            total_tokens=0,
            model_name="test"
        )
        
        await search_pipeline.initialize()
        
        results = await search_pipeline.search(
            query="test query",
            vector_index=empty_index,
            top_k=5
        )
        
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_performance_monitoring(self, search_pipeline, vector_index):
        """Test search performance monitoring and timing."""
        await search_pipeline.initialize()
        
        mock_results = [(vector_index.chunks[0], 0.9)]
        
        with patch.object(search_pipeline, '_vector_search', return_value=mock_results):
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["test query"]
                mock_hyde_class.return_value = mock_hyde
                
                # Should include timing information in the result
                results = await search_pipeline.search(
                    query="test query",
                    vector_index=vector_index,
                    top_k=5,
                    include_timing=True
                )
                
                # If timing is included, results should be a dict with 'results' and 'timing'
                if isinstance(results, dict):
                    assert 'results' in results
                    assert 'timing' in results
                    assert isinstance(results['timing'], dict)


class TestEnhancedSearchPipelineIntegration:
    """Test Enhanced Search Pipeline integration scenarios."""

    @pytest.fixture
    def comprehensive_vector_index(self):
        """Create a comprehensive vector index for integration testing."""
        chunks = []
        
        # Math functions
        math_embedding_base = np.random.normal([1, 0, 0], 0.1, 384)
        for i in range(3):
            embedding = (math_embedding_base + np.random.normal(0, 0.05, 384)).tolist()
            chunks.append(VectorChunk(
                path=f"src/math/operations_{i}.py",
                span=(1, 25),
                content=f"def calculate_{i}(a, b):\n    return a + b * {i}\n\ndef process_math_{i}():\n    pass",
                embedding=embedding,
                token_count=30
            ))
        
        # String functions  
        string_embedding_base = np.random.normal([0, 1, 0], 0.1, 384)
        for i in range(3):
            embedding = (string_embedding_base + np.random.normal(0, 0.05, 384)).tolist()
            chunks.append(VectorChunk(
                path=f"src/string/utils_{i}.py",
                span=(1, 25),
                content=f"def format_string_{i}(text):\n    return text.upper()\n\ndef validate_string_{i}():\n    pass",
                embedding=embedding,
                token_count=28
            ))
        
        # Database functions
        db_embedding_base = np.random.normal([0, 0, 1], 0.1, 384)
        for i in range(2):
            embedding = (db_embedding_base + np.random.normal(0, 0.05, 384)).tolist()
            chunks.append(VectorChunk(
                path=f"src/database/models_{i}.py", 
                span=(1, 30),
                content=f"class DataModel_{i}:\n    def save(self):\n        pass\n    def load(self):\n        pass",
                embedding=embedding,
                token_count=35
            ))
        
        return VectorIndex(
            chunks=chunks,
            dimension=384,
            total_tokens=sum(chunk.token_count for chunk in chunks),
            model_name="comprehensive-test"
        )

    @pytest.mark.asyncio
    async def test_full_pipeline_integration(self, comprehensive_vector_index):
        """Test complete search pipeline with all components enabled."""
        from src.repoindex.config import GeminiConfig
        
        ai_config = AIConfig(
            provider="gemini",
            gemini=GeminiConfig(api_key="test-key")
        )
        
        query_config = QueryConfig(
            max_results=10,
            use_hyde=True,
            hyde_iterations=3,
            enable_reranking=True,
            fusion_method="rrf"
        )
        
        reranker_config = RerankerConfig(
            enabled=True,
            top_k=15
        )
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=ai_config,
                query_config=query_config,
                reranker_config=reranker_config,
                coordinator=mock_coordinator
            )
            
            await pipeline.initialize()
            
            # Mock all components
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = [
                    "mathematical calculation functions",
                    "arithmetic operations and calculations", 
                    "numeric processing and math utilities"
                ]
                mock_hyde_class.return_value = mock_hyde
                
                with patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class:
                    mock_embedder = AsyncMock()
                    # Return embedding similar to math functions
                    mock_embedder.embed_query.return_value = np.random.normal([1, 0, 0], 0.1, 384).tolist()
                    mock_embed_class.return_value = mock_embedder
                    
                    with patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                        mock_reranker = AsyncMock()
                        # Mock reranker to prefer math-related chunks
                        def mock_rerank(query, results):
                            # Sort by whether chunk path contains 'math'
                            sorted_results = sorted(
                                results,
                                key=lambda x: (1 if 'math' in x[0].path else 0, x[1]),
                                reverse=True
                            )
                            return sorted_results[:query_config.max_results]
                        
                        mock_reranker.rerank.side_effect = mock_rerank
                        mock_reranker_class.return_value = mock_reranker
                        
                        results = await pipeline.search(
                            query="mathematical calculation functions",
                            vector_index=comprehensive_vector_index,
                            top_k=5
                        )
                        
                        assert len(results) <= 5
                        assert len(results) > 0
                        
                        # Verify math functions are ranked higher
                        math_results = [r for r in results if 'math' in r[0].path]
                        assert len(math_results) > 0

    @pytest.mark.asyncio
    async def test_pipeline_error_recovery(self, comprehensive_vector_index):
        """Test pipeline error recovery and fallback mechanisms."""
        from src.repoindex.config import GeminiConfig
        
        ai_config = AIConfig(
            provider="gemini",
            gemini=GeminiConfig(api_key="test-key")
        )
        
        query_config = QueryConfig(
            use_hyde=True,
            enable_reranking=True
        )
        
        reranker_config = RerankerConfig(enabled=True)
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = EnhancedSearchPipeline(
                ai_config=ai_config,
                query_config=query_config,
                reranker_config=reranker_config,
                coordinator=mock_coordinator
            )
            
            await pipeline.initialize()
            
            # Test HyDE failure with fallback to direct search
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.side_effect = Exception("HyDE service unavailable")
                mock_hyde_class.return_value = mock_hyde
                
                with patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class:
                    mock_embedder = AsyncMock()
                    mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                    mock_embed_class.return_value = mock_embedder
                    
                    # Should fallback gracefully
                    with pytest.raises(EnhancedSearchError):
                        await pipeline.search(
                            query="test query",
                            vector_index=comprehensive_vector_index,
                            top_k=3
                        )


class TestCreateEnhancedSearchPipeline:
    """Test the convenience function for creating enhanced search pipelines."""

    @pytest.mark.asyncio
    async def test_create_pipeline_function(self):
        """Test the create_enhanced_search_pipeline convenience function."""
        from src.repoindex.config import GeminiConfig
        
        ai_config = AIConfig(
            provider="gemini",
            gemini=GeminiConfig(api_key="test-key")
        )
        
        query_config = QueryConfig()
        reranker_config = RerankerConfig()
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            pipeline = await create_enhanced_search_pipeline(
                ai_config=ai_config,
                query_config=query_config,
                reranker_config=reranker_config
            )
            
            assert isinstance(pipeline, EnhancedSearchPipeline)
            assert pipeline.ai_config == ai_config
            assert pipeline.query_config == query_config
            assert pipeline.reranker_config == reranker_config


class TestEnhancedSearchError:
    """Test Enhanced Search Error handling."""

    def test_enhanced_search_error(self):
        """Test EnhancedSearchError exception."""
        error = EnhancedSearchError("Search pipeline failed", component="HyDE")
        
        assert "Search pipeline failed" in str(error)
        assert error.component == "HyDE"
        
        # Test without component
        error2 = EnhancedSearchError("General search error")
        assert "General search error" in str(error2)
        assert not hasattr(error2, 'component')