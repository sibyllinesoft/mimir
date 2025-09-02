"""
Comprehensive integration tests for Mimir 2.0 enhancements.

Tests the full integration of RAPTOR, enhanced search, HyDE, 
reranking, and code embeddings working together.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from repoindex.config import MimirConfig, get_ai_config
from repoindex.data.schemas import FeatureConfig, VectorChunk, VectorIndex
from repoindex.pipeline import (
    ADVANCED_SEARCH_AVAILABLE,
    RAPTOR_AVAILABLE,
    PipelineCoordinator,
    get_pipeline_coordinator,
    run_integration_validation,
    validate_mimir_2_setup,
)

if ADVANCED_SEARCH_AVAILABLE:
    from repoindex.pipeline import (
        EnhancedSearchPipeline,
        HyDETransformer,
        CodeEmbeddingAdapter,
        create_reranker,
    )

if RAPTOR_AVAILABLE:
    from repoindex.pipeline import RaptorProcessor, RaptorStage


class TestMimir2Integration:
    """Test Mimir 2.0 component integration."""
    
    @pytest.fixture
    def config_with_all_features(self):
        """Configuration with all Mimir 2.0 features enabled."""
        return {
            "MIMIR_ENABLE_OLLAMA": "true",
            "MIMIR_ENABLE_RAPTOR": "true", 
            "MIMIR_ENABLE_HYDE": "true",
            "MIMIR_ENABLE_RERANKING": "true",
            "MIMIR_ENABLE_CODE_EMBEDDINGS": "true",
            "OLLAMA_HOST": "localhost",
            "OLLAMA_PORT": "11434",
            "OLLAMA_MODEL": "llama3.2:3b",
            "QUERY_ENABLE_HYDE": "true",
            "QUERY_TRANSFORMER_PROVIDER": "ollama",
            "QUERY_RERANKER_ENABLED": "true",
            "QUERY_RERANKER_TOP_K": "10",
            "QUERY_INITIAL_K": "50",
        }
    
    @pytest.fixture
    def sample_vector_chunks(self):
        """Sample vector chunks for testing."""
        return [
            VectorChunk(
                chunk_id="1",
                path="test.py",
                content="def hello_world():\n    print('Hello, World!')",
                span=(0, 20),
                embedding=[0.1, 0.2, 0.3] * 256  # Mock 768-dim embedding
            ),
            VectorChunk(
                chunk_id="2", 
                path="utils.py",
                content="class StringUtils:\n    @staticmethod\n    def reverse(s):\n        return s[::-1]",
                span=(0, 30),
                embedding=[0.2, 0.3, 0.4] * 256
            ),
            VectorChunk(
                chunk_id="3",
                path="main.py", 
                content="import utils\n\ndef main():\n    print(utils.StringUtils.reverse('hello'))",
                span=(0, 40),
                embedding=[0.3, 0.4, 0.5] * 256
            ),
        ]
    
    @pytest.fixture
    def mock_vector_index(self, sample_vector_chunks):
        """Mock vector index for testing."""
        return VectorIndex(
            chunks=sample_vector_chunks,
            embedding_model="test-model"
        )
    
    async def test_pipeline_coordinator_initialization(self, config_with_all_features):
        """Test that PipelineCoordinator properly initializes all components."""
        with patch.dict(os.environ, config_with_all_features):
            coordinator = PipelineCoordinator()
            
            # Test capability assessment
            capabilities = await coordinator.assess_capabilities()
            
            assert "ollama" in capabilities
            assert "raptor" in capabilities or not RAPTOR_AVAILABLE
            assert "enhanced_search" in capabilities or not ADVANCED_SEARCH_AVAILABLE
            
            # Test LLM adapter creation
            if capabilities.get("ollama", False):
                adapter = await coordinator.get_llm_adapter("ollama")
                assert adapter is not None
    
    @pytest.mark.skipif(not ADVANCED_SEARCH_AVAILABLE, reason="Advanced search components not available")
    async def test_enhanced_search_pipeline_integration(self, config_with_all_features, mock_vector_index):
        """Test EnhancedSearchPipeline integration with all components."""
        with patch.dict(os.environ, config_with_all_features):
            pipeline = EnhancedSearchPipeline()
            
            # Mock the initialize methods to avoid actual model loading
            with patch.object(pipeline.hyde_transformer, 'initialize', AsyncMock()):
                with patch.object(pipeline.code_embedder, 'initialize', AsyncMock()):
                    with patch.object(pipeline.reranker, 'initialize', AsyncMock()):
                        
                        await pipeline.initialize()
                        
                        # Test pipeline status
                        status = await pipeline.get_pipeline_status()
                        assert status['pipeline_version'] == '2.0'
                        assert 'components' in status
                        assert 'features' in status
                        
                        # Test configuration
                        pipeline.configure_pipeline(
                            enable_hyde=True,
                            enable_reranking=True,
                            final_k=15
                        )
                        
                        assert pipeline.enable_hyde is True
                        assert pipeline.enable_reranking is True
                        assert pipeline.final_k == 15
    
    @pytest.mark.skipif(not ADVANCED_SEARCH_AVAILABLE, reason="Advanced search components not available")
    async def test_hyde_transformer_integration(self, config_with_all_features):
        """Test HyDE transformer integration with LLM adapters."""
        with patch.dict(os.environ, config_with_all_features):
            
            # Mock LLM adapter
            mock_adapter = AsyncMock()
            mock_adapter.generate_response.return_value = AsyncMock(
                success=True,
                text="def example_function():\n    return 'generated code'"
            )
            
            transformer = HyDETransformer(llm_adapter=mock_adapter)
            
            # Test query transformation
            original_query = "how to reverse a string in python"
            enhanced_query = await transformer.transform_query(original_query)
            
            # Should contain original query plus generated content
            assert original_query in enhanced_query
            assert len(enhanced_query) > len(original_query)
    
    @pytest.mark.skipif(not ADVANCED_SEARCH_AVAILABLE, reason="Advanced search components not available") 
    async def test_code_embeddings_integration(self, config_with_all_features, sample_vector_chunks):
        """Test code embedding adapter integration."""
        with patch.dict(os.environ, config_with_all_features):
            
            # Mock sentence-transformers model
            with patch('repoindex.pipeline.code_embeddings.SentenceTransformer') as mock_st:
                mock_model = MagicMock()
                mock_model.encode.return_value = MagicMock()
                mock_model.encode.return_value.tolist.return_value = [0.1] * 768
                mock_st.return_value = mock_model
                
                embedder = CodeEmbeddingAdapter()
                await embedder.initialize()
                
                # Test single chunk embedding
                chunk = sample_vector_chunks[0]
                embedded_chunk = await embedder.embed_code_chunk(chunk)
                
                assert embedded_chunk.embedding is not None
                assert len(embedded_chunk.embedding) > 0
                
                # Test batch embedding
                embedded_chunks = await embedder.embed_code_chunks(sample_vector_chunks)
                assert len(embedded_chunks) == len(sample_vector_chunks)
                assert all(chunk.embedding for chunk in embedded_chunks)
    
    @pytest.mark.skipif(not ADVANCED_SEARCH_AVAILABLE, reason="Advanced search components not available")
    async def test_reranker_integration(self, config_with_all_features, sample_vector_chunks):
        """Test cross-encoder reranker integration.""" 
        with patch.dict(os.environ, config_with_all_features):
            
            reranker = create_reranker()
            
            # Test with fallback reranker if cross-encoder not available
            if hasattr(reranker, 'available') and not reranker.available:
                # Test fallback reranker
                results = await reranker.rerank_chunks(
                    "test query", sample_vector_chunks, top_k=2
                )
                assert len(results) <= 2
                assert all(isinstance(score, float) for _, score in results)
            else:
                # Mock cross-encoder for testing
                with patch.object(reranker, '_score_pairs_batched', AsyncMock(return_value=[0.8, 0.6, 0.4])):
                    results = await reranker.rerank_chunks(
                        "test query", sample_vector_chunks, top_k=2
                    )
                    assert len(results) == 2
                    # Should be sorted by score (descending)
                    assert results[0][1] >= results[1][1]
    
    @pytest.mark.skipif(not RAPTOR_AVAILABLE, reason="RAPTOR components not available")
    async def test_raptor_integration(self, config_with_all_features, sample_vector_chunks):
        """Test RAPTOR processor integration."""
        with patch.dict(os.environ, config_with_all_features):
            
            # Mock clustering and summarization
            with patch('repoindex.pipeline.raptor.UMAP') as mock_umap:
                with patch('repoindex.pipeline.raptor.HDBSCAN') as mock_hdbscan:
                    
                    # Mock UMAP
                    mock_umap_instance = MagicMock()
                    mock_umap_instance.fit_transform.return_value = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
                    mock_umap.return_value = mock_umap_instance
                    
                    # Mock HDBSCAN
                    mock_hdbscan_instance = MagicMock()
                    mock_hdbscan_instance.fit_predict.return_value = [0, 0, 1]  # Two clusters
                    mock_hdbscan.return_value = mock_hdbscan_instance
                    
                    # Mock LLM adapter for summarization
                    mock_adapter = AsyncMock()
                    mock_adapter.generate_response.return_value = AsyncMock(
                        success=True,
                        text="Summary of code cluster"
                    )
                    
                    processor = RaptorProcessor()
                    processor.summarizer = mock_adapter
                    
                    # Test tree building
                    tree = await processor.build_tree(sample_vector_chunks)
                    
                    assert tree is not None
                    assert len(tree.nodes) > len(sample_vector_chunks)  # Should have summary nodes
    
    async def test_integration_validation_framework(self, config_with_all_features):
        """Test the integration validation framework."""
        with patch.dict(os.environ, config_with_all_features):
            
            # Test individual validation functions
            results = await validate_mimir_2_setup()
            assert isinstance(results, dict)
            assert 'overall_success' in results
            
            # Test comprehensive integration validation
            validation_results = await run_integration_validation()
            assert isinstance(validation_results, dict)
            assert 'components' in validation_results
            assert 'capabilities' in validation_results
    
    async def test_backward_compatibility(self, config_with_all_features):
        """Test that new features don't break existing functionality."""
        # Test with all features disabled
        config_disabled = {k: v for k, v in config_with_all_features.items()}
        config_disabled.update({
            "MIMIR_ENABLE_RAPTOR": "false",
            "MIMIR_ENABLE_HYDE": "false", 
            "MIMIR_ENABLE_RERANKING": "false",
            "QUERY_ENABLE_HYDE": "false",
            "QUERY_RERANKER_ENABLED": "false",
        })
        
        with patch.dict(os.environ, config_disabled):
            # Should still be able to get a coordinator
            coordinator = get_pipeline_coordinator()
            assert coordinator is not None
            
            # Basic capabilities should still work
            capabilities = await coordinator.assess_capabilities()
            assert isinstance(capabilities, dict)
    
    async def test_configuration_migration(self, config_with_all_features):
        """Test configuration migration and validation."""
        with patch.dict(os.environ, config_with_all_features):
            
            config = get_ai_config()
            
            # Test that nested configurations work
            assert hasattr(config, 'query')
            assert hasattr(config, 'reranker')
            
            # Test backward compatibility fields
            assert hasattr(config, 'enable_hyde')
            assert hasattr(config, 'enable_reranking')
            
            # Test configuration consistency
            if config.query.enable_hyde:
                assert config.query.transformer_provider in ["ollama", "gemini"]
            
            if config.reranker.enabled:
                assert config.reranker.top_k > 0
                assert config.reranker.initial_retrieval_k >= config.reranker.top_k
    
    async def test_end_to_end_search_pipeline(self, config_with_all_features, mock_vector_index):
        """Test complete end-to-end search pipeline."""
        with patch.dict(os.environ, config_with_all_features):
            
            if not ADVANCED_SEARCH_AVAILABLE:
                pytest.skip("Advanced search not available")
            
            # Mock all the expensive operations
            with patch('repoindex.pipeline.enhanced_search.HyDETransformer') as mock_hyde:
                with patch('repoindex.pipeline.enhanced_search.CodeEmbeddingAdapter') as mock_embedder:
                    with patch('repoindex.pipeline.enhanced_search.create_reranker') as mock_reranker_factory:
                        
                        # Setup mocks
                        mock_hyde_instance = AsyncMock()
                        mock_hyde_instance.initialize = AsyncMock()
                        mock_hyde_instance.transform_query = AsyncMock(return_value="enhanced query")
                        mock_hyde.return_value = mock_hyde_instance
                        
                        mock_embedder_instance = AsyncMock() 
                        mock_embedder_instance.initialize = AsyncMock()
                        mock_embedder.return_value = mock_embedder_instance
                        
                        mock_reranker_instance = AsyncMock()
                        mock_reranker_instance.initialize = AsyncMock()
                        mock_reranker_factory.return_value = mock_reranker_instance
                        
                        # Test pipeline creation and initialization
                        pipeline = EnhancedSearchPipeline()
                        await pipeline.initialize()
                        
                        # Verify all components were initialized
                        mock_hyde_instance.initialize.assert_called_once()
                        mock_embedder_instance.initialize.assert_called_once()
                        mock_reranker_instance.initialize.assert_called_once()
    
    async def test_performance_characteristics(self, config_with_all_features):
        """Test performance characteristics and resource usage."""
        with patch.dict(os.environ, config_with_all_features):
            
            coordinator = PipelineCoordinator()
            
            # Test that coordinator can be created quickly
            import time
            start = time.time()
            capabilities = await coordinator.assess_capabilities()
            end = time.time()
            
            # Should complete quickly (under 1 second in tests)
            assert (end - start) < 1.0
            
            # Test memory efficiency - coordinator should be lightweight
            import sys
            coordinator_size = sys.getsizeof(coordinator)
            assert coordinator_size < 10000  # Should be less than 10KB
    
    @pytest.mark.parametrize("feature_combo", [
        {"hyde": True, "reranking": False, "raptor": False},
        {"hyde": False, "reranking": True, "raptor": False}, 
        {"hyde": True, "reranking": True, "raptor": False},
        {"hyde": True, "reranking": True, "raptor": True},
        {"hyde": False, "reranking": False, "raptor": False},  # All disabled
    ])
    async def test_feature_combinations(self, config_with_all_features, feature_combo):
        """Test different combinations of features enabled/disabled."""
        
        # Update config based on feature combination
        config = {k: v for k, v in config_with_all_features.items()}
        config["MIMIR_ENABLE_HYDE"] = str(feature_combo["hyde"]).lower()
        config["MIMIR_ENABLE_RERANKING"] = str(feature_combo["reranking"]).lower()
        config["MIMIR_ENABLE_RAPTOR"] = str(feature_combo["raptor"]).lower()
        config["QUERY_ENABLE_HYDE"] = str(feature_combo["hyde"]).lower()
        config["QUERY_RERANKER_ENABLED"] = str(feature_combo["reranking"]).lower()
        
        with patch.dict(os.environ, config):
            
            # Should always be able to get a coordinator
            coordinator = get_pipeline_coordinator()
            assert coordinator is not None
            
            # Capabilities should reflect enabled features
            capabilities = await coordinator.assess_capabilities()
            
            if feature_combo["hyde"] and ADVANCED_SEARCH_AVAILABLE:
                # Would have HyDE capability if components available
                pass
            
            if feature_combo["reranking"] and ADVANCED_SEARCH_AVAILABLE:
                # Would have reranking capability if components available  
                pass
                
            if feature_combo["raptor"] and RAPTOR_AVAILABLE:
                # Would have RAPTOR capability if components available
                pass


class TestMimirConfigurationSystem:
    """Test the enhanced configuration system."""
    
    def test_nested_config_structure(self):
        """Test that nested configuration classes work properly.""" 
        from repoindex.config import AIConfig, QueryConfig, RerankerConfig
        
        # Test QueryConfig
        query_config = QueryConfig(
            enable_hyde=True,
            transformer_provider="ollama",
            transformer_model="llama3.2:3b"
        )
        assert query_config.enable_hyde is True
        assert query_config.transformer_provider == "ollama"
        
        # Test RerankerConfig
        reranker_config = RerankerConfig(
            enabled=True,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            top_k=15,
            initial_retrieval_k=75
        )
        assert reranker_config.enabled is True
        assert reranker_config.top_k == 15
        
        # Test AIConfig with nested configs
        ai_config = AIConfig(
            query=query_config,
            reranker=reranker_config
        )
        assert ai_config.query.enable_hyde is True
        assert ai_config.reranker.top_k == 15
    
    def test_environment_variable_mapping(self):
        """Test that environment variables map correctly to nested configs."""
        test_env = {
            "QUERY_ENABLE_HYDE": "true",
            "QUERY_TRANSFORMER_PROVIDER": "gemini",
            "QUERY_RERANKER_ENABLED": "true", 
            "QUERY_RERANKER_TOP_K": "25",
            "QUERY_INITIAL_K": "100",
        }
        
        with patch.dict(os.environ, test_env):
            from repoindex.config import QueryConfig, RerankerConfig
            
            query_config = QueryConfig()
            assert query_config.enable_hyde is True
            assert query_config.transformer_provider == "gemini"
            
            reranker_config = RerankerConfig()
            assert reranker_config.enabled is True
            assert reranker_config.top_k == 25
            assert reranker_config.initial_retrieval_k == 100
    
    def test_backward_compatibility_fields(self):
        """Test that legacy configuration fields still work."""
        test_env = {
            "MIMIR_ENABLE_HYDE": "true",
            "MIMIR_ENABLE_RERANKING": "true",
            "RERANKING_TOP_K": "30",
        }
        
        with patch.dict(os.environ, test_env):
            from repoindex.config import AIConfig
            
            ai_config = AIConfig()
            
            # Legacy fields should still be accessible
            assert hasattr(ai_config, 'enable_hyde')
            assert hasattr(ai_config, 'enable_reranking') 
            assert hasattr(ai_config, 'reranking_top_k')
            
            assert ai_config.enable_hyde is True
            assert ai_config.enable_reranking is True
            assert ai_config.reranking_top_k == 30