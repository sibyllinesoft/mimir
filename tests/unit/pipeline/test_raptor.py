"""
Unit tests for RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval).

Tests the hierarchical clustering, summarization, and tree-based retrieval
functionality of the RAPTOR system.
"""

import numpy as np
from unittest.mock import AsyncMock, Mock, patch
import pytest

from src.repoindex.pipeline.raptor import (
    RaptorProcessor, 
    RaptorConfig,
    create_raptor_tree,
    query_raptor_tree,
    HAS_ML_DEPS
)
from src.repoindex.pipeline.raptor_structures import RaptorNode, RaptorTree
from src.repoindex.data.schemas import VectorChunk, VectorIndex
from src.repoindex.config import AIConfig


class TestRaptorProcessor:
    """Test RaptorProcessor functionality."""

    @pytest.fixture
    def sample_chunks(self):
        """Create sample vector chunks for testing."""
        # Create embeddings with some similarity patterns
        embedding1 = np.random.normal(0, 1, 384).tolist()
        embedding2 = (np.array(embedding1) + np.random.normal(0, 0.1, 384)).tolist()  # Similar to embedding1
        embedding3 = np.random.normal(0, 1, 384).tolist()  # Different
        embedding4 = (np.array(embedding3) + np.random.normal(0, 0.1, 384)).tolist()  # Similar to embedding3
        
        return [
            VectorChunk(
                path="src/math.py",
                span=(1, 10),
                content="def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b",
                embedding=embedding1,
                token_count=25
            ),
            VectorChunk(
                path="src/math.py",
                span=(11, 20),
                content="def multiply(a, b):\n    return a * b\n\ndef divide(a, b):\n    return a / b",
                embedding=embedding2,
                token_count=25
            ),
            VectorChunk(
                path="src/string_utils.py",
                span=(1, 15),
                content="def concat(a, b):\n    return str(a) + str(b)\n\ndef uppercase(s):\n    return s.upper()",
                embedding=embedding3,
                token_count=30
            ),
            VectorChunk(
                path="src/string_utils.py", 
                span=(16, 25),
                content="def lowercase(s):\n    return s.lower()\n\ndef trim(s):\n    return s.strip()",
                embedding=embedding4,
                token_count=25
            )
        ]

    @pytest.fixture
    def vector_index(self, sample_chunks):
        """Create a vector index with sample chunks."""
        return VectorIndex(
            chunks=sample_chunks,
            dimension=384,
            total_tokens=sum(chunk.token_count for chunk in sample_chunks),
            model_name="test-model"
        )

    @pytest.fixture
    def raptor_config(self):
        """Create RAPTOR configuration."""
        return RaptorConfig(
            max_clusters=3,
            min_cluster_size=2,
            summarization_length=100,
            tree_depth=2,
            similarity_threshold=0.7
        )

    @pytest.fixture
    def ai_config(self):
        """Create AI configuration for testing."""
        from src.repoindex.config import GeminiConfig
        return AIConfig(
            provider="gemini",
            gemini=GeminiConfig(api_key="test-key")
        )

    @pytest.fixture
    def raptor_processor(self, raptor_config, ai_config):
        """Create RaptorProcessor instance."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator:
            processor = RaptorProcessor(ai_config, raptor_config, mock_coordinator())
            return processor

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_initialize_processor(self, raptor_processor):
        """Test processor initialization."""
        await raptor_processor.initialize()
        
        assert raptor_processor._initialized is True

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_process_embeddings_success(self, raptor_processor, vector_index):
        """Test successful processing of embeddings into RAPTOR tree."""
        # Mock the LLM calls for summarization
        mock_responses = [
            "Mathematical operations: functions for basic arithmetic operations including addition, subtraction, multiplication, and division.",
            "String utilities: functions for string manipulation including concatenation, case conversion, and whitespace trimming.",
            "Core utilities: comprehensive collection of mathematical and string processing functions for common programming tasks."
        ]
        
        with patch.object(raptor_processor.coordinator, 'generate_text', side_effect=mock_responses):
            await raptor_processor.initialize()
            tree = await raptor_processor.process_embeddings(vector_index)
            
            assert isinstance(tree, RaptorTree)
            assert tree.root is not None
            assert len(tree.nodes) > len(vector_index.chunks)  # Should have more nodes due to clusters
            
            # Check that leaf nodes correspond to original chunks
            leaf_nodes = [node for node in tree.nodes.values() if node.is_leaf]
            assert len(leaf_nodes) == len(vector_index.chunks)
            
            # Check that we have cluster nodes
            cluster_nodes = [node for node in tree.nodes.values() if not node.is_leaf]
            assert len(cluster_nodes) > 0

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_create_leaf_nodes(self, raptor_processor, vector_index):
        """Test creation of leaf nodes from vector chunks."""
        await raptor_processor.initialize()
        leaf_nodes = raptor_processor._create_leaf_nodes(vector_index.chunks)
        
        assert len(leaf_nodes) == len(vector_index.chunks)
        
        for i, (node, chunk) in enumerate(zip(leaf_nodes, vector_index.chunks)):
            assert node.node_id == f"leaf_{i}"
            assert node.is_leaf is True
            assert node.content == chunk.content
            assert node.path == chunk.path
            assert node.span == chunk.span
            assert np.array_equal(node.embedding, chunk.embedding)
            assert node.children == []

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_cluster_embeddings(self, raptor_processor, sample_chunks):
        """Test embedding clustering functionality."""
        await raptor_processor.initialize()
        
        embeddings = np.array([chunk.embedding for chunk in sample_chunks])
        cluster_labels = raptor_processor._cluster_embeddings(embeddings, n_clusters=2)
        
        assert len(cluster_labels) == len(sample_chunks)
        assert all(isinstance(label, (int, np.integer)) for label in cluster_labels)
        assert len(set(cluster_labels)) <= 2  # Should have at most 2 clusters

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_create_cluster_nodes(self, raptor_processor, sample_chunks):
        """Test creation of cluster nodes from grouped chunks."""
        await raptor_processor.initialize()
        
        # Create leaf nodes first
        leaf_nodes = raptor_processor._create_leaf_nodes(sample_chunks)
        
        # Group nodes into clusters (simulating clustering result)
        clusters = {
            0: leaf_nodes[:2],  # Math functions
            1: leaf_nodes[2:]   # String functions
        }
        
        mock_summaries = [
            "Mathematical operations for basic arithmetic",
            "String manipulation and formatting functions"
        ]
        
        with patch.object(raptor_processor, '_summarize_cluster', side_effect=mock_summaries):
            cluster_nodes = await raptor_processor._create_cluster_nodes(clusters, level=0)
            
            assert len(cluster_nodes) == 2
            
            for node in cluster_nodes:
                assert not node.is_leaf
                assert len(node.children) == 2
                assert node.content in mock_summaries
                assert node.embedding is not None
                assert len(node.embedding) == 384

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available") 
    @pytest.mark.asyncio
    async def test_summarize_cluster(self, raptor_processor, sample_chunks):
        """Test cluster summarization functionality."""
        await raptor_processor.initialize()
        
        # Mock LLM response
        expected_summary = "This cluster contains mathematical functions for arithmetic operations."
        
        with patch.object(raptor_processor.coordinator, 'generate_text', return_value=expected_summary):
            summary = await raptor_processor._summarize_cluster(sample_chunks[:2])
            
            assert summary == expected_summary
            
            # Verify the LLM was called with appropriate context
            raptor_processor.coordinator.generate_text.assert_called_once()
            call_args = raptor_processor.coordinator.generate_text.call_args[0]
            prompt = call_args[0]
            
            assert "summarize" in prompt.lower()
            assert sample_chunks[0].content in prompt
            assert sample_chunks[1].content in prompt

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    def test_calculate_cluster_metrics(self, raptor_processor, sample_chunks):
        """Test cluster quality metrics calculation."""
        # Create two similar embeddings and one dissimilar
        embedding1 = np.random.normal(0, 1, 384)
        embedding2 = embedding1 + np.random.normal(0, 0.1, 384)  # Very similar
        embedding3 = np.random.normal(0, 1, 384)  # Different
        
        cluster_embeddings = np.array([embedding1, embedding2, embedding3])
        
        metrics = raptor_processor._calculate_cluster_metrics(cluster_embeddings)
        
        assert 'coherence' in metrics
        assert 'separation' in metrics
        assert 'silhouette' in metrics
        assert isinstance(metrics['coherence'], float)
        assert isinstance(metrics['separation'], float)
        assert isinstance(metrics['silhouette'], float)

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_create_root_node(self, raptor_processor, sample_chunks):
        """Test creation of root node."""
        await raptor_processor.initialize()
        
        # Create some cluster nodes to serve as children
        leaf_nodes = raptor_processor._create_leaf_nodes(sample_chunks)
        clusters = {0: leaf_nodes}
        
        mock_summary = "Root summary of all mathematical and string utility functions"
        
        with patch.object(raptor_processor, '_summarize_cluster', return_value=mock_summary):
            cluster_nodes = await raptor_processor._create_cluster_nodes(clusters, level=0)
            root_node = await raptor_processor._create_root_node(cluster_nodes)
            
            assert root_node.node_id == "root"
            assert not root_node.is_leaf
            assert root_node.content == mock_summary
            assert len(root_node.children) == len(cluster_nodes)
            assert root_node.embedding is not None

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_query_tree(self, raptor_processor, vector_index):
        """Test querying the RAPTOR tree."""
        # Create a tree first
        mock_summaries = [
            "Mathematical operations and arithmetic functions",
            "String processing and text utilities",
            "Comprehensive programming utilities for math and text"
        ]
        
        with patch.object(raptor_processor.coordinator, 'generate_text', side_effect=mock_summaries):
            await raptor_processor.initialize()
            tree = await raptor_processor.process_embeddings(vector_index)
            
            # Query the tree
            query = "mathematical functions for addition"
            results = await raptor_processor.query_tree(tree, query, top_k=3)
            
            assert isinstance(results, list)
            assert len(results) <= 3
            
            for result in results:
                assert 'node' in result
                assert 'score' in result
                assert isinstance(result['node'], RaptorNode)
                assert isinstance(result['score'], float)
                assert 0 <= result['score'] <= 1

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    def test_cosine_similarity(self, raptor_processor):
        """Test cosine similarity calculation."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        vec3 = np.array([1, 0, 0])
        
        # Perpendicular vectors
        sim1 = raptor_processor._cosine_similarity(vec1, vec2)
        assert abs(sim1) < 0.001  # Should be ~0
        
        # Identical vectors
        sim2 = raptor_processor._cosine_similarity(vec1, vec3)
        assert abs(sim2 - 1.0) < 0.001  # Should be ~1

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_tree_traversal(self, raptor_processor, vector_index):
        """Test top-down tree traversal during querying."""
        mock_summaries = ["Math functions", "String functions", "All utilities"]
        
        with patch.object(raptor_processor.coordinator, 'generate_text', side_effect=mock_summaries):
            await raptor_processor.initialize()
            tree = await raptor_processor.process_embeddings(vector_index)
            
            query_embedding = np.random.normal(0, 1, 384).tolist()
            
            results = raptor_processor._top_down_traversal(
                tree.root, query_embedding, top_k=2, current_results=[]
            )
            
            assert isinstance(results, list)
            assert len(results) <= 2
            
            # Results should be sorted by score (descending)
            if len(results) > 1:
                assert results[0]['score'] >= results[1]['score']


class TestRaptorConfig:
    """Test RAPTOR configuration."""

    def test_raptor_config_defaults(self):
        """Test RAPTOR config default values."""
        config = RaptorConfig()
        
        assert config.max_clusters == 10
        assert config.min_cluster_size == 3
        assert config.summarization_length == 150
        assert config.tree_depth == 3
        assert config.similarity_threshold == 0.8
        assert config.clustering_algorithm == "kmeans"

    def test_raptor_config_custom(self):
        """Test RAPTOR config with custom values."""
        config = RaptorConfig(
            max_clusters=5,
            min_cluster_size=2,
            summarization_length=200,
            tree_depth=2,
            similarity_threshold=0.7,
            clustering_algorithm="hierarchical",
            enable_dynamic_clustering=True
        )
        
        assert config.max_clusters == 5
        assert config.min_cluster_size == 2
        assert config.summarization_length == 200
        assert config.tree_depth == 2
        assert config.similarity_threshold == 0.7
        assert config.clustering_algorithm == "hierarchical"
        assert config.enable_dynamic_clustering is True


class TestRaptorFunctions:
    """Test standalone RAPTOR functions."""

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_create_raptor_tree_function(self, vector_index):
        """Test the create_raptor_tree convenience function."""
        ai_config = AIConfig(provider="gemini")
        raptor_config = RaptorConfig(max_clusters=2)
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.generate_text.return_value = "Test summary"
            mock_coordinator_class.return_value = mock_coordinator
            
            tree = await create_raptor_tree(vector_index, ai_config, raptor_config)
            
            assert isinstance(tree, RaptorTree)
            assert tree.root is not None

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_query_raptor_tree_function(self, vector_index):
        """Test the query_raptor_tree convenience function."""
        ai_config = AIConfig(provider="gemini")
        raptor_config = RaptorConfig(max_clusters=2)
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.generate_text.return_value = "Test summary"
            mock_coordinator_class.return_value = mock_coordinator
            
            # Create tree first
            tree = await create_raptor_tree(vector_index, ai_config, raptor_config)
            
            # Query the tree
            results = await query_raptor_tree(tree, "test query", ai_config, top_k=2)
            
            assert isinstance(results, list)
            assert len(results) <= 2


class TestRaptorErrorHandling:
    """Test RAPTOR error handling and edge cases."""

    @pytest.fixture
    def empty_vector_index(self):
        """Create an empty vector index."""
        return VectorIndex(
            chunks=[],
            dimension=384,
            total_tokens=0,
            model_name="test-model"
        )

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_process_empty_embeddings(self, raptor_processor, empty_vector_index):
        """Test processing with no embeddings."""
        await raptor_processor.initialize()
        
        with pytest.raises(ValueError) as exc_info:
            await raptor_processor.process_embeddings(empty_vector_index)
        
        assert "empty" in str(exc_info.value).lower()

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_process_single_embedding(self, raptor_processor):
        """Test processing with only one embedding."""
        single_chunk = VectorChunk(
            path="test.py",
            span=(1, 5),
            content="def test(): pass",
            embedding=np.random.normal(0, 1, 384).tolist(),
            token_count=10
        )
        
        vector_index = VectorIndex(
            chunks=[single_chunk],
            dimension=384,
            total_tokens=10,
            model_name="test-model"
        )
        
        await raptor_processor.initialize()
        
        with patch.object(raptor_processor.coordinator, 'generate_text', return_value="Single function"):
            tree = await raptor_processor.process_embeddings(vector_index)
            
            assert isinstance(tree, RaptorTree)
            assert tree.root is not None
            # Should have at least the leaf node
            assert len(tree.nodes) >= 1

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_summarization_failure(self, raptor_processor, vector_index):
        """Test handling of LLM summarization failures."""
        await raptor_processor.initialize()
        
        # Mock LLM failure
        with patch.object(raptor_processor.coordinator, 'generate_text', side_effect=Exception("LLM error")):
            with pytest.raises(Exception) as exc_info:
                await raptor_processor.process_embeddings(vector_index)
            
            assert "LLM error" in str(exc_info.value)

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    @pytest.mark.asyncio
    async def test_clustering_failure(self, raptor_processor, vector_index):
        """Test handling of clustering failures."""
        await raptor_processor.initialize()
        
        # Mock clustering failure
        with patch.object(raptor_processor, '_cluster_embeddings', side_effect=Exception("Clustering failed")):
            with pytest.raises(Exception) as exc_info:
                await raptor_processor.process_embeddings(vector_index)
            
            assert "Clustering failed" in str(exc_info.value)

    @pytest.mark.skipif(not HAS_ML_DEPS, reason="ML dependencies not available")
    def test_invalid_embeddings_dimension(self, raptor_processor):
        """Test handling of invalid embedding dimensions."""
        # Create chunks with mismatched embedding dimensions
        chunks = [
            VectorChunk(
                path="test1.py",
                span=(1, 5),
                content="def test1(): pass",
                embedding=np.random.normal(0, 1, 384).tolist(),
                token_count=10
            ),
            VectorChunk(
                path="test2.py", 
                span=(1, 5),
                content="def test2(): pass",
                embedding=np.random.normal(0, 1, 256).tolist(),  # Wrong dimension
                token_count=10
            )
        ]
        
        vector_index = VectorIndex(
            chunks=chunks,
            dimension=384,
            total_tokens=20,
            model_name="test-model"
        )
        
        # Should handle dimension mismatch gracefully
        with pytest.raises(ValueError) as exc_info:
            embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # NumPy should raise error for inconsistent dimensions
        assert "shape" in str(exc_info.value).lower() or "dimension" in str(exc_info.value).lower()


@pytest.mark.skipif(HAS_ML_DEPS, reason="Testing ML dependencies unavailable scenario")
class TestRaptorWithoutMLDeps:
    """Test RAPTOR behavior when ML dependencies are not available."""

    def test_raptor_processor_init_without_ml_deps(self):
        """Test RaptorProcessor initialization without ML dependencies."""
        from src.repoindex.config import AIConfig
        
        ai_config = AIConfig(provider="gemini")
        raptor_config = RaptorConfig()
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator:
            processor = RaptorProcessor(ai_config, raptor_config, mock_coordinator())
            
            # Should initialize but functionality will be limited
            assert processor.ai_config == ai_config
            assert processor.config == raptor_config

    @pytest.mark.asyncio
    async def test_raptor_functions_without_ml_deps(self):
        """Test RAPTOR functions raise appropriate errors without ML deps."""
        from src.repoindex.data.schemas import VectorIndex, VectorChunk
        
        vector_index = VectorIndex(chunks=[], dimension=384, total_tokens=0, model_name="test")
        ai_config = AIConfig(provider="gemini")
        raptor_config = RaptorConfig()
        
        # Should raise ImportError or similar when trying to use ML functionality
        with pytest.raises((ImportError, RuntimeError)) as exc_info:
            await create_raptor_tree(vector_index, ai_config, raptor_config)
        
        error_msg = str(exc_info.value).lower()
        assert any(keyword in error_msg for keyword in ["sklearn", "numpy", "ml", "dependencies"])