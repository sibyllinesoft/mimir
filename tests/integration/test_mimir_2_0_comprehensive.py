"""
Comprehensive integration tests for Mimir 2.0 features.

Tests the complete integration of all Mimir 2.0 components including
RAPTOR, enhanced search, Ollama adapter, and configuration management.
"""

import asyncio
import json
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
from src.repoindex.pipeline.ollama import OllamaAdapter
from src.repoindex.pipeline.llm_adapter_factory import LLMAdapterFactory
from src.repoindex.pipeline.enhanced_search import EnhancedSearchPipeline
from src.repoindex.pipeline.raptor import RaptorProcessor


@pytest.fixture
def comprehensive_config():
    """Create comprehensive Mimir 2.0 configuration."""
    from src.repoindex.config import OllamaConfig, GeminiConfig
    
    return MimirConfig(
        ai=AIConfig(
            provider="auto",
            fallback_providers=["ollama", "gemini"],
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                model="llama2:7b",
                timeout=30
            ),
            gemini=GeminiConfig(
                api_key="test-gemini-key",
                model="gemini-1.5-flash"
            )
        ),
        query=QueryConfig(
            max_results=20,
            use_hyde=True,
            hyde_iterations=3,
            enable_reranking=True,
            fusion_method="rrf"
        ),
        reranker=RerankerConfig(
            enabled=True,
            provider="cross_encoder",
            top_k=15
        ),
        pipeline=PipelineConfig(
            enable_raptor=True,
            enable_code_embeddings=True,
            enable_hybrid_search=True,
            chunk_size=512,
            max_concurrent_tasks=4
        )
    )


@pytest.fixture
def comprehensive_vector_index():
    """Create comprehensive vector index for integration testing."""
    chunks = []
    
    # Algorithm implementations
    algorithm_code = [
        ("src/algorithms/sorting/quicksort.py", """
def quicksort(arr, low=0, high=None):
    if high is None:
        high = len(arr) - 1
    
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi - 1)
        quicksort(arr, pi + 1, high)
    return arr

def partition(arr, low, high):
    pivot = arr[high]
    i = low - 1
    
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1
"""),
        ("src/algorithms/searching/binary_search.py", """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

def binary_search_recursive(arr, target, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left > right:
        return -1
    
    mid = (left + right) // 2
    
    if arr[mid] == target:
        return mid
    elif arr[mid] < target:
        return binary_search_recursive(arr, target, mid + 1, right)
    else:
        return binary_search_recursive(arr, target, left, mid - 1)
"""),
        ("src/data_structures/linked_list.py", """
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    def __init__(self):
        self.head = None
        self.size = 0
    
    def append(self, val):
        new_node = ListNode(val)
        if not self.head:
            self.head = new_node
        else:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        self.size += 1
    
    def prepend(self, val):
        new_node = ListNode(val, self.head)
        self.head = new_node
        self.size += 1
    
    def delete(self, val):
        if not self.head:
            return False
        
        if self.head.val == val:
            self.head = self.head.next
            self.size -= 1
            return True
        
        current = self.head
        while current.next:
            if current.next.val == val:
                current.next = current.next.next
                self.size -= 1
                return True
            current = current.next
        
        return False
""")
    ]
    
    # Generate embeddings for each code snippet
    for i, (path, content) in enumerate(algorithm_code):
        # Create embeddings that cluster algorithms, data structures, etc.
        if "sorting" in path:
            base_embedding = np.array([1, 0, 0] + [0] * 381)
        elif "searching" in path:
            base_embedding = np.array([0, 1, 0] + [0] * 381)
        elif "data_structures" in path:
            base_embedding = np.array([0, 0, 1] + [0] * 381)
        else:
            base_embedding = np.array([0.3, 0.3, 0.4] + [0] * 381)
        
        # Add some noise for realism
        embedding = (base_embedding + np.random.normal(0, 0.1, 384)).tolist()
        
        chunks.append(VectorChunk(
            path=path,
            span=(1, content.count('\n') + 1),
            content=content.strip(),
            embedding=embedding,
            token_count=len(content.split())
        ))
    
    # Add some utility functions
    utility_code = [
        ("src/utils/math_helpers.py", """
import math

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    return abs(a * b) // gcd(a, b)

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
"""),
        ("src/utils/string_helpers.py", """
def camel_to_snake(name):
    import re
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def snake_to_camel(name):
    components = name.split('_')
    return components[0] + ''.join(x.title() for x in components[1:])

def is_palindrome(s):
    s = ''.join(c.lower() for c in s if c.isalnum())
    return s == s[::-1]

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]
""")
    ]
    
    for i, (path, content) in enumerate(utility_code):
        # Utility embeddings
        base_embedding = np.array([0.5, 0.5, 0] + [0] * 381)
        embedding = (base_embedding + np.random.normal(0, 0.1, 384)).tolist()
        
        chunks.append(VectorChunk(
            path=path,
            span=(1, content.count('\n') + 1),
            content=content.strip(),
            embedding=embedding,
            token_count=len(content.split())
        ))
    
    return VectorIndex(
        chunks=chunks,
        dimension=384,
        total_tokens=sum(chunk.token_count for chunk in chunks),
        model_name="comprehensive-test-index"
    )


class TestMimir20FullIntegration:
    """Test complete Mimir 2.0 feature integration."""

    @pytest.mark.asyncio
    async def test_llm_adapter_factory_integration(self, comprehensive_config):
        """Test LLM adapter factory with multiple providers."""
        factory = LLMAdapterFactory()
        
        # Test Ollama adapter creation
        with patch.object(OllamaAdapter, 'is_available', return_value=True):
            adapter = await factory.create_adapter(comprehensive_config.ai)
            assert isinstance(adapter, OllamaAdapter)
            assert adapter.base_url == "http://localhost:11434"
            assert adapter.timeout == 30

    @pytest.mark.asyncio
    async def test_enhanced_search_pipeline_integration(self, comprehensive_config, comprehensive_vector_index):
        """Test enhanced search pipeline with all features enabled."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            search_pipeline = EnhancedSearchPipeline(
                ai_config=comprehensive_config.ai,
                query_config=comprehensive_config.query,
                reranker_config=comprehensive_config.reranker,
                coordinator=mock_coordinator
            )
            
            await search_pipeline.initialize()
            
            # Mock all pipeline components
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = [
                    "binary search algorithm implementation",
                    "search algorithm for sorted arrays",
                    "efficient searching in ordered data structures"
                ]
                mock_hyde_class.return_value = mock_hyde
                
                with patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class:
                    mock_embedder = AsyncMock()
                    # Return query embedding similar to search algorithms
                    mock_embedder.embed_query.return_value = np.array([0, 1, 0] + [0] * 381).tolist()
                    mock_embed_class.return_value = mock_embedder
                    
                    with patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                        mock_reranker = AsyncMock()
                        
                        # Mock reranker to prefer binary search results
                        def mock_rerank(query, results):
                            def score_fn(item):
                                chunk, original_score = item
                                if "binary_search" in chunk.content:
                                    return 0.9
                                elif "search" in chunk.path:
                                    return 0.7
                                else:
                                    return original_score
                            
                            scored_results = [(chunk, score_fn((chunk, score))) 
                                            for chunk, score in results]
                            return sorted(scored_results, key=lambda x: x[1], reverse=True)[:10]
                        
                        mock_reranker.rerank.side_effect = mock_rerank
                        mock_reranker_class.return_value = mock_reranker
                        
                        # Execute search
                        results = await search_pipeline.search(
                            query="binary search algorithm implementation",
                            vector_index=comprehensive_vector_index,
                            top_k=5
                        )
                        
                        assert len(results) <= 5
                        assert len(results) > 0
                        
                        # Verify binary search results are ranked highly
                        top_result = results[0]
                        assert "binary_search" in top_result[0].content or "search" in top_result[0].path

    @pytest.mark.skipif(
        'not HAS_ML_DEPS', 
        reason="ML dependencies not available"
    )
    @pytest.mark.asyncio
    async def test_raptor_integration(self, comprehensive_config, comprehensive_vector_index):
        """Test RAPTOR system integration."""
        from src.repoindex.pipeline.raptor import RaptorConfig
        
        raptor_config = RaptorConfig(
            max_clusters=3,
            min_cluster_size=2,
            tree_depth=2
        )
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.generate_text.side_effect = [
                "Sorting algorithms including quicksort with partitioning logic",
                "Search algorithms for finding elements in data structures",
                "Utility functions for mathematical and string operations",
                "Comprehensive algorithm and utility library for data processing"
            ]
            mock_coordinator_class.return_value = mock_coordinator
            
            raptor_processor = RaptorProcessor(
                ai_config=comprehensive_config.ai,
                config=raptor_config,
                coordinator=mock_coordinator
            )
            
            await raptor_processor.initialize()
            tree = await raptor_processor.process_embeddings(comprehensive_vector_index)
            
            assert tree is not None
            assert tree.root is not None
            assert len(tree.nodes) > len(comprehensive_vector_index.chunks)
            
            # Test querying the tree
            query_results = await raptor_processor.query_tree(
                tree, 
                "quicksort algorithm implementation", 
                top_k=3
            )
            
            assert len(query_results) <= 3
            assert all('node' in result and 'score' in result for result in query_results)

    @pytest.mark.asyncio
    async def test_end_to_end_search_workflow(self, comprehensive_config, comprehensive_vector_index):
        """Test complete end-to-end search workflow."""
        # This test simulates a complete search workflow from query to results
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator.generate_text.return_value = "Enhanced query about sorting algorithms"
            mock_coordinator_class.return_value = mock_coordinator
            
            # Initialize all components
            factory = LLMAdapterFactory()
            
            with patch.object(OllamaAdapter, 'is_available', return_value=True):
                llm_adapter = await factory.create_adapter(comprehensive_config.ai)
                
                # Create enhanced search pipeline
                search_pipeline = EnhancedSearchPipeline(
                    ai_config=comprehensive_config.ai,
                    query_config=comprehensive_config.query,
                    reranker_config=comprehensive_config.reranker,
                    coordinator=mock_coordinator
                )
                
                await search_pipeline.initialize()
                
                # Mock all pipeline components for end-to-end test
                with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                     patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                     patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                    
                    # Setup HyDE
                    mock_hyde = AsyncMock()
                    mock_hyde.expand_query.return_value = [
                        "quicksort implementation",
                        "sorting algorithm with divide and conquer",
                        "efficient array sorting method"
                    ]
                    mock_hyde_class.return_value = mock_hyde
                    
                    # Setup embeddings
                    mock_embedder = AsyncMock()
                    mock_embedder.embed_query.return_value = np.array([1, 0, 0] + [0] * 381).tolist()
                    mock_embed_class.return_value = mock_embedder
                    
                    # Setup reranker
                    mock_reranker = AsyncMock()
                    def rerank_for_sorting(query, results):
                        scored = []
                        for chunk, score in results:
                            if "quicksort" in chunk.content.lower():
                                scored.append((chunk, 0.95))
                            elif "sort" in chunk.path.lower():
                                scored.append((chunk, 0.85))
                            else:
                                scored.append((chunk, score * 0.7))
                        return sorted(scored, key=lambda x: x[1], reverse=True)[:5]
                    
                    mock_reranker.rerank.side_effect = rerank_for_sorting
                    mock_reranker_class.return_value = mock_reranker
                    
                    # Execute end-to-end search
                    final_results = await search_pipeline.search(
                        query="quicksort algorithm implementation",
                        vector_index=comprehensive_vector_index,
                        top_k=5
                    )
                    
                    # Verify results
                    assert len(final_results) <= 5
                    assert len(final_results) > 0
                    
                    # Check that quicksort is ranked highly
                    top_result = final_results[0]
                    assert ("quicksort" in top_result[0].content.lower() or 
                           "sort" in top_result[0].path.lower())
                    
                    # Verify all components were used
                    mock_hyde.expand_query.assert_called_once()
                    mock_embedder.embed_query.assert_called()
                    mock_reranker.rerank.assert_called()

    @pytest.mark.asyncio
    async def test_configuration_management_integration(self, comprehensive_config):
        """Test configuration management across all components."""
        # Test configuration loading and validation
        from src.repoindex.config import validate_config, get_config, set_config
        
        # Validate comprehensive configuration
        validate_config(comprehensive_config)
        
        # Test setting and getting global configuration
        original_config = get_config()
        set_config(comprehensive_config)
        
        retrieved_config = get_config()
        assert retrieved_config.ai.provider == "auto"
        assert "ollama" in retrieved_config.ai.fallback_providers
        assert retrieved_config.query.use_hyde is True
        assert retrieved_config.reranker.enabled is True
        assert retrieved_config.pipeline.enable_raptor is True
        
        # Restore original config
        set_config(original_config)

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, comprehensive_config, comprehensive_vector_index):
        """Test concurrent operations with multiple pipeline components."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            # Create multiple search pipelines
            pipelines = []
            for i in range(3):
                pipeline = EnhancedSearchPipeline(
                    ai_config=comprehensive_config.ai,
                    query_config=comprehensive_config.query,
                    reranker_config=comprehensive_config.reranker,
                    coordinator=mock_coordinator
                )
                await pipeline.initialize()
                pipelines.append(pipeline)
            
            # Mock components for concurrent operations
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["expanded query"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                mock_reranker.rerank.return_value = [(comprehensive_vector_index.chunks[0], 0.9)]
                mock_reranker_class.return_value = mock_reranker
                
                # Run concurrent searches
                queries = [
                    "binary search implementation",
                    "quicksort algorithm",
                    "linked list operations"
                ]
                
                async def search_task(pipeline, query):
                    return await pipeline.search(
                        query=query,
                        vector_index=comprehensive_vector_index,
                        top_k=3
                    )
                
                # Execute concurrent searches
                tasks = [search_task(pipelines[i], queries[i]) for i in range(3)]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Verify all searches completed successfully
                assert len(results) == 3
                for result in results:
                    assert not isinstance(result, Exception)
                    assert isinstance(result, list)
                    assert len(result) <= 3

    @pytest.mark.asyncio
    async def test_error_handling_integration(self, comprehensive_config, comprehensive_vector_index):
        """Test error handling across integrated components."""
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            search_pipeline = EnhancedSearchPipeline(
                ai_config=comprehensive_config.ai,
                query_config=comprehensive_config.query,
                reranker_config=comprehensive_config.reranker,
                coordinator=mock_coordinator
            )
            
            await search_pipeline.initialize()
            
            # Test HyDE failure with graceful fallback
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.side_effect = Exception("HyDE service failed")
                mock_hyde_class.return_value = mock_hyde
                
                with pytest.raises(Exception) as exc_info:
                    await search_pipeline.search(
                        query="test query",
                        vector_index=comprehensive_vector_index,
                        top_k=3
                    )
                
                assert "HyDE service failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_memory_management_integration(self, comprehensive_config):
        """Test memory management with large-scale operations."""
        # Create a large vector index to test memory handling
        large_chunks = []
        for i in range(1000):  # Large number of chunks
            embedding = np.random.normal(0, 1, 384).tolist()
            large_chunks.append(VectorChunk(
                path=f"src/large_file_{i}.py",
                span=(1, 50),
                content=f"def large_function_{i}():\n    # Function implementation {i}\n    return {i}",
                embedding=embedding,
                token_count=20
            ))
        
        large_index = VectorIndex(
            chunks=large_chunks,
            dimension=384,
            total_tokens=sum(chunk.token_count for chunk in large_chunks),
            model_name="large-test-index"
        )
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            search_pipeline = EnhancedSearchPipeline(
                ai_config=comprehensive_config.ai,
                query_config=comprehensive_config.query,
                reranker_config=comprehensive_config.reranker,
                coordinator=mock_coordinator
            )
            
            await search_pipeline.initialize()
            
            # Mock components to handle large-scale operations
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class, \
                 patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class, \
                 patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                
                mock_hyde = AsyncMock()
                mock_hyde.expand_query.return_value = ["large scale query"]
                mock_hyde_class.return_value = mock_hyde
                
                mock_embedder = AsyncMock()
                mock_embedder.embed_query.return_value = np.random.normal(0, 1, 384).tolist()
                mock_embed_class.return_value = mock_embedder
                
                mock_reranker = AsyncMock()
                # Return subset of results to simulate realistic reranking
                mock_reranker.rerank.return_value = [(large_chunks[i], 0.8 - i * 0.01) for i in range(10)]
                mock_reranker_class.return_value = mock_reranker
                
                # Execute search on large index
                results = await search_pipeline.search(
                    query="large scale function",
                    vector_index=large_index,
                    top_k=10
                )
                
                # Should handle large index without memory issues
                assert len(results) <= 10
                assert isinstance(results, list)


@pytest.mark.integration
class TestMimir20ComponentInteraction:
    """Test interactions between Mimir 2.0 components."""

    @pytest.mark.asyncio
    async def test_adapter_factory_with_enhanced_search(self, comprehensive_config):
        """Test LLM adapter factory integration with enhanced search."""
        factory = LLMAdapterFactory()
        
        # Test that search pipeline can use adapters from factory
        with patch.object(OllamaAdapter, 'is_available', return_value=True):
            adapter = await factory.create_adapter(comprehensive_config.ai)
            
            # Verify adapter can be used in search context
            assert adapter.get_provider_name() == "ollama"
            
            # Mock text generation for search enhancement
            with patch.object(adapter, 'generate_text', return_value="Enhanced search query"):
                result = await adapter.generate_text("test query", "llama2:7b")
                assert result == "Enhanced search query"

    @pytest.mark.asyncio
    async def test_configuration_propagation(self, comprehensive_config):
        """Test configuration propagation across components."""
        # Verify configuration values are properly propagated
        assert comprehensive_config.ai.ollama.timeout == 30
        assert comprehensive_config.query.hyde_iterations == 3
        assert comprehensive_config.reranker.top_k == 15
        assert comprehensive_config.pipeline.chunk_size == 512
        
        # Test component initialization with config values
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            search_pipeline = EnhancedSearchPipeline(
                ai_config=comprehensive_config.ai,
                query_config=comprehensive_config.query,
                reranker_config=comprehensive_config.reranker,
                coordinator=mock_coordinator
            )
            
            # Verify components receive correct configuration
            assert search_pipeline.query_config.hyde_iterations == 3
            assert search_pipeline.reranker_config.top_k == 15

    @pytest.mark.asyncio
    async def test_pipeline_stage_coordination(self, comprehensive_config, comprehensive_vector_index):
        """Test coordination between different pipeline stages."""
        # This test verifies that pipeline stages work together correctly
        
        with patch('src.repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator = Mock()
            mock_coordinator_class.return_value = mock_coordinator
            
            # Track stage execution order
            stage_execution_order = []
            
            # Mock HyDE stage
            with patch('src.repoindex.pipeline.hyde.HyDEProcessor') as mock_hyde_class:
                mock_hyde = AsyncMock()
                async def mock_expand_query(*args, **kwargs):
                    stage_execution_order.append("hyde")
                    return ["expanded query 1", "expanded query 2"]
                mock_hyde.expand_query = mock_expand_query
                mock_hyde_class.return_value = mock_hyde
                
                # Mock embedding stage
                with patch('src.repoindex.pipeline.code_embeddings.CodeEmbeddingProcessor') as mock_embed_class:
                    mock_embedder = AsyncMock()
                    async def mock_embed_query(*args, **kwargs):
                        stage_execution_order.append("embedding")
                        return np.random.normal(0, 1, 384).tolist()
                    mock_embedder.embed_query = mock_embed_query
                    mock_embed_class.return_value = mock_embedder
                    
                    # Mock reranking stage
                    with patch('src.repoindex.pipeline.reranking.CrossEncoderReranker') as mock_reranker_class:
                        mock_reranker = AsyncMock()
                        async def mock_rerank(query, results):
                            stage_execution_order.append("reranking")
                            return results[:5]  # Return top 5
                        mock_reranker.rerank = mock_rerank
                        mock_reranker_class.return_value = mock_reranker
                        
                        # Execute pipeline
                        search_pipeline = EnhancedSearchPipeline(
                            ai_config=comprehensive_config.ai,
                            query_config=comprehensive_config.query,
                            reranker_config=comprehensive_config.reranker,
                            coordinator=mock_coordinator
                        )
                        
                        await search_pipeline.initialize()
                        
                        await search_pipeline.search(
                            query="test query",
                            vector_index=comprehensive_vector_index,
                            top_k=5
                        )
                        
                        # Verify stage execution order
                        expected_order = ["hyde", "embedding", "reranking"]
                        assert stage_execution_order == expected_order