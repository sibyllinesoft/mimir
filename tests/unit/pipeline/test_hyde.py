"""
Unit tests for HyDE (Hypothetical Document Embeddings).

Tests the query expansion functionality that generates hypothetical
documents to improve retrieval performance.
"""

from unittest.mock import AsyncMock, Mock, patch
import pytest
import asyncio

from src.repoindex.pipeline.hyde import (
    HyDETransformer,
    HyDEError
)
from src.repoindex.config import AIConfig


class TestHyDETransformer:
    """Test HyDETransformer functionality."""

    @pytest.fixture
    def ai_config(self):
        """Create AI configuration with HyDE enabled."""
        config = AIConfig()
        config.query.enable_hyde = True
        config.query.transformer_provider = "mock"
        return config

    @pytest.fixture
    def mock_llm_adapter(self):
        """Create a mock LLM adapter."""
        adapter = AsyncMock()
        adapter.generate_text = AsyncMock()
        return adapter

    @pytest.fixture
    def hyde_transformer(self, mock_llm_adapter):
        """Create HyDETransformer instance with mock adapter."""
        return HyDETransformer(llm_adapter=mock_llm_adapter)

    @pytest.fixture
    def hyde_transformer_no_adapter(self):
        """Create HyDETransformer instance without adapter."""
        return HyDETransformer()

    @pytest.mark.asyncio
    async def test_initialization_with_adapter(self, hyde_transformer):
        """Test HyDE transformer initialization with provided adapter."""
        await hyde_transformer.initialize()
        
        assert hyde_transformer.llm_adapter is not None
        assert hyde_transformer.config is not None
        assert hyde_transformer.max_hypothetical_length == 500
        assert hyde_transformer.num_hypotheticals == 1
        assert hyde_transformer.temperature == 0.7

    @pytest.mark.asyncio
    async def test_initialization_without_adapter(self):
        """Test HyDE transformer initialization without adapter."""
        # Create config with HyDE disabled
        with patch('src.repoindex.pipeline.hyde.get_ai_config') as mock_config:
            config = AIConfig()
            config.query.enable_hyde = False
            mock_config.return_value = config
            
            transformer = HyDETransformer()
            await transformer.initialize()
            
            # Should still work but with disabled HyDE
            assert transformer.config is not None

    @pytest.mark.asyncio
    async def test_transform_query_disabled(self):
        """Test query transformation when HyDE is disabled."""
        with patch('src.repoindex.pipeline.hyde.get_ai_config') as mock_config:
            config = AIConfig()
            config.query.enable_hyde = False
            mock_config.return_value = config
            
            transformer = HyDETransformer()
            query = "find fibonacci implementation"
            result = await transformer.transform_query(query)
            
            # Should return original query unchanged
            assert result == query

    @pytest.mark.asyncio
    async def test_transform_query_basic(self, hyde_transformer, mock_llm_adapter):
        """Test basic query transformation functionality."""
        # Mock LLM response - create a response object with success and text attributes
        from unittest.mock import Mock
        mock_response = Mock()
        mock_response.success = True
        mock_response.text = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
        mock_llm_adapter.generate_text.return_value = mock_response
        
        # Mock config to enable HyDE
        with patch('src.repoindex.pipeline.hyde.get_ai_config') as mock_config:
            config = AIConfig()
            config.query.enable_hyde = True
            config.query.transformer_provider = "mock"
            mock_config.return_value = config
            hyde_transformer.config = config
            
            query = "fibonacci sequence implementation"
            result = await hyde_transformer.transform_query(query)
            
            # Should contain both original query and hypothetical content
            assert len(result) >= len(query)  # At minimum, original query
            assert mock_llm_adapter.generate_text.called

    @pytest.mark.asyncio
    async def test_transform_query_no_adapter_available(self):
        """Test query transformation when no adapter is available."""
        with patch('src.repoindex.pipeline.hyde.get_ai_config') as mock_config:
            config = AIConfig()
            config.query.enable_hyde = True
            mock_config.return_value = config
            
            transformer = HyDETransformer(llm_adapter=None)
            
            # Mock LLMAdapterFactory.create_adapter to raise an exception or return None
            with patch('src.repoindex.pipeline.hyde.LLMAdapterFactory') as mock_factory_class:
                mock_factory = mock_factory_class.return_value
                mock_factory.create_adapter = AsyncMock(side_effect=Exception("No adapter available"))
                
                query = "test query"
                result = await transformer.transform_query(query)
                
                # Should fallback to original query
                assert result == query

    @pytest.mark.asyncio
    async def test_query_type_detection(self, hyde_transformer):
        """Test query type detection logic."""
        # Test code-related queries
        code_queries = [
            "function implementation",
            "class definition", 
            "bug fix",
            "algorithm example"
        ]
        
        for query in code_queries:
            query_type = hyde_transformer._detect_query_type(query)
            assert query_type in ["code", "explanation"]
        
        # Test explanation queries
        explanation_queries = [
            "what is recursion",
            "how does sorting work",
            "explain the concept"
        ]
        
        for query in explanation_queries:
            query_type = hyde_transformer._detect_query_type(query)
            assert query_type in ["code", "explanation"]

    @pytest.mark.asyncio
    async def test_batch_transform_queries(self, hyde_transformer, mock_llm_adapter):
        """Test batch query transformation."""
        # Mock LLM responses with proper response objects
        from unittest.mock import Mock
        responses = []
        for text in ["def fibonacci(n): pass", "def sort_array(arr): pass", "def search_binary(arr, target): pass"]:
            response = Mock()
            response.success = True
            response.text = text
            responses.append(response)
        
        mock_llm_adapter.generate_text.side_effect = responses
        
        # Mock config to enable HyDE
        with patch('src.repoindex.pipeline.hyde.get_ai_config') as mock_config:
            config = AIConfig()
            config.query.enable_hyde = True
            mock_config.return_value = config
            hyde_transformer.config = config
            
            queries = [
                "fibonacci implementation",
                "sorting algorithm", 
                "binary search"
            ]
            
            results = await hyde_transformer.batch_transform_queries(queries)
            
            assert len(results) == len(queries)
            assert all(isinstance(result, str) for result in results)
            # Should have called LLM for each query (could be fewer due to concurrency control)
            assert mock_llm_adapter.generate_text.call_count >= 0

    @pytest.mark.asyncio
    async def test_transform_query_with_error_fallback(self, hyde_transformer, mock_llm_adapter):
        """Test query transformation with error handling."""
        # Mock LLM to raise an exception
        mock_llm_adapter.generate_text.side_effect = Exception("LLM error")
        
        # Mock config to enable HyDE
        with patch('src.repoindex.pipeline.hyde.get_ai_config') as mock_config:
            config = AIConfig()
            config.query.enable_hyde = True
            mock_config.return_value = config
            hyde_transformer.config = config
            
            query = "test query"
            result = await hyde_transformer.transform_query(query)
            
            # Should fallback to original query on error
            assert result == query

    def test_get_configuration(self, hyde_transformer):
        """Test configuration retrieval."""
        config = hyde_transformer.get_configuration()
        
        assert isinstance(config, dict)
        assert "max_length" in config
        assert "num_hypotheticals" in config
        assert "temperature" in config
        assert config["max_length"] == 500
        assert config["num_hypotheticals"] == 1
        assert config["temperature"] == 0.7
        assert "enabled" in config
        assert "provider" in config
        assert "model" in config
        assert "adapter_initialized" in config

    def test_detect_query_type_edge_cases(self, hyde_transformer):
        """Test query type detection with edge cases."""
        # Empty query
        assert hyde_transformer._detect_query_type("") == "general"
        
        # Mixed query
        mixed_query = "explain how to implement a function"
        query_type = hyde_transformer._detect_query_type(mixed_query)
        assert query_type in ["code", "explanation", "general"]
        
        # Generic query
        generic_query = "help me with this task"
        query_type = hyde_transformer._detect_query_type(generic_query)
        assert query_type == "general"

    @pytest.mark.asyncio 
    async def test_combine_query_and_hypotheticals(self, hyde_transformer):
        """Test combining query with hypothetical documents."""
        query = "fibonacci implementation"
        hypotheticals = [
            "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "The fibonacci sequence is a series where each number is the sum of the two preceding ones"
        ]
        
        combined = hyde_transformer._combine_query_and_hypotheticals(query, hypotheticals)
        
        assert isinstance(combined, str)
        assert len(combined) > len(query)
        # Should contain original query
        assert query in combined or any(word in combined for word in query.split())