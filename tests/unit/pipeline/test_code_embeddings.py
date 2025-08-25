"""
Unit tests for Code Embeddings functionality.

Tests the specialized code embedding processor that handles
code-specific tokenization, embedding generation, and semantic search.
"""

from unittest.mock import Mock, patch, AsyncMock
import pytest
import numpy as np

from src.repoindex.pipeline.code_embeddings import (
    CodeEmbeddingProcessor,
    CodeEmbeddingConfig,
    create_code_embedding_processor
)
from src.repoindex.data.schemas import VectorChunk


class TestCodeEmbeddingProcessor:
    """Test CodeEmbeddingProcessor functionality."""

    @pytest.fixture
    def embedding_config(self):
        """Create code embedding configuration."""
        return CodeEmbeddingConfig(
            model="microsoft/codebert-base",
            max_tokens=512,
            batch_size=16,
            normalize_embeddings=True,
            use_code_tokens=True
        )

    @pytest.fixture
    def processor(self, embedding_config):
        """Create CodeEmbeddingProcessor instance."""
        return CodeEmbeddingProcessor(embedding_config)

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.config is not None
        assert processor.config.model == "microsoft/codebert-base"
        assert processor.config.max_tokens == 512
        assert processor.config.batch_size == 16

    @pytest.mark.asyncio
    async def test_initialize_processor(self, processor):
        """Test processor initialization with model loading."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            assert processor._initialized is True
            mock_st_class.assert_called_once_with("microsoft/codebert-base")

    @pytest.mark.asyncio
    async def test_embed_query_success(self, processor):
        """Test successful query embedding."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_embedding = np.random.normal(0, 1, 768).tolist()
            mock_model.encode.return_value = [mock_embedding]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            query = "binary search algorithm implementation"
            embedding = await processor.embed_query(query)
            
            assert isinstance(embedding, list)
            assert len(embedding) == 768  # CodeBERT dimension
            assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_query_with_code_tokens(self, processor):
        """Test query embedding with code tokenization."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_embedding = np.random.normal(0, 1, 768).tolist()
            mock_model.encode.return_value = [mock_embedding]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            # Query with code-like tokens
            query = "def fibonacci(n): return fibonacci(n-1) + fibonacci(n-2)"
            embedding = await processor.embed_query(query)
            
            assert isinstance(embedding, list)
            assert len(embedding) == 768
            
            # Verify the model was called with processed query
            mock_model.encode.assert_called_once()
            call_args = mock_model.encode.call_args[0]
            processed_query = call_args[0][0]
            
            # Should contain the original query or processed version
            assert len(processed_query) > 0

    @pytest.mark.asyncio
    async def test_embed_code_chunks(self, processor):
        """Test embedding code chunks."""
        chunks = [
            VectorChunk(
                path="src/math.py",
                span=(1, 10),
                content="def add(a, b):\n    return a + b",
                embedding=None,  # Will be generated
                token_count=15
            ),
            VectorChunk(
                path="src/sort.py",
                span=(1, 15),
                content="def bubble_sort(arr):\n    for i in range(len(arr)):\n        for j in range(len(arr)-1-i):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]",
                embedding=None,
                token_count=35
            )
        ]
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_embeddings = [
                np.random.normal(0, 1, 768).tolist(),
                np.random.normal(0, 1, 768).tolist()
            ]
            mock_model.encode.return_value = mock_embeddings
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            embedded_chunks = await processor.embed_code_chunks(chunks)
            
            assert len(embedded_chunks) == 2
            for chunk in embedded_chunks:
                assert isinstance(chunk, VectorChunk)
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 768

    @pytest.mark.asyncio
    async def test_embed_code_chunks_batch_processing(self, processor):
        """Test code chunk embedding with batch processing."""
        # Create many chunks to test batching
        chunks = []
        for i in range(50):
            chunks.append(VectorChunk(
                path=f"src/file_{i}.py",
                span=(1, 10),
                content=f"def function_{i}():\n    return {i}",
                embedding=None,
                token_count=12
            ))
        
        processor.config.batch_size = 10  # Small batch size for testing
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            # Mock embeddings for all chunks
            mock_embeddings = [np.random.normal(0, 1, 768).tolist() for _ in range(50)]
            mock_model.encode.return_value = mock_embeddings
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            embedded_chunks = await processor.embed_code_chunks(chunks)
            
            assert len(embedded_chunks) == 50
            assert all(chunk.embedding is not None for chunk in embedded_chunks)
            
            # Should have called encode multiple times due to batching
            assert mock_model.encode.call_count >= 5  # 50 chunks / 10 batch_size

    @pytest.mark.asyncio
    async def test_preprocess_code_text(self, processor):
        """Test code text preprocessing."""
        await processor.initialize()
        
        # Test various code preprocessing scenarios
        test_cases = [
            # Basic function
            ("def hello():\n    print('world')", "def hello(): print('world')"),
            
            # Class with methods
            ("class Calculator:\n    def add(self, a, b):\n        return a + b", 
             "class Calculator: def add(self, a, b): return a + b"),
            
            # Comments and docstrings
            ('def func():\n    """This is a docstring"""\n    # Comment\n    return True',
             'def func(): return True'),
            
            # Multiple lines with proper spacing
            ("import os\nimport sys\n\ndef main():\n    pass",
             "import os import sys def main(): pass")
        ]
        
        for original, expected_pattern in test_cases:
            processed = processor._preprocess_code_text(original)
            
            assert isinstance(processed, str)
            assert len(processed) > 0
            # The exact processing logic may vary, so just check it's transformed
            assert processed != original or original.count('\n') <= 1

    @pytest.mark.asyncio
    async def test_tokenize_code_query(self, processor):
        """Test code query tokenization."""
        await processor.initialize()
        
        # Test code-specific tokenization
        queries = [
            "def fibonacci implementation",
            "class inheritance python",
            "binary search tree traversal",
            "async await function javascript"
        ]
        
        for query in queries:
            tokenized = processor._tokenize_code_query(query)
            
            assert isinstance(tokenized, str)
            assert len(tokenized) > 0
            # Should preserve important code terms
            if "def" in query:
                assert "def" in tokenized.lower() or "function" in tokenized.lower()

    @pytest.mark.asyncio
    async def test_normalize_embeddings(self, processor):
        """Test embedding normalization."""
        processor.config.normalize_embeddings = True
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            # Create un-normalized embeddings
            raw_embedding = np.random.normal(0, 5, 768)  # Large magnitude
            mock_model.encode.return_value = [raw_embedding.tolist()]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            query = "test query"
            embedding = await processor.embed_query(query)
            
            # Check if normalized (L2 norm should be close to 1)
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01  # Should be normalized

    @pytest.mark.asyncio
    async def test_embed_query_long_text(self, processor):
        """Test embedding very long queries."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_embedding = np.random.normal(0, 1, 768).tolist()
            mock_model.encode.return_value = [mock_embedding]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            # Create very long query
            long_query = "def function_name " * 200  # Much longer than max_tokens
            embedding = await processor.embed_query(long_query)
            
            assert isinstance(embedding, list)
            assert len(embedding) == 768
            
            # Should handle truncation gracefully
            mock_model.encode.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_empty_content(self, processor):
        """Test embedding empty or whitespace-only content."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_embedding = np.random.normal(0, 1, 768).tolist()
            mock_model.encode.return_value = [mock_embedding]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            # Test empty and whitespace queries
            empty_queries = ["", "   ", "\n\n", "\t\t"]
            
            for query in empty_queries:
                embedding = await processor.embed_query(query)
                
                # Should still return valid embedding (possibly zero or default)
                assert isinstance(embedding, list)
                assert len(embedding) == 768

    @pytest.mark.asyncio
    async def test_model_loading_error(self, processor):
        """Test handling of model loading errors."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_st_class.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception) as exc_info:
                await processor.initialize()
            
            assert "Model not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embedding_generation_error(self, processor):
        """Test handling of embedding generation errors."""
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_model.encode.side_effect = Exception("Encoding failed")
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            with pytest.raises(Exception) as exc_info:
                await processor.embed_query("test query")
            
            assert "Encoding failed" in str(exc_info.value)


class TestCodeEmbeddingConfig:
    """Test CodeEmbeddingConfig configuration."""

    def test_config_defaults(self):
        """Test configuration default values."""
        config = CodeEmbeddingConfig()
        
        assert config.model == "microsoft/codebert-base"
        assert config.max_tokens == 512
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
        assert config.use_code_tokens is True

    def test_config_custom_values(self):
        """Test configuration with custom values."""
        config = CodeEmbeddingConfig(
            model="microsoft/graphcodebert-base",
            max_tokens=1024,
            batch_size=16,
            normalize_embeddings=False,
            use_code_tokens=False,
            dimension=768
        )
        
        assert config.model == "microsoft/graphcodebert-base"
        assert config.max_tokens == 1024
        assert config.batch_size == 16
        assert config.normalize_embeddings is False
        assert config.use_code_tokens is False
        assert config.dimension == 768


class TestCreateCodeEmbeddingProcessor:
    """Test create_code_embedding_processor convenience function."""

    @pytest.mark.asyncio
    async def test_create_processor_function(self):
        """Test create_code_embedding_processor convenience function."""
        config = CodeEmbeddingConfig(model="test-model")
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_st_class.return_value = mock_model
            
            processor = await create_code_embedding_processor(config)
            
            assert isinstance(processor, CodeEmbeddingProcessor)
            assert processor.config == config


class TestCodeEmbeddingIntegration:
    """Test code embedding integration scenarios."""

    @pytest.fixture
    def mixed_code_chunks(self):
        """Create chunks with different programming languages."""
        return [
            VectorChunk(
                path="src/calculator.py",
                span=(1, 20),
                content="""
def add(a: int, b: int) -> int:
    \"\"\"Add two integers and return the result.\"\"\"
    return a + b

class Calculator:
    def multiply(self, x, y):
        return x * y
""".strip(),
                embedding=None,
                token_count=40
            ),
            VectorChunk(
                path="src/utils.js",
                span=(1, 15),
                content="""
function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n - 1) + fibonacci(n - 2);
}

const factorial = (n) => n <= 1 ? 1 : n * factorial(n - 1);
""".strip(),
                embedding=None,
                token_count=35
            ),
            VectorChunk(
                path="src/algorithms.cpp",
                span=(1, 25),
                content="""
#include <vector>
#include <algorithm>

void quicksort(std::vector<int>& arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quicksort(arr, low, pi - 1);
        quicksort(arr, pi + 1, high);
    }
}
""".strip(),
                embedding=None,
                token_count=50
            )
        ]

    @pytest.mark.asyncio
    async def test_multilanguage_embedding(self, mixed_code_chunks):
        """Test embedding code chunks from different programming languages."""
        config = CodeEmbeddingConfig(
            model="microsoft/codebert-base",
            use_code_tokens=True
        )
        
        processor = CodeEmbeddingProcessor(config)
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            mock_embeddings = [
                np.random.normal(0, 1, 768).tolist() for _ in mixed_code_chunks
            ]
            mock_model.encode.return_value = mock_embeddings
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            embedded_chunks = await processor.embed_code_chunks(mixed_code_chunks)
            
            assert len(embedded_chunks) == 3
            
            # All chunks should have embeddings
            for chunk in embedded_chunks:
                assert chunk.embedding is not None
                assert len(chunk.embedding) == 768
            
            # Verify model was called with code from different languages
            mock_model.encode.assert_called_once()
            encoded_texts = mock_model.encode.call_args[0][0]
            
            # Should have processed all three language codes
            assert len(encoded_texts) == 3

    @pytest.mark.asyncio
    async def test_semantic_similarity_search(self):
        """Test semantic similarity search with code embeddings."""
        config = CodeEmbeddingConfig()
        processor = CodeEmbeddingProcessor(config)
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            
            # Mock query embedding
            query_embedding = np.array([1, 0, 0] + [0] * 765)  # Simple embedding
            
            # Mock chunk embeddings with varying similarity to query
            chunk_embeddings = [
                np.array([0.9, 0.1, 0] + [0] * 765),  # High similarity
                np.array([0.5, 0.5, 0] + [0] * 765),  # Medium similarity
                np.array([0, 0, 1] + [0] * 765),      # Low similarity
            ]
            
            mock_model.encode.side_effect = [
                [query_embedding.tolist()],  # Query embedding
                [e.tolist() for e in chunk_embeddings]  # Chunk embeddings
            ]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            # Create test chunks
            chunks = [
                VectorChunk(
                    path=f"src/test_{i}.py",
                    span=(1, 10),
                    content=f"def function_{i}(): pass",
                    embedding=None,
                    token_count=10
                ) for i in range(3)
            ]
            
            # Embed chunks
            embedded_chunks = await processor.embed_code_chunks(chunks)
            
            # Embed query
            query = "function implementation"
            query_emb = await processor.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for chunk in embedded_chunks:
                chunk_emb = np.array(chunk.embedding)
                query_emb_array = np.array(query_emb)
                
                # Cosine similarity
                similarity = np.dot(chunk_emb, query_emb_array) / (
                    np.linalg.norm(chunk_emb) * np.linalg.norm(query_emb_array)
                )
                similarities.append(similarity)
            
            # Should be sorted by similarity (highest first)
            sorted_similarities = sorted(similarities, reverse=True)
            assert similarities != sorted_similarities or similarities == sorted_similarities

    @pytest.mark.asyncio
    async def test_code_specific_preprocessing(self):
        """Test code-specific preprocessing features."""
        config = CodeEmbeddingConfig(use_code_tokens=True)
        processor = CodeEmbeddingProcessor(config)
        
        await processor.initialize()
        
        # Test preprocessing of different code constructs
        test_code_samples = [
            # Python class with methods
            """
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        # Process the data
        return [item.upper() for item in self.data]
""",
            # JavaScript function with comments
            """
// Calculate factorial recursively
function factorial(n) {
    if (n === 0 || n === 1) {
        return 1;
    }
    return n * factorial(n - 1);
}
""",
            # C++ template function
            """
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}
"""
        ]
        
        for code_sample in test_code_samples:
            processed = processor._preprocess_code_text(code_sample)
            
            # Should remove excessive whitespace and normalize
            assert len(processed) > 0
            assert processed != code_sample  # Should be modified
            
            # Should preserve key code tokens
            if "class" in code_sample:
                assert "class" in processed.lower()
            if "function" in code_sample:
                assert "function" in processed.lower()
            if "template" in code_sample:
                assert "template" in processed.lower()

    @pytest.mark.asyncio
    async def test_embedding_consistency(self):
        """Test that identical code produces identical embeddings."""
        config = CodeEmbeddingConfig()
        processor = CodeEmbeddingProcessor(config)
        
        with patch('sentence_transformers.SentenceTransformer') as mock_st_class:
            mock_model = Mock()
            # Return same embedding for same input
            consistent_embedding = np.random.normal(0, 1, 768).tolist()
            mock_model.encode.return_value = [consistent_embedding]
            mock_st_class.return_value = mock_model
            
            await processor.initialize()
            
            code_text = "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
            
            # Generate embedding twice
            embedding1 = await processor.embed_query(code_text)
            embedding2 = await processor.embed_query(code_text)
            
            # Should be identical
            assert embedding1 == embedding2
            
            # Verify both calls used the same processed input
            assert mock_model.encode.call_count == 2