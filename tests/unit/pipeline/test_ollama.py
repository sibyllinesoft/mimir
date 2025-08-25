"""
Unit tests for OllamaAdapter.

Tests the Ollama adapter for connection handling, model management,
text generation, and error scenarios.
"""

import json
from unittest.mock import AsyncMock, Mock, patch
import pytest
import aiohttp
from aiohttp import ClientResponseError, ClientTimeout

from src.repoindex.pipeline.ollama import OllamaAdapter, OllamaModelInfo


class TestOllamaAdapter:
    """Test OllamaAdapter functionality."""

    @pytest.fixture
    def adapter(self):
        """Create an OllamaAdapter instance."""
        return OllamaAdapter(base_url="http://localhost:11434", timeout=30)

    @pytest.fixture
    def mock_session(self):
        """Create a mock aiohttp.ClientSession."""
        session = AsyncMock(spec=aiohttp.ClientSession)
        return session

    @pytest.mark.asyncio
    async def test_init_adapter(self, adapter):
        """Test adapter initialization."""
        assert adapter.base_url == "http://localhost:11434"
        assert adapter.timeout == 30
        assert adapter._session is None

    @pytest.mark.asyncio
    async def test_context_manager(self, adapter):
        """Test async context manager protocol."""
        with patch.object(adapter, '_get_session') as mock_get:
            mock_session = AsyncMock()
            mock_get.return_value = mock_session
            
            async with adapter as ctx:
                assert ctx is adapter
                mock_get.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_session(self, adapter, mock_session):
        """Test session cleanup."""
        adapter._session = mock_session
        await adapter.close()
        
        mock_session.close.assert_called_once()
        assert adapter._session is None

    @pytest.mark.asyncio
    async def test_get_provider_name(self, adapter):
        """Test provider name retrieval."""
        assert adapter.get_provider_name() == "ollama"

    @pytest.mark.asyncio
    async def test_is_available_success(self, adapter, mock_session):
        """Test successful availability check."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"status": "Ollama is running"}
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        result = await adapter.is_available()
        assert result is True
        
        mock_session.get.assert_called_once_with(
            "http://localhost:11434/",
            timeout=aiohttp.ClientTimeout(total=30)
        )

    @pytest.mark.asyncio
    async def test_is_available_connection_error(self, adapter, mock_session):
        """Test availability check with connection error."""
        mock_session.get.side_effect = aiohttp.ClientConnectorError(
            connection_key=Mock(), os_error=None
        )
        adapter._session = mock_session
        
        result = await adapter.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_timeout(self, adapter, mock_session):
        """Test availability check with timeout."""
        mock_session.get.side_effect = aiohttp.ClientTimeoutError()
        adapter._session = mock_session
        
        result = await adapter.is_available()
        assert result is False

    @pytest.mark.asyncio
    async def test_list_models_success(self, adapter, mock_session):
        """Test successful model listing."""
        mock_models_data = {
            "models": [
                {
                    "name": "llama2:7b",
                    "modified_at": "2023-11-01T12:00:00Z",
                    "size": 3800000000,
                    "digest": "abc123",
                    "details": {
                        "format": "ggml",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "7B",
                        "quantization_level": "Q4_0"
                    }
                },
                {
                    "name": "codellama:13b",
                    "modified_at": "2023-11-02T12:00:00Z",
                    "size": 7300000000,
                    "digest": "def456",
                    "details": {
                        "format": "ggml",
                        "family": "llama",
                        "families": ["llama"],
                        "parameter_size": "13B",
                        "quantization_level": "Q4_0"
                    }
                }
            ]
        }
        
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = mock_models_data
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        models = await adapter.list_models()
        
        assert len(models) == 2
        assert models[0].name == "llama2:7b"
        assert models[0].size == 3800000000
        assert models[0].parameter_size == "7B"
        assert models[1].name == "codellama:13b"
        assert models[1].parameter_size == "13B"
        
        mock_session.get.assert_called_once_with(
            "http://localhost:11434/api/tags",
            timeout=aiohttp.ClientTimeout(total=30)
        )

    @pytest.mark.asyncio
    async def test_list_models_empty(self, adapter, mock_session):
        """Test model listing with no models."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"models": []}
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        models = await adapter.list_models()
        assert len(models) == 0

    @pytest.mark.asyncio
    async def test_list_models_error(self, adapter, mock_session):
        """Test model listing with error response."""
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.raise_for_status.side_effect = ClientResponseError(
            request_info=Mock(), history=[], status=500
        )
        
        mock_session.get.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        with pytest.raises(Exception) as exc_info:
            await adapter.list_models()
        
        assert "Failed to list models" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_pull_model_success(self, adapter, mock_session):
        """Test successful model pulling."""
        # Mock streaming response for pull operation
        mock_response = AsyncMock()
        mock_response.status = 200
        
        # Simulate streaming pull responses
        pull_responses = [
            '{"status": "pulling manifest"}\n',
            '{"status": "downloading", "digest": "abc123", "total": 1000, "completed": 500}\n',
            '{"status": "downloading", "digest": "abc123", "total": 1000, "completed": 1000}\n',
            '{"status": "verifying sha256 digest"}\n',
            '{"status": "success"}\n'
        ]
        
        async def mock_iter_chunked(size):
            for response in pull_responses:
                yield response.encode()
        
        mock_response.content.iter_chunked = mock_iter_chunked
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        await adapter.pull_model("llama2:7b")
        
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "api/pull" in call_args[0][0]
        assert json.loads(call_args[1]["data"])["name"] == "llama2:7b"

    @pytest.mark.asyncio
    async def test_pull_model_error(self, adapter, mock_session):
        """Test model pulling with error."""
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_response.raise_for_status.side_effect = ClientResponseError(
            request_info=Mock(), history=[], status=404
        )
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        with pytest.raises(Exception) as exc_info:
            await adapter.pull_model("nonexistent:model")
        
        assert "Failed to pull model" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_generate_text_success(self, adapter, mock_session):
        """Test successful text generation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "model": "llama2:7b",
            "created_at": "2023-11-01T12:00:00Z",
            "response": "This is a generated response about Python programming.",
            "done": True,
            "context": [1, 2, 3, 4, 5],
            "total_duration": 5000000000,
            "load_duration": 1000000000,
            "prompt_eval_count": 20,
            "prompt_eval_duration": 1000000000,
            "eval_count": 50,
            "eval_duration": 3000000000
        }
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        result = await adapter.generate_text("Explain Python programming", "llama2:7b")
        
        assert result == "This is a generated response about Python programming."
        
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert "api/generate" in call_args[0][0]
        
        request_data = json.loads(call_args[1]["data"])
        assert request_data["model"] == "llama2:7b"
        assert request_data["prompt"] == "Explain Python programming"
        assert request_data["stream"] is False

    @pytest.mark.asyncio
    async def test_generate_text_with_options(self, adapter, mock_session):
        """Test text generation with custom options."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": "Generated with custom options",
            "done": True
        }
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        options = {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "num_predict": 100
        }
        
        result = await adapter.generate_text(
            "Test prompt", 
            "llama2:7b", 
            temperature=0.8,
            max_tokens=100,
            **options
        )
        
        assert result == "Generated with custom options"
        
        call_args = mock_session.post.call_args
        request_data = json.loads(call_args[1]["data"])
        assert request_data["options"]["temperature"] == 0.8
        assert request_data["options"]["num_predict"] == 100

    @pytest.mark.asyncio
    async def test_generate_text_timeout(self, adapter, mock_session):
        """Test text generation with timeout."""
        mock_session.post.side_effect = aiohttp.ClientTimeoutError()
        adapter._session = mock_session
        
        with pytest.raises(Exception) as exc_info:
            await adapter.generate_text("Test prompt", "llama2:7b")
        
        assert "timeout" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_generate_text_retries(self, adapter, mock_session):
        """Test text generation with retries on retryable errors."""
        # First call fails with 503, second succeeds
        error_response = AsyncMock()
        error_response.status = 503
        error_response.raise_for_status.side_effect = ClientResponseError(
            request_info=Mock(), history=[], status=503
        )
        
        success_response = AsyncMock()
        success_response.status = 200
        success_response.json.return_value = {
            "response": "Success after retry",
            "done": True
        }
        
        mock_session.post.return_value.__aenter__.side_effect = [
            error_response, success_response
        ]
        adapter._session = mock_session
        
        with patch('asyncio.sleep', new_callable=AsyncMock):  # Mock sleep for faster tests
            result = await adapter.generate_text("Test prompt", "llama2:7b")
        
        assert result == "Success after retry"
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_synthesize_answer(self, adapter, mock_session):
        """Test answer synthesis from search results."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": "Based on the search results, Python is a programming language that emphasizes readability and simplicity.",
            "done": True
        }
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        search_results = [
            "Python is a high-level programming language",
            "Python emphasizes code readability",
            "Python has a simple and easy-to-learn syntax"
        ]
        
        result = await adapter.synthesize_answer(
            "What is Python?", 
            search_results, 
            "llama2:7b"
        )
        
        assert "Python is a programming language" in result
        
        call_args = mock_session.post.call_args
        request_data = json.loads(call_args[1]["data"])
        assert "synthesize" in request_data["prompt"].lower()
        assert all(snippet in request_data["prompt"] for snippet in search_results)

    @pytest.mark.asyncio
    async def test_explain_code_snippet(self, adapter, mock_session):
        """Test code snippet explanation."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": "This function adds two numbers together and returns the result.",
            "done": True
        }
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        code_snippet = """
        def add(a, b):
            return a + b
        """
        
        result = await adapter.explain_code_snippet(code_snippet, "llama2:7b")
        
        assert "adds two numbers" in result
        
        call_args = mock_session.post.call_args
        request_data = json.loads(call_args[1]["data"])
        assert code_snippet.strip() in request_data["prompt"]

    @pytest.mark.asyncio
    async def test_suggest_improvements(self, adapter, mock_session):
        """Test code improvement suggestions."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "response": "Consider adding type hints and error handling to improve the function.",
            "done": True
        }
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        code_snippet = """
        def divide(a, b):
            return a / b
        """
        
        result = await adapter.suggest_improvements(code_snippet, "llama2:7b")
        
        assert "type hints" in result or "error handling" in result
        
        call_args = mock_session.post.call_args
        request_data = json.loads(call_args[1]["data"])
        assert "improve" in request_data["prompt"].lower()

    @pytest.mark.asyncio
    async def test_get_model_info(self, adapter, mock_session):
        """Test model information retrieval."""
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "modelfile": "# Modelfile for llama2:7b",
            "parameters": "# Parameters",
            "template": "{{ .Prompt }}",
            "details": {
                "format": "ggml",
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "7B",
                "quantization_level": "Q4_0"
            }
        }
        
        mock_session.post.return_value.__aenter__.return_value = mock_response
        adapter._session = mock_session
        
        info = await adapter.get_model_info("llama2:7b")
        
        assert isinstance(info, OllamaModelInfo)
        assert info.name == "llama2:7b"
        assert info.parameter_size == "7B"
        assert info.format == "ggml"
        assert info.family == "llama"
        
        call_args = mock_session.post.call_args
        request_data = json.loads(call_args[1]["data"])
        assert request_data["name"] == "llama2:7b"

    def test_is_retryable_error(self, adapter):
        """Test retryable error detection."""
        # Server errors (5xx) should be retryable
        server_error = ClientResponseError(
            request_info=Mock(), history=[], status=503
        )
        assert adapter._is_retryable_error(server_error) is True
        
        # Too many requests should be retryable
        rate_limit_error = ClientResponseError(
            request_info=Mock(), history=[], status=429
        )
        assert adapter._is_retryable_error(rate_limit_error) is True
        
        # Timeout should be retryable
        timeout_error = aiohttp.ClientTimeoutError()
        assert adapter._is_retryable_error(timeout_error) is True
        
        # Client errors (4xx except 429) should not be retryable
        client_error = ClientResponseError(
            request_info=Mock(), history=[], status=404
        )
        assert adapter._is_retryable_error(client_error) is False
        
        # Connection errors should be retryable
        connection_error = aiohttp.ClientConnectorError(
            connection_key=Mock(), os_error=None
        )
        assert adapter._is_retryable_error(connection_error) is True


class TestOllamaModelInfo:
    """Test OllamaModelInfo data class."""

    def test_model_info_creation(self):
        """Test model info creation and properties."""
        info = OllamaModelInfo(
            name="llama2:7b",
            size=3800000000,
            parameter_size="7B",
            format="ggml",
            family="llama",
            quantization_level="Q4_0"
        )
        
        assert info.name == "llama2:7b"
        assert info.size == 3800000000
        assert info.parameter_size == "7B"
        assert info.format == "ggml"
        assert info.family == "llama"
        assert info.quantization_level == "Q4_0"

    def test_model_info_str_representation(self):
        """Test string representation of model info."""
        info = OllamaModelInfo(
            name="llama2:7b",
            size=3800000000,
            parameter_size="7B",
            format="ggml",
            family="llama",
            quantization_level="Q4_0"
        )
        
        str_repr = str(info)
        assert "llama2:7b" in str_repr
        assert "7B" in str_repr