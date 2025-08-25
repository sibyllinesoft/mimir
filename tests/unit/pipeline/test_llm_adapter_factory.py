"""
Unit tests for LLMAdapterFactory.

Tests the factory pattern for creating LLM adapters based on provider
configuration and availability.
"""

from unittest.mock import Mock, patch, MagicMock
import pytest

from src.repoindex.pipeline.llm_adapter_factory import (
    LLMAdapterFactory,
    LLMProvider,
    create_default_llm_adapter,
    get_available_providers
)
from src.repoindex.pipeline.llm_adapter import LLMError, MockLLMAdapter
from src.repoindex.config import AIConfig


class TestLLMAdapterFactory:
    """Test LLMAdapterFactory functionality."""

    @pytest.fixture
    def factory(self):
        """Create an LLMAdapterFactory instance."""
        return LLMAdapterFactory()

    @pytest.fixture
    def mock_config(self):
        """Create mock AI configuration."""
        config = AIConfig()
        config.default_llm_provider = "mock"
        return config

    @pytest.fixture  
    def ollama_config(self):
        """Create configuration for Ollama."""
        config = AIConfig()
        config.default_llm_provider = "ollama"
        config.ollama_host = "localhost"
        config.ollama_port = 11434
        config.ollama_model = "llama3.2:3b"
        return config

    def test_create_mock_adapter(self, factory, mock_config):
        """Test creating mock adapter."""
        adapter = factory.create_adapter(provider="mock", config=mock_config)
        
        assert isinstance(adapter, MockLLMAdapter)
        assert adapter.provider == "mock"

    def test_create_adapter_with_default_provider(self, factory, mock_config):
        """Test creating adapter using default provider from config."""
        adapter = factory.create_adapter(config=mock_config)
        
        assert isinstance(adapter, MockLLMAdapter)
        assert adapter.provider == "mock"

    def test_create_adapter_unsupported_provider(self, factory, mock_config):
        """Test creating adapter with unsupported provider raises error."""
        with pytest.raises(LLMError) as excinfo:
            factory.create_adapter(provider="unsupported", config=mock_config)
        
        assert "Unsupported LLM provider" in str(excinfo.value)
        assert "unsupported" in str(excinfo.value)

    def test_create_default_adapter(self, factory):
        """Test creating default adapter."""
        with patch('src.repoindex.pipeline.llm_adapter_factory.get_ai_config') as mock_get_config:
            mock_config = AIConfig()
            mock_config.default_llm_provider = "mock"
            mock_get_config.return_value = mock_config
            
            adapter = factory.create_default_adapter()
            
            assert isinstance(adapter, MockLLMAdapter)
            assert adapter.provider == "mock"

    @pytest.mark.asyncio
    async def test_create_ollama_adapter_specific(self, factory, ollama_config):
        """Test creating Ollama adapter specifically."""
        # Mock the OllamaAdapter since it may have external dependencies
        with patch('src.repoindex.pipeline.llm_adapter_factory.OllamaAdapter') as mock_ollama:
            mock_instance = Mock()
            mock_instance.provider = "ollama"
            mock_ollama.return_value = mock_instance
            
            adapter = factory.create_ollama_adapter(config=ollama_config)
            
            # Verify it was called with correct config
            mock_ollama.assert_called_once()
            assert adapter.provider == "ollama"

    def test_get_available_providers(self):
        """Test getting list of available providers."""
        providers = get_available_providers()
        
        assert isinstance(providers, list)
        assert len(providers) > 0
        assert "mock" in providers  # Mock should always be available

    def test_provider_enum_values(self):
        """Test LLMProvider enum values."""
        assert LLMProvider.MOCK == "mock"
        assert LLMProvider.OLLAMA == "ollama"
        assert LLMProvider.GEMINI == "gemini"

    def test_register_custom_adapter(self, factory):
        """Test registering custom adapter."""
        # Create a mock custom adapter
        class CustomAdapter(MockLLMAdapter):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.provider = "custom"
        
        # Register it (note: this would need to be extended in actual implementation)
        # For now, just test the concept
        assert hasattr(LLMAdapterFactory, 'register_adapter')

    @pytest.mark.asyncio
    async def test_create_default_llm_adapter_function(self):
        """Test standalone create_default_llm_adapter function."""
        with patch('src.repoindex.pipeline.llm_adapter_factory.get_ai_config') as mock_get_config:
            mock_config = AIConfig()
            mock_config.default_llm_provider = "mock"
            mock_get_config.return_value = mock_config
            
            adapter = await create_default_llm_adapter()
            
            assert isinstance(adapter, MockLLMAdapter)
            assert adapter.provider == "mock"

    def test_create_adapter_with_kwargs(self, factory, mock_config):
        """Test creating adapter with additional kwargs."""
        adapter = factory.create_adapter(
            provider="mock", 
            config=mock_config,
            custom_param="test_value"
        )
        
        assert isinstance(adapter, MockLLMAdapter)
        # Custom params would be passed to the adapter constructor

    def test_create_adapter_provider_case_insensitive(self, factory, mock_config):
        """Test that provider names are case insensitive."""
        adapter1 = factory.create_adapter(provider="MOCK", config=mock_config)
        adapter2 = factory.create_adapter(provider="mock", config=mock_config)
        adapter3 = factory.create_adapter(provider="Mock", config=mock_config)
        
        assert all(isinstance(a, MockLLMAdapter) for a in [adapter1, adapter2, adapter3])

    def test_error_handling_in_adapter_creation(self, factory):
        """Test error handling when adapter creation fails."""
        # Mock config that would cause creation to fail
        with patch('src.repoindex.pipeline.llm_adapter_factory.get_ai_config') as mock_get_config:
            mock_config = AIConfig()
            mock_config.default_llm_provider = "mock"
            mock_get_config.return_value = mock_config
            
            # Mock the adapter creation to raise an exception
            with patch.object(factory, '_create_mock_adapter', side_effect=Exception("Creation failed")):
                with pytest.raises(LLMError) as excinfo:
                    factory.create_adapter(provider="mock", config=mock_config)
                
                assert "Failed to create mock adapter" in str(excinfo.value)
                assert "Creation failed" in str(excinfo.value)

    def test_factory_adapter_registry(self):
        """Test that factory has proper adapter registry."""
        factory = LLMAdapterFactory()
        
        # Check that registry exists and has expected adapters
        assert hasattr(factory, '_adapters')
        assert isinstance(factory._adapters, dict)
        assert LLMProvider.MOCK in factory._adapters
        assert LLMProvider.OLLAMA in factory._adapters

    @pytest.mark.asyncio
    async def test_async_adapter_creation_methods(self, factory):
        """Test async methods in factory work correctly."""
        with patch('src.repoindex.pipeline.llm_adapter_factory.get_ai_config') as mock_get_config:
            mock_config = AIConfig()
            mock_config.default_llm_provider = "mock"
            mock_get_config.return_value = mock_config
            
            # Test async create_default_llm_adapter function
            adapter = await create_default_llm_adapter(config=mock_config)
            assert isinstance(adapter, MockLLMAdapter)

    def test_create_adapter_without_config(self, factory):
        """Test creating adapter without providing config uses default."""
        with patch('src.repoindex.pipeline.llm_adapter_factory.get_ai_config') as mock_get_config:
            mock_config = AIConfig()
            mock_config.default_llm_provider = "mock"
            mock_get_config.return_value = mock_config
            
            adapter = factory.create_adapter(provider="mock")
            
            assert isinstance(adapter, MockLLMAdapter)
            mock_get_config.assert_called_once()

    def test_provider_enum_iteration(self):
        """Test iterating over provider enum."""
        providers = list(LLMProvider)
        
        assert len(providers) >= 3
        assert LLMProvider.MOCK in providers
        assert LLMProvider.OLLAMA in providers  
        assert LLMProvider.GEMINI in providers