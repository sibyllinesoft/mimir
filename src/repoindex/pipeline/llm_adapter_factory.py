"""
LLM Adapter Factory for Mimir.

Provides a factory pattern for creating LLM adapters based on configuration,
enabling seamless switching between different LLM providers (Ollama, Gemini, etc.)
while maintaining a consistent interface.
"""

import logging
from typing import Dict, Type, Optional, Any
from enum import Enum

from .llm_adapter import LLMAdapter, MockLLMAdapter, LLMError
from .ollama import OllamaAdapter
from .gemini import GeminiAdapter  # Assuming this exists based on config
from ..config import AIConfig

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OLLAMA = "ollama"
    GEMINI = "gemini" 
    MOCK = "mock"


class LLMAdapterFactory:
    """
    Factory for creating LLM adapters based on configuration.
    
    Provides a centralized way to instantiate different LLM adapters
    with proper configuration and validation.
    """
    
    # Registry of adapter classes
    _adapters: Dict[LLMProvider, Type[LLMAdapter]] = {
        LLMProvider.OLLAMA: OllamaAdapter,
        LLMProvider.MOCK: MockLLMAdapter,
    }
    
    @classmethod
    def register_adapter(cls, provider: LLMProvider, adapter_class: Type[LLMAdapter]):
        """
        Register a new adapter class.
        
        Args:
            provider: Provider identifier
            adapter_class: Adapter class to register
        """
        cls._adapters[provider] = adapter_class
        logger.info(f"Registered LLM adapter for provider: {provider}")
    
    @classmethod
    def get_available_providers(cls) -> list[LLMProvider]:
        """Get list of available LLM providers."""
        return list(cls._adapters.keys())
    
    @classmethod
    def create_adapter(
        cls,
        provider: Optional[str] = None,
        config: Optional[AIConfig] = None,
        **kwargs
    ) -> LLMAdapter:
        """
        Create an LLM adapter instance.
        
        Args:
            provider: LLM provider name (if None, uses config default)
            config: AI configuration object  
            **kwargs: Additional adapter-specific arguments
            
        Returns:
            Configured LLM adapter instance
            
        Raises:
            LLMError: If provider is unsupported or configuration is invalid
        """
        # Use config if provided, otherwise create default
        if config is None:
            from ..config import get_ai_config
            config = get_ai_config()
        
        # Determine provider
        if provider is None:
            provider = config.default_llm_provider
            
        try:
            provider_enum = LLMProvider(provider.lower())
        except ValueError:
            available = [p.value for p in cls.get_available_providers()]
            raise LLMError(
                message=f"Unsupported LLM provider: {provider}. Available: {available}",
                provider=provider,
                model="unknown",
                error_type="configuration"
            )
        
        # Check if adapter is registered
        if provider_enum not in cls._adapters:
            raise LLMError(
                message=f"No adapter registered for provider: {provider}",
                provider=provider,
                model="unknown",
                error_type="configuration"
            )
        
        # Create adapter with provider-specific configuration
        adapter_class = cls._adapters[provider_enum]
        
        try:
            if provider_enum == LLMProvider.OLLAMA:
                return cls._create_ollama_adapter(config, **kwargs)
            elif provider_enum == LLMProvider.GEMINI:
                return cls._create_gemini_adapter(config, **kwargs)
            elif provider_enum == LLMProvider.MOCK:
                return cls._create_mock_adapter(config, **kwargs)
            else:
                # Generic creation for custom adapters
                return adapter_class(**kwargs)
                
        except Exception as e:
            # Determine model name from config based on provider
            model_name = "unknown"
            if provider_enum == LLMProvider.OLLAMA:
                model_name = config.ollama_model
            elif provider_enum == LLMProvider.GEMINI:
                model_name = config.gemini_model
            
            raise LLMError(
                message=f"Failed to create {provider} adapter: {str(e)}",
                provider=provider,
                model=model_name,
                error_type="initialization",
                cause=e
            )
    
    @classmethod
    def create_default_adapter(cls, config: Optional[AIConfig] = None) -> LLMAdapter:
        """
        Create adapter using default configuration.
        
        Args:
            config: Optional AI configuration
            
        Returns:
            Default LLM adapter instance
        """
        return cls.create_adapter(config=config)
    
    @classmethod  
    def create_ollama_adapter(cls, config: Optional[AIConfig] = None, **kwargs) -> OllamaAdapter:
        """
        Create Ollama adapter specifically.
        
        Args:
            config: Optional AI configuration
            **kwargs: Additional Ollama-specific arguments
            
        Returns:
            Configured Ollama adapter
        """
        return cls._create_ollama_adapter(config or AIConfig(), **kwargs)
    
    @classmethod
    def create_gemini_adapter(cls, config: Optional[AIConfig] = None, **kwargs) -> LLMAdapter:
        """
        Create Gemini adapter specifically.
        
        Args:
            config: Optional AI configuration
            **kwargs: Additional Gemini-specific arguments
            
        Returns:
            Configured Gemini adapter
        """
        return cls._create_gemini_adapter(config or AIConfig(), **kwargs)
    
    @classmethod
    def _create_ollama_adapter(cls, config: AIConfig, **kwargs) -> OllamaAdapter:
        """Create Ollama adapter with configuration."""
        # Check if Ollama is enabled
        if not config.enable_ollama:
            raise LLMError(
                message="Ollama adapter is disabled in configuration",
                provider="ollama",
                model=config.ollama_model,
                error_type="configuration"
            )
        
        # Merge config with kwargs (kwargs take precedence)
        adapter_kwargs = {
            "base_url": config.ollama_base_url,
            "model_name": config.ollama_model,
            "max_tokens": config.ollama_max_tokens,
            "temperature": config.ollama_temperature,
            "timeout": config.ollama_timeout,
            **kwargs
        }
        
        return OllamaAdapter(**adapter_kwargs)
    
    @classmethod
    def _create_gemini_adapter(cls, config: AIConfig, **kwargs) -> LLMAdapter:
        """Create Gemini adapter with configuration."""
        # Import here to avoid circular imports
        try:
            from .gemini import GeminiAdapter
        except ImportError:
            raise LLMError(
                message="Gemini adapter not available - missing implementation",
                provider="gemini",
                model=config.gemini_model,
                error_type="configuration"
            )
        
        # Check if Gemini is enabled and has API key
        if not config.enable_gemini:
            raise LLMError(
                message="Gemini adapter is disabled in configuration",
                provider="gemini",
                model=config.gemini_model,
                error_type="configuration"
            )
        
        if not config.api_key:
            raise LLMError(
                message="Gemini API key not configured",
                provider="gemini",
                model=config.gemini_model,
                error_type="configuration"
            )
        
        # Register Gemini adapter if not already registered
        if LLMProvider.GEMINI not in cls._adapters:
            cls.register_adapter(LLMProvider.GEMINI, GeminiAdapter)
        
        # Merge config with kwargs
        adapter_kwargs = {
            "model_name": config.gemini_model,
            "api_key": config.api_key,
            "max_tokens": config.gemini_max_tokens,
            "temperature": config.gemini_temperature,
            **kwargs
        }
        
        return GeminiAdapter(**adapter_kwargs)
    
    @classmethod
    def _create_mock_adapter(cls, config: AIConfig, **kwargs) -> MockLLMAdapter:
        """Create Mock adapter with configuration."""
        # MockLLMAdapter sets its own model_name, don't override it
        return MockLLMAdapter(**kwargs)
    
    @classmethod
    async def validate_adapter(cls, adapter: LLMAdapter) -> bool:
        """
        Validate that an adapter is working correctly.
        
        Args:
            adapter: LLM adapter to validate
            
        Returns:
            True if adapter is working, False otherwise
        """
        try:
            # Test basic functionality
            model_info = adapter.get_model_info()
            logger.info(f"Adapter validation - Model: {model_info.model_name}, Provider: {model_info.provider}")
            
            # For Ollama, check server availability
            if hasattr(adapter, 'is_available'):
                available = await adapter.is_available()
                if not available:
                    logger.warning(f"Adapter {adapter.get_provider_name()} is not available")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Adapter validation failed: {e}")
            return False
    
    @classmethod
    async def create_and_validate_adapter(
        cls,
        provider: Optional[str] = None,
        config: Optional[AIConfig] = None,
        **kwargs
    ) -> LLMAdapter:
        """
        Create and validate an adapter.
        
        Args:
            provider: LLM provider name
            config: AI configuration
            **kwargs: Additional adapter arguments
            
        Returns:
            Validated LLM adapter
            
        Raises:
            LLMError: If adapter creation or validation fails
        """
        adapter = cls.create_adapter(provider=provider, config=config, **kwargs)
        
        is_valid = await cls.validate_adapter(adapter)
        if not is_valid:
            # Clean up if validation fails
            if hasattr(adapter, 'close'):
                await adapter.close()
            raise LLMError(
                message=f"Adapter validation failed for provider: {adapter.get_provider_name()}",
                provider=adapter.get_provider_name(),
                model=getattr(adapter, 'model_name', 'unknown'),
                error_type="validation"
            )
        
        return adapter


# Convenience functions for common use cases
async def create_default_llm_adapter(config: Optional[AIConfig] = None) -> LLMAdapter:
    """
    Create the default LLM adapter with validation.
    
    Args:
        config: Optional AI configuration
        
    Returns:
        Validated default LLM adapter
    """
    return await LLMAdapterFactory.create_and_validate_adapter(config=config)


async def create_ollama_adapter(config: Optional[AIConfig] = None) -> OllamaAdapter:
    """
    Create an Ollama adapter with validation.
    
    Args:
        config: Optional AI configuration
        
    Returns:
        Validated Ollama adapter
    """
    adapter = await LLMAdapterFactory.create_and_validate_adapter(
        provider="ollama", config=config
    )
    return adapter


def get_available_providers() -> list[str]:
    """Get list of available LLM provider names."""
    return [provider.value for provider in LLMAdapterFactory.get_available_providers()]