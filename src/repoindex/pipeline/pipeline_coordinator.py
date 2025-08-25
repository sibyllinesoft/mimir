"""
Pipeline Coordinator for Mimir 2.0.

Coordinates the enhanced pipeline with RAPTOR, advanced search, and LLM integration.
Provides centralized configuration and dependency management for all new components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass
from enum import Enum

from ..config import AIConfig, get_ai_config
from .llm_adapter_factory import LLMAdapterFactory, LLMAdapter
from .llm_adapter import LLMError

logger = logging.getLogger(__name__)


class PipelineMode(str, Enum):
    """Pipeline execution modes."""
    LEGACY = "legacy"  # Original 6-stage pipeline  
    ENHANCED = "enhanced"  # With RAPTOR and advanced search
    MINIMAL = "minimal"  # Core functionality only


@dataclass 
class PipelineCapabilities:
    """Capabilities available in current pipeline configuration."""
    has_ollama: bool = False
    has_raptor: bool = False
    has_hyde: bool = False
    has_reranking: bool = False
    has_code_embeddings: bool = False
    llm_providers: List[str] = None
    
    def __post_init__(self):
        if self.llm_providers is None:
            self.llm_providers = []


class PipelineCoordinator:
    """
    Coordinates the enhanced Mimir 2.0 pipeline.
    
    Manages configuration, dependencies, and execution flow for all
    pipeline enhancements including RAPTOR, advanced search, and LLM integration.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        """
        Initialize pipeline coordinator.
        
        Args:
            config: AI configuration, uses default if not provided
        """
        self.config = config or get_ai_config()
        self._llm_adapters: Dict[str, LLMAdapter] = {}
        self._capabilities: Optional[PipelineCapabilities] = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize the pipeline coordinator and validate capabilities."""
        if self._initialized:
            return
            
        logger.info("Initializing Mimir 2.0 pipeline coordinator")
        
        # Validate and initialize LLM adapters
        await self._initialize_llm_adapters()
        
        # Assess available capabilities
        self._capabilities = await self._assess_capabilities()
        
        self._initialized = True
        logger.info(f"Pipeline coordinator initialized with capabilities: {self._capabilities}")
    
    async def _initialize_llm_adapters(self) -> None:
        """Initialize and validate LLM adapters."""
        adapters_to_init = []
        
        # Add Ollama if enabled
        if self.config.enable_ollama:
            adapters_to_init.append("ollama")
        
        # Add Gemini if enabled and configured
        if self.config.enable_gemini and self.config.api_key:
            adapters_to_init.append("gemini")
        
        # Always have mock as fallback
        adapters_to_init.append("mock")
        
        for provider in adapters_to_init:
            try:
                adapter = await LLMAdapterFactory.create_and_validate_adapter(
                    provider=provider,
                    config=self.config
                )
                self._llm_adapters[provider] = adapter
                logger.info(f"Initialized {provider} LLM adapter")
                
            except LLMError as e:
                logger.warning(f"Failed to initialize {provider} adapter: {e}")
                
                # For Ollama failures, this is common if server isn't running
                if provider == "ollama":
                    logger.info("Ollama adapter unavailable - ensure Ollama server is running for full functionality")
                
    async def _assess_capabilities(self) -> PipelineCapabilities:
        """Assess available pipeline capabilities based on configuration and adapters."""
        capabilities = PipelineCapabilities()
        
        # Check LLM capabilities
        capabilities.llm_providers = list(self._llm_adapters.keys())
        capabilities.has_ollama = "ollama" in self._llm_adapters
        
        # Check feature capabilities based on config and dependencies
        capabilities.has_raptor = (
            self.config.enable_raptor and 
            capabilities.has_ollama and  # RAPTOR needs LLM for summarization
            self._check_ml_dependencies()
        )
        
        capabilities.has_hyde = (
            self.config.enable_hyde and 
            len(self._llm_adapters) > 0  # HyDE needs any LLM
        )
        
        capabilities.has_reranking = (
            self.config.enable_reranking and
            self._check_ml_dependencies()
        )
        
        capabilities.has_code_embeddings = (
            self.config.enable_code_embeddings and
            self._check_ml_dependencies()
        )
        
        return capabilities
    
    def _check_ml_dependencies(self) -> bool:
        """Check if ML dependencies are available."""
        try:
            import sklearn  # scikit-learn
            import sentence_transformers
            return True
        except ImportError:
            return False
    
    def get_pipeline_mode(self) -> PipelineMode:
        """Determine the appropriate pipeline mode based on capabilities."""
        if not self._capabilities:
            return PipelineMode.MINIMAL
            
        # Enhanced mode if we have advanced capabilities
        if (self._capabilities.has_raptor or 
            self._capabilities.has_hyde or 
            self._capabilities.has_reranking):
            return PipelineMode.ENHANCED
            
        # Legacy mode if we have basic LLM support
        if len(self._capabilities.llm_providers) > 1:  # More than just mock
            return PipelineMode.LEGACY
            
        return PipelineMode.MINIMAL
    
    async def get_llm_adapter(self, provider: Optional[str] = None) -> LLMAdapter:
        """
        Get an LLM adapter for the specified provider.
        
        Args:
            provider: LLM provider name, uses default if not specified
            
        Returns:
            LLM adapter instance
            
        Raises:
            LLMError: If no suitable adapter is available
        """
        if not self._initialized:
            await self.initialize()
        
        # Use default provider if not specified
        if provider is None:
            provider = self.config.default_llm_provider
        
        # Get the adapter
        if provider in self._llm_adapters:
            return self._llm_adapters[provider]
        
        # Fallback to any available adapter
        if self._llm_adapters:
            fallback_provider = next(iter(self._llm_adapters.keys()))
            logger.warning(f"Provider {provider} not available, falling back to {fallback_provider}")
            return self._llm_adapters[fallback_provider]
        
        raise LLMError(
            message="No LLM adapters available",
            provider=provider or "unknown",
            model="unknown",
            error_type="configuration"
        )
    
    async def get_summarization_adapter(self) -> LLMAdapter:
        """Get the best LLM adapter for summarization (RAPTOR)."""
        # Prefer Ollama for local, fast summarization
        if "ollama" in self._llm_adapters:
            return self._llm_adapters["ollama"]
        
        # Fall back to any available adapter
        return await self.get_llm_adapter()
    
    async def get_hyde_adapter(self) -> LLMAdapter:
        """Get the best LLM adapter for HyDE query transformation."""
        # Prefer Ollama for fast query transformation
        if "ollama" in self._llm_adapters:
            return self._llm_adapters["ollama"]
            
        # Fall back to any available adapter
        return await self.get_llm_adapter()
    
    def get_capabilities(self) -> PipelineCapabilities:
        """Get current pipeline capabilities."""
        if not self._capabilities:
            raise RuntimeError("Pipeline coordinator not initialized")
        return self._capabilities
    
    def is_feature_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled and available."""
        if not self._capabilities:
            return False
            
        feature_map = {
            "raptor": self._capabilities.has_raptor,
            "hyde": self._capabilities.has_hyde, 
            "reranking": self._capabilities.has_reranking,
            "code_embeddings": self._capabilities.has_code_embeddings,
            "ollama": self._capabilities.has_ollama,
        }
        
        return feature_map.get(feature, False)
    
    async def assess_capabilities(self) -> Dict[str, bool]:
        """
        Assess and return available pipeline capabilities.
        
        Returns:
            Dictionary mapping capability names to availability status
        """
        if not self._initialized:
            await self.initialize()
        
        if not self._capabilities:
            return {}
        
        return {
            "ollama": self._capabilities.has_ollama,
            "raptor": self._capabilities.has_raptor,
            "hyde": self._capabilities.has_hyde,
            "reranking": self._capabilities.has_reranking,
            "code_embeddings": self._capabilities.has_code_embeddings,
            "enhanced_search": self._capabilities.has_hyde or self._capabilities.has_reranking,
        }
    
    async def validate_configuration(self) -> Dict[str, Any]:
        """
        Validate the current configuration and return status report.
        
        Returns:
            Configuration validation report
        """
        if not self._initialized:
            await self.initialize()
        
        report = {
            "pipeline_mode": self.get_pipeline_mode().value,
            "capabilities": {
                "ollama": self._capabilities.has_ollama,
                "raptor": self._capabilities.has_raptor,
                "hyde": self._capabilities.has_hyde,
                "reranking": self._capabilities.has_reranking,
                "code_embeddings": self._capabilities.has_code_embeddings,
            },
            "llm_providers": self._capabilities.llm_providers,
            "config_status": {
                "ollama_enabled": self.config.enable_ollama,
                "raptor_enabled": self.config.enable_raptor,
                "hyde_enabled": self.config.enable_hyde,
                "reranking_enabled": self.config.enable_reranking,
                "code_embeddings_enabled": self.config.enable_code_embeddings,
            },
            "warnings": [],
            "errors": []
        }
        
        # Check for common configuration issues
        if self.config.enable_ollama and not self._capabilities.has_ollama:
            report["warnings"].append("Ollama is enabled but not available - check if Ollama server is running")
        
        if self.config.enable_raptor and not self._capabilities.has_raptor:
            report["errors"].append("RAPTOR is enabled but dependencies are missing - install ML dependencies")
        
        if not self._llm_adapters or len(self._llm_adapters) == 1 and "mock" in self._llm_adapters:
            report["warnings"].append("Only mock LLM adapter available - configure Ollama or Gemini for full functionality")
        
        return report
    
    async def cleanup(self) -> None:
        """Clean up resources and close connections."""
        logger.info("Cleaning up pipeline coordinator")
        
        for provider, adapter in self._llm_adapters.items():
            try:
                if hasattr(adapter, 'close'):
                    await adapter.close()
                logger.debug(f"Closed {provider} adapter")
            except Exception as e:
                logger.warning(f"Error closing {provider} adapter: {e}")
        
        self._llm_adapters.clear()
        self._initialized = False


# Global coordinator instance
_coordinator: Optional[PipelineCoordinator] = None


async def get_pipeline_coordinator(config: Optional[AIConfig] = None) -> PipelineCoordinator:
    """
    Get the global pipeline coordinator instance.
    
    Args:
        config: Optional AI configuration
        
    Returns:
        Initialized pipeline coordinator
    """
    global _coordinator
    
    if _coordinator is None:
        _coordinator = PipelineCoordinator(config)
        await _coordinator.initialize()
    
    return _coordinator


async def cleanup_pipeline_coordinator() -> None:
    """Clean up the global pipeline coordinator."""
    global _coordinator
    
    if _coordinator:
        await _coordinator.cleanup()
        _coordinator = None