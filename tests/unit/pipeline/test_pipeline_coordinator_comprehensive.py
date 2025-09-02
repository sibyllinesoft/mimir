#!/usr/bin/env python3
"""
Comprehensive tests for PipelineCoordinator.

Tests the central pipeline orchestration component that manages configuration,
dependencies, LLM adapters, and capability assessment for Mimir 2.0.
"""

import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Optional

# Import the modules we're testing
from repoindex.pipeline.pipeline_coordinator import (
    PipelineCoordinator, 
    PipelineMode, 
    PipelineCapabilities,
    get_pipeline_coordinator,
    cleanup_pipeline_coordinator,
    _coordinator
)
from repoindex.config import AIConfig
from repoindex.pipeline.llm_adapter import LLMError, LLMAdapter


class TestPipelineCapabilities:
    """Test PipelineCapabilities dataclass."""
    
    def test_pipeline_capabilities_initialization_default(self):
        """Test default initialization of PipelineCapabilities."""
        caps = PipelineCapabilities()
        
        assert caps.has_ollama is False
        assert caps.has_raptor is False
        assert caps.has_hyde is False
        assert caps.has_reranking is False
        assert caps.has_code_embeddings is False
        assert caps.llm_providers == []
    
    def test_pipeline_capabilities_initialization_custom(self):
        """Test custom initialization of PipelineCapabilities."""
        providers = ["ollama", "gemini"]
        caps = PipelineCapabilities(
            has_ollama=True,
            has_raptor=True,
            has_hyde=True,
            has_reranking=False,
            has_code_embeddings=True,
            llm_providers=providers
        )
        
        assert caps.has_ollama is True
        assert caps.has_raptor is True
        assert caps.has_hyde is True
        assert caps.has_reranking is False
        assert caps.has_code_embeddings is True
        assert caps.llm_providers == providers
    
    def test_pipeline_capabilities_post_init(self):
        """Test __post_init__ sets empty list for None llm_providers."""
        caps = PipelineCapabilities(llm_providers=None)
        assert caps.llm_providers == []


class TestPipelineMode:
    """Test PipelineMode enum."""
    
    def test_pipeline_mode_values(self):
        """Test PipelineMode enum values."""
        assert PipelineMode.LEGACY == "legacy"
        assert PipelineMode.ENHANCED == "enhanced"
        assert PipelineMode.MINIMAL == "minimal"
    
    def test_pipeline_mode_string_conversion(self):
        """Test PipelineMode string conversion."""
        assert str(PipelineMode.LEGACY) == "legacy"
        assert str(PipelineMode.ENHANCED) == "enhanced"
        assert str(PipelineMode.MINIMAL) == "minimal"


class TestPipelineCoordinator:
    """Test PipelineCoordinator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_config = Mock(spec=AIConfig)
        self.mock_config.enable_ollama = True
        self.mock_config.enable_gemini = True
        self.mock_config.enable_raptor = True
        self.mock_config.enable_hyde = True
        self.mock_config.enable_reranking = True
        self.mock_config.enable_code_embeddings = True
        self.mock_config.api_key = "test_api_key"
        self.mock_config.default_llm_provider = "ollama"
    
    def test_coordinator_initialization_default_config(self):
        """Test coordinator initialization with default config."""
        with patch('repoindex.pipeline.pipeline_coordinator.get_ai_config') as mock_get_config:
            mock_get_config.return_value = self.mock_config
            coordinator = PipelineCoordinator()
            
            assert coordinator.config == self.mock_config
            assert coordinator._llm_adapters == {}
            assert coordinator._capabilities is None
            assert coordinator._initialized is False
    
    def test_coordinator_initialization_custom_config(self):
        """Test coordinator initialization with custom config."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        assert coordinator.config == self.mock_config
        assert coordinator._llm_adapters == {}
        assert coordinator._capabilities is None
        assert coordinator._initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful coordinator initialization."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        mock_ollama_adapter = Mock(spec=LLMAdapter)
        mock_gemini_adapter = Mock(spec=LLMAdapter)
        mock_mock_adapter = Mock(spec=LLMAdapter)
        
        with patch('repoindex.pipeline.pipeline_coordinator.LLMAdapterFactory.create_and_validate_adapter') as mock_create:
            mock_create.side_effect = [mock_ollama_adapter, mock_gemini_adapter, mock_mock_adapter]
            
            with patch.object(coordinator, '_check_ml_dependencies', return_value=True):
                await coordinator.initialize()
        
        assert coordinator._initialized is True
        assert len(coordinator._llm_adapters) == 3
        assert "ollama" in coordinator._llm_adapters
        assert "gemini" in coordinator._llm_adapters
        assert "mock" in coordinator._llm_adapters
        assert coordinator._capabilities is not None
    
    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initialization when already initialized."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._initialized = True
        
        with patch.object(coordinator, '_initialize_llm_adapters') as mock_init_adapters:
            await coordinator.initialize()
            
        # Should not call initialization again
        mock_init_adapters.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_initialize_llm_adapters_ollama_enabled(self):
        """Test LLM adapter initialization when Ollama is enabled."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        mock_ollama_adapter = Mock(spec=LLMAdapter)
        mock_gemini_adapter = Mock(spec=LLMAdapter)
        mock_mock_adapter = Mock(spec=LLMAdapter)
        
        with patch('repoindex.pipeline.pipeline_coordinator.LLMAdapterFactory.create_and_validate_adapter') as mock_create:
            mock_create.side_effect = [mock_ollama_adapter, mock_gemini_adapter, mock_mock_adapter]
            
            await coordinator._initialize_llm_adapters()
        
        assert len(coordinator._llm_adapters) == 3
        assert coordinator._llm_adapters["ollama"] == mock_ollama_adapter
        assert coordinator._llm_adapters["gemini"] == mock_gemini_adapter
        assert coordinator._llm_adapters["mock"] == mock_mock_adapter
    
    @pytest.mark.asyncio
    async def test_initialize_llm_adapters_ollama_failure(self):
        """Test LLM adapter initialization when Ollama fails."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        mock_gemini_adapter = Mock(spec=LLMAdapter)
        mock_mock_adapter = Mock(spec=LLMAdapter)
        
        with patch('repoindex.pipeline.pipeline_coordinator.LLMAdapterFactory.create_and_validate_adapter') as mock_create:
            mock_create.side_effect = [
                LLMError("Ollama connection failed", "ollama", "llama2", "connection"),
                mock_gemini_adapter,
                mock_mock_adapter
            ]
            
            await coordinator._initialize_llm_adapters()
        
        # Should continue with other adapters even if one fails
        assert len(coordinator._llm_adapters) == 2
        assert "ollama" not in coordinator._llm_adapters
        assert coordinator._llm_adapters["gemini"] == mock_gemini_adapter
        assert coordinator._llm_adapters["mock"] == mock_mock_adapter
    
    @pytest.mark.asyncio
    async def test_initialize_llm_adapters_disabled_features(self):
        """Test LLM adapter initialization with disabled features."""
        self.mock_config.enable_ollama = False
        self.mock_config.enable_gemini = False
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        mock_mock_adapter = Mock(spec=LLMAdapter)
        
        with patch('repoindex.pipeline.pipeline_coordinator.LLMAdapterFactory.create_and_validate_adapter') as mock_create:
            mock_create.return_value = mock_mock_adapter
            
            await coordinator._initialize_llm_adapters()
        
        # Should only initialize mock adapter
        assert len(coordinator._llm_adapters) == 1
        assert coordinator._llm_adapters["mock"] == mock_mock_adapter
    
    @pytest.mark.asyncio
    async def test_assess_capabilities_full_features(self):
        """Test capability assessment with all features available."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._llm_adapters = {"ollama": Mock(), "gemini": Mock(), "mock": Mock()}
        
        with patch.object(coordinator, '_check_ml_dependencies', return_value=True):
            capabilities = await coordinator._assess_capabilities()
        
        assert capabilities.llm_providers == ["ollama", "gemini", "mock"]
        assert capabilities.has_ollama is True
        assert capabilities.has_raptor is True  # Has Ollama and ML dependencies
        assert capabilities.has_hyde is True  # Has LLM adapters
        assert capabilities.has_reranking is True  # Has ML dependencies
        assert capabilities.has_code_embeddings is True  # Has ML dependencies
    
    @pytest.mark.asyncio
    async def test_assess_capabilities_no_ml_dependencies(self):
        """Test capability assessment without ML dependencies."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._llm_adapters = {"ollama": Mock(), "mock": Mock()}
        
        with patch.object(coordinator, '_check_ml_dependencies', return_value=False):
            capabilities = await coordinator._assess_capabilities()
        
        assert capabilities.has_ollama is True
        assert capabilities.has_raptor is False  # Missing ML dependencies
        assert capabilities.has_hyde is True  # Has LLM adapters
        assert capabilities.has_reranking is False  # Missing ML dependencies
        assert capabilities.has_code_embeddings is False  # Missing ML dependencies
    
    @pytest.mark.asyncio
    async def test_assess_capabilities_no_ollama(self):
        """Test capability assessment without Ollama."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._llm_adapters = {"gemini": Mock(), "mock": Mock()}
        
        with patch.object(coordinator, '_check_ml_dependencies', return_value=True):
            capabilities = await coordinator._assess_capabilities()
        
        assert capabilities.has_ollama is False
        assert capabilities.has_raptor is False  # Needs Ollama
        assert capabilities.has_hyde is True  # Has other LLM adapters
        assert capabilities.has_reranking is True  # Has ML dependencies
        assert capabilities.has_code_embeddings is True  # Has ML dependencies
    
    def test_check_ml_dependencies_available(self):
        """Test ML dependencies check when available."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        with patch('builtins.__import__') as mock_import:
            result = coordinator._check_ml_dependencies()
            assert result is True
    
    def test_check_ml_dependencies_missing(self):
        """Test ML dependencies check when missing."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        def mock_import(name, *args, **kwargs):
            if name in ['sklearn', 'sentence_transformers']:
                raise ImportError(f"No module named '{name}'")
            return Mock()
        
        with patch('builtins.__import__', side_effect=mock_import):
            result = coordinator._check_ml_dependencies()
            assert result is False
    
    def test_get_pipeline_mode_enhanced(self):
        """Test pipeline mode determination for enhanced mode."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(
            has_raptor=True,
            has_hyde=True,
            has_reranking=True,
            llm_providers=["ollama", "gemini"]
        )
        
        mode = coordinator.get_pipeline_mode()
        assert mode == PipelineMode.ENHANCED
    
    def test_get_pipeline_mode_legacy(self):
        """Test pipeline mode determination for legacy mode."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(
            has_raptor=False,
            has_hyde=False,
            has_reranking=False,
            llm_providers=["gemini", "mock"]  # More than just mock
        )
        
        mode = coordinator.get_pipeline_mode()
        assert mode == PipelineMode.LEGACY
    
    def test_get_pipeline_mode_minimal(self):
        """Test pipeline mode determination for minimal mode."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(
            has_raptor=False,
            has_hyde=False,
            has_reranking=False,
            llm_providers=["mock"]  # Only mock
        )
        
        mode = coordinator.get_pipeline_mode()
        assert mode == PipelineMode.MINIMAL
    
    def test_get_pipeline_mode_no_capabilities(self):
        """Test pipeline mode determination with no capabilities."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = None
        
        mode = coordinator.get_pipeline_mode()
        assert mode == PipelineMode.MINIMAL
    
    @pytest.mark.asyncio
    async def test_get_llm_adapter_default_provider(self):
        """Test getting LLM adapter with default provider."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._initialized = True
        mock_adapter = Mock(spec=LLMAdapter)
        coordinator._llm_adapters["ollama"] = mock_adapter
        
        result = await coordinator.get_llm_adapter()
        assert result == mock_adapter
    
    @pytest.mark.asyncio
    async def test_get_llm_adapter_specific_provider(self):
        """Test getting LLM adapter with specific provider."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._initialized = True
        mock_adapter = Mock(spec=LLMAdapter)
        coordinator._llm_adapters["gemini"] = mock_adapter
        
        result = await coordinator.get_llm_adapter("gemini")
        assert result == mock_adapter
    
    @pytest.mark.asyncio
    async def test_get_llm_adapter_fallback(self):
        """Test getting LLM adapter with fallback."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._initialized = True
        mock_adapter = Mock(spec=LLMAdapter)
        coordinator._llm_adapters["mock"] = mock_adapter
        
        result = await coordinator.get_llm_adapter("nonexistent")
        assert result == mock_adapter
    
    @pytest.mark.asyncio
    async def test_get_llm_adapter_no_adapters(self):
        """Test getting LLM adapter when none available."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._initialized = True
        coordinator._llm_adapters = {}
        
        with pytest.raises(LLMError) as exc_info:
            await coordinator.get_llm_adapter()
        
        assert "No LLM adapters available" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_llm_adapter_auto_initialize(self):
        """Test getting LLM adapter triggers auto-initialization."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._initialized = False
        
        with patch.object(coordinator, 'initialize', new_callable=AsyncMock) as mock_init:
            mock_adapter = Mock(spec=LLMAdapter)
            coordinator._llm_adapters["ollama"] = mock_adapter
            coordinator._initialized = True  # Set after mock init
            
            result = await coordinator.get_llm_adapter()
            
            mock_init.assert_called_once()
            assert result == mock_adapter
    
    @pytest.mark.asyncio
    async def test_get_summarization_adapter_ollama_preferred(self):
        """Test getting summarization adapter prefers Ollama."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        mock_ollama = Mock(spec=LLMAdapter)
        mock_gemini = Mock(spec=LLMAdapter)
        coordinator._llm_adapters = {"ollama": mock_ollama, "gemini": mock_gemini}
        
        result = await coordinator.get_summarization_adapter()
        assert result == mock_ollama
    
    @pytest.mark.asyncio
    async def test_get_hyde_adapter_ollama_preferred(self):
        """Test getting HyDE adapter prefers Ollama."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        mock_ollama = Mock(spec=LLMAdapter)
        mock_gemini = Mock(spec=LLMAdapter)
        coordinator._llm_adapters = {"ollama": mock_ollama, "gemini": mock_gemini}
        
        result = await coordinator.get_hyde_adapter()
        assert result == mock_ollama
    
    def test_get_capabilities_initialized(self):
        """Test getting capabilities when initialized."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        mock_capabilities = PipelineCapabilities(has_ollama=True)
        coordinator._capabilities = mock_capabilities
        
        result = coordinator.get_capabilities()
        assert result == mock_capabilities
    
    def test_get_capabilities_not_initialized(self):
        """Test getting capabilities when not initialized."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = None
        
        with pytest.raises(RuntimeError) as exc_info:
            coordinator.get_capabilities()
        
        assert "Pipeline coordinator not initialized" in str(exc_info.value)
    
    def test_is_feature_enabled_true(self):
        """Test feature enabled check returns True."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(has_raptor=True)
        
        result = coordinator.is_feature_enabled("raptor")
        assert result is True
    
    def test_is_feature_enabled_false(self):
        """Test feature enabled check returns False."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(has_raptor=False)
        
        result = coordinator.is_feature_enabled("raptor")
        assert result is False
    
    def test_is_feature_enabled_unknown_feature(self):
        """Test feature enabled check for unknown feature."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities()
        
        result = coordinator.is_feature_enabled("unknown")
        assert result is False
    
    def test_is_feature_enabled_no_capabilities(self):
        """Test feature enabled check with no capabilities."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = None
        
        result = coordinator.is_feature_enabled("raptor")
        assert result is False
    
    @pytest.mark.asyncio
    async def test_assess_capabilities_method(self):
        """Test assess_capabilities method."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(
            has_ollama=True,
            has_raptor=True,
            has_hyde=False,
            has_reranking=True,
            has_code_embeddings=False
        )
        coordinator._initialized = True
        
        result = await coordinator.assess_capabilities()
        
        expected = {
            "ollama": True,
            "raptor": True,
            "hyde": False,
            "reranking": True,
            "code_embeddings": False,
            "enhanced_search": True,  # True because reranking is True
        }
        assert result == expected
    
    @pytest.mark.asyncio
    async def test_validate_configuration_full_report(self):
        """Test configuration validation with full report."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(
            has_ollama=True,
            has_raptor=True,
            has_hyde=True,
            has_reranking=False,  # Not available
            has_code_embeddings=True,
            llm_providers=["ollama", "gemini"]
        )
        coordinator._llm_adapters = {"ollama": Mock(), "gemini": Mock()}
        coordinator._initialized = True
        
        report = await coordinator.validate_configuration()
        
        assert report["pipeline_mode"] == "enhanced"
        assert report["capabilities"]["ollama"] is True
        assert report["capabilities"]["raptor"] is True
        assert report["capabilities"]["reranking"] is False
        assert report["llm_providers"] == ["ollama", "gemini"]
        assert report["config_status"]["ollama_enabled"] is True
        assert len(report["warnings"]) == 0  # No warnings expected
        assert len(report["errors"]) == 0  # No errors expected
    
    @pytest.mark.asyncio
    async def test_validate_configuration_with_issues(self):
        """Test configuration validation with issues."""
        self.mock_config.enable_raptor = True
        coordinator = PipelineCoordinator(config=self.mock_config)
        coordinator._capabilities = PipelineCapabilities(
            has_ollama=False,  # Enabled but not available
            has_raptor=False,  # Enabled but not available
            llm_providers=["mock"]  # Only mock available
        )
        coordinator._llm_adapters = {"mock": Mock()}
        coordinator._initialized = True
        
        report = await coordinator.validate_configuration()
        
        assert len(report["warnings"]) >= 1
        assert len(report["errors"]) >= 1
        assert any("Ollama is enabled but not available" in w for w in report["warnings"])
        assert any("RAPTOR is enabled but dependencies are missing" in e for e in report["errors"])
        assert any("Only mock LLM adapter available" in w for w in report["warnings"])
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test coordinator cleanup."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        mock_adapter1 = Mock(spec=LLMAdapter)
        mock_adapter1.close = AsyncMock()
        mock_adapter2 = Mock(spec=LLMAdapter)
        # mock_adapter2 doesn't have close method
        
        coordinator._llm_adapters = {"ollama": mock_adapter1, "mock": mock_adapter2}
        coordinator._initialized = True
        
        await coordinator.cleanup()
        
        mock_adapter1.close.assert_called_once()
        assert coordinator._llm_adapters == {}
        assert coordinator._initialized is False
    
    @pytest.mark.asyncio
    async def test_cleanup_with_errors(self):
        """Test coordinator cleanup with adapter errors."""
        coordinator = PipelineCoordinator(config=self.mock_config)
        
        mock_adapter = Mock(spec=LLMAdapter)
        mock_adapter.close = AsyncMock(side_effect=Exception("Close failed"))
        
        coordinator._llm_adapters = {"ollama": mock_adapter}
        coordinator._initialized = True
        
        # Should not raise exception even if cleanup fails
        await coordinator.cleanup()
        
        assert coordinator._llm_adapters == {}
        assert coordinator._initialized is False


class TestGlobalCoordinatorFunctions:
    """Test global coordinator management functions."""
    
    def teardown_method(self):
        """Clean up after each test."""
        # Reset global coordinator
        import repoindex.pipeline.pipeline_coordinator
        repoindex.pipeline.pipeline_coordinator._coordinator = None
    
    @pytest.mark.asyncio
    async def test_get_pipeline_coordinator_first_call(self):
        """Test first call to get_pipeline_coordinator creates instance."""
        mock_config = Mock(spec=AIConfig)
        
        with patch('repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator_instance = Mock()
            mock_coordinator_instance.initialize = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator_instance
            
            result = await get_pipeline_coordinator(mock_config)
            
            mock_coordinator_class.assert_called_once_with(mock_config)
            mock_coordinator_instance.initialize.assert_called_once()
            assert result == mock_coordinator_instance
    
    @pytest.mark.asyncio
    async def test_get_pipeline_coordinator_subsequent_calls(self):
        """Test subsequent calls return same instance."""
        mock_config = Mock(spec=AIConfig)
        
        with patch('repoindex.pipeline.pipeline_coordinator.PipelineCoordinator') as mock_coordinator_class:
            mock_coordinator_instance = Mock()
            mock_coordinator_instance.initialize = AsyncMock()
            mock_coordinator_class.return_value = mock_coordinator_instance
            
            result1 = await get_pipeline_coordinator(mock_config)
            result2 = await get_pipeline_coordinator(mock_config)
            
            # Should only create and initialize once
            mock_coordinator_class.assert_called_once()
            mock_coordinator_instance.initialize.assert_called_once()
            assert result1 == result2 == mock_coordinator_instance
    
    @pytest.mark.asyncio
    async def test_cleanup_pipeline_coordinator_with_instance(self):
        """Test cleanup with existing coordinator instance."""
        # Set up global coordinator
        import repoindex.pipeline.pipeline_coordinator
        mock_coordinator = Mock()
        mock_coordinator.cleanup = AsyncMock()
        repoindex.pipeline.pipeline_coordinator._coordinator = mock_coordinator
        
        await cleanup_pipeline_coordinator()
        
        mock_coordinator.cleanup.assert_called_once()
        assert repoindex.pipeline.pipeline_coordinator._coordinator is None
    
    @pytest.mark.asyncio
    async def test_cleanup_pipeline_coordinator_no_instance(self):
        """Test cleanup with no coordinator instance."""
        # Ensure no global coordinator exists
        import repoindex.pipeline.pipeline_coordinator
        repoindex.pipeline.pipeline_coordinator._coordinator = None
        
        # Should not raise exception
        await cleanup_pipeline_coordinator()
        
        assert repoindex.pipeline.pipeline_coordinator._coordinator is None


def run_tests():
    """Run all tests when script is executed directly."""
    import subprocess
    import sys
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)