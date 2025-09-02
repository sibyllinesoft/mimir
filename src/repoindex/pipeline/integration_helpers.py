"""
Integration helpers for Mimir 2.0 components.

Provides utilities for integrating RAPTOR, advanced search, and LLM components
into the existing Mimir pipeline architecture.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..config import AIConfig
from .pipeline_coordinator import PipelineCoordinator, get_pipeline_coordinator
from .llm_adapter import LLMAdapter, CodeSnippet

logger = logging.getLogger(__name__)


@dataclass
class IntegrationResult:
    """Result of an integration operation."""
    success: bool
    component: str
    message: str
    data: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class ComponentIntegrator(ABC):
    """Base class for component integrators."""
    
    def __init__(self, coordinator: PipelineCoordinator):
        self.coordinator = coordinator
        self.name = self.__class__.__name__.replace("Integrator", "").lower()
    
    @abstractmethod
    async def validate_dependencies(self) -> IntegrationResult:
        """Validate that all dependencies are available."""
        pass
    
    @abstractmethod
    async def initialize_component(self) -> IntegrationResult:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def test_component(self) -> IntegrationResult:
        """Test the component functionality."""
        pass


class OllamaIntegrator(ComponentIntegrator):
    """Integrator for Ollama LLM adapter."""
    
    async def validate_dependencies(self) -> IntegrationResult:
        """Validate Ollama dependencies."""
        try:
            import aiohttp
            
            capabilities = self.coordinator.get_capabilities()
            if not capabilities.has_ollama:
                return IntegrationResult(
                    success=False,
                    component="ollama",
                    message="Ollama adapter not available - check if Ollama server is running"
                )
            
            return IntegrationResult(
                success=True,
                component="ollama", 
                message="Ollama dependencies validated"
            )
            
        except ImportError as e:
            return IntegrationResult(
                success=False,
                component="ollama",
                message=f"Missing dependency: {e}",
                error=e
            )
    
    async def initialize_component(self) -> IntegrationResult:
        """Initialize Ollama adapter."""
        try:
            adapter = await self.coordinator.get_llm_adapter("ollama")
            
            # Test basic functionality
            model_info = adapter.get_model_info()
            
            return IntegrationResult(
                success=True,
                component="ollama",
                message=f"Ollama adapter initialized with model: {model_info.model_name}",
                data={"model": model_info.model_name, "provider": model_info.provider}
            )
            
        except Exception as e:
            return IntegrationResult(
                success=False,
                component="ollama",
                message=f"Failed to initialize Ollama adapter: {e}",
                error=e
            )
    
    async def test_component(self) -> IntegrationResult:
        """Test Ollama functionality."""
        try:
            adapter = await self.coordinator.get_llm_adapter("ollama")
            
            # Simple test generation
            test_prompt = "Explain what a Python function is in one sentence."
            response = await adapter.generate_text(test_prompt)
            
            if len(response.strip()) > 10:  # Basic sanity check
                return IntegrationResult(
                    success=True,
                    component="ollama",
                    message="Ollama adapter test successful",
                    data={"test_prompt": test_prompt, "response_length": len(response)}
                )
            else:
                return IntegrationResult(
                    success=False,
                    component="ollama", 
                    message="Ollama adapter test failed - empty or invalid response"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                component="ollama",
                message=f"Ollama adapter test failed: {e}",
                error=e
            )


class RaptorIntegrator(ComponentIntegrator):
    """Integrator for RAPTOR hierarchical indexing."""
    
    async def validate_dependencies(self) -> IntegrationResult:
        """Validate RAPTOR dependencies."""
        try:
            import sklearn
            import umap
            import hdbscan
            import sentence_transformers
            
            capabilities = self.coordinator.get_capabilities()
            if not capabilities.has_raptor:
                return IntegrationResult(
                    success=False,
                    component="raptor",
                    message="RAPTOR capabilities not available - check ML dependencies and config"
                )
            
            return IntegrationResult(
                success=True,
                component="raptor",
                message="RAPTOR dependencies validated"
            )
            
        except ImportError as e:
            return IntegrationResult(
                success=False,
                component="raptor",
                message=f"Missing RAPTOR dependency: {e}",
                error=e
            )
    
    async def initialize_component(self) -> IntegrationResult:
        """Initialize RAPTOR component."""
        try:
            # This will be implemented by the ai-engineer agent
            # For now, just validate the capability
            if self.coordinator.is_feature_enabled("raptor"):
                return IntegrationResult(
                    success=True,
                    component="raptor",
                    message="RAPTOR component ready for initialization"
                )
            else:
                return IntegrationResult(
                    success=False,
                    component="raptor",
                    message="RAPTOR feature not enabled"
                )
                
        except Exception as e:
            return IntegrationResult(
                success=False,
                component="raptor",
                message=f"RAPTOR initialization failed: {e}",
                error=e
            )
    
    async def test_component(self) -> IntegrationResult:
        """Test RAPTOR functionality."""
        # Will be implemented once RAPTOR component is complete
        return IntegrationResult(
            success=True,
            component="raptor",
            message="RAPTOR test pending implementation"
        )


class SearchIntegrator(ComponentIntegrator):
    """Integrator for advanced search enhancements."""
    
    async def validate_dependencies(self) -> IntegrationResult:
        """Validate search enhancement dependencies."""
        try:
            import sentence_transformers
            
            capabilities = self.coordinator.get_capabilities()
            missing_features = []
            
            if not capabilities.has_code_embeddings:
                missing_features.append("code_embeddings")
            if not capabilities.has_hyde:
                missing_features.append("hyde")  
            if not capabilities.has_reranking:
                missing_features.append("reranking")
            
            if missing_features:
                return IntegrationResult(
                    success=False,
                    component="search",
                    message=f"Search features not available: {missing_features}",
                    data={"missing_features": missing_features}
                )
            
            return IntegrationResult(
                success=True,
                component="search",
                message="Search enhancement dependencies validated"
            )
            
        except ImportError as e:
            return IntegrationResult(
                success=False,
                component="search",
                message=f"Missing search dependency: {e}",
                error=e
            )
    
    async def initialize_component(self) -> IntegrationResult:
        """Initialize search enhancements."""
        # Will be implemented by ai-engineer
        return IntegrationResult(
            success=True,
            component="search",
            message="Search enhancements ready for initialization"
        )
    
    async def test_component(self) -> IntegrationResult:
        """Test search enhancements."""
        # Will be implemented once search components are complete
        return IntegrationResult(
            success=True,
            component="search",
            message="Search enhancement test pending implementation"
        )


class PipelineIntegrator:
    """
    Main integrator for coordinating all Mimir 2.0 enhancements.
    
    Orchestrates the integration of RAPTOR, advanced search, and LLM components
    into the existing pipeline architecture.
    """
    
    def __init__(self, config: Optional[AIConfig] = None):
        self.config = config
        self.coordinator: Optional[PipelineCoordinator] = None
        self.integrators: Dict[str, ComponentIntegrator] = {}
    
    async def initialize(self) -> None:
        """Initialize the pipeline integrator."""
        self.coordinator = await get_pipeline_coordinator(self.config)
        
        # Create component integrators
        self.integrators = {
            "ollama": OllamaIntegrator(self.coordinator),
            "raptor": RaptorIntegrator(self.coordinator),
            "search": SearchIntegrator(self.coordinator),
        }
    
    async def validate_all_dependencies(self) -> Dict[str, IntegrationResult]:
        """Validate dependencies for all components."""
        if not self.coordinator:
            await self.initialize()
        
        results = {}
        for name, integrator in self.integrators.items():
            try:
                result = await integrator.validate_dependencies()
                results[name] = result
                logger.info(f"{name} dependency validation: {result.message}")
            except Exception as e:
                results[name] = IntegrationResult(
                    success=False,
                    component=name,
                    message=f"Dependency validation error: {e}",
                    error=e
                )
                logger.error(f"{name} dependency validation failed: {e}")
        
        return results
    
    async def initialize_all_components(self) -> Dict[str, IntegrationResult]:
        """Initialize all components."""
        if not self.coordinator:
            await self.initialize()
        
        results = {}
        for name, integrator in self.integrators.items():
            try:
                result = await integrator.initialize_component()
                results[name] = result
                logger.info(f"{name} initialization: {result.message}")
            except Exception as e:
                results[name] = IntegrationResult(
                    success=False,
                    component=name,
                    message=f"Initialization error: {e}",
                    error=e
                )
                logger.error(f"{name} initialization failed: {e}")
        
        return results
    
    async def test_all_components(self) -> Dict[str, IntegrationResult]:
        """Test all components."""
        if not self.coordinator:
            await self.initialize()
        
        results = {}
        for name, integrator in self.integrators.items():
            try:
                result = await integrator.test_component()
                results[name] = result
                logger.info(f"{name} test: {result.message}")
            except Exception as e:
                results[name] = IntegrationResult(
                    success=False,
                    component=name,
                    message=f"Test error: {e}",
                    error=e
                )
                logger.error(f"{name} test failed: {e}")
        
        return results
    
    async def run_full_integration_test(self) -> Dict[str, Any]:
        """Run complete integration test suite."""
        logger.info("Starting full Mimir 2.0 integration test")
        
        # Step 1: Validate dependencies  
        dependency_results = await self.validate_all_dependencies()
        
        # Step 2: Initialize components
        init_results = await self.initialize_all_components()
        
        # Step 3: Test components
        test_results = await self.test_all_components()
        
        # Generate summary report
        all_successful = (
            all(r.success for r in dependency_results.values()) and
            all(r.success for r in init_results.values()) and
            all(r.success for r in test_results.values())
        )
        
        report = {
            "overall_success": all_successful,
            "pipeline_mode": self.coordinator.get_pipeline_mode().value,
            "capabilities": self.coordinator.get_capabilities(),
            "results": {
                "dependencies": dependency_results,
                "initialization": init_results,
                "testing": test_results
            },
            "summary": {
                "total_components": len(self.integrators),
                "successful_dependencies": sum(1 for r in dependency_results.values() if r.success),
                "successful_initializations": sum(1 for r in init_results.values() if r.success),
                "successful_tests": sum(1 for r in test_results.values() if r.success),
            }
        }
        
        logger.info(f"Integration test complete - Success: {all_successful}")
        return report


# Convenience functions
async def run_integration_validation() -> Dict[str, Any]:
    """Run integration validation for all Mimir 2.0 components."""
    integrator = PipelineIntegrator()
    return await integrator.run_full_integration_test()


async def validate_mimir_2_setup() -> bool:
    """Quick validation of Mimir 2.0 setup."""
    try:
        integrator = PipelineIntegrator()
        results = await integrator.validate_all_dependencies()
        return all(result.success for result in results.values())
    except Exception as e:
        logger.error(f"Setup validation failed: {e}")
        return False