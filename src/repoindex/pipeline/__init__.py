"""Pipeline orchestration and stage implementations."""

from .stage import PipelineStageInterface, ConfigurablePipelineStage, AsyncPipelineStage
from .stages import (
    AcquireStage,
    RepoMapperStage,
    SerenaStage,
    LeannStage,
    SnippetsStage,
    BundleStage,
)

# LLM Components  
from .llm_adapter import LLMAdapter, MockLLMAdapter, LLMRequest, LLMResponse, LLMError
from .llm_adapter_factory import LLMAdapterFactory, create_default_llm_adapter
from .ollama import OllamaAdapter

# RAPTOR Components (optional import for graceful degradation)
try:
    from .raptor_stage import RaptorStage
    from .raptor import RaptorProcessor, RaptorConfig
    from .raptor_structures import RaptorTree, RaptorNode
    RAPTOR_AVAILABLE = True
except ImportError:
    RAPTOR_AVAILABLE = False
    RaptorStage = None
    RaptorProcessor = None
    RaptorConfig = None
    RaptorTree = None
    RaptorNode = None

# Advanced Search Components (optional import for graceful degradation)
try:
    from .hyde import HyDETransformer
    from .code_embeddings import CodeEmbeddingAdapter
    from .reranking import CrossEncoderReranker, create_reranker
    from .enhanced_search import EnhancedSearchPipeline, create_enhanced_search_pipeline
    ADVANCED_SEARCH_AVAILABLE = True
except ImportError:
    ADVANCED_SEARCH_AVAILABLE = False
    HyDETransformer = None
    CodeEmbeddingAdapter = None
    CrossEncoderReranker = None
    create_reranker = None
    EnhancedSearchPipeline = None
    create_enhanced_search_pipeline = None

# Coordination Components
from .pipeline_coordinator import PipelineCoordinator, get_pipeline_coordinator
from .integration_helpers import run_integration_validation, validate_mimir_2_setup

__all__ = [
    # Pipeline stages
    "PipelineStageInterface",
    "ConfigurablePipelineStage", 
    "AsyncPipelineStage",
    "AcquireStage",
    "RepoMapperStage",
    "SerenaStage", 
    "LeannStage",
    "SnippetsStage",
    "BundleStage",
    
    # LLM components
    "LLMAdapter",
    "MockLLMAdapter", 
    "LLMRequest",
    "LLMResponse",
    "LLMError",
    "LLMAdapterFactory",
    "OllamaAdapter",
    "create_default_llm_adapter",
    
    # Coordination
    "PipelineCoordinator",
    "get_pipeline_coordinator", 
    "run_integration_validation",
    "validate_mimir_2_setup",
    
    # Constants
    "RAPTOR_AVAILABLE",
    "ADVANCED_SEARCH_AVAILABLE",
]

# Add RAPTOR components to __all__ if available
if RAPTOR_AVAILABLE:
    __all__.extend([
        "RaptorStage",
        "RaptorProcessor", 
        "RaptorConfig",
        "RaptorTree",
        "RaptorNode",
    ])

# Add advanced search components to __all__ if available
if ADVANCED_SEARCH_AVAILABLE:
    __all__.extend([
        "HyDETransformer",
        "CodeEmbeddingAdapter",
        "CrossEncoderReranker", 
        "create_reranker",
        "EnhancedSearchPipeline",
        "create_enhanced_search_pipeline",
    ])
