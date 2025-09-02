"""
Abstract pipeline stage interface for modular pipeline architecture.

This module defines the base interface for all pipeline stages, enabling
a plugin architecture for future pipeline extensions while maintaining
consistent error handling and progress reporting.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Protocol, TYPE_CHECKING

from ..data.schemas import PipelineStage
from ..util.errors import MimirError

if TYPE_CHECKING:
    # Forward reference to avoid circular imports
    from .run import PipelineContext


class ProgressCallback(Protocol):
    """Type hint for progress callback functions."""
    
    def __call__(self, progress: int) -> None:
        """Report progress percentage (0-100)."""
        ...


class PipelineStageInterface(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Defines the standard interface for stage execution, error handling,
    and progress reporting. Each concrete stage implements the execute
    method with stage-specific logic while maintaining consistent
    behavior across the pipeline.
    """

    def __init__(self, stage_type: PipelineStage):
        """Initialize pipeline stage with type identifier."""
        self.stage_type = stage_type

    @abstractmethod
    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """
        Execute the pipeline stage.

        Args:
            context: Pipeline context containing all stage data
            progress_callback: Optional callback to report progress (0-100)

        Raises:
            MimirError: Structured error with context and recovery strategy
        """
        pass

    async def validate_prerequisites(self, context: "PipelineContext") -> None:
        """
        Validate that prerequisites for this stage are met.
        
        Override in concrete stages to check specific requirements.

        Args:
            context: Pipeline context to validate

        Raises:
            MimirError: If prerequisites are not met
        """
        pass

    async def cleanup(self, context: "PipelineContext") -> None:
        """
        Clean up resources after stage execution.
        
        Override in concrete stages to handle cleanup logic.

        Args:
            context: Pipeline context for cleanup
        """
        pass

    def get_metrics(self) -> dict[str, Any]:
        """
        Get stage-specific metrics.
        
        Override in concrete stages to provide detailed metrics.

        Returns:
            Dictionary of stage metrics
        """
        return {
            "stage_type": self.stage_type.value,
            "execution_count": getattr(self, "_execution_count", 0),
        }

    def get_stage_info(self) -> dict[str, Any]:
        """
        Get stage information for debugging and monitoring.

        Returns:
            Dictionary containing stage metadata
        """
        return {
            "stage_type": self.stage_type.value,
            "class_name": self.__class__.__name__,
            "capabilities": self._get_capabilities(),
        }

    def _get_capabilities(self) -> list[str]:
        """
        Get list of stage capabilities.
        
        Override in concrete stages to define specific capabilities.

        Returns:
            List of capability strings
        """
        return ["basic_execution"]

    def _update_progress(
        self, 
        progress: int, 
        callback: ProgressCallback | None = None
    ) -> None:
        """
        Helper method to update progress with validation.

        Args:
            progress: Progress percentage (0-100)
            callback: Optional progress callback
        """
        if callback is not None:
            # Validate progress range
            progress = max(0, min(100, progress))
            callback(progress)


class ConfigurablePipelineStage(PipelineStageInterface):
    """
    Base class for pipeline stages that require configuration.
    
    Extends the basic pipeline stage interface with configuration
    management capabilities.
    """

    def __init__(self, stage_type: PipelineStage, config: dict[str, Any] | None = None):
        """Initialize configurable pipeline stage."""
        super().__init__(stage_type)
        self.config = config or {}

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with fallback.

        Args:
            key: Configuration key
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        return self.config.get(key, default)

    def validate_config(self) -> None:
        """
        Validate stage configuration.
        
        Override in concrete stages to implement validation logic.

        Raises:
            MimirError: If configuration is invalid
        """
        pass

    def get_metrics(self) -> dict[str, Any]:
        """Get stage metrics including configuration info."""
        metrics = super().get_metrics()
        metrics.update({
            "configuration": {
                "has_config": bool(self.config),
                "config_keys": list(self.config.keys()),
            }
        })
        return metrics


class AsyncPipelineStage(PipelineStageInterface):
    """
    Base class for async-intensive pipeline stages.
    
    Provides additional utilities for stages that perform
    significant async operations like I/O or network requests.
    """

    def __init__(self, stage_type: PipelineStage, concurrency_limit: int = 4):
        """Initialize async pipeline stage with concurrency control."""
        super().__init__(stage_type)
        self.concurrency_limit = concurrency_limit
        self._semaphore = None

    async def _get_semaphore(self):
        """Get or create semaphore for concurrency control."""
        if self._semaphore is None:
            import asyncio
            self._semaphore = asyncio.Semaphore(self.concurrency_limit)
        return self._semaphore

    async def execute_with_concurrency_limit(
        self, 
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """
        Execute stage with concurrency limiting.

        Args:
            context: Pipeline context
            progress_callback: Optional progress callback
        """
        semaphore = await self._get_semaphore()
        async with semaphore:
            await self.execute(context, progress_callback)

    def _get_capabilities(self) -> list[str]:
        """Get async stage capabilities."""
        return ["basic_execution", "async_operations", "concurrency_control"]

    def get_metrics(self) -> dict[str, Any]:
        """Get async stage metrics."""
        metrics = super().get_metrics()
        metrics.update({
            "concurrency": {
                "limit": self.concurrency_limit,
                "has_semaphore": self._semaphore is not None,
            }
        })
        return metrics