"""
Hybrid Indexing Pipeline for Mimir-Lens Integration.

This module implements a hybrid indexing pipeline that leverages Lens for 
high-performance bulk operations while preserving Mimir's deep code analysis 
capabilities. The pipeline orchestrates both systems for optimal performance
and comprehensive code understanding.

Key Features:
- Delegated heavy lifting to Lens for performance-critical operations
- Parallel processing of Lens and Mimir operations
- Smart result synthesis combining both systems' outputs
- Circuit breaker pattern for reliability
- Comprehensive error handling and fallback mechanisms
- Performance monitoring and metrics collection
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from pathlib import Path

from pydantic import BaseModel, Field
import aiofiles

from .lens_client import (
    LensIntegrationClient, 
    get_lens_client,
    LensResponse,
    LensHealthStatus,
    LensIntegrationError
)
from .stage import AsyncPipelineStage, ProgressCallback
from ..data.schemas import PipelineStage
from ..util.log import get_logger
from ..util.errors import MimirError

if TYPE_CHECKING:
    from .run import PipelineContext

logger = get_logger(__name__)


class HybridStrategy(Enum):
    """Strategies for hybrid Lens-Mimir operations."""
    LENS_FIRST = "lens_first"           # Try Lens first, fallback to Mimir
    MIMIR_FIRST = "mimir_first"         # Try Mimir first, fallback to Lens
    PARALLEL = "parallel"               # Run both in parallel, merge results
    LENS_ONLY = "lens_only"            # Use only Lens (fast mode)
    MIMIR_ONLY = "mimir_only"          # Use only Mimir (deep analysis mode)
    DELEGATED = "delegated"            # Delegate heavy lifting to Lens


class OperationType(Enum):
    """Types of operations that can be hybridized."""
    DISCOVERY = "discovery"
    INDEXING = "indexing"
    EMBEDDING = "embedding"
    SEARCH = "search"
    ANALYSIS = "analysis"
    BUNDLING = "bundling"


@dataclass
class PerformanceMetrics:
    """Performance metrics for hybrid operations."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    lens_time_ms: float = 0.0
    mimir_time_ms: float = 0.0
    parallel_time_ms: float = 0.0
    total_time_ms: float = 0.0
    items_processed: int = 0
    lens_success_rate: float = 0.0
    mimir_success_rate: float = 0.0
    fallback_triggered: bool = False
    strategy_used: HybridStrategy = HybridStrategy.PARALLEL
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.start_time and self.end_time:
            self.total_time_ms = (
                self.end_time - self.start_time
            ).total_seconds() * 1000


@dataclass
class HybridResult:
    """Result from hybrid Lens-Mimir operation."""
    success: bool
    data: Any = None
    lens_data: Any = None
    mimir_data: Any = None
    error: Optional[str] = None
    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    strategy_used: HybridStrategy = HybridStrategy.PARALLEL
    fallback_used: bool = False
    
    def finalize_metrics(self) -> None:
        """Finalize performance metrics."""
        self.metrics.finalize()


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, don't attempt
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for Lens operations."""
    failure_threshold: int = 5
    recovery_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    
    def record_success(self) -> None:
        """Record successful operation."""
        self.failure_count = 0
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.CLOSED
            logger.info("Circuit breaker recovered - state: CLOSED")
    
    def record_failure(self) -> None:
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if (self.failure_count >= self.failure_threshold and 
            self.state == CircuitBreakerState.CLOSED):
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_attempt(self) -> bool:
        """Check if operation can be attempted."""
        if self.state == CircuitBreakerState.CLOSED:
            return True
        
        if self.state == CircuitBreakerState.OPEN:
            if (self.last_failure_time and 
                datetime.utcnow() - self.last_failure_time > self.recovery_timeout):
                self.state = CircuitBreakerState.HALF_OPEN
                logger.info("Circuit breaker attempting recovery - state: HALF_OPEN")
                return True
            return False
        
        # HALF_OPEN - allow limited attempts
        return True


class HybridIndexingPipeline(AsyncPipelineStage):
    """
    Hybrid indexing pipeline that orchestrates Lens and Mimir operations.
    
    This pipeline implements several hybrid strategies:
    1. Delegated Heavy Lifting: Send bulk operations to Lens
    2. Parallel Processing: Run Lens and Mimir simultaneously
    3. Smart Fallbacks: Graceful degradation when services unavailable
    4. Result Synthesis: Combine outputs from both systems intelligently
    """
    
    def __init__(
        self,
        stage_type: PipelineStage,
        concurrency_limit: int = 8,
        lens_client: Optional[LensIntegrationClient] = None,
        default_strategy: HybridStrategy = HybridStrategy.PARALLEL,
        enable_circuit_breaker: bool = True
    ):
        """Initialize hybrid indexing pipeline."""
        super().__init__(stage_type, concurrency_limit)
        self.lens_client = lens_client or get_lens_client()
        self.default_strategy = default_strategy
        self.enable_circuit_breaker = enable_circuit_breaker
        
        # Circuit breaker for Lens operations
        self.circuit_breaker = CircuitBreaker() if enable_circuit_breaker else None
        
        # Performance tracking
        self.operation_metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.total_operations = 0
        self.successful_operations = 0
        
        # Thread pool for CPU-intensive operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4)
        
        logger.info(f"Initialized HybridIndexingPipeline with strategy: {default_strategy.value}")
    
    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """Execute hybrid indexing pipeline."""
        logger.info("Starting hybrid indexing pipeline execution")
        start_time = time.time()
        
        try:
            # Check Lens availability
            lens_available = await self._check_lens_availability()
            if not lens_available:
                logger.warning("Lens not available, falling back to Mimir-only mode")
                self.default_strategy = HybridStrategy.MIMIR_ONLY
            
            # Execute hybrid operations based on context
            await self._execute_hybrid_operations(context, progress_callback)
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"Hybrid pipeline completed in {execution_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Hybrid pipeline execution failed: {e}")
            raise MimirError(f"Hybrid indexing failed: {e}")
    
    async def _check_lens_availability(self) -> bool:
        """Check if Lens is available and healthy."""
        if self.circuit_breaker and not self.circuit_breaker.can_attempt():
            return False
        
        try:
            health_check = await self.lens_client.health_check()
            is_healthy = health_check.status in [
                LensHealthStatus.HEALTHY, 
                LensHealthStatus.DEGRADED
            ]
            
            if self.circuit_breaker:
                if is_healthy:
                    self.circuit_breaker.record_success()
                else:
                    self.circuit_breaker.record_failure()
            
            return is_healthy
            
        except Exception as e:
            logger.warning(f"Lens health check failed: {e}")
            if self.circuit_breaker:
                self.circuit_breaker.record_failure()
            return False
    
    async def _execute_hybrid_operations(
        self,
        context: "PipelineContext",
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """Execute hybrid operations based on pipeline context."""
        # This method will be extended in subsequent implementations
        # For now, establish the basic orchestration framework
        
        operations = [
            (OperationType.DISCOVERY, self._hybrid_discovery),
            (OperationType.INDEXING, self._hybrid_indexing),
            (OperationType.EMBEDDING, self._hybrid_embedding),
            (OperationType.BUNDLING, self._hybrid_bundling)
        ]
        
        total_ops = len(operations)
        completed_ops = 0
        
        for op_type, operation_func in operations:
            try:
                logger.info(f"Starting hybrid {op_type.value} operation")
                result = await operation_func(context)
                
                if result.success:
                    logger.info(f"Hybrid {op_type.value} completed successfully")
                else:
                    logger.warning(f"Hybrid {op_type.value} failed: {result.error}")
                
                completed_ops += 1
                if progress_callback:
                    progress = int((completed_ops / total_ops) * 100)
                    progress_callback(progress)
                    
            except Exception as e:
                logger.error(f"Hybrid {op_type.value} operation failed: {e}")
                # Continue with other operations even if one fails
                completed_ops += 1
                if progress_callback:
                    progress = int((completed_ops / total_ops) * 100)
                    progress_callback(progress)
    
    async def _hybrid_discovery(self, context: "PipelineContext") -> HybridResult:
        """Execute hybrid file discovery."""
        metrics = PerformanceMetrics()
        metrics.strategy_used = self.default_strategy
        
        try:
            if self.default_strategy == HybridStrategy.PARALLEL:
                return await self._parallel_discovery(context, metrics)
            elif self.default_strategy == HybridStrategy.LENS_FIRST:
                return await self._lens_first_discovery(context, metrics)
            elif self.default_strategy == HybridStrategy.DELEGATED:
                return await self._delegated_discovery(context, metrics)
            else:
                return await self._mimir_only_discovery(context, metrics)
                
        except Exception as e:
            logger.error(f"Hybrid discovery failed: {e}")
            return HybridResult(
                success=False,
                error=str(e),
                metrics=metrics,
                strategy_used=self.default_strategy
            )
    
    async def _parallel_discovery(
        self, 
        context: "PipelineContext", 
        metrics: PerformanceMetrics
    ) -> HybridResult:
        """Run Lens and Mimir discovery in parallel."""
        logger.info("Starting parallel discovery (Lens + Mimir)")
        
        # Create tasks for parallel execution
        tasks = []
        
        # Lens discovery task
        if await self._check_lens_availability():
            lens_task = asyncio.create_task(
                self._lens_discovery(context),
                name="lens_discovery"
            )
            tasks.append(("lens", lens_task))
        
        # Mimir discovery task (always available)
        mimir_task = asyncio.create_task(
            self._mimir_discovery(context),
            name="mimir_discovery"
        )
        tasks.append(("mimir", mimir_task))
        
        # Wait for all tasks with timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*[task for _, task in tasks], return_exceptions=True),
                timeout=30.0
            )
            
            # Process results
            lens_data = None
            mimir_data = None
            
            for i, (source, _) in enumerate(tasks):
                result = results[i]
                if isinstance(result, Exception):
                    logger.warning(f"{source} discovery failed: {result}")
                else:
                    if source == "lens":
                        lens_data = result
                        metrics.lens_success_rate = 1.0
                    else:
                        mimir_data = result
                        metrics.mimir_success_rate = 1.0
            
            # Synthesize results
            synthesized_data = await self._synthesize_discovery_results(
                lens_data, mimir_data
            )
            
            return HybridResult(
                success=True,
                data=synthesized_data,
                lens_data=lens_data,
                mimir_data=mimir_data,
                metrics=metrics,
                strategy_used=HybridStrategy.PARALLEL
            )
            
        except asyncio.TimeoutError:
            logger.warning("Parallel discovery timed out")
            return HybridResult(
                success=False,
                error="Discovery operation timed out",
                metrics=metrics,
                strategy_used=HybridStrategy.PARALLEL
            )
    
    async def _lens_discovery(self, context: "PipelineContext") -> Dict[str, Any]:
        """Perform discovery using Lens."""
        # This is a placeholder - will be implemented based on Lens API
        logger.info("Performing Lens-based file discovery")
        
        try:
            # Example Lens discovery call
            response = await self.lens_client.bulk_index(
                documents=[], # Will be populated with actual file data
                operation="discovery"
            )
            
            if response.success:
                return response.data
            else:
                raise LensIntegrationError(f"Lens discovery failed: {response.error}")
                
        except Exception as e:
            logger.error(f"Lens discovery error: {e}")
            raise
    
    async def _mimir_discovery(self, context: "PipelineContext") -> Dict[str, Any]:
        """Perform discovery using Mimir's native capabilities."""
        logger.info("Performing Mimir-based file discovery")
        
        # This would integrate with existing Mimir discovery logic
        # For now, return a placeholder structure
        return {
            "source": "mimir",
            "files_discovered": [],
            "analysis_depth": "deep",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def _synthesize_discovery_results(
        self,
        lens_data: Optional[Dict[str, Any]],
        mimir_data: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize discovery results from Lens and Mimir."""
        logger.info("Synthesizing discovery results from Lens and Mimir")
        
        synthesized = {
            "hybrid_discovery": True,
            "sources": [],
            "combined_files": set(),
            "analysis_capabilities": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if lens_data:
            synthesized["sources"].append("lens")
            synthesized["analysis_capabilities"].extend(["fast_indexing", "bulk_operations"])
            # Add Lens-discovered files
            if "files_discovered" in lens_data:
                synthesized["combined_files"].update(lens_data["files_discovered"])
        
        if mimir_data:
            synthesized["sources"].append("mimir")
            synthesized["analysis_capabilities"].extend(["deep_analysis", "ast_parsing"])
            # Add Mimir-discovered files
            if "files_discovered" in mimir_data:
                synthesized["combined_files"].update(mimir_data["files_discovered"])
        
        # Convert set to list for JSON serialization
        synthesized["combined_files"] = list(synthesized["combined_files"])
        synthesized["total_files"] = len(synthesized["combined_files"])
        
        return synthesized
    
    async def _lens_first_discovery(
        self, 
        context: "PipelineContext", 
        metrics: PerformanceMetrics
    ) -> HybridResult:
        """Try Lens first, fallback to Mimir."""
        logger.info("Starting Lens-first discovery with Mimir fallback")
        
        try:
            # Try Lens first
            lens_data = await self._lens_discovery(context)
            metrics.lens_success_rate = 1.0
            
            return HybridResult(
                success=True,
                data=lens_data,
                lens_data=lens_data,
                metrics=metrics,
                strategy_used=HybridStrategy.LENS_FIRST
            )
            
        except Exception as e:
            logger.warning(f"Lens discovery failed, falling back to Mimir: {e}")
            metrics.fallback_triggered = True
            
            try:
                # Fallback to Mimir
                mimir_data = await self._mimir_discovery(context)
                metrics.mimir_success_rate = 1.0
                
                return HybridResult(
                    success=True,
                    data=mimir_data,
                    mimir_data=mimir_data,
                    metrics=metrics,
                    strategy_used=HybridStrategy.LENS_FIRST,
                    fallback_used=True
                )
                
            except Exception as fallback_error:
                logger.error(f"Mimir fallback also failed: {fallback_error}")
                return HybridResult(
                    success=False,
                    error=f"Both Lens and Mimir failed: {e}, {fallback_error}",
                    metrics=metrics,
                    strategy_used=HybridStrategy.LENS_FIRST,
                    fallback_used=True
                )
    
    async def _delegated_discovery(
        self, 
        context: "PipelineContext", 
        metrics: PerformanceMetrics
    ) -> HybridResult:
        """Delegate heavy lifting to Lens, use Mimir for specialized analysis."""
        logger.info("Starting delegated discovery (Lens handles bulk, Mimir analyzes)")
        
        try:
            # Step 1: Use Lens for bulk file discovery and indexing
            lens_data = await self._lens_discovery(context)
            
            # Step 2: Use Mimir for deep analysis of critical files
            critical_files = self._identify_critical_files(lens_data)
            mimir_analysis = await self._analyze_critical_files_with_mimir(
                critical_files, context
            )
            
            # Step 3: Combine results
            combined_data = {
                **lens_data,
                "mimir_analysis": mimir_analysis,
                "strategy": "delegated_heavy_lifting"
            }
            
            metrics.lens_success_rate = 1.0
            metrics.mimir_success_rate = 1.0 if mimir_analysis else 0.0
            
            return HybridResult(
                success=True,
                data=combined_data,
                lens_data=lens_data,
                mimir_data=mimir_analysis,
                metrics=metrics,
                strategy_used=HybridStrategy.DELEGATED
            )
            
        except Exception as e:
            logger.error(f"Delegated discovery failed: {e}")
            return HybridResult(
                success=False,
                error=str(e),
                metrics=metrics,
                strategy_used=HybridStrategy.DELEGATED
            )
    
    def _identify_critical_files(self, lens_data: Dict[str, Any]) -> List[str]:
        """Identify files that need deep Mimir analysis."""
        # This would contain logic to identify files that benefit from Mimir's deep analysis
        # For example: configuration files, complex algorithms, etc.
        
        critical_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.rs', '.go'}
        critical_files = []
        
        if "files_discovered" in lens_data:
            for file_path in lens_data["files_discovered"]:
                path = Path(file_path)
                if path.suffix.lower() in critical_extensions:
                    critical_files.append(file_path)
        
        return critical_files[:50]  # Limit for performance
    
    async def _analyze_critical_files_with_mimir(
        self, 
        critical_files: List[str], 
        context: "PipelineContext"
    ) -> Dict[str, Any]:
        """Use Mimir to perform deep analysis on critical files."""
        logger.info(f"Performing deep Mimir analysis on {len(critical_files)} critical files")
        
        analysis_results = {
            "analyzed_files": len(critical_files),
            "file_analyses": {},
            "patterns_detected": [],
            "complexity_metrics": {}
        }
        
        # This would integrate with Mimir's existing analysis capabilities
        # For now, return a placeholder structure
        
        return analysis_results
    
    async def _mimir_only_discovery(
        self, 
        context: "PipelineContext", 
        metrics: PerformanceMetrics
    ) -> HybridResult:
        """Use only Mimir for discovery (fallback mode)."""
        logger.info("Running Mimir-only discovery")
        
        try:
            mimir_data = await self._mimir_discovery(context)
            metrics.mimir_success_rate = 1.0
            
            return HybridResult(
                success=True,
                data=mimir_data,
                mimir_data=mimir_data,
                metrics=metrics,
                strategy_used=HybridStrategy.MIMIR_ONLY
            )
            
        except Exception as e:
            logger.error(f"Mimir-only discovery failed: {e}")
            return HybridResult(
                success=False,
                error=str(e),
                metrics=metrics,
                strategy_used=HybridStrategy.MIMIR_ONLY
            )
    
    # Placeholder methods for other hybrid operations
    async def _hybrid_indexing(self, context: "PipelineContext") -> HybridResult:
        """Execute hybrid indexing operation."""
        logger.info("Hybrid indexing operation - placeholder")
        metrics = PerformanceMetrics()
        return HybridResult(success=True, data={}, metrics=metrics)
    
    async def _hybrid_embedding(self, context: "PipelineContext") -> HybridResult:
        """Execute hybrid embedding operation."""
        logger.info("Hybrid embedding operation - placeholder")
        metrics = PerformanceMetrics()
        return HybridResult(success=True, data={}, metrics=metrics)
    
    async def _hybrid_bundling(self, context: "PipelineContext") -> HybridResult:
        """Execute hybrid bundling operation."""
        logger.info("Hybrid bundling operation - placeholder")
        metrics = PerformanceMetrics()
        return HybridResult(success=True, data={}, metrics=metrics)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive hybrid pipeline metrics."""
        base_metrics = super().get_metrics()
        
        hybrid_metrics = {
            "hybrid_pipeline": {
                "total_operations": self.total_operations,
                "successful_operations": self.successful_operations,
                "success_rate": (
                    self.successful_operations / self.total_operations 
                    if self.total_operations > 0 else 0.0
                ),
                "default_strategy": self.default_strategy.value,
                "circuit_breaker_enabled": self.enable_circuit_breaker,
            }
        }
        
        if self.circuit_breaker:
            hybrid_metrics["circuit_breaker"] = {
                "state": self.circuit_breaker.state.value,
                "failure_count": self.circuit_breaker.failure_count,
                "can_attempt": self.circuit_breaker.can_attempt()
            }
        
        base_metrics.update(hybrid_metrics)
        return base_metrics
    
    def _get_capabilities(self) -> List[str]:
        """Get hybrid pipeline capabilities."""
        capabilities = super()._get_capabilities()
        capabilities.extend([
            "hybrid_orchestration",
            "lens_integration",
            "parallel_processing",
            "circuit_breaker",
            "performance_monitoring",
            "smart_fallbacks",
            "result_synthesis"
        ])
        return capabilities
    
    async def cleanup(self, context: "PipelineContext") -> None:
        """Clean up hybrid pipeline resources."""
        logger.info("Cleaning up hybrid pipeline resources")
        
        # Shutdown thread pool
        if hasattr(self, '_thread_pool'):
            self._thread_pool.shutdown(wait=True)
        
        # Close Lens client if needed
        if self.lens_client:
            # Lens client cleanup would go here
            pass
        
        await super().cleanup(context)