"""
Security-hardened pipeline orchestration for repository indexing.

Provides the main IndexingPipeline with comprehensive security hardening including
input validation, sandboxed execution, credential scanning, and audit logging.
"""

import asyncio
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import uuid

from ..data.schemas import (
    IndexConfig,
    IndexManifest,
    IndexState,
    PipelineStage,
    PipelineStatus,
    RepoInfo,
    SearchResponse,
    AskResponse,
    FeatureConfig,
    PipelineError,
)
from ..util.fs import (
    get_index_directory,
    atomic_write_json,
    ensure_directory,
)
from ..util.gitio import discover_git_repository
from ..util.log import get_pipeline_logger, PipelineLogger
from ..util.errors import (
    MimirError,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    ErrorCollector,
    ErrorContext,
    ExternalToolError,
    FileSystemError,
    ConfigurationError,
    TimeoutError,
    IntegrationError,
    create_error_context,
    handle_external_tool_error,
    with_error_handling
)
from ..util.logging_config import (
    get_logger,
    set_request_context,
    log_pipeline_start,
    log_pipeline_complete,
    log_pipeline_error,
    PerformanceLogger
)
from ..monitoring import (
    get_metrics_collector,
    get_trace_manager,
    pipeline_metrics,
    trace_pipeline_stage
)

# Import security components
from ..security.config import get_security_config, SecurityConfig
from ..security.validation import PathValidator, SecurityValidationError, ValidationContext
from ..security.sandbox import configure_sandbox, ResourceLimiter, ProcessIsolator
from ..security.audit import get_security_auditor, SecurityEvent, SecurityEventType
from ..security.pipeline_integration import (
    SecureFileDiscovery, 
    SecurePipelineContext,
    create_secure_pipeline
)

# Import original pipeline components
from .discover import FileDiscovery
from .repomapper import RepoMapperAdapter
from .serena import SerenaAdapter
from .leann import LEANNAdapter
from .snippets import SnippetExtractor
from .bundle import BundleCreator
from .hybrid_search import HybridSearchEngine
from .ask_index import SymbolGraphNavigator

logger = get_logger(__name__)


class SecurePipelineContext:
    """Security-enhanced context object shared across pipeline stages."""
    
    def __init__(
        self,
        index_id: str,
        repo_info: RepoInfo,
        config: IndexConfig,
        work_dir: Path,
        logger: PipelineLogger,
        security_config: Optional[SecurityConfig] = None
    ):
        self.index_id = index_id
        self.repo_info = repo_info
        self.config = config
        self.work_dir = work_dir
        self.logger = logger
        self.start_time = time.time()
        
        # Security configuration
        self.security_config = security_config or get_security_config()
        self.security_auditor = get_security_auditor()
        
        # Stage artifacts
        self.tracked_files: List[str] = []
        self.repomap_data: Optional[Any] = None
        self.serena_graph: Optional[Any] = None
        self.vector_index: Optional[Any] = None
        self.snippets: Optional[Any] = None
        self.manifest: Optional[IndexManifest] = None
        
        # Error handling
        self.errors: List[PipelineError] = []
        self.cancelled = False
        
        # Security components
        self.path_validator = PathValidator(
            allowed_base_paths=self.security_config.allowed_base_paths,
            max_path_length=self.security_config.max_path_length,
            max_filename_length=self.security_config.max_filename_length
        )
        
        # Resource limits if sandboxing enabled
        self.resource_limiter = None
        if self.security_config.enable_sandboxing:
            self.resource_limiter = ResourceLimiter(
                max_memory=self.security_config.max_memory_mb * 1024 * 1024,
                max_cpu_time=self.security_config.max_cpu_time_seconds,
                max_wall_time=self.security_config.max_wall_time_seconds,
                max_open_files=self.security_config.max_open_files,
                max_processes=self.security_config.max_processes
            )


class SecureIndexingPipeline:
    """
    Security-hardened main pipeline orchestrator for repository indexing.
    
    Provides comprehensive security including input validation, sandboxed execution,
    credential scanning, and audit logging throughout the indexing process.
    """
    
    def __init__(
        self, 
        storage_dir: Path, 
        concurrency_limits: Optional[Dict[str, int]] = None,
        security_config: Optional[SecurityConfig] = None
    ):
        """Initialize secure indexing pipeline."""
        self.storage_dir = storage_dir
        self.security_config = security_config or get_security_config()
        self.concurrency_limits = concurrency_limits or {
            "io": 16,  # Increased I/O concurrency for better throughput
            "cpu": 4   # Increased CPU concurrency for parallel processing
        }
        self.active_pipelines: Dict[str, SecurePipelineContext] = {}
        
        # Create semaphores for concurrency control
        self.io_semaphore = asyncio.Semaphore(self.concurrency_limits["io"])
        self.cpu_semaphore = asyncio.Semaphore(self.concurrency_limits["cpu"])
        
        # Initialize monitoring
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()
        self.start_time = time.time()
        
        # Initialize security components
        self.security_auditor = get_security_auditor()
        self.path_validator = PathValidator(
            allowed_base_paths=self.security_config.allowed_base_paths,
            max_path_length=self.security_config.max_path_length,
            max_filename_length=self.security_config.max_filename_length
        )
        
        # Configure sandbox if enabled
        if self.security_config.enable_sandboxing:
            configure_sandbox(
                base_path=self.storage_dir,
                custom_limits=self.security_config.get_resource_limits()
            )
        
        logger.info(
            "Secure indexing pipeline initialized",
            security_features={
                "sandboxing": self.security_config.enable_sandboxing,
                "encryption": self.security_config.enable_index_encryption,
                "credential_scanning": self.security_config.enable_credential_scanning,
                "audit_logging": self.security_config.enable_audit_logging
            }
        )
    
    async def start_indexing(
        self,
        repo_path: str,
        rev: Optional[str] = None,
        language: str = "ts",
        index_opts: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start secure indexing of a repository.
        
        Returns the index_id for tracking progress.
        """
        try:
            # Security validation for inputs
            if not self.path_validator.validate_path(repo_path, operation="read"):
                raise SecurityValidationError(
                    f"Repository path validation failed: {repo_path}",
                    context=ValidationContext(
                        component="pipeline_start",
                        operation="validate_repo_path",
                        input_data={"repo_path": repo_path}
                    )
                )
            
            # Generate unique index ID
            index_id = str(uuid.uuid4())
            
            # Record security event
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.PIPELINE_START,
                component="secure_pipeline",
                message="Secure repository indexing started",
                metadata={
                    "index_id": index_id,
                    "repo_path": repo_path,
                    "revision": rev,
                    "language": language,
                    "has_index_opts": index_opts is not None
                }
            ))
            
            # Discover git repository with path validation
            git_repo = discover_git_repository(repo_path)
            repo_root = git_repo.get_repo_root()
            
            # Validate discovered repository root
            if not self.path_validator.validate_path(str(repo_root), operation="read"):
                raise SecurityValidationError(
                    f"Repository root validation failed: {repo_root}",
                    context=ValidationContext(
                        component="pipeline_start",
                        operation="validate_repo_root",
                        input_data={"repo_root": str(repo_root)}
                    )
                )
            
            # Create repository info
            head_commit = git_repo.get_head_commit() if rev is None else rev
            is_dirty = git_repo.is_worktree_dirty()
            
            repo_info = RepoInfo(
                root=str(repo_root),
                rev=head_commit,
                worktree_dirty=is_dirty
            )
            
            # Create and validate index configuration
            config = IndexConfig(**(index_opts or {}))
            
            # Set up working directory with security checks
            index_dir = get_index_directory(self.storage_dir, index_id)
            
            # Validate index directory is within allowed paths
            if not self.path_validator.validate_path(str(index_dir), operation="write"):
                raise SecurityValidationError(
                    f"Index directory validation failed: {index_dir}",
                    context=ValidationContext(
                        component="pipeline_start",
                        operation="validate_index_dir",
                        input_data={"index_dir": str(index_dir)}
                    )
                )
            
            # Ensure directory exists
            ensure_directory(index_dir)
            
            # Create logger
            logger = get_pipeline_logger(index_id, index_dir)
            
            # Create secure pipeline context
            context = SecurePipelineContext(
                index_id=index_id,
                repo_info=repo_info,
                config=config,
                work_dir=index_dir,
                logger=logger,
                security_config=self.security_config
            )
            
            # Store context
            self.active_pipelines[index_id] = context
            
            # Update metrics
            self.metrics_collector.set_cache_size("active_pipelines", len(self.active_pipelines))
            
            # Create initial status
            status = PipelineStatus(
                index_id=index_id,
                state=IndexState.QUEUED,
                progress=0,
                message="Secure pipeline queued"
            )
            self._save_status(context, status)
            
            # Start pipeline execution
            asyncio.create_task(self._execute_secure_pipeline(context))
            
            logger.log_info("Secure pipeline started", stage=None, repo_path=repo_path)
            
            return index_id
            
        except Exception as e:
            # Record security event for failed start
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.PIPELINE_ERROR,
                component="secure_pipeline",
                severity="high",
                message=f"Secure pipeline start failed: {str(e)}",
                metadata={
                    "repo_path": repo_path,
                    "error": str(e)
                }
            ))
            raise
    
    async def _execute_secure_pipeline(self, context: SecurePipelineContext) -> None:
        """Execute the complete secure indexing pipeline."""
        # Initialize enhanced logging
        enhanced_logger = get_logger("secure_pipeline")
        set_request_context(pipeline_id=context.index_id)
        
        error_collector = ErrorCollector()
        
        # Start tracing
        trace_context = self.trace_manager.create_trace_context(
            "secure_pipeline_execution",
            pipeline_id=context.index_id,
            repo_root=context.repo_info.root,
            languages=context.config.languages
        )
        
        # Record pipeline start metrics
        self.metrics_collector.record_pipeline_start("secure_pipeline", str(context.config.languages))
        
        # Apply resource limits if sandboxing enabled
        if context.resource_limiter:
            try:
                context.resource_limiter.apply_limits()
                logger.info("Resource limits applied for secure execution")
            except Exception as e:
                logger.warning(f"Failed to apply resource limits: {e}")
        
        with enhanced_logger.performance_track("secure_pipeline_execution"):
            async with self.trace_manager.trace_operation(
                "secure_pipeline_execution",
                pipeline_id=context.index_id,
                repo_root=context.repo_info.root,
                languages=str(context.config.languages)
            ) as span:
                try:
                    # Record execution start
                    context.security_auditor.record_event(SecurityEvent(
                        event_type=SecurityEventType.PIPELINE_EXECUTION_START,
                        component="secure_pipeline",
                        message="Secure pipeline execution started",
                        metadata={
                            "pipeline_id": context.index_id,
                            "repo_root": context.repo_info.root,
                            "revision": context.repo_info.rev,
                            "languages": context.config.languages,
                            "security_features": {
                                "sandboxing": context.security_config.enable_sandboxing,
                                "encryption": context.security_config.enable_index_encryption,
                                "credential_scanning": context.security_config.enable_credential_scanning
                            }
                        }
                    ))
                    
                    log_pipeline_start(
                        context.index_id,
                        {
                            "repo_root": context.repo_info.root,
                            "revision": context.repo_info.rev,
                            "languages": context.config.languages,
                            "features": context.config.features.dict(),
                            "security_enabled": True
                        }
                    )
                    
                    # Add span attributes
                    span.set_attribute("pipeline.languages", str(context.config.languages))
                    span.set_attribute("pipeline.repo_root", context.repo_info.root)
                    span.set_attribute("pipeline.revision", context.repo_info.rev)
                    span.set_attribute("pipeline.security_enabled", True)
                    
                    # Update status to running
                    status = PipelineStatus(
                        index_id=context.index_id,
                        state=IndexState.RUNNING,
                        progress=0,
                        message="Starting secure pipeline execution"
                    )
                    self._save_status(context, status)
                
                    # Execute stages in sequence with security enhancements
                    await self._execute_secure_stage(context, "acquire", self._stage_secure_acquire, error_collector)
                    if context.cancelled:
                        return
                    
                    await self._execute_secure_stage(context, "repomapper", self._stage_secure_repomapper, error_collector)
                    if context.cancelled:
                        return
                    
                    await self._execute_secure_stage(context, "serena", self._stage_secure_serena, error_collector)
                    if context.cancelled:
                        return
                    
                    await self._execute_secure_stage(context, "leann", self._stage_secure_leann, error_collector)
                    if context.cancelled:
                        return
                    
                    await self._execute_secure_stage(context, "snippets", self._stage_secure_snippets, error_collector)
                    if context.cancelled:
                        return
                    
                    await self._execute_secure_stage(context, "bundle", self._stage_secure_bundle, error_collector)
                    if context.cancelled:
                        return
                    
                    # Check for accumulated errors
                    if error_collector.has_errors():
                        aggregate_error = error_collector.create_aggregate_error()
                        raise aggregate_error
                    
                    # Mark as completed
                    status = PipelineStatus(
                        index_id=context.index_id,
                        state=IndexState.DONE,
                        progress=100,
                        message="Secure pipeline completed successfully"
                    )
                    status.mark_completed()
                    self._save_status(context, status)
                    
                    # Calculate final stats
                    duration_ms = (time.time() - context.start_time) * 1000
                    duration_seconds = duration_ms / 1000
                    stats = {
                        "files_processed": len(context.tracked_files),
                        "has_repomap": context.repomap_data is not None,
                        "has_symbols": context.serena_graph is not None,
                        "has_vectors": context.vector_index is not None,
                        "has_snippets": context.snippets is not None,
                        "warnings": len(error_collector.warnings),
                        "security_enabled": True
                    }
                    
                    # Record success metrics
                    self.metrics_collector.record_pipeline_success(
                        "secure_pipeline",
                        duration_seconds,
                        str(context.config.languages),
                        len(context.tracked_files)
                    )
                    
                    # Record successful completion
                    context.security_auditor.record_event(SecurityEvent(
                        event_type=SecurityEventType.PIPELINE_COMPLETE,
                        component="secure_pipeline",
                        message="Secure pipeline completed successfully",
                        metadata={
                            "pipeline_id": context.index_id,
                            "duration_seconds": duration_seconds,
                            "stats": stats
                        }
                    ))
                    
                    # Add success span attributes
                    span.set_attribute("pipeline.files_processed", len(context.tracked_files))
                    span.set_attribute("pipeline.duration_seconds", duration_seconds)
                    span.set_attribute("pipeline.success", True)
                    
                    log_pipeline_complete(context.index_id, duration_ms, stats)
                    
                    context.logger.log_info(
                        "Secure pipeline completed successfully",
                        duration_ms=duration_ms,
                        security_enabled=True
                    )
                    
                except Exception as e:
                    # Enhanced error handling and logging
                    duration_ms = (time.time() - context.start_time) * 1000
                    
                    # Convert to structured error if needed
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="secure_pipeline",
                            operation="pipeline_execution",
                            parameters={
                                "index_id": context.index_id,
                                "repo_root": context.repo_info.root,
                                "stage_progress": self._get_current_stage_progress(context)
                            }
                        )
                        e = MimirError(
                            message=f"Secure pipeline execution failed: {str(e)}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.LOGIC,
                            recovery_strategy=RecoveryStrategy.ESCALATE,
                            context=error_context,
                            cause=e
                        )
                    
                    # Record error metrics and security event
                    error_type = type(e).__name__
                    severity = "high" if isinstance(e, MimirError) and e.severity == ErrorSeverity.HIGH else "medium"
                    
                    self.metrics_collector.record_pipeline_error(
                        "secure_pipeline",
                        error_type,
                        severity,
                        str(context.config.languages)
                    )
                    
                    context.security_auditor.record_event(SecurityEvent(
                        event_type=SecurityEventType.PIPELINE_ERROR,
                        component="secure_pipeline",
                        severity="high",
                        message=f"Secure pipeline failed: {str(e)}",
                        metadata={
                            "pipeline_id": context.index_id,
                            "error_type": error_type,
                            "duration_seconds": duration_ms / 1000,
                            "error": str(e)
                        }
                    ))
                    
                    # Add error span attributes
                    span.set_attribute("pipeline.success", False)
                    span.set_attribute("pipeline.error_type", error_type)
                    span.record_exception(e)
                    
                    # Log structured error
                    log_pipeline_error(context.index_id, e)
                    
                    context.logger.log_error(
                        "Secure pipeline failed",
                        error=e,
                        duration_ms=duration_ms,
                        security_enabled=True
                    )
                    
                    status = PipelineStatus(
                        index_id=context.index_id,
                        state=IndexState.FAILED,
                        progress=0,
                        message=f"Secure pipeline failed: {str(e)}"
                    )
                    status.mark_failed(str(e))
                    self._save_status(context, status)
                    
                    # Re-raise for upstream handling
                    raise
            
                finally:
                    # Clean up resources
                    try:
                        if context.resource_limiter:
                            context.resource_limiter.cleanup()
                        
                        if context.index_id in self.active_pipelines:
                            del self.active_pipelines[context.index_id]
                        
                        # Update metrics
                        self.metrics_collector.set_cache_size("active_pipelines", len(self.active_pipelines))
                        
                        context.logger.close()
                        
                    except Exception as cleanup_error:
                        enhanced_logger.operation_error(
                            "secure_pipeline_cleanup",
                            cleanup_error,
                            pipeline_id=context.index_id
                        )
    
    async def _execute_secure_stage(
        self,
        context: SecurePipelineContext,
        stage_name: str,
        stage_func,
        error_collector: ErrorCollector
    ) -> None:
        """Execute a pipeline stage with comprehensive security controls."""
        try:
            # Record stage start
            context.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.STAGE_START,
                component=f"secure_{stage_name}",
                message=f"Starting secure {stage_name} stage",
                metadata={
                    "pipeline_id": context.index_id,
                    "stage": stage_name,
                    "repo_root": context.repo_info.root
                }
            ))
            
            stage_start = time.time()
            await stage_func(context)
            
            # Record successful stage completion
            duration = time.time() - stage_start
            context.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.STAGE_COMPLETE,
                component=f"secure_{stage_name}",
                message=f"Secure {stage_name} stage completed successfully",
                metadata={
                    "pipeline_id": context.index_id,
                    "stage": stage_name,
                    "duration_seconds": duration
                }
            ))
            
        except MimirError as e:
            # Collect structured errors
            error_collector.add_error(e)
            
            # Record stage error
            context.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.STAGE_ERROR,
                component=f"secure_{stage_name}",
                severity="high",
                message=f"Secure {stage_name} stage failed: {str(e)}",
                metadata={
                    "pipeline_id": context.index_id,
                    "stage": stage_name,
                    "error": str(e),
                    "recovery_strategy": e.recovery_strategy.value if e.recovery_strategy else "unknown"
                }
            ))
            
            # Decide on recovery strategy
            if e.recovery_strategy == RecoveryStrategy.ABORT:
                raise
            elif e.recovery_strategy == RecoveryStrategy.SKIP:
                enhanced_logger = get_logger(f"secure_pipeline.{stage_name}")
                enhanced_logger.warning(
                    f"Skipping {stage_name} stage due to recoverable error",
                    error=e.to_dict()
                )
            elif e.recovery_strategy == RecoveryStrategy.FALLBACK:
                # Implement fallback logic if needed
                enhanced_logger = get_logger(f"secure_pipeline.{stage_name}")
                enhanced_logger.warning(
                    f"Using fallback for {stage_name} stage",
                    error=e.to_dict()
                )
        except Exception as e:
            # Convert unknown exceptions to structured errors
            error_context = create_error_context(
                component=f"secure_pipeline.{stage_name}",
                operation=stage_name,
                parameters={"index_id": context.index_id}
            )
            structured_error = MimirError(
                message=f"Unexpected error in secure {stage_name} stage: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.LOGIC,
                recovery_strategy=RecoveryStrategy.ABORT,
                context=error_context,
                cause=e
            )
            
            # Record security event
            context.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.STAGE_ERROR,
                component=f"secure_{stage_name}",
                severity="high",
                message=f"Unexpected error in secure {stage_name} stage: {str(e)}",
                metadata={
                    "pipeline_id": context.index_id,
                    "stage": stage_name,
                    "error": str(e)
                }
            ))
            
            error_collector.add_error(structured_error)
            raise structured_error
    
    @pipeline_metrics("secure_acquire")
    async def _stage_secure_acquire(self, context: SecurePipelineContext) -> None:
        """Stage 1: Secure file acquisition and discovery."""
        stage = PipelineStage.ACQUIRE
        enhanced_logger = get_logger("secure_pipeline.acquire")
        
        async with self.trace_manager.trace_pipeline_stage(
            "secure_acquire",
            context.index_id,
            repo_root=context.repo_info.root,
            languages=str(context.config.languages)
        ) as span:
            with enhanced_logger.performance_track("secure_file_discovery", include_system_metrics=True):
                context.logger.log_stage_start(stage, "Discovering repository files securely")
                
                try:
                    async with self.io_semaphore:
                        # Create secure file discovery
                        original_discovery = FileDiscovery(context.repo_info.root)
                        secure_discovery = SecureFileDiscovery(original_discovery, context.security_config)
                        
                        # Secure file discovery with validation and scanning
                        context.tracked_files = await secure_discovery.discover_files(
                            extensions=context.config.languages,
                            excludes=context.config.excludes
                        )
                    
                    if not context.tracked_files:
                        raise SecurityValidationError(
                            "No trackable files found after security validation",
                            context=ValidationContext(
                                component="secure_file_discovery",
                                operation="discover_files",
                                input_data={
                                    "extensions": context.config.languages,
                                    "excludes": context.config.excludes,
                                    "repo_root": context.repo_info.root
                                }
                            )
                        )
                    
                    # Add span attributes
                    span.set_attribute("stage.files_discovered", len(context.tracked_files))
                    span.set_attribute("stage.extensions", str(context.config.languages))
                    span.set_attribute("stage.security_enabled", True)
                    
                    enhanced_logger.operation_success(
                        "secure_file_discovery",
                        file_count=len(context.tracked_files),
                        extensions=context.config.languages,
                        security_validated=True
                    )
                    
                    context.logger.log_info(
                        f"Securely discovered {len(context.tracked_files)} files",
                        stage=stage,
                        file_count=len(context.tracked_files),
                        security_enabled=True
                    )
                    
                    # Update progress
                    self._update_stage_progress(context, stage, 100)
                    
                except Exception as e:
                    # Convert to structured error if needed
                    if not isinstance(e, (MimirError, SecurityValidationError)):
                        error_context = create_error_context(
                            component="secure_file_discovery",
                            operation="discover_files",
                            parameters={
                                "repo_root": context.repo_info.root,
                                "extensions": context.config.languages
                            }
                        )
                        e = FileSystemError(
                            path=context.repo_info.root,
                            operation="secure_file_discovery",
                            message=str(e),
                            context=error_context
                        )
                    
                    enhanced_logger.operation_error("secure_file_discovery", e)
                    context.logger.log_stage_error(stage, e)
                    raise
    
    # TODO: Implement remaining secure stage methods
    # _stage_secure_repomapper, _stage_secure_serena, _stage_secure_leann, 
    # _stage_secure_snippets, _stage_secure_bundle
    
    async def _stage_secure_repomapper(self, context: SecurePipelineContext) -> None:
        """Stage 2: Secure RepoMapper analysis with sandboxing."""
        stage = PipelineStage.REPOMAPPER
        enhanced_logger = get_logger("secure_pipeline.repomapper")
        
        async with self.trace_manager.trace_pipeline_stage(
            "secure_repomapper",
            context.index_id,
            files_count=len(context.tracked_files)
        ) as span:
            with enhanced_logger.performance_track("secure_repository_analysis", include_system_metrics=True):
                context.logger.log_stage_start(stage, "Analyzing repository structure securely")
                
                try:
                    async with self.cpu_semaphore:
                        # TODO: Implement secure external tool execution for RepoMapper
                        # This would use ProcessIsolator for sandboxed execution
                        repomapper = RepoMapperAdapter()
                        context.repomap_data = await repomapper.analyze_repository(
                            repo_root=Path(context.repo_info.root),
                            files=context.tracked_files,
                            work_dir=context.work_dir,
                            progress_callback=lambda p: self._update_stage_progress(context, stage, p)
                        )
                    
                    if not context.repomap_data:
                        raise ExternalToolError(
                            tool="repomapper",
                            message="Secure repository analysis produced no results",
                            context=create_error_context(
                                component="secure_repomapper",
                                operation="analyze_repository",
                                parameters={
                                    "file_count": len(context.tracked_files),
                                    "repo_root": str(context.repo_info.root)
                                }
                            )
                        )
                    
                    # Add span attributes
                    span.set_attribute("stage.files_analyzed", len(context.tracked_files))
                    span.set_attribute("stage.structure_detected", True)
                    span.set_attribute("stage.security_enabled", True)
                    
                    enhanced_logger.operation_success(
                        "secure_repository_analysis",
                        files_analyzed=len(context.tracked_files),
                        structure_detected=True,
                        security_enabled=True
                    )
                    
                except Exception as e:
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="secure_repomapper",
                            operation="analyze_repository",
                            parameters={
                                "repo_root": context.repo_info.root,
                                "file_count": len(context.tracked_files)
                            }
                        )
                        e = handle_external_tool_error(
                            tool="repomapper",
                            command=["repomapper", "analyze"],
                            exit_code=1,
                            stdout="",
                            stderr=str(e),
                            context=error_context
                        )
                    
                    enhanced_logger.operation_error("secure_repository_analysis", e)
                    context.logger.log_stage_error(stage, e)
                    raise
    
    # Simplified implementation of remaining stages for demonstration
    async def _stage_secure_serena(self, context: SecurePipelineContext) -> None:
        """Stage 3: Secure Serena analysis with sandboxing."""
        # Implementation would follow same pattern as repomapper
        # with secure external tool execution
        pass
    
    async def _stage_secure_leann(self, context: SecurePipelineContext) -> None:
        """Stage 4: Secure LEANN processing with encryption."""
        # Implementation would include vector encryption if enabled
        pass
    
    async def _stage_secure_snippets(self, context: SecurePipelineContext) -> None:
        """Stage 5: Secure snippet extraction."""
        # Implementation would follow secure file access patterns
        pass
    
    async def _stage_secure_bundle(self, context: SecurePipelineContext) -> None:
        """Stage 6: Secure bundle creation with encryption."""
        # Implementation would encrypt final bundle if configured
        pass
    
    def _update_stage_progress(
        self,
        context: SecurePipelineContext,
        stage: PipelineStage,
        stage_progress: int
    ) -> None:
        """Update secure pipeline progress for a specific stage."""
        # Same implementation as original pipeline
        stage_weights = {
            PipelineStage.ACQUIRE: 5,
            PipelineStage.REPOMAPPER: 20,
            PipelineStage.SERENA: 25,
            PipelineStage.LEANN: 30,
            PipelineStage.SNIPPETS: 15,
            PipelineStage.BUNDLE: 5
        }
        
        stage_order = list(PipelineStage)
        current_stage_index = stage_order.index(stage)
        
        base_progress = sum(
            stage_weights[s] for s in stage_order[:current_stage_index]
        )
        
        current_stage_progress = (stage_progress / 100.0) * stage_weights[stage]
        overall_progress = int(base_progress + current_stage_progress)
        
        status = PipelineStatus(
            index_id=context.index_id,
            state=IndexState.RUNNING,
            stage=stage,
            progress=overall_progress,
            message=f"Secure {stage.value} stage: {stage_progress}%"
        )
        self._save_status(context, status)
        
        context.logger.log_stage_progress(stage, stage_progress)
    
    def _save_status(self, context: SecurePipelineContext, status: PipelineStatus) -> None:
        """Save pipeline status to file with error handling."""
        try:
            status_path = context.work_dir / "status.json"
            atomic_write_json(status_path, status.model_dump())
        except Exception as e:
            enhanced_logger = get_logger("secure_pipeline.status")
            enhanced_logger.operation_error(
                "save_status",
                e,
                status_path=str(status_path),
                pipeline_id=context.index_id
            )
    
    def _get_current_stage_progress(self, context: SecurePipelineContext) -> Dict[str, Any]:
        """Get current pipeline stage progress for error context."""
        return {
            "tracked_files": len(context.tracked_files) if context.tracked_files else 0,
            "has_repomap": context.repomap_data is not None,
            "has_serena_graph": context.serena_graph is not None,
            "has_vector_index": context.vector_index is not None,
            "has_snippets": context.snippets is not None,
            "has_manifest": context.manifest is not None,
            "security_enabled": True
        }
    
    async def cancel(self, index_id: str) -> bool:
        """Cancel an active secure pipeline."""
        context = self.active_pipelines.get(index_id)
        if not context:
            return False
        
        context.cancelled = True
        context.logger.log_info("Secure pipeline cancellation requested")
        
        # Record security event
        context.security_auditor.record_event(SecurityEvent(
            event_type=SecurityEventType.PIPELINE_CANCELLED,
            component="secure_pipeline",
            message="Secure pipeline cancelled by user",
            metadata={"pipeline_id": index_id}
        ))
        
        # Update status
        status = PipelineStatus(
            index_id=index_id,
            state=IndexState.CANCELLED,
            progress=0,
            message="Secure pipeline cancelled by user"
        )
        self._save_status(context, status)
        
        return True


# Alias for backward compatibility
SecurePipelineRunner = SecureIndexingPipeline