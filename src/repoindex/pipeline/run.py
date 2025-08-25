"""
Main pipeline orchestration for repository indexing.

Coordinates all six pipeline stages with comprehensive error handling, progress
tracking, performance monitoring, and concurrent execution where dependencies allow.
"""

import asyncio
import time
import uuid
from pathlib import Path
from typing import Any

from ..data.schemas import (
    IndexConfig,
    IndexManifest,
    IndexState,
    PipelineError,
    PipelineStage,
    PipelineStatus,
    RepoInfo,
)
from ..monitoring import (
    get_metrics_collector,
    get_trace_manager,
    pipeline_metrics,
)
from ..util.errors import (
    ErrorCategory,
    ErrorCollector,
    ErrorSeverity,
    ExternalToolError,
    FileSystemError,
    IntegrationError,
    MimirError,
    RecoveryStrategy,
    ValidationError,
    create_error_context,
    handle_external_tool_error,
)
from ..util.fs import (
    atomic_write_json,
    get_index_directory,
)
from ..util.gitio import discover_git_repository
from ..util.log import PipelineLogger, get_pipeline_logger
from ..util.logging_config import (
    get_logger,
    log_pipeline_complete,
    log_pipeline_error,
    log_pipeline_start,
    set_request_context,
)
from .bundle import BundleCreator
from .discover import FileDiscovery
from .leann import LEANNAdapter
from .repomapper import RepoMapperAdapter
from .serena import SerenaAdapter
from .snippets import SnippetExtractor
from .query_engine import IndexedRepository, QueryEngine


class PipelineContext:
    """Context object shared across pipeline stages."""

    def __init__(
        self,
        index_id: str,
        repo_info: RepoInfo,
        config: IndexConfig,
        work_dir: Path,
        logger: PipelineLogger,
    ):
        self.index_id = index_id
        self.repo_info = repo_info
        self.config = config
        self.work_dir = work_dir
        self.logger = logger
        self.start_time = time.time()

        # Stage artifacts
        self.tracked_files: list[str] = []
        self.repomap_data: Any | None = None
        self.serena_graph: Any | None = None
        self.vector_index: Any | None = None
        self.snippets: Any | None = None
        self.manifest: IndexManifest | None = None

        # Error handling
        self.errors: list[PipelineError] = []
        self.cancelled = False


class IndexingPipeline:
    """
    Main pipeline orchestrator for repository indexing.

    Manages the complete indexing workflow from git discovery through
    bundle creation, with support for concurrent execution and error recovery.
    
    Separated from querying concerns - use QueryEngine for search and ask operations.
    """

    def __init__(self, storage_dir: Path, query_engine: QueryEngine | None = None, concurrency_limits: dict[str, int] | None = None):
        """Initialize indexing pipeline."""
        self.storage_dir = storage_dir
        self.query_engine = query_engine
        self.concurrency_limits = concurrency_limits or {
            "io": 16,  # Increased I/O concurrency for better throughput
            "cpu": 4,  # Increased CPU concurrency for parallel processing
        }
        self.active_pipelines: dict[str, PipelineContext] = {}

        # Create semaphores for concurrency control
        self.io_semaphore = asyncio.Semaphore(self.concurrency_limits["io"])
        self.cpu_semaphore = asyncio.Semaphore(self.concurrency_limits["cpu"])

        # Initialize monitoring
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()
        self.start_time = time.time()

    async def start_indexing(
        self,
        repo_path: str,
        rev: str | None = None,
        language: str = "ts",
        index_opts: dict[str, Any] | None = None,
    ) -> str:
        """
        Start indexing a repository.

        Returns the index_id for tracking progress.
        """
        # Generate unique index ID
        index_id = str(uuid.uuid4())

        # Discover git repository
        git_repo = discover_git_repository(repo_path)
        repo_root = git_repo.get_repo_root()

        # Create repository info
        head_commit = git_repo.get_head_commit() if rev is None else rev
        is_dirty = git_repo.is_worktree_dirty()

        repo_info = RepoInfo(root=str(repo_root), rev=head_commit, worktree_dirty=is_dirty)

        # Create index configuration
        config = IndexConfig(**(index_opts or {}))

        # Set up working directory
        index_dir = get_index_directory(self.storage_dir, index_id)

        # Create logger
        logger = get_pipeline_logger(index_id, index_dir)

        # Create pipeline context
        context = PipelineContext(
            index_id=index_id, repo_info=repo_info, config=config, work_dir=index_dir, logger=logger
        )

        # Store context
        self.active_pipelines[index_id] = context

        # Update metrics
        self.metrics_collector.set_cache_size("active_pipelines", len(self.active_pipelines))

        # Create initial status
        status = PipelineStatus(
            index_id=index_id, state=IndexState.QUEUED, progress=0, message="Pipeline queued"
        )
        self._save_status(context, status)

        # Start pipeline execution
        asyncio.create_task(self._execute_pipeline(context))

        logger.log_info("Pipeline started", stage=None, repo_path=repo_path)

        return index_id

    async def _execute_pipeline(self, context: PipelineContext) -> None:
        """Execute the complete indexing pipeline with comprehensive error handling."""
        # Initialize enhanced logging
        enhanced_logger = get_logger("pipeline")
        set_request_context(pipeline_id=context.index_id)

        error_collector = ErrorCollector()

        # Start tracing
        self.trace_manager.create_trace_context(
            "pipeline_execution",
            pipeline_id=context.index_id,
            repo_root=context.repo_info.root,
            languages=context.config.languages,
        )

        # Record pipeline start metrics
        self.metrics_collector.record_pipeline_start("pipeline", str(context.config.languages))

        with enhanced_logger.performance_track("pipeline_execution"):
            async with self.trace_manager.trace_operation(
                "pipeline_execution",
                pipeline_id=context.index_id,
                repo_root=context.repo_info.root,
                languages=str(context.config.languages),
            ) as span:
                try:
                    log_pipeline_start(
                        context.index_id,
                        {
                            "repo_root": context.repo_info.root,
                            "revision": context.repo_info.rev,
                            "languages": context.config.languages,
                            "features": context.config.features.dict(),
                        },
                    )

                    # Add span attributes
                    span.set_attribute("pipeline.languages", str(context.config.languages))
                    span.set_attribute("pipeline.repo_root", context.repo_info.root)
                    span.set_attribute("pipeline.revision", context.repo_info.rev)

                    # Update status to running
                    status = PipelineStatus(
                        index_id=context.index_id,
                        state=IndexState.RUNNING,
                        progress=0,
                        message="Starting pipeline execution",
                    )
                    self._save_status(context, status)

                    # Execute stages in sequence with enhanced error handling
                    await self._execute_stage_safely(
                        context, "acquire", self._stage_acquire, error_collector
                    )
                    if context.cancelled:
                        return

                    await self._execute_stage_safely(
                        context, "repomapper", self._stage_repomapper, error_collector
                    )
                    if context.cancelled:
                        return

                    await self._execute_stage_safely(
                        context, "serena", self._stage_serena, error_collector
                    )
                    if context.cancelled:
                        return

                    await self._execute_stage_safely(
                        context, "leann", self._stage_leann, error_collector
                    )
                    if context.cancelled:
                        return

                    await self._execute_stage_safely(
                        context, "snippets", self._stage_snippets, error_collector
                    )
                    if context.cancelled:
                        return

                    await self._execute_stage_safely(
                        context, "bundle", self._stage_bundle, error_collector
                    )
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
                        message="Pipeline completed successfully",
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
                    }

                    # Record success metrics
                    self.metrics_collector.record_pipeline_success(
                        "pipeline",
                        duration_seconds,
                        str(context.config.languages),
                        len(context.tracked_files),
                    )

                    # Record business metrics
                    self.metrics_collector.record_repository_indexed(
                        str(context.config.languages), "success"
                    )

                    # Add success span attributes
                    span.set_attribute("pipeline.files_processed", len(context.tracked_files))
                    span.set_attribute("pipeline.duration_seconds", duration_seconds)
                    span.set_attribute("pipeline.success", True)

                    # Register with query engine if available
                    if self.query_engine:
                        indexed_repo = IndexedRepository(
                            index_id=context.index_id,
                            repo_root=context.repo_info.root,
                            rev=context.repo_info.rev,
                            serena_graph=context.serena_graph,
                            vector_index=context.vector_index,
                            repomap_data=context.repomap_data,
                            snippets=context.snippets,
                            manifest=context.manifest,
                        )
                        self.query_engine.register_indexed_repository(indexed_repo)
                        
                        context.logger.log_info(
                            "Repository registered with query engine", 
                            index_id=context.index_id
                        )

                    log_pipeline_complete(context.index_id, duration_ms, stats)

                    context.logger.log_info(
                        "Pipeline completed successfully", duration_ms=duration_ms
                    )

                except Exception as e:
                    # Enhanced error handling and logging
                    duration_ms = (time.time() - context.start_time) * 1000

                    # Convert to structured error if needed
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="pipeline",
                            operation="pipeline_execution",
                            parameters={
                                "index_id": context.index_id,
                                "repo_root": context.repo_info.root,
                                "stage_progress": self._get_current_stage_progress(context),
                            },
                        )
                        e = MimirError(
                            message=f"Pipeline execution failed: {str(e)}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.LOGIC,
                            recovery_strategy=RecoveryStrategy.ESCALATE,
                            context=error_context,
                            cause=e,
                        )

                    # Record error metrics
                    error_type = type(e).__name__
                    severity = (
                        "high"
                        if isinstance(e, MimirError) and e.severity == ErrorSeverity.HIGH
                        else "medium"
                    )
                    self.metrics_collector.record_pipeline_error(
                        "pipeline", error_type, severity, str(context.config.languages)
                    )

                    # Record failed repository indexing
                    self.metrics_collector.record_repository_indexed(
                        str(context.config.languages), "error"
                    )

                    # Add error span attributes
                    span.set_attribute("pipeline.success", False)
                    span.set_attribute("pipeline.error_type", error_type)
                    span.record_exception(e)

                    # Log structured error
                    log_pipeline_error(context.index_id, e)

                    context.logger.log_error("Pipeline failed", error=e, duration_ms=duration_ms)

                    status = PipelineStatus(
                        index_id=context.index_id,
                        state=IndexState.FAILED,
                        progress=0,
                        message=f"Pipeline failed: {str(e)}",
                    )
                    status.mark_failed(str(e))
                    self._save_status(context, status)

                    # Mark context as failed for cleanup logic
                    context._pipeline_failed = True

                    # Re-raise for upstream handling
                    raise

                finally:
                    # Clean up with error handling
                    try:
                        # Keep completed pipelines for ask functionality, only remove failed ones
                        if context.index_id in self.active_pipelines:
                            # Check if pipeline failed by looking at the context or if this is in the exception handler
                            # If we're in the exception handler (after raise), remove the failed pipeline
                            # Otherwise, keep successful pipelines for querying
                            if hasattr(context, "_pipeline_failed") and context._pipeline_failed:
                                del self.active_pipelines[context.index_id]
                                
                                # Also unregister from query engine if failed
                                if self.query_engine:
                                    self.query_engine.unregister_indexed_repository(context.index_id)
                            # Successful pipelines remain available for querying

                        # Update metrics
                        self.metrics_collector.set_cache_size(
                            "active_pipelines", len(self.active_pipelines)
                        )

                        context.logger.close()
                    except Exception as cleanup_error:
                        enhanced_logger.operation_error(
                            "pipeline_cleanup", cleanup_error, pipeline_id=context.index_id
                        )

    @pipeline_metrics("acquire")
    async def _stage_acquire(self, context: PipelineContext) -> None:
        """Stage 1: Acquire and discover files with enhanced error handling."""
        stage = PipelineStage.ACQUIRE
        enhanced_logger = get_logger("pipeline.acquire")

        async with self.trace_manager.trace_pipeline_stage(
            "acquire",
            context.index_id,
            repo_root=context.repo_info.root,
            languages=str(context.config.languages),
        ) as span:
            with enhanced_logger.performance_track("file_discovery", include_system_metrics=True):
                context.logger.log_stage_start(stage, "Discovering repository files")

                try:
                    async with self.io_semaphore:
                        discovery = FileDiscovery(context.repo_info.root)
                        context.tracked_files = await discovery.discover_files(
                            extensions=context.config.languages, excludes=context.config.excludes
                        )

                    if not context.tracked_files:
                        raise ValidationError(
                            "No trackable files found in repository",
                            context=create_error_context(
                                component="file_discovery",
                                operation="discover_files",
                                parameters={
                                    "extensions": context.config.languages,
                                    "excludes": context.config.excludes,
                                    "repo_root": context.repo_info.root,
                                },
                            ),
                            suggestions=[
                                "Check if repository contains files with specified extensions",
                                "Verify exclude patterns are not too restrictive",
                                "Ensure repository is properly initialized",
                            ],
                        )

                    # Add span attributes
                    span.set_attribute("stage.files_discovered", len(context.tracked_files))
                    span.set_attribute("stage.extensions", str(context.config.languages))

                    enhanced_logger.operation_success(
                        "file_discovery",
                        file_count=len(context.tracked_files),
                        extensions=context.config.languages,
                    )

                    context.logger.log_info(
                        f"Discovered {len(context.tracked_files)} files",
                        stage=stage,
                        file_count=len(context.tracked_files),
                    )

                    # Update progress
                    self._update_stage_progress(context, stage, 100)

                except Exception as e:
                    # Convert to structured error if needed
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="file_discovery",
                            operation="discover_files",
                            parameters={
                                "repo_root": context.repo_info.root,
                                "extensions": context.config.languages,
                            },
                        )
                        e = FileSystemError(
                            path=context.repo_info.root,
                            operation="file_discovery",
                            message=str(e),
                            context=error_context,
                        )

                    enhanced_logger.operation_error("file_discovery", e)
                    context.logger.log_stage_error(stage, e)
                    raise

    @pipeline_metrics("repomapper")
    async def _stage_repomapper(self, context: PipelineContext) -> None:
        """Stage 2: RepoMapper analysis with enhanced error handling."""
        stage = PipelineStage.REPOMAPPER
        enhanced_logger = get_logger("pipeline.repomapper")

        async with self.trace_manager.trace_pipeline_stage(
            "repomapper", context.index_id, files_count=len(context.tracked_files)
        ) as span:
            with enhanced_logger.performance_track(
                "repository_analysis", include_system_metrics=True
            ):
                context.logger.log_stage_start(stage, "Analyzing repository structure")

                try:
                    async with self.cpu_semaphore:
                        repomapper = RepoMapperAdapter()
                        context.repomap_data = await repomapper.analyze_repository(
                            repo_root=Path(context.repo_info.root),
                            files=context.tracked_files,
                            work_dir=context.work_dir,
                            progress_callback=lambda p: self._update_stage_progress(
                                context, stage, p
                            ),
                        )

                    if not context.repomap_data:
                        raise ExternalToolError(
                            tool="repomapper",
                            message="Repository analysis produced no results",
                            context=create_error_context(
                                component="repomapper",
                                operation="analyze_repository",
                                parameters={
                                    "file_count": len(context.tracked_files),
                                    "repo_root": str(context.repo_info.root),
                                },
                            ),
                            suggestions=[
                                "Check if repository contains analyzable code structure",
                                "Verify RepoMapper tool is properly configured",
                                "Ensure files are accessible and readable",
                            ],
                        )

                    # Add span attributes
                    span.set_attribute("stage.files_analyzed", len(context.tracked_files))
                    span.set_attribute("stage.structure_detected", True)

                    enhanced_logger.operation_success(
                        "repository_analysis",
                        files_analyzed=len(context.tracked_files),
                        structure_detected=True,
                    )

                except Exception as e:
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="repomapper",
                            operation="analyze_repository",
                            parameters={
                                "repo_root": context.repo_info.root,
                                "file_count": len(context.tracked_files),
                            },
                        )
                        e = handle_external_tool_error(
                            tool="repomapper",
                            command=["repomapper", "analyze"],
                            exit_code=1,
                            stdout="",
                            stderr=str(e),
                            context=error_context,
                        )

                    enhanced_logger.operation_error("repository_analysis", e)
                    context.logger.log_stage_error(stage, e)
                    raise

    @pipeline_metrics("serena")
    async def _stage_serena(self, context: PipelineContext) -> None:
        """Stage 3: Serena TypeScript analysis with enhanced error handling."""
        stage = PipelineStage.SERENA
        enhanced_logger = get_logger("pipeline.serena")

        async with self.trace_manager.trace_pipeline_stage(
            "serena", context.index_id, files_count=len(context.tracked_files)
        ) as span:
            with enhanced_logger.performance_track("symbol_analysis", include_system_metrics=True):
                context.logger.log_stage_start(stage, "Analyzing TypeScript symbols")

                try:
                    async with self.cpu_semaphore:
                        serena = SerenaAdapter()
                        context.serena_graph = await serena.analyze_project(
                            project_root=Path(context.repo_info.root),
                            files=context.tracked_files,
                            work_dir=context.work_dir,
                            config=context.config,
                            progress_callback=lambda p: self._update_stage_progress(
                                context, stage, p
                            ),
                        )

                    if not context.serena_graph:
                        raise ExternalToolError(
                            tool="serena",
                            message="Symbol analysis produced no results",
                            context=create_error_context(
                                component="serena",
                                operation="analyze_project",
                                parameters={
                                    "file_count": len(context.tracked_files),
                                    "project_root": str(context.repo_info.root),
                                },
                            ),
                            suggestions=[
                                "Check if project contains TypeScript/JavaScript files",
                                "Verify Serena analyzer is properly configured",
                                "Ensure project has valid syntax and structure",
                            ],
                        )

                    # Record symbol extraction metrics
                    symbols_found = getattr(context.serena_graph, "symbol_count", 0)
                    if symbols_found > 0:
                        self.metrics_collector.record_symbols_extracted(
                            str(context.config.languages), "typescript", symbols_found
                        )

                    # Add span attributes
                    span.set_attribute("stage.files_analyzed", len(context.tracked_files))
                    span.set_attribute("stage.symbols_found", symbols_found)

                    enhanced_logger.operation_success(
                        "symbol_analysis",
                        files_analyzed=len(context.tracked_files),
                        symbols_found=symbols_found,
                    )

                except Exception as e:
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="serena",
                            operation="analyze_project",
                            parameters={
                                "project_root": context.repo_info.root,
                                "file_count": len(context.tracked_files),
                            },
                        )
                        e = handle_external_tool_error(
                            tool="serena",
                            command=["serena", "analyze"],
                            exit_code=1,
                            stdout="",
                            stderr=str(e),
                            context=error_context,
                        )

                    enhanced_logger.operation_error("symbol_analysis", e)
                    context.logger.log_stage_error(stage, e)
                    raise

    @pipeline_metrics("leann")
    async def _stage_leann(self, context: PipelineContext) -> None:
        """Stage 4: LEANN vector embedding with enhanced error handling."""
        stage = PipelineStage.LEANN
        enhanced_logger = get_logger("pipeline.leann")

        async with self.trace_manager.trace_pipeline_stage(
            "leann",
            context.index_id,
            files_count=len(context.tracked_files),
            vector_enabled=context.config.features.vector,
        ) as span:
            with enhanced_logger.performance_track("vector_embedding", include_system_metrics=True):
                context.logger.log_stage_start(stage, "Building vector embeddings")

                try:
                    if not context.config.features.vector:
                        enhanced_logger.info("Vector search disabled, skipping LEANN stage")
                        context.logger.log_info(
                            "Vector search disabled, skipping LEANN", stage=stage
                        )
                        span.set_attribute("stage.skipped", True)
                        self._update_stage_progress(context, stage, 100)
                        return

                    async with self.cpu_semaphore:
                        leann = LEANNAdapter()

                        # Get ordered files with fallback
                        ordered_files = context.tracked_files
                        if context.repomap_data and hasattr(
                            context.repomap_data, "get_ordered_files"
                        ):
                            try:
                                ordered_files = context.repomap_data.get_ordered_files()
                            except Exception:
                                enhanced_logger.warning(
                                    "Failed to get ordered files from RepoMap, using original order"
                                )

                        context.vector_index = await leann.build_index(
                            repo_root=Path(context.repo_info.root),
                            files=context.tracked_files,
                            repomap_order=ordered_files,
                            work_dir=context.work_dir,
                            config=context.config,
                            progress_callback=lambda p: self._update_stage_progress(
                                context, stage, p
                            ),
                        )

                    if not context.vector_index:
                        raise ExternalToolError(
                            tool="leann",
                            message="Vector embedding produced no results",
                            context=create_error_context(
                                component="leann",
                                operation="build_index",
                                parameters={
                                    "file_count": len(context.tracked_files),
                                    "repo_root": str(context.repo_info.root),
                                },
                            ),
                            suggestions=[
                                "Check if files contain embedable content",
                                "Verify LEANN model is accessible",
                                "Ensure sufficient memory for embedding generation",
                            ],
                        )

                    # Record embeddings metrics
                    chunks_created = getattr(context.vector_index, "total_chunks", 0)
                    if chunks_created > 0:
                        self.metrics_collector.record_embeddings_created(
                            "leann",
                            "code_chunk",
                            chunks_created,  # model name
                        )

                    # Add span attributes
                    span.set_attribute("stage.files_embedded", len(context.tracked_files))
                    span.set_attribute("stage.chunks_created", chunks_created)
                    span.set_attribute(
                        "stage.embedding_dimension", getattr(context.vector_index, "dimension", 0)
                    )

                    enhanced_logger.operation_success(
                        "vector_embedding",
                        files_embedded=len(context.tracked_files),
                        chunks_created=chunks_created,
                        embedding_dimension=getattr(context.vector_index, "dimension", 0),
                    )

                except Exception as e:
                    if not isinstance(e, MimirError):
                        error_context = create_error_context(
                            component="leann",
                            operation="build_index",
                            parameters={
                                "repo_root": context.repo_info.root,
                                "file_count": len(context.tracked_files),
                                "vector_enabled": context.config.features.vector,
                            },
                        )
                        structured_error = handle_external_tool_error(
                            tool="leann",
                            command=["leann", "embed"],
                            exit_code=1,
                            stdout="",
                            stderr=str(e),
                            context=error_context,
                        )
                        enhanced_logger.operation_error("vector_embedding", structured_error)
                        context.logger.log_stage_error(stage, structured_error)
                        raise structured_error
                    else:
                        enhanced_logger.operation_error("vector_embedding", e)
                        context.logger.log_stage_error(stage, e)
                        raise

    async def _stage_snippets(self, context: PipelineContext) -> None:
        """Stage 5: Extract code snippets with enhanced error handling."""
        stage = PipelineStage.SNIPPETS
        enhanced_logger = get_logger("pipeline.snippets")

        with enhanced_logger.performance_track("snippet_extraction", include_system_metrics=True):
            context.logger.log_stage_start(stage, "Extracting code snippets")

            try:
                async with self.io_semaphore:
                    extractor = SnippetExtractor()
                    context.snippets = await extractor.extract_snippets(
                        repo_root=Path(context.repo_info.root),
                        serena_graph=context.serena_graph,
                        work_dir=context.work_dir,
                        context_lines=context.config.context_lines,
                        progress_callback=lambda p: self._update_stage_progress(context, stage, p),
                    )

                    if not context.snippets:
                        raise IntegrationError(
                            source="snippet_extractor",
                            target="serena_graph",
                            message="Snippet extraction produced no results",
                            context=create_error_context(
                                component="snippet_extractor",
                                operation="extract_snippets",
                                parameters={
                                    "context_lines": context.config.context_lines,
                                    "has_serena_graph": context.serena_graph is not None,
                                },
                            ),
                            suggestions=[
                                "Check if Serena graph contains extractable symbols",
                                "Verify source files are accessible",
                                "Ensure context_lines setting is reasonable",
                            ],
                        )

                    enhanced_logger.operation_success(
                        "snippet_extraction",
                        snippets_extracted=getattr(context.snippets, "count", 0),
                        context_lines=context.config.context_lines,
                    )

            except Exception as e:
                if not isinstance(e, MimirError):
                    error_context = create_error_context(
                        component="snippet_extractor",
                        operation="extract_snippets",
                        parameters={
                            "repo_root": context.repo_info.root,
                            "context_lines": context.config.context_lines,
                            "has_serena_graph": context.serena_graph is not None,
                        },
                    )
                    e = MimirError(
                        message=f"Snippet extraction failed: {str(e)}",
                        severity=ErrorSeverity.MEDIUM,
                        category=ErrorCategory.LOGIC,
                        recovery_strategy=RecoveryStrategy.FALLBACK,
                        context=error_context,
                        cause=e,
                    )

                enhanced_logger.operation_error("snippet_extraction", e)
                context.logger.log_stage_error(stage, e)
                raise

    async def _stage_bundle(self, context: PipelineContext) -> None:
        """Stage 6: Create final bundle with enhanced error handling."""
        stage = PipelineStage.BUNDLE
        enhanced_logger = get_logger("pipeline.bundle")

        with enhanced_logger.performance_track("bundle_creation", include_system_metrics=True):
            context.logger.log_stage_start(stage, "Creating artifact bundle")

            try:
                async with self.io_semaphore:
                    bundler = BundleCreator()
                    context.manifest = await bundler.create_bundle(
                        context=context,
                        progress_callback=lambda p: self._update_stage_progress(context, stage, p),
                    )

                    if not context.manifest:
                        raise IntegrationError(
                            source="bundle_creator",
                            target="pipeline_context",
                            message="Bundle creation produced no manifest",
                            context=create_error_context(
                                component="bundle_creator",
                                operation="create_bundle",
                                parameters={
                                    "work_dir": str(context.work_dir),
                                    "has_tracked_files": len(context.tracked_files) > 0,
                                    "has_repomap": context.repomap_data is not None,
                                    "has_serena": context.serena_graph is not None,
                                    "has_vectors": context.vector_index is not None,
                                    "has_snippets": context.snippets is not None,
                                },
                            ),
                            suggestions=[
                                "Verify all pipeline stages completed successfully",
                                "Check work directory permissions",
                                "Ensure sufficient disk space for bundle creation",
                            ],
                        )

                    # Save final manifest with error handling
                    manifest_path = context.work_dir / "manifest.json"
                    try:
                        atomic_write_json(manifest_path, context.manifest.dict())
                    except Exception as manifest_error:
                        raise FileSystemError(
                            path=manifest_path,
                            operation="write_manifest",
                            message=f"Failed to save manifest: {str(manifest_error)}",
                            context=create_error_context(
                                component="bundle_creator", operation="save_manifest"
                            ),
                        )

                    enhanced_logger.operation_success(
                        "bundle_creation",
                        manifest_saved=True,
                        work_dir=str(context.work_dir),
                        components_bundled={
                            "files": len(context.tracked_files),
                            "repomap": context.repomap_data is not None,
                            "serena": context.serena_graph is not None,
                            "vectors": context.vector_index is not None,
                            "snippets": context.snippets is not None,
                        },
                    )

            except Exception as e:
                if not isinstance(e, MimirError):
                    error_context = create_error_context(
                        component="bundle_creator",
                        operation="create_bundle",
                        parameters={
                            "work_dir": str(context.work_dir),
                            "context_state": {
                                "tracked_files": len(context.tracked_files),
                                "has_repomap": context.repomap_data is not None,
                                "has_serena": context.serena_graph is not None,
                                "has_vectors": context.vector_index is not None,
                                "has_snippets": context.snippets is not None,
                            },
                        },
                    )
                    e = MimirError(
                        message=f"Bundle creation failed: {str(e)}",
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.FILESYSTEM,
                        recovery_strategy=RecoveryStrategy.RETRY,
                        context=error_context,
                        cause=e,
                    )

                enhanced_logger.operation_error("bundle_creation", e)
                context.logger.log_stage_error(stage, e)
                raise

    def _update_stage_progress(
        self, context: PipelineContext, stage: PipelineStage, stage_progress: int
    ) -> None:
        """Update pipeline progress for a specific stage."""
        # Calculate overall progress based on stage weights
        stage_weights = {
            PipelineStage.ACQUIRE: 5,
            PipelineStage.REPOMAPPER: 20,
            PipelineStage.SERENA: 25,
            PipelineStage.LEANN: 30,
            PipelineStage.SNIPPETS: 15,
            PipelineStage.BUNDLE: 5,
        }

        # Calculate base progress for completed stages
        stage_order = list(PipelineStage)
        current_stage_index = stage_order.index(stage)

        base_progress = sum(stage_weights[s] for s in stage_order[:current_stage_index])

        # Add current stage progress
        current_stage_progress = (stage_progress / 100.0) * stage_weights[stage]
        overall_progress = int(base_progress + current_stage_progress)

        # Update status
        status = PipelineStatus(
            index_id=context.index_id,
            state=IndexState.RUNNING,
            stage=stage,
            progress=overall_progress,
            message=f"{stage.value} stage: {stage_progress}%",
        )
        self._save_status(context, status)

        # Log progress
        context.logger.log_stage_progress(stage, stage_progress)

    def _save_status(self, context: PipelineContext, status: PipelineStatus) -> None:
        """Save pipeline status to file with error handling."""
        try:
            status_path = context.work_dir / "status.json"
            atomic_write_json(status_path, status.model_dump())
        except Exception as e:
            # Log but don't fail pipeline for status save errors
            enhanced_logger = get_logger("pipeline.status")
            enhanced_logger.operation_error(
                "save_status", e, status_path=str(status_path), pipeline_id=context.index_id
            )

    async def cancel(self, index_id: str) -> bool:
        """Cancel an active pipeline."""
        context = self.active_pipelines.get(index_id)
        if not context:
            return False

        context.cancelled = True
        context.logger.log_info("Pipeline cancellation requested")

        # Update status
        status = PipelineStatus(
            index_id=index_id,
            state=IndexState.CANCELLED,
            progress=0,
            message="Pipeline cancelled by user",
        )
        self._save_status(context, status)

        return True

    def get_query_engine(self) -> QueryEngine | None:
        """Get the associated query engine if available."""
        return self.query_engine

    def set_query_engine(self, query_engine: QueryEngine) -> None:
        """Set the query engine for this pipeline."""
        self.query_engine = query_engine


    async def _execute_stage_safely(
        self, context: PipelineContext, stage_name: str, stage_func, error_collector: ErrorCollector
    ) -> None:
        """Execute a pipeline stage with error collection and recovery."""
        try:
            await stage_func(context)
        except MimirError as e:
            # Collect structured errors
            error_collector.add_error(e)

            # Decide on recovery strategy
            if e.recovery_strategy == RecoveryStrategy.ABORT:
                raise
            elif e.recovery_strategy == RecoveryStrategy.SKIP:
                enhanced_logger = get_logger(f"pipeline.{stage_name}")
                enhanced_logger.warning(
                    f"Skipping {stage_name} stage due to recoverable error", error=e.to_dict()
                )
            elif e.recovery_strategy == RecoveryStrategy.FALLBACK:
                # Implement fallback logic if needed
                enhanced_logger = get_logger(f"pipeline.{stage_name}")
                enhanced_logger.warning(f"Using fallback for {stage_name} stage", error=e.to_dict())
        except Exception as e:
            # Convert unknown exceptions to structured errors
            error_context = create_error_context(
                component=f"pipeline.{stage_name}",
                operation=stage_name,
                parameters={"index_id": context.index_id},
            )
            structured_error = MimirError(
                message=f"Unexpected error in {stage_name} stage: {str(e)}",
                severity=ErrorSeverity.HIGH,
                category=ErrorCategory.LOGIC,
                recovery_strategy=RecoveryStrategy.ABORT,
                context=error_context,
                cause=e,
            )
            error_collector.add_error(structured_error)
            raise structured_error

    def _get_current_stage_progress(self, context: PipelineContext) -> dict[str, Any]:
        """Get current pipeline stage progress for error context."""
        return {
            "tracked_files": len(context.tracked_files) if context.tracked_files else 0,
            "has_repomap": context.repomap_data is not None,
            "has_serena_graph": context.serena_graph is not None,
            "has_vector_index": context.vector_index is not None,
            "has_snippets": context.snippets is not None,
            "has_manifest": context.manifest is not None,
        }


# Alias for backward compatibility with tests
PipelineRunner = IndexingPipeline
