"""
Concrete implementations of pipeline stages.

This module contains the actual implementation of each pipeline stage
using the abstract interface defined in stage.py. Each stage focuses
on a specific aspect of the indexing pipeline.
"""

from pathlib import Path
from typing import Any, TYPE_CHECKING

from .discover import FileDiscovery
from .repomapper import RepoMapperAdapter
from .serena import SerenaAdapter
from .leann import LEANNAdapter
from .snippets import SnippetExtractor
from .bundle import BundleCreator
from .stage import PipelineStageInterface, AsyncPipelineStage, ConfigurablePipelineStage

from ..data.schemas import PipelineStage
from ..util.errors import (
    ValidationError, 
    FileSystemError, 
    ExternalToolError, 
    IntegrationError, 
    MimirError,
    ErrorSeverity,
    ErrorCategory,
    RecoveryStrategy,
    create_error_context,
    handle_external_tool_error
)
from ..util.log import get_logger
from ..monitoring.metrics import get_metrics_collector

if TYPE_CHECKING:
    from .run import PipelineContext


class AcquireStage(AsyncPipelineStage):
    """
    Stage 1: Discover and acquire repository files.
    
    Identifies all trackable files in the repository based on
    configured extensions and exclude patterns.
    """

    def __init__(self):
        super().__init__(PipelineStage.ACQUIRE, concurrency_limit=16)

    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback = None
    ) -> None:
        """Execute file discovery and acquisition."""
        enhanced_logger = get_logger("pipeline.acquire")
        
        with enhanced_logger.performance_track("file_discovery", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Discovering repository files")

            try:
                discovery = FileDiscovery(context.repo_info.root)
                context.tracked_files = await discovery.discover_files(
                    extensions=context.config.languages, 
                    excludes=context.config.excludes
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

                enhanced_logger.operation_success(
                    "file_discovery",
                    file_count=len(context.tracked_files),
                    extensions=context.config.languages,
                )

                context.logger.log_info(
                    f"Discovered {len(context.tracked_files)} files",
                    stage=self.stage_type,
                    file_count=len(context.tracked_files),
                )

                if progress_callback:
                    progress_callback(100)

            except Exception as e:
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
                context.logger.log_stage_error(self.stage_type, e)
                raise

    def _get_capabilities(self) -> list[str]:
        return ["basic_execution", "async_operations", "file_discovery", "pattern_matching"]


class RepoMapperStage(AsyncPipelineStage):
    """
    Stage 2: Analyze repository structure with RepoMapper.
    
    Performs static analysis to understand repository organization,
    dependencies, and structural relationships.
    """

    def __init__(self):
        super().__init__(PipelineStage.REPOMAPPER, concurrency_limit=4)

    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback = None
    ) -> None:
        """Execute repository structure analysis."""
        enhanced_logger = get_logger("pipeline.repomapper")
        
        with enhanced_logger.performance_track("repository_analysis", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Analyzing repository structure")

            try:
                repomapper = RepoMapperAdapter()
                context.repomap_data = await repomapper.analyze_repository(
                    repo_root=Path(context.repo_info.root),
                    files=context.tracked_files,
                    work_dir=context.work_dir,
                    progress_callback=progress_callback,
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
                context.logger.log_stage_error(self.stage_type, e)
                raise

    def _get_capabilities(self) -> list[str]:
        return ["basic_execution", "async_operations", "code_analysis", "dependency_mapping"]


class SerenaStage(AsyncPipelineStage):
    """
    Stage 3: TypeScript/JavaScript symbol analysis with Serena.
    
    Extracts symbols, type information, and cross-references
    from TypeScript and JavaScript files.
    """

    def __init__(self):
        super().__init__(PipelineStage.SERENA, concurrency_limit=4)

    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback = None
    ) -> None:
        """Execute symbol analysis."""
        enhanced_logger = get_logger("pipeline.serena")
        metrics_collector = get_metrics_collector()
        
        with enhanced_logger.performance_track("symbol_analysis", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Analyzing TypeScript symbols")

            try:
                serena = SerenaAdapter()
                context.serena_graph = await serena.analyze_project(
                    project_root=Path(context.repo_info.root),
                    files=context.tracked_files,
                    work_dir=context.work_dir,
                    config=context.config,
                    progress_callback=progress_callback,
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
                    metrics_collector.record_symbols_extracted(
                        str(context.config.languages), "typescript", symbols_found
                    )

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
                context.logger.log_stage_error(self.stage_type, e)
                raise

    def _get_capabilities(self) -> list[str]:
        return ["basic_execution", "async_operations", "symbol_analysis", "typescript_support"]


class LeannStage(ConfigurablePipelineStage, AsyncPipelineStage):
    """
    Stage 4: Vector embedding generation with LEANN.
    
    Creates vector embeddings for code chunks to enable
    semantic similarity search.
    """

    def __init__(self):
        ConfigurablePipelineStage.__init__(self, PipelineStage.LEANN)
        AsyncPipelineStage.__init__(self, PipelineStage.LEANN, concurrency_limit=4)

    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback = None
    ) -> None:
        """Execute vector embedding generation."""
        enhanced_logger = get_logger("pipeline.leann")
        metrics_collector = get_metrics_collector()
        
        with enhanced_logger.performance_track("vector_embedding", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Building vector embeddings")

            try:
                if not context.config.features.vector:
                    enhanced_logger.info("Vector search disabled, skipping LEANN stage")
                    context.logger.log_info("Vector search disabled, skipping LEANN", stage=self.stage_type)
                    if progress_callback:
                        progress_callback(100)
                    return

                leann = LEANNAdapter()

                # Get ordered files with fallback
                ordered_files = context.tracked_files
                if context.repomap_data and hasattr(context.repomap_data, "get_ordered_files"):
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
                    progress_callback=progress_callback,
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
                    metrics_collector.record_embeddings_created("leann", "code_chunk", chunks_created)

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
                    e = handle_external_tool_error(
                        tool="leann",
                        command=["leann", "embed"],
                        exit_code=1,
                        stdout="",
                        stderr=str(e),
                        context=error_context,
                    )

                enhanced_logger.operation_error("vector_embedding", e)
                context.logger.log_stage_error(self.stage_type, e)
                raise

    def _get_capabilities(self) -> list[str]:
        return ["basic_execution", "async_operations", "vector_embedding", "semantic_search"]


class SnippetsStage(AsyncPipelineStage):
    """
    Stage 5: Extract code snippets with context.
    
    Extracts meaningful code snippets from the analyzed symbols
    with appropriate context lines for search results.
    """

    def __init__(self):
        super().__init__(PipelineStage.SNIPPETS, concurrency_limit=16)

    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback = None
    ) -> None:
        """Execute snippet extraction."""
        enhanced_logger = get_logger("pipeline.snippets")
        
        with enhanced_logger.performance_track("snippet_extraction", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Extracting code snippets")

            try:
                extractor = SnippetExtractor()
                context.snippets = await extractor.extract_snippets(
                    repo_root=Path(context.repo_info.root),
                    serena_graph=context.serena_graph,
                    work_dir=context.work_dir,
                    context_lines=context.config.context_lines,
                    progress_callback=progress_callback,
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
                context.logger.log_stage_error(self.stage_type, e)
                raise

    def _get_capabilities(self) -> list[str]:
        return ["basic_execution", "async_operations", "snippet_extraction", "context_preservation"]


class BundleStage(AsyncPipelineStage):
    """
    Stage 6: Create final artifact bundle.
    
    Packages all generated artifacts into a compressed bundle
    with manifest for easy distribution and consumption.
    """

    def __init__(self):
        super().__init__(PipelineStage.BUNDLE, concurrency_limit=16)

    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback = None
    ) -> None:
        """Execute bundle creation."""
        enhanced_logger = get_logger("pipeline.bundle")
        
        with enhanced_logger.performance_track("bundle_creation", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Creating artifact bundle")

            try:
                bundler = BundleCreator()
                context.manifest = await bundler.create_bundle(
                    context=context,
                    progress_callback=progress_callback,
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
                from ..util.fs import atomic_write_json
                
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
                context.logger.log_stage_error(self.stage_type, e)
                raise

    def _get_capabilities(self) -> list[str]:
        return ["basic_execution", "async_operations", "bundle_creation", "artifact_packaging"]