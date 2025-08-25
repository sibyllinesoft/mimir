"""
Query Engine for repository indexing.

Provides search and question answering capabilities using the indexed
repository data. Separates querying concerns from indexing pipeline.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from ..data.schemas import (
    AskResponse,
    FeatureConfig,
    SearchResponse,
)
from ..monitoring import get_metrics_collector, get_trace_manager
from ..util.errors import (
    ErrorCategory,
    ErrorSeverity,
    MimirError,
    RecoveryStrategy,
    ValidationError,
    create_error_context,
)
from ..util.logging_config import get_logger
from .ask_index import SymbolGraphNavigator


class IndexedRepository:
    """
    Container for indexed repository data.
    
    Holds all the artifacts from the indexing pipeline that are needed
    for querying operations.
    """

    def __init__(
        self,
        index_id: str,
        repo_root: str,
        rev: str,
        serena_graph: Any | None = None,
        vector_index: Any | None = None,
        repomap_data: Any | None = None,
        snippets: Any | None = None,
        manifest: Any | None = None,
    ):
        self.index_id = index_id
        self.repo_root = repo_root
        self.rev = rev
        self.serena_graph = serena_graph
        self.vector_index = vector_index
        self.repomap_data = repomap_data
        self.snippets = snippets
        self.manifest = manifest

    @property
    def is_complete(self) -> bool:
        """Check if this repository has all required data for querying."""
        return self.serena_graph is not None

    @property
    def has_vector_search(self) -> bool:
        """Check if vector search is available."""
        return self.vector_index is not None

    def validate_for_operation(self, operation: str) -> None:
        """Validate that the repository has required data for an operation."""
        if operation == "ask":
            if not self.serena_graph:
                raise ValidationError(
                    "Symbol graph not available - repository must be fully indexed first",
                    context=create_error_context(
                        component="query_engine",
                        operation=operation,
                        parameters={"index_id": self.index_id, "has_serena_graph": False},
                    ),
                )
        elif operation == "search":
            if not self.is_complete:
                raise ValidationError(
                    "Repository not fully indexed - cannot perform search",
                    context=create_error_context(
                        component="query_engine",
                        operation=operation,
                        parameters={"index_id": self.index_id, "is_complete": False},
                    ),
                )


class QueryEngine:
    """
    Query engine for indexed repositories.
    
    Handles all querying operations including search and question answering.
    Separated from indexing concerns to follow Single Responsibility Principle.
    """

    def __init__(self, concurrency_limits: dict[str, int] | None = None):
        """Initialize query engine."""
        self.concurrency_limits = concurrency_limits or {"query": 4}
        self.indexed_repositories: dict[str, IndexedRepository] = {}
        
        # Create semaphore for query concurrency control
        self.query_semaphore = asyncio.Semaphore(self.concurrency_limits["query"])
        
        # Initialize monitoring
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()
        
        # Initialize symbol graph navigator
        self.symbol_navigator = SymbolGraphNavigator(use_gemini=True)

    def register_indexed_repository(self, indexed_repo: IndexedRepository) -> None:
        """Register a completed indexed repository for querying."""
        self.indexed_repositories[indexed_repo.index_id] = indexed_repo
        
        # Update metrics
        self.metrics_collector.set_cache_size(
            "indexed_repositories", len(self.indexed_repositories)
        )

    def unregister_indexed_repository(self, index_id: str) -> bool:
        """Unregister an indexed repository."""
        if index_id in self.indexed_repositories:
            del self.indexed_repositories[index_id]
            
            # Update metrics
            self.metrics_collector.set_cache_size(
                "indexed_repositories", len(self.indexed_repositories)
            )
            return True
        return False

    def get_indexed_repository(self, index_id: str) -> IndexedRepository | None:
        """Get an indexed repository by ID."""
        return self.indexed_repositories.get(index_id)

    def list_indexed_repositories(self) -> list[str]:
        """List all available indexed repository IDs."""
        return list(self.indexed_repositories.keys())

    async def search(
        self,
        index_id: str,
        query: str,
        k: int = 20,
        features: FeatureConfig = FeatureConfig(),
        context_lines: int = 5,
    ) -> SearchResponse:
        """
        Execute hybrid search against an indexed repository.
        
        Args:
            index_id: Repository index identifier
            query: Search query string
            k: Number of results to return
            features: Feature configuration for search modalities
            context_lines: Lines of context around matches
            
        Returns:
            Search response with results and metadata
            
        Raises:
            ValidationError: If repository not found or not ready
            MimirError: If search operation fails
        """
        enhanced_logger = get_logger("query_engine.search")
        
        async with self.trace_manager.trace_operation(
            "query_search",
            index_id=index_id,
            query=query,
            features=features.model_dump(),
        ):
            with enhanced_logger.performance_track("hybrid_search"):
                try:
                    # Get indexed repository
                    indexed_repo = self.get_indexed_repository(index_id)
                    if not indexed_repo:
                        raise ValidationError(
                            f"No indexed repository found for ID: {index_id}",
                            context=create_error_context(
                                component="query_engine",
                                operation="search",
                                parameters={"index_id": index_id},
                            ),
                        )

                    # Validate repository is ready for search
                    indexed_repo.validate_for_operation("search")

                    async with self.query_semaphore:
                        # Execute search operation
                        # This is a placeholder implementation - would integrate with actual search systems
                        start_time = time.time()
                        
                        # TODO: Implement actual hybrid search logic here
                        # This would use:
                        # - indexed_repo.vector_index for vector search
                        # - indexed_repo.serena_graph for symbol search  
                        # - indexed_repo.repomap_data for graph-based search
                        # - indexed_repo.snippets for context retrieval
                        
                        execution_time_ms = (time.time() - start_time) * 1000
                        
                        response = SearchResponse(
                            query=query,
                            results=[],  # Would be populated with actual results
                            total_count=0,
                            features_used=features,
                            execution_time_ms=execution_time_ms,
                            index_id=index_id,
                        )

                    enhanced_logger.operation_success(
                        "hybrid_search",
                        query=query,
                        features_enabled=features.model_dump(),
                        results_count=response.total_count,
                        execution_time_ms=execution_time_ms,
                    )

                    return response

                except Exception as e:
                    if not isinstance(e, MimirError):
                        e = MimirError(
                            message=f"Search operation failed: {str(e)}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.LOGIC,
                            recovery_strategy=RecoveryStrategy.RETRY,
                            context=create_error_context(
                                component="query_engine",
                                operation="search",
                                parameters={
                                    "index_id": index_id,
                                    "query": query,
                                    "features": features.dict(),
                                },
                            ),
                            cause=e,
                        )

                    enhanced_logger.operation_error("hybrid_search", e, query=query)
                    raise

    async def ask(
        self,
        index_id: str,
        question: str,
        context_lines: int = 5,
    ) -> AskResponse:
        """
        Execute multi-hop reasoning against an indexed repository.
        
        Args:
            index_id: Repository index identifier
            question: Question to ask about the codebase
            context_lines: Lines of context for evidence
            
        Returns:
            Ask response with answer and supporting citations
            
        Raises:
            ValidationError: If repository not found or not ready
            MimirError: If ask operation fails
        """
        enhanced_logger = get_logger("query_engine.ask")

        async with self.trace_manager.trace_operation(
            "query_ask",
            index_id=index_id,
            question=question,
            context_lines=context_lines,
        ):
            with enhanced_logger.performance_track("question_answering"):
                try:
                    # Get indexed repository
                    indexed_repo = self.get_indexed_repository(index_id)
                    if not indexed_repo:
                        raise ValidationError(
                            f"No indexed repository found for ID: {index_id}",
                            context=create_error_context(
                                component="query_engine",
                                operation="ask",
                                parameters={"index_id": index_id},
                            ),
                        )

                    # Validate repository is ready for ask operations
                    indexed_repo.validate_for_operation("ask")

                    async with self.query_semaphore:
                        # Execute question answering using symbol graph navigator
                        response = await self.symbol_navigator.ask(
                            question=question,
                            serena_graph=indexed_repo.serena_graph,
                            repo_root=indexed_repo.repo_root,
                            rev=indexed_repo.rev,
                            context_lines=context_lines,
                        )

                        # Update the response with the correct index_id
                        response.index_id = index_id

                    enhanced_logger.operation_success(
                        "question_answering",
                        question=question,
                        context_lines=context_lines,
                        citations_count=len(response.citations),
                        execution_time_ms=response.execution_time_ms,
                    )

                    return response

                except Exception as e:
                    if not isinstance(e, MimirError):
                        e = MimirError(
                            message=f"Question answering failed: {str(e)}",
                            severity=ErrorSeverity.HIGH,
                            category=ErrorCategory.LOGIC,
                            recovery_strategy=RecoveryStrategy.RETRY,
                            context=create_error_context(
                                component="query_engine",
                                operation="ask",
                                parameters={
                                    "index_id": index_id,
                                    "question": question,
                                    "context_lines": context_lines,
                                },
                            ),
                            cause=e,
                        )

                    enhanced_logger.operation_error("question_answering", e, question=question)
                    raise

    def get_repository_stats(self, index_id: str) -> dict[str, Any] | None:
        """
        Get statistics about an indexed repository.
        
        Args:
            index_id: Repository index identifier
            
        Returns:
            Statistics dictionary or None if repository not found
        """
        indexed_repo = self.get_indexed_repository(index_id)
        if not indexed_repo:
            return None

        return {
            "index_id": indexed_repo.index_id,
            "repo_root": indexed_repo.repo_root,
            "rev": indexed_repo.rev,
            "has_serena_graph": indexed_repo.serena_graph is not None,
            "has_vector_index": indexed_repo.vector_index is not None,
            "has_repomap_data": indexed_repo.repomap_data is not None,
            "has_snippets": indexed_repo.snippets is not None,
            "has_manifest": indexed_repo.manifest is not None,
            "is_complete": indexed_repo.is_complete,
            "has_vector_search": indexed_repo.has_vector_search,
        }

    async def health_check(self) -> dict[str, Any]:
        """
        Perform health check on the query engine.
        
        Returns:
            Health status information
        """
        return {
            "status": "healthy",
            "indexed_repositories": len(self.indexed_repositories),
            "concurrency_limits": self.concurrency_limits,
            "available_operations": ["search", "ask"],
        }