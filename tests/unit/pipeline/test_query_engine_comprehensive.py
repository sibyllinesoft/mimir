"""
Comprehensive tests for the QueryEngine and IndexedRepository classes.

Tests cover repository management, search operations, ask operations,
error handling, concurrency control, and monitoring integration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any, Dict

from src.repoindex.pipeline.query_engine import QueryEngine, IndexedRepository
from src.repoindex.data.schemas import AskResponse, FeatureConfig, SearchResponse
from src.repoindex.util.errors import ValidationError, MimirError
from tests.fixtures.advanced_fixtures import (
    AdvancedMockManager,
    MockPipelineComponents,
    sample_search_results,
)


class TestIndexedRepository:
    """Test the IndexedRepository container class."""

    def test_init_complete_repository(self):
        """Test initialization with complete repository data."""
        repo = IndexedRepository(
            index_id="test-repo-001",
            repo_root="/path/to/repo",
            rev="main",
            serena_graph={"nodes": [], "edges": []},
            vector_index={"vectors": []},
            repomap_data={"files": []},
            snippets={"snippets": []},
            manifest={"version": "1.0"},
        )
        
        assert repo.index_id == "test-repo-001"
        assert repo.repo_root == "/path/to/repo"
        assert repo.rev == "main"
        assert repo.is_complete is True
        assert repo.has_vector_search is True

    def test_init_minimal_repository(self):
        """Test initialization with minimal required data."""
        repo = IndexedRepository(
            index_id="minimal-repo",
            repo_root="/minimal",
            rev="develop",
        )
        
        assert repo.index_id == "minimal-repo"
        assert repo.is_complete is False
        assert repo.has_vector_search is False

    def test_is_complete_property(self):
        """Test is_complete property with various configurations."""
        # Complete repository
        complete_repo = IndexedRepository(
            index_id="complete",
            repo_root="/path",
            rev="main",
            serena_graph={"data": "present"},
        )
        assert complete_repo.is_complete is True
        
        # Incomplete repository
        incomplete_repo = IndexedRepository(
            index_id="incomplete",
            repo_root="/path",
            rev="main",
            serena_graph=None,
        )
        assert incomplete_repo.is_complete is False

    def test_has_vector_search_property(self):
        """Test has_vector_search property with various configurations."""
        # With vector index
        with_vectors = IndexedRepository(
            index_id="vectors",
            repo_root="/path",
            rev="main",
            vector_index={"vectors": ["data"]},
        )
        assert with_vectors.has_vector_search is True
        
        # Without vector index
        without_vectors = IndexedRepository(
            index_id="no-vectors",
            repo_root="/path",
            rev="main",
            vector_index=None,
        )
        assert without_vectors.has_vector_search is False

    def test_validate_for_ask_operation_success(self):
        """Test successful validation for ask operations."""
        repo = IndexedRepository(
            index_id="ask-ready",
            repo_root="/path",
            rev="main",
            serena_graph={"graph": "data"},
        )
        
        # Should not raise any exception
        repo.validate_for_operation("ask")

    def test_validate_for_ask_operation_failure(self):
        """Test validation failure for ask operations."""
        repo = IndexedRepository(
            index_id="ask-not-ready",
            repo_root="/path",
            rev="main",
            serena_graph=None,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            repo.validate_for_operation("ask")
        
        assert "Symbol graph not available" in str(exc_info.value)
        assert exc_info.value.context["component"] == "query_engine"
        assert exc_info.value.context["operation"] == "ask"

    def test_validate_for_search_operation_success(self):
        """Test successful validation for search operations."""
        repo = IndexedRepository(
            index_id="search-ready",
            repo_root="/path",
            rev="main",
            serena_graph={"graph": "data"},
        )
        
        # Should not raise any exception
        repo.validate_for_operation("search")

    def test_validate_for_search_operation_failure(self):
        """Test validation failure for search operations."""
        repo = IndexedRepository(
            index_id="search-not-ready",
            repo_root="/path",
            rev="main",
            serena_graph=None,
        )
        
        with pytest.raises(ValidationError) as exc_info:
            repo.validate_for_operation("search")
        
        assert "Repository not fully indexed" in str(exc_info.value)
        assert exc_info.value.context["component"] == "query_engine"
        assert exc_info.value.context["operation"] == "search"


class TestQueryEngine:
    """Test the QueryEngine class."""

    @pytest.fixture
    def query_engine(self):
        """Create QueryEngine instance for testing."""
        with patch("src.repoindex.pipeline.query_engine.get_metrics_collector"), \
             patch("src.repoindex.pipeline.query_engine.get_trace_manager"), \
             patch("src.repoindex.pipeline.query_engine.SymbolGraphNavigator"):
            return QueryEngine(concurrency_limits={"query": 2})

    @pytest.fixture
    def mock_indexed_repo(self):
        """Create mock indexed repository."""
        return IndexedRepository(
            index_id="test-repo",
            repo_root="/test/repo",
            rev="main",
            serena_graph={"nodes": [], "edges": []},
            vector_index={"vectors": [], "metadata": {}},
            repomap_data={"files": ["file1.py", "file2.py"]},
            snippets={"snippets": ["snippet1", "snippet2"]},
            manifest={"version": "1.0", "created": "2024-01-01"},
        )

    def test_init_default_concurrency(self):
        """Test initialization with default concurrency limits."""
        with patch("src.repoindex.pipeline.query_engine.get_metrics_collector"), \
             patch("src.repoindex.pipeline.query_engine.get_trace_manager"), \
             patch("src.repoindex.pipeline.query_engine.SymbolGraphNavigator"):
            engine = QueryEngine()
            
            assert engine.concurrency_limits == {"query": 4}
            assert engine.query_semaphore._value == 4

    def test_init_custom_concurrency(self):
        """Test initialization with custom concurrency limits."""
        custom_limits = {"query": 8}
        
        with patch("src.repoindex.pipeline.query_engine.get_metrics_collector"), \
             patch("src.repoindex.pipeline.query_engine.get_trace_manager"), \
             patch("src.repoindex.pipeline.query_engine.SymbolGraphNavigator"):
            engine = QueryEngine(concurrency_limits=custom_limits)
            
            assert engine.concurrency_limits == custom_limits
            assert engine.query_semaphore._value == 8

    def test_register_indexed_repository(self, query_engine, mock_indexed_repo):
        """Test registering an indexed repository."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        assert query_engine.indexed_repositories["test-repo"] == mock_indexed_repo
        assert len(query_engine.indexed_repositories) == 1

    def test_unregister_indexed_repository_success(self, query_engine, mock_indexed_repo):
        """Test successful unregistration of indexed repository."""
        # First register
        query_engine.register_indexed_repository(mock_indexed_repo)
        assert len(query_engine.indexed_repositories) == 1
        
        # Then unregister
        result = query_engine.unregister_indexed_repository("test-repo")
        
        assert result is True
        assert len(query_engine.indexed_repositories) == 0
        assert "test-repo" not in query_engine.indexed_repositories

    def test_unregister_indexed_repository_failure(self, query_engine):
        """Test unregistration of non-existent repository."""
        result = query_engine.unregister_indexed_repository("non-existent")
        
        assert result is False
        assert len(query_engine.indexed_repositories) == 0

    def test_get_indexed_repository_success(self, query_engine, mock_indexed_repo):
        """Test successful retrieval of indexed repository."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        retrieved = query_engine.get_indexed_repository("test-repo")
        
        assert retrieved == mock_indexed_repo
        assert retrieved.index_id == "test-repo"

    def test_get_indexed_repository_failure(self, query_engine):
        """Test retrieval of non-existent repository."""
        retrieved = query_engine.get_indexed_repository("non-existent")
        
        assert retrieved is None

    def test_list_indexed_repositories(self, query_engine):
        """Test listing indexed repositories."""
        # Empty list initially
        assert query_engine.list_indexed_repositories() == []
        
        # Add repositories
        repo1 = IndexedRepository("repo1", "/path1", "main")
        repo2 = IndexedRepository("repo2", "/path2", "develop")
        
        query_engine.register_indexed_repository(repo1)
        query_engine.register_indexed_repository(repo2)
        
        repo_list = query_engine.list_indexed_repositories()
        assert set(repo_list) == {"repo1", "repo2"}
        assert len(repo_list) == 2

    @pytest.mark.asyncio
    async def test_search_success(self, query_engine, mock_indexed_repo):
        """Test successful search operation."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock search engine
        mock_search_engine = AsyncMock()
        mock_response = SearchResponse(
            query="test query",
            results=[],
            total_count=5,
            execution_time_ms=150.0,
        )
        mock_search_engine.search.return_value = mock_response
        
        with patch("src.repoindex.pipeline.query_engine.EnhancedSearchPipeline") as mock_enhanced:
            mock_enhanced.return_value = mock_search_engine
            
            response = await query_engine.search(
                index_id="test-repo",
                query="test query",
                k=10,
            )
        
        assert response.index_id == "test-repo"
        assert response.query == "test query"
        assert response.total_count == 5
        assert response.execution_time_ms == 150.0
        
        # Verify search engine was called correctly
        mock_search_engine.initialize.assert_called_once()
        mock_search_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_fallback_to_hybrid(self, query_engine, mock_indexed_repo):
        """Test fallback to HybridSearchEngine when EnhancedSearchPipeline is not available."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock hybrid search engine
        mock_hybrid_engine = AsyncMock()
        mock_response = SearchResponse(
            query="fallback query",
            results=[],
            total_count=3,
            execution_time_ms=200.0,
        )
        mock_hybrid_engine.search.return_value = mock_response
        
        # Mock ImportError for enhanced search
        with patch("src.repoindex.pipeline.query_engine.EnhancedSearchPipeline", side_effect=ImportError), \
             patch("src.repoindex.pipeline.query_engine.HybridSearchEngine") as mock_hybrid, \
             patch("src.repoindex.pipeline.query_engine.get_logger"):
            
            mock_hybrid.return_value = mock_hybrid_engine
            
            response = await query_engine.search(
                index_id="test-repo",
                query="fallback query",
                k=5,
            )
        
        assert response.index_id == "test-repo"
        assert response.query == "fallback query"
        assert response.total_count == 3
        
        # Verify hybrid engine was used
        mock_hybrid_engine.search.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_repository_not_found(self, query_engine):
        """Test search with non-existent repository."""
        with pytest.raises(ValidationError) as exc_info:
            await query_engine.search(
                index_id="non-existent",
                query="test query",
            )
        
        assert "No indexed repository found" in str(exc_info.value)
        assert exc_info.value.context["index_id"] == "non-existent"

    @pytest.mark.asyncio
    async def test_search_repository_not_ready(self, query_engine):
        """Test search with repository that's not ready."""
        incomplete_repo = IndexedRepository(
            index_id="incomplete",
            repo_root="/path",
            rev="main",
            serena_graph=None,  # Missing required data
        )
        query_engine.register_indexed_repository(incomplete_repo)
        
        with pytest.raises(ValidationError) as exc_info:
            await query_engine.search(
                index_id="incomplete",
                query="test query",
            )
        
        assert "Repository not fully indexed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_search_with_features(self, query_engine, mock_indexed_repo):
        """Test search with custom feature configuration."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        features = FeatureConfig(
            vector_search=True,
            semantic_search=True,
            reranking=False,
        )
        
        mock_search_engine = AsyncMock()
        mock_response = SearchResponse(
            query="feature test",
            results=[],
            total_count=2,
            execution_time_ms=100.0,
        )
        mock_search_engine.search.return_value = mock_response
        
        with patch("src.repoindex.pipeline.query_engine.EnhancedSearchPipeline") as mock_enhanced:
            mock_enhanced.return_value = mock_search_engine
            
            await query_engine.search(
                index_id="test-repo",
                query="feature test",
                k=20,
                features=features,
                context_lines=10,
            )
        
        # Verify features were passed correctly
        call_args = mock_search_engine.search.call_args
        assert call_args.kwargs["features"] == features
        assert call_args.kwargs["k"] == 20
        assert call_args.kwargs["context_lines"] == 10

    @pytest.mark.asyncio
    async def test_search_concurrency_control(self, query_engine, mock_indexed_repo):
        """Test that search operations respect concurrency limits."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock search engine with delay to test concurrency
        mock_search_engine = AsyncMock()
        mock_response = SearchResponse(
            query="concurrent test",
            results=[],
            total_count=1,
            execution_time_ms=50.0,
        )
        
        async def delayed_search(*args, **kwargs):
            await asyncio.sleep(0.1)  # Small delay
            return mock_response
        
        mock_search_engine.search.side_effect = delayed_search
        
        with patch("src.repoindex.pipeline.query_engine.EnhancedSearchPipeline") as mock_enhanced:
            mock_enhanced.return_value = mock_search_engine
            
            # Start multiple concurrent searches (more than the limit of 2)
            tasks = [
                query_engine.search("test-repo", f"query {i}")
                for i in range(5)
            ]
            
            # All should complete successfully due to semaphore management
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 5
            assert all(r.query.startswith("query") for r in responses)

    @pytest.mark.asyncio
    async def test_ask_success(self, query_engine, mock_indexed_repo):
        """Test successful ask operation."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock symbol navigator
        mock_ask_response = AskResponse(
            question="What does this function do?",
            answer="This function processes data.",
            citations=[],
            execution_time_ms=300.0,
        )
        query_engine.symbol_navigator.ask = AsyncMock(return_value=mock_ask_response)
        
        response = await query_engine.ask(
            index_id="test-repo",
            question="What does this function do?",
            context_lines=5,
        )
        
        assert response.index_id == "test-repo"
        assert response.question == "What does this function do?"
        assert response.answer == "This function processes data."
        assert response.execution_time_ms == 300.0
        
        # Verify navigator was called correctly
        query_engine.symbol_navigator.ask.assert_called_once_with(
            question="What does this function do?",
            serena_graph=mock_indexed_repo.serena_graph,
            repo_root=mock_indexed_repo.repo_root,
            rev=mock_indexed_repo.rev,
            context_lines=5,
        )

    @pytest.mark.asyncio
    async def test_ask_repository_not_found(self, query_engine):
        """Test ask with non-existent repository."""
        with pytest.raises(ValidationError) as exc_info:
            await query_engine.ask(
                index_id="non-existent",
                question="Test question?",
            )
        
        assert "No indexed repository found" in str(exc_info.value)
        assert exc_info.value.context["index_id"] == "non-existent"

    @pytest.mark.asyncio
    async def test_ask_repository_not_ready(self, query_engine):
        """Test ask with repository missing symbol graph."""
        incomplete_repo = IndexedRepository(
            index_id="no-graph",
            repo_root="/path",
            rev="main",
            serena_graph=None,
        )
        query_engine.register_indexed_repository(incomplete_repo)
        
        with pytest.raises(ValidationError) as exc_info:
            await query_engine.ask(
                index_id="no-graph",
                question="Test question?",
            )
        
        assert "Symbol graph not available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_ask_error_handling(self, query_engine, mock_indexed_repo):
        """Test error handling in ask operation."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock navigator to raise exception
        query_engine.symbol_navigator.ask = AsyncMock(
            side_effect=Exception("Navigator failed")
        )
        
        with pytest.raises(MimirError) as exc_info:
            await query_engine.ask(
                index_id="test-repo",
                question="Failing question?",
            )
        
        assert "Question answering failed" in str(exc_info.value)
        assert exc_info.value.category.value == "logic"

    def test_get_repository_stats_success(self, query_engine, mock_indexed_repo):
        """Test successful repository stats retrieval."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        stats = query_engine.get_repository_stats("test-repo")
        
        assert stats is not None
        assert stats["index_id"] == "test-repo"
        assert stats["repo_root"] == "/test/repo"
        assert stats["rev"] == "main"
        assert stats["has_serena_graph"] is True
        assert stats["has_vector_index"] is True
        assert stats["has_repomap_data"] is True
        assert stats["has_snippets"] is True
        assert stats["has_manifest"] is True
        assert stats["is_complete"] is True
        assert stats["has_vector_search"] is True

    def test_get_repository_stats_not_found(self, query_engine):
        """Test repository stats for non-existent repository."""
        stats = query_engine.get_repository_stats("non-existent")
        
        assert stats is None

    def test_get_repository_stats_partial_data(self, query_engine):
        """Test repository stats with partial data."""
        partial_repo = IndexedRepository(
            index_id="partial",
            repo_root="/partial",
            rev="branch",
            serena_graph={"data": "present"},
            # Missing other components
        )
        query_engine.register_indexed_repository(partial_repo)
        
        stats = query_engine.get_repository_stats("partial")
        
        assert stats["has_serena_graph"] is True
        assert stats["has_vector_index"] is False
        assert stats["has_repomap_data"] is False
        assert stats["has_snippets"] is False
        assert stats["has_manifest"] is False
        assert stats["is_complete"] is True  # Only needs serena_graph
        assert stats["has_vector_search"] is False

    @pytest.mark.asyncio
    async def test_health_check(self, query_engine, mock_indexed_repo):
        """Test query engine health check."""
        # Add some repositories
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        repo2 = IndexedRepository("repo2", "/path2", "develop")
        query_engine.register_indexed_repository(repo2)
        
        health = await query_engine.health_check()
        
        assert health["status"] == "healthy"
        assert health["indexed_repositories"] == 2
        assert health["concurrency_limits"] == {"query": 2}
        assert "search" in health["available_operations"]
        assert "ask" in health["available_operations"]

    @pytest.mark.asyncio
    async def test_concurrent_operations_mixed(self, query_engine, mock_indexed_repo):
        """Test concurrent mix of search and ask operations."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock both engines
        mock_search_engine = AsyncMock()
        mock_search_response = SearchResponse(
            query="search query",
            results=[],
            total_count=1,
            execution_time_ms=100.0,
        )
        mock_search_engine.search.return_value = mock_search_response
        
        mock_ask_response = AskResponse(
            question="ask question",
            answer="test answer",
            citations=[],
            execution_time_ms=200.0,
        )
        query_engine.symbol_navigator.ask = AsyncMock(return_value=mock_ask_response)
        
        with patch("src.repoindex.pipeline.query_engine.EnhancedSearchPipeline") as mock_enhanced:
            mock_enhanced.return_value = mock_search_engine
            
            # Mix of operations
            tasks = [
                query_engine.search("test-repo", "search query 1"),
                query_engine.ask("test-repo", "ask question 1"),
                query_engine.search("test-repo", "search query 2"),
                query_engine.ask("test-repo", "ask question 2"),
            ]
            
            responses = await asyncio.gather(*tasks)
            
            assert len(responses) == 4
            assert isinstance(responses[0], SearchResponse)
            assert isinstance(responses[1], AskResponse)
            assert isinstance(responses[2], SearchResponse)
            assert isinstance(responses[3], AskResponse)

    @pytest.mark.asyncio
    async def test_error_handling_preserves_context(self, query_engine, mock_indexed_repo):
        """Test that error handling preserves all relevant context."""
        query_engine.register_indexed_repository(mock_indexed_repo)
        
        # Mock search engine to raise exception
        mock_search_engine = AsyncMock()
        mock_search_engine.search.side_effect = ValueError("Search failed")
        
        with patch("src.repoindex.pipeline.query_engine.EnhancedSearchPipeline") as mock_enhanced:
            mock_enhanced.return_value = mock_search_engine
            
            with pytest.raises(MimirError) as exc_info:
                await query_engine.search(
                    index_id="test-repo",
                    query="failing query",
                    k=15,
                    features=FeatureConfig(vector_search=True),
                )
        
        error = exc_info.value
        assert "Search operation failed" in str(error)
        assert error.context["component"] == "query_engine"
        assert error.context["operation"] == "search"
        assert error.context["parameters"]["index_id"] == "test-repo"
        assert error.context["parameters"]["query"] == "failing query"
        assert "features" in error.context["parameters"]