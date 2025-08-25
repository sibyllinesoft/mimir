"""
Unit tests for the QueryEngine architecture separation.

Tests the new QueryEngine class and its integration with IndexingPipeline
to ensure proper separation of indexing and querying concerns.
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

from src.repoindex.pipeline.query_engine import QueryEngine, IndexedRepository
from src.repoindex.data.schemas import (
    AskResponse,
    SearchResponse, 
    FeatureConfig,
    Citation,
    SerenaGraph,
    SymbolEntry,
    SymbolType,
)
from src.repoindex.util.errors import ValidationError


@pytest.fixture
def query_engine():
    """Create a QueryEngine for testing."""
    return QueryEngine(concurrency_limits={"query": 2})


@pytest.fixture
def mock_serena_graph():
    """Create a mock Serena graph for testing."""
    # Create mock symbol entries
    mock_entries = [
        SymbolEntry(
            symbol="test_function",
            path="test.py",
            span=(10, 20),
            type=SymbolType.DEF,
        ),
        SymbolEntry(
            symbol="test_function",
            path="main.py", 
            span=(5, 6),
            type=SymbolType.REF,
        ),
    ]
    
    graph = MagicMock(spec=SerenaGraph)
    graph.entries = mock_entries
    return graph


@pytest.fixture
def indexed_repo(mock_serena_graph):
    """Create an IndexedRepository for testing."""
    return IndexedRepository(
        index_id="test-index-123",
        repo_root="/test/repo",
        rev="abc123",
        serena_graph=mock_serena_graph,
        vector_index={"dimension": 768, "total_chunks": 100},
        repomap_data={"files": ["test.py", "main.py"]},
        snippets={"count": 50},
        manifest={"version": "1.0"},
    )


class TestIndexedRepository:
    """Test IndexedRepository container class."""

    def test_indexed_repository_creation(self):
        """Test creating an IndexedRepository."""
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test",
            rev="abc",
        )
        
        assert repo.index_id == "test-123"
        assert repo.repo_root == "/test"
        assert repo.rev == "abc"
        assert repo.serena_graph is None
        assert not repo.is_complete
        assert not repo.has_vector_search

    def test_indexed_repository_completeness(self, mock_serena_graph):
        """Test repository completeness checks."""
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test", 
            rev="abc",
        )
        
        # Not complete without serena graph
        assert not repo.is_complete
        
        # Complete with serena graph
        repo.serena_graph = mock_serena_graph
        assert repo.is_complete

    def test_vector_search_availability(self):
        """Test vector search availability checks."""
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test",
            rev="abc",
        )
        
        # No vector search initially
        assert not repo.has_vector_search
        
        # Has vector search with vector index
        repo.vector_index = {"dimension": 768}
        assert repo.has_vector_search

    def test_validate_for_operation_ask_success(self, mock_serena_graph):
        """Test successful validation for ask operation."""
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test",
            rev="abc",
            serena_graph=mock_serena_graph,
        )
        
        # Should not raise
        repo.validate_for_operation("ask")

    def test_validate_for_operation_ask_failure(self):
        """Test failed validation for ask operation."""
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test",
            rev="abc",
        )
        
        # Should raise ValidationError
        with pytest.raises(ValidationError, match="Symbol graph not available"):
            repo.validate_for_operation("ask")

    def test_validate_for_operation_search_success(self, mock_serena_graph):
        """Test successful validation for search operation."""
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test",
            rev="abc",
            serena_graph=mock_serena_graph,
        )
        
        # Should not raise
        repo.validate_for_operation("search")

    def test_validate_for_operation_search_failure(self):
        """Test failed validation for search operation.""" 
        repo = IndexedRepository(
            index_id="test-123",
            repo_root="/test",
            rev="abc",
        )
        
        # Should raise ValidationError
        with pytest.raises(ValidationError, match="Repository not fully indexed"):
            repo.validate_for_operation("search")


class TestQueryEngine:
    """Test QueryEngine class."""

    def test_query_engine_initialization(self, query_engine):
        """Test QueryEngine initialization."""
        assert query_engine.concurrency_limits == {"query": 2}
        assert query_engine.indexed_repositories == {}
        assert query_engine.query_semaphore._value == 2

    def test_register_indexed_repository(self, query_engine, indexed_repo):
        """Test registering an indexed repository."""
        query_engine.register_indexed_repository(indexed_repo)
        
        assert "test-index-123" in query_engine.indexed_repositories
        assert query_engine.indexed_repositories["test-index-123"] == indexed_repo

    def test_unregister_indexed_repository(self, query_engine, indexed_repo):
        """Test unregistering an indexed repository."""
        # Register first
        query_engine.register_indexed_repository(indexed_repo)
        assert "test-index-123" in query_engine.indexed_repositories
        
        # Unregister
        result = query_engine.unregister_indexed_repository("test-index-123")
        assert result is True
        assert "test-index-123" not in query_engine.indexed_repositories
        
        # Unregister non-existent
        result = query_engine.unregister_indexed_repository("non-existent")
        assert result is False

    def test_get_indexed_repository(self, query_engine, indexed_repo):
        """Test getting an indexed repository."""
        # Not found initially
        result = query_engine.get_indexed_repository("test-index-123")
        assert result is None
        
        # Found after registration
        query_engine.register_indexed_repository(indexed_repo)
        result = query_engine.get_indexed_repository("test-index-123")
        assert result == indexed_repo

    def test_list_indexed_repositories(self, query_engine, indexed_repo):
        """Test listing indexed repositories."""
        # Empty initially
        assert query_engine.list_indexed_repositories() == []
        
        # Contains registered repos
        query_engine.register_indexed_repository(indexed_repo)
        assert query_engine.list_indexed_repositories() == ["test-index-123"]

    @pytest.mark.asyncio
    async def test_search_repository_not_found(self, query_engine):
        """Test search with non-existent repository."""
        with pytest.raises(ValidationError, match="No indexed repository found"):
            await query_engine.search(
                index_id="non-existent",
                query="test query",
            )

    @pytest.mark.asyncio
    async def test_search_repository_not_ready(self, query_engine):
        """Test search with repository not ready for search."""
        # Create incomplete repository
        incomplete_repo = IndexedRepository(
            index_id="incomplete",
            repo_root="/test",
            rev="abc",
        )
        query_engine.register_indexed_repository(incomplete_repo)
        
        with pytest.raises(ValidationError, match="Repository not fully indexed"):
            await query_engine.search(
                index_id="incomplete",
                query="test query",
            )

    @pytest.mark.asyncio
    async def test_search_success(self, query_engine, indexed_repo):
        """Test successful search operation."""
        query_engine.register_indexed_repository(indexed_repo)
        
        response = await query_engine.search(
            index_id="test-index-123", 
            query="test query",
            k=10,
            features=FeatureConfig(vector=True, symbol=True),
            context_lines=3,
        )
        
        assert isinstance(response, SearchResponse)
        assert response.query == "test query"
        assert response.index_id == "test-index-123"
        assert response.features_used.vector is True
        assert response.features_used.symbol is True

    @pytest.mark.asyncio  
    async def test_ask_repository_not_found(self, query_engine):
        """Test ask with non-existent repository."""
        with pytest.raises(ValidationError, match="No indexed repository found"):
            await query_engine.ask(
                index_id="non-existent",
                question="What does this function do?",
            )

    @pytest.mark.asyncio
    async def test_ask_repository_not_ready(self, query_engine):
        """Test ask with repository not ready for ask operations."""
        # Create repository without serena graph
        incomplete_repo = IndexedRepository(
            index_id="incomplete",
            repo_root="/test",
            rev="abc",
        )
        query_engine.register_indexed_repository(incomplete_repo)
        
        with pytest.raises(ValidationError, match="Symbol graph not available"):
            await query_engine.ask(
                index_id="incomplete",
                question="What does this function do?",
            )

    @pytest.mark.asyncio
    async def test_ask_success(self, query_engine, indexed_repo):
        """Test successful ask operation."""
        query_engine.register_indexed_repository(indexed_repo)
        
        # Mock the symbol navigator to return a response
        with patch.object(query_engine.symbol_navigator, 'ask', new_callable=AsyncMock) as mock_ask:
            mock_response = AskResponse(
                question="What does test_function do?",
                answer="This is a test function that performs testing operations.",
                citations=[
                    Citation(
                        repo_root="/test/repo",
                        rev="abc123", 
                        path="test.py",
                        span=(10, 20),
                        content_sha="sha123",
                    )
                ],
                execution_time_ms=150.0,
                index_id="",  # Will be set by query engine
            )
            mock_ask.return_value = mock_response
            
            response = await query_engine.ask(
                index_id="test-index-123",
                question="What does test_function do?",
                context_lines=5,
            )
            
            # Verify response
            assert isinstance(response, AskResponse)
            assert response.question == "What does test_function do?"
            assert response.answer == "This is a test function that performs testing operations."
            assert response.index_id == "test-index-123"  # Should be set by query engine
            assert len(response.citations) == 1
            
            # Verify navigator was called correctly
            mock_ask.assert_called_once_with(
                question="What does test_function do?",
                serena_graph=indexed_repo.serena_graph,
                repo_root="/test/repo",
                rev="abc123",
                context_lines=5,
            )

    def test_get_repository_stats(self, query_engine, indexed_repo):
        """Test getting repository statistics."""
        # Not found initially
        stats = query_engine.get_repository_stats("test-index-123")
        assert stats is None
        
        # Found after registration
        query_engine.register_indexed_repository(indexed_repo)
        stats = query_engine.get_repository_stats("test-index-123")
        
        assert stats is not None
        assert stats["index_id"] == "test-index-123"
        assert stats["repo_root"] == "/test/repo"
        assert stats["rev"] == "abc123"
        assert stats["has_serena_graph"] is True
        assert stats["has_vector_index"] is True
        assert stats["has_repomap_data"] is True
        assert stats["has_snippets"] is True
        assert stats["has_manifest"] is True
        assert stats["is_complete"] is True
        assert stats["has_vector_search"] is True

    @pytest.mark.asyncio
    async def test_health_check(self, query_engine, indexed_repo):
        """Test health check functionality."""
        query_engine.register_indexed_repository(indexed_repo)
        
        health = await query_engine.health_check()
        
        assert health["status"] == "healthy"
        assert health["indexed_repositories"] == 1
        assert health["concurrency_limits"] == {"query": 2}
        assert health["available_operations"] == ["search", "ask"]


class TestArchitecturalSeparation:
    """Test that the architectural separation is working correctly."""

    @pytest.mark.asyncio
    async def test_query_operations_isolated_from_indexing(self, query_engine):
        """Test that query operations are completely isolated from indexing."""
        # QueryEngine should not have any indexing-related methods
        indexing_methods = [
            "start_indexing", "_execute_pipeline", "_stage_acquire", "_stage_repomapper",
            "_stage_serena", "_stage_leann", "_stage_snippets", "_stage_bundle"
        ]
        
        for method_name in indexing_methods:
            assert not hasattr(query_engine, method_name), f"QueryEngine should not have indexing method: {method_name}"

    def test_indexing_pipeline_methods_removed(self):
        """Test that IndexingPipeline no longer has query methods."""
        from src.repoindex.pipeline.run import IndexingPipeline
        
        # These methods should have been moved to QueryEngine
        query_methods = ["search", "ask"]
        
        pipeline = IndexingPipeline(storage_dir=Path("/tmp/test"))
        
        for method_name in query_methods:
            assert not hasattr(pipeline, method_name), f"IndexingPipeline should not have query method: {method_name}"

    def test_single_responsibility_principle(self):
        """Test that classes follow Single Responsibility Principle."""
        from src.repoindex.pipeline.run import IndexingPipeline
        
        # IndexingPipeline should only have indexing-related methods
        pipeline = IndexingPipeline(storage_dir=Path("/tmp/test"))
        indexing_methods = [
            "start_indexing", "_execute_pipeline", "_stage_acquire", "_stage_repomapper",
            "_stage_serena", "_stage_leann", "_stage_snippets", "_stage_bundle"
        ]
        
        for method_name in indexing_methods:
            assert hasattr(pipeline, method_name), f"IndexingPipeline should have indexing method: {method_name}"
        
        # QueryEngine should only have query-related methods
        query_engine = QueryEngine()
        query_methods = ["search", "ask", "register_indexed_repository", "get_repository_stats"]
        
        for method_name in query_methods:
            assert hasattr(query_engine, method_name), f"QueryEngine should have query method: {method_name}"

    def test_clean_dependency_injection(self):
        """Test that dependency injection is clean and follows best practices."""
        from src.repoindex.pipeline.run import IndexingPipeline
        
        query_engine = QueryEngine()
        
        # Pipeline can be created with or without query engine
        pipeline1 = IndexingPipeline(storage_dir=Path("/tmp/test"))
        assert pipeline1.get_query_engine() is None
        
        pipeline2 = IndexingPipeline(storage_dir=Path("/tmp/test"), query_engine=query_engine)
        assert pipeline2.get_query_engine() == query_engine
        
        # Pipeline can be configured with query engine after creation
        pipeline1.set_query_engine(query_engine)
        assert pipeline1.get_query_engine() == query_engine


if __name__ == "__main__":
    pytest.main([__file__])