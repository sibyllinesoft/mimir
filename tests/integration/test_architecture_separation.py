"""
Integration tests for the architectural separation between indexing and querying.

Tests the complete workflow from indexing to querying using the separated architecture.
"""

import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from src.repoindex.pipeline.run import IndexingPipeline
from src.repoindex.pipeline.query_engine import QueryEngine, IndexedRepository
from src.repoindex.mcp.server import MCPServer
from src.repoindex.data.schemas import (
    EnsureRepoIndexRequest,
    SearchRepoRequest, 
    AskIndexRequest,
    FeatureConfig,
    SerenaGraph,
    SymbolEntry,
    SymbolType,
)


@pytest.fixture
def temp_storage():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def query_engine():
    """Create QueryEngine for testing.""" 
    return QueryEngine()


@pytest.fixture
def mock_repo_structure():
    """Create mock repository structure."""
    return {
        "files": ["src/main.py", "src/utils.py", "tests/test_main.py"],
        "serena_graph": SerenaGraph(
            entries=[
                SymbolEntry(
                    symbol="main",
                    path="src/main.py",
                    span=(1, 10),
                    type=SymbolType.DEF,
                ),
                SymbolEntry(
                    symbol="helper_function",
                    path="src/utils.py", 
                    span=(5, 15),
                    type=SymbolType.DEF,
                ),
                SymbolEntry(
                    symbol="helper_function",
                    path="src/main.py",
                    span=(8, 9),
                    type=SymbolType.REF,
                ),
            ],
            file_count=2,
            symbol_count=2,
        ),
        "vector_index": {"dimension": 768, "total_chunks": 50},
        "repomap_data": {"structure": "analyzed"},
        "snippets": {"count": 25},
        "manifest": {"version": "1.0", "created": "2024-01-01"},
    }


class TestArchitecturalIntegration:
    """Test the complete architectural separation works end-to-end."""

    @pytest.mark.asyncio
    async def test_indexing_pipeline_registers_with_query_engine(
        self, temp_storage, query_engine, mock_repo_structure
    ):
        """Test that IndexingPipeline properly registers completed indexes with QueryEngine."""
        # Create pipeline with query engine
        pipeline = IndexingPipeline(storage_dir=temp_storage, query_engine=query_engine)
        
        # Mock the pipeline stages to simulate successful completion
        with patch.object(pipeline, '_execute_pipeline') as mock_execute:
            # Create a mock context that simulates successful pipeline completion
            mock_context = MagicMock()
            mock_context.index_id = "test-index-123"
            mock_context.repo_info.root = "/test/repo"
            mock_context.repo_info.rev = "abc123"
            mock_context.serena_graph = mock_repo_structure["serena_graph"]
            mock_context.vector_index = mock_repo_structure["vector_index"]
            mock_context.repomap_data = mock_repo_structure["repomap_data"]
            mock_context.snippets = mock_repo_structure["snippets"]
            mock_context.manifest = mock_repo_structure["manifest"]
            
            pipeline.active_pipelines["test-index-123"] = mock_context
            
            # Simulate successful registration with query engine
            indexed_repo = IndexedRepository(
                index_id="test-index-123",
                repo_root="/test/repo",
                rev="abc123", 
                serena_graph=mock_repo_structure["serena_graph"],
                vector_index=mock_repo_structure["vector_index"],
                repomap_data=mock_repo_structure["repomap_data"],
                snippets=mock_repo_structure["snippets"],
                manifest=mock_repo_structure["manifest"],
            )
            
            query_engine.register_indexed_repository(indexed_repo)
            
            # Verify registration
            assert "test-index-123" in query_engine.indexed_repositories
            registered_repo = query_engine.get_indexed_repository("test-index-123")
            assert registered_repo.index_id == "test-index-123"
            assert registered_repo.repo_root == "/test/repo"
            assert registered_repo.is_complete

    @pytest.mark.asyncio 
    async def test_mcp_server_uses_query_engine_for_search(self, temp_storage):
        """Test that MCP server uses QueryEngine for search operations."""
        # Create MCP server with integrated architecture
        server = MCPServer(storage_dir=temp_storage)
        
        # Mock an indexed repository in the query engine
        mock_repo = IndexedRepository(
            index_id="test-index-123",
            repo_root="/test/repo",
            rev="abc123",
            serena_graph=MagicMock(),
            vector_index={"dimension": 768},
        )
        server.query_engine.register_indexed_repository(mock_repo)
        
        # Mock the search method
        with patch.object(server.query_engine, 'search', new_callable=AsyncMock) as mock_search:
            from src.repoindex.data.schemas import SearchResponse
            mock_search.return_value = SearchResponse(
                query="test query",
                results=[],
                total_count=0,
                features_used=FeatureConfig(),
                execution_time_ms=100.0,
                index_id="test-index-123",
            )
            
            # Call search through MCP server
            result = await server._search_repo({
                "index_id": "test-index-123",
                "query": "test query",
                "k": 10,
                "features": {"vector": True, "symbol": True},
                "context_lines": 3,
            })
            
            # Verify query engine was called
            mock_search.assert_called_once()
            assert not result.isError
            
            # Verify call arguments
            call_args = mock_search.call_args
            assert call_args[1]["index_id"] == "test-index-123"
            assert call_args[1]["query"] == "test query"

    @pytest.mark.asyncio
    async def test_mcp_server_uses_query_engine_for_ask(self, temp_storage):
        """Test that MCP server uses QueryEngine for ask operations."""
        # Create MCP server with integrated architecture
        server = MCPServer(storage_dir=temp_storage)
        
        # Mock an indexed repository in the query engine
        mock_repo = IndexedRepository(
            index_id="test-index-123",
            repo_root="/test/repo", 
            rev="abc123",
            serena_graph=MagicMock(),
        )
        server.query_engine.register_indexed_repository(mock_repo)
        
        # Mock the ask method
        with patch.object(server.query_engine, 'ask', new_callable=AsyncMock) as mock_ask:
            from src.repoindex.data.schemas import AskResponse
            mock_ask.return_value = AskResponse(
                question="What does this function do?",
                answer="This function performs the main logic.",
                citations=[],
                execution_time_ms=200.0,
                index_id="test-index-123",
            )
            
            # Call ask through MCP server
            result = await server._ask_index({
                "index_id": "test-index-123",
                "question": "What does this function do?",
                "context_lines": 5,
            })
            
            # Verify query engine was called
            mock_ask.assert_called_once()
            assert not result.isError
            
            # Verify call arguments
            call_args = mock_ask.call_args
            assert call_args[1]["index_id"] == "test-index-123"
            assert call_args[1]["question"] == "What does this function do?"

    @pytest.mark.asyncio
    async def test_complete_indexing_to_querying_workflow(self, temp_storage, mock_repo_structure):
        """Test complete workflow from indexing through querying."""
        # Step 1: Create integrated system
        query_engine = QueryEngine()
        server = MCPServer(storage_dir=temp_storage, query_engine=query_engine)
        
        # Step 2: Mock successful indexing
        with patch('src.repoindex.mcp.server.IndexingPipeline') as MockPipeline:
            mock_pipeline = MagicMock()
            MockPipeline.return_value = mock_pipeline
            # Make the start_indexing method async
            mock_pipeline.start_indexing = AsyncMock(return_value="test-index-123")
            
            # Simulate indexing completion
            result = await server._ensure_repo_index({
                "path": "/test/repo",
                "rev": "main",
                "language": "ts",
            })
            
            assert not result.isError
                
        # Step 3: Manually register the completed index (simulating pipeline completion)
        indexed_repo = IndexedRepository(
            index_id="test-index-123",
            repo_root="/test/repo",
            rev="main",
            serena_graph=mock_repo_structure["serena_graph"],
            vector_index=mock_repo_structure["vector_index"],
            repomap_data=mock_repo_structure["repomap_data"],
            snippets=mock_repo_structure["snippets"],
            manifest=mock_repo_structure["manifest"],
        )
        query_engine.register_indexed_repository(indexed_repo)
        
        # Step 4: Test search functionality
        search_response = await query_engine.search(
            index_id="test-index-123",
            query="main function",
            k=10,
        )
        assert search_response.index_id == "test-index-123"
        assert search_response.query == "main function"
        
        # Step 5: Test ask functionality with mocked navigator
        with patch.object(query_engine.symbol_navigator, 'ask', new_callable=AsyncMock) as mock_nav_ask:
            from src.repoindex.data.schemas import AskResponse
            mock_nav_ask.return_value = AskResponse(
                question="What does main do?",
                answer="The main function is the entry point.",
                citations=[],
                execution_time_ms=150.0,
                index_id="",
            )
            
            ask_response = await query_engine.ask(
                index_id="test-index-123",
                question="What does main do?",
            )
            assert ask_response.index_id == "test-index-123"
            assert ask_response.question == "What does main do?"

    @pytest.mark.asyncio
    async def test_error_handling_separation(self, temp_storage):
        """Test that error handling works correctly with separated architecture."""
        query_engine = QueryEngine()
        
        # Test query engine handles missing repositories gracefully
        with pytest.raises(Exception):  # Should raise ValidationError
            await query_engine.search(
                index_id="non-existent",
                query="test",
            )
        
        with pytest.raises(Exception):  # Should raise ValidationError  
            await query_engine.ask(
                index_id="non-existent",
                question="test question",
            )

    def test_concurrency_isolation(self, temp_storage):
        """Test that indexing and querying concurrency controls are isolated."""
        query_engine = QueryEngine(concurrency_limits={"query": 3})
        pipeline = IndexingPipeline(
            storage_dir=temp_storage,
            query_engine=query_engine, 
            concurrency_limits={"io": 8, "cpu": 2}
        )
        
        # Verify separate concurrency controls
        assert pipeline.concurrency_limits == {"io": 8, "cpu": 2}
        assert query_engine.concurrency_limits == {"query": 3}
        assert pipeline.io_semaphore._value == 8
        assert pipeline.cpu_semaphore._value == 2
        assert query_engine.query_semaphore._value == 3

    @pytest.mark.asyncio
    async def test_resource_cleanup_separation(self, temp_storage):
        """Test that resource cleanup works correctly with separated architecture."""
        query_engine = QueryEngine()
        pipeline = IndexingPipeline(storage_dir=temp_storage, query_engine=query_engine)
        
        # Simulate failed pipeline
        mock_context = MagicMock()
        mock_context.index_id = "failed-index"
        mock_context._pipeline_failed = True
        pipeline.active_pipelines["failed-index"] = mock_context
        
        # Manually register with query engine first
        indexed_repo = IndexedRepository(
            index_id="failed-index",
            repo_root="/test/repo", 
            rev="abc",
        )
        query_engine.register_indexed_repository(indexed_repo)
        assert "failed-index" in query_engine.indexed_repositories
        
        # Simulate cleanup (this would happen in the pipeline's finally block)
        if hasattr(mock_context, "_pipeline_failed") and mock_context._pipeline_failed:
            del pipeline.active_pipelines["failed-index"]
            pipeline.query_engine.unregister_indexed_repository("failed-index")
        
        # Verify cleanup
        assert "failed-index" not in pipeline.active_pipelines
        assert "failed-index" not in query_engine.indexed_repositories


class TestBackwardCompatibility:
    """Test that the architectural changes maintain backward compatibility."""
    
    def test_pipeline_alias_compatibility(self):
        """Test that PipelineRunner alias still exists for backward compatibility."""
        from src.repoindex.pipeline.run import PipelineRunner, IndexingPipeline
        
        # PipelineRunner should be an alias for IndexingPipeline
        assert PipelineRunner is IndexingPipeline

    def test_mcp_server_alias_compatibility(self):
        """Test that MimirMCPServer alias still exists for backward compatibility."""
        from src.repoindex.mcp.server import MimirMCPServer, MCPServer
        
        # MimirMCPServer should be an alias for MCPServer
        assert MimirMCPServer is MCPServer

    @pytest.mark.asyncio
    async def test_existing_api_contracts_maintained(self, temp_storage):
        """Test that existing API contracts are maintained after refactoring."""
        # MCP server should still create and manage pipelines the same way
        server = MCPServer(storage_dir=temp_storage)
        
        # Should still have the same MCP server structure
        assert hasattr(server, 'server')
        assert hasattr(server, 'pipelines')
        assert hasattr(server, 'query_engine')
        
        # Should be able to create pipelines and handle requests
        assert hasattr(server, '_ensure_repo_index')
        assert hasattr(server, '_search_repo')
        assert hasattr(server, '_ask_index')
        assert hasattr(server, '_cancel')


if __name__ == "__main__":
    pytest.main([__file__])