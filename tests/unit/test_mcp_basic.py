"""
Basic unit tests for MCP server functionality.

Tests the MCP server in isolation without full integration dependencies.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

import pytest

from src.repoindex.mcp.server import MCPServer
from src.repoindex.data.schemas import (
    EnsureRepoIndexResponse,
    IndexState,
    PipelineStatus
)


@pytest.mark.asyncio
async def test_mcp_server_initialization():
    """Test MCP server can be initialized."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        server = MCPServer(storage_dir=storage_dir)
        
        assert server.storage_dir == storage_dir
        assert server.indexes_dir == storage_dir / "indexes"
        assert server.indexes_dir.exists()
        assert server.pipelines == {}


@pytest.mark.asyncio
async def test_ensure_repo_index_response_format():
    """Test that ensure_repo_index returns the correct response format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        server = MCPServer(storage_dir=storage_dir)
        
        # Create a test repository
        repo_dir = Path(tmpdir) / "test_repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()  # Simulate git repo
        (repo_dir / "test.py").write_text("print('hello')")
        
        # Mock the pipeline to avoid full execution
        with patch('src.repoindex.mcp.server.IndexingPipeline') as MockPipeline:
            mock_pipeline = Mock()
            mock_pipeline.start_indexing = AsyncMock(return_value="test_index_123")
            MockPipeline.return_value = mock_pipeline
            
            # Call ensure_repo_index
            result = await server._ensure_repo_index({
                "path": str(repo_dir),
                "language": "py"
            })
            
            # Check result format
            assert result.isError is False
            content = json.loads(result.content[0].text)
            
            # Check response structure matches EnsureRepoIndexResponse
            assert "index_id" in content
            assert content["index_id"] == "test_index_123"
            assert "status_uri" in content
            assert content["status_uri"] == "mimir://indexes/test_index_123/status.json"
            assert "manifest_uri" in content
            assert content["manifest_uri"] == "mimir://indexes/test_index_123/manifest.json"


@pytest.mark.asyncio
async def test_mcp_server_error_handling():
    """Test MCP server handles errors gracefully."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        server = MCPServer(storage_dir=storage_dir)
        
        # Mock pipeline to raise an error
        with patch('src.repoindex.mcp.server.IndexingPipeline') as MockPipeline:
            mock_pipeline = Mock()
            mock_pipeline.start_indexing = AsyncMock(side_effect=Exception("Test error"))
            MockPipeline.return_value = mock_pipeline
            
            # Call with invalid path
            result = await server._ensure_repo_index({
                "path": "/nonexistent/path"
            })
            
            # Should not crash but return error
            assert result.isError is True
            assert "Error" in result.content[0].text


@pytest.mark.asyncio
async def test_search_repo_requires_index():
    """Test that search_repo requires a valid index_id."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        server = MCPServer(storage_dir=storage_dir)
        
        # Try to search without a valid index
        result = await server._search_repo({
            "index_id": "nonexistent_index",
            "query": "test query"
        })
        
        # Should return an error
        assert result.isError is True
        assert "not found" in result.content[0].text.lower() or "error" in result.content[0].text.lower()


@pytest.mark.asyncio  
async def test_cancel_operation():
    """Test canceling an indexing operation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        server = MCPServer(storage_dir=storage_dir)
        
        # Create a mock pipeline
        mock_pipeline = Mock()
        mock_pipeline.cancel = AsyncMock(return_value=True)
        server.pipelines["test_index"] = mock_pipeline
        
        # Cancel the operation
        result = await server._cancel({
            "index_id": "test_index"
        })
        
        # Check it was cancelled
        assert result.isError is False
        content = json.loads(result.content[0].text)
        assert content["ok"] is True
        
        # Verify cancel was called
        mock_pipeline.cancel.assert_called_once()


@pytest.mark.asyncio
async def test_get_repo_bundle_requires_complete_index():
    """Test that get_repo_bundle requires a completed index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        storage_dir = Path(tmpdir)
        server = MCPServer(storage_dir=storage_dir)
        
        # Create index directory but no bundle file
        index_dir = storage_dir / "indexes" / "test_index"
        index_dir.mkdir(parents=True)
        
        # Try to get bundle
        result = await server._get_repo_bundle({
            "index_id": "test_index"
        })
        
        # Should return error since bundle doesn't exist
        assert result.isError is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])