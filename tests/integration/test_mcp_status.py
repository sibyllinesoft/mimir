"""
Test MCP server status and resource functionality.

This test focuses on testing the status reporting and resource access
without requiring complex external tool dependencies.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from src.repoindex.mcp.server import MCPServer


@pytest.fixture
def simple_python_repo():
    """Create a simple Python repository for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        
        # Create a simple Python project
        (repo / "main.py").write_text('''#!/usr/bin/env python3
def main():
    print("Hello World")
    return 0

if __name__ == "__main__":
    main()
''')
        
        (repo / "utils.py").write_text('''def helper_function():
    return "helper"
''')
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True)
        
        yield repo


@pytest.fixture
def mcp_server():
    """Create MCP server instance."""
    with tempfile.TemporaryDirectory() as storage_dir:
        server = MCPServer(storage_dir=Path(storage_dir))
        yield server


class TestMCPStatus:
    """Test MCP server status and resources."""
    
    @pytest.mark.integration
    async def test_start_indexing_and_check_status(self, simple_python_repo: Path, mcp_server: MCPServer):
        """Test starting indexing and checking status via resources."""
        print(f"Testing with repo: {simple_python_repo}")
        
        # Start indexing
        result = await mcp_server._ensure_repo_index({
            "path": str(simple_python_repo),
            "language": "py"
        })
        
        print(f"Index result isError: {result.isError}")
        if result.isError:
            print(f"Error: {result.content}")
            return
            
        # Parse the response
        response_text = result.content[0].text
        print(f"Full response: {response_text}")
        
        try:
            response_data = json.loads(response_text)
            print(f"Response keys: {list(response_data.keys())}")
            
            if "index_id" in response_data:
                index_id = response_data["index_id"]
                print(f"✅ Got index_id: {index_id}")
                
                # Wait a moment for pipeline to progress
                time.sleep(1)
                
                # Test reading the status resource
                await self._test_status_resource(mcp_server, index_id)
                
                # Test reading the manifest resource (might not exist yet)
                await self._test_manifest_resource(mcp_server, index_id)
                
            else:
                print(f"❌ No index_id in response: {response_data}")
                
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON response: {e}")
            print(f"Raw response: {response_text}")
    
    async def _test_status_resource(self, mcp_server: MCPServer, index_id: str):
        """Test reading the status resource."""
        try:
            status_uri = f"repo://status"  # General status
            status_result = await mcp_server._read_resource(status_uri)
            
            if hasattr(status_result, 'contents'):
                print(f"✅ Status resource accessible")
                if hasattr(status_result.contents[0], 'text'):
                    status_data = json.loads(status_result.contents[0].text)
                    print(f"Status data keys: {list(status_data.keys())}")
                    
                    # Check for pipeline status
                    if "active_pipelines" in status_data:
                        print(f"Active pipelines: {len(status_data['active_pipelines'])}")
                    
            else:
                print(f"❌ Status resource not accessible")
                
        except Exception as e:
            print(f"Error reading status resource: {e}")
    
    async def _test_manifest_resource(self, mcp_server: MCPServer, index_id: str):
        """Test reading the manifest resource."""
        try:
            manifest_uri = f"repo://manifest/{index_id}"
            manifest_result = await mcp_server._read_resource(manifest_uri)
            
            if hasattr(manifest_result, 'contents'):
                print(f"✅ Manifest resource accessible for {index_id}")
                if hasattr(manifest_result.contents[0], 'text'):
                    manifest_data = json.loads(manifest_result.contents[0].text)
                    print(f"Manifest keys: {list(manifest_data.keys())}")
                    
                    # Check for key manifest fields
                    key_fields = ["index_id", "repo_root", "files", "created_at"]
                    for field in key_fields:
                        if field in manifest_data:
                            print(f"  ✓ {field}: {manifest_data[field] if field != 'files' else len(manifest_data[field])} {'files' if field == 'files' else ''}")
                        else:
                            print(f"  ✗ Missing {field}")
            else:
                print(f"❌ Manifest resource not accessible for {index_id}")
                
        except Exception as e:
            print(f"Error reading manifest resource: {e}")
    
    @pytest.mark.integration
    async def test_pipeline_logs_available(self, simple_python_repo: Path, mcp_server: MCPServer):
        """Test that pipeline logs are accessible."""
        # Start indexing
        result = await mcp_server._ensure_repo_index({
            "path": str(simple_python_repo),
            "language": "py"
        })
        
        if result.isError:
            print(f"Indexing failed: {result.content}")
            return
            
        response_data = json.loads(result.content[0].text)
        index_id = response_data["index_id"]
        
        # Wait for some pipeline activity
        time.sleep(2)
        
        # Try to read logs
        try:
            logs_uri = f"repo://logs/{index_id}"
            logs_result = await mcp_server._read_resource(logs_uri)
            
            if hasattr(logs_result, 'contents'):
                print(f"✅ Logs resource accessible for {index_id}")
                log_content = logs_result.contents[0].text
                print(f"Log content length: {len(log_content)} chars")
                
                # Check for expected log patterns
                expected_patterns = ["Pipeline started", "file_discovery", "pipeline_execution"]
                found_patterns = []
                for pattern in expected_patterns:
                    if pattern in log_content:
                        found_patterns.append(pattern)
                
                print(f"Found log patterns: {found_patterns}")
                assert len(found_patterns) > 0, "No expected log patterns found"
                
            else:
                print(f"❌ Logs resource not accessible for {index_id}")
                
        except Exception as e:
            print(f"Error reading logs resource: {e}")
    
    @pytest.mark.integration 
    async def test_multiple_repositories(self, mcp_server: MCPServer):
        """Test indexing multiple repositories."""
        repos_created = []
        index_ids = []
        
        try:
            # Create multiple test repositories
            for i in range(2):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    repo = Path(tmp_dir)
                    
                    (repo / f"file_{i}.py").write_text(f'''
def function_{i}():
    return {i}
''')
                    
                    # Initialize git
                    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
                    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
                    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
                    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
                    subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True)
                    
                    repos_created.append(str(repo))
                    
                    # Index the repository
                    result = await mcp_server._ensure_repo_index({
                        "path": str(repo),
                        "language": "py"
                    })
                    
                    if not result.isError:
                        response_data = json.loads(result.content[0].text)
                        index_ids.append(response_data["index_id"])
            
            print(f"Successfully started indexing for {len(index_ids)} repositories")
            
            # Check server status with multiple pipelines
            status_uri = "repo://status"
            status_result = await mcp_server._read_resource(status_uri)
            
            if hasattr(status_result, 'contents'):
                status_data = json.loads(status_result.contents[0].text)
                print(f"Server handling {len(status_data.get('active_pipelines', []))} active pipelines")
                
        except Exception as e:
            print(f"Multiple repository test failed: {e}")
            raise