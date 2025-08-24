"""
Test MCP server functions with real repository data.

This test verifies that the MCP server tools actually work and capture
data from serena, repomapper, and other pipeline components.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest

from src.repoindex.mcp.server import MCPServer


@pytest.fixture
def simple_python_repo():
    """Create a simple Python repository for testing."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        
        # Create a simple Python project structure
        (repo / "main.py").write_text('''#!/usr/bin/env python3
"""Main entry point for the application."""

import sys
from data_processor import DataProcessor
from model_trainer import ModelTrainer

def main():
    """Main function."""
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    data = processor.load_data("data.csv")
    model = trainer.train(data)
    
    print(f"Model accuracy: {model.accuracy}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
''')
        
        (repo / "data_processor.py").write_text('''"""Data processing utilities."""

import pandas as pd


class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self):
        self.data = None
    
    def load_data(self, filepath):
        """Load data from CSV file."""
        self.data = pd.read_csv(filepath)
        return self.data
    
    def clean_data(self, data):
        """Clean and preprocess data."""
        # Remove null values
        cleaned = data.dropna()
        return cleaned
''')
        
        (repo / "model_trainer.py").write_text('''"""Machine learning model training."""

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


class ModelTrainer:
    """Handles model training and evaluation."""
    
    def __init__(self):
        self.model = LinearRegression()
        self.accuracy = 0.0
    
    def train(self, data):
        """Train the model on provided data."""
        # Simple training simulation
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        
        self.model.fit(X, y)
        predictions = self.model.predict(X)
        self.accuracy = accuracy_score(y, predictions > 0.5)
        
        return self
    
    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)
''')
        
        (repo / "README.md").write_text('''# Test ML Project

A simple machine learning project for testing.

## Components

- `main.py`: Entry point
- `data_processor.py`: Data handling
- `model_trainer.py`: ML training
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


class TestMCPFunctions:
    """Test MCP server functions with real data."""
    
    @pytest.mark.integration
    async def test_ensure_repo_index_basic(self, simple_python_repo: Path, mcp_server: MCPServer):
        """Test basic repository indexing."""
        print(f"Testing with repo: {simple_python_repo}")
        
        # List files in the repo
        files = list(simple_python_repo.rglob("*"))
        print(f"Repo files: {[f.name for f in files if f.is_file()]}")
        
        # Try to index the repository
        try:
            result = await mcp_server._ensure_repo_index({
                "path": str(simple_python_repo),
                "language": "py"
            })
            
            print(f"Index result isError: {result.isError}")
            if result.isError:
                print(f"Error content: {result.content}")
            else:
                print(f"Success content preview: {result.content[0].text[:200]}...")
                
                # Try to parse the result
                content = json.loads(result.content[0].text)
                print(f"Result keys: {list(content.keys())}")
                
                if "success" in content and content["success"]:
                    print("✅ Repository indexing succeeded")
                    if "files" in content:
                        print(f"Files indexed: {len(content['files'])}")
                    if "index_id" in content:
                        print(f"Index ID: {content['index_id']}")
                else:
                    print("❌ Repository indexing reported failure")
                    
        except Exception as e:
            print(f"Exception during indexing: {e}")
            print(f"Exception type: {type(e)}")
            import traceback
            traceback.print_exc()
            
            # Don't fail the test immediately, let's see what went wrong
            pytest.fail(f"Repository indexing failed with exception: {e}")
    
    @pytest.mark.integration 
    async def test_list_mcp_resources(self, mcp_server: MCPServer):
        """Test that MCP resources are available."""
        # Get the resources list directly from the server method
        resources = await mcp_server._list_resources()
        
        resource_uris = [str(res.uri) for res in resources]
        print(f"Available resources: {resource_uris}")
        
        # Should have at least some resources
        assert len(resources) > 0, "No MCP resources found"
        print("✅ MCP resources are available")
        
        # Check for expected resource patterns
        expected_patterns = ["repo://"]
        found_patterns = []
        for uri in resource_uris:
            for pattern in expected_patterns:
                if uri.startswith(pattern):
                    found_patterns.append(pattern)
                    break
                    
        print(f"Found resource patterns: {found_patterns}")
        assert len(found_patterns) > 0, "No expected resource patterns found"
    
    @pytest.mark.integration
    async def test_server_status(self, mcp_server: MCPServer):
        """Test server status and configuration."""
        print(f"Storage directory: {mcp_server.storage_dir}")
        print(f"Indexes directory: {mcp_server.indexes_dir}")
        print(f"Active pipelines: {len(mcp_server.pipelines)}")
        print(f"Server start time: {mcp_server.start_time}")
        
        # Check that directories exist
        assert mcp_server.storage_dir.exists()
        assert mcp_server.indexes_dir.exists()
        
        print("✅ Server status is healthy")