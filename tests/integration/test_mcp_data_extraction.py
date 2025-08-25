"""
Test what data the MCP server can extract without external tool dependencies.

This test focuses on verifying what file discovery and basic analysis
the system can perform before hitting external tool requirements.
"""

import json
import subprocess  
import tempfile
import time
from pathlib import Path

import pytest

from src.repoindex.mcp.server import MCPServer


@pytest.fixture
def rich_python_repo():
    """Create a Python repository with diverse code patterns."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        
        # Create main application file
        (repo / "app.py").write_text('''#!/usr/bin/env python3
"""Main application entry point."""

import sys
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict

from utils.config import Config
from data.processor import DataProcessor
from models.ml_model import MLModel

@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    data_dir: str = "./data"

class Application:
    """Main application class."""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.processor = DataProcessor()
        self.model = MLModel()
        self.logger = logging.getLogger(__name__)
    
    async def run(self) -> int:
        """Run the application."""
        self.logger.info("Starting application")
        
        try:
            data = await self.processor.load_data(self.config.data_dir)
            results = await self.model.predict(data)
            
            self.logger.info(f"Processed {len(results)} predictions")
            return 0
            
        except Exception as e:
            self.logger.error(f"Application failed: {e}")
            return 1

def main() -> int:
    """Main entry point."""
    config = AppConfig(debug=True)
    app = Application(config)
    return asyncio.run(app.run())

if __name__ == "__main__":
    sys.exit(main())
''')
        
        # Create utility modules
        (repo / "utils").mkdir()
        (repo / "utils" / "__init__.py").write_text("")
        (repo / "utils" / "config.py").write_text('''"""Configuration management."""

import os
from typing import Dict, Any

class Config:
    """Configuration manager."""
    
    def __init__(self):
        self.settings: Dict[str, Any] = {}
    
    def load_from_env(self) -> None:
        """Load settings from environment variables."""
        self.settings["debug"] = os.getenv("DEBUG", "false").lower() == "true"
        self.settings["log_level"] = os.getenv("LOG_LEVEL", "INFO")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.settings.get(key, default)
''')
        
        # Create data processing module
        (repo / "data").mkdir()
        (repo / "data" / "__init__.py").write_text("")
        (repo / "data" / "processor.py").write_text('''"""Data processing utilities."""

import asyncio
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

class DataProcessor:
    """Handles data loading and preprocessing."""
    
    def __init__(self):
        self.cache: Dict[str, Any] = {}
    
    async def load_data(self, data_dir: str) -> pd.DataFrame:
        """Load data from directory."""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # Simulate async data loading
        await asyncio.sleep(0.1)
        
        # Mock data
        data = pd.DataFrame({
            "feature_1": [1, 2, 3, 4, 5],
            "feature_2": [0.1, 0.2, 0.3, 0.4, 0.5],
            "target": [0, 1, 0, 1, 0]
        })
        
        self.cache["last_loaded"] = data
        return data
    
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data."""
        # Simple preprocessing
        processed = data.copy()
        processed["feature_1_normalized"] = processed["feature_1"] / processed["feature_1"].max()
        return processed
''')
        
        # Create ML model module
        (repo / "models").mkdir()
        (repo / "models" / "__init__.py").write_text("")
        (repo / "models" / "ml_model.py").write_text('''"""Machine learning model."""

import numpy as np
import pandas as pd
from typing import List
from sklearn.linear_model import LogisticRegression

class MLModel:
    """Machine learning model wrapper."""
    
    def __init__(self):
        self.model = LogisticRegression()
        self.trained = False
    
    async def train(self, data: pd.DataFrame) -> None:
        """Train the model."""
        X = data.drop("target", axis=1)
        y = data["target"]
        
        self.model.fit(X, y)
        self.trained = True
    
    async def predict(self, data: pd.DataFrame) -> List[int]:
        """Make predictions."""
        if not self.trained:
            raise ValueError("Model not trained")
        
        X = data.drop("target", axis=1) if "target" in data.columns else data
        predictions = self.model.predict(X)
        return predictions.tolist()
    
    def get_feature_importance(self) -> dict:
        """Get feature importance."""
        if not self.trained:
            return {}
        
        feature_names = [f"feature_{i}" for i in range(len(self.model.coef_[0]))]
        return dict(zip(feature_names, self.model.coef_[0]))
''')
        
        # Create test files
        (repo / "tests").mkdir()
        (repo / "tests" / "__init__.py").write_text("")
        (repo / "tests" / "test_processor.py").write_text('''"""Test data processor."""

import pytest
from data.processor import DataProcessor

@pytest.mark.asyncio
async def test_data_loading():
    """Test data loading functionality."""
    processor = DataProcessor()
    
    # This would fail in real test but shows intent
    # data = await processor.load_data("./test_data")
    # assert len(data) > 0
    
def test_cache_initialization():
    """Test processor cache."""
    processor = DataProcessor()
    assert isinstance(processor.cache, dict)
''')
        
        # Create requirements and config files
        (repo / "requirements.txt").write_text('''pandas>=1.5.0
scikit-learn>=1.2.0
numpy>=1.20.0
pytest>=7.0.0
pytest-asyncio>=0.20.0
''')
        
        (repo / "pyproject.toml").write_text('''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "test-ml-app"
version = "0.1.0"
description = "Test ML application"
requires-python = ">=3.8"

[tool.pytest.ini_options]
asyncio_mode = "auto"
''')
        
        (repo / "README.md").write_text('''# Test ML Application

A test machine learning application with modular structure.

## Structure

- `app.py`: Main application
- `utils/`: Configuration utilities
- `data/`: Data processing modules  
- `models/`: ML model implementations
- `tests/`: Test suite

## Features

- Async data loading
- Configurable preprocessing
- ML model training and prediction
- Comprehensive test coverage
''')
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial ML app structure"], cwd=repo, check=True, capture_output=True)
        
        yield repo


@pytest.fixture
def mcp_server():
    """Create MCP server instance."""
    with tempfile.TemporaryDirectory() as storage_dir:
        server = MCPServer(storage_dir=Path(storage_dir))
        yield server


class TestMCPDataExtraction:
    """Test data extraction capabilities of MCP server."""
    
    @pytest.mark.integration
    async def test_file_discovery_data(self, rich_python_repo: Path, mcp_server: MCPServer):
        """Test what file discovery captures."""
        print(f"Testing with rich Python repo: {rich_python_repo}")
        
        # List the files we created
        all_files = list(rich_python_repo.rglob("*"))
        python_files = [f for f in all_files if f.suffix == ".py"]
        print(f"Created {len(python_files)} Python files:")
        for f in python_files:
            print(f"  - {f.relative_to(rich_python_repo)}")
        
        # Start indexing
        result = await mcp_server._ensure_repo_index({
            "path": str(rich_python_repo),
            "language": "py"
        })
        
        print(f"Index result isError: {result.isError}")
        if result.isError:
            print(f"Error: {result.content}")
            return
        
        response_data = json.loads(result.content[0].text)
        index_id = response_data["index_id"]
        print(f"Index ID: {index_id}")
        
        # Wait for pipeline to run
        time.sleep(3)
        
        # Check what files were discovered
        await self._check_discovered_files(mcp_server, index_id)
        
        # Check status and logs
        await self._check_pipeline_status(mcp_server, index_id)
    
    async def _check_discovered_files(self, mcp_server: MCPServer, index_id: str):
        """Check what files were discovered by the pipeline."""
        try:
            # Check if manifest exists (may not be created if pipeline fails)
            manifest_uri = f"repo://manifest/{index_id}"
            manifest_result = await mcp_server._read_resource(manifest_uri)
            
            if hasattr(manifest_result, 'contents') and manifest_result.contents:
                manifest_data = json.loads(manifest_result.contents[0].text)
                print(f"✅ Manifest available - found {len(manifest_data.get('files', []))} files")
                
                # Show discovered files
                if "files" in manifest_data:
                    for file_info in manifest_data["files"][:5]:  # Show first 5
                        print(f"  📄 {file_info.get('path', 'unknown')}")
                        if len(manifest_data["files"]) > 5:
                            print(f"  ... and {len(manifest_data['files']) - 5} more files")
                            break
                            
            else:
                print("❌ Manifest not available (pipeline may have failed)")
                
        except Exception as e:
            print(f"Error checking manifest: {e}")
    
    async def _check_pipeline_status(self, mcp_server: MCPServer, index_id: str):
        """Check pipeline status and logs."""
        try:
            # Read logs to see what happened
            logs_uri = f"repo://logs/{index_id}"
            logs_result = await mcp_server._read_resource(logs_uri)
            
            if hasattr(logs_result, 'contents') and logs_result.contents:
                log_content = logs_result.contents[0].text
                print(f"📄 Pipeline logs ({len(log_content)} chars):")
                
                # Show relevant log lines
                log_lines = log_content.split('\n')
                relevant_lines = []
                
                for line in log_lines:
                    if any(keyword in line.lower() for keyword in [
                        'error', 'failed', 'exception', 'completed', 'stage', 'discovered'
                    ]):
                        relevant_lines.append(line.strip())
                
                for line in relevant_lines[-10:]:  # Last 10 relevant lines
                    print(f"  🔍 {line}")
                    
                # Analyze what stages were reached
                stages_mentioned = []
                for line in log_lines:
                    if "Starting" in line or "starting" in line:
                        stages_mentioned.append(line.strip())
                
                print(f"\n📊 Pipeline stages reached: {len(stages_mentioned)}")
                for stage in stages_mentioned:
                    print(f"  ✓ {stage}")
                    
            else:
                print("❌ No logs available")
                
        except Exception as e:
            print(f"Error checking logs: {e}")
    
    @pytest.mark.integration
    async def test_server_handles_missing_tools(self, rich_python_repo: Path, mcp_server: MCPServer):
        """Test how server handles missing external tools."""
        print("Testing server behavior with missing external tools...")
        
        # Start indexing and immediately check status
        result = await mcp_server._ensure_repo_index({
            "path": str(rich_python_repo),
            "language": "py" 
        })
        
        if result.isError:
            print(f"❌ Indexing failed immediately: {result.content}")
            return
        
        response_data = json.loads(result.content[0].text)
        index_id = response_data["index_id"]
        
        # Check status quickly to see pipeline state
        status_uri = "repo://status"
        status_result = await mcp_server._read_resource(status_uri)
        
        if hasattr(status_result, 'contents'):
            status_data = json.loads(status_result.contents[0].text)
            print(f"Server status: {len(status_data.get('active_pipelines', []))} active pipelines")
            
            # Check individual pipeline status
            if status_data.get('active_pipelines'):
                for pipeline_info in status_data['active_pipelines']:
                    if isinstance(pipeline_info, dict):
                        print(f"  Pipeline {pipeline_info.get('index_id', 'unknown')}: {pipeline_info.get('status', 'unknown')}")
                    else:
                        print(f"  Pipeline ID: {pipeline_info}")
        
        # Wait and check final status
        time.sleep(5) 
        
        # Re-check status
        final_status_result = await mcp_server._read_resource(status_uri)
        if hasattr(final_status_result, 'contents'):
            final_status_data = json.loads(final_status_result.contents[0].text)
            print(f"Final status: {len(final_status_data.get('active_pipelines', []))} active pipelines")
            
            if len(final_status_data.get('active_pipelines', [])) == 0:
                print("✅ Pipeline completed (successfully or with failure)")
            else:
                print("⚠️ Pipeline still running")
        
        # Check the logs to see what failed
        await self._check_pipeline_status(mcp_server, index_id)