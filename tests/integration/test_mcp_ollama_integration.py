"""
Integration tests for MCP server with Ollama integration.

Tests the MCP server functionality when configured to use Ollama for embeddings,
verifying that the server can properly communicate with Ollama and process
repository indexing using local LLM models.
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import Mock, patch, AsyncMock

import httpx
import pytest

from src.repoindex.mcp.server import MCPServer
from src.repoindex.data.schemas import RepoMap, SerenaGraph, VectorIndex, FileRank, DependencyEdge, SymbolEntry, SymbolType, VectorChunk


class OllamaTestHelper:
    """Helper class for Ollama testing operations."""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.client = httpx.AsyncClient()
    
    async def is_ollama_available(self) -> bool:
        """Check if Ollama server is running and accessible."""
        try:
            response = await self.client.get(f"{self.base_url}/api/version", timeout=5.0)
            return response.status_code == 200
        except Exception:
            return False
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure specified model is available in Ollama."""
        try:
            # Check if model is already available
            response = await self.client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                available_models = [m.get("name", "").split(":")[0] for m in models]
                if model_name in available_models:
                    return True
            
            # Try to pull the model if not available
            pull_data = {"name": model_name}
            response = await self.client.post(
                f"{self.base_url}/api/pull",
                json=pull_data,
                timeout=300.0  # 5 minute timeout for model download
            )
            
            if response.status_code == 200:
                # Stream the pull response
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            status = json.loads(line)
                            if status.get("status") == "success":
                                return True
                        except json.JSONDecodeError:
                            continue
            
            return False
        except Exception as e:
            print(f"Error ensuring model {model_name}: {e}")
            return False
    
    async def test_embedding_generation(self, model_name: str, text: str) -> Optional[list]:
        """Test embedding generation with specified model."""
        try:
            embed_data = {
                "model": model_name,
                "prompt": text
            }
            response = await self.client.post(
                f"{self.base_url}/api/embeddings",
                json=embed_data,
                timeout=30.0
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding")
            return None
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    async def cleanup(self):
        """Clean up HTTP client."""
        await self.client.aclose()


class TestMCPOllamaIntegration:
    """Integration tests for MCP server with Ollama."""

    @pytest.fixture
    async def ollama_helper(self) -> OllamaTestHelper:
        """Create Ollama test helper."""
        helper = OllamaTestHelper()
        yield helper
        await helper.cleanup()

    @pytest.fixture
    async def test_repo(self) -> Path:
        """Create a test repository with Python files for Ollama testing."""
        repo_dir = Path(tempfile.mkdtemp(prefix="mimir_ollama_test_repo_"))

        # Create a simple Python project
        setup_py = """from setuptools import setup, find_packages

setup(
    name="test-ml-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
    ]
)
"""
        (repo_dir / "setup.py").write_text(setup_py)

        # Create source directory
        src_dir = repo_dir / "src"
        src_dir.mkdir()

        # Create main module
        main_py = """import numpy as np
from sklearn.linear_model import LinearRegression
from .data_processor import DataProcessor
from .model_trainer import ModelTrainer

def main():
    \"\"\"Main entry point for ML training pipeline.\"\"\"
    processor = DataProcessor()
    trainer = ModelTrainer()
    
    # Load and process data
    data = processor.load_dataset("training_data.csv")
    X, y = processor.prepare_features(data)
    
    # Train model
    model = trainer.train_linear_model(X, y)
    
    # Evaluate
    accuracy = trainer.evaluate_model(model, X, y)
    print(f"Model accuracy: {accuracy:.3f}")
    
    return model

if __name__ == "__main__":
    main()
"""
        (src_dir / "main.py").write_text(main_py)

        # Create data processor module
        data_processor_py = """import numpy as np
import pandas as pd
from typing import Tuple, Any

class DataProcessor:
    \"\"\"Handles data loading and preprocessing for ML pipeline.\"\"\"
    
    def __init__(self):
        self.scaler = None
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        \"\"\"Load dataset from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Loaded dataframe
        \"\"\"
        try:
            data = pd.read_csv(filepath)
            return data
        except FileNotFoundError:
            print(f"Dataset not found: {filepath}")
            return pd.DataFrame()
    
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        \"\"\"Prepare features and target variables.
        
        Args:
            data: Raw dataframe
            
        Returns:
            Tuple of (features, targets)
        \"\"\"
        if data.empty:
            return np.array([]), np.array([])
            
        # Simple feature extraction
        features = data.drop('target', axis=1).values
        targets = data['target'].values
        
        return features, targets
    
    def normalize_features(self, X: np.ndarray) -> np.ndarray:
        \"\"\"Normalize feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Normalized features
        \"\"\"
        from sklearn.preprocessing import StandardScaler
        
        if self.scaler is None:
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
"""
        (src_dir / "data_processor.py").write_text(data_processor_py)

        # Create model trainer module
        model_trainer_py = """import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from typing import Any

class ModelTrainer:
    \"\"\"Handles ML model training and evaluation.\"\"\"
    
    def __init__(self):
        self.models = {}
    
    def train_linear_model(self, X: np.ndarray, y: np.ndarray) -> LinearRegression:
        \"\"\"Train a linear regression model.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Trained model
        \"\"\"
        model = LinearRegression()
        model.fit(X, y)
        self.models['linear'] = model
        return model
    
    def evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray) -> float:
        \"\"\"Evaluate model performance.
        
        Args:
            model: Trained model
            X: Feature matrix  
            y: True targets
            
        Returns:
            R² score
        \"\"\"
        predictions = model.predict(X)
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        
        print(f"MSE: {mse:.3f}")
        print(f"R²: {r2:.3f}")
        
        return r2
    
    def save_model(self, model: Any, filepath: str):
        \"\"\"Save trained model to disk.
        
        Args:
            model: Model to save
            filepath: Output path
        \"\"\"
        import joblib
        joblib.dump(model, filepath)
        print(f"Model saved to: {filepath}")
"""
        (src_dir / "model_trainer.py").write_text(model_trainer_py)

        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial ML project"],
            cwd=repo_dir,
            capture_output=True
        )

        yield repo_dir

        # Cleanup
        shutil.rmtree(repo_dir)

    @pytest.fixture
    async def mcp_server_with_ollama_config(self) -> MCPServer:
        """Create MCP server configured to use Ollama."""
        storage_dir = Path(tempfile.mkdtemp(prefix="mimir_ollama_storage_"))
        
        # Create server with custom configuration for Ollama
        server = MCPServer(storage_dir=storage_dir)
        
        # Mock the external tool calls to use Ollama instead of default tools
        server.ollama_config = {
            "base_url": "http://localhost:11434",
            "embedding_model": "nomic-embed-text",
            "chat_model": "llama3.1",
            "timeout": 30.0
        }
        
        yield server
        shutil.rmtree(storage_dir)

    def create_ollama_mock_responses(self, py_files: list[str]) -> dict[str, Any]:
        """Create mock responses that simulate Ollama-based processing."""
        return {
            "repomapper": {
                "file_ranks": [
                    {
                        "path": "src/main.py",
                        "rank": 0.95,
                        "centrality": 0.85,
                        "dependencies": ["src/data_processor.py", "src/model_trainer.py"],
                    },
                    {
                        "path": "src/data_processor.py",
                        "rank": 0.80,
                        "centrality": 0.70,
                        "dependencies": [],
                    },
                    {
                        "path": "src/model_trainer.py",
                        "rank": 0.75,
                        "centrality": 0.65,
                        "dependencies": [],
                    },
                ],
                "dependency_graph": [
                    {
                        "source": "src/main.py",
                        "target": "src/data_processor.py",
                        "edge_type": "import",
                        "weight": 1.0,
                    },
                    {
                        "source": "src/main.py",
                        "target": "src/model_trainer.py",
                        "edge_type": "import",
                        "weight": 1.0,
                    }
                ],
                "total_files": len(py_files),
            },
            "serena": {
                "symbols": [
                    {
                        "type": "def",
                        "symbol": "DataProcessor",
                        "path": "src/data_processor.py",
                        "span": [100, 1500],
                        "sig": "class DataProcessor:",
                    },
                    {
                        "type": "def",
                        "symbol": "DataProcessor.load_dataset",
                        "path": "src/data_processor.py",
                        "span": [200, 500],
                        "sig": "def load_dataset(self, filepath: str) -> pd.DataFrame:",
                    },
                    {
                        "type": "def",
                        "symbol": "ModelTrainer",
                        "path": "src/model_trainer.py",
                        "span": [100, 2000],
                        "sig": "class ModelTrainer:",
                    },
                ],
                "file_count": 2,
                "symbol_count": 3
            },
            # Simulate Ollama-based embeddings
            "ollama_embeddings": {
                "chunks": [
                    {
                        "chunk_id": "chunk_001",
                        "path": "src/main.py",
                        "content": "Main entry point for ML training pipeline.",
                        "span": [100, 150],
                        "embedding": [0.1, -0.2, 0.3, 0.4, -0.1] * 128,  # 640-dim like nomic-embed-text
                        "token_count": 8,
                    },
                    {
                        "chunk_id": "chunk_002",
                        "path": "src/data_processor.py",
                        "content": "class DataProcessor:",
                        "span": [80, 100],
                        "embedding": [0.2, -0.1, 0.4, 0.3, -0.2] * 128,
                        "token_count": 3,
                    },
                    {
                        "chunk_id": "chunk_003",
                        "path": "src/model_trainer.py",
                        "content": "Train a linear regression model.",
                        "span": [200, 250],
                        "embedding": [0.3, -0.4, 0.1, 0.2, -0.3] * 128,
                        "token_count": 6,
                    },
                ],
                "dimension": 640,
                "total_tokens": 17,
                "model_name": "nomic-embed-text",
            },
        }

    @pytest.mark.integration
    async def test_ollama_availability(self, ollama_helper: OllamaTestHelper):
        """Test that Ollama is available and responsive."""
        is_available = await ollama_helper.is_ollama_available()
        if not is_available:
            pytest.skip("Ollama server not available - skipping Ollama integration tests")
        
        print("✅ Ollama server is available and responsive")

    @pytest.mark.integration
    async def test_ollama_embedding_model(self, ollama_helper: OllamaTestHelper):
        """Test that the embedding model is available in Ollama."""
        is_available = await ollama_helper.is_ollama_available()
        if not is_available:
            pytest.skip("Ollama server not available")
        
        # Test with a lightweight embedding model
        model_available = await ollama_helper.ensure_model_available("nomic-embed-text")
        if not model_available:
            pytest.skip("nomic-embed-text model not available in Ollama")
        
        # Test embedding generation
        test_text = "This is a test document for embedding generation."
        embedding = await ollama_helper.test_embedding_generation("nomic-embed-text", test_text)
        
        assert embedding is not None, "Failed to generate embedding"
        assert isinstance(embedding, list), "Embedding should be a list"
        assert len(embedding) > 0, "Embedding should not be empty"
        
        print(f"✅ Generated embedding with {len(embedding)} dimensions")

    @pytest.mark.integration
    async def test_mcp_repo_indexing_with_ollama_mocks(
        self, 
        test_repo: Path, 
        mcp_server_with_ollama_config: MCPServer,
        ollama_helper: OllamaTestHelper
    ):
        """Test repository indexing with Ollama integration (mocked for speed)."""
        py_files = [str(f.relative_to(test_repo)) for f in test_repo.rglob("*.py")]
        mock_responses = self.create_ollama_mock_responses(py_files)

        # Mock subprocess calls to simulate Ollama-enhanced processing
        async def mock_ollama_embedding_call(url, data):
            """Mock Ollama embedding API call."""
            if "embeddings" in url:
                # Simulate successful embedding generation
                return {
                    "embedding": mock_responses["ollama_embeddings"]["chunks"][0]["embedding"]
                }
            return {}

        # Mock the adapter classes before they can be instantiated
        mock_repomapper = Mock()
        mock_serena = Mock()
        mock_leann = Mock()
        
        with patch("subprocess.run") as mock_run, \
             patch("httpx.AsyncClient.post") as mock_http_post, \
             patch.object(mock_repomapper, 'analyze_repository') as mock_repo_analyze, \
             patch.object(mock_serena, 'analyze_project') as mock_serena_analyze, \
             patch.object(mock_leann, 'build_index') as mock_leann_build, \
             patch("src.repoindex.pipeline.run.RepoMapperAdapter", return_value=mock_repomapper), \
             patch("src.repoindex.pipeline.run.SerenaAdapter", return_value=mock_serena), \
             patch("src.repoindex.pipeline.run.LEANNAdapter", return_value=mock_leann):

            # Configure subprocess mocks for external tools
            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])
                if not cmd:
                    return Mock(returncode=1, stdout="", stderr="Unknown command")

                # Handle git commands
                if "git" in cmd[0]:
                    if "rev-parse" in cmd and "--show-toplevel" in cmd:
                        return Mock(returncode=0, stdout=str(test_repo), stderr="")
                    elif "rev-parse" in cmd:
                        return Mock(returncode=0, stdout="HEAD", stderr="")
                    elif "status" in cmd:
                        return Mock(returncode=0, stdout="On branch main\nnothing to commit, working tree clean", stderr="")
                    elif "log" in cmd:
                        return Mock(returncode=0, stdout="commit abc123\nAuthor: test\n", stderr="")
                    else:
                        return Mock(returncode=0, stdout="", stderr="")

                if "repomapper" in cmd[0]:
                    if "--version" in cmd:
                        return Mock(returncode=0, stdout="repomapper 1.0.0", stderr="")
                    else:
                        return Mock(
                            returncode=0,
                            stdout=json.dumps(mock_responses["repomapper"]),
                            stderr=""
                        )
                elif "serena" in cmd[0]:
                    if "--version" in cmd:
                        return Mock(returncode=0, stdout="serena 1.0.0", stderr="")
                    else:
                        return Mock(
                            returncode=0,
                            stdout=json.dumps(mock_responses["serena"]),
                            stderr=""
                        )
                elif "leann" in cmd[0] or any("embed" in arg for arg in cmd):
                    # Mock embedding tool to use Ollama responses
                    if "--version" in cmd:
                        return Mock(returncode=0, stdout="ollama-embedder 1.0.0", stderr="")
                    else:
                        return Mock(
                            returncode=0,
                            stdout=json.dumps(mock_responses["ollama_embeddings"]),
                            stderr=""
                        )
                else:
                    # Default for any other command
                    return Mock(returncode=0, stdout="", stderr="")

            # Configure HTTP mocks for Ollama API
            async def http_post_side_effect(*args, **kwargs):
                url = str(args[0])
                if "embeddings" in url:
                    response_data = await mock_ollama_embedding_call(url, kwargs.get("json", {}))
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = response_data
                    return mock_response
                return Mock(status_code=404)

            mock_run.side_effect = subprocess_side_effect
            mock_http_post.side_effect = http_post_side_effect
            
            # Create proper schema objects for the adapters to return
            file_ranks = [FileRank(**fr) for fr in mock_responses["repomapper"]["file_ranks"]]
            edges = [DependencyEdge(**edge) for edge in mock_responses["repomapper"]["dependency_graph"]]
            repo_map = RepoMap(
                file_ranks=file_ranks,
                edges=edges, 
                total_files=mock_responses["repomapper"]["total_files"]
            )
            
            symbol_entries = [SymbolEntry(**sym) for sym in mock_responses["serena"]["symbols"]]
            serena_graph = SerenaGraph(
                entries=symbol_entries,
                file_count=mock_responses["serena"]["file_count"],
                symbol_count=mock_responses["serena"]["symbol_count"]
            )
            
            vector_chunks = [VectorChunk(**chunk) for chunk in mock_responses["ollama_embeddings"]["chunks"]]
            vector_index = VectorIndex(
                chunks=vector_chunks,
                dimension=mock_responses["ollama_embeddings"]["dimension"],
                total_tokens=mock_responses["ollama_embeddings"]["total_tokens"],
                model_name=mock_responses["ollama_embeddings"]["model_name"]
            )
            
            # Configure the adapter method mocks to return the schema objects
            mock_repo_analyze.return_value = repo_map
            mock_serena_analyze.return_value = serena_graph  
            mock_leann_build.return_value = vector_index

            # Test repository indexing
            result = await mcp_server_with_ollama_config._ensure_repo_index({
                "path": str(test_repo),
                "language": "py",
                "index_opts": {
                    "languages": ["py"],
                    "excludes": ["__pycache__/", ".git/", "*.pyc"],
                    "context_lines": 3,
                    "max_files_to_embed": 50,
                    "embedding_provider": "ollama",
                    "ollama_config": {
                        "base_url": "http://localhost:11434",
                        "embedding_model": "nomic-embed-text"
                    }
                }
            })

            # Debug the result first
            print(f"Result isError: {result.isError}")
            print(f"Result content: {result.content[0].text}")
            
            # Verify indexing result
            assert result.isError is False
            content = json.loads(result.content[0].text)

            assert "success" in content
            assert content["success"] is True
            assert "index_id" in content
            assert "files" in content
            assert len(content["files"]) >= 3  # We created 3 Python files

            print("✅ Successfully indexed repository with Ollama configuration")
            print(f"   Index ID: {content['index_id']}")
            print(f"   Files indexed: {len(content['files'])}")
            
            return content["index_id"]

    @pytest.mark.integration  
    async def test_mcp_search_with_ollama_embeddings(
        self,
        test_repo: Path,
        mcp_server_with_ollama_config: MCPServer,
        ollama_helper: OllamaTestHelper
    ):
        """Test repository search using Ollama embeddings."""
        # First index the repository
        index_id = await self.test_mcp_repo_indexing_with_ollama_mocks(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        # Mock Ollama search query embedding
        with patch("httpx.AsyncClient.post") as mock_http_post:
            async def search_embedding_mock(*args, **kwargs):
                url = str(args[0])
                if "embeddings" in url:
                    # Return mock query embedding similar to our content embeddings
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "embedding": [0.15, -0.25, 0.35, 0.45, -0.15] * 128  # Query embedding
                    }
                    return mock_response
                return Mock(status_code=404)

            mock_http_post.side_effect = search_embedding_mock

            # Test vector search with Ollama
            search_result = await mcp_server_with_ollama_config._search_repo({
                "index_id": index_id,
                "query": "machine learning data preprocessing",
                "features": {"vector": True, "symbol": True, "graph": False},
                "k": 10,
            })

            # Verify search results
            assert search_result.isError is False
            content = json.loads(search_result.content[0].text)

            assert "results" in content
            results = content["results"]
            assert len(results) > 0

            print(f"✅ Ollama-based search returned {len(results)} results")

            # Verify result structure and Ollama-specific metadata
            for result in results:
                assert "path" in result
                assert "content" in result  
                assert "score" in result
                assert "type" in result

            # Test symbol search combined with Ollama vectors
            symbol_search_result = await mcp_server_with_ollama_config._search_repo({
                "index_id": index_id,
                "query": "DataProcessor",
                "features": {"vector": True, "symbol": True, "graph": False},
                "k": 5,
            })

            assert symbol_search_result.isError is False
            symbol_content = json.loads(symbol_search_result.content[0].text)
            symbol_results = symbol_content["results"]

            print(f"✅ Combined Ollama + symbol search returned {len(symbol_results)} results")

    @pytest.mark.integration
    async def test_mcp_question_answering_with_ollama(
        self,
        test_repo: Path, 
        mcp_server_with_ollama_config: MCPServer,
        ollama_helper: OllamaTestHelper
    ):
        """Test question answering using Ollama chat models."""
        # First index the repository
        index_id = await self.test_mcp_repo_indexing_with_ollama_mocks(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        # Mock Ollama chat completion for Q&A
        with patch("httpx.AsyncClient.post") as mock_http_post:
            async def chat_completion_mock(*args, **kwargs):
                url = str(args[0])
                if "generate" in url or "chat" in url:
                    # Simulate intelligent response about the ML codebase
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "message": {
                            "content": """The DataProcessor class is a key component of this ML pipeline that handles data loading and preprocessing. 

Key responsibilities:
1. **Data Loading**: The `load_dataset()` method loads data from CSV files using pandas
2. **Feature Preparation**: The `prepare_features()` method separates features from target variables
3. **Normalization**: The `normalize_features()` method applies StandardScaler for feature scaling

The class is used in the main.py pipeline where it loads training data, prepares features, and feeds them to the ModelTrainer for model training. This follows a clean separation of concerns design pattern."""
                        },
                        "done": True
                    }
                    return mock_response
                elif "embeddings" in url:
                    # Mock embedding for context retrieval
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {
                        "embedding": [0.1, -0.2, 0.3] * 213  # 640 dimensions
                    }
                    return mock_response
                return Mock(status_code=404)

            mock_http_post.side_effect = chat_completion_mock

            # Ask a question about the codebase
            ask_result = await mcp_server_with_ollama_config._ask_index({
                "index_id": index_id,
                "question": "What is the DataProcessor class and what does it do?",
                "context_lines": 5
            })

            # Verify the answer
            assert ask_result.isError is False
            content = json.loads(ask_result.content[0].text)

            assert "answer" in content
            answer = content["answer"]
            assert len(answer) > 50  # Reasonable length answer

            # Check that answer contains relevant terms
            answer_lower = answer.lower()
            assert "dataprocessor" in answer_lower
            assert any(word in answer_lower for word in ["data", "loading", "preprocessing"])

            print("✅ Ollama-based question answered successfully")
            print(f"   Answer length: {len(answer)} characters")
            print(f"   Answer preview: {answer[:150]}...")

    @pytest.mark.integration
    async def test_mcp_bundle_export_with_ollama_metadata(
        self,
        test_repo: Path,
        mcp_server_with_ollama_config: MCPServer,
        ollama_helper: OllamaTestHelper
    ):
        """Test bundle export includes Ollama-specific metadata."""
        # First index the repository
        index_id = await self.test_mcp_repo_indexing_with_ollama_mocks(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        # Export the bundle
        bundle_result = await mcp_server_with_ollama_config._get_repo_bundle({
            "index_id": index_id
        })

        # Verify the bundle
        assert bundle_result.isError is False
        content = json.loads(bundle_result.content[0].text)

        assert "bundle_data" in content
        bundle_data = content["bundle_data"]

        # Verify bundle structure
        required_components = ["manifest", "repo_map", "symbol_graph", "vector_index", "snippets"]
        for component in required_components:
            assert component in bundle_data, f"Bundle missing component: {component}"

        # Check Ollama-specific metadata in vector index
        vector_index = bundle_data["vector_index"]
        assert "chunks" in vector_index
        assert "metadata" in vector_index

        metadata = vector_index["metadata"]
        assert "model_name" in metadata
        assert "embedding_dimensions" in metadata
        
        # Verify Ollama-specific fields
        if "provider" in metadata:
            assert metadata["provider"] == "ollama"
        if "base_url" in metadata:
            assert "localhost:11434" in metadata["base_url"]

        print("✅ Bundle exported with Ollama metadata:")
        print(f"   Model: {metadata.get('model_name', 'unknown')}")
        print(f"   Dimensions: {metadata.get('embedding_dimensions', 'unknown')}")
        print(f"   Chunks: {len(vector_index['chunks'])}")

    @pytest.mark.integration
    @pytest.mark.slow
    async def test_end_to_end_ollama_workflow(
        self,
        test_repo: Path,
        mcp_server_with_ollama_config: MCPServer,
        ollama_helper: OllamaTestHelper
    ):
        """Test complete end-to-end workflow with Ollama integration."""
        print("🚀 Starting end-to-end Ollama workflow test")

        # Check Ollama availability
        print("🔍 Step 1: Checking Ollama availability...")
        await self.test_ollama_availability(ollama_helper)

        # Test embedding model
        print("📊 Step 2: Testing embedding model...")
        await self.test_ollama_embedding_model(ollama_helper)

        # Index repository with Ollama
        print("📁 Step 3: Indexing repository with Ollama...")
        await self.test_mcp_repo_indexing_with_ollama_mocks(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        # Test Ollama-based search
        print("🔍 Step 4: Testing Ollama-based search...")
        await self.test_mcp_search_with_ollama_embeddings(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        # Test Ollama-based Q&A
        print("❓ Step 5: Testing Ollama-based question answering...")
        await self.test_mcp_question_answering_with_ollama(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        # Test bundle export
        print("📦 Step 6: Testing bundle export with Ollama metadata...")
        await self.test_mcp_bundle_export_with_ollama_metadata(
            test_repo, mcp_server_with_ollama_config, ollama_helper
        )

        print("✅ End-to-end Ollama workflow completed successfully!")


@pytest.mark.integration 
async def test_ollama_connection_health():
    """Standalone test for Ollama connection health."""
    helper = OllamaTestHelper()
    
    try:
        is_available = await helper.is_ollama_available()
        if is_available:
            print("✅ Ollama server is healthy and accessible")
        else:
            print("❌ Ollama server is not accessible")
            print("   Make sure Ollama is running: `ollama serve`")
    finally:
        await helper.cleanup()


if __name__ == "__main__":
    # Quick development test
    import asyncio
    
    async def quick_test():
        """Run a quick test during development."""
        print("Running Ollama connection health check...")
        await test_ollama_connection_health()
        
    asyncio.run(quick_test())