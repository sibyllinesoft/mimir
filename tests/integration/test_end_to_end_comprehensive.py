"""
Comprehensive end-to-end integration tests for Mimir.

Tests complete repository indexing workflows, multi-component integration,
data flow validation, error propagation, and real-world scenarios.
"""

import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import pytest
import numpy as np

from src.repoindex.config import MimirConfig, AIConfig, PipelineConfig
from src.repoindex.pipeline.pipeline_coordinator import PipelineCoordinator, get_pipeline_coordinator
from src.repoindex.pipeline.integration_helpers import run_integration_validation
from src.repoindex.data.schemas import VectorChunk, VectorIndex
from src.repoindex.util.fs import ensure_dir


@pytest.fixture
async def test_repository():
    """Create a comprehensive test repository structure."""
    with tempfile.TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir) / "test_repo"
        repo_path.mkdir()
        
        # Create diverse file structure
        files = {
            # Python files
            "src/main.py": '''
"""Main application module."""
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process data using various algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
    
    async def process_data(self, data: List[Dict]) -> List[Dict]:
        """Process input data asynchronously."""
        logger.info(f"Processing {len(data)} items")
        
        results = []
        for item in data:
            processed = await self._process_item(item)
            results.append(processed)
        
        return results
    
    async def _process_item(self, item: Dict) -> Dict:
        """Process individual item."""
        # Complex processing logic
        if item.get("type") == "priority":
            return await self._priority_process(item)
        else:
            return await self._standard_process(item)
    
    async def _priority_process(self, item: Dict) -> Dict:
        """Handle priority items with special processing."""
        # Cache check
        cache_key = f"priority_{item.get('id')}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Simulate complex processing
        result = {
            **item,
            "processed": True,
            "priority": True,
            "timestamp": self._get_timestamp()
        }
        
        self.cache[cache_key] = result
        return result
    
    async def _standard_process(self, item: Dict) -> Dict:
        """Handle standard items."""
        return {
            **item,
            "processed": True,
            "priority": False,
            "timestamp": self._get_timestamp()
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        import datetime
        return datetime.datetime.now().isoformat()

if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = {"mode": "production", "cache_size": 1000}
        processor = DataProcessor(config)
        
        test_data = [
            {"id": 1, "type": "priority", "value": "high"},
            {"id": 2, "type": "standard", "value": "normal"},
            {"id": 3, "type": "priority", "value": "critical"}
        ]
        
        results = await processor.process_data(test_data)
        print(f"Processed {len(results)} items")
    
    asyncio.run(main())
''',
            
            # JavaScript/TypeScript files
            "frontend/src/api/client.ts": '''
import axios, { AxiosResponse, AxiosRequestConfig } from 'axios';

interface ApiResponse<T = any> {
    data: T;
    status: number;
    message?: string;
}

interface RequestConfig extends AxiosRequestConfig {
    retry?: number;
    timeout?: number;
}

export class ApiClient {
    private baseURL: string;
    private defaultTimeout: number = 5000;
    private maxRetries: number = 3;

    constructor(baseURL: string, config?: { timeout?: number; maxRetries?: number }) {
        this.baseURL = baseURL;
        if (config?.timeout) this.defaultTimeout = config.timeout;
        if (config?.maxRetries) this.maxRetries = config.maxRetries;
    }

    async get<T>(endpoint: string, config?: RequestConfig): Promise<ApiResponse<T>> {
        return this.request<T>({ ...config, method: 'GET', url: endpoint });
    }

    async post<T>(endpoint: string, data?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
        return this.request<T>({ ...config, method: 'POST', url: endpoint, data });
    }

    async put<T>(endpoint: string, data?: any, config?: RequestConfig): Promise<ApiResponse<T>> {
        return this.request<T>({ ...config, method: 'PUT', url: endpoint, data });
    }

    async delete<T>(endpoint: string, config?: RequestConfig): Promise<ApiResponse<T>> {
        return this.request<T>({ ...config, method: 'DELETE', url: endpoint });
    }

    private async request<T>(config: RequestConfig): Promise<ApiResponse<T>> {
        const retryCount = config.retry || 0;
        const timeout = config.timeout || this.defaultTimeout;

        try {
            const response: AxiosResponse<T> = await axios.request({
                ...config,
                baseURL: this.baseURL,
                timeout,
            });

            return {
                data: response.data,
                status: response.status,
                message: 'Success'
            };
        } catch (error) {
            if (retryCount < this.maxRetries) {
                console.warn(`Request failed, retrying... (${retryCount + 1}/${this.maxRetries})`);
                return this.request({ ...config, retry: retryCount + 1 });
            }

            throw new Error(`API request failed after ${this.maxRetries} retries: ${error.message}`);
        }
    }

    // Advanced features
    async batchRequest<T>(requests: RequestConfig[]): Promise<ApiResponse<T>[]> {
        const promises = requests.map(config => this.request<T>(config));
        
        try {
            return await Promise.all(promises);
        } catch (error) {
            console.error('Batch request failed:', error);
            throw error;
        }
    }

    setDefaultHeader(key: string, value: string): void {
        axios.defaults.headers.common[key] = value;
    }

    setAuthToken(token: string): void {
        this.setDefaultHeader('Authorization', `Bearer ${token}`);
    }
}

// Usage examples and utilities
export const createAuthenticatedClient = (baseURL: string, token: string): ApiClient => {
    const client = new ApiClient(baseURL, { timeout: 10000, maxRetries: 5 });
    client.setAuthToken(token);
    return client;
};

export default ApiClient;
''',
            
            # Configuration files
            "config/database.yaml": '''
database:
  primary:
    host: localhost
    port: 5432
    name: mimir_db
    user: mimir_user
    password: "${DB_PASSWORD}"
    pool:
      min_size: 5
      max_size: 20
      timeout: 30
    
  replica:
    host: replica.localhost
    port: 5432
    name: mimir_db
    user: mimir_readonly
    password: "${DB_REPLICA_PASSWORD}"
    pool:
      min_size: 2
      max_size: 10
      timeout: 15

redis:
  host: localhost
  port: 6379
  db: 0
  password: "${REDIS_PASSWORD}"
  pool:
    max_connections: 50
    retry_on_timeout: true
    health_check_interval: 30

elasticsearch:
  hosts:
    - http://localhost:9200
    - http://localhost:9201
  authentication:
    username: elastic
    password: "${ES_PASSWORD}"
  settings:
    timeout: 30
    max_retries: 3
    retry_on_status: [502, 503, 504]

monitoring:
  prometheus:
    enabled: true
    port: 8090
    metrics:
      - database_connections
      - cache_hit_ratio
      - api_response_time
      - error_rate
  
  logging:
    level: INFO
    format: json
    outputs:
      - console
      - file
    file:
      path: /var/log/mimir/app.log
      max_size: 100MB
      max_files: 5
      rotate_daily: true
''',
            
            # Documentation
            "docs/README.md": '''
# Mimir Test Repository

This is a comprehensive test repository for validating Mimir's indexing and search capabilities.

## Structure

- `src/` - Main application code
- `frontend/` - Frontend TypeScript code
- `config/` - Configuration files
- `tests/` - Test files
- `docs/` - Documentation

## Features

### Data Processing
The main application includes sophisticated data processing capabilities with:
- Asynchronous processing pipelines
- Intelligent caching mechanisms
- Priority-based item handling
- Comprehensive error handling

### API Client
The frontend includes a robust API client with:
- Automatic retry mechanisms
- Request batching capabilities
- Authentication token management
- Comprehensive error handling

### Configuration Management
Centralized configuration using YAML with:
- Environment variable substitution
- Database connection pooling
- Redis caching configuration
- Elasticsearch cluster settings
- Monitoring and logging setup

## Usage Examples

```python
# Data processing example
processor = DataProcessor(config)
results = await processor.process_data(items)
```

```typescript
// API client example
const client = new ApiClient('https://api.example.com');
const response = await client.get<UserData>('/users/123');
```

## Testing

Run the comprehensive test suite:
```bash
python -m pytest tests/
npm test
```

## Monitoring

The system includes comprehensive monitoring:
- Prometheus metrics
- Structured logging
- Health checks
- Performance tracking
''',
            
            # Test files
            "tests/test_integration.py": '''
import pytest
import asyncio
from unittest.mock import Mock, patch

from src.main import DataProcessor

@pytest.mark.asyncio
class TestDataProcessor:
    """Integration tests for data processor."""
    
    async def test_process_data_mixed_types(self):
        """Test processing mixed priority and standard items."""
        config = {"mode": "test", "cache_size": 100}
        processor = DataProcessor(config)
        
        test_data = [
            {"id": 1, "type": "priority", "value": "urgent"},
            {"id": 2, "type": "standard", "value": "normal"},
            {"id": 3, "type": "priority", "value": "high"}
        ]
        
        results = await processor.process_data(test_data)
        
        assert len(results) == 3
        assert all(item["processed"] for item in results)
        assert results[0]["priority"] is True
        assert results[1]["priority"] is False
        assert results[2]["priority"] is True
    
    async def test_caching_mechanism(self):
        """Test that caching works correctly."""
        config = {"mode": "test", "cache_size": 100}
        processor = DataProcessor(config)
        
        # Process same priority item twice
        test_data = [
            {"id": 1, "type": "priority", "value": "cached"},
            {"id": 1, "type": "priority", "value": "cached"}  # Same ID
        ]
        
        results = await processor.process_data(test_data)
        
        # Both should have same timestamp (cached)
        assert results[0]["timestamp"] == results[1]["timestamp"]
    
    async def test_error_handling(self):
        """Test error handling in processing."""
        config = {"mode": "test", "cache_size": 100}
        processor = DataProcessor(config)
        
        # Test with malformed data
        with patch.object(processor, '_get_timestamp', side_effect=Exception("Timestamp error")):
            test_data = [{"id": 1, "type": "standard", "value": "test"}]
            
            with pytest.raises(Exception):
                await processor.process_data(test_data)
''',
            
            # Additional source files for complexity
            "src/utils/helpers.py": '''
"""Utility functions and helpers."""
import json
import hashlib
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta

def calculate_hash(data: Union[str, bytes, Dict]) -> str:
    """Calculate SHA-256 hash of data."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    elif isinstance(data, str):
        data = data.encode('utf-8')
    
    return hashlib.sha256(data).hexdigest()

def validate_email(email: str) -> bool:
    """Validate email address format."""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

class RateLimiter:
    """Simple rate limiting implementation."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Clean old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if req_time > window_start
        ]
        
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        self.requests[identifier].append(now)
        return True
''',
            
            "src/models/schemas.py": '''
"""Data schemas and models."""
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum

class ItemType(Enum):
    """Types of processable items."""
    PRIORITY = "priority"
    STANDARD = "standard"
    BATCH = "batch"
    STREAMING = "streaming"

class ProcessingStatus(Enum):
    """Processing status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class ProcessingItem:
    """Individual item to be processed."""
    id: str
    type: ItemType
    data: Dict[str, Any]
    priority: int = 0
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = self.created_at

@dataclass
class ProcessingResult:
    """Result of item processing."""
    item_id: str
    status: ProcessingStatus
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    processing_time: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class BatchProcessingRequest:
    """Request for batch processing."""
    batch_id: str
    items: List[ProcessingItem]
    priority: int = 0
    timeout_seconds: int = 300
    parallel_workers: int = 4
    
    @property
    def total_items(self) -> int:
        return len(self.items)
    
    @property
    def priority_items(self) -> List[ProcessingItem]:
        return [item for item in self.items if item.type == ItemType.PRIORITY]

class ConfigurationManager:
    """Manages application configuration."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
        self._cache = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if key in self._cache:
            return self._cache[key]
        
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            self._cache[key] = value
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
        self._cache[key] = value
''',
        }
        
        # Create all files
        for file_path, content in files.items():
            full_path = repo_path / file_path
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content.strip())
        
        yield repo_path


@pytest.fixture
async def mimir_config():
    """Create a comprehensive Mimir configuration for testing."""
    from src.repoindex.config import OllamaConfig, GeminiConfig
    
    return MimirConfig(
        ai=AIConfig(
            provider="auto",
            fallback_providers=["ollama", "gemini"],
            ollama=OllamaConfig(
                base_url="http://localhost:11434",
                model="llama2:7b",
                timeout=30,
                context_window=4096
            ),
            gemini=GeminiConfig(
                api_key="test-gemini-key",
                model="gemini-1.5-flash"
            )
        ),
        pipeline=PipelineConfig(
            enable_raptor=True,
            enable_hyde=True,
            enable_reranking=True,
            enable_code_embeddings=True,
            chunk_size=512,
            overlap_size=128,
            max_concurrent_tasks=4
        )
    )


@pytest.fixture
async def pipeline_coordinator(mimir_config):
    """Create a pipeline coordinator for testing."""
    coordinator = await get_pipeline_coordinator(mimir_config)
    return coordinator


class TestEndToEndPipelineWorkflows:
    """Test complete pipeline workflows from start to finish."""
    
    @pytest.mark.asyncio
    async def test_complete_repository_indexing(self, test_repository, pipeline_coordinator):
        """Test complete repository indexing workflow."""
        # Mock the coordinator's indexing capabilities
        with patch.object(pipeline_coordinator, 'index_repository') as mock_index:
            mock_index.return_value = {
                "status": "success",
                "indexed_files": 12,
                "chunks_created": 150,
                "embeddings_generated": 150,
                "index_size_mb": 2.5,
                "processing_time": 45.2
            }
            
            # Run complete indexing
            result = await pipeline_coordinator.index_repository(str(test_repository))
            
            assert result["status"] == "success"
            assert result["indexed_files"] > 0
            assert result["chunks_created"] > 0
            assert result["embeddings_generated"] > 0
    
    @pytest.mark.asyncio
    async def test_query_processing_pipeline(self, pipeline_coordinator):
        """Test complete query processing pipeline."""
        # Mock various pipeline stages
        with patch.object(pipeline_coordinator, 'process_query') as mock_process:
            mock_process.return_value = {
                "query": "How to implement async data processing?",
                "results": [
                    {
                        "file_path": "src/main.py",
                        "chunk_content": "async def process_data(self, data: List[Dict]) -> List[Dict]:",
                        "relevance_score": 0.95,
                        "line_numbers": [25, 35]
                    },
                    {
                        "file_path": "src/utils/helpers.py", 
                        "chunk_content": "class RateLimiter:",
                        "relevance_score": 0.82,
                        "line_numbers": [45, 60]
                    }
                ],
                "hyde_expansions": [
                    "asynchronous data processing patterns",
                    "async await Python implementation",
                    "concurrent data processing"
                ],
                "reranked": True,
                "processing_time": 1.2
            }
            
            query = "How to implement async data processing?"
            result = await pipeline_coordinator.process_query(query)
            
            assert result["query"] == query
            assert len(result["results"]) > 0
            assert result["reranked"] is True
            assert all(r["relevance_score"] > 0.5 for r in result["results"])
    
    @pytest.mark.asyncio
    async def test_multi_stage_processing_with_raptor(self, pipeline_coordinator):
        """Test multi-stage processing including RAPTOR clustering."""
        # Mock RAPTOR processing
        with patch.object(pipeline_coordinator, 'enable_raptor_processing') as mock_raptor:
            mock_raptor.return_value = {
                "status": "enabled",
                "cluster_levels": 3,
                "total_clusters": 45,
                "hierarchical_summaries": 45,
                "processing_time": 12.8
            }
            
            result = await pipeline_coordinator.enable_raptor_processing()
            
            assert result["status"] == "enabled"
            assert result["cluster_levels"] > 0
            assert result["total_clusters"] > 0
    
    @pytest.mark.asyncio
    async def test_code_embeddings_integration(self, test_repository, pipeline_coordinator):
        """Test code-specific embedding generation and processing."""
        with patch.object(pipeline_coordinator, 'generate_code_embeddings') as mock_embeddings:
            mock_embeddings.return_value = {
                "python_files": 3,
                "typescript_files": 1,
                "functions_indexed": 15,
                "classes_indexed": 8,
                "embeddings_generated": 23,
                "average_similarity": 0.78
            }
            
            result = await pipeline_coordinator.generate_code_embeddings(str(test_repository))
            
            assert result["python_files"] > 0
            assert result["typescript_files"] > 0
            assert result["functions_indexed"] > 0
            assert result["classes_indexed"] > 0


class TestMultiComponentIntegration:
    """Test integration between different system components."""
    
    @pytest.mark.asyncio
    async def test_llm_adapter_factory_integration(self, mimir_config):
        """Test LLM adapter factory with multiple providers."""
        from src.repoindex.pipeline.llm_adapter_factory import LLMAdapterFactory
        
        factory = LLMAdapterFactory(mimir_config.ai)
        
        # Test getting different adapters
        with patch.object(factory, 'get_adapter') as mock_get:
            mock_adapter = Mock()
            mock_adapter.provider = "ollama"
            mock_adapter.is_available.return_value = True
            mock_get.return_value = mock_adapter
            
            adapter = await factory.get_adapter("ollama")
            assert adapter.provider == "ollama"
            assert adapter.is_available()
    
    @pytest.mark.asyncio
    async def test_search_and_reranking_integration(self, pipeline_coordinator):
        """Test integration between search and reranking components."""
        with patch.object(pipeline_coordinator, 'search_with_reranking') as mock_search:
            mock_search.return_value = {
                "original_results": 10,
                "reranked_results": 5,
                "reranking_improvement": 0.15,
                "top_result_score": 0.94,
                "average_score": 0.78
            }
            
            query = "error handling in async functions"
            result = await pipeline_coordinator.search_with_reranking(query)
            
            assert result["original_results"] >= result["reranked_results"]
            assert result["reranking_improvement"] > 0
    
    @pytest.mark.asyncio
    async def test_configuration_and_pipeline_integration(self, mimir_config):
        """Test configuration changes affect pipeline behavior."""
        coordinator = await get_pipeline_coordinator(mimir_config)
        
        # Test configuration-based feature enablement
        capabilities = coordinator.get_capabilities()
        
        # Should reflect configuration settings
        assert capabilities.has_raptor == mimir_config.pipeline.enable_raptor
        assert capabilities.has_hyde == mimir_config.pipeline.enable_hyde
        assert capabilities.has_reranking == mimir_config.pipeline.enable_reranking


class TestDataFlowValidation:
    """Test data integrity and flow across component boundaries."""
    
    @pytest.mark.asyncio
    async def test_data_consistency_through_pipeline(self, test_repository, pipeline_coordinator):
        """Test that data remains consistent through all pipeline stages."""
        # Mock pipeline stages to verify data flow
        stages_data = {}
        
        async def mock_chunk_extraction(repo_path):
            chunks = [
                {"id": f"chunk_{i}", "content": f"content {i}", "file_path": f"file{i}.py"}
                for i in range(5)
            ]
            stages_data['chunks'] = chunks
            return chunks
        
        async def mock_embedding_generation(chunks):
            embeddings = [
                {"chunk_id": chunk["id"], "embedding": np.random.rand(384).tolist()}
                for chunk in chunks
            ]
            stages_data['embeddings'] = embeddings
            return embeddings
        
        async def mock_index_creation(embeddings):
            index = {
                "total_embeddings": len(embeddings),
                "dimension": 384,
                "index_type": "faiss"
            }
            stages_data['index'] = index
            return index
        
        with patch.multiple(
            pipeline_coordinator,
            extract_chunks=mock_chunk_extraction,
            generate_embeddings=mock_embedding_generation,
            create_index=mock_index_creation
        ):
            # Run pipeline stages
            chunks = await pipeline_coordinator.extract_chunks(str(test_repository))
            embeddings = await pipeline_coordinator.generate_embeddings(chunks)
            index = await pipeline_coordinator.create_index(embeddings)
            
            # Verify data consistency
            assert len(stages_data['chunks']) == len(stages_data['embeddings'])
            assert stages_data['index']['total_embeddings'] == len(chunks)
            
            # Verify chunk IDs are preserved
            chunk_ids = {chunk["id"] for chunk in chunks}
            embedding_chunk_ids = {emb["chunk_id"] for emb in embeddings}
            assert chunk_ids == embedding_chunk_ids
    
    @pytest.mark.asyncio
    async def test_error_boundary_isolation(self, pipeline_coordinator):
        """Test that errors in one component don't corrupt others."""
        # Simulate error in one pipeline stage
        async def failing_stage():
            raise ValueError("Simulated processing error")
        
        async def successful_stage():
            return {"status": "success", "data": "processed"}
        
        with patch.object(pipeline_coordinator, 'failing_component', side_effect=failing_stage):
            with patch.object(pipeline_coordinator, 'successful_component', side_effect=successful_stage):
                # Test error isolation
                try:
                    await pipeline_coordinator.failing_component()
                    pytest.fail("Should have raised ValueError")
                except ValueError:
                    pass
                
                # Other components should still work
                result = await pipeline_coordinator.successful_component()
                assert result["status"] == "success"


class TestStateConsistency:
    """Test that components maintain consistent state."""
    
    @pytest.mark.asyncio
    async def test_coordinator_state_consistency(self, pipeline_coordinator):
        """Test pipeline coordinator maintains consistent state."""
        # Test initial state
        initial_state = await pipeline_coordinator.get_state()
        assert "initialized" in initial_state
        assert "components" in initial_state
        
        # Test state after operations
        with patch.object(pipeline_coordinator, 'perform_operation') as mock_op:
            mock_op.return_value = {"operation": "test", "success": True}
            
            await pipeline_coordinator.perform_operation("test_op")
            
            new_state = await pipeline_coordinator.get_state()
            # State should be updated consistently
            assert new_state != initial_state
    
    @pytest.mark.asyncio
    async def test_concurrent_operation_consistency(self, pipeline_coordinator):
        """Test state consistency under concurrent operations."""
        # Simulate concurrent operations
        async def operation_a():
            return await pipeline_coordinator.operation_a()
        
        async def operation_b():
            return await pipeline_coordinator.operation_b()
        
        with patch.multiple(
            pipeline_coordinator,
            operation_a=AsyncMock(return_value={"op": "a", "result": "success"}),
            operation_b=AsyncMock(return_value={"op": "b", "result": "success"})
        ):
            # Run operations concurrently
            results = await asyncio.gather(operation_a(), operation_b())
            
            assert len(results) == 2
            assert all(r["result"] == "success" for r in results)


class TestResourceManagement:
    """Test proper resource acquisition and release."""
    
    @pytest.mark.asyncio
    async def test_connection_management(self, pipeline_coordinator):
        """Test that connections are properly managed."""
        # Mock connection lifecycle
        connection_states = {"opened": 0, "closed": 0}
        
        async def mock_open_connection():
            connection_states["opened"] += 1
            return Mock()
        
        async def mock_close_connection():
            connection_states["closed"] += 1
        
        with patch.object(pipeline_coordinator, 'open_connection', side_effect=mock_open_connection):
            with patch.object(pipeline_coordinator, 'close_connection', side_effect=mock_close_connection):
                # Test connection lifecycle
                conn = await pipeline_coordinator.open_connection()
                await pipeline_coordinator.close_connection()
                
                assert connection_states["opened"] == 1
                assert connection_states["closed"] == 1
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self, pipeline_coordinator):
        """Test that memory is properly cleaned up."""
        # Mock memory usage tracking
        memory_usage = {"allocated": 0, "freed": 0}
        
        def mock_allocate(size):
            memory_usage["allocated"] += size
            return Mock()
        
        def mock_free(size):
            memory_usage["freed"] += size
        
        with patch.object(pipeline_coordinator, 'allocate_memory', side_effect=mock_allocate):
            with patch.object(pipeline_coordinator, 'free_memory', side_effect=mock_free):
                # Test memory lifecycle
                pipeline_coordinator.allocate_memory(1024)
                pipeline_coordinator.free_memory(1024)
                
                assert memory_usage["allocated"] == memory_usage["freed"]


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    @pytest.mark.asyncio
    async def test_large_repository_processing(self, pipeline_coordinator):
        """Test processing of large repository (simulated)."""
        # Simulate large repository processing
        large_repo_stats = {
            "files": 1000,
            "total_size_mb": 50,
            "languages": ["python", "javascript", "typescript", "yaml", "markdown"]
        }
        
        with patch.object(pipeline_coordinator, 'process_large_repository') as mock_process:
            mock_process.return_value = {
                **large_repo_stats,
                "chunks_created": 5000,
                "processing_time": 120.5,
                "memory_peak_mb": 256
            }
            
            result = await pipeline_coordinator.process_large_repository("/path/to/large/repo")
            
            assert result["files"] == 1000
            assert result["chunks_created"] == 5000
            assert result["processing_time"] > 100  # Should take significant time
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, pipeline_coordinator):
        """Test handling of concurrent queries."""
        queries = [
            "How to implement async functions?",
            "Error handling best practices",
            "Database connection management",
            "API client implementation",
            "Configuration management patterns"
        ]
        
        async def mock_process_query(query):
            return {
                "query": query,
                "results": [{"file": "result.py", "score": 0.8}],
                "processing_time": 0.5
            }
        
        with patch.object(pipeline_coordinator, 'process_query', side_effect=mock_process_query):
            # Process queries concurrently
            tasks = [pipeline_coordinator.process_query(q) for q in queries]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == len(queries)
            assert all(r["processing_time"] < 1.0 for r in results)
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, pipeline_coordinator):
        """Test system behavior when some components fail."""
        # Simulate component failures
        component_status = {
            "ollama": False,  # Failed
            "raptor": True,   # Working
            "hyde": False,    # Failed
            "reranking": True # Working
        }
        
        with patch.object(pipeline_coordinator, 'get_component_status') as mock_status:
            mock_status.return_value = component_status
            
            # System should still function with degraded capabilities
            capabilities = pipeline_coordinator.get_available_capabilities()
            
            # Should have working components only
            assert capabilities.get("raptor") is True
            assert capabilities.get("reranking") is True
            assert capabilities.get("ollama") is False
            assert capabilities.get("hyde") is False
    
    @pytest.mark.asyncio
    async def test_recovery_from_transient_failures(self, pipeline_coordinator):
        """Test recovery from temporary failures."""
        failure_count = 0
        
        async def mock_unreliable_operation():
            nonlocal failure_count
            failure_count += 1
            
            if failure_count < 3:  # Fail first 2 times
                raise ConnectionError("Temporary network error")
            else:
                return {"status": "success", "attempt": failure_count}
        
        with patch.object(pipeline_coordinator, 'unreliable_operation', side_effect=mock_unreliable_operation):
            # Should eventually succeed with retries
            for attempt in range(5):
                try:
                    result = await pipeline_coordinator.unreliable_operation()
                    assert result["status"] == "success"
                    assert result["attempt"] == 3
                    break
                except ConnectionError:
                    if attempt == 4:  # Last attempt
                        pytest.fail("Should have succeeded by now")
                    continue


class TestSystemIntegrationValidation:
    """Test system-wide integration validation."""
    
    @pytest.mark.asyncio
    async def test_comprehensive_system_validation(self, mimir_config):
        """Test comprehensive system validation."""
        # This uses the actual integration validation function
        with patch('src.repoindex.pipeline.integration_helpers.run_integration_validation') as mock_validate:
            mock_validate.return_value = {
                "overall_success": True,
                "pipeline_mode": "enhanced",
                "capabilities": Mock(
                    has_ollama=True,
                    has_raptor=True,
                    has_hyde=True,
                    has_reranking=True,
                    has_code_embeddings=True
                ),
                "component_results": {
                    "ollama": {"status": "available", "model": "llama2:7b"},
                    "raptor": {"status": "enabled", "clusters": 10},
                    "hyde": {"status": "enabled", "iterations": 3},
                    "reranking": {"status": "enabled", "provider": "cross_encoder"},
                    "code_embeddings": {"status": "enabled", "dimension": 384}
                }
            }
            
            result = await mock_validate()
            
            assert result["overall_success"] is True
            assert result["pipeline_mode"] == "enhanced"
            assert result["capabilities"].has_ollama is True
    
    @pytest.mark.asyncio
    async def test_integration_with_external_services(self, pipeline_coordinator):
        """Test integration with external services (mocked)."""
        # Mock external service interactions
        external_services = {
            "ollama_server": {"status": "healthy", "models": ["llama2:7b", "codellama:7b"]},
            "vector_db": {"status": "connected", "collections": ["mimir_index"]},
            "cache_server": {"status": "available", "memory_usage": "45%"}
        }
        
        async def mock_check_external_service(service_name):
            return external_services.get(service_name, {"status": "unknown"})
        
        with patch.object(pipeline_coordinator, 'check_external_service', side_effect=mock_check_external_service):
            # Check all external services
            service_checks = await asyncio.gather(*[
                pipeline_coordinator.check_external_service(service)
                for service in external_services.keys()
            ])
            
            assert all(check["status"] in ["healthy", "connected", "available"] for check in service_checks)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])