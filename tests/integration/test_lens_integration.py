"""
Integration tests for Mimir-Lens communication.

Tests the HTTP client, health checks, error handling, and fallback mechanisms.
"""

import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime

from src.repoindex.config import LensConfig
from src.repoindex.pipeline.lens_client import (
    LensIntegrationClient,
    LensHealthStatus,
    LensIndexRequest,
    LensSearchRequest,
    LensIntegrationError,
    LensConnectionError,
    LensTimeoutError,
    get_lens_client,
    init_lens_client,
)


@pytest.fixture
def lens_config():
    """Create test Lens configuration."""
    return LensConfig(
        enabled=True,
        base_url="http://localhost:3001",
        timeout=10,
        max_retries=2,
        retry_delay=0.1,
        health_check_enabled=True,
        health_check_timeout=5,
        fallback_enabled=True
    )


@pytest.fixture
async def lens_client(lens_config):
    """Create test Lens client."""
    client = LensIntegrationClient(lens_config)
    yield client
    await client.close()


class TestLensConfig:
    """Test Lens configuration validation."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = LensConfig()
        
        assert config.enabled is False
        assert config.base_url == "http://localhost:3001"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.health_check_enabled is True
        assert config.fallback_enabled is True
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = LensConfig(
            timeout=60,
            max_retries=5,
            retry_delay=2.0,
            connection_pool_size=20
        )
        assert config.timeout == 60
        assert config.max_retries == 5
        
        # Invalid timeout
        with pytest.raises(ValueError, match="Timeout must be positive"):
            LensConfig(timeout=0)
        
        # Invalid max retries
        with pytest.raises(ValueError, match="Max retries should not exceed 10"):
            LensConfig(max_retries=15)
        
        # Invalid retry delay
        with pytest.raises(ValueError, match="Retry delay must be at least 0.1 seconds"):
            LensConfig(retry_delay=0.05)
    
    def test_env_prefix(self):
        """Test environment variable prefix."""
        config = LensConfig()
        assert config.model_config["env_prefix"] == "LENS_"


class TestLensIntegrationClient:
    """Test Lens integration client functionality."""
    
    def test_client_initialization(self, lens_config):
        """Test client initialization."""
        client = LensIntegrationClient(lens_config)
        
        assert client.config == lens_config
        assert client.session is None
        assert client._health_status == LensHealthStatus.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_session_management(self, lens_client):
        """Test HTTP session management."""
        # Session should be None initially
        assert lens_client.session is None
        
        # Initialize should create session
        await lens_client.initialize()
        assert lens_client.session is not None
        assert not lens_client.session.closed
        
        # Close should cleanup session
        await lens_client.close()
        assert lens_client.session.closed
    
    @pytest.mark.asyncio
    async def test_context_manager(self, lens_config):
        """Test async context manager."""
        async with LensIntegrationClient(lens_config) as client:
            assert client.session is not None
            assert not client.session.closed
        
        # Session should be closed after exiting context
        assert client.session.closed
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, lens_client):
        """Test successful health check."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "healthy",
            "version": "1.0.0",
            "uptime": 3600
        }
        
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            health = await lens_client.check_health()
            
            assert health.status == LensHealthStatus.HEALTHY
            assert health.version == "1.0.0"
            assert health.uptime_seconds == 3600
            assert health.error is None
            assert lens_client._health_status == LensHealthStatus.HEALTHY
            assert lens_client._consecutive_failures == 0
    
    @pytest.mark.asyncio
    async def test_health_check_failure(self, lens_client):
        """Test health check failure."""
        # Mock timeout
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.side_effect = asyncio.TimeoutError()
            
            health = await lens_client.check_health()
            
            assert health.status == LensHealthStatus.UNHEALTHY
            assert health.error == "Health check timeout"
            assert lens_client._health_status == LensHealthStatus.UNHEALTHY
            assert lens_client._consecutive_failures == 1
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, lens_client):
        """Test circuit breaker functionality."""
        # Simulate multiple failures to trigger circuit breaker
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.side_effect = Exception("Connection failed")
            
            # First 3 failures should trigger circuit breaker
            for i in range(3):
                await lens_client.check_health()
            
            assert lens_client._consecutive_failures == 3
            assert lens_client._circuit_breaker_opened_at is not None
            assert lens_client._is_circuit_breaker_open() is True
    
    @pytest.mark.asyncio
    async def test_successful_request(self, lens_client):
        """Test successful API request."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {"result": "success", "data": {"id": 123}}
        
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            response = await lens_client._make_request(
                "POST", 
                "/api/test",
                data={"key": "value"}
            )
            
            assert response.success is True
            assert response.data == {"result": "success", "data": {"id": 123}}
            assert response.status_code == 200
            assert response.from_fallback is False
    
    @pytest.mark.asyncio
    async def test_request_with_retries(self, lens_client):
        """Test request with retry logic."""
        # Mock responses: first fails, second succeeds
        mock_responses = [
            AsyncMock(status=500, json=AsyncMock(return_value={"error": "Server error"})),
            AsyncMock(status=200, json=AsyncMock(return_value={"result": "success"}))
        ]
        
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.return_value.__aenter__.side_effect = mock_responses
            
            response = await lens_client._make_request("GET", "/api/test")
            
            assert response.success is True
            assert response.data == {"result": "success"}
            # Should have been called twice due to retry
            assert mock_session.get.call_count == 2
    
    @pytest.mark.asyncio
    async def test_request_timeout_with_fallback(self, lens_client):
        """Test request timeout with fallback enabled."""
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.side_effect = asyncio.TimeoutError()
            
            # Should use fallback instead of raising exception
            response = await lens_client._make_request("GET", "/api/test")
            
            assert response.success is True
            assert response.from_fallback is True
            assert "fallback" in response.data
    
    @pytest.mark.asyncio
    async def test_request_without_fallback(self, lens_config):
        """Test request timeout without fallback."""
        lens_config.fallback_enabled = False
        client = LensIntegrationClient(lens_config)
        
        with patch.object(client, 'session') as mock_session:
            mock_session.get.side_effect = asyncio.TimeoutError()
            
            # Should raise exception without fallback
            with pytest.raises(LensTimeoutError):
                await client._make_request("GET", "/api/test")
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_index_repository(self, lens_client):
        """Test repository indexing."""
        request = LensIndexRequest(
            repository_path="/path/to/repo",
            repository_id="test-repo",
            branch="main",
            force_reindex=True
        )
        
        # Mock successful indexing response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "status": "success",
            "repository_id": "test-repo",
            "indexed_files": 42
        }
        
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.post.return_value.__aenter__.return_value = mock_response
            
            response = await lens_client.index_repository(request)
            
            assert response.success is True
            assert response.data["repository_id"] == "test-repo"
            assert response.data["indexed_files"] == 42
    
    @pytest.mark.asyncio
    async def test_search_repository(self, lens_client):
        """Test repository search."""
        request = LensSearchRequest(
            query="test function",
            repository_id="test-repo",
            max_results=10
        )
        
        # Mock successful search response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "results": [
                {"file": "test.py", "line": 10, "score": 0.95},
                {"file": "utils.py", "line": 25, "score": 0.87}
            ],
            "total": 2,
            "query_time_ms": 45
        }
        
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            response = await lens_client.search_repository(request)
            
            assert response.success is True
            assert len(response.data["results"]) == 2
            assert response.data["total"] == 2
            assert response.data["query_time_ms"] == 45
    
    @pytest.mark.asyncio
    async def test_get_embeddings(self, lens_client):
        """Test getting embeddings."""
        # Mock successful embeddings response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "repository_id": "test-repo",
            "embeddings": [0.1, 0.2, 0.3, 0.4, 0.5],
            "model": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        with patch.object(lens_client, 'session') as mock_session:
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            response = await lens_client.get_embeddings("test-repo", "test.py")
            
            assert response.success is True
            assert response.data["repository_id"] == "test-repo"
            assert len(response.data["embeddings"]) == 5
    
    @pytest.mark.asyncio
    async def test_disabled_features(self, lens_config):
        """Test behavior when features are disabled."""
        lens_config.enabled = False
        client = LensIntegrationClient(lens_config)
        
        # All operations should return disabled error
        index_request = LensIndexRequest(
            repository_path="/path",
            repository_id="test"
        )
        response = await client.index_repository(index_request)
        assert response.success is False
        assert "disabled" in response.error
        
        search_request = LensSearchRequest(query="test")
        response = await client.search_repository(search_request)
        assert response.success is False
        assert "disabled" in response.error
        
        response = await client.get_embeddings("test")
        assert response.success is False
        assert "disabled" in response.error
        
        await client.close()
    
    def test_health_status_methods(self, lens_client):
        """Test health status utility methods."""
        # Initially unknown
        assert not lens_client.is_healthy()
        assert lens_client.get_health_status() == LensHealthStatus.UNKNOWN
        assert lens_client.get_last_health_check() is None
        
        # Set to healthy
        lens_client._health_status = LensHealthStatus.HEALTHY
        lens_client._last_health_check = datetime.now()
        
        assert lens_client.is_healthy()
        assert lens_client.get_health_status() == LensHealthStatus.HEALTHY
        assert lens_client.get_last_health_check() is not None
    
    @pytest.mark.asyncio
    async def test_ensure_healthy(self, lens_client):
        """Test ensure_healthy method."""
        # Mock health check to return healthy immediately
        with patch.object(lens_client, 'check_health') as mock_health:
            mock_health.return_value.status = LensHealthStatus.HEALTHY
            
            result = await lens_client.ensure_healthy(max_wait_seconds=5)
            assert result is True
            mock_health.assert_called_once()
        
        # Mock health check to always return unhealthy
        with patch.object(lens_client, 'check_health') as mock_health:
            mock_health.return_value.status = LensHealthStatus.UNHEALTHY
            
            result = await lens_client.ensure_healthy(max_wait_seconds=1)
            assert result is False


class TestLensGlobalFunctions:
    """Test global Lens client functions."""
    
    def test_get_lens_client(self, lens_config):
        """Test getting global Lens client."""
        # Reset global client
        import src.repoindex.pipeline.lens_client as lens_module
        lens_module._global_lens_client = None
        
        client1 = get_lens_client(lens_config)
        client2 = get_lens_client()
        
        # Should return same instance
        assert client1 is client2
    
    @pytest.mark.asyncio
    async def test_init_lens_client(self, lens_config):
        """Test initializing Lens client."""
        with patch('src.repoindex.pipeline.lens_client.LensIntegrationClient') as MockClient:
            mock_instance = AsyncMock()
            MockClient.return_value = mock_instance
            
            client = await init_lens_client(lens_config)
            
            MockClient.assert_called_once_with(lens_config)
            mock_instance.initialize.assert_called_once()
            assert client == mock_instance


class TestLensFallbackBehavior:
    """Test Lens fallback behavior in detail."""
    
    @pytest.mark.asyncio
    async def test_search_fallback(self, lens_client):
        """Test search fallback response."""
        response = await lens_client._fallback_response("GET", "/api/v1/search")
        
        assert response.success is True
        assert response.from_fallback is True
        assert response.data["fallback"] is True
        assert response.data["results"] == []
        assert response.data["total"] == 0
    
    @pytest.mark.asyncio
    async def test_index_fallback(self, lens_client):
        """Test indexing fallback response."""
        response = await lens_client._fallback_response(
            "POST", 
            "/api/v1/index",
            data={"repository_id": "test"}
        )
        
        assert response.success is True
        assert response.from_fallback is True
        assert response.data["fallback"] is True
        assert response.data["indexed"] is False
        assert response.data["status"] == "deferred"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_with_fallback(self, lens_client):
        """Test circuit breaker triggering fallback."""
        # Open circuit breaker
        lens_client._consecutive_failures = 3
        lens_client._circuit_breaker_opened_at = datetime.now()
        
        response = await lens_client._make_request("GET", "/api/v1/search")
        
        assert response.success is True
        assert response.from_fallback is True


@pytest.mark.asyncio
async def test_integration_workflow():
    """Test complete integration workflow."""
    config = LensConfig(
        enabled=True,
        base_url="http://localhost:3001",
        health_check_enabled=True,
        fallback_enabled=True
    )
    
    async with LensIntegrationClient(config) as client:
        # This would be a real integration test if Lens service was running
        # For now, just test the client can be created and configured properly
        assert client.config.enabled is True
        assert client.config.base_url == "http://localhost:3001"
        assert client.session is not None