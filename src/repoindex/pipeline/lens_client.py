"""
Lens Integration Client for Mimir-Lens Communication.

Provides async HTTP client for communication with Lens indexing service,
including health checks, error handling, and fallback mechanisms.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import aiohttp
from pydantic import BaseModel, Field

from ..config import get_lens_config
from ..util.log import get_logger

logger = get_logger(__name__)


class LensHealthStatus(Enum):
    """Lens service health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"


@dataclass
class LensResponse:
    """Standard Lens API response wrapper."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    status_code: int = 200
    response_time_ms: float = 0.0
    from_fallback: bool = False


@dataclass
class LensHealthCheck:
    """Lens service health check result."""
    status: LensHealthStatus
    timestamp: datetime
    response_time_ms: float
    version: Optional[str] = None
    uptime_seconds: Optional[int] = None
    error: Optional[str] = None


class LensIndexRequest(BaseModel):
    """Request model for indexing operations."""
    repository_path: str
    repository_id: str
    branch: str = "main"
    force_reindex: bool = False
    include_embeddings: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # GPU preferences (for future Lens GPU support)
    prefer_gpu: bool = False
    gpu_device_id: Optional[int] = None
    gpu_batch_size: Optional[int] = None
    gpu_memory_limit: Optional[str] = None


class LensSearchRequest(BaseModel):
    """Request model for search operations."""
    query: str
    repository_id: Optional[str] = None
    max_results: int = 20
    include_embeddings: bool = False
    filters: Dict[str, Any] = Field(default_factory=dict)


class LensIntegrationError(Exception):
    """Base exception for Lens integration errors."""
    pass


class LensConnectionError(LensIntegrationError):
    """Raised when connection to Lens service fails."""
    pass


class LensServiceError(LensIntegrationError):
    """Raised when Lens service returns an error."""
    pass


class LensTimeoutError(LensIntegrationError):
    """Raised when Lens service request times out."""
    pass


class LensIntegrationClient:
    """
    HTTP client for communicating with Lens indexing service.
    
    Features:
    - Async HTTP communication with connection pooling
    - Health checking and service monitoring
    - Error handling and retry logic
    - Fallback mechanisms for service unavailability
    - Request/response logging and metrics
    """
    
    def __init__(self, config=None):
        """
        Initialize Lens integration client.
        
        Args:
            config: LensConfig instance, defaults to global config
        """
        self.config = config or get_lens_config()
        self.session: Optional[aiohttp.ClientSession] = None
        self._health_status = LensHealthStatus.UNKNOWN
        self._last_health_check: Optional[datetime] = None
        self._consecutive_failures = 0
        self._circuit_breaker_opened_at: Optional[datetime] = None
        
        logger.info(f"Initializing Lens client with base_url: {self.config.base_url}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def initialize(self) -> None:
        """Initialize the HTTP session and connection pool."""
        if self.session and not self.session.closed:
            return
        
        # Configure connection pool and timeouts
        connector = aiohttp.TCPConnector(
            limit=self.config.connection_pool_size,
            keepalive_timeout=self.config.keep_alive_timeout,
            enable_cleanup_closed=True
        )
        
        timeout = aiohttp.ClientTimeout(
            total=self.config.timeout,
            connect=10  # 10 second connect timeout
        )
        
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mimir-Lens-Client/1.0"
        }
        
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=headers
        )
        
        logger.info("Lens client HTTP session initialized")
        
        # Perform initial health check if enabled
        if self.config.health_check_enabled:
            await self.check_health()
    
    async def close(self) -> None:
        """Close the HTTP session and cleanup resources."""
        if self.session and not self.session.closed:
            await self.session.close()
            logger.info("Lens client HTTP session closed")
    
    async def check_health(self) -> LensHealthCheck:
        """
        Check the health status of the Lens service.
        
        Returns:
            LensHealthCheck with current status
        """
        start_time = time.time()
        
        try:
            if not self.session:
                await self.initialize()
            
            health_url = f"{self.config.base_url}/health"
            
            async with self.session.get(
                health_url,
                timeout=aiohttp.ClientTimeout(total=self.config.health_check_timeout)
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    data = await response.json()
                    
                    health_check = LensHealthCheck(
                        status=LensHealthStatus.HEALTHY,
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        version=data.get("version"),
                        uptime_seconds=data.get("uptime")
                    )
                    
                    self._health_status = LensHealthStatus.HEALTHY
                    self._consecutive_failures = 0
                    self._circuit_breaker_opened_at = None
                    
                else:
                    health_check = LensHealthCheck(
                        status=LensHealthStatus.DEGRADED,
                        timestamp=datetime.now(),
                        response_time_ms=response_time,
                        error=f"HTTP {response.status}"
                    )
                    
                    self._health_status = LensHealthStatus.DEGRADED
        
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            health_check = LensHealthCheck(
                status=LensHealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                error="Health check timeout"
            )
            self._health_status = LensHealthStatus.UNHEALTHY
            self._consecutive_failures += 1
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health_check = LensHealthCheck(
                status=LensHealthStatus.UNHEALTHY,
                timestamp=datetime.now(),
                response_time_ms=response_time,
                error=str(e)
            )
            self._health_status = LensHealthStatus.UNHEALTHY
            self._consecutive_failures += 1
        
        self._last_health_check = health_check.timestamp
        
        # Open circuit breaker if too many failures
        if (self._consecutive_failures >= 3 and 
            self._circuit_breaker_opened_at is None):
            self._circuit_breaker_opened_at = datetime.now()
            logger.warning(f"Lens circuit breaker opened after {self._consecutive_failures} failures")
        
        logger.debug(f"Lens health check: {health_check.status.value} in {response_time:.1f}ms")
        return health_check
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is currently open."""
        if self._circuit_breaker_opened_at is None:
            return False
        
        # Auto-reset after 60 seconds
        if datetime.now() - self._circuit_breaker_opened_at > timedelta(seconds=60):
            self._circuit_breaker_opened_at = None
            self._consecutive_failures = 0
            logger.info("Lens circuit breaker auto-reset")
            return False
        
        return True
    
    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> LensResponse:
        """
        Make an HTTP request to Lens service with error handling and retries.
        
        Args:
            method: HTTP method
            endpoint: API endpoint (without base URL)
            data: Request body data
            params: Query parameters
            
        Returns:
            LensResponse with result or error
        """
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            if self.config.fallback_enabled:
                logger.warning("Circuit breaker open, using fallback")
                return await self._fallback_response(method, endpoint, data)
            else:
                raise LensConnectionError("Circuit breaker open, service unavailable")
        
        # Ensure session is initialized
        if not self.session:
            await self.initialize()
        
        url = f"{self.config.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        start_time = time.time()
        
        for attempt in range(self.config.max_retries + 1):
            try:
                kwargs = {
                    "url": url,
                    "params": params
                }
                
                if data is not None:
                    kwargs["json"] = data
                
                async with getattr(self.session, method.lower())(**kwargs) as response:
                    response_time = (time.time() - start_time) * 1000
                    
                    # Handle successful responses
                    if 200 <= response.status < 300:
                        try:
                            result_data = await response.json()
                        except:
                            result_data = await response.text()
                        
                        self._consecutive_failures = 0
                        
                        return LensResponse(
                            success=True,
                            data=result_data,
                            status_code=response.status,
                            response_time_ms=response_time
                        )
                    
                    # Handle error responses
                    else:
                        try:
                            error_data = await response.json()
                            error_msg = error_data.get("error", f"HTTP {response.status}")
                        except:
                            error_msg = f"HTTP {response.status}"
                        
                        # Don't retry client errors (4xx)
                        if 400 <= response.status < 500:
                            return LensResponse(
                                success=False,
                                error=error_msg,
                                status_code=response.status,
                                response_time_ms=response_time
                            )
                        
                        # Retry server errors (5xx) and other errors
                        if attempt < self.config.max_retries:
                            await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                            continue
                        else:
                            self._consecutive_failures += 1
                            return LensResponse(
                                success=False,
                                error=error_msg,
                                status_code=response.status,
                                response_time_ms=response_time
                            )
            
            except asyncio.TimeoutError:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    self._consecutive_failures += 1
                    response_time = (time.time() - start_time) * 1000
                    
                    if self.config.fallback_enabled:
                        logger.warning(f"Request timeout after {self.config.max_retries} retries, using fallback")
                        return await self._fallback_response(method, endpoint, data)
                    else:
                        raise LensTimeoutError(f"Request timeout after {self.config.max_retries} retries")
            
            except Exception as e:
                if attempt < self.config.max_retries:
                    logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    continue
                else:
                    self._consecutive_failures += 1
                    
                    if self.config.fallback_enabled:
                        logger.error(f"Request failed after {self.config.max_retries} retries, using fallback: {e}")
                        return await self._fallback_response(method, endpoint, data)
                    else:
                        raise LensConnectionError(f"Request failed after {self.config.max_retries} retries: {e}")
    
    async def _fallback_response(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None
    ) -> LensResponse:
        """
        Generate fallback response when Lens service is unavailable.
        
        Args:
            method: HTTP method of original request
            endpoint: API endpoint of original request  
            data: Original request data
            
        Returns:
            LensResponse indicating fallback was used
        """
        fallback_data = {
            "message": "Lens service unavailable, using local fallback",
            "fallback": True,
            "original_endpoint": endpoint,
            "timestamp": datetime.now().isoformat()
        }
        
        # For search requests, return empty results
        if "search" in endpoint.lower():
            fallback_data.update({
                "results": [],
                "total": 0,
                "query_time_ms": 0
            })
        
        # For indexing requests, simulate success
        elif "index" in endpoint.lower():
            fallback_data.update({
                "indexed": False,
                "status": "deferred",
                "message": "Indexing deferred until Lens service is available"
            })
        
        return LensResponse(
            success=True,
            data=fallback_data,
            status_code=200,
            response_time_ms=1.0,
            from_fallback=True
        )
    
    async def check_gpu_capabilities(self) -> Dict[str, Any]:
        """
        Check GPU capabilities of the Lens service.
        
        Returns:
            Dict containing GPU capability information
        """
        try:
            response = await self._make_request("GET", "/api/capabilities/gpu")
            if response.success:
                return response.data or {
                    "gpu_available": False,
                    "gpu_devices": [],
                    "fallback_reason": "GPU info unavailable"
                }
            else:
                return {
                    "gpu_available": False,
                    "gpu_devices": [],
                    "fallback_reason": f"API error: {response.error}"
                }
        except Exception as e:
            logger.warning(f"Failed to check GPU capabilities: {e}")
            return {
                "gpu_available": False,
                "gpu_devices": [],
                "fallback_reason": f"Connection error: {str(e)}"
            }
    
    # API Methods
    
    async def index_repository(self, request: LensIndexRequest) -> LensResponse:
        """
        Index a repository with Lens service.
        
        Args:
            request: LensIndexRequest with repository details
            
        Returns:
            LensResponse with indexing result
        """
        if not self.config.enabled or not self.config.enable_indexing:
            return LensResponse(
                success=False,
                error="Lens indexing is disabled",
                status_code=503
            )
        
        data = request.dict()
        return await self._make_request("POST", "/api/v1/index", data=data)
    
    async def search_repository(self, request: LensSearchRequest) -> LensResponse:
        """
        Search indexed repositories using Lens service.
        
        Args:
            request: LensSearchRequest with search parameters
            
        Returns:
            LensResponse with search results
        """
        if not self.config.enabled or not self.config.enable_search:
            return LensResponse(
                success=False,
                error="Lens search is disabled", 
                status_code=503
            )
        
        params = request.dict()
        return await self._make_request("GET", "/api/v1/search", params=params)
    
    async def get_embeddings(self, repository_id: str, file_path: str = None) -> LensResponse:
        """
        Get embeddings for repository or specific file.
        
        Args:
            repository_id: Repository identifier
            file_path: Optional file path within repository
            
        Returns:
            LensResponse with embedding data
        """
        if not self.config.enabled or not self.config.enable_embeddings:
            return LensResponse(
                success=False,
                error="Lens embeddings are disabled",
                status_code=503
            )
        
        params = {"repository_id": repository_id}
        if file_path:
            params["file_path"] = file_path
        
        return await self._make_request("GET", "/api/v1/embeddings", params=params)
    
    async def get_repository_status(self, repository_id: str) -> LensResponse:
        """
        Get indexing status for a repository.
        
        Args:
            repository_id: Repository identifier
            
        Returns:
            LensResponse with repository status
        """
        params = {"repository_id": repository_id}
        return await self._make_request("GET", "/api/v1/status", params=params)
    
    # Utility Methods
    
    def is_healthy(self) -> bool:
        """Check if Lens service is currently healthy."""
        return self._health_status == LensHealthStatus.HEALTHY
    
    def get_health_status(self) -> LensHealthStatus:
        """Get current health status."""
        return self._health_status
    
    def get_last_health_check(self) -> Optional[datetime]:
        """Get timestamp of last health check."""
        return self._last_health_check
    
    async def ensure_healthy(self, max_wait_seconds: int = 30) -> bool:
        """
        Ensure Lens service is healthy, waiting if necessary.
        
        Args:
            max_wait_seconds: Maximum time to wait for health
            
        Returns:
            True if service becomes healthy, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < max_wait_seconds:
            health = await self.check_health()
            
            if health.status == LensHealthStatus.HEALTHY:
                return True
            
            # Wait before next check
            await asyncio.sleep(min(5, max_wait_seconds - (time.time() - start_time)))
        
        return False


# Global client instance
_global_lens_client: Optional[LensIntegrationClient] = None


def get_lens_client(config=None) -> LensIntegrationClient:
    """
    Get global Lens integration client instance.
    
    Args:
        config: Optional LensConfig, uses global config if not provided
        
    Returns:
        LensIntegrationClient instance
    """
    global _global_lens_client
    
    if _global_lens_client is None:
        _global_lens_client = LensIntegrationClient(config)
    
    return _global_lens_client


async def init_lens_client(config=None) -> LensIntegrationClient:
    """
    Initialize and return Lens integration client.
    
    Args:
        config: Optional LensConfig
        
    Returns:
        Initialized LensIntegrationClient instance
    """
    client = get_lens_client(config)
    await client.initialize()
    return client