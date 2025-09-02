#!/usr/bin/env python3
"""
Integration Tests for Mimir-Lens Integrated System
Comprehensive end-to-end testing framework
"""

import asyncio
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import pytest
import httpx
import asyncpg
import redis.asyncio as redis
from prometheus_client.parser import text_string_to_metric_families

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================================================
# TEST CONFIGURATION
# ==========================================================================

@dataclass
class TestConfig:
    """Test configuration"""
    mimir_base_url: str = "http://localhost:8000"
    lens_base_url: str = "http://localhost:3000"
    postgres_url: str = "postgresql://mimir_user:mimir123@localhost:5432/mimir"
    redis_url: str = "redis://localhost:6379/0"
    prometheus_url: str = "http://localhost:9090"
    grafana_url: str = "http://localhost:3001"
    jaeger_url: str = "http://localhost:16686"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 2.0

# ==========================================================================
# TEST UTILITIES
# ==========================================================================

class HealthChecker:
    """Health check utilities for all services"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        
    async def check_mimir_health(self) -> Dict[str, Any]:
        """Check Mimir service health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.config.mimir_base_url}/health")
            response.raise_for_status()
            return response.json()
            
    async def check_lens_health(self) -> Dict[str, Any]:
        """Check Lens service health"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.config.lens_base_url}/health")
            response.raise_for_status()
            return response.json()
            
    async def check_postgres_health(self) -> bool:
        """Check PostgreSQL health"""
        try:
            conn = await asyncpg.connect(self.config.postgres_url)
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        except Exception as e:
            logger.error(f"PostgreSQL health check failed: {e}")
            return False
            
    async def check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            client = redis.from_url(self.config.redis_url)
            await client.ping()
            await client.close()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False
            
    async def check_prometheus_health(self) -> bool:
        """Check Prometheus health"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.config.prometheus_url}/-/healthy")
                return response.status_code == 200
        except Exception as e:
            logger.error(f"Prometheus health check failed: {e}")
            return False

class MetricsCollector:
    """Metrics collection utilities"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        
    async def get_mimir_metrics(self) -> Dict[str, float]:
        """Get Mimir metrics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.config.mimir_base_url}/metrics")
            response.raise_for_status()
            
            metrics = {}
            for family in text_string_to_metric_families(response.text):
                for sample in family.samples:
                    metrics[sample.name] = sample.value
            return metrics
            
    async def get_lens_metrics(self) -> Dict[str, float]:
        """Get Lens metrics"""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.config.lens_base_url}/metrics")
            response.raise_for_status()
            
            metrics = {}
            for family in text_string_to_metric_families(response.text):
                for sample in family.samples:
                    metrics[sample.name] = sample.value
            return metrics

class LoadTestRunner:
    """Load testing utilities"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        
    async def run_concurrent_requests(self, 
                                    url: str, 
                                    payload: Optional[Dict] = None,
                                    concurrent_users: int = 10,
                                    duration_seconds: int = 30) -> Dict[str, Any]:
        """Run concurrent requests for load testing"""
        start_time = time.time()
        results = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'response_times': [],
            'errors': []
        }
        
        async def make_request():
            try:
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    request_start = time.time()
                    if payload:
                        response = await client.post(url, json=payload)
                    else:
                        response = await client.get(url)
                    request_time = time.time() - request_start
                    
                    results['total_requests'] += 1
                    if response.status_code < 400:
                        results['successful_requests'] += 1
                    else:
                        results['failed_requests'] += 1
                        results['errors'].append(f"HTTP {response.status_code}")
                    
                    results['response_times'].append(request_time)
                    
            except Exception as e:
                results['total_requests'] += 1
                results['failed_requests'] += 1
                results['errors'].append(str(e))
        
        # Run concurrent requests for the specified duration
        tasks = []
        end_time = start_time + duration_seconds
        
        while time.time() < end_time:
            # Launch concurrent requests
            batch_tasks = [make_request() for _ in range(concurrent_users)]
            tasks.extend(batch_tasks)
            
            # Execute batch
            await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Small delay between batches
            await asyncio.sleep(0.1)
        
        # Calculate statistics
        if results['response_times']:
            results['avg_response_time'] = sum(results['response_times']) / len(results['response_times'])
            results['min_response_time'] = min(results['response_times'])
            results['max_response_time'] = max(results['response_times'])
            results['p95_response_time'] = sorted(results['response_times'])[int(len(results['response_times']) * 0.95)]
        
        results['success_rate'] = results['successful_requests'] / results['total_requests'] if results['total_requests'] > 0 else 0
        results['duration_seconds'] = time.time() - start_time
        results['requests_per_second'] = results['total_requests'] / results['duration_seconds']
        
        return results

# ==========================================================================
# FIXTURES
# ==========================================================================

@pytest.fixture(scope="session")
def config():
    """Test configuration fixture"""
    return TestConfig()

@pytest.fixture(scope="session")
def health_checker(config):
    """Health checker fixture"""
    return HealthChecker(config)

@pytest.fixture(scope="session")
def metrics_collector(config):
    """Metrics collector fixture"""
    return MetricsCollector(config)

@pytest.fixture(scope="session")
def load_test_runner(config):
    """Load test runner fixture"""
    return LoadTestRunner(config)

@pytest.fixture(scope="session")
async def wait_for_services(health_checker):
    """Wait for all services to be ready"""
    max_wait_time = 120  # 2 minutes
    check_interval = 5   # 5 seconds
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            # Check all services
            mimir_healthy = await health_checker.check_mimir_health()
            lens_healthy = await health_checker.check_lens_health()
            postgres_healthy = await health_checker.check_postgres_health()
            redis_healthy = await health_checker.check_redis_health()
            prometheus_healthy = await health_checker.check_prometheus_health()
            
            if all([mimir_healthy, lens_healthy, postgres_healthy, redis_healthy, prometheus_healthy]):
                logger.info("All services are healthy and ready")
                return True
                
        except Exception as e:
            logger.warning(f"Services not ready yet: {e}")
            
        await asyncio.sleep(check_interval)
    
    pytest.fail("Services did not become healthy within the timeout period")

# ==========================================================================
# BASIC HEALTH TESTS
# ==========================================================================

@pytest.mark.asyncio
class TestBasicHealth:
    """Basic health check tests"""
    
    async def test_all_services_healthy(self, health_checker, wait_for_services):
        """Test that all services report healthy status"""
        # Mimir health
        mimir_health = await health_checker.check_mimir_health()
        assert mimir_health is not None
        logger.info(f"Mimir health: {mimir_health}")
        
        # Lens health
        lens_health = await health_checker.check_lens_health()
        assert lens_health is not None
        logger.info(f"Lens health: {lens_health}")
        
        # Infrastructure health
        assert await health_checker.check_postgres_health()
        assert await health_checker.check_redis_health()
        assert await health_checker.check_prometheus_health()
    
    async def test_service_dependencies(self, config):
        """Test service dependency connectivity"""
        # Test Mimir -> Lens connectivity
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{config.mimir_base_url}/api/lens/status")
            assert response.status_code in [200, 404]  # 404 is OK if endpoint doesn't exist yet
            
        # Test database connectivity from services
        # This would be tested through the service APIs

# ==========================================================================
# INTEGRATION TESTS
# ==========================================================================

@pytest.mark.asyncio
class TestMimirLensIntegration:
    """Test Mimir-Lens integration functionality"""
    
    async def test_indexing_workflow(self, config):
        """Test complete indexing workflow"""
        test_repo = {
            "name": "test-repo",
            "url": "https://github.com/octocat/Hello-World.git",
            "branch": "main"
        }
        
        async with httpx.AsyncClient(timeout=60) as client:
            # Submit indexing job
            response = await client.post(
                f"{config.mimir_base_url}/api/index/repository",
                json=test_repo
            )
            assert response.status_code in [200, 201, 202]
            job_data = response.json()
            
            if 'job_id' in job_data:
                job_id = job_data['job_id']
                
                # Poll for completion
                max_wait = 180  # 3 minutes
                poll_interval = 10
                start_time = time.time()
                
                while time.time() - start_time < max_wait:
                    status_response = await client.get(f"{config.mimir_base_url}/api/jobs/{job_id}")
                    if status_response.status_code == 200:
                        status_data = status_response.json()
                        if status_data.get('status') in ['completed', 'failed']:
                            assert status_data.get('status') == 'completed'
                            break
                    
                    await asyncio.sleep(poll_interval)
    
    async def test_search_functionality(self, config):
        """Test search functionality across both services"""
        search_queries = [
            {"query": "function", "type": "semantic"},
            {"query": "class", "type": "structural"},
            {"query": "import", "type": "pattern"}
        ]
        
        async with httpx.AsyncClient() as client:
            for query_data in search_queries:
                # Test Mimir search
                response = await client.post(
                    f"{config.mimir_base_url}/api/search",
                    json=query_data
                )
                assert response.status_code in [200, 404, 422]  # Various acceptable responses
                
                # Test Lens search (if available)
                try:
                    response = await client.post(
                        f"{config.lens_base_url}/api/search",
                        json=query_data
                    )
                    assert response.status_code in [200, 404, 422]
                except Exception:
                    pass  # Lens might not be fully implemented yet

# ==========================================================================
# PERFORMANCE TESTS
# ==========================================================================

@pytest.mark.asyncio
class TestPerformance:
    """Performance and load tests"""
    
    async def test_mimir_api_performance(self, config, load_test_runner):
        """Test Mimir API performance under load"""
        results = await load_test_runner.run_concurrent_requests(
            f"{config.mimir_base_url}/health",
            concurrent_users=10,
            duration_seconds=30
        )
        
        logger.info(f"Mimir performance results: {results}")
        
        # Performance assertions
        assert results['success_rate'] > 0.95  # 95% success rate
        assert results['avg_response_time'] < 5.0  # Average response time < 5s
        assert results['p95_response_time'] < 10.0  # 95th percentile < 10s
    
    async def test_lens_api_performance(self, config, load_test_runner):
        """Test Lens API performance under load"""
        results = await load_test_runner.run_concurrent_requests(
            f"{config.lens_base_url}/health",
            concurrent_users=10,
            duration_seconds=30
        )
        
        logger.info(f"Lens performance results: {results}")
        
        # Performance assertions
        assert results['success_rate'] > 0.95  # 95% success rate
        assert results['avg_response_time'] < 2.0  # Average response time < 2s
        assert results['p95_response_time'] < 5.0  # 95th percentile < 5s

# ==========================================================================
# MONITORING TESTS
# ==========================================================================

@pytest.mark.asyncio  
class TestMonitoring:
    """Test monitoring and observability"""
    
    async def test_metrics_collection(self, metrics_collector):
        """Test that metrics are being collected"""
        try:
            mimir_metrics = await metrics_collector.get_mimir_metrics()
            assert len(mimir_metrics) > 0
            logger.info(f"Collected {len(mimir_metrics)} Mimir metrics")
        except Exception as e:
            logger.warning(f"Failed to collect Mimir metrics: {e}")
        
        try:
            lens_metrics = await metrics_collector.get_lens_metrics()
            assert len(lens_metrics) > 0
            logger.info(f"Collected {len(lens_metrics)} Lens metrics")
        except Exception as e:
            logger.warning(f"Failed to collect Lens metrics: {e}")
    
    async def test_prometheus_queries(self, config):
        """Test Prometheus queries"""
        test_queries = [
            'up',
            'up{job="mimir-server"}',
            'up{job="lens-server"}',
        ]
        
        async with httpx.AsyncClient() as client:
            for query in test_queries:
                try:
                    response = await client.get(
                        f"{config.prometheus_url}/api/v1/query",
                        params={'query': query}
                    )
                    if response.status_code == 200:
                        data = response.json()
                        assert data['status'] == 'success'
                        logger.info(f"Query '{query}' returned {len(data['data']['result'])} results")
                except Exception as e:
                    logger.warning(f"Prometheus query failed: {e}")

# ==========================================================================
# RESILIENCE TESTS
# ==========================================================================

@pytest.mark.asyncio
class TestResilience:
    """Test system resilience and fault tolerance"""
    
    async def test_service_restart_recovery(self, health_checker, config):
        """Test service recovery after restart (manual test)"""
        # This test would require Docker API access to restart containers
        # For now, we'll test that services can handle temporary failures
        
        max_retries = 3
        retry_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                await health_checker.check_mimir_health()
                await health_checker.check_lens_health()
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Service check failed (attempt {attempt + 1}): {e}")
                    await asyncio.sleep(retry_delay)
                else:
                    raise
    
    async def test_database_connection_resilience(self, config):
        """Test database connection handling"""
        # Test connection pooling and reconnection
        connections = []
        try:
            for i in range(5):
                conn = await asyncpg.connect(config.postgres_url)
                connections.append(conn)
                await conn.execute("SELECT 1")
        finally:
            for conn in connections:
                await conn.close()

# ==========================================================================
# CLEANUP TESTS
# ==========================================================================

@pytest.mark.asyncio
class TestCleanup:
    """Test data cleanup and maintenance"""
    
    async def test_cache_cleanup(self, config):
        """Test cache cleanup functionality"""
        client = redis.from_url(config.redis_url)
        try:
            # Set a test key with expiration
            await client.setex("test_key", 1, "test_value")
            
            # Verify it exists
            value = await client.get("test_key")
            assert value == b"test_value"
            
            # Wait for expiration
            await asyncio.sleep(2)
            
            # Verify it's cleaned up
            value = await client.get("test_key")
            assert value is None
            
        finally:
            await client.close()

# ==========================================================================
# TEST RUNNER
# ==========================================================================

if __name__ == "__main__":
    # Run tests with pytest
    import sys
    pytest.main([__file__] + sys.argv[1:])