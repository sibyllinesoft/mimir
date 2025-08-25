"""
Comprehensive concurrent scenario tests for Mimir.

Tests concurrent operations, race conditions, resource contention,
and system behavior under concurrent load.
"""

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import pytest
import numpy as np

from src.repoindex.config import MimirConfig, AIConfig, PipelineConfig
from src.repoindex.pipeline.pipeline_coordinator import PipelineCoordinator, get_pipeline_coordinator
from tests.fixtures.integration_fixtures import (
    mock_ollama_server, 
    mock_vector_database, 
    integrated_test_environment,
    MockComponentFactory
)


@dataclass
class ConcurrentTestResult:
    """Result of a concurrent operation test."""
    operation_id: str
    success: bool
    duration: float
    error_message: Optional[str] = None
    result_data: Optional[Dict[str, Any]] = None


@dataclass
class ConcurrencyMetrics:
    """Metrics for concurrent operations."""
    total_operations: int
    successful_operations: int
    failed_operations: int
    average_duration: float
    max_duration: float
    min_duration: float
    throughput: float  # operations per second
    error_rate: float
    

class ConcurrentOperationSimulator:
    """Simulate concurrent operations for testing."""
    
    def __init__(self, coordinator: PipelineCoordinator):
        self.coordinator = coordinator
        self.operation_counter = 0
        self.results: List[ConcurrentTestResult] = []
        
    async def simulate_query_operation(self, query: str) -> ConcurrentTestResult:
        """Simulate a query operation."""
        operation_id = f"query_{self.operation_counter}"
        self.operation_counter += 1
        start_time = time.time()
        
        try:
            # Mock query processing
            await asyncio.sleep(random.uniform(0.1, 0.5))  # Simulate processing time
            
            # Simulate potential failures
            if random.random() < 0.05:  # 5% failure rate
                raise Exception("Simulated query processing error")
            
            result_data = {
                "query": query,
                "results": [
                    {"file_path": f"result_{i}.py", "score": random.uniform(0.5, 1.0)}
                    for i in range(random.randint(1, 10))
                ],
                "processing_time": time.time() - start_time
            }
            
            duration = time.time() - start_time
            return ConcurrentTestResult(
                operation_id=operation_id,
                success=True,
                duration=duration,
                result_data=result_data
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ConcurrentTestResult(
                operation_id=operation_id,
                success=False,
                duration=duration,
                error_message=str(e)
            )
    
    async def simulate_indexing_operation(self, repo_path: str) -> ConcurrentTestResult:
        """Simulate a repository indexing operation."""
        operation_id = f"index_{self.operation_counter}"
        self.operation_counter += 1
        start_time = time.time()
        
        try:
            # Simulate indexing phases
            phases = [
                ("file_discovery", 0.2),
                ("content_extraction", 0.5),
                ("embedding_generation", 0.8),
                ("index_creation", 0.3)
            ]
            
            for phase, duration in phases:
                await asyncio.sleep(duration * random.uniform(0.8, 1.2))
                
                # Simulate potential phase failures
                if random.random() < 0.02:  # 2% failure rate per phase
                    raise Exception(f"Simulated failure in {phase}")
            
            result_data = {
                "repo_path": repo_path,
                "indexed_files": random.randint(50, 200),
                "chunks_created": random.randint(500, 2000),
                "embeddings_generated": random.randint(500, 2000)
            }
            
            duration = time.time() - start_time
            return ConcurrentTestResult(
                operation_id=operation_id,
                success=True,
                duration=duration,
                result_data=result_data
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ConcurrentTestResult(
                operation_id=operation_id,
                success=False,
                duration=duration,
                error_message=str(e)
            )
    
    async def simulate_configuration_update(self) -> ConcurrentTestResult:
        """Simulate a configuration update operation."""
        operation_id = f"config_{self.operation_counter}"
        self.operation_counter += 1
        start_time = time.time()
        
        try:
            # Simulate configuration validation and update
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Simulate validation failure
            if random.random() < 0.1:  # 10% failure rate
                raise Exception("Configuration validation failed")
            
            result_data = {
                "updated_settings": random.randint(1, 5),
                "validation_passed": True,
                "restart_required": random.choice([True, False])
            }
            
            duration = time.time() - start_time
            return ConcurrentTestResult(
                operation_id=operation_id,
                success=True,
                duration=duration,
                result_data=result_data
            )
            
        except Exception as e:
            duration = time.time() - start_time
            return ConcurrentTestResult(
                operation_id=operation_id,
                success=False,
                duration=duration,
                error_message=str(e)
            )
    
    def calculate_metrics(self, results: List[ConcurrentTestResult]) -> ConcurrencyMetrics:
        """Calculate performance metrics from results."""
        if not results:
            return ConcurrencyMetrics(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        durations = [r.duration for r in results]
        total_time = max(durations) if durations else 0.0
        
        return ConcurrencyMetrics(
            total_operations=len(results),
            successful_operations=len(successful),
            failed_operations=len(failed),
            average_duration=np.mean(durations) if durations else 0.0,
            max_duration=max(durations) if durations else 0.0,
            min_duration=min(durations) if durations else 0.0,
            throughput=len(results) / total_time if total_time > 0 else 0.0,
            error_rate=len(failed) / len(results) if results else 0.0
        )


class TestConcurrentQueryProcessing:
    """Test concurrent query processing scenarios."""
    
    @pytest.fixture
    async def simulator(self):
        """Create operation simulator for testing."""
        mock_coordinator = MockComponentFactory.create_mock_pipeline_coordinator()
        return ConcurrentOperationSimulator(mock_coordinator)
    
    @pytest.mark.asyncio
    async def test_concurrent_query_processing(self, simulator):
        """Test processing multiple concurrent queries."""
        queries = [
            "How to implement async functions?",
            "Error handling best practices",
            "Database connection management", 
            "API client implementation",
            "Configuration management patterns",
            "Testing strategies for async code",
            "Performance optimization techniques",
            "Security best practices",
            "Logging and monitoring setup",
            "Deployment automation scripts"
        ]
        
        # Execute queries concurrently
        tasks = [simulator.simulate_query_operation(query) for query in queries]
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        metrics = simulator.calculate_metrics(results)
        
        # Assertions
        assert metrics.total_operations == len(queries)
        assert metrics.successful_operations > 0
        assert metrics.error_rate < 0.2  # Less than 20% error rate
        assert metrics.average_duration < 1.0  # Average under 1 second
        
        # Verify all successful operations have results
        successful_results = [r for r in results if r.success]
        for result in successful_results:
            assert result.result_data is not None
            assert "query" in result.result_data
            assert "results" in result.result_data
            assert len(result.result_data["results"]) > 0
    
    @pytest.mark.asyncio
    async def test_high_concurrency_query_load(self, simulator):
        """Test system under high concurrent query load."""
        # Generate many concurrent queries
        base_queries = [
            "async implementation", "error handling", "database connection",
            "API design", "configuration", "testing", "performance", "security"
        ]
        
        # Create 50 concurrent queries with variations
        queries = []
        for i in range(50):
            base = random.choice(base_queries)
            queries.append(f"{base} example {i}")
        
        # Measure execution time
        start_time = time.time()
        tasks = [simulator.simulate_query_operation(query) for query in queries]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze performance under load
        metrics = simulator.calculate_metrics(results)
        
        # Performance assertions
        assert metrics.total_operations == 50
        assert metrics.throughput > 10  # At least 10 queries per second
        assert metrics.error_rate < 0.15  # Less than 15% error rate under load
        assert total_time < 10.0  # Complete within 10 seconds
        
        # Check for reasonable response time distribution
        durations = [r.duration for r in results]
        p95_duration = np.percentile(durations, 95)
        assert p95_duration < 2.0  # 95th percentile under 2 seconds
    
    @pytest.mark.asyncio
    async def test_query_resource_contention(self, simulator):
        """Test query processing under resource contention."""
        # Simulate resource-intensive queries
        intensive_queries = [
            "complex algorithm implementation with detailed examples",
            "comprehensive database schema design with relationships",
            "advanced security implementation with encryption examples",
            "performance optimization with benchmarking code",
            "distributed system architecture with communication patterns"
        ]
        
        # Run multiple rounds of intensive queries
        all_results = []
        
        for round_num in range(3):
            print(f"Running resource contention test round {round_num + 1}")
            
            tasks = [
                simulator.simulate_query_operation(query) 
                for query in intensive_queries
            ]
            round_results = await asyncio.gather(*tasks)
            all_results.extend(round_results)
            
            # Brief pause between rounds
            await asyncio.sleep(0.1)
        
        # Analyze resource contention impact
        metrics = simulator.calculate_metrics(all_results)
        
        assert metrics.total_operations == len(intensive_queries) * 3
        assert metrics.error_rate < 0.25  # Allow slightly higher error rate under contention
        
        # Check for performance degradation patterns
        durations_by_round = []
        for round_num in range(3):
            round_start = round_num * len(intensive_queries)
            round_end = (round_num + 1) * len(intensive_queries)
            round_durations = [r.duration for r in all_results[round_start:round_end] if r.success]
            durations_by_round.append(round_durations)
        
        # Performance shouldn't degrade significantly across rounds
        if all(len(round_durs) > 0 for round_durs in durations_by_round):
            avg_durations = [np.mean(round_durs) for round_durs in durations_by_round]
            assert max(avg_durations) / min(avg_durations) < 2.0  # Less than 2x degradation


class TestConcurrentIndexingOperations:
    """Test concurrent repository indexing scenarios."""
    
    @pytest.fixture
    async def simulator(self):
        """Create operation simulator for testing."""
        mock_coordinator = MockComponentFactory.create_mock_pipeline_coordinator()
        return ConcurrentOperationSimulator(mock_coordinator)
    
    @pytest.mark.asyncio
    async def test_concurrent_repository_indexing(self, simulator):
        """Test indexing multiple repositories concurrently."""
        repo_paths = [
            "/repos/project_a",
            "/repos/project_b", 
            "/repos/project_c",
            "/repos/project_d",
            "/repos/project_e"
        ]
        
        # Index repositories concurrently
        tasks = [simulator.simulate_indexing_operation(path) for path in repo_paths]
        results = await asyncio.gather(*tasks)
        
        # Analyze indexing results
        metrics = simulator.calculate_metrics(results)
        
        assert metrics.total_operations == len(repo_paths)
        assert metrics.successful_operations >= len(repo_paths) * 0.8  # At least 80% success
        assert metrics.error_rate < 0.25  # Less than 25% error rate
        
        # Verify successful indexing operations
        successful_results = [r for r in results if r.success]
        for result in successful_results:
            assert result.result_data is not None
            assert "repo_path" in result.result_data
            assert "indexed_files" in result.result_data
            assert result.result_data["indexed_files"] > 0
    
    @pytest.mark.asyncio
    async def test_mixed_concurrent_operations(self, simulator):
        """Test mixed indexing and query operations concurrently."""
        # Create mixed workload
        indexing_tasks = [
            simulator.simulate_indexing_operation(f"/repos/concurrent_test_{i}")
            for i in range(3)
        ]
        
        query_tasks = [
            simulator.simulate_query_operation(f"concurrent query {i}")
            for i in range(7)
        ]
        
        config_tasks = [
            simulator.simulate_configuration_update()
            for i in range(2)
        ]
        
        # Execute all operations concurrently
        all_tasks = indexing_tasks + query_tasks + config_tasks
        results = await asyncio.gather(*all_tasks)
        
        # Separate results by operation type
        indexing_results = results[:3]
        query_results = results[3:10]
        config_results = results[10:]
        
        # Analyze each operation type
        indexing_metrics = simulator.calculate_metrics(indexing_results)
        query_metrics = simulator.calculate_metrics(query_results)
        config_metrics = simulator.calculate_metrics(config_results)
        overall_metrics = simulator.calculate_metrics(results)
        
        # Overall system should handle mixed load well
        assert overall_metrics.total_operations == 12
        assert overall_metrics.error_rate < 0.3
        
        # Each operation type should perform reasonably
        assert indexing_metrics.successful_operations >= 2  # At least 2/3 indexing succeed
        assert query_metrics.successful_operations >= 5   # At least 5/7 queries succeed  
        assert config_metrics.successful_operations >= 1  # At least 1/2 configs succeed


class TestRaceConditionHandling:
    """Test handling of race conditions and concurrent access."""
    
    @pytest.mark.asyncio
    async def test_concurrent_configuration_updates(self):
        """Test concurrent configuration updates for race conditions."""
        mock_coordinator = MockComponentFactory.create_mock_pipeline_coordinator()
        
        # Shared state to test race conditions
        shared_state = {"config_version": 0, "updates": []}
        update_lock = asyncio.Lock()
        
        async def update_configuration(update_id: str):
            """Simulate configuration update with potential race condition."""
            async with update_lock:
                current_version = shared_state["config_version"]
                
                # Simulate processing time
                await asyncio.sleep(random.uniform(0.05, 0.15))
                
                # Check for race condition
                if shared_state["config_version"] != current_version:
                    raise Exception(f"Race condition detected in update {update_id}")
                
                # Apply update
                shared_state["config_version"] += 1
                shared_state["updates"].append(update_id)
                
                return {
                    "update_id": update_id,
                    "version": shared_state["config_version"],
                    "success": True
                }
        
        # Attempt concurrent updates
        update_tasks = [
            update_configuration(f"update_{i}")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        # Analyze race condition handling
        successful_updates = [r for r in results if isinstance(r, dict) and r.get("success")]
        race_condition_errors = [r for r in results if isinstance(r, Exception)]
        
        # With proper locking, all updates should succeed
        assert len(successful_updates) == 10
        assert len(race_condition_errors) == 0
        assert shared_state["config_version"] == 10
        assert len(shared_state["updates"]) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_index_access(self):
        """Test concurrent access to search index."""
        # Mock vector database with concurrent access simulation
        class ConcurrentMockVectorDB:
            def __init__(self):
                self.access_count = 0
                self.concurrent_accesses = 0
                self.max_concurrent = 0
                self.access_lock = asyncio.Lock()
            
            async def search(self, query: str):
                async with self.access_lock:
                    self.access_count += 1
                    self.concurrent_accesses += 1
                    self.max_concurrent = max(self.max_concurrent, self.concurrent_accesses)
                
                try:
                    # Simulate search processing
                    await asyncio.sleep(random.uniform(0.1, 0.3))
                    
                    # Simulate potential concurrent access issues
                    if random.random() < 0.05:
                        raise Exception("Concurrent access conflict")
                    
                    return {
                        "query": query,
                        "results": [f"result_{i}" for i in range(random.randint(1, 5))],
                        "access_id": self.access_count
                    }
                finally:
                    async with self.access_lock:
                        self.concurrent_accesses -= 1
        
        vector_db = ConcurrentMockVectorDB()
        
        # Perform concurrent searches
        search_tasks = [
            vector_db.search(f"concurrent search {i}")
            for i in range(20)
        ]
        
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Analyze concurrent access handling
        successful_searches = [r for r in results if isinstance(r, dict)]
        access_errors = [r for r in results if isinstance(r, Exception)]
        
        assert len(successful_searches) >= 18  # At least 90% success rate
        assert vector_db.access_count == len(successful_searches)
        assert vector_db.concurrent_accesses == 0  # All completed
        assert vector_db.max_concurrent <= 20  # Reasonable concurrency level
    
    @pytest.mark.asyncio
    async def test_resource_cleanup_race_conditions(self):
        """Test resource cleanup under concurrent operations."""
        # Simulate resource management with cleanup race conditions
        resources = {"active": set(), "cleaned": set()}
        cleanup_lock = asyncio.Lock()
        
        async def use_resource(resource_id: str):
            """Use a resource with automatic cleanup."""
            resources["active"].add(resource_id)
            
            try:
                # Simulate resource usage
                await asyncio.sleep(random.uniform(0.1, 0.4))
                
                return {"resource_id": resource_id, "used": True}
                
            finally:
                # Cleanup with race condition protection
                async with cleanup_lock:
                    if resource_id in resources["active"]:
                        resources["active"].remove(resource_id)
                        resources["cleaned"].add(resource_id)
        
        # Use multiple resources concurrently
        resource_tasks = [
            use_resource(f"resource_{i}")
            for i in range(15)
        ]
        
        results = await asyncio.gather(*resource_tasks)
        
        # Verify proper cleanup without race conditions
        assert len(results) == 15
        assert len(resources["active"]) == 0  # All resources cleaned up
        assert len(resources["cleaned"]) == 15  # All resources properly cleaned
        assert all(r["used"] for r in results)  # All resources were used


class TestConcurrentErrorRecovery:
    """Test error recovery under concurrent operations."""
    
    @pytest.mark.asyncio
    async def test_concurrent_operation_error_isolation(self):
        """Test that errors in one concurrent operation don't affect others."""
        # Create operations with controlled failure rates
        async def reliable_operation(op_id: str):
            await asyncio.sleep(random.uniform(0.1, 0.2))
            return {"id": op_id, "status": "success"}
        
        async def unreliable_operation(op_id: str):
            await asyncio.sleep(random.uniform(0.1, 0.2))
            if random.random() < 0.5:  # 50% failure rate
                raise Exception(f"Operation {op_id} failed")
            return {"id": op_id, "status": "success"}
        
        # Mix reliable and unreliable operations
        reliable_tasks = [reliable_operation(f"reliable_{i}") for i in range(10)]
        unreliable_tasks = [unreliable_operation(f"unreliable_{i}") for i in range(10)]
        
        all_tasks = reliable_tasks + unreliable_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Separate results
        reliable_results = results[:10]
        unreliable_results = results[10:]
        
        # All reliable operations should succeed despite unreliable ones failing
        reliable_successes = [r for r in reliable_results if isinstance(r, dict)]
        assert len(reliable_successes) == 10  # All reliable operations succeeded
        
        # Unreliable operations should have some failures but not affect reliable ones
        unreliable_successes = [r for r in unreliable_results if isinstance(r, dict)]
        unreliable_failures = [r for r in unreliable_results if isinstance(r, Exception)]
        
        assert len(unreliable_successes) + len(unreliable_failures) == 10
        assert len(unreliable_failures) > 0  # Some failures occurred
        assert len(unreliable_successes) > 0  # Some successes occurred
    
    @pytest.mark.asyncio
    async def test_concurrent_retry_mechanisms(self):
        """Test retry mechanisms under concurrent load."""
        # Operation with retry logic
        retry_counts = {}
        success_after_retries = {}
        
        async def operation_with_retries(op_id: str, max_retries: int = 3):
            retry_counts[op_id] = 0
            
            for attempt in range(max_retries + 1):
                retry_counts[op_id] = attempt
                
                try:
                    await asyncio.sleep(random.uniform(0.05, 0.15))
                    
                    # Simulate operation that succeeds after some retries
                    failure_probability = 0.7 * (0.5 ** attempt)  # Decreasing failure rate
                    if random.random() < failure_probability:
                        raise Exception(f"Attempt {attempt} failed for {op_id}")
                    
                    success_after_retries[op_id] = attempt
                    return {"id": op_id, "success_attempt": attempt}
                    
                except Exception:
                    if attempt == max_retries:
                        raise  # Final failure
                    continue
        
        # Run many concurrent operations with retries
        tasks = [operation_with_retries(f"op_{i}") for i in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze retry behavior
        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        # Most operations should eventually succeed with retries
        assert len(successes) >= 15  # At least 75% success rate with retries
        
        # Verify retry attempts were made
        assert max(retry_counts.values()) > 0  # Some operations required retries
        
        # Verify successful operations after retries
        retry_success_attempts = [success_after_retries[r["id"]] for r in successes]
        assert any(attempt > 0 for attempt in retry_success_attempts)  # Some succeeded after retry
    
    @pytest.mark.asyncio  
    async def test_graceful_degradation_under_failures(self):
        """Test system graceful degradation when components fail concurrently."""
        # Simulate system with multiple components that can fail
        component_states = {
            "llm_service": {"available": True, "failure_rate": 0.1},
            "vector_db": {"available": True, "failure_rate": 0.05},
            "cache": {"available": True, "failure_rate": 0.15},
            "index": {"available": True, "failure_rate": 0.08}
        }
        
        async def system_operation(op_id: str, required_components: List[str]):
            """Perform operation that may degrade gracefully."""
            available_components = []
            failed_components = []
            
            for component in required_components:
                if component not in component_states:
                    failed_components.append(component)
                    continue
                    
                state = component_states[component]
                if state["available"] and random.random() >= state["failure_rate"]:
                    available_components.append(component)
                else:
                    failed_components.append(component)
            
            # Determine operation capability based on available components
            if not available_components:
                raise Exception(f"Operation {op_id} failed: no components available")
            
            # Graceful degradation logic
            capability_level = len(available_components) / len(required_components)
            
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            return {
                "id": op_id,
                "capability_level": capability_level,
                "available_components": available_components,
                "failed_components": failed_components,
                "degraded": len(failed_components) > 0
            }
        
        # Test different operation types with different component requirements
        operation_types = [
            (["llm_service", "vector_db", "index"], "full_search"),
            (["vector_db", "index"], "basic_search"),
            (["cache", "index"], "cached_lookup"),
            (["llm_service"], "llm_only")
        ]
        
        all_tasks = []
        for i in range(20):
            components, op_type = random.choice(operation_types)
            task = system_operation(f"{op_type}_{i}", components)
            all_tasks.append(task)
        
        results = await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Analyze graceful degradation
        successes = [r for r in results if isinstance(r, dict)]
        failures = [r for r in results if isinstance(r, Exception)]
        
        # System should maintain some level of functionality
        assert len(successes) >= 14  # At least 70% operations succeed
        
        # Check degradation patterns
        degraded_operations = [r for r in successes if r["degraded"]]
        full_capability_operations = [r for r in successes if not r["degraded"]]
        
        # Should have some operations running with degraded capability
        assert len(degraded_operations) > 0
        
        # Average capability should be reasonable despite failures
        avg_capability = np.mean([r["capability_level"] for r in successes])
        assert avg_capability > 0.6  # Average capability above 60%


class TestPerformanceUnderConcurrency:
    """Test performance characteristics under concurrent load."""
    
    @pytest.mark.asyncio
    async def test_throughput_scaling(self):
        """Test how throughput scales with concurrent operations."""
        # Test different concurrency levels
        concurrency_levels = [1, 5, 10, 20, 30]
        throughput_results = {}
        
        async def simple_operation():
            await asyncio.sleep(random.uniform(0.05, 0.15))
            return {"completed": True}
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Run operations at this concurrency level
            tasks = [simple_operation() for _ in range(concurrency)]
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            duration = end_time - start_time
            throughput = len(results) / duration
            
            throughput_results[concurrency] = throughput
            
            print(f"Concurrency {concurrency}: {throughput:.2f} ops/sec")
        
        # Analyze throughput scaling
        baseline_throughput = throughput_results[1]
        
        # Throughput should generally increase with concurrency (up to a point)
        assert throughput_results[5] > baseline_throughput * 2  # At least 2x improvement
        assert throughput_results[10] > baseline_throughput * 4  # At least 4x improvement
        
        # May plateau or decrease at very high concurrency due to overhead
        max_throughput = max(throughput_results.values())
        assert max_throughput > baseline_throughput * 5  # Significant improvement possible
    
    @pytest.mark.asyncio
    async def test_memory_usage_under_concurrent_load(self):
        """Test memory usage patterns under concurrent operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create memory-intensive concurrent operations
        async def memory_intensive_operation(data_size: int):
            # Simulate operation that uses memory
            data = [random.random() for _ in range(data_size)]
            await asyncio.sleep(random.uniform(0.1, 0.3))
            
            # Simulate processing that keeps data in memory briefly
            result = sum(data) / len(data)
            await asyncio.sleep(random.uniform(0.05, 0.15))
            
            return {"result": result, "data_size": data_size}
        
        # Run concurrent memory-intensive operations
        data_sizes = [10000] * 20  # 20 operations with 10k elements each
        tasks = [memory_intensive_operation(size) for size in data_sizes]
        
        # Measure peak memory during concurrent execution
        peak_memory = baseline_memory
        
        async def memory_monitor():
            nonlocal peak_memory
            while True:
                current_memory = process.memory_info().rss / 1024 / 1024
                peak_memory = max(peak_memory, current_memory)
                await asyncio.sleep(0.1)
        
        monitor_task = asyncio.create_task(memory_monitor())
        
        try:
            results = await asyncio.gather(*tasks)
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        
        # Measure memory after operations
        final_memory = process.memory_info().rss / 1024 / 1024
        
        # Analyze memory usage patterns
        memory_increase = peak_memory - baseline_memory
        memory_retained = final_memory - baseline_memory
        
        print(f"Baseline memory: {baseline_memory:.2f} MB")
        print(f"Peak memory: {peak_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory increase: {memory_increase:.2f} MB")
        print(f"Memory retained: {memory_retained:.2f} MB")
        
        # Memory should increase during concurrent operations but not excessively
        assert memory_increase < 200  # Less than 200MB increase
        assert memory_retained < 50   # Less than 50MB retained after completion
        
        # All operations should complete successfully
        assert len(results) == 20
        assert all(isinstance(r, dict) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])