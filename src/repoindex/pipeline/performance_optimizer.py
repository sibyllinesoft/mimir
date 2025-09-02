"""
Performance Benchmarking and Optimization for Hybrid Query Engine.

Provides comprehensive performance measurement, analysis, and optimization
for Phase 3 hybrid search capabilities.
"""

import asyncio
import time
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple

from ..monitoring import get_metrics_collector, get_trace_manager
from ..util.log import get_logger
from .hybrid_query_engine import HybridQueryEngine, QueryContext, QueryStrategy

logger = get_logger(__name__)


class BenchmarkType(Enum):
    """Types of performance benchmarks."""
    LATENCY = "latency"           # Response time measurement
    THROUGHPUT = "throughput"     # Queries per second
    CONCURRENCY = "concurrency"   # Concurrent query handling
    ACCURACY = "accuracy"         # Result quality vs speed tradeoffs
    MEMORY = "memory"             # Memory usage profiling
    CACHE = "cache"               # Cache effectiveness


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE_CACHING = "aggressive_caching"
    QUERY_PARALLELIZATION = "query_parallelization"
    EARLY_TERMINATION = "early_termination"
    RESULT_STREAMING = "result_streaming"
    SMART_ROUTING = "smart_routing"
    BUDGET_ENFORCEMENT = "budget_enforcement"


@dataclass
class BenchmarkResult:
    """Individual benchmark measurement result."""
    benchmark_type: BenchmarkType
    strategy: QueryStrategy
    query_complexity: float
    response_time_ms: float
    memory_usage_mb: float
    cache_hit: bool
    result_count: int
    accuracy_score: float
    error: Optional[str] = None


@dataclass
class PerformanceProfile:
    """Performance profile for a specific configuration."""
    strategy: QueryStrategy
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    throughput_qps: float
    memory_usage_mb: float
    cache_hit_rate: float
    accuracy_score: float
    optimization_recommendations: List[str] = field(default_factory=list)


@dataclass
class BenchmarkSuite:
    """Complete benchmark suite configuration."""
    query_patterns: List[Dict[str, Any]]
    strategies_to_test: List[QueryStrategy]
    concurrency_levels: List[int]
    duration_seconds: int = 60
    warmup_seconds: int = 10
    target_accuracy: float = 0.9
    performance_budget_ms: int = 2000


class PerformanceOptimizer:
    """
    Performance optimization and benchmarking system.
    
    Features:
    - Comprehensive performance benchmarking
    - Strategy comparison and recommendation
    - Real-time optimization suggestions
    - Memory and cache profiling
    - Concurrency testing
    - Performance regression detection
    """
    
    def __init__(self, hybrid_engine: HybridQueryEngine):
        """Initialize performance optimizer."""
        self.hybrid_engine = hybrid_engine
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()
        
        # Performance baselines and thresholds
        self.performance_baselines = {
            QueryStrategy.VECTOR_FIRST: {"latency_ms": 800, "throughput_qps": 25},
            QueryStrategy.SEMANTIC_FIRST: {"latency_ms": 1500, "throughput_qps": 15},
            QueryStrategy.PARALLEL_HYBRID: {"latency_ms": 1200, "throughput_qps": 20},
            QueryStrategy.ADAPTIVE: {"latency_ms": 1000, "throughput_qps": 22}
        }
        
        # Optimization configurations
        self.optimization_configs = {
            OptimizationStrategy.AGGRESSIVE_CACHING: {
                "cache_size": 2000,
                "cache_ttl": 600,
                "precompute_common_queries": True
            },
            OptimizationStrategy.QUERY_PARALLELIZATION: {
                "max_parallel_searches": 3,
                "timeout_per_search": 1000
            },
            OptimizationStrategy.EARLY_TERMINATION: {
                "confidence_threshold": 0.9,
                "min_results_before_termination": 5
            }
        }
        
        # Historical performance data
        self.performance_history: List[BenchmarkResult] = []
        
    async def run_comprehensive_benchmark(
        self, 
        benchmark_suite: BenchmarkSuite,
        search_data: Dict[str, Any]
    ) -> Dict[QueryStrategy, PerformanceProfile]:
        """
        Run comprehensive performance benchmark suite.
        
        Args:
            benchmark_suite: Configuration for benchmark tests
            search_data: Mock or real search data for testing
            
        Returns:
            Performance profiles for each strategy
        """
        logger.info(f"Starting comprehensive benchmark with {len(benchmark_suite.strategies_to_test)} strategies")
        
        results = {}
        
        for strategy in benchmark_suite.strategies_to_test:
            logger.info(f"Benchmarking strategy: {strategy.value}")
            
            # Run individual benchmarks for this strategy
            strategy_results = []
            
            # Latency benchmark
            latency_results = await self._benchmark_latency(
                strategy, benchmark_suite, search_data
            )
            strategy_results.extend(latency_results)
            
            # Throughput benchmark
            throughput_results = await self._benchmark_throughput(
                strategy, benchmark_suite, search_data
            )
            strategy_results.extend(throughput_results)
            
            # Concurrency benchmark
            concurrency_results = await self._benchmark_concurrency(
                strategy, benchmark_suite, search_data
            )
            strategy_results.extend(concurrency_results)
            
            # Memory benchmark
            memory_results = await self._benchmark_memory_usage(
                strategy, benchmark_suite, search_data
            )
            strategy_results.extend(memory_results)
            
            # Create performance profile
            profile = self._create_performance_profile(strategy, strategy_results)
            results[strategy] = profile
            
            logger.info(f"Strategy {strategy.value} - Avg: {profile.avg_response_time_ms:.1f}ms, "
                       f"P95: {profile.p95_response_time_ms:.1f}ms, QPS: {profile.throughput_qps:.1f}")
        
        # Generate optimization recommendations
        self._generate_optimization_recommendations(results, benchmark_suite)
        
        return results
    
    async def _benchmark_latency(
        self, 
        strategy: QueryStrategy, 
        benchmark_suite: BenchmarkSuite,
        search_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark response time latency."""
        results = []
        
        # Warmup
        await self._warmup_strategy(strategy, search_data)
        
        for query_pattern in benchmark_suite.query_patterns:
            for _ in range(10):  # 10 measurements per pattern
                start_time = time.time()
                memory_before = self._get_memory_usage()
                
                try:
                    context = QueryContext(
                        strategy=strategy,
                        max_results=query_pattern.get("max_results", 20),
                        performance_budget_ms=benchmark_suite.performance_budget_ms
                    )
                    
                    # Execute search
                    response = await self.hybrid_engine.search(
                        query=query_pattern["query"],
                        index_id="benchmark_index",
                        context=context,
                        **search_data
                    )
                    
                    execution_time = (time.time() - start_time) * 1000
                    memory_after = self._get_memory_usage()
                    
                    # Check cache hit (simplified)
                    cache_hit = execution_time < 100  # Very fast suggests cache hit
                    
                    # Calculate accuracy score (simplified)
                    accuracy_score = min(len(response.results) / context.max_results, 1.0)
                    
                    result = BenchmarkResult(
                        benchmark_type=BenchmarkType.LATENCY,
                        strategy=strategy,
                        query_complexity=query_pattern.get("complexity", 0.5),
                        response_time_ms=execution_time,
                        memory_usage_mb=memory_after - memory_before,
                        cache_hit=cache_hit,
                        result_count=len(response.results),
                        accuracy_score=accuracy_score
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.warning(f"Benchmark failed for {strategy.value}: {e}")
                    results.append(BenchmarkResult(
                        benchmark_type=BenchmarkType.LATENCY,
                        strategy=strategy,
                        query_complexity=query_pattern.get("complexity", 0.5),
                        response_time_ms=benchmark_suite.performance_budget_ms,  # Max time
                        memory_usage_mb=0,
                        cache_hit=False,
                        result_count=0,
                        accuracy_score=0.0,
                        error=str(e)
                    ))
                
                # Small delay between measurements
                await asyncio.sleep(0.1)
        
        return results
    
    async def _benchmark_throughput(
        self,
        strategy: QueryStrategy,
        benchmark_suite: BenchmarkSuite,
        search_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark throughput (queries per second)."""
        results = []
        
        # Run for specified duration
        start_time = time.time()
        end_time = start_time + benchmark_suite.duration_seconds
        completed_queries = 0
        
        while time.time() < end_time:
            query_pattern = benchmark_suite.query_patterns[
                completed_queries % len(benchmark_suite.query_patterns)
            ]
            
            query_start = time.time()
            
            try:
                context = QueryContext(
                    strategy=strategy,
                    max_results=query_pattern.get("max_results", 20),
                    performance_budget_ms=1000  # Shorter budget for throughput test
                )
                
                await self.hybrid_engine.search(
                    query=query_pattern["query"],
                    index_id="benchmark_index",
                    context=context,
                    **search_data
                )
                
                completed_queries += 1
                
                # Record individual query time for throughput calculation
                query_time = (time.time() - query_start) * 1000
                
                result = BenchmarkResult(
                    benchmark_type=BenchmarkType.THROUGHPUT,
                    strategy=strategy,
                    query_complexity=query_pattern.get("complexity", 0.5),
                    response_time_ms=query_time,
                    memory_usage_mb=0,  # Not measured for throughput
                    cache_hit=False,
                    result_count=0,  # Not measured for throughput
                    accuracy_score=0.0  # Not measured for throughput
                )
                
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Throughput test query failed: {e}")
                completed_queries += 1  # Count failed queries too
        
        actual_duration = time.time() - start_time
        throughput_qps = completed_queries / actual_duration
        
        logger.info(f"Throughput benchmark for {strategy.value}: "
                   f"{throughput_qps:.2f} QPS ({completed_queries} queries in {actual_duration:.1f}s)")
        
        return results
    
    async def _benchmark_concurrency(
        self,
        strategy: QueryStrategy,
        benchmark_suite: BenchmarkSuite,
        search_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark concurrent query handling."""
        results = []
        
        for concurrency_level in benchmark_suite.concurrency_levels:
            logger.info(f"Testing concurrency level: {concurrency_level}")
            
            # Create concurrent query tasks
            tasks = []
            for i in range(concurrency_level):
                query_pattern = benchmark_suite.query_patterns[
                    i % len(benchmark_suite.query_patterns)
                ]
                
                context = QueryContext(
                    strategy=strategy,
                    max_results=query_pattern.get("max_results", 20),
                    performance_budget_ms=benchmark_suite.performance_budget_ms
                )
                
                task = self._execute_concurrent_query(
                    query_pattern["query"],
                    context,
                    search_data
                )
                tasks.append(task)
            
            # Execute all queries concurrently
            start_time = time.time()
            concurrent_results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time = (time.time() - start_time) * 1000
            
            # Process results
            successful_queries = 0
            total_response_time = 0
            
            for result in concurrent_results:
                if isinstance(result, Exception):
                    logger.warning(f"Concurrent query failed: {result}")
                else:
                    successful_queries += 1
                    total_response_time += result["response_time_ms"]
            
            avg_response_time = total_response_time / max(successful_queries, 1)
            success_rate = successful_queries / concurrency_level
            
            benchmark_result = BenchmarkResult(
                benchmark_type=BenchmarkType.CONCURRENCY,
                strategy=strategy,
                query_complexity=0.5,  # Average complexity
                response_time_ms=avg_response_time,
                memory_usage_mb=0,
                cache_hit=False,
                result_count=successful_queries,
                accuracy_score=success_rate
            )
            
            results.append(benchmark_result)
            
            logger.info(f"Concurrency {concurrency_level}: {success_rate:.1%} success rate, "
                       f"{avg_response_time:.1f}ms avg response time")
        
        return results
    
    async def _execute_concurrent_query(
        self,
        query: str,
        context: QueryContext,
        search_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single query for concurrency testing."""
        start_time = time.time()
        
        response = await self.hybrid_engine.search(
            query=query,
            index_id="benchmark_index",
            context=context,
            **search_data
        )
        
        return {
            "response_time_ms": (time.time() - start_time) * 1000,
            "result_count": len(response.results)
        }
    
    async def _benchmark_memory_usage(
        self,
        strategy: QueryStrategy,
        benchmark_suite: BenchmarkSuite,
        search_data: Dict[str, Any]
    ) -> List[BenchmarkResult]:
        """Benchmark memory usage patterns."""
        results = []
        
        # Test with increasing data sizes
        data_size_multipliers = [0.5, 1.0, 2.0, 5.0]  # Simulate different data sizes
        
        for multiplier in data_size_multipliers:
            # Simulate larger dataset (simplified)
            scaled_search_data = self._scale_search_data(search_data, multiplier)
            
            memory_before = self._get_memory_usage()
            
            # Execute multiple queries to see memory pattern
            for query_pattern in benchmark_suite.query_patterns[:3]:  # First 3 patterns
                context = QueryContext(
                    strategy=strategy,
                    max_results=int(query_pattern.get("max_results", 20) * multiplier)
                )
                
                try:
                    await self.hybrid_engine.search(
                        query=query_pattern["query"],
                        index_id="benchmark_index",
                        context=context,
                        **scaled_search_data
                    )
                except Exception as e:
                    logger.warning(f"Memory benchmark query failed: {e}")
            
            memory_after = self._get_memory_usage()
            memory_delta = memory_after - memory_before
            
            result = BenchmarkResult(
                benchmark_type=BenchmarkType.MEMORY,
                strategy=strategy,
                query_complexity=multiplier,  # Use multiplier as complexity indicator
                response_time_ms=0,  # Not measured for memory test
                memory_usage_mb=memory_delta,
                cache_hit=False,
                result_count=0,
                accuracy_score=0.0
            )
            
            results.append(result)
            
            logger.info(f"Memory usage for {strategy.value} at {multiplier}x scale: {memory_delta:.1f}MB")
        
        return results
    
    def _scale_search_data(self, search_data: Dict[str, Any], multiplier: float) -> Dict[str, Any]:
        """Scale search data for memory testing (simplified simulation)."""
        # In practice, this would create larger mock datasets
        # For now, just return the original data
        return search_data
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified)."""
        import psutil
        import os
        
        try:
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            return memory_mb
        except Exception:
            return 0.0  # Fallback if psutil not available
    
    async def _warmup_strategy(self, strategy: QueryStrategy, search_data: Dict[str, Any]):
        """Warm up strategy before benchmarking."""
        warmup_queries = [
            "warmup query 1",
            "test search",
            "sample function"
        ]
        
        for query in warmup_queries:
            context = QueryContext(strategy=strategy, max_results=5)
            try:
                await self.hybrid_engine.search(
                    query=query,
                    index_id="warmup_index",
                    context=context,
                    **search_data
                )
            except Exception:
                pass  # Ignore warmup failures
    
    def _create_performance_profile(
        self, 
        strategy: QueryStrategy, 
        results: List[BenchmarkResult]
    ) -> PerformanceProfile:
        """Create performance profile from benchmark results."""
        latency_results = [r for r in results if r.benchmark_type == BenchmarkType.LATENCY and not r.error]
        throughput_results = [r for r in results if r.benchmark_type == BenchmarkType.THROUGHPUT]
        memory_results = [r for r in results if r.benchmark_type == BenchmarkType.MEMORY]
        
        if not latency_results:
            # Fallback for failed benchmarks
            return PerformanceProfile(
                strategy=strategy,
                avg_response_time_ms=5000,
                p95_response_time_ms=10000,
                p99_response_time_ms=15000,
                throughput_qps=1.0,
                memory_usage_mb=100.0,
                cache_hit_rate=0.0,
                accuracy_score=0.0,
                optimization_recommendations=["Strategy failed benchmarking - investigate issues"]
            )
        
        # Calculate statistics
        response_times = [r.response_time_ms for r in latency_results]
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        p99_response_time = statistics.quantiles(response_times, n=100)[98]  # 99th percentile
        
        # Calculate throughput
        if throughput_results:
            total_time_seconds = sum(r.response_time_ms for r in throughput_results) / 1000
            throughput_qps = len(throughput_results) / max(total_time_seconds, 1)
        else:
            throughput_qps = 0.0
        
        # Calculate memory usage
        if memory_results:
            avg_memory_mb = statistics.mean([r.memory_usage_mb for r in memory_results])
        else:
            avg_memory_mb = 0.0
        
        # Calculate cache hit rate
        cache_hits = sum(1 for r in latency_results if r.cache_hit)
        cache_hit_rate = cache_hits / len(latency_results) if latency_results else 0.0
        
        # Calculate average accuracy
        accuracy_scores = [r.accuracy_score for r in latency_results if r.accuracy_score > 0]
        avg_accuracy = statistics.mean(accuracy_scores) if accuracy_scores else 0.0
        
        # Generate optimization recommendations
        recommendations = self._generate_strategy_recommendations(
            strategy, avg_response_time, throughput_qps, cache_hit_rate, avg_accuracy
        )
        
        return PerformanceProfile(
            strategy=strategy,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            throughput_qps=throughput_qps,
            memory_usage_mb=avg_memory_mb,
            cache_hit_rate=cache_hit_rate,
            accuracy_score=avg_accuracy,
            optimization_recommendations=recommendations
        )
    
    def _generate_strategy_recommendations(
        self,
        strategy: QueryStrategy,
        avg_response_time: float,
        throughput_qps: float,
        cache_hit_rate: float,
        accuracy: float
    ) -> List[str]:
        """Generate optimization recommendations for a strategy."""
        recommendations = []
        
        # Check against baselines
        baseline = self.performance_baselines.get(strategy, {})
        target_latency = baseline.get("latency_ms", 2000)
        target_throughput = baseline.get("throughput_qps", 10)
        
        if avg_response_time > target_latency * 1.5:
            recommendations.append(f"Response time {avg_response_time:.0f}ms exceeds target {target_latency}ms - consider aggressive caching")
        
        if throughput_qps < target_throughput * 0.7:
            recommendations.append(f"Throughput {throughput_qps:.1f} QPS below target {target_throughput} - enable query parallelization")
        
        if cache_hit_rate < 0.3:
            recommendations.append(f"Low cache hit rate {cache_hit_rate:.1%} - optimize cache strategy")
        
        if accuracy < 0.8:
            recommendations.append(f"Accuracy {accuracy:.1%} below 80% - review result ranking algorithms")
        
        # Strategy-specific recommendations
        if strategy == QueryStrategy.VECTOR_FIRST:
            if avg_response_time > 1000:
                recommendations.append("Vector search taking too long - consider early termination")
        elif strategy == QueryStrategy.SEMANTIC_FIRST:
            if avg_response_time > 2000:
                recommendations.append("Semantic analysis too slow - optimize symbol graph operations")
        elif strategy == QueryStrategy.PARALLEL_HYBRID:
            if throughput_qps < 15:
                recommendations.append("Parallel strategy underperforming - check system resource limits")
        
        return recommendations
    
    def _generate_optimization_recommendations(
        self,
        profiles: Dict[QueryStrategy, PerformanceProfile],
        benchmark_suite: BenchmarkSuite
    ) -> None:
        """Generate overall optimization recommendations."""
        logger.info("=== Performance Optimization Recommendations ===")
        
        # Find best and worst performing strategies
        strategies_by_latency = sorted(
            profiles.items(), 
            key=lambda x: x[1].avg_response_time_ms
        )
        strategies_by_throughput = sorted(
            profiles.items(), 
            key=lambda x: x[1].throughput_qps, 
            reverse=True
        )
        
        best_latency = strategies_by_latency[0]
        best_throughput = strategies_by_throughput[0]
        
        logger.info(f"Best latency: {best_latency[0].value} ({best_latency[1].avg_response_time_ms:.1f}ms)")
        logger.info(f"Best throughput: {best_throughput[0].value} ({best_throughput[1].throughput_qps:.1f} QPS)")
        
        # Overall recommendations
        overall_recommendations = []
        
        # Check if any strategy meets all targets
        meets_targets = []
        for strategy, profile in profiles.items():
            baseline = self.performance_baselines.get(strategy, {})
            if (profile.avg_response_time_ms <= baseline.get("latency_ms", 2000) and
                profile.throughput_qps >= baseline.get("throughput_qps", 10)):
                meets_targets.append(strategy)
        
        if not meets_targets:
            overall_recommendations.append("No strategy meets all performance targets - system optimization needed")
        
        # Cache optimization
        avg_cache_hit_rate = statistics.mean([p.cache_hit_rate for p in profiles.values()])
        if avg_cache_hit_rate < 0.4:
            overall_recommendations.append(f"System-wide cache hit rate {avg_cache_hit_rate:.1%} is low - implement aggressive caching")
        
        # Memory usage
        max_memory = max([p.memory_usage_mb for p in profiles.values()])
        if max_memory > 500:  # 500MB threshold
            overall_recommendations.append(f"High memory usage detected ({max_memory:.1f}MB) - optimize data structures")
        
        for recommendation in overall_recommendations:
            logger.info(f"RECOMMENDATION: {recommendation}")
    
    def get_optimization_config(self, strategy: OptimizationStrategy) -> Dict[str, Any]:
        """Get configuration for specific optimization strategy."""
        return self.optimization_configs.get(strategy, {})
    
    async def profile_query_types(
        self, 
        search_data: Dict[str, Any]
    ) -> Dict[str, Dict[str, float]]:
        """Profile performance across different query types."""
        query_type_patterns = {
            "simple": [
                {"query": "function", "complexity": 0.1},
                {"query": "class", "complexity": 0.1},
                {"query": "method", "complexity": 0.1}
            ],
            "moderate": [
                {"query": "find function calculateSum", "complexity": 0.5},
                {"query": "class UserManager methods", "complexity": 0.5},
                {"query": "authentication implementation", "complexity": 0.5}
            ],
            "complex": [
                {"query": "explain how user authentication system handles OAuth2 token validation", "complexity": 0.9},
                {"query": "find all error handling patterns in database connection modules", "complexity": 0.8},
                {"query": "similar code patterns to async request processing with retry logic", "complexity": 0.9}
            ]
        }
        
        results = {}
        
        for query_type, patterns in query_type_patterns.items():
            type_results = {}
            
            for strategy in [QueryStrategy.VECTOR_FIRST, QueryStrategy.SEMANTIC_FIRST, QueryStrategy.PARALLEL_HYBRID]:
                response_times = []
                
                for pattern in patterns:
                    context = QueryContext(
                        strategy=strategy,
                        max_results=20
                    )
                    
                    start_time = time.time()
                    
                    try:
                        await self.hybrid_engine.search(
                            query=pattern["query"],
                            index_id="profile_index",
                            context=context,
                            **search_data
                        )
                        
                        response_time = (time.time() - start_time) * 1000
                        response_times.append(response_time)
                        
                    except Exception as e:
                        logger.warning(f"Query profiling failed: {e}")
                        response_times.append(5000)  # Penalty time
                
                type_results[strategy.value] = statistics.mean(response_times) if response_times else 5000
            
            results[query_type] = type_results
            
            logger.info(f"Query type '{query_type}' average times:")
            for strategy_name, avg_time in type_results.items():
                logger.info(f"  {strategy_name}: {avg_time:.1f}ms")
        
        return results


# Factory function for easy instantiation
def create_performance_optimizer(hybrid_engine: HybridQueryEngine) -> PerformanceOptimizer:
    """Create and return a PerformanceOptimizer instance."""
    return PerformanceOptimizer(hybrid_engine)


# Predefined benchmark suites for different use cases
def create_standard_benchmark_suite() -> BenchmarkSuite:
    """Create standard benchmark suite for general testing."""
    return BenchmarkSuite(
        query_patterns=[
            {"query": "find function", "complexity": 0.2, "max_results": 10},
            {"query": "class definition User", "complexity": 0.3, "max_results": 15},
            {"query": "authentication implementation", "complexity": 0.6, "max_results": 20},
            {"query": "how does error handling work", "complexity": 0.8, "max_results": 25},
            {"query": "similar patterns to async processing", "complexity": 0.9, "max_results": 30}
        ],
        strategies_to_test=[
            QueryStrategy.VECTOR_FIRST,
            QueryStrategy.SEMANTIC_FIRST,
            QueryStrategy.PARALLEL_HYBRID,
            QueryStrategy.ADAPTIVE
        ],
        concurrency_levels=[1, 5, 10, 20],
        duration_seconds=30,
        warmup_seconds=5,
        target_accuracy=0.85,
        performance_budget_ms=2000
    )


def create_stress_benchmark_suite() -> BenchmarkSuite:
    """Create stress test benchmark suite for load testing."""
    return BenchmarkSuite(
        query_patterns=[
            {"query": f"complex query {i} with multiple entities and patterns", "complexity": 0.9, "max_results": 50}
            for i in range(20)  # 20 different complex queries
        ],
        strategies_to_test=[QueryStrategy.PARALLEL_HYBRID, QueryStrategy.ADAPTIVE],
        concurrency_levels=[10, 25, 50, 100],
        duration_seconds=120,  # 2 minutes
        warmup_seconds=10,
        target_accuracy=0.8,
        performance_budget_ms=5000
    )