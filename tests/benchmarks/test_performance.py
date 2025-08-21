"""
Comprehensive performance benchmarks for Mimir Deep Code Research System.

Tests critical performance paths with realistic data volumes:
- Vector search with 10,000+ chunks
- Symbol search with 5,000+ symbols
- Full pipeline execution workflow
- Memory usage patterns and optimization
- Concurrent operation performance
"""

import asyncio
import statistics
import time
import tracemalloc
from unittest.mock import Mock

import psutil
import pytest

from repoindex.data.schemas import (
    VectorChunk,
    FeatureConfig,
    RepoMap,
    SearchResponse,
    SerenaGraph,
    SymbolEntry,
    SymbolType,
    VectorIndex,
)
from repoindex.pipeline.ask_index import SymbolGraphNavigator

# Import the components to benchmark
from repoindex.pipeline.hybrid_search import HybridSearchEngine
from repoindex.pipeline.run import IndexingPipeline


class PerformanceBenchmark:
    """Base class for performance benchmarks with metrics collection."""

    def __init__(self):
        self.metrics = {}

    def start_memory_tracking(self):
        """Start memory tracking."""
        tracemalloc.start()

    def stop_memory_tracking(self) -> tuple[float, float]:
        """Stop memory tracking and return current, peak memory in MB."""
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return current / 1024 / 1024, peak / 1024 / 1024

    def measure_time(self, func_name: str):
        """Context manager for timing operations."""
        return TimingContext(self.metrics, func_name)

    def get_system_metrics(self) -> dict[str, float]:
        """Get current system resource usage."""
        process = psutil.Process()
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
        }


class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, metrics: dict, operation: str):
        self.metrics = metrics
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        if self.operation not in self.metrics:
            self.metrics[self.operation] = []
        self.metrics[self.operation].append(duration * 1000)  # Convert to ms


class MockDataGenerator:
    """Generate realistic mock data for benchmarking."""

    @staticmethod
    def create_vector_index(chunk_count: int = 10000) -> VectorIndex:
        """Create a mock vector index with specified number of chunks."""
        chunks = []

        for i in range(chunk_count):
            chunk = CodeChunk(
                path=f"src/file_{i % 100}.ts",
                span=(i * 10, (i + 1) * 10),
                hash=f"hash_{i}",
                embedding=[0.1 * j for j in range(384)],  # 384-dim embedding
            )
            chunks.append(chunk)

        return VectorIndex(chunks=chunks, dimension=384, total_tokens=chunk_count * 50)

    @staticmethod
    def create_serena_graph(symbol_count: int = 5000) -> SerenaGraph:
        """Create a mock symbol graph with specified number of symbols."""
        entries = []

        # Create definitions
        for i in range(symbol_count // 3):
            entry = SymbolEntry(
                path=f"src/module_{i % 50}.ts",
                span=(i * 20, (i + 1) * 20),
                symbol=f"function_{i}",
                type=SymbolType.DEF,
                sig=f"function function_{i}(param: string): boolean",
                caller=None,
                callee=None,
            )
            entries.append(entry)

        # Create references
        for i in range(symbol_count // 3, 2 * symbol_count // 3):
            entry = SymbolEntry(
                path=f"src/module_{i % 50}.ts",
                span=(i * 20, (i + 1) * 20),
                symbol=f"function_{i % (symbol_count // 3)}",
                type=SymbolType.REF,
                sig=None,
                caller=None,
                callee=None,
            )
            entries.append(entry)

        # Create function calls
        for i in range(2 * symbol_count // 3, symbol_count):
            entry = SymbolEntry(
                path=f"src/module_{i % 50}.ts",
                span=(i * 20, (i + 1) * 20),
                symbol=None,
                type=SymbolType.CALL,
                sig=None,
                caller=f"function_{i % 100}",
                callee=f"function_{(i + 1) % 100}",
            )
            entries.append(entry)

        return SerenaGraph(entries=entries)

    @staticmethod
    def create_repo_map(file_count: int = 1000) -> RepoMap:
        """Create a mock repository map."""
        from repoindex.data.schemas import FileRank, RepoMapEdge

        edges = []
        file_ranks = []

        # Create file dependencies
        for i in range(file_count):
            file_path = f"src/file_{i}.ts"

            # Create edges to other files
            for j in range(min(5, file_count - i - 1)):
                target_path = f"src/file_{i + j + 1}.ts"
                edge = RepoMapEdge(
                    source=file_path,
                    target=target_path,
                    weight=0.5 + (j * 0.1),
                    relation_type="import",
                )
                edges.append(edge)

            # Create file rank
            rank = FileRank(
                path=file_path,
                rank=1.0 - (i / file_count),  # Decreasing rank
                importance_score=0.8 + (i % 5) * 0.04,
            )
            file_ranks.append(rank)

        return RepoMap(edges=edges, file_ranks=file_ranks)


@pytest.mark.benchmark
class TestVectorSearchPerformance(PerformanceBenchmark):
    """Benchmark vector similarity search performance."""

    @pytest.fixture
    def hybrid_search_engine(self):
        """Create hybrid search engine instance."""
        return HybridSearchEngine()

    @pytest.fixture
    def large_vector_index(self):
        """Create large vector index for benchmarking."""
        return MockDataGenerator.create_vector_index(10000)

    async def test_vector_search_performance(self, hybrid_search_engine, large_vector_index):
        """Benchmark vector search with 10,000 chunks."""
        queries = [
            "async function authentication",
            "database connection pool",
            "error handling middleware",
            "React component state",
            "TypeScript interface definition",
        ]

        self.start_memory_tracking()

        for query in queries:
            with self.measure_time("vector_search"):
                results = await hybrid_search_engine._vector_search(query, large_vector_index)
                assert len(results) <= hybrid_search_engine.max_results_per_feature

        current_mem, peak_mem = self.stop_memory_tracking()

        # Performance assertions
        avg_time = statistics.mean(self.metrics["vector_search"])
        assert avg_time < 500, f"Vector search too slow: {avg_time:.2f}ms > 500ms"
        assert peak_mem < 200, f"Memory usage too high: {peak_mem:.2f}MB > 200MB"

        print("Vector Search Performance:")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")
        print(
            f"  Throughput: {len(queries) / (sum(self.metrics['vector_search']) / 1000):.2f} queries/sec"
        )

    async def test_vector_search_concurrent(self, hybrid_search_engine, large_vector_index):
        """Benchmark concurrent vector searches."""
        queries = ["async function"] * 10  # 10 concurrent searches

        self.start_memory_tracking()

        with self.measure_time("concurrent_vector_search"):
            tasks = [
                hybrid_search_engine._vector_search(query, large_vector_index) for query in queries
            ]
            results = await asyncio.gather(*tasks)

        current_mem, peak_mem = self.stop_memory_tracking()

        # Verify all searches completed
        assert len(results) == len(queries)

        concurrent_time = self.metrics["concurrent_vector_search"][0]
        print("Concurrent Vector Search:")
        print(f"  10 concurrent searches: {concurrent_time:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")


@pytest.mark.benchmark
class TestSymbolSearchPerformance(PerformanceBenchmark):
    """Benchmark symbol search and graph navigation performance."""

    @pytest.fixture
    def symbol_navigator(self):
        """Create symbol graph navigator instance."""
        return SymbolGraphNavigator()

    @pytest.fixture
    def large_serena_graph(self):
        """Create large symbol graph for benchmarking."""
        return MockDataGenerator.create_serena_graph(5000)

    async def test_symbol_search_performance(self, symbol_navigator, large_serena_graph):
        """Benchmark symbol search with 5,000 symbols."""
        queries = [
            "function_0",
            "function_100",
            "function_500",
            "function_1000",
            "nonexistent_function",
        ]

        self.start_memory_tracking()

        for query in queries:
            with self.measure_time("symbol_search"):
                # Test find_symbol_definition
                definition = await symbol_navigator.find_symbol_definition(
                    query, large_serena_graph
                )

            with self.measure_time("symbol_references"):
                # Test find_symbol_references
                references = await symbol_navigator.find_symbol_references(
                    query, large_serena_graph
                )

        current_mem, peak_mem = self.stop_memory_tracking()

        # Performance assertions
        avg_search_time = statistics.mean(self.metrics["symbol_search"])
        avg_ref_time = statistics.mean(self.metrics["symbol_references"])

        assert avg_search_time < 100, f"Symbol search too slow: {avg_search_time:.2f}ms > 100ms"
        assert avg_ref_time < 200, f"Reference search too slow: {avg_ref_time:.2f}ms > 200ms"

        print("Symbol Search Performance:")
        print(f"  Average definition search: {avg_search_time:.2f}ms")
        print(f"  Average reference search: {avg_ref_time:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")

    async def test_symbol_graph_navigation(self, symbol_navigator, large_serena_graph):
        """Benchmark multi-hop symbol graph navigation."""
        test_symbols = [
            SymbolEntry(
                path="src/test.ts",
                span=(0, 10),
                symbol="function_0",
                type=SymbolType.DEF,
                sig="function function_0(): void",
            )
        ]

        intents = [Mock(intent_type="flow", targets=["function_0"])]

        self.start_memory_tracking()

        with self.measure_time("graph_navigation"):
            evidence = await symbol_navigator._walk_symbol_graph(
                test_symbols, large_serena_graph, intents
            )

        current_mem, peak_mem = self.stop_memory_tracking()

        nav_time = self.metrics["graph_navigation"][0]
        assert nav_time < 1000, f"Graph navigation too slow: {nav_time:.2f}ms > 1000ms"

        print("Symbol Graph Navigation:")
        print(f"  Navigation time: {nav_time:.2f}ms")
        print(f"  Evidence collected: {len(evidence)} symbols")
        print(f"  Peak memory: {peak_mem:.2f}MB")


@pytest.mark.benchmark
class TestHybridSearchPerformance(PerformanceBenchmark):
    """Benchmark full hybrid search combining all modalities."""

    @pytest.fixture
    def hybrid_engine(self):
        """Create hybrid search engine."""
        return HybridSearchEngine()

    @pytest.fixture
    def large_datasets(self):
        """Create large datasets for hybrid search."""
        return {
            "vector_index": MockDataGenerator.create_vector_index(10000),
            "serena_graph": MockDataGenerator.create_serena_graph(5000),
            "repo_map": MockDataGenerator.create_repo_map(1000),
        }

    async def test_hybrid_search_full_performance(self, hybrid_engine, large_datasets):
        """Benchmark full hybrid search with all features enabled."""
        queries = [
            "authentication middleware function",
            "database connection error handling",
            "React component lifecycle methods",
            "TypeScript type definitions",
            "async promise handling",
        ]

        features = FeatureConfig(vector=True, symbol=True, graph=True)

        self.start_memory_tracking()

        for query in queries:
            with self.measure_time("hybrid_search_full"):
                response = await hybrid_engine.search(
                    query=query,
                    vector_index=large_datasets["vector_index"],
                    serena_graph=large_datasets["serena_graph"],
                    repomap=large_datasets["repo_map"],
                    repo_root="/test/repo",
                    rev="main",
                    features=features,
                    k=20,
                )
                assert isinstance(response, SearchResponse)
                assert response.execution_time_ms > 0

        current_mem, peak_mem = self.stop_memory_tracking()

        # Performance assertions
        avg_time = statistics.mean(self.metrics["hybrid_search_full"])
        assert avg_time < 2000, f"Hybrid search too slow: {avg_time:.2f}ms > 2000ms"

        print("Hybrid Search Performance (Full):")
        print(f"  Average time: {avg_time:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")
        print(
            f"  Throughput: {len(queries) / (sum(self.metrics['hybrid_search_full']) / 1000):.2f} queries/sec"
        )

    async def test_hybrid_search_scaling(self, hybrid_engine, large_datasets):
        """Test how hybrid search performance scales with result count."""
        query = "function authentication"
        features = FeatureConfig(vector=True, symbol=True, graph=True)
        k_values = [5, 10, 20, 50, 100]

        self.start_memory_tracking()

        for k in k_values:
            with self.measure_time(f"hybrid_search_k_{k}"):
                response = await hybrid_engine.search(
                    query=query,
                    vector_index=large_datasets["vector_index"],
                    serena_graph=large_datasets["serena_graph"],
                    repomap=large_datasets["repo_map"],
                    repo_root="/test/repo",
                    rev="main",
                    features=features,
                    k=k,
                )
                assert len(response.results) <= k

        current_mem, peak_mem = self.stop_memory_tracking()

        print("Hybrid Search Scaling:")
        for k in k_values:
            time_ms = self.metrics[f"hybrid_search_k_{k}"][0]
            print(f"  k={k}: {time_ms:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")


@pytest.mark.benchmark
class TestPipelinePerformance(PerformanceBenchmark):
    """Benchmark full pipeline execution performance."""

    @pytest.fixture
    def mock_pipeline(self, tmp_path):
        """Create mocked pipeline for performance testing."""
        pipeline = IndexingPipeline(storage_dir=tmp_path)

        # Mock external tool adapters to avoid actual tool execution
        pipeline._mock_adapters()

        return pipeline

    async def test_pipeline_stage_performance(self, mock_pipeline, tmp_path):
        """Benchmark individual pipeline stages."""
        from repoindex.data.schemas import IndexConfig, RepoInfo
        from repoindex.pipeline.run import PipelineContext

        # Create mock context
        context = PipelineContext(
            index_id="test_index",
            repo_info=RepoInfo(root=str(tmp_path), rev="main", worktree_dirty=False),
            config=IndexConfig(),
            work_dir=tmp_path / "work",
            logger=Mock(),
        )
        context.tracked_files = [f"file_{i}.ts" for i in range(100)]

        self.start_memory_tracking()

        # Benchmark each stage
        stages = [
            ("acquire", mock_pipeline._stage_acquire),
            ("repomapper", mock_pipeline._stage_repomapper),
            ("serena", mock_pipeline._stage_serena),
            ("leann", mock_pipeline._stage_leann),
            ("snippets", mock_pipeline._stage_snippets),
            ("bundle", mock_pipeline._stage_bundle),
        ]

        for stage_name, stage_func in stages:
            with self.measure_time(f"stage_{stage_name}"):
                try:
                    await stage_func(context)
                except Exception:
                    # Expected for mocked stages
                    pass

        current_mem, peak_mem = self.stop_memory_tracking()

        print("Pipeline Stage Performance:")
        for stage_name, _ in stages:
            if f"stage_{stage_name}" in self.metrics:
                time_ms = self.metrics[f"stage_{stage_name}"][0]
                print(f"  {stage_name}: {time_ms:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")

    async def test_concurrent_pipeline_limits(self, mock_pipeline):
        """Test pipeline concurrency limits and resource usage."""
        # Test CPU and IO semaphore behavior
        assert mock_pipeline.cpu_semaphore._value == 2
        assert mock_pipeline.io_semaphore._value == 8

        # Simulate concurrent stage execution
        async def mock_cpu_task():
            async with mock_pipeline.cpu_semaphore:
                await asyncio.sleep(0.1)

        async def mock_io_task():
            async with mock_pipeline.io_semaphore:
                await asyncio.sleep(0.05)

        self.start_memory_tracking()

        with self.measure_time("concurrent_cpu_tasks"):
            # Submit more tasks than CPU limit
            cpu_tasks = [mock_cpu_task() for _ in range(5)]
            await asyncio.gather(*cpu_tasks)

        with self.measure_time("concurrent_io_tasks"):
            # Submit more tasks than IO limit
            io_tasks = [mock_io_task() for _ in range(15)]
            await asyncio.gather(*io_tasks)

        current_mem, peak_mem = self.stop_memory_tracking()

        cpu_time = self.metrics["concurrent_cpu_tasks"][0]
        io_time = self.metrics["concurrent_io_tasks"][0]

        print("Concurrent Pipeline Limits:")
        print(f"  CPU tasks (5 concurrent, 2 limit): {cpu_time:.2f}ms")
        print(f"  IO tasks (15 concurrent, 8 limit): {io_time:.2f}ms")
        print(f"  Peak memory: {peak_mem:.2f}MB")


@pytest.mark.benchmark
class TestMemoryOptimization(PerformanceBenchmark):
    """Benchmark memory usage patterns and optimization opportunities."""

    async def test_large_dataset_memory_usage(self):
        """Test memory usage with large datasets."""
        self.start_memory_tracking()

        # Create progressively larger datasets
        sizes = [1000, 5000, 10000, 20000]
        peak_memories = []

        for size in sizes:
            # Measure memory for vector index
            vector_index = MockDataGenerator.create_vector_index(size)
            current_mem, peak_mem = self.stop_memory_tracking()
            peak_memories.append(peak_mem)

            # Restart tracking for next iteration
            self.start_memory_tracking()

            # Clean up
            del vector_index

        print("Memory Usage Scaling:")
        for size, peak_mem in zip(sizes, peak_memories, strict=False):
            print(f"  {size} chunks: {peak_mem:.2f}MB")

        # Memory should scale reasonably (not exponentially)
        memory_per_chunk = [peak_mem / size for size, peak_mem in zip(sizes, peak_memories, strict=False)]
        assert max(memory_per_chunk) / min(memory_per_chunk) < 2.0, "Memory scaling is non-linear"

    async def test_search_memory_efficiency(self):
        """Test memory efficiency during search operations."""
        vector_index = MockDataGenerator.create_vector_index(10000)
        serena_graph = MockDataGenerator.create_serena_graph(5000)
        hybrid_engine = HybridSearchEngine()

        queries = ["test query"] * 20

        self.start_memory_tracking()

        for query in queries:
            with self.measure_time("memory_efficient_search"):
                # Test that memory doesn't grow with repeated searches
                await hybrid_engine._vector_search(query, vector_index)
                await hybrid_engine._symbol_search(query, serena_graph)

        current_mem, peak_mem = self.stop_memory_tracking()

        print("Search Memory Efficiency:")
        print(f"  20 searches peak memory: {peak_mem:.2f}MB")
        print(f"  Memory per search: {peak_mem / len(queries):.2f}MB")

        # Memory usage should be bounded for repeated searches
        assert peak_mem < 300, f"Memory usage too high for repeated searches: {peak_mem:.2f}MB"


# Monkeypatch pipeline to add mock adapters
def _mock_adapters(self):
    """Add mock adapters to pipeline for performance testing."""

    async def mock_stage_acquire(context):
        await asyncio.sleep(0.01)  # Simulate file discovery

    async def mock_stage_repomapper(context):
        await asyncio.sleep(0.05)  # Simulate repo analysis
        context.repomap_data = MockDataGenerator.create_repo_map(100)

    async def mock_stage_serena(context):
        await asyncio.sleep(0.1)  # Simulate symbol analysis
        context.serena_graph = MockDataGenerator.create_serena_graph(500)

    async def mock_stage_leann(context):
        await asyncio.sleep(0.2)  # Simulate vector embedding
        context.vector_index = MockDataGenerator.create_vector_index(1000)

    async def mock_stage_snippets(context):
        await asyncio.sleep(0.03)  # Simulate snippet extraction
        context.snippets = Mock()

    async def mock_stage_bundle(context):
        await asyncio.sleep(0.02)  # Simulate bundle creation
        context.manifest = Mock()

    # Replace stages with mocked versions
    self._stage_acquire = mock_stage_acquire
    self._stage_repomapper = mock_stage_repomapper
    self._stage_serena = mock_stage_serena
    self._stage_leann = mock_stage_leann
    self._stage_snippets = mock_stage_snippets
    self._stage_bundle = mock_stage_bundle


# Add mock method to IndexingPipeline
IndexingPipeline._mock_adapters = _mock_adapters


if __name__ == "__main__":
    # Run benchmarks with pytest
    import subprocess

    subprocess.run(["python", "-m", "pytest", "-v", "-m", "benchmark", "--tb=short", __file__])
