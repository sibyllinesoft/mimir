#!/usr/bin/env python3
"""
Simple benchmark runner for Mimir performance analysis.
"""

import asyncio
import time
import tracemalloc
import psutil
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from repoindex.pipeline.hybrid_search import HybridSearchEngine
from repoindex.pipeline.ask_index import SymbolGraphNavigator
from repoindex.data.schemas import (
    VectorIndex, VectorChunk, SerenaGraph, RepoMap, SymbolEntry, SymbolType,
    CodeSnippet, FeatureConfig, DependencyEdge, FileRank
)


class BenchmarkRunner:
    """Performance benchmark runner with metrics collection."""
    
    def __init__(self):
        self.results = {}
    
    def time_it(self, name: str):
        """Context manager for timing operations."""
        return TimingContext(self.results, name)
    
    def measure_memory(self):
        """Context manager for memory measurement."""
        return MemoryContext()
    
    def print_results(self):
        """Print benchmark results."""
        print("\n" + "="*60)
        print("MIMIR PERFORMANCE BENCHMARK RESULTS")
        print("="*60)
        
        for operation, times in self.results.items():
            if times:
                avg_time = statistics.mean(times)
                min_time = min(times)
                max_time = max(times)
                print(f"\n{operation}:")
                print(f"  Average: {avg_time:.2f}ms")
                print(f"  Min:     {min_time:.2f}ms")
                print(f"  Max:     {max_time:.2f}ms")
                print(f"  Samples: {len(times)}")


class TimingContext:
    """Context manager for timing operations."""
    
    def __init__(self, results: Dict, operation: str):
        self.results = results
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.perf_counter() - self.start_time
        if self.operation not in self.results:
            self.results[self.operation] = []
        self.results[self.operation].append(duration * 1000)  # Convert to ms


class MemoryContext:
    """Context manager for memory measurement."""
    
    def __enter__(self):
        tracemalloc.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.current_mb = current / 1024 / 1024
        self.peak_mb = peak / 1024 / 1024


class MockDataGenerator:
    """Generate realistic mock data for benchmarking."""
    
    @staticmethod
    def create_vector_index(chunk_count: int = 10000) -> VectorIndex:
        """Create a mock vector index with specified number of chunks."""
        chunks = []
        
        for i in range(chunk_count):
            chunk = VectorChunk(
                chunk_id=f"chunk_{i}",
                path=f"src/file_{i % 100}.ts",
                span=(i * 10, (i + 1) * 10),
                content=f"function example_{i}() {{ return 'content_{i}'; }}",
                embedding=[0.1 * j for j in range(384)],  # 384-dim embedding
                token_count=50
            )
            chunks.append(chunk)
            
        return VectorIndex(
            chunks=chunks,
            dimension=384,
            total_tokens=chunk_count * 50,
            model_name="mock-embedding-model"
        )
    
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
                callee=None
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
                callee=None
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
                callee=f"function_{(i + 1) % 100}"
            )
            entries.append(entry)
        
        # Calculate file and symbol counts
        file_count = len(set(entry.path for entry in entries))
        symbol_count = len(set(entry.symbol for entry in entries if entry.symbol))
            
        return SerenaGraph(
            entries=entries,
            file_count=file_count,
            symbol_count=symbol_count
        )
    
    @staticmethod
    def create_repo_map(file_count: int = 1000) -> RepoMap:
        """Create a mock repository map."""
        edges = []
        file_ranks = []
        
        # Create file dependencies
        for i in range(file_count):
            file_path = f"src/file_{i}.ts"
            
            # Create edges to other files
            for j in range(min(5, file_count - i - 1)):
                target_path = f"src/file_{i + j + 1}.ts"
                edge = DependencyEdge(
                    source=file_path,
                    target=target_path,
                    weight=0.5 + (j * 0.1),
                    edge_type="import"
                )
                edges.append(edge)
            
            # Create file rank
            rank = FileRank(
                path=file_path,
                rank=1.0 - (i / file_count),  # Decreasing rank
                centrality=0.8 + (i % 5) * 0.04,
                dependencies=[f"src/file_{j}.ts" for j in range(max(0, i-2), i)]
            )
            file_ranks.append(rank)
        
        return RepoMap(
            edges=edges, 
            file_ranks=file_ranks,
            total_files=file_count
        )


async def benchmark_vector_search():
    """Benchmark vector search performance."""
    print("\n🔍 Vector Search Benchmarks")
    print("-" * 40)
    
    runner = BenchmarkRunner()
    engine = HybridSearchEngine()
    vector_index = MockDataGenerator.create_vector_index(10000)
    
    queries = [
        "async function authentication",
        "database connection pool", 
        "error handling middleware",
        "React component state",
        "TypeScript interface definition"
    ]
    
    print(f"Testing with {len(vector_index.chunks)} chunks and {len(queries)} queries...")
    
    with runner.measure_memory() as mem:
        for query in queries:
            with runner.time_it("vector_search"):
                results = await engine._vector_search(query, vector_index)
                print(f"  Query '{query[:20]}...': {len(results)} results")
    
    print(f"Peak memory usage: {mem.peak_mb:.2f}MB")
    
    # Test concurrent searches
    print("\nTesting concurrent vector searches...")
    concurrent_queries = ["async function"] * 10
    
    with runner.time_it("concurrent_vector_search"):
        tasks = [engine._vector_search(q, vector_index) for q in concurrent_queries]
        results = await asyncio.gather(*tasks)
    
    print(f"10 concurrent searches completed")
    
    return runner


async def benchmark_symbol_search():
    """Benchmark symbol search performance."""
    print("\n🔍 Symbol Search Benchmarks")
    print("-" * 40)
    
    runner = BenchmarkRunner()
    navigator = SymbolGraphNavigator()
    serena_graph = MockDataGenerator.create_serena_graph(5000)
    
    print(f"Testing with {len(serena_graph.entries)} symbols...")
    
    test_queries = [
        "function_0",
        "function_100", 
        "function_500",
        "function_1000",
        "nonexistent_function"
    ]
    
    with runner.measure_memory() as mem:
        for query in test_queries:
            with runner.time_it("symbol_definition_search"):
                definition = await navigator.find_symbol_definition(query, serena_graph)
                
            with runner.time_it("symbol_references_search"):
                references = await navigator.find_symbol_references(query, serena_graph)
                
            found = "✓" if definition else "✗"
            ref_count = len(references) if references else 0
            print(f"  Query '{query}': {found} definition, {ref_count} references")
    
    print(f"Peak memory usage: {mem.peak_mb:.2f}MB")
    
    # Test symbol graph navigation
    print("\nTesting symbol graph navigation...")
    test_symbols = [
        SymbolEntry(
            path="src/test.ts",
            span=(0, 10),
            symbol="function_0",
            type=SymbolType.DEF,
            sig="function function_0(): void"
        )
    ]
    
    intents = [Mock(intent_type="flow", targets=["function_0"])]
    
    with runner.time_it("symbol_graph_navigation"):
        evidence = await navigator._walk_symbol_graph(test_symbols, serena_graph, intents)
    
    print(f"Graph navigation found {len(evidence)} related symbols")
    
    return runner


async def benchmark_hybrid_search():
    """Benchmark full hybrid search performance."""
    print("\n🔍 Hybrid Search Benchmarks")
    print("-" * 40)
    
    runner = BenchmarkRunner()
    engine = HybridSearchEngine()
    
    # Create large datasets
    print("Creating test datasets...")
    vector_index = MockDataGenerator.create_vector_index(10000)
    serena_graph = MockDataGenerator.create_serena_graph(5000)
    repo_map = MockDataGenerator.create_repo_map(1000)
    
    print(f"  Vector index: {len(vector_index.chunks)} chunks")
    print(f"  Symbol graph: {len(serena_graph.entries)} symbols")
    print(f"  Repo map: {len(repo_map.edges)} edges")
    
    queries = [
        "authentication middleware function",
        "database connection error handling", 
        "React component lifecycle methods",
        "TypeScript type definitions",
        "async promise handling"
    ]
    
    features = FeatureConfig(vector=True, symbol=True, graph=True)
    
    print(f"\nTesting {len(queries)} hybrid searches...")
    
    with runner.measure_memory() as mem:
        for query in queries:
            with runner.time_it("hybrid_search_full"):
                response = await engine.search(
                    query=query,
                    vector_index=vector_index,
                    serena_graph=serena_graph,
                    repomap=repo_map,
                    repo_root="/test/repo",
                    rev="main",
                    features=features,
                    k=20
                )
                print(f"  Query '{query[:30]}...': {len(response.results)} results in {response.execution_time_ms:.2f}ms")
    
    print(f"Peak memory usage: {mem.peak_mb:.2f}MB")
    
    # Test scaling with result count
    print("\nTesting result count scaling...")
    query = "function authentication"
    k_values = [5, 10, 20, 50, 100]
    
    for k in k_values:
        with runner.time_it(f"hybrid_search_k_{k}"):
            response = await engine.search(
                query=query,
                vector_index=vector_index,
                serena_graph=serena_graph,
                repomap=repo_map,
                repo_root="/test/repo",
                rev="main",
                features=features,
                k=k
            )
        print(f"  k={k}: {len(response.results)} results")
    
    return runner


async def benchmark_memory_usage():
    """Benchmark memory usage patterns."""
    print("\n🧠 Memory Usage Benchmarks")
    print("-" * 40)
    
    runner = BenchmarkRunner()
    
    # Test memory scaling with data size
    sizes = [1000, 5000, 10000, 20000]
    print("Testing memory scaling with dataset size...")
    
    for size in sizes:
        with runner.measure_memory() as mem:
            vector_index = MockDataGenerator.create_vector_index(size)
            serena_graph = MockDataGenerator.create_serena_graph(size)
        
        print(f"  {size:5d} items: {mem.peak_mb:6.2f}MB peak memory")
        
        # Clean up
        del vector_index, serena_graph
    
    # Test memory efficiency during repeated operations
    print("\nTesting memory efficiency during repeated searches...")
    vector_index = MockDataGenerator.create_vector_index(10000)
    engine = HybridSearchEngine()
    
    queries = ["test query"] * 20
    
    with runner.measure_memory() as mem:
        for i, query in enumerate(queries):
            with runner.time_it("memory_efficient_search"):
                await engine._vector_search(query, vector_index)
            
            if i % 5 == 0:
                process = psutil.Process()
                current_mem = process.memory_info().rss / 1024 / 1024
                print(f"  After {i+1:2d} searches: {current_mem:.2f}MB")
    
    print(f"Final peak memory: {mem.peak_mb:.2f}MB")
    
    return runner


async def main():
    """Run all benchmarks and analyze results."""
    print("🚀 Starting Mimir Performance Benchmarks")
    print("=" * 60)
    
    # Run individual benchmarks
    vector_runner = await benchmark_vector_search()
    symbol_runner = await benchmark_symbol_search()
    hybrid_runner = await benchmark_hybrid_search()
    memory_runner = await benchmark_memory_usage()
    
    # Combine all results
    all_results = BenchmarkRunner()
    for runner in [vector_runner, symbol_runner, hybrid_runner, memory_runner]:
        all_results.results.update(runner.results)
    
    # Print comprehensive results
    all_results.print_results()
    
    # Performance analysis
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Identify potential bottlenecks
    bottlenecks = []
    
    if "vector_search" in all_results.results:
        avg_vector = statistics.mean(all_results.results["vector_search"])
        if avg_vector > 500:
            bottlenecks.append(f"Vector search avg: {avg_vector:.2f}ms (target: <500ms)")
    
    if "symbol_definition_search" in all_results.results:
        avg_symbol = statistics.mean(all_results.results["symbol_definition_search"])
        if avg_symbol > 100:
            bottlenecks.append(f"Symbol search avg: {avg_symbol:.2f}ms (target: <100ms)")
    
    if "hybrid_search_full" in all_results.results:
        avg_hybrid = statistics.mean(all_results.results["hybrid_search_full"])
        if avg_hybrid > 2000:
            bottlenecks.append(f"Hybrid search avg: {avg_hybrid:.2f}ms (target: <2000ms)")
    
    if bottlenecks:
        print("\n🚨 PERFORMANCE BOTTLENECKS IDENTIFIED:")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"  {i}. {bottleneck}")
    else:
        print("\n✅ All performance targets met!")
    
    # Performance recommendations
    print("\n💡 OPTIMIZATION RECOMMENDATIONS:")
    print("  1. Implement caching for repeated vector searches")
    print("  2. Add symbol lookup indexing for faster symbol searches")
    print("  3. Optimize hybrid search with early termination")
    print("  4. Add async batching for concurrent operations")
    print("  5. Implement memory pooling for large datasets")


if __name__ == "__main__":
    asyncio.run(main())