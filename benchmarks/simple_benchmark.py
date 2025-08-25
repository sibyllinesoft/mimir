#!/usr/bin/env python3
"""
Simplified performance benchmark for Mimir critical paths.
"""

import asyncio
import time
import tracemalloc
import statistics
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from repoindex.pipeline.hybrid_search import HybridSearchEngine
from repoindex.pipeline.ask_index import SymbolGraphNavigator
from repoindex.data.schemas import (
    VectorIndex, VectorChunk, SerenaGraph, SymbolEntry, SymbolType,
    FeatureConfig
)


class SimpleBenchmark:
    """Simple performance benchmark focusing on critical operations."""
    
    def __init__(self):
        self.results = {}
    
    def time_operation(self, name: str):
        """Context manager for timing operations."""
        return TimingContext(self.results, name)
    
    def get_memory_usage(self):
        """Get current memory usage in MB."""
        tracemalloc.start()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak / 1024 / 1024


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


def create_test_vector_index(size: int = 10000) -> VectorIndex:
    """Create test vector index."""
    chunks = []
    for i in range(size):
        chunk = VectorChunk(
            chunk_id=f"chunk_{i}",
            path=f"src/file_{i % 100}.ts",
            span=(i * 10, (i + 1) * 10),
            content=f"async function test_{i}() {{ return await fetch('/api/{i}'); }}",
            embedding=[0.1 * j for j in range(384)],
            token_count=20
        )
        chunks.append(chunk)
    
    return VectorIndex(
        chunks=chunks,
        dimension=384,
        total_tokens=size * 20,
        model_name="test-model"
    )


def create_test_serena_graph(size: int = 5000) -> SerenaGraph:
    """Create test symbol graph."""
    entries = []
    
    # Create function definitions
    for i in range(size // 2):
        entry = SymbolEntry(
            path=f"src/module_{i % 50}.ts",
            span=(i * 20, (i + 1) * 20),
            symbol=f"function_{i}",
            type=SymbolType.DEF,
            sig=f"function function_{i}(): Promise<string>"
        )
        entries.append(entry)
    
    # Create references
    for i in range(size // 2, size):
        entry = SymbolEntry(
            path=f"src/module_{i % 50}.ts",
            span=(i * 20, (i + 1) * 20),
            symbol=f"function_{i % (size // 2)}",
            type=SymbolType.REF
        )
        entries.append(entry)
    
    return SerenaGraph(
        entries=entries,
        file_count=50,
        symbol_count=size // 2
    )


async def benchmark_vector_search():
    """Benchmark vector search operations."""
    print("🔍 Vector Search Performance")
    print("-" * 40)
    
    benchmark = SimpleBenchmark()
    engine = HybridSearchEngine()
    
    # Test different data sizes
    sizes = [1000, 5000, 10000]
    
    for size in sizes:
        print(f"\nTesting with {size} chunks:")
        vector_index = create_test_vector_index(size)
        
        queries = [
            "async function",
            "fetch api", 
            "return await",
            "Promise string",
            "test function"
        ]
        
        # Measure search performance
        for query in queries:
            with benchmark.time_operation(f"vector_search_{size}"):
                results = await engine._vector_search(query, vector_index)
            
        avg_time = statistics.mean(benchmark.results[f"vector_search_{size}"])
        print(f"  Average search time: {avg_time:.2f}ms")
        print(f"  Memory usage: {benchmark.get_memory_usage():.2f}MB")
        
        # Test performance target
        if avg_time > 500:
            print(f"  ⚠️  SLOW: {avg_time:.2f}ms > 500ms target")
        else:
            print(f"  ✅ GOOD: {avg_time:.2f}ms < 500ms target")
    
    return benchmark


async def benchmark_symbol_search():
    """Benchmark symbol search operations."""
    print("\n🔍 Symbol Search Performance")
    print("-" * 40)
    
    benchmark = SimpleBenchmark()
    navigator = SymbolGraphNavigator()
    
    # Test different graph sizes
    sizes = [1000, 3000, 5000]
    
    for size in sizes:
        print(f"\nTesting with {size} symbols:")
        serena_graph = create_test_serena_graph(size)
        
        test_symbols = [
            "function_0",
            "function_100",
            "function_500",
            "nonexistent"
        ]
        
        # Measure definition search
        for symbol in test_symbols:
            with benchmark.time_operation(f"symbol_def_{size}"):
                definition = await navigator.find_symbol_definition(symbol, serena_graph)
            
            with benchmark.time_operation(f"symbol_ref_{size}"):
                references = await navigator.find_symbol_references(symbol, serena_graph)
        
        avg_def_time = statistics.mean(benchmark.results[f"symbol_def_{size}"])
        avg_ref_time = statistics.mean(benchmark.results[f"symbol_ref_{size}"])
        
        print(f"  Definition search: {avg_def_time:.2f}ms")
        print(f"  Reference search: {avg_ref_time:.2f}ms")
        print(f"  Memory usage: {benchmark.get_memory_usage():.2f}MB")
        
        # Test performance targets
        if avg_def_time > 100:
            print(f"  ⚠️  SLOW DEF: {avg_def_time:.2f}ms > 100ms target")
        else:
            print(f"  ✅ GOOD DEF: {avg_def_time:.2f}ms < 100ms target")
            
        if avg_ref_time > 200:
            print(f"  ⚠️  SLOW REF: {avg_ref_time:.2f}ms > 200ms target")
        else:
            print(f"  ✅ GOOD REF: {avg_ref_time:.2f}ms < 200ms target")
    
    return benchmark


async def benchmark_hybrid_search():
    """Benchmark hybrid search combining vector and symbol search."""
    print("\n🔍 Hybrid Search Performance")
    print("-" * 40)
    
    benchmark = SimpleBenchmark()
    engine = HybridSearchEngine()
    
    # Create test data
    vector_index = create_test_vector_index(10000)
    serena_graph = create_test_serena_graph(5000)
    
    print(f"Testing with {len(vector_index.chunks)} chunks and {len(serena_graph.entries)} symbols")
    
    queries = [
        "async function test",
        "Promise return value",
        "fetch api call",
        "function definition",
        "module export"
    ]
    
    # Test different feature combinations
    feature_configs = [
        ("vector_only", FeatureConfig(vector=True, symbol=False, graph=False)),
        ("symbol_only", FeatureConfig(vector=False, symbol=True, graph=False)),
        ("vector_symbol", FeatureConfig(vector=True, symbol=True, graph=False)),
    ]
    
    for config_name, features in feature_configs:
        print(f"\n  Testing {config_name}:")
        
        for query in queries:
            with benchmark.time_operation(f"hybrid_{config_name}"):
                response = await engine.search(
                    query=query,
                    vector_index=vector_index if features.vector else None,
                    serena_graph=serena_graph if features.symbol else None,
                    repomap=None,
                    repo_root="/test",
                    rev="main",
                    features=features,
                    k=20
                )
        
        avg_time = statistics.mean(benchmark.results[f"hybrid_{config_name}"])
        print(f"    Average time: {avg_time:.2f}ms")
        
        # Performance targets
        target = 1000 if config_name == "vector_symbol" else 500
        if avg_time > target:
            print(f"    ⚠️  SLOW: {avg_time:.2f}ms > {target}ms target")
        else:
            print(f"    ✅ GOOD: {avg_time:.2f}ms < {target}ms target")
    
    return benchmark


async def benchmark_concurrent_operations():
    """Benchmark concurrent search operations."""
    print("\n🔍 Concurrent Operations Performance")
    print("-" * 40)
    
    benchmark = SimpleBenchmark()
    engine = HybridSearchEngine()
    navigator = SymbolGraphNavigator()
    
    # Create test data
    vector_index = create_test_vector_index(5000)
    serena_graph = create_test_serena_graph(2500)
    
    # Test concurrent vector searches
    print("\n  Testing concurrent vector searches:")
    concurrent_queries = ["async function"] * 10
    
    with benchmark.time_operation("concurrent_vector"):
        tasks = [engine._vector_search(q, vector_index) for q in concurrent_queries]
        results = await asyncio.gather(*tasks)
    
    concurrent_time = benchmark.results["concurrent_vector"][0]
    print(f"    10 concurrent searches: {concurrent_time:.2f}ms")
    
    # Test concurrent symbol searches  
    print("\n  Testing concurrent symbol searches:")
    concurrent_symbols = ["function_0"] * 10
    
    with benchmark.time_operation("concurrent_symbol"):
        tasks = [navigator.find_symbol_definition(s, serena_graph) for s in concurrent_symbols]
        results = await asyncio.gather(*tasks)
    
    concurrent_time = benchmark.results["concurrent_symbol"][0]
    print(f"    10 concurrent symbol searches: {concurrent_time:.2f}ms")
    
    return benchmark


def analyze_results(benchmarks: List[SimpleBenchmark]):
    """Analyze benchmark results and identify bottlenecks."""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Collect all results
    all_results = {}
    for bench in benchmarks:
        all_results.update(bench.results)
    
    # Identify bottlenecks
    bottlenecks = []
    
    # Check vector search performance
    vector_10k = [op for op in all_results.keys() if "vector_search_10000" in op]
    if vector_10k and all_results[vector_10k[0]]:
        avg_time = statistics.mean(all_results[vector_10k[0]])
        if avg_time > 500:
            bottlenecks.append(f"Vector search (10K chunks): {avg_time:.2f}ms > 500ms")
    
    # Check symbol search performance
    symbol_5k = [op for op in all_results.keys() if "symbol_def_5000" in op]
    if symbol_5k and all_results[symbol_5k[0]]:
        avg_time = statistics.mean(all_results[symbol_5k[0]])
        if avg_time > 100:
            bottlenecks.append(f"Symbol definition search (5K symbols): {avg_time:.2f}ms > 100ms")
    
    # Check hybrid search performance
    hybrid_ops = [op for op in all_results.keys() if "hybrid_vector_symbol" in op]
    if hybrid_ops and all_results[hybrid_ops[0]]:
        avg_time = statistics.mean(all_results[hybrid_ops[0]])
        if avg_time > 1000:
            bottlenecks.append(f"Hybrid search: {avg_time:.2f}ms > 1000ms")
    
    if bottlenecks:
        print("\n🚨 PERFORMANCE BOTTLENECKS IDENTIFIED:")
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"  {i}. {bottleneck}")
    else:
        print("\n✅ ALL PERFORMANCE TARGETS MET!")
    
    # Print optimization recommendations
    print("\n💡 OPTIMIZATION OPPORTUNITIES:")
    print("  1. Vector Search Optimization:")
    print("     - Implement embedding caching for repeated queries")
    print("     - Add early termination for similarity thresholds")
    print("     - Use approximate nearest neighbor (ANN) algorithms")
    
    print("  2. Symbol Search Optimization:")
    print("     - Create symbol name index/hash map for O(1) lookups")
    print("     - Implement symbol prefix tries for partial matching")
    print("     - Cache frequently accessed symbol references")
    
    print("  3. Hybrid Search Optimization:")
    print("     - Parallelize vector and symbol searches")
    print("     - Implement result caching with TTL")
    print("     - Add query result size limits")
    
    print("  4. Memory Optimization:")
    print("     - Implement lazy loading for large datasets")
    print("     - Use memory-mapped files for vector embeddings")
    print("     - Add garbage collection hints for large operations")
    
    return bottlenecks


async def main():
    """Run all benchmarks and analyze performance."""
    print("🚀 Mimir Performance Benchmark Analysis")
    print("="*60)
    
    benchmarks = []
    
    # Run individual benchmarks
    benchmarks.append(await benchmark_vector_search())
    benchmarks.append(await benchmark_symbol_search())
    benchmarks.append(await benchmark_hybrid_search())
    benchmarks.append(await benchmark_concurrent_operations())
    
    # Analyze results
    bottlenecks = analyze_results(benchmarks)
    
    # Performance summary
    print(f"\n📊 SUMMARY:")
    print(f"  Benchmarks completed: {len(benchmarks)}")
    print(f"  Bottlenecks identified: {len(bottlenecks)}")
    print(f"  Ready for optimization: {'Yes' if bottlenecks else 'Performance targets met'}")


if __name__ == "__main__":
    asyncio.run(main())