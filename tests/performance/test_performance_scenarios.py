"""
Performance testing scenarios for Mimir components.

This test suite focuses on benchmarking critical performance paths,
memory usage patterns, and scalability characteristics under load.
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import Mock, patch
import tempfile
import json

import pytest

from src.repoindex.config import MimirConfig
from src.repoindex.util.fs import (
    compute_file_hash,
    atomic_write_text,
    atomic_write_json,
    ensure_directory,
    get_file_metadata,
)
from src.repoindex.data.schemas import (
    RepoInfo,
    IndexConfig,
    IndexManifest,
    SearchResult,
    CodeSnippet,
    Citation,
    SearchScores,
)


class TestFileOperationPerformance:
    """Test performance characteristics of file operations."""
    
    def test_hash_computation_performance(self, tmp_path):
        """Benchmark file hashing performance with various file sizes."""
        # Test data of different sizes
        test_sizes = [
            (1024, "1KB"),           # Small config files
            (10 * 1024, "10KB"),     # Medium source files
            (100 * 1024, "100KB"),   # Large source files
            (1024 * 1024, "1MB"),    # Very large files
        ]
        
        performance_data = {}
        
        for size_bytes, label in test_sizes:
            # Create test file
            test_file = tmp_path / f"test_{label.replace(' ', '_')}.txt"
            content = "A" * size_bytes
            test_file.write_text(content)
            
            # Benchmark hashing
            start_time = time.perf_counter()
            hash_value = compute_file_hash(test_file)
            end_time = time.perf_counter()
            
            duration_ms = (end_time - start_time) * 1000
            performance_data[label] = {
                "duration_ms": duration_ms,
                "throughput_mb_per_s": (size_bytes / (1024 * 1024)) / (duration_ms / 1000),
                "hash": hash_value
            }
        
        # Performance assertions
        assert performance_data["1KB"]["duration_ms"] < 10, "Small files should hash in <10ms"
        assert performance_data["100KB"]["duration_ms"] < 50, "Medium files should hash in <50ms"
        assert performance_data["1MB"]["duration_ms"] < 200, "Large files should hash in <200ms"
        
        # Throughput should be reasonable
        assert performance_data["1MB"]["throughput_mb_per_s"] > 5, "Should achieve >5MB/s throughput"
        
        print(f"\\nFile Hashing Performance:\\n{json.dumps(performance_data, indent=2)}")
    
    def test_concurrent_file_operations(self, tmp_path):
        """Test performance of concurrent file operations."""
        num_threads = 10
        files_per_thread = 20
        
        def worker_thread(thread_id):
            """Worker function for concurrent file operations."""
            thread_dir = tmp_path / f"thread_{thread_id}"
            ensure_directory(thread_dir)
            
            results = []
            for i in range(files_per_thread):
                file_path = thread_dir / f"file_{i}.txt"
                content = f"Thread {thread_id}, file {i} content with some data"
                
                # Time atomic write
                start_time = time.perf_counter()
                atomic_write_text(file_path, content)
                write_time = time.perf_counter() - start_time
                
                # Time hash computation
                start_time = time.perf_counter()
                hash_value = compute_file_hash(file_path)
                hash_time = time.perf_counter() - start_time
                
                results.append({
                    "write_time": write_time,
                    "hash_time": hash_time,
                    "total_time": write_time + hash_time
                })
            
            return results
        
        # Execute concurrent operations
        start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker_thread, i) for i in range(num_threads)]
            all_results = [future.result() for future in futures]
        
        total_time = time.perf_counter() - start_time
        
        # Flatten results
        flat_results = [item for sublist in all_results for item in sublist]
        total_operations = len(flat_results)
        
        # Calculate statistics
        avg_write_time = sum(r["write_time"] for r in flat_results) / total_operations
        avg_hash_time = sum(r["hash_time"] for r in flat_results) / total_operations
        operations_per_second = total_operations / total_time
        
        # Performance assertions
        assert avg_write_time < 0.01, "Average write time should be <10ms"
        assert avg_hash_time < 0.005, "Average hash time should be <5ms"
        assert operations_per_second > 100, "Should achieve >100 operations/second"
        
        # Verify all files were created correctly
        total_files = sum(len(list(thread_dir.glob("*.txt"))) 
                         for thread_dir in tmp_path.glob("thread_*"))
        assert total_files == num_threads * files_per_thread
        
        print(f"\\nConcurrent Operations Performance:")
        print(f"  Total operations: {total_operations}")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Operations/sec: {operations_per_second:.1f}")
        print(f"  Avg write time: {avg_write_time*1000:.1f}ms")
        print(f"  Avg hash time: {avg_hash_time*1000:.1f}ms")


class TestDataModelPerformance:
    """Test performance of data model operations."""
    
    def test_schema_serialization_performance(self):
        """Benchmark Pydantic model serialization performance."""
        # Create complex nested data structure
        repo_info = RepoInfo(
            root="/large/repository/path",
            rev="main",
            worktree_dirty=False
        )
        
        config = IndexConfig(
            languages=["ts", "tsx", "js", "jsx", "py", "rs", "go"],
            excludes=[
                "node_modules/", ".git/", "dist/", "build/", 
                "target/", "__pycache__/", ".pytest_cache/"
            ],
            context_lines=5,
            max_files_to_embed=1000
        )
        
        manifest = IndexManifest(repo=repo_info, config=config)
        
        # Create multiple search results
        search_results = []
        for i in range(100):
            content = CodeSnippet(
                path=f"src/component_{i}.ts",
                span=(i*10, i*10+20),
                hash=f"hash_{i}",
                pre="// Previous context",
                text=f"export function component_{i}() {{ return 'test'; }}",
                post="// Following context",
                line_start=i*10,
                line_end=i*10+20,
            )
            
            citation = Citation(
                repo_root="/repo",
                rev="main",
                path=f"src/component_{i}.ts",
                span=(i*10, i*10+20),
                content_sha=f"hash_{i}"
            )
            
            scores = SearchScores(vector=0.8 + i*0.001, symbol=0.9, graph=0.3)
            
            result = SearchResult(
                path=f"src/component_{i}.ts",
                span=(i*10, i*10+20),
                score=0.85 + i*0.001,
                scores=scores,
                content=content,
                citation=citation
            )
            search_results.append(result)
        
        # Benchmark serialization
        start_time = time.perf_counter()
        for _ in range(10):  # Multiple iterations for better measurement
            manifest_json = manifest.model_dump()
            results_json = [result.model_dump() for result in search_results]
        
        serialization_time = time.perf_counter() - start_time
        
        # Benchmark deserialization
        start_time = time.perf_counter()
        for _ in range(10):
            restored_manifest = IndexManifest.model_validate(manifest_json)
            restored_results = [SearchResult.model_validate(data) for data in results_json]
        
        deserialization_time = time.perf_counter() - start_time
        
        # Performance assertions
        assert serialization_time < 0.1, "Serialization should complete in <100ms"
        assert deserialization_time < 0.2, "Deserialization should complete in <200ms"
        
        # Data size calculations
        json_size = len(json.dumps(manifest_json)) + sum(len(json.dumps(r)) for r in results_json)
        objects_per_second = (100 * 10 * 2) / (serialization_time + deserialization_time)  # Total operations
        
        print(f"\\nData Model Performance:")
        print(f"  Serialization time: {serialization_time*1000:.1f}ms")
        print(f"  Deserialization time: {deserialization_time*1000:.1f}ms")
        print(f"  JSON size: {json_size/1024:.1f}KB")
        print(f"  Objects/sec: {objects_per_second:.0f}")
    
    def test_config_loading_performance(self, tmp_path):
        """Benchmark configuration loading and validation."""
        # Create test config file
        config_file = tmp_path / "test_config.yaml"
        config_data = {
            "server": {
                "host": "localhost",
                "port": 8080
            },
            "ai": {
                "google_api_key": "test-key",
                "gemini_model": "gemini-1.5-flash"
            },
            "index": {
                "languages": ["ts", "tsx", "js", "jsx", "py"],
                "excludes": ["node_modules/", ".git/", "dist/"],
                "context_lines": 5
            }
        }
        
        # Write YAML config
        import yaml
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
        
        # Benchmark config loading
        load_times = []
        for _ in range(20):  # Multiple iterations
            start_time = time.perf_counter()
            config = MimirConfig.load_from_file(str(config_file))
            load_time = time.perf_counter() - start_time
            load_times.append(load_time)
        
        avg_load_time = sum(load_times) / len(load_times)
        min_load_time = min(load_times)
        max_load_time = max(load_times)
        
        # Performance assertions
        assert avg_load_time < 0.01, "Average config loading should be <10ms"
        assert max_load_time < 0.05, "Max config loading should be <50ms"
        
        print(f"\\nConfig Loading Performance:")
        print(f"  Average load time: {avg_load_time*1000:.1f}ms")
        print(f"  Min load time: {min_load_time*1000:.1f}ms")
        print(f"  Max load time: {max_load_time*1000:.1f}ms")
        print(f"  Config file size: {config_file.stat().st_size} bytes")


class TestMemoryUsagePatterns:
    """Test memory usage and resource management."""
    
    def test_large_data_structure_memory(self):
        """Test memory usage patterns with large data structures."""
        import tracemalloc
        
        # Start memory tracing
        tracemalloc.start()
        
        # Create baseline memory snapshot
        baseline_snapshot = tracemalloc.take_snapshot()
        
        # Create large collection of search results
        search_results = []
        for i in range(1000):  # Large number of objects
            content = CodeSnippet(
                path=f"src/large_component_{i}.ts",
                span=(i*100, i*100+200),
                hash=f"large_hash_{i}_with_more_content",
                pre="// Extended previous context " * 5,
                text=f"export function largeComponent_{i}() {{ " + 
                     "const data = " + "{'key': 'value'}" * 10 + 
                     f"; return process(data); }}",
                post="// Extended following context " * 5,
                line_start=i*100,
                line_end=i*100+200,
            )
            
            citation = Citation(
                repo_root="/very/long/repository/path",
                rev="feature/large-components-implementation",
                path=f"src/large_component_{i}.ts",
                span=(i*100, i*100+200),
                content_sha=f"large_hash_{i}_with_more_content"
            )
            
            scores = SearchScores(
                vector=0.8 + (i % 100) * 0.001, 
                symbol=0.9 + (i % 50) * 0.001, 
                graph=0.3 + (i % 25) * 0.001
            )
            
            result = SearchResult(
                path=f"src/large_component_{i}.ts",
                span=(i*100, i*100+200),
                score=0.85 + (i % 150) * 0.0001,
                scores=scores,
                content=content,
                citation=citation
            )
            search_results.append(result)
        
        # Take memory snapshot after creating objects
        after_creation_snapshot = tracemalloc.take_snapshot()
        
        # Serialize all objects
        serialized_data = [result.model_dump() for result in search_results]
        
        # Take memory snapshot after serialization
        after_serialization_snapshot = tracemalloc.take_snapshot()
        
        # Calculate memory usage
        creation_stats = after_creation_snapshot.compare_to(baseline_snapshot, 'lineno')
        serialization_stats = after_serialization_snapshot.compare_to(after_creation_snapshot, 'lineno')
        
        total_creation_memory = sum(stat.size for stat in creation_stats[:10])  # Top 10 allocations
        total_serialization_memory = sum(stat.size for stat in serialization_stats[:10])
        
        # Memory efficiency assertions
        memory_per_object = total_creation_memory / len(search_results)
        assert memory_per_object < 10000, "Memory per object should be <10KB"  # Reasonable limit
        
        # Cleanup and verify memory release
        del search_results
        del serialized_data
        
        final_snapshot = tracemalloc.take_snapshot()
        cleanup_stats = final_snapshot.compare_to(after_serialization_snapshot, 'lineno')
        
        print(f"\\nMemory Usage Analysis:")
        print(f"  Objects created: 1000")
        print(f"  Total creation memory: {total_creation_memory/1024:.1f}KB")
        print(f"  Memory per object: {memory_per_object:.0f} bytes")
        print(f"  Serialization overhead: {total_serialization_memory/1024:.1f}KB")
        print(f"  Memory after cleanup: verified")
        
        tracemalloc.stop()


class TestStressScenarios:
    """Test system behavior under stress conditions."""
    
    def test_concurrent_config_access(self, tmp_path):
        """Test configuration access under high concurrency."""
        # Create config file
        config_file = tmp_path / "stress_config.yaml"
        config_data = {
            "server": {"host": "localhost", "port": 8080},
            "ai": {"google_api_key": "test-key"},
        }
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
        
        # Stress test parameters
        num_threads = 20
        operations_per_thread = 50
        
        # Results tracking
        results = {"successes": 0, "failures": 0, "errors": []}
        results_lock = threading.Lock()
        
        def stress_worker():
            """Worker function for stress testing."""
            for _ in range(operations_per_thread):
                try:
                    config = MimirConfig.load_from_file(str(config_file))
                    assert config.server.host == "localhost"
                    
                    with results_lock:
                        results["successes"] += 1
                        
                except Exception as e:
                    with results_lock:
                        results["failures"] += 1
                        results["errors"].append(str(e))
        
        # Execute stress test
        start_time = time.perf_counter()
        threads = []
        for _ in range(num_threads):
            thread = threading.Thread(target=stress_worker)
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        end_time = time.perf_counter()
        
        total_operations = num_threads * operations_per_thread
        success_rate = results["successes"] / total_operations
        operations_per_second = total_operations / (end_time - start_time)
        
        # Stress test assertions
        assert success_rate > 0.99, f"Success rate should be >99%, got {success_rate:.3f}"
        assert results["failures"] == 0, f"Should have no failures, got {results['failures']}"
        assert operations_per_second > 500, f"Should achieve >500 ops/sec, got {operations_per_second:.1f}"
        
        print(f"\\nStress Test Results:")
        print(f"  Total operations: {total_operations}")
        print(f"  Success rate: {success_rate:.4f}")
        print(f"  Operations/sec: {operations_per_second:.1f}")
        print(f"  Duration: {end_time - start_time:.3f}s")
        
        if results["errors"]:
            print(f"  Errors encountered: {len(results['errors'])}")
            for error in results["errors"][:5]:  # Show first 5 errors
                print(f"    - {error}")


@pytest.mark.performance
class TestPerformanceRegression:
    """Test suite to detect performance regressions."""
    
    def test_baseline_operations_benchmark(self, tmp_path):
        """Establish baseline performance metrics for regression detection."""
        benchmarks = {}
        
        # File operation benchmark
        test_file = tmp_path / "benchmark.txt"
        content = "Benchmark content " * 100  # Reasonable size
        
        start = time.perf_counter()
        atomic_write_text(test_file, content)
        benchmarks["file_write"] = time.perf_counter() - start
        
        start = time.perf_counter()
        hash_value = compute_file_hash(test_file)
        benchmarks["file_hash"] = time.perf_counter() - start
        
        start = time.perf_counter()
        metadata = get_file_metadata(test_file)
        benchmarks["file_metadata"] = time.perf_counter() - start
        
        # Data model benchmark
        start = time.perf_counter()
        repo_info = RepoInfo(root=str(tmp_path), rev="main", worktree_dirty=False)
        config = IndexConfig()
        manifest = IndexManifest(repo=repo_info, config=config)
        json_data = manifest.model_dump()
        benchmarks["model_serialization"] = time.perf_counter() - start
        
        start = time.perf_counter()
        restored = IndexManifest.model_validate(json_data)
        benchmarks["model_deserialization"] = time.perf_counter() - start
        
        # Performance regression thresholds (in milliseconds)
        thresholds = {
            "file_write": 5.0,
            "file_hash": 2.0,
            "file_metadata": 1.0,
            "model_serialization": 1.0,
            "model_deserialization": 2.0,
        }
        
        # Check against thresholds
        for operation, duration in benchmarks.items():
            duration_ms = duration * 1000
            threshold = thresholds[operation]
            
            assert duration_ms < threshold, (
                f"{operation} took {duration_ms:.2f}ms, "
                f"exceeds threshold of {threshold}ms"
            )
        
        print(f"\\nBaseline Performance Benchmarks:")
        for operation, duration in benchmarks.items():
            print(f"  {operation}: {duration*1000:.2f}ms")
        
        # Save benchmarks for regression tracking
        benchmark_file = tmp_path / "performance_baseline.json"
        with open(benchmark_file, 'w') as f:
            json.dump({k: v*1000 for k, v in benchmarks.items()}, f, indent=2)