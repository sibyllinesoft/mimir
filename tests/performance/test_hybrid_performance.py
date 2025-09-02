"""
Performance Benchmarking Tests for Hybrid Mimir-Lens Pipeline.

Comprehensive performance validation and benchmarking to ensure the hybrid
pipeline meets performance targets and scales effectively.
"""

import asyncio
import pytest
import time
import statistics
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from typing import Dict, List, Any, Optional
import tempfile
import json
import random
import string

# Import hybrid pipeline components
from src.repoindex.pipeline.hybrid_indexing import HybridIndexingPipeline, HybridStrategy
from src.repoindex.pipeline.hybrid_discover import HybridDiscoveryStage
from src.repoindex.pipeline.hybrid_code_embeddings import HybridCodeEmbeddingStage
from src.repoindex.pipeline.parallel_processor import ParallelProcessor, ResourceLimits
from src.repoindex.pipeline.result_synthesizer import ResultSynthesizer
from src.repoindex.pipeline.hybrid_metrics import MetricsCollector, PipelineMetrics
from src.repoindex.data.schemas import VectorChunk
from src.repoindex.util.log import get_logger

logger = get_logger(__name__)


class PerformanceBenchmark:
    """Performance benchmarking utility."""
    
    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.measurements = []
    
    def start(self):
        """Start timing measurement."""
        self.start_time = time.time()
    
    def stop(self):
        """Stop timing measurement."""
        self.end_time = time.time()
        if self.start_time:
            duration = (self.end_time - self.start_time) * 1000  # milliseconds
            self.measurements.append(duration)
            logger.info(f"Benchmark '{self.name}': {duration:.2f}ms")
            return duration
        return 0.0
    
    def get_stats(self) -> Dict[str, float]:
        """Get statistical summary of measurements."""
        if not self.measurements:
            return {}
        
        return {
            'count': len(self.measurements),
            'min': min(self.measurements),
            'max': max(self.measurements),
            'mean': statistics.mean(self.measurements),
            'median': statistics.median(self.measurements),
            'stdev': statistics.stdev(self.measurements) if len(self.measurements) > 1 else 0.0
        }


def generate_mock_files(count: int = 100) -> List[str]:
    """Generate mock file paths for testing."""
    extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.rs', '.go']
    directories = ['src', 'lib', 'tests', 'examples', 'utils']
    
    files = []
    for i in range(count):
        directory = random.choice(directories)
        extension = random.choice(extensions)
        name = ''.join(random.choices(string.ascii_lowercase, k=8))
        files.append(f"{directory}/{name}{extension}")
    
    return files


def generate_mock_chunks(count: int = 500) -> List[VectorChunk]:
    """Generate mock vector chunks for testing."""
    chunks = []
    files = generate_mock_files(count // 10)
    
    for i in range(count):
        file_path = random.choice(files)
        content_length = random.randint(50, 500)
        content = ''.join(random.choices(string.ascii_letters + string.digits + ' \n', k=content_length))
        
        chunk = VectorChunk(
            chunk_id=f'chunk_{i}',
            content=content,
            file_path=file_path,
            start_line=random.randint(1, 100),
            end_line=random.randint(101, 200),
            chunk_type='code' if random.random() > 0.3 else 'text'
        )
        chunks.append(chunk)
    
    return chunks


@pytest.fixture
def performance_metrics_collector():
    """Create metrics collector for performance testing."""
    collector = MetricsCollector(
        collection_interval_seconds=0.1,
        enable_alerts=False  # Disable alerts during testing
    )
    return collector


@pytest.fixture
def mock_high_performance_lens_client():
    """Create mock Lens client optimized for performance testing."""
    mock_client = AsyncMock()
    
    # Fast health check
    mock_client.health_check.return_value = Mock(status=Mock(value='healthy'))
    
    # Fast bulk indexing
    async def mock_bulk_index(documents, **kwargs):
        # Simulate processing time based on document count
        processing_time = len(documents) * 0.1  # 0.1ms per document
        await asyncio.sleep(processing_time / 1000)  # Convert to seconds
        
        return Mock(
            success=True,
            data={
                'indexed_documents': len(documents),
                'processing_time_ms': processing_time
            },
            response_time_ms=processing_time
        )
    
    mock_client.bulk_index.side_effect = mock_bulk_index
    
    # Fast embedding generation
    async def mock_generate_embeddings(documents, **kwargs):
        # Simulate embedding generation time
        processing_time = len(documents) * 0.5  # 0.5ms per document
        await asyncio.sleep(processing_time / 1000)
        
        embeddings = []
        for i, doc in enumerate(documents):
            embeddings.append({
                'id': doc.get('id', f'doc_{i}'),
                'vector': [random.random() for _ in range(384)],  # 384-dim embedding
                'model': 'fast_mock_model'
            })
        
        return Mock(
            success=True,
            data={'embeddings': embeddings},
            response_time_ms=processing_time
        )
    
    mock_client.generate_embeddings.side_effect = mock_generate_embeddings
    
    return mock_client


class TestHybridDiscoveryPerformance:
    """Performance tests for hybrid discovery stage."""
    
    @pytest.mark.asyncio
    async def test_discovery_scalability(self, mock_high_performance_lens_client, performance_metrics_collector):
        """Test discovery performance with increasing file counts."""
        file_counts = [100, 500, 1000, 2000]
        benchmarks = {}
        
        for file_count in file_counts:
            benchmark = PerformanceBenchmark(f"discovery_{file_count}_files")
            mock_files = generate_mock_files(file_count)
            
            # Create mock context
            context = Mock()
            context.repo_path = Path('/test')
            context.work_dir = Path('/test/work')
            context.incremental_mode = False
            
            with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_high_performance_lens_client):
                with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                    # Mock file discovery with realistic timing
                    mock_discovery_instance = AsyncMock()
                    
                    async def mock_discover_files(*args, **kwargs):
                        await asyncio.sleep(file_count * 0.0001)  # Simulate work
                        return mock_files
                    
                    mock_discovery_instance.discover_files = mock_discover_files
                    mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                    mock_discovery_instance.get_file_metadata.return_value = {
                        f: {'size': 1000, 'hash': f'hash_{i}'} 
                        for i, f in enumerate(mock_files[:50])  # Limit metadata processing
                    }
                    mock_discovery.return_value = mock_discovery_instance
                    
                    # Run benchmark
                    stage = HybridDiscoveryStage(
                        enable_lens_indexing=True,
                        batch_size=50,
                        max_files_per_batch=200
                    )
                    
                    benchmark.start()
                    await stage.execute(context)
                    duration = benchmark.stop()
                    
                    benchmarks[file_count] = duration
        
        # Analyze scalability
        logger.info("Discovery Scalability Results:")
        for file_count, duration in benchmarks.items():
            throughput = file_count / (duration / 1000)  # files per second
            logger.info(f"  {file_count} files: {duration:.2f}ms ({throughput:.1f} files/sec)")
        
        # Verify performance targets
        # Should handle 1000 files in under 5 seconds
        assert benchmarks[1000] < 5000, f"Discovery too slow for 1000 files: {benchmarks[1000]}ms"
        
        # Should scale roughly linearly (allowing for some overhead)
        ratio_1000_to_100 = benchmarks[1000] / benchmarks[100]
        assert ratio_1000_to_100 < 15, f"Poor scalability: {ratio_1000_to_100}x slowdown for 10x files"
    
    @pytest.mark.asyncio
    async def test_parallel_discovery_performance(self, mock_high_performance_lens_client):
        """Test parallel processing performance in discovery."""
        concurrency_levels = [1, 2, 4, 8]
        benchmarks = {}
        file_count = 1000
        
        for concurrency in concurrency_levels:
            benchmark = PerformanceBenchmark(f"discovery_concurrency_{concurrency}")
            
            context = Mock()
            context.repo_path = Path('/test')
            context.work_dir = Path('/test/work')
            
            with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_high_performance_lens_client):
                with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                    mock_discovery_instance = AsyncMock()
                    mock_discovery_instance.discover_files.return_value = generate_mock_files(file_count)
                    mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                    mock_discovery_instance.get_file_metadata.return_value = {}
                    mock_discovery.return_value = mock_discovery_instance
                    
                    stage = HybridDiscoveryStage(
                        concurrency_limit=concurrency,
                        enable_lens_indexing=True
                    )
                    
                    benchmark.start()
                    await stage.execute(context)
                    duration = benchmark.stop()
                    
                    benchmarks[concurrency] = duration
        
        # Analyze concurrency benefits
        logger.info("Discovery Concurrency Results:")
        baseline = benchmarks[1]
        for concurrency, duration in benchmarks.items():
            speedup = baseline / duration
            logger.info(f"  {concurrency} workers: {duration:.2f}ms ({speedup:.2f}x speedup)")
        
        # Should see some speedup with concurrency
        assert benchmarks[4] < benchmarks[1] * 0.8, "No significant speedup with concurrency"
    
    @pytest.mark.asyncio
    async def test_lens_vs_mimir_performance_comparison(self, mock_high_performance_lens_client):
        """Compare performance of Lens vs Mimir-only discovery."""
        file_count = 500
        iterations = 3
        
        lens_times = []
        mimir_times = []
        
        for i in range(iterations):
            context = Mock()
            context.repo_path = Path('/test')
            context.work_dir = Path('/test/work')
            
            with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                mock_discovery_instance = AsyncMock()
                mock_discovery_instance.discover_files.return_value = generate_mock_files(file_count)
                mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                mock_discovery_instance.get_file_metadata.return_value = {}
                mock_discovery.return_value = mock_discovery_instance
                
                # Test with Lens enabled
                with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_high_performance_lens_client):
                    stage_lens = HybridDiscoveryStage(enable_lens_indexing=True)
                    
                    start_time = time.time()
                    await stage_lens.execute(context)
                    lens_time = (time.time() - start_time) * 1000
                    lens_times.append(lens_time)
                
                # Test Mimir-only (Lens unavailable)
                mock_failing_lens = AsyncMock()
                mock_failing_lens.health_check.side_effect = Exception("Unavailable")
                
                with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_failing_lens):
                    stage_mimir = HybridDiscoveryStage(enable_lens_indexing=True)
                    
                    start_time = time.time()
                    await stage_mimir.execute(context)
                    mimir_time = (time.time() - start_time) * 1000
                    mimir_times.append(mimir_time)
        
        avg_lens_time = statistics.mean(lens_times)
        avg_mimir_time = statistics.mean(mimir_times)
        
        logger.info(f"Average Lens discovery time: {avg_lens_time:.2f}ms")
        logger.info(f"Average Mimir-only discovery time: {avg_mimir_time:.2f}ms")
        
        # Lens should provide some performance benefit for bulk operations
        # (Though in this mock setup, the benefit may be minimal)
        performance_ratio = avg_lens_time / avg_mimir_time
        logger.info(f"Lens/Mimir performance ratio: {performance_ratio:.2f}")


class TestHybridEmbeddingPerformance:
    """Performance tests for hybrid embedding stage."""
    
    @pytest.mark.asyncio
    async def test_embedding_throughput(self, mock_high_performance_lens_client):
        """Test embedding generation throughput."""
        chunk_counts = [100, 500, 1000]
        batch_sizes = [16, 32, 64]
        
        results = {}
        
        for chunk_count in chunk_counts:
            for batch_size in batch_sizes:
                test_key = f"{chunk_count}chunks_batch{batch_size}"
                benchmark = PerformanceBenchmark(test_key)
                
                # Create test chunks
                chunks = generate_mock_chunks(chunk_count)
                
                context = Mock()
                context.vector_chunks = chunks
                
                with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_high_performance_lens_client):
                    with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                        # Mock fast Mimir adapter
                        mock_adapter_instance = AsyncMock()
                        mock_adapter_instance.initialize = AsyncMock()
                        
                        async def mock_embed_chunk(chunk):
                            await asyncio.sleep(0.001)  # 1ms per chunk
                            chunk.embedding = [random.random() for _ in range(384)]
                            return chunk
                        
                        mock_adapter_instance.embed_code_chunk = mock_embed_chunk
                        mock_adapter.return_value = mock_adapter_instance
                        
                        stage = HybridCodeEmbeddingStage(
                            enable_lens_vectors=True,
                            batch_size=batch_size,
                            concurrency_limit=4
                        )
                        
                        benchmark.start()
                        await stage.execute(context)
                        duration = benchmark.stop()
                        
                        throughput = chunk_count / (duration / 1000)  # chunks per second
                        results[test_key] = {
                            'duration_ms': duration,
                            'throughput_cps': throughput
                        }
                        
                        logger.info(f"{test_key}: {duration:.2f}ms ({throughput:.1f} chunks/sec)")
        
        # Analyze optimal batch size
        for chunk_count in chunk_counts:
            batch_results = {
                batch_size: results[f"{chunk_count}chunks_batch{batch_size}"]["throughput_cps"]
                for batch_size in batch_sizes
            }
            best_batch_size = max(batch_results, key=batch_results.get)
            logger.info(f"Optimal batch size for {chunk_count} chunks: {best_batch_size}")
        
        # Verify performance targets
        # Should handle 1000 chunks in under 10 seconds
        best_1000_time = min(
            results[f"1000chunks_batch{batch_size}"]["duration_ms"]
            for batch_size in batch_sizes
        )
        assert best_1000_time < 10000, f"Embedding too slow for 1000 chunks: {best_1000_time}ms"
    
    @pytest.mark.asyncio
    async def test_embedding_memory_efficiency(self, mock_high_performance_lens_client):
        """Test memory efficiency of embedding processing."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Large chunk set for memory testing
        chunk_count = 2000
        chunks = generate_mock_chunks(chunk_count)
        
        context = Mock()
        context.vector_chunks = chunks
        
        # Record initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_high_performance_lens_client):
            with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                mock_adapter_instance = AsyncMock()
                mock_adapter_instance.initialize = AsyncMock()
                mock_adapter_instance.embed_code_chunk = AsyncMock(
                    side_effect=lambda chunk: chunk
                )
                mock_adapter.return_value = mock_adapter_instance
                
                stage = HybridCodeEmbeddingStage(
                    enable_lens_vectors=True,
                    batch_size=32,
                    embedding_cache_size=1000  # Limit cache size
                )
                
                await stage.execute(context)
                
                # Record peak memory
                peak_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = peak_memory - initial_memory
                
                logger.info(f"Memory usage: {initial_memory:.2f}MB -> {peak_memory:.2f}MB (+{memory_increase:.2f}MB)")
        
        # Memory increase should be reasonable for the workload
        # Allow up to 500MB increase for 2000 chunks with embeddings
        assert memory_increase < 500, f"Excessive memory usage: {memory_increase}MB"
    
    @pytest.mark.asyncio
    async def test_cache_performance_impact(self, mock_high_performance_lens_client):
        """Test impact of caching on embedding performance."""
        chunk_count = 500
        chunks = generate_mock_chunks(chunk_count)
        
        # Add embeddings to half the chunks (simulate cache hits)
        for i, chunk in enumerate(chunks[:chunk_count // 2]):
            chunk.embedding = [random.random() for _ in range(384)]
        
        context = Mock()
        context.vector_chunks = chunks
        
        with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_high_performance_lens_client):
            with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                mock_adapter_instance = AsyncMock()
                mock_adapter_instance.initialize = AsyncMock()
                mock_adapter_instance.embed_code_chunk = AsyncMock(
                    side_effect=lambda chunk: chunk
                )
                mock_adapter.return_value = mock_adapter_instance
                
                stage = HybridCodeEmbeddingStage(enable_lens_vectors=True)
                
                benchmark = PerformanceBenchmark("embedding_with_cache")
                benchmark.start()
                await stage.execute(context)
                duration = benchmark.stop()
                
                logger.info(f"Embedding with 50% cache hits: {duration:.2f}ms")
        
        # Should complete faster due to cache hits
        # Exact timing depends on mock implementation, but should be reasonable
        assert duration < 5000, f"Embedding with cache too slow: {duration}ms"


class TestParallelProcessorPerformance:
    """Performance tests for parallel processor."""
    
    @pytest.mark.asyncio
    async def test_parallel_processor_scalability(self):
        """Test parallel processor scalability with different concurrency levels."""
        task_count = 100
        concurrency_levels = [1, 2, 4, 8, 16]
        
        async def cpu_bound_task(task_id: int) -> int:
            # Simulate CPU-bound work
            result = 0
            for i in range(1000):
                result += i * task_id
            await asyncio.sleep(0.001)  # Small async operation
            return result
        
        benchmarks = {}
        
        for concurrency in concurrency_levels:
            processor = ParallelProcessor(
                ResourceLimits(max_concurrent_tasks=concurrency)
            )
            
            benchmark = PerformanceBenchmark(f"parallel_{concurrency}_workers")
            
            try:
                await processor.start()
                
                # Submit all tasks
                task_ids = []
                benchmark.start()
                
                for i in range(task_count):
                    task_id = await processor.submit_task(
                        cpu_bound_task, i, task_id=f"task_{i}"
                    )
                    task_ids.append(task_id)
                
                # Wait for completion
                await processor.wait_for_completion(task_ids)
                duration = benchmark.stop()
                
                benchmarks[concurrency] = duration
                throughput = task_count / (duration / 1000)
                logger.info(f"{concurrency} workers: {duration:.2f}ms ({throughput:.1f} tasks/sec)")
                
            finally:
                await processor.stop()
        
        # Analyze scalability
        baseline = benchmarks[1]
        for concurrency in concurrency_levels:
            if concurrency > 1:
                speedup = baseline / benchmarks[concurrency]
                efficiency = speedup / concurrency
                logger.info(f"Concurrency {concurrency}: {speedup:.2f}x speedup, {efficiency:.2f} efficiency")
        
        # Should see some speedup with multiple workers
        assert benchmarks[4] < benchmarks[1] * 0.7, "Insufficient parallelization benefit"
    
    @pytest.mark.asyncio
    async def test_task_queue_performance(self):
        """Test task queue performance under high load."""
        processor = ParallelProcessor(
            ResourceLimits(
                max_concurrent_tasks=4,
                max_queue_size=10000
            )
        )
        
        async def quick_task(value: int) -> int:
            return value * 2
        
        task_count = 5000
        benchmark = PerformanceBenchmark("high_load_queue")
        
        try:
            await processor.start()
            
            benchmark.start()
            
            # Submit many tasks quickly
            task_ids = []
            for i in range(task_count):
                task_id = await processor.submit_task(
                    quick_task, i, task_id=f"queue_test_{i}"
                )
                task_ids.append(task_id)
            
            # Wait for all to complete
            await processor.wait_for_completion(task_ids)
            duration = benchmark.stop()
            
            throughput = task_count / (duration / 1000)
            logger.info(f"High load queue: {duration:.2f}ms ({throughput:.1f} tasks/sec)")
            
            # Should handle high task volumes efficiently
            assert duration < 30000, f"Queue performance too slow: {duration}ms for {task_count} tasks"
            
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_error_handling_performance(self):
        """Test performance impact of error handling and retries."""
        processor = ParallelProcessor()
        
        failure_rate = 0.3  # 30% of tasks fail initially
        
        async def unreliable_task(task_id: int) -> str:
            if random.random() < failure_rate:
                raise Exception(f"Task {task_id} failed")
            await asyncio.sleep(0.001)
            return f"success_{task_id}"
        
        task_count = 200
        benchmark = PerformanceBenchmark("error_handling_performance")
        
        try:
            await processor.start()
            
            benchmark.start()
            
            task_ids = []
            for i in range(task_count):
                task_id = await processor.submit_task(
                    unreliable_task, i,
                    task_id=f"unreliable_{i}",
                    max_retries=2
                )
                task_ids.append(task_id)
            
            results = await processor.wait_for_completion(task_ids)
            duration = benchmark.stop()
            
            # Count successes and failures
            successes = sum(1 for result in results.values() if not str(result).startswith("ERROR"))
            failures = len(results) - successes
            
            logger.info(f"Error handling: {duration:.2f}ms, {successes} successes, {failures} failures")
            
            # Should still complete in reasonable time despite retries
            assert duration < 10000, f"Error handling too slow: {duration}ms"
            
        finally:
            await processor.stop()


class TestResultSynthesizerPerformance:
    """Performance tests for result synthesizer."""
    
    @pytest.mark.asyncio
    async def test_synthesis_performance_scalability(self):
        """Test synthesis performance with different data sizes."""
        synthesizer = ResultSynthesizer()
        
        data_sizes = [100, 500, 1000, 2000]
        benchmarks = {}
        
        for size in data_sizes:
            # Create mock discovery data
            mimir_files = [f"mimir_file_{i}.py" for i in range(size)]
            lens_files = [f"lens_file_{i}.py" for i in range(size)]
            
            # Add some overlap
            overlap_size = size // 3
            for i in range(overlap_size):
                lens_files[i] = mimir_files[i]
            
            mimir_data = {
                'files_discovered': mimir_files,
                'structure_analysis': {'complex': 'analysis_data'},
                'workspaces': ['frontend', 'backend']
            }
            
            lens_data = {
                'files_discovered': lens_files,
                'indexing_performance': {'time_ms': 100 * size}
            }
            
            benchmark = PerformanceBenchmark(f"synthesis_{size}_files")
            benchmark.start()
            
            result = await synthesizer.synthesize_discovery_results(mimir_data, lens_data)
            
            duration = benchmark.stop()
            benchmarks[size] = duration
            
            assert result.success
            logger.info(f"Synthesis {size} files: {duration:.2f}ms")
        
        # Synthesis should scale well (mostly linear processing)
        largest = benchmarks[2000]
        smallest = benchmarks[100]
        ratio = largest / smallest
        
        # Allow for some overhead but should scale reasonably
        assert ratio < 25, f"Poor synthesis scalability: {ratio}x slowdown for 20x data"
    
    @pytest.mark.asyncio
    async def test_embedding_fusion_performance(self):
        """Test performance of embedding fusion operations."""
        synthesizer = ResultSynthesizer()
        
        chunk_counts = [100, 500, 1000]
        benchmarks = {}
        
        for count in chunk_counts:
            # Create chunks with embeddings for both systems
            mimir_chunks = []
            lens_chunks = []
            
            for i in range(count):
                base_chunk_data = {
                    'chunk_id': f'chunk_{i}',
                    'content': f'test content {i}',
                    'file_path': f'file_{i}.py',
                    'start_line': i * 10,
                    'end_line': (i * 10) + 5,
                    'chunk_type': 'code'
                }
                
                mimir_chunk = VectorChunk(**base_chunk_data)
                mimir_chunk.embedding = [random.random() for _ in range(384)]
                mimir_chunk.embedding_model = 'mimir_model'
                mimir_chunks.append(mimir_chunk)
                
                lens_chunk = VectorChunk(**base_chunk_data)
                lens_chunk.embedding = [random.random() for _ in range(384)]
                lens_chunk.embedding_model = 'lens_model'
                lens_chunks.append(lens_chunk)
            
            benchmark = PerformanceBenchmark(f"fusion_{count}_chunks")
            benchmark.start()
            
            result = await synthesizer.synthesize_embedding_results(
                mimir_chunks, lens_chunks
            )
            
            duration = benchmark.stop()
            benchmarks[count] = duration
            
            assert result.success
            logger.info(f"Embedding fusion {count} chunks: {duration:.2f}ms")
        
        # Fusion should be efficient even for large chunk sets
        assert benchmarks[1000] < 5000, f"Embedding fusion too slow: {benchmarks[1000]}ms"


class TestEndToEndPerformance:
    """End-to-end performance tests."""
    
    @pytest.mark.asyncio
    async def test_complete_pipeline_performance_targets(self, mock_high_performance_lens_client):
        """Test that complete pipeline meets performance targets."""
        # Performance targets
        targets = {
            'total_time_ms': 30000,     # 30 seconds max
            'discovery_time_ms': 5000,   # 5 seconds max
            'embedding_time_ms': 20000,  # 20 seconds max
            'bundling_time_ms': 5000     # 5 seconds max
        }
        
        # Create realistic test data
        file_count = 1000
        chunk_count = 2000
        
        context = Mock()
        context.repo_path = Path('/test/repo')
        context.work_dir = Path('/test/work')
        context.files = generate_mock_files(file_count)
        context.vector_chunks = generate_mock_chunks(chunk_count)
        
        metrics_collector = MetricsCollector()
        pipeline_metrics = metrics_collector.start_pipeline_metrics()
        
        total_benchmark = PerformanceBenchmark("complete_pipeline")
        total_benchmark.start()
        
        # Simulate complete pipeline execution
        with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_high_performance_lens_client):
            with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_high_performance_lens_client):
                with patch('src.repoindex.pipeline.hybrid_bundle.get_lens_client', return_value=mock_high_performance_lens_client):
                    
                    # Discovery stage
                    discovery_benchmark = PerformanceBenchmark("discovery_stage")
                    discovery_benchmark.start()
                    
                    with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                        mock_discovery_instance = AsyncMock()
                        mock_discovery_instance.discover_files.return_value = context.files
                        mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                        mock_discovery_instance.get_file_metadata.return_value = {}
                        mock_discovery.return_value = mock_discovery_instance
                        
                        discovery_stage = HybridDiscoveryStage(enable_lens_indexing=True)
                        await discovery_stage.execute(context)
                    
                    discovery_time = discovery_benchmark.stop()
                    pipeline_metrics.discovery_time_ms = discovery_time
                    
                    # Embedding stage
                    embedding_benchmark = PerformanceBenchmark("embedding_stage")
                    embedding_benchmark.start()
                    
                    with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                        mock_adapter_instance = AsyncMock()
                        mock_adapter_instance.initialize = AsyncMock()
                        mock_adapter_instance.embed_code_chunk = AsyncMock(
                            side_effect=lambda chunk: chunk
                        )
                        mock_adapter.return_value = mock_adapter_instance
                        
                        embedding_stage = HybridCodeEmbeddingStage(enable_lens_vectors=True)
                        await embedding_stage.execute(context)
                    
                    embedding_time = embedding_benchmark.stop()
                    pipeline_metrics.embedding_time_ms = embedding_time
                    
                    # Bundle stage
                    bundle_benchmark = PerformanceBenchmark("bundle_stage")
                    bundle_benchmark.start()
                    
                    with patch('src.repoindex.pipeline.hybrid_bundle.BundleCreator') as mock_creator:
                        mock_creator_instance = Mock()
                        mock_creator_instance._create_manifest.return_value = Mock()
                        mock_creator.return_value = mock_creator_instance
                        
                        bundle_stage = HybridBundleStage(enable_lens_export=True)
                        await bundle_stage.execute(context)
                    
                    bundle_time = bundle_benchmark.stop()
                    pipeline_metrics.bundling_time_ms = bundle_time
        
        total_time = total_benchmark.stop()
        pipeline_metrics.files_processed = file_count
        pipeline_metrics.chunks_processed = chunk_count
        
        # Complete metrics
        metrics_collector.finish_pipeline_metrics(pipeline_metrics)
        
        # Log results
        logger.info("=== PERFORMANCE TARGET VALIDATION ===")
        logger.info(f"Total time: {total_time:.2f}ms (target: {targets['total_time_ms']}ms)")
        logger.info(f"Discovery: {discovery_time:.2f}ms (target: {targets['discovery_time_ms']}ms)")
        logger.info(f"Embedding: {embedding_time:.2f}ms (target: {targets['embedding_time_ms']}ms)")
        logger.info(f"Bundling: {bundle_time:.2f}ms (target: {targets['bundling_time_ms']}ms)")
        logger.info(f"Throughput: {file_count / (total_time / 1000):.1f} files/sec")
        logger.info(f"Chunk rate: {chunk_count / (total_time / 1000):.1f} chunks/sec")
        
        # Validate performance targets
        assert total_time <= targets['total_time_ms'], f"Total time exceeded: {total_time}ms > {targets['total_time_ms']}ms"
        assert discovery_time <= targets['discovery_time_ms'], f"Discovery time exceeded: {discovery_time}ms > {targets['discovery_time_ms']}ms"
        assert embedding_time <= targets['embedding_time_ms'], f"Embedding time exceeded: {embedding_time}ms > {targets['embedding_time_ms']}ms"
        assert bundle_time <= targets['bundling_time_ms'], f"Bundle time exceeded: {bundle_time}ms > {targets['bundling_time_ms']}ms"
        
        # Minimum throughput requirements
        min_file_throughput = 30  # files per second
        actual_file_throughput = file_count / (total_time / 1000)
        assert actual_file_throughput >= min_file_throughput, f"Insufficient file throughput: {actual_file_throughput:.1f} < {min_file_throughput} files/sec"
        
        logger.info("✅ All performance targets met!")


if __name__ == '__main__':
    # Run with performance-focused pytest configuration
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '--benchmark-only',  # If pytest-benchmark is available
        '--durations=10'     # Show slowest tests
    ])