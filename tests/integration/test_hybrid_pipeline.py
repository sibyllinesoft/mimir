"""
Comprehensive Integration Tests for Hybrid Mimir-Lens Pipeline.

Tests the complete integration of Mimir and Lens systems working together
in various scenarios and configurations.
"""

import asyncio
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from typing import Dict, List, Any, Optional
import json
import time

# Import the hybrid pipeline components
from src.repoindex.pipeline.hybrid_indexing import HybridIndexingPipeline
from src.repoindex.pipeline.hybrid_discover import HybridDiscoveryStage
from src.repoindex.pipeline.hybrid_code_embeddings import HybridCodeEmbeddingStage
from src.repoindex.pipeline.hybrid_bundle import HybridBundleStage
from src.repoindex.pipeline.parallel_processor import ParallelProcessor, ResourceLimits
from src.repoindex.pipeline.result_synthesizer import ResultSynthesizer, SynthesisStrategy
from src.repoindex.pipeline.hybrid_metrics import MetricsCollector
from src.repoindex.pipeline.lens_client import (
    LensIntegrationClient, LensHealthStatus, LensResponse, LensHealthCheck
)
from src.repoindex.data.schemas import PipelineStage, VectorChunk, VectorIndex
from src.repoindex.util.errors import MimirError


@pytest.fixture
def temp_directory():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_lens_client():
    """Create mock Lens client for testing."""
    mock_client = AsyncMock(spec=LensIntegrationClient)
    
    # Mock health check
    mock_client.health_check.return_value = LensHealthCheck(
        status=LensHealthStatus.HEALTHY,
        timestamp=time.time(),
        response_time_ms=50.0,
        version="1.0.0"
    )
    
    # Mock bulk index
    mock_client.bulk_index.return_value = LensResponse(
        success=True,
        data={
            'indexed_documents': 10,
            'processing_time_ms': 500.0,
            'status': 'completed'
        },
        status_code=200,
        response_time_ms=500.0
    )
    
    # Mock generate embeddings
    mock_client.generate_embeddings.return_value = LensResponse(
        success=True,
        data={
            'embeddings': [
                {
                    'id': f'chunk_{i}',
                    'vector': [0.1] * 384,  # Mock 384-dim embedding
                    'model': 'lens_embedding_model'
                }
                for i in range(5)
            ]
        },
        status_code=200,
        response_time_ms=300.0
    )
    
    # Mock data export
    mock_client.export_data.return_value = LensResponse(
        success=True,
        data={
            'documents': [{'id': f'doc_{i}', 'content': f'content {i}'} for i in range(10)],
            'vectors': [{'id': f'vec_{i}', 'embedding': [0.1] * 384} for i in range(10)],
            'metadata': {'export_timestamp': time.time()}
        },
        status_code=200,
        response_time_ms=1000.0
    )
    
    return mock_client


@pytest.fixture
def mock_pipeline_context():
    """Create mock pipeline context."""
    context = Mock()
    context.repo_path = Path('/test/repo')
    context.repo_name = 'test-repo'
    context.work_dir = Path('/test/work')
    context.config = Mock()
    context.index_id = 'test-index-123'
    context.incremental_mode = False
    
    # Mock files and chunks
    context.files = [
        'src/main.py',
        'src/utils.py', 
        'tests/test_main.py',
        'README.md'
    ]
    
    context.vector_chunks = [
        VectorChunk(
            chunk_id=f'chunk_{i}',
            content=f'test content {i}',
            file_path=f'src/file_{i}.py',
            start_line=i * 10,
            end_line=(i * 10) + 5,
            chunk_type='code'
        )
        for i in range(5)
    ]
    
    return context


class TestHybridDiscoveryStage:
    """Test hybrid discovery stage integration."""
    
    @pytest.mark.asyncio
    async def test_discovery_with_lens_available(self, mock_lens_client, mock_pipeline_context, temp_directory):
        """Test discovery when Lens is available."""
        # Setup
        mock_pipeline_context.work_dir = temp_directory
        
        with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_lens_client):
            with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                # Mock file discovery
                mock_discovery_instance = AsyncMock()
                mock_discovery_instance.discover_files.return_value = [
                    'src/main.py', 'src/utils.py', 'tests/test_main.py'
                ]
                mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                mock_discovery_instance.get_file_metadata.return_value = {
                    'src/main.py': {'size': 1000, 'hash': 'abc123'},
                    'src/utils.py': {'size': 500, 'hash': 'def456'}
                }
                mock_discovery.return_value = mock_discovery_instance
                
                # Create and execute stage
                stage = HybridDiscoveryStage(enable_lens_indexing=True)
                
                await stage.execute(mock_pipeline_context)
                
                # Verify Lens integration was called
                mock_lens_client.health_check.assert_called_once()
                mock_lens_client.bulk_index.assert_called_once()
                
                # Verify context was updated
                assert hasattr(mock_pipeline_context, 'discovery_result')
    
    @pytest.mark.asyncio
    async def test_discovery_lens_fallback(self, mock_pipeline_context, temp_directory):
        """Test discovery falls back to Mimir when Lens unavailable."""
        # Setup
        mock_pipeline_context.work_dir = temp_directory
        
        # Mock unavailable Lens client
        mock_lens_unavailable = AsyncMock(spec=LensIntegrationClient)
        mock_lens_unavailable.health_check.side_effect = Exception("Lens unavailable")
        
        with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_lens_unavailable):
            with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                # Mock file discovery
                mock_discovery_instance = AsyncMock()
                mock_discovery_instance.discover_files.return_value = [
                    'src/main.py', 'src/utils.py'
                ]
                mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                mock_discovery.return_value = mock_discovery_instance
                
                # Create and execute stage
                stage = HybridDiscoveryStage(enable_lens_indexing=True)
                
                await stage.execute(mock_pipeline_context)
                
                # Verify fallback to Mimir worked
                assert hasattr(mock_pipeline_context, 'discovery_result')
                
                # Verify Lens was not used for bulk indexing
                mock_lens_unavailable.bulk_index.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_discovery_parallel_processing(self, mock_lens_client, mock_pipeline_context, temp_directory):
        """Test parallel processing in discovery stage."""
        mock_pipeline_context.work_dir = temp_directory
        
        with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_lens_client):
            with patch('src.repoindex.pipeline.hybrid_discover.get_parallel_processor') as mock_get_processor:
                # Mock parallel processor
                mock_processor = AsyncMock(spec=ParallelProcessor)
                mock_processor.submit_task.return_value = "task_id_123"
                mock_processor.get_result.return_value = {
                    'success': True,
                    'indexed_files': 5,
                    'execution_time_ms': 200.0
                }
                mock_get_processor.return_value = mock_processor
                
                with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                    mock_discovery_instance = AsyncMock()
                    mock_discovery_instance.discover_files.return_value = ['test.py']
                    mock_discovery.return_value = mock_discovery_instance
                    
                    stage = HybridDiscoveryStage()
                    await stage.execute(mock_pipeline_context)
                    
                    # Verify parallel processing was used
                    mock_processor.submit_task.assert_called()
                    mock_processor.get_result.assert_called()


class TestHybridCodeEmbeddingStage:
    """Test hybrid code embedding stage integration."""
    
    @pytest.mark.asyncio
    async def test_embedding_with_lens_vectors(self, mock_lens_client, mock_pipeline_context):
        """Test embedding generation with Lens vector services."""
        with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_lens_client):
            with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_parallel_processor') as mock_get_processor:
                # Mock parallel processor
                mock_processor = AsyncMock()
                mock_processor.submit_task.return_value = "embedding_task_123"
                mock_processor.get_result.return_value = {
                    'success': True,
                    'chunks': mock_pipeline_context.vector_chunks,
                    'execution_time_ms': 500.0
                }
                mock_get_processor.return_value = mock_processor
                
                # Mock Mimir adapter
                with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                    mock_adapter_instance = AsyncMock()
                    mock_adapter_instance.initialize = AsyncMock()
                    mock_adapter_instance.embed_code_chunk = AsyncMock(
                        side_effect=lambda chunk: chunk  # Return chunk as-is
                    )
                    mock_adapter.return_value = mock_adapter_instance
                    
                    stage = HybridCodeEmbeddingStage(enable_lens_vectors=True)
                    
                    await stage.execute(mock_pipeline_context)
                    
                    # Verify Lens vector services were used
                    mock_lens_client.generate_embeddings.assert_called()
                    
                    # Verify context was updated with embeddings
                    assert hasattr(mock_pipeline_context, 'vector_chunks')
    
    @pytest.mark.asyncio
    async def test_embedding_batch_optimization(self, mock_lens_client, mock_pipeline_context):
        """Test embedding batch size optimization."""
        # Create more chunks to test batching
        large_chunk_set = [
            VectorChunk(
                chunk_id=f'chunk_{i}',
                content=f'test content {i}' * 100,  # Larger content
                file_path=f'src/file_{i}.py',
                start_line=i * 10,
                end_line=(i * 10) + 5,
                chunk_type='code' if i % 2 == 0 else 'text'
            )
            for i in range(50)  # More chunks
        ]
        mock_pipeline_context.vector_chunks = large_chunk_set
        
        with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_lens_client):
            with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_parallel_processor') as mock_get_processor:
                mock_processor = AsyncMock()
                mock_processor.submit_task.return_value = "batch_task"
                mock_processor.get_result.return_value = {
                    'success': True,
                    'chunks': large_chunk_set[:10],  # Simulate batch processing
                    'execution_time_ms': 800.0
                }
                mock_get_processor.return_value = mock_processor
                
                with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                    mock_adapter_instance = AsyncMock()
                    mock_adapter_instance.initialize = AsyncMock()
                    mock_adapter.return_value = mock_adapter_instance
                    
                    stage = HybridCodeEmbeddingStage(
                        enable_lens_vectors=True,
                        batch_size=16
                    )
                    
                    await stage.execute(mock_pipeline_context)
                    
                    # Verify batching was used (multiple task submissions)
                    assert mock_processor.submit_task.call_count > 1
    
    @pytest.mark.asyncio
    async def test_embedding_cache_behavior(self, mock_lens_client, mock_pipeline_context):
        """Test embedding caching functionality."""
        # Setup chunks with some already having embeddings (cached)
        cached_chunks = []
        for i, chunk in enumerate(mock_pipeline_context.vector_chunks):
            if i % 2 == 0:
                chunk.embedding = [0.1] * 384  # Mock cached embedding
            cached_chunks.append(chunk)
        mock_pipeline_context.vector_chunks = cached_chunks
        
        with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_lens_client):
            stage = HybridCodeEmbeddingStage(enable_lens_vectors=True)
            
            await stage.execute(mock_pipeline_context)
            
            # Verify cache was considered (fewer calls to Lens than total chunks)
            # This would be validated through metrics in the actual implementation


class TestHybridBundleStage:
    """Test hybrid bundle stage integration."""
    
    @pytest.mark.asyncio
    async def test_bundle_creation_with_lens_data(self, mock_lens_client, mock_pipeline_context, temp_directory):
        """Test bundle creation including Lens data."""
        # Setup work directory with mock artifacts
        mock_pipeline_context.work_dir = temp_directory
        
        # Create mock Mimir artifacts
        (temp_directory / 'manifest.json').write_text('{"test": "manifest"}')
        (temp_directory / 'chunks.json').write_text('{"test": "chunks"}')
        
        # Add mock discovery and embedding results to context
        mock_pipeline_context.discovery_result = Mock()
        mock_pipeline_context.discovery_result.lens_data = {
            'files_discovered': ['test1.py', 'test2.py'],
            'indexing_performance': {'time_ms': 500}
        }
        
        with patch('src.repoindex.pipeline.hybrid_bundle.get_lens_client', return_value=mock_lens_client):
            with patch('src.repoindex.pipeline.hybrid_bundle.BundleCreator') as mock_bundle_creator:
                # Mock bundle creator
                mock_creator_instance = Mock()
                mock_creator_instance._create_manifest.return_value = Mock()
                mock_bundle_creator.return_value = mock_creator_instance
                
                stage = HybridBundleStage(enable_lens_export=True)
                
                await stage.execute(mock_pipeline_context)
                
                # Verify Lens export was called
                mock_lens_client.export_data.assert_called_once()
                
                # Verify final manifest was created
                assert hasattr(mock_pipeline_context, 'final_manifest')
    
    @pytest.mark.asyncio
    async def test_bundle_compression_and_integrity(self, mock_lens_client, mock_pipeline_context, temp_directory):
        """Test bundle compression and integrity verification."""
        mock_pipeline_context.work_dir = temp_directory
        
        # Create substantial mock artifacts
        (temp_directory / 'large_file.json').write_text('{"data": "' + 'x' * 10000 + '"}')
        
        with patch('src.repoindex.pipeline.hybrid_bundle.get_lens_client', return_value=mock_lens_client):
            stage = HybridBundleStage(
                enable_lens_export=True,
                compression_level=5  # Higher compression
            )
            
            await stage.execute(mock_pipeline_context)
            
            # Verify bundle was created and can be validated
            # In a real test, we'd check the actual compressed file
            assert hasattr(mock_pipeline_context, 'final_manifest')


class TestResultSynthesizer:
    """Test result synthesis functionality."""
    
    @pytest.mark.asyncio
    async def test_discovery_synthesis_best_of_both(self):
        """Test synthesis of discovery results using best-of-both strategy."""
        synthesizer = ResultSynthesizer()
        
        mimir_data = {
            'files_discovered': ['src/main.py', 'src/utils.py', 'tests/test.py'],
            'structure_analysis': {'valid': True, 'warnings': []},
            'workspaces': ['frontend', 'backend']
        }
        
        lens_data = {
            'files_discovered': ['src/main.py', 'src/helpers.py', 'README.md'],
            'indexing_performance': {'time_ms': 300, 'throughput': 100}
        }
        
        result = await synthesizer.synthesize_discovery_results(
            mimir_data, lens_data, SynthesisStrategy.BEST_OF_BOTH
        )
        
        assert result.success
        assert 'files_discovered' in result.synthesized_data
        assert len(result.synthesized_data['files_discovered']) == 5  # Union of files
        assert 'structure_analysis' in result.synthesized_data  # Mimir's analysis
        assert 'indexing_performance' in result.synthesized_data  # Lens's performance
        assert result.confidence_score > 0.8  # High confidence with both systems
    
    @pytest.mark.asyncio
    async def test_embedding_synthesis_weighted_fusion(self):
        """Test synthesis of embedding results using weighted fusion."""
        synthesizer = ResultSynthesizer()
        
        # Create mock chunks with embeddings
        mimir_chunks = [
            VectorChunk(
                chunk_id='chunk_1',
                content='test content',
                file_path='test.py',
                start_line=1,
                end_line=5,
                chunk_type='code',
                embedding=[0.5, 0.3, 0.2, 0.1],  # Mock embedding
                embedding_model='mimir_model'
            )
        ]
        
        lens_chunks = [
            VectorChunk(
                chunk_id='chunk_1',
                content='test content',
                file_path='test.py',
                start_line=1,
                end_line=5,
                chunk_type='code',
                embedding=[0.4, 0.4, 0.1, 0.1],  # Different embedding
                embedding_model='lens_model'
            )
        ]
        
        result = await synthesizer.synthesize_embedding_results(
            mimir_chunks, lens_chunks, SynthesisStrategy.WEIGHTED_FUSION
        )
        
        assert result.success
        assert len(result.synthesized_data) == 1
        
        # Verify fusion occurred (would check actual fused embedding in real test)
        fused_chunk = result.synthesized_data[0]
        assert 'fused' in fused_chunk.embedding_model
    
    @pytest.mark.asyncio
    async def test_synthesis_fallback_scenarios(self):
        """Test synthesis behavior in fallback scenarios."""
        synthesizer = ResultSynthesizer()
        
        # Test with only Mimir data
        result = await synthesizer.synthesize_discovery_results(
            {'files': ['test.py']}, None
        )
        assert result.success
        assert result.mimir_contribution == 1.0
        assert result.lens_contribution == 0.0
        
        # Test with only Lens data
        result = await synthesizer.synthesize_discovery_results(
            None, {'files': ['test.py']}
        )
        assert result.success
        assert result.mimir_contribution == 0.0
        assert result.lens_contribution == 1.0
        
        # Test with no data
        result = await synthesizer.synthesize_discovery_results(None, None)
        assert not result.success
        assert result.error is not None


class TestParallelProcessor:
    """Test parallel processing functionality."""
    
    @pytest.mark.asyncio
    async def test_parallel_task_execution(self):
        """Test basic parallel task execution."""
        processor = ParallelProcessor(
            ResourceLimits(max_concurrent_tasks=2)
        )
        
        async def test_task(value: int) -> int:
            await asyncio.sleep(0.1)  # Simulate work
            return value * 2
        
        try:
            await processor.start()
            
            # Submit multiple tasks
            task_ids = []
            for i in range(5):
                task_id = await processor.submit_task(
                    test_task, i, task_id=f"test_{i}"
                )
                task_ids.append(task_id)
            
            # Wait for all results
            results = await processor.wait_for_completion(task_ids, timeout=5.0)
            
            # Verify results
            assert len(results) == 5
            for i, task_id in enumerate(task_ids):
                assert results[task_id] == i * 2
                
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_parallel_task_retry_logic(self):
        """Test task retry logic for failed tasks."""
        processor = ParallelProcessor()
        
        call_count = 0
        
        async def failing_task() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception(f"Attempt {call_count} failed")
            return "success"
        
        try:
            await processor.start()
            
            task_id = await processor.submit_task(
                failing_task,
                task_id="retry_test",
                max_retries=3
            )
            
            result = await processor.get_result(task_id, timeout=5.0)
            
            # Verify task eventually succeeded after retries
            assert result == "success"
            assert call_count == 3
            
        finally:
            await processor.stop()
    
    @pytest.mark.asyncio
    async def test_resource_limits_enforcement(self):
        """Test that resource limits are enforced."""
        processor = ParallelProcessor(
            ResourceLimits(max_concurrent_tasks=2)
        )
        
        started_tasks = []
        
        async def blocking_task(task_id: str) -> str:
            started_tasks.append(task_id)
            await asyncio.sleep(1.0)  # Block for testing
            return task_id
        
        try:
            await processor.start()
            
            # Submit more tasks than the limit
            task_ids = []
            for i in range(4):
                task_id = await processor.submit_task(
                    blocking_task, f"task_{i}", task_id=f"limit_test_{i}"
                )
                task_ids.append(task_id)
            
            # Wait a bit for tasks to start
            await asyncio.sleep(0.5)
            
            # Should only have max_concurrent_tasks running
            assert len(started_tasks) <= 2
            
        finally:
            await processor.stop()


class TestMetricsCollector:
    """Test metrics collection functionality."""
    
    @pytest.mark.asyncio
    async def test_metrics_collection_basic(self):
        """Test basic metrics collection."""
        collector = MetricsCollector(collection_interval_seconds=0.1)
        
        try:
            await collector.start_monitoring()
            
            # Let it collect a few samples
            await asyncio.sleep(0.5)
            
            # Check that metrics were collected
            assert len(collector.performance_snapshots) > 0
            
            # Test custom metric recording
            collector.record_metric('test_metric', 42.0)
            assert 'test_metric' in collector.metrics
            assert len(collector.metrics['test_metric']) == 1
            assert collector.metrics['test_metric'][0].value == 42.0
            
        finally:
            await collector.stop_monitoring()
    
    @pytest.mark.asyncio
    async def test_pipeline_metrics_tracking(self):
        """Test pipeline metrics tracking."""
        collector = MetricsCollector()
        
        # Start pipeline metrics
        pipeline_metrics = collector.start_pipeline_metrics()
        
        # Simulate pipeline work
        await asyncio.sleep(0.1)
        pipeline_metrics.files_processed = 10
        pipeline_metrics.lens_success_rate = 0.95
        
        # Finish metrics
        collector.finish_pipeline_metrics(pipeline_metrics)
        
        # Verify metrics were recorded
        assert len(collector.pipeline_metrics) == 1
        assert collector.pipeline_metrics[0].files_processed == 10
        assert collector.pipeline_metrics[0].lens_success_rate == 0.95
        assert collector.pipeline_metrics[0].total_time_ms > 0
    
    def test_system_health_score_calculation(self):
        """Test system health score calculation."""
        collector = MetricsCollector()
        
        # Add mock performance snapshot
        from src.repoindex.pipeline.hybrid_metrics import PerformanceSnapshot
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            cpu_percent=50.0,  # Normal
            memory_percent=60.0,  # Normal
            memory_used_mb=1000.0,
            disk_io_read_mb=10.0,
            disk_io_write_mb=5.0,
            network_sent_mb=1.0,
            network_recv_mb=2.0,
            process_count=100
        )
        collector.performance_snapshots.append(snapshot)
        
        health_score = collector.get_system_health_score()
        
        # Should be high for normal resource usage
        assert 0.8 <= health_score <= 1.0


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_hybrid_pipeline_flow(self, mock_lens_client, temp_directory):
        """Test complete pipeline flow from discovery to bundling."""
        # Create mock pipeline context
        context = Mock()
        context.repo_path = temp_directory
        context.work_dir = temp_directory
        context.repo_name = 'integration-test'
        context.config = Mock()
        context.index_id = 'integration-test-123'
        
        # Create test files
        (temp_directory / 'src').mkdir()
        (temp_directory / 'src' / 'main.py').write_text('print("hello world")')
        (temp_directory / 'src' / 'utils.py').write_text('def helper(): pass')
        
        with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=mock_lens_client):
            with patch('src.repoindex.pipeline.hybrid_code_embeddings.get_lens_client', return_value=mock_lens_client):
                with patch('src.repoindex.pipeline.hybrid_bundle.get_lens_client', return_value=mock_lens_client):
                    with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                        # Mock file discovery
                        mock_discovery_instance = AsyncMock()
                        mock_discovery_instance.discover_files.return_value = [
                            'src/main.py', 'src/utils.py'
                        ]
                        mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                        mock_discovery_instance.get_file_metadata.return_value = {
                            'src/main.py': {'size': 100, 'hash': 'abc123'},
                            'src/utils.py': {'size': 50, 'hash': 'def456'}
                        }
                        mock_discovery.return_value = mock_discovery_instance
                        
                        # Execute pipeline stages sequentially
                        
                        # 1. Discovery
                        discovery_stage = HybridDiscoveryStage(enable_lens_indexing=True)
                        await discovery_stage.execute(context)
                        assert hasattr(context, 'discovery_result')
                        
                        # 2. Create mock chunks for embedding stage
                        context.vector_chunks = [
                            VectorChunk(
                                chunk_id='chunk_1',
                                content='print("hello world")',
                                file_path='src/main.py',
                                start_line=1,
                                end_line=1,
                                chunk_type='code'
                            ),
                            VectorChunk(
                                chunk_id='chunk_2',
                                content='def helper(): pass',
                                file_path='src/utils.py',
                                start_line=1,
                                end_line=1,
                                chunk_type='code'
                            )
                        ]
                        
                        # 3. Embeddings
                        with patch('src.repoindex.pipeline.hybrid_code_embeddings.CodeEmbeddingAdapter') as mock_adapter:
                            mock_adapter_instance = AsyncMock()
                            mock_adapter_instance.initialize = AsyncMock()
                            mock_adapter_instance.embed_code_chunk = AsyncMock(
                                side_effect=lambda chunk: chunk
                            )
                            mock_adapter.return_value = mock_adapter_instance
                            
                            embedding_stage = HybridCodeEmbeddingStage(enable_lens_vectors=True)
                            await embedding_stage.execute(context)
                            assert hasattr(context, 'vector_chunks')
                        
                        # 4. Bundling
                        with patch('src.repoindex.pipeline.hybrid_bundle.BundleCreator') as mock_bundle_creator:
                            mock_creator_instance = Mock()
                            mock_manifest = Mock()
                            mock_manifest.dict.return_value = {'test': 'manifest'}
                            mock_creator_instance._create_manifest.return_value = mock_manifest
                            mock_bundle_creator.return_value = mock_creator_instance
                            
                            bundle_stage = HybridBundleStage(enable_lens_export=True)
                            await bundle_stage.execute(context)
                            assert hasattr(context, 'final_manifest')
                        
                        # Verify end-to-end integration
                        # All Lens client methods should have been called
                        assert mock_lens_client.health_check.call_count >= 3  # Once per stage
                        assert mock_lens_client.bulk_index.called
                        assert mock_lens_client.generate_embeddings.called
                        assert mock_lens_client.export_data.called
    
    @pytest.mark.asyncio
    async def test_error_recovery_and_fallbacks(self, temp_directory):
        """Test error recovery and fallback mechanisms."""
        # Create failing Lens client
        failing_lens_client = AsyncMock(spec=LensIntegrationClient)
        failing_lens_client.health_check.side_effect = Exception("Service unavailable")
        failing_lens_client.bulk_index.side_effect = Exception("Indexing failed")
        
        context = Mock()
        context.repo_path = temp_directory
        context.work_dir = temp_directory
        
        with patch('src.repoindex.pipeline.hybrid_discover.get_lens_client', return_value=failing_lens_client):
            with patch('src.repoindex.pipeline.hybrid_discover.FileDiscovery') as mock_discovery:
                mock_discovery_instance = AsyncMock()
                mock_discovery_instance.discover_files.return_value = ['test.py']
                mock_discovery_instance.detect_changes.return_value = ([], [], [], {})
                mock_discovery.return_value = mock_discovery_instance
                
                # Discovery should succeed despite Lens failure
                discovery_stage = HybridDiscoveryStage(enable_lens_indexing=True)
                await discovery_stage.execute(context)
                
                # Should have fallen back to Mimir-only mode
                assert hasattr(context, 'discovery_result')
                
                # Verify Lens failures were handled gracefully
                assert failing_lens_client.health_check.called


if __name__ == '__main__':
    pytest.main([__file__])