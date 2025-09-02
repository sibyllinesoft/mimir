#!/usr/bin/env python3
"""
Comprehensive tests for pipeline stages.

Tests all concrete pipeline stage implementations including acquire, repomapper,
serena, leann, snippets, and bundle stages.
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from dataclasses import dataclass

import pytest

# Import the modules we're testing
from repoindex.pipeline.stages import (
    AcquireStage,
    RepoMapperStage, 
    SerenaStage,
    LeannStage,
    SnippetsStage,
    BundleStage
)
from repoindex.data.schemas import PipelineStage, IndexManifest
from repoindex.util.errors import (
    ValidationError,
    FileSystemError, 
    ExternalToolError,
    IntegrationError,
    MimirError
)


# Mock context class for testing
@dataclass
class MockPipelineContext:
    """Mock pipeline context for testing."""
    repo_info: Mock
    config: Mock
    work_dir: Path
    logger: Mock
    tracked_files: list = None
    repomap_data: Mock = None
    serena_graph: Mock = None
    vector_index: Mock = None
    snippets: Mock = None
    manifest: Mock = None
    index_id: str = "test_index"


class TestAcquireStage:
    """Test AcquireStage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = AcquireStage()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context
        self.mock_context = MockPipelineContext(
            repo_info=Mock(),
            config=Mock(),
            work_dir=self.temp_dir,
            logger=Mock()
        )
        self.mock_context.repo_info.root = self.temp_dir
        self.mock_context.config.languages = [".py", ".js", ".ts"]
        self.mock_context.config.excludes = ["node_modules", "__pycache__"]
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_acquire_stage_initialization(self):
        """Test AcquireStage initialization."""
        stage = AcquireStage()
        
        assert stage.stage_type == PipelineStage.ACQUIRE
        assert stage.concurrency_limit == 16
        assert "file_discovery" in stage._get_capabilities()
        assert "pattern_matching" in stage._get_capabilities()
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful file discovery and acquisition."""
        mock_files = ["file1.py", "file2.js", "file3.ts"]
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.FileDiscovery') as mock_discovery_class:
                mock_discovery = Mock()
                mock_discovery.discover_files = AsyncMock(return_value=mock_files)
                mock_discovery_class.return_value = mock_discovery
                
                await self.stage.execute(self.mock_context, progress_callback)
        
        assert self.mock_context.tracked_files == mock_files
        assert len(progress_calls) == 1
        assert progress_calls[0] == 100
        self.mock_context.logger.log_stage_start.assert_called_once()
        self.mock_context.logger.log_info.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_no_files_found(self):
        """Test execution when no files are found."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.FileDiscovery') as mock_discovery_class:
                mock_discovery = Mock()
                mock_discovery.discover_files = AsyncMock(return_value=[])
                mock_discovery_class.return_value = mock_discovery
                
                with pytest.raises(ValidationError) as exc_info:
                    await self.stage.execute(self.mock_context)
        
        assert "No trackable files found" in str(exc_info.value)
        self.mock_context.logger.log_stage_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_discovery_exception(self):
        """Test execution with discovery exception."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.FileDiscovery') as mock_discovery_class:
                mock_discovery = Mock()
                mock_discovery.discover_files = AsyncMock(side_effect=Exception("Discovery failed"))
                mock_discovery_class.return_value = mock_discovery
                
                with pytest.raises(FileSystemError):
                    await self.stage.execute(self.mock_context)
        
        self.mock_context.logger.log_stage_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_mimir_error_passthrough(self):
        """Test execution passes through MimirError without wrapping."""
        original_error = ValidationError("Original validation error")
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.FileDiscovery') as mock_discovery_class:
                mock_discovery = Mock()
                mock_discovery.discover_files = AsyncMock(side_effect=original_error)
                mock_discovery_class.return_value = mock_discovery
                
                with pytest.raises(ValidationError) as exc_info:
                    await self.stage.execute(self.mock_context)
        
        assert exc_info.value is original_error


class TestRepoMapperStage:
    """Test RepoMapperStage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = RepoMapperStage()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context
        self.mock_context = MockPipelineContext(
            repo_info=Mock(),
            config=Mock(),
            work_dir=self.temp_dir,
            logger=Mock(),
            tracked_files=["file1.py", "file2.py"]
        )
        self.mock_context.repo_info.root = self.temp_dir
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_repomapper_stage_initialization(self):
        """Test RepoMapperStage initialization."""
        stage = RepoMapperStage()
        
        assert stage.stage_type == PipelineStage.REPOMAPPER
        assert stage.concurrency_limit == 4
        assert "code_analysis" in stage._get_capabilities()
        assert "dependency_mapping" in stage._get_capabilities()
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful repository analysis."""
        mock_repomap_data = Mock()
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.RepoMapperAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.analyze_repository = AsyncMock(return_value=mock_repomap_data)
                mock_adapter_class.return_value = mock_adapter
                
                await self.stage.execute(self.mock_context, progress_callback)
        
        assert self.mock_context.repomap_data == mock_repomap_data
        self.mock_context.logger.log_stage_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        """Test execution when analysis produces no results."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.RepoMapperAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.analyze_repository = AsyncMock(return_value=None)
                mock_adapter_class.return_value = mock_adapter
                
                with pytest.raises(ExternalToolError) as exc_info:
                    await self.stage.execute(self.mock_context)
        
        assert "Repository analysis produced no results" in str(exc_info.value)
        assert exc_info.value.tool == "repomapper"
    
    @pytest.mark.asyncio
    async def test_execute_analysis_exception(self):
        """Test execution with analysis exception."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.RepoMapperAdapter') as mock_adapter_class:
                mock_adapter = Mock()
                mock_adapter.analyze_repository = AsyncMock(side_effect=Exception("Analysis failed"))
                mock_adapter_class.return_value = mock_adapter
                
                with patch('repoindex.pipeline.stages.handle_external_tool_error') as mock_handle_error:
                    mock_error = ExternalToolError("repomapper", "Handled error")
                    mock_handle_error.return_value = mock_error
                    
                    with pytest.raises(ExternalToolError):
                        await self.stage.execute(self.mock_context)
        
        self.mock_context.logger.log_stage_error.assert_called_once()


class TestSerenaStage:
    """Test SerenaStage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = SerenaStage()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context
        self.mock_context = MockPipelineContext(
            repo_info=Mock(),
            config=Mock(),
            work_dir=self.temp_dir,
            logger=Mock(),
            tracked_files=["file1.ts", "file2.js"]
        )
        self.mock_context.repo_info.root = self.temp_dir
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_serena_stage_initialization(self):
        """Test SerenaStage initialization."""
        stage = SerenaStage()
        
        assert stage.stage_type == PipelineStage.SERENA
        assert stage.concurrency_limit == 4
        assert "symbol_analysis" in stage._get_capabilities()
        assert "typescript_support" in stage._get_capabilities()
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful symbol analysis."""
        mock_serena_graph = Mock()
        mock_serena_graph.symbol_count = 42
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.get_metrics_collector') as mock_get_metrics:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                with patch('repoindex.pipeline.stages.SerenaAdapter') as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.analyze_project = AsyncMock(return_value=mock_serena_graph)
                    mock_adapter_class.return_value = mock_adapter
                    
                    await self.stage.execute(self.mock_context)
        
        assert self.mock_context.serena_graph == mock_serena_graph
        mock_metrics.record_symbols_extracted.assert_called_once()
        self.mock_context.logger.log_stage_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        """Test execution when analysis produces no results."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.get_metrics_collector') as mock_get_metrics:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                with patch('repoindex.pipeline.stages.SerenaAdapter') as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.analyze_project = AsyncMock(return_value=None)
                    mock_adapter_class.return_value = mock_adapter
                    
                    with pytest.raises(ExternalToolError) as exc_info:
                        await self.stage.execute(self.mock_context)
        
        assert "Symbol analysis produced no results" in str(exc_info.value)
        assert exc_info.value.tool == "serena"
    
    @pytest.mark.asyncio
    async def test_execute_no_symbols_found(self):
        """Test execution when no symbols are found."""
        mock_serena_graph = Mock()
        mock_serena_graph.symbol_count = 0  # No symbols
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.get_metrics_collector') as mock_get_metrics:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                with patch('repoindex.pipeline.stages.SerenaAdapter') as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.analyze_project = AsyncMock(return_value=mock_serena_graph)
                    mock_adapter_class.return_value = mock_adapter
                    
                    await self.stage.execute(self.mock_context)
        
        # Should not call metrics recording when no symbols found
        mock_metrics.record_symbols_extracted.assert_not_called()


class TestLeannStage:
    """Test LeannStage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = LeannStage()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context
        self.mock_context = MockPipelineContext(
            repo_info=Mock(),
            config=Mock(),
            work_dir=self.temp_dir,
            logger=Mock(),
            tracked_files=["file1.py", "file2.py"],
            repomap_data=Mock()
        )
        self.mock_context.repo_info.root = self.temp_dir
        self.mock_context.config.features = Mock()
        self.mock_context.config.features.vector = True
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_leann_stage_initialization(self):
        """Test LeannStage initialization."""
        stage = LeannStage()
        
        assert stage.stage_type == PipelineStage.LEANN
        assert stage.concurrency_limit == 4
        assert "vector_embedding" in stage._get_capabilities()
        assert "semantic_search" in stage._get_capabilities()
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful vector embedding generation."""
        mock_vector_index = Mock()
        mock_vector_index.total_chunks = 100
        mock_vector_index.dimension = 768
        
        mock_ordered_files = ["ordered_file1.py", "ordered_file2.py"]
        self.mock_context.repomap_data.get_ordered_files.return_value = mock_ordered_files
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.get_metrics_collector') as mock_get_metrics:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                with patch('repoindex.pipeline.stages.LEANNAdapter') as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.build_index = AsyncMock(return_value=mock_vector_index)
                    mock_adapter_class.return_value = mock_adapter
                    
                    await self.stage.execute(self.mock_context)
        
        assert self.mock_context.vector_index == mock_vector_index
        mock_metrics.record_embeddings_created.assert_called_once_with("leann", "code_chunk", 100)
        self.mock_context.logger.log_stage_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_vector_disabled(self):
        """Test execution when vector search is disabled."""
        self.mock_context.config.features.vector = False
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            await self.stage.execute(self.mock_context, progress_callback)
        
        assert len(progress_calls) == 1
        assert progress_calls[0] == 100
        self.mock_context.logger.log_info.assert_called_with(
            "Vector search disabled, skipping LEANN", stage=PipelineStage.LEANN
        )
    
    @pytest.mark.asyncio
    async def test_execute_ordered_files_fallback(self):
        """Test execution with ordered files fallback."""
        mock_vector_index = Mock()
        mock_vector_index.total_chunks = 50
        
        # Make get_ordered_files raise exception
        self.mock_context.repomap_data.get_ordered_files.side_effect = Exception("Failed to get ordered")
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.get_metrics_collector') as mock_get_metrics:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                with patch('repoindex.pipeline.stages.LEANNAdapter') as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.build_index = AsyncMock(return_value=mock_vector_index)
                    mock_adapter_class.return_value = mock_adapter
                    
                    await self.stage.execute(self.mock_context)
        
        # Should warn about fallback and use original tracked_files
        mock_logger.warning.assert_called_once()
        assert self.mock_context.vector_index == mock_vector_index
    
    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        """Test execution when embedding produces no results."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.get_metrics_collector') as mock_get_metrics:
                mock_metrics = Mock()
                mock_get_metrics.return_value = mock_metrics
                
                with patch('repoindex.pipeline.stages.LEANNAdapter') as mock_adapter_class:
                    mock_adapter = Mock()
                    mock_adapter.build_index = AsyncMock(return_value=None)
                    mock_adapter_class.return_value = mock_adapter
                    
                    with pytest.raises(ExternalToolError) as exc_info:
                        await self.stage.execute(self.mock_context)
        
        assert "Vector embedding produced no results" in str(exc_info.value)
        assert exc_info.value.tool == "leann"


class TestSnippetsStage:
    """Test SnippetsStage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = SnippetsStage()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context
        self.mock_context = MockPipelineContext(
            repo_info=Mock(),
            config=Mock(),
            work_dir=self.temp_dir,
            logger=Mock(),
            tracked_files=["file1.py", "file2.py"],
            serena_graph=Mock()
        )
        self.mock_context.repo_info.root = self.temp_dir
        self.mock_context.config.context_lines = 3
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_snippets_stage_initialization(self):
        """Test SnippetsStage initialization."""
        stage = SnippetsStage()
        
        assert stage.stage_type == PipelineStage.SNIPPETS
        assert stage.concurrency_limit == 16
        assert "snippet_extraction" in stage._get_capabilities()
        assert "context_preservation" in stage._get_capabilities()
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful snippet extraction."""
        mock_snippets = Mock()
        mock_snippets.count = 25
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.SnippetExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor.extract_snippets = AsyncMock(return_value=mock_snippets)
                mock_extractor_class.return_value = mock_extractor
                
                await self.stage.execute(self.mock_context)
        
        assert self.mock_context.snippets == mock_snippets
        self.mock_context.logger.log_stage_start.assert_called_once()
        mock_logger.operation_success.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_no_results(self):
        """Test execution when extraction produces no results."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.SnippetExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor.extract_snippets = AsyncMock(return_value=None)
                mock_extractor_class.return_value = mock_extractor
                
                with pytest.raises(IntegrationError) as exc_info:
                    await self.stage.execute(self.mock_context)
        
        assert "Snippet extraction produced no results" in str(exc_info.value)
        assert exc_info.value.source == "snippet_extractor"
        assert exc_info.value.target == "serena_graph"
    
    @pytest.mark.asyncio
    async def test_execute_extraction_exception(self):
        """Test execution with extraction exception."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.SnippetExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor.extract_snippets = AsyncMock(side_effect=Exception("Extraction failed"))
                mock_extractor_class.return_value = mock_extractor
                
                with pytest.raises(MimirError):
                    await self.stage.execute(self.mock_context)
        
        self.mock_context.logger.log_stage_error.assert_called_once()


class TestBundleStage:
    """Test BundleStage class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.stage = BundleStage()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context with all required data
        self.mock_context = MockPipelineContext(
            repo_info=Mock(),
            config=Mock(),
            work_dir=self.temp_dir,
            logger=Mock(),
            tracked_files=["file1.py", "file2.py"],
            repomap_data=Mock(),
            serena_graph=Mock(),
            vector_index=Mock(),
            snippets=Mock()
        )
        self.mock_context.repo_info.root = self.temp_dir
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_bundle_stage_initialization(self):
        """Test BundleStage initialization."""
        stage = BundleStage()
        
        assert stage.stage_type == PipelineStage.BUNDLE
        assert stage.concurrency_limit == 16
        assert "bundle_creation" in stage._get_capabilities()
        assert "artifact_packaging" in stage._get_capabilities()
    
    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful bundle creation."""
        mock_manifest = Mock(spec=IndexManifest)
        mock_manifest.dict.return_value = {"manifest": "data"}
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.BundleCreator') as mock_bundler_class:
                mock_bundler = Mock()
                mock_bundler.create_bundle = AsyncMock(return_value=mock_manifest)
                mock_bundler_class.return_value = mock_bundler
                
                with patch('repoindex.pipeline.stages.atomic_write_json') as mock_write:
                    await self.stage.execute(self.mock_context)
        
        assert self.mock_context.manifest == mock_manifest
        mock_write.assert_called_once()
        self.mock_context.logger.log_stage_start.assert_called_once()
        mock_logger.operation_success.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_no_manifest(self):
        """Test execution when bundle creation produces no manifest."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.BundleCreator') as mock_bundler_class:
                mock_bundler = Mock()
                mock_bundler.create_bundle = AsyncMock(return_value=None)
                mock_bundler_class.return_value = mock_bundler
                
                with pytest.raises(IntegrationError) as exc_info:
                    await self.stage.execute(self.mock_context)
        
        assert "Bundle creation produced no manifest" in str(exc_info.value)
        assert exc_info.value.source == "bundle_creator"
        assert exc_info.value.target == "pipeline_context"
    
    @pytest.mark.asyncio
    async def test_execute_manifest_write_failure(self):
        """Test execution with manifest write failure."""
        mock_manifest = Mock(spec=IndexManifest)
        mock_manifest.dict.return_value = {"manifest": "data"}
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.BundleCreator') as mock_bundler_class:
                mock_bundler = Mock()
                mock_bundler.create_bundle = AsyncMock(return_value=mock_manifest)
                mock_bundler_class.return_value = mock_bundler
                
                with patch('repoindex.pipeline.stages.atomic_write_json') as mock_write:
                    mock_write.side_effect = Exception("Write failed")
                    
                    with pytest.raises(FileSystemError) as exc_info:
                        await self.stage.execute(self.mock_context)
        
        assert "Failed to save manifest" in str(exc_info.value)
        assert exc_info.value.operation == "write_manifest"
    
    @pytest.mark.asyncio
    async def test_execute_bundle_creation_exception(self):
        """Test execution with bundle creation exception."""
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.BundleCreator') as mock_bundler_class:
                mock_bundler = Mock()
                mock_bundler.create_bundle = AsyncMock(side_effect=Exception("Bundle failed"))
                mock_bundler_class.return_value = mock_bundler
                
                with pytest.raises(MimirError):
                    await self.stage.execute(self.mock_context)
        
        self.mock_context.logger.log_stage_error.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_mimir_error_passthrough(self):
        """Test execution passes through MimirError without wrapping."""
        original_error = IntegrationError("bundle_creator", "target", "Original error")
        
        with patch('repoindex.pipeline.stages.get_logger') as mock_get_logger:
            mock_logger = Mock()
            mock_logger.performance_track.return_value.__enter__ = Mock()
            mock_logger.performance_track.return_value.__exit__ = Mock()
            mock_get_logger.return_value = mock_logger
            
            with patch('repoindex.pipeline.stages.BundleCreator') as mock_bundler_class:
                mock_bundler = Mock()
                mock_bundler.create_bundle = AsyncMock(side_effect=original_error)
                mock_bundler_class.return_value = mock_bundler
                
                with pytest.raises(IntegrationError) as exc_info:
                    await self.stage.execute(self.mock_context)
        
        assert exc_info.value is original_error


def run_tests():
    """Run all tests when script is executed directly."""
    import subprocess
    import sys
    
    # Run pytest with verbose output
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)