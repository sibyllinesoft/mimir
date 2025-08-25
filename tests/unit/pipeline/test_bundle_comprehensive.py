#!/usr/bin/env python3
"""
Comprehensive tests for BundleCreator.

Tests bundle creation, compression, validation, and extraction for pipeline artifacts.
"""

import asyncio
import tarfile
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, mock_open
from datetime import datetime

import pytest
import zstandard as zstd

# Import the modules we're testing
from repoindex.pipeline.bundle import BundleCreator, BundleError
from repoindex.data.schemas import ArtifactPaths, IndexCounts, IndexManifest, ToolVersions


class TestBundleCreator:
    """Test BundleCreator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.creator = BundleCreator()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # Create mock context
        self.mock_context = Mock()
        self.mock_context.work_dir = self.temp_dir
        self.mock_context.index_id = "test_index"
        self.mock_context.repo_info = {"name": "test_repo"}
        self.mock_context.config = {"version": "1.0"}
        self.mock_context.tracked_files = ["file1.py", "file2.py"]
        
        # Create mock serena graph
        mock_entry1 = Mock()
        mock_entry1.type.value = "def"
        mock_entry2 = Mock()
        mock_entry2.type.value = "ref"
        mock_graph = Mock()
        mock_graph.entries = [mock_entry1, mock_entry2]
        self.mock_context.serena_graph = mock_graph
        
        # Create mock vector index
        mock_vector_index = Mock()
        mock_vector_index.chunks = ["chunk1", "chunk2", "chunk3"]
        self.mock_context.vector_index = mock_vector_index
    
    def teardown_method(self):
        """Clean up after each test."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_bundle_creator_initialization(self):
        """Test BundleCreator initialization."""
        creator = BundleCreator()
        
        assert creator.max_bundle_size == 2 * 1024 * 1024 * 1024  # 2GB
        assert creator.compression_level == 3
    
    @pytest.mark.asyncio
    async def test_create_bundle_success(self):
        """Test successful bundle creation."""
        # Create required artifact files
        (self.temp_dir / "repomap.json").touch()
        (self.temp_dir / "serena_graph.jsonl").touch()
        (self.temp_dir / "snippets.jsonl").touch()
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        with patch.object(self.creator, '_create_manifest') as mock_create_manifest:
            mock_manifest = Mock(spec=IndexManifest)
            mock_manifest.paths = ArtifactPaths()
            mock_manifest.updated_at = datetime.utcnow()
            mock_create_manifest.return_value = mock_manifest
            
            with patch.object(self.creator, '_validate_artifacts') as mock_validate:
                with patch.object(self.creator, '_create_compressed_bundle') as mock_create_bundle:
                    mock_bundle_path = self.temp_dir / "bundle.tar.zst"
                    mock_create_bundle.return_value = mock_bundle_path
                    
                    result = await self.creator.create_bundle(self.mock_context, progress_callback)
        
        assert result == mock_manifest
        assert mock_manifest.paths.bundle == "bundle.tar.zst"
        assert len(progress_calls) == 5
        assert progress_calls == [10, 30, 50, 90, 100]
        mock_create_manifest.assert_called_once_with(self.mock_context)
        mock_validate.assert_called_once()
        mock_create_bundle.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_bundle_no_progress_callback(self):
        """Test bundle creation without progress callback."""
        # Create required artifact files
        (self.temp_dir / "repomap.json").touch()
        (self.temp_dir / "serena_graph.jsonl").touch()
        (self.temp_dir / "snippets.jsonl").touch()
        
        with patch.object(self.creator, '_create_manifest') as mock_create_manifest:
            mock_manifest = Mock(spec=IndexManifest)
            mock_manifest.paths = ArtifactPaths()
            mock_manifest.updated_at = datetime.utcnow()
            mock_create_manifest.return_value = mock_manifest
            
            with patch.object(self.creator, '_validate_artifacts'):
                with patch.object(self.creator, '_create_compressed_bundle') as mock_create_bundle:
                    mock_bundle_path = self.temp_dir / "bundle.tar.zst"
                    mock_create_bundle.return_value = mock_bundle_path
                    
                    result = await self.creator.create_bundle(self.mock_context)
        
        assert result == mock_manifest
    
    @pytest.mark.asyncio
    async def test_create_manifest(self):
        """Test manifest creation."""
        with patch.object(self.creator, '_calculate_counts') as mock_calc_counts:
            mock_counts = IndexCounts()
            mock_calc_counts.return_value = mock_counts
            
            with patch.object(self.creator, '_get_tool_versions') as mock_get_versions:
                mock_versions = ToolVersions()
                mock_get_versions.return_value = mock_versions
                
                result = await self.creator._create_manifest(self.mock_context)
        
        assert result.index_id == "test_index"
        assert result.repo == {"name": "test_repo"}
        assert result.config == {"version": "1.0"}
        assert result.counts == mock_counts
        assert result.versions == mock_versions
        assert isinstance(result.paths, ArtifactPaths)
    
    @pytest.mark.asyncio
    async def test_calculate_counts(self):
        """Test count calculation from context."""
        result = await self.creator._calculate_counts(self.mock_context)
        
        assert result.files_total == 2
        assert result.files_indexed == 2
        assert result.symbols_defs == 1  # One "def" entry
        assert result.symbols_refs == 1  # One "ref" entry
        assert result.vectors == 3  # Three chunks
        assert result.chunks == 3
    
    @pytest.mark.asyncio
    async def test_calculate_counts_no_serena_graph(self):
        """Test count calculation without serena graph."""
        self.mock_context.serena_graph = None
        
        result = await self.creator._calculate_counts(self.mock_context)
        
        assert result.files_total == 2
        assert result.files_indexed == 2
        assert result.symbols_defs == 0
        assert result.symbols_refs == 0
    
    @pytest.mark.asyncio
    async def test_calculate_counts_no_vector_index(self):
        """Test count calculation without vector index."""
        self.mock_context.vector_index = None
        
        result = await self.creator._calculate_counts(self.mock_context)
        
        assert result.files_total == 2
        assert result.files_indexed == 2
        assert result.vectors == 0
        assert result.chunks == 0
    
    @pytest.mark.asyncio
    async def test_get_tool_versions_success(self):
        """Test successful tool version retrieval."""
        mock_process = Mock()
        mock_process.communicate = AsyncMock(return_value=(b"repomapper 1.2.3\n", b""))
        mock_process.returncode = 0
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await self.creator._get_tool_versions()
        
        # Should have extracted version from output
        assert hasattr(result, 'repomapper')
    
    @pytest.mark.asyncio
    async def test_get_tool_versions_command_failure(self):
        """Test tool version retrieval with command failure."""
        mock_process = Mock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"command not found"))
        mock_process.returncode = 1
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await self.creator._get_tool_versions()
        
        # Should handle failure gracefully
        assert isinstance(result, ToolVersions)
    
    @pytest.mark.asyncio
    async def test_get_tool_versions_exception(self):
        """Test tool version retrieval with exception."""
        with patch('asyncio.create_subprocess_exec', side_effect=Exception("Process failed")):
            result = await self.creator._get_tool_versions()
        
        # Should handle exception gracefully
        assert isinstance(result, ToolVersions)
    
    def test_extract_version_pattern_match(self):
        """Test version extraction with pattern match."""
        version_string = "repomapper version 1.2.3 (build 456)"
        result = self.creator._extract_version(version_string)
        assert result == "1.2.3"
    
    def test_extract_version_no_pattern(self):
        """Test version extraction without pattern match."""
        version_string = "repomapper beta release"
        result = self.creator._extract_version(version_string)
        assert result == "repomapper beta release"
    
    def test_extract_version_multiline(self):
        """Test version extraction with multiline output."""
        version_string = "repomapper beta release\nCopyright info\nUsage info"
        result = self.creator._extract_version(version_string)
        assert result == "repomapper beta release"
    
    @pytest.mark.asyncio
    async def test_validate_artifacts_all_present(self):
        """Test artifact validation when all files present."""
        paths = ArtifactPaths()
        paths.repomap = "repomap.json"
        paths.serena_graph = "serena_graph.jsonl" 
        paths.snippets = "snippets.jsonl"
        
        # Create required files
        (self.temp_dir / "repomap.json").touch()
        (self.temp_dir / "serena_graph.jsonl").touch()
        (self.temp_dir / "snippets.jsonl").touch()
        
        # Should not raise exception
        await self.creator._validate_artifacts(self.temp_dir, paths)
    
    @pytest.mark.asyncio
    async def test_validate_artifacts_missing_required(self):
        """Test artifact validation with missing required files."""
        paths = ArtifactPaths()
        paths.repomap = "repomap.json"
        paths.serena_graph = "serena_graph.jsonl"
        paths.snippets = "snippets.jsonl"
        
        # Create only some files
        (self.temp_dir / "repomap.json").touch()
        # Missing serena_graph.jsonl and snippets.jsonl
        
        with pytest.raises(BundleError) as exc_info:
            await self.creator._validate_artifacts(self.temp_dir, paths)
        
        assert "Missing required artifacts" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_validate_artifacts_missing_optional(self):
        """Test artifact validation with missing optional files."""
        paths = ArtifactPaths()
        paths.repomap = "repomap.json"
        paths.serena_graph = "serena_graph.jsonl"
        paths.snippets = "snippets.jsonl"
        paths.leann_index = "leann_index.pkl"
        paths.vectors = "vectors.npy"
        
        # Create required files only
        (self.temp_dir / "repomap.json").touch()
        (self.temp_dir / "serena_graph.jsonl").touch()
        (self.temp_dir / "snippets.jsonl").touch()
        # Missing optional files
        
        # Should not raise exception but should print warning
        with patch('builtins.print') as mock_print:
            await self.creator._validate_artifacts(self.temp_dir, paths)
            
        mock_print.assert_called_once()
        assert "Warning: Missing optional artifacts" in mock_print.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_create_compressed_bundle_success(self):
        """Test successful compressed bundle creation."""
        paths = ArtifactPaths()
        paths.repomap = "repomap.json"
        paths.serena_graph = "serena_graph.jsonl"
        
        # Create test files
        (self.temp_dir / "repomap.json").write_text('{"test": "data"}')
        (self.temp_dir / "serena_graph.jsonl").write_text('{"entry": "test"}')
        (self.temp_dir / "manifest.json").write_text('{"manifest": "test"}')
        
        with patch('repoindex.util.fs.get_directory_size', return_value=1024):
            with patch.object(self.creator, '_create_zstd_tar') as mock_create_tar:
                result = await self.creator._create_compressed_bundle(
                    self.temp_dir, paths, None
                )
        
        assert result == self.temp_dir / "bundle.tar.zst"
        mock_create_tar.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_compressed_bundle_size_warning(self):
        """Test compressed bundle creation with size warning."""
        paths = ArtifactPaths()
        paths.repomap = "repomap.json"
        
        # Create test file
        (self.temp_dir / "repomap.json").write_text('{"test": "data"}')
        
        # Mock large directory size
        large_size = self.creator.max_bundle_size + 1024
        
        with patch('repoindex.util.fs.get_directory_size', return_value=large_size):
            with patch.object(self.creator, '_create_zstd_tar'):
                with patch('builtins.print') as mock_print:
                    await self.creator._create_compressed_bundle(
                        self.temp_dir, paths, None
                    )
        
        mock_print.assert_called_once()
        assert "Warning: Bundle size" in mock_print.call_args[0][0]
    
    @pytest.mark.asyncio
    async def test_create_compressed_bundle_with_progress(self):
        """Test compressed bundle creation with progress callback."""
        paths = ArtifactPaths()
        paths.repomap = "repomap.json"
        
        (self.temp_dir / "repomap.json").write_text('{"test": "data"}')
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        with patch('repoindex.util.fs.get_directory_size', return_value=1024):
            with patch.object(self.creator, '_create_zstd_tar'):
                await self.creator._create_compressed_bundle(
                    self.temp_dir, paths, progress_callback
                )
        
        # Progress callback should be passed to _create_zstd_tar
        # (actual progress updates tested in _create_zstd_tar test)
    
    def test_create_zstd_tar_success(self):
        """Test successful zstd tar creation."""
        files = ["test1.txt", "test2.txt"]
        output_path = self.temp_dir / "test_bundle.tar.zst"
        
        # Create test files
        (self.temp_dir / "test1.txt").write_text("content1")
        (self.temp_dir / "test2.txt").write_text("content2")
        
        progress_calls = []
        def progress_callback(progress):
            progress_calls.append(progress)
        
        self.creator._create_zstd_tar(
            self.temp_dir, files, output_path, progress_callback
        )
        
        # Verify bundle was created
        assert output_path.exists()
        assert output_path.stat().st_size > 0
        
        # Verify progress was reported
        assert len(progress_calls) >= 2  # At least one call per file
    
    def test_create_zstd_tar_missing_files(self):
        """Test zstd tar creation with missing files."""
        files = ["exists.txt", "missing.txt"]
        output_path = self.temp_dir / "test_bundle.tar.zst"
        
        # Create only one file
        (self.temp_dir / "exists.txt").write_text("content")
        # missing.txt doesn't exist
        
        # Should not raise exception, should skip missing files
        self.creator._create_zstd_tar(self.temp_dir, files, output_path, None)
        
        assert output_path.exists()
    
    @pytest.mark.asyncio
    async def test_extract_bundle_success(self):
        """Test successful bundle extraction."""
        bundle_path = self.temp_dir / "test_bundle.tar.zst"
        extract_dir = self.temp_dir / "extracted"
        
        # Create a mock bundle (just an empty file for this test)
        bundle_path.touch()
        
        # Mock manifest content
        mock_manifest = Mock(spec=IndexManifest)
        
        with patch.object(self.creator, '_extract_zstd_tar') as mock_extract:
            with patch('repoindex.data.schemas.IndexManifest.load_from_file') as mock_load:
                mock_load.return_value = mock_manifest
                
                # Create manifest file in extract dir
                extract_dir.mkdir(parents=True)
                (extract_dir / "manifest.json").touch()
                
                result = await self.creator.extract_bundle(bundle_path, extract_dir)
        
        assert result == mock_manifest
        mock_extract.assert_called_once_with(bundle_path, extract_dir)
    
    @pytest.mark.asyncio
    async def test_extract_bundle_no_manifest(self):
        """Test bundle extraction with missing manifest."""
        bundle_path = self.temp_dir / "test_bundle.tar.zst"
        extract_dir = self.temp_dir / "extracted"
        
        bundle_path.touch()
        extract_dir.mkdir(parents=True)
        # No manifest.json created
        
        with patch.object(self.creator, '_extract_zstd_tar'):
            with pytest.raises(BundleError) as exc_info:
                await self.creator.extract_bundle(bundle_path, extract_dir)
        
        assert "No manifest found in extracted bundle" in str(exc_info.value)
    
    def test_extract_zstd_tar_mock(self):
        """Test zstd tar extraction (mocked)."""
        bundle_path = self.temp_dir / "test_bundle.tar.zst"
        extract_dir = self.temp_dir / "extracted"
        
        # Create mock compressed data
        bundle_path.touch()
        extract_dir.mkdir(parents=True)
        
        # Mock the zstd and tarfile operations
        with patch('zstandard.ZstdDecompressor') as mock_decompressor_class:
            mock_decompressor = Mock()
            mock_decompressor_class.return_value = mock_decompressor
            
            mock_stream = Mock()
            mock_decompressor.stream_reader.return_value.__enter__ = Mock(return_value=mock_stream)
            mock_decompressor.stream_reader.return_value.__exit__ = Mock(return_value=None)
            
            with patch('tarfile.open') as mock_tarfile:
                mock_tar = Mock()
                mock_tarfile.return_value.__enter__ = Mock(return_value=mock_tar)
                mock_tarfile.return_value.__exit__ = Mock(return_value=None)
                
                self.creator._extract_zstd_tar(bundle_path, extract_dir)
        
        mock_decompressor.stream_reader.assert_called_once()
        mock_tar.extractall.assert_called_once_with(path=extract_dir)
    
    @pytest.mark.asyncio
    async def test_validate_bundle_valid_bundle(self):
        """Test bundle validation for valid bundle."""
        bundle_path = self.temp_dir / "valid_bundle.tar.zst"
        bundle_path.write_bytes(b"mock bundle content")
        
        mock_file_list = ["manifest.json", "repomap.json", "serena_graph.jsonl", "snippets.jsonl"]
        
        with patch.object(self.creator, '_list_bundle_contents', return_value=mock_file_list):
            result = await self.creator.validate_bundle(bundle_path)
        
        assert result["valid"] is True
        assert result["bundle_size"] > 0
        assert result["file_count"] == 4
        assert result["has_manifest"] is True
        assert len(result["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_validate_bundle_missing_file(self):
        """Test bundle validation with missing bundle file."""
        bundle_path = self.temp_dir / "nonexistent.tar.zst"
        
        result = await self.creator.validate_bundle(bundle_path)
        
        assert result["valid"] is False
        assert result["bundle_size"] == 0
        assert "Bundle file does not exist" in result["issues"]
    
    @pytest.mark.asyncio
    async def test_validate_bundle_missing_manifest(self):
        """Test bundle validation with missing manifest."""
        bundle_path = self.temp_dir / "no_manifest.tar.zst"
        bundle_path.write_bytes(b"mock bundle content")
        
        mock_file_list = ["repomap.json", "serena_graph.jsonl"]  # No manifest.json
        
        with patch.object(self.creator, '_list_bundle_contents', return_value=mock_file_list):
            result = await self.creator.validate_bundle(bundle_path)
        
        assert result["has_manifest"] is False
        assert "Bundle missing manifest.json" in result["issues"]
    
    @pytest.mark.asyncio
    async def test_validate_bundle_missing_required_files(self):
        """Test bundle validation with missing required files."""
        bundle_path = self.temp_dir / "incomplete.tar.zst"
        bundle_path.write_bytes(b"mock bundle content")
        
        mock_file_list = ["manifest.json", "repomap.json"]  # Missing serena_graph.jsonl, snippets.jsonl
        
        with patch.object(self.creator, '_list_bundle_contents', return_value=mock_file_list):
            result = await self.creator.validate_bundle(bundle_path)
        
        assert any("Missing required files" in issue for issue in result["issues"])
    
    @pytest.mark.asyncio
    async def test_validate_bundle_size_warning(self):
        """Test bundle validation with size warning."""
        bundle_path = self.temp_dir / "large_bundle.tar.zst"
        
        # Create a large bundle file (mock)
        large_size = self.creator.max_bundle_size + 1024
        bundle_path.write_bytes(b"x" * 1024)  # Actual size doesn't matter for test
        
        mock_file_list = ["manifest.json", "repomap.json", "serena_graph.jsonl", "snippets.jsonl"]
        
        with patch.object(self.creator, '_list_bundle_contents', return_value=mock_file_list):
            with patch.object(bundle_path, 'stat') as mock_stat:
                mock_stat.return_value.st_size = large_size
                result = await self.creator.validate_bundle(bundle_path)
        
        assert any("Bundle size" in issue and "exceeds recommended limit" in issue 
                  for issue in result["issues"])
    
    @pytest.mark.asyncio
    async def test_validate_bundle_validation_exception(self):
        """Test bundle validation with exception during validation."""
        bundle_path = self.temp_dir / "error_bundle.tar.zst"
        bundle_path.write_bytes(b"mock bundle content")
        
        with patch.object(self.creator, '_list_bundle_contents', side_effect=Exception("Read error")):
            result = await self.creator.validate_bundle(bundle_path)
        
        assert result["valid"] is False
        assert any("Validation failed: Read error" in issue for issue in result["issues"])
    
    def test_list_bundle_contents_mock(self):
        """Test bundle content listing (mocked)."""
        bundle_path = self.temp_dir / "test_bundle.tar.zst"
        bundle_path.touch()
        
        # Mock tarfile members
        mock_file1 = Mock()
        mock_file1.name = "file1.txt"
        mock_file1.isfile.return_value = True
        
        mock_file2 = Mock()
        mock_file2.name = "file2.txt"
        mock_file2.isfile.return_value = True
        
        mock_dir = Mock()
        mock_dir.name = "directory"
        mock_dir.isfile.return_value = False  # Not a file
        
        with patch('zstandard.ZstdDecompressor') as mock_decompressor_class:
            mock_decompressor = Mock()
            mock_decompressor_class.return_value = mock_decompressor
            
            mock_stream = Mock()
            mock_decompressor.stream_reader.return_value.__enter__ = Mock(return_value=mock_stream)
            mock_decompressor.stream_reader.return_value.__exit__ = Mock(return_value=None)
            
            with patch('tarfile.open') as mock_tarfile:
                mock_tar = Mock()
                mock_tar.__iter__ = Mock(return_value=iter([mock_file1, mock_file2, mock_dir]))
                mock_tarfile.return_value.__enter__ = Mock(return_value=mock_tar)
                mock_tarfile.return_value.__exit__ = Mock(return_value=None)
                
                result = self.creator._list_bundle_contents(bundle_path)
        
        # Should only include files, not directories
        assert result == ["file1.txt", "file2.txt"]


class TestBundleError:
    """Test BundleError exception."""
    
    def test_bundle_error_creation(self):
        """Test BundleError creation."""
        error = BundleError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_bundle_error_inheritance(self):
        """Test BundleError inherits from Exception."""
        error = BundleError("Test error")
        assert isinstance(error, Exception)
        assert isinstance(error, BundleError)


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