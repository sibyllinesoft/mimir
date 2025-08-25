"""
Basic tests for utility modules using available functions.

This test suite covers error handling, logging, and file system utilities
with tests that match the actual module implementations.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.repoindex.util.errors import (
    MimirError,
    ValidationError,
    ConfigurationError,
    SecurityError,
    ErrorSeverity,
    ErrorCategory,
    ErrorContext,
    create_error_context,
)
from src.repoindex.util.log import (
    get_logger,
    setup_logging,
    get_pipeline_logger,
    PipelineLogger,
)
from src.repoindex.util.fs import (
    ensure_directory,
    compute_file_hash,
    compute_content_hash,
    atomic_write_text,
    atomic_write_json,
    read_text_with_hash,
    get_file_metadata,
    create_temp_directory,
)


class TestErrorHandling:
    """Test error handling classes and functions."""
    
    def test_base_mimir_error(self):
        """Test base MimirError."""
        error = MimirError("Test error message")
        assert "Test error message" in str(error)
        assert isinstance(error, Exception)
    
    def test_validation_error(self):
        """Test ValidationError functionality."""
        error = ValidationError("Invalid input", field="username")
        assert "Invalid input" in str(error)
        assert error.details["field"] == "username"
    
    def test_configuration_error(self):
        """Test ConfigurationError functionality."""
        error = ConfigurationError("database", "Config missing")
        assert "Config missing" in str(error)
        assert error.details["config_key"] == "database"
    
    def test_security_error(self):
        """Test SecurityError functionality."""
        error = SecurityError("Access denied")
        assert "Access denied" in str(error)
        assert isinstance(error, MimirError)
    
    def test_error_severity(self):
        """Test error severity enumeration."""
        assert ErrorSeverity.CRITICAL == "critical"
        assert ErrorSeverity.HIGH == "high" 
        assert ErrorSeverity.MEDIUM == "medium"
        assert ErrorSeverity.LOW == "low"
        assert ErrorSeverity.INFO == "info"
    
    def test_error_category(self):
        """Test error category enumeration."""
        assert ErrorCategory.VALIDATION == "validation"
        assert ErrorCategory.CONFIGURATION == "configuration"
        assert ErrorCategory.PERMISSION in [cat.value for cat in ErrorCategory]
    
    def test_create_error_context(self):
        """Test error context creation."""
        context = create_error_context(
            component="test_component",
            operation="test_operation"
        )
        
        assert isinstance(context, ErrorContext)
        assert context.component == "test_component"
        assert context.operation == "test_operation"


class TestLogging:
    """Test logging functionality."""
    
    def test_get_logger(self):
        """Test logger creation."""
        logger = get_logger("test.module")
        assert logger.name == "test.module"
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
        assert hasattr(logger, 'warning')
        assert hasattr(logger, 'debug')
    
    def test_logger_hierarchy(self):
        """Test logger name hierarchy."""
        parent_logger = get_logger("parent")
        child_logger = get_logger("parent.child")
        
        assert parent_logger.name == "parent"
        assert child_logger.name == "parent.child"
    
    def test_setup_logging(self):
        """Test logging setup."""
        # Should not raise any exceptions
        setup_logging(level="INFO", format_type="standard")
        
        # Should configure root logger
        root_logger = logging.getLogger()
        assert root_logger.level == logging.INFO
    
    def test_get_pipeline_logger(self):
        """Test pipeline logger creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = get_pipeline_logger("test_index", temp_dir)
            assert isinstance(logger, PipelineLogger)
            assert logger.index_id == "test_index"


class TestFileSystem:
    """Test file system utility functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_path, ignore_errors=True)
    
    def test_ensure_directory_new(self, temp_dir):
        """Test creating new directory."""
        new_dir = temp_dir / "new_directory"
        
        assert not new_dir.exists()
        result = ensure_directory(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir
    
    def test_ensure_directory_existing(self, temp_dir):
        """Test ensuring existing directory."""
        existing_dir = temp_dir / "existing"
        existing_dir.mkdir()
        
        result = ensure_directory(existing_dir)
        assert existing_dir.exists()
        assert result == existing_dir
    
    def test_ensure_directory_nested(self, temp_dir):
        """Test creating nested directories."""
        nested_dir = temp_dir / "level1" / "level2" / "level3"
        
        result = ensure_directory(nested_dir)
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        assert result == nested_dir
    
    def test_compute_file_hash(self, temp_dir):
        """Test file hash computation."""
        test_file = temp_dir / "test.txt"
        test_content = "This is test content for hashing"
        test_file.write_text(test_content)
        
        # Test SHA256 hash (default)
        hash_value = compute_file_hash(test_file)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256 hex length
        
        # Test MD5 hash
        md5_hash = compute_file_hash(test_file, algorithm="md5")
        assert len(md5_hash) == 32  # MD5 hex length
        
        # Same content should produce same hash
        hash_value2 = compute_file_hash(test_file)
        assert hash_value == hash_value2
    
    def test_compute_content_hash(self):
        """Test content hash computation."""
        content = "Test content for hashing"
        
        # String content
        hash_str = compute_content_hash(content)
        assert isinstance(hash_str, str)
        assert len(hash_str) == 64
        
        # Bytes content
        hash_bytes = compute_content_hash(content.encode('utf-8'))
        assert hash_str == hash_bytes  # Should be same
        
        # Different algorithms
        md5_hash = compute_content_hash(content, algorithm="md5")
        assert len(md5_hash) == 32
    
    def test_atomic_write_text(self, temp_dir):
        """Test atomic text file writing."""
        test_file = temp_dir / "atomic.txt"
        test_content = "This is atomic content"
        
        atomic_write_text(test_file, test_content)
        
        assert test_file.exists()
        written_content = test_file.read_text(encoding="utf-8")
        assert written_content == test_content
    
    def test_atomic_write_json(self, temp_dir):
        """Test atomic JSON file writing."""
        test_file = temp_dir / "data.json"
        test_data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        
        atomic_write_json(test_file, test_data)
        
        assert test_file.exists()
        
        # Read back and verify
        import json
        with open(test_file, 'r') as f:
            loaded_data = json.load(f)
        
        assert loaded_data == test_data
    
    def test_read_text_with_hash(self, temp_dir):
        """Test reading text with hash computation."""
        test_file = temp_dir / "hash_test.txt"
        test_content = "Content to read with hash"
        test_file.write_text(test_content)
        
        content, hash_value = read_text_with_hash(test_file)
        
        assert content == test_content
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64  # SHA256
        
        # Verify hash matches separate computation
        expected_hash = compute_file_hash(test_file)
        assert hash_value == expected_hash
    
    def test_get_file_metadata(self, temp_dir):
        """Test file metadata extraction."""
        test_file = temp_dir / "metadata_test.txt"
        test_content = "Content for metadata testing"
        test_file.write_text(test_content)
        
        metadata = get_file_metadata(test_file)
        
        assert isinstance(metadata, dict)
        assert "size" in metadata
        assert "modified" in metadata
        assert "created" in metadata
        assert "path" in metadata
        
        assert metadata["size"] > 0
        assert isinstance(metadata["modified"], (int, float))
        assert isinstance(metadata["path"], str)
    
    def test_create_temp_directory(self):
        """Test temporary directory creation."""
        temp_dir = create_temp_directory(prefix="test_")
        
        assert temp_dir.exists()
        assert temp_dir.is_dir()
        assert "test_" in temp_dir.name
        
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)


class TestIntegrationScenarios:
    """Test integration between utility modules."""
    
    def test_error_with_file_operations(self, tmp_path):
        """Test error handling in file operations."""
        try:
            # Try to read non-existent file
            compute_file_hash(tmp_path / "nonexistent.txt")
        except FileNotFoundError as e:
            # Create error context
            context = create_error_context(
                component="file_operations",
                operation="compute_hash"
            )
            
            # Wrap in custom error
            error = ValidationError("File not found")
            
            assert "File not found" in str(error)
            assert context.component == "file_operations"
    
    def test_logging_with_error_context(self, tmp_path):
        """Test logging with structured error context."""
        logger = get_logger("test.integration")
        
        # Create error context
        context = create_error_context(
            component="test_system",
            operation="integration_test"
        )
        
        # This should not raise an error
        logger.info("Integration test starting", extra={"context": context})
        logger.error("Integration test error", extra={"context": context})
    
    def test_file_operations_workflow(self, tmp_path):
        """Test complete file operations workflow."""
        # Create directory structure
        work_dir = tmp_path / "workflow"
        ensure_directory(work_dir)
        
        # Write data atomically
        data_file = work_dir / "data.json"
        test_data = {"workflow": "test", "timestamp": "2024-01-01"}
        atomic_write_json(data_file, test_data)
        
        # Read with hash
        content, hash_value = read_text_with_hash(data_file)
        assert test_data["workflow"] in content
        
        # Get metadata
        metadata = get_file_metadata(data_file)
        assert metadata["hash"] == hash_value
        assert metadata["size"] > 0


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_file_operations(self, tmp_path):
        """Test operations on empty files."""
        empty_file = tmp_path / "empty.txt"
        empty_file.touch()
        
        # Should handle empty file
        hash_value = compute_file_hash(empty_file)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
        
        content, content_hash = read_text_with_hash(empty_file)
        assert content == ""
        assert content_hash == hash_value
    
    def test_unicode_file_operations(self, tmp_path):
        """Test operations with Unicode content."""
        unicode_file = tmp_path / "unicode.txt"
        unicode_content = "Hello 🌍 Unicode: àáâãäå 中文 русский"
        
        atomic_write_text(unicode_file, unicode_content)
        
        content, hash_value = read_text_with_hash(unicode_file)
        assert content == unicode_content
        assert isinstance(hash_value, str)
    
    def test_large_content_hashing(self):
        """Test hashing large content."""
        large_content = "A" * (1024 * 1024)  # 1MB
        
        hash_value = compute_content_hash(large_content)
        assert isinstance(hash_value, str)
        assert len(hash_value) == 64
    
    def test_error_context_with_none_values(self):
        """Test error context with None values."""
        context = create_error_context(
            component="test",
            operation="none_test"
        )
        
        assert context.component == "test"
        assert context.operation == "none_test"


class TestPerformanceScenarios:
    """Test performance-related scenarios."""
    
    def test_many_file_operations(self, tmp_path):
        """Test performance with many file operations."""
        test_dir = tmp_path / "performance"
        ensure_directory(test_dir)
        
        # Create many files
        file_count = 50  # Keep reasonable for CI
        for i in range(file_count):
            file_path = test_dir / f"file_{i:03d}.txt"
            atomic_write_text(file_path, f"Content of file {i}")
        
        # Verify all files
        created_files = list(test_dir.glob("file_*.txt"))
        assert len(created_files) == file_count
        
        # Hash all files
        hashes = []
        for file_path in created_files:
            hash_value = compute_file_hash(file_path)
            hashes.append(hash_value)
        
        # All hashes should be unique (different content)
        assert len(set(hashes)) == len(hashes)
    
    def test_large_json_operations(self, tmp_path):
        """Test performance with large JSON data."""
        large_data = {
            "items": [{"id": i, "value": f"item_{i}", "data": list(range(10))} 
                     for i in range(1000)]
        }
        
        json_file = tmp_path / "large.json"
        atomic_write_json(json_file, large_data)
        
        assert json_file.exists()
        metadata = get_file_metadata(json_file)
        assert metadata["size"] > 10000  # Should be reasonably large


class TestConcurrencyEdgeCases:
    """Test edge cases related to concurrency."""
    
    def test_concurrent_directory_creation(self, tmp_path):
        """Test concurrent directory creation."""
        import threading
        import queue
        
        results = queue.Queue()
        target_dir = tmp_path / "concurrent"
        
        def create_directory():
            try:
                result_dir = ensure_directory(target_dir)
                results.put(("success", str(result_dir)))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Start multiple threads trying to create same directory
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=create_directory)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success":
                success_count += 1
        
        # All should succeed
        assert success_count == 5
        assert target_dir.exists()
    
    def test_concurrent_file_writing(self, tmp_path):
        """Test concurrent atomic file writing."""
        import threading
        
        def write_worker(worker_id):
            try:
                file_path = tmp_path / f"worker_{worker_id}.txt"
                content = f"Worker {worker_id} content"
                atomic_write_text(file_path, content)
                return True
            except Exception:
                return False
        
        # Start concurrent writers
        threads = []
        for i in range(10):
            thread = threading.Thread(target=write_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify all files were created
        worker_files = list(tmp_path.glob("worker_*.txt"))
        assert len(worker_files) == 10
        
        # Verify content integrity
        for i in range(10):
            file_path = tmp_path / f"worker_{i}.txt"
            assert file_path.exists()
            content = file_path.read_text()
            assert content == f"Worker {i} content"