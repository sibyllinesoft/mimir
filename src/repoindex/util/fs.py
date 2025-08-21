"""
File system utilities for atomic operations and content hashing.

Provides safe file operations with content verification and atomic writes
to prevent corruption during pipeline execution.
"""

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_index_directory(base_dir: Path, index_id: str) -> Path:
    """Get the directory for a specific index, ensuring it exists."""
    index_dir = base_dir / index_id
    return ensure_directory(index_dir)


def compute_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of file contents."""
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        while chunk := f.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def compute_content_hash(content: Union[str, bytes], algorithm: str = "sha256") -> str:
    """Compute hash of content."""
    hash_func = hashlib.new(algorithm)
    
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    hash_func.update(content)
    return hash_func.hexdigest()


def atomic_write_text(file_path: Union[str, Path], content: str, encoding: str = "utf-8") -> None:
    """Atomically write text content to file."""
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='w',
        encoding=encoding,
        dir=file_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    
    # Atomic move to final location
    tmp_path.replace(file_path)


def atomic_write_bytes(file_path: Union[str, Path], content: bytes) -> None:
    """Atomically write binary content to file."""
    file_path = Path(file_path)
    
    # Ensure parent directory exists
    ensure_directory(file_path.parent)
    
    # Write to temporary file first
    with tempfile.NamedTemporaryFile(
        mode='wb',
        dir=file_path.parent,
        delete=False,
        suffix='.tmp'
    ) as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    
    # Atomic move to final location
    tmp_path.replace(file_path)


def atomic_write_json(file_path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """Atomically write JSON data to file."""
    content = json.dumps(data, indent=indent, default=str)
    atomic_write_text(file_path, content)


def read_text_with_hash(file_path: Union[str, Path]) -> tuple[str, str]:
    """Read text file and return content with its hash."""
    file_path = Path(file_path)
    content = file_path.read_text(encoding='utf-8')
    content_hash = compute_content_hash(content)
    return content, content_hash


def read_bytes_with_hash(file_path: Union[str, Path]) -> tuple[bytes, str]:
    """Read binary file and return content with its hash."""
    file_path = Path(file_path)
    content = file_path.read_bytes()
    content_hash = compute_content_hash(content)
    return content, content_hash


def safe_copy(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Safely copy file with verification."""
    src, dst = Path(src), Path(dst)
    
    # Ensure destination directory exists
    ensure_directory(dst.parent)
    
    # Copy file
    shutil.copy2(src, dst)
    
    # Verify copy integrity
    src_hash = compute_file_hash(src)
    dst_hash = compute_file_hash(dst)
    
    if src_hash != dst_hash:
        dst.unlink()  # Remove corrupted copy
        raise ValueError(f"Copy verification failed: {src} -> {dst}")


def safe_move(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Safely move file with verification."""
    src, dst = Path(src), Path(dst)
    
    # Ensure destination directory exists
    ensure_directory(dst.parent)
    
    # Compute source hash before move
    src_hash = compute_file_hash(src)
    
    # Move file
    shutil.move(src, dst)
    
    # Verify move integrity
    dst_hash = compute_file_hash(dst)
    
    if src_hash != dst_hash:
        raise ValueError(f"Move verification failed: {src} -> {dst}")


def cleanup_directory(directory: Union[str, Path], keep_patterns: Optional[list[str]] = None) -> None:
    """Clean up directory, optionally keeping files matching patterns."""
    directory = Path(directory)
    
    if not directory.exists():
        return
    
    keep_patterns = keep_patterns or []
    
    for item in directory.iterdir():
        # Check if item matches any keep pattern
        should_keep = any(
            item.match(pattern) for pattern in keep_patterns
        )
        
        if not should_keep:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)


def get_directory_size(directory: Union[str, Path]) -> int:
    """Get total size of directory in bytes."""
    directory = Path(directory)
    total_size = 0
    
    for item in directory.rglob('*'):
        if item.is_file():
            total_size += item.stat().st_size
    
    return total_size


def create_temp_directory(prefix: str = "mimir_", parent: Optional[Path] = None) -> Path:
    """Create a temporary directory for pipeline work."""
    return Path(tempfile.mkdtemp(prefix=prefix, dir=parent))


class TemporaryDirectory:
    """Context manager for temporary directory with automatic cleanup."""
    
    def __init__(self, prefix: str = "mimir_", parent: Optional[Path] = None):
        self.prefix = prefix
        self.parent = parent
        self.path: Optional[Path] = None
    
    def __enter__(self) -> Path:
        self.path = create_temp_directory(self.prefix, self.parent)
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path and self.path.exists():
            shutil.rmtree(self.path)


def extract_file_span(
    file_path: Union[str, Path],
    start_byte: int,
    end_byte: int,
    context_lines: int = 5
) -> tuple[str, str, str]:
    """
    Extract file span with context lines.
    
    Returns (pre_context, span_content, post_context).
    """
    file_path = Path(file_path)
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract the span
    span_content = content[start_byte:end_byte]
    
    # Find line boundaries for context
    lines = content.split('\n')
    
    # Find which lines contain the span
    current_byte = 0
    start_line = 0
    end_line = 0
    
    for i, line in enumerate(lines):
        line_end = current_byte + len(line) + 1  # +1 for newline
        
        if current_byte <= start_byte < line_end:
            start_line = i
        
        if current_byte <= end_byte <= line_end:
            end_line = i
            break
        
        current_byte = line_end
    
    # Extract context lines
    pre_start = max(0, start_line - context_lines)
    post_end = min(len(lines), end_line + context_lines + 1)
    
    pre_context = '\n'.join(lines[pre_start:start_line])
    post_context = '\n'.join(lines[end_line + 1:post_end])
    
    return pre_context, span_content, post_context


def validate_file_integrity(
    file_path: Union[str, Path],
    expected_hash: str,
    algorithm: str = "sha256"
) -> bool:
    """Validate file integrity against expected hash."""
    try:
        actual_hash = compute_file_hash(file_path, algorithm)
        return actual_hash == expected_hash
    except Exception:
        return False


def get_file_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Get comprehensive file metadata."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    stat = file_path.stat()
    
    return {
        "path": str(file_path),
        "size": stat.st_size,
        "modified": stat.st_mtime,
        "created": getattr(stat, 'st_birthtime', stat.st_ctime),
        "mode": oct(stat.st_mode),
        "is_file": file_path.is_file(),
        "is_dir": file_path.is_dir(),
        "exists": True,
        "hash": compute_file_hash(file_path) if file_path.is_file() else None
    }