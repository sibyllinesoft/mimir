"""
Input validation and sanitization for security hardening.

Provides comprehensive validation for repository paths, file content,
schema validation, and prevents path traversal and injection attacks.
"""

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import jsonschema
from jsonschema import Draft7Validator
from pydantic import BaseModel, ValidationError as PydanticValidationError

from ..util.errors import ValidationError, SecurityError
from ..util.log import get_logger

logger = get_logger(__name__)


class SecurityViolation(SecurityError):
    """Raised when input fails security validation."""
    
    def __init__(self, message: str, violation_type: str, details: Dict[str, Any]) -> None:
        super().__init__(message)
        self.violation_type = violation_type
        self.details = details


class PathValidator:
    """Validates and sanitizes file system paths to prevent traversal attacks."""
    
    # Dangerous patterns that could indicate path traversal
    DANGEROUS_PATTERNS = [
        r"\.\.[\\/]",  # Parent directory traversal
        r"[\\/]\.\.[\\/]",  # Directory traversal in middle
        r"[\\/]\.\.$",  # Directory traversal at end
        r"^\.\.[\\/]",  # Directory traversal at start
        r"~[\\/]",  # Home directory reference
        r"\$[A-Za-z_][A-Za-z0-9_]*",  # Environment variable expansion
        r"%[A-Za-z_][A-Za-z0-9_]*%",  # Windows environment variables
        r"[<>:\"|?*]",  # Invalid filename characters on Windows
        r"[\x00-\x1f\x7f-\x9f]",  # Control characters
    ]
    
    # Maximum path length to prevent buffer overflows
    MAX_PATH_LENGTH = 4096
    
    # Maximum filename length
    MAX_FILENAME_LENGTH = 255
    
    # Allowed file extensions for processing
    ALLOWED_EXTENSIONS = {
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h", 
        ".hpp", ".cs", ".go", ".rs", ".php", ".rb", ".kt", ".swift",
        ".scala", ".clj", ".hs", ".ml", ".elm", ".dart", ".vue", ".svelte",
        ".json", ".yaml", ".yml", ".toml", ".xml", ".md", ".txt", ".sql"
    }
    
    def __init__(self, allowed_base_paths: Optional[List[str]] = None):
        """Initialize path validator with optional base path restrictions.
        
        Args:
            allowed_base_paths: List of allowed base paths for operations.
                               If None, no base path restrictions apply.
        """
        self.allowed_base_paths = []
        if allowed_base_paths:
            for base_path in allowed_base_paths:
                resolved = Path(base_path).resolve()
                self.allowed_base_paths.append(resolved)
    
    def validate_path(self, path: Union[str, Path], operation: str = "access") -> Path:
        """Validate a file system path for security issues.
        
        Args:
            path: Path to validate
            operation: Operation being performed (for error messages)
            
        Returns:
            Validated and resolved Path object
            
        Raises:
            SecurityViolation: If path fails security validation
        """
        if not path:
            raise SecurityViolation(
                "Empty path provided",
                "empty_path",
                {"operation": operation}
            )
        
        path_str = str(path)
        
        # Check path length
        if len(path_str) > self.MAX_PATH_LENGTH:
            raise SecurityViolation(
                f"Path too long: {len(path_str)} > {self.MAX_PATH_LENGTH}",
                "path_too_long",
                {"path_length": len(path_str), "max_length": self.MAX_PATH_LENGTH}
            )
        
        # Check for dangerous patterns
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, path_str, re.IGNORECASE):
                raise SecurityViolation(
                    f"Path contains dangerous pattern: {pattern}",
                    "dangerous_pattern",
                    {"path": path_str, "pattern": pattern, "operation": operation}
                )
        
        # Resolve path and check for traversal
        try:
            resolved_path = Path(path).resolve()
        except (OSError, ValueError) as e:
            raise SecurityViolation(
                f"Failed to resolve path: {e}",
                "path_resolution_failed",
                {"path": path_str, "error": str(e)}
            )
        
        # Check filename length
        if len(resolved_path.name) > self.MAX_FILENAME_LENGTH:
            raise SecurityViolation(
                f"Filename too long: {len(resolved_path.name)} > {self.MAX_FILENAME_LENGTH}",
                "filename_too_long",
                {"filename": resolved_path.name, "length": len(resolved_path.name)}
            )
        
        # Check against allowed base paths if configured
        if self.allowed_base_paths:
            path_allowed = False
            for base_path in self.allowed_base_paths:
                try:
                    resolved_path.relative_to(base_path)
                    path_allowed = True
                    break
                except ValueError:
                    continue
            
            if not path_allowed:
                raise SecurityViolation(
                    "Path outside allowed base paths",
                    "path_outside_allowed_base",
                    {
                        "path": str(resolved_path),
                        "allowed_bases": [str(p) for p in self.allowed_base_paths]
                    }
                )
        
        return resolved_path
    
    def validate_repository_path(self, repo_path: Union[str, Path]) -> Path:
        """Validate a repository path for indexing operations.
        
        Args:
            repo_path: Repository path to validate
            
        Returns:
            Validated repository path
            
        Raises:
            SecurityViolation: If repository path is invalid
        """
        validated_path = self.validate_path(repo_path, "repository_indexing")
        
        # Additional checks for repositories
        if not validated_path.exists():
            raise SecurityViolation(
                "Repository path does not exist",
                "repo_path_not_exists",
                {"path": str(validated_path)}
            )
        
        if not validated_path.is_dir():
            raise SecurityViolation(
                "Repository path is not a directory",
                "repo_path_not_directory",
                {"path": str(validated_path)}
            )
        
        # Check for git repository
        git_dir = validated_path / ".git"
        if not git_dir.exists():
            logger.warning(
                "Repository path does not contain .git directory",
                repo_path=str(validated_path)
            )
        
        return validated_path
    
    def validate_file_extension(self, file_path: Union[str, Path]) -> bool:
        """Check if file extension is allowed for processing.
        
        Args:
            file_path: File path to check
            
        Returns:
            True if extension is allowed, False otherwise
        """
        path = Path(file_path)
        extension = path.suffix.lower()
        return extension in self.ALLOWED_EXTENSIONS
    
    def sanitize_filename(self, filename: str) -> str:
        """Sanitize a filename by removing dangerous characters.
        
        Args:
            filename: Original filename
            
        Returns:
            Sanitized filename
        """
        # Remove control characters and dangerous symbols
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f\x7f-\x9f]', '_', filename)
        
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_{2,}', '_', sanitized)
        
        # Remove leading/trailing underscores and dots
        sanitized = sanitized.strip('_.')
        
        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"
        
        # Truncate if too long
        if len(sanitized) > self.MAX_FILENAME_LENGTH:
            name_part = sanitized[:self.MAX_FILENAME_LENGTH - 10]
            sanitized = f"{name_part}_truncated"
        
        return sanitized


class ContentValidator:
    """Validates file content for security issues."""
    
    # Maximum file size for processing (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Maximum number of lines in a file
    MAX_LINES = 50000
    
    # Patterns that might indicate malicious content
    SUSPICIOUS_PATTERNS = [
        r"eval\s*\(",  # eval() calls
        r"exec\s*\(",  # exec() calls
        r"__import__\s*\(",  # dynamic imports
        r"subprocess\.",  # subprocess calls
        r"os\.system\s*\(",  # os.system calls
        r"open\s*\([^)]*[\"']w[\"']",  # file writes
        r"rm\s+-rf",  # dangerous shell commands
        r"sudo\s+",  # sudo commands
        r"chmod\s+777",  # permission changes
        r"<script[^>]*>",  # script tags
        r"javascript:",  # javascript URLs
        r"data:text/html",  # data URLs
        r"\\x[0-9a-fA-F]{2}",  # hex encoded content
        r"\\u[0-9a-fA-F]{4}",  # unicode encoded content
    ]
    
    def validate_file_content(self, file_path: Path, content: Optional[bytes] = None) -> Dict[str, Any]:
        """Validate file content for security issues.
        
        Args:
            file_path: Path to the file
            content: Optional file content to validate (if not provided, reads from file)
            
        Returns:
            Dictionary with validation results
            
        Raises:
            SecurityViolation: If content fails security validation
        """
        if content is None:
            if not file_path.exists():
                raise SecurityViolation(
                    "File does not exist",
                    "file_not_exists",
                    {"path": str(file_path)}
                )
            
            # Check file size before reading
            file_size = file_path.stat().st_size
            if file_size > self.MAX_FILE_SIZE:
                raise SecurityViolation(
                    f"File too large: {file_size} > {self.MAX_FILE_SIZE}",
                    "file_too_large",
                    {"path": str(file_path), "size": file_size, "max_size": self.MAX_FILE_SIZE}
                )
            
            try:
                content = file_path.read_bytes()
            except (OSError, PermissionError) as e:
                raise SecurityViolation(
                    f"Failed to read file: {e}",
                    "file_read_failed",
                    {"path": str(file_path), "error": str(e)}
                )
        
        # Check content size
        if len(content) > self.MAX_FILE_SIZE:
            raise SecurityViolation(
                f"Content too large: {len(content)} > {self.MAX_FILE_SIZE}",
                "content_too_large",
                {"size": len(content), "max_size": self.MAX_FILE_SIZE}
            )
        
        # Try to decode as text for pattern analysis
        text_content = ""
        encoding_detected = "unknown"
        
        for encoding in ["utf-8", "latin-1", "ascii"]:
            try:
                text_content = content.decode(encoding)
                encoding_detected = encoding
                break
            except UnicodeDecodeError:
                continue
        
        if not text_content and content:
            # Binary file - limited validation
            return {
                "is_text": False,
                "encoding": None,
                "size": len(content),
                "suspicious_patterns": 0,
                "warnings": ["Binary file - limited security validation"]
            }
        
        # Count lines
        lines = text_content.split('\n')
        if len(lines) > self.MAX_LINES:
            raise SecurityViolation(
                f"Too many lines: {len(lines)} > {self.MAX_LINES}",
                "too_many_lines",
                {"lines": len(lines), "max_lines": self.MAX_LINES}
            )
        
        # Check for suspicious patterns
        suspicious_count = 0
        suspicious_matches = []
        
        for pattern in self.SUSPICIOUS_PATTERNS:
            matches = re.findall(pattern, text_content, re.IGNORECASE | re.MULTILINE)
            if matches:
                suspicious_count += len(matches)
                suspicious_matches.append({
                    "pattern": pattern,
                    "matches": len(matches),
                    "examples": matches[:3]  # First 3 matches for reference
                })
        
        # Check for null bytes (often indicates binary data in text files)
        null_bytes = content.count(b'\x00')
        if null_bytes > 0:
            suspicious_matches.append({
                "pattern": "null_bytes",
                "matches": null_bytes,
                "examples": ["\\x00"]
            })
        
        warnings = []
        if suspicious_count > 10:
            warnings.append(f"High number of suspicious patterns detected: {suspicious_count}")
        
        if suspicious_count > 0:
            logger.warning(
                "Suspicious patterns detected in file content",
                file_path=str(file_path),
                suspicious_count=suspicious_count,
                patterns=suspicious_matches
            )
        
        return {
            "is_text": True,
            "encoding": encoding_detected,
            "size": len(content),
            "lines": len(lines),
            "suspicious_patterns": suspicious_count,
            "suspicious_matches": suspicious_matches,
            "warnings": warnings
        }


class SchemaValidator:
    """Validates data against JSON schemas."""
    
    def __init__(self):
        """Initialize schema validator."""
        self.validators_cache: Dict[str, Draft7Validator] = {}
    
    def load_schema(self, schema_path: Path) -> Dict[str, Any]:
        """Load and validate a JSON schema.
        
        Args:
            schema_path: Path to schema file
            
        Returns:
            Loaded schema dictionary
            
        Raises:
            SecurityViolation: If schema is invalid
        """
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise SecurityViolation(
                f"Failed to load schema: {e}",
                "schema_load_failed",
                {"schema_path": str(schema_path), "error": str(e)}
            )
        
        # Validate the schema itself
        try:
            Draft7Validator.check_schema(schema)
        except jsonschema.SchemaError as e:
            raise SecurityViolation(
                f"Invalid schema: {e}",
                "invalid_schema",
                {"schema_path": str(schema_path), "error": str(e)}
            )
        
        return schema
    
    def get_validator(self, schema: Dict[str, Any], schema_id: str) -> Draft7Validator:
        """Get or create a validator for the given schema.
        
        Args:
            schema: JSON schema dictionary
            schema_id: Unique identifier for caching
            
        Returns:
            JSON Schema validator
        """
        if schema_id not in self.validators_cache:
            self.validators_cache[schema_id] = Draft7Validator(schema)
        return self.validators_cache[schema_id]
    
    def validate_data(self, data: Any, schema: Dict[str, Any], schema_id: str = "default") -> Dict[str, Any]:
        """Validate data against a JSON schema.
        
        Args:
            data: Data to validate
            schema: JSON schema
            schema_id: Schema identifier for caching
            
        Returns:
            Validation results dictionary
            
        Raises:
            SecurityViolation: If data fails validation
        """
        validator = self.get_validator(schema, schema_id)
        
        errors = list(validator.iter_errors(data))
        
        if errors:
            error_details = []
            for error in errors[:10]:  # Limit to first 10 errors
                error_details.append({
                    "path": list(error.absolute_path),
                    "message": error.message,
                    "invalid_value": str(error.instance)[:100]  # Truncate for security
                })
            
            raise SecurityViolation(
                f"Schema validation failed with {len(errors)} errors",
                "schema_validation_failed",
                {
                    "error_count": len(errors),
                    "errors": error_details,
                    "schema_id": schema_id
                }
            )
        
        return {
            "valid": True,
            "schema_id": schema_id,
            "data_type": type(data).__name__
        }


class InputValidator:
    """Main input validator combining all validation methods."""
    
    def __init__(self, allowed_base_paths: Optional[List[str]] = None):
        """Initialize input validator.
        
        Args:
            allowed_base_paths: List of allowed base paths for path operations
        """
        self.path_validator = PathValidator(allowed_base_paths)
        self.content_validator = ContentValidator()
        self.schema_validator = SchemaValidator()
    
    def validate_mcp_request(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an MCP tool request.
        
        Args:
            tool_name: Name of the MCP tool being called
            arguments: Tool arguments to validate
            
        Returns:
            Validation results
            
        Raises:
            SecurityViolation: If request fails validation
        """
        results = {
            "tool_name": tool_name,
            "validation_passed": True,
            "warnings": []
        }
        
        # Validate repository path if present
        if "path" in arguments:
            try:
                validated_path = self.path_validator.validate_repository_path(arguments["path"])
                arguments["path"] = str(validated_path)  # Update with validated path
            except SecurityViolation as e:
                logger.error("Repository path validation failed", error=e.details)
                raise
        
        # Validate index_id format
        if "index_id" in arguments:
            index_id = arguments["index_id"]
            if not isinstance(index_id, str) or not re.match(r'^[a-zA-Z0-9\-_]{1,100}$', index_id):
                raise SecurityViolation(
                    "Invalid index_id format",
                    "invalid_index_id",
                    {"index_id": str(index_id)[:100]}  # Truncate for security
                )
        
        # Validate query strings
        for field in ["query", "question"]:
            if field in arguments:
                query = arguments[field]
                if not isinstance(query, str):
                    raise SecurityViolation(
                        f"Invalid {field} type: must be string",
                        "invalid_query_type",
                        {"field": field, "type": type(query).__name__}
                    )
                
                if len(query) > 10000:  # Reasonable query length limit
                    raise SecurityViolation(
                        f"Query too long: {len(query)} > 10000",
                        "query_too_long",
                        {"field": field, "length": len(query)}
                    )
                
                # Check for injection patterns
                suspicious_patterns = [
                    r"<script", r"javascript:", r"data:text/html",
                    r"\\x[0-9a-fA-F]{2}", r"\\u[0-9a-fA-F]{4}"
                ]
                
                for pattern in suspicious_patterns:
                    if re.search(pattern, query, re.IGNORECASE):
                        results["warnings"].append(f"Suspicious pattern in {field}: {pattern}")
        
        # Validate numeric parameters
        for field in ["k", "context_lines"]:
            if field in arguments:
                value = arguments[field]
                if not isinstance(value, int) or value < 0:
                    raise SecurityViolation(
                        f"Invalid {field}: must be non-negative integer",
                        "invalid_numeric_parameter",
                        {"field": field, "value": value}
                    )
                
                # Set reasonable limits
                max_values = {"k": 1000, "context_lines": 100}
                if value > max_values.get(field, 1000):
                    raise SecurityViolation(
                        f"{field} too large: {value} > {max_values[field]}",
                        "parameter_too_large",
                        {"field": field, "value": value, "max": max_values[field]}
                    )
        
        logger.info("MCP request validation passed", tool_name=tool_name, warnings=results["warnings"])
        return results