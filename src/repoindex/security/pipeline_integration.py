"""
Security integration for Mimir pipeline components.

Integrates the comprehensive security framework with existing pipeline stages
to provide defense-in-depth protection during repository processing.
"""

import asyncio
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from ..data.schemas import PipelineStage
from ..util.log import get_logger
from .audit import SecurityEvent, SecurityEventType, get_security_auditor
from .config import SecurityConfig, get_security_config
from .crypto import CryptoManager, IndexEncryption
from .sandbox import ProcessIsolator, ResourceLimiter, SandboxViolation
from .secrets import CredentialScanner
from .validation import (
    ContentValidator,
    PathValidator,
    SchemaValidator,
    ValidationContext,
)
from .validation import (
    SecurityViolation as SecurityValidationError,
)

logger = get_logger(__name__)


class SecureFileDiscovery:
    """Security-enhanced file discovery wrapper."""

    def __init__(self, file_discovery, security_config: SecurityConfig | None = None):
        """Initialize secure file discovery wrapper.

        Args:
            file_discovery: Original FileDiscovery instance
            security_config: Security configuration
        """
        self.file_discovery = file_discovery
        self.security_config = security_config or get_security_config()
        self.path_validator = PathValidator(
            allowed_base_paths=self.security_config.allowed_base_paths,
            max_path_length=self.security_config.max_path_length,
            max_filename_length=self.security_config.max_filename_length,
        )
        self.content_validator = ContentValidator(max_file_size=self.security_config.max_file_size)
        self.credential_scanner = CredentialScanner()
        self.security_auditor = get_security_auditor()

    async def discover_files(
        self, extensions: list[str] | None = None, excludes: list[str] | None = None
    ) -> list[str]:
        """Securely discover files with validation and scanning."""
        try:
            # Record security event
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.FILE_ACCESS,
                    component="file_discovery",
                    message="Starting secure file discovery",
                    metadata={
                        "repo_root": str(self.file_discovery.repo_root),
                        "extensions": extensions,
                        "excludes": excludes,
                    },
                )
            )

            # Validate repository root path
            if not self.path_validator.validate_path(
                str(self.file_discovery.repo_root), operation="read"
            ):
                raise SecurityValidationError(
                    f"Repository root path validation failed: {self.file_discovery.repo_root}",
                    context=ValidationContext(
                        component="file_discovery",
                        operation="validate_repo_root",
                        input_data={"path": str(self.file_discovery.repo_root)},
                    ),
                )

            # Call original discovery
            discovered_files = await self.file_discovery.discover_files(
                extensions=extensions, excludes=excludes
            )

            # Validate each discovered file
            validated_files = []
            security_violations = []

            for file_path in discovered_files:
                try:
                    # Validate file path
                    full_path = self.file_discovery.repo_root / file_path

                    if not self.path_validator.validate_path(str(full_path), operation="read"):
                        security_violations.append(f"Path validation failed: {file_path}")
                        continue

                    # Validate file content size and type
                    if full_path.exists():
                        validation_result = await self.content_validator.validate_file_async(
                            full_path
                        )

                        if not validation_result.is_valid:
                            security_violations.append(
                                f"Content validation failed for {file_path}: {validation_result.error}"
                            )
                            continue

                        # Scan for credentials if enabled
                        if (
                            self.security_config.enable_credential_scanning
                            and self._should_scan_file(file_path)
                        ):
                            # Read file content for scanning
                            try:
                                content = await asyncio.to_thread(
                                    full_path.read_text, encoding="utf-8"
                                )
                                credentials = await asyncio.to_thread(
                                    self.credential_scanner.scan_text, content
                                )

                                if credentials:
                                    # Record security violation
                                    self.security_auditor.record_event(
                                        SecurityEvent(
                                            event_type=SecurityEventType.CREDENTIAL_DETECTED,
                                            component="file_discovery",
                                            severity="high",
                                            message=f"Credentials detected in {file_path}",
                                            metadata={
                                                "file_path": file_path,
                                                "credential_types": [
                                                    c.pattern_type for c in credentials
                                                ],
                                                "credential_count": len(credentials),
                                            },
                                        )
                                    )

                                    # Skip file if credentials found
                                    security_violations.append(
                                        f"Credentials detected in {file_path}, excluding from indexing"
                                    )
                                    continue

                            except UnicodeDecodeError:
                                # Skip binary files for credential scanning
                                pass
                            except Exception as e:
                                logger.warning(f"Failed to scan {file_path} for credentials: {e}")

                    validated_files.append(file_path)

                except Exception as e:
                    logger.warning(f"Security validation failed for {file_path}: {e}")
                    security_violations.append(f"Validation error for {file_path}: {e}")

            # Log security violations
            if security_violations:
                self.security_auditor.record_event(
                    SecurityEvent(
                        event_type=SecurityEventType.VALIDATION_FAILURE,
                        component="file_discovery",
                        severity="medium",
                        message="Security violations detected during file discovery",
                        metadata={
                            "violations": security_violations,
                            "total_files": len(discovered_files),
                            "validated_files": len(validated_files),
                        },
                    )
                )

            # Record successful completion
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.FILE_ACCESS,
                    component="file_discovery",
                    message="Secure file discovery completed",
                    metadata={
                        "total_discovered": len(discovered_files),
                        "validated_files": len(validated_files),
                        "security_violations": len(security_violations),
                    },
                )
            )

            logger.info(
                f"Secure file discovery completed: {len(validated_files)}/{len(discovered_files)} files validated",
                security_violations=len(security_violations),
            )

            return validated_files

        except Exception as e:
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.SECURITY_ERROR,
                    component="file_discovery",
                    severity="high",
                    message=f"Secure file discovery failed: {str(e)}",
                    metadata={"error": str(e)},
                )
            )
            raise

    def _should_scan_file(self, file_path: str) -> bool:
        """Determine if file should be scanned for credentials."""
        if len(self.credential_scanner.scan_results) >= self.security_config.max_scan_files:
            return False

        file_ext = Path(file_path).suffix.lower()
        return file_ext in self.security_config.scan_extensions

    async def get_file_content_with_context(
        self, file_path: str, start_line: int, end_line: int, context_lines: int = 5
    ) -> dict[str, str]:
        """Securely get file content with validation."""
        try:
            # Validate path and parameters
            full_path = self.file_discovery.repo_root / file_path

            if not self.path_validator.validate_path(str(full_path), operation="read"):
                raise SecurityValidationError(
                    f"Path validation failed for content access: {file_path}",
                    context=ValidationContext(
                        component="file_discovery",
                        operation="get_content",
                        input_data={"file_path": file_path},
                    ),
                )

            # Validate line parameters
            if start_line < 0 or end_line < start_line or context_lines < 0:
                raise SecurityValidationError(
                    "Invalid line parameters for content extraction",
                    context=ValidationContext(
                        component="file_discovery",
                        operation="validate_line_params",
                        input_data={
                            "start_line": start_line,
                            "end_line": end_line,
                            "context_lines": context_lines,
                        },
                    ),
                )

            # Limit context lines to prevent excessive memory usage
            max_context_lines = 50
            if context_lines > max_context_lines:
                context_lines = max_context_lines
                logger.warning(f"Context lines limited to {max_context_lines} for security")

            # Call original method
            result = await self.file_discovery.get_file_content_with_context(
                file_path, start_line, end_line, context_lines
            )

            # Log access
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.FILE_ACCESS,
                    component="file_discovery",
                    message=f"File content accessed: {file_path}",
                    metadata={
                        "file_path": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "context_lines": context_lines,
                    },
                )
            )

            return result

        except Exception as e:
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.SECURITY_ERROR,
                    component="file_discovery",
                    severity="medium",
                    message=f"Secure content access failed: {str(e)}",
                    metadata={"file_path": file_path, "error": str(e)},
                )
            )
            raise


class SecurePipelineContext:
    """Security-enhanced pipeline context wrapper."""

    def __init__(self, original_context, security_config: SecurityConfig | None = None):
        """Initialize secure pipeline context wrapper.

        Args:
            original_context: Original PipelineContext instance
            security_config: Security configuration
        """
        self.original_context = original_context
        self.security_config = security_config or get_security_config()
        self.resource_limiter = ResourceLimiter(
            max_memory=self.security_config.max_memory_mb * 1024 * 1024,
            max_cpu_time=self.security_config.max_cpu_time_seconds,
            max_wall_time=self.security_config.max_wall_time_seconds,
            max_open_files=self.security_config.max_open_files,
            max_processes=self.security_config.max_processes,
        )
        self.security_auditor = get_security_auditor()
        self.start_time = time.time()

        # Initialize encryption if enabled
        self.crypto_manager = None
        self.index_encryption = None
        if self.security_config.enable_index_encryption:
            self.crypto_manager = CryptoManager()
            self.index_encryption = IndexEncryption(self.crypto_manager)

    def __getattr__(self, name):
        """Delegate attribute access to original context."""
        return getattr(self.original_context, name)

    @asynccontextmanager
    async def secure_stage_execution(self, stage: PipelineStage, stage_func: Callable):
        """Execute pipeline stage with security controls."""
        stage_name = stage.value if hasattr(stage, "value") else str(stage)

        # Record stage start
        self.security_auditor.record_event(
            SecurityEvent(
                event_type=SecurityEventType.STAGE_START,
                component=f"pipeline_{stage_name}",
                message=f"Starting secure execution of {stage_name} stage",
                metadata={
                    "stage": stage_name,
                    "pipeline_id": self.original_context.index_id,
                    "repo_root": self.original_context.repo_info.root,
                },
            )
        )

        # Apply resource limits
        if self.security_config.enable_sandboxing:
            try:
                self.resource_limiter.apply_limits()
            except Exception as e:
                logger.warning(f"Failed to apply resource limits: {e}")

        stage_start = time.time()
        try:
            yield self

            # Record successful completion
            duration = time.time() - stage_start
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.STAGE_COMPLETE,
                    component=f"pipeline_{stage_name}",
                    message=f"Stage {stage_name} completed successfully",
                    metadata={
                        "stage": stage_name,
                        "duration_seconds": duration,
                        "pipeline_id": self.original_context.index_id,
                    },
                )
            )

        except Exception as e:
            # Record stage failure
            duration = time.time() - stage_start
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.STAGE_ERROR,
                    component=f"pipeline_{stage_name}",
                    severity="high",
                    message=f"Stage {stage_name} failed: {str(e)}",
                    metadata={
                        "stage": stage_name,
                        "duration_seconds": duration,
                        "error": str(e),
                        "pipeline_id": self.original_context.index_id,
                    },
                )
            )
            raise

        finally:
            # Cleanup resource limits
            if self.security_config.enable_sandboxing:
                try:
                    self.resource_limiter.cleanup()
                except Exception as e:
                    logger.warning(f"Failed to cleanup resource limits: {e}")

    async def secure_external_tool_execution(
        self, tool_name: str, command: list[str], work_dir: Path, timeout: int | None = None
    ) -> dict[str, Any]:
        """Execute external tool with sandboxing and monitoring."""
        if not self.security_config.enable_sandboxing:
            # If sandboxing disabled, log warning and proceed
            logger.warning(f"Executing {tool_name} without sandboxing - security risk!")
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.SECURITY_WARNING,
                    component=f"external_tool_{tool_name}",
                    severity="medium",
                    message="External tool executed without sandboxing",
                    metadata={
                        "tool": tool_name,
                        "command": command[:2],  # Only log first 2 command parts for security
                        "sandboxing_disabled": True,
                    },
                )
            )
            # Fall back to basic subprocess execution
            # This would need to be implemented based on the original tool execution
            raise NotImplementedError("Fallback execution not implemented")

        # Record tool execution start
        self.security_auditor.record_event(
            SecurityEvent(
                event_type=SecurityEventType.EXTERNAL_TOOL_START,
                component=f"external_tool_{tool_name}",
                message=f"Starting sandboxed execution of {tool_name}",
                metadata={"tool": tool_name, "work_dir": str(work_dir), "timeout": timeout},
            )
        )

        try:
            # Create process isolator
            isolator = ProcessIsolator(
                allowed_paths=[str(work_dir), str(self.original_context.repo_info.root)],
                resource_limits=self.security_config.get_resource_limits(),
                timeout=timeout or self.security_config.max_wall_time_seconds,
            )

            # Execute with sandboxing
            execution_start = time.time()
            result = await isolator.execute_isolated(
                command=command,
                work_dir=work_dir,
                env_vars={"PYTHONPATH": "", "NODE_ENV": "production"},  # Minimal environment
            )
            execution_time = time.time() - execution_start

            # Record successful execution
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.EXTERNAL_TOOL_COMPLETE,
                    component=f"external_tool_{tool_name}",
                    message=f"Tool {tool_name} executed successfully",
                    metadata={
                        "tool": tool_name,
                        "execution_time": execution_time,
                        "exit_code": result.get("exit_code", 0),
                        "stdout_length": len(result.get("stdout", "")),
                        "stderr_length": len(result.get("stderr", "")),
                    },
                )
            )

            return result

        except SandboxViolation as e:
            # Record sandbox violation
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.SANDBOX_VIOLATION,
                    component=f"external_tool_{tool_name}",
                    severity="high",
                    message=f"Sandbox violation during {tool_name} execution: {str(e)}",
                    metadata={
                        "tool": tool_name,
                        "violation_type": (
                            e.violation_type if hasattr(e, "violation_type") else "unknown"
                        ),
                        "error": str(e),
                    },
                )
            )
            raise

        except Exception as e:
            # Record general execution failure
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.EXTERNAL_TOOL_ERROR,
                    component=f"external_tool_{tool_name}",
                    severity="high",
                    message=f"Tool {tool_name} execution failed: {str(e)}",
                    metadata={"tool": tool_name, "error": str(e)},
                )
            )
            raise

    async def encrypt_pipeline_data(self, data: dict[str, Any], data_type: str) -> dict[str, Any]:
        """Encrypt sensitive pipeline data if encryption is enabled."""
        if not self.security_config.enable_index_encryption or not self.index_encryption:
            return data

        try:
            # Determine what to encrypt based on configuration
            encrypted_data = data.copy()

            if data_type == "vector_index" and self.security_config.encrypt_embeddings:
                # Encrypt vector embeddings
                if "embeddings" in data:
                    encrypted_data["embeddings"] = await asyncio.to_thread(
                        self.index_encryption.encrypt_embeddings,
                        data["embeddings"],
                        {"data_type": "vector_embeddings", "stage": "leann"},
                    )
                    logger.info("Vector embeddings encrypted")

            elif data_type == "metadata" and self.security_config.encrypt_metadata:
                # Encrypt metadata
                encrypted_data = await asyncio.to_thread(
                    self.index_encryption.encrypt_metadata,
                    data,
                    {"data_type": "pipeline_metadata", "stage": "bundle"},
                )
                logger.info("Pipeline metadata encrypted")

            # Record encryption activity
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.DATA_ENCRYPTED,
                    component="pipeline_encryption",
                    message=f"Pipeline data encrypted: {data_type}",
                    metadata={
                        "data_type": data_type,
                        "original_size": len(str(data)),
                        "encrypted_size": len(str(encrypted_data)),
                    },
                )
            )

            return encrypted_data

        except Exception as e:
            logger.error(f"Failed to encrypt pipeline data: {e}")
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.ENCRYPTION_ERROR,
                    component="pipeline_encryption",
                    severity="high",
                    message=f"Pipeline data encryption failed: {str(e)}",
                    metadata={"data_type": data_type, "error": str(e)},
                )
            )
            # Return original data if encryption fails (with warning)
            logger.warning(f"Proceeding without encryption for {data_type}")
            return data

    async def validate_pipeline_inputs(self, inputs: dict[str, Any], stage: str) -> dict[str, Any]:
        """Validate pipeline inputs with security checks."""
        try:
            validator = SchemaValidator()

            # Define stage-specific validation schemas
            validation_schemas = {
                "acquire": {
                    "type": "object",
                    "properties": {
                        "extensions": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 10},
                            "maxItems": 50,
                        },
                        "excludes": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 256},
                            "maxItems": 100,
                        },
                    },
                },
                "search": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "maxLength": self.security_config.max_query_length,
                        },
                        "k": {"type": "integer", "minimum": 1, "maximum": 100},
                        "context_lines": {"type": "integer", "minimum": 0, "maximum": 20},
                    },
                    "required": ["query"],
                },
            }

            # Validate against schema if available
            if stage in validation_schemas:
                validation_result = validator.validate_data(inputs, validation_schemas[stage])
                if not validation_result.is_valid:
                    raise SecurityValidationError(
                        f"Input validation failed for stage {stage}: {validation_result.error}",
                        context=ValidationContext(
                            component=f"pipeline_{stage}",
                            operation="validate_inputs",
                            input_data=inputs,
                        ),
                    )

            # Additional security checks
            validated_inputs = inputs.copy()

            # Sanitize string inputs
            for key, value in validated_inputs.items():
                if isinstance(value, str):
                    # Remove potentially dangerous characters
                    if any(char in value for char in ["<", ">", "&", '"', "\x00"]):
                        sanitized = "".join(
                            char for char in value if char not in ["<", ">", "&", '"', "\x00"]
                        )
                        validated_inputs[key] = sanitized
                        logger.warning(f"Sanitized input {key}: removed dangerous characters")

            return validated_inputs

        except Exception as e:
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.VALIDATION_FAILURE,
                    component=f"pipeline_{stage}",
                    severity="high",
                    message=f"Input validation failed: {str(e)}",
                    metadata={
                        "stage": stage,
                        "error": str(e),
                        "inputs": str(inputs)[:500],  # Truncate for security
                    },
                )
            )
            raise


class SecurityPipelineWrapper:
    """Security wrapper for the main indexing pipeline."""

    def __init__(self, original_pipeline, security_config: SecurityConfig | None = None):
        """Initialize security wrapper for pipeline.

        Args:
            original_pipeline: Original IndexingPipeline instance
            security_config: Security configuration
        """
        self.original_pipeline = original_pipeline
        self.security_config = security_config or get_security_config()
        self.security_auditor = get_security_auditor()

        # Override pipeline methods with secure versions
        self._wrap_pipeline_methods()

    def _wrap_pipeline_methods(self):
        """Wrap original pipeline methods with security enhancements."""
        # Store original methods
        self._original_start_indexing = self.original_pipeline.start_indexing
        self._original_execute_pipeline = self.original_pipeline._execute_pipeline

        # Replace with secure versions
        self.original_pipeline.start_indexing = self._secure_start_indexing
        self.original_pipeline._execute_pipeline = self._secure_execute_pipeline

    async def _secure_start_indexing(
        self,
        repo_path: str,
        rev: str | None = None,
        language: str = "ts",
        index_opts: dict[str, Any] | None = None,
    ) -> str:
        """Security-enhanced start indexing."""
        try:
            # Validate inputs
            path_validator = PathValidator(
                allowed_base_paths=self.security_config.allowed_base_paths,
                max_path_length=self.security_config.max_path_length,
            )

            if not path_validator.validate_path(repo_path, operation="read"):
                raise SecurityValidationError(
                    f"Repository path validation failed: {repo_path}",
                    context=ValidationContext(
                        component="pipeline_start",
                        operation="validate_repo_path",
                        input_data={"repo_path": repo_path},
                    ),
                )

            # Sanitize index options
            if index_opts:
                schema_validator = SchemaValidator()
                index_opts_schema = {
                    "type": "object",
                    "properties": {
                        "languages": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 10},
                            "maxItems": 20,
                        },
                        "excludes": {
                            "type": "array",
                            "items": {"type": "string", "maxLength": 256},
                            "maxItems": 100,
                        },
                    },
                }

                validation_result = schema_validator.validate_data(index_opts, index_opts_schema)
                if not validation_result.is_valid:
                    raise SecurityValidationError(
                        f"Index options validation failed: {validation_result.error}",
                        context=ValidationContext(
                            component="pipeline_start",
                            operation="validate_index_opts",
                            input_data=index_opts,
                        ),
                    )

            # Record pipeline start
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.PIPELINE_START,
                    component="pipeline",
                    message="Secure pipeline indexing started",
                    metadata={
                        "repo_path": repo_path,
                        "revision": rev,
                        "language": language,
                        "has_index_opts": index_opts is not None,
                    },
                )
            )

            # Call original method
            index_id = await self._original_start_indexing(repo_path, rev, language, index_opts)

            logger.info(f"Secure indexing started: {index_id}")
            return index_id

        except Exception as e:
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.PIPELINE_ERROR,
                    component="pipeline",
                    severity="high",
                    message=f"Secure pipeline start failed: {str(e)}",
                    metadata={"repo_path": repo_path, "error": str(e)},
                )
            )
            raise

    async def _secure_execute_pipeline(self, context) -> None:
        """Security-enhanced pipeline execution."""
        # Wrap context with security enhancements
        secure_context = SecurePipelineContext(context, self.security_config)

        try:
            # Record pipeline execution start
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.PIPELINE_EXECUTION_START,
                    component="pipeline",
                    message="Secure pipeline execution started",
                    metadata={
                        "pipeline_id": context.index_id,
                        "repo_root": context.repo_info.root,
                        "security_features": {
                            "sandboxing": self.security_config.enable_sandboxing,
                            "encryption": self.security_config.enable_index_encryption,
                            "credential_scanning": self.security_config.enable_credential_scanning,
                            "audit_logging": self.security_config.enable_audit_logging,
                        },
                    },
                )
            )

            # Execute original pipeline with secure context
            await self._original_execute_pipeline(secure_context)

            # Record successful completion
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.PIPELINE_COMPLETE,
                    component="pipeline",
                    message="Secure pipeline execution completed successfully",
                    metadata={
                        "pipeline_id": context.index_id,
                        "duration_seconds": time.time() - secure_context.start_time,
                    },
                )
            )

        except Exception as e:
            # Record pipeline failure
            self.security_auditor.record_event(
                SecurityEvent(
                    event_type=SecurityEventType.PIPELINE_ERROR,
                    component="pipeline",
                    severity="high",
                    message=f"Secure pipeline execution failed: {str(e)}",
                    metadata={
                        "pipeline_id": context.index_id,
                        "error": str(e),
                        "duration_seconds": time.time() - secure_context.start_time,
                    },
                )
            )
            raise

    def __getattr__(self, name):
        """Delegate attribute access to original pipeline."""
        return getattr(self.original_pipeline, name)


def create_secure_pipeline(original_pipeline, security_config: SecurityConfig | None = None):
    """Create security-enhanced version of indexing pipeline.

    Args:
        original_pipeline: Original IndexingPipeline instance
        security_config: Security configuration

    Returns:
        Security-enhanced pipeline wrapper
    """
    return SecurityPipelineWrapper(original_pipeline, security_config)


def create_secure_file_discovery(original_discovery, security_config: SecurityConfig | None = None):
    """Create security-enhanced version of file discovery.

    Args:
        original_discovery: Original FileDiscovery instance
        security_config: Security configuration

    Returns:
        Security-enhanced file discovery wrapper
    """
    return SecureFileDiscovery(original_discovery, security_config)
