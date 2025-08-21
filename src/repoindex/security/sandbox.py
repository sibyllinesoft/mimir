"""
Sandboxing and process isolation for security hardening.

Provides process isolation, resource limits, and safe execution
environments for code analysis operations.
"""

import asyncio
import os
import resource
import signal
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, AsyncGenerator

from ..util.errors import SecurityError, ExternalToolError
from ..util.log import get_logger

logger = get_logger(__name__)


class ResourceExhausted(SecurityError):
    """Raised when resource limits are exceeded."""
    pass


class SandboxViolation(SecurityError):
    """Raised when sandbox constraints are violated."""
    pass


class ResourceLimiter:
    """Manages system resource limits for safe execution."""
    
    # Default resource limits
    DEFAULT_LIMITS = {
        "max_memory": 1024 * 1024 * 1024,  # 1GB memory
        "max_cpu_time": 300,  # 5 minutes CPU time
        "max_wall_time": 600,  # 10 minutes wall time
        "max_open_files": 1024,  # File descriptor limit
        "max_processes": 32,  # Process limit
        "max_file_size": 100 * 1024 * 1024,  # 100MB file size
    }
    
    def __init__(self, custom_limits: Optional[Dict[str, int]] = None):
        """Initialize resource limiter with optional custom limits.
        
        Args:
            custom_limits: Dictionary of custom resource limits
        """
        self.limits = self.DEFAULT_LIMITS.copy()
        if custom_limits:
            self.limits.update(custom_limits)
    
    def set_process_limits(self) -> None:
        """Set resource limits for the current process."""
        try:
            # Memory limit (RSS - Resident Set Size)
            if self.limits["max_memory"] > 0:
                resource.setrlimit(resource.RLIMIT_RSS, (self.limits["max_memory"], self.limits["max_memory"]))
            
            # CPU time limit
            if self.limits["max_cpu_time"] > 0:
                resource.setrlimit(resource.RLIMIT_CPU, (self.limits["max_cpu_time"], self.limits["max_cpu_time"]))
            
            # File descriptor limit
            if self.limits["max_open_files"] > 0:
                resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits["max_open_files"], self.limits["max_open_files"]))
            
            # Process limit
            if self.limits["max_processes"] > 0:
                resource.setrlimit(resource.RLIMIT_NPROC, (self.limits["max_processes"], self.limits["max_processes"]))
            
            # File size limit
            if self.limits["max_file_size"] > 0:
                resource.setrlimit(resource.RLIMIT_FSIZE, (self.limits["max_file_size"], self.limits["max_file_size"]))
            
            logger.debug("Process resource limits set", limits=self.limits)
            
        except (ValueError, OSError) as e:
            logger.warning("Failed to set some resource limits", error=str(e), limits=self.limits)
    
    def check_memory_usage(self) -> Dict[str, Any]:
        """Check current memory usage against limits.
        
        Returns:
            Dictionary with memory usage information
        """
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_kb = usage.ru_maxrss
            
            # Convert to bytes (ru_maxrss is in KB on Linux, bytes on macOS)
            if os.name == 'posix' and os.uname().sysname != 'Darwin':
                memory_bytes = memory_kb * 1024
            else:
                memory_bytes = memory_kb
            
            memory_percent = (memory_bytes / self.limits["max_memory"]) * 100 if self.limits["max_memory"] > 0 else 0
            
            return {
                "memory_bytes": memory_bytes,
                "memory_mb": memory_bytes / (1024 * 1024),
                "memory_percent": memory_percent,
                "limit_bytes": self.limits["max_memory"],
                "limit_exceeded": memory_bytes > self.limits["max_memory"] if self.limits["max_memory"] > 0 else False
            }
        except Exception as e:
            logger.warning("Failed to check memory usage", error=str(e))
            return {"error": str(e)}
    
    def check_cpu_usage(self) -> Dict[str, Any]:
        """Check current CPU usage against limits.
        
        Returns:
            Dictionary with CPU usage information
        """
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            cpu_time = usage.ru_utime + usage.ru_stime
            cpu_percent = (cpu_time / self.limits["max_cpu_time"]) * 100 if self.limits["max_cpu_time"] > 0 else 0
            
            return {
                "cpu_time_seconds": cpu_time,
                "cpu_percent": cpu_percent,
                "limit_seconds": self.limits["max_cpu_time"],
                "limit_exceeded": cpu_time > self.limits["max_cpu_time"] if self.limits["max_cpu_time"] > 0 else False
            }
        except Exception as e:
            logger.warning("Failed to check CPU usage", error=str(e))
            return {"error": str(e)}


class ProcessIsolator:
    """Isolates processes with restricted permissions and capabilities."""
    
    def __init__(self, resource_limiter: Optional[ResourceLimiter] = None):
        """Initialize process isolator.
        
        Args:
            resource_limiter: Resource limiter to apply to processes
        """
        self.resource_limiter = resource_limiter or ResourceLimiter()
    
    async def run_isolated_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        input_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Run a command in an isolated environment.
        
        Args:
            command: Command and arguments to execute
            cwd: Working directory for the command
            env: Environment variables for the command
            timeout: Timeout in seconds
            input_data: Optional input data to send to the process
            
        Returns:
            Dictionary with execution results
            
        Raises:
            SecurityError: If execution fails security checks
            ExternalToolError: If command execution fails
        """
        if not command:
            raise SecurityError("Empty command provided")
        
        # Validate command executable
        executable = command[0]
        if not self._validate_executable(executable):
            raise SecurityError(f"Executable not allowed: {executable}")
        
        # Set up isolated environment
        isolated_env = self._create_isolated_environment(env)
        
        # Use timeout from resource limiter if not specified
        if timeout is None:
            timeout = self.resource_limiter.limits["max_wall_time"]
        
        start_time = time.time()
        
        try:
            # Create subprocess with security restrictions
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=cwd,
                env=isolated_env,
                stdin=asyncio.subprocess.PIPE if input_data else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=self._setup_child_process,
            )
            
            # Run with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(input_data),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                # Kill the process and its children
                try:
                    process.kill()
                    await process.wait()
                except ProcessLookupError:
                    pass
                
                raise ExternalToolError(
                    tool=executable,
                    message=f"Command timed out after {timeout} seconds",
                    exit_code=-1,
                    stdout="",
                    stderr=f"Process killed due to timeout ({timeout}s)"
                )
            
            execution_time = time.time() - start_time
            return_code = process.returncode
            
            # Decode output safely
            stdout_str = stdout.decode('utf-8', errors='replace') if stdout else ""
            stderr_str = stderr.decode('utf-8', errors='replace') if stderr else ""
            
            # Log execution details
            logger.info(
                "Isolated command execution completed",
                command=command[0],
                return_code=return_code,
                execution_time=execution_time,
                stdout_length=len(stdout_str),
                stderr_length=len(stderr_str)
            )
            
            return {
                "return_code": return_code,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "execution_time": execution_time,
                "command": command,
                "success": return_code == 0
            }
            
        except FileNotFoundError:
            raise ExternalToolError(
                tool=executable,
                message=f"Executable not found: {executable}",
                exit_code=-1,
                stdout="",
                stderr=f"Command not found: {executable}"
            )
        except Exception as e:
            logger.error("Failed to execute isolated command", command=command, error=str(e))
            raise ExternalToolError(
                tool=executable,
                message=f"Failed to execute command: {e}",
                exit_code=-1,
                stdout="",
                stderr=str(e)
            )
    
    def _validate_executable(self, executable: str) -> bool:
        """Validate that an executable is allowed to run.
        
        Args:
            executable: Executable name or path
            
        Returns:
            True if executable is allowed
        """
        # List of allowed executables for code analysis
        allowed_executables = {
            "repomapper", "tree-sitter", "node", "npm", "npx", "python3", "python",
            "git", "find", "grep", "awk", "sed", "sort", "uniq", "head", "tail",
            "cat", "wc", "ls", "mkdir", "cp", "mv", "rm", "chmod", "tar", "gzip",
            "gunzip", "zstd", "unzstd"
        }
        
        # Get base executable name
        base_name = Path(executable).name
        
        # Check if it's in allowed list
        if base_name in allowed_executables:
            return True
        
        # Allow executables in /usr/bin and /bin
        if executable.startswith(('/usr/bin/', '/bin/')):
            return True
        
        logger.warning("Executable not in allowed list", executable=executable)
        return False
    
    def _create_isolated_environment(self, user_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """Create an isolated environment with minimal variables.
        
        Args:
            user_env: User-provided environment variables
            
        Returns:
            Isolated environment dictionary
        """
        # Start with minimal safe environment
        isolated_env = {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "HOME": "/tmp",
            "USER": "nobody",
            "SHELL": "/bin/sh",
            "TERM": "dumb",
            "LC_ALL": "C",
            "LANG": "C",
        }
        
        # Add safe user environment variables
        if user_env:
            safe_vars = {
                "NODE_ENV", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE",
                "NPM_CONFIG_CACHE", "npm_config_cache"
            }
            
            for key, value in user_env.items():
                if key in safe_vars:
                    isolated_env[key] = value
                elif key.startswith(("MIMIR_", "REPOINDEX_")):
                    # Allow our own environment variables
                    isolated_env[key] = value
        
        return isolated_env
    
    def _setup_child_process(self) -> None:
        """Set up child process with security restrictions."""
        try:
            # Set resource limits
            self.resource_limiter.set_process_limits()
            
            # Create new process group
            os.setpgrp()
            
            # Set up signal handling to prevent escape
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            
        except Exception as e:
            logger.warning("Failed to set up child process restrictions", error=str(e))


class Sandbox:
    """Main sandbox manager combining isolation and resource limiting."""
    
    def __init__(
        self,
        base_path: Optional[Path] = None,
        custom_limits: Optional[Dict[str, int]] = None,
        allowed_executables: Optional[List[str]] = None
    ):
        """Initialize sandbox environment.
        
        Args:
            base_path: Base path for sandbox operations
            custom_limits: Custom resource limits
            allowed_executables: Additional allowed executables
        """
        self.base_path = base_path or Path.cwd()
        self.resource_limiter = ResourceLimiter(custom_limits)
        self.process_isolator = ProcessIsolator(self.resource_limiter)
        self.temp_dirs: List[Path] = []
    
    @asynccontextmanager
    async def isolated_environment(
        self,
        cleanup_on_exit: bool = True
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Create an isolated environment context manager.
        
        Args:
            cleanup_on_exit: Whether to clean up temporary resources on exit
            
        Yields:
            Dictionary with sandbox environment information
        """
        temp_dir = None
        try:
            # Create temporary directory for sandbox operations
            temp_dir = Path(tempfile.mkdtemp(prefix="mimir_sandbox_"))
            self.temp_dirs.append(temp_dir)
            
            # Set initial resource limits
            initial_usage = self.resource_limiter.check_memory_usage()
            
            sandbox_info = {
                "temp_dir": temp_dir,
                "base_path": self.base_path,
                "initial_memory": initial_usage,
                "resource_limiter": self.resource_limiter,
                "process_isolator": self.process_isolator
            }
            
            logger.info("Sandbox environment created", temp_dir=str(temp_dir))
            
            yield sandbox_info
            
        except Exception as e:
            logger.error("Sandbox environment error", error=str(e), temp_dir=str(temp_dir))
            raise
        finally:
            if cleanup_on_exit and temp_dir:
                await self._cleanup_temp_dir(temp_dir)
    
    async def run_sandboxed_command(
        self,
        command: List[str],
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        input_data: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """Run a command in the sandbox environment.
        
        Args:
            command: Command to execute
            cwd: Working directory (relative to sandbox)
            env: Environment variables
            timeout: Execution timeout
            input_data: Input data for the process
            
        Returns:
            Execution results
        """
        # Use temporary directory if no cwd specified
        async with self.isolated_environment() as sandbox:
            work_dir = cwd or sandbox["temp_dir"]
            
            # Ensure working directory is within sandbox
            if not self._path_within_sandbox(work_dir):
                raise SandboxViolation(f"Working directory outside sandbox: {work_dir}")
            
            # Run the command
            result = await self.process_isolator.run_isolated_command(
                command=command,
                cwd=work_dir,
                env=env,
                timeout=timeout,
                input_data=input_data
            )
            
            # Check resource usage after execution
            memory_usage = self.resource_limiter.check_memory_usage()
            cpu_usage = self.resource_limiter.check_cpu_usage()
            
            result.update({
                "sandbox_info": {
                    "temp_dir": str(sandbox["temp_dir"]),
                    "memory_usage": memory_usage,
                    "cpu_usage": cpu_usage
                }
            })
            
            return result
    
    def _path_within_sandbox(self, path: Path) -> bool:
        """Check if a path is within the sandbox boundaries.
        
        Args:
            path: Path to check
            
        Returns:
            True if path is within sandbox
        """
        try:
            resolved_path = path.resolve()
            
            # Check against base path
            try:
                resolved_path.relative_to(self.base_path.resolve())
                return True
            except ValueError:
                pass
            
            # Check against temporary directories
            for temp_dir in self.temp_dirs:
                try:
                    resolved_path.relative_to(temp_dir.resolve())
                    return True
                except ValueError:
                    continue
            
            return False
            
        except Exception:
            return False
    
    async def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Clean up a temporary directory.
        
        Args:
            temp_dir: Temporary directory to clean up
        """
        try:
            if temp_dir.exists():
                # Use system command for safe cleanup
                await self.process_isolator.run_isolated_command(
                    ["rm", "-rf", str(temp_dir)],
                    timeout=30
                )
                
            # Remove from tracking list
            if temp_dir in self.temp_dirs:
                self.temp_dirs.remove(temp_dir)
                
            logger.debug("Temporary directory cleaned up", temp_dir=str(temp_dir))
            
        except Exception as e:
            logger.warning("Failed to cleanup temporary directory", temp_dir=str(temp_dir), error=str(e))
    
    async def cleanup_all(self) -> None:
        """Clean up all sandbox resources."""
        for temp_dir in self.temp_dirs.copy():
            await self._cleanup_temp_dir(temp_dir)
        
        logger.info("All sandbox resources cleaned up")


# Global sandbox instance for use across the application
_global_sandbox: Optional[Sandbox] = None


def get_sandbox() -> Sandbox:
    """Get the global sandbox instance.
    
    Returns:
        Global sandbox instance
    """
    global _global_sandbox
    if _global_sandbox is None:
        _global_sandbox = Sandbox()
    return _global_sandbox


def configure_sandbox(
    base_path: Optional[Path] = None,
    custom_limits: Optional[Dict[str, int]] = None
) -> None:
    """Configure the global sandbox instance.
    
    Args:
        base_path: Base path for sandbox operations
        custom_limits: Custom resource limits
    """
    global _global_sandbox
    _global_sandbox = Sandbox(base_path=base_path, custom_limits=custom_limits)