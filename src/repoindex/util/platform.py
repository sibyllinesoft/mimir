"""
Platform abstraction utilities for cross-platform compatibility.

Provides unified interfaces for platform-specific functionality including
resource management, process control, and system information across
Windows, macOS, and Linux platforms.
"""

import os
import platform
import resource
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional
from pathlib import Path

from .errors import ResourceError, create_error_context
from .log import get_logger

logger = get_logger(__name__)


@dataclass
class SystemInfo:
    """Cross-platform system information."""
    
    platform: str  # 'Windows', 'Linux', 'Darwin'
    architecture: str  # 'x86_64', 'arm64', etc.
    python_version: str
    is_windows: bool
    is_linux: bool
    is_macos: bool
    
    @classmethod
    def detect(cls) -> "SystemInfo":
        """Detect current system information."""
        system = platform.system()
        return cls(
            platform=system,
            architecture=platform.machine(),
            python_version=sys.version,
            is_windows=system == "Windows",
            is_linux=system == "Linux", 
            is_macos=system == "Darwin",
        )


@dataclass
class ResourceUsage:
    """Cross-platform resource usage information."""
    
    memory_bytes: int
    memory_mb: float
    memory_percent: float
    cpu_time_seconds: float
    cpu_percent: float
    
    # Limits
    memory_limit_bytes: int
    cpu_limit_seconds: int
    
    # Status
    memory_limit_exceeded: bool
    cpu_limit_exceeded: bool


class PlatformAdapter(ABC):
    """Abstract platform adapter for cross-platform functionality."""
    
    @abstractmethod
    def get_system_info(self) -> SystemInfo:
        """Get system information."""
        pass
    
    @abstractmethod
    def set_memory_limit(self, limit_bytes: int) -> bool:
        """Set memory limit for current process."""
        pass
    
    @abstractmethod
    def set_cpu_limit(self, limit_seconds: int) -> bool:
        """Set CPU time limit for current process."""
        pass
    
    @abstractmethod
    def set_file_descriptor_limit(self, limit: int) -> bool:
        """Set file descriptor limit for current process."""
        pass
    
    @abstractmethod
    def set_process_limit(self, limit: int) -> bool:
        """Set process/thread limit for current process."""
        pass
    
    @abstractmethod
    def set_file_size_limit(self, limit_bytes: int) -> bool:
        """Set maximum file size limit."""
        pass
    
    @abstractmethod
    def get_resource_usage(
        self, 
        memory_limit: int = 0, 
        cpu_limit: int = 0
    ) -> ResourceUsage:
        """Get current resource usage information."""
        pass


class UnixPlatformAdapter(PlatformAdapter):
    """Unix-based platform adapter (Linux, macOS)."""
    
    def __init__(self):
        self.system_info = SystemInfo.detect()
    
    def get_system_info(self) -> SystemInfo:
        """Get system information."""
        return self.system_info
    
    def set_memory_limit(self, limit_bytes: int) -> bool:
        """Set memory limit using setrlimit."""
        try:
            resource.setrlimit(resource.RLIMIT_RSS, (limit_bytes, limit_bytes))
            return True
        except (ValueError, OSError) as e:
            logger.warning("Failed to set memory limit", limit=limit_bytes, error=str(e))
            return False
    
    def set_cpu_limit(self, limit_seconds: int) -> bool:
        """Set CPU time limit using setrlimit."""
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (limit_seconds, limit_seconds))
            return True
        except (ValueError, OSError) as e:
            logger.warning("Failed to set CPU limit", limit=limit_seconds, error=str(e))
            return False
    
    def set_file_descriptor_limit(self, limit: int) -> bool:
        """Set file descriptor limit using setrlimit."""
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (limit, limit))
            return True
        except (ValueError, OSError) as e:
            logger.warning("Failed to set file descriptor limit", limit=limit, error=str(e))
            return False
    
    def set_process_limit(self, limit: int) -> bool:
        """Set process limit using setrlimit."""
        try:
            resource.setrlimit(resource.RLIMIT_NPROC, (limit, limit))
            return True
        except (ValueError, OSError) as e:
            logger.warning("Failed to set process limit", limit=limit, error=str(e))
            return False
    
    def set_file_size_limit(self, limit_bytes: int) -> bool:
        """Set file size limit using setrlimit."""
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (limit_bytes, limit_bytes))
            return True
        except (ValueError, OSError) as e:
            logger.warning("Failed to set file size limit", limit=limit_bytes, error=str(e))
            return False
    
    def get_resource_usage(
        self, 
        memory_limit: int = 0, 
        cpu_limit: int = 0
    ) -> ResourceUsage:
        """Get resource usage with platform-specific memory handling."""
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF)
            memory_kb = usage.ru_maxrss
            
            # Handle platform differences in ru_maxrss
            # Linux: ru_maxrss is in KB
            # macOS: ru_maxrss is in bytes
            if self.system_info.is_linux:
                memory_bytes = memory_kb * 1024
            else:  # macOS
                memory_bytes = memory_kb
            
            memory_mb = memory_bytes / (1024 * 1024)
            memory_percent = (
                (memory_bytes / memory_limit) * 100 if memory_limit > 0 else 0
            )
            
            cpu_time = usage.ru_utime + usage.ru_stime
            cpu_percent = (
                (cpu_time / cpu_limit) * 100 if cpu_limit > 0 else 0
            )
            
            return ResourceUsage(
                memory_bytes=memory_bytes,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_time_seconds=cpu_time,
                cpu_percent=cpu_percent,
                memory_limit_bytes=memory_limit,
                cpu_limit_seconds=cpu_limit,
                memory_limit_exceeded=memory_bytes > memory_limit if memory_limit > 0 else False,
                cpu_limit_exceeded=cpu_time > cpu_limit if cpu_limit > 0 else False,
            )
            
        except Exception as e:
            logger.warning("Failed to get resource usage", error=str(e))
            # Return empty usage data
            return ResourceUsage(
                memory_bytes=0,
                memory_mb=0,
                memory_percent=0,
                cpu_time_seconds=0,
                cpu_percent=0,
                memory_limit_bytes=memory_limit,
                cpu_limit_seconds=cpu_limit,
                memory_limit_exceeded=False,
                cpu_limit_exceeded=False,
            )


class WindowsPlatformAdapter(PlatformAdapter):
    """Windows platform adapter with limited resource control."""
    
    def __init__(self):
        self.system_info = SystemInfo.detect()
    
    def get_system_info(self) -> SystemInfo:
        """Get system information.""" 
        return self.system_info
    
    def set_memory_limit(self, limit_bytes: int) -> bool:
        """Windows memory limiting is limited - log warning."""
        logger.warning(
            "Memory limits not fully supported on Windows",
            limit=limit_bytes,
            suggestion="Use process-level monitoring instead"
        )
        return False
    
    def set_cpu_limit(self, limit_seconds: int) -> bool:
        """Windows CPU limiting is limited - log warning."""
        logger.warning(
            "CPU limits not fully supported on Windows",
            limit=limit_seconds,
            suggestion="Use process-level monitoring instead"
        )
        return False
    
    def set_file_descriptor_limit(self, limit: int) -> bool:
        """Windows file descriptor limiting is different - log warning."""
        logger.warning(
            "File descriptor limits not supported on Windows",
            limit=limit,
            suggestion="Use built-in Windows file handle management"
        )
        return False
    
    def set_process_limit(self, limit: int) -> bool:
        """Windows process limiting is different - log warning."""
        logger.warning(
            "Process limits not supported on Windows",
            limit=limit,
            suggestion="Use Windows Job Objects for process management"
        )
        return False
    
    def set_file_size_limit(self, limit_bytes: int) -> bool:
        """Windows file size limiting is handled differently - log warning."""
        logger.warning(
            "File size limits not supported on Windows",
            limit=limit_bytes,
            suggestion="Use application-level size checks"
        )
        return False
    
    def get_resource_usage(
        self,
        memory_limit: int = 0,
        cpu_limit: int = 0
    ) -> ResourceUsage:
        """Get resource usage using Windows-specific methods."""
        try:
            # Try to use psutil if available for better Windows support
            try:
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                memory_bytes = memory_info.rss
                
                # Get CPU times
                cpu_times = process.cpu_times()
                cpu_time = cpu_times.user + cpu_times.system
                
            except ImportError:
                logger.warning("psutil not available for Windows resource monitoring")
                # Fallback to basic resource module (limited on Windows)
                try:
                    usage = resource.getrusage(resource.RUSAGE_SELF)
                    # On Windows, ru_maxrss might not be reliable
                    memory_bytes = usage.ru_maxrss * 1024 if usage.ru_maxrss else 0
                    cpu_time = usage.ru_utime + usage.ru_stime
                except Exception:
                    memory_bytes = 0
                    cpu_time = 0
            
            memory_mb = memory_bytes / (1024 * 1024)
            memory_percent = (
                (memory_bytes / memory_limit) * 100 if memory_limit > 0 else 0
            )
            cpu_percent = (
                (cpu_time / cpu_limit) * 100 if cpu_limit > 0 else 0
            )
            
            return ResourceUsage(
                memory_bytes=memory_bytes,
                memory_mb=memory_mb,
                memory_percent=memory_percent,
                cpu_time_seconds=cpu_time,
                cpu_percent=cpu_percent,
                memory_limit_bytes=memory_limit,
                cpu_limit_seconds=cpu_limit,
                memory_limit_exceeded=memory_bytes > memory_limit if memory_limit > 0 else False,
                cpu_limit_exceeded=cpu_time > cpu_limit if cpu_limit > 0 else False,
            )
            
        except Exception as e:
            logger.warning("Failed to get Windows resource usage", error=str(e))
            return ResourceUsage(
                memory_bytes=0,
                memory_mb=0,
                memory_percent=0,
                cpu_time_seconds=0,
                cpu_percent=0,
                memory_limit_bytes=memory_limit,
                cpu_limit_seconds=cpu_limit,
                memory_limit_exceeded=False,
                cpu_limit_exceeded=False,
            )


# Platform detection and adapter factory
def create_platform_adapter() -> PlatformAdapter:
    """Create appropriate platform adapter for current system."""
    system_info = SystemInfo.detect()
    
    if system_info.is_windows:
        return WindowsPlatformAdapter()
    else:  # Unix-like (Linux, macOS)
        return UnixPlatformAdapter()


# Global platform adapter instance
_platform_adapter: Optional[PlatformAdapter] = None


def get_platform_adapter() -> PlatformAdapter:
    """Get the global platform adapter instance."""
    global _platform_adapter
    if _platform_adapter is None:
        _platform_adapter = create_platform_adapter()
    return _platform_adapter


# Convenience functions for common operations
def get_system_info() -> SystemInfo:
    """Get system information using the platform adapter."""
    return get_platform_adapter().get_system_info()


def get_platform_context() -> dict[str, Any]:
    """Get platform context for error reporting."""
    info = get_system_info()
    return {
        "platform": info.platform,
        "architecture": info.architecture,
        "python_version": info.python_version,
    }


def is_windows() -> bool:
    """Check if running on Windows."""
    return get_system_info().is_windows


def is_linux() -> bool:
    """Check if running on Linux."""
    return get_system_info().is_linux


def is_macos() -> bool:
    """Check if running on macOS."""
    return get_system_info().is_macos


def is_unix_like() -> bool:
    """Check if running on Unix-like system (Linux or macOS)."""
    info = get_system_info()
    return info.is_linux or info.is_macos


# Path handling utilities for cross-platform compatibility
def normalize_path(path: str) -> str:
    """Normalize path for current platform."""
    return os.path.normpath(path)


def get_executable_name(base_name: str) -> str:
    """Get executable name with platform-specific extension."""
    if is_windows():
        return f"{base_name}.exe"
    return base_name


def get_path_separator() -> str:
    """Get path separator for current platform."""
    return os.sep


def join_paths(*paths: str) -> str:
    """Join paths using platform-appropriate separator."""
    return os.path.join(*paths)


# Environment variable utilities
def get_home_directory() -> Path:
    """Get user home directory in a cross-platform way."""
    return Path.home()


def get_temp_directory() -> Path:
    """Get temporary directory in a cross-platform way."""
    return Path.cwd() / "tmp" if is_windows() else Path("/tmp")


def get_config_directory(app_name: str = "mimir") -> Path:
    """Get application configuration directory."""
    if is_windows():
        return Path.home() / "AppData" / "Roaming" / app_name
    elif is_macos():
        return Path.home() / "Library" / "Application Support" / app_name
    else:  # Linux
        return Path.home() / f".{app_name}"