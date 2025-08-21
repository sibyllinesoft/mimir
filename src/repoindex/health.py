"""
Health check endpoints for Mimir Deep Code Research System.

Provides comprehensive health checks for container orchestration and monitoring.
"""

import asyncio
import json
import logging
import psutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

from .util.fs import ensure_directory
from .pipeline.run import IndexingPipeline
from .monitoring import get_metrics_collector


logger = logging.getLogger(__name__)


class HealthChecker:
    """
    Comprehensive health checker for Mimir services.
    
    Provides various levels of health checks from basic liveness
    to detailed readiness and dependency verification.
    """
    
    def __init__(self, storage_dir: Optional[Path] = None):
        """Initialize health checker with storage directory."""
        self.storage_dir = storage_dir or Path.home() / ".cache" / "mimir"
        self.start_time = time.time()
        self.metrics_collector = get_metrics_collector()
        
    async def liveness_check(self) -> Dict[str, Any]:
        """
        Basic liveness check - is the service running?
        
        Returns:
            Dict with basic status information
        """
        uptime_seconds = time.time() - self.start_time
        
        # Update uptime metric
        self.metrics_collector.update_uptime(self.start_time)
        
        # Set health status
        self.metrics_collector.set_health_status("liveness", True)
        
        return {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": uptime_seconds,
            "service": "mimir-repoindex",
            "version": "0.1.0"
        }
    
    async def readiness_check(self) -> Dict[str, Any]:
        """
        Readiness check - is the service ready to accept requests?
        
        Checks storage accessibility, system resources, and critical dependencies.
        """
        checks = {}
        overall_status = "ready"
        
        # Storage directory accessibility
        try:
            ensure_directory(self.storage_dir)
            test_file = self.storage_dir / ".health_check"
            test_file.write_text("health_check")
            test_file.unlink()
            checks["storage"] = {"status": "healthy", "path": str(self.storage_dir)}
            self.metrics_collector.set_health_status("storage", True)
        except Exception as e:
            checks["storage"] = {"status": "unhealthy", "error": str(e)}
            self.metrics_collector.set_health_status("storage", False)
            overall_status = "not_ready"
        
        # Memory check
        try:
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent
            checks["memory"] = {
                "status": "healthy" if memory_usage_percent < 90 else "warning",
                "usage_percent": memory_usage_percent,
                "available_mb": memory.available // 1024 // 1024
            }
            
            # Update memory metrics
            self.metrics_collector.memory_usage_bytes.labels(type="rss").set(memory.used)
            self.metrics_collector.set_health_status("memory", memory_usage_percent < 95)
            
            if memory_usage_percent > 95:
                overall_status = "not_ready"
        except Exception as e:
            checks["memory"] = {"status": "unhealthy", "error": str(e)}
            self.metrics_collector.set_health_status("memory", False)
            overall_status = "not_ready"
        
        # CPU check
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            checks["cpu"] = {
                "status": "healthy" if cpu_percent < 90 else "warning",
                "usage_percent": cpu_percent,
                "cores": psutil.cpu_count()
            }
            
            # Update CPU metrics
            self.metrics_collector.cpu_usage_percent.set(cpu_percent)
            self.metrics_collector.set_health_status("cpu", cpu_percent < 90)
            
        except Exception as e:
            checks["cpu"] = {"status": "unhealthy", "error": str(e)}
            self.metrics_collector.set_health_status("cpu", False)
        
        # Disk space check
        try:
            disk_usage = psutil.disk_usage(str(self.storage_dir))
            disk_free_percent = (disk_usage.free / disk_usage.total) * 100
            checks["disk"] = {
                "status": "healthy" if disk_free_percent > 10 else "warning",
                "free_percent": disk_free_percent,
                "free_gb": disk_usage.free // 1024 // 1024 // 1024
            }
            
            # Update disk metrics
            self.metrics_collector.update_disk_metrics([str(self.storage_dir)])
            self.metrics_collector.set_health_status("disk", disk_free_percent > 10)
            
            if disk_free_percent < 5:
                overall_status = "not_ready"
        except Exception as e:
            checks["disk"] = {"status": "unhealthy", "error": str(e)}
            self.metrics_collector.set_health_status("disk", False)
            overall_status = "not_ready"
        
        # Python environment check
        try:
            import repoindex
            checks["python_env"] = {
                "status": "healthy",
                "version": f"{psutil.sys.version_info.major}.{psutil.sys.version_info.minor}",
                "package_available": True
            }
        except ImportError as e:
            checks["python_env"] = {"status": "unhealthy", "error": str(e)}
            overall_status = "not_ready"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "checks": checks
        }
    
    async def detailed_health(self) -> Dict[str, Any]:
        """
        Detailed health check including pipeline status and metrics.
        
        Provides comprehensive system state for monitoring and debugging.
        """
        basic_health = await self.readiness_check()
        
        # Additional detailed checks
        try:
            # Process information
            process = psutil.Process()
            basic_health["process"] = {
                "pid": process.pid,
                "memory_rss_mb": process.memory_info().rss // 1024 // 1024,
                "memory_vms_mb": process.memory_info().vms // 1024 // 1024,
                "cpu_percent": process.cpu_percent(),
                "num_threads": process.num_threads(),
                "open_files": len(process.open_files()),
                "connections": len(process.connections())
            }
            
            # Pipeline health (if any pipelines are running)
            indexes_dir = self.storage_dir / "indexes"
            if indexes_dir.exists():
                pipeline_status = []
                for index_dir in indexes_dir.iterdir():
                    if index_dir.is_dir():
                        status_file = index_dir / "status.json"
                        if status_file.exists():
                            try:
                                status_data = json.loads(status_file.read_text())
                                pipeline_status.append({
                                    "index_id": index_dir.name,
                                    "state": status_data.get("state", "unknown"),
                                    "progress": status_data.get("progress", 0)
                                })
                            except Exception:
                                pipeline_status.append({
                                    "index_id": index_dir.name,
                                    "state": "error",
                                    "progress": 0
                                })
                
                basic_health["pipelines"] = {
                    "active_count": len(pipeline_status),
                    "status": pipeline_status
                }
            
        except Exception as e:
            logger.warning(f"Error collecting detailed health metrics: {e}")
            basic_health["detailed_check_error"] = str(e)
        
        return basic_health


# Global health checker instance
_health_checker: Optional[HealthChecker] = None


def get_health_checker(storage_dir: Optional[Path] = None) -> HealthChecker:
    """Get or create global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(storage_dir)
    return _health_checker


async def health_check_handler() -> Dict[str, Any]:
    """Simple health check for basic monitoring."""
    checker = get_health_checker()
    return await checker.liveness_check()


async def readiness_check_handler() -> Dict[str, Any]:
    """Readiness check for container orchestration."""
    checker = get_health_checker()
    return await checker.readiness_check()


async def detailed_health_handler() -> Dict[str, Any]:
    """Detailed health check for debugging and monitoring."""
    checker = get_health_checker()
    return await checker.detailed_health()