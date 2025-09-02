"""
Async Parallel Processing Manager for Hybrid Operations.

This module provides advanced async coordination capabilities for running
Lens and Mimir operations in parallel, with sophisticated load balancing,
resource management, and error handling.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import (
    Any, Callable, Dict, List, Optional, Union, Tuple, 
    Coroutine, TypeVar, Generic, Awaitable
)
from contextlib import asynccontextmanager
import weakref
from collections import defaultdict

from ..util.log import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class TaskPriority(Enum):
    """Task priority levels for execution scheduling."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskInfo:
    """Information about a task in the parallel processor."""
    task_id: str
    priority: TaskPriority
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    error: Optional[str] = None
    result: Any = None
    retry_count: int = 0
    max_retries: int = 3
    
    @property
    def execution_time_ms(self) -> float:
        """Get task execution time in milliseconds."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0
    
    @property
    def total_time_ms(self) -> float:
        """Get total time from creation to completion in milliseconds."""
        if self.completed_at:
            return (self.completed_at - self.created_at).total_seconds() * 1000
        return 0.0


@dataclass
class ResourceLimits:
    """Resource limits for parallel processing."""
    max_concurrent_tasks: int = 10
    max_memory_mb: int = 1024
    max_cpu_percent: float = 80.0
    max_execution_time_s: float = 300.0  # 5 minutes
    max_queue_size: int = 1000


@dataclass
class ProcessingMetrics:
    """Metrics for parallel processing operations."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_cancelled: int = 0
    total_execution_time_ms: float = 0.0
    average_task_time_ms: float = 0.0
    peak_concurrent_tasks: int = 0
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.tasks_completed > 0:
            self.average_task_time_ms = (
                self.total_execution_time_ms / self.tasks_completed
            )


class ParallelProcessor:
    """
    Advanced async parallel processor for Lens-Mimir hybrid operations.
    
    Features:
    - Priority-based task scheduling
    - Resource limits and throttling
    - Automatic retry with backoff
    - Performance monitoring
    - Graceful shutdown
    """
    
    def __init__(
        self,
        resource_limits: Optional[ResourceLimits] = None,
        enable_metrics: bool = True
    ):
        """Initialize parallel processor."""
        self.resource_limits = resource_limits or ResourceLimits()
        self.enable_metrics = enable_metrics
        
        # Task management
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=self.resource_limits.max_queue_size
        )
        self.active_tasks: Dict[str, TaskInfo] = {}
        self.completed_tasks: Dict[str, TaskInfo] = {}
        self.task_futures: Dict[str, asyncio.Future] = {}
        
        # Concurrency control
        self.semaphore = asyncio.Semaphore(
            self.resource_limits.max_concurrent_tasks
        )
        
        # Thread pool for CPU-intensive operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=min(4, self.resource_limits.max_concurrent_tasks)
        )
        
        # Processing state
        self.is_running = False
        self.shutdown_event = asyncio.Event()
        self.worker_tasks: List[asyncio.Task] = []
        
        # Metrics
        self.metrics = ProcessingMetrics() if enable_metrics else None
        
        # Weak references to avoid memory leaks
        self._cleanup_registry = weakref.WeakSet()
        
        logger.info(f"ParallelProcessor initialized with limits: {resource_limits}")
    
    async def start(self) -> None:
        """Start the parallel processor."""
        if self.is_running:
            logger.warning("ParallelProcessor already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start worker tasks
        num_workers = min(
            4, 
            self.resource_limits.max_concurrent_tasks
        )
        
        self.worker_tasks = [
            asyncio.create_task(self._worker(f"worker-{i}"))
            for i in range(num_workers)
        ]
        
        logger.info(f"ParallelProcessor started with {num_workers} workers")
    
    async def stop(self) -> None:
        """Stop the parallel processor gracefully."""
        if not self.is_running:
            return
        
        logger.info("Stopping ParallelProcessor...")
        self.is_running = False
        self.shutdown_event.set()
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            if not task.done():
                task.cancel()
        
        # Wait for workers to finish
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        # Finalize metrics
        if self.metrics:
            self.metrics.finalize()
        
        logger.info("ParallelProcessor stopped")
    
    async def submit_task(
        self,
        task_func: Callable[..., Awaitable[T]],
        *args,
        task_id: Optional[str] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        timeout: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Submit a task for parallel execution.
        
        Args:
            task_func: Async function to execute
            *args: Arguments for task_func
            task_id: Optional unique task identifier
            priority: Task priority level
            max_retries: Maximum retry attempts
            timeout: Task timeout in seconds
            **kwargs: Keyword arguments for task_func
            
        Returns:
            Task ID string
        """
        if not self.is_running:
            raise RuntimeError("ParallelProcessor not running")
        
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task-{int(time.time() * 1000000)}"
        
        # Create task info
        task_info = TaskInfo(
            task_id=task_id,
            priority=priority,
            max_retries=max_retries
        )
        
        # Create task package
        task_package = (
            priority.value,  # Priority for queue ordering (lower number = higher priority)
            time.time(),     # Timestamp for FIFO within same priority
            {
                "task_info": task_info,
                "task_func": task_func,
                "args": args,
                "kwargs": kwargs,
                "timeout": timeout or self.resource_limits.max_execution_time_s
            }
        )
        
        # Add to queue
        await self.task_queue.put(task_package)
        
        # Track task
        self.active_tasks[task_id] = task_info
        
        # Create future for result
        future = asyncio.Future()
        self.task_futures[task_id] = future
        
        # Update metrics
        if self.metrics:
            self.metrics.tasks_submitted += 1
        
        logger.debug(f"Task {task_id} submitted with priority {priority.value}")
        return task_id
    
    async def get_result(
        self, 
        task_id: str, 
        timeout: Optional[float] = None
    ) -> Any:
        """
        Wait for and get task result.
        
        Args:
            task_id: Task identifier
            timeout: Wait timeout in seconds
            
        Returns:
            Task result
            
        Raises:
            TimeoutError: If timeout exceeded
            RuntimeError: If task failed
        """
        if task_id not in self.task_futures:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        future = self.task_futures[task_id]
        
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} result timeout after {timeout}s")
            raise
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            raise RuntimeError(f"Task {task_id} failed: {e}")
    
    async def wait_for_completion(
        self, 
        task_ids: Optional[List[str]] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Wait for multiple tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for (None = all active)
            timeout: Total wait timeout in seconds
            
        Returns:
            Dictionary mapping task_id -> result
        """
        if task_ids is None:
            task_ids = list(self.task_futures.keys())
        
        results = {}
        futures = {
            task_id: self.task_futures[task_id] 
            for task_id in task_ids 
            if task_id in self.task_futures
        }
        
        if not futures:
            return results
        
        try:
            # Wait for all futures
            done, pending = await asyncio.wait(
                futures.values(),
                timeout=timeout,
                return_when=asyncio.ALL_COMPLETED
            )
            
            # Collect results
            for task_id, future in futures.items():
                if future in done:
                    try:
                        results[task_id] = await future
                    except Exception as e:
                        results[task_id] = f"ERROR: {e}"
                else:
                    results[task_id] = "TIMEOUT"
            
        except Exception as e:
            logger.error(f"Error waiting for task completion: {e}")
            raise
        
        return results
    
    async def _worker(self, worker_name: str) -> None:
        """Worker coroutine for processing tasks."""
        logger.info(f"Worker {worker_name} started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get task from queue with timeout
                try:
                    _, _, task_package = await asyncio.wait_for(
                        self.task_queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Extract task components
                task_info = task_package["task_info"]
                task_func = task_package["task_func"]
                args = task_package["args"]
                kwargs = task_package["kwargs"]
                timeout = task_package["timeout"]
                
                # Execute task
                await self._execute_task(
                    worker_name,
                    task_info,
                    task_func,
                    args,
                    kwargs,
                    timeout
                )
                
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                continue
        
        logger.info(f"Worker {worker_name} stopped")
    
    async def _execute_task(
        self,
        worker_name: str,
        task_info: TaskInfo,
        task_func: Callable,
        args: tuple,
        kwargs: dict,
        timeout: float
    ) -> None:
        """Execute a single task with retry logic."""
        task_id = task_info.task_id
        
        async with self.semaphore:  # Concurrency limiting
            task_info.status = TaskStatus.RUNNING
            task_info.started_at = datetime.utcnow()
            
            # Update peak concurrent tasks
            if self.metrics:
                current_active = len([
                    t for t in self.active_tasks.values() 
                    if t.status == TaskStatus.RUNNING
                ])
                self.metrics.peak_concurrent_tasks = max(
                    self.metrics.peak_concurrent_tasks,
                    current_active
                )
            
            logger.debug(f"Worker {worker_name} executing task {task_id}")
            
            for attempt in range(task_info.max_retries + 1):
                try:
                    # Execute task with timeout
                    result = await asyncio.wait_for(
                        task_func(*args, **kwargs),
                        timeout=timeout
                    )
                    
                    # Task succeeded
                    task_info.status = TaskStatus.COMPLETED
                    task_info.completed_at = datetime.utcnow()
                    task_info.result = result
                    
                    # Update metrics
                    if self.metrics:
                        self.metrics.tasks_completed += 1
                        self.metrics.total_execution_time_ms += task_info.execution_time_ms
                    
                    # Set future result
                    if task_id in self.task_futures:
                        future = self.task_futures[task_id]
                        if not future.done():
                            future.set_result(result)
                    
                    logger.debug(f"Task {task_id} completed in {task_info.execution_time_ms:.2f}ms")
                    break
                    
                except asyncio.TimeoutError:
                    error_msg = f"Task timeout after {timeout}s"
                    logger.warning(f"Task {task_id} attempt {attempt + 1}: {error_msg}")
                    
                    if attempt >= task_info.max_retries:
                        task_info.status = TaskStatus.FAILED
                        task_info.completed_at = datetime.utcnow()
                        task_info.error = error_msg
                        
                        if self.metrics:
                            self.metrics.tasks_failed += 1
                        
                        # Set future exception
                        if task_id in self.task_futures:
                            future = self.task_futures[task_id]
                            if not future.done():
                                future.set_exception(asyncio.TimeoutError(error_msg))
                    else:
                        # Exponential backoff for retry
                        delay = min(2 ** attempt, 30)
                        await asyncio.sleep(delay)
                        task_info.retry_count += 1
                
                except Exception as e:
                    error_msg = f"Task execution error: {e}"
                    logger.warning(f"Task {task_id} attempt {attempt + 1}: {error_msg}")
                    
                    if attempt >= task_info.max_retries:
                        task_info.status = TaskStatus.FAILED
                        task_info.completed_at = datetime.utcnow()
                        task_info.error = error_msg
                        
                        if self.metrics:
                            self.metrics.tasks_failed += 1
                        
                        # Set future exception
                        if task_id in self.task_futures:
                            future = self.task_futures[task_id]
                            if not future.done():
                                future.set_exception(e)
                    else:
                        # Exponential backoff for retry
                        delay = min(2 ** attempt, 30)
                        await asyncio.sleep(delay)
                        task_info.retry_count += 1
            
            # Move to completed tasks
            if task_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                
                # Clean up future reference
                if task_id in self.task_futures:
                    del self.task_futures[task_id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current processor status."""
        active_count = len(self.active_tasks)
        completed_count = len(self.completed_tasks)
        
        status = {
            "is_running": self.is_running,
            "active_tasks": active_count,
            "completed_tasks": completed_count,
            "queue_size": self.task_queue.qsize(),
            "worker_count": len(self.worker_tasks),
            "resource_limits": {
                "max_concurrent_tasks": self.resource_limits.max_concurrent_tasks,
                "max_queue_size": self.resource_limits.max_queue_size,
            }
        }
        
        if self.metrics:
            status["metrics"] = {
                "tasks_submitted": self.metrics.tasks_submitted,
                "tasks_completed": self.metrics.tasks_completed,
                "tasks_failed": self.metrics.tasks_failed,
                "average_task_time_ms": self.metrics.average_task_time_ms,
                "peak_concurrent_tasks": self.metrics.peak_concurrent_tasks,
            }
        
        return status
    
    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """Get information about a specific task."""
        if task_id in self.active_tasks:
            return self.active_tasks[task_id]
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        return None
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task if it's still pending or running."""
        if task_id in self.task_futures:
            future = self.task_futures[task_id]
            if not future.done():
                future.cancel()
                
                if task_id in self.active_tasks:
                    task_info = self.active_tasks[task_id]
                    task_info.status = TaskStatus.CANCELLED
                    task_info.completed_at = datetime.utcnow()
                    
                    self.completed_tasks[task_id] = self.active_tasks.pop(task_id)
                    
                    if self.metrics:
                        self.metrics.tasks_cancelled += 1
                
                logger.info(f"Task {task_id} cancelled")
                return True
        
        return False
    
    def cleanup_completed_tasks(self, older_than: Optional[timedelta] = None) -> int:
        """
        Clean up old completed tasks to free memory.
        
        Args:
            older_than: Remove tasks older than this duration (default: 1 hour)
            
        Returns:
            Number of tasks cleaned up
        """
        if older_than is None:
            older_than = timedelta(hours=1)
        
        cutoff_time = datetime.utcnow() - older_than
        to_remove = []
        
        for task_id, task_info in self.completed_tasks.items():
            if task_info.completed_at and task_info.completed_at < cutoff_time:
                to_remove.append(task_id)
        
        for task_id in to_remove:
            del self.completed_tasks[task_id]
        
        logger.info(f"Cleaned up {len(to_remove)} completed tasks")
        return len(to_remove)
    
    @asynccontextmanager
    async def managed_execution(self):
        """Context manager for automatic start/stop of processor."""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


# Global parallel processor instance
_global_parallel_processor: Optional[ParallelProcessor] = None


async def get_parallel_processor(
    resource_limits: Optional[ResourceLimits] = None
) -> ParallelProcessor:
    """Get or create global parallel processor instance."""
    global _global_parallel_processor
    
    if _global_parallel_processor is None:
        _global_parallel_processor = ParallelProcessor(resource_limits)
        await _global_parallel_processor.start()
    
    return _global_parallel_processor


async def shutdown_parallel_processor():
    """Shutdown global parallel processor."""
    global _global_parallel_processor
    
    if _global_parallel_processor is not None:
        await _global_parallel_processor.stop()
        _global_parallel_processor = None