"""
Hybrid Discovery Stage for Mimir-Lens Integration.

This module enhances the existing FileDiscovery with Lens coordination,
providing intelligent file discovery that leverages both Mimir's deep
git analysis and Lens's high-performance bulk operations.
"""

import asyncio
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field

from .discover import FileDiscovery
from .lens_client import LensIntegrationClient, get_lens_client, LensResponse
from .parallel_processor import ParallelProcessor, get_parallel_processor, TaskPriority
from .stage import AsyncPipelineStage, ProgressCallback
from ..data.schemas import PipelineStage, IndexConfig
from ..util.log import get_logger
from ..util.errors import MimirError

if TYPE_CHECKING:
    from .run import PipelineContext

logger = get_logger(__name__)


@dataclass
class DiscoveryMetrics:
    """Metrics for hybrid discovery operations."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # File counts
    total_files_discovered: int = 0
    lens_indexed_files: int = 0
    mimir_analyzed_files: int = 0
    new_files: int = 0
    modified_files: int = 0
    removed_files: int = 0
    
    # Performance metrics
    discovery_time_ms: float = 0.0
    lens_indexing_time_ms: float = 0.0
    mimir_analysis_time_ms: float = 0.0
    parallel_operations_time_ms: float = 0.0
    
    # Success rates
    lens_success_rate: float = 0.0
    mimir_success_rate: float = 1.0  # Mimir is always available
    
    # Resource usage
    peak_memory_mb: float = 0.0
    concurrent_operations: int = 0
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.start_time:
            total_time = (self.end_time - self.start_time).total_seconds() * 1000
            self.discovery_time_ms = total_time


@dataclass
class DiscoveryResult:
    """Result from hybrid discovery operation."""
    success: bool
    files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    lens_data: Optional[Dict[str, Any]] = None
    mimir_data: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    metrics: DiscoveryMetrics = field(default_factory=DiscoveryMetrics)
    error: Optional[str] = None
    
    def finalize_metrics(self) -> None:
        """Finalize discovery metrics."""
        self.metrics.finalize()


class HybridDiscoveryStage(AsyncPipelineStage):
    """
    Enhanced discovery stage with Lens integration.
    
    This stage coordinates between Mimir's git-aware file discovery
    and Lens's high-performance bulk indexing capabilities to provide
    comprehensive and efficient repository analysis.
    
    Features:
    - Parallel Lens indexing and Mimir analysis
    - Incremental discovery with change detection  
    - Intelligent file prioritization
    - Performance-optimized bulk operations
    - Fallback to Mimir-only mode when Lens unavailable
    """
    
    def __init__(
        self,
        stage_type: PipelineStage = PipelineStage.DISCOVERY,
        concurrency_limit: int = 6,
        lens_client: Optional[LensIntegrationClient] = None,
        enable_lens_indexing: bool = True,
        batch_size: int = 100,
        max_files_per_batch: int = 500
    ):
        """Initialize hybrid discovery stage."""
        super().__init__(stage_type, concurrency_limit)
        
        self.lens_client = lens_client or get_lens_client()
        self.enable_lens_indexing = enable_lens_indexing
        self.batch_size = batch_size
        self.max_files_per_batch = max_files_per_batch
        
        # Internal state
        self._file_discovery: Optional[FileDiscovery] = None
        self._parallel_processor: Optional[ParallelProcessor] = None
        
        logger.info(f"HybridDiscoveryStage initialized with Lens integration: {enable_lens_indexing}")
    
    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """Execute hybrid discovery stage."""
        logger.info("Starting hybrid discovery execution")
        start_time = time.time()
        
        try:
            # Initialize file discovery
            repo_path = getattr(context, 'repo_path', '.')
            self._file_discovery = FileDiscovery(repo_path)
            
            # Get parallel processor
            self._parallel_processor = await get_parallel_processor()
            
            # Execute hybrid discovery
            result = await self._execute_hybrid_discovery(context, progress_callback)
            
            # Store results in context
            if hasattr(context, 'discovery_result'):
                context.discovery_result = result
            
            # Finalize metrics
            result.finalize_metrics()
            execution_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"Hybrid discovery completed in {execution_time:.2f}ms. "
                f"Files discovered: {len(result.files)}"
            )
            
            if not result.success:
                raise MimirError(f"Hybrid discovery failed: {result.error}")
                
        except Exception as e:
            logger.error(f"Hybrid discovery execution failed: {e}")
            raise MimirError(f"Discovery stage failed: {e}")
    
    async def _execute_hybrid_discovery(
        self,
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> DiscoveryResult:
        """Execute the core hybrid discovery logic."""
        metrics = DiscoveryMetrics()
        
        try:
            # Phase 1: Basic file discovery (25% progress)
            self._update_progress(10, progress_callback)
            basic_files = await self._discover_basic_files(context)
            metrics.total_files_discovered = len(basic_files)
            self._update_progress(25, progress_callback)
            
            # Phase 2: Change detection if incremental (35% progress)
            changes = await self._detect_changes(context, basic_files)
            if changes:
                metrics.new_files = len(changes.get('added', []))
                metrics.modified_files = len(changes.get('modified', []))
                metrics.removed_files = len(changes.get('removed', []))
            self._update_progress(35, progress_callback)
            
            # Phase 3: Parallel Lens indexing and Mimir analysis (80% progress)
            if self.enable_lens_indexing:
                parallel_result = await self._execute_parallel_operations(
                    basic_files, context, progress_callback
                )
                metrics.lens_success_rate = parallel_result.get('lens_success_rate', 0.0)
                metrics.lens_indexing_time_ms = parallel_result.get('lens_time_ms', 0.0)
                metrics.mimir_analysis_time_ms = parallel_result.get('mimir_time_ms', 0.0)
            else:
                # Mimir-only analysis
                parallel_result = await self._execute_mimir_only_analysis(
                    basic_files, context
                )
                metrics.mimir_analysis_time_ms = parallel_result.get('mimir_time_ms', 0.0)
            
            self._update_progress(80, progress_callback)
            
            # Phase 4: Result synthesis and finalization (100% progress)
            final_result = await self._synthesize_discovery_results(
                basic_files, changes, parallel_result, metrics
            )
            self._update_progress(100, progress_callback)
            
            logger.info(f"Discovery completed: {len(final_result.files)} files processed")
            return final_result
            
        except Exception as e:
            logger.error(f"Hybrid discovery failed: {e}")
            return DiscoveryResult(
                success=False,
                error=str(e),
                metrics=metrics
            )
    
    async def _discover_basic_files(self, context: "PipelineContext") -> List[str]:
        """Perform basic file discovery using Mimir's FileDiscovery."""
        logger.info("Performing basic file discovery")
        
        if not self._file_discovery:
            raise MimirError("FileDiscovery not initialized")
        
        # Get configuration from context
        extensions = getattr(context, 'file_extensions', None)
        excludes = getattr(context, 'exclude_patterns', None)
        
        # Discover files
        files = await self._file_discovery.discover_files(
            extensions=extensions,
            excludes=excludes
        )
        
        logger.info(f"Discovered {len(files)} files in repository")
        return files
    
    async def _detect_changes(
        self, 
        context: "PipelineContext", 
        current_files: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Detect changes since last indexing if incremental mode enabled."""
        if not getattr(context, 'incremental_mode', False):
            logger.info("Incremental mode disabled, skipping change detection")
            return None
        
        logger.info("Detecting file changes for incremental indexing")
        
        # Get previous state from context or cache
        previous_files = getattr(context, 'previous_files', [])
        previous_hashes = getattr(context, 'previous_hashes', {})
        
        if not previous_files:
            logger.info("No previous state found, treating as full discovery")
            return None
        
        # Detect changes
        added, modified, removed, current_hashes = await self._file_discovery.detect_changes(
            previous_files, previous_hashes
        )
        
        changes = {
            'added': added,
            'modified': modified, 
            'removed': removed,
            'current_hashes': current_hashes,
            'total_changes': len(added) + len(modified) + len(removed)
        }
        
        logger.info(
            f"Change detection complete: {len(added)} added, "
            f"{len(modified)} modified, {len(removed)} removed"
        )
        
        return changes
    
    async def _execute_parallel_operations(
        self,
        files: List[str],
        context: "PipelineContext",
        progress_callback: ProgressCallback | None = None
    ) -> Dict[str, Any]:
        """Execute Lens indexing and Mimir analysis in parallel."""
        logger.info(f"Starting parallel operations for {len(files)} files")
        
        if not self._parallel_processor:
            raise MimirError("ParallelProcessor not initialized")
        
        start_time = time.time()
        
        # Submit parallel tasks
        tasks = {}
        
        # Task 1: Lens bulk indexing
        if await self._check_lens_availability():
            lens_task_id = await self._parallel_processor.submit_task(
                self._lens_bulk_index,
                files, context,
                task_id="lens_indexing",
                priority=TaskPriority.HIGH,
                timeout=120.0
            )
            tasks['lens'] = lens_task_id
        
        # Task 2: Mimir detailed analysis (for priority files)
        priority_files = await self._get_priority_files(files, context)
        mimir_task_id = await self._parallel_processor.submit_task(
            self._mimir_detailed_analysis,
            priority_files, context,
            task_id="mimir_analysis", 
            priority=TaskPriority.NORMAL,
            timeout=180.0
        )
        tasks['mimir'] = mimir_task_id
        
        # Wait for completion with progress updates
        results = {}
        completed_tasks = 0
        total_tasks = len(tasks)
        
        for task_name, task_id in tasks.items():
            try:
                result = await self._parallel_processor.get_result(task_id, timeout=300.0)
                results[task_name] = result
                completed_tasks += 1
                
                # Update progress
                if progress_callback:
                    base_progress = 35  # Starting progress
                    task_progress = int(45 * (completed_tasks / total_tasks))  # 45% for parallel ops
                    progress_callback(base_progress + task_progress)
                    
                logger.info(f"Task {task_name} completed successfully")
                
            except Exception as e:
                logger.error(f"Task {task_name} failed: {e}")
                results[task_name] = {'error': str(e), 'success': False}
                completed_tasks += 1
        
        # Calculate timing metrics
        total_time_ms = (time.time() - start_time) * 1000
        
        return {
            'results': results,
            'lens_success_rate': 1.0 if 'lens' in results and results['lens'].get('success') else 0.0,
            'mimir_success_rate': 1.0 if results.get('mimir', {}).get('success') else 0.0,
            'lens_time_ms': results.get('lens', {}).get('execution_time_ms', 0.0),
            'mimir_time_ms': results.get('mimir', {}).get('execution_time_ms', 0.0),
            'parallel_time_ms': total_time_ms
        }
    
    async def _execute_mimir_only_analysis(
        self,
        files: List[str],
        context: "PipelineContext"
    ) -> Dict[str, Any]:
        """Execute Mimir-only analysis when Lens is unavailable."""
        logger.info(f"Executing Mimir-only analysis for {len(files)} files")
        
        start_time = time.time()
        
        # Get priority files for detailed analysis
        priority_files = await self._get_priority_files(files, context)
        
        # Execute Mimir analysis
        result = await self._mimir_detailed_analysis(priority_files, context)
        
        execution_time_ms = (time.time() - start_time) * 1000
        
        return {
            'results': {'mimir': result},
            'lens_success_rate': 0.0,
            'mimir_success_rate': 1.0 if result.get('success') else 0.0,
            'lens_time_ms': 0.0,
            'mimir_time_ms': execution_time_ms,
            'parallel_time_ms': execution_time_ms
        }
    
    async def _check_lens_availability(self) -> bool:
        """Check if Lens is available for operations."""
        try:
            health_check = await self.lens_client.health_check()
            return health_check.status.value in ['healthy', 'degraded']
        except Exception as e:
            logger.warning(f"Lens availability check failed: {e}")
            return False
    
    async def _lens_bulk_index(
        self, 
        files: List[str], 
        context: "PipelineContext"
    ) -> Dict[str, Any]:
        """Perform bulk indexing using Lens."""
        logger.info(f"Starting Lens bulk indexing for {len(files)} files")
        
        start_time = time.time()
        
        try:
            # Prepare documents for Lens indexing
            documents = await self._prepare_documents_for_lens(files, context)
            
            # Process in batches to avoid overwhelming Lens
            indexed_count = 0
            batch_results = []
            
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i:i + self.batch_size]
                
                try:
                    # Send batch to Lens
                    response = await self.lens_client.bulk_index(
                        documents=batch,
                        operation="discovery_indexing"
                    )
                    
                    if response.success:
                        indexed_count += len(batch)
                        batch_results.append(response.data)
                        logger.debug(f"Indexed batch {i//self.batch_size + 1}: {len(batch)} files")
                    else:
                        logger.warning(f"Batch indexing failed: {response.error}")
                        
                except Exception as e:
                    logger.error(f"Lens batch indexing error: {e}")
                    continue
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'indexed_files': indexed_count,
                'total_files': len(files),
                'batch_results': batch_results,
                'execution_time_ms': execution_time_ms,
                'source': 'lens'
            }
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Lens bulk indexing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time_ms,
                'source': 'lens'
            }
    
    async def _prepare_documents_for_lens(
        self, 
        files: List[str], 
        context: "PipelineContext"
    ) -> List[Dict[str, Any]]:
        """Prepare file documents for Lens indexing."""
        documents = []
        
        # Get file metadata
        if not self._file_discovery:
            raise MimirError("FileDiscovery not initialized")
        
        metadata = await self._file_discovery.get_file_metadata(files)
        
        # Process files in batches to manage memory
        batch_size = 50
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file_path in batch:
                file_meta = metadata.get(file_path, {})
                
                if file_meta.get('error'):
                    continue  # Skip files with metadata errors
                
                # Read file content (with size limits for performance)
                content = await self._read_file_content_safe(file_path, max_size_mb=1.0)
                
                if content:
                    document = {
                        'id': file_path,
                        'path': file_path,
                        'content': content,
                        'metadata': file_meta,
                        'type': 'file',
                        'repository': getattr(context, 'repo_name', 'unknown')
                    }
                    documents.append(document)
        
        logger.info(f"Prepared {len(documents)} documents for Lens indexing")
        return documents
    
    async def _read_file_content_safe(
        self, 
        file_path: str, 
        max_size_mb: float = 1.0
    ) -> Optional[str]:
        """Safely read file content with size limits."""
        if not self._file_discovery:
            return None
        
        try:
            full_path = self._file_discovery.repo_root / file_path
            
            # Check file size
            stat = await asyncio.to_thread(full_path.stat)
            max_size_bytes = int(max_size_mb * 1024 * 1024)
            
            if stat.st_size > max_size_bytes:
                logger.warning(f"File {file_path} too large ({stat.st_size} bytes), skipping content")
                return None
            
            # Check if binary
            is_binary = await self._file_discovery._is_binary_file(full_path)
            if is_binary:
                logger.debug(f"File {file_path} is binary, skipping content")
                return None
            
            # Read content
            content = await asyncio.to_thread(
                full_path.read_text, 
                encoding='utf-8', 
                errors='ignore'
            )
            
            return content
            
        except Exception as e:
            logger.warning(f"Failed to read {file_path}: {e}")
            return None
    
    async def _mimir_detailed_analysis(
        self, 
        files: List[str], 
        context: "PipelineContext"
    ) -> Dict[str, Any]:
        """Perform detailed analysis using Mimir's capabilities."""
        logger.info(f"Starting Mimir detailed analysis for {len(files)} files")
        
        start_time = time.time()
        
        try:
            if not self._file_discovery:
                raise MimirError("FileDiscovery not initialized")
            
            # Get detailed file metadata
            metadata = await self._file_discovery.get_file_metadata(files)
            
            # Perform repository structure analysis
            structure_analysis = await self._file_discovery.validate_repository_structure()
            
            # Detect workspaces for monorepo support
            workspaces = await self._file_discovery.detect_workspaces()
            
            # Generate cache key for incremental indexing
            config = getattr(context, 'config', IndexConfig())
            cache_key = await self._file_discovery.get_cache_key(config)
            
            # Compute dirty overlay for uncommitted changes
            dirty_overlay = await self._file_discovery.compute_dirty_overlay()
            
            execution_time_ms = (time.time() - start_time) * 1000
            
            return {
                'success': True,
                'analyzed_files': len(files),
                'metadata': metadata,
                'structure_analysis': structure_analysis,
                'workspaces': workspaces,
                'cache_key': cache_key,
                'dirty_overlay': dirty_overlay,
                'execution_time_ms': execution_time_ms,
                'source': 'mimir'
            }
            
        except Exception as e:
            execution_time_ms = (time.time() - start_time) * 1000
            logger.error(f"Mimir detailed analysis failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_ms': execution_time_ms,
                'source': 'mimir'
            }
    
    async def _get_priority_files(
        self, 
        all_files: List[str], 
        context: "PipelineContext"
    ) -> List[str]:
        """Get priority files for detailed analysis."""
        if not self._file_discovery:
            return all_files[:100]  # Default limit
        
        max_files = getattr(context, 'max_priority_files', 200)
        
        # Get priority files using Mimir's heuristics
        priority_files = await self._file_discovery.get_priority_files(
            all_files=all_files,
            max_files=max_files
        )
        
        logger.info(f"Selected {len(priority_files)} priority files for detailed analysis")
        return priority_files
    
    async def _synthesize_discovery_results(
        self,
        files: List[str],
        changes: Optional[Dict[str, Any]],
        parallel_result: Dict[str, Any],
        metrics: DiscoveryMetrics
    ) -> DiscoveryResult:
        """Synthesize results from all discovery operations."""
        logger.info("Synthesizing hybrid discovery results")
        
        # Extract results from parallel operations
        results = parallel_result.get('results', {})
        lens_data = results.get('lens')
        mimir_data = results.get('mimir')
        
        # Update metrics
        if lens_data:
            metrics.lens_indexed_files = lens_data.get('indexed_files', 0)
        if mimir_data:
            metrics.mimir_analyzed_files = mimir_data.get('analyzed_files', 0)
        
        metrics.lens_success_rate = parallel_result.get('lens_success_rate', 0.0)
        metrics.lens_indexing_time_ms = parallel_result.get('lens_time_ms', 0.0)
        metrics.mimir_analysis_time_ms = parallel_result.get('mimir_time_ms', 0.0)
        metrics.parallel_operations_time_ms = parallel_result.get('parallel_time_ms', 0.0)
        
        # Create comprehensive metadata
        combined_metadata = {
            'total_files': len(files),
            'discovery_timestamp': datetime.utcnow().isoformat(),
            'hybrid_discovery': True,
            'sources_used': []
        }
        
        if lens_data and lens_data.get('success'):
            combined_metadata['sources_used'].append('lens')
            combined_metadata['lens_indexing'] = {
                'indexed_files': lens_data.get('indexed_files', 0),
                'execution_time_ms': lens_data.get('execution_time_ms', 0.0)
            }
        
        if mimir_data and mimir_data.get('success'):
            combined_metadata['sources_used'].append('mimir')
            combined_metadata['mimir_analysis'] = {
                'analyzed_files': mimir_data.get('analyzed_files', 0),
                'structure_analysis': mimir_data.get('structure_analysis'),
                'workspaces': mimir_data.get('workspaces', []),
                'cache_key': mimir_data.get('cache_key'),
                'execution_time_ms': mimir_data.get('execution_time_ms', 0.0)
            }
        
        # Include change detection results
        if changes:
            combined_metadata['incremental'] = changes
        
        return DiscoveryResult(
            success=True,
            files=files,
            metadata=combined_metadata,
            lens_data=lens_data,
            mimir_data=mimir_data,
            changes=changes,
            metrics=metrics
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid discovery stage metrics."""
        base_metrics = super().get_metrics()
        
        hybrid_metrics = {
            'hybrid_discovery': {
                'lens_integration_enabled': self.enable_lens_indexing,
                'batch_size': self.batch_size,
                'max_files_per_batch': self.max_files_per_batch,
                'file_discovery_initialized': self._file_discovery is not None,
                'parallel_processor_available': self._parallel_processor is not None
            }
        }
        
        base_metrics.update(hybrid_metrics)
        return base_metrics
    
    def _get_capabilities(self) -> List[str]:
        """Get hybrid discovery capabilities."""
        capabilities = super()._get_capabilities()
        capabilities.extend([
            'hybrid_discovery',
            'lens_integration',
            'parallel_operations',
            'incremental_indexing',
            'change_detection',
            'priority_analysis',
            'bulk_indexing',
            'repository_analysis'
        ])
        return capabilities
    
    async def cleanup(self, context: "PipelineContext") -> None:
        """Clean up hybrid discovery resources."""
        logger.info("Cleaning up hybrid discovery stage")
        
        # Reset internal state
        self._file_discovery = None
        self._parallel_processor = None
        
        await super().cleanup(context)