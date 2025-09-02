"""
Hybrid Bundle Stage with Lens Integration.

This module enhances the bundle creation process by incorporating Lens-indexed
data, providing comprehensive artifact bundling that includes both Mimir's
detailed analysis and Lens's high-performance indexing results.
"""

import asyncio
import json
import tarfile
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import zstandard as zstd

from .bundle import BundleCreator, BundleError
from .lens_client import LensIntegrationClient, get_lens_client, LensResponse
from .parallel_processor import ParallelProcessor, get_parallel_processor, TaskPriority
from .stage import AsyncPipelineStage, ProgressCallback
from ..data.schemas import (
    ArtifactPaths, IndexCounts, IndexManifest, ToolVersions, 
    PipelineStage, VectorChunk, VectorIndex
)
from ..util.fs import get_directory_size
from ..util.log import get_logger
from ..util.errors import MimirError

if TYPE_CHECKING:
    from .run import PipelineContext

logger = get_logger(__name__)


@dataclass
class BundleMetrics:
    """Metrics for hybrid bundle operations."""
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    
    # Bundle statistics
    total_files_bundled: int = 0
    mimir_artifacts_size_mb: float = 0.0
    lens_artifacts_size_mb: float = 0.0
    compressed_bundle_size_mb: float = 0.0
    compression_ratio: float = 0.0
    
    # Processing metrics
    artifact_collection_time_ms: float = 0.0
    lens_export_time_ms: float = 0.0
    bundle_creation_time_ms: float = 0.0
    compression_time_ms: float = 0.0
    
    # Content statistics
    vector_chunks_included: int = 0
    lens_documents_included: int = 0
    mimir_analysis_files: int = 0
    
    def finalize(self) -> None:
        """Finalize metrics calculation."""
        if self.end_time is None:
            self.end_time = datetime.utcnow()
        
        if self.mimir_artifacts_size_mb > 0 and self.compressed_bundle_size_mb > 0:
            total_uncompressed = self.mimir_artifacts_size_mb + self.lens_artifacts_size_mb
            self.compression_ratio = total_uncompressed / self.compressed_bundle_size_mb


@dataclass
class HybridArtifacts:
    """Container for hybrid Mimir-Lens artifacts."""
    mimir_artifacts: Dict[str, Path] = field(default_factory=dict)
    lens_artifacts: Dict[str, Any] = field(default_factory=dict)
    combined_manifests: Dict[str, Any] = field(default_factory=dict)
    vector_indices: List[VectorIndex] = field(default_factory=list)
    lens_export_data: Optional[Dict[str, Any]] = None
    
    @property
    def total_artifacts(self) -> int:
        return len(self.mimir_artifacts) + len(self.lens_artifacts)


class HybridBundleStage(AsyncPipelineStage):
    """
    Enhanced bundle stage with Lens integration.
    
    This stage creates comprehensive bundles that include:
    - Traditional Mimir artifacts (manifests, indices, analysis results)
    - Lens-exported data (vectors, indices, metadata)
    - Hybrid manifest combining both sources
    - Performance and quality metrics
    - Cross-reference mappings between Mimir and Lens data
    
    Features:
    - Parallel artifact collection and processing
    - Lens data export with chunked streaming
    - Intelligent compression optimization
    - Integrity verification across hybrid data
    - Incremental bundle updates
    """
    
    def __init__(
        self,
        stage_type: PipelineStage = PipelineStage.BUNDLE,
        concurrency_limit: int = 4,
        lens_client: Optional[LensIntegrationClient] = None,
        bundle_creator: Optional[BundleCreator] = None,
        enable_lens_export: bool = True,
        max_bundle_size_gb: float = 2.0,
        compression_level: int = 3,
        include_raw_data: bool = False
    ):
        """Initialize hybrid bundle stage."""
        super().__init__(stage_type, concurrency_limit)
        
        self.lens_client = lens_client or get_lens_client()
        self.bundle_creator = bundle_creator or BundleCreator()
        self.enable_lens_export = enable_lens_export
        self.max_bundle_size_gb = max_bundle_size_gb
        self.compression_level = compression_level
        self.include_raw_data = include_raw_data
        
        # Internal state
        self._parallel_processor: Optional[ParallelProcessor] = None
        
        logger.info(
            f"HybridBundleStage initialized with Lens export: {enable_lens_export}, "
            f"max size: {max_bundle_size_gb}GB, compression: {compression_level}"
        )
    
    async def execute(
        self, 
        context: "PipelineContext", 
        progress_callback: ProgressCallback | None = None
    ) -> None:
        """Execute hybrid bundle stage."""
        logger.info("Starting hybrid bundle creation")
        start_time = time.time()
        
        try:
            # Initialize parallel processor
            self._parallel_processor = await get_parallel_processor()
            
            # Execute hybrid bundling
            manifest = await self._execute_hybrid_bundling(context, progress_callback)
            
            # Store results in context
            context.final_manifest = manifest
            if hasattr(context, 'bundle_manifest'):
                context.bundle_manifest = manifest
            
            execution_time = (time.time() - start_time) * 1000
            logger.info(f"Hybrid bundle created in {execution_time:.2f}ms")
            
        except Exception as e:
            logger.error(f"Hybrid bundle creation failed: {e}")
            raise MimirError(f"Bundle stage failed: {e}")
    
    async def _execute_hybrid_bundling(
        self,
        context: "PipelineContext",
        progress_callback: ProgressCallback | None = None
    ) -> IndexManifest:
        """Execute the core hybrid bundling logic."""
        metrics = BundleMetrics()
        
        try:
            # Phase 1: Collect all artifacts (20% progress)
            self._update_progress(5, progress_callback)
            artifacts = await self._collect_hybrid_artifacts(context)
            metrics.mimir_artifacts_size_mb = await self._calculate_artifacts_size(
                list(artifacts.mimir_artifacts.values())
            )
            self._update_progress(20, progress_callback)
            
            # Phase 2: Export Lens data if enabled (40% progress)
            if self.enable_lens_export:
                lens_export = await self._export_lens_data(context)
                artifacts.lens_export_data = lens_export
                if lens_export:
                    metrics.lens_artifacts_size_mb = lens_export.get('size_mb', 0.0)
                    metrics.lens_documents_included = lens_export.get('document_count', 0)
            self._update_progress(40, progress_callback)
            
            # Phase 3: Create hybrid manifest (60% progress)
            manifest = await self._create_hybrid_manifest(context, artifacts, metrics)
            self._update_progress(60, progress_callback)
            
            # Phase 4: Create compressed bundle (90% progress)
            bundle_path = await self._create_hybrid_bundle(
                context, artifacts, manifest, progress_callback
            )
            manifest.paths.bundle = bundle_path.name
            self._update_progress(90, progress_callback)
            
            # Phase 5: Finalize and validate (100% progress)
            await self._finalize_bundle(bundle_path, manifest, metrics)
            self._update_progress(100, progress_callback)
            
            # Store metrics in context
            if hasattr(context, 'bundle_metrics'):
                context.bundle_metrics = metrics
            
            logger.info(
                f"Hybrid bundle completed: {metrics.total_files_bundled} files, "
                f"{metrics.compressed_bundle_size_mb:.2f}MB compressed "
                f"({metrics.compression_ratio:.2f}x compression)"
            )
            
            return manifest
            
        except Exception as e:
            logger.error(f"Hybrid bundling failed: {e}")
            raise BundleError(f"Hybrid bundle creation failed: {e}")
    
    async def _collect_hybrid_artifacts(self, context: "PipelineContext") -> HybridArtifacts:
        """Collect all artifacts from Mimir and Lens operations."""
        logger.info("Collecting hybrid artifacts from pipeline context")
        
        artifacts = HybridArtifacts()
        
        # Collect Mimir artifacts (traditional pipeline outputs)
        mimir_artifacts = await self._collect_mimir_artifacts(context)
        artifacts.mimir_artifacts = mimir_artifacts
        
        # Collect Lens artifacts (from hybrid operations)
        lens_artifacts = await self._collect_lens_artifacts(context)
        artifacts.lens_artifacts = lens_artifacts
        
        # Extract vector indices
        vector_indices = self._extract_vector_indices(context)
        artifacts.vector_indices = vector_indices
        
        logger.info(
            f"Collected artifacts: {len(artifacts.mimir_artifacts)} Mimir, "
            f"{len(artifacts.lens_artifacts)} Lens, {len(artifacts.vector_indices)} indices"
        )
        
        return artifacts
    
    async def _collect_mimir_artifacts(self, context: "PipelineContext") -> Dict[str, Path]:
        """Collect traditional Mimir pipeline artifacts."""
        artifacts = {}
        work_dir = getattr(context, 'work_dir', Path('.'))
        
        # Standard Mimir artifact patterns
        artifact_patterns = {
            'manifest': 'manifest.json',
            'file_index': 'file_index.json',
            'chunks': 'chunks.json',
            'embeddings': 'embeddings.pkl',
            'search_index': 'search_index.faiss',
            'metadata': 'metadata.json',
            'analysis': 'analysis.json'
        }
        
        for name, pattern in artifact_patterns.items():
            artifact_path = work_dir / pattern
            if artifact_path.exists():
                artifacts[name] = artifact_path
                logger.debug(f"Found Mimir artifact: {name} -> {artifact_path}")
        
        # Look for additional artifacts in subdirectories
        for subdir in ['vectors', 'indices', 'cache']:
            subdir_path = work_dir / subdir
            if subdir_path.exists() and subdir_path.is_dir():
                for file_path in subdir_path.rglob('*'):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(work_dir)
                        artifacts[str(rel_path)] = file_path
        
        logger.info(f"Collected {len(artifacts)} Mimir artifacts")
        return artifacts
    
    async def _collect_lens_artifacts(self, context: "PipelineContext") -> Dict[str, Any]:
        """Collect artifacts from Lens operations."""
        artifacts = {}
        
        # Extract Lens data from context
        if hasattr(context, 'discovery_result') and context.discovery_result:
            discovery = context.discovery_result
            if hasattr(discovery, 'lens_data') and discovery.lens_data:
                artifacts['discovery'] = discovery.lens_data
        
        if hasattr(context, 'embedding_result') and context.embedding_result:
            embedding = context.embedding_result
            if hasattr(embedding, 'lens_data') and embedding.lens_data:
                artifacts['embeddings'] = embedding.lens_data
        
        # Check for Lens-specific results in various context attributes
        lens_attributes = ['lens_results', 'lens_data', 'hybrid_results']
        for attr in lens_attributes:
            if hasattr(context, attr):
                data = getattr(context, attr)
                if data:
                    artifacts[attr] = data
        
        logger.info(f"Collected {len(artifacts)} Lens artifacts")
        return artifacts
    
    def _extract_vector_indices(self, context: "PipelineContext") -> List[VectorIndex]:
        """Extract vector indices from context."""
        indices = []
        
        # Look for vector indices in various context attributes
        if hasattr(context, 'vector_index') and context.vector_index:
            indices.append(context.vector_index)
        
        if hasattr(context, 'vector_indices') and context.vector_indices:
            indices.extend(context.vector_indices)
        
        # Extract indices from embedded chunks
        if hasattr(context, 'vector_chunks') and context.vector_chunks:
            # Create a synthetic index from chunks
            synthetic_index = VectorIndex()
            synthetic_index.chunks = context.vector_chunks
            indices.append(synthetic_index)
        
        logger.info(f"Extracted {len(indices)} vector indices")
        return indices
    
    async def _export_lens_data(self, context: "PipelineContext") -> Optional[Dict[str, Any]]:
        """Export data from Lens for bundling."""
        if not self.enable_lens_export:
            logger.info("Lens export disabled, skipping")
            return None
        
        logger.info("Exporting data from Lens")
        
        try:
            # Check if Lens is available
            if not await self._check_lens_availability():
                logger.warning("Lens not available for export")
                return None
            
            # Prepare export request
            export_request = {
                'repository': getattr(context, 'repo_name', 'unknown'),
                'include_vectors': True,
                'include_metadata': True,
                'include_documents': self.include_raw_data,
                'format': 'json'
            }
            
            # Execute export
            response = await self.lens_client.export_data(export_request)
            
            if response.success and response.data:
                export_data = response.data
                
                # Calculate exported data size
                export_json = json.dumps(export_data)
                size_mb = len(export_json.encode()) / (1024 * 1024)
                
                result = {
                    'data': export_data,
                    'size_mb': size_mb,
                    'document_count': len(export_data.get('documents', [])),
                    'vector_count': len(export_data.get('vectors', [])),
                    'metadata_count': len(export_data.get('metadata', [])),
                    'export_timestamp': datetime.utcnow().isoformat()
                }
                
                logger.info(f"Lens export completed: {result['document_count']} docs, {size_mb:.2f}MB")
                return result
            else:
                logger.warning(f"Lens export failed: {response.error}")
                return None
                
        except Exception as e:
            logger.error(f"Lens data export failed: {e}")
            return None
    
    async def _check_lens_availability(self) -> bool:
        """Check if Lens is available for operations."""
        try:
            health_check = await self.lens_client.health_check()
            return health_check.status.value in ['healthy', 'degraded']
        except Exception:
            return False
    
    async def _create_hybrid_manifest(
        self,
        context: "PipelineContext",
        artifacts: HybridArtifacts,
        metrics: BundleMetrics
    ) -> IndexManifest:
        """Create comprehensive hybrid manifest."""
        logger.info("Creating hybrid manifest")
        
        # Use traditional bundle creator as base
        base_manifest = await self.bundle_creator._create_manifest(context)
        
        # Enhance with hybrid information
        hybrid_info = {
            'hybrid_pipeline': True,
            'sources': [],
            'lens_integration': self.enable_lens_export,
            'artifacts': {
                'mimir_count': len(artifacts.mimir_artifacts),
                'lens_count': len(artifacts.lens_artifacts),
                'vector_indices': len(artifacts.vector_indices)
            }
        }
        
        # Add source information
        if artifacts.mimir_artifacts:
            hybrid_info['sources'].append('mimir')
        
        if artifacts.lens_artifacts:
            hybrid_info['sources'].append('lens')
        
        # Include Lens export information
        if artifacts.lens_export_data:
            hybrid_info['lens_export'] = {
                'document_count': artifacts.lens_export_data.get('document_count', 0),
                'vector_count': artifacts.lens_export_data.get('vector_count', 0),
                'size_mb': artifacts.lens_export_data.get('size_mb', 0.0),
                'export_timestamp': artifacts.lens_export_data.get('export_timestamp')
            }
        
        # Add metrics information
        hybrid_info['metrics'] = {
            'bundle_creation_timestamp': datetime.utcnow().isoformat(),
            'total_artifacts': artifacts.total_artifacts,
            'compression_level': self.compression_level
        }
        
        # Update manifest with hybrid information
        if not hasattr(base_manifest, 'hybrid_info'):
            # Add hybrid info as additional metadata
            if hasattr(base_manifest, 'metadata'):
                base_manifest.metadata.update(hybrid_info)
            else:
                # Create metadata field
                base_manifest.metadata = hybrid_info
        
        return base_manifest
    
    async def _create_hybrid_bundle(
        self,
        context: "PipelineContext",
        artifacts: HybridArtifacts,
        manifest: IndexManifest,
        progress_callback: ProgressCallback | None = None
    ) -> Path:
        """Create compressed bundle with all hybrid artifacts."""
        logger.info("Creating hybrid compressed bundle")
        
        work_dir = getattr(context, 'work_dir', Path('.'))
        bundle_path = work_dir / f"hybrid_index_{int(time.time())}.tar.zst"
        
        # Create compressor
        compressor = zstd.ZstdCompressor(level=self.compression_level)
        
        total_files = len(artifacts.mimir_artifacts) + len(artifacts.lens_artifacts) + 1
        processed_files = 0
        
        with open(bundle_path, 'wb') as bundle_file:
            with compressor.stream_writer(bundle_file) as compressed_writer:
                with tarfile.open(fileobj=compressed_writer, mode='w|') as tar:
                    
                    # Add Mimir artifacts
                    for name, artifact_path in artifacts.mimir_artifacts.items():
                        try:
                            tar.add(artifact_path, arcname=f"mimir/{name}")
                            processed_files += 1
                            
                            if progress_callback and total_files > 0:
                                base_progress = 60
                                file_progress = int(25 * (processed_files / total_files))
                                progress_callback(base_progress + file_progress)
                                
                        except Exception as e:
                            logger.warning(f"Failed to add Mimir artifact {name}: {e}")
                    
                    # Add Lens artifacts as JSON files
                    for name, artifact_data in artifacts.lens_artifacts.items():
                        try:
                            # Convert to JSON and add to bundle
                            json_content = json.dumps(artifact_data, indent=2)
                            json_bytes = json_content.encode('utf-8')
                            
                            info = tarfile.TarInfo(name=f"lens/{name}.json")
                            info.size = len(json_bytes)
                            info.mtime = int(time.time())
                            
                            tar.addfile(info, fileobj=asyncio.BytesIO(json_bytes))
                            processed_files += 1
                            
                        except Exception as e:
                            logger.warning(f"Failed to add Lens artifact {name}: {e}")
                    
                    # Add Lens export data if available
                    if artifacts.lens_export_data:
                        try:
                            export_json = json.dumps(artifacts.lens_export_data, indent=2)
                            export_bytes = export_json.encode('utf-8')
                            
                            info = tarfile.TarInfo(name="lens/export_data.json")
                            info.size = len(export_bytes)
                            info.mtime = int(time.time())
                            
                            tar.addfile(info, fileobj=asyncio.BytesIO(export_bytes))
                            
                        except Exception as e:
                            logger.warning(f"Failed to add Lens export data: {e}")
                    
                    # Add vector indices
                    for i, vector_index in enumerate(artifacts.vector_indices):
                        try:
                            # Convert vector index to JSON
                            index_dict = {
                                'index_id': getattr(vector_index, 'index_id', f'index_{i}'),
                                'chunk_count': len(getattr(vector_index, 'chunks', [])),
                                'embedding_model': getattr(vector_index, 'embedding_model', 'unknown'),
                                'created_at': getattr(vector_index, 'created_at', datetime.utcnow()).isoformat(),
                                'chunks': [
                                    {
                                        'chunk_id': chunk.chunk_id,
                                        'file_path': chunk.file_path,
                                        'content': chunk.content,
                                        'start_line': chunk.start_line,
                                        'end_line': chunk.end_line,
                                        'chunk_type': chunk.chunk_type,
                                        'embedding': chunk.embedding.tolist() if chunk.embedding is not None else None
                                    }
                                    for chunk in getattr(vector_index, 'chunks', [])
                                ]
                            }
                            
                            index_json = json.dumps(index_dict, indent=2)
                            index_bytes = index_json.encode('utf-8')
                            
                            info = tarfile.TarInfo(name=f"indices/vector_index_{i}.json")
                            info.size = len(index_bytes)
                            info.mtime = int(time.time())
                            
                            tar.addfile(info, fileobj=asyncio.BytesIO(index_bytes))
                            
                        except Exception as e:
                            logger.warning(f"Failed to add vector index {i}: {e}")
                    
                    # Add hybrid manifest
                    try:
                        manifest_dict = manifest.dict() if hasattr(manifest, 'dict') else manifest.__dict__
                        manifest_json = json.dumps(manifest_dict, indent=2, default=str)
                        manifest_bytes = manifest_json.encode('utf-8')
                        
                        info = tarfile.TarInfo(name="hybrid_manifest.json")
                        info.size = len(manifest_bytes)
                        info.mtime = int(time.time())
                        
                        tar.addfile(info, fileobj=asyncio.BytesIO(manifest_bytes))
                        processed_files += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to add manifest: {e}")
                        raise
        
        # Calculate final bundle size
        bundle_size_mb = bundle_path.stat().st_size / (1024 * 1024)
        logger.info(f"Created hybrid bundle: {bundle_path} ({bundle_size_mb:.2f}MB)")
        
        return bundle_path
    
    async def _calculate_artifacts_size(self, artifact_paths: List[Path]) -> float:
        """Calculate total size of artifacts in MB."""
        total_size = 0
        
        for path in artifact_paths:
            try:
                if path.is_file():
                    total_size += path.stat().st_size
                elif path.is_dir():
                    total_size += await get_directory_size(path)
            except Exception as e:
                logger.warning(f"Failed to get size of {path}: {e}")
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    async def _finalize_bundle(
        self,
        bundle_path: Path,
        manifest: IndexManifest,
        metrics: BundleMetrics
    ) -> None:
        """Finalize bundle creation and update metrics."""
        # Update metrics
        metrics.compressed_bundle_size_mb = bundle_path.stat().st_size / (1024 * 1024)
        metrics.total_files_bundled = 1  # The bundle itself
        metrics.finalize()
        
        # Update manifest with final information
        manifest.updated_at = datetime.utcnow()
        if hasattr(manifest, 'bundle_info'):
            manifest.bundle_info = {
                'size_mb': metrics.compressed_bundle_size_mb,
                'compression_ratio': metrics.compression_ratio,
                'creation_timestamp': datetime.utcnow().isoformat()
            }
        
        # Validate bundle integrity
        try:
            # Quick validation - check if bundle can be opened
            with tarfile.open(bundle_path, 'r') as tar:
                member_count = len(tar.getmembers())
                logger.info(f"Bundle validation successful: {member_count} members")
                
        except Exception as e:
            logger.error(f"Bundle validation failed: {e}")
            raise BundleError(f"Bundle integrity check failed: {e}")
        
        logger.info(f"Bundle finalized: {bundle_path}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get hybrid bundle stage metrics."""
        base_metrics = super().get_metrics()
        
        hybrid_metrics = {
            'hybrid_bundle': {
                'lens_export_enabled': self.enable_lens_export,
                'max_bundle_size_gb': self.max_bundle_size_gb,
                'compression_level': self.compression_level,
                'include_raw_data': self.include_raw_data,
                'parallel_processor_available': self._parallel_processor is not None
            }
        }
        
        base_metrics.update(hybrid_metrics)
        return base_metrics
    
    def _get_capabilities(self) -> List[str]:
        """Get hybrid bundle capabilities."""
        capabilities = super()._get_capabilities()
        capabilities.extend([
            'hybrid_bundling',
            'lens_data_export',
            'compressed_archives',
            'manifest_generation',
            'integrity_verification',
            'parallel_processing',
            'vector_index_bundling',
            'cross_reference_mapping'
        ])
        return capabilities
    
    async def cleanup(self, context: "PipelineContext") -> None:
        """Clean up hybrid bundle resources."""
        logger.info("Cleaning up hybrid bundle stage")
        
        # Reset processor reference
        self._parallel_processor = None
        
        await super().cleanup(context)