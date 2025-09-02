"""
RAPTOR Pipeline Stage for Mimir 2.0.

Integrates RAPTOR hierarchical indexing as a pipeline stage,
running after LEANN to create hierarchical document organization.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np

from .stage import ConfigurablePipelineStage, AsyncPipelineStage, PipelineStage
from .raptor import RaptorProcessor, RaptorConfig
from .raptor_structures import RaptorTree
from ..config import get_ai_config
from ..util.logging_util import get_logger, get_metrics_collector
from ..util.error_handling import MimirError, ExternalToolError, create_error_context

logger = logging.getLogger(__name__)


class RaptorStage(ConfigurablePipelineStage, AsyncPipelineStage):
    """
    RAPTOR hierarchical indexing pipeline stage.
    
    Creates hierarchical organization of documents using clustering
    and summarization for improved retrieval performance.
    
    This stage runs after LEANN to enhance the vector embeddings
    with hierarchical structure.
    """
    
    def __init__(self):
        """Initialize RAPTOR stage."""
        ConfigurablePipelineStage.__init__(self, PipelineStage.RAPTOR)
        AsyncPipelineStage.__init__(self, PipelineStage.RAPTOR, concurrency_limit=1)
        self.processor: Optional[RaptorProcessor] = None
    
    async def execute(self, context: "PipelineContext", progress_callback=None) -> None:
        """Execute RAPTOR hierarchical indexing."""
        enhanced_logger = get_logger("pipeline.raptor")
        metrics_collector = get_metrics_collector()
        
        with enhanced_logger.performance_track("raptor_processing", include_system_metrics=True):
            context.logger.log_stage_start(self.stage_type, "Building RAPTOR hierarchical index")
            
            try:
                # Check if RAPTOR is enabled
                ai_config = get_ai_config()
                if not ai_config.enable_raptor:
                    enhanced_logger.info("RAPTOR disabled, skipping hierarchical indexing")
                    context.logger.log_info("RAPTOR disabled", stage=self.stage_type)
                    if progress_callback:
                        progress_callback(100)
                    return
                
                # Verify we have vector embeddings from LEANN
                if not context.vector_index:
                    enhanced_logger.warning("No vector index available, skipping RAPTOR")
                    context.logger.log_warning(
                        "RAPTOR requires vector embeddings from LEANN stage", 
                        stage=self.stage_type
                    )
                    if progress_callback:
                        progress_callback(100)
                    return
                
                if progress_callback:
                    progress_callback(10)
                
                # Initialize RAPTOR processor
                raptor_config = RaptorConfig.from_ai_config(ai_config)
                self.processor = RaptorProcessor(config=raptor_config, ai_config=ai_config)
                await self.processor.initialize()
                
                if progress_callback:
                    progress_callback(20)
                
                # Extract embeddings and documents from vector index
                embeddings, documents, metadata = await self._extract_vector_data(context)
                
                if not embeddings or len(embeddings) < 2:
                    enhanced_logger.info("Insufficient documents for RAPTOR clustering")
                    if progress_callback:
                        progress_callback(100)
                    return
                
                if progress_callback:
                    progress_callback(40)
                
                enhanced_logger.info(f"Processing {len(embeddings)} documents with RAPTOR")
                
                # Create hierarchical tree
                raptor_tree = await self.processor.process_embeddings(
                    embeddings, documents, metadata
                )
                
                if progress_callback:
                    progress_callback(80)
                
                # Store RAPTOR tree in context
                context.raptor_tree = raptor_tree
                
                # Save tree to work directory for persistence
                tree_path = context.work_dir / "raptor_tree.json"
                raptor_tree.save_to_file(tree_path)
                
                # Log statistics
                stats = raptor_tree.get_tree_stats()
                enhanced_logger.operation_success(
                    "raptor_processing",
                    total_nodes=stats["total_nodes"],
                    leaf_nodes=stats["leaf_nodes"],
                    internal_nodes=stats["internal_nodes"],
                    max_level=stats["max_level"],
                    tree_file=str(tree_path)
                )
                
                # Record metrics
                metrics_collector.record_embeddings_created(
                    "raptor", "hierarchical_node", stats["internal_nodes"]
                )
                
                context.logger.log_info(
                    f"RAPTOR tree created with {stats['total_nodes']} nodes "
                    f"({stats['leaf_nodes']} leaves, {stats['internal_nodes']} internal)",
                    stage=self.stage_type
                )
                
                if progress_callback:
                    progress_callback(100)
            
            except Exception as e:
                if not isinstance(e, MimirError):
                    error_context = create_error_context(
                        component="raptor",
                        operation="hierarchical_indexing",
                        parameters={
                            "repo_root": str(context.repo_info.root),
                            "file_count": len(context.tracked_files),
                            "has_vector_index": context.vector_index is not None,
                            "raptor_enabled": ai_config.enable_raptor if 'ai_config' in locals() else False,
                        }
                    )
                    
                    e = ExternalToolError(
                        tool="raptor",
                        message=f"RAPTOR processing failed: {str(e)}",
                        context=error_context,
                        suggestions=[
                            "Ensure ML dependencies are installed (pip install scikit-learn umap-learn hdbscan)",
                            "Verify Ollama is running for summarization",
                            "Check vector embeddings are available from LEANN stage",
                            "Reduce cluster size thresholds in configuration"
                        ]
                    )
                
                enhanced_logger.operation_error("raptor_processing", e)
                context.logger.log_stage_error(self.stage_type, e)
                raise
    
    async def _extract_vector_data(
        self, context: "PipelineContext"
    ) -> tuple[List[np.ndarray], List[str], List[Dict[str, Any]]]:
        """
        Extract embeddings, documents, and metadata from vector index.
        
        Args:
            context: Pipeline context with vector index
            
        Returns:
            Tuple of (embeddings, documents, metadata)
        """
        embeddings = []
        documents = []
        metadata = []
        
        try:
            # This is a simplified extraction - the actual implementation
            # would depend on the specific vector index format used by LEANN
            vector_index = context.vector_index
            
            # Extract chunks from vector index
            if hasattr(vector_index, 'chunks'):
                for i, chunk in enumerate(vector_index.chunks):
                    # Get embedding vector
                    if hasattr(chunk, 'embedding'):
                        embeddings.append(np.array(chunk.embedding))
                    elif hasattr(chunk, 'vector'):
                        embeddings.append(np.array(chunk.vector))
                    else:
                        logger.warning(f"No embedding found for chunk {i}")
                        continue
                    
                    # Get document text
                    if hasattr(chunk, 'content'):
                        documents.append(chunk.content)
                    elif hasattr(chunk, 'text'):
                        documents.append(chunk.text)
                    else:
                        documents.append(f"Chunk {i}")  # Fallback
                    
                    # Extract metadata
                    chunk_metadata = {
                        'chunk_id': i,
                        'source_type': 'vector_chunk'
                    }
                    
                    if hasattr(chunk, 'file_path'):
                        chunk_metadata['file_path'] = chunk.file_path
                    elif hasattr(chunk, 'source'):
                        chunk_metadata['file_path'] = chunk.source
                    
                    if hasattr(chunk, 'start_line'):
                        chunk_metadata['start_line'] = chunk.start_line
                    if hasattr(chunk, 'end_line'):
                        chunk_metadata['end_line'] = chunk.end_line
                    
                    metadata.append(chunk_metadata)
            
            # Alternative: extract from index directly if chunks not available
            elif hasattr(vector_index, 'embeddings') and hasattr(vector_index, 'documents'):
                embeddings = [np.array(emb) for emb in vector_index.embeddings]
                documents = vector_index.documents
                metadata = [{'chunk_id': i, 'source_type': 'direct'} for i in range(len(documents))]
            
            else:
                logger.warning("Cannot extract data from vector index - unknown format")
                return [], [], []
            
            logger.debug(f"Extracted {len(embeddings)} embeddings from vector index")
            return embeddings, documents, metadata
        
        except Exception as e:
            logger.error(f"Failed to extract vector data: {e}")
            return [], [], []
    
    def _get_capabilities(self) -> List[str]:
        """Get stage capabilities."""
        return [
            "basic_execution",
            "async_operations", 
            "hierarchical_indexing",
            "clustering",
            "summarization",
            "tree_organization"
        ]
    
    @property
    def stage_name(self) -> str:
        """Get human-readable stage name."""
        return "RAPTOR Hierarchical Indexing"
    
    @property
    def stage_description(self) -> str:
        """Get stage description."""
        return ("Creates hierarchical organization of code documents using "
                "clustering and AI summarization for improved retrieval")


# Add RAPTOR to the PipelineStage enum (would need to be added to the actual enum)
# This is a conceptual addition - the actual implementation would require
# modifying the PipelineStage enum in the stage.py file
if not hasattr(PipelineStage, 'RAPTOR'):
    # For now, we'll use a placeholder. In actual implementation,
    # this would be added to the PipelineStage enum properly
    PipelineStage.RAPTOR = "raptor"