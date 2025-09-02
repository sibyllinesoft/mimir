"""
Smart Result Synthesis for Hybrid Mimir-Lens Operations.

This module provides intelligent synthesis of results from both Mimir and Lens
systems, combining their respective strengths to create comprehensive and
optimized final outputs.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from enum import Enum
import numpy as np

from ..data.schemas import VectorChunk, VectorIndex
from ..util.log import get_logger
from ..util.errors import MimirError

logger = get_logger(__name__)


class SynthesisStrategy(Enum):
    """Strategies for combining Mimir and Lens results."""
    BEST_OF_BOTH = "best_of_both"          # Take best results from each system
    LENS_PRIORITY = "lens_priority"        # Prefer Lens results, fallback to Mimir
    MIMIR_PRIORITY = "mimir_priority"      # Prefer Mimir results, fallback to Lens
    WEIGHTED_FUSION = "weighted_fusion"    # Combine results with confidence weights
    CONSENSUS = "consensus"                # Only include results both systems agree on
    COMPLEMENTARY = "complementary"        # Use each system for its strengths


class ConfidenceSource(Enum):
    """Sources of confidence scores."""
    PERFORMANCE_METRICS = "performance_metrics"
    QUALITY_INDICATORS = "quality_indicators" 
    SYSTEM_HEALTH = "system_health"
    HISTORICAL_ACCURACY = "historical_accuracy"
    CONTENT_TYPE_SUITABILITY = "content_type_suitability"


@dataclass
class ConfidenceScore:
    """Confidence score for a result."""
    value: float = 0.0  # 0.0 to 1.0
    source: ConfidenceSource = ConfidenceSource.QUALITY_INDICATORS
    reasoning: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SynthesisResult:
    """Result of synthesis operation."""
    success: bool
    synthesized_data: Any = None
    mimir_contribution: float = 0.0  # 0.0 to 1.0
    lens_contribution: float = 0.0   # 0.0 to 1.0
    strategy_used: SynthesisStrategy = SynthesisStrategy.BEST_OF_BOTH
    confidence_score: float = 0.0
    synthesis_time_ms: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None


class ResultSynthesizer:
    """
    Intelligent synthesis engine for combining Mimir and Lens results.
    
    This class implements various strategies for merging results from both
    systems, taking into account their respective strengths, system health,
    and the specific characteristics of the data being processed.
    """
    
    def __init__(
        self,
        default_strategy: SynthesisStrategy = SynthesisStrategy.BEST_OF_BOTH,
        enable_quality_scoring: bool = True,
        enable_performance_weighting: bool = True
    ):
        """Initialize result synthesizer."""
        self.default_strategy = default_strategy
        self.enable_quality_scoring = enable_quality_scoring
        self.enable_performance_weighting = enable_performance_weighting
        
        # Synthesis preferences based on content type
        self.content_type_preferences = {
            'code': {
                'mimir_weight': 0.7,  # Mimir excels at code analysis
                'lens_weight': 0.3
            },
            'text': {
                'mimir_weight': 0.4,
                'lens_weight': 0.6   # Lens may be faster for plain text
            },
            'mixed': {
                'mimir_weight': 0.5,
                'lens_weight': 0.5
            }
        }
        
        logger.info(f"ResultSynthesizer initialized with strategy: {default_strategy.value}")
    
    async def synthesize_discovery_results(
        self,
        mimir_data: Optional[Dict[str, Any]],
        lens_data: Optional[Dict[str, Any]],
        strategy: Optional[SynthesisStrategy] = None
    ) -> SynthesisResult:
        """Synthesize file discovery results from Mimir and Lens."""
        logger.info("Synthesizing discovery results")
        start_time = time.time()
        
        try:
            strategy = strategy or self.default_strategy
            
            if not mimir_data and not lens_data:
                return SynthesisResult(
                    success=False,
                    error="No data available for synthesis"
                )
            
            # Handle single-source scenarios
            if not mimir_data:
                return self._create_lens_only_result(lens_data, start_time)
            
            if not lens_data:
                return self._create_mimir_only_result(mimir_data, start_time)
            
            # Execute synthesis strategy
            if strategy == SynthesisStrategy.BEST_OF_BOTH:
                return await self._synthesize_discovery_best_of_both(
                    mimir_data, lens_data, start_time
                )
            elif strategy == SynthesisStrategy.WEIGHTED_FUSION:
                return await self._synthesize_discovery_weighted_fusion(
                    mimir_data, lens_data, start_time
                )
            elif strategy == SynthesisStrategy.COMPLEMENTARY:
                return await self._synthesize_discovery_complementary(
                    mimir_data, lens_data, start_time
                )
            else:
                # Default to best of both
                return await self._synthesize_discovery_best_of_both(
                    mimir_data, lens_data, start_time
                )
                
        except Exception as e:
            logger.error(f"Discovery synthesis failed: {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                synthesis_time_ms=(time.time() - start_time) * 1000
            )
    
    async def synthesize_embedding_results(
        self,
        mimir_chunks: List[VectorChunk],
        lens_chunks: List[VectorChunk],
        strategy: Optional[SynthesisStrategy] = None
    ) -> SynthesisResult:
        """Synthesize embedding results from Mimir and Lens."""
        logger.info(f"Synthesizing embedding results: {len(mimir_chunks)} Mimir, {len(lens_chunks)} Lens")
        start_time = time.time()
        
        try:
            strategy = strategy or self.default_strategy
            
            if not mimir_chunks and not lens_chunks:
                return SynthesisResult(
                    success=False,
                    error="No chunks available for synthesis"
                )
            
            # Create chunk mapping for comparison
            chunk_mapping = self._create_chunk_mapping(mimir_chunks, lens_chunks)
            
            # Execute synthesis strategy
            if strategy == SynthesisStrategy.BEST_OF_BOTH:
                return await self._synthesize_embeddings_best_of_both(
                    chunk_mapping, start_time
                )
            elif strategy == SynthesisStrategy.WEIGHTED_FUSION:
                return await self._synthesize_embeddings_weighted_fusion(
                    chunk_mapping, start_time
                )
            elif strategy == SynthesisStrategy.CONSENSUS:
                return await self._synthesize_embeddings_consensus(
                    chunk_mapping, start_time
                )
            else:
                # Default to best of both
                return await self._synthesize_embeddings_best_of_both(
                    chunk_mapping, start_time
                )
                
        except Exception as e:
            logger.error(f"Embedding synthesis failed: {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                synthesis_time_ms=(time.time() - start_time) * 1000
            )
    
    async def synthesize_search_results(
        self,
        mimir_results: List[Dict[str, Any]],
        lens_results: List[Dict[str, Any]],
        query: str,
        strategy: Optional[SynthesisStrategy] = None
    ) -> SynthesisResult:
        """Synthesize search results from Mimir and Lens."""
        logger.info(f"Synthesizing search results for query: '{query[:50]}...'")
        start_time = time.time()
        
        try:
            strategy = strategy or self.default_strategy
            
            if not mimir_results and not lens_results:
                return SynthesisResult(
                    success=False,
                    error="No search results available for synthesis"
                )
            
            # Execute synthesis strategy
            if strategy == SynthesisStrategy.BEST_OF_BOTH:
                return await self._synthesize_search_best_of_both(
                    mimir_results, lens_results, query, start_time
                )
            elif strategy == SynthesisStrategy.WEIGHTED_FUSION:
                return await self._synthesize_search_weighted_fusion(
                    mimir_results, lens_results, query, start_time
                )
            else:
                # Default to best of both
                return await self._synthesize_search_best_of_both(
                    mimir_results, lens_results, query, start_time
                )
                
        except Exception as e:
            logger.error(f"Search synthesis failed: {e}")
            return SynthesisResult(
                success=False,
                error=str(e),
                synthesis_time_ms=(time.time() - start_time) * 1000
            )
    
    def _create_lens_only_result(
        self, 
        lens_data: Dict[str, Any], 
        start_time: float
    ) -> SynthesisResult:
        """Create result using only Lens data."""
        return SynthesisResult(
            success=True,
            synthesized_data=lens_data,
            mimir_contribution=0.0,
            lens_contribution=1.0,
            strategy_used=SynthesisStrategy.LENS_PRIORITY,
            confidence_score=0.7,  # Lower confidence without Mimir validation
            synthesis_time_ms=(time.time() - start_time) * 1000
        )
    
    def _create_mimir_only_result(
        self, 
        mimir_data: Dict[str, Any], 
        start_time: float
    ) -> SynthesisResult:
        """Create result using only Mimir data."""
        return SynthesisResult(
            success=True,
            synthesized_data=mimir_data,
            mimir_contribution=1.0,
            lens_contribution=0.0,
            strategy_used=SynthesisStrategy.MIMIR_PRIORITY,
            confidence_score=0.8,  # Higher confidence in Mimir's deep analysis
            synthesis_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _synthesize_discovery_best_of_both(
        self,
        mimir_data: Dict[str, Any],
        lens_data: Dict[str, Any],
        start_time: float
    ) -> SynthesisResult:
        """Synthesize discovery results taking best from both systems."""
        synthesized = {
            'hybrid_discovery': True,
            'sources': ['mimir', 'lens'],
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Combine file lists intelligently
        mimir_files = set(mimir_data.get('files_discovered', []))
        lens_files = set(lens_data.get('files_discovered', []))
        
        # Take union of files (best coverage)
        all_files = mimir_files.union(lens_files)
        synthesized['files_discovered'] = list(all_files)
        synthesized['total_files'] = len(all_files)
        
        # Use Mimir's analysis capabilities
        if 'structure_analysis' in mimir_data:
            synthesized['structure_analysis'] = mimir_data['structure_analysis']
        
        if 'workspaces' in mimir_data:
            synthesized['workspaces'] = mimir_data['workspaces']
        
        # Use Lens's performance metrics
        if 'indexing_performance' in lens_data:
            synthesized['indexing_performance'] = lens_data['indexing_performance']
        
        # Combine quality metrics
        quality_metrics = {}
        
        # Coverage metric - how much overlap between systems
        if mimir_files and lens_files:
            overlap = len(mimir_files.intersection(lens_files))
            total_unique = len(all_files)
            quality_metrics['coverage_overlap'] = overlap / total_unique if total_unique > 0 else 0.0
        
        # Consistency metric - similar file counts indicate good consistency  
        if mimir_files and lens_files:
            size_diff = abs(len(mimir_files) - len(lens_files))
            max_size = max(len(mimir_files), len(lens_files))
            quality_metrics['count_consistency'] = 1.0 - (size_diff / max_size) if max_size > 0 else 1.0
        
        return SynthesisResult(
            success=True,
            synthesized_data=synthesized,
            mimir_contribution=0.6,  # Higher weight for Mimir's analysis
            lens_contribution=0.4,
            strategy_used=SynthesisStrategy.BEST_OF_BOTH,
            confidence_score=0.9,  # High confidence when both systems available
            synthesis_time_ms=(time.time() - start_time) * 1000,
            quality_metrics=quality_metrics
        )
    
    async def _synthesize_discovery_weighted_fusion(
        self,
        mimir_data: Dict[str, Any],
        lens_data: Dict[str, Any],
        start_time: float
    ) -> SynthesisResult:
        """Synthesize discovery results with weighted fusion."""
        # Calculate weights based on system health and performance
        mimir_weight = 0.6  # Default higher weight for analysis depth
        lens_weight = 0.4   # Weight for performance and coverage
        
        # Adjust weights based on available data quality
        mimir_quality = self._assess_discovery_data_quality(mimir_data)
        lens_quality = self._assess_discovery_data_quality(lens_data)
        
        if mimir_quality > lens_quality:
            mimir_weight = 0.7
            lens_weight = 0.3
        elif lens_quality > mimir_quality:
            mimir_weight = 0.4
            lens_weight = 0.6
        
        # Create weighted synthesis
        synthesized = {
            'hybrid_discovery': True,
            'synthesis_strategy': 'weighted_fusion',
            'weights': {'mimir': mimir_weight, 'lens': lens_weight},
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Weighted file combination
        mimir_files = set(mimir_data.get('files_discovered', []))
        lens_files = set(lens_data.get('files_discovered', []))
        
        # Prioritize files found by higher-weighted system
        if mimir_weight > lens_weight:
            primary_files = mimir_files
            secondary_files = lens_files - mimir_files
        else:
            primary_files = lens_files
            secondary_files = mimir_files - lens_files
        
        all_files = list(primary_files) + list(secondary_files)
        synthesized['files_discovered'] = all_files
        synthesized['total_files'] = len(all_files)
        
        # Weighted metadata combination
        if mimir_weight > 0.5:
            synthesized.update({k: v for k, v in mimir_data.items() if k not in ['files_discovered']})
            # Add Lens performance data
            if 'performance_metrics' in lens_data:
                synthesized['lens_performance'] = lens_data['performance_metrics']
        else:
            synthesized.update({k: v for k, v in lens_data.items() if k not in ['files_discovered']})
            # Add Mimir analysis data
            if 'structure_analysis' in mimir_data:
                synthesized['mimir_analysis'] = mimir_data['structure_analysis']
        
        confidence_score = (mimir_weight * mimir_quality + lens_weight * lens_quality)
        
        return SynthesisResult(
            success=True,
            synthesized_data=synthesized,
            mimir_contribution=mimir_weight,
            lens_contribution=lens_weight,
            strategy_used=SynthesisStrategy.WEIGHTED_FUSION,
            confidence_score=confidence_score,
            synthesis_time_ms=(time.time() - start_time) * 1000
        )
    
    async def _synthesize_discovery_complementary(
        self,
        mimir_data: Dict[str, Any],
        lens_data: Dict[str, Any],
        start_time: float
    ) -> SynthesisResult:
        """Synthesize discovery results using complementary strengths."""
        synthesized = {
            'hybrid_discovery': True,
            'synthesis_strategy': 'complementary',
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Use Lens for high-performance file discovery and indexing
        if 'files_discovered' in lens_data:
            synthesized['files_discovered'] = lens_data['files_discovered']
            synthesized['total_files'] = len(lens_data['files_discovered'])
        
        if 'indexing_performance' in lens_data:
            synthesized['bulk_indexing'] = lens_data['indexing_performance']
        
        # Use Mimir for deep analysis and git integration
        if 'structure_analysis' in mimir_data:
            synthesized['repository_analysis'] = mimir_data['structure_analysis']
        
        if 'workspaces' in mimir_data:
            synthesized['monorepo_workspaces'] = mimir_data['workspaces']
        
        if 'dirty_overlay' in mimir_data:
            synthesized['uncommitted_changes'] = mimir_data['dirty_overlay']
        
        if 'cache_key' in mimir_data:
            synthesized['incremental_cache_key'] = mimir_data['cache_key']
        
        # Combine file metadata - use Mimir's detailed analysis
        if 'file_metadata' in mimir_data:
            synthesized['detailed_file_analysis'] = mimir_data['file_metadata']
        
        return SynthesisResult(
            success=True,
            synthesized_data=synthesized,
            mimir_contribution=0.5,  # Equal contribution but different roles
            lens_contribution=0.5,
            strategy_used=SynthesisStrategy.COMPLEMENTARY,
            confidence_score=0.95,  # Highest confidence when using each system's strengths
            synthesis_time_ms=(time.time() - start_time) * 1000
        )
    
    def _assess_discovery_data_quality(self, data: Dict[str, Any]) -> float:
        """Assess quality of discovery data."""
        quality_score = 0.0
        
        # Check for completeness
        if 'files_discovered' in data:
            quality_score += 0.3
            file_count = len(data['files_discovered'])
            if file_count > 0:
                quality_score += 0.2
        
        # Check for analysis depth
        if 'structure_analysis' in data:
            quality_score += 0.2
        
        if 'workspaces' in data:
            quality_score += 0.1
        
        if 'file_metadata' in data:
            quality_score += 0.1
        
        # Check for performance metrics
        if 'performance_metrics' in data or 'indexing_performance' in data:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def _create_chunk_mapping(
        self, 
        mimir_chunks: List[VectorChunk], 
        lens_chunks: List[VectorChunk]
    ) -> Dict[str, Dict[str, VectorChunk]]:
        """Create mapping between Mimir and Lens chunks."""
        mapping = defaultdict(dict)
        
        # Map Mimir chunks
        for chunk in mimir_chunks:
            key = self._generate_chunk_key(chunk)
            mapping[key]['mimir'] = chunk
        
        # Map Lens chunks
        for chunk in lens_chunks:
            key = self._generate_chunk_key(chunk)
            mapping[key]['lens'] = chunk
        
        return dict(mapping)
    
    def _generate_chunk_key(self, chunk: VectorChunk) -> str:
        """Generate unique key for chunk mapping."""
        return f"{chunk.file_path}:{chunk.start_line}-{chunk.end_line}"
    
    async def _synthesize_embeddings_best_of_both(
        self,
        chunk_mapping: Dict[str, Dict[str, VectorChunk]],
        start_time: float
    ) -> SynthesisResult:
        """Synthesize embeddings taking best from both systems."""
        synthesized_chunks = []
        mimir_count = 0
        lens_count = 0
        
        for chunk_key, chunks in chunk_mapping.items():
            mimir_chunk = chunks.get('mimir')
            lens_chunk = chunks.get('lens')
            
            if mimir_chunk and lens_chunk:
                # Both available - choose based on quality
                chosen_chunk = self._choose_better_embedding(mimir_chunk, lens_chunk)
                if chosen_chunk == mimir_chunk:
                    mimir_count += 1
                else:
                    lens_count += 1
                synthesized_chunks.append(chosen_chunk)
                
            elif mimir_chunk:
                # Only Mimir available
                synthesized_chunks.append(mimir_chunk)
                mimir_count += 1
                
            elif lens_chunk:
                # Only Lens available
                synthesized_chunks.append(lens_chunk)
                lens_count += 1
        
        total_chunks = len(synthesized_chunks)
        mimir_contrib = mimir_count / total_chunks if total_chunks > 0 else 0.0
        lens_contrib = lens_count / total_chunks if total_chunks > 0 else 0.0
        
        quality_metrics = {
            'total_chunks': total_chunks,
            'mimir_chunks_used': mimir_count,
            'lens_chunks_used': lens_count,
            'embedding_coverage': 1.0  # All requested chunks have embeddings
        }
        
        return SynthesisResult(
            success=True,
            synthesized_data=synthesized_chunks,
            mimir_contribution=mimir_contrib,
            lens_contribution=lens_contrib,
            strategy_used=SynthesisStrategy.BEST_OF_BOTH,
            confidence_score=0.9,
            synthesis_time_ms=(time.time() - start_time) * 1000,
            quality_metrics=quality_metrics
        )
    
    def _choose_better_embedding(
        self, 
        mimir_chunk: VectorChunk, 
        lens_chunk: VectorChunk
    ) -> VectorChunk:
        """Choose the better embedding between Mimir and Lens chunks."""
        # Default to Mimir if both have embeddings (assuming better quality for code)
        if (mimir_chunk.embedding is not None and len(mimir_chunk.embedding) > 0 and
            lens_chunk.embedding is not None and len(lens_chunk.embedding) > 0):
            
            # For code chunks, prefer Mimir
            if mimir_chunk.chunk_type == 'code':
                return mimir_chunk
            else:
                # For text chunks, could prefer Lens
                return lens_chunk
        
        # Return the one with an embedding
        if mimir_chunk.embedding is not None and len(mimir_chunk.embedding) > 0:
            return mimir_chunk
        elif lens_chunk.embedding is not None and len(lens_chunk.embedding) > 0:
            return lens_chunk
        
        # Neither has embedding, return Mimir as default
        return mimir_chunk
    
    async def _synthesize_embeddings_weighted_fusion(
        self,
        chunk_mapping: Dict[str, Dict[str, VectorChunk]],
        start_time: float
    ) -> SynthesisResult:
        """Synthesize embeddings with weighted fusion of vectors."""
        synthesized_chunks = []
        fusion_count = 0
        
        for chunk_key, chunks in chunk_mapping.items():
            mimir_chunk = chunks.get('mimir')
            lens_chunk = chunks.get('lens')
            
            if (mimir_chunk and lens_chunk and
                mimir_chunk.embedding is not None and lens_chunk.embedding is not None and
                len(mimir_chunk.embedding) > 0 and len(lens_chunk.embedding) > 0):
                
                # Both have embeddings - create weighted fusion
                fused_chunk = self._fuse_embeddings(mimir_chunk, lens_chunk)
                synthesized_chunks.append(fused_chunk)
                fusion_count += 1
                
            elif mimir_chunk and mimir_chunk.embedding is not None:
                synthesized_chunks.append(mimir_chunk)
                
            elif lens_chunk and lens_chunk.embedding is not None:
                synthesized_chunks.append(lens_chunk)
        
        quality_metrics = {
            'total_chunks': len(synthesized_chunks),
            'fused_embeddings': fusion_count,
            'fusion_rate': fusion_count / len(synthesized_chunks) if synthesized_chunks else 0.0
        }
        
        return SynthesisResult(
            success=True,
            synthesized_data=synthesized_chunks,
            mimir_contribution=0.6,  # Slightly higher weight for Mimir
            lens_contribution=0.4,
            strategy_used=SynthesisStrategy.WEIGHTED_FUSION,
            confidence_score=0.85,
            synthesis_time_ms=(time.time() - start_time) * 1000,
            quality_metrics=quality_metrics
        )
    
    def _fuse_embeddings(
        self, 
        mimir_chunk: VectorChunk, 
        lens_chunk: VectorChunk,
        mimir_weight: float = 0.6
    ) -> VectorChunk:
        """Fuse embeddings from Mimir and Lens chunks."""
        # Create weighted average of embeddings
        lens_weight = 1.0 - mimir_weight
        
        mimir_emb = np.array(mimir_chunk.embedding)
        lens_emb = np.array(lens_chunk.embedding)
        
        # Handle dimension mismatch
        if mimir_emb.shape != lens_emb.shape:
            logger.warning(f"Embedding dimension mismatch: {mimir_emb.shape} vs {lens_emb.shape}")
            # Return the higher-dimensional one or Mimir as default
            return mimir_chunk if len(mimir_emb) >= len(lens_emb) else lens_chunk
        
        # Create weighted fusion
        fused_embedding = (mimir_weight * mimir_emb + lens_weight * lens_emb)
        
        # Normalize the fused embedding
        norm = np.linalg.norm(fused_embedding)
        if norm > 0:
            fused_embedding = fused_embedding / norm
        
        # Create new chunk with fused embedding
        fused_chunk = VectorChunk(
            chunk_id=f"fused_{mimir_chunk.chunk_id}",
            content=mimir_chunk.content,  # Use Mimir's content as primary
            file_path=mimir_chunk.file_path,
            start_line=mimir_chunk.start_line,
            end_line=mimir_chunk.end_line,
            chunk_type=mimir_chunk.chunk_type,
            embedding=fused_embedding,
            embedding_model=f"fused_{mimir_chunk.embedding_model}_{lens_chunk.embedding_model}"
        )
        
        return fused_chunk
    
    async def _synthesize_embeddings_consensus(
        self,
        chunk_mapping: Dict[str, Dict[str, VectorChunk]],
        start_time: float
    ) -> SynthesisResult:
        """Synthesize embeddings using only chunks both systems processed."""
        consensus_chunks = []
        
        for chunk_key, chunks in chunk_mapping.items():
            mimir_chunk = chunks.get('mimir')
            lens_chunk = chunks.get('lens')
            
            # Only include chunks that both systems processed successfully
            if (mimir_chunk and lens_chunk and
                mimir_chunk.embedding is not None and lens_chunk.embedding is not None and
                len(mimir_chunk.embedding) > 0 and len(lens_chunk.embedding) > 0):
                
                # Use Mimir's result for consensus (assuming higher quality analysis)
                consensus_chunks.append(mimir_chunk)
        
        quality_metrics = {
            'consensus_chunks': len(consensus_chunks),
            'total_possible': len(chunk_mapping),
            'consensus_rate': len(consensus_chunks) / len(chunk_mapping) if chunk_mapping else 0.0
        }
        
        return SynthesisResult(
            success=True,
            synthesized_data=consensus_chunks,
            mimir_contribution=1.0,  # Using Mimir chunks for consensus
            lens_contribution=0.0,   # But Lens validated the processing
            strategy_used=SynthesisStrategy.CONSENSUS,
            confidence_score=0.95,  # Very high confidence for consensus
            synthesis_time_ms=(time.time() - start_time) * 1000,
            quality_metrics=quality_metrics
        )
    
    async def _synthesize_search_best_of_both(
        self,
        mimir_results: List[Dict[str, Any]],
        lens_results: List[Dict[str, Any]],
        query: str,
        start_time: float
    ) -> SynthesisResult:
        """Synthesize search results taking best from both systems."""
        # Combine and deduplicate results
        combined_results = []
        seen_results = set()
        
        # Process Mimir results first (assuming higher quality)
        for result in mimir_results:
            result_key = self._generate_search_result_key(result)
            if result_key not in seen_results:
                result['source'] = 'mimir'
                combined_results.append(result)
                seen_results.add(result_key)
        
        # Add unique Lens results
        for result in lens_results:
            result_key = self._generate_search_result_key(result)
            if result_key not in seen_results:
                result['source'] = 'lens'
                combined_results.append(result)
                seen_results.add(result_key)
        
        # Sort by relevance score if available
        if combined_results and 'score' in combined_results[0]:
            combined_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        quality_metrics = {
            'total_results': len(combined_results),
            'mimir_results': len(mimir_results),
            'lens_results': len(lens_results),
            'unique_results': len(combined_results),
            'deduplication_rate': 1.0 - (len(combined_results) / (len(mimir_results) + len(lens_results))) if (len(mimir_results) + len(lens_results)) > 0 else 0.0
        }
        
        return SynthesisResult(
            success=True,
            synthesized_data=combined_results,
            mimir_contribution=len(mimir_results) / len(combined_results) if combined_results else 0.0,
            lens_contribution=len(lens_results) / len(combined_results) if combined_results else 0.0,
            strategy_used=SynthesisStrategy.BEST_OF_BOTH,
            confidence_score=0.9,
            synthesis_time_ms=(time.time() - start_time) * 1000,
            quality_metrics=quality_metrics
        )
    
    async def _synthesize_search_weighted_fusion(
        self,
        mimir_results: List[Dict[str, Any]],
        lens_results: List[Dict[str, Any]],
        query: str,
        start_time: float
    ) -> SynthesisResult:
        """Synthesize search results with weighted fusion of scores."""
        # Create result mapping for fusion
        result_mapping = defaultdict(dict)
        
        for result in mimir_results:
            key = self._generate_search_result_key(result)
            result_mapping[key]['mimir'] = result
        
        for result in lens_results:
            key = self._generate_search_result_key(result)
            result_mapping[key]['lens'] = result
        
        # Fuse results with weighted scoring
        fused_results = []
        mimir_weight = 0.6  # Higher weight for Mimir's semantic understanding
        lens_weight = 0.4
        
        for result_key, results in result_mapping.items():
            mimir_result = results.get('mimir')
            lens_result = results.get('lens')
            
            if mimir_result and lens_result:
                # Both have the result - fuse scores
                mimir_score = mimir_result.get('score', 0.0)
                lens_score = lens_result.get('score', 0.0)
                
                fused_score = (mimir_weight * mimir_score + lens_weight * lens_score)
                
                # Use Mimir result as base, update score
                fused_result = mimir_result.copy()
                fused_result['score'] = fused_score
                fused_result['source'] = 'fused'
                fused_result['mimir_score'] = mimir_score
                fused_result['lens_score'] = lens_score
                
                fused_results.append(fused_result)
                
            elif mimir_result:
                mimir_result['source'] = 'mimir'
                fused_results.append(mimir_result)
                
            elif lens_result:
                lens_result['source'] = 'lens'
                fused_results.append(lens_result)
        
        # Sort by fused scores
        if fused_results and 'score' in fused_results[0]:
            fused_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        
        return SynthesisResult(
            success=True,
            synthesized_data=fused_results,
            mimir_contribution=mimir_weight,
            lens_contribution=lens_weight,
            strategy_used=SynthesisStrategy.WEIGHTED_FUSION,
            confidence_score=0.85,
            synthesis_time_ms=(time.time() - start_time) * 1000
        )
    
    def _generate_search_result_key(self, result: Dict[str, Any]) -> str:
        """Generate unique key for search result deduplication."""
        # Use file path and line numbers if available
        file_path = result.get('file_path', result.get('path', ''))
        start_line = result.get('start_line', result.get('line', 0))
        
        # Fallback to content hash if no position info
        if not file_path:
            content = result.get('content', result.get('text', ''))
            import hashlib
            return hashlib.sha256(content.encode()).hexdigest()[:16]
        
        return f"{file_path}:{start_line}"
    
    def get_synthesis_statistics(self) -> Dict[str, Any]:
        """Get statistics about synthesis operations."""
        # This could be expanded to track historical synthesis performance
        return {
            'default_strategy': self.default_strategy.value,
            'quality_scoring_enabled': self.enable_quality_scoring,
            'performance_weighting_enabled': self.enable_performance_weighting,
            'content_type_preferences': self.content_type_preferences
        }