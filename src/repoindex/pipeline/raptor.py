"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval.

Implements hierarchical clustering and summarization of code embeddings
to create a tree-structured index for improved retrieval.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ML dependencies (optional - graceful degradation)
try:
    from sklearn.cluster import HDBSCAN
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import umap
    HAS_ML_DEPS = True
except ImportError:
    HAS_ML_DEPS = False
    HDBSCAN = None
    silhouette_score = None
    StandardScaler = None
    umap = None

from .raptor_structures import RaptorTree, RaptorNode, NodeType, ClusterMetrics
from .llm_adapter import LLMAdapter, CodeSnippet
from .pipeline_coordinator import get_pipeline_coordinator
from ..config import AIConfig, get_ai_config

logger = logging.getLogger(__name__)


@dataclass
class RaptorConfig:
    """Configuration for RAPTOR processing."""
    # Clustering parameters
    cluster_min_size: int = 5
    cluster_min_samples: int = 3
    cluster_threshold: float = 0.1
    max_clusters: int = 10
    
    # UMAP parameters for dimensionality reduction
    umap_n_neighbors: int = 15
    umap_n_components: int = 5
    umap_min_dist: float = 0.1
    
    # Summarization parameters
    max_summary_length: int = 500
    summarization_model: str = "llama3.2:3b"
    
    # Tree building parameters
    max_tree_levels: int = 3
    min_cluster_size_for_split: int = 10
    
    @classmethod
    def from_ai_config(cls, ai_config: AIConfig) -> 'RaptorConfig':
        """Create RaptorConfig from AIConfig."""
        return cls(
            cluster_threshold=ai_config.raptor_cluster_threshold,
            max_clusters=ai_config.raptor_max_clusters,
            summarization_model=ai_config.raptor_summarization_model,
        )


class RaptorProcessor:
    """
    Main RAPTOR processor for hierarchical indexing.
    
    Implements clustering, summarization, and tree building
    for improved document retrieval and organization.
    """
    
    def __init__(
        self, 
        config: Optional[RaptorConfig] = None,
        ai_config: Optional[AIConfig] = None
    ):
        """
        Initialize RAPTOR processor.
        
        Args:
            config: RAPTOR-specific configuration
            ai_config: AI configuration for LLM access
        """
        self.ai_config = ai_config or get_ai_config()
        self.config = config or RaptorConfig.from_ai_config(self.ai_config)
        self.coordinator = None
        self._initialized = False
        
        if not HAS_ML_DEPS:
            logger.warning("ML dependencies not available - RAPTOR functionality limited")
    
    async def initialize(self) -> None:
        """Initialize the RAPTOR processor."""
        if self._initialized:
            return
        
        if not HAS_ML_DEPS:
            raise RuntimeError("RAPTOR requires ML dependencies: pip install scikit-learn umap-learn hdbscan")
        
        # Get pipeline coordinator for LLM access
        self.coordinator = await get_pipeline_coordinator(self.ai_config)
        self._initialized = True
        
        logger.info("RAPTOR processor initialized")
    
    async def process_embeddings(
        self,
        embeddings: List[np.ndarray],
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ) -> RaptorTree:
        """
        Process embeddings into hierarchical RAPTOR tree.
        
        Args:
            embeddings: Document embeddings
            documents: Original document texts
            metadata: Optional metadata for each document
            
        Returns:
            Hierarchical RAPTOR tree
        """
        if not self._initialized:
            await self.initialize()
        
        logger.info(f"Processing {len(embeddings)} embeddings into RAPTOR tree")
        
        if len(embeddings) != len(documents):
            raise ValueError("Number of embeddings must match number of documents")
        
        metadata = metadata or [{} for _ in documents]
        
        # Create tree and add leaf nodes
        tree = RaptorTree()
        leaf_nodes = await self._create_leaf_nodes(embeddings, documents, metadata, tree)
        
        # Build hierarchical structure
        await self._build_hierarchical_tree(leaf_nodes, tree)
        
        logger.info(f"Created RAPTOR tree with {len(tree.nodes)} nodes")
        return tree
    
    async def _create_leaf_nodes(
        self,
        embeddings: List[np.ndarray], 
        documents: List[str],
        metadata: List[Dict[str, Any]],
        tree: RaptorTree
    ) -> List[RaptorNode]:
        """Create leaf nodes from documents and embeddings."""
        leaf_nodes = []
        
        for i, (embedding, document, meta) in enumerate(zip(embeddings, documents, metadata)):
            node = RaptorNode(
                node_type=NodeType.LEAF,
                level=0,
                content=document,
                embedding=embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
                metadata=meta
            )
            
            # Extract source file info if available
            if 'file_path' in meta:
                node.add_source_file(meta['file_path'])
            
            tree.add_node(node)
            leaf_nodes.append(node)
        
        logger.debug(f"Created {len(leaf_nodes)} leaf nodes")
        return leaf_nodes
    
    async def _build_hierarchical_tree(
        self,
        current_nodes: List[RaptorNode],
        tree: RaptorTree,
        current_level: int = 0
    ) -> None:
        """Build hierarchical tree structure recursively."""
        if current_level >= self.config.max_tree_levels or len(current_nodes) <= 1:
            # Create root node if we have multiple nodes and no root yet
            if len(current_nodes) > 1 and not tree.root_id:
                await self._create_root_node(current_nodes, tree)
            return
        
        logger.debug(f"Building tree level {current_level + 1} with {len(current_nodes)} nodes")
        
        # Extract embeddings for clustering
        embeddings = []
        node_map = {}
        
        for node in current_nodes:
            if node.embedding:
                embeddings.append(np.array(node.embedding))
                node_map[len(embeddings) - 1] = node
        
        if len(embeddings) < self.config.cluster_min_size:
            # Not enough nodes for clustering, create root
            if not tree.root_id:
                await self._create_root_node(current_nodes, tree)
            return
        
        # Perform clustering
        clusters = await self._cluster_embeddings(embeddings)
        
        if len(set(clusters)) <= 1:
            # Only one cluster, create root
            if not tree.root_id:
                await self._create_root_node(current_nodes, tree)
            return
        
        # Create cluster summary nodes
        cluster_nodes = await self._create_cluster_nodes(
            current_nodes, clusters, tree, current_level + 1
        )
        
        # Recurse to next level
        await self._build_hierarchical_tree(cluster_nodes, tree, current_level + 1)
    
    async def _cluster_embeddings(self, embeddings: List[np.ndarray]) -> List[int]:
        """Cluster embeddings using UMAP + HDBSCAN."""
        if not embeddings:
            return []
        
        # Convert to numpy array
        embedding_matrix = np.array(embeddings)
        
        # Standardize embeddings
        scaler = StandardScaler()
        scaled_embeddings = scaler.fit_transform(embedding_matrix)
        
        # Dimensionality reduction with UMAP
        reducer = umap.UMAP(
            n_neighbors=min(self.config.umap_n_neighbors, len(embeddings) - 1),
            n_components=min(self.config.umap_n_components, scaled_embeddings.shape[1]),
            min_dist=self.config.umap_min_dist,
            random_state=42
        )
        reduced_embeddings = reducer.fit_transform(scaled_embeddings)
        
        # Clustering with HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=self.config.cluster_min_size,
            min_samples=self.config.cluster_min_samples,
            cluster_selection_epsilon=self.config.cluster_threshold
        )
        cluster_labels = clusterer.fit_predict(reduced_embeddings)
        
        # Calculate clustering metrics
        if len(set(cluster_labels)) > 1:
            try:
                silhouette = silhouette_score(reduced_embeddings, cluster_labels)
                logger.debug(f"Clustering silhouette score: {silhouette:.3f}")
            except Exception as e:
                logger.warning(f"Failed to calculate silhouette score: {e}")
        
        logger.debug(f"Created {len(set(cluster_labels))} clusters")
        return cluster_labels.tolist()
    
    async def _create_cluster_nodes(
        self,
        nodes: List[RaptorNode],
        cluster_labels: List[int],
        tree: RaptorTree,
        level: int
    ) -> List[RaptorNode]:
        """Create summary nodes for each cluster."""
        cluster_groups = {}
        
        # Group nodes by cluster
        for i, label in enumerate(cluster_labels):
            if label not in cluster_groups:
                cluster_groups[label] = []
            cluster_groups[label].append(nodes[i])
        
        cluster_nodes = []
        
        for cluster_id, cluster_nodes_list in cluster_groups.items():
            if cluster_id == -1:  # Noise points in HDBSCAN
                continue
            
            # Create summary for this cluster
            summary_text = await self._summarize_cluster(cluster_nodes_list)
            
            # Calculate cluster metrics
            cluster_embeddings = [
                np.array(node.embedding) for node in cluster_nodes_list 
                if node.embedding
            ]
            metrics = self._calculate_cluster_metrics(cluster_embeddings, cluster_id)
            
            # Create cluster summary node
            summary_node = RaptorNode(
                node_type=NodeType.INTERNAL,
                level=level,
                content=summary_text,
                cluster_id=cluster_id,
                cluster_metrics=metrics
            )
            
            # Calculate summary embedding (average of cluster)
            if cluster_embeddings:
                avg_embedding = np.mean(cluster_embeddings, axis=0)
                summary_node.embedding = avg_embedding.tolist()
            
            # Collect source files from all cluster nodes
            for node in cluster_nodes_list:
                summary_node.source_files.extend(node.source_files)
                # Create parent-child relationship
                tree.create_parent_child_relationship(summary_node.node_id, node.node_id)
            
            tree.add_node(summary_node)
            cluster_nodes.append(summary_node)
        
        logger.debug(f"Created {len(cluster_nodes)} cluster summary nodes at level {level}")
        return cluster_nodes
    
    async def _summarize_cluster(self, nodes: List[RaptorNode]) -> str:
        """Generate summary for a cluster of nodes."""
        if not nodes:
            return ""
        
        if len(nodes) == 1:
            return nodes[0].content
        
        # Combine content from all nodes in cluster
        combined_content = "\n\n".join([
            f"Document {i+1}:\n{node.content}"
            for i, node in enumerate(nodes[:5])  # Limit to first 5 for summarization
        ])
        
        if len(combined_content) > 10000:  # Truncate if too long
            combined_content = combined_content[:10000] + "..."
        
        try:
            # Get LLM adapter for summarization
            llm_adapter = await self.coordinator.get_summarization_adapter()
            
            # Create summarization prompt
            prompt = f"""Summarize the following code documents into a concise overview that captures the main functionality and purpose:

{combined_content}

Provide a summary that:
1. Identifies the main functionality or purpose
2. Notes key patterns or architectural elements  
3. Is concise but informative (max {self.config.max_summary_length} words)

Summary:"""
            
            summary = await llm_adapter.generate_text(
                prompt=prompt,
                max_tokens=self.config.max_summary_length,
                temperature=0.1
            )
            
            return summary.strip()
            
        except Exception as e:
            logger.warning(f"Failed to generate cluster summary: {e}")
            # Fallback: use first document's content
            return nodes[0].content
    
    def _calculate_cluster_metrics(
        self, 
        cluster_embeddings: List[np.ndarray], 
        cluster_id: int
    ) -> ClusterMetrics:
        """Calculate quality metrics for a cluster."""
        if not cluster_embeddings:
            return ClusterMetrics(
                silhouette_score=0.0,
                coherence_score=0.0,
                size=0,
                density=0.0,
                diameter=0.0,
                intra_cluster_distance=0.0,
                inter_cluster_distance=0.0
            )
        
        embeddings = np.array(cluster_embeddings)
        size = len(embeddings)
        
        # Calculate pairwise distances within cluster
        if size > 1:
            from scipy.spatial.distance import pdist
            distances = pdist(embeddings, metric='cosine')
            intra_distance = np.mean(distances)
            diameter = np.max(distances)
            density = 1.0 / (1.0 + intra_distance)
        else:
            intra_distance = 0.0
            diameter = 0.0
            density = 1.0
        
        return ClusterMetrics(
            silhouette_score=0.0,  # Would need all clusters to compute
            coherence_score=density,  # Use density as coherence proxy
            size=size,
            density=density,
            diameter=diameter,
            intra_cluster_distance=intra_distance,
            inter_cluster_distance=0.0,  # Would need other clusters
        )
    
    async def _create_root_node(self, nodes: List[RaptorNode], tree: RaptorTree) -> None:
        """Create root summary node for the entire tree."""
        if len(nodes) == 1:
            # Single node becomes root
            node = nodes[0]
            node.node_type = NodeType.ROOT
            tree.root_id = node.node_id
            return
        
        # Create summary of all nodes
        summary_text = await self._summarize_cluster(nodes)
        
        root_node = RaptorNode(
            node_type=NodeType.ROOT,
            level=max(node.level for node in nodes) + 1,
            content=summary_text
        )
        
        # Calculate root embedding (average of all nodes)
        embeddings = [
            np.array(node.embedding) for node in nodes 
            if node.embedding
        ]
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            root_node.embedding = avg_embedding.tolist()
        
        # Collect all source files
        for node in nodes:
            root_node.source_files.extend(node.source_files)
            tree.create_parent_child_relationship(root_node.node_id, node.node_id)
        
        tree.add_node(root_node)
        tree.root_id = root_node.node_id
        
        logger.debug(f"Created root node summarizing {len(nodes)} nodes")
    
    async def query_tree(
        self,
        tree: RaptorTree,
        query_embedding: np.ndarray,
        top_k: int = 5,
        traverse_strategy: str = "top_down"
    ) -> List[Tuple[RaptorNode, float]]:
        """
        Query the RAPTOR tree for relevant nodes.
        
        Args:
            tree: RAPTOR tree to query
            query_embedding: Query embedding vector
            top_k: Number of results to return
            traverse_strategy: How to traverse tree ("top_down", "all_levels", "leaves_only")
            
        Returns:
            List of (node, similarity_score) tuples
        """
        if traverse_strategy == "leaves_only":
            candidates = tree.get_leaves()
        elif traverse_strategy == "all_levels":
            candidates = list(tree.nodes.values())
        else:  # top_down
            candidates = await self._top_down_traversal(tree, query_embedding, top_k)
        
        # Calculate similarities
        scored_nodes = []
        for node in candidates:
            if node.embedding:
                node_embedding = np.array(node.embedding)
                similarity = self._cosine_similarity(query_embedding, node_embedding)
                scored_nodes.append((node, similarity))
        
        # Sort by similarity and return top-k
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        return scored_nodes[:top_k]
    
    async def _top_down_traversal(
        self,
        tree: RaptorTree,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[RaptorNode]:
        """Traverse tree top-down, following most similar paths."""
        if not tree.root_id:
            return tree.get_leaves()  # Fallback to leaves
        
        candidates = []
        current_level = [tree.get_node(tree.root_id)]
        
        while current_level and len(candidates) < top_k * 2:  # Get extra candidates
            next_level = []
            
            # Score current level
            level_scores = []
            for node in current_level:
                if node and node.embedding:
                    similarity = self._cosine_similarity(
                        query_embedding, np.array(node.embedding)
                    )
                    level_scores.append((node, similarity))
            
            # Sort by similarity
            level_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Add top nodes as candidates and get their children
            for node, score in level_scores:
                candidates.append(node)
                children = tree.get_children(node.node_id)
                next_level.extend(children)
            
            current_level = next_level
        
        return candidates
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        except ZeroDivisionError:
            return 0.0


# Convenience functions
async def create_raptor_tree(
    embeddings: List[np.ndarray],
    documents: List[str],
    metadata: Optional[List[Dict[str, Any]]] = None,
    config: Optional[RaptorConfig] = None
) -> RaptorTree:
    """Create RAPTOR tree from embeddings and documents."""
    processor = RaptorProcessor(config=config)
    return await processor.process_embeddings(embeddings, documents, metadata)


async def query_raptor_tree(
    tree: RaptorTree,
    query_embedding: np.ndarray,
    top_k: int = 5,
    strategy: str = "top_down"
) -> List[Tuple[RaptorNode, float]]:
    """Query RAPTOR tree for relevant nodes."""
    processor = RaptorProcessor()
    await processor.initialize()
    return await processor.query_tree(tree, query_embedding, top_k, strategy)