"""
RAPTOR Tree Data Structures for Hierarchical Indexing.

Implements tree structures for RAPTOR (Recursive Abstractive Processing 
for Tree-Organized Retrieval) hierarchical document organization.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from pathlib import Path
from datetime import datetime
from enum import Enum

import numpy as np

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of nodes in the RAPTOR tree."""
    LEAF = "leaf"  # Original document/code snippet
    INTERNAL = "internal"  # Cluster summary node
    ROOT = "root"  # Top-level summary


@dataclass
class ClusterMetrics:
    """Metrics for cluster quality assessment."""
    silhouette_score: float
    coherence_score: float
    size: int
    density: float
    diameter: float
    intra_cluster_distance: float
    inter_cluster_distance: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ClusterMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class RaptorNode:
    """
    A node in the RAPTOR hierarchical tree.
    
    Each node represents either an original document (leaf) or 
    a cluster summary (internal/root).
    """
    
    # Core identifiers
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    node_type: NodeType = NodeType.LEAF
    level: int = 0  # 0 for leaves, increases up the tree
    
    # Content
    content: str = ""  # Original text or summary
    embedding: Optional[List[float]] = None
    
    # Tree structure
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    
    # Metadata
    cluster_id: Optional[int] = None
    cluster_metrics: Optional[ClusterMetrics] = None
    source_files: List[str] = field(default_factory=list)  # Original files represented
    created_at: datetime = field(default_factory=datetime.now)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation."""
        if not self.content and self.node_type != NodeType.ROOT:
            logger.warning(f"Node {self.node_id} has empty content")
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return self.node_type == NodeType.LEAF
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.node_type == NodeType.ROOT
    
    @property  
    def has_children(self) -> bool:
        """Check if this node has children."""
        return len(self.children_ids) > 0
    
    @property
    def embedding_array(self) -> Optional[np.ndarray]:
        """Get embedding as numpy array."""
        if self.embedding is None:
            return None
        return np.array(self.embedding)
    
    def add_child(self, child_node_id: str) -> None:
        """Add a child node ID."""
        if child_node_id not in self.children_ids:
            self.children_ids.append(child_node_id)
    
    def remove_child(self, child_node_id: str) -> None:
        """Remove a child node ID."""
        if child_node_id in self.children_ids:
            self.children_ids.remove(child_node_id)
    
    def set_parent(self, parent_node_id: str) -> None:
        """Set the parent node ID."""
        self.parent_id = parent_node_id
    
    def update_content(self, new_content: str, new_embedding: Optional[List[float]] = None) -> None:
        """Update node content and embedding."""
        self.content = new_content
        if new_embedding is not None:
            self.embedding = new_embedding
    
    def add_source_file(self, file_path: str) -> None:
        """Add a source file to this node."""
        if file_path not in self.source_files:
            self.source_files.append(file_path)
    
    def get_descendant_count(self, tree: 'RaptorTree') -> int:
        """Get total number of descendants."""
        count = 0
        for child_id in self.children_ids:
            child = tree.get_node(child_id)
            if child:
                count += 1 + child.get_descendant_count(tree)
        return count
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary for serialization."""
        data = asdict(self)
        
        # Convert datetime to string
        if isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        
        # Convert enums to strings
        data['node_type'] = self.node_type.value
        
        # Handle cluster metrics
        if self.cluster_metrics:
            data['cluster_metrics'] = self.cluster_metrics.to_dict()
        
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RaptorNode':
        """Create node from dictionary."""
        # Handle datetime
        if 'created_at' in data and isinstance(data['created_at'], str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        
        # Handle enums
        if 'node_type' in data and isinstance(data['node_type'], str):
            data['node_type'] = NodeType(data['node_type'])
        
        # Handle cluster metrics
        if 'cluster_metrics' in data and data['cluster_metrics']:
            data['cluster_metrics'] = ClusterMetrics.from_dict(data['cluster_metrics'])
        
        return cls(**data)


class RaptorTree:
    """
    Hierarchical tree structure for RAPTOR indexing.
    
    Manages the hierarchical organization of documents and clusters,
    providing efficient access and traversal methods.
    """
    
    def __init__(self, tree_id: Optional[str] = None):
        """
        Initialize RAPTOR tree.
        
        Args:
            tree_id: Unique identifier for this tree
        """
        self.tree_id = tree_id or str(uuid.uuid4())
        self.nodes: Dict[str, RaptorNode] = {}
        self.root_id: Optional[str] = None
        self.created_at = datetime.now()
        self.metadata: Dict[str, Any] = {}
        
        # Index structures for efficient access
        self._nodes_by_level: Dict[int, List[str]] = {}
        self._leaves: Set[str] = set()
        self._dirty = False  # Track if indices need rebuilding
    
    def add_node(self, node: RaptorNode) -> None:
        """Add a node to the tree."""
        self.nodes[node.node_id] = node
        self._dirty = True
        
        # Update root if this is a root node
        if node.is_root:
            self.root_id = node.node_id
        
        logger.debug(f"Added node {node.node_id} (type: {node.node_type}, level: {node.level})")
    
    def get_node(self, node_id: str) -> Optional[RaptorNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)
    
    def remove_node(self, node_id: str) -> bool:
        """Remove a node from the tree."""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        
        # Remove from parent's children
        if node.parent_id:
            parent = self.get_node(node.parent_id)
            if parent:
                parent.remove_child(node_id)
        
        # Update children's parent
        for child_id in node.children_ids:
            child = self.get_node(child_id)
            if child:
                child.parent_id = node.parent_id
        
        # Remove from tree
        del self.nodes[node_id]
        self._dirty = True
        
        # Update root if necessary
        if self.root_id == node_id:
            self.root_id = None
        
        return True
    
    def create_parent_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Create parent-child relationship between nodes."""
        parent = self.get_node(parent_id)
        child = self.get_node(child_id)
        
        if not parent or not child:
            return False
        
        parent.add_child(child_id)
        child.set_parent(parent_id)
        self._dirty = True
        return True
    
    def get_children(self, node_id: str) -> List[RaptorNode]:
        """Get all children of a node."""
        node = self.get_node(node_id)
        if not node:
            return []
        
        children = []
        for child_id in node.children_ids:
            child = self.get_node(child_id)
            if child:
                children.append(child)
        return children
    
    def get_ancestors(self, node_id: str) -> List[RaptorNode]:
        """Get all ancestors of a node (path to root)."""
        ancestors = []
        current_id = node_id
        
        while current_id:
            node = self.get_node(current_id)
            if not node or not node.parent_id:
                break
            
            parent = self.get_node(node.parent_id)
            if parent:
                ancestors.append(parent)
                current_id = parent.node_id
            else:
                break
        
        return ancestors
    
    def get_descendants(self, node_id: str) -> List[RaptorNode]:
        """Get all descendants of a node."""
        descendants = []
        node = self.get_node(node_id)
        if not node:
            return descendants
        
        # Use BFS to get all descendants
        queue = list(node.children_ids)
        while queue:
            child_id = queue.pop(0)
            child = self.get_node(child_id)
            if child:
                descendants.append(child)
                queue.extend(child.children_ids)
        
        return descendants
    
    def get_leaves(self) -> List[RaptorNode]:
        """Get all leaf nodes."""
        if self._dirty:
            self._rebuild_indices()
        
        return [self.get_node(node_id) for node_id in self._leaves if self.get_node(node_id)]
    
    def get_nodes_by_level(self, level: int) -> List[RaptorNode]:
        """Get all nodes at a specific level."""
        if self._dirty:
            self._rebuild_indices()
        
        node_ids = self._nodes_by_level.get(level, [])
        return [self.get_node(node_id) for node_id in node_ids if self.get_node(node_id)]
    
    def get_max_level(self) -> int:
        """Get the maximum level in the tree."""
        if not self.nodes:
            return 0
        return max(node.level for node in self.nodes.values())
    
    def get_tree_stats(self) -> Dict[str, Any]:
        """Get statistics about the tree."""
        if self._dirty:
            self._rebuild_indices()
        
        total_nodes = len(self.nodes)
        leaf_count = len(self._leaves)
        internal_count = total_nodes - leaf_count
        max_level = self.get_max_level()
        
        level_counts = {}
        for level in range(max_level + 1):
            level_counts[level] = len(self._nodes_by_level.get(level, []))
        
        return {
            "tree_id": self.tree_id,
            "total_nodes": total_nodes,
            "leaf_nodes": leaf_count,
            "internal_nodes": internal_count,
            "max_level": max_level,
            "nodes_per_level": level_counts,
            "created_at": self.created_at.isoformat(),
        }
    
    def validate_tree(self) -> Tuple[bool, List[str]]:
        """Validate tree structure and return errors."""
        errors = []
        
        # Check for orphaned nodes (except root)
        for node_id, node in self.nodes.items():
            if not node.is_root and not node.parent_id:
                errors.append(f"Orphaned node: {node_id}")
        
        # Check for broken parent-child relationships
        for node_id, node in self.nodes.items():
            # Check parent relationship
            if node.parent_id:
                parent = self.get_node(node.parent_id)
                if not parent:
                    errors.append(f"Node {node_id} has non-existent parent {node.parent_id}")
                elif node_id not in parent.children_ids:
                    errors.append(f"Parent {node.parent_id} doesn't list {node_id} as child")
            
            # Check children relationships
            for child_id in node.children_ids:
                child = self.get_node(child_id)
                if not child:
                    errors.append(f"Node {node_id} has non-existent child {child_id}")
                elif child.parent_id != node_id:
                    errors.append(f"Child {child_id} doesn't list {node_id} as parent")
        
        # Check for cycles
        visited = set()
        
        def has_cycle(node_id: str, path: Set[str]) -> bool:
            if node_id in path:
                return True
            if node_id in visited:
                return False
            
            node = self.get_node(node_id)
            if not node:
                return False
            
            path.add(node_id)
            for child_id in node.children_ids:
                if has_cycle(child_id, path):
                    return True
            path.remove(node_id)
            visited.add(node_id)
            return False
        
        if self.root_id and has_cycle(self.root_id, set()):
            errors.append("Tree contains cycles")
        
        return len(errors) == 0, errors
    
    def _rebuild_indices(self) -> None:
        """Rebuild internal indices for efficient access."""
        self._nodes_by_level.clear()
        self._leaves.clear()
        
        for node_id, node in self.nodes.items():
            # Group by level
            if node.level not in self._nodes_by_level:
                self._nodes_by_level[node.level] = []
            self._nodes_by_level[node.level].append(node_id)
            
            # Track leaves
            if node.is_leaf or len(node.children_ids) == 0:
                self._leaves.add(node_id)
        
        self._dirty = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tree to dictionary for serialization."""
        return {
            "tree_id": self.tree_id,
            "root_id": self.root_id,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RaptorTree':
        """Create tree from dictionary."""
        tree = cls(tree_id=data["tree_id"])
        tree.root_id = data.get("root_id")
        tree.created_at = datetime.fromisoformat(data["created_at"])
        tree.metadata = data.get("metadata", {})
        
        # Load nodes
        for node_id, node_data in data["nodes"].items():
            node = RaptorNode.from_dict(node_data)
            tree.add_node(node)
        
        return tree
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save tree to JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        logger.info(f"Saved RAPTOR tree {self.tree_id} to {path}")
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'RaptorTree':
        """Load tree from JSON file."""
        path = Path(file_path)
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        tree = cls.from_dict(data)
        logger.info(f"Loaded RAPTOR tree {tree.tree_id} from {path}")
        return tree