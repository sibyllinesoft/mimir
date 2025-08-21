"""
Core data schemas for the Mimir indexing system.

Defines Pydantic models for all data structures used throughout the pipeline
and MCP interface, ensuring type safety and consistent serialization.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator
import uuid


class IndexState(str, Enum):
    """Pipeline execution states."""
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineStage(str, Enum):
    """Individual pipeline stage identifiers."""
    ACQUIRE = "acquire"
    REPOMAPPER = "repomapper"
    SERENA = "serena"
    LEANN = "leann"
    SNIPPETS = "snippets"
    BUNDLE = "bundle"


class SymbolType(str, Enum):
    """Symbol entry types in the graph."""
    DEF = "def"      # Symbol definition
    REF = "ref"      # Symbol reference
    CALL = "call"    # Function/method call
    IMPORT = "import"  # Import statement


# Core Configuration Models

class FeatureConfig(BaseModel):
    """Feature flags for different search modalities."""
    vector: bool = True
    symbol: bool = True
    graph: bool = True


class IndexConfig(BaseModel):
    """Configuration for index creation and processing."""
    languages: List[str] = Field(default_factory=lambda: ["ts", "tsx", "js", "jsx", "md", "mdx", "json", "yaml"])
    excludes: List[str] = Field(default_factory=lambda: [
        "node_modules/", "dist/", "build/", ".next/", "coverage/",
        "__pycache__/", ".git/", ".vscode/", ".idea/"
    ])
    context_lines: int = 5
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    max_files_to_embed: Optional[int] = None
    imports_policy: Dict[str, bool] = Field(default_factory=lambda: {
        "types_for_first_order": True,
        "code_for_direct_imports": True
    })
    watch_mode: bool = False


# Repository Information Models

class RepoInfo(BaseModel):
    """Repository metadata and state."""
    root: str  # Absolute path to repository root
    rev: str   # HEAD commit SHA
    worktree_dirty: bool  # Whether there are uncommitted changes
    
    @field_validator('root')
    @classmethod
    def validate_absolute_path(cls, v: str) -> str:
        """Ensure repository root is an absolute path."""
        path = Path(v)
        if not path.is_absolute():
            raise ValueError("Repository root must be an absolute path")
        return str(path.resolve())


class IndexCounts(BaseModel):
    """Statistics about indexed content."""
    files_total: int = 0
    files_indexed: int = 0
    symbols_defs: int = 0
    symbols_refs: int = 0
    vectors: int = 0
    chunks: int = 0


class ArtifactPaths(BaseModel):
    """Paths to all generated artifacts."""
    repomap: str = "repomap.json"
    serena_graph: str = "serena_graph.jsonl"
    leann_index: str = "leann.index"
    vectors: str = "vectors.bin"
    snippets: str = "snippets.jsonl"
    vendor_types: str = "types/"
    vendor_sources: str = "vendor_src/"
    bundle: str = "bundle.tar.zst"


class ToolVersions(BaseModel):
    """Versions of external tools used in pipeline."""
    repomapper: str = ""
    serena: str = ""
    leann: str = ""
    tsserver: str = ""
    mimir: str = "0.1.0"


# Main Index Manifest

class IndexManifest(BaseModel):
    """Complete index metadata and artifact registry."""
    index_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    repo: RepoInfo
    config: IndexConfig
    counts: IndexCounts = Field(default_factory=IndexCounts)
    paths: ArtifactPaths = Field(default_factory=ArtifactPaths)
    versions: ToolVersions = Field(default_factory=ToolVersions)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save manifest to JSON file."""
        with open(file_path, 'w') as f:
            json.dump(self.dict(), f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'IndexManifest':
        """Load manifest from JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Pipeline Status Models

class PipelineStatus(BaseModel):
    """Current pipeline execution status."""
    index_id: str
    state: IndexState
    stage: Optional[PipelineStage] = None
    progress: int = 0  # 0-100 percentage
    message: str = ""
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def update_progress(self, stage: PipelineStage, progress: int, message: str = "") -> None:
        """Update pipeline progress."""
        self.stage = stage
        self.progress = progress
        self.message = message
        if self.state == IndexState.QUEUED:
            self.state = IndexState.RUNNING
            self.started_at = datetime.utcnow()
    
    def mark_completed(self) -> None:
        """Mark pipeline as successfully completed."""
        self.state = IndexState.DONE
        self.progress = 100
        self.completed_at = datetime.now(timezone.utc)
    
    def mark_failed(self, error: str) -> None:
        """Mark pipeline as failed with error message."""
        self.state = IndexState.FAILED
        self.error = error
        self.completed_at = datetime.now(timezone.utc)


# Symbol Graph Models

class SymbolEntry(BaseModel):
    """Individual symbol entry in the graph."""
    type: SymbolType
    symbol: Optional[str] = None  # Fully qualified symbol name
    path: str  # Source file path (relative to repo root)
    span: Tuple[int, int]  # Byte offset range [start, end)
    sig: Optional[str] = None  # Type signature (for definitions)
    caller: Optional[str] = None  # Calling symbol (for calls)
    callee: Optional[str] = None  # Called symbol (for calls)
    from_module: Optional[str] = None  # Import source (for imports)
    
    @field_validator('span')
    @classmethod
    def validate_span(cls, v: Tuple[int, int]) -> Tuple[int, int]:
        """Ensure span is valid (start < end)."""
        start, end = v
        if start >= end:
            raise ValueError("Span start must be less than end")
        return v


class SerenaGraph(BaseModel):
    """Complete symbol graph from Serena analysis."""
    entries: List[SymbolEntry]
    file_count: int
    symbol_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_definitions(self) -> List[SymbolEntry]:
        """Get all definition entries."""
        return [entry for entry in self.entries if entry.type == SymbolType.DEF]
    
    def get_references(self, symbol: str) -> List[SymbolEntry]:
        """Get all references to a specific symbol."""
        return [entry for entry in self.entries 
                if entry.type == SymbolType.REF and entry.symbol == symbol]
    
    def save_to_jsonl(self, file_path: Union[str, Path]) -> None:
        """Save graph entries to JSONL file."""
        with open(file_path, 'w') as f:
            for entry in self.entries:
                f.write(entry.json() + '\n')
    
    @classmethod
    def load_from_jsonl(cls, file_path: Union[str, Path]) -> 'SerenaGraph':
        """Load graph entries from JSONL file."""
        entries = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(SymbolEntry.parse_raw(line))
        
        return cls(
            entries=entries,
            file_count=len(set(entry.path for entry in entries)),
            symbol_count=len(set(entry.symbol for entry in entries if entry.symbol))
        )


# Code Snippet Models

class CodeSnippet(BaseModel):
    """Source code snippet with context."""
    path: str  # File path relative to repo root
    span: Tuple[int, int]  # Byte offset range
    hash: str  # SHA256 hash of content for verification
    pre: str   # Lines before the match
    text: str  # Matched content
    post: str  # Lines after the match
    line_start: int  # Starting line number
    line_end: int    # Ending line number


class SnippetCollection(BaseModel):
    """Collection of code snippets."""
    snippets: List[CodeSnippet]
    total_count: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def save_to_jsonl(self, file_path: Union[str, Path]) -> None:
        """Save snippets to JSONL file."""
        with open(file_path, 'w') as f:
            for snippet in self.snippets:
                f.write(snippet.json() + '\n')
    
    @classmethod
    def load_from_jsonl(cls, file_path: Union[str, Path]) -> 'SnippetCollection':
        """Load snippets from JSONL file."""
        snippets = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    snippets.append(CodeSnippet.parse_raw(line))
        
        return cls(snippets=snippets, total_count=len(snippets))


# Search Models

class Citation(BaseModel):
    """Source citation for search results."""
    repo_root: str
    rev: str
    path: str
    span: Tuple[int, int]
    content_sha: str


class SearchScores(BaseModel):
    """Individual search modality scores."""
    vector: float = 0.0
    symbol: float = 0.0
    graph: float = 0.0


class SearchResult(BaseModel):
    """Individual search result with content and metadata."""
    path: str
    span: Tuple[int, int]
    score: float
    scores: SearchScores
    content: CodeSnippet
    citation: Citation
    
    @property
    def combined_score(self) -> float:
        """Compute combined score from individual modalities."""
        return (
            1.0 * self.scores.vector +
            0.9 * self.scores.symbol +
            0.3 * self.scores.graph
        )


class SearchResponse(BaseModel):
    """Complete search response."""
    query: str
    results: List[SearchResult]
    total_count: int
    features_used: FeatureConfig
    execution_time_ms: float
    index_id: str


class AskResponse(BaseModel):
    """Response from ask_index operation."""
    question: str
    answer: str
    citations: List[Citation]
    execution_time_ms: float
    index_id: str


# MCP Tool Request/Response Models

class EnsureRepoIndexRequest(BaseModel):
    """Request to ensure repository is indexed."""
    path: str
    rev: Optional[str] = None
    language: str = "ts"
    index_opts: Optional[Dict[str, Any]] = None


class EnsureRepoIndexResponse(BaseModel):
    """Response from ensure_repo_index tool."""
    index_id: str
    status_uri: str
    manifest_uri: str


class GetRepoBundleRequest(BaseModel):
    """Request to get repository bundle."""
    index_id: str


class GetRepoBundleResponse(BaseModel):
    """Response from get_repo_bundle tool."""
    bundle_uri: str
    manifest_uri: str


class SearchRepoRequest(BaseModel):
    """Request to search repository."""
    index_id: str
    query: str
    k: int = 20
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    context_lines: int = 5


class AskIndexRequest(BaseModel):
    """Request to ask question of index."""
    index_id: str
    question: str
    context_lines: int = 5


class CancelRequest(BaseModel):
    """Request to cancel indexing operation."""
    index_id: str


class CancelResponse(BaseModel):
    """Response from cancel operation."""
    ok: bool
    message: str = ""


# RepoMapper Models

class FileRank(BaseModel):
    """File importance ranking from RepoMapper."""
    path: str
    rank: float
    centrality: float
    dependencies: List[str]


class DependencyEdge(BaseModel):
    """Dependency relationship between files."""
    source: str
    target: str
    weight: float
    edge_type: str  # import, call, etc.


class RepoMap(BaseModel):
    """Repository structure map from RepoMapper."""
    file_ranks: List[FileRank]
    edges: List[DependencyEdge]
    total_files: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    
    def get_ordered_files(self) -> List[str]:
        """Get files ordered by importance ranking."""
        return [fr.path for fr in sorted(self.file_ranks, key=lambda x: x.rank, reverse=True)]


# Vector Index Models

class VectorChunk(BaseModel):
    """Individual chunk for vector embedding."""
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    path: str
    span: Tuple[int, int]
    content: str
    embedding: Optional[List[float]] = None
    token_count: int = 0


class VectorIndex(BaseModel):
    """Vector search index metadata."""
    chunks: List[VectorChunk]
    dimension: int
    total_tokens: int
    model_name: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)


# Error Models

class PipelineError(BaseModel):
    """Pipeline execution error."""
    stage: PipelineStage
    error_type: str
    message: str
    traceback: Optional[str] = None
    recoverable: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ValidationError(BaseModel):
    """Data validation error."""
    field: str
    value: Any
    message: str
    error_type: str = "validation_error"