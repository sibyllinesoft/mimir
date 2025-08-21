"""
Unit tests for data schemas and validation.

Tests Pydantic models, serialization, and validation logic
for all core data structures.
"""


import pytest

from src.repoindex.data.schemas import (
    Citation,
    CodeSnippet,
    IndexConfig,
    IndexCounts,
    IndexManifest,
    PipelineStatus,
    RepoInfo,
    SearchResult,
    SearchScores,
    SerenaGraph,
    SymbolEntry,
    SymbolType,
    VectorChunk,
)


class TestRepoInfo:
    """Test repository information model."""

    def test_basic_repo_info(self):
        """Test basic repo info creation."""
        repo = RepoInfo(root="/path/to/repo", rev="main", worktree_dirty=False)

        assert repo.root == "/path/to/repo"
        assert repo.rev == "main"
        assert not repo.worktree_dirty

    def test_repo_info_dirty_worktree(self):
        """Test repo info with dirty worktree."""
        repo = RepoInfo(root="/path/to/repo", rev="feature-branch", worktree_dirty=True)

        assert repo.worktree_dirty
        assert repo.rev == "feature-branch"


class TestIndexConfig:
    """Test index configuration model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IndexConfig()

        assert "ts" in config.languages
        assert "tsx" in config.languages
        assert "js" in config.languages
        assert "jsx" in config.languages

        assert "node_modules/" in config.excludes
        assert ".git/" in config.excludes

        assert config.context_lines == 5  # Per VISION.md specification
        assert config.max_files_to_embed is None  # Per actual implementation

    def test_custom_config(self):
        """Test custom configuration."""
        config = IndexConfig(
            languages=["py", "js"],
            excludes=["dist/", "build/"],
            context_lines=5,
            max_files_to_embed=500,
        )

        assert config.languages == ["py", "js"]
        assert config.excludes == ["dist/", "build/"]
        assert config.context_lines == 5
        assert config.max_files_to_embed == 500


class TestIndexManifest:
    """Test index manifest model."""

    def test_manifest_creation(self, default_config, sample_repo_info):
        """Test manifest creation with defaults."""
        manifest = IndexManifest(repo=sample_repo_info, config=default_config)

        assert manifest.repo == sample_repo_info
        assert manifest.config == default_config
        assert manifest.index_id  # Should have auto-generated ULID
        assert isinstance(manifest.counts, IndexCounts)
        assert manifest.counts.files_total == 0  # Default

    def test_manifest_serialization(self, default_config, sample_repo_info):
        """Test manifest JSON serialization."""
        manifest = IndexManifest(repo=sample_repo_info, config=default_config)

        # Should serialize without errors
        json_data = manifest.model_dump()
        assert "index_id" in json_data
        assert "repo" in json_data
        assert "config" in json_data

        # Should deserialize back
        restored = IndexManifest.model_validate(json_data)
        assert restored.index_id == manifest.index_id
        assert restored.repo.root == manifest.repo.root


class TestSymbolEntry:
    """Test symbol entry model."""

    def test_symbol_definition(self):
        """Test symbol definition entry."""
        entry = SymbolEntry(
            type=SymbolType.DEF,
            path="src/utils.ts",
            span=(10, 20),
            symbol="calculateSum",
            sig="function calculateSum(a: number, b: number): number",
        )

        assert entry.type == SymbolType.DEF
        assert entry.path == "src/utils.ts"
        assert entry.span == (10, 20)
        assert entry.symbol == "calculateSum"
        assert "function" in entry.sig

    def test_symbol_reference(self):
        """Test symbol reference entry."""
        entry = SymbolEntry(
            type=SymbolType.REF, path="src/main.ts", span=(5, 15), symbol="calculateSum"
        )

        assert entry.type == SymbolType.REF
        assert entry.symbol == "calculateSum"
        assert entry.sig is None  # References don't have signatures

    def test_symbol_call(self):
        """Test symbol call entry."""
        entry = SymbolEntry(
            type=SymbolType.CALL,
            path="src/main.ts",
            span=(8, 18),
            caller="main",
            callee="calculateSum",
        )

        assert entry.type == SymbolType.CALL
        assert entry.caller == "main"
        assert entry.callee == "calculateSum"


class TestVectorChunk:
    """Test vector chunk model."""

    def test_vector_chunk_creation(self):
        """Test vector chunk with embedding."""
        chunk = VectorChunk(
            path="src/utils.ts",
            span=(10, 30),
            content="function add(a, b) { return a + b; }",
            embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
        )

        assert chunk.path == "src/utils.ts"
        assert chunk.span == (10, 30)
        assert len(chunk.embedding) == 5
        assert chunk.embedding[0] == 0.1

    def test_vector_chunk_validation(self):
        """Test vector chunk validation."""
        # Empty embedding should be valid
        chunk = VectorChunk(path="test.js", span=(0, 10), content="var x = 1;", embedding=[])
        assert chunk.embedding == []


class TestCodeSnippet:
    """Test code snippet model."""

    def test_code_snippet_with_context(self):
        """Test code snippet with context lines."""
        snippet = CodeSnippet(
            path="src/main.ts",
            span=(10, 15),
            hash="abc123",
            pre="// Setup code\nconst config = {};",
            text="function main() {\n  return process();\n}",
            post="// Cleanup\nmain();",
            line_start=8,
            line_end=17,
        )

        assert snippet.path == "src/main.ts"
        assert snippet.span == (10, 15)
        assert snippet.hash == "abc123"
        assert "Setup code" in snippet.pre
        assert "function main" in snippet.text
        assert "Cleanup" in snippet.post
        assert snippet.line_start == 8
        assert snippet.line_end == 17


class TestSearchResult:
    """Test search result model."""

    def test_search_result_creation(self):
        """Test search result with all scores."""
        scores = SearchScores(vector=0.8, symbol=0.9, graph=0.3)

        content = CodeSnippet(
            path="src/test.ts",
            span=(5, 10),
            hash="xyz789",
            pre="",
            text="export const value = 42;",
            post="",
            line_start=5,
            line_end=5,
        )

        citation = Citation(
            repo_root="/repo", rev="main", path="src/test.ts", span=(5, 10), content_sha="xyz789"
        )

        result = SearchResult(
            path="src/test.ts",
            span=(5, 10),
            score=0.85,
            scores=scores,
            content=content,
            citation=citation,
        )

        assert result.path == "src/test.ts"
        assert result.score == 0.85
        assert result.scores.vector == 0.8
        assert result.scores.symbol == 0.9
        assert result.scores.graph == 0.3
        assert result.content.text == "export const value = 42;"
        assert result.citation.repo_root == "/repo"


class TestPipelineStatus:
    """Test pipeline status model."""

    def test_pipeline_status_running(self):
        """Test running pipeline status."""
        from src.repoindex.data.schemas import IndexState, PipelineStage

        status = PipelineStatus(
            index_id="test_index_123",
            state=IndexState.RUNNING,
            stage=PipelineStage.SERENA,
            progress=60,  # Integer percentage
            message="Processing TypeScript files",
            error=None,
        )

        assert status.stage == PipelineStage.SERENA
        assert status.progress == 60
        assert "TypeScript" in status.message
        assert status.error is None
        assert status.state == IndexState.RUNNING

    def test_pipeline_status_complete(self):
        """Test completed pipeline status."""
        from src.repoindex.data.schemas import IndexState, PipelineStage

        status = PipelineStatus(
            index_id="test_index_123",
            state=IndexState.DONE,
            stage=PipelineStage.BUNDLE,
            progress=100,  # Integer percentage
            message="Index creation complete",
            error=None,
        )

        assert status.state == IndexState.DONE
        assert status.progress == 100

    def test_pipeline_status_error(self):
        """Test error pipeline status."""
        from src.repoindex.data.schemas import IndexState, PipelineStage

        status = PipelineStatus(
            index_id="test_index_123",
            state=IndexState.FAILED,
            stage=PipelineStage.REPOMAPPER,
            progress=30,  # Integer percentage
            message="Processing failed",
            error="Tree-sitter parsing error",
        )

        assert status.state == IndexState.FAILED
        assert status.error == "Tree-sitter parsing error"
        assert status.progress == 30


class TestSerenaGraph:
    """Test Serena graph model."""

    def test_empty_graph(self):
        """Test empty graph creation."""
        graph = SerenaGraph(entries=[], file_count=0, symbol_count=0)

        assert len(graph.entries) == 0
        assert graph.file_count == 0
        assert graph.symbol_count == 0

    def test_graph_with_entries(self):
        """Test graph with symbol entries."""
        entries = [
            SymbolEntry(type=SymbolType.DEF, path="src/utils.ts", span=(10, 20), symbol="helper"),
            SymbolEntry(type=SymbolType.REF, path="src/main.ts", span=(5, 11), symbol="helper"),
        ]

        graph = SerenaGraph(
            entries=entries, file_count=2, symbol_count=1  # utils.ts and main.ts  # helper symbol
        )

        assert len(graph.entries) == 2
        assert graph.file_count == 2
        assert graph.symbol_count == 1
        definitions = graph.get_definitions()
        assert len(definitions) == 1
        assert definitions[0].symbol == "helper"

        references = graph.get_references("helper")
        assert len(references) == 1
        assert references[0].path == "src/main.ts"


@pytest.mark.asyncio
class TestSchemaIntegration:
    """Test schema integration and workflows."""

    async def test_manifest_workflow(self, temp_dir, default_config):
        """Test complete manifest creation workflow."""
        # Create repo info
        repo = RepoInfo(root=str(temp_dir), rev="main", worktree_dirty=False)

        # Create manifest
        manifest = IndexManifest(repo=repo, config=default_config)

        # Update counts
        manifest.counts.files_total = 10
        manifest.counts.files_indexed = 8
        manifest.counts.symbols_defs = 45

        # Serialize and restore
        json_data = manifest.model_dump()
        restored = IndexManifest.model_validate(json_data)

        assert restored.counts.files_total == 10
        assert restored.counts.files_indexed == 8
        assert restored.counts.symbols_defs == 45
        assert restored.repo.root == str(temp_dir)

    async def test_search_workflow(self):
        """Test search result workflow."""
        # Create search scores
        scores = SearchScores(vector=0.7, symbol=0.8)

        # Create content snippet
        content = CodeSnippet(
            path="src/api.ts",
            span=(20, 35),
            hash="content123",
            pre="// API helpers",
            text="export async function fetchData() {\n  return await api.get('/data');\n}",
            post="// Export more functions",
            line_start=18,
            line_end=23,
        )

        # Create citation
        citation = Citation(
            repo_root="/project",
            rev="main",
            path="src/api.ts",
            span=(20, 35),
            content_sha="content123",
        )

        # Create result
        result = SearchResult(
            path="src/api.ts",
            span=(20, 35),
            score=0.75,
            scores=scores,
            content=content,
            citation=citation,
        )

        # Verify all components work together
        assert result.path == content.path == citation.path
        assert result.span == content.span == citation.span
        assert result.content.hash == citation.content_sha
        assert result.score == 0.75
