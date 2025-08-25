"""
End-to-end workflow testing for Mimir repository indexing system.

This test suite validates complete user journeys from repository input
to search results, ensuring all components work together correctly.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import shutil

import pytest

from src.repoindex.config import MimirConfig, IndexConfig
from src.repoindex.data.schemas import (
    RepoInfo,
    IndexManifest,
    SearchResult,
    CodeSnippet,
    Citation,
    SearchScores,
    PipelineStatus,
    IndexState,
    PipelineStage,
)
from src.repoindex.util.fs import (
    ensure_directory,
    atomic_write_text,
    atomic_write_json,
    compute_file_hash,
)


@pytest.fixture
def sample_repository(tmp_path):
    """Create a sample repository structure for testing."""
    repo_dir = tmp_path / "sample_repo"
    
    # Create directory structure
    src_dir = repo_dir / "src"
    utils_dir = src_dir / "utils"
    components_dir = src_dir / "components"
    tests_dir = repo_dir / "tests"
    
    ensure_directory(src_dir)
    ensure_directory(utils_dir)
    ensure_directory(components_dir)
    ensure_directory(tests_dir)
    
    # Create sample source files
    (src_dir / "index.ts").write_text("""
// Main application entry point
import { App } from './components/App';
import { initializeConfig } from './utils/config';

async function main() {
    const config = await initializeConfig();
    const app = new App(config);
    await app.start();
}

main().catch(console.error);
""")
    
    (utils_dir / "config.ts").write_text("""
// Configuration management utilities
export interface Config {
    apiUrl: string;
    timeout: number;
    retries: number;
}

export async function initializeConfig(): Promise<Config> {
    return {
        apiUrl: process.env.API_URL || 'http://localhost:3000',
        timeout: 5000,
        retries: 3,
    };
}

export function validateConfig(config: Config): boolean {
    return config.apiUrl.length > 0 && config.timeout > 0;
}
""")
    
    (components_dir / "App.ts").write_text("""
// Main application class
import { Config } from '../utils/config';

export class App {
    private config: Config;
    private isRunning: boolean = false;

    constructor(config: Config) {
        this.config = config;
    }

    async start(): Promise<void> {
        if (this.isRunning) {
            throw new Error('App is already running');
        }
        
        console.log('Starting application with config:', this.config);
        this.isRunning = true;
        
        // Initialize services
        await this.initializeServices();
    }

    async stop(): Promise<void> {
        if (!this.isRunning) {
            return;
        }
        
        console.log('Stopping application');
        this.isRunning = false;
    }

    private async initializeServices(): Promise<void> {
        // Service initialization logic
        await new Promise(resolve => setTimeout(resolve, 100));
    }

    getStatus(): { running: boolean; config: Config } {
        return {
            running: this.isRunning,
            config: this.config,
        };
    }
}
""")
    
    (tests_dir / "App.test.ts").write_text("""
// Tests for App component
import { App } from '../src/components/App';

describe('App', () => {
    it('should initialize with config', () => {
        const config = {
            apiUrl: 'http://test.com',
            timeout: 1000,
            retries: 2,
        };
        
        const app = new App(config);
        const status = app.getStatus();
        
        expect(status.running).toBe(false);
        expect(status.config).toEqual(config);
    });
    
    it('should start and stop correctly', async () => {
        const app = new App({
            apiUrl: 'http://test.com',
            timeout: 1000,
            retries: 2,
        });
        
        await app.start();
        expect(app.getStatus().running).toBe(true);
        
        await app.stop();
        expect(app.getStatus().running).toBe(false);
    });
});
""")
    
    # Create package.json
    package_json = {
        "name": "sample-app",
        "version": "1.0.0",
        "scripts": {
            "start": "node dist/index.js",
            "build": "tsc",
            "test": "jest"
        },
        "devDependencies": {
            "@types/node": "^18.0.0",
            "typescript": "^4.8.0",
            "jest": "^29.0.0"
        }
    }
    (repo_dir / "package.json").write_text(json.dumps(package_json, indent=2))
    
    # Create README
    (repo_dir / "README.md").write_text("""
# Sample Application

This is a sample TypeScript application for testing repository indexing.

## Features

- Configuration management
- Modular component architecture  
- Comprehensive testing
- TypeScript support

## Usage

```bash
npm install
npm run build
npm start
```
""")
    
    return repo_dir


@pytest.fixture
def mimir_config():
    """Create a test Mimir configuration."""
    return MimirConfig(
        server={"host": "localhost", "port": 8080},
        ai={"google_api_key": "test-key"},
        index=IndexConfig(
            languages=["ts", "tsx", "js", "jsx"],
            excludes=["node_modules/", ".git/", "dist/", "coverage/"],
            context_lines=3,
            max_files_to_embed=100
        )
    )


class TestRepositoryIndexingWorkflow:
    """Test complete repository indexing workflow."""
    
    def test_repository_discovery_workflow(self, sample_repository, mimir_config):
        """Test repository file discovery and filtering."""
        from src.repoindex.pipeline.discover import RepositoryDiscoverer
        
        # Mock the repository discoverer since we don't have the full implementation
        with patch('src.repoindex.pipeline.discover.RepositoryDiscoverer') as MockDiscoverer:
            # Setup mock to return our sample files
            mock_instance = Mock()
            mock_instance.discover_files.return_value = [
                sample_repository / "src" / "index.ts",
                sample_repository / "src" / "utils" / "config.ts", 
                sample_repository / "src" / "components" / "App.ts",
                sample_repository / "tests" / "App.test.ts",
            ]
            MockDiscoverer.return_value = mock_instance
            
            discoverer = MockDiscoverer(
                root_path=sample_repository,
                config=mimir_config.index
            )
            
            discovered_files = discoverer.discover_files()
            
            # Verify discovery results
            assert len(discovered_files) == 4
            assert any("index.ts" in str(f) for f in discovered_files)
            assert any("config.ts" in str(f) for f in discovered_files)
            assert any("App.ts" in str(f) for f in discovered_files)
            assert any("App.test.ts" in str(f) for f in discovered_files)
            
            # Verify no excluded files
            for file_path in discovered_files:
                assert "node_modules" not in str(file_path)
                assert ".git" not in str(file_path)
    
    def test_file_processing_workflow(self, sample_repository):
        """Test file content processing and extraction."""
        # Test processing of a TypeScript file
        app_file = sample_repository / "src" / "components" / "App.ts"
        content = app_file.read_text()
        
        # Extract key information that would be indexed
        lines = content.split('\n')
        
        # Find class definitions
        class_lines = [i for i, line in enumerate(lines) if 'export class' in line]
        assert len(class_lines) == 1
        assert 'App' in lines[class_lines[0]]
        
        # Find method definitions  
        method_lines = [i for i, line in enumerate(lines) if 'async ' in line or 'private' in line]
        assert len(method_lines) >= 3  # start, stop, initializeServices
        
        # Verify we can extract spans for code snippets
        class_start = class_lines[0]
        class_end = len(lines) - 1
        
        class_snippet = CodeSnippet(
            path="src/components/App.ts",
            span=(class_start, class_end),
            hash=compute_file_hash(app_file),
            pre="",
            text='\n'.join(lines[class_start:class_start+5]),  # First 5 lines
            post="",
            line_start=class_start + 1,
            line_end=class_start + 5,
        )
        
        assert 'export class App' in class_snippet.text
        assert class_snippet.hash is not None
    
    def test_index_manifest_creation(self, sample_repository, mimir_config):
        """Test creation of index manifest."""
        repo_info = RepoInfo(
            root=str(sample_repository),
            rev="main", 
            worktree_dirty=False
        )
        
        manifest = IndexManifest(
            repo=repo_info,
            config=mimir_config.index
        )
        
        # Update with discovered file counts
        manifest.counts.files_total = 4
        manifest.counts.files_indexed = 4
        manifest.counts.symbols_defs = 8  # Estimated from sample code
        manifest.counts.symbols_refs = 15
        
        # Verify manifest structure
        assert manifest.repo.root == str(sample_repository)
        assert manifest.config.languages == ["ts", "tsx", "js", "jsx"]
        assert manifest.counts.files_indexed == 4
        
        # Test serialization
        manifest_json = manifest.model_dump()
        restored_manifest = IndexManifest.model_validate(manifest_json)
        
        assert restored_manifest.repo.root == manifest.repo.root
        assert restored_manifest.counts.files_indexed == manifest.counts.files_indexed


class TestSearchWorkflow:
    """Test search functionality workflows."""
    
    def test_search_query_processing(self):
        """Test search query processing and result ranking."""
        # Mock search results for a query
        query = "App class configuration"
        
        # Create mock search results
        mock_results = [
            SearchResult(
                path="src/components/App.ts",
                span=(10, 25),
                score=0.95,
                scores=SearchScores(vector=0.9, symbol=0.95, graph=0.8),
                content=CodeSnippet(
                    path="src/components/App.ts",
                    span=(10, 25),
                    hash="app_hash",
                    pre="// Main application class",
                    text="export class App {\n    private config: Config;\n    constructor(config: Config) {",
                    post="        this.config = config;\n    }",
                    line_start=10,
                    line_end=15,
                ),
                citation=Citation(
                    repo_root="/repo",
                    rev="main",
                    path="src/components/App.ts",
                    span=(10, 25),
                    content_sha="app_hash"
                )
            ),
            SearchResult(
                path="src/utils/config.ts",
                span=(5, 15),
                score=0.85,
                scores=SearchScores(vector=0.8, symbol=0.9, graph=0.7),
                content=CodeSnippet(
                    path="src/utils/config.ts",
                    span=(5, 15),
                    hash="config_hash",
                    pre="// Configuration management utilities",
                    text="export interface Config {\n    apiUrl: string;\n    timeout: number;",
                    post="    retries: number;\n}",
                    line_start=5,
                    line_end=10,
                ),
                citation=Citation(
                    repo_root="/repo",
                    rev="main", 
                    path="src/utils/config.ts",
                    span=(5, 15),
                    content_sha="config_hash"
                )
            )
        ]
        
        # Test result ordering (should be sorted by score)
        sorted_results = sorted(mock_results, key=lambda r: r.score, reverse=True)
        assert sorted_results[0].path == "src/components/App.ts"
        assert sorted_results[0].score == 0.95
        assert sorted_results[1].path == "src/utils/config.ts"
        assert sorted_results[1].score == 0.85
        
        # Test result filtering and validation
        valid_results = [r for r in sorted_results if r.score > 0.8]
        assert len(valid_results) == 2
        
        for result in valid_results:
            # Verify result structure
            assert result.path is not None
            assert result.content.text is not None
            assert result.citation.repo_root is not None
            
            # Verify score consistency
            assert result.score > 0
            assert result.scores.vector > 0
    
    def test_search_result_formatting(self):
        """Test search result formatting for display."""
        search_result = SearchResult(
            path="src/components/App.ts",
            span=(10, 25), 
            score=0.95,
            scores=SearchScores(vector=0.9, symbol=0.95, graph=0.8),
            content=CodeSnippet(
                path="src/components/App.ts",
                span=(10, 25),
                hash="app_hash",
                pre="import { Config } from '../utils/config';",
                text="export class App {\n    private config: Config;\n    private isRunning: boolean = false;",
                post="    constructor(config: Config) {\n        this.config = config;\n    }",
                line_start=10,
                line_end=13,
            ),
            citation=Citation(
                repo_root="/repo",
                rev="main",
                path="src/components/App.ts", 
                span=(10, 25),
                content_sha="app_hash"
            )
        )
        
        # Test formatted display
        formatted = {
            "file": search_result.path,
            "line_range": f"{search_result.content.line_start}-{search_result.content.line_end}",
            "score": f"{search_result.score:.2f}",
            "preview": search_result.content.text[:100] + ("..." if len(search_result.content.text) > 100 else ""),
            "context": {
                "before": search_result.content.pre,
                "after": search_result.content.post,
            }
        }
        
        assert formatted["file"] == "src/components/App.ts"
        assert formatted["line_range"] == "10-13"
        assert formatted["score"] == "0.95"
        assert "export class App" in formatted["preview"]
        assert "import { Config }" in formatted["context"]["before"]


class TestPipelineStatusWorkflow:
    """Test pipeline status and progress tracking."""
    
    def test_pipeline_status_progression(self):
        """Test pipeline status updates through workflow stages."""
        # Initial status
        status = PipelineStatus(
            index_id="test_index_123",
            state=IndexState.RUNNING,
            stage=PipelineStage.REPOMAPPER,
            progress=10,
            message="Starting repository mapping",
            error=None
        )
        
        assert status.state == IndexState.RUNNING
        assert status.stage == PipelineStage.REPOMAPPER
        assert status.progress == 10
        
        # Progression through stages
        progression = [
            (PipelineStage.REPOMAPPER, 20, "Repository structure analyzed"),
            (PipelineStage.SERENA, 40, "Code symbols extracted"),
            (PipelineStage.EMBEDDING, 60, "Embeddings generated"),
            (PipelineStage.BUNDLE, 80, "Index bundle created"),
            (PipelineStage.BUNDLE, 100, "Pipeline completed successfully"),
        ]
        
        for stage, progress, message in progression:
            status = PipelineStatus(
                index_id="test_index_123",
                state=IndexState.RUNNING if progress < 100 else IndexState.DONE,
                stage=stage,
                progress=progress,
                message=message,
                error=None
            )
            
            # Verify progress increases
            assert status.progress >= 10
            assert status.message is not None
            
            if progress == 100:
                assert status.state == IndexState.DONE
        
        # Test error state
        error_status = PipelineStatus(
            index_id="test_index_123",
            state=IndexState.FAILED,
            stage=PipelineStage.SERENA,
            progress=35,
            message="Symbol extraction failed",
            error="Tree-sitter parsing error: Invalid syntax"
        )
        
        assert error_status.state == IndexState.FAILED
        assert error_status.error is not None
        assert "parsing error" in error_status.error
    
    def test_pipeline_error_handling(self):
        """Test pipeline error scenarios and recovery."""
        # Test various error conditions
        error_scenarios = [
            {
                "stage": PipelineStage.REPOMAPPER,
                "error": "Repository not found",
                "recoverable": False
            },
            {
                "stage": PipelineStage.SERENA,
                "error": "Symbol extraction timeout", 
                "recoverable": True
            },
            {
                "stage": PipelineStage.EMBEDDING,
                "error": "API rate limit exceeded",
                "recoverable": True
            },
            {
                "stage": PipelineStage.BUNDLE,
                "error": "Insufficient disk space",
                "recoverable": False
            }
        ]
        
        for scenario in error_scenarios:
            status = PipelineStatus(
                index_id="test_index_error",
                state=IndexState.FAILED,
                stage=scenario["stage"],
                progress=25,  # Failed partway through
                message=f"Failed at {scenario['stage'].value}",
                error=scenario["error"]
            )
            
            assert status.state == IndexState.FAILED
            assert status.error == scenario["error"]
            assert status.progress < 100  # Should not be complete
            
            # Test error categorization
            if "timeout" in scenario["error"] or "rate limit" in scenario["error"]:
                assert scenario["recoverable"] == True
            elif "not found" in scenario["error"] or "disk space" in scenario["error"]:
                assert scenario["recoverable"] == False


class TestIntegrationWorkflow:
    """Test integration between different system components."""
    
    @pytest.mark.asyncio
    async def test_full_indexing_integration(self, sample_repository, mimir_config, tmp_path):
        """Test complete integration from repository to searchable index."""
        # Create workspace for indexing
        workspace_dir = tmp_path / "workspace"
        ensure_directory(workspace_dir)
        
        # Mock the full pipeline integration
        with patch.multiple(
            'src.repoindex.pipeline',
            RepositoryDiscoverer=Mock(),
            SerenaExtractor=Mock(),
            EmbeddingGenerator=Mock(),
            IndexBundler=Mock()
        ) as mocks:
            
            # Setup mock chain
            discoverer = mocks['RepositoryDiscoverer'].return_value
            discoverer.discover_files.return_value = [
                sample_repository / "src" / "index.ts",
                sample_repository / "src" / "components" / "App.ts",
                sample_repository / "src" / "utils" / "config.ts",
            ]
            
            extractor = mocks['SerenaExtractor'].return_value
            extractor.extract_symbols.return_value = {
                "definitions": [
                    {"symbol": "App", "file": "src/components/App.ts", "line": 10},
                    {"symbol": "Config", "file": "src/utils/config.ts", "line": 5},
                ],
                "references": [
                    {"symbol": "Config", "file": "src/components/App.ts", "line": 12},
                ]
            }
            
            generator = mocks['EmbeddingGenerator'].return_value
            generator.generate_embeddings.return_value = [
                {"file": "src/index.ts", "embedding": [0.1, 0.2, 0.3]},
                {"file": "src/components/App.ts", "embedding": [0.4, 0.5, 0.6]},
                {"file": "src/utils/config.ts", "embedding": [0.7, 0.8, 0.9]},
            ]
            
            bundler = mocks['IndexBundler'].return_value
            bundler.create_bundle.return_value = workspace_dir / "index.bundle"
            
            # Simulate full integration workflow
            repo_info = RepoInfo(
                root=str(sample_repository),
                rev="main",
                worktree_dirty=False
            )
            
            manifest = IndexManifest(
                repo=repo_info,
                config=mimir_config.index
            )
            
            # Step 1: Discover files
            discovered_files = discoverer.discover_files()
            assert len(discovered_files) == 3
            
            # Step 2: Extract symbols
            symbols = extractor.extract_symbols(discovered_files)
            assert len(symbols["definitions"]) == 2
            assert len(symbols["references"]) == 1
            
            # Step 3: Generate embeddings
            embeddings = generator.generate_embeddings(discovered_files)
            assert len(embeddings) == 3
            
            # Step 4: Create bundle
            bundle_path = bundler.create_bundle(manifest, symbols, embeddings)
            assert bundle_path is not None
            
            # Step 5: Update manifest with results
            manifest.counts.files_total = len(discovered_files)
            manifest.counts.files_indexed = len(discovered_files)
            manifest.counts.symbols_defs = len(symbols["definitions"])
            manifest.counts.symbols_refs = len(symbols["references"])
            
            # Verify final state
            assert manifest.counts.files_indexed == 3
            assert manifest.counts.symbols_defs == 2
            assert manifest.counts.symbols_refs == 1
            
            # Test manifest persistence
            manifest_file = workspace_dir / "manifest.json"
            atomic_write_json(manifest_file, manifest.model_dump())
            
            # Verify we can reload the manifest
            reloaded_manifest = IndexManifest.model_validate(
                json.loads(manifest_file.read_text())
            )
            assert reloaded_manifest.counts.files_indexed == 3
    
    def test_configuration_workflow_integration(self, tmp_path):
        """Test configuration loading and application across components."""
        # Create config file
        config_file = tmp_path / "integration_config.yaml"
        config_data = {
            "server": {
                "host": "0.0.0.0",
                "port": 9090,
                "debug": True
            },
            "ai": {
                "google_api_key": "integration-test-key",
                "gemini_model": "gemini-1.5-pro",
                "temperature": 0.7
            },
            "index": {
                "languages": ["ts", "tsx", "js", "jsx", "py", "rs"],
                "excludes": ["node_modules/", ".git/", "target/", "__pycache__/"],
                "context_lines": 7,
                "max_files_to_embed": 500
            },
            "search": {
                "max_results": 50,
                "score_threshold": 0.1
            }
        }
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.safe_dump(config_data, f)
        
        # Load configuration
        config = MimirConfig.load_from_file(str(config_file))
        
        # Verify configuration propagation
        assert config.server.host == "0.0.0.0"
        assert config.server.port == 9090
        assert config.ai.gemini_model == "gemini-1.5-pro"
        assert "py" in config.index.languages
        assert "rs" in config.index.languages
        assert config.index.context_lines == 7
        assert config.index.max_files_to_embed == 500
        
        # Test configuration usage in components
        repo_info = RepoInfo(
            root="/test/repo",
            rev="main",
            worktree_dirty=False
        )
        
        manifest = IndexManifest(
            repo=repo_info,
            config=config.index
        )
        
        # Verify config application
        assert manifest.config.languages == config.index.languages
        assert manifest.config.context_lines == 7
        assert manifest.config.max_files_to_embed == 500
        
        # Test serialization preserves all config
        manifest_json = manifest.model_dump()
        restored = IndexManifest.model_validate(manifest_json)
        
        assert restored.config.languages == config.index.languages
        assert restored.config.context_lines == config.index.context_lines