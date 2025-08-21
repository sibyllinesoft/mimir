"""
Integration tests for pipeline components working together.

Tests the complete pipeline flow from git discovery through
bundle creation with real adapters and data.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock

from src.repoindex.pipeline.run import PipelineRunner
from src.repoindex.data.schemas import (
    IndexManifest,
    RepoInfo,
    IndexConfig,
    PipelineStatus
)


@pytest.mark.integration
class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @pytest.fixture
    async def pipeline_runner(self, storage_dir):
        """Create pipeline runner for testing."""
        return PipelineRunner(storage_dir=storage_dir)
    
    @pytest.fixture
    def integration_repo(self, temp_dir):
        """Create a more comprehensive test repository."""
        repo_dir = temp_dir / "integration_repo"
        repo_dir.mkdir()
        
        # Create package.json
        (repo_dir / "package.json").write_text(json.dumps({
            "name": "integration-test",
            "version": "1.0.0",
            "dependencies": {
                "lodash": "^4.17.21"
            }
        }, indent=2))
        
        # Create src directory with multiple TypeScript files
        src_dir = repo_dir / "src"
        src_dir.mkdir()
        
        # Main entry point
        (src_dir / "index.ts").write_text('''
import { Calculator } from './calculator';
import { formatNumber } from './utils';

export async function main() {
    const calc = new Calculator();
    const result = calc.add(10, 20);
    console.log(formatNumber(result));
    return result;
}

export { Calculator } from './calculator';
export { formatNumber, multiply } from './utils';
''')
        
        # Calculator class
        (src_dir / "calculator.ts").write_text('''
import { multiply } from './utils';

export class Calculator {
    private history: number[] = [];
    
    add(a: number, b: number): number {
        const result = a + b;
        this.history.push(result);
        return result;
    }
    
    subtract(a: number, b: number): number {
        const result = a - b;
        this.history.push(result);
        return result;
    }
    
    multiplyBy(value: number, factor: number): number {
        return multiply(value, factor);
    }
    
    getHistory(): number[] {
        return [...this.history];
    }
    
    clear(): void {
        this.history = [];
    }
}
''')
        
        # Utility functions
        (src_dir / "utils.ts").write_text('''
export function formatNumber(num: number): string {
    return num.toLocaleString();
}

export function multiply(a: number, b: number): number {
    return a * b;
}

export function divide(a: number, b: number): number {
    if (b === 0) {
        throw new Error('Division by zero');
    }
    return a / b;
}

export const PI = 3.14159;
export const E = 2.71828;
''')
        
        # Tests directory
        tests_dir = repo_dir / "tests"
        tests_dir.mkdir()
        
        (tests_dir / "calculator.test.ts").write_text('''
import { Calculator } from '../src/calculator';
import { formatNumber } from '../src/utils';

describe('Calculator', () => {
    let calc: Calculator;
    
    beforeEach(() => {
        calc = new Calculator();
    });
    
    test('should add numbers correctly', () => {
        expect(calc.add(2, 3)).toBe(5);
        expect(calc.add(-1, 1)).toBe(0);
    });
    
    test('should track history', () => {
        calc.add(1, 2);
        calc.subtract(5, 3);
        expect(calc.getHistory()).toEqual([3, 2]);
    });
});
''')
        
        # Documentation
        (repo_dir / "README.md").write_text('''
# Integration Test Repository

This is a test repository for integration testing the Mimir indexing pipeline.

## Features

- TypeScript calculator library
- Comprehensive test suite
- Documentation and examples

## Usage

```typescript
import { Calculator, formatNumber } from './src';

const calc = new Calculator();
const result = calc.add(10, 20);
console.log(formatNumber(result)); // "30"
```
''')
        
        return repo_dir
    
    async def test_complete_pipeline_flow(self, pipeline_runner, integration_repo, storage_dir):
        """Test complete pipeline execution."""
        # Create repository info
        repo_info = RepoInfo(
            root=str(integration_repo),
            rev="main",
            worktree_dirty=False
        )
        
        # Create configuration
        config = IndexConfig(
            languages=["ts", "tsx", "js", "jsx", "md"],
            excludes=["node_modules/", "dist/", ".git/"],
            context_lines=3,
            max_files_to_embed=50
        )
        
        # Create manifest
        manifest = IndexManifest(
            repo=repo_info,
            config=config
        )
        
        # Mock external adapters
        with patch('src.repoindex.pipeline.repomapper.RepoMapperAdapter') as mock_repomapper:
            with patch('src.repoindex.pipeline.serena.SerenaAdapter') as mock_serena:
                with patch('src.repoindex.pipeline.leann.LEANNAdapter') as mock_leann:
                    # Configure mocks
                    mock_repomapper_instance = AsyncMock()
                    mock_repomapper.return_value = mock_repomapper_instance
                    
                    mock_serena_instance = AsyncMock()
                    mock_serena.return_value = mock_serena_instance
                    
                    mock_leann_instance = AsyncMock()
                    mock_leann.return_value = mock_leann_instance
                    
                    # Mock adapter responses
                    from src.repoindex.data.schemas import RepoMap, SerenaGraph, VectorIndex
                    
                    mock_repomapper_instance.analyze.return_value = RepoMap(
                        file_ranks=[],
                        edges=[]
                    )
                    
                    mock_serena_instance.analyze.return_value = SerenaGraph(
                        entries=[]
                    )
                    
                    mock_leann_instance.embed_files.return_value = VectorIndex(
                        chunks=[]
                    )
                    
                    # Execute pipeline
                    await pipeline_runner.run_pipeline(manifest)
                    
                    # Verify pipeline completed
                    index_dir = storage_dir / manifest.index_id
                    assert index_dir.exists()
                    
                    # Check artifacts were created
                    assert (index_dir / "manifest.json").exists()
                    assert (index_dir / "discovered_files.json").exists()
                    
                    # Verify adapters were called
                    mock_repomapper_instance.analyze.assert_called_once()
                    mock_serena_instance.analyze.assert_called_once()
                    mock_leann_instance.embed_files.assert_called_once()
    
    async def test_pipeline_status_tracking(self, pipeline_runner, integration_repo):
        """Test pipeline status is tracked correctly."""
        repo_info = RepoInfo(
            root=str(integration_repo),
            rev="main",
            worktree_dirty=False
        )
        
        config = IndexConfig()
        manifest = IndexManifest(repo=repo_info, config=config)
        
        # Track status updates
        status_updates = []
        
        async def mock_status_callback(status: PipelineStatus):
            status_updates.append(status)
        
        # Mock adapters with delays
        with patch('src.repoindex.pipeline.repomapper.RepoMapperAdapter') as mock_repomapper:
            with patch('src.repoindex.pipeline.serena.SerenaAdapter') as mock_serena:
                with patch('src.repoindex.pipeline.leann.LEANNAdapter') as mock_leann:
                    
                    async def delayed_analyze(*args, **kwargs):
                        await asyncio.sleep(0.1)  # Simulate work
                        from src.repoindex.data.schemas import RepoMap, SerenaGraph, VectorIndex
                        if 'RepoMap' in str(mock_repomapper):
                            return RepoMap(file_ranks=[], edges=[])
                        elif 'SerenaGraph' in str(mock_serena):
                            return SerenaGraph(entries=[])
                        else:
                            return VectorIndex(chunks=[])
                    
                    mock_repomapper.return_value.analyze = delayed_analyze
                    mock_serena.return_value.analyze = delayed_analyze
                    mock_leann.return_value.embed_files = delayed_analyze
                    
                    # Run pipeline with status callback
                    await pipeline_runner.run_pipeline(manifest, status_callback=mock_status_callback)
                    
                    # Verify status updates were received
                    assert len(status_updates) > 0
                    
                    # Check we got status for each stage
                    stages = [status.stage for status in status_updates]
                    assert "acquire" in stages
                    assert "serena" in stages or "repomapper" in stages
    
    async def test_pipeline_error_handling(self, pipeline_runner, integration_repo):
        """Test pipeline handles errors gracefully."""
        repo_info = RepoInfo(
            root=str(integration_repo),
            rev="main",
            worktree_dirty=False
        )
        
        config = IndexConfig()
        manifest = IndexManifest(repo=repo_info, config=config)
        
        # Mock adapter that raises error
        with patch('src.repoindex.pipeline.serena.SerenaAdapter') as mock_serena:
            mock_serena_instance = AsyncMock()
            mock_serena.return_value = mock_serena_instance
            mock_serena_instance.analyze.side_effect = Exception("Serena analysis failed")
            
            # Pipeline should handle error gracefully
            with pytest.raises(Exception) as exc_info:
                await pipeline_runner.run_pipeline(manifest)
            
            assert "Serena analysis failed" in str(exc_info.value)
    
    async def test_pipeline_cancellation(self, pipeline_runner, integration_repo):
        """Test pipeline can be cancelled."""
        repo_info = RepoInfo(
            root=str(integration_repo),
            rev="main",
            worktree_dirty=False
        )
        
        config = IndexConfig()
        manifest = IndexManifest(repo=repo_info, config=config)
        
        # Mock adapter with long-running operation
        with patch('src.repoindex.pipeline.serena.SerenaAdapter') as mock_serena:
            async def long_running_analyze(*args, **kwargs):
                await asyncio.sleep(10)  # Long operation
                return SerenaGraph(entries=[])
            
            mock_serena_instance = AsyncMock()
            mock_serena.return_value = mock_serena_instance
            mock_serena_instance.analyze = long_running_analyze
            
            # Start pipeline
            pipeline_task = asyncio.create_task(
                pipeline_runner.run_pipeline(manifest)
            )
            
            # Cancel after short delay
            await asyncio.sleep(0.1)
            pipeline_task.cancel()
            
            # Should raise cancellation
            with pytest.raises(asyncio.CancelledError):
                await pipeline_task


@pytest.mark.integration
class TestSearchIntegration:
    """Test search functionality integration."""
    
    async def test_hybrid_search_integration(self, storage_dir, integration_repo):
        """Test hybrid search with real data structures."""
        from src.repoindex.pipeline.hybrid_search import HybridSearchEngine
        from src.repoindex.data.schemas import (
            VectorIndex, VectorChunk, SerenaGraph, SymbolEntry, SymbolType,
            RepoMap, FeatureConfig
        )
        
        # Create mock search data
        vector_index = VectorIndex(chunks=[
            VectorChunk(
                path="src/calculator.ts",
                span=(5, 15),
                content="add(a: number, b: number): number",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5]
            ),
            VectorChunk(
                path="src/utils.ts",
                span=(1, 5),
                content="formatNumber(num: number): string",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6]
            )
        ])
        
        serena_graph = SerenaGraph(entries=[
            SymbolEntry(
                type=SymbolType.DEF,
                path="src/calculator.ts",
                span=(5, 15),
                symbol="add",
                sig="add(a: number, b: number): number"
            ),
            SymbolEntry(
                type=SymbolType.REF,
                path="src/index.ts",
                span=(8, 11),
                symbol="add"
            )
        ])
        
        repomap = RepoMap(file_ranks=[], edges=[])
        
        # Create search engine
        search_engine = HybridSearchEngine()
        
        # Test search
        features = FeatureConfig(vector=True, symbol=True, graph=False)
        
        with patch.object(search_engine, '_vector_search') as mock_vector:
            with patch.object(search_engine, '_symbol_search') as mock_symbol:
                mock_vector.return_value = [("src/calculator.ts", (5, 15), 0.8)]
                mock_symbol.return_value = [("src/calculator.ts", (5, 15), 0.9)]
                
                response = await search_engine.search(
                    query="add function",
                    vector_index=vector_index,
                    serena_graph=serena_graph,
                    repomap=repomap,
                    repo_root=str(integration_repo),
                    rev="main",
                    features=features
                )
                
                assert response.query == "add function"
                assert len(response.results) > 0
                assert response.features_used.vector
                assert response.features_used.symbol
    
    async def test_ask_integration(self, storage_dir, integration_repo):
        """Test ask functionality with symbol navigation."""
        from src.repoindex.pipeline.ask_index import SymbolGraphNavigator
        from src.repoindex.data.schemas import SerenaGraph, SymbolEntry, SymbolType
        
        # Create mock symbol graph with relationships
        serena_graph = SerenaGraph(entries=[
            # Calculator class definition
            SymbolEntry(
                type=SymbolType.DEF,
                path="src/calculator.ts",
                span=(3, 25),
                symbol="Calculator",
                sig="class Calculator"
            ),
            # add method definition
            SymbolEntry(
                type=SymbolType.DEF,
                path="src/calculator.ts",
                span=(6, 12),
                symbol="add",
                sig="add(a: number, b: number): number"
            ),
            # add method call
            SymbolEntry(
                type=SymbolType.CALL,
                path="src/index.ts",
                span=(4, 7),
                caller="main",
                callee="add"
            ),
            # Calculator usage
            SymbolEntry(
                type=SymbolType.REF,
                path="src/index.ts",
                span=(3, 13),
                symbol="Calculator"
            )
        ])
        
        # Create navigator
        navigator = SymbolGraphNavigator()
        
        # Test question answering
        response = await navigator.ask(
            question="What is the Calculator class?",
            serena_graph=serena_graph,
            repo_root=str(integration_repo),
            rev="main"
        )
        
        assert response.question == "What is the Calculator class?"
        assert "Calculator" in response.answer
        assert len(response.citations) > 0
        assert response.execution_time_ms > 0


@pytest.mark.integration 
@pytest.mark.slow
class TestEndToEndWorkflow:
    """Test complete end-to-end workflow."""
    
    async def test_mcp_server_full_workflow(self, temp_dir, integration_repo):
        """Test complete MCP server workflow."""
        from src.repoindex.mcp.server import MimirMCPServer
        
        storage_dir = temp_dir / "mcp_storage"
        storage_dir.mkdir()
        
        server = MimirMCPServer(storage_dir=storage_dir)
        
        # Mock all external dependencies
        with patch('src.repoindex.pipeline.repomapper.RepoMapperAdapter'):
            with patch('src.repoindex.pipeline.serena.SerenaAdapter'):
                with patch('src.repoindex.pipeline.leann.LEANNAdapter'):
                    
                    # 1. Ensure repo index
                    ensure_args = {
                        "repo_root": str(integration_repo),
                        "config": {
                            "languages": ["ts", "js"],
                            "excludes": ["node_modules/"],
                            "context_lines": 3,
                            "max_files_to_embed": 50
                        }
                    }
                    
                    ensure_result = await server._ensure_repo_index(ensure_args)
                    assert not ensure_result.isError
                    
                    # Extract index ID from result
                    index_id = ensure_result.content[0].text.split(": ")[1]
                    
                    # 2. Test search
                    search_args = {
                        "index_id": index_id,
                        "query": "add function",
                        "features": {
                            "vector": True,
                            "symbol": True,
                            "graph": False
                        },
                        "k": 10
                    }
                    
                    with patch.object(server, '_execute_search') as mock_search:
                        from src.repoindex.data.schemas import SearchResponse, FeatureConfig
                        mock_search.return_value = SearchResponse(
                            query="add function",
                            results=[],
                            total_count=0,
                            features_used=FeatureConfig(vector=True, symbol=True, graph=False),
                            execution_time_ms=50.0,
                            index_id=index_id
                        )
                        
                        search_result = await server._search_repo(search_args)
                        assert not search_result.isError
                    
                    # 3. Test ask
                    ask_args = {
                        "index_id": index_id,
                        "question": "What does the Calculator class do?"
                    }
                    
                    with patch.object(server, '_execute_ask') as mock_ask:
                        from src.repoindex.data.schemas import AskResponse
                        mock_ask.return_value = AskResponse(
                            question="What does the Calculator class do?",
                            answer="The Calculator class provides mathematical operations.",
                            citations=[],
                            execution_time_ms=100.0,
                            index_id=index_id
                        )
                        
                        ask_result = await server._ask_index(ask_args)
                        assert not ask_result.isError
                    
                    # 4. Test resource access
                    status_result = await server._read_resource("repo://status")
                    assert status_result.contents[0].mimeType == "application/json"
                    
                    manifest_result = await server._read_resource(f"repo://manifest/{index_id}")
                    assert manifest_result.contents[0].mimeType == "application/json"