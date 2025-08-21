"""
Integration test fixtures and configuration.

Provides fixtures specifically for integration testing including
repositories, pipelines, and mock external services.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from src.repoindex.data.schemas import (
    DependencyEdge,
    FileRank,
    IndexConfig,
    RepoMap,
    SerenaGraph,
    SymbolEntry,
    SymbolType,
    VectorChunk,
    VectorIndex,
)
from src.repoindex.pipeline.run import PipelineRunner


@pytest_asyncio.fixture
async def integration_repo(temp_dir: Path) -> Path:
    """Create a comprehensive test repository for integration testing."""
    repo_dir = temp_dir / "integration_repo"
    repo_dir.mkdir()

    # Create package.json
    (repo_dir / "package.json").write_text(
        json.dumps(
            {
                "name": "integration-test",
                "version": "1.0.0",
                "dependencies": {"lodash": "^4.17.21"},
            },
            indent=2,
        )
    )

    # Create src directory with multiple TypeScript files
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    # Main entry point
    (src_dir / "index.ts").write_text(
        """
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
"""
    )

    # Calculator class
    (src_dir / "calculator.ts").write_text(
        """
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
"""
    )

    # Utility functions
    (src_dir / "utils.ts").write_text(
        """
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
"""
    )

    # Tests directory
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir()

    (tests_dir / "calculator.test.ts").write_text(
        """
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
"""
    )

    # Documentation
    (repo_dir / "README.md").write_text(
        """
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
"""
    )

    return repo_dir


@pytest_asyncio.fixture
async def test_repo(temp_dir: Path) -> Path:
    """Create a simpler test repository for basic testing."""
    repo_dir = temp_dir / "test_repo"
    repo_dir.mkdir()

    # Create simple files
    (repo_dir / "README.md").write_text("# Test Repository")

    src_dir = repo_dir / "src"
    src_dir.mkdir()

    (src_dir / "index.ts").write_text(
        """
export function hello(name: string): string {
    return `Hello, ${name}!`;
}
"""
    )

    return repo_dir


@pytest_asyncio.fixture
async def indexing_pipeline(storage_dir: Path) -> PipelineRunner:
    """Create a pipeline runner for testing."""
    return PipelineRunner(storage_dir=storage_dir)


@pytest_asyncio.fixture
async def mock_vector_index() -> VectorIndex:
    """Create a mock vector index for testing."""
    return VectorIndex(
        chunks=[
            VectorChunk(
                path="src/calculator.ts",
                span=(5, 15),
                content="add(a: number, b: number): number",
                embedding=[0.1, 0.2, 0.3, 0.4, 0.5],
                token_count=10,
            ),
            VectorChunk(
                path="src/utils.ts",
                span=(1, 5),
                content="formatNumber(num: number): string",
                embedding=[0.2, 0.3, 0.4, 0.5, 0.6],
                token_count=8,
            ),
        ],
        dimension=5,
        total_tokens=18,
        model_name="test-embedding-model",
    )


@pytest_asyncio.fixture
async def mock_serena_graph() -> SerenaGraph:
    """Create a mock Serena graph for testing."""
    return SerenaGraph(
        entries=[
            SymbolEntry(
                type=SymbolType.DEF,
                path="src/calculator.ts",
                span=(5, 15),
                symbol="add",
                sig="add(a: number, b: number): number",
            ),
            SymbolEntry(type=SymbolType.REF, path="src/index.ts", span=(8, 11), symbol="add"),
            SymbolEntry(
                type=SymbolType.DEF,
                path="src/utils.ts",
                span=(1, 5),
                symbol="formatNumber",
                sig="formatNumber(num: number): string",
            ),
        ],
        file_count=3,
        symbol_count=2,
    )


@pytest_asyncio.fixture
async def mock_repomap() -> RepoMap:
    """Create a mock repository map for testing."""
    return RepoMap(
        file_ranks=[
            FileRank(
                path="src/index.ts",
                rank=0.9,
                centrality=0.8,
                dependencies=["src/calculator.ts", "src/utils.ts"],
            ),
            FileRank(
                path="src/calculator.ts", rank=0.7, centrality=0.6, dependencies=["src/utils.ts"]
            ),
            FileRank(path="src/utils.ts", rank=0.5, centrality=0.4, dependencies=[]),
        ],
        edges=[
            DependencyEdge(
                source="src/index.ts", target="src/calculator.ts", weight=0.8, edge_type="import"
            ),
            DependencyEdge(
                source="src/index.ts", target="src/utils.ts", weight=0.6, edge_type="import"
            ),
            DependencyEdge(
                source="src/calculator.ts", target="src/utils.ts", weight=0.7, edge_type="import"
            ),
        ],
        total_files=3,
    )


@pytest_asyncio.fixture
async def mock_external_adapters():
    """Create mock external adapters for testing."""
    # Mock RepoMapper adapter
    mock_repomapper = AsyncMock()
    mock_repomapper.build_index.return_value = RepoMap(
        file_ranks=[FileRank(path="src/index.ts", rank=0.9, centrality=0.8, dependencies=[])],
        edges=[],
        total_files=1,
    )

    # Mock Serena adapter
    mock_serena = AsyncMock()
    mock_serena.build_index.return_value = SerenaGraph(
        entries=[
            SymbolEntry(type=SymbolType.DEF, path="src/index.ts", span=(0, 10), symbol="hello")
        ],
        file_count=1,
        symbol_count=1,
    )

    # Mock LEANN adapter
    mock_leann = AsyncMock()
    mock_leann.build_index.return_value = VectorIndex(
        chunks=[
            VectorChunk(
                path="src/index.ts",
                span=(0, 10),
                content="hello function",
                embedding=[0.1, 0.2, 0.3],
                token_count=5,
            )
        ],
        dimension=3,
        total_tokens=5,
        model_name="test-model",
    )

    return {"repomapper": mock_repomapper, "serena": mock_serena, "leann": mock_leann}


@pytest.fixture
def integration_config() -> IndexConfig:
    """Provide configuration optimized for integration testing."""
    return IndexConfig(
        languages=["ts", "tsx", "js", "jsx", "md"],
        excludes=["node_modules/", "dist/", ".git/"],
        context_lines=3,
        max_files_to_embed=50,
    )
