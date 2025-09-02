"""
Pytest configuration and shared fixtures for Mimir tests.

Provides common test utilities, fixtures, and configuration for
unit, integration, and end-to-end tests.
"""

import asyncio
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest

from src.repoindex.data.schemas import IndexConfig, RepoInfo
from src.repoindex.util.fs import TemporaryDirectory


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory for test artifacts."""
    with TemporaryDirectory(prefix="mimir_test_") as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_repo_dir(temp_dir: Path) -> Path:
    """Create a sample repository structure for testing."""
    repo_dir = temp_dir / "sample_repo"
    repo_dir.mkdir()

    # Create sample files
    (repo_dir / "package.json").write_text('{"name": "test-repo", "version": "1.0.0"}')
    (repo_dir / "README.md").write_text("# Test Repository\n\nThis is a test repository.")

    # Create source directory with sample TypeScript files
    src_dir = repo_dir / "src"
    src_dir.mkdir()

    (src_dir / "index.ts").write_text(
        """
export function main() {
    console.log("Hello, world!");
    return greet("Mimir");
}

export function greet(name: string): string {
    return `Hello, ${name}!`;
}
"""
    )

    (src_dir / "utils.ts").write_text(
        """
export function add(a: number, b: number): number {
    return a + b;
}

export function multiply(a: number, b: number): number {
    return a * b;
}

export class Calculator {
    add(a: number, b: number): number {
        return add(a, b);
    }

    multiply(a: number, b: number): number {
        return multiply(a, b);
    }
}
"""
    )

    # Create test directory
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir()

    (tests_dir / "index.test.ts").write_text(
        """
import { greet, main } from "../src/index";

describe("main function", () => {
    it("should return greeting", () => {
        const result = main();
        expect(result).toBe("Hello, Mimir!");
    });
});

describe("greet function", () => {
    it("should greet user", () => {
        const result = greet("World");
        expect(result).toBe("Hello, World!");
    });
});
"""
    )

    return repo_dir


@pytest.fixture
def default_config() -> IndexConfig:
    """Provide default index configuration for testing."""
    return IndexConfig(
        languages=["ts", "tsx", "js", "jsx", "md"],
        excludes=["node_modules/", "dist/", "build/"],
        context_lines=3,
        max_files_to_embed=100,
    )


@pytest.fixture
def sample_repo_info(sample_repo_dir: Path) -> RepoInfo:
    """Provide repository info for the sample repository."""
    return RepoInfo(root=str(sample_repo_dir), rev="main", worktree_dirty=False)


@pytest.fixture
async def storage_dir(temp_dir: Path) -> AsyncGenerator[Path, None]:
    """Provide a storage directory for pipeline artifacts."""
    storage = temp_dir / "storage"
    storage.mkdir()
    yield storage


class MockGitRepo:
    """Mock git repository for testing."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def get_repo_root(self) -> Path:
        return self.repo_path

    def get_head_commit(self) -> str:
        return "abc123def456"

    def get_tree_hash(self, ref: str = "HEAD") -> str:
        return "tree123hash456"

    def list_tracked_files(self, extensions=None, excludes=None) -> list[str]:
        files = []
        for file_path in self.repo_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.repo_path)

                # Apply extension filter
                if extensions:
                    if not any(str(rel_path).endswith(f".{ext}") for ext in extensions):
                        continue

                # Apply exclude filter
                if excludes:
                    if any(exclude in str(rel_path) for exclude in excludes):
                        continue

                files.append(str(rel_path))

        return files

    def is_worktree_dirty(self) -> bool:
        return False

    def get_dirty_files(self) -> set[str]:
        return set()

    def compute_dirty_overlay(self) -> dict[str, str]:
        return {}

    def hash_file_content(self, file_path: str) -> str:
        return f"hash_{file_path.replace('/', '_')}"


@pytest.fixture
def mock_git_repo(sample_repo_dir: Path) -> MockGitRepo:
    """Provide a mock git repository for testing."""
    return MockGitRepo(sample_repo_dir)


@pytest.fixture
def sample_files_list() -> list[str]:
    """Provide a sample list of files for testing."""
    return ["src/index.ts", "src/utils.ts", "tests/index.test.ts", "package.json", "README.md"]


# Pytest markers for different test types
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add unit marker to tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker to tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark async tests that might be slow
        if item.get_closest_marker("asyncio") and "pipeline" in str(item.fspath):
            item.add_marker(pytest.mark.slow)
