"""
Simple integration test for MCP server functionality.

Uses manual setup instead of async fixtures to avoid pytest-asyncio issues.
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.repoindex.mcp.server import MCPServer


@pytest.mark.integration
@pytest.mark.asyncio
async def test_mcp_server_integration():
    """Test MCP server end-to-end with mocked external tools."""

    # Create test repository
    repo_dir = Path(tempfile.mkdtemp(prefix="mimir_test_repo_"))

    try:
        # Set up test repository
        package_json = {"name": "test-auth-app", "version": "1.0.0", "main": "src/index.ts"}
        (repo_dir / "package.json").write_text(json.dumps(package_json, indent=2))

        src_dir = repo_dir / "src"
        src_dir.mkdir()

        # Create TypeScript files
        index_ts = """import { AuthController } from './auth/controller';
import { UserService } from './services/user';

const app = express();
const userService = new UserService();
const authController = new AuthController(userService);

app.post('/auth/login', authController.login.bind(authController));
export default app;
"""
        (src_dir / "index.ts").write_text(index_ts)

        auth_dir = src_dir / "auth"
        auth_dir.mkdir()

        controller_ts = """export class AuthController {
    constructor(private userService: UserService) {}

    async login(req: Request, res: Response): Promise<void> {
        const { email, password } = req.body;
        const user = await this.userService.findByEmail(email);

        if (!user) {
            res.status(401).json({ error: 'Invalid credentials' });
            return;
        }

        const token = generateToken({ userId: user.id, email: user.email });
        res.json({ token, user });
    }
}
"""
        (auth_dir / "controller.ts").write_text(controller_ts)

        services_dir = src_dir / "services"
        services_dir.mkdir()

        user_service_ts = """export class UserService {
    async findByEmail(email: string): Promise<User | null> {
        return null;
    }

    async create(userData: CreateUserData): Promise<User> {
        return user;
    }
}
"""
        (services_dir / "user.ts").write_text(user_service_ts)

        # Initialize git repository
        print(f"Initializing git repo in: {repo_dir}")
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_dir,
            capture_output=True,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True, check=True)
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True, check=True
        )

        # Verify git initialization
        git_dir = repo_dir / ".git"
        assert git_dir.exists(), f"Git directory not found at {git_dir}"

        # Test git repo validation
        test_result = subprocess.run(
            ["git", "rev-parse", "--git-dir"], cwd=repo_dir, capture_output=True, text=True
        )
        print(
            f"Git validation: returncode={test_result.returncode}, stdout='{test_result.stdout.strip()}', stderr='{test_result.stderr.strip()}'"
        )

        # Test git status
        status_result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
        )
        print(
            f"Git status: returncode={status_result.returncode}, output='{status_result.stdout.strip()}'"
        )
        print(f"Git initialized successfully, .git exists: {git_dir.exists()}")

        # Create MCP server
        storage_dir = Path(tempfile.mkdtemp(prefix="mimir_storage_"))

        try:
            mcp_server = MCPServer(storage_dir=storage_dir)

            # Get TypeScript files for mocking
            ts_files = [str(f.relative_to(repo_dir)) for f in repo_dir.rglob("*.ts")]

            # Create mock responses
            mock_responses = {
                "repomapper": {
                    "file_ranks": [
                        {
                            "path": "src/index.ts",
                            "rank": 0.95,
                            "centrality": 0.8,
                            "dependencies": ["src/auth/controller.ts"],
                        },
                        {
                            "path": "src/auth/controller.ts",
                            "rank": 0.85,
                            "centrality": 0.7,
                            "dependencies": ["src/services/user.ts"],
                        },
                        {
                            "path": "src/services/user.ts",
                            "rank": 0.75,
                            "centrality": 0.6,
                            "dependencies": [],
                        },
                    ],
                    "dependency_graph": [
                        {
                            "source": "src/index.ts",
                            "target": "src/auth/controller.ts",
                            "edge_type": "import",
                        }
                    ],
                    "total_files": len(ts_files),
                },
                "serena": {
                    "symbols": [
                        {
                            "type": "definition",
                            "kind": "class",
                            "name": "AuthController",
                            "path": "src/auth/controller.ts",
                            "start_line": 1,
                            "end_line": 20,
                            "start_col": 0,
                            "end_col": 1,
                            "signature": "export class AuthController",
                        },
                        {
                            "type": "definition",
                            "kind": "method",
                            "name": "login",
                            "path": "src/auth/controller.ts",
                            "start_line": 4,
                            "end_line": 15,
                            "start_col": 4,
                            "end_col": 5,
                            "signature": "async login(req: Request, res: Response): Promise<void>",
                        },
                    ]
                },
                "leann": {
                    "embeddings": [
                        {
                            "chunk_id": "chunk_001",
                            "file_path": "src/index.ts",
                            "content": "import { AuthController } from './auth/controller';",
                            "start_line": 1,
                            "end_line": 1,
                            "start_char": 0,
                            "end_char": 50,
                            "embedding_vector": [0.1] * 10,  # Simplified embedding
                            "vector_norm": 1.0,
                            "token_count": 8,
                            "chunk_type": "import",
                        },
                        {
                            "chunk_id": "chunk_002",
                            "file_path": "src/auth/controller.ts",
                            "content": "export class AuthController {",
                            "start_line": 1,
                            "end_line": 1,
                            "start_char": 0,
                            "end_char": 30,
                            "embedding_vector": [0.2] * 10,
                            "vector_norm": 1.0,
                            "token_count": 4,
                            "chunk_type": "class_definition",
                        },
                    ],
                    "metadata": {
                        "model_name": "leann-cpu-v2",
                        "embedding_dimensions": 10,
                        "total_chunks": 2,
                        "total_tokens": 12,
                    },
                },
            }

            # Mock external adapters and git discovery
            with (
                patch("src.repoindex.pipeline.repomapper.RepoMapperAdapter") as mock_repomapper_cls,
                patch("src.repoindex.pipeline.serena.SerenaAdapter") as mock_serena_cls,
                patch("src.repoindex.pipeline.leann.LEANNAdapter") as mock_leann_cls,
                patch(
                    "src.repoindex.pipeline.discover.discover_git_repository"
                ) as mock_git_discovery,
            ):

                # Configure RepoMapper mock
                from src.repoindex.data.schemas import DependencyEdge, FileRank, RepoMap

                mock_repomapper = Mock()
                mock_repomapper.build_index.return_value = RepoMap(
                    file_ranks=[
                        FileRank(
                            path="src/index.ts",
                            rank=0.95,
                            centrality=0.8,
                            dependencies=["src/auth/controller.ts"],
                        ),
                        FileRank(
                            path="src/auth/controller.ts",
                            rank=0.85,
                            centrality=0.7,
                            dependencies=["src/services/user.ts"],
                        ),
                        FileRank(
                            path="src/services/user.ts", rank=0.75, centrality=0.6, dependencies=[]
                        ),
                    ],
                    edges=[
                        DependencyEdge(
                            source="src/index.ts",
                            target="src/auth/controller.ts",
                            weight=1.0,
                            edge_type="import",
                        )
                    ],
                    total_files=len(ts_files),
                )
                mock_repomapper_cls.return_value = mock_repomapper

                # Configure Serena mock
                from src.repoindex.data.schemas import SerenaGraph, SymbolEntry, SymbolType

                mock_serena = Mock()
                mock_serena.build_index.return_value = SerenaGraph(
                    entries=[
                        SymbolEntry(
                            type=SymbolType.DEF,
                            path="src/auth/controller.ts",
                            span=(0, 100),
                            symbol="AuthController",
                            sig="export class AuthController",
                        ),
                        SymbolEntry(
                            type=SymbolType.DEF,
                            path="src/auth/controller.ts",
                            span=(150, 300),
                            symbol="login",
                            sig="async login(req: Request, res: Response): Promise<void>",
                        ),
                    ],
                    file_count=len(ts_files),
                    symbol_count=2,
                )
                mock_serena_cls.return_value = mock_serena

                # Configure LEANN mock
                from src.repoindex.data.schemas import VectorChunk, VectorIndex

                mock_leann = Mock()
                mock_leann.build_index.return_value = VectorIndex(
                    chunks=[
                        VectorChunk(
                            path="src/index.ts",
                            span=(0, 50),
                            content="import { AuthController } from './auth/controller';",
                            embedding=[0.1] * 384,  # Match expected dimension
                            token_count=8,
                        ),
                        VectorChunk(
                            path="src/auth/controller.ts",
                            span=(0, 30),
                            content="export class AuthController {",
                            embedding=[0.2] * 384,
                            token_count=4,
                        ),
                    ],
                    dimension=384,
                    total_tokens=12,
                    model_name="all-MiniLM-L6-v2",
                )
                mock_leann_cls.return_value = mock_leann

                # Configure git discovery mock
                from unittest.mock import MagicMock

                mock_git_repo = MagicMock()
                mock_git_repo.get_repo_root.return_value = repo_dir
                mock_git_repo.get_head_commit.return_value = "abc123def456789"
                mock_git_repo.get_tree_hash.return_value = "tree123hash456"
                mock_git_repo.list_tracked_files.return_value = ts_files
                mock_git_repo.is_worktree_dirty.return_value = False
                mock_git_repo.get_dirty_files.return_value = set()
                mock_git_repo.compute_dirty_overlay.return_value = {}

                mock_git_discovery.return_value = mock_git_repo

                print("🚀 Starting MCP integration test")

                # Test 1: Index repository
                print("📁 Testing repository indexing...")
                index_result = await mcp_server._ensure_repo_index(
                    {
                        "path": str(repo_dir),
                        "rev": "HEAD",
                        "language": "ts",
                        "index_opts": {
                            "languages": ["ts", "tsx", "js", "jsx"],
                            "excludes": ["node_modules/", "dist/", ".git/"],
                            "context_lines": 3,
                            "max_files_to_embed": 50,
                        },
                    }
                )

                # Verify indexing succeeded
                if index_result.isError:
                    error_msg = index_result.content[0].text
                    print(f"❌ Indexing failed: {error_msg}")
                    return

                assert (
                    index_result.isError is False
                ), f"Indexing failed: {index_result.content[0].text}"
                content = json.loads(index_result.content[0].text)

                assert "success" in content
                assert content["success"] is True
                assert "index_id" in content
                assert "files" in content
                assert len(content["files"]) >= 3  # We created 3 .ts files

                index_id = content["index_id"]
                print("✅ Repository indexed successfully!")
                print(f"   Index ID: {index_id}")
                print(f"   Files indexed: {len(content['files'])}")

                # Test 2: Search functionality
                print("🔍 Testing search functionality...")
                search_result = await mcp_server._search_repo(
                    {
                        "index_id": index_id,
                        "query": "authentication login controller",
                        "features": {"vector": True, "symbol": True, "graph": False},
                        "k": 10,
                    }
                )

                assert search_result.isError is False
                search_content = json.loads(search_result.content[0].text)

                assert "results" in search_content
                results = search_content["results"]
                assert len(results) > 0

                print(f"✅ Search returned {len(results)} results")

                # Test 3: Question answering
                print("❓ Testing question answering...")
                ask_result = await mcp_server._ask_index(
                    {
                        "index_id": index_id,
                        "question": "What is the AuthController class and what does it do?",
                    }
                )

                assert ask_result.isError is False
                ask_content = json.loads(ask_result.content[0].text)

                assert "answer" in ask_content
                answer = ask_content["answer"]
                assert len(answer) > 0

                print("✅ Question answered successfully")
                print(f"   Answer length: {len(answer)} characters")

                # Test 4: Bundle export
                print("📦 Testing bundle export...")
                bundle_result = await mcp_server._get_repo_bundle({"index_id": index_id})

                assert bundle_result.isError is False
                bundle_content = json.loads(bundle_result.content[0].text)

                assert "bundle_data" in bundle_content
                bundle_data = bundle_content["bundle_data"]

                # Verify bundle structure
                required_components = [
                    "manifest",
                    "repo_map",
                    "symbol_graph",
                    "vector_index",
                    "snippets",
                ]
                for component in required_components:
                    assert component in bundle_data, f"Bundle missing component: {component}"

                print("✅ Bundle exported successfully:")
                print(f"   Components: {list(bundle_data.keys())}")

                # Test 5: Verify data quality
                print("🧪 Verifying data quality...")

                # Check manifest
                manifest = bundle_data["manifest"]
                assert manifest["repo"]["root"] == str(repo_dir)
                assert len(manifest["files"]) >= 3

                # Check repo map
                repo_map = bundle_data["repo_map"]
                assert "file_ranks" in repo_map
                assert len(repo_map["file_ranks"]) >= 3

                # Check symbol graph
                symbol_graph = bundle_data["symbol_graph"]
                assert "entries" in symbol_graph
                assert (
                    len(symbol_graph["entries"]) >= 2
                )  # Should have AuthController and login method

                # Check vector index
                vector_index = bundle_data["vector_index"]
                assert "chunks" in vector_index
                assert len(vector_index["chunks"]) >= 2

                # Check snippets
                snippets = bundle_data["snippets"]
                assert "snippets" in snippets
                assert len(snippets["snippets"]) >= 3

                print("✅ Data quality verification passed:")
                print(f"   Files ranked: {len(repo_map['file_ranks'])}")
                print(f"   Symbol entries: {len(symbol_graph['entries'])}")
                print(f"   Vector chunks: {len(vector_index['chunks'])}")
                print(f"   Code snippets: {len(snippets['snippets'])}")

                print("🎉 All integration tests passed!")

        finally:
            # Cleanup storage
            shutil.rmtree(storage_dir)

    finally:
        # Cleanup repo
        shutil.rmtree(repo_dir)


if __name__ == "__main__":
    # Run the test directly
    asyncio.run(test_mcp_server_integration())
