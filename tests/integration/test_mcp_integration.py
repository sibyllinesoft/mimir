"""
Integration tests for MCP server functionality.

Tests the MCP server's tools and resources by simulating real MCP client calls
and verifying the responses are correct.
"""

import asyncio
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest

from src.repoindex.mcp.server import MCPServer


class TestMCPIntegration:
    """Integration tests for MCP server with mocked external tools."""

    @pytest.fixture
    async def test_repo(self) -> Path:
        """Create a test repository with TypeScript files."""
        repo_dir = Path(tempfile.mkdtemp(prefix="mimir_test_repo_"))

        # Create package.json
        package_json = {"name": "test-auth-app", "version": "1.0.0", "main": "src/index.ts"}
        (repo_dir / "package.json").write_text(json.dumps(package_json, indent=2))

        # Create source directory
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
        // Database lookup logic
        return null;
    }

    async create(userData: CreateUserData): Promise<User> {
        // User creation logic
        return user;
    }
}
"""
        (services_dir / "user.ts").write_text(user_service_ts)

        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

        yield repo_dir

        # Cleanup
        shutil.rmtree(repo_dir)

    @pytest.fixture
    async def mcp_server(self) -> MCPServer:
        """Create MCP server for testing."""
        storage_dir = Path(tempfile.mkdtemp(prefix="mimir_storage_"))
        server = MCPServer(storage_dir=storage_dir)
        yield server
        shutil.rmtree(storage_dir)

    def create_mock_responses(self, ts_files: list[str]) -> dict[str, Any]:
        """Create mock responses for external tools."""
        return {
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

    @pytest.mark.integration
    async def test_ensure_repo_index_tool(self, test_repo: Path, mcp_server: MCPServer):
        """Test the ensure_repo_index MCP tool."""
        ts_files = [str(f.relative_to(test_repo)) for f in test_repo.rglob("*.ts")]
        mock_responses = self.create_mock_responses(ts_files)

        with patch("subprocess.run") as mock_run:

            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args", [])

                if not cmd:
                    return Mock(returncode=1, stdout="", stderr="Unknown command")

                if "repomapper" in cmd[0]:
                    if "--version" in cmd:
                        return Mock(returncode=0, stdout="repomapper 1.0.0", stderr="")
                    else:
                        return Mock(
                            returncode=0, stdout=json.dumps(mock_responses["repomapper"]), stderr=""
                        )
                elif "serena" in cmd[0]:
                    if "--version" in cmd:
                        return Mock(returncode=0, stdout="serena 1.0.0", stderr="")
                    else:
                        return Mock(
                            returncode=0, stdout=json.dumps(mock_responses["serena"]), stderr=""
                        )
                elif "leann" in cmd[0]:
                    if "--version" in cmd:
                        return Mock(returncode=0, stdout="leann 1.0.0", stderr="")
                    else:
                        return Mock(
                            returncode=0, stdout=json.dumps(mock_responses["leann"]), stderr=""
                        )
                else:
                    return subprocess.run(*args, **kwargs)

            mock_run.side_effect = subprocess_side_effect

            # Call ensure_repo_index tool
            result = await mcp_server._ensure_repo_index(
                {
                    "repo_root": str(test_repo),
                    "config": {
                        "languages": ["ts", "tsx", "js", "jsx"],
                        "excludes": ["node_modules/", "dist/", ".git/"],
                        "context_lines": 3,
                        "max_files_to_embed": 50,
                    },
                }
            )

            # Verify the result
            assert result.is_error is False
            content = json.loads(result.content[0].text)

            assert "success" in content
            assert content["success"] is True
            assert "index_id" in content
            assert "files" in content
            assert len(content["files"]) >= 3  # We created 3 .ts files

            print("✅ Successfully indexed repository")
            print(f"   Index ID: {content['index_id']}")
            print(f"   Files indexed: {len(content['files'])}")

            return content["index_id"]

    @pytest.mark.integration
    async def test_search_repo_tool(self, test_repo: Path, mcp_server: MCPServer):
        """Test the search_repo MCP tool."""
        # First ensure the repo is indexed
        index_id = await self.test_ensure_repo_index_tool(test_repo, mcp_server)

        # Test vector search
        search_result = await mcp_server._search_repo(
            {
                "index_id": index_id,
                "query": "authentication login controller",
                "features": {"vector": True, "symbol": True, "graph": False},
                "k": 10,
            }
        )

        # Verify search results
        assert search_result.is_error is False
        content = json.loads(search_result.content[0].text)

        assert "results" in content
        results = content["results"]
        assert len(results) > 0

        print(f"✅ Search returned {len(results)} results")

        # Verify result structure
        for result in results:
            assert "path" in result
            assert "content" in result
            assert "score" in result
            assert "type" in result

        # Test symbol-only search
        symbol_search_result = await mcp_server._search_repo(
            {
                "index_id": index_id,
                "query": "AuthController",
                "features": {"vector": False, "symbol": True, "graph": False},
                "k": 5,
            }
        )

        assert symbol_search_result.is_error is False
        symbol_content = json.loads(symbol_search_result.content[0].text)
        symbol_results = symbol_content["results"]

        print(f"✅ Symbol search returned {len(symbol_results)} results")

    @pytest.mark.integration
    async def test_ask_index_tool(self, test_repo: Path, mcp_server: MCPServer):
        """Test the ask_index MCP tool."""
        # First ensure the repo is indexed
        index_id = await self.test_ensure_repo_index_tool(test_repo, mcp_server)

        # Ask a question about the codebase
        ask_result = await mcp_server._ask_index(
            {
                "index_id": index_id,
                "question": "What is the AuthController class and what does it do?",
            }
        )

        # Verify the answer
        assert ask_result.is_error is False
        content = json.loads(ask_result.content[0].text)

        assert "answer" in content
        answer = content["answer"]
        assert len(answer) > 0

        print("✅ Question answered successfully")
        print(f"   Answer length: {len(answer)} characters")
        print(f"   Answer preview: {answer[:100]}...")

    @pytest.mark.integration
    async def test_get_repo_bundle_tool(self, test_repo: Path, mcp_server: MCPServer):
        """Test the get_repo_bundle MCP tool."""
        # First ensure the repo is indexed
        index_id = await self.test_ensure_repo_index_tool(test_repo, mcp_server)

        # Get the bundle
        bundle_result = await mcp_server._get_repo_bundle({"index_id": index_id})

        # Verify the bundle
        assert bundle_result.is_error is False
        content = json.loads(bundle_result.content[0].text)

        assert "bundle_data" in content
        bundle_data = content["bundle_data"]

        # Verify bundle structure
        required_components = ["manifest", "repo_map", "symbol_graph", "vector_index", "snippets"]
        for component in required_components:
            assert component in bundle_data, f"Bundle missing component: {component}"

        # Verify each component has data
        manifest = bundle_data["manifest"]
        assert manifest["repo"]["root"] == str(test_repo)
        assert len(manifest["files"]) > 0

        repo_map = bundle_data["repo_map"]
        assert "file_ranks" in repo_map
        assert len(repo_map["file_ranks"]) > 0

        symbol_graph = bundle_data["symbol_graph"]
        assert "entries" in symbol_graph
        assert len(symbol_graph["entries"]) > 0

        vector_index = bundle_data["vector_index"]
        assert "chunks" in vector_index
        assert len(vector_index["chunks"]) > 0

        snippets = bundle_data["snippets"]
        assert "snippets" in snippets
        assert len(snippets["snippets"]) > 0

        print("✅ Bundle exported successfully:")
        print(f"   Files ranked: {len(repo_map['file_ranks'])}")
        print(f"   Symbol entries: {len(symbol_graph['entries'])}")
        print(f"   Vector chunks: {len(vector_index['chunks'])}")
        print(f"   Code snippets: {len(snippets['snippets'])}")

    @pytest.mark.integration
    async def test_mcp_resources(self, test_repo: Path, mcp_server: MCPServer):
        """Test MCP resources functionality."""
        # First ensure the repo is indexed
        index_id = await self.test_ensure_repo_index_tool(test_repo, mcp_server)

        # List available resources
        resources = await mcp_server.server.list_resources()

        # Should have status resource
        status_resources = [r for r in resources if r.name == "status"]
        assert len(status_resources) > 0

        print(f"✅ Found {len(resources)} MCP resources")

        # Read status resource
        status_uri = f"status://index/{index_id}"
        status_result = await mcp_server._get_status_resource(
            index_id, mcp_server.indexes_dir / index_id
        )

        # Verify status content
        assert status_result.contents[0].text is not None
        status_data = json.loads(status_result.contents[0].text)

        assert "index_id" in status_data
        assert "state" in status_data
        assert "files" in status_data

        print("✅ Status resource read successfully")
        print(f"   State: {status_data['state']}")
        print(f"   Files: {len(status_data['files'])}")

    @pytest.mark.integration
    async def test_end_to_end_workflow(self, test_repo: Path, mcp_server: MCPServer):
        """Test complete end-to-end workflow."""
        print("🚀 Starting end-to-end workflow test")

        # Step 1: Index repository
        print("📁 Step 1: Indexing repository...")
        index_id = await self.test_ensure_repo_index_tool(test_repo, mcp_server)

        # Step 2: Perform searches
        print("🔍 Step 2: Testing search functionality...")
        await self.test_search_repo_tool(test_repo, mcp_server)

        # Step 3: Ask questions
        print("❓ Step 3: Testing question answering...")
        await self.test_ask_index_tool(test_repo, mcp_server)

        # Step 4: Export bundle
        print("📦 Step 4: Testing bundle export...")
        await self.test_get_repo_bundle_tool(test_repo, mcp_server)

        # Step 5: Check resources
        print("📋 Step 5: Testing MCP resources...")
        await self.test_mcp_resources(test_repo, mcp_server)

        print("✅ End-to-end workflow completed successfully!")


if __name__ == "__main__":
    # Quick test for development
    import asyncio

    async def quick_test():
        test_instance = TestMCPIntegration()

        # Create test repo
        test_repo_gen = test_instance.test_repo()
        test_repo = await test_repo_gen.__anext__()

        print(f"Created test repo at: {test_repo}")
        print(f"TypeScript files: {list(test_repo.rglob('*.ts'))}")

        # Clean up
        await test_repo_gen.__anext__()

    asyncio.run(quick_test())
