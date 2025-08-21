"""
Integration tests for Mimir pipeline with mocked external tools.

Since external tools (RepoMapper, Serena, LEANN) may not be installed,
this test mocks their responses but verifies the full pipeline flow
and data processing works correctly.
"""

import asyncio
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import patch, Mock, AsyncMock
import subprocess

import pytest

from src.repoindex.pipeline.run import IndexingPipeline
from src.repoindex.data.schemas import (
    IndexManifest, RepoInfo, IndexConfig, IndexState
)
from src.repoindex.mcp.server import MCPServer


class TestPipelineWithMockedTools:
    """Test the complete pipeline with mocked external tools."""
    
    @pytest.fixture
    async def test_repo(self) -> Path:
        """Create a test repository with real TypeScript/JavaScript files."""
        repo_dir = Path(tempfile.mkdtemp(prefix="mimir_test_repo_"))
        
        # Create package.json
        package_json = {
            "name": "test-auth-app",
            "version": "1.0.0",
            "main": "src/index.ts"
        }
        (repo_dir / "package.json").write_text(json.dumps(package_json, indent=2))
        
        # Create source directory with TypeScript files
        src_dir = repo_dir / "src"
        src_dir.mkdir()
        
        # Create main application file
        index_ts = '''import express from 'express';
import { AuthController } from './auth/controller';
import { UserService } from './services/user';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());

const userService = new UserService();
const authController = new AuthController(userService);

app.post('/auth/login', authController.login.bind(authController));
app.post('/auth/register', authController.register.bind(authController));

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

export default app;
'''
        (src_dir / "index.ts").write_text(index_ts)
        
        # Create authentication controller
        auth_dir = src_dir / "auth"
        auth_dir.mkdir()
        
        controller_ts = '''import { Request, Response } from 'express';
import { UserService } from '../services/user';
import { generateToken } from '../utils/jwt';
import { hashPassword, comparePassword } from '../utils/crypto';

export interface LoginRequest {
    email: string;
    password: string;
}

export class AuthController {
    constructor(private userService: UserService) {}

    async login(req: Request, res: Response): Promise<void> {
        try {
            const { email, password }: LoginRequest = req.body;
            
            const user = await this.userService.findByEmail(email);
            if (!user) {
                res.status(401).json({ error: 'Invalid credentials' });
                return;
            }

            const isValid = await comparePassword(password, user.passwordHash);
            if (!isValid) {
                res.status(401).json({ error: 'Invalid credentials' });
                return;
            }

            const token = generateToken({ userId: user.id, email: user.email });
            
            res.json({ token, user: { id: user.id, email: user.email, name: user.name } });
        } catch (error) {
            res.status(500).json({ error: 'Internal server error' });
        }
    }

    async register(req: Request, res: Response): Promise<void> {
        try {
            const { email, password, name } = req.body;
            
            const existingUser = await this.userService.findByEmail(email);
            if (existingUser) {
                res.status(409).json({ error: 'User already exists' });
                return;
            }

            const passwordHash = await hashPassword(password);
            const user = await this.userService.create({ email, passwordHash, name });
            const token = generateToken({ userId: user.id, email: user.email });

            res.status(201).json({ token, user: { id: user.id, email: user.email, name: user.name } });
        } catch (error) {
            res.status(500).json({ error: 'Internal server error' });
        }
    }
}
'''
        (auth_dir / "controller.ts").write_text(controller_ts)
        
        # Create user service
        services_dir = src_dir / "services"
        services_dir.mkdir()
        
        user_service_ts = '''export interface User {
    id: string;
    email: string;
    name: string;
    passwordHash: string;
}

export class UserService {
    private users: User[] = [];

    async findById(id: string): Promise<User | null> {
        return this.users.find(user => user.id === id) || null;
    }

    async findByEmail(email: string): Promise<User | null> {
        return this.users.find(user => user.email === email) || null;
    }

    async create(userData: { email: string; passwordHash: string; name: string }): Promise<User> {
        const user: User = {
            id: Math.random().toString(36),
            ...userData
        };
        this.users.push(user);
        return user;
    }
}
'''
        (services_dir / "user.ts").write_text(user_service_ts)
        
        # Create utilities
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()
        
        jwt_utils_ts = '''export interface TokenPayload {
    userId: string;
    email: string;
}

export function generateToken(payload: TokenPayload): string {
    return 'mock-jwt-token';
}

export function verifyToken(token: string): TokenPayload {
    return { userId: 'user123', email: 'test@example.com' };
}
'''
        (utils_dir / "jwt.ts").write_text(jwt_utils_ts)
        
        crypto_utils_ts = '''export async function hashPassword(password: string): Promise<string> {
    return 'hashed-' + password;
}

export async function comparePassword(password: string, hash: string): Promise<boolean> {
    return hash === 'hashed-' + password;
}
'''
        (utils_dir / "crypto.ts").write_text(crypto_utils_ts)
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)
        
        yield repo_dir
        
        # Cleanup
        shutil.rmtree(repo_dir)

    @pytest.fixture
    async def indexing_pipeline(self) -> IndexingPipeline:
        """Create an indexing pipeline for testing."""
        storage_dir = Path(tempfile.mkdtemp(prefix="mimir_storage_"))
        pipeline = IndexingPipeline(storage_dir=str(storage_dir))
        yield pipeline
        shutil.rmtree(storage_dir)

    def create_mock_repomapper_output(self, files: List[str]) -> Dict[str, Any]:
        """Create realistic RepoMapper output."""
        return {
            "file_ranks": [
                {
                    "path": "src/index.ts",
                    "rank": 0.95,
                    "centrality": 0.8,
                    "dependencies": ["src/auth/controller.ts", "src/services/user.ts"]
                },
                {
                    "path": "src/auth/controller.ts", 
                    "rank": 0.85,
                    "centrality": 0.7,
                    "dependencies": ["src/services/user.ts", "src/utils/jwt.ts"]
                },
                {
                    "path": "src/services/user.ts",
                    "rank": 0.75,
                    "centrality": 0.6,
                    "dependencies": []
                },
                {
                    "path": "src/utils/jwt.ts",
                    "rank": 0.65,
                    "centrality": 0.5,
                    "dependencies": []
                },
                {
                    "path": "src/utils/crypto.ts",
                    "rank": 0.55,
                    "centrality": 0.4,
                    "dependencies": []
                }
            ],
            "dependency_graph": [
                {
                    "source": "src/index.ts",
                    "target": "src/auth/controller.ts",
                    "edge_type": "import"
                },
                {
                    "source": "src/auth/controller.ts",
                    "target": "src/services/user.ts",
                    "edge_type": "import"
                }
            ],
            "total_files": len(files)
        }

    def create_mock_serena_output(self) -> Dict[str, Any]:
        """Create realistic Serena symbol analysis output."""
        return {
            "symbols": [
                {
                    "type": "definition",
                    "kind": "class",
                    "name": "AuthController",
                    "path": "src/auth/controller.ts",
                    "start_line": 8,
                    "end_line": 45,
                    "start_col": 0,
                    "end_col": 1,
                    "signature": "export class AuthController",
                    "modifiers": ["export"]
                },
                {
                    "type": "definition", 
                    "kind": "method",
                    "name": "login",
                    "path": "src/auth/controller.ts",
                    "start_line": 12,
                    "end_line": 28,
                    "start_col": 4,
                    "end_col": 5,
                    "signature": "async login(req: Request, res: Response): Promise<void>",
                    "modifiers": ["async"],
                    "parent_symbol": "AuthController"
                },
                {
                    "type": "definition",
                    "kind": "method", 
                    "name": "register",
                    "path": "src/auth/controller.ts",
                    "start_line": 30,
                    "end_line": 43,
                    "start_col": 4,
                    "end_col": 5,
                    "signature": "async register(req: Request, res: Response): Promise<void>",
                    "modifiers": ["async"],
                    "parent_symbol": "AuthController"
                },
                {
                    "type": "definition",
                    "kind": "class",
                    "name": "UserService", 
                    "path": "src/services/user.ts",
                    "start_line": 8,
                    "end_line": 35,
                    "start_col": 0,
                    "end_col": 1,
                    "signature": "export class UserService",
                    "modifiers": ["export"]
                },
                {
                    "type": "reference",
                    "kind": "identifier",
                    "name": "UserService",
                    "path": "src/index.ts",
                    "start_line": 3,
                    "end_line": 3,
                    "start_col": 9,
                    "end_col": 20,
                    "context": "import { UserService } from './services/user';"
                }
            ]
        }

    def create_mock_leann_output(self, files: List[str]) -> Dict[str, Any]:
        """Create realistic LEANN embedding output."""
        chunks = []
        chunk_id = 1
        
        for file_path in files[:3]:  # Mock first 3 files
            # Create mock embedding vector (384 dimensions)
            embedding = [0.1 * i for i in range(10)]  # Simplified for testing
            
            chunks.append({
                "chunk_id": f"chunk_{chunk_id:03d}",
                "file_path": file_path,
                "content": f"// Mock content from {file_path}",
                "start_line": 1,
                "end_line": 10,
                "start_char": 0,
                "end_char": 200,
                "embedding_vector": embedding,
                "vector_norm": 1.0,
                "token_count": 25,
                "chunk_type": "code_block"
            })
            chunk_id += 1
        
        return {
            "embeddings": chunks,
            "metadata": {
                "model_name": "leann-cpu-v2",
                "embedding_dimensions": 10,  # Simplified
                "total_chunks": len(chunks),
                "total_tokens": len(chunks) * 25
            }
        }

    @pytest.mark.integration
    async def test_full_pipeline_with_mocked_tools(self, test_repo: Path, indexing_pipeline: IndexingPipeline):
        """Test complete pipeline with mocked external tool responses."""
        # Create repository info
        repo_info = RepoInfo(
            root=str(test_repo),
            rev="HEAD",
            worktree_dirty=False
        )
        
        # Create indexing configuration
        config = IndexConfig(
            languages=["ts", "tsx", "js", "jsx"],
            excludes=["node_modules/", "dist/", ".git/"],
            context_lines=3,
            max_files_to_embed=50
        )
        
        # Create manifest
        manifest = IndexManifest(repo=repo_info, config=config)
        
        # Get list of TypeScript files for mocking
        ts_files = [str(f.relative_to(test_repo)) for f in test_repo.rglob("*.ts")]
        
        print(f"Testing pipeline with mocked tools")
        print(f"Repository: {test_repo}")
        print(f"TypeScript files found: {len(ts_files)}")
        
        # Mock external tool subprocess calls
        with patch('subprocess.run') as mock_run:
            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get('args', [])
                
                if not cmd:
                    return Mock(returncode=1, stdout="", stderr="Unknown command")
                
                # Mock RepoMapper
                if 'repomapper' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="repomapper 1.0.0", stderr="")
                    else:
                        output = self.create_mock_repomapper_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                
                # Mock Serena
                elif 'serena' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="serena 1.0.0", stderr="")
                    else:
                        output = self.create_mock_serena_output()
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                
                # Mock LEANN
                elif 'leann' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="leann 1.0.0", stderr="")
                    else:
                        output = self.create_mock_leann_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                
                # Default git commands (real)
                else:
                    return subprocess.run(*args, **kwargs)
            
            mock_run.side_effect = subprocess_side_effect
            
            # Run the complete pipeline
            result = await indexing_pipeline.run_pipeline(manifest)
            
            # Verify pipeline completed successfully
            assert result.state == IndexState.DONE
            assert result.error is None
            assert result.progress == 100
            
            print(f"Pipeline completed successfully!")
            print(f"Final state: {result.state}")
            
            # Verify all stages completed
            assert result.stages.acquire.state == IndexState.DONE
            assert result.stages.repo_mapper.state == IndexState.DONE
            assert result.stages.serena.state == IndexState.DONE
            assert result.stages.leann.state == IndexState.DONE
            assert result.stages.snippets.state == IndexState.DONE
            assert result.stages.bundle.state == IndexState.DONE
            
            print("All pipeline stages completed successfully!")
            
            # Verify files were discovered
            assert len(result.files) > 0
            print(f"Discovered {len(result.files)} files")
            
            # Verify TypeScript files were found
            ts_files_found = [f for f in result.files if f.endswith('.ts')]
            assert len(ts_files_found) >= 5  # We created 5 .ts files
            print(f"Found {len(ts_files_found)} TypeScript files")

    @pytest.mark.integration
    async def test_search_with_mocked_data(self, test_repo: Path, indexing_pipeline: IndexingPipeline):
        """Test search functionality with mocked pipeline data."""
        # Create MCP server
        mcp_server = MCPServer(indexing_pipeline)
        
        # Mock external tools for indexing
        ts_files = [str(f.relative_to(test_repo)) for f in test_repo.rglob("*.ts")]
        
        with patch('subprocess.run') as mock_run:
            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get('args', [])
                
                if not cmd:
                    return Mock(returncode=1, stdout="", stderr="Unknown command")
                
                if 'repomapper' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="repomapper 1.0.0", stderr="")
                    else:
                        output = self.create_mock_repomapper_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                elif 'serena' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="serena 1.0.0", stderr="")
                    else:
                        output = self.create_mock_serena_output()
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                elif 'leann' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="leann 1.0.0", stderr="")
                    else:
                        output = self.create_mock_leann_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                else:
                    return subprocess.run(*args, **kwargs)
            
            mock_run.side_effect = subprocess_side_effect
            
            # Index the repository
            index_result = await mcp_server.ensure_repo_index(
                repo_root=str(test_repo),
                config={
                    "languages": ["ts", "tsx", "js", "jsx"],
                    "excludes": ["node_modules/", "dist/", ".git/"],
                    "context_lines": 3,
                    "max_files_to_embed": 50
                }
            )
            
            assert index_result.get("success") is True
            index_id = index_result["index_id"]
            print(f"Repository indexed with ID: {index_id}")
            
            # Test vector search
            search_result = await mcp_server.search_repo(
                index_id=index_id,
                query="authentication login password",
                features={
                    "vector": True,
                    "symbol": True,
                    "graph": False
                },
                k=10
            )
            
            assert "results" in search_result
            results = search_result["results"]
            assert len(results) > 0
            
            print(f"Vector search returned {len(results)} results")
            
            # Test symbol search
            symbol_search = await mcp_server.search_repo(
                index_id=index_id,
                query="AuthController",
                features={
                    "vector": False,
                    "symbol": True,
                    "graph": False
                },
                k=5
            )
            
            assert "results" in symbol_search
            symbol_results = symbol_search["results"]
            assert len(symbol_results) > 0
            
            print(f"Symbol search returned {len(symbol_results)} results")

    @pytest.mark.integration
    async def test_ask_functionality_with_mocked_data(self, test_repo: Path, indexing_pipeline: IndexingPipeline):
        """Test question answering with mocked data."""
        mcp_server = MCPServer(indexing_pipeline)
        ts_files = [str(f.relative_to(test_repo)) for f in test_repo.rglob("*.ts")]
        
        with patch('subprocess.run') as mock_run:
            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get('args', [])
                
                if not cmd:
                    return Mock(returncode=1, stdout="", stderr="Unknown command")
                
                if 'repomapper' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="repomapper 1.0.0", stderr="")
                    else:
                        output = self.create_mock_repomapper_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                elif 'serena' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="serena 1.0.0", stderr="")
                    else:
                        output = self.create_mock_serena_output()
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                elif 'leann' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="leann 1.0.0", stderr="")
                    else:
                        output = self.create_mock_leann_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                else:
                    return subprocess.run(*args, **kwargs)
            
            mock_run.side_effect = subprocess_side_effect
            
            # Index repository
            index_result = await mcp_server.ensure_repo_index(
                repo_root=str(test_repo),
                config={
                    "languages": ["ts", "tsx", "js", "jsx"],
                    "excludes": ["node_modules/", "dist/", ".git/"],
                    "context_lines": 3,
                    "max_files_to_embed": 50
                }
            )
            
            assert index_result.get("success") is True
            index_id = index_result["index_id"]
            
            # Ask questions about the codebase
            questions = [
                "What is the AuthController class?",
                "How does the login method work?",
                "What services are used in this application?"
            ]
            
            for question in questions:
                print(f"Asking: {question}")
                
                answer_result = await mcp_server.ask_index(
                    index_id=index_id,
                    question=question
                )
                
                assert "answer" in answer_result
                answer = answer_result["answer"]
                assert len(answer) > 0
                
                print(f"Answer received, length: {len(answer)} characters")

    @pytest.mark.integration  
    async def test_bundle_export_with_mocked_data(self, test_repo: Path, indexing_pipeline: IndexingPipeline):
        """Test bundle export with mocked pipeline data."""
        mcp_server = MCPServer(indexing_pipeline)
        ts_files = [str(f.relative_to(test_repo)) for f in test_repo.rglob("*.ts")]
        
        with patch('subprocess.run') as mock_run:
            def subprocess_side_effect(*args, **kwargs):
                cmd = args[0] if args else kwargs.get('args', [])
                
                if not cmd:
                    return Mock(returncode=1, stdout="", stderr="Unknown command")
                
                if 'repomapper' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="repomapper 1.0.0", stderr="")
                    else:
                        output = self.create_mock_repomapper_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                elif 'serena' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="serena 1.0.0", stderr="")
                    else:
                        output = self.create_mock_serena_output()
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                elif 'leann' in cmd[0]:
                    if '--version' in cmd:
                        return Mock(returncode=0, stdout="leann 1.0.0", stderr="")
                    else:
                        output = self.create_mock_leann_output(ts_files)
                        return Mock(returncode=0, stdout=json.dumps(output), stderr="")
                else:
                    return subprocess.run(*args, **kwargs)
            
            mock_run.side_effect = subprocess_side_effect
            
            # Index repository
            index_result = await mcp_server.ensure_repo_index(
                repo_root=str(test_repo),
                config={
                    "languages": ["ts", "tsx", "js", "jsx"],
                    "excludes": ["node_modules/", "dist/", ".git/"],
                    "context_lines": 3,
                    "max_files_to_embed": 50
                }
            )
            
            assert index_result.get("success") is True
            index_id = index_result["index_id"]
            
            # Export bundle
            bundle_result = await mcp_server.get_repo_bundle(index_id=index_id)
            
            assert "bundle_data" in bundle_result
            bundle_data = bundle_result["bundle_data"]
            
            # Verify bundle structure
            required_keys = ["manifest", "repo_map", "symbol_graph", "vector_index", "snippets"]
            for key in required_keys:
                assert key in bundle_data, f"Bundle missing required key: {key}"
            
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
            
            print(f"Bundle exported successfully:")
            print(f"  - {len(repo_map['file_ranks'])} files ranked")
            print(f"  - {len(symbol_graph['entries'])} symbol entries")
            print(f"  - {len(vector_index['chunks'])} vector chunks")
            print(f"  - {len(snippets['snippets'])} code snippets")

if __name__ == "__main__":
    # Quick test for development
    import asyncio
    
    async def quick_test():
        test_instance = TestPipelineWithMockedTools()
        
        # Create test repo
        test_repo_gen = test_instance.test_repo()
        test_repo = await test_repo_gen.__anext__()
        
        print(f"Created test repo at: {test_repo}")
        print(f"TypeScript files: {list(test_repo.rglob('*.ts'))}")
        
        # Clean up
        await test_repo_gen.__anext__()
    
    asyncio.run(quick_test())