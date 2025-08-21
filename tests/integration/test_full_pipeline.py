"""
Integration tests for the full Mimir pipeline.

Tests the complete workflow:
1. Create test repository with real TypeScript/JavaScript files
2. Run full pipeline: Acquire -> RepoMapper -> Serena -> LEANN -> Snippets -> Bundle
3. Verify indexing results are sane
4. Test querying capabilities
5. Validate search results make sense
"""

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.repoindex.data.schemas import IndexConfig, IndexManifest, IndexState, RepoInfo
from src.repoindex.mcp.server import MimirMCPServer
from src.repoindex.pipeline.run import PipelineRunner


class TestFullPipelineIntegration:
    """Test the complete pipeline with real external tools."""

    @pytest.fixture
    async def test_repo(self) -> Path:
        """Create a test repository with real TypeScript/JavaScript files."""
        repo_dir = Path(tempfile.mkdtemp(prefix="mimir_test_repo_"))

        # Create package.json
        package_json = {
            "name": "test-auth-app",
            "version": "1.0.0",
            "main": "src/index.ts",
            "scripts": {"start": "ts-node src/index.ts", "test": "jest"},
            "dependencies": {"express": "^4.18.0", "jsonwebtoken": "^9.0.0", "bcrypt": "^5.1.0"},
            "devDependencies": {"typescript": "^5.0.0", "@types/node": "^18.0.0"},
        }

        (repo_dir / "package.json").write_text(json.dumps(package_json, indent=2))

        # Create TypeScript configuration
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "strict": true,
                "esModuleInterop": true,
                "outDir": "./dist",
            },
            "include": ["src/**/*"],
            "exclude": ["node_modules", "dist"],
        }

        (repo_dir / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2))

        # Create source directory
        src_dir = repo_dir / "src"
        src_dir.mkdir()

        # Create main application file
        index_ts = """import express from 'express';
import { AuthController } from './auth/controller';
import { UserService } from './services/user';
import { DatabaseConnection } from './database/connection';

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());

// Services
const dbConnection = new DatabaseConnection();
const userService = new UserService(dbConnection);
const authController = new AuthController(userService);

// Routes
app.post('/auth/login', authController.login.bind(authController));
app.post('/auth/register', authController.register.bind(authController));
app.get('/auth/profile', authController.getProfile.bind(authController));

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});

export default app;
"""
        (src_dir / "index.ts").write_text(index_ts)

        # Create authentication module
        auth_dir = src_dir / "auth"
        auth_dir.mkdir()

        controller_ts = """import { Request, Response } from 'express';
import { UserService } from '../services/user';
import { generateToken, verifyToken } from '../utils/jwt';
import { hashPassword, comparePassword } from '../utils/crypto';

export interface LoginRequest {
    email: string;
    password: string;
}

export interface RegisterRequest {
    email: string;
    password: string;
    name: string;
}

export class AuthController {
    constructor(private userService: UserService) {}

    async login(req: Request, res: Response): Promise<void> {
        try {
            const { email, password }: LoginRequest = req.body;

            // Validate input
            if (!email || !password) {
                res.status(400).json({ error: 'Email and password required' });
                return;
            }

            // Find user
            const user = await this.userService.findByEmail(email);
            if (!user) {
                res.status(401).json({ error: 'Invalid credentials' });
                return;
            }

            // Verify password
            const isValid = await comparePassword(password, user.passwordHash);
            if (!isValid) {
                res.status(401).json({ error: 'Invalid credentials' });
                return;
            }

            // Generate token
            const token = generateToken({ userId: user.id, email: user.email });

            res.json({
                token,
                user: {
                    id: user.id,
                    email: user.email,
                    name: user.name
                }
            });
        } catch (error) {
            console.error('Login error:', error);
            res.status(500).json({ error: 'Internal server error' });
        }
    }

    async register(req: Request, res: Response): Promise<void> {
        try {
            const { email, password, name }: RegisterRequest = req.body;

            // Validate input
            if (!email || !password || !name) {
                res.status(400).json({ error: 'Email, password, and name required' });
                return;
            }

            // Check if user exists
            const existingUser = await this.userService.findByEmail(email);
            if (existingUser) {
                res.status(409).json({ error: 'User already exists' });
                return;
            }

            // Hash password
            const passwordHash = await hashPassword(password);

            // Create user
            const user = await this.userService.create({
                email,
                passwordHash,
                name
            });

            // Generate token
            const token = generateToken({ userId: user.id, email: user.email });

            res.status(201).json({
                token,
                user: {
                    id: user.id,
                    email: user.email,
                    name: user.name
                }
            });
        } catch (error) {
            console.error('Registration error:', error);
            res.status(500).json({ error: 'Internal server error' });
        }
    }

    async getProfile(req: Request, res: Response): Promise<void> {
        try {
            const token = req.headers.authorization?.replace('Bearer ', '');
            if (!token) {
                res.status(401).json({ error: 'Token required' });
                return;
            }

            const payload = verifyToken(token);
            const user = await this.userService.findById(payload.userId);

            if (!user) {
                res.status(404).json({ error: 'User not found' });
                return;
            }

            res.json({
                user: {
                    id: user.id,
                    email: user.email,
                    name: user.name
                }
            });
        } catch (error) {
            console.error('Profile error:', error);
            res.status(401).json({ error: 'Invalid token' });
        }
    }
}
"""
        (auth_dir / "controller.ts").write_text(controller_ts)

        # Create user service
        services_dir = src_dir / "services"
        services_dir.mkdir()

        user_service_ts = """import { DatabaseConnection } from '../database/connection';

export interface User {
    id: string;
    email: string;
    name: string;
    passwordHash: string;
    createdAt: Date;
    updatedAt: Date;
}

export interface CreateUserData {
    email: string;
    passwordHash: string;
    name: string;
}

export class UserService {
    constructor(private db: DatabaseConnection) {}

    async findById(id: string): Promise<User | null> {
        const query = 'SELECT * FROM users WHERE id = ?';
        const result = await this.db.query(query, [id]);
        return result.length > 0 ? this.mapRowToUser(result[0]) : null;
    }

    async findByEmail(email: string): Promise<User | null> {
        const query = 'SELECT * FROM users WHERE email = ?';
        const result = await this.db.query(query, [email]);
        return result.length > 0 ? this.mapRowToUser(result[0]) : null;
    }

    async create(userData: CreateUserData): Promise<User> {
        const id = this.generateId();
        const now = new Date();

        const query = `
            INSERT INTO users (id, email, password_hash, name, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
        `;

        await this.db.query(query, [
            id,
            userData.email,
            userData.passwordHash,
            userData.name,
            now,
            now
        ]);

        return {
            id,
            email: userData.email,
            name: userData.name,
            passwordHash: userData.passwordHash,
            createdAt: now,
            updatedAt: now
        };
    }

    async update(id: string, updates: Partial<CreateUserData>): Promise<User | null> {
        const user = await this.findById(id);
        if (!user) return null;

        const fields = [];
        const values = [];

        if (updates.email) {
            fields.push('email = ?');
            values.push(updates.email);
        }
        if (updates.name) {
            fields.push('name = ?');
            values.push(updates.name);
        }
        if (updates.passwordHash) {
            fields.push('password_hash = ?');
            values.push(updates.passwordHash);
        }

        if (fields.length === 0) return user;

        fields.push('updated_at = ?');
        values.push(new Date());
        values.push(id);

        const query = `UPDATE users SET ${fields.join(', ')} WHERE id = ?`;
        await this.db.query(query, values);

        return this.findById(id);
    }

    async delete(id: string): Promise<boolean> {
        const query = 'DELETE FROM users WHERE id = ?';
        const result = await this.db.query(query, [id]);
        return result.affectedRows > 0;
    }

    private mapRowToUser(row: any): User {
        return {
            id: row.id,
            email: row.email,
            name: row.name,
            passwordHash: row.password_hash,
            createdAt: new Date(row.created_at),
            updatedAt: new Date(row.updated_at)
        };
    }

    private generateId(): string {
        return Math.random().toString(36).substring(2) + Date.now().toString(36);
    }
}
"""
        (services_dir / "user.ts").write_text(user_service_ts)

        # Create utilities
        utils_dir = src_dir / "utils"
        utils_dir.mkdir()

        jwt_utils_ts = """import jwt from 'jsonwebtoken';

const JWT_SECRET = process.env.JWT_SECRET || 'default-secret-key';
const JWT_EXPIRES_IN = process.env.JWT_EXPIRES_IN || '24h';

export interface TokenPayload {
    userId: string;
    email: string;
}

export function generateToken(payload: TokenPayload): string {
    return jwt.sign(payload, JWT_SECRET, { expiresIn: JWT_EXPIRES_IN });
}

export function verifyToken(token: string): TokenPayload {
    try {
        return jwt.verify(token, JWT_SECRET) as TokenPayload;
    } catch (error) {
        throw new Error('Invalid token');
    }
}

export function refreshToken(token: string): string {
    const payload = verifyToken(token);
    return generateToken({ userId: payload.userId, email: payload.email });
}
"""
        (utils_dir / "jwt.ts").write_text(jwt_utils_ts)

        crypto_utils_ts = """import bcrypt from 'bcrypt';

const SALT_ROUNDS = 10;

export async function hashPassword(password: string): Promise<string> {
    return bcrypt.hash(password, SALT_ROUNDS);
}

export async function comparePassword(password: string, hash: string): Promise<boolean> {
    return bcrypt.compare(password, hash);
}

export function generateSalt(): string {
    return bcrypt.genSaltSync(SALT_ROUNDS);
}
"""
        (utils_dir / "crypto.ts").write_text(crypto_utils_ts)

        # Create database connection
        database_dir = src_dir / "database"
        database_dir.mkdir()

        connection_ts = """export interface QueryResult {
    affectedRows: number;
    insertId?: number;
}

export class DatabaseConnection {
    private connectionString: string;

    constructor() {
        this.connectionString = process.env.DATABASE_URL || 'sqlite://test.db';
    }

    async connect(): Promise<void> {
        console.log('Connecting to database:', this.connectionString);
        // Database connection logic would go here
    }

    async disconnect(): Promise<void> {
        console.log('Disconnecting from database');
        // Database disconnection logic would go here
    }

    async query(sql: string, params: any[] = []): Promise<any[]> {
        console.log('Executing query:', sql, 'with params:', params);
        // Database query logic would go here
        return [];
    }

    async transaction<T>(callback: (conn: DatabaseConnection) => Promise<T>): Promise<T> {
        console.log('Starting transaction');
        try {
            const result = await callback(this);
            console.log('Committing transaction');
            return result;
        } catch (error) {
            console.log('Rolling back transaction');
            throw error;
        }
    }
}
"""
        (database_dir / "connection.ts").write_text(connection_ts)

        # Create README
        readme_md = """# Test Authentication App

A sample TypeScript/Express authentication application for testing Mimir indexing.

## Features

- User registration and login
- JWT token authentication
- Password hashing with bcrypt
- TypeScript with strict typing
- Modular architecture

## Architecture

- `src/index.ts` - Main application entry point
- `src/auth/` - Authentication controllers and logic
- `src/services/` - Business logic services
- `src/utils/` - Utility functions (JWT, crypto)
- `src/database/` - Database connection and operations

## Usage

```bash
npm install
npm start
```

## API Endpoints

- `POST /auth/login` - User login
- `POST /auth/register` - User registration
- `GET /auth/profile` - Get user profile (requires auth)
"""
        (repo_dir / "README.md").write_text(readme_md)

        # Initialize git repository
        import subprocess

        subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True)

        yield repo_dir

        # Cleanup
        shutil.rmtree(repo_dir)

    @pytest.fixture
    async def pipeline_runner(self) -> PipelineRunner:
        """Create a pipeline runner for testing."""
        storage_dir = Path(tempfile.mkdtemp(prefix="mimir_storage_"))
        runner = PipelineRunner(storage_dir=str(storage_dir))
        yield runner
        shutil.rmtree(storage_dir)

    @pytest.fixture
    async def mcp_server(self, pipeline_runner) -> MimirMCPServer:
        """Create MCP server for testing."""
        return MimirMCPServer(pipeline_runner)

    @pytest.mark.integration
    async def test_full_pipeline_execution(self, test_repo: Path, pipeline_runner: PipelineRunner):
        """Test complete pipeline execution with real external tools."""
        # Create repository info
        repo_info = RepoInfo(root=str(test_repo), rev="HEAD", worktree_dirty=False)

        # Create indexing configuration
        config = IndexConfig(
            languages=["ts", "tsx", "js", "jsx"],
            excludes=["node_modules/", "dist/", ".git/"],
            context_lines=3,
            max_files_to_embed=50,  # Small for testing
        )

        # Create manifest
        manifest = IndexManifest(repo=repo_info, config=config)

        print(f"Testing pipeline with repo: {test_repo}")
        print(f"Index ID: {manifest.index_id}")

        # Run the complete pipeline
        try:
            result = await pipeline_runner.run_pipeline(manifest)

            # Verify pipeline completed successfully
            assert result.state == IndexState.DONE
            assert result.error is None
            assert result.progress == 100

            print("Pipeline completed successfully!")
            print(f"Final state: {result.state}")

            # Verify pipeline stages were executed
            assert result.stages.acquire.state == IndexState.DONE
            assert result.stages.repo_mapper.state == IndexState.DONE
            assert result.stages.serena.state == IndexState.DONE
            assert result.stages.leann.state == IndexState.DONE
            assert result.stages.snippets.state == IndexState.DONE
            assert result.stages.bundle.state == IndexState.DONE

            print("All pipeline stages completed successfully!")

            # Check that files were discovered
            assert len(result.files) > 0
            print(f"Discovered {len(result.files)} files")

            # Verify TypeScript files were found
            ts_files = [f for f in result.files if f.endswith(".ts")]
            assert len(ts_files) >= 6  # We created 6 .ts files
            print(f"Found {len(ts_files)} TypeScript files")

            return result

        except Exception as e:
            print(f"Pipeline failed with error: {e}")
            # Check if it's due to missing external tools
            if (
                "repomapper" in str(e).lower()
                or "serena" in str(e).lower()
                or "leann" in str(e).lower()
            ):
                pytest.skip(f"External tool not available: {e}")
            else:
                raise

    @pytest.mark.integration
    async def test_search_functionality(self, test_repo: Path, mcp_server: MimirMCPServer):
        """Test search functionality after indexing."""
        # First, ensure the repository is indexed
        index_result = await mcp_server.ensure_repo_index(
            repo_root=str(test_repo),
            config={
                "languages": ["ts", "tsx", "js", "jsx"],
                "excludes": ["node_modules/", "dist/", ".git/"],
                "context_lines": 3,
                "max_files_to_embed": 50,
            },
        )

        if index_result.get("error"):
            if any(
                tool in index_result["error"].lower() for tool in ["repomapper", "serena", "leann"]
            ):
                pytest.skip(f"External tool not available: {index_result['error']}")
            else:
                pytest.fail(f"Indexing failed: {index_result['error']}")

        index_id = index_result["index_id"]
        print(f"Repository indexed with ID: {index_id}")

        # Test vector search for authentication-related code
        search_result = await mcp_server.search_repo(
            index_id=index_id,
            query="authentication login password",
            features={"vector": True, "symbol": True, "graph": False},
            k=10,
        )

        assert "results" in search_result
        results = search_result["results"]
        assert len(results) > 0

        print(f"Vector search returned {len(results)} results")

        # Verify results contain relevant authentication code
        auth_related = 0
        for result in results:
            content = result.get("content", "").lower()
            if any(term in content for term in ["login", "password", "auth", "token"]):
                auth_related += 1

        assert auth_related > 0, "Search results should contain authentication-related code"
        print(f"{auth_related} results contained authentication-related content")

        # Test symbol search
        symbol_search = await mcp_server.search_repo(
            index_id=index_id,
            query="AuthController",
            features={"vector": False, "symbol": True, "graph": False},
            k=5,
        )

        assert "results" in symbol_search
        symbol_results = symbol_search["results"]

        # Should find the AuthController class and its methods
        controller_found = any("AuthController" in r.get("content", "") for r in symbol_results)
        assert controller_found, "Should find AuthController in symbol search"

        print(f"Symbol search found AuthController: {controller_found}")

    @pytest.mark.integration
    async def test_ask_functionality(self, test_repo: Path, mcp_server: MimirMCPServer):
        """Test natural language question answering."""
        # Ensure repository is indexed
        index_result = await mcp_server.ensure_repo_index(
            repo_root=str(test_repo),
            config={
                "languages": ["ts", "tsx", "js", "jsx"],
                "excludes": ["node_modules/", "dist/", ".git/"],
                "context_lines": 3,
                "max_files_to_embed": 50,
            },
        )

        if index_result.get("error"):
            if any(
                tool in index_result["error"].lower() for tool in ["repomapper", "serena", "leann"]
            ):
                pytest.skip(f"External tool not available: {index_result['error']}")
            else:
                pytest.fail(f"Indexing failed: {index_result['error']}")

        index_id = index_result["index_id"]

        # Ask questions about the codebase
        questions = [
            "How does user authentication work in this application?",
            "What endpoints are available for authentication?",
            "How are passwords handled securely?",
            "What is the AuthController class responsible for?",
        ]

        for question in questions:
            print(f"Asking: {question}")

            answer_result = await mcp_server.ask_index(index_id=index_id, question=question)

            assert "answer" in answer_result
            answer = answer_result["answer"]
            assert len(answer) > 0

            print(f"Answer length: {len(answer)} characters")

            # Verify answer contains relevant context
            answer_lower = answer.lower()
            if "authentication" in question.lower():
                assert any(term in answer_lower for term in ["auth", "login", "password", "token"])
            elif "endpoints" in question.lower():
                assert any(term in answer_lower for term in ["post", "get", "/auth", "route"])
            elif "password" in question.lower():
                assert any(term in answer_lower for term in ["hash", "bcrypt", "secure"])
            elif "AuthController" in question:
                assert "authcontroller" in answer_lower

    @pytest.mark.integration
    async def test_bundle_export_import(self, test_repo: Path, mcp_server: MimirMCPServer):
        """Test bundle export and import functionality."""
        # Index the repository
        index_result = await mcp_server.ensure_repo_index(
            repo_root=str(test_repo),
            config={
                "languages": ["ts", "tsx", "js", "jsx"],
                "excludes": ["node_modules/", "dist/", ".git/"],
                "context_lines": 3,
                "max_files_to_embed": 50,
            },
        )

        if index_result.get("error"):
            if any(
                tool in index_result["error"].lower() for tool in ["repomapper", "serena", "leann"]
            ):
                pytest.skip(f"External tool not available: {index_result['error']}")
            else:
                pytest.fail(f"Indexing failed: {index_result['error']}")

        index_id = index_result["index_id"]

        # Export bundle
        bundle_result = await mcp_server.get_repo_bundle(index_id=index_id)

        assert "bundle_data" in bundle_result
        bundle_data = bundle_result["bundle_data"]

        # Verify bundle structure
        assert "manifest" in bundle_data
        assert "repo_map" in bundle_data
        assert "symbol_graph" in bundle_data
        assert "vector_index" in bundle_data
        assert "snippets" in bundle_data

        manifest = bundle_data["manifest"]
        assert manifest["repo"]["root"] == str(test_repo)
        assert len(manifest["files"]) > 0

        # Verify repo map has file rankings
        repo_map = bundle_data["repo_map"]
        assert "file_ranks" in repo_map
        assert len(repo_map["file_ranks"]) > 0

        # Verify symbol graph has entries
        symbol_graph = bundle_data["symbol_graph"]
        assert "entries" in symbol_graph
        assert len(symbol_graph["entries"]) > 0

        # Verify vector index has chunks
        vector_index = bundle_data["vector_index"]
        assert "chunks" in vector_index
        assert len(vector_index["chunks"]) > 0

        # Verify snippets collection
        snippets = bundle_data["snippets"]
        assert "snippets" in snippets
        assert len(snippets["snippets"]) > 0

        print("Bundle exported successfully:")
        print(f"  - {len(repo_map['file_ranks'])} files ranked")
        print(f"  - {len(symbol_graph['entries'])} symbol entries")
        print(f"  - {len(vector_index['chunks'])} vector chunks")
        print(f"  - {len(snippets['snippets'])} code snippets")

    @pytest.mark.integration
    async def test_incremental_indexing(self, test_repo: Path, mcp_server: MimirMCPServer):
        """Test incremental indexing when files change."""
        # Initial indexing
        index_result = await mcp_server.ensure_repo_index(
            repo_root=str(test_repo),
            config={
                "languages": ["ts", "tsx", "js", "jsx"],
                "excludes": ["node_modules/", "dist/", ".git/"],
                "context_lines": 3,
                "max_files_to_embed": 50,
            },
        )

        if index_result.get("error"):
            if any(
                tool in index_result["error"].lower() for tool in ["repomapper", "serena", "leann"]
            ):
                pytest.skip(f"External tool not available: {index_result['error']}")
            else:
                pytest.fail(f"Indexing failed: {index_result['error']}")

        index_id = index_result["index_id"]
        original_file_count = len(index_result["files"])

        # Add a new file to the repository
        new_file = test_repo / "src" / "middleware" / "auth.ts"
        new_file.parent.mkdir(exist_ok=True)

        auth_middleware = """import { Request, Response, NextFunction } from 'express';
import { verifyToken } from '../utils/jwt';

export interface AuthenticatedRequest extends Request {
    user?: {
        userId: string;
        email: string;
    };
}

export function requireAuth(req: AuthenticatedRequest, res: Response, next: NextFunction): void {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (!token) {
        res.status(401).json({ error: 'Authentication required' });
        return;
    }

    try {
        const payload = verifyToken(token);
        req.user = payload;
        next();
    } catch (error) {
        res.status(401).json({ error: 'Invalid token' });
    }
}

export function optionalAuth(req: AuthenticatedRequest, res: Response, next: NextFunction): void {
    const token = req.headers.authorization?.replace('Bearer ', '');

    if (token) {
        try {
            const payload = verifyToken(token);
            req.user = payload;
        } catch (error) {
            // Ignore invalid tokens for optional auth
        }
    }

    next();
}
"""
        new_file.write_text(auth_middleware)

        # Commit the change
        import subprocess

        subprocess.run(["git", "add", "."], cwd=test_repo, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", "Add auth middleware"], cwd=test_repo, capture_output=True
        )

        # Re-index (should detect the change)
        reindex_result = await mcp_server.ensure_repo_index(
            repo_root=str(test_repo),
            config={
                "languages": ["ts", "tsx", "js", "jsx"],
                "excludes": ["node_modules/", "dist/", ".git/"],
                "context_lines": 3,
                "max_files_to_embed": 50,
            },
        )

        if reindex_result.get("error"):
            pytest.fail(f"Re-indexing failed: {reindex_result['error']}")

        # Should detect the new file
        new_file_count = len(reindex_result["files"])
        assert new_file_count > original_file_count

        # Search for the new middleware
        search_result = await mcp_server.search_repo(
            index_id=index_id,
            query="requireAuth middleware authentication",
            features={"vector": True, "symbol": True, "graph": False},
            k=5,
        )

        # Should find the new middleware function
        middleware_found = any(
            "requireAuth" in r.get("content", "") for r in search_result["results"]
        )
        assert middleware_found, "Should find new requireAuth middleware"

        print("Incremental indexing successful:")
        print(f"  - Original files: {original_file_count}")
        print(f"  - New files: {new_file_count}")
        print(f"  - Found new middleware: {middleware_found}")


if __name__ == "__main__":
    # Run a quick test if executed directly
    import asyncio

    async def quick_test():
        test_instance = TestFullPipelineIntegration()

        # Create test repo
        test_repo_gen = test_instance.test_repo()
        test_repo = await test_repo_gen.__anext__()

        print(f"Created test repo at: {test_repo}")
        print(f"Files created: {list(test_repo.rglob('*.ts'))}")

        # Clean up
        await test_repo_gen.__anext__()

    asyncio.run(quick_test())
