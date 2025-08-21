"""
Tests for data format validation and compatibility.

Verifies that our data schemas can handle real-world data structures
from external tools and that serialization/deserialization works correctly.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, Any

from src.repoindex.data.schemas import (
    RepoMap, FileRank, DependencyEdge,
    SerenaGraph, SymbolEntry, SymbolType,
    VectorIndex, VectorChunk,
    IndexManifest, RepoInfo, IndexConfig,
    SearchResult, SearchScores, Citation,
    CodeSnippet, AskResponse
)


class TestDataFormatCompatibility:
    """Test compatibility with expected external tool data formats."""
    
    def test_repomapper_real_world_format(self):
        """Test RepoMapper data format parsing."""
        # Real-world RepoMapper output structure
        repomapper_data = {
            "file_ranks": [
                {
                    "path": "src/main.ts",
                    "rank": 0.95,
                    "centrality": 0.88,
                    "importance_score": 0.92,
                    "connections": 15,
                    "incoming_refs": 8,
                    "outgoing_refs": 7
                },
                {
                    "path": "src/utils/helpers.ts", 
                    "rank": 0.73,
                    "centrality": 0.65,
                    "importance_score": 0.69,
                    "connections": 10,
                    "incoming_refs": 6,
                    "outgoing_refs": 4
                },
                {
                    "path": "tests/main.test.ts",
                    "rank": 0.42,
                    "centrality": 0.38,
                    "importance_score": 0.40,
                    "connections": 5,
                    "incoming_refs": 0,
                    "outgoing_refs": 5
                }
            ],
            "edges": [
                {
                    "source": "src/main.ts",
                    "target": "src/utils/helpers.ts",
                    "weight": 0.85,
                    "relationship_type": "import",
                    "line_number": 3,
                    "import_specifiers": ["formatDate", "validateInput"]
                },
                {
                    "source": "tests/main.test.ts",
                    "target": "src/main.ts",
                    "weight": 0.90,
                    "relationship_type": "test_import",
                    "line_number": 1,
                    "import_specifiers": ["main", "initialize"]
                },
                {
                    "source": "src/main.ts",
                    "target": "src/types.ts",
                    "weight": 0.65,
                    "relationship_type": "type_import",
                    "line_number": 2,
                    "import_specifiers": ["UserConfig", "AppState"]
                }
            ],
            "metadata": {
                "analysis_timestamp": "2024-01-15T10:30:00Z",
                "total_files_analyzed": 25,
                "total_dependencies": 18,
                "analysis_duration_ms": 1250,
                "repository_complexity_score": 0.67,
                "language_distribution": {
                    "typescript": 0.8,
                    "javascript": 0.15,
                    "json": 0.05
                }
            }
        }
        
        # Convert to our schema
        file_ranks = [
            FileRank(
                path=fr["path"], 
                rank=fr["rank"],
                centrality=fr.get("centrality", 0.5),
                dependencies=[]
            )
            for fr in repomapper_data["file_ranks"]
        ]
        
        edges = [
            DependencyEdge(
                source=edge["source"],
                target=edge["target"], 
                weight=edge["weight"],
                edge_type=edge.get("relationship_type", "import")
            )
            for edge in repomapper_data["edges"]
        ]
        
        repomap = RepoMap(
            file_ranks=file_ranks, 
            edges=edges,
            total_files=len(file_ranks)
        )
        
        # Verify structure
        assert len(repomap.file_ranks) == 3
        assert len(repomap.edges) == 3
        
        # Verify data integrity
        main_rank = next(fr for fr in repomap.file_ranks if fr.path == "src/main.ts")
        assert main_rank.rank == 0.95
        
        import_edge = next(e for e in repomap.edges if e.source == "src/main.ts")
        assert import_edge.target == "src/utils/helpers.ts"
        assert import_edge.weight == 0.85
        
        # Test serialization round-trip
        json_data = repomap.model_dump()
        restored = RepoMap.model_validate(json_data)
        
        assert len(restored.file_ranks) == len(repomap.file_ranks)
        assert len(restored.edges) == len(repomap.edges)
    
    def test_serena_real_world_format(self):
        """Test Serena symbol analysis data format."""
        # Real-world Serena output structure
        serena_data = {
            "symbols": [
                {
                    "type": "definition",
                    "kind": "class",
                    "name": "UserManager",
                    "path": "src/user/manager.ts",
                    "start_line": 15,
                    "end_line": 45,
                    "start_col": 0,
                    "end_col": 1,
                    "signature": "export class UserManager implements IUserManager",
                    "modifiers": ["export"],
                    "accessibility": "public",
                    "documentation": "Manages user authentication and profile operations"
                },
                {
                    "type": "definition",
                    "kind": "method",
                    "name": "authenticate", 
                    "path": "src/user/manager.ts",
                    "start_line": 20,
                    "end_line": 28,
                    "start_col": 2,
                    "end_col": 3,
                    "signature": "async authenticate(email: string, password: string): Promise<User | null>",
                    "modifiers": ["async"],
                    "accessibility": "public",
                    "parent_symbol": "UserManager",
                    "parameters": [
                        {"name": "email", "type": "string"},
                        {"name": "password", "type": "string"}
                    ],
                    "return_type": "Promise<User | null>"
                },
                {
                    "type": "reference",
                    "kind": "identifier",
                    "name": "UserManager",
                    "path": "src/main.ts",
                    "start_line": 8,
                    "end_line": 8,
                    "start_col": 15,
                    "end_col": 26,
                    "context": "const userMgr = new UserManager();"
                },
                {
                    "type": "call",
                    "kind": "method_call",
                    "caller_name": "main",
                    "callee_name": "authenticate",
                    "path": "src/main.ts",
                    "start_line": 12,
                    "end_line": 12,
                    "start_col": 20,
                    "end_col": 32,
                    "call_context": "await userMgr.authenticate(email, pass)",
                    "arguments": ["email", "pass"]
                }
            ],
            "imports": [
                {
                    "path": "src/main.ts",
                    "imported_from": "src/user/manager.ts",
                    "imported_symbols": ["UserManager"],
                    "import_type": "named",
                    "line_number": 3
                }
            ],
            "exports": [
                {
                    "path": "src/user/manager.ts",
                    "exported_symbol": "UserManager",
                    "export_type": "named",
                    "line_number": 15
                }
            ],
            "metadata": {
                "typescript_version": "5.3.0",
                "analysis_timestamp": "2024-01-15T10:31:00Z",
                "total_symbols": 125,
                "files_analyzed": 15,
                "analysis_duration_ms": 850,
                "symbol_distribution": {
                    "classes": 8,
                    "methods": 45,
                    "functions": 32,
                    "interfaces": 12,
                    "types": 28
                }
            }
        }
        
        # Convert to our schema
        entries = []
        
        for symbol in serena_data["symbols"]:
            if symbol["type"] == "definition":
                entry = SymbolEntry(
                    type=SymbolType.DEF,
                    path=symbol["path"],
                    span=(symbol["start_line"], symbol["end_line"]),
                    symbol=symbol["name"],
                    sig=symbol["signature"]
                )
                entries.append(entry)
            
            elif symbol["type"] == "reference":
                # Ensure span is valid (start < end)
                start_line = symbol["start_line"]
                end_line = max(symbol["end_line"], start_line + 1)
                entry = SymbolEntry(
                    type=SymbolType.REF,
                    path=symbol["path"],
                    span=(start_line, end_line),
                    symbol=symbol["name"]
                )
                entries.append(entry)
            
            elif symbol["type"] == "call":
                # Ensure span is valid (start < end)
                start_line = symbol["start_line"]
                end_line = max(symbol["end_line"], start_line + 1)
                entry = SymbolEntry(
                    type=SymbolType.CALL,
                    path=symbol["path"],
                    span=(start_line, end_line),
                    caller=symbol["caller_name"],
                    callee=symbol["callee_name"]
                )
                entries.append(entry)
        
        # Calculate required fields from the test data
        unique_files = set(entry.path for entry in entries)
        graph = SerenaGraph(
            entries=entries,
            file_count=len(unique_files),
            symbol_count=len(entries)
        )
        
        # Verify structure
        assert len(graph.entries) == 4
        
        # Verify definitions
        defs = [e for e in graph.entries if e.type == SymbolType.DEF]
        assert len(defs) == 2
        
        user_manager_def = next(e for e in defs if e.symbol == "UserManager")
        assert user_manager_def.path == "src/user/manager.ts"
        assert "class UserManager" in user_manager_def.sig
        
        auth_def = next(e for e in defs if e.symbol == "authenticate")
        assert auth_def.path == "src/user/manager.ts"
        assert "async authenticate" in auth_def.sig
        
        # Verify references
        refs = [e for e in graph.entries if e.type == SymbolType.REF and e.symbol == "UserManager"]
        assert len(refs) == 1
        assert refs[0].path == "src/main.ts"
        
        # Test serialization round-trip
        json_data = graph.model_dump()
        restored = SerenaGraph.model_validate(json_data)
        
        assert len(restored.entries) == len(graph.entries)
    
    def test_leann_real_world_format(self):
        """Test LEANN embedding data format."""
        # Real-world LEANN output structure
        leann_data = {
            "embeddings": [
                {
                    "chunk_id": "chunk_001",
                    "file_path": "src/auth/login.ts",
                    "content": "export async function login(email: string, password: string): Promise<LoginResult> {\n    const user = await validateCredentials(email, password);\n    if (!user) {\n        throw new Error('Invalid credentials');\n    }",
                    "start_line": 15,
                    "end_line": 19,
                    "start_char": 450,
                    "end_char": 650,
                    "embedding_vector": [
                        0.0234, -0.1567, 0.2341, 0.0892, -0.0456,
                        0.1789, 0.0123, -0.2134, 0.1456, 0.0789,
                        # ... (would be 384 or 768 dimensions in reality)
                    ],
                    "vector_norm": 1.0,
                    "token_count": 45,
                    "function_name": "login",
                    "chunk_type": "function_definition"
                },
                {
                    "chunk_id": "chunk_002", 
                    "file_path": "src/auth/login.ts",
                    "content": "    return {\n        user,\n        token: generateJWT(user.id),\n        expiresAt: new Date(Date.now() + JWT_EXPIRY)\n    };\n}",
                    "start_line": 20,
                    "end_line": 25,
                    "start_char": 651,
                    "end_char": 780,
                    "embedding_vector": [
                        0.0456, -0.1234, 0.1892, 0.0567, -0.0789,
                        0.1345, 0.0234, -0.1789, 0.1123, 0.0892,
                        # ... (continuation of vector)
                    ],
                    "vector_norm": 1.0,
                    "token_count": 25,
                    "function_name": "login",
                    "chunk_type": "function_return"
                },
                {
                    "chunk_id": "chunk_003",
                    "file_path": "src/utils/jwt.ts", 
                    "content": "export function generateJWT(userId: string): string {\n    const payload = { userId, iat: Date.now() };\n    return jwt.sign(payload, JWT_SECRET, { expiresIn: '24h' });\n}",
                    "start_line": 8,
                    "end_line": 11,
                    "start_char": 200,
                    "end_char": 350,
                    "embedding_vector": [
                        0.0789, -0.0987, 0.1567, 0.0234, -0.1123,
                        0.0892, 0.0456, -0.1345, 0.0678, 0.1234,
                        # ... (continuation of vector)
                    ],
                    "vector_norm": 1.0,
                    "token_count": 32,
                    "function_name": "generateJWT",
                    "chunk_type": "function_definition"
                }
            ],
            "metadata": {
                "model_name": "leann-cpu-v2",
                "model_version": "2.1.0",
                "embedding_dimensions": 384,
                "total_chunks": 3,
                "total_tokens": 102,
                "analysis_timestamp": "2024-01-15T10:32:00Z",
                "processing_time_ms": 1850,
                "chunk_strategy": "function_aware",
                "overlap_lines": 2,
                "max_chunk_size": 512
            }
        }
        
        # Convert to our schema
        chunks = []
        
        for emb in leann_data["embeddings"]:
            chunk = VectorChunk(
                chunk_id=emb["chunk_id"],
                path=emb["file_path"],
                span=(emb["start_line"], emb["end_line"]),
                content=emb["content"],
                embedding=emb["embedding_vector"][:10],  # Truncate for test
                token_count=emb["token_count"]
            )
            chunks.append(chunk)
        
        vector_index = VectorIndex(
            chunks=chunks,
            dimension=leann_data["metadata"]["embedding_dimensions"],
            total_tokens=leann_data["metadata"]["total_tokens"],
            model_name=leann_data["metadata"]["model_name"]
        )
        
        # Verify structure
        assert len(vector_index.chunks) == 3
        
        # Verify chunk data
        login_chunks = [c for c in vector_index.chunks if "login.ts" in c.path]
        assert len(login_chunks) == 2
        
        first_chunk = login_chunks[0]
        assert "export async function login" in first_chunk.content
        assert len(first_chunk.embedding) == 10
        assert first_chunk.span == (15, 19)
        
        jwt_chunks = [c for c in vector_index.chunks if "jwt.ts" in c.path]
        assert len(jwt_chunks) == 1
        assert "generateJWT" in jwt_chunks[0].content
        
        # Test serialization round-trip
        json_data = vector_index.model_dump()
        restored = VectorIndex.model_validate(json_data)
        
        assert len(restored.chunks) == len(vector_index.chunks)
        
        # Verify embedding data integrity
        for orig, rest in zip(vector_index.chunks, restored.chunks):
            assert orig.path == rest.path
            assert orig.span == rest.span
            assert orig.content == rest.content
            assert orig.embedding == rest.embedding
    
    def test_search_response_format(self):
        """Test search response data format."""
        # Create comprehensive search response
        scores = SearchScores(
            vector=0.85,
            symbol=0.92,
            graph=0.67
        )
        
        content = CodeSnippet(
            path="src/auth/middleware.ts",
            span=(25, 35),
            hash="sha256:abc123def456",
            pre="// Authentication middleware\nimport { Request, Response } from 'express';",
            text="export function authMiddleware(req: Request, res: Response, next: NextFunction) {\n    const token = req.headers.authorization?.split(' ')[1];\n    if (!token) {\n        return res.status(401).json({ error: 'No token provided' });\n    }",
            post="    try {\n        const decoded = jwt.verify(token, JWT_SECRET);\n        req.user = decoded;\n        next();\n    } catch (error) {",
            line_start=23,
            line_end=37
        )
        
        citation = Citation(
            repo_root="/home/user/project",
            rev="main",
            path="src/auth/middleware.ts",
            span=(25, 35),
            content_sha="sha256:abc123def456"
        )
        
        search_result = SearchResult(
            path="src/auth/middleware.ts",
            span=(25, 35),
            score=0.88,
            scores=scores,
            content=content,
            citation=citation
        )
        
        # Verify structure
        assert search_result.path == content.path == citation.path
        assert search_result.span == content.span == citation.span
        assert search_result.score == 0.88
        assert search_result.scores.vector == 0.85
        assert search_result.scores.symbol == 0.92
        assert search_result.scores.graph == 0.67
        
        # Test serialization
        json_data = search_result.model_dump()
        restored = SearchResult.model_validate(json_data)
        
        assert restored.path == search_result.path
        assert restored.score == search_result.score
        assert restored.content.text == search_result.content.text
        assert restored.citation.content_sha == search_result.citation.content_sha
    
    def test_ask_response_format(self):
        """Test ask response data format."""
        # Create citations
        citations = [
            Citation(
                repo_root="/project",
                rev="main", 
                path="src/user.ts",
                span=(10, 15),
                content_sha="hash1"
            ),
            Citation(
                repo_root="/project",
                rev="main",
                path="src/auth.ts", 
                span=(20, 25),
                content_sha="hash2"
            )
        ]
        
        ask_response = AskResponse(
            question="How does user authentication work in this codebase?",
            answer="User authentication in this codebase follows a JWT-based approach. The `UserManager` class in `src/user/manager.ts` handles the core authentication logic through the `authenticate` method, which validates email and password credentials. Upon successful authentication, the system generates a JWT token using the `generateJWT` function in `src/utils/jwt.ts`. The authentication middleware in `src/auth/middleware.ts` validates incoming requests by checking for the presence and validity of JWT tokens in the Authorization header.",
            citations=citations,
            execution_time_ms=1250.5,
            index_id="01HQ123456789ABCDEF"
        )
        
        # Verify structure
        assert ask_response.question.startswith("How does user authentication")
        assert "JWT" in ask_response.answer
        assert "UserManager" in ask_response.answer
        assert len(ask_response.citations) == 2
        assert ask_response.execution_time_ms > 0
        assert ask_response.index_id.startswith("01HQ")
        
        # Test serialization
        json_data = ask_response.model_dump()
        restored = AskResponse.model_validate(json_data)
        
        assert restored.question == ask_response.question
        assert restored.answer == ask_response.answer
        assert len(restored.citations) == len(ask_response.citations)
        assert restored.execution_time_ms == ask_response.execution_time_ms


class TestIndexManifestFormat:
    """Test index manifest data structure and serialization."""
    
    def test_complete_manifest_structure(self, temp_dir):
        """Test complete manifest with all fields populated."""
        # Create comprehensive repository info
        repo_info = RepoInfo(
            root=str(temp_dir),
            rev="feature/authentication-v2",
            worktree_dirty=True
        )
        
        # Create detailed configuration
        config = IndexConfig(
            languages=["ts", "tsx", "js", "jsx", "vue", "py"],
            excludes=[
                "node_modules/", "dist/", "build/", ".git/", 
                "__pycache__/", "*.pyc", ".venv/", "coverage/"
            ],
            context_lines=5,
            max_files_to_embed=2000
        )
        
        # Create manifest
        manifest = IndexManifest(
            repo=repo_info,
            config=config
        )
        
        # Update counts to simulate completed indexing
        manifest.counts.files_total = 156
        manifest.counts.files_indexed = 142
        manifest.counts.symbols_defs = 1247
        manifest.counts.symbols_refs = 2156
        manifest.counts.vectors = 892
        manifest.counts.chunks = 892
        
        # Update versions
        manifest.versions.repomapper = "1.2.3"
        manifest.versions.serena = "2.1.0" 
        manifest.versions.leann = "3.0.1"
        
        # Verify structure
        assert manifest.repo.root == str(temp_dir)
        assert manifest.repo.rev == "feature/authentication-v2"
        assert manifest.repo.worktree_dirty is True
        
        assert len(manifest.config.languages) == 6
        assert "tsx" in manifest.config.languages
        assert "vue" in manifest.config.languages
        
        assert manifest.counts.files_total == 156
        assert manifest.counts.files_indexed == 142
        assert manifest.counts.symbols_defs == 1247
        assert manifest.counts.symbols_refs == 2156
        assert manifest.counts.vectors == 892
        assert manifest.counts.chunks == 892
        
        # Test serialization with all fields
        json_data = manifest.model_dump()
        
        # Verify JSON structure
        assert "index_id" in json_data
        assert "repo" in json_data
        assert "config" in json_data
        assert "counts" in json_data
        assert "versions" in json_data
        assert "paths" in json_data
        
        # Verify nested structures
        assert json_data["repo"]["worktree_dirty"] is True
        assert len(json_data["config"]["languages"]) == 6
        assert json_data["counts"]["symbols_defs"] == 1247
        assert json_data["versions"]["serena"] == "2.1.0"
        
        # Test deserialization
        restored = IndexManifest.model_validate(json_data)
        
        assert restored.index_id == manifest.index_id
        assert restored.repo.root == manifest.repo.root
        assert restored.repo.rev == manifest.repo.rev
        assert restored.repo.worktree_dirty == manifest.repo.worktree_dirty
        assert restored.config.languages == manifest.config.languages
        assert restored.counts.files_total == manifest.counts.files_total
        assert restored.versions.serena == manifest.versions.serena
    
    def test_manifest_file_persistence(self, temp_dir):
        """Test manifest file save/load operations."""
        # Create manifest
        repo_info = RepoInfo(
            root=str(temp_dir / "test_repo"),
            rev="main",
            worktree_dirty=False
        )
        
        manifest = IndexManifest(
            repo=repo_info,
            config=IndexConfig()
        )
        
        # Save to file
        manifest_file = temp_dir / "manifest.json"
        manifest_file.write_text(manifest.model_dump_json(indent=2))
        
        # Load from file
        loaded_data = json.loads(manifest_file.read_text())
        restored_manifest = IndexManifest.model_validate(loaded_data)
        
        # Verify persistence
        assert restored_manifest.index_id == manifest.index_id
        assert restored_manifest.repo.root == manifest.repo.root
        assert restored_manifest.config.languages == manifest.config.languages
        
        # Verify file format is human-readable
        file_content = manifest_file.read_text()
        assert '"index_id"' in file_content
        assert '"repo"' in file_content
        assert '"config"' in file_content
        # Should be formatted with indentation
        assert "\n  " in file_content


class TestDataValidation:
    """Test data validation and error handling."""
    
    def test_invalid_data_rejection(self):
        """Test that invalid data is properly rejected."""
        # Test invalid SymbolType
        with pytest.raises(ValueError):
            SymbolEntry(
                type="invalid_type",  # Should be SymbolType enum
                path="test.ts",
                span=(1, 5),
                symbol="test"
            )
        
        # Test invalid span (end before start)
        with pytest.raises(ValueError):
            SymbolEntry(
                type=SymbolType.DEF,
                path="test.ts",
                span=(10, 5),  # Invalid: end < start
                symbol="test"
            )
    
    def test_required_field_validation(self):
        """Test that required fields are enforced."""
        # Test missing required path
        with pytest.raises(ValueError):
            SymbolEntry(
                type=SymbolType.DEF,
                # path is required but missing
                span=(1, 5),
                symbol="test"
            )
        
        # Test missing required repo in manifest
        with pytest.raises(ValueError):
            IndexManifest(
                # repo is required but missing
                config=IndexConfig()
            )
    
    def test_data_type_coercion(self):
        """Test that data types are properly coerced."""
        # Test span tuple coercion
        chunk = VectorChunk(
            path="test.ts",
            span=[5, 10],  # List should be coerced to tuple
            content="test",
            embedding=[0.1, 0.2, 0.3]
        )
        
        assert chunk.span == (5, 10)
        assert isinstance(chunk.span, tuple)
        
        # Test embedding list validation
        assert isinstance(chunk.embedding, list)
        assert all(isinstance(x, float) for x in chunk.embedding)