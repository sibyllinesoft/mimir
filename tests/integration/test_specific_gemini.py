#!/usr/bin/env python3
"""
Quick test of Gemini integration with a specific question about the test code.
"""

import asyncio
import json
import tempfile
import subprocess
import os
from pathlib import Path

from src.repoindex.mcp.server import MCPServer


async def test_specific_gemini_question():
    """Test Gemini with a question about TypeScript authentication code."""
    
    print("🤖 Testing Gemini with Specific TypeScript Code Question")
    print("=" * 60)
    
    # Create a test repository with TypeScript authentication code
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        
        # Create TypeScript authentication code (same as before)
        (repo / "auth.ts").write_text('''
/**
 * Authentication and user management system.
 */

import { randomBytes, pbkdf2Sync } from 'crypto';

interface UserData {
    username: string;
    email: string;
    passwordHash?: string;
    createdAt: Date;
    lastLogin?: Date;
    isActive: boolean;
    roles: string[];
}

/**
 * Represents a user in the system.
 */
class User implements UserData {
    public username: string;
    public email: string;
    public passwordHash?: string;
    public createdAt: Date;
    public lastLogin?: Date;
    public isActive: boolean;
    public roles: string[];
    
    constructor(username: string, email: string) {
        this.username = username;
        this.email = email;
        this.createdAt = new Date();
        this.isActive = true;
        this.roles = [];
    }
    
    /**
     * Set user password with secure hashing.
     */
    setPassword(password: string): void {
        const salt = randomBytes(16).toString('hex');
        const hash = pbkdf2Sync(password, salt, 100000, 64, 'sha256').toString('hex');
        this.passwordHash = hash + salt;
    }
    
    /**
     * Verify password against stored hash.
     */
    checkPassword(password: string): boolean {
        if (!this.passwordHash) {
            return false;
        }
        
        const storedHash = this.passwordHash.slice(0, -32);
        const salt = this.passwordHash.slice(-32);
        
        const computedHash = pbkdf2Sync(password, salt, 100000, 64, 'sha256').toString('hex');
        
        return computedHash === storedHash;
    }
}

export { User };
''')
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "TypeScript authentication system"], cwd=repo, check=True, capture_output=True)
        
        # Create MCP server and run pipeline
        with tempfile.TemporaryDirectory() as storage_dir:
            mcp_server = MCPServer(storage_dir=Path(storage_dir))
            
            print(f"📁 Testing with repository: {repo}")
            
            # Start indexing
            result = await mcp_server._ensure_repo_index({
                "path": str(repo),
                "index_opts": {
                    "languages": ["ts"],
                    "excludes": ["node_modules/", ".git/"]
                }
            })
            
            if result.isError:
                print(f"❌ Indexing failed: {result.content}")
                return
            
            response_data = json.loads(result.content[0].text)
            index_id = response_data["index_id"]
            print(f"✅ Pipeline started with index ID: {index_id}")
            
            # Wait for pipeline completion
            print("⏳ Waiting for pipeline completion...")
            import time
            for i in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                
                status_uri = "repo://status"
                status_result = await mcp_server._read_resource(status_uri)
                
                if hasattr(status_result, 'contents'):
                    status_data = json.loads(status_result.contents[0].text)
                    active_count = len(status_data.get('active_pipelines', []))
                    
                    if active_count == 0:
                        print(f"🎉 Pipeline completed after {i+1} seconds")
                        break
            
            # Test specific questions about the TypeScript code
            test_questions = [
                "What TypeScript classes are defined in this codebase?",
                "How does the User class work?",
                "What methods does the User class have?",
                "How does the setPassword method work?",
                "What interfaces are defined in this code?",
            ]
            
            print("\\n🤖 Testing specific questions about TypeScript code:")
            print("=" * 60)
            
            for i, question in enumerate(test_questions, 1):
                print(f"\\n📝 Question {i}: {question}")
                print("-" * 40)
                
                try:
                    # Call ask_index
                    ask_result = await mcp_server._ask_index({
                        "index_id": index_id,
                        "question": question,
                        "context_lines": 5
                    })
                    
                    if ask_result.isError:
                        print(f"❌ Ask failed: {ask_result.content}")
                    else:
                        response_text = ask_result.content[0].text
                        response_data = json.loads(response_text)
                        
                        answer = response_data.get("answer", "")
                        citations = response_data.get("citations", [])
                        exec_time = response_data.get("execution_time_ms", 0)
                        
                        print(f"🧠 Answer ({exec_time:.1f}ms):")
                        print(answer[:500] + "..." if len(answer) > 500 else answer)
                        
                        if citations:
                            print(f"\\n📚 Citations ({len(citations)} sources):")
                            for citation in citations[:2]:  # Show first 2
                                print(f"  - {citation.get('path', 'unknown')}")
                        
                except Exception as e:
                    print(f"❌ Error processing question: {e}")
                    
                print()
                
                # Small delay between questions
                await asyncio.sleep(0.5)
            
            print("\\n✅ Specific Gemini test completed!")


if __name__ == "__main__":
    asyncio.run(test_specific_gemini_question())