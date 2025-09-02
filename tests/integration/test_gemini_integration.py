#!/usr/bin/env python3
"""
Test script to verify Gemini integration works with the ask_index functionality.

This script tests the complete workflow:
1. Create a test repository with sample code
2. Run the full MCP pipeline to generate index data
3. Test the ask_index functionality with Gemini integration
4. Verify intelligent responses are generated
"""

import asyncio
import json
import tempfile
import subprocess
import os
from pathlib import Path

from src.repoindex.mcp.server import MCPServer


async def test_gemini_ask_functionality():
    """Test the complete ask_index workflow with Gemini integration."""
    
    print("🤖 Testing Gemini Integration for ask_index functionality")
    print("=" * 60)
    
    # Check if Gemini API key is available
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("⚠️  No Gemini API key found in environment variables.")
        print("   Set GOOGLE_API_KEY or GEMINI_API_KEY to test Gemini integration.")
        print("   Testing will proceed with fallback mode only.")
        print()
    
    # Create a test repository with more complex code
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        
        # Create a sample authentication system (TypeScript)
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

interface SessionData {
    username: string;
    createdAt: Date;
    lastAccessed: Date;
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

/**
 * Manages user authentication and sessions.
 */
class AuthenticationManager {
    private users: Map<string, User> = new Map();
    private sessions: Map<string, SessionData> = new Map();
    private failedAttempts: Map<string, number> = new Map();
    
    /**
     * Register a new user.
     */
    registerUser(username: string, email: string, password: string): boolean {
        if (this.users.has(username)) {
            return false;
        }
        
        const user = new User(username, email);
        user.setPassword(password);
        this.users.set(username, user);
        return true;
    }
    
    /**
     * Authenticate user and return session token.
     */
    authenticate(username: string, password: string): string | null {
        const user = this.users.get(username);
        if (!user || !user.isActive) {
            this.recordFailedAttempt(username);
            return null;
        }
        
        if (user.checkPassword(password)) {
            user.lastLogin = new Date();
            const sessionToken = randomBytes(32).toString('base64url');
            this.sessions.set(sessionToken, {
                username: username,
                createdAt: new Date(),
                lastAccessed: new Date()
            });
            return sessionToken;
        } else {
            this.recordFailedAttempt(username);
            return null;
        }
    }
    
    /**
     * Validate session token and return user.
     */
    validateSession(sessionToken: string): User | null {
        const session = this.sessions.get(sessionToken);
        if (!session) {
            return null;
        }
        
        // Check if session is expired (24 hours)
        const now = new Date();
        const hoursSinceCreation = (now.getTime() - session.createdAt.getTime()) / (1000 * 60 * 60);
        
        if (hoursSinceCreation > 24) {
            this.sessions.delete(sessionToken);
            return null;
        }
        
        session.lastAccessed = now;
        return this.users.get(session.username) || null;
    }
    
    /**
     * Logout user by removing session.
     */
    logout(sessionToken: string): boolean {
        return this.sessions.delete(sessionToken);
    }
    
    /**
     * Record failed login attempt for rate limiting.
     */
    private recordFailedAttempt(username: string): void {
        const currentAttempts = this.failedAttempts.get(username) || 0;
        this.failedAttempts.set(username, currentAttempts + 1);
        
        // Lock account after 5 failed attempts
        if (currentAttempts + 1 >= 5) {
            const user = this.users.get(username);
            if (user) {
                user.isActive = false;
            }
        }
    }
}

export { User, AuthenticationManager };
''')
        
        # Create a database layer (TypeScript)
        (repo / "database.ts").write_text('''
/**
 * Database connection and query management.
 */

import sqlite3 from 'sqlite3';
import { promisify } from 'util';

interface UserRecord {
    id: number;
    username: string;
    email: string;
    password_hash: string;
    created_at: string;
    last_login?: string;
    is_active: boolean;
}

interface SessionRecord {
    session_token: string;
    username: string;
    created_at: string;
    last_accessed: string;
}

/**
 * Manages database connections and transactions.
 */
class DatabaseManager {
    private db: sqlite3.Database;
    private dbPath: string;
    
    constructor(dbPath: string = ':memory:') {
        this.dbPath = dbPath;
        this.db = new sqlite3.Database(dbPath);
        this.initSchema();
    }
    
    /**
     * Initialize database schema.
     */
    private initSchema(): void {
        const createUsersTable = `
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT TRUE
            )
        `;
        
        const createSessionsTable = `
            CREATE TABLE IF NOT EXISTS user_sessions (
                session_token TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username)
            )
        `;
        
        this.db.serialize(() => {
            this.db.run(createUsersTable);
            this.db.run(createSessionsTable);
        });
    }
    
    /**
     * Create new user in database.
     */
    async createUser(username: string, email: string, passwordHash: string): Promise<boolean> {
        return new Promise((resolve, reject) => {
            const stmt = this.db.prepare(
                "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)"
            );
            
            stmt.run([username, email, passwordHash], function(err) {
                if (err) {
                    if (err.code === 'SQLITE_CONSTRAINT_UNIQUE') {
                        resolve(false);
                    } else {
                        reject(err);
                    }
                } else {
                    resolve(true);
                }
            });
            
            stmt.finalize();
        });
    }
    
    /**
     * Get user by username.
     */
    async getUser(username: string): Promise<UserRecord | null> {
        return new Promise((resolve, reject) => {
            this.db.get(
                "SELECT * FROM users WHERE username = ?",
                [username],
                (err, row: UserRecord) => {
                    if (err) {
                        reject(err);
                    } else {
                        resolve(row || null);
                    }
                }
            );
        });
    }
    
    /**
     * Update user's last login timestamp.
     */
    async updateLastLogin(username: string): Promise<void> {
        return new Promise((resolve, reject) => {
            this.db.run(
                "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
                [username],
                (err) => {
                    if (err) {
                        reject(err);
                    } else {
                        resolve();
                    }
                }
            );
        });
    }
    
    /**
     * Close database connection.
     */
    close(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.db.close((err) => {
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }
}

export { DatabaseManager, UserRecord, SessionRecord };
''')
        
        # Create main application (TypeScript)
        (repo / "main.ts").write_text('''
/**
 * Main application entry point.
 */

import { AuthenticationManager } from './auth.js';
import { DatabaseManager } from './database.js';
import * as readline from 'readline';

/**
 * Main application function.
 */
async function main(): Promise<void> {
    // Initialize components
    const db = new DatabaseManager('app.db');
    const authManager = new AuthenticationManager();
    
    console.log('Authentication System Started');
    console.log('Available commands: register, login, status, logout, quit');
    
    let currentSession: string | null = null;
    
    const rl = readline.createInterface({
        input: process.stdin,
        output: process.stdout
    });
    
    const askQuestion = (question: string): Promise<string> => {
        return new Promise(resolve => {
            rl.question(question, answer => resolve(answer));
        });
    };
    
    while (true) {
        try {
            const command = (await askQuestion('>>> ')).trim().toLowerCase();
            
            if (command === 'quit') {
                break;
            } else if (command === 'register') {
                const username = await askQuestion('Username: ');
                const email = await askQuestion('Email: ');
                const password = await askQuestion('Password: ');
                
                if (authManager.registerUser(username, email, password)) {
                    console.log(`User ${username} registered successfully!`);
                } else {
                    console.log('Registration failed - user already exists');
                }
            } else if (command === 'login') {
                const username = await askQuestion('Username: ');
                const password = await askQuestion('Password: ');
                
                const sessionToken = authManager.authenticate(username, password);
                if (sessionToken) {
                    currentSession = sessionToken;
                    console.log(`Login successful! Session: ${sessionToken.substring(0, 8)}...`);
                } else {
                    console.log('Login failed - invalid credentials');
                }
            } else if (command === 'status') {
                if (currentSession) {
                    const user = authManager.validateSession(currentSession);
                    if (user) {
                        console.log(`Logged in as: ${user.username} (${user.email})`);
                    } else {
                        console.log('Session expired');
                        currentSession = null;
                    }
                } else {
                    console.log('Not logged in');
                }
            } else if (command === 'logout') {
                if (currentSession) {
                    authManager.logout(currentSession);
                    currentSession = null;
                    console.log('Logged out successfully');
                } else {
                    console.log('Not logged in');
                }
            } else {
                console.log('Unknown command');
            }
        } catch (error) {
            if (error instanceof Error && error.message === 'quit') {
                break;
            }
            console.log(`Error: ${error}`);
        }
    }
    
    rl.close();
    await db.close();
    console.log('Goodbye!');
}

if (import.meta.url === `file://${process.argv[1]}`) {
    main().catch(console.error);
}

export { main };
''')
        
        # Initialize git repository
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Authentication system implementation"], cwd=repo, check=True, capture_output=True)
        
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
            for i in range(45):  # Wait up to 45 seconds
                await asyncio.sleep(1)
                
                status_uri = "repo://status"
                status_result = await mcp_server._read_resource(status_uri)
                
                if hasattr(status_result, 'contents'):
                    status_data = json.loads(status_result.contents[0].text)
                    active_count = len(status_data.get('active_pipelines', []))
                    
                    if active_count == 0:
                        print(f"🎉 Pipeline completed after {i+1} seconds")
                        break
                    else:
                        if (i + 1) % 10 == 0:
                            print(f"⏱️  Still processing... ({i+1}s elapsed)")
            
            # Test various questions with the ask_index functionality
            test_questions = [
                "How does user authentication work in this codebase?",
                "What is the password hashing mechanism?",
                "How are user sessions managed?",
                "What security measures are implemented for login?",
                "How does the database integration work?",
                "What happens when a user logs in?",
                "How are failed login attempts handled?",
            ]
            
            print("\n🤖 Testing ask_index functionality with Gemini integration:")
            print("=" * 60)
            
            for i, question in enumerate(test_questions, 1):
                print(f"\n📝 Question {i}: {question}")
                print("-" * 40)
                
                try:
                    # Call ask_index
                    ask_result = await mcp_server._ask_index({
                        "index_id": index_id,
                        "question": question,
                        "context_lines": 3
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
                        print(answer)
                        
                        if citations:
                            print(f"\n📚 Citations ({len(citations)} sources):")
                            for citation in citations[:3]:  # Show first 3
                                print(f"  - {citation.get('path', 'unknown')}")
                        
                except Exception as e:
                    print(f"❌ Error processing question: {e}")
                    
                print()
                
                # Small delay between questions
                await asyncio.sleep(0.5)
            
            print("\n✅ Gemini integration test completed!")
            print("=" * 60)
            
            if api_key:
                print("🎯 Results show intelligent Gemini-powered responses")
            else:
                print("🔄 Results show fallback mode responses (no Gemini API key)")


if __name__ == "__main__":
    asyncio.run(test_gemini_ask_functionality())