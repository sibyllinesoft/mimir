#!/usr/bin/env python3
"""
Direct test of Gemini integration without full pipeline.
This tests the GeminiAdapter directly to verify it works with your API key.
"""

import asyncio
import os
from src.repoindex.pipeline.gemini import GeminiAdapter
from src.repoindex.data.schemas import CodeSnippet


async def test_gemini_direct():
    """Test Gemini adapter directly with sample code."""
    
    print("🤖 Testing Gemini Integration Directly")
    print("=" * 50)
    
    # Check if API key is available
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No Gemini API key found in environment")
        return False
        
    print(f"✅ Found API key: {api_key[:8]}...")
    
    try:
        # Initialize Gemini adapter
        print("🔧 Initializing Gemini adapter...")
        gemini = GeminiAdapter()
        print(f"✅ Initialized with model: {gemini.model_name}")
        
        # Create sample code snippets (simulating what would come from pipeline)
        sample_snippets = [
            CodeSnippet(
                path="auth.py",
                line_start=61,
                line_end=69,
                text='''def set_password(self, password: str) -> None:
    """Set user password with secure hashing."""
    salt = secrets.token_hex(16)
    self.password_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000  # 100k iterations
    ).hex() + salt''',
                pre="    def __init__(self, username: str, email: str):\n        self.username = username",
                post="    def check_password(self, password: str) -> bool:"
            ),
            CodeSnippet(
                path="auth.py", 
                line_start=106,
                line_end=124,
                text='''def authenticate(self, username: str, password: str) -> Optional[str]:
    """Authenticate user and return session token."""
    user = self.users.get(username)
    if not user or not user.is_active:
        self._record_failed_attempt(username)
        return None
    
    if user.check_password(password):
        user.last_login = datetime.now()
        session_token = secrets.token_urlsafe(32)
        self.sessions[session_token] = {
            'username': username,
            'created_at': datetime.now(),
            'last_accessed': datetime.now()
        }
        return session_token
    else:
        self._record_failed_attempt(username)
        return None''',
                pre="    def register_user(self, username: str, email: str, password: str) -> bool:",
                post="    def validate_session(self, session_token: str) -> Optional[User]:"
            )
        ]
        
        # Test questions
        test_questions = [
            "How does password hashing work in this authentication system?",
            "What happens during user authentication?",
            "What security measures are implemented?"
        ]
        
        print(f"\n🧪 Testing {len(test_questions)} questions with Gemini...")
        
        for i, question in enumerate(test_questions, 1):
            print(f"\n📝 Question {i}: {question}")
            print("-" * 50)
            
            try:
                # Call Gemini to synthesize an answer
                answer = await gemini.synthesize_answer(
                    question=question,
                    evidence_snippets=sample_snippets,
                    intents=[{"intent_type": "definition", "targets": ["authentication"]}],
                    repo_info={"root": "/tmp/test", "rev": "main"}
                )
                
                print(f"🧠 Gemini Response:")
                print(answer)
                print()
                
            except Exception as e:
                print(f"❌ Error: {e}")
                return False
        
        # Test explain_code_snippet
        print("\n🔍 Testing code snippet explanation...")
        explanation = await gemini.explain_code_snippet(
            sample_snippets[0],
            context="This is part of a user authentication system"
        )
        print("📖 Code Explanation:")
        print(explanation)
        
        print("\n🎉 All Gemini tests passed successfully!")
        print("The integration is working with your API key.")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing or using Gemini: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gemini_direct())
    if not success:
        exit(1)