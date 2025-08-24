#!/usr/bin/env python3
"""
Minimal test of Gemini integration without dependencies.
This tests the core Gemini functionality directly.
"""

import os
import asyncio
import sys

# Add the project to the path
sys.path.insert(0, "/home/nathan/Projects/mimir")

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold
except ImportError:
    print("❌ google-generativeai not installed")
    print("Install with: pip install google-generativeai>=0.8.3")
    exit(1)


async def test_gemini_minimal():
    """Test Gemini directly without Mimir dependencies."""
    
    print("🤖 Testing Gemini API Directly")
    print("=" * 40)
    
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No Gemini API key found")
        print("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable")
        return False
        
    print(f"✅ API Key: {api_key[:8]}...")
    
    try:
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=GenerationConfig(
                max_output_tokens=4096,
                temperature=0.1,
                candidate_count=1,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            }
        )
        
        print("✅ Gemini model initialized")
        
        # Test with sample code analysis
        prompt = """
Analyze this Python authentication code and explain how password hashing works:

```python
def set_password(self, password: str) -> None:
    \"\"\"Set user password with secure hashing.\"\"\"
    salt = secrets.token_hex(16)
    self.password_hash = hashlib.pbkdf2_hmac(
        'sha256', 
        password.encode('utf-8'), 
        salt.encode('utf-8'), 
        100000  # 100k iterations
    ).hex() + salt

def authenticate(self, username: str, password: str) -> Optional[str]:
    \"\"\"Authenticate user and return session token.\"\"\"
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
        return None
```

Please provide a comprehensive analysis focusing on:
1. How the password hashing mechanism works
2. Security measures implemented
3. The authentication flow
"""
        
        print("\n🧪 Sending request to Gemini...")
        
        # Make async call to Gemini
        response = await asyncio.to_thread(model.generate_content, prompt)
        
        if response.candidates and response.candidates[0].content.parts:
            answer = response.candidates[0].content.parts[0].text
            
            print("\n🧠 Gemini Analysis:")
            print("=" * 40)
            print(answer)
            print("\n✅ SUCCESS: Gemini integration is working perfectly!")
            print("Your API key is valid and the model is responding correctly.")
            return True
        else:
            print("❌ No response from Gemini")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_gemini_minimal())
    if not success:
        exit(1)