#!/usr/bin/env python3
"""
Ollama Setup Validation Script

Quick validation script to check if Ollama is properly set up for
Mimir integration testing. Runs pre-flight checks and reports status.

Usage:
    python scripts/validate_ollama_setup.py
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Add project root to path  
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Run: pip install httpx")
    sys.exit(1)


class OllamaValidator:
    """Validates Ollama setup for Mimir integration."""
    
    def __init__(self):
        self.base_url = "http://localhost:11434"
        self.required_models = ["nomic-embed-text"]
        self.optional_models = ["llama3.1"]
        
    async def check_server_running(self) -> Tuple[bool, str]:
        """Check if Ollama server is running."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/version", timeout=5.0)
                if response.status_code == 200:
                    version_info = response.json()
                    return True, f"Ollama {version_info.get('version', 'unknown')}"
                else:
                    return False, f"HTTP {response.status_code}"
        except httpx.ConnectError:
            return False, "Connection refused (server not running?)"
        except Exception as e:
            return False, str(e)
    
    async def list_models(self) -> Tuple[bool, list]:
        """List available models."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags", timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    models = [model.get("name", "").split(":")[0] for model in data.get("models", [])]
                    return True, models
                else:
                    return False, []
        except Exception as e:
            return False, []
    
    async def test_embedding(self, model: str) -> Tuple[bool, str, int]:
        """Test embedding generation."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": model, "prompt": "test embedding"},
                    timeout=30.0
                )
                if response.status_code == 200:
                    data = response.json()
                    embedding = data.get("embedding", [])
                    return True, "Success", len(embedding)
                else:
                    return False, f"HTTP {response.status_code}", 0
        except Exception as e:
            return False, str(e), 0
    
    async def test_chat(self, model: str) -> Tuple[bool, str, str]:
        """Test chat generation."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/generate",
                    json={
                        "model": model, 
                        "prompt": "What is machine learning?",
                        "stream": False
                    },
                    timeout=60.0
                )
                if response.status_code == 200:
                    data = response.json()
                    response_text = data.get("response", "")[:100]
                    return True, "Success", response_text
                else:
                    return False, f"HTTP {response.status_code}", ""
        except Exception as e:
            return False, str(e), ""
    
    def check_python_deps(self) -> Dict[str, bool]:
        """Check required Python dependencies."""
        deps = {}
        
        try:
            import httpx
            deps["httpx"] = True
        except ImportError:
            deps["httpx"] = False
            
        try:
            import pytest
            deps["pytest"] = True
        except ImportError:
            deps["pytest"] = False
            
        try:
            from src.repoindex.mcp.server import MCPServer
            deps["mimir-mcp"] = True
        except ImportError:
            deps["mimir-mcp"] = False
            
        return deps
    
    def check_ollama_cli(self) -> Tuple[bool, str]:
        """Check if Ollama CLI is available."""
        try:
            result = subprocess.run(
                ["ollama", "version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                return False, result.stderr.strip()
        except FileNotFoundError:
            return False, "Ollama CLI not found in PATH"
        except subprocess.TimeoutExpired:
            return False, "Command timed out"
        except Exception as e:
            return False, str(e)


async def main():
    """Run validation checks and report results."""
    print("🔍 Validating Ollama Setup for Mimir Integration")
    print("=" * 60)
    
    validator = OllamaValidator()
    all_good = True
    
    # Check 1: Ollama CLI
    print("\n📋 1. Ollama CLI Check")
    cli_ok, cli_info = validator.check_ollama_cli()
    if cli_ok:
        print(f"   ✅ Ollama CLI available: {cli_info}")
    else:
        print(f"   ❌ Ollama CLI: {cli_info}")
        print("   💡 Install: curl -fsSL https://ollama.ai/install.sh | sh")
        all_good = False
    
    # Check 2: Server running
    print("\n🌐 2. Ollama Server Check")
    server_ok, server_info = await validator.check_server_running()
    if server_ok:
        print(f"   ✅ Server running: {server_info}")
    else:
        print(f"   ❌ Server not accessible: {server_info}")
        print("   💡 Start server: ollama serve")
        all_good = False
        # Can't continue without server
        print("\n⚠️  Cannot continue validation without running server")
        print_summary(False)
        return
    
    # Check 3: Available models
    print("\n🤖 3. Model Availability")
    models_ok, models = await validator.list_models()
    if models_ok:
        print(f"   📦 Available models: {', '.join(models) if models else 'none'}")
        
        # Check required models
        for model in validator.required_models:
            if model in models:
                print(f"   ✅ Required model '{model}' available")
            else:
                print(f"   ❌ Required model '{model}' missing")
                print(f"   💡 Install: ollama pull {model}")
                all_good = False
        
        # Check optional models
        for model in validator.optional_models:
            if model in models:
                print(f"   ✅ Optional model '{model}' available")
            else:
                print(f"   ⚠️  Optional model '{model}' missing (tests will be limited)")
                print(f"   💡 Install: ollama pull {model}")
    else:
        print("   ❌ Could not list models")
        all_good = False
    
    # Check 4: Embedding functionality
    print("\n🔢 4. Embedding Test")
    if models_ok and any(model in models for model in validator.required_models):
        embedding_model = next(model for model in validator.required_models if model in models)
        embed_ok, embed_msg, embed_dims = await validator.test_embedding(embedding_model)
        if embed_ok:
            print(f"   ✅ Embedding test passed: {embed_dims} dimensions")
        else:
            print(f"   ❌ Embedding test failed: {embed_msg}")
            all_good = False
    else:
        print("   ⚠️  Skipped (no embedding model available)")
    
    # Check 5: Chat functionality (optional)
    print("\n💬 5. Chat Test (Optional)")
    if models_ok and any(model in models for model in validator.optional_models):
        chat_model = next(model for model in validator.optional_models if model in models)
        chat_ok, chat_msg, chat_response = await validator.test_chat(chat_model)
        if chat_ok:
            print(f"   ✅ Chat test passed")
            print(f"   📝 Response preview: {chat_response}...")
        else:
            print(f"   ❌ Chat test failed: {chat_msg}")
    else:
        print("   ⚠️  Skipped (no chat model available)")
    
    # Check 6: Python dependencies
    print("\n🐍 6. Python Dependencies")
    deps = validator.check_python_deps()
    for dep, available in deps.items():
        if available:
            print(f"   ✅ {dep} available")
        else:
            print(f"   ❌ {dep} missing")
            if dep == "httpx":
                print("   💡 Install: pip install httpx")
            elif dep == "pytest":
                print("   💡 Install: pip install pytest")
            elif dep == "mimir-mcp":
                print("   💡 Make sure you're in the Mimir project directory")
            all_good = False
    
    # Final summary
    print_summary(all_good)


def print_summary(success: bool):
    """Print final validation summary."""
    print("\n" + "=" * 60)
    if success:
        print("🎉 VALIDATION PASSED")
        print("\n✅ Your Ollama setup is ready for Mimir integration testing!")
        print("\nNext steps:")
        print("   python scripts/test_ollama_integration.py --quick")
    else:
        print("❌ VALIDATION FAILED")
        print("\n⚠️  Your Ollama setup needs attention before running tests.")
        print("\nCommon fixes:")
        print("   1. Start Ollama: ollama serve")
        print("   2. Install models: ollama pull nomic-embed-text")  
        print("   3. Install deps: pip install httpx pytest")
        print("\nThen re-run this validation script.")
    print("=" * 60)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Validation script crashed: {e}")
        sys.exit(1)