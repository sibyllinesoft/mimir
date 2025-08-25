#!/usr/bin/env python3
"""
Ollama Integration Test Runner

A standalone script to run and validate the Mimir MCP server integration with Ollama.
Handles Ollama setup, model downloading, and comprehensive testing.

Usage:
    python scripts/test_ollama_integration.py [options]

Options:
    --setup         Setup Ollama models before testing
    --quick         Run only quick tests (skip slow end-to-end tests)  
    --verbose       Enable verbose output
    --cleanup       Clean up test data after completion
    --models MODEL  Comma-separated list of models to test (default: nomic-embed-text)
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tests.integration.test_ollama_config import (
    OLLAMA_CONFIG,
    get_ollama_test_config, 
    get_model_requirements,
    check_test_environment
)


class OllamaTestRunner:
    """Handles Ollama testing setup and execution."""
    
    def __init__(self, verbose: bool = False, cleanup: bool = True):
        self.verbose = verbose
        self.cleanup = cleanup
        self.config = get_ollama_test_config()
        
    def log(self, message: str, level: str = "INFO"):
        """Log a message with timestamp."""
        if self.verbose or level in ("ERROR", "SUCCESS"):
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    async def check_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{OLLAMA_CONFIG['base_url']}/api/version",
                    timeout=5.0
                )
                return response.status_code == 200
        except Exception as e:
            self.log(f"Ollama connection failed: {e}", "ERROR")
            return False
    
    async def list_available_models(self) -> List[str]:
        """List models available in Ollama."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{OLLAMA_CONFIG['base_url']}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [model.get("name", "").split(":")[0] for model in models]
        except Exception as e:
            self.log(f"Failed to list models: {e}", "ERROR")
        return []
    
    async def pull_model(self, model_name: str, timeout: int = 600) -> bool:
        """Pull a model from Ollama registry."""
        self.log(f"Pulling model: {model_name} (timeout: {timeout}s)")
        
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{OLLAMA_CONFIG['base_url']}/api/pull",
                    json={"name": model_name},
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    # Stream the pull response
                    async for line in response.aiter_lines():
                        if line.strip():
                            try:
                                status = json.loads(line)
                                if self.verbose:
                                    if "status" in status:
                                        self.log(f"Pull status: {status['status']}")
                                if status.get("status") == "success":
                                    return True
                            except json.JSONDecodeError:
                                continue
        except Exception as e:
            self.log(f"Model pull failed: {e}", "ERROR")
        
        return False
    
    async def setup_models(self, models: List[str]) -> bool:
        """Setup required models for testing."""
        self.log("Setting up Ollama models...")
        
        # Check which models are already available
        available_models = await self.list_available_models()
        self.log(f"Available models: {available_models}")
        
        # Pull missing models
        success = True
        for model in models:
            if model not in available_models:
                self.log(f"Model {model} not found, pulling...")
                if not await self.pull_model(model):
                    self.log(f"Failed to pull model: {model}", "ERROR")
                    success = False
                else:
                    self.log(f"Successfully pulled model: {model}", "SUCCESS")
            else:
                self.log(f"Model {model} already available")
        
        return success
    
    def run_pytest_tests(self, test_patterns: List[str], quick: bool = False) -> bool:
        """Run pytest with specified test patterns."""
        cmd = ["python", "-m", "pytest", "-v"]
        
        # Add integration marker
        cmd.extend(["-m", "integration"])
        
        # Skip slow tests if quick mode
        if quick:
            cmd.extend(["-m", "not slow"])
        
        # Add test patterns
        for pattern in test_patterns:
            cmd.append(pattern)
        
        # Set environment variables
        env = os.environ.copy()
        env.update({
            "OLLAMA_BASE_URL": OLLAMA_CONFIG["base_url"],
            "OLLAMA_EMBEDDING_MODEL": OLLAMA_CONFIG["embedding_model"], 
            "OLLAMA_CHAT_MODEL": OLLAMA_CONFIG["chat_model"],
            "OLLAMA_TIMEOUT": str(OLLAMA_CONFIG["timeout"]),
        })
        
        self.log(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, env=env, capture_output=False)
            return result.returncode == 0
        except Exception as e:
            self.log(f"Test execution failed: {e}", "ERROR")
            return False
    
    def generate_test_report(self, results: dict) -> str:
        """Generate a test summary report."""
        report = "\n" + "="*60 + "\n"
        report += "OLLAMA INTEGRATION TEST REPORT\n"
        report += "="*60 + "\n"
        
        # Environment info
        report += f"Ollama URL: {OLLAMA_CONFIG['base_url']}\n"
        report += f"Embedding Model: {OLLAMA_CONFIG['embedding_model']}\n"
        report += f"Chat Model: {OLLAMA_CONFIG['chat_model']}\n"
        report += f"Test Environment: {'PASS' if results.get('environment_check') else 'FAIL'}\n"
        report += "\n"
        
        # Test results
        if "setup_success" in results:
            report += f"Model Setup: {'PASS' if results['setup_success'] else 'FAIL'}\n"
        
        if "test_success" in results:
            report += f"Integration Tests: {'PASS' if results['test_success'] else 'FAIL'}\n"
        
        # Recommendations
        report += "\nRECOMMENDATIONS:\n"
        if not results.get("environment_check"):
            report += "- Ensure Ollama server is running: `ollama serve`\n"
        if not results.get("setup_success"):
            report += "- Check model availability: `ollama list`\n" 
            report += f"- Manually pull models: `ollama pull {OLLAMA_CONFIG['embedding_model']}`\n"
        if not results.get("test_success"):
            report += "- Review test logs for specific failures\n"
            report += "- Check MCP server configuration\n"
        
        report += "\n" + "="*60 + "\n"
        return report


async def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run Mimir MCP + Ollama integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--setup", action="store_true",
                       help="Setup Ollama models before testing")
    parser.add_argument("--quick", action="store_true",
                       help="Run only quick tests (skip slow end-to-end tests)")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose output")
    parser.add_argument("--cleanup", action="store_true", default=True,
                       help="Clean up test data after completion")
    parser.add_argument("--models", type=str,
                       default="nomic-embed-text",
                       help="Comma-separated list of models to test")
    parser.add_argument("--test-pattern", type=str,
                       default="tests/integration/test_mcp_ollama_integration.py",
                       help="Pytest pattern for test selection")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = OllamaTestRunner(verbose=args.verbose, cleanup=args.cleanup)
    models = [model.strip() for model in args.models.split(",")]
    results = {}
    
    print("🚀 Starting Ollama Integration Test Suite")
    print(f"Models to test: {models}")
    print(f"Quick mode: {args.quick}")
    print()
    
    # Step 1: Environment check
    runner.log("Checking test environment...")
    env_check = check_test_environment()
    runner.log(f"Environment check: {env_check}")
    
    ollama_running = await runner.check_ollama_running()
    results["environment_check"] = ollama_running
    
    if not ollama_running:
        runner.log("Ollama server not running!", "ERROR")
        runner.log("Start Ollama server: `ollama serve`", "ERROR")
        print(runner.generate_test_report(results))
        return 1
    
    runner.log("Ollama server is running ✓", "SUCCESS")
    
    # Step 2: Model setup
    if args.setup:
        runner.log("Setting up required models...")
        setup_success = await runner.setup_models(models)
        results["setup_success"] = setup_success
        
        if not setup_success:
            runner.log("Model setup failed!", "ERROR")
            print(runner.generate_test_report(results))
            return 1
        
        runner.log("Model setup completed ✓", "SUCCESS")
    
    # Step 3: Run integration tests
    runner.log("Running integration tests...")
    test_patterns = [args.test_pattern]
    
    test_success = runner.run_pytest_tests(test_patterns, quick=args.quick)
    results["test_success"] = test_success
    
    # Step 4: Generate report
    print(runner.generate_test_report(results))
    
    if test_success:
        runner.log("All tests completed successfully! ✓", "SUCCESS")
        return 0
    else:
        runner.log("Some tests failed ✗", "ERROR")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test runner interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner crashed: {e}")
        sys.exit(1)