#!/usr/bin/env python3
"""
Release script for Mimir MCP Server.

Builds and validates the package before release to ensure it works correctly
as an MCP server for Claude Desktop.
"""

import subprocess
import sys
import tempfile
import json
from pathlib import Path


def run_command(cmd: list[str], description: str) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"→ {description}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ {description}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description}")
        print(f"    Error: {e.stderr.strip()}")
        return False, e.stderr


def validate_wheel(wheel_path: Path) -> bool:
    """Validate that the wheel contains all necessary components for MCP server."""
    print("\n🔍 Validating wheel contents...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Extract wheel
        success, _ = run_command([
            "python3", "-m", "zipfile", "-e", str(wheel_path), str(temp_path)
        ], "Extracting wheel for validation")
        
        if not success:
            return False
            
        # Check that critical modules exist
        required_files = [
            "repoindex/__init__.py",
            "repoindex/mcp/server.py", 
            "repoindex/data/__init__.py",
            "repoindex/data/schemas.py",
        ]
        
        for file_path in required_files:
            full_path = temp_path / file_path
            if not full_path.exists():
                print(f"  ✗ Missing required file: {file_path}")
                return False
            print(f"  ✓ Found: {file_path}")
        
        # Test imports
        import os
        old_path = sys.path[:]
        try:
            sys.path.insert(0, str(temp_path))
            
            # Test critical imports
            from repoindex.data.schemas import AskIndexRequest
            from repoindex.mcp.server import main
            
            print("  ✓ All critical imports successful")
            return True
            
        except ImportError as e:
            print(f"  ✗ Import test failed: {e}")
            return False
        finally:
            sys.path[:] = old_path


def main():
    """Main release process."""
    print("🚀 Mimir MCP Server Release Process\n")
    
    # Check we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run from project root.")
        return False
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    for path in ["dist", "build", "*.egg-info"]:
        run_command(["rm", "-rf", path], f"Removing {path}")
    
    # Build the package
    print("\n📦 Building package...")
    success, _ = run_command(["python3", "-m", "build"], "Building wheel and sdist")
    if not success:
        return False
    
    # Find the built wheel
    dist_dir = Path("dist")
    wheels = list(dist_dir.glob("*.whl"))
    if not wheels:
        print("❌ No wheel file found in dist/")
        return False
    
    wheel_path = wheels[0]
    print(f"📄 Built wheel: {wheel_path}")
    
    # Validate wheel
    if not validate_wheel(wheel_path):
        print("\n❌ Wheel validation failed!")
        return False
    
    # Test installation in clean environment
    print("\n🧪 Testing installation in clean environment...")
    with tempfile.TemporaryDirectory() as temp_dir:
        venv_path = Path(temp_dir) / "test_env"
        
        # Create virtual environment
        success, _ = run_command([
            "python3", "-m", "venv", str(venv_path)
        ], "Creating test virtual environment")
        if not success:
            return False
        
        # Install wheel
        pip_path = venv_path / "bin" / "pip"
        success, _ = run_command([
            str(pip_path), "install", str(wheel_path)
        ], "Installing wheel in test environment")
        if not success:
            return False
        
        # Test entry points
        mimir_server_path = venv_path / "bin" / "mimir-server"
        if not mimir_server_path.exists():
            print("❌ mimir-server entry point not found!")
            return False
        
        print("  ✓ mimir-server entry point exists")
        
        # Test that the entry point can import (without running)
        python_path = venv_path / "bin" / "python"
        success, _ = run_command([
            str(python_path), "-c", 
            "from repoindex.mcp.server import main; print('Entry point imports successfully')"
        ], "Testing entry point import")
        if not success:
            return False
    
    # Generate release information
    print("\n📋 Release Summary:")
    print(f"  📦 Package: {wheel_path.name}")
    print(f"  📐 Size: {wheel_path.stat().st_size // 1024} KB")
    
    # Show installation instructions
    print("\n🎉 Package ready for release!")
    print("\nTo publish to PyPI:")
    print("  1. Test upload: python3 -m twine upload --repository testpypi dist/*")
    print("  2. Production upload: python3 -m twine upload dist/*")
    print("\nMCP Configuration for users:")
    print('  Add to Claude Desktop config: {"mcpServers": {"mimir-repoindex": {"command": "mimir-server"}}}')
    print("\n📚 See MCP_CONFIGURATION.md for detailed user setup instructions.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)