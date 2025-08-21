#!/usr/bin/env python3
"""
Development setup script for Mimir repository indexing system.

This script sets up the development environment, installs dependencies,
and provides instructions for getting started.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"→ {description}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"  ✓ {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ✗ {description}")
        print(f"    Error: {e.stderr.strip()}")
        return False


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def setup_development_environment():
    """Set up the development environment."""
    print("🔧 Setting up Mimir development environment\n")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the project root.")
        return False
    
    # Check if uv is installed
    if not check_uv_installed():
        print("❌ Error: uv is not installed.")
        print("   Please install uv first: https://github.com/astral-sh/uv")
        print("   Or use pip: pip install uv")
        return False
    
    print("✓ uv is installed")
    
    # Install dependencies
    success = True
    
    success &= run_command(
        ["uv", "sync", "--dev"],
        "Installing project dependencies with uv"
    )
    
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        return False
    
    print("\n🎉 Development environment setup complete!")
    print("\nNext steps:")
    print("1. Activate the virtual environment:")
    print("   source .venv/bin/activate  # Linux/macOS")
    print("   .venv\\Scripts\\activate     # Windows")
    print()
    print("2. Run tests to verify setup:")
    print("   uv run pytest tests/unit/ -v")
    print()
    print("3. Start the MCP server:")
    print("   uv run mimir-server")
    print()
    print("4. Or start the UI server:")
    print("   uv run mimir-ui")
    print()
    print("5. See ARCHITECTURE.md for detailed system documentation")
    
    return True


def main():
    """Main setup function."""
    try:
        success = setup_development_environment()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()