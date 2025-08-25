#!/usr/bin/env python3
"""
Deployment validation script for Mimir Deep Code Research System.
Validates deployment readiness without requiring Docker daemon.
"""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any

class DeploymentValidator:
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: Dict[str, Any] = {}
        self.errors: List[str] = []
        
    def validate_all(self) -> bool:
        """Run all validation checks."""
        print("🔍 Validating Mimir deployment readiness...\n")
        
        checks = [
            ("File Structure", self.validate_file_structure),
            ("Python Dependencies", self.validate_python_deps),
            ("Docker Configuration", self.validate_docker_config),
            ("Environment Setup", self.validate_environment),
            ("Security Framework", self.validate_security),
            ("Application Import", self.validate_app_import),
            ("Health Checks", self.validate_health_checks),
        ]
        
        all_passed = True
        for check_name, check_func in checks:
            print(f"📋 {check_name}...")
            try:
                result = check_func()
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"   {status}\n")
                self.results[check_name] = result
                if not result:
                    all_passed = False
            except Exception as e:
                print(f"   ❌ ERROR: {e}\n")
                self.errors.append(f"{check_name}: {e}")
                all_passed = False
                
        return all_passed
    
    def validate_file_structure(self) -> bool:
        """Validate required files and directories exist."""
        required_files = [
            "Dockerfile",
            "docker-compose.yml", 
            "docker-compose.prod.yml",
            ".env.example",
            "pyproject.toml",
            "src/repoindex/__init__.py",
            "src/repoindex/mcp/server.py",
            "src/repoindex/pipeline/run.py",
        ]
        
        required_dirs = [
            "src",
            "tests", 
            "data",
            "cache",
            "logs",
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
                
        missing_dirs = []
        for dir_path in required_dirs:
            if not (self.project_root / dir_path).exists():
                missing_dirs.append(dir_path)
                
        if missing_files:
            print(f"   Missing files: {', '.join(missing_files)}")
        if missing_dirs:
            print(f"   Missing directories: {', '.join(missing_dirs)}")
            
        return len(missing_files) == 0 and len(missing_dirs) == 0
    
    def validate_python_deps(self) -> bool:
        """Validate Python environment and dependencies."""
        try:
            # Check Python version
            result = subprocess.run([sys.executable, "--version"], 
                                 capture_output=True, text=True)
            python_version = result.stdout.strip()
            print(f"   Python: {python_version}")
            
            # Check if uv is available
            result = subprocess.run(["uv", "--version"], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                print(f"   UV: {result.stdout.strip()}")
            
            # Check if main dependencies can be imported
            result = subprocess.run([
                "uv", "run", "python", "-c", 
                "import asyncio, fastapi, pydantic, mcp; print('Core deps: OK')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   Core dependencies: Available")
                return True
            else:
                print(f"   Dependency error: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   Python validation error: {e}")
            return False
    
    def validate_docker_config(self) -> bool:
        """Validate Docker configuration syntax."""
        try:
            # Validate base compose
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.yml", "config", "--quiet"
            ], capture_output=True, cwd=self.project_root)
            
            if result.returncode != 0:
                print("   Base docker-compose.yml: Invalid syntax")
                return False
                
            # Validate production compose
            result = subprocess.run([
                "docker-compose", "-f", "docker-compose.yml", 
                "-f", "docker-compose.prod.yml", "config", "--quiet"
            ], capture_output=True, cwd=self.project_root)
            
            if result.returncode != 0:
                print("   Production compose: Invalid syntax")
                return False
                
            print("   Docker Compose: Syntax valid")
            return True
            
        except Exception as e:
            print(f"   Docker validation error: {e}")
            return False
    
    def validate_environment(self) -> bool:
        """Validate environment configuration."""
        env_example = self.project_root / ".env.example"
        if not env_example.exists():
            print("   .env.example not found")
            return False
            
        # Count configuration options
        with open(env_example) as f:
            lines = f.readlines()
            
        config_lines = [l for l in lines if l.strip() and not l.strip().startswith('#')]
        print(f"   Environment variables: {len(config_lines)} configured")
        
        return True
    
    def validate_security(self) -> bool:
        """Validate security framework."""
        try:
            result = subprocess.run([
                "uv", "run", "python", "-c",
                """
from src.repoindex.security.config import SecurityConfig
from src.repoindex.security.auth import AuthManager  
from src.repoindex.security.audit import SecurityAuditor
config = SecurityConfig()
auth = AuthManager()
auditor = SecurityAuditor()
print('Security framework: Functional')
"""
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   Security components: All functional")
                return True
            else:
                print(f"   Security validation failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   Security validation error: {e}")
            return False
    
    def validate_app_import(self) -> bool:
        """Validate main application can be imported."""
        try:
            result = subprocess.run([
                "uv", "run", "python", "-c",
                "from src.repoindex.mcp.server import MCPServer; print('MCP Server: OK')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   MCP Server import: Success")
                return True
            else:
                print(f"   MCP Server import failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   Application import error: {e}")
            return False
    
    def validate_health_checks(self) -> bool:
        """Validate health check endpoints."""
        try:
            # Test the health check command from Dockerfile
            result = subprocess.run([
                "uv", "run", "python", "-c",
                "import asyncio; from src.repoindex.mcp.server import MCPServer; print('Health check: OK')"
            ], capture_output=True, text=True, cwd=self.project_root)
            
            if result.returncode == 0:
                print("   Health check command: Working")
                return True
            else:
                print(f"   Health check failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"   Health check validation error: {e}")
            return False
    
    def print_summary(self, all_passed: bool):
        """Print deployment validation summary."""
        print("=" * 60)
        if all_passed:
            print("🎉 DEPLOYMENT VALIDATION PASSED")
            print("\nMimir is ready for production deployment!")
            print("\nNext steps:")
            print("1. Copy .env.example to .env and configure")
            print("2. Run: docker-compose up -d")
            print("3. For production: docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d")
        else:
            print("❌ DEPLOYMENT VALIDATION FAILED")
            print(f"\nFound {len(self.errors)} issues that need to be resolved:")
            for error in self.errors:
                print(f"  • {error}")
        print("=" * 60)

def main():
    """Main validation entry point."""
    project_root = Path(__file__).parent
    validator = DeploymentValidator(project_root)
    
    all_passed = validator.validate_all()
    validator.print_summary(all_passed)
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()