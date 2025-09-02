#!/usr/bin/env python3
"""
Mimir Monitoring Setup Script

This script helps set up the complete Mimir monitoring stack with:
- NATS JetStream for trace streaming
- Skald integration for deep monitoring
- Docker Compose orchestration
- Configuration validation

Usage:
    python scripts/setup_monitoring.py [--install-deps] [--start-stack]
"""

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'


class MimirMonitoringSetup:
    """Setup and validation for Mimir monitoring infrastructure."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.config_file = self.project_root / "monitoring_config.yaml"
        
    def print_header(self):
        """Print setup header."""
        print(f"{Colors.BOLD}{Colors.BLUE}")
        print("🔍 Mimir Deep Research Monitoring Setup")
        print("Setting up Skald + NATS monitoring infrastructure")
        print("="*50)
        print(f"{Colors.ENDC}")
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available."""
        print(f"{Colors.YELLOW}📋 Checking dependencies...{Colors.ENDC}")
        
        missing = []
        
        # Check Docker
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"  {Colors.GREEN}✓ Docker: {result.stdout.strip()}{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append("Docker")
            print(f"  {Colors.RED}❌ Docker not found{Colors.ENDC}")
        
        # Check Docker Compose
        try:
            result = subprocess.run(["docker-compose", "--version"], 
                                  capture_output=True, text=True, check=True)
            print(f"  {Colors.GREEN}✓ Docker Compose: {result.stdout.strip()}{Colors.ENDC}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            missing.append("Docker Compose")
            print(f"  {Colors.RED}❌ Docker Compose not found{Colors.ENDC}")
        
        # Check Python dependencies
        try:
            import skald
            print(f"  {Colors.GREEN}✓ Skald monitoring library installed{Colors.ENDC}")
        except ImportError:
            missing.append("sibylline-skald")
            print(f"  {Colors.RED}❌ Skald not installed{Colors.ENDC}")
        
        try:
            import nats
            print(f"  {Colors.GREEN}✓ NATS Python client installed{Colors.ENDC}")
        except ImportError:
            missing.append("nats-py")
            print(f"  {Colors.RED}❌ NATS Python client not installed{Colors.ENDC}")
        
        if missing:
            print(f"\n{Colors.RED}❌ Missing dependencies: {', '.join(missing)}{Colors.ENDC}")
            return False
        
        print(f"{Colors.GREEN}✅ All dependencies available{Colors.ENDC}")
        return True
    
    def install_dependencies(self):
        """Install missing Python dependencies."""
        print(f"{Colors.YELLOW}📦 Installing Python dependencies...{Colors.ENDC}")
        
        packages = [
            "sibylline-skald>=0.2.0",
            "nats-py>=2.10.0", 
            "pyyaml>=6.0"
        ]
        
        for package in packages:
            print(f"  Installing {package}...")
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", package], 
                             check=True, capture_output=True)
                print(f"  {Colors.GREEN}✓ {package} installed{Colors.ENDC}")
            except subprocess.CalledProcessError as e:
                print(f"  {Colors.RED}❌ Failed to install {package}: {e}{Colors.ENDC}")
                return False
        
        return True
    
    def validate_config(self) -> bool:
        """Validate monitoring configuration."""
        print(f"{Colors.YELLOW}⚙️ Validating configuration...{Colors.ENDC}")
        
        if not self.config_file.exists():
            print(f"  {Colors.RED}❌ Config file not found: {self.config_file}{Colors.ENDC}")
            return False
        
        if not HAS_YAML:
            print(f"  {Colors.YELLOW}⚠️ PyYAML not available, skipping config validation{Colors.ENDC}")
            return True
        
        try:
            with open(self.config_file) as f:
                config = yaml.safe_load(f)
            
            # Validate required sections
            required_sections = ["monitoring", "nats", "skald", "traces"]
            for section in required_sections:
                if section not in config:
                    print(f"  {Colors.RED}❌ Missing config section: {section}{Colors.ENDC}")
                    return False
                print(f"  {Colors.GREEN}✓ Section '{section}' present{Colors.ENDC}")
            
            # Validate NATS configuration
            nats_config = config.get("nats", {})
            if nats_config.get("enabled", False):
                url = nats_config.get("url", "nats://localhost:4222")
                stream = nats_config.get("stream_name", "MIMIR_TRACES")
                print(f"  {Colors.GREEN}✓ NATS configured: {url}, stream: {stream}{Colors.ENDC}")
            
            print(f"  {Colors.GREEN}✓ Configuration valid{Colors.ENDC}")
            return True
            
        except Exception as e:
            print(f"  {Colors.RED}❌ Config validation error: {e}{Colors.ENDC}")
            return False
    
    def start_monitoring_stack(self):
        """Start the Docker Compose monitoring stack."""
        print(f"{Colors.YELLOW}🚀 Starting monitoring stack...{Colors.ENDC}")
        
        compose_file = self.project_root / "docker-compose.monitoring.yml"
        if not compose_file.exists():
            print(f"  {Colors.RED}❌ Docker Compose file not found: {compose_file}{Colors.ENDC}")
            return False
        
        try:
            # Start NATS first
            print("  Starting NATS JetStream...")
            result = subprocess.run([
                "docker-compose", "-f", str(compose_file), 
                "up", "-d", "nats"
            ], check=True, capture_output=True, text=True)
            
            # Wait for NATS to be healthy
            print("  Waiting for NATS to be ready...")
            for i in range(30):  # 30 second timeout
                try:
                    health_check = subprocess.run([
                        "docker", "exec", "mimir-nats", 
                        "wget", "--quiet", "--tries=1", "--spider", 
                        "http://localhost:8222/varz"
                    ], check=True, capture_output=True)
                    break
                except subprocess.CalledProcessError:
                    time.sleep(1)
            else:
                print(f"  {Colors.RED}❌ NATS failed to become ready{Colors.ENDC}")
                return False
            
            print(f"  {Colors.GREEN}✓ NATS JetStream started{Colors.ENDC}")
            
            # Start the monitored server
            print("  Starting monitored Mimir server...")
            subprocess.run([
                "docker-compose", "-f", str(compose_file), 
                "up", "-d", "mimir-monitored"
            ], check=True)
            
            print(f"  {Colors.GREEN}✓ Mimir monitored server started{Colors.ENDC}")
            
            # Optional: Start trace viewer
            print("  Starting trace viewer...")
            subprocess.run([
                "docker-compose", "-f", str(compose_file), 
                "up", "-d", "trace-viewer"
            ], check=True)
            
            print(f"  {Colors.GREEN}✓ Trace viewer started{Colors.ENDC}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"  {Colors.RED}❌ Failed to start stack: {e}{Colors.ENDC}")
            if e.stdout:
                print(f"  stdout: {e.stdout}")
            if e.stderr:
                print(f"  stderr: {e.stderr}")
            return False
    
    def show_monitoring_info(self):
        """Show information about the running monitoring stack."""
        print(f"\n{Colors.BOLD}{Colors.GREEN}🎯 Monitoring Stack Ready!{Colors.ENDC}")
        print()
        print("Services running:")
        print("  📡 NATS JetStream: http://localhost:8222")
        print("  🔍 Mimir Monitored Server: stdio interface")
        print("  📊 Trace Viewer: docker logs mimir-trace-viewer")
        print()
        print("To view live traces:")
        print("  docker logs -f mimir-trace-viewer")
        print()
        print("To connect Claude Code to monitored server:")
        print("  Use the mimir-monitored-server command instead of mimir-server")
        print()
        print("To stop the stack:")
        print("  docker-compose -f docker-compose.monitoring.yml down")
        print()
        print(f"{Colors.BLUE}Happy monitoring! 🚀{Colors.ENDC}")
    
    def run_setup(self, install_deps: bool = False, start_stack: bool = False):
        """Run the complete setup process."""
        self.print_header()
        
        # Check dependencies first
        if not self.check_dependencies():
            if install_deps:
                if not self.install_dependencies():
                    return False
                print()  # Add spacing
            else:
                print(f"\n{Colors.YELLOW}💡 Run with --install-deps to install missing dependencies{Colors.ENDC}")
                return False
        
        # Validate configuration
        if not self.validate_config():
            return False
        
        print()  # Add spacing
        
        # Start stack if requested
        if start_stack:
            if self.start_monitoring_stack():
                self.show_monitoring_info()
                return True
            else:
                return False
        else:
            print(f"{Colors.GREEN}✅ Setup validation complete!{Colors.ENDC}")
            print(f"{Colors.YELLOW}💡 Run with --start-stack to start the monitoring infrastructure{Colors.ENDC}")
            return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Set up Mimir deep research monitoring infrastructure"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install missing Python dependencies"
    )
    parser.add_argument(
        "--start-stack", 
        action="store_true",
        help="Start the Docker Compose monitoring stack"
    )
    
    args = parser.parse_args()
    
    setup = MimirMonitoringSetup()
    success = setup.run_setup(
        install_deps=args.install_deps,
        start_stack=args.start_stack
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()