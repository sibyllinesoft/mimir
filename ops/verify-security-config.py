#!/usr/bin/env python3
"""
Security Configuration Verification Script for Mimir v1.0.0
Validates production security settings and configurations.
"""

import os
import sys
import yaml
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Tuple

class SecurityConfigVerifier:
    """Verifies security configuration across all system components."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues: List[str] = []
        self.warnings: List[str] = []
        self.passed_checks: List[str] = []
    
    def verify_docker_security(self) -> bool:
        """Verify Docker security configurations."""
        print("🔍 Verifying Docker security configurations...")
        
        # Check docker-compose.prod.yml
        prod_compose = self.project_root / "docker-compose.prod.yml"
        if not prod_compose.exists():
            self.issues.append("docker-compose.prod.yml not found")
            return False
        
        with open(prod_compose) as f:
            config = yaml.safe_load(f)
        
        # Verify security configurations for each service
        services = config.get('services', {})
        
        for service_name, service_config in services.items():
            # Check resource limits
            deploy_config = service_config.get('deploy', {})
            resources = deploy_config.get('resources', {})
            
            if 'limits' not in resources:
                self.warnings.append(f"Service {service_name}: No resource limits defined")
            else:
                self.passed_checks.append(f"Service {service_name}: Resource limits configured")
            
            # Check security options
            security_opt = service_config.get('security_opt', [])
            if 'no-new-privileges:true' not in security_opt and service_name != 'redis':
                self.warnings.append(f"Service {service_name}: no-new-privileges not set")
            else:
                self.passed_checks.append(f"Service {service_name}: Security options configured")
            
            # Check health checks
            if 'healthcheck' not in service_config and service_name not in ['loki', 'grafana']:
                self.warnings.append(f"Service {service_name}: No health check defined")
            elif 'healthcheck' in service_config:
                self.passed_checks.append(f"Service {service_name}: Health check configured")
        
        return True
    
    def verify_nginx_security(self) -> bool:
        """Verify NGINX security configurations."""
        print("🔍 Verifying NGINX security configurations...")
        
        nginx_config = self.project_root / "ops" / "nginx" / "nginx.conf"
        if not nginx_config.exists():
            self.issues.append("NGINX configuration file not found")
            return False
        
        with open(nginx_config) as f:
            content = f.read()
        
        # Check for security headers
        security_headers = [
            'X-Frame-Options',
            'X-XSS-Protection',
            'X-Content-Type-Options',
            'Referrer-Policy',
            'Content-Security-Policy',
            'Strict-Transport-Security'
        ]
        
        for header in security_headers:
            if header in content:
                self.passed_checks.append(f"NGINX: {header} header configured")
            else:
                self.issues.append(f"NGINX: Missing {header} header")
        
        # Check for rate limiting
        if 'limit_req_zone' in content:
            self.passed_checks.append("NGINX: Rate limiting configured")
        else:
            self.issues.append("NGINX: Rate limiting not configured")
        
        # Check for SSL configuration template
        if 'ssl_protocols' in content:
            self.passed_checks.append("NGINX: SSL configuration template present")
        else:
            self.warnings.append("NGINX: SSL configuration template not found")
        
        return True
    
    def verify_application_security(self) -> bool:
        """Verify application-level security configurations."""
        print("🔍 Verifying application security configurations...")
        
        # Check security module exists
        security_dir = self.project_root / "src" / "repoindex" / "security"
        if not security_dir.exists():
            self.issues.append("Security module directory not found")
            return False
        
        # Check key security files
        security_files = [
            'config.py',
            'auth.py', 
            'audit.py',
            'sandbox.py',
            'crypto.py',  # encryption functionality
            'secrets.py'  # credential scanning functionality
        ]
        
        for file in security_files:
            file_path = security_dir / file
            if file_path.exists():
                self.passed_checks.append(f"Security module: {file} present")
            else:
                self.issues.append(f"Security module: {file} missing")
        
        # Check MCP server security
        mcp_server = self.project_root / "src" / "repoindex" / "mcp" / "server.py"
        mcp_secure_server = self.project_root / "src" / "repoindex" / "mcp" / "secure_server.py"
        
        if mcp_secure_server.exists():
            with open(mcp_secure_server) as f:
                content = f.read()
            
            if 'SecurityConfig' in content:
                self.passed_checks.append("MCP server: Security configuration integrated (secure_server.py)")
            else:
                self.warnings.append("MCP server: Security configuration not properly integrated")
        elif mcp_server.exists():
            with open(mcp_server) as f:
                content = f.read()
            
            if 'SecurityConfig' in content:
                self.passed_checks.append("MCP server: Security configuration integrated")
            else:
                self.warnings.append("MCP server: Security configuration not integrated")
        else:
            self.issues.append("MCP server files not found")
        
        return True
    
    def verify_dependencies(self) -> bool:
        """Verify dependency security."""
        print("🔍 Verifying dependency security...")
        
        # Check if safety is available
        try:
            result = subprocess.run(['safety', '--version'], 
                                 capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                self.passed_checks.append("Security scanner (safety) available")
                
                # Run safety check
                result = subprocess.run(['safety', 'check'], 
                                     capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    self.passed_checks.append("Dependencies: No known vulnerabilities")
                else:
                    self.issues.append(f"Dependencies: Security vulnerabilities found:\n{result.stdout}")
            else:
                self.warnings.append("Security scanner (safety) not available")
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("Security scanner (safety) not available or timeout")
        
        return True
    
    def verify_environment_setup(self) -> bool:
        """Verify environment security setup."""
        print("🔍 Verifying environment security setup...")
        
        # Check for example environment files
        env_example = self.project_root / ".env.example"
        if env_example.exists():
            self.passed_checks.append("Environment: Example configuration present")
        else:
            self.warnings.append("Environment: No .env.example found")
        
        # Check for secrets in codebase (basic check)
        try:
            result = subprocess.run([
                'grep', '-r', '-i', 
                '--exclude-dir=.git',
                '--exclude-dir=__pycache__',
                '--exclude=*.pyc',
                r'password\|secret\|key.*=',
                str(self.project_root)
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:  # No matches found
                self.passed_checks.append("Codebase: No obvious secrets found")
            else:
                # Filter out acceptable patterns
                lines = result.stdout.split('\n')
                suspicious_lines = [
                    line for line in lines 
                    if line and 
                    'example' not in line.lower() and 
                    'test' not in line.lower() and
                    'TODO' not in line and
                    'config' not in line.lower()
                ]
                
                if suspicious_lines:
                    self.warnings.append(f"Potential secrets in codebase:\n" + 
                                       '\n'.join(suspicious_lines[:5]))
                else:
                    self.passed_checks.append("Codebase: No obvious secrets found")
        
        except (subprocess.TimeoutExpired, FileNotFoundError):
            self.warnings.append("Unable to scan for secrets in codebase")
        
        return True
    
    def run_verification(self) -> Tuple[bool, Dict[str, Any]]:
        """Run complete security verification."""
        print("🛡️  Starting Mimir Security Configuration Verification")
        print("=" * 60)
        
        # Run all verification checks
        checks = [
            self.verify_docker_security,
            self.verify_nginx_security, 
            self.verify_application_security,
            self.verify_dependencies,
            self.verify_environment_setup
        ]
        
        all_passed = True
        for check in checks:
            try:
                if not check():
                    all_passed = False
            except Exception as e:
                self.issues.append(f"Verification error: {str(e)}")
                all_passed = False
        
        # Generate report
        print("\n" + "=" * 60)
        print("🛡️  SECURITY VERIFICATION REPORT")
        print("=" * 60)
        
        if self.passed_checks:
            print(f"\n✅ PASSED CHECKS ({len(self.passed_checks)}):")
            for check in self.passed_checks:
                print(f"   ✅ {check}")
        
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   ⚠️  {warning}")
        
        if self.issues:
            print(f"\n❌ CRITICAL ISSUES ({len(self.issues)}):")
            for issue in self.issues:
                print(f"   ❌ {issue}")
            all_passed = False
        
        print("\n" + "=" * 60)
        
        if all_passed and not self.issues:
            print("🎯 SECURITY VERIFICATION: PASSED")
            print("   System is ready for production deployment")
        elif not self.issues:
            print("⚠️  SECURITY VERIFICATION: PASSED WITH WARNINGS") 
            print("   Review warnings before production deployment")
        else:
            print("❌ SECURITY VERIFICATION: FAILED")
            print("   Critical issues must be resolved before deployment")
        
        return all_passed and not self.issues, {
            'passed_checks': len(self.passed_checks),
            'warnings': len(self.warnings),
            'critical_issues': len(self.issues),
            'overall_status': 'PASSED' if all_passed and not self.issues else 'FAILED'
        }

def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent
    verifier = SecurityConfigVerifier(project_root)
    
    success, report = verifier.run_verification()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()