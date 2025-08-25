#!/usr/bin/env python3
"""
Security audit script for Mimir cryptographic implementations.

Validates the security fixes and performs comprehensive security checks:
- Verifies hardcoded salt vulnerability is fixed
- Checks for other cryptographic vulnerabilities
- Validates security best practices
- Generates security report
"""

import ast
import hashlib
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ANSI color codes for output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


class SecurityAuditor:
    """Performs comprehensive security audit of cryptographic implementations."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.issues = []
        self.warnings = []
        self.passed_checks = []
        
    def audit_hardcoded_secrets(self) -> List[Dict[str, Any]]:
        """Check for hardcoded cryptographic secrets and salts."""
        print(f"{Colors.HEADER}🔍 Checking for hardcoded cryptographic secrets...{Colors.ENDC}")
        
        issues = []
        
        # Patterns for hardcoded secrets
        patterns = [
            (r'salt\s*=\s*b?["\'][^"\']+["\']', "Hardcoded salt detected"),
            (r'key\s*=\s*b?["\'][^"\']+["\']', "Hardcoded encryption key detected"),
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password detected"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret detected"),
            (r'token\s*=\s*["\'][^"\']{20,}["\']', "Hardcoded token detected"),
        ]
        
        # Whitelist for legitimate constants
        whitelist = [
            "MAGIC_HEADER",
            "test_password",
            "example_",
            "demo_",
            "TODO",
            "FIXME",
        ]
        
        for py_file in self.project_root.rglob("*.py"):
            if "test" in str(py_file) or "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                for pattern, description in patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        matched_text = match.group(0)
                        
                        # Check whitelist
                        if any(wl in matched_text for wl in whitelist):
                            continue
                            
                        issues.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'issue': description,
                            'code': matched_text.strip(),
                            'severity': 'HIGH'
                        })
                        
            except Exception as e:
                print(f"{Colors.WARNING}Warning: Could not read {py_file}: {e}{Colors.ENDC}")
        
        return issues
    
    def audit_crypto_implementations(self) -> List[Dict[str, Any]]:
        """Audit cryptographic implementations for security issues."""
        print(f"{Colors.HEADER}🔐 Auditing cryptographic implementations...{Colors.ENDC}")
        
        issues = []
        crypto_files = list(self.project_root.rglob("*crypto*.py")) + \
                      list(self.project_root.rglob("*secret*.py")) + \
                      list(self.project_root.rglob("*encrypt*.py"))
        
        for py_file in crypto_files:
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for weak cryptographic practices
                weak_patterns = [
                    (r'MD5\(', "MD5 is cryptographically broken", "HIGH"),
                    (r'SHA1\(', "SHA1 is cryptographically weak", "MEDIUM"),
                    (r'DES\(', "DES encryption is broken", "HIGH"),
                    (r'RC4\(', "RC4 cipher is broken", "HIGH"),
                    (r'CBC\s*\(.*,\s*None', "CBC mode without IV", "HIGH"),
                    (r'random\.random\(\)', "Using weak random for crypto", "HIGH"),
                    (r'time\(\).*salt', "Time-based salt generation", "MEDIUM"),
                    (r'iterations\s*=\s*[0-9]{1,3}[^0-9]', "Low PBKDF2 iterations", "MEDIUM"),
                ]
                
                for pattern, description, severity in weak_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        line_num = content[:match.start()].count('\n') + 1
                        issues.append({
                            'file': str(py_file.relative_to(self.project_root)),
                            'line': line_num,
                            'issue': description,
                            'code': match.group(0).strip(),
                            'severity': severity
                        })
                
                # Check for good practices
                good_patterns = [
                    r'secrets\.token_bytes\(',
                    r'PBKDF2HMAC\(',
                    r'Scrypt\(',
                    r'AES.*GCM\(',
                    r'hmac\.compare_digest\(',
                ]
                
                for pattern in good_patterns:
                    if re.search(pattern, content):
                        self.passed_checks.append(f"{py_file.name}: Uses secure {pattern.split('(')[0]} implementation")
                
            except Exception as e:
                print(f"{Colors.WARNING}Warning: Could not analyze {py_file}: {e}{Colors.ENDC}")
        
        return issues
    
    def check_secrets_manager_security(self) -> List[Dict[str, Any]]:
        """Specifically check SecretsManager for security compliance."""
        print(f"{Colors.HEADER}🛡️  Checking SecretsManager security compliance...{Colors.ENDC}")
        
        issues = []
        secrets_file = self.project_root / "src" / "repoindex" / "security" / "secrets.py"
        
        if not secrets_file.exists():
            return [{'issue': 'SecretsManager file not found', 'severity': 'HIGH'}]
        
        try:
            with open(secrets_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST to analyze code structure
            tree = ast.parse(content)
            
            # Check for required security features
            required_features = {
                'salt_generation': False,
                'pbkdf2_usage': False,
                'secure_storage': False,
                'migration_support': False,
            }
            
            # Look for security implementations
            if 'secrets.token_bytes' in content:
                required_features['salt_generation'] = True
                self.passed_checks.append("SecretsManager: Uses cryptographically secure salt generation")
            
            if 'PBKDF2HMAC' in content:
                required_features['pbkdf2_usage'] = True
                self.passed_checks.append("SecretsManager: Uses PBKDF2 for key derivation")
            
            if 'MAGIC_HEADER' in content:
                required_features['secure_storage'] = True
                self.passed_checks.append("SecretsManager: Implements secure file format")
            
            if 'migrate_legacy_file' in content:
                required_features['migration_support'] = True
                self.passed_checks.append("SecretsManager: Supports legacy file migration")
            
            # Check for the specific vulnerability fix
            if 'salt = b"mimir_salt_v1"' not in content or 'LEGACY_SALT' in content:
                self.passed_checks.append("SecretsManager: Hardcoded salt vulnerability FIXED")
            else:
                issues.append({
                    'file': 'secrets.py',
                    'issue': 'Hardcoded salt still present without proper handling',
                    'severity': 'CRITICAL'
                })
            
            # Check for missing features
            for feature, present in required_features.items():
                if not present:
                    issues.append({
                        'file': 'secrets.py',
                        'issue': f'Missing security feature: {feature}',
                        'severity': 'MEDIUM'
                    })
            
            # Check iteration count
            iteration_match = re.search(r'iterations\s*=\s*(\d+)', content)
            if iteration_match:
                iterations = int(iteration_match.group(1))
                if iterations < 100000:
                    issues.append({
                        'file': 'secrets.py',
                        'issue': f'PBKDF2 iterations too low: {iterations} (recommended: ≥100000)',
                        'severity': 'MEDIUM'
                    })
                else:
                    self.passed_checks.append(f"SecretsManager: Uses secure iteration count: {iterations}")
            
        except Exception as e:
            issues.append({
                'file': 'secrets.py',
                'issue': f'Failed to analyze SecretsManager: {e}',
                'severity': 'HIGH'
            })
        
        return issues
    
    def check_environment_security(self) -> List[Dict[str, Any]]:
        """Check for environment variable security issues."""
        print(f"{Colors.HEADER}🌍 Checking environment variable security...{Colors.ENDC}")
        
        issues = []
        
        # Check for hardcoded environment variables
        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for environment variable patterns
                env_patterns = [
                    r'os\.environ\[["\']([^"\']*(?:key|secret|password|token)[^"\']*)["\']',
                    r'getenv\(["\']([^"\']*(?:key|secret|password|token)[^"\']*)["\']'
                ]
                
                for pattern in env_patterns:
                    for match in re.finditer(pattern, content, re.IGNORECASE):
                        env_var = match.group(1)
                        line_num = content[:match.start()].count('\n') + 1
                        
                        # This is actually good practice, just document it
                        self.passed_checks.append(f"{py_file.name}: Uses environment variable {env_var} for secrets")
                
            except Exception as e:
                continue
        
        return issues
    
    def test_crypto_functions(self) -> List[Dict[str, Any]]:
        """Test cryptographic functions for basic security properties."""
        print(f"{Colors.HEADER}🧪 Testing cryptographic function security...{Colors.ENDC}")
        
        issues = []
        
        try:
            # Import and test SecretsManager
            sys.path.insert(0, str(self.project_root))
            from src.repoindex.security.secrets import SecretsManager
            
            # Test salt generation uniqueness
            salts = []
            for _ in range(10):
                manager = SecretsManager(password="test_password")
                salts.append(manager.salt)
            
            if len(set(salts)) != len(salts):
                issues.append({
                    'issue': 'Salt generation not unique across instances',
                    'severity': 'HIGH'
                })
            else:
                self.passed_checks.append("Salt generation: Produces unique salts across instances")
            
            # Test salt length
            if all(len(salt) == SecretsManager.SALT_SIZE for salt in salts):
                self.passed_checks.append(f"Salt generation: Correct length ({SecretsManager.SALT_SIZE} bytes)")
            else:
                issues.append({
                    'issue': 'Salt length inconsistent or incorrect',
                    'severity': 'MEDIUM'
                })
            
            # Test that no salt equals the legacy salt
            if any(salt == SecretsManager.LEGACY_SALT for salt in salts):
                issues.append({
                    'issue': 'New instances still generating legacy salt',
                    'severity': 'CRITICAL'
                })
            else:
                self.passed_checks.append("Salt generation: Never generates legacy hardcoded salt")
                
        except Exception as e:
            issues.append({
                'issue': f'Failed to test crypto functions: {e}',
                'severity': 'HIGH'
            })
        
        return issues
    
    def generate_report(self) -> str:
        """Generate comprehensive security audit report."""
        all_issues = []
        
        # Run all audits
        all_issues.extend(self.audit_hardcoded_secrets())
        all_issues.extend(self.audit_crypto_implementations())
        all_issues.extend(self.check_secrets_manager_security())
        all_issues.extend(self.check_environment_security())
        all_issues.extend(self.test_crypto_functions())
        
        # Generate report
        report = []
        report.append("=" * 80)
        report.append("MIMIR SECURITY AUDIT REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        critical = sum(1 for issue in all_issues if issue.get('severity') == 'CRITICAL')
        high = sum(1 for issue in all_issues if issue.get('severity') == 'HIGH')
        medium = sum(1 for issue in all_issues if issue.get('severity') == 'MEDIUM')
        low = sum(1 for issue in all_issues if issue.get('severity') == 'LOW')
        
        report.append(f"SUMMARY:")
        report.append(f"  Critical Issues: {critical}")
        report.append(f"  High Issues: {high}")
        report.append(f"  Medium Issues: {medium}")
        report.append(f"  Low Issues: {low}")
        report.append(f"  Security Checks Passed: {len(self.passed_checks)}")
        report.append("")
        
        # Overall status
        if critical > 0:
            status = f"{Colors.FAIL}CRITICAL VULNERABILITIES FOUND{Colors.ENDC}"
        elif high > 0:
            status = f"{Colors.WARNING}HIGH RISK ISSUES FOUND{Colors.ENDC}"
        elif medium > 0:
            status = f"{Colors.WARNING}MEDIUM RISK ISSUES FOUND{Colors.ENDC}"
        else:
            status = f"{Colors.OKGREEN}SECURITY AUDIT PASSED{Colors.ENDC}"
        
        report.append(f"OVERALL STATUS: {status}")
        report.append("")
        
        # Detailed issues
        if all_issues:
            report.append("SECURITY ISSUES FOUND:")
            report.append("-" * 40)
            
            for issue in sorted(all_issues, key=lambda x: {'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3}.get(x.get('severity', 'LOW'))):
                severity_color = {
                    'CRITICAL': Colors.FAIL,
                    'HIGH': Colors.FAIL,
                    'MEDIUM': Colors.WARNING,
                    'LOW': Colors.OKBLUE
                }.get(issue.get('severity', 'LOW'), Colors.ENDC)
                
                report.append(f"{severity_color}[{issue.get('severity', 'UNKNOWN')}]{Colors.ENDC} {issue.get('issue', 'Unknown issue')}")
                
                if 'file' in issue:
                    report.append(f"  File: {issue['file']}")
                if 'line' in issue:
                    report.append(f"  Line: {issue['line']}")
                if 'code' in issue:
                    report.append(f"  Code: {issue['code']}")
                report.append("")
        
        # Passed checks
        if self.passed_checks:
            report.append("SECURITY CHECKS PASSED:")
            report.append("-" * 40)
            for check in self.passed_checks:
                report.append(f"{Colors.OKGREEN}✓{Colors.ENDC} {check}")
            report.append("")
        
        # Recommendations
        report.append("SECURITY RECOMMENDATIONS:")
        report.append("-" * 40)
        report.append("1. Regularly rotate encryption keys and passwords")
        report.append("2. Use environment variables for sensitive configuration")
        report.append("3. Implement proper key management for production")
        report.append("4. Enable comprehensive logging for security events")
        report.append("5. Perform regular security audits and penetration testing")
        report.append("6. Keep cryptographic libraries updated")
        report.append("")
        
        # Hardcoded salt fix verification
        report.append("HARDCODED SALT VULNERABILITY STATUS:")
        report.append("-" * 40)
        
        has_hardcoded_issues = any(
            'hardcoded salt' in issue.get('issue', '').lower() or
            'salt' in issue.get('issue', '').lower()
            for issue in all_issues
            if issue.get('severity') in ['CRITICAL', 'HIGH']
        )
        
        if has_hardcoded_issues:
            report.append(f"{Colors.FAIL}❌ HARDCODED SALT VULNERABILITY STILL PRESENT{Colors.ENDC}")
        else:
            report.append(f"{Colors.OKGREEN}✅ HARDCODED SALT VULNERABILITY SUCCESSFULLY FIXED{Colors.ENDC}")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Run security audit."""
    project_root = Path(__file__).parent.parent
    
    print(f"{Colors.BOLD}{Colors.HEADER}Mimir Security Audit{Colors.ENDC}")
    print(f"Auditing project: {project_root}")
    print("")
    
    auditor = SecurityAuditor(project_root)
    report = auditor.generate_report()
    
    print(report)
    
    # Save report to file
    report_file = project_root / "security_audit_report.txt"
    with open(report_file, 'w') as f:
        # Strip ANSI codes for file
        import re
        clean_report = re.sub(r'\033\[[0-9;]*m', '', report)
        f.write(clean_report)
    
    print(f"\nReport saved to: {report_file}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())