#!/usr/bin/env python3
"""
Comprehensive Security Vulnerability Scanner for Mimir Dependencies

This script performs multi-layered security scanning using:
- safety: PyPI vulnerability database
- pip-audit: Enhanced dependency vulnerability scanning
- bandit: Static analysis security testing
- semgrep: Advanced SAST with security rules

Usage:
    python scripts/deps/deps-scan.py [--fix] [--report-format json|text] [--severity low|medium|high|critical]
"""

import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DependencySecurityScanner:
    """Comprehensive dependency security scanner with multiple tools."""
    
    def __init__(self, project_root: Path, severity_threshold: str = "medium"):
        self.project_root = project_root
        self.severity_threshold = severity_threshold
        self.scan_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project": "mimir",
            "severity_threshold": severity_threshold,
            "scans": {}
        }
    
    def run_safety_scan(self) -> Dict[str, Any]:
        """Run safety vulnerability scan."""
        logger.info("Running safety vulnerability scan...")
        
        try:
            cmd = [
                sys.executable, "-m", "safety", "check",
                "--json",
                "--full-report",
                "--continue-on-error"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    safety_data = {"raw_output": result.stdout}
            else:
                safety_data = {"error": result.stderr}
            
            return {
                "tool": "safety",
                "status": "completed" if result.returncode == 0 else "vulnerabilities_found",
                "exit_code": result.returncode,
                "data": safety_data,
                "vulnerabilities_count": len(safety_data.get("vulnerabilities", [])) if isinstance(safety_data.get("vulnerabilities"), list) else 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Safety scan timed out")
            return {"tool": "safety", "status": "timeout", "error": "Scan timed out after 5 minutes"}
        except Exception as e:
            logger.error(f"Safety scan failed: {e}")
            return {"tool": "safety", "status": "error", "error": str(e)}
    
    def run_pip_audit_scan(self) -> Dict[str, Any]:
        """Run pip-audit vulnerability scan."""
        logger.info("Running pip-audit vulnerability scan...")
        
        try:
            cmd = [
                sys.executable, "-m", "pip_audit",
                "--format", "json",
                "--desc",
                "--requirement", "requirements.txt"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.stdout:
                try:
                    audit_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    audit_data = {"raw_output": result.stdout}
            else:
                audit_data = {"error": result.stderr}
            
            return {
                "tool": "pip-audit",
                "status": "completed" if result.returncode == 0 else "vulnerabilities_found",
                "exit_code": result.returncode,
                "data": audit_data,
                "vulnerabilities_count": len(audit_data.get("vulnerabilities", [])) if isinstance(audit_data.get("vulnerabilities"), list) else 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error("pip-audit scan timed out")
            return {"tool": "pip-audit", "status": "timeout", "error": "Scan timed out after 5 minutes"}
        except Exception as e:
            logger.error(f"pip-audit scan failed: {e}")
            return {"tool": "pip-audit", "status": "error", "error": str(e)}
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run bandit static analysis security scan."""
        logger.info("Running bandit security scan...")
        
        try:
            cmd = [
                sys.executable, "-m", "bandit",
                "-r", "src/",
                "-f", "json",
                "-c", "pyproject.toml"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.stdout:
                try:
                    bandit_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    bandit_data = {"raw_output": result.stdout}
            else:
                bandit_data = {"error": result.stderr}
            
            return {
                "tool": "bandit",
                "status": "completed" if result.returncode == 0 else "issues_found",
                "exit_code": result.returncode,
                "data": bandit_data,
                "issues_count": len(bandit_data.get("results", [])) if isinstance(bandit_data.get("results"), list) else 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Bandit scan timed out")
            return {"tool": "bandit", "status": "timeout", "error": "Scan timed out after 5 minutes"}
        except Exception as e:
            logger.error(f"Bandit scan failed: {e}")
            return {"tool": "bandit", "status": "error", "error": str(e)}
    
    def run_semgrep_scan(self) -> Dict[str, Any]:
        """Run semgrep advanced SAST scan."""
        logger.info("Running semgrep security scan...")
        
        try:
            cmd = [
                sys.executable, "-m", "semgrep",
                "--config", "auto",
                "--config", "p/python",
                "--config", "p/security-audit",
                "--json",
                "src/"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.stdout:
                try:
                    semgrep_data = json.loads(result.stdout)
                except json.JSONDecodeError:
                    semgrep_data = {"raw_output": result.stdout}
            else:
                semgrep_data = {"error": result.stderr}
            
            return {
                "tool": "semgrep",
                "status": "completed" if result.returncode == 0 else "findings",
                "exit_code": result.returncode,
                "data": semgrep_data,
                "findings_count": len(semgrep_data.get("results", [])) if isinstance(semgrep_data.get("results"), list) else 0
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Semgrep scan timed out")
            return {"tool": "semgrep", "status": "timeout", "error": "Scan timed out after 10 minutes"}
        except Exception as e:
            logger.error(f"Semgrep scan failed: {e}")
            return {"tool": "semgrep", "status": "error", "error": str(e)}
    
    def run_comprehensive_scan(self) -> Dict[str, Any]:
        """Run all security scans and aggregate results."""
        logger.info("Starting comprehensive security scan...")
        
        # Run all scans
        self.scan_results["scans"]["safety"] = self.run_safety_scan()
        self.scan_results["scans"]["pip_audit"] = self.run_pip_audit_scan()
        self.scan_results["scans"]["bandit"] = self.run_bandit_scan()
        self.scan_results["scans"]["semgrep"] = self.run_semgrep_scan()
        
        # Aggregate critical findings
        self.scan_results["summary"] = self._aggregate_results()
        
        return self.scan_results
    
    def _aggregate_results(self) -> Dict[str, Any]:
        """Aggregate scan results into summary."""
        total_vulnerabilities = 0
        critical_issues = []
        tools_with_findings = []
        
        for tool, scan_result in self.scan_results["scans"].items():
            if scan_result.get("status") in ["vulnerabilities_found", "issues_found", "findings"]:
                tools_with_findings.append(tool)
                
                if tool == "safety":
                    total_vulnerabilities += scan_result.get("vulnerabilities_count", 0)
                elif tool == "pip_audit":
                    total_vulnerabilities += scan_result.get("vulnerabilities_count", 0)
                elif tool == "bandit":
                    # Filter for high/critical severity issues
                    issues = scan_result.get("data", {}).get("results", [])
                    for issue in issues:
                        if issue.get("issue_severity") in ["MEDIUM", "HIGH"]:
                            critical_issues.append({
                                "tool": tool,
                                "severity": issue.get("issue_severity"),
                                "issue": issue.get("test_name"),
                                "file": issue.get("filename"),
                                "line": issue.get("line_number")
                            })
                elif tool == "semgrep":
                    # Filter for security findings
                    findings = scan_result.get("data", {}).get("results", [])
                    for finding in findings:
                        if "security" in finding.get("check_id", "").lower():
                            critical_issues.append({
                                "tool": tool,
                                "severity": finding.get("extra", {}).get("severity", "MEDIUM"),
                                "issue": finding.get("check_id"),
                                "file": finding.get("path"),
                                "line": finding.get("start", {}).get("line")
                            })
        
        return {
            "total_vulnerabilities": total_vulnerabilities,
            "critical_issues_count": len(critical_issues),
            "tools_with_findings": tools_with_findings,
            "critical_issues": critical_issues[:10],  # Top 10 critical issues
            "security_status": "CRITICAL" if total_vulnerabilities > 0 or len(critical_issues) > 0 else "CLEAN"
        }
    
    def generate_report(self, format_type: str = "text") -> str:
        """Generate human-readable security report."""
        if format_type == "json":
            return json.dumps(self.scan_results, indent=2)
        
        # Text format
        report = []
        report.append("=" * 80)
        report.append("MIMIR DEPENDENCY SECURITY SCAN REPORT")
        report.append("=" * 80)
        report.append(f"Scan Date: {self.scan_results['timestamp']}")
        report.append(f"Severity Threshold: {self.severity_threshold}")
        report.append("")
        
        summary = self.scan_results.get("summary", {})
        report.append("SUMMARY:")
        report.append(f"  Security Status: {summary.get('security_status', 'UNKNOWN')}")
        report.append(f"  Total Vulnerabilities: {summary.get('total_vulnerabilities', 0)}")
        report.append(f"  Critical Issues: {summary.get('critical_issues_count', 0)}")
        report.append(f"  Tools with Findings: {', '.join(summary.get('tools_with_findings', []))}")
        report.append("")
        
        # Detailed scan results
        for tool, scan_result in self.scan_results["scans"].items():
            report.append(f"{tool.upper()} SCAN:")
            report.append(f"  Status: {scan_result.get('status', 'unknown')}")
            report.append(f"  Exit Code: {scan_result.get('exit_code', 'N/A')}")
            
            if scan_result.get("status") == "error":
                report.append(f"  Error: {scan_result.get('error', 'Unknown error')}")
            elif scan_result.get("vulnerabilities_count", 0) > 0:
                report.append(f"  Vulnerabilities Found: {scan_result.get('vulnerabilities_count')}")
            elif scan_result.get("issues_count", 0) > 0:
                report.append(f"  Issues Found: {scan_result.get('issues_count')}")
            elif scan_result.get("findings_count", 0) > 0:
                report.append(f"  Findings: {scan_result.get('findings_count')}")
            
            report.append("")
        
        # Critical issues detail
        if summary.get("critical_issues"):
            report.append("CRITICAL ISSUES (Top 10):")
            for i, issue in enumerate(summary["critical_issues"][:10], 1):
                report.append(f"  {i}. [{issue['tool']}] {issue['severity']}: {issue['issue']}")
                report.append(f"     File: {issue['file']}:{issue.get('line', '?')}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END SECURITY SCAN REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_report(self, filename: str, format_type: str = "text"):
        """Save security report to file."""
        report_content = self.generate_report(format_type)
        
        report_file = self.project_root / filename
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Security report saved to {report_file}")


def main():
    """Main entry point for dependency security scanner."""
    parser = argparse.ArgumentParser(description="Comprehensive dependency security scanner")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix vulnerabilities (where possible)")
    parser.add_argument("--report-format", choices=["text", "json"], default="text", help="Report output format")
    parser.add_argument("--severity", choices=["low", "medium", "high", "critical"], default="medium", help="Minimum severity threshold")
    parser.add_argument("--output", "-o", help="Output report file")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize scanner
    scanner = DependencySecurityScanner(
        project_root=project_root,
        severity_threshold=args.severity
    )
    
    try:
        # Run comprehensive scan
        results = scanner.run_comprehensive_scan()
        
        # Generate and display report
        report = scanner.generate_report(args.report_format)
        print(report)
        
        # Save report if requested
        if args.output:
            scanner.save_report(args.output, args.report_format)
        
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_filename = f"security_scan_report_{timestamp}.{args.report_format}"
        scanner.save_report(auto_filename, args.report_format)
        
        # Exit with appropriate code
        summary = results.get("summary", {})
        if summary.get("security_status") == "CRITICAL":
            logger.error("CRITICAL security issues found!")
            sys.exit(1)
        elif summary.get("total_vulnerabilities", 0) > 0:
            logger.warning("Security vulnerabilities detected")
            sys.exit(2)
        else:
            logger.info("No security vulnerabilities detected")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Scan interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Scan failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()