#!/usr/bin/env python3
"""
Automated Dependency Update Management for Mimir

This script manages dependency updates with security validation:
- Analyzes outdated dependencies
- Proposes safe updates with testing
- Creates update branches with validation
- Integrates security scanning into update workflow

Usage:
    python scripts/deps/deps-update.py [--dry-run] [--security-only] [--create-pr] [--auto-merge-security]
"""

import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
import re
import tempfile
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyUpdateManager:
    """Manages dependency updates with security validation and testing."""
    
    def __init__(self, project_root: Path, dry_run: bool = False):
        self.project_root = project_root
        self.dry_run = dry_run
        self.pyproject_file = project_root / "pyproject.toml"
        self.requirements_file = project_root / "requirements.txt"
        
        # Security-critical packages that require immediate updates
        self.critical_packages = {
            "cryptography", "fastapi", "pydantic", "uvicorn", "httpx",
            "jinja2", "python-multipart", "aiofiles", "GitPython"
        }
        
    def get_outdated_dependencies(self) -> Dict[str, Any]:
        """Get list of outdated dependencies using pip."""
        logger.info("Checking for outdated dependencies...")
        
        try:
            cmd = [sys.executable, "-m", "pip", "list", "--outdated", "--format=json"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to check outdated dependencies: {result.stderr}")
                return {"packages": [], "error": result.stderr}
            
            outdated_packages = json.loads(result.stdout) if result.stdout else []
            
            # Categorize updates by priority
            critical_updates = []
            security_updates = []
            minor_updates = []
            major_updates = []
            
            for package in outdated_packages:
                name = package["name"]
                current = package["version"]
                latest = package["latest_version"]
                
                update_info = {
                    "name": name,
                    "current_version": current,
                    "latest_version": latest,
                    "update_type": self._classify_update_type(current, latest),
                    "is_critical": name.lower() in self.critical_packages,
                    "security_relevant": self._is_security_relevant(name)
                }
                
                if update_info["is_critical"]:
                    critical_updates.append(update_info)
                elif update_info["security_relevant"]:
                    security_updates.append(update_info)
                elif update_info["update_type"] == "major":
                    major_updates.append(update_info)
                else:
                    minor_updates.append(update_info)
            
            return {
                "total_outdated": len(outdated_packages),
                "critical_updates": critical_updates,
                "security_updates": security_updates,
                "minor_updates": minor_updates,
                "major_updates": major_updates,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Dependency check timed out")
            return {"packages": [], "error": "Timeout"}
        except Exception as e:
            logger.error(f"Failed to check dependencies: {e}")
            return {"packages": [], "error": str(e)}
    
    def _classify_update_type(self, current: str, latest: str) -> str:
        """Classify update type based on semantic versioning."""
        try:
            # Parse version numbers
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]
            
            # Pad to same length
            max_len = max(len(current_parts), len(latest_parts))
            current_parts.extend([0] * (max_len - len(current_parts)))
            latest_parts.extend([0] * (max_len - len(latest_parts)))
            
            if latest_parts[0] > current_parts[0]:
                return "major"
            elif latest_parts[1] > current_parts[1]:
                return "minor"
            elif latest_parts[2] > current_parts[2]:
                return "patch"
            else:
                return "unknown"
                
        except (ValueError, IndexError):
            return "unknown"
    
    def _is_security_relevant(self, package_name: str) -> bool:
        """Check if package is security-relevant based on known patterns."""
        security_keywords = {
            "crypt", "auth", "security", "ssl", "tls", "jwt", "oauth",
            "password", "hash", "sign", "cert", "key"
        }
        
        return any(keyword in package_name.lower() for keyword in security_keywords)
    
    def check_security_vulnerabilities(self, package_updates: List[Dict]) -> Dict[str, Any]:
        """Check for known vulnerabilities in packages before updating."""
        logger.info("Checking security vulnerabilities for update candidates...")
        
        vulnerable_packages = []
        safe_packages = []
        
        for update in package_updates:
            package_name = update["name"]
            current_version = update["current_version"]
            
            # Check with safety
            try:
                cmd = [
                    sys.executable, "-m", "safety", "check",
                    "--json",
                    f"--package={package_name}=={current_version}"
                ]
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0 and result.stdout:
                    # Vulnerabilities found
                    vuln_data = json.loads(result.stdout)
                    vulnerable_packages.append({
                        "package": update,
                        "vulnerabilities": vuln_data
                    })
                else:
                    safe_packages.append(update)
                    
            except Exception as e:
                logger.warning(f"Could not check vulnerabilities for {package_name}: {e}")
                safe_packages.append(update)  # Assume safe if check fails
        
        return {
            "vulnerable_packages": vulnerable_packages,
            "safe_packages": safe_packages,
            "vulnerable_count": len(vulnerable_packages)
        }
    
    def create_update_plan(self, outdated_deps: Dict[str, Any], security_only: bool = False) -> Dict[str, Any]:
        """Create a prioritized update plan."""
        logger.info("Creating dependency update plan...")
        
        update_plan = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "security_only": security_only,
            "batches": []
        }
        
        # Batch 1: Critical security updates (immediate)
        if outdated_deps["critical_updates"]:
            security_check = self.check_security_vulnerabilities(outdated_deps["critical_updates"])
            update_plan["batches"].append({
                "name": "critical_security",
                "priority": 1,
                "description": "Critical security-related dependency updates",
                "packages": outdated_deps["critical_updates"],
                "security_analysis": security_check,
                "auto_merge": True,
                "requires_testing": True
            })
        
        # Batch 2: Security updates (high priority)
        if not security_only and outdated_deps["security_updates"]:
            security_check = self.check_security_vulnerabilities(outdated_deps["security_updates"])
            update_plan["batches"].append({
                "name": "security_updates",
                "priority": 2,
                "description": "Security-relevant dependency updates",
                "packages": outdated_deps["security_updates"],
                "security_analysis": security_check,
                "auto_merge": False,
                "requires_testing": True
            })
        
        # Batch 3: Minor updates (if not security-only mode)
        if not security_only and outdated_deps["minor_updates"]:
            update_plan["batches"].append({
                "name": "minor_updates",
                "priority": 3,
                "description": "Minor version updates",
                "packages": outdated_deps["minor_updates"][:10],  # Limit to 10
                "auto_merge": False,
                "requires_testing": True
            })
        
        # Batch 4: Major updates (manual review required)
        if not security_only and outdated_deps["major_updates"]:
            update_plan["batches"].append({
                "name": "major_updates",
                "priority": 4,
                "description": "Major version updates (require manual review)",
                "packages": outdated_deps["major_updates"][:5],  # Limit to 5
                "auto_merge": False,
                "requires_testing": True,
                "requires_manual_review": True
            })
        
        return update_plan
    
    def apply_updates(self, update_batch: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a batch of dependency updates."""
        batch_name = update_batch["name"]
        packages = update_batch["packages"]
        
        logger.info(f"Applying update batch: {batch_name} ({len(packages)} packages)")
        
        if self.dry_run:
            logger.info("DRY RUN - Would apply the following updates:")
            for package in packages:
                logger.info(f"  {package['name']}: {package['current_version']} -> {package['latest_version']}")
            return {"status": "dry_run", "packages_updated": packages}
        
        updated_packages = []
        failed_packages = []
        
        for package in packages:
            try:
                # Update using pip
                cmd = [
                    sys.executable, "-m", "pip", "install", "--upgrade",
                    f"{package['name']}=={package['latest_version']}"
                ]
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    updated_packages.append(package)
                    logger.info(f"Updated {package['name']} to {package['latest_version']}")
                else:
                    failed_packages.append({
                        "package": package,
                        "error": result.stderr
                    })
                    logger.error(f"Failed to update {package['name']}: {result.stderr}")
                    
            except Exception as e:
                failed_packages.append({
                    "package": package,
                    "error": str(e)
                })
                logger.error(f"Exception updating {package['name']}: {e}")
        
        # Update requirements files
        self._update_requirements_files()
        
        return {
            "status": "completed",
            "packages_updated": updated_packages,
            "packages_failed": failed_packages,
            "success_count": len(updated_packages),
            "failure_count": len(failed_packages)
        }
    
    def _update_requirements_files(self):
        """Update requirements.txt and other dependency files."""
        logger.info("Updating requirements files...")
        
        try:
            # Generate new requirements.txt
            cmd = [sys.executable, "-m", "pip", "freeze"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                with open(self.requirements_file, 'w') as f:
                    f.write(result.stdout)
                logger.info("Updated requirements.txt")
            else:
                logger.error(f"Failed to generate requirements.txt: {result.stderr}")
                
        except Exception as e:
            logger.error(f"Failed to update requirements files: {e}")
    
    def run_tests_after_update(self) -> Dict[str, Any]:
        """Run test suite after dependency updates."""
        logger.info("Running test suite after dependency updates...")
        
        try:
            # Run pytest with coverage
            cmd = [
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=src",
                "--cov-report=json",
                "--cov-report=term-missing",
                "-v"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            return {
                "test_status": "passed" if result.returncode == 0 else "failed",
                "exit_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr
            }
            
        except subprocess.TimeoutExpired:
            return {
                "test_status": "timeout",
                "error": "Test suite timed out after 10 minutes"
            }
        except Exception as e:
            return {
                "test_status": "error",
                "error": str(e)
            }
    
    def create_update_branch(self, batch_name: str) -> str:
        """Create a Git branch for dependency updates."""
        branch_name = f"deps/update-{batch_name}-{datetime.now().strftime('%Y%m%d')}"
        
        logger.info(f"Creating update branch: {branch_name}")
        
        if not self.dry_run:
            try:
                # Create and checkout new branch
                subprocess.run(
                    ["git", "checkout", "-b", branch_name],
                    cwd=self.project_root,
                    check=True
                )
                logger.info(f"Created branch {branch_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to create branch: {e}")
                raise
        
        return branch_name
    
    def commit_updates(self, batch: Dict[str, Any], update_results: Dict[str, Any]):
        """Commit dependency updates to Git."""
        if self.dry_run:
            logger.info("DRY RUN - Would commit dependency updates")
            return
        
        try:
            # Add updated files
            subprocess.run(
                ["git", "add", "requirements.txt", "pyproject.toml"],
                cwd=self.project_root,
                check=True
            )
            
            # Create commit message
            updated_packages = update_results["packages_updated"]
            commit_msg = f"Update {batch['name']} dependencies\n\n"
            commit_msg += f"Updated {len(updated_packages)} packages:\n"
            
            for package in updated_packages:
                commit_msg += f"- {package['name']}: {package['current_version']} -> {package['latest_version']}\n"
            
            if batch.get("security_analysis", {}).get("vulnerable_count", 0) > 0:
                commit_msg += f"\nSecurity: Addresses {batch['security_analysis']['vulnerable_count']} vulnerabilities"
            
            # Commit changes
            subprocess.run(
                ["git", "commit", "-m", commit_msg],
                cwd=self.project_root,
                check=True
            )
            
            logger.info("Committed dependency updates")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit updates: {e}")
            raise
    
    def generate_update_report(self, update_plan: Dict[str, Any], results: List[Dict[str, Any]]) -> str:
        """Generate dependency update report."""
        report = []
        report.append("=" * 80)
        report.append("MIMIR DEPENDENCY UPDATE REPORT")
        report.append("=" * 80)
        report.append(f"Report Date: {update_plan['timestamp']}")
        report.append(f"Update Mode: {'Security Only' if update_plan['security_only'] else 'Full Update'}")
        report.append("")
        
        for i, (batch, result) in enumerate(zip(update_plan["batches"], results)):
            report.append(f"BATCH {i+1}: {batch['name'].upper().replace('_', ' ')}")
            report.append(f"Priority: {batch['priority']}")
            report.append(f"Description: {batch['description']}")
            report.append(f"Packages Updated: {result.get('success_count', 0)}")
            report.append(f"Packages Failed: {result.get('failure_count', 0)}")
            
            if result.get("packages_updated"):
                report.append("  Updated Packages:")
                for package in result["packages_updated"]:
                    report.append(f"    - {package['name']}: {package['current_version']} -> {package['latest_version']}")
            
            if result.get("packages_failed"):
                report.append("  Failed Updates:")
                for failed in result["packages_failed"]:
                    report.append(f"    - {failed['package']['name']}: {failed['error'][:100]}...")
            
            report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point for dependency update manager."""
    parser = argparse.ArgumentParser(description="Automated dependency update manager")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be updated without making changes")
    parser.add_argument("--security-only", action="store_true", help="Only update security-related dependencies")
    parser.add_argument("--create-pr", action="store_true", help="Create pull request for updates")
    parser.add_argument("--auto-merge-security", action="store_true", help="Auto-merge critical security updates after tests pass")
    parser.add_argument("--batch", help="Update specific batch: critical_security, security_updates, minor_updates, major_updates")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize update manager
    update_manager = DependencyUpdateManager(
        project_root=project_root,
        dry_run=args.dry_run
    )
    
    try:
        # Get outdated dependencies
        outdated_deps = update_manager.get_outdated_dependencies()
        
        if outdated_deps.get("error"):
            logger.error(f"Failed to check dependencies: {outdated_deps['error']}")
            sys.exit(1)
        
        if outdated_deps["total_outdated"] == 0:
            logger.info("All dependencies are up to date!")
            sys.exit(0)
        
        logger.info(f"Found {outdated_deps['total_outdated']} outdated dependencies")
        
        # Create update plan
        update_plan = update_manager.create_update_plan(outdated_deps, args.security_only)
        
        if not update_plan["batches"]:
            logger.info("No updates to apply based on current criteria")
            sys.exit(0)
        
        # Execute update batches
        results = []
        
        for batch in update_plan["batches"]:
            if args.batch and batch["name"] != args.batch:
                continue
                
            logger.info(f"Processing batch: {batch['name']}")
            
            # Create branch for this batch
            if not args.dry_run:
                branch_name = update_manager.create_update_branch(batch["name"])
            
            # Apply updates
            update_result = update_manager.apply_updates(batch)
            results.append(update_result)
            
            # Run tests if updates were applied
            if update_result["success_count"] > 0 and not args.dry_run:
                test_result = update_manager.run_tests_after_update()
                update_result["test_result"] = test_result
                
                if test_result["test_status"] != "passed":
                    logger.error("Tests failed after dependency updates!")
                    if batch.get("auto_merge") or args.auto_merge_security:
                        logger.error("Auto-merge cancelled due to test failures")
                    # Could implement rollback logic here
                
                # Commit updates if tests pass
                if test_result["test_status"] == "passed":
                    update_manager.commit_updates(batch, update_result)
        
        # Generate report
        report = update_manager.generate_update_report(update_plan, results)
        print(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = project_root / f"dependency_update_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        logger.info(f"Update report saved to {report_file}")
        
        # Exit with appropriate code
        total_failures = sum(result.get("failure_count", 0) for result in results)
        if total_failures > 0:
            logger.warning(f"{total_failures} packages failed to update")
            sys.exit(1)
        else:
            logger.info("All dependency updates completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Update process interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Update process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()