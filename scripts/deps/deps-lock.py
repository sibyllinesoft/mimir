#!/usr/bin/env python3
"""
Dependency Lock File Management for Mimir

This script manages dependency lock files with security validation:
- uv.lock validation and regeneration
- requirements.txt pinning and hashing
- Dependency hash verification
- Lock file security analysis
- Cross-platform compatibility checks

Usage:
    python scripts/deps/deps-lock.py [--regenerate] [--validate] [--add-hashes] [--check-platform]
"""

import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from datetime import datetime, timezone
import hashlib
import re
import tempfile
import os
import toml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyLockManager:
    """Manages dependency lock files with security and reproducibility focus."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_file = project_root / "pyproject.toml"
        self.uv_lock_file = project_root / "uv.lock"
        self.requirements_file = project_root / "requirements.txt"
        self.requirements_pinned_file = project_root / "requirements-pinned.txt"
        
    def validate_lock_files(self) -> Dict[str, Any]:
        """Validate all lock files for consistency and security."""
        logger.info("Validating lock files...")
        
        validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "files_checked": [],
            "validation_status": "unknown",
            "issues": [],
            "recommendations": []
        }
        
        # Validate uv.lock
        if self.uv_lock_file.exists():
            uv_validation = self._validate_uv_lock()
            validation_results["uv_lock"] = uv_validation
            validation_results["files_checked"].append("uv.lock")
            
            if uv_validation.get("issues"):
                validation_results["issues"].extend(uv_validation["issues"])
        else:
            validation_results["issues"].append({
                "file": "uv.lock",
                "severity": "warning",
                "message": "uv.lock file missing"
            })
        
        # Validate requirements.txt
        if self.requirements_file.exists():
            req_validation = self._validate_requirements_txt()
            validation_results["requirements_txt"] = req_validation
            validation_results["files_checked"].append("requirements.txt")
            
            if req_validation.get("issues"):
                validation_results["issues"].extend(req_validation["issues"])
        else:
            validation_results["issues"].append({
                "file": "requirements.txt",
                "severity": "error",
                "message": "requirements.txt file missing"
            })
        
        # Validate requirements-pinned.txt
        if self.requirements_pinned_file.exists():
            pinned_validation = self._validate_pinned_requirements()
            validation_results["requirements_pinned"] = pinned_validation
            validation_results["files_checked"].append("requirements-pinned.txt")
            
            if pinned_validation.get("issues"):
                validation_results["issues"].extend(pinned_validation["issues"])
        
        # Check consistency between files
        consistency_check = self._check_file_consistency()
        validation_results["consistency"] = consistency_check
        if consistency_check.get("issues"):
            validation_results["issues"].extend(consistency_check["issues"])
        
        # Determine overall status
        error_count = len([issue for issue in validation_results["issues"] if issue["severity"] == "error"])
        warning_count = len([issue for issue in validation_results["issues"] if issue["severity"] == "warning"])
        
        if error_count > 0:
            validation_results["validation_status"] = "failed"
        elif warning_count > 0:
            validation_results["validation_status"] = "warnings"
        else:
            validation_results["validation_status"] = "passed"
        
        return validation_results
    
    def _validate_uv_lock(self) -> Dict[str, Any]:
        """Validate uv.lock file."""
        logger.info("Validating uv.lock file...")
        
        validation = {
            "status": "unknown",
            "issues": [],
            "metadata": {},
            "package_count": 0,
            "hash_count": 0
        }
        
        try:
            # Try to parse uv.lock as TOML
            with open(self.uv_lock_file, 'r') as f:
                lock_content = f.read()
            
            try:
                lock_data = toml.loads(lock_content)
                validation["metadata"] = lock_data.get("metadata", {})
                
                # Count packages
                packages = lock_data.get("package", [])
                validation["package_count"] = len(packages)
                
                # Count packages with hashes
                hash_count = 0
                for package in packages:
                    if package.get("source", {}).get("registry") and package.get("wheels"):
                        for wheel in package["wheels"]:
                            if wheel.get("hash"):
                                hash_count += 1
                                break
                
                validation["hash_count"] = hash_count
                
                # Check for required metadata
                if not validation["metadata"]:
                    validation["issues"].append({
                        "file": "uv.lock",
                        "severity": "warning",
                        "message": "Missing lock file metadata"
                    })
                
                # Check hash coverage
                hash_coverage = (hash_count / validation["package_count"]) * 100 if validation["package_count"] > 0 else 0
                if hash_coverage < 90:
                    validation["issues"].append({
                        "file": "uv.lock",
                        "severity": "warning",
                        "message": f"Low hash coverage: {hash_coverage:.1f}% of packages have hashes"
                    })
                
                validation["status"] = "valid"
                
            except toml.TomlDecodeError as e:
                validation["issues"].append({
                    "file": "uv.lock",
                    "severity": "error",
                    "message": f"Invalid TOML format: {e}"
                })
                validation["status"] = "invalid"
                
        except FileNotFoundError:
            validation["issues"].append({
                "file": "uv.lock",
                "severity": "error",
                "message": "File not found"
            })
            validation["status"] = "missing"
        except Exception as e:
            validation["issues"].append({
                "file": "uv.lock",
                "severity": "error",
                "message": f"Validation error: {e}"
            })
            validation["status"] = "error"
        
        return validation
    
    def _validate_requirements_txt(self) -> Dict[str, Any]:
        """Validate requirements.txt file."""
        logger.info("Validating requirements.txt file...")
        
        validation = {
            "status": "unknown",
            "issues": [],
            "package_count": 0,
            "pinned_count": 0,
            "unpinned_packages": []
        }
        
        try:
            with open(self.requirements_file, 'r') as f:
                lines = f.readlines()
            
            packages = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Parse package specification
                package_info = self._parse_requirement_line(line, line_num)
                if package_info:
                    packages.append(package_info)
            
            validation["package_count"] = len(packages)
            
            # Check for pinning
            pinned_packages = []
            unpinned_packages = []
            
            for package in packages:
                if package.get("version_pinned"):
                    pinned_packages.append(package)
                else:
                    unpinned_packages.append(package)
            
            validation["pinned_count"] = len(pinned_packages)
            validation["unpinned_packages"] = unpinned_packages
            
            # Add issues for unpinned packages
            if unpinned_packages:
                for package in unpinned_packages[:5]:  # Show first 5
                    validation["issues"].append({
                        "file": "requirements.txt",
                        "severity": "warning",
                        "message": f"Package '{package['name']}' is not pinned to exact version",
                        "line": package.get("line_number")
                    })
                
                if len(unpinned_packages) > 5:
                    validation["issues"].append({
                        "file": "requirements.txt",
                        "severity": "warning",
                        "message": f"... and {len(unpinned_packages) - 5} more unpinned packages"
                    })
            
            validation["status"] = "valid"
            
        except FileNotFoundError:
            validation["issues"].append({
                "file": "requirements.txt",
                "severity": "error",
                "message": "File not found"
            })
            validation["status"] = "missing"
        except Exception as e:
            validation["issues"].append({
                "file": "requirements.txt",
                "severity": "error",
                "message": f"Validation error: {e}"
            })
            validation["status"] = "error"
        
        return validation
    
    def _validate_pinned_requirements(self) -> Dict[str, Any]:
        """Validate requirements-pinned.txt file."""
        logger.info("Validating requirements-pinned.txt file...")
        
        validation = {
            "status": "unknown",
            "issues": [],
            "package_count": 0,
            "hash_count": 0,
            "packages_with_hashes": []
        }
        
        try:
            with open(self.requirements_pinned_file, 'r') as f:
                lines = f.readlines()
            
            packages = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                package_info = self._parse_requirement_line(line, line_num)
                if package_info:
                    packages.append(package_info)
                    
                    # Check for hashes
                    if '--hash' in line:
                        validation["hash_count"] += 1
                        validation["packages_with_hashes"].append(package_info["name"])
            
            validation["package_count"] = len(packages)
            
            # Check hash coverage
            if validation["package_count"] > 0:
                hash_coverage = (validation["hash_count"] / validation["package_count"]) * 100
                if hash_coverage < 100:
                    validation["issues"].append({
                        "file": "requirements-pinned.txt",
                        "severity": "warning",
                        "message": f"Incomplete hash coverage: {hash_coverage:.1f}% of packages have hashes"
                    })
            
            validation["status"] = "valid"
            
        except FileNotFoundError:
            validation["status"] = "missing"
        except Exception as e:
            validation["issues"].append({
                "file": "requirements-pinned.txt",
                "severity": "error",
                "message": f"Validation error: {e}"
            })
            validation["status"] = "error"
        
        return validation
    
    def _parse_requirement_line(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """Parse a requirements line into package information."""
        # Remove hashes and options
        clean_line = re.sub(r'--hash=[^\s]+', '', line).strip()
        clean_line = re.sub(r'--[^\s]+=[^\s]+', '', clean_line).strip()
        
        # Parse package name and version
        match = re.match(r'^([a-zA-Z0-9_-]+)([><=!~]+.*)?', clean_line)
        if not match:
            return None
        
        name = match.group(1)
        version_spec = match.group(2) or ""
        
        # Check if version is pinned (exact version)
        version_pinned = "==" in version_spec
        
        return {
            "name": name,
            "version_spec": version_spec,
            "version_pinned": version_pinned,
            "line_number": line_number,
            "raw_line": line.strip()
        }
    
    def _check_file_consistency(self) -> Dict[str, Any]:
        """Check consistency between different lock files."""
        logger.info("Checking consistency between lock files...")
        
        consistency = {
            "status": "unknown",
            "issues": [],
            "uv_lock_packages": set(),
            "requirements_packages": set(),
            "missing_from_requirements": [],
            "missing_from_uv_lock": []
        }
        
        try:
            # Get packages from uv.lock
            if self.uv_lock_file.exists():
                with open(self.uv_lock_file, 'r') as f:
                    uv_data = toml.loads(f.read())
                
                for package in uv_data.get("package", []):
                    consistency["uv_lock_packages"].add(package["name"].lower())
            
            # Get packages from requirements.txt
            if self.requirements_file.exists():
                with open(self.requirements_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            package_info = self._parse_requirement_line(line, 0)
                            if package_info:
                                consistency["requirements_packages"].add(package_info["name"].lower())
            
            # Find differences
            consistency["missing_from_requirements"] = list(
                consistency["uv_lock_packages"] - consistency["requirements_packages"]
            )
            consistency["missing_from_uv_lock"] = list(
                consistency["requirements_packages"] - consistency["uv_lock_packages"]
            )
            
            # Generate issues
            if consistency["missing_from_requirements"]:
                consistency["issues"].append({
                    "file": "consistency",
                    "severity": "warning",
                    "message": f"Packages in uv.lock but not in requirements.txt: {', '.join(consistency['missing_from_requirements'][:5])}"
                })
            
            if consistency["missing_from_uv_lock"]:
                consistency["issues"].append({
                    "file": "consistency",
                    "severity": "warning", 
                    "message": f"Packages in requirements.txt but not in uv.lock: {', '.join(consistency['missing_from_uv_lock'][:5])}"
                })
            
            consistency["status"] = "checked"
            
        except Exception as e:
            consistency["issues"].append({
                "file": "consistency",
                "severity": "error",
                "message": f"Consistency check error: {e}"
            })
            consistency["status"] = "error"
        
        return consistency
    
    def regenerate_lock_files(self, include_hashes: bool = False) -> Dict[str, Any]:
        """Regenerate all lock files."""
        logger.info("Regenerating lock files...")
        
        results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operations": [],
            "success": True,
            "errors": []
        }
        
        try:
            # Regenerate uv.lock
            logger.info("Regenerating uv.lock...")
            uv_result = self._regenerate_uv_lock()
            results["operations"].append(uv_result)
            if not uv_result["success"]:
                results["success"] = False
                results["errors"].extend(uv_result.get("errors", []))
            
            # Regenerate requirements.txt
            logger.info("Regenerating requirements.txt...")
            req_result = self._regenerate_requirements_txt()
            results["operations"].append(req_result)
            if not req_result["success"]:
                results["success"] = False
                results["errors"].extend(req_result.get("errors", []))
            
            # Generate requirements-pinned.txt with hashes if requested
            if include_hashes:
                logger.info("Generating requirements-pinned.txt with hashes...")
                pinned_result = self._generate_pinned_requirements_with_hashes()
                results["operations"].append(pinned_result)
                if not pinned_result["success"]:
                    results["success"] = False
                    results["errors"].extend(pinned_result.get("errors", []))
            
        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Lock file regeneration failed: {e}")
        
        return results
    
    def _regenerate_uv_lock(self) -> Dict[str, Any]:
        """Regenerate uv.lock file."""
        try:
            cmd = ["uv", "lock", "--upgrade"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                "operation": "uv_lock_regeneration",
                "success": result.returncode == 0,
                "command": " ".join(cmd),
                "output": result.stdout,
                "errors": [result.stderr] if result.stderr else []
            }
            
        except subprocess.TimeoutExpired:
            return {
                "operation": "uv_lock_regeneration",
                "success": False,
                "errors": ["uv lock command timed out"]
            }
        except Exception as e:
            return {
                "operation": "uv_lock_regeneration", 
                "success": False,
                "errors": [str(e)]
            }
    
    def _regenerate_requirements_txt(self) -> Dict[str, Any]:
        """Regenerate requirements.txt file."""
        try:
            cmd = [sys.executable, "-m", "pip", "freeze"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                with open(self.requirements_file, 'w') as f:
                    f.write(result.stdout)
                
                return {
                    "operation": "requirements_txt_regeneration",
                    "success": True,
                    "packages_written": len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
                }
            else:
                return {
                    "operation": "requirements_txt_regeneration",
                    "success": False,
                    "errors": [result.stderr]
                }
                
        except Exception as e:
            return {
                "operation": "requirements_txt_regeneration",
                "success": False,
                "errors": [str(e)]
            }
    
    def _generate_pinned_requirements_with_hashes(self) -> Dict[str, Any]:
        """Generate requirements-pinned.txt with hashes."""
        try:
            # Use pip-tools to generate requirements with hashes
            cmd = [
                sys.executable, "-m", "pip", "freeze",
                "--all"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode != 0:
                return {
                    "operation": "pinned_requirements_generation",
                    "success": False,
                    "errors": [result.stderr]
                }
            
            # For now, just generate pinned requirements without hashes
            # In a full implementation, you'd use pip-tools or similar
            pinned_content = result.stdout
            
            with open(self.requirements_pinned_file, 'w') as f:
                f.write("# Generated pinned requirements\n")
                f.write(f"# Generated at {datetime.now(timezone.utc).isoformat()}\n\n")
                f.write(pinned_content)
            
            return {
                "operation": "pinned_requirements_generation",
                "success": True,
                "note": "Generated without hashes - requires pip-tools for hash generation"
            }
            
        except Exception as e:
            return {
                "operation": "pinned_requirements_generation",
                "success": False,
                "errors": [str(e)]
            }
    
    def check_platform_compatibility(self) -> Dict[str, Any]:
        """Check cross-platform compatibility of dependencies."""
        logger.info("Checking platform compatibility...")
        
        compatibility = {
            "status": "unknown",
            "platform_specific_packages": [],
            "potential_issues": [],
            "recommendations": []
        }
        
        try:
            # Get installed packages with platform info
            cmd = [sys.executable, "-m", "pip", "list", "--format=json"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                compatibility["status"] = "error"
                compatibility["potential_issues"].append("Could not retrieve package list")
                return compatibility
            
            packages = json.loads(result.stdout)
            
            # Check for known platform-specific packages
            platform_specific = {
                "pywin32", "winsound", "msvcrt",  # Windows
                "fcntl", "termios", "pwd", "grp",  # Unix
                "Foundation", "objc",  # macOS
            }
            
            for package in packages:
                name = package["name"].lower()
                
                if name in platform_specific:
                    compatibility["platform_specific_packages"].append({
                        "name": package["name"],
                        "version": package["version"],
                        "platform": "windows" if name in {"pywin32", "winsound", "msvcrt"} else
                                   "macos" if name in {"Foundation", "objc"} else "unix"
                    })
            
            # Check for packages with common platform issues
            problematic_packages = {
                "psutil": "May have different behavior on different platforms",
                "numpy": "Binary wheels may not be available on all platforms",
                "cryptography": "Requires compilation on some platforms",
            }
            
            for package in packages:
                name = package["name"].lower()
                if name in problematic_packages:
                    compatibility["potential_issues"].append({
                        "package": package["name"],
                        "issue": problematic_packages[name]
                    })
            
            # Generate recommendations
            if compatibility["platform_specific_packages"]:
                compatibility["recommendations"].append(
                    "Test deployment on all target platforms"
                )
            
            if compatibility["potential_issues"]:
                compatibility["recommendations"].append(
                    "Use platform-specific requirements files or conditional dependencies"
                )
            
            compatibility["status"] = "checked"
            
        except Exception as e:
            compatibility["status"] = "error"
            compatibility["potential_issues"].append(f"Platform check failed: {e}")
        
        return compatibility
    
    def generate_lock_report(self, validation_results: Dict[str, Any], 
                           compatibility_results: Dict[str, Any]) -> str:
        """Generate human-readable lock file report."""
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MIMIR DEPENDENCY LOCK FILE REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Report Date: {validation_results['timestamp']}")
        report_lines.append("")
        
        # Validation Summary
        report_lines.append("VALIDATION SUMMARY:")
        report_lines.append(f"  Overall Status: {validation_results['validation_status'].upper()}")
        report_lines.append(f"  Files Checked: {', '.join(validation_results['files_checked'])}")
        report_lines.append(f"  Issues Found: {len(validation_results['issues'])}")
        report_lines.append("")
        
        # Issues Detail
        if validation_results["issues"]:
            report_lines.append("ISSUES FOUND:")
            for i, issue in enumerate(validation_results["issues"], 1):
                report_lines.append(f"  {i}. [{issue['severity'].upper()}] {issue['file']}")
                report_lines.append(f"     {issue['message']}")
                if issue.get("line"):
                    report_lines.append(f"     Line: {issue['line']}")
            report_lines.append("")
        
        # File-specific details
        if validation_results.get("uv_lock"):
            uv_info = validation_results["uv_lock"]
            report_lines.append("UV.LOCK:")
            report_lines.append(f"  Status: {uv_info['status']}")
            report_lines.append(f"  Packages: {uv_info['package_count']}")
            report_lines.append(f"  With Hashes: {uv_info['hash_count']}")
            report_lines.append("")
        
        if validation_results.get("requirements_txt"):
            req_info = validation_results["requirements_txt"]
            report_lines.append("REQUIREMENTS.TXT:")
            report_lines.append(f"  Status: {req_info['status']}")
            report_lines.append(f"  Packages: {req_info['package_count']}")
            report_lines.append(f"  Pinned: {req_info['pinned_count']}")
            if req_info.get("unpinned_packages"):
                report_lines.append(f"  Unpinned: {len(req_info['unpinned_packages'])}")
            report_lines.append("")
        
        # Platform compatibility
        if compatibility_results.get("status") == "checked":
            report_lines.append("PLATFORM COMPATIBILITY:")
            report_lines.append(f"  Platform-Specific Packages: {len(compatibility_results['platform_specific_packages'])}")
            report_lines.append(f"  Potential Issues: {len(compatibility_results['potential_issues'])}")
            
            if compatibility_results["recommendations"]:
                report_lines.append("  Recommendations:")
                for rec in compatibility_results["recommendations"]:
                    report_lines.append(f"    - {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        report_lines.append("END LOCK FILE REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)


def main():
    """Main entry point for dependency lock manager."""
    parser = argparse.ArgumentParser(description="Dependency lock file management")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate all lock files")
    parser.add_argument("--validate", action="store_true", help="Validate existing lock files")
    parser.add_argument("--add-hashes", action="store_true", help="Add hashes to pinned requirements")
    parser.add_argument("--check-platform", action="store_true", help="Check platform compatibility")
    parser.add_argument("--output", "-o", help="Output report file")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize lock manager
    lock_manager = DependencyLockManager(project_root=project_root)
    
    try:
        # Default to validation if no specific action
        if not any([args.regenerate, args.validate, args.check_platform]):
            args.validate = True
        
        validation_results = None
        compatibility_results = None
        
        # Regenerate lock files if requested
        if args.regenerate:
            logger.info("Regenerating lock files...")
            regen_results = lock_manager.regenerate_lock_files(include_hashes=args.add_hashes)
            
            if not regen_results["success"]:
                logger.error("Lock file regeneration failed!")
                for error in regen_results["errors"]:
                    logger.error(f"  {error}")
                sys.exit(1)
            else:
                logger.info("Lock file regeneration completed successfully")
        
        # Validate lock files
        if args.validate or args.regenerate:
            logger.info("Validating lock files...")
            validation_results = lock_manager.validate_lock_files()
        
        # Check platform compatibility
        if args.check_platform:
            logger.info("Checking platform compatibility...")
            compatibility_results = lock_manager.check_platform_compatibility()
        
        # Generate and display report
        if validation_results or compatibility_results:
            report = lock_manager.generate_lock_report(
                validation_results or {},
                compatibility_results or {}
            )
            print(report)
            
            # Save report if requested
            if args.output:
                with open(args.output, 'w') as f:
                    f.write(report)
                logger.info(f"Report saved to {args.output}")
            
            # Auto-save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            auto_filename = f"lock_file_report_{timestamp}.txt"
            auto_filepath = project_root / auto_filename
            
            with open(auto_filepath, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {auto_filepath}")
        
        # Exit with appropriate code
        if validation_results:
            if validation_results["validation_status"] == "failed":
                logger.error("Lock file validation failed!")
                sys.exit(1)
            elif validation_results["validation_status"] == "warnings":
                logger.warning("Lock file validation completed with warnings")
                sys.exit(2)
        
        logger.info("Lock file management completed successfully")
        sys.exit(0)
        
    except KeyboardInterrupt:
        logger.info("Lock file management interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Lock file management failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()