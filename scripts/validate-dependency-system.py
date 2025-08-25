#!/usr/bin/env python3
"""
Dependency Management System Validation Script

This script validates the entire dependency management system implementation,
ensuring all components are properly configured and functioning correctly.

Usage:
    python scripts/validate-dependency-system.py [--fix] [--verbose]
"""

import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime
import toml
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencySystemValidator:
    """Validates the entire dependency management system."""
    
    def __init__(self, project_root: Path, fix_issues: bool = False):
        self.project_root = project_root
        self.fix_issues = fix_issues
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "unknown",
            "components": {},
            "issues": [],
            "recommendations": []
        }
    
    def validate_system(self) -> Dict[str, Any]:
        """Run comprehensive system validation."""
        logger.info("Starting comprehensive dependency system validation...")
        
        # Validate each component
        self.validate_configuration_files()
        self.validate_dependency_scripts()
        self.validate_github_workflows()
        self.validate_docker_configuration()
        self.validate_security_tools()
        self.validate_documentation()
        
        # Determine overall system status
        self.determine_system_status()
        
        return self.validation_results
    
    def validate_configuration_files(self):
        """Validate core configuration files."""
        logger.info("Validating configuration files...")
        
        config_results = {
            "pyproject_toml": self.validate_pyproject_toml(),
            "uv_lock": self.validate_uv_lock(),
            "dependabot_yml": self.validate_dependabot_config(),
            "requirements_files": self.validate_requirements_files()
        }
        
        self.validation_results["components"]["configuration"] = config_results
    
    def validate_pyproject_toml(self) -> Dict[str, Any]:
        """Validate pyproject.toml configuration."""
        pyproject_file = self.project_root / "pyproject.toml"
        
        if not pyproject_file.exists():
            return {"status": "missing", "issues": ["pyproject.toml file not found"]}
        
        try:
            with open(pyproject_file, 'r') as f:
                config = toml.load(f)
            
            issues = []
            
            # Check for security-related dependency groups
            dependency_groups = config.get("dependency-groups", {})
            required_groups = ["security-scan", "prod", "dev"]
            
            for group in required_groups:
                if group not in dependency_groups:
                    issues.append(f"Missing dependency group: {group}")
            
            # Check for security configuration
            if "tool" not in config or "dependency-security" not in config.get("tool", {}):
                issues.append("Missing [tool.dependency-security] section")
            
            # Check for security tool configurations
            security_tools = ["bandit", "safety", "pip-audit", "semgrep"]
            tool_config = config.get("tool", {})
            
            for tool in security_tools:
                if tool not in tool_config:
                    issues.append(f"Missing [tool.{tool}] configuration")
            
            # Check for enhanced dependency metadata
            project = config.get("project", {})
            if "authors" not in project:
                issues.append("Missing authors metadata")
            if "keywords" not in project:
                issues.append("Missing keywords metadata")
            if "classifiers" not in project:
                issues.append("Missing classifiers metadata")
            
            return {
                "status": "valid" if not issues else "issues",
                "issues": issues,
                "dependency_groups": len(dependency_groups),
                "security_tools_configured": len([t for t in security_tools if t in tool_config])
            }
            
        except Exception as e:
            return {"status": "error", "issues": [f"Failed to parse pyproject.toml: {e}"]}
    
    def validate_uv_lock(self) -> Dict[str, Any]:
        """Validate uv.lock file."""
        uv_lock_file = self.project_root / "uv.lock"
        
        if not uv_lock_file.exists():
            return {"status": "missing", "issues": ["uv.lock file not found"]}
        
        try:
            with open(uv_lock_file, 'r') as f:
                lock_content = toml.load(f)
            
            issues = []
            
            # Check for metadata
            if "metadata" not in lock_content:
                issues.append("Missing metadata section in uv.lock")
            
            # Check for packages
            packages = lock_content.get("package", [])
            if not packages:
                issues.append("No packages found in uv.lock")
            
            # Check hash coverage
            packages_with_hashes = 0
            for package in packages:
                if package.get("wheels"):
                    for wheel in package["wheels"]:
                        if wheel.get("hash"):
                            packages_with_hashes += 1
                            break
            
            hash_coverage = (packages_with_hashes / len(packages)) * 100 if packages else 0
            if hash_coverage < 90:
                issues.append(f"Low hash coverage: {hash_coverage:.1f}%")
            
            return {
                "status": "valid" if not issues else "issues",
                "issues": issues,
                "package_count": len(packages),
                "hash_coverage": hash_coverage
            }
            
        except Exception as e:
            return {"status": "error", "issues": [f"Failed to parse uv.lock: {e}"]}
    
    def validate_dependabot_config(self) -> Dict[str, Any]:
        """Validate Dependabot configuration."""
        dependabot_file = self.project_root / ".github" / "dependabot.yml"
        
        if not dependabot_file.exists():
            return {"status": "missing", "issues": ["dependabot.yml file not found"]}
        
        try:
            with open(dependabot_file, 'r') as f:
                config = yaml.safe_load(f)
            
            issues = []
            
            # Check version
            if config.get("version") != 2:
                issues.append("Dependabot version should be 2")
            
            # Check updates configuration
            updates = config.get("updates", [])
            if not updates:
                issues.append("No update configurations found")
            
            # Check for required ecosystems
            ecosystems = [update.get("package-ecosystem") for update in updates]
            required_ecosystems = ["pip", "docker", "github-actions"]
            
            for ecosystem in required_ecosystems:
                if ecosystem not in ecosystems:
                    issues.append(f"Missing ecosystem: {ecosystem}")
            
            # Check for security grouping
            has_security_groups = False
            for update in updates:
                groups = update.get("groups", {})
                for group_name in groups.keys():
                    if "security" in group_name.lower():
                        has_security_groups = True
                        break
            
            if not has_security_groups:
                issues.append("No security-focused dependency groups found")
            
            return {
                "status": "valid" if not issues else "issues",
                "issues": issues,
                "ecosystems": ecosystems,
                "update_configs": len(updates)
            }
            
        except Exception as e:
            return {"status": "error", "issues": [f"Failed to parse dependabot.yml: {e}"]}
    
    def validate_requirements_files(self) -> Dict[str, Any]:
        """Validate requirements files."""
        results = {}
        
        files_to_check = [
            "requirements.txt",
            "requirements-pinned.txt"
        ]
        
        for filename in files_to_check:
            file_path = self.project_root / filename
            
            if not file_path.exists():
                results[filename] = {"status": "missing", "issues": [f"{filename} not found"]}
                continue
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                packages = []
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        packages.append(line)
                
                issues = []
                pinned_count = 0
                
                for package in packages:
                    if "==" in package:
                        pinned_count += 1
                
                if filename == "requirements-pinned.txt":
                    if pinned_count != len(packages):
                        issues.append("Not all packages are pinned in requirements-pinned.txt")
                    
                    # Check for hashes
                    hash_count = sum(1 for pkg in packages if "--hash" in pkg)
                    if hash_count == 0:
                        issues.append("No package hashes found in requirements-pinned.txt")
                
                results[filename] = {
                    "status": "valid" if not issues else "issues",
                    "issues": issues,
                    "package_count": len(packages),
                    "pinned_count": pinned_count
                }
                
            except Exception as e:
                results[filename] = {"status": "error", "issues": [f"Failed to parse {filename}: {e}"]}
        
        return results
    
    def validate_dependency_scripts(self):
        """Validate dependency management scripts."""
        logger.info("Validating dependency scripts...")
        
        scripts_dir = self.project_root / "scripts" / "deps"
        required_scripts = [
            "deps-scan.py",
            "deps-update.py", 
            "deps-audit.py",
            "deps-lock.py"
        ]
        
        script_results = {}
        
        for script_name in required_scripts:
            script_path = scripts_dir / script_name
            script_results[script_name] = self.validate_script(script_path)
        
        self.validation_results["components"]["scripts"] = script_results
    
    def validate_script(self, script_path: Path) -> Dict[str, Any]:
        """Validate individual script."""
        if not script_path.exists():
            return {"status": "missing", "issues": [f"Script not found: {script_path}"]}
        
        issues = []
        
        # Check if executable
        if not script_path.stat().st_mode & 0o111:
            issues.append("Script is not executable")
        
        # Check if it has proper shebang
        try:
            with open(script_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line.startswith('#!/usr/bin/env python'):
                    issues.append("Missing or incorrect shebang")
        except Exception as e:
            issues.append(f"Failed to read script: {e}")
        
        # Try to import the script (basic syntax check)
        try:
            # This is a simple syntax validation
            subprocess.run(
                [sys.executable, "-m", "py_compile", str(script_path)],
                check=True,
                capture_output=True,
                timeout=30
            )
        except subprocess.CalledProcessError:
            issues.append("Script has syntax errors")
        except subprocess.TimeoutExpired:
            issues.append("Script validation timed out")
        except Exception as e:
            issues.append(f"Failed to validate script: {e}")
        
        return {
            "status": "valid" if not issues else "issues",
            "issues": issues,
            "executable": script_path.stat().st_mode & 0o111 != 0
        }
    
    def validate_github_workflows(self):
        """Validate GitHub Actions workflows."""
        logger.info("Validating GitHub workflows...")
        
        workflows_dir = self.project_root / ".github" / "workflows"
        
        if not workflows_dir.exists():
            self.validation_results["components"]["workflows"] = {
                "status": "missing",
                "issues": ["GitHub workflows directory not found"]
            }
            return
        
        workflow_results = {}
        
        # Check for dependency management workflow
        dep_workflow = workflows_dir / "dependency-management.yml"
        if dep_workflow.exists():
            workflow_results["dependency-management"] = self.validate_workflow_file(dep_workflow)
        else:
            workflow_results["dependency-management"] = {
                "status": "missing",
                "issues": ["dependency-management.yml workflow not found"]
            }
        
        # Check CI workflow has dependency scanning
        ci_workflow = workflows_dir / "ci.yml"
        if ci_workflow.exists():
            workflow_results["ci"] = self.validate_ci_workflow(ci_workflow)
        else:
            workflow_results["ci"] = {
                "status": "missing", 
                "issues": ["ci.yml workflow not found"]
            }
        
        self.validation_results["components"]["workflows"] = workflow_results
    
    def validate_workflow_file(self, workflow_path: Path) -> Dict[str, Any]:
        """Validate individual workflow file."""
        try:
            with open(workflow_path, 'r') as f:
                workflow = yaml.safe_load(f)
            
            issues = []
            
            # Check for required fields
            if "name" not in workflow:
                issues.append("Missing workflow name")
            
            if "on" not in workflow:
                issues.append("Missing workflow triggers")
            
            if "jobs" not in workflow:
                issues.append("Missing workflow jobs")
            
            # Check for security-related jobs
            jobs = workflow.get("jobs", {})
            security_jobs = [job for job in jobs.keys() if "security" in job.lower() or "audit" in job.lower()]
            
            if not security_jobs:
                issues.append("No security-related jobs found")
            
            return {
                "status": "valid" if not issues else "issues",
                "issues": issues,
                "job_count": len(jobs),
                "security_jobs": len(security_jobs)
            }
            
        except Exception as e:
            return {"status": "error", "issues": [f"Failed to parse workflow: {e}"]}
    
    def validate_ci_workflow(self, ci_path: Path) -> Dict[str, Any]:
        """Validate CI workflow has dependency scanning."""
        try:
            with open(ci_path, 'r') as f:
                content = f.read()
            
            issues = []
            
            # Check for dependency scanning integration
            if "deps-scan.py" not in content:
                issues.append("CI workflow doesn't include comprehensive dependency scanning")
            
            if "deps-lock.py" not in content:
                issues.append("CI workflow doesn't include lock file validation")
            
            # Check for security scanning tools
            security_tools = ["bandit", "safety", "semgrep"]
            for tool in security_tools:
                if tool not in content:
                    issues.append(f"CI workflow missing {tool} scanning")
            
            return {
                "status": "valid" if not issues else "issues",
                "issues": issues,
                "has_dependency_scanning": "deps-scan.py" in content
            }
            
        except Exception as e:
            return {"status": "error", "issues": [f"Failed to validate CI workflow: {e}"]}
    
    def validate_docker_configuration(self):
        """Validate Docker configuration."""
        logger.info("Validating Docker configuration...")
        
        dockerfile = self.project_root / "Dockerfile"
        
        if not dockerfile.exists():
            self.validation_results["components"]["docker"] = {
                "status": "missing",
                "issues": ["Dockerfile not found"]
            }
            return
        
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            issues = []
            
            # Check for security features
            if "USER" not in content:
                issues.append("Dockerfile doesn't use non-root user")
            
            if "HEALTHCHECK" not in content:
                issues.append("Dockerfile missing health check")
            
            if "uv" not in content:
                issues.append("Dockerfile doesn't use uv for dependency management")
            
            # Check for security scanning in build
            if "deps-scan.py" not in content and "security" not in content.lower():
                issues.append("Dockerfile doesn't include security scanning")
            
            # Check for multi-stage build
            stage_count = content.count("FROM ")
            if stage_count < 2:
                issues.append("Dockerfile should use multi-stage build")
            
            self.validation_results["components"]["docker"] = {
                "status": "valid" if not issues else "issues",
                "issues": issues,
                "stages": stage_count,
                "has_security_features": len(issues) < 3
            }
            
        except Exception as e:
            self.validation_results["components"]["docker"] = {
                "status": "error",
                "issues": [f"Failed to validate Dockerfile: {e}"]
            }
    
    def validate_security_tools(self):
        """Validate security tools installation and configuration."""
        logger.info("Validating security tools...")
        
        tools_to_check = [
            ("bandit", "bandit --version"),
            ("safety", "safety --version"),
            ("pip-audit", "pip-audit --version"),
            ("semgrep", "semgrep --version")
        ]
        
        tool_results = {}
        
        for tool_name, version_cmd in tools_to_check:
            try:
                result = subprocess.run(
                    version_cmd.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode == 0:
                    tool_results[tool_name] = {
                        "status": "available",
                        "version": result.stdout.strip()
                    }
                else:
                    tool_results[tool_name] = {
                        "status": "error",
                        "issues": [f"Tool returned error code {result.returncode}"]
                    }
                    
            except subprocess.TimeoutExpired:
                tool_results[tool_name] = {
                    "status": "timeout",
                    "issues": ["Version check timed out"]
                }
            except FileNotFoundError:
                tool_results[tool_name] = {
                    "status": "missing",
                    "issues": ["Tool not installed"]
                }
            except Exception as e:
                tool_results[tool_name] = {
                    "status": "error", 
                    "issues": [f"Failed to check tool: {e}"]
                }
        
        self.validation_results["components"]["security_tools"] = tool_results
    
    def validate_documentation(self):
        """Validate documentation completeness."""
        logger.info("Validating documentation...")
        
        docs_dir = self.project_root / "docs"
        required_docs = [
            "DEPENDENCY_MANAGEMENT.md"
        ]
        
        doc_results = {}
        
        for doc_name in required_docs:
            doc_path = docs_dir / doc_name
            
            if not doc_path.exists():
                doc_results[doc_name] = {
                    "status": "missing",
                    "issues": [f"Documentation file not found: {doc_name}"]
                }
                continue
            
            try:
                with open(doc_path, 'r') as f:
                    content = f.read()
                
                issues = []
                
                # Check for key sections
                required_sections = [
                    "## Overview",
                    "## Tools and Scripts", 
                    "## Security Features",
                    "## Usage Guide",
                    "## Troubleshooting"
                ]
                
                for section in required_sections:
                    if section not in content:
                        issues.append(f"Missing section: {section}")
                
                # Check for script references
                scripts = ["deps-scan.py", "deps-update.py", "deps-audit.py", "deps-lock.py"]
                for script in scripts:
                    if script not in content:
                        issues.append(f"Missing script documentation: {script}")
                
                doc_results[doc_name] = {
                    "status": "valid" if not issues else "issues",
                    "issues": issues,
                    "word_count": len(content.split()),
                    "sections_found": len([s for s in required_sections if s in content])
                }
                
            except Exception as e:
                doc_results[doc_name] = {
                    "status": "error",
                    "issues": [f"Failed to validate documentation: {e}"]
                }
        
        self.validation_results["components"]["documentation"] = doc_results
    
    def determine_system_status(self):
        """Determine overall system status."""
        components = self.validation_results["components"]
        total_issues = []
        
        for component_name, component_data in components.items():
            if isinstance(component_data, dict):
                if component_data.get("status") == "error":
                    total_issues.extend(component_data.get("issues", []))
                elif component_data.get("status") == "issues":
                    total_issues.extend(component_data.get("issues", []))
                elif isinstance(component_data, dict):
                    # Handle nested components
                    for subcomponent_name, subcomponent_data in component_data.items():
                        if isinstance(subcomponent_data, dict):
                            if subcomponent_data.get("status") in ["error", "issues", "missing"]:
                                total_issues.extend(subcomponent_data.get("issues", []))
        
        self.validation_results["issues"] = total_issues
        
        if not total_issues:
            self.validation_results["system_status"] = "healthy"
        elif len(total_issues) <= 5:
            self.validation_results["system_status"] = "warnings"
        else:
            self.validation_results["system_status"] = "critical"
    
    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        results = self.validation_results
        
        report = []
        report.append("=" * 80)
        report.append("MIMIR DEPENDENCY MANAGEMENT SYSTEM VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Validation Date: {results['timestamp']}")
        report.append(f"System Status: {results['system_status'].upper()}")
        report.append("")
        
        # Summary
        report.append("SUMMARY:")
        total_issues = len(results['issues'])
        report.append(f"  Total Issues Found: {total_issues}")
        report.append("")
        
        # Component Status
        report.append("COMPONENT STATUS:")
        components = results.get("components", {})
        
        for component_name, component_data in components.items():
            report.append(f"  {component_name.upper()}:")
            
            if isinstance(component_data, dict) and "status" in component_data:
                status = component_data["status"]
                report.append(f"    Status: {status}")
                
                if component_data.get("issues"):
                    report.append(f"    Issues: {len(component_data['issues'])}")
                    for issue in component_data["issues"][:3]:  # Show first 3 issues
                        report.append(f"      - {issue}")
                    if len(component_data["issues"]) > 3:
                        report.append(f"      ... and {len(component_data['issues']) - 3} more")
            else:
                # Handle nested components
                for subcomponent_name, subcomponent_data in component_data.items():
                    if isinstance(subcomponent_data, dict):
                        status = subcomponent_data.get("status", "unknown")
                        report.append(f"    {subcomponent_name}: {status}")
            
            report.append("")
        
        # Critical Issues
        if results['issues']:
            report.append("ISSUES REQUIRING ATTENTION:")
            for i, issue in enumerate(results['issues'][:10], 1):  # Show first 10
                report.append(f"  {i}. {issue}")
            
            if len(results['issues']) > 10:
                report.append(f"  ... and {len(results['issues']) - 10} more issues")
            
            report.append("")
        
        # Recommendations
        if results.get('recommendations'):
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(results['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        report.append("=" * 80)
        report.append("END VALIDATION REPORT")
        report.append("=" * 80)
        
        return "\n".join(report)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate dependency management system")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix issues automatically")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--output", "-o", help="Save report to file")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    # Initialize validator
    validator = DependencySystemValidator(
        project_root=project_root,
        fix_issues=args.fix
    )
    
    try:
        # Run validation
        results = validator.validate_system()
        
        # Generate report
        report = validator.generate_report()
        print(report)
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {args.output}")
        
        # Auto-save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_filename = f"dependency_system_validation_{timestamp}.txt"
        auto_filepath = project_root / auto_filename
        
        with open(auto_filepath, 'w') as f:
            f.write(report)
        logger.info(f"Validation report saved to {auto_filepath}")
        
        # Exit with appropriate code
        system_status = results["system_status"]
        if system_status == "critical":
            logger.error("Dependency system validation failed with critical issues!")
            sys.exit(1)
        elif system_status == "warnings":
            logger.warning("Dependency system validation completed with warnings")
            sys.exit(2)
        else:
            logger.info("Dependency system validation passed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()