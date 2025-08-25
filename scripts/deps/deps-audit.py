#!/usr/bin/env python3
"""
Comprehensive Dependency Audit System for Mimir

This script performs detailed dependency analysis including:
- License compliance checking
- Supply chain security analysis
- Dependency tree analysis
- Bloat detection and optimization
- Security risk assessment

Usage:
    python scripts/deps/deps-audit.py [--format json|text|html] [--output file] [--check-licenses] [--analyze-bloat]
"""

import sys
import subprocess
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timezone
import re
import tempfile
import os
import requests
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DependencyAuditor:
    """Comprehensive dependency auditor with security and compliance analysis."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.pyproject_file = project_root / "pyproject.toml"
        self.requirements_file = project_root / "requirements.txt"
        
        # License compatibility matrix
        self.license_compatibility = {
            "MIT": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "restrictive": False},
            "Apache-2.0": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "restrictive": False},
            "BSD": {"compatible": ["MIT", "BSD", "Apache-2.0", "ISC"], "restrictive": False},
            "GPL-3.0": {"compatible": ["GPL-3.0"], "restrictive": True},
            "GPL-2.0": {"compatible": ["GPL-2.0", "GPL-3.0"], "restrictive": True},
            "LGPL": {"compatible": ["MIT", "BSD", "Apache-2.0", "LGPL", "GPL"], "restrictive": True},
        }
        
        # Known problematic packages
        self.problematic_packages = {
            "requests": {"alternatives": ["httpx"], "issues": ["sync_only", "heavy"]},
            "urllib3": {"issues": ["security_history"], "monitor": True},
            "pillow": {"issues": ["security_history"], "monitor": True},
        }
    
    def get_installed_packages(self) -> Dict[str, Any]:
        """Get list of all installed packages with detailed information."""
        logger.info("Analyzing installed packages...")
        
        try:
            # Get basic package list
            cmd = [sys.executable, "-m", "pip", "list", "--format=json"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to get package list: {result.stderr}")
                return {"packages": [], "error": result.stderr}
            
            packages = json.loads(result.stdout) if result.stdout else []
            
            # Enrich with additional metadata
            enriched_packages = []
            for package in packages:
                enriched = self._enrich_package_info(package)
                enriched_packages.append(enriched)
            
            return {
                "packages": enriched_packages,
                "total_count": len(enriched_packages),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except subprocess.TimeoutExpired:
            logger.error("Package analysis timed out")
            return {"packages": [], "error": "Timeout"}
        except Exception as e:
            logger.error(f"Failed to analyze packages: {e}")
            return {"packages": [], "error": str(e)}
    
    def _enrich_package_info(self, package: Dict[str, str]) -> Dict[str, Any]:
        """Enrich package information with metadata."""
        name = package["name"]
        version = package["version"]
        
        # Get package details using pip show
        enriched = {
            "name": name,
            "version": version,
            "license": None,
            "homepage": None,
            "summary": None,
            "author": None,
            "requires": [],
            "required_by": [],
            "security_relevant": self._is_security_relevant(name),
            "is_dev_dependency": self._is_dev_dependency(name),
            "problematic": name.lower() in self.problematic_packages,
            "size_estimate": None
        }
        
        try:
            cmd = [sys.executable, "-m", "pip", "show", name]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                pip_info = self._parse_pip_show_output(result.stdout)
                enriched.update(pip_info)
                
        except Exception as e:
            logger.debug(f"Could not get detailed info for {name}: {e}")
        
        return enriched
    
    def _parse_pip_show_output(self, output: str) -> Dict[str, Any]:
        """Parse pip show output into structured data."""
        info = {}
        
        for line in output.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if key == 'license':
                    info['license'] = value if value and value != 'UNKNOWN' else None
                elif key == 'home-page':
                    info['homepage'] = value if value and value != 'UNKNOWN' else None
                elif key == 'summary':
                    info['summary'] = value if value and value != 'UNKNOWN' else None
                elif key == 'author':
                    info['author'] = value if value and value != 'UNKNOWN' else None
                elif key == 'requires':
                    info['requires'] = [dep.strip() for dep in value.split(',')] if value else []
                elif key == 'required-by':
                    info['required_by'] = [dep.strip() for dep in value.split(',')] if value else []
        
        return info
    
    def _is_security_relevant(self, package_name: str) -> bool:
        """Check if package is security-relevant."""
        security_keywords = {
            "crypt", "auth", "security", "ssl", "tls", "jwt", "oauth",
            "password", "hash", "sign", "cert", "key", "token"
        }
        
        return any(keyword in package_name.lower() for keyword in security_keywords)
    
    def _is_dev_dependency(self, package_name: str) -> bool:
        """Check if package is a development dependency."""
        dev_keywords = {
            "test", "pytest", "mock", "coverage", "lint", "format", "black",
            "mypy", "flake", "ruff", "pre-commit", "debug", "profil"
        }
        
        return any(keyword in package_name.lower() for keyword in dev_keywords)
    
    def analyze_dependency_tree(self) -> Dict[str, Any]:
        """Analyze dependency tree structure."""
        logger.info("Analyzing dependency tree...")
        
        try:
            cmd = [sys.executable, "-m", "pipdeptree", "--json"]
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                # Fallback to manual tree analysis
                return self._analyze_tree_manual()
            
            tree_data = json.loads(result.stdout) if result.stdout else []
            
            # Analyze tree structure
            analysis = {
                "total_packages": len(tree_data),
                "top_level_packages": [],
                "deep_dependencies": [],
                "circular_dependencies": [],
                "bloat_candidates": [],
                "duplicate_dependencies": []
            }
            
            package_depths = {}
            
            for package in tree_data:
                name = package["package"]["package_name"]
                dependencies = package.get("dependencies", [])
                
                # Calculate depth
                depth = self._calculate_dependency_depth(package, tree_data)
                package_depths[name] = depth
                
                if not dependencies:
                    analysis["top_level_packages"].append(name)
                
                if depth > 3:
                    analysis["deep_dependencies"].append({
                        "package": name,
                        "depth": depth,
                        "dependencies": len(dependencies)
                    })
                
                # Check for potential bloat
                if len(dependencies) > 10:
                    analysis["bloat_candidates"].append({
                        "package": name,
                        "dependency_count": len(dependencies)
                    })
            
            # Find duplicates
            analysis["duplicate_dependencies"] = self._find_duplicate_dependencies(tree_data)
            
            analysis["dependency_depths"] = package_depths
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze dependency tree: {e}")
            return {"error": str(e)}
    
    def _analyze_tree_manual(self) -> Dict[str, Any]:
        """Manual dependency tree analysis as fallback."""
        logger.info("Performing manual dependency tree analysis...")
        
        # This would implement a simpler version without pipdeptree
        return {
            "total_packages": 0,
            "top_level_packages": [],
            "deep_dependencies": [],
            "circular_dependencies": [],
            "bloat_candidates": [],
            "duplicate_dependencies": [],
            "warning": "Limited analysis - pipdeptree not available"
        }
    
    def _calculate_dependency_depth(self, package: Dict, tree_data: List[Dict]) -> int:
        """Calculate maximum dependency depth for a package."""
        visited = set()
        
        def _depth_recursive(pkg_name: str, current_depth: int = 0) -> int:
            if pkg_name in visited:
                return current_depth  # Circular dependency
            
            visited.add(pkg_name)
            
            # Find package in tree
            pkg_data = None
            for pkg in tree_data:
                if pkg["package"]["package_name"].lower() == pkg_name.lower():
                    pkg_data = pkg
                    break
            
            if not pkg_data or not pkg_data.get("dependencies"):
                return current_depth
            
            max_depth = current_depth
            for dep in pkg_data["dependencies"]:
                dep_name = dep["package_name"]
                dep_depth = _depth_recursive(dep_name, current_depth + 1)
                max_depth = max(max_depth, dep_depth)
            
            return max_depth
        
        return _depth_recursive(package["package"]["package_name"])
    
    def _find_duplicate_dependencies(self, tree_data: List[Dict]) -> List[Dict]:
        """Find packages that appear in multiple dependency chains."""
        dependency_counts = defaultdict(set)
        
        for package in tree_data:
            for dep in package.get("dependencies", []):
                dep_name = dep["package_name"]
                dependency_counts[dep_name].add(package["package"]["package_name"])
        
        duplicates = []
        for dep_name, dependents in dependency_counts.items():
            if len(dependents) > 1:
                duplicates.append({
                    "package": dep_name,
                    "used_by": list(dependents),
                    "usage_count": len(dependents)
                })
        
        return sorted(duplicates, key=lambda x: x["usage_count"], reverse=True)
    
    def check_license_compliance(self, packages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check license compliance for all dependencies."""
        logger.info("Checking license compliance...")
        
        license_analysis = {
            "compliant_packages": [],
            "non_compliant_packages": [],
            "unknown_licenses": [],
            "restrictive_licenses": [],
            "license_distribution": defaultdict(int)
        }
        
        project_license = "MIT"  # Assume MIT license for Mimir
        
        for package in packages:
            name = package["name"]
            license_info = package.get("license")
            
            if not license_info:
                license_analysis["unknown_licenses"].append(name)
                continue
            
            # Normalize license name
            normalized_license = self._normalize_license_name(license_info)
            license_analysis["license_distribution"][normalized_license] += 1
            
            # Check compatibility
            if normalized_license in self.license_compatibility:
                compat_info = self.license_compatibility[normalized_license]
                
                if project_license in compat_info["compatible"]:
                    license_analysis["compliant_packages"].append({
                        "package": name,
                        "license": normalized_license
                    })
                    
                    if compat_info["restrictive"]:
                        license_analysis["restrictive_licenses"].append({
                            "package": name,
                            "license": normalized_license,
                            "implications": "May require special handling"
                        })
                else:
                    license_analysis["non_compliant_packages"].append({
                        "package": name,
                        "license": normalized_license,
                        "reason": "Incompatible with project license"
                    })
            else:
                license_analysis["unknown_licenses"].append({
                    "package": name,
                    "license": license_info,
                    "requires_review": True
                })
        
        # Convert defaultdict to regular dict for JSON serialization
        license_analysis["license_distribution"] = dict(license_analysis["license_distribution"])
        
        return license_analysis
    
    def _normalize_license_name(self, license_str: str) -> str:
        """Normalize license name for comparison."""
        if not license_str:
            return "Unknown"
        
        license_str = license_str.upper().strip()
        
        # Common license mappings
        mappings = {
            "MIT LICENSE": "MIT",
            "APACHE SOFTWARE LICENSE": "Apache-2.0",
            "APACHE LICENSE 2.0": "Apache-2.0",
            "BSD LICENSE": "BSD",
            "GNU GENERAL PUBLIC LICENSE": "GPL",
            "GNU LESSER GENERAL PUBLIC LICENSE": "LGPL"
        }
        
        for pattern, normalized in mappings.items():
            if pattern in license_str:
                return normalized
        
        return license_str
    
    def analyze_security_risks(self, packages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze security risks in dependencies."""
        logger.info("Analyzing security risks...")
        
        risk_analysis = {
            "high_risk_packages": [],
            "security_relevant_packages": [],
            "problematic_packages": [],
            "outdated_packages": [],
            "unmaintained_packages": []
        }
        
        for package in packages:
            name = package["name"]
            
            # Check if security-relevant
            if package.get("security_relevant"):
                risk_analysis["security_relevant_packages"].append(name)
            
            # Check if problematic
            if name.lower() in self.problematic_packages:
                problem_info = self.problematic_packages[name.lower()]
                risk_analysis["problematic_packages"].append({
                    "package": name,
                    "issues": problem_info.get("issues", []),
                    "alternatives": problem_info.get("alternatives", [])
                })
            
            # Check for high-risk indicators
            high_risk_indicators = []
            if not package.get("homepage"):
                high_risk_indicators.append("no_homepage")
            if not package.get("author"):
                high_risk_indicators.append("no_author")
            if not package.get("license"):
                high_risk_indicators.append("no_license")
            
            if high_risk_indicators:
                risk_analysis["high_risk_packages"].append({
                    "package": name,
                    "risk_indicators": high_risk_indicators
                })
        
        return risk_analysis
    
    def estimate_package_sizes(self, packages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Estimate package installation sizes."""
        logger.info("Estimating package sizes...")
        
        size_analysis = {
            "total_estimated_size": 0,
            "large_packages": [],
            "size_by_category": {
                "runtime": 0,
                "development": 0,
                "security": 0
            }
        }
        
        for package in packages:
            name = package["name"]
            
            # Rough size estimation (this could be improved with actual size checking)
            estimated_size = self._estimate_package_size(name)
            package["estimated_size"] = estimated_size
            
            size_analysis["total_estimated_size"] += estimated_size
            
            if estimated_size > 10 * 1024 * 1024:  # > 10MB
                size_analysis["large_packages"].append({
                    "package": name,
                    "estimated_size": estimated_size,
                    "size_mb": round(estimated_size / (1024 * 1024), 1)
                })
            
            # Categorize by type
            if package.get("is_dev_dependency"):
                size_analysis["size_by_category"]["development"] += estimated_size
            elif package.get("security_relevant"):
                size_analysis["size_by_category"]["security"] += estimated_size
            else:
                size_analysis["size_by_category"]["runtime"] += estimated_size
        
        return size_analysis
    
    def _estimate_package_size(self, package_name: str) -> int:
        """Rough estimation of package size."""
        # This is a simplified estimation
        # In reality, you'd want to check actual installed sizes
        
        known_large_packages = {
            "numpy": 50 * 1024 * 1024,
            "pandas": 100 * 1024 * 1024,
            "tensorflow": 500 * 1024 * 1024,
            "torch": 800 * 1024 * 1024,
            "matplotlib": 30 * 1024 * 1024,
            "pillow": 20 * 1024 * 1024,
        }
        
        return known_large_packages.get(package_name.lower(), 1024 * 1024)  # 1MB default
    
    def generate_comprehensive_report(self, 
                                    packages_info: Dict[str, Any],
                                    tree_analysis: Dict[str, Any],
                                    license_analysis: Dict[str, Any],
                                    security_analysis: Dict[str, Any],
                                    size_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        
        return {
            "audit_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "project": "mimir",
                "audit_version": "1.0"
            },
            "summary": {
                "total_packages": packages_info.get("total_count", 0),
                "security_relevant": len(security_analysis.get("security_relevant_packages", [])),
                "problematic": len(security_analysis.get("problematic_packages", [])),
                "license_issues": len(license_analysis.get("non_compliant_packages", [])),
                "unknown_licenses": len(license_analysis.get("unknown_licenses", [])),
                "estimated_total_size_mb": round(size_analysis.get("total_estimated_size", 0) / (1024 * 1024), 1)
            },
            "packages": packages_info,
            "dependency_tree": tree_analysis,
            "license_compliance": license_analysis,
            "security_risks": security_analysis,
            "size_analysis": size_analysis,
            "recommendations": self._generate_recommendations(
                security_analysis, license_analysis, size_analysis, tree_analysis
            )
        }
    
    def _generate_recommendations(self, security_analysis, license_analysis, 
                                size_analysis, tree_analysis) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on audit results."""
        recommendations = []
        
        # Security recommendations
        if security_analysis.get("problematic_packages"):
            recommendations.append({
                "category": "security",
                "priority": "high",
                "title": "Replace problematic packages",
                "description": f"Found {len(security_analysis['problematic_packages'])} problematic packages",
                "action": "Review alternatives for flagged packages"
            })
        
        # License recommendations
        if license_analysis.get("non_compliant_packages"):
            recommendations.append({
                "category": "legal",
                "priority": "medium",
                "title": "Resolve license conflicts",
                "description": f"Found {len(license_analysis['non_compliant_packages'])} license conflicts",
                "action": "Review and resolve incompatible licenses"
            })
        
        # Size recommendations
        large_packages = size_analysis.get("large_packages", [])
        if large_packages:
            recommendations.append({
                "category": "performance",
                "priority": "low",
                "title": "Optimize package sizes",
                "description": f"Found {len(large_packages)} large packages",
                "action": "Consider lighter alternatives or selective imports"
            })
        
        # Tree structure recommendations
        if tree_analysis.get("deep_dependencies"):
            recommendations.append({
                "category": "maintenance",
                "priority": "medium",
                "title": "Review deep dependency chains",
                "description": f"Found {len(tree_analysis['deep_dependencies'])} packages with deep dependencies",
                "action": "Consider flattening dependency structure"
            })
        
        return recommendations


def main():
    """Main entry point for dependency auditor."""
    parser = argparse.ArgumentParser(description="Comprehensive dependency audit system")
    parser.add_argument("--format", choices=["text", "json", "html"], default="text", help="Report output format")
    parser.add_argument("--output", "-o", help="Output report file")
    parser.add_argument("--check-licenses", action="store_true", help="Perform detailed license compliance check")
    parser.add_argument("--analyze-bloat", action="store_true", help="Analyze package size and bloat")
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent.parent
    
    # Initialize auditor
    auditor = DependencyAuditor(project_root=project_root)
    
    try:
        # Run comprehensive audit
        logger.info("Starting comprehensive dependency audit...")
        
        # Get package information
        packages_info = auditor.get_installed_packages()
        if packages_info.get("error"):
            logger.error(f"Failed to get package information: {packages_info['error']}")
            sys.exit(1)
        
        packages = packages_info.get("packages", [])
        
        # Analyze dependency tree
        tree_analysis = auditor.analyze_dependency_tree()
        
        # Check license compliance
        license_analysis = auditor.check_license_compliance(packages) if args.check_licenses else {}
        
        # Analyze security risks
        security_analysis = auditor.analyze_security_risks(packages)
        
        # Estimate package sizes
        size_analysis = auditor.estimate_package_sizes(packages) if args.analyze_bloat else {}
        
        # Generate comprehensive report
        audit_report = auditor.generate_comprehensive_report(
            packages_info, tree_analysis, license_analysis, 
            security_analysis, size_analysis
        )
        
        # Format and display report
        if args.format == "json":
            report_output = json.dumps(audit_report, indent=2)
        else:
            # Generate text report
            report_lines = []
            report_lines.append("=" * 80)
            report_lines.append("MIMIR COMPREHENSIVE DEPENDENCY AUDIT REPORT")
            report_lines.append("=" * 80)
            report_lines.append(f"Audit Date: {audit_report['audit_metadata']['timestamp']}")
            report_lines.append("")
            
            summary = audit_report["summary"]
            report_lines.append("SUMMARY:")
            report_lines.append(f"  Total Packages: {summary['total_packages']}")
            report_lines.append(f"  Security-Relevant: {summary['security_relevant']}")
            report_lines.append(f"  Problematic: {summary['problematic']}")
            report_lines.append(f"  License Issues: {summary['license_issues']}")
            report_lines.append(f"  Estimated Size: {summary['estimated_total_size_mb']} MB")
            report_lines.append("")
            
            # Add recommendations
            if audit_report.get("recommendations"):
                report_lines.append("RECOMMENDATIONS:")
                for i, rec in enumerate(audit_report["recommendations"], 1):
                    report_lines.append(f"  {i}. [{rec['priority'].upper()}] {rec['title']}")
                    report_lines.append(f"     {rec['description']}")
                    report_lines.append(f"     Action: {rec['action']}")
                    report_lines.append("")
            
            report_lines.append("=" * 80)
            report_output = "\n".join(report_lines)
        
        print(report_output)
        
        # Save report if requested
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report_output)
            logger.info(f"Audit report saved to {args.output}")
        
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        auto_filename = f"dependency_audit_report_{timestamp}.{args.format}"
        auto_filepath = project_root / auto_filename
        
        with open(auto_filepath, 'w') as f:
            f.write(report_output)
        logger.info(f"Audit report saved to {auto_filepath}")
        
        # Exit with appropriate code based on findings
        summary = audit_report["summary"]
        if summary["problematic"] > 0 or summary["license_issues"] > 0:
            logger.warning("Audit found issues that require attention")
            sys.exit(1)
        else:
            logger.info("Dependency audit completed successfully")
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("Audit interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Audit failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()