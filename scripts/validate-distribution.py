#!/usr/bin/env python3
"""
Mimir Distribution Validation Script

Validates all distribution formats and installation methods to ensure
they work correctly before release.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "dist"
VALIDATION_RESULTS = {}

class Color:
    """Terminal colors for better output."""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def log(message: str, level: str = "info") -> None:
    """Colored logging."""
    colors = {
        "info": Color.BLUE,
        "success": Color.GREEN,
        "warning": Color.YELLOW,
        "error": Color.RED
    }
    color = colors.get(level, "")
    print(f"{color}[{level.upper()}]{Color.END} {message}")

def run_command(cmd: List[str], cwd: Optional[Path] = None, timeout: int = 300) -> Tuple[int, str, str]:
    """Execute a shell command and return result."""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        log(f"Command timed out after {timeout} seconds", "error")
        return 1, "", "Command timed out"
    except Exception as e:
        log(f"Command failed with exception: {e}", "error")
        return 1, "", str(e)

def validate_wheel_distribution() -> Dict[str, any]:
    """Validate Python wheel distribution."""
    log("Validating wheel distribution...")
    result = {"name": "wheel", "status": "unknown", "errors": []}
    
    try:
        # Find wheel file
        wheel_files = list(BUILD_DIR.glob("*.whl"))
        if not wheel_files:
            result["status"] = "failed"
            result["errors"].append("No wheel file found")
            return result
        
        wheel_path = wheel_files[0]
        log(f"Testing wheel: {wheel_path.name}")
        
        # Create temporary virtual environment
        with tempfile.TemporaryDirectory() as temp_dir:
            venv_path = Path(temp_dir) / "test_venv"
            
            # Create virtual environment
            ret_code, stdout, stderr = run_command([
                sys.executable, "-m", "venv", str(venv_path)
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to create venv: {stderr}")
                return result
            
            # Determine python executable path
            if os.name == "nt":  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
            else:  # Unix-like
                python_exe = venv_path / "bin" / "python"
            
            # Install wheel
            ret_code, stdout, stderr = run_command([
                str(python_exe), "-m", "pip", "install", str(wheel_path)
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to install wheel: {stderr}")
                return result
            
            # Test entry points
            ret_code, stdout, stderr = run_command([
                str(python_exe), "-c", 
                "import repoindex; from repoindex.mcp.server import main; print('Import successful')"
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to import package: {stderr}")
                return result
            
            # Test CLI command (with timeout)
            ret_code, stdout, stderr = run_command([
                str(python_exe), "-m", "repoindex.mcp.server", "--help"
            ], timeout=10)
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"CLI command failed: {stderr}")
                return result
        
        result["status"] = "passed"
        result["wheel_file"] = wheel_path.name
        log("✓ Wheel validation passed", "success")
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Validation exception: {str(e)}")
        log(f"Wheel validation failed: {e}", "error")
    
    return result

def validate_docker_image() -> Dict[str, any]:
    """Validate Docker image."""
    log("Validating Docker image...")
    result = {"name": "docker", "status": "unknown", "errors": []}
    
    try:
        # Check if Docker is available
        ret_code, stdout, stderr = run_command(["docker", "--version"])
        if ret_code != 0:
            result["status"] = "skipped"
            result["errors"].append("Docker not available")
            return result
        
        # Check if image exists
        ret_code, stdout, stderr = run_command([
            "docker", "image", "inspect", "mimir-server:latest"
        ])
        
        if ret_code != 0:
            result["status"] = "failed"
            result["errors"].append("Docker image mimir-server:latest not found")
            return result
        
        # Test container startup
        container_name = f"mimir-test-{int(time.time())}"
        
        # Run container with health check
        ret_code, stdout, stderr = run_command([
            "docker", "run", "--name", container_name,
            "-d", "-p", "0:8000",  # Use random port
            "mimir-server:latest"
        ])
        
        if ret_code != 0:
            result["status"] = "failed"
            result["errors"].append(f"Failed to start container: {stderr}")
            return result
        
        try:
            # Wait for container to start
            time.sleep(5)
            
            # Check container status
            ret_code, stdout, stderr = run_command([
                "docker", "inspect", container_name, 
                "--format", "{{.State.Status}}"
            ])
            
            if ret_code == 0 and stdout.strip() == "running":
                result["status"] = "passed"
                log("✓ Docker validation passed", "success")
            else:
                result["status"] = "failed"
                result["errors"].append(f"Container not running: {stdout.strip()}")
        
        finally:
            # Cleanup container
            run_command(["docker", "rm", "-f", container_name])
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Validation exception: {str(e)}")
        log(f"Docker validation failed: {e}", "error")
    
    return result

def validate_source_distribution() -> Dict[str, any]:
    """Validate source distribution."""
    log("Validating source distribution...")
    result = {"name": "sdist", "status": "unknown", "errors": []}
    
    try:
        # Find source distribution
        sdist_files = list(BUILD_DIR.glob("*.tar.gz"))
        if not sdist_files:
            result["status"] = "failed"
            result["errors"].append("No source distribution found")
            return result
        
        sdist_path = sdist_files[0]
        log(f"Testing source distribution: {sdist_path.name}")
        
        # Create temporary directory and extract
        with tempfile.TemporaryDirectory() as temp_dir:
            extract_dir = Path(temp_dir) / "extracted"
            
            # Extract tarball
            ret_code, stdout, stderr = run_command([
                "tar", "-xzf", str(sdist_path), "-C", temp_dir
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to extract: {stderr}")
                return result
            
            # Find extracted directory
            extracted_dirs = [d for d in Path(temp_dir).iterdir() if d.is_dir()]
            if not extracted_dirs:
                result["status"] = "failed"
                result["errors"].append("No extracted directory found")
                return result
            
            source_dir = extracted_dirs[0]
            
            # Check required files
            required_files = ["pyproject.toml", "README.md", "src/repoindex"]
            missing_files = []
            
            for req_file in required_files:
                if not (source_dir / req_file).exists():
                    missing_files.append(req_file)
            
            if missing_files:
                result["status"] = "failed"
                result["errors"].append(f"Missing files: {missing_files}")
                return result
            
            # Try to build from source
            venv_path = Path(temp_dir) / "build_venv"
            
            # Create virtual environment
            ret_code, stdout, stderr = run_command([
                sys.executable, "-m", "venv", str(venv_path)
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to create build venv: {stderr}")
                return result
            
            # Determine python executable path
            if os.name == "nt":  # Windows
                python_exe = venv_path / "Scripts" / "python.exe"
            else:  # Unix-like
                python_exe = venv_path / "bin" / "python"
            
            # Install build dependencies
            ret_code, stdout, stderr = run_command([
                str(python_exe), "-m", "pip", "install", "build"
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to install build: {stderr}")
                return result
            
            # Build from source
            ret_code, stdout, stderr = run_command([
                str(python_exe), "-m", "build"
            ], cwd=source_dir)
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to build from source: {stderr}")
                return result
        
        result["status"] = "passed"
        result["sdist_file"] = sdist_path.name
        log("✓ Source distribution validation passed", "success")
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Validation exception: {str(e)}")
        log(f"Source distribution validation failed: {e}", "error")
    
    return result

def validate_standalone_executable() -> Dict[str, any]:
    """Validate standalone executable."""
    log("Validating standalone executable...")
    result = {"name": "executable", "status": "unknown", "errors": []}
    
    try:
        # Find executable files
        exe_files = list(BUILD_DIR.glob("*.zip"))
        exe_files = [f for f in exe_files if "mimir-server" in f.name and "tar" not in f.name]
        
        if not exe_files:
            result["status"] = "failed"
            result["errors"].append("No standalone executable found")
            return result
        
        exe_archive = exe_files[0]
        log(f"Testing executable: {exe_archive.name}")
        
        # Extract and test executable
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract archive
            ret_code, stdout, stderr = run_command([
                "unzip", "-o", str(exe_archive), "-d", temp_dir
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Failed to extract executable: {stderr}")
                return result
            
            # Find executable file
            exe_files = []
            for item in Path(temp_dir).rglob("*"):
                if item.is_file() and ("mimir-server" in item.name):
                    exe_files.append(item)
            
            if not exe_files:
                result["status"] = "failed"
                result["errors"].append("No executable file found in archive")
                return result
            
            exe_path = exe_files[0]
            
            # Make executable (Unix-like systems)
            if os.name != "nt":
                exe_path.chmod(0o755)
            
            # Test executable (with timeout)
            ret_code, stdout, stderr = run_command([
                str(exe_path), "--help"
            ], timeout=30)
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Executable failed: {stderr}")
                return result
        
        result["status"] = "passed"
        result["executable_archive"] = exe_archive.name
        log("✓ Standalone executable validation passed", "success")
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Validation exception: {str(e)}")
        log(f"Standalone executable validation failed: {e}", "error")
    
    return result

def validate_installation_scripts() -> Dict[str, any]:
    """Validate installation scripts."""
    log("Validating installation scripts...")
    result = {"name": "install_scripts", "status": "unknown", "errors": []}
    
    try:
        scripts = [
            PROJECT_ROOT / "scripts" / "install.sh",
            PROJECT_ROOT / "scripts" / "install.bat"
        ]
        
        missing_scripts = []
        for script in scripts:
            if not script.exists():
                missing_scripts.append(str(script))
        
        if missing_scripts:
            result["status"] = "failed"
            result["errors"].append(f"Missing scripts: {missing_scripts}")
            return result
        
        # Test script syntax (Unix)
        if scripts[0].exists():
            ret_code, stdout, stderr = run_command([
                "bash", "-n", str(scripts[0])
            ])
            
            if ret_code != 0:
                result["status"] = "failed"
                result["errors"].append(f"Bash script syntax error: {stderr}")
                return result
        
        # Test help output
        ret_code, stdout, stderr = run_command([
            "bash", str(scripts[0]), "--help"
        ])
        
        if ret_code != 0:
            result["status"] = "failed"
            result["errors"].append(f"Install script help failed: {stderr}")
            return result
        
        result["status"] = "passed"
        log("✓ Installation scripts validation passed", "success")
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Validation exception: {str(e)}")
        log(f"Installation scripts validation failed: {e}", "error")
    
    return result

def validate_checksums() -> Dict[str, any]:
    """Validate distribution checksums."""
    log("Validating checksums...")
    result = {"name": "checksums", "status": "unknown", "errors": []}
    
    try:
        checksums_file = BUILD_DIR / "checksums.json"
        sha256sums_file = BUILD_DIR / "SHA256SUMS"
        
        if not checksums_file.exists():
            result["status"] = "failed"
            result["errors"].append("checksums.json not found")
            return result
        
        if not sha256sums_file.exists():
            result["status"] = "failed"
            result["errors"].append("SHA256SUMS not found")
            return result
        
        # Load checksums
        with open(checksums_file) as f:
            checksums = json.load(f)
        
        # Verify each file
        import hashlib
        
        verified_files = 0
        for filename, info in checksums.items():
            file_path = BUILD_DIR / filename
            
            if not file_path.exists():
                result["errors"].append(f"File missing: {filename}")
                continue
            
            # Calculate actual checksum
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            actual_hash = hasher.hexdigest()
            expected_hash = info['sha256']
            
            if actual_hash != expected_hash:
                result["errors"].append(f"Checksum mismatch for {filename}")
            else:
                verified_files += 1
        
        if result["errors"]:
            result["status"] = "failed"
        else:
            result["status"] = "passed"
            result["verified_files"] = verified_files
            log(f"✓ Checksums validation passed ({verified_files} files)", "success")
        
    except Exception as e:
        result["status"] = "failed"
        result["errors"].append(f"Validation exception: {str(e)}")
        log(f"Checksums validation failed: {e}", "error")
    
    return result

def generate_validation_report() -> None:
    """Generate final validation report."""
    log("\n" + "="*60)
    log("DISTRIBUTION VALIDATION REPORT", "info")
    log("="*60)
    
    total_tests = len(VALIDATION_RESULTS)
    passed_tests = sum(1 for r in VALIDATION_RESULTS.values() if r["status"] == "passed")
    failed_tests = sum(1 for r in VALIDATION_RESULTS.values() if r["status"] == "failed")
    skipped_tests = sum(1 for r in VALIDATION_RESULTS.values() if r["status"] == "skipped")
    
    log(f"Total tests: {total_tests}")
    log(f"Passed: {passed_tests}", "success")
    log(f"Failed: {failed_tests}", "error" if failed_tests > 0 else "info")
    log(f"Skipped: {skipped_tests}", "warning" if skipped_tests > 0 else "info")
    
    log("\nDetailed Results:")
    for test_name, result in VALIDATION_RESULTS.items():
        status_color = {
            "passed": "success",
            "failed": "error", 
            "skipped": "warning",
            "unknown": "info"
        }.get(result["status"], "info")
        
        log(f"  {test_name}: {result['status'].upper()}", status_color)
        
        if result["errors"]:
            for error in result["errors"]:
                log(f"    - {error}", "error")
    
    # Save report to file
    report_file = BUILD_DIR / "validation-report.json"
    with open(report_file, 'w') as f:
        json.dump({
            "summary": {
                "total": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests
            },
            "results": VALIDATION_RESULTS
        }, f, indent=2)
    
    log(f"\nValidation report saved to: {report_file}")
    
    if failed_tests > 0:
        log(f"\nValidation FAILED with {failed_tests} failures", "error")
        sys.exit(1)
    else:
        log(f"\nAll validations PASSED!", "success")

def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Mimir distributions")
    parser.add_argument(
        "--tests",
        nargs="+",
        choices=["wheel", "docker", "sdist", "executable", "scripts", "checksums", "all"],
        default=["all"],
        help="Tests to run"
    )
    parser.add_argument(
        "--build-first",
        action="store_true",
        help="Build distributions before validation"
    )
    
    args = parser.parse_args()
    
    if not BUILD_DIR.exists():
        log("Build directory not found. Run with --build-first or build manually.", "error")
        sys.exit(1)
    
    # Build distributions if requested
    if args.build_first:
        log("Building distributions first...")
        build_script = PROJECT_ROOT / "scripts" / "build.py"
        ret_code, stdout, stderr = run_command([sys.executable, str(build_script)])
        if ret_code != 0:
            log("Build failed", "error")
            sys.exit(1)
    
    # Determine tests to run
    tests_to_run = args.tests
    if "all" in tests_to_run:
        tests_to_run = ["wheel", "docker", "sdist", "executable", "scripts", "checksums"]
    
    log(f"Running validation tests: {', '.join(tests_to_run)}")
    
    # Run validation tests
    test_functions = {
        "wheel": validate_wheel_distribution,
        "docker": validate_docker_image,
        "sdist": validate_source_distribution,
        "executable": validate_standalone_executable,
        "scripts": validate_installation_scripts,
        "checksums": validate_checksums
    }
    
    for test_name in tests_to_run:
        if test_name in test_functions:
            VALIDATION_RESULTS[test_name] = test_functions[test_name]()
    
    # Generate report
    generate_validation_report()

if __name__ == "__main__":
    main()