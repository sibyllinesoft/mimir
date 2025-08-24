#!/usr/bin/env python3
"""
Mimir Distribution Build Automation Script

Builds multiple distribution formats:
- Python wheel distribution
- Docker images 
- Standalone executable using PyInstaller
- Source distribution
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
BUILD_DIR = PROJECT_ROOT / "dist"
DOCKER_IMAGES = [
    "mimir-server",
    "mimir-server:latest",
]

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

def run_command(cmd: List[str], cwd: Optional[Path] = None, check: bool = True) -> Tuple[int, str, str]:
    """Execute a shell command and return result."""
    log(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        if check:
            log(f"Command failed with exit code {e.returncode}", "error")
            log(f"STDOUT: {e.stdout}", "error")
            log(f"STDERR: {e.stderr}", "error")
            raise
        return e.returncode, e.stdout, e.stderr

def get_version() -> str:
    """Extract version from pyproject.toml."""
    import tomllib
    with open(PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]

def validate_environment() -> bool:
    """Validate build environment."""
    log("Validating build environment...")
    
    required_tools = {
        "python": ["python", "--version"],
        "git": ["git", "--version"],
        "docker": ["docker", "--version"]
    }
    
    missing_tools = []
    for tool, cmd in required_tools.items():
        try:
            run_command(cmd, check=False)
            log(f"✓ {tool} found", "success")
        except FileNotFoundError:
            missing_tools.append(tool)
            log(f"✗ {tool} not found", "error")
    
    if missing_tools:
        log(f"Missing required tools: {', '.join(missing_tools)}", "error")
        return False
    
    return True

def clean_build_dir() -> None:
    """Clean previous build artifacts."""
    log("Cleaning build directory...")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    log("Build directory cleaned", "success")

def build_python_wheel() -> None:
    """Build Python wheel distribution."""
    log("Building Python wheel...")
    
    # Clean old build artifacts
    for pattern in ["build", "*.egg-info", "dist"]:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    
    # Build wheel
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "build"])
    run_command([sys.executable, "-m", "build", "--wheel"])
    
    # Move wheel to dist directory
    wheel_files = list((PROJECT_ROOT / "dist").glob("*.whl"))
    if wheel_files:
        for wheel in wheel_files:
            shutil.copy2(wheel, BUILD_DIR / wheel.name)
        log(f"✓ Python wheel built: {wheel_files[0].name}", "success")
    else:
        raise RuntimeError("Wheel build failed - no wheel file found")

def build_source_dist() -> None:
    """Build source distribution."""
    log("Building source distribution...")
    
    run_command([sys.executable, "-m", "build", "--sdist"])
    
    # Move source dist to build directory
    sdist_files = list((PROJECT_ROOT / "dist").glob("*.tar.gz"))
    if sdist_files:
        for sdist in sdist_files:
            shutil.copy2(sdist, BUILD_DIR / sdist.name)
        log(f"✓ Source distribution built: {sdist_files[0].name}", "success")
    else:
        raise RuntimeError("Source distribution build failed")

def build_docker_images() -> None:
    """Build Docker images."""
    log("Building Docker images...")
    
    version = get_version()
    
    # Build main image
    run_command([
        "docker", "build",
        "-t", f"mimir-server:latest",
        "-t", f"mimir-server:{version}",
        "--target", "production",
        "."
    ])
    
    # Build development image
    run_command([
        "docker", "build",
        "-t", f"mimir-server:dev",
        "--target", "runtime",
        "."
    ])
    
    # Export images
    log("Exporting Docker images...")
    
    for tag in [f"mimir-server:{version}", "mimir-server:latest"]:
        output_file = BUILD_DIR / f"{tag.replace(':', '_')}.tar"
        run_command([
            "docker", "save",
            "-o", str(output_file),
            tag
        ])
        
        # Compress the image
        with open(output_file, 'rb') as f_in:
            with zipfile.ZipFile(f"{output_file}.zip", 'w', zipfile.ZIP_DEFLATED) as f_out:
                f_out.write(output_file, output_file.name)
        
        output_file.unlink()  # Remove uncompressed version
        log(f"✓ Docker image exported: {output_file.name}.zip", "success")

def install_pyinstaller() -> None:
    """Install PyInstaller if not available."""
    try:
        import PyInstaller
        log("PyInstaller already installed", "success")
    except ImportError:
        log("Installing PyInstaller...")
        run_command([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_standalone_executable() -> None:
    """Build standalone executable using PyInstaller."""
    log("Building standalone executable...")
    
    install_pyinstaller()
    
    # Create temporary spec file
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

block_cipher = None

# Add source directory to path
src_path = Path('{PROJECT_ROOT}') / 'src'
sys.path.insert(0, str(src_path))

# Analysis
a = Analysis(
    ['{PROJECT_ROOT}/src/repoindex/mcp/server.py'],
    pathex=[str(src_path)],
    binaries=[],
    datas=[
        ('{PROJECT_ROOT}/src/repoindex/data/schemas', 'repoindex/data/schemas'),
        ('{PROJECT_ROOT}/README.md', '.'),
    ],
    hiddenimports=[
        'repoindex',
        'repoindex.mcp',
        'repoindex.mcp.server',
        'repoindex.pipeline',
        'repoindex.util',
        'repoindex.security',
        'asyncio',
        'mcp',
        'pydantic',
        'fastapi',
        'uvicorn',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='mimir-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    spec_file = PROJECT_ROOT / "mimir-server.spec"
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    try:
        # Build executable
        run_command([
            sys.executable, "-m", "PyInstaller",
            "--clean",
            "--noconfirm",
            str(spec_file)
        ])
        
        # Find and copy executable
        system = platform.system().lower()
        exe_name = "mimir-server.exe" if system == "windows" else "mimir-server"
        
        exe_path = PROJECT_ROOT / "dist" / exe_name
        if exe_path.exists():
            target_name = f"mimir-server-{platform.system().lower()}-{platform.machine().lower()}"
            if system == "windows":
                target_name += ".exe"
            
            shutil.copy2(exe_path, BUILD_DIR / target_name)
            
            # Create archive
            archive_name = f"{target_name}.zip"
            with zipfile.ZipFile(BUILD_DIR / archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.write(BUILD_DIR / target_name, target_name)
            
            # Remove uncompressed executable
            (BUILD_DIR / target_name).unlink()
            
            log(f"✓ Standalone executable built: {archive_name}", "success")
        else:
            raise RuntimeError(f"Executable not found at {exe_path}")
    
    finally:
        # Cleanup
        if spec_file.exists():
            spec_file.unlink()
        for cleanup_dir in ["build", "dist"]:
            cleanup_path = PROJECT_ROOT / cleanup_dir
            if cleanup_path.exists() and cleanup_path != BUILD_DIR:
                shutil.rmtree(cleanup_path)

def create_checksums() -> None:
    """Create checksums for all build artifacts."""
    log("Creating checksums...")
    
    import hashlib
    
    checksums = {}
    
    for file_path in BUILD_DIR.iterdir():
        if file_path.is_file() and file_path.suffix in ['.whl', '.tar.gz', '.zip']:
            # Calculate SHA256
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            
            checksums[file_path.name] = {
                'sha256': hasher.hexdigest(),
                'size': file_path.stat().st_size
            }
    
    # Write checksums file
    checksums_file = BUILD_DIR / "checksums.json"
    with open(checksums_file, 'w') as f:
        json.dump(checksums, f, indent=2)
    
    # Also create SHA256SUMS file (traditional format)
    sha256sums_file = BUILD_DIR / "SHA256SUMS"
    with open(sha256sums_file, 'w') as f:
        for filename, info in checksums.items():
            f.write(f"{info['sha256']}  {filename}\n")
    
    log(f"✓ Checksums created", "success")

def create_build_info() -> None:
    """Create build information file."""
    log("Creating build information...")
    
    # Get git information
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], 
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        
        commit_date = subprocess.check_output(
            ["git", "show", "-s", "--format=%ci", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
        
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True
        ).strip()
    except subprocess.CalledProcessError:
        commit_hash = "unknown"
        commit_date = "unknown"
        branch = "unknown"
    
    build_info = {
        "version": get_version(),
        "build_date": subprocess.check_output(
            ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"] if platform.system() != "Windows" 
            else ["powershell", "-Command", "Get-Date -Format 'yyyy-MM-ddTHH:mm:ssZ'"],
            text=True
        ).strip(),
        "git": {
            "commit": commit_hash,
            "commit_date": commit_date,
            "branch": branch
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": platform.python_version()
        },
        "artifacts": [f.name for f in BUILD_DIR.iterdir() if f.is_file() and f.suffix in ['.whl', '.tar.gz', '.zip']]
    }
    
    build_info_file = BUILD_DIR / "build-info.json"
    with open(build_info_file, 'w') as f:
        json.dump(build_info, f, indent=2)
    
    log(f"✓ Build information created", "success")

def main():
    """Main build function."""
    parser = argparse.ArgumentParser(description="Build Mimir distributions")
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["wheel", "sdist", "docker", "executable", "all"],
        default=["all"],
        help="Distribution formats to build"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip environment validation"
    )
    
    args = parser.parse_args()
    
    if not args.skip_validation and not validate_environment():
        sys.exit(1)
    
    log(f"Starting build process for Mimir v{get_version()}")
    clean_build_dir()
    
    formats = args.formats
    if "all" in formats:
        formats = ["wheel", "sdist", "docker", "executable"]
    
    try:
        if "wheel" in formats:
            build_python_wheel()
        
        if "sdist" in formats:
            build_source_dist()
        
        if "docker" in formats:
            build_docker_images()
        
        if "executable" in formats:
            build_standalone_executable()
        
        create_checksums()
        create_build_info()
        
        log(f"Build completed successfully! Artifacts in: {BUILD_DIR}", "success")
        
        # List all artifacts
        log("Build artifacts:", "info")
        for artifact in sorted(BUILD_DIR.iterdir()):
            if artifact.is_file():
                size_mb = artifact.stat().st_size / 1024 / 1024
                log(f"  {artifact.name} ({size_mb:.1f} MB)")
        
    except Exception as e:
        log(f"Build failed: {e}", "error")
        sys.exit(1)

if __name__ == "__main__":
    main()