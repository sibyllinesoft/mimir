#!/usr/bin/env python3
"""
Mimir Release Automation Script

Handles:
- Version bumping
- Git tagging
- Building all distribution formats
- Creating GitHub releases
- Publishing to PyPI (optional)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent
CHANGELOG_FILE = PROJECT_ROOT / "CHANGELOG.md"
PYPROJECT_FILE = PROJECT_ROOT / "pyproject.toml"

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

def get_current_version() -> str:
    """Extract current version from pyproject.toml."""
    import tomllib
    with open(PYPROJECT_FILE, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]

def bump_version(current_version: str, bump_type: str) -> str:
    """Bump version number based on semantic versioning."""
    parts = list(map(int, current_version.split(".")))
    
    if bump_type == "major":
        parts[0] += 1
        parts[1] = 0
        parts[2] = 0
    elif bump_type == "minor":
        parts[1] += 1
        parts[2] = 0
    elif bump_type == "patch":
        parts[2] += 1
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    return ".".join(map(str, parts))

def update_version_in_file(new_version: str) -> None:
    """Update version in pyproject.toml."""
    log(f"Updating version to {new_version} in pyproject.toml")
    
    # Read current content
    with open(PYPROJECT_FILE, 'r') as f:
        content = f.read()
    
    # Replace version
    version_pattern = r'version = "[^"]*"'
    new_content = re.sub(version_pattern, f'version = "{new_version}"', content)
    
    # Write back
    with open(PYPROJECT_FILE, 'w') as f:
        f.write(new_content)
    
    log(f"✓ Version updated in pyproject.toml", "success")

def update_changelog(version: str, changes: str) -> None:
    """Update CHANGELOG.md with new version."""
    log("Updating CHANGELOG.md")
    
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Read existing changelog
    if CHANGELOG_FILE.exists():
        with open(CHANGELOG_FILE, 'r') as f:
            content = f.read()
    else:
        content = "# Changelog\n\nAll notable changes to this project will be documented in this file.\n\n"
    
    # Create new entry
    new_entry = f"""## [{version}] - {today}

{changes}

"""
    
    # Insert after the header
    lines = content.split('\n')
    header_end = 0
    for i, line in enumerate(lines):
        if line.startswith('## [') or line.startswith('## Unreleased'):
            header_end = i
            break
        elif i > 5:  # Safety break
            header_end = len(lines)
            break
    
    # Insert new entry
    if header_end == 0:
        lines.insert(-1, new_entry.strip())
    else:
        lines.insert(header_end, new_entry.strip())
    
    # Write back
    with open(CHANGELOG_FILE, 'w') as f:
        f.write('\n'.join(lines))
    
    log(f"✓ Changelog updated", "success")

def get_git_status() -> bool:
    """Check if git repository is clean."""
    ret_code, stdout, stderr = run_command(["git", "status", "--porcelain"], check=False)
    return ret_code == 0 and not stdout.strip()

def commit_version_changes(version: str) -> None:
    """Commit version bump changes."""
    log("Committing version changes")
    
    # Add files
    run_command(["git", "add", str(PYPROJECT_FILE), str(CHANGELOG_FILE)])
    
    # Commit
    commit_message = f"Bump version to {version}"
    run_command(["git", "commit", "-m", commit_message])
    
    log(f"✓ Version changes committed", "success")

def create_git_tag(version: str, tag_message: str) -> None:
    """Create and push git tag."""
    log(f"Creating git tag v{version}")
    
    tag_name = f"v{version}"
    
    # Create annotated tag
    run_command(["git", "tag", "-a", tag_name, "-m", tag_message])
    
    # Push tag
    run_command(["git", "push", "origin", tag_name])
    
    log(f"✓ Git tag {tag_name} created and pushed", "success")

def build_distributions() -> Path:
    """Build all distribution formats."""
    log("Building distributions")
    
    build_script = PROJECT_ROOT / "scripts" / "build.py"
    if not build_script.exists():
        raise RuntimeError("Build script not found. Run with --skip-build to skip building.")
    
    # Run build script
    run_command([sys.executable, str(build_script), "--formats", "all"])
    
    dist_dir = PROJECT_ROOT / "dist"
    log(f"✓ Distributions built in {dist_dir}", "success")
    return dist_dir

def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or config."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        github_token_file = Path.home() / ".github_token"
        if github_token_file.exists():
            token = github_token_file.read_text().strip()
    return token

def create_github_release(version: str, changes: str, dist_dir: Path) -> None:
    """Create GitHub release with artifacts."""
    log("Creating GitHub release")
    
    token = get_github_token()
    if not token:
        log("GitHub token not found. Set GITHUB_TOKEN environment variable or create ~/.github_token", "error")
        return
    
    # Extract repository info from git remote
    ret_code, stdout, stderr = run_command(["git", "remote", "get-url", "origin"], check=False)
    if ret_code != 0:
        log("Could not get git remote URL", "error")
        return
    
    remote_url = stdout.strip()
    # Parse GitHub repo from URL
    if "github.com" in remote_url:
        if remote_url.startswith("git@"):
            # SSH format: git@github.com:owner/repo.git
            repo_part = remote_url.split(":")[-1].replace(".git", "")
        else:
            # HTTPS format: https://github.com/owner/repo.git
            repo_part = remote_url.split("github.com/")[-1].replace(".git", "")
        
        repo_owner, repo_name = repo_part.split("/")
    else:
        log("Not a GitHub repository", "error")
        return
    
    # Create release
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    
    release_data = {
        "tag_name": f"v{version}",
        "name": f"Mimir v{version}",
        "body": changes,
        "draft": False,
        "prerelease": "rc" in version.lower() or "alpha" in version.lower() or "beta" in version.lower(),
    }
    
    response = requests.post(
        f"https://api.github.com/repos/{repo_owner}/{repo_name}/releases",
        headers=headers,
        json=release_data
    )
    
    if response.status_code != 201:
        log(f"Failed to create GitHub release: {response.json()}", "error")
        return
    
    release_info = response.json()
    upload_url = release_info["upload_url"].split("{")[0]  # Remove template part
    
    log(f"✓ GitHub release created: {release_info['html_url']}", "success")
    
    # Upload artifacts
    for artifact in dist_dir.iterdir():
        if artifact.is_file() and artifact.suffix in ['.whl', '.tar.gz', '.zip']:
            log(f"Uploading {artifact.name}")
            
            with open(artifact, 'rb') as f:
                upload_response = requests.post(
                    f"{upload_url}?name={artifact.name}",
                    headers={
                        **headers,
                        "Content-Type": "application/octet-stream",
                    },
                    data=f
                )
            
            if upload_response.status_code == 201:
                log(f"✓ Uploaded {artifact.name}", "success")
            else:
                log(f"Failed to upload {artifact.name}: {upload_response.json()}", "error")

def publish_to_pypi(dist_dir: Path, repository: str = "pypi") -> None:
    """Publish to PyPI using twine."""
    log(f"Publishing to {repository}")
    
    # Check if twine is available
    ret_code, _, _ = run_command(["twine", "--version"], check=False)
    if ret_code != 0:
        log("Installing twine...")
        run_command([sys.executable, "-m", "pip", "install", "twine"])
    
    # Find wheel and sdist files
    wheel_files = list(dist_dir.glob("*.whl"))
    sdist_files = list(dist_dir.glob("*.tar.gz"))
    
    if not wheel_files or not sdist_files:
        log("No wheel or sdist files found for upload", "error")
        return
    
    # Upload to PyPI
    upload_files = wheel_files + sdist_files
    cmd = ["twine", "upload"]
    
    if repository != "pypi":
        cmd.extend(["--repository", repository])
    
    cmd.extend([str(f) for f in upload_files])
    
    try:
        run_command(cmd)
        log(f"✓ Published to {repository}", "success")
    except subprocess.CalledProcessError:
        log(f"Failed to publish to {repository}. Check your credentials.", "error")

def main():
    """Main release function."""
    parser = argparse.ArgumentParser(description="Release Mimir")
    parser.add_argument(
        "bump_type",
        choices=["major", "minor", "patch"],
        help="Version bump type"
    )
    parser.add_argument(
        "--changes",
        required=True,
        help="Release changes description"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building distributions"
    )
    parser.add_argument(
        "--skip-github",
        action="store_true",
        help="Skip GitHub release creation"
    )
    parser.add_argument(
        "--publish-pypi",
        action="store_true",
        help="Publish to PyPI"
    )
    parser.add_argument(
        "--pypi-repository",
        default="pypi",
        help="PyPI repository (default: pypi, use 'testpypi' for testing)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    
    args = parser.parse_args()
    
    # Validate git status
    if not get_git_status():
        log("Git repository is not clean. Commit or stash changes first.", "error")
        if not args.dry_run:
            sys.exit(1)
    
    current_version = get_current_version()
    new_version = bump_version(current_version, args.bump_type)
    
    log(f"Current version: {current_version}")
    log(f"New version: {new_version}")
    log(f"Changes: {args.changes}")
    
    if args.dry_run:
        log("DRY RUN - No changes will be made", "warning")
        return
    
    try:
        # Version bump
        update_version_in_file(new_version)
        update_changelog(new_version, args.changes)
        commit_version_changes(new_version)
        
        # Git tag
        create_git_tag(new_version, f"Release v{new_version}\n\n{args.changes}")
        
        # Build distributions
        dist_dir = None
        if not args.skip_build:
            dist_dir = build_distributions()
        
        # GitHub release
        if not args.skip_github and dist_dir:
            create_github_release(new_version, args.changes, dist_dir)
        
        # PyPI publishing
        if args.publish_pypi and dist_dir:
            publish_to_pypi(dist_dir, args.pypi_repository)
        
        log(f"✓ Release v{new_version} completed successfully!", "success")
        
    except Exception as e:
        log(f"Release failed: {e}", "error")
        log("You may need to manually clean up any partial changes.", "warning")
        sys.exit(1)

if __name__ == "__main__":
    main()