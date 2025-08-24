"""
Git integration utilities for repository operations.

Provides subprocess-based git operations with proper error handling
and subprocess management for repository discovery and change detection.
"""

import subprocess
from pathlib import Path


class GitError(Exception):
    """Git operation error."""

    pass


class GitRepository:
    """Interface for git repository operations."""

    def __init__(self, repo_path: str | Path):
        """Initialize git repository interface."""
        self.repo_path = Path(repo_path).resolve()

        # Verify this is a git repository
        if not self._is_git_repo():
            raise GitError(f"Not a git repository: {self.repo_path}")

    def _is_git_repo(self) -> bool:
        """Check if path is a git repository."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
            )
            return result.returncode == 0
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_git(
        self, args: list[str], check: bool = True, input_data: str | None = None
    ) -> subprocess.CompletedProcess:
        """Run git command with proper error handling."""
        cmd = ["git"] + args

        try:
            result = subprocess.run(
                cmd,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=check,
                input=input_data,
            )
            return result
        except subprocess.CalledProcessError as e:
            raise GitError(f"Git command failed: {' '.join(cmd)}\nError: {e.stderr}")
        except FileNotFoundError:
            raise GitError("Git command not found. Please ensure git is installed.")

    def get_repo_root(self) -> Path:
        """Get the repository root directory."""
        result = self._run_git(["rev-parse", "--show-toplevel"])
        return Path(result.stdout.strip())

    def get_head_commit(self) -> str:
        """Get the current HEAD commit SHA."""
        result = self._run_git(["rev-parse", "HEAD"])
        return result.stdout.strip()

    def get_tree_hash(self, ref: str = "HEAD") -> str:
        """Get the tree hash for a given reference."""
        result = self._run_git(["rev-parse", f"{ref}^{{tree}}"])
        return result.stdout.strip()

    def list_tracked_files(
        self, extensions: list[str] | None = None, excludes: list[str] | None = None
    ) -> list[str]:
        """
        List tracked files, optionally filtered by extension and excludes.

        Returns paths relative to repository root.
        """
        result = self._run_git(["ls-files", "-z"])

        # Split on null bytes and filter empty strings
        files = [f for f in result.stdout.split("\0") if f]

        # Apply extension filter
        if extensions:
            ext_set = {ext.lstrip(".") for ext in extensions}
            files = [f for f in files if Path(f).suffix.lstrip(".") in ext_set]

        # Apply exclude patterns
        if excludes:
            filtered_files = []
            for file_path in files:
                should_exclude = any(
                    exclude in file_path or file_path.startswith(exclude) for exclude in excludes
                )
                if not should_exclude:
                    filtered_files.append(file_path)
            files = filtered_files

        return files

    def get_dirty_files(self) -> set[str]:
        """Get list of files with uncommitted changes."""
        # Get staged changes
        result = self._run_git(["diff", "--cached", "--name-only", "-z"])
        staged_files = {f for f in result.stdout.split("\0") if f}

        # Get unstaged changes
        result = self._run_git(["diff", "--name-only", "-z"])
        unstaged_files = {f for f in result.stdout.split("\0") if f}

        # Get untracked files
        result = self._run_git(["ls-files", "--others", "--exclude-standard", "-z"])
        untracked_files = {f for f in result.stdout.split("\0") if f}

        return staged_files | unstaged_files | untracked_files

    def is_worktree_dirty(self) -> bool:
        """Check if worktree has any uncommitted changes."""
        return len(self.get_dirty_files()) > 0

    def hash_file_content(self, file_path: str) -> str:
        """Get git hash of file content (staged or worktree)."""
        try:
            # Try to get hash of staged version first
            result = self._run_git(["ls-files", "-s", file_path])
            if result.stdout.strip():
                # Parse staged file hash
                parts = result.stdout.strip().split()
                if len(parts) >= 2:
                    return parts[1]
        except GitError:
            pass

        # Fall back to hashing worktree version
        full_path = self.repo_path / file_path
        if not full_path.exists():
            raise GitError(f"File not found: {file_path}")

        # Use git hash-object to get hash
        result = self._run_git(["hash-object", str(full_path)])
        return result.stdout.strip()

    def get_file_content(self, file_path: str, ref: str = "HEAD") -> str:
        """Get file content at specific reference."""
        try:
            result = self._run_git(["show", f"{ref}:{file_path}"])
            return result.stdout
        except GitError:
            # If file doesn't exist at ref, try worktree
            full_path = self.repo_path / file_path
            if full_path.exists():
                return full_path.read_text(encoding="utf-8")
            raise

    def compute_dirty_overlay(self) -> dict[str, str]:
        """
        Compute overlay of dirty files with their content hashes.

        Returns mapping of file path -> content hash for all dirty files.
        """
        dirty_files = self.get_dirty_files()
        overlay = {}

        for file_path in dirty_files:
            try:
                content_hash = self.hash_file_content(file_path)
                overlay[file_path] = content_hash
            except GitError:
                # Skip files that can't be hashed
                continue

        return overlay

    def get_cache_key(self, config_hash: str) -> str:
        """
        Generate cache key from repository state and configuration.

        Combines HEAD tree hash, dirty overlay hash, and config hash.
        """
        tree_hash = self.get_tree_hash()
        dirty_overlay = self.compute_dirty_overlay()

        # Create overlay hash
        overlay_str = json.dumps(dirty_overlay, sort_keys=True)
        overlay_hash = hashlib.sha256(overlay_str.encode()).hexdigest()[:16]

        return f"{tree_hash[:16]}_{overlay_hash}_{config_hash[:16]}"

    def detect_workspaces(self) -> list[dict[str, str]]:
        """
        Detect workspace configuration for monorepos.

        Returns list of workspace info with name, path, and type.
        """
        workspaces = []
        repo_root = self.get_repo_root()

        # Check for package.json workspaces (npm/yarn)
        package_json = repo_root / "package.json"
        if package_json.exists():
            try:
                import json

                with open(package_json) as f:
                    pkg_data = json.load(f)

                if "workspaces" in pkg_data:
                    workspace_patterns = pkg_data["workspaces"]
                    if isinstance(workspace_patterns, dict):
                        workspace_patterns = workspace_patterns.get("packages", [])

                    # NOTE: Future enhancement - expand glob patterns to resolve actual workspace directories
                    # Current implementation returns workspace patterns as-is for basic support
                    for pattern in workspace_patterns:
                        workspaces.append({"name": pattern, "path": pattern, "type": "npm"})
            except (OSError, json.JSONDecodeError):
                pass

        # Check for pnpm-workspace.yaml
        pnpm_workspace = repo_root / "pnpm-workspace.yaml"
        if pnpm_workspace.exists():
            try:
                import yaml

                with open(pnpm_workspace) as f:
                    workspace_data = yaml.safe_load(f)

                packages = workspace_data.get("packages", [])
                for pattern in packages:
                    workspaces.append({"name": pattern, "path": pattern, "type": "pnpm"})
            except (ImportError, Exception):
                pass

        # Check for Cargo.toml workspace (Rust)
        cargo_toml = repo_root / "Cargo.toml"
        if cargo_toml.exists():
            try:
                import tomllib

                with open(cargo_toml, "rb") as f:
                    cargo_data = tomllib.load(f)

                workspace_data = cargo_data.get("workspace", {})
                members = workspace_data.get("members", [])

                for member in members:
                    workspaces.append({"name": member, "path": member, "type": "cargo"})
            except (ImportError, Exception):
                pass

        return workspaces

    def get_commit_info(self, ref: str = "HEAD") -> dict[str, str]:
        """Get detailed commit information."""
        result = self._run_git(["show", "--format=%H|%h|%an|%ae|%ad|%s", "--no-patch", ref])

        parts = result.stdout.strip().split("|")
        if len(parts) >= 6:
            return {
                "hash": parts[0],
                "short_hash": parts[1],
                "author_name": parts[2],
                "author_email": parts[3],
                "date": parts[4],
                "subject": parts[5],
            }

        return {"hash": self.get_head_commit()}

    def get_recent_commits(self, count: int = 10) -> list[dict[str, str]]:
        """Get list of recent commits."""
        result = self._run_git(["log", f"--max-count={count}", "--format=%H|%h|%an|%ae|%ad|%s"])

        commits = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("|")
                if len(parts) >= 6:
                    commits.append(
                        {
                            "hash": parts[0],
                            "short_hash": parts[1],
                            "author_name": parts[2],
                            "author_email": parts[3],
                            "date": parts[4],
                            "subject": parts[5],
                        }
                    )

        return commits


def discover_git_repository(path: str | Path) -> GitRepository:
    """
    Discover git repository from any path within it.

    Walks up directory tree to find repository root.
    """
    path = Path(path).resolve()

    # Try current directory and parents
    for potential_repo in [path] + list(path.parents):
        try:
            return GitRepository(potential_repo)
        except GitError:
            continue

    raise GitError(f"No git repository found for path: {path}")


# Import required modules at module level to avoid issues
import hashlib
import json
