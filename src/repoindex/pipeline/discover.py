"""
Git-scoped file discovery and change detection.

Provides intelligent file discovery using git tracking with support for
incremental indexing, change detection, and monorepo workspace handling.
"""

import asyncio
import hashlib
from pathlib import Path

from ..data.schemas import IndexConfig
from ..util.gitio import discover_git_repository


class FileDiscovery:
    """
    Git-aware file discovery with intelligent filtering and change detection.

    Provides the foundation for incremental indexing by tracking file changes
    and maintaining content hashes for cache invalidation.
    """

    def __init__(self, repo_path: str | Path):
        """Initialize file discovery for repository."""
        self.repo_path = Path(repo_path)
        self.git_repo = discover_git_repository(repo_path)
        self.repo_root = self.git_repo.get_repo_root()

    async def discover_files(
        self, extensions: list[str] | None = None, excludes: list[str] | None = None
    ) -> list[str]:
        """
        Discover all trackable files in the repository.

        Returns list of file paths relative to repository root.
        """
        # Default extensions for code analysis
        if extensions is None:
            extensions = ["ts", "tsx", "js", "jsx", "md", "mdx", "json", "yaml", "yml"]

        # Default exclusion patterns
        if excludes is None:
            excludes = [
                "node_modules/",
                "dist/",
                "build/",
                ".next/",
                "coverage/",
                "__pycache__/",
                ".git/",
                ".vscode/",
                ".idea/",
                "target/",
                ".cache/",
                "tmp/",
                "temp/",
            ]

        # Get tracked files from git
        tracked_files = await asyncio.to_thread(
            self.git_repo.list_tracked_files, extensions=extensions, excludes=excludes
        )

        # Filter by actual file existence (in case of git inconsistencies)
        existing_files = []
        for file_path in tracked_files:
            full_path = self.repo_root / file_path
            if full_path.exists() and full_path.is_file():
                existing_files.append(file_path)

        return existing_files

    async def detect_changes(
        self, previous_files: list[str], previous_hashes: dict[str, str]
    ) -> tuple[list[str], list[str], list[str], dict[str, str]]:
        """
        Detect file changes since previous indexing.

        Returns:
        - added_files: New files not in previous index
        - modified_files: Files with content changes
        - removed_files: Files no longer tracked
        - current_hashes: Updated hash mapping
        """
        # Get current file list
        current_files = await self.discover_files()
        current_file_set = set(current_files)
        previous_file_set = set(previous_files)

        # Detect additions and removals
        added_files = list(current_file_set - previous_file_set)
        removed_files = list(previous_file_set - current_file_set)

        # Check for modifications in existing files
        modified_files = []
        current_hashes = {}

        # Batch process file hashing
        for file_path in current_files:
            try:
                # Get current hash
                current_hash = await asyncio.to_thread(self.git_repo.hash_file_content, file_path)
                current_hashes[file_path] = current_hash

                # Check if modified
                if file_path in previous_hashes:
                    if previous_hashes[file_path] != current_hash:
                        modified_files.append(file_path)

            except Exception:
                # If we can't hash the file, consider it modified
                if file_path in previous_files:
                    modified_files.append(file_path)

        return added_files, modified_files, removed_files, current_hashes

    async def get_file_metadata(self, file_paths: list[str]) -> dict[str, dict]:
        """Get metadata for a list of files."""
        metadata = {}

        # Process files in batches to avoid overwhelming the system
        batch_size = 50
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i : i + batch_size]
            batch_metadata = await asyncio.gather(
                *[self._get_single_file_metadata(fp) for fp in batch], return_exceptions=True
            )

            for file_path, meta in zip(batch, batch_metadata, strict=False):
                if not isinstance(meta, Exception):
                    metadata[file_path] = meta

        return metadata

    async def _get_single_file_metadata(self, file_path: str) -> dict:
        """Get metadata for a single file."""
        full_path = self.repo_root / file_path

        try:
            stat = await asyncio.to_thread(full_path.stat)
            content_hash = await asyncio.to_thread(self.git_repo.hash_file_content, file_path)

            return {
                "path": file_path,
                "size": stat.st_size,
                "modified": stat.st_mtime,
                "hash": content_hash,
                "extension": full_path.suffix.lstrip("."),
                "is_binary": await self._is_binary_file(full_path),
            }
        except Exception as e:
            return {"path": file_path, "error": str(e), "exists": False}

    async def _is_binary_file(self, file_path: Path) -> bool:
        """Check if file is binary by examining initial bytes."""
        try:
            # Read first 8KB to check for binary content
            chunk_size = 8192
            with open(file_path, "rb") as f:
                chunk = f.read(chunk_size)

            # Check for null bytes (common binary indicator)
            if b"\x00" in chunk:
                return True

            # Check for high ratio of non-printable characters
            non_printable = sum(1 for byte in chunk if byte < 32 and byte not in {9, 10, 13})
            ratio = non_printable / len(chunk) if chunk else 0

            return ratio > 0.3

        except Exception:
            return False

    async def compute_dirty_overlay(self) -> dict[str, str]:
        """
        Compute overlay of uncommitted changes.

        Returns mapping of file path -> content hash for dirty files.
        """
        return await asyncio.to_thread(self.git_repo.compute_dirty_overlay)

    async def get_cache_key(self, config: IndexConfig) -> str:
        """
        Generate cache key for incremental indexing.

        Combines repository state, configuration, and dirty overlay.
        """
        # Serialize config for hashing
        config_dict = config.dict()
        config_str = str(sorted(config_dict.items()))
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()

        return await asyncio.to_thread(self.git_repo.get_cache_key, config_hash)

    async def detect_workspaces(self) -> list[dict[str, str]]:
        """
        Detect monorepo workspace configuration.

        Returns list of workspace information for advanced repository analysis.
        """
        return await asyncio.to_thread(self.git_repo.detect_workspaces)

    async def get_file_content_with_context(
        self, file_path: str, start_line: int, end_line: int, context_lines: int = 5
    ) -> dict[str, str]:
        """
        Get file content with surrounding context lines.

        Useful for snippet extraction and context-aware analysis.
        """
        full_path = self.repo_root / file_path

        try:
            content = await asyncio.to_thread(full_path.read_text, encoding="utf-8")
            lines = content.split("\n")

            # Calculate context boundaries
            context_start = max(0, start_line - context_lines)
            context_end = min(len(lines), end_line + context_lines)

            # Extract sections
            pre_context = "\n".join(lines[context_start:start_line])
            main_content = "\n".join(lines[start_line:end_line])
            post_context = "\n".join(lines[end_line:context_end])

            return {
                "pre": pre_context,
                "content": main_content,
                "post": post_context,
                "line_start": start_line,
                "line_end": end_line,
                "context_start": context_start,
                "context_end": context_end,
            }

        except Exception as e:
            return {"error": str(e), "path": file_path}

    async def get_priority_files(
        self,
        all_files: list[str],
        dirty_files: set[str] | None = None,
        max_files: int | None = None,
    ) -> list[str]:
        """
        Get priority-ordered list of files for processing.

        Prioritizes dirty files and important files based on heuristics.
        """
        if dirty_files is None:
            dirty_files = set((await self.compute_dirty_overlay()).keys())

        priority_files = []

        # 1. Always prioritize dirty files
        for file_path in all_files:
            if file_path in dirty_files:
                priority_files.append(file_path)

        # 2. Add remaining files with importance heuristics
        remaining_files = [f for f in all_files if f not in dirty_files]

        # Sort by importance heuristics
        def file_importance(file_path: str) -> int:
            score = 0

            # Higher score for certain file types
            if file_path.endswith((".ts", ".tsx")):
                score += 10
            elif file_path.endswith((".js", ".jsx")):
                score += 8
            elif file_path.endswith(".json"):
                score += 5

            # Higher score for root-level files
            if "/" not in file_path:
                score += 5

            # Higher score for config files
            filename = Path(file_path).name.lower()
            if filename in {
                "package.json",
                "tsconfig.json",
                "index.ts",
                "index.js",
                "main.ts",
                "main.js",
            }:
                score += 15

            # Higher score for shorter paths (likely more important)
            score += max(0, 10 - file_path.count("/"))

            return score

        remaining_files.sort(key=file_importance, reverse=True)
        priority_files.extend(remaining_files)

        # Apply max files limit if specified
        if max_files is not None:
            priority_files = priority_files[:max_files]

        return priority_files

    async def validate_repository_structure(self) -> dict[str, any]:
        """
        Validate repository structure and detect potential issues.

        Returns validation report with warnings and recommendations.
        """
        report = {"valid": True, "warnings": [], "recommendations": [], "stats": {}}

        try:
            # Get basic repository info
            files = await self.discover_files()
            workspaces = await self.detect_workspaces()

            report["stats"] = {
                "total_files": len(files),
                "workspaces": len(workspaces),
                "file_types": {},
            }

            # Analyze file type distribution
            for file_path in files:
                ext = Path(file_path).suffix.lstrip(".")
                report["stats"]["file_types"][ext] = report["stats"]["file_types"].get(ext, 0) + 1

            # Check for common issues

            # 1. Too many files warning
            if len(files) > 10000:
                report["warnings"].append(
                    f"Large repository with {len(files)} files. Consider using excludes to focus indexing."
                )

            # 2. Missing important files
            important_files = {"package.json", "tsconfig.json", "README.md"}
            existing_files = {Path(f).name for f in files}
            missing_important = important_files - existing_files

            if missing_important:
                report["recommendations"].append(
                    f"Consider adding missing project files: {', '.join(missing_important)}"
                )

            # 3. Monorepo detection
            if workspaces:
                report["recommendations"].append(
                    f"Detected {len(workspaces)} workspaces. Consider workspace-specific indexing for better performance."
                )

            # 4. Binary file detection
            binary_count = 0
            for file_path in files[:100]:  # Sample first 100 files
                full_path = self.repo_root / file_path
                if await self._is_binary_file(full_path):
                    binary_count += 1

            if binary_count > 10:
                report["warnings"].append(
                    "Detected potential binary files. Consider adding binary extensions to excludes."
                )

        except Exception as e:
            report["valid"] = False
            report["warnings"].append(f"Validation failed: {str(e)}")

        return report
