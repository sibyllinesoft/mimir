"""
RepoMapper adapter for repository structure analysis.

Provides integration with RepoMapper for Tree-sitter-based AST analysis
and PageRank-based file importance ranking.
"""

import asyncio
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..data.schemas import DependencyEdge, FileRank, RepoMap


class RepoMapperError(Exception):
    """RepoMapper operation error."""

    pass


class RepoMapperAdapter:
    """
    Adapter for RepoMapper integration.

    Manages RepoMapper execution and result parsing for repository
    structure analysis and file importance ranking.
    """

    def __init__(self, repomapper_path: str | None = None, validate: bool = True):
        """Initialize RepoMapper adapter."""
        self.repomapper_path = repomapper_path or "repomapper"
        if validate:
            self._validate_repomapper()

    def _validate_repomapper(self) -> None:
        """Validate RepoMapper installation."""
        try:
            result = subprocess.run(
                [self.repomapper_path, "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise RepoMapperError("RepoMapper not properly installed")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise RepoMapperError(
                f"RepoMapper not found at {self.repomapper_path}. "
                "Please install RepoMapper or provide correct path."
            )

    async def analyze_repository(
        self,
        repo_root: Path,
        files: list[str],
        work_dir: Path,
        progress_callback: Callable[[int], None] | None = None,
    ) -> RepoMap:
        """
        Run RepoMapper analysis on repository.

        Returns RepoMap with file rankings and dependency edges.
        """
        if progress_callback:
            progress_callback(10)

        # Create temporary file list for RepoMapper
        files_list_path = work_dir / "repomapper_files.txt"
        with open(files_list_path, "w") as f:
            for file_path in files:
                f.write(f"{file_path}\n")

        # Set up RepoMapper command
        output_path = work_dir / "repomap.json"
        cmd = [
            self.repomapper_path,
            "--repo-root",
            str(repo_root),
            "--files-list",
            str(files_list_path),
            "--output",
            str(output_path),
            "--format",
            "json",
            "--include-edges",
            "--page-rank",
        ]

        if progress_callback:
            progress_callback(30)

        try:
            # Run RepoMapper analysis
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=repo_root
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise RepoMapperError(
                    f"RepoMapper failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

            if progress_callback:
                progress_callback(80)

            # Load and parse results
            with open(output_path) as f:
                raw_data = json.load(f)

            repomap = self._parse_repomapper_output(raw_data)

            if progress_callback:
                progress_callback(100)

            return repomap

        except Exception as e:
            if isinstance(e, RepoMapperError):
                raise
            raise RepoMapperError(f"Failed to run RepoMapper: {str(e)}")

    def _parse_repomapper_output(self, raw_data: dict[str, Any]) -> RepoMap:
        """Parse RepoMapper JSON output into structured data."""
        file_ranks = []
        edges = []

        # Parse file rankings
        for file_info in raw_data.get("files", []):
            file_rank = FileRank(
                path=file_info["path"],
                rank=file_info.get("rank", 0.0),
                centrality=file_info.get("centrality", 0.0),
                dependencies=file_info.get("dependencies", []),
            )
            file_ranks.append(file_rank)

        # Parse dependency edges
        for edge_info in raw_data.get("edges", []):
            edge = DependencyEdge(
                source=edge_info["source"],
                target=edge_info["target"],
                weight=edge_info.get("weight", 1.0),
                edge_type=edge_info.get("type", "import"),
            )
            edges.append(edge)

        return RepoMap(file_ranks=file_ranks, edges=edges, total_files=len(file_ranks))

    async def analyze_file_subset(
        self,
        repo_root: Path,
        files: list[str],
        work_dir: Path,
        focus_files: list[str],
        progress_callback: Callable[[int], None] | None = None,
    ) -> RepoMap:
        """
        Analyze specific subset of files with focused analysis.

        Useful for incremental updates or targeted analysis.
        """
        if progress_callback:
            progress_callback(10)

        # Create file lists
        all_files_path = work_dir / "all_files.txt"
        focus_files_path = work_dir / "focus_files.txt"

        with open(all_files_path, "w") as f:
            for file_path in files:
                f.write(f"{file_path}\n")

        with open(focus_files_path, "w") as f:
            for file_path in focus_files:
                f.write(f"{file_path}\n")

        # Run focused analysis
        output_path = work_dir / "repomap_subset.json"
        cmd = [
            self.repomapper_path,
            "--repo-root",
            str(repo_root),
            "--files-list",
            str(all_files_path),
            "--focus-files",
            str(focus_files_path),
            "--output",
            str(output_path),
            "--format",
            "json",
            "--include-edges",
        ]

        if progress_callback:
            progress_callback(30)

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=repo_root
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise RepoMapperError(
                    f"RepoMapper subset analysis failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

            if progress_callback:
                progress_callback(80)

            # Parse results
            with open(output_path) as f:
                raw_data = json.load(f)

            repomap = self._parse_repomapper_output(raw_data)

            if progress_callback:
                progress_callback(100)

            return repomap

        except Exception as e:
            if isinstance(e, RepoMapperError):
                raise
            raise RepoMapperError(f"Failed to run RepoMapper subset analysis: {str(e)}")

    async def get_file_dependencies(
        self, repo_root: Path, file_path: str, work_dir: Path
    ) -> dict[str, Any]:
        """
        Get detailed dependency information for a specific file.

        Returns imports, exports, and dependency relationships.
        """
        output_path = work_dir / f"deps_{Path(file_path).name}.json"
        cmd = [
            self.repomapper_path,
            "--repo-root",
            str(repo_root),
            "--single-file",
            file_path,
            "--output",
            str(output_path),
            "--format",
            "json",
            "--include-symbols",
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=repo_root
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise RepoMapperError(
                    f"RepoMapper dependency analysis failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

            with open(output_path) as f:
                return json.load(f)

        except Exception as e:
            if isinstance(e, RepoMapperError):
                raise
            raise RepoMapperError(f"Failed to analyze file dependencies: {str(e)}")

    async def validate_analysis(
        self, repo_root: Path, files: list[str], work_dir: Path
    ) -> dict[str, Any]:
        """
        Validate RepoMapper analysis results.

        Returns validation report with statistics and potential issues.
        """
        validation_report = {"valid": True, "statistics": {}, "issues": [], "recommendations": []}

        try:
            # Run quick validation analysis
            cmd = [
                self.repomapper_path,
                "--repo-root",
                str(repo_root),
                "--validate",
                "--stats-only",
            ]

            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=repo_root
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0 and stdout:
                try:
                    stats = json.loads(stdout.decode())
                    validation_report["statistics"] = stats

                    # Add recommendations based on statistics
                    if stats.get("parse_errors", 0) > 0:
                        validation_report["issues"].append(
                            f"Found {stats['parse_errors']} parse errors in files"
                        )

                    if stats.get("total_symbols", 0) == 0:
                        validation_report["issues"].append(
                            "No symbols detected - check file types and content"
                        )

                    cycles = stats.get("circular_dependencies", 0)
                    if cycles > 0:
                        validation_report["recommendations"].append(
                            f"Found {cycles} circular dependencies - consider refactoring"
                        )

                except json.JSONDecodeError:
                    validation_report["issues"].append("Could not parse validation statistics")

        except Exception as e:
            validation_report["valid"] = False
            validation_report["issues"].append(f"Validation failed: {str(e)}")

        return validation_report
