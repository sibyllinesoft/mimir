"""
Bundle creation for index artifacts.

Creates compressed bundles of all pipeline artifacts with manifest
generation and integrity verification.
"""

import asyncio
import tarfile
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import zstandard as zstd

from ..data.schemas import ArtifactPaths, IndexCounts, IndexManifest, ToolVersions
from ..util.fs import get_directory_size


class BundleError(Exception):
    """Bundle creation error."""

    pass


class BundleCreator:
    """
    Creates compressed bundles of index artifacts.

    Manages final manifest creation, artifact bundling, and integrity
    verification for complete index packages.
    """

    def __init__(self):
        """Initialize bundle creator."""
        self.max_bundle_size = 2 * 1024 * 1024 * 1024  # 2GB
        self.compression_level = 3  # zstd compression level

    async def create_bundle(
        self, context: "PipelineContext", progress_callback: Callable[[int], None] | None = None
    ) -> IndexManifest:
        """
        Create complete index bundle with manifest.

        Returns final IndexManifest with all artifact information.
        """
        if progress_callback:
            progress_callback(10)

        # Create final manifest
        manifest = await self._create_manifest(context)

        if progress_callback:
            progress_callback(30)

        # Validate all artifacts exist
        await self._validate_artifacts(context.work_dir, manifest.paths)

        if progress_callback:
            progress_callback(50)

        # Create compressed bundle
        bundle_path = await self._create_compressed_bundle(
            context.work_dir, manifest.paths, progress_callback
        )

        if progress_callback:
            progress_callback(90)

        # Update manifest with final bundle info
        manifest.paths.bundle = bundle_path.name
        manifest.updated_at = datetime.utcnow()

        if progress_callback:
            progress_callback(100)

        return manifest

    async def _create_manifest(self, context: "PipelineContext") -> IndexManifest:
        """Create complete index manifest."""
        # Calculate counts from artifacts
        counts = await self._calculate_counts(context)

        # Get tool versions
        versions = await self._get_tool_versions()

        # Create manifest
        manifest = IndexManifest(
            index_id=context.index_id,
            repo=context.repo_info,
            config=context.config,
            counts=counts,
            paths=ArtifactPaths(),
            versions=versions,
        )

        return manifest

    async def _calculate_counts(self, context: "PipelineContext") -> IndexCounts:
        """Calculate statistics from pipeline artifacts."""
        counts = IndexCounts()

        # File counts
        counts.files_total = len(context.tracked_files)
        counts.files_indexed = len(context.tracked_files)  # All tracked files are indexed

        # Symbol counts from Serena graph
        if context.serena_graph:
            for entry in context.serena_graph.entries:
                if entry.type.value == "def":
                    counts.symbols_defs += 1
                elif entry.type.value == "ref":
                    counts.symbols_refs += 1

        # Vector counts from LEANN index
        if context.vector_index:
            counts.vectors = len(context.vector_index.chunks)
            counts.chunks = len(context.vector_index.chunks)

        return counts

    async def _get_tool_versions(self) -> ToolVersions:
        """Get versions of all tools used in pipeline."""
        versions = ToolVersions()

        # Get tool versions through subprocess calls
        tools = {
            "repomapper": ["repomapper", "--version"],
            "serena": ["serena", "--version"],
            "leann": ["leann", "--version"],
            "tsserver": ["tsc", "--version"],
        }

        for tool_name, cmd in tools.items():
            try:
                result = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await result.communicate()

                if result.returncode == 0 and stdout:
                    version_str = stdout.decode().strip()
                    # Extract version number from output
                    version = self._extract_version(version_str)
                    setattr(versions, tool_name, version)

            except Exception:
                # If we can't get version, leave as empty string
                continue

        return versions

    def _extract_version(self, version_string: str) -> str:
        """Extract version number from tool output."""
        import re

        # Try to find version pattern (x.y.z)
        version_pattern = r"(\d+\.\d+\.\d+)"
        match = re.search(version_pattern, version_string)

        if match:
            return match.group(1)

        # Fall back to first line
        return version_string.split("\n")[0].strip()

    async def _validate_artifacts(self, work_dir: Path, paths: ArtifactPaths) -> None:
        """Validate that all expected artifacts exist."""
        required_files = [paths.repomap, paths.serena_graph, paths.snippets]

        # Vector files are optional if vector search is disabled
        optional_files = [paths.leann_index, paths.vectors]

        missing_required = []
        for file_path in required_files:
            full_path = work_dir / file_path
            if not full_path.exists():
                missing_required.append(file_path)

        if missing_required:
            raise BundleError(f"Missing required artifacts: {missing_required}")

        # Check optional files and log warnings
        missing_optional = []
        for file_path in optional_files:
            full_path = work_dir / file_path
            if not full_path.exists():
                missing_optional.append(file_path)

        if missing_optional:
            print(f"Warning: Missing optional artifacts: {missing_optional}")

    async def _create_compressed_bundle(
        self,
        work_dir: Path,
        paths: ArtifactPaths,
        progress_callback: Callable[[int], None] | None = None,
    ) -> Path:
        """Create compressed tar.zst bundle of all artifacts."""
        bundle_path = work_dir / "bundle.tar.zst"

        # Check total size before bundling
        total_size = await asyncio.to_thread(get_directory_size, work_dir)
        if total_size > self.max_bundle_size:
            print(
                f"Warning: Bundle size ({total_size / 1024 / 1024:.1f}MB) exceeds recommended limit"
            )

        # Create list of files to bundle
        bundle_files = []

        # Add all artifact files
        for field_name in paths.model_fields.keys():
            file_path = getattr(paths, field_name)
            if isinstance(file_path, str):  # Ensure it's a string path
                full_path = work_dir / file_path
                if full_path.exists():
                    bundle_files.append(file_path)

        # Add additional files if they exist
        additional_files = ["manifest.json", "status.json", "log.md", "events.jsonl", "chunks.json"]

        for file_name in additional_files:
            file_path = work_dir / file_name
            if file_path.exists():
                bundle_files.append(file_name)

        # Create compressed tar archive
        await asyncio.to_thread(
            self._create_zstd_tar, work_dir, bundle_files, bundle_path, progress_callback
        )

        return bundle_path

    def _create_zstd_tar(
        self,
        work_dir: Path,
        files: list[str],
        output_path: Path,
        progress_callback: Callable[[int], None] | None = None,
    ) -> None:
        """Create zstd-compressed tar archive."""
        # Create zstd compressor
        compressor = zstd.ZstdCompressor(level=self.compression_level)

        # Create compressed tar file
        with open(output_path, "wb") as f:
            with compressor.stream_writer(f) as compressed_stream:
                with tarfile.open(fileobj=compressed_stream, mode="w|") as tar:
                    for i, file_path in enumerate(files):
                        full_path = work_dir / file_path
                        if full_path.exists():
                            tar.add(full_path, arcname=file_path)

                        # Update progress
                        if progress_callback:
                            progress = 50 + int((i / len(files)) * 40)
                            progress_callback(progress)

    async def extract_bundle(self, bundle_path: Path, extract_dir: Path) -> IndexManifest:
        """
        Extract bundle and return manifest.

        Useful for loading previously created indexes.
        """
        extract_dir.mkdir(parents=True, exist_ok=True)

        # Extract zstd tar archive
        await asyncio.to_thread(self._extract_zstd_tar, bundle_path, extract_dir)

        # Load and return manifest
        manifest_path = extract_dir / "manifest.json"
        if manifest_path.exists():
            return IndexManifest.load_from_file(manifest_path)
        else:
            raise BundleError("No manifest found in extracted bundle")

    def _extract_zstd_tar(self, bundle_path: Path, extract_dir: Path) -> None:
        """Extract zstd-compressed tar archive."""
        decompressor = zstd.ZstdDecompressor()

        with open(bundle_path, "rb") as f:
            with decompressor.stream_reader(f) as decompressed_stream:
                with tarfile.open(fileobj=decompressed_stream, mode="r|") as tar:
                    tar.extractall(path=extract_dir)

    async def validate_bundle(self, bundle_path: Path) -> dict:
        """
        Validate bundle integrity and contents.

        Returns validation report.
        """
        report = {
            "valid": True,
            "bundle_size": 0,
            "file_count": 0,
            "has_manifest": False,
            "issues": [],
        }

        try:
            # Check bundle file exists and get size
            if not bundle_path.exists():
                report["valid"] = False
                report["issues"].append("Bundle file does not exist")
                return report

            report["bundle_size"] = bundle_path.stat().st_size

            # Try to list bundle contents
            file_list = await asyncio.to_thread(self._list_bundle_contents, bundle_path)

            report["file_count"] = len(file_list)
            report["has_manifest"] = "manifest.json" in file_list

            if not report["has_manifest"]:
                report["issues"].append("Bundle missing manifest.json")

            # Check for required files
            required_files = ["repomap.json", "serena_graph.jsonl", "snippets.jsonl"]
            missing_files = [f for f in required_files if f not in file_list]

            if missing_files:
                report["issues"].append(f"Missing required files: {missing_files}")

            # Size warnings
            if report["bundle_size"] > self.max_bundle_size:
                report["issues"].append(
                    f"Bundle size ({report['bundle_size'] / 1024 / 1024:.1f}MB) exceeds recommended limit"
                )

        except Exception as e:
            report["valid"] = False
            report["issues"].append(f"Validation failed: {str(e)}")

        return report

    def _list_bundle_contents(self, bundle_path: Path) -> list[str]:
        """List contents of zstd tar bundle."""
        decompressor = zstd.ZstdDecompressor()
        file_list = []

        with open(bundle_path, "rb") as f:
            with decompressor.stream_reader(f) as decompressed_stream:
                with tarfile.open(fileobj=decompressed_stream, mode="r|") as tar:
                    for member in tar:
                        if member.isfile():
                            file_list.append(member.name)

        return file_list
