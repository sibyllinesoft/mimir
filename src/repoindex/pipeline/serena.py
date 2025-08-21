"""
Serena adapter for TypeScript symbol analysis.

Provides integration with Serena for TypeScript Server orchestration,
symbol resolution, and first-order dependency analysis.
"""

import asyncio
import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..data.schemas import IndexConfig, SerenaGraph, SymbolEntry, SymbolType
from ..util.fs import atomic_write_json


class SerenaError(Exception):
    """Serena operation error."""

    pass


class SerenaAdapter:
    """
    Adapter for Serena TypeScript analysis.

    Manages TypeScript Server integration for symbol resolution,
    import analysis, and first-order dependency type extraction.
    """

    def __init__(self, serena_path: str | None = None, validate: bool = True):
        """Initialize Serena adapter."""
        self.serena_path = serena_path or "serena"
        if validate:
            self._validate_serena()

    def _validate_serena(self) -> None:
        """Validate Serena installation."""
        try:
            result = subprocess.run(
                [self.serena_path, "--version"], capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                raise SerenaError("Serena not properly installed")
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            raise SerenaError(
                f"Serena not found at {self.serena_path}. "
                "Please install Serena or provide correct path."
            )

    async def analyze_project(
        self,
        project_root: Path,
        files: list[str],
        work_dir: Path,
        config: IndexConfig,
        progress_callback: Callable[[int], None] | None = None,
    ) -> SerenaGraph:
        """
        Run complete Serena analysis on TypeScript project.

        Returns SerenaGraph with symbol definitions, references, and call relationships.
        """
        if progress_callback:
            progress_callback(5)

        # Filter TypeScript/JavaScript files
        ts_files = [
            f for f in files if any(f.endswith(ext) for ext in [".ts", ".tsx", ".js", ".jsx"])
        ]

        if not ts_files:
            # Return empty graph if no TypeScript files
            return SerenaGraph(entries=[], file_count=0, symbol_count=0)

        if progress_callback:
            progress_callback(10)

        # Set up Serena configuration
        serena_config = await self._create_serena_config(project_root, ts_files, work_dir, config)

        if progress_callback:
            progress_callback(20)

        # Run symbol analysis
        symbol_entries = await self._run_symbol_analysis(
            project_root, serena_config, work_dir, progress_callback
        )

        if progress_callback:
            progress_callback(70)

        # Extract first-order dependency types
        if config.imports_policy.get("types_for_first_order", True):
            await self._extract_dependency_types(project_root, serena_config, work_dir)

        if progress_callback:
            progress_callback(85)

        # Extract vendor sources for direct imports
        if config.imports_policy.get("code_for_direct_imports", True):
            await self._extract_vendor_sources(project_root, ts_files, work_dir)

        if progress_callback:
            progress_callback(100)

        # Create SerenaGraph
        graph = SerenaGraph(
            entries=symbol_entries,
            file_count=len(ts_files),
            symbol_count=len({entry.symbol for entry in symbol_entries if entry.symbol}),
        )

        # Save graph to work directory
        graph.save_to_jsonl(work_dir / "serena_graph.jsonl")

        return graph

    async def _create_serena_config(
        self, project_root: Path, files: list[str], work_dir: Path, config: IndexConfig
    ) -> dict[str, Any]:
        """Create Serena configuration file."""
        serena_config = {
            "project_root": str(project_root),
            "files": files,
            "output_dir": str(work_dir),
            "analysis": {
                "symbols": True,
                "references": True,
                "calls": True,
                "imports": True,
                "types": True,
            },
            "tsconfig": {
                "auto_detect": True,
                "fallback": {
                    "target": "ES2020",
                    "module": "ESNext",
                    "moduleResolution": "node",
                    "allowJs": True,
                    "strict": False,
                },
            },
            "output_format": "jsonl",
        }

        # Save config file
        config_path = work_dir / "serena_config.json"
        atomic_write_json(config_path, serena_config)

        return serena_config

    async def _run_symbol_analysis(
        self,
        project_root: Path,
        serena_config: dict[str, Any],
        work_dir: Path,
        progress_callback: Callable[[int], None] | None = None,
    ) -> list[SymbolEntry]:
        """Run main symbol analysis."""
        config_path = work_dir / "serena_config.json"
        output_path = work_dir / "serena_symbols.jsonl"

        cmd = [
            self.serena_path,
            "analyze",
            "--config",
            str(config_path),
            "--output",
            str(output_path),
            "--format",
            "jsonl",
        ]

        if progress_callback:
            progress_callback(30)

        try:
            # Start Serena process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            # Monitor progress (Serena outputs progress to stderr)
            stderr_data = []
            while True:
                try:
                    line = await asyncio.wait_for(process.stderr.readline(), timeout=1.0)
                    if not line:
                        break

                    line_str = line.decode().strip()
                    stderr_data.append(line_str)

                    # Parse progress if available
                    if "Progress:" in line_str and progress_callback:
                        try:
                            # Extract percentage from line like "Progress: 45%"
                            progress_str = line_str.split("Progress:")[1].strip()
                            progress_pct = int(progress_str.replace("%", ""))
                            # Map to range 30-70
                            mapped_progress = 30 + int((progress_pct / 100.0) * 40)
                            progress_callback(mapped_progress)
                        except (ValueError, IndexError):
                            pass

                except TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        break
                    continue

            # Wait for completion
            await process.wait()

            if process.returncode != 0:
                error_msg = "\n".join(stderr_data) if stderr_data else "Unknown error"
                raise SerenaError(f"Serena analysis failed: {error_msg}")

            # Parse symbol entries
            symbol_entries = []
            if output_path.exists():
                with open(output_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = json.loads(line)
                                entry = self._parse_symbol_entry(data)
                                if entry:
                                    symbol_entries.append(entry)
                            except (json.JSONDecodeError, ValueError):
                                continue

            return symbol_entries

        except Exception as e:
            if isinstance(e, SerenaError):
                raise
            raise SerenaError(f"Failed to run Serena analysis: {str(e)}")

    def _parse_symbol_entry(self, data: dict[str, Any]) -> SymbolEntry | None:
        """Parse a single symbol entry from Serena output."""
        try:
            # Map Serena entry types to our enum
            entry_type_map = {
                "definition": SymbolType.DEF,
                "reference": SymbolType.REF,
                "call": SymbolType.CALL,
                "import": SymbolType.IMPORT,
            }

            entry_type = entry_type_map.get(data.get("type"))
            if not entry_type:
                return None

            # Create symbol entry
            entry = SymbolEntry(
                type=entry_type,
                path=data.get("path", ""),
                span=(data.get("start", 0), data.get("end", 0)),
            )

            # Add optional fields based on entry type
            if "symbol" in data:
                entry.symbol = data["symbol"]
            if "signature" in data:
                entry.sig = data["signature"]
            if "caller" in data:
                entry.caller = data["caller"]
            if "callee" in data:
                entry.callee = data["callee"]
            if "from" in data:
                entry.from_module = data["from"]

            return entry

        except Exception:
            return None

    async def _extract_dependency_types(
        self, project_root: Path, serena_config: dict[str, Any], work_dir: Path
    ) -> None:
        """Extract .d.ts files for first-order dependencies."""
        types_dir = work_dir / "types"
        types_dir.mkdir(exist_ok=True)

        cmd = [
            self.serena_path,
            "extract-types",
            "--project",
            str(project_root),
            "--output",
            str(types_dir),
            "--first-order-only",
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                # Don't fail the whole pipeline for type extraction
                print(
                    f"Warning: Type extraction failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

        except Exception as e:
            print(f"Warning: Failed to extract dependency types: {str(e)}")

    async def _extract_vendor_sources(
        self, project_root: Path, files: list[str], work_dir: Path
    ) -> None:
        """Extract vendor source code for direct imports."""
        vendor_dir = work_dir / "vendor_src"
        vendor_dir.mkdir(exist_ok=True)

        # Create file list for import analysis
        files_list_path = work_dir / "ts_files.txt"
        with open(files_list_path, "w") as f:
            for file_path in files:
                f.write(f"{file_path}\n")

        cmd = [
            self.serena_path,
            "extract-vendor",
            "--project",
            str(project_root),
            "--files",
            str(files_list_path),
            "--output",
            str(vendor_dir),
            "--direct-imports-only",
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                # Don't fail the whole pipeline for vendor extraction
                print(
                    f"Warning: Vendor source extraction failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

        except Exception as e:
            print(f"Warning: Failed to extract vendor sources: {str(e)}")

    async def resolve_symbol_references(
        self, project_root: Path, symbol_name: str, work_dir: Path
    ) -> list[dict[str, Any]]:
        """
        Resolve all references to a specific symbol.

        Useful for targeted analysis and navigation.
        """
        output_path = work_dir / f"refs_{symbol_name.replace('/', '_')}.json"

        cmd = [
            self.serena_path,
            "find-references",
            "--project",
            str(project_root),
            "--symbol",
            symbol_name,
            "--output",
            str(output_path),
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                raise SerenaError(
                    f"Reference resolution failed: {stderr.decode() if stderr else 'Unknown error'}"
                )

            if output_path.exists():
                with open(output_path) as f:
                    return json.load(f)

            return []

        except Exception as e:
            if isinstance(e, SerenaError):
                raise
            raise SerenaError(f"Failed to resolve symbol references: {str(e)}")

    async def get_symbol_definition(
        self, project_root: Path, file_path: str, position: int, work_dir: Path
    ) -> dict[str, Any] | None:
        """
        Get symbol definition at specific file position.

        Returns definition information including signature and location.
        """
        cmd = [
            self.serena_path,
            "definition",
            "--project",
            str(project_root),
            "--file",
            file_path,
            "--position",
            str(position),
        ]

        try:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            stdout, stderr = await result.communicate()

            if result.returncode != 0:
                return None  # Definition not found

            if stdout:
                try:
                    return json.loads(stdout.decode())
                except json.JSONDecodeError:
                    return None

            return None

        except Exception:
            return None

    async def validate_typescript_project(self, project_root: Path) -> dict[str, Any]:
        """
        Validate TypeScript project configuration.

        Returns validation report with project health and recommendations.
        """
        validation_report = {
            "valid": True,
            "has_tsconfig": False,
            "typescript_files": 0,
            "javascript_files": 0,
            "issues": [],
            "recommendations": [],
        }

        try:
            # Check for tsconfig.json
            tsconfig_path = project_root / "tsconfig.json"
            validation_report["has_tsconfig"] = tsconfig_path.exists()

            if not validation_report["has_tsconfig"]:
                validation_report["recommendations"].append(
                    "Add tsconfig.json for better TypeScript analysis"
                )

            # Count file types
            for file_path in project_root.rglob("*"):
                if file_path.is_file():
                    if file_path.suffix in [".ts", ".tsx"]:
                        validation_report["typescript_files"] += 1
                    elif file_path.suffix in [".js", ".jsx"]:
                        validation_report["javascript_files"] += 1

            # Check project structure
            if validation_report["typescript_files"] == 0:
                validation_report["issues"].append("No TypeScript files found for analysis")

            # Run Serena validation
            cmd = [self.serena_path, "validate", "--project", str(project_root)]

            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=project_root,
            )

            stdout, stderr = await result.communicate()

            if result.returncode == 0 and stdout:
                try:
                    serena_validation = json.loads(stdout.decode())
                    validation_report.update(serena_validation)
                except json.JSONDecodeError:
                    pass

        except Exception as e:
            validation_report["valid"] = False
            validation_report["issues"].append(f"Validation failed: {str(e)}")

        return validation_report
