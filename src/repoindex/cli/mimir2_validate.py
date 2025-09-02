#!/usr/bin/env python3
"""
Mimir 2.0 Validation CLI Tool.

Validates the Mimir 2.0 setup including Ollama connectivity,
ML dependencies, and feature availability.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any

import click
from rich.console import Console  
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint

# Add the parent directory to the path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from repoindex.config import get_ai_config
from repoindex.pipeline.integration_helpers import run_integration_validation
from repoindex.pipeline.pipeline_coordinator import get_pipeline_coordinator


console = Console()


@click.group()
def cli():
    """Mimir 2.0 validation and setup tools."""
    pass


@cli.command()
@click.option('--json-output', is_flag=True, help='Output results as JSON')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
async def validate(json_output: bool, verbose: bool):
    """Validate Mimir 2.0 setup and configuration."""
    console.print("[bold blue]🔍 Validating Mimir 2.0 Setup[/bold blue]")
    
    try:
        # Run comprehensive validation
        results = await run_integration_validation()
        
        if json_output:
            # JSON output for programmatic use
            print(json.dumps(results, indent=2, default=str))
            return
        
        # Rich formatted output
        _display_validation_results(results, verbose)
        
        # Exit with appropriate code
        sys.exit(0 if results["overall_success"] else 1)
        
    except Exception as e:
        console.print(f"[bold red]❌ Validation failed with error: {e}[/bold red]")
        if verbose:
            console.print_exception()
        sys.exit(1)


@cli.command()
@click.option('--json-output', is_flag=True, help='Output results as JSON')
async def config(json_output: bool):
    """Display current Mimir 2.0 configuration."""
    try:
        config = get_ai_config()
        coordinator = await get_pipeline_coordinator(config)
        report = await coordinator.validate_configuration()
        
        if json_output:
            print(json.dumps(report, indent=2, default=str))
            return
        
        _display_config_report(report)
        
    except Exception as e:
        console.print(f"[bold red]❌ Configuration check failed: {e}[/bold red]")
        sys.exit(1)


@cli.command()
async def dependencies():
    """Check ML and AI dependencies."""
    console.print("[bold blue]🔍 Checking Dependencies[/bold blue]")
    
    dependencies = {
        "Core": {
            "aiohttp": "HTTP client for Ollama",
            "sentence_transformers": "Embedding models",
        },
        "ML/Clustering": {
            "sklearn": "Machine learning algorithms", 
            "umap": "Dimensionality reduction",
            "hdbscan": "Hierarchical clustering",
        },
        "Deep Learning": {
            "torch": "PyTorch for transformer models",
            "transformers": "Hugging Face transformers",
        }
    }
    
    for category, deps in dependencies.items():
        table = Table(title=f"{category} Dependencies")
        table.add_column("Package", style="cyan")
        table.add_column("Purpose", style="white")
        table.add_column("Status", style="bold")
        
        for package, purpose in deps.items():
            try:
                __import__(package)
                status = "[green]✅ Available[/green]"
            except ImportError:
                status = "[red]❌ Missing[/red]"
            
            table.add_row(package, purpose, status)
        
        console.print(table)
        console.print()


@cli.command()
@click.option('--host', default='localhost', help='Ollama host')
@click.option('--port', default=11434, help='Ollama port')
async def ollama(host: str, port: int):
    """Test Ollama connectivity."""
    import aiohttp
    
    console.print(f"[bold blue]🔍 Testing Ollama at {host}:{port}[/bold blue]")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"http://{host}:{port}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    
                    console.print(f"[green]✅ Ollama server accessible[/green]")
                    console.print(f"[blue]📦 Available models: {len(models)}[/blue]")
                    
                    if models:
                        table = Table(title="Available Models")
                        table.add_column("Model", style="cyan")
                        table.add_column("Size", style="white")
                        table.add_column("Modified", style="dim")
                        
                        for model in models:
                            size_mb = model.get("size", 0) // (1024 * 1024)
                            table.add_row(
                                model["name"], 
                                f"{size_mb} MB",
                                model.get("modified_at", "Unknown")[:10]
                            )
                        
                        console.print(table)
                else:
                    console.print(f"[red]❌ Ollama server error: HTTP {response.status}[/red]")
                    
    except aiohttp.ClientConnectorError:
        console.print(f"[red]❌ Cannot connect to Ollama at {host}:{port}[/red]")
        console.print("[yellow]💡 Make sure Ollama is running: `ollama serve`[/yellow]")
    except Exception as e:
        console.print(f"[red]❌ Ollama test failed: {e}[/red]")


def _display_validation_results(results: Dict[str, Any], verbose: bool):
    """Display validation results in rich format."""
    
    # Overall status
    if results["overall_success"]:
        console.print("✅ [bold green]All components validated successfully![/bold green]")
    else:
        console.print("❌ [bold red]Some components failed validation[/bold red]")
    
    console.print()
    
    # Pipeline info
    mode = results.get("pipeline_mode", "unknown")
    console.print(f"📋 [bold]Pipeline Mode:[/bold] {mode.upper()}")
    
    # Capabilities
    capabilities = results.get("capabilities")
    if capabilities:
        cap_table = Table(title="Available Capabilities")
        cap_table.add_column("Feature", style="cyan") 
        cap_table.add_column("Available", style="bold")
        
        features = {
            "Ollama LLM": capabilities.has_ollama,
            "RAPTOR Indexing": capabilities.has_raptor,
            "HyDE Queries": capabilities.has_hyde,
            "Reranking": capabilities.has_reranking,
            "Code Embeddings": capabilities.has_code_embeddings,
        }
        
        for feature, available in features.items():
            status = "[green]✅[/green]" if available else "[red]❌[/red]"
            cap_table.add_row(feature, status)
        
        console.print(cap_table)
        console.print()
    
    # Component results
    if verbose and "results" in results:
        _display_component_results(results["results"])


def _display_component_results(results: Dict[str, Any]):
    """Display detailed component results."""
    for phase, phase_results in results.items():
        console.print(f"[bold]{phase.title()} Results:[/bold]")
        
        for component, result in phase_results.items():
            status = "✅" if result.success else "❌"
            message = result.message
            
            console.print(f"  {status} {component}: {message}")
        
        console.print()


def _display_config_report(report: Dict[str, Any]):
    """Display configuration report."""
    # Pipeline mode
    console.print(f"📋 [bold]Pipeline Mode:[/bold] {report['pipeline_mode'].upper()}")
    console.print()
    
    # Config status
    config_table = Table(title="Configuration Status")
    config_table.add_column("Feature", style="cyan")
    config_table.add_column("Configured", style="white") 
    config_table.add_column("Available", style="bold")
    
    config_status = report.get("config_status", {})
    capabilities = report.get("capabilities", {})
    
    features = [
        ("Ollama", "ollama_enabled", "ollama"),
        ("RAPTOR", "raptor_enabled", "raptor"),
        ("HyDE", "hyde_enabled", "hyde"),
        ("Reranking", "reranking_enabled", "reranking"),
        ("Code Embeddings", "code_embeddings_enabled", "code_embeddings"),
    ]
    
    for name, config_key, cap_key in features:
        configured = "✅" if config_status.get(config_key, False) else "❌"
        available = "✅" if capabilities.get(cap_key, False) else "❌"
        config_table.add_row(name, configured, available)
    
    console.print(config_table)
    console.print()
    
    # Warnings and errors
    if report.get("warnings"):
        console.print("[yellow]⚠️  Warnings:[/yellow]")
        for warning in report["warnings"]:
            console.print(f"  • {warning}")
        console.print()
    
    if report.get("errors"):
        console.print("[red]❌ Errors:[/red]")
        for error in report["errors"]:
            console.print(f"  • {error}")
        console.print()


# Make commands async-aware
for cmd in [validate, config, ollama]:
    cmd = click.command()(lambda: asyncio.run(cmd.callback()))


if __name__ == "__main__":
    # Fix async command handling
    import inspect
    
    for name, obj in cli.commands.items():
        if inspect.iscoroutinefunction(obj.callback):
            original_callback = obj.callback
            obj.callback = lambda *args, **kwargs: asyncio.run(original_callback(*args, **kwargs))
    
    cli()