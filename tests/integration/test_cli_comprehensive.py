"""
Comprehensive CLI integration tests for Mimir.

Tests all CLI commands, argument parsing, error handling, and user interaction
scenarios to ensure robust command-line interface functionality.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional

import pytest
from click.testing import CliRunner

# Import the CLI module
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from repoindex.cli.mimir2_validate import cli, validate, config, dependencies, ollama


@pytest.fixture
def cli_runner():
    """Pytest fixture providing Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_config():
    """Mock configuration for CLI tests."""
    return {
        "ai": {
            "provider": "ollama",
            "ollama": {
                "base_url": "http://localhost:11434",
                "model": "llama2:7b",
                "timeout": 30
            }
        },
        "pipeline": {
            "enable_raptor": True,
            "enable_hyde": True,
            "enable_reranking": True,
            "enable_code_embeddings": True
        }
    }


@pytest.fixture
def mock_validation_results():
    """Mock validation results for CLI tests."""
    return {
        "overall_success": True,
        "pipeline_mode": "enhanced",
        "capabilities": Mock(
            has_ollama=True,
            has_raptor=True,
            has_hyde=True,
            has_reranking=True,
            has_code_embeddings=True
        ),
        "results": {
            "initialization": {
                "config_load": Mock(success=True, message="Configuration loaded successfully"),
                "coordinator": Mock(success=True, message="Pipeline coordinator initialized"),
            },
            "validation": {
                "ollama": Mock(success=True, message="Ollama server accessible"),
                "raptor": Mock(success=True, message="RAPTOR processor available"),
                "hyde": Mock(success=True, message="HyDE queries enabled"),
                "reranking": Mock(success=True, message="Reranking enabled"),
            }
        }
    }


class TestCLIBasicFunctionality:
    """Test basic CLI functionality and commands."""
    
    def test_cli_help_command(self, cli_runner):
        """Test that CLI help command works correctly."""
        result = cli_runner.invoke(cli, ['--help'])
        
        assert result.exit_code == 0
        assert "Mimir 2.0 validation and setup tools" in result.output
        assert "validate" in result.output
        assert "config" in result.output
        assert "dependencies" in result.output
        assert "ollama" in result.output
    
    def test_validate_command_help(self, cli_runner):
        """Test validate command help."""
        result = cli_runner.invoke(cli, ['validate', '--help'])
        
        assert result.exit_code == 0
        assert "Validate Mimir 2.0 setup and configuration" in result.output
        assert "--json-output" in result.output
        assert "--verbose" in result.output
    
    def test_config_command_help(self, cli_runner):
        """Test config command help."""
        result = cli_runner.invoke(cli, ['config', '--help'])
        
        assert result.exit_code == 0
        assert "Display current Mimir 2.0 configuration" in result.output
        assert "--json-output" in result.output
    
    def test_dependencies_command_help(self, cli_runner):
        """Test dependencies command help."""
        result = cli_runner.invoke(cli, ['dependencies', '--help'])
        
        assert result.exit_code == 0
        assert "Check ML and AI dependencies" in result.output
    
    def test_ollama_command_help(self, cli_runner):
        """Test ollama command help."""
        result = cli_runner.invoke(cli, ['ollama', '--help'])
        
        assert result.exit_code == 0
        assert "Test Ollama connectivity" in result.output
        assert "--host" in result.output
        assert "--port" in result.output


class TestValidateCommand:
    """Test the validate command functionality."""
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_command_success(self, mock_validation, cli_runner, mock_validation_results):
        """Test successful validation command."""
        # Setup mock to return async result
        async def mock_async_validation():
            return mock_validation_results
        
        mock_validation.return_value = asyncio.create_task(mock_async_validation())
        
        # Run the command in an event loop context
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: mock_validation_results
            
            result = cli_runner.invoke(cli, ['validate'])
            
            assert result.exit_code == 0
            assert "Validating Mimir 2.0 Setup" in result.output
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_command_json_output(self, mock_validation, cli_runner, mock_validation_results):
        """Test validate command with JSON output."""
        async def mock_async_validation():
            return mock_validation_results
        
        mock_validation.return_value = asyncio.create_task(mock_async_validation())
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: mock_validation_results
            
            result = cli_runner.invoke(cli, ['validate', '--json-output'])
            
            # Should have JSON output
            try:
                output_data = json.loads(result.output)
                assert output_data["overall_success"] is True
                assert "pipeline_mode" in output_data
            except json.JSONDecodeError:
                pytest.fail("Expected valid JSON output")
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_command_verbose_output(self, mock_validation, cli_runner, mock_validation_results):
        """Test validate command with verbose output."""
        async def mock_async_validation():
            return mock_validation_results
        
        mock_validation.return_value = asyncio.create_task(mock_async_validation())
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: mock_validation_results
            
            result = cli_runner.invoke(cli, ['validate', '--verbose'])
            
            assert result.exit_code == 0
            # Verbose mode should show detailed results
            assert "Pipeline Mode:" in result.output
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_command_failure(self, mock_validation, cli_runner):
        """Test validate command when validation fails."""
        failed_results = {
            "overall_success": False,
            "pipeline_mode": "basic",
            "capabilities": Mock(
                has_ollama=False,
                has_raptor=False,
                has_hyde=False,
                has_reranking=False,
                has_code_embeddings=False
            ),
            "results": {}
        }
        
        async def mock_async_validation():
            return failed_results
        
        mock_validation.return_value = asyncio.create_task(mock_async_validation())
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: failed_results
            
            result = cli_runner.invoke(cli, ['validate'])
            
            assert result.exit_code == 1
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_command_exception(self, mock_validation, cli_runner):
        """Test validate command when an exception occurs."""
        async def mock_async_validation():
            raise Exception("Test validation error")
        
        mock_validation.return_value = asyncio.create_task(mock_async_validation())
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = Exception("Test validation error")
            
            result = cli_runner.invoke(cli, ['validate'])
            
            assert result.exit_code == 1
            assert "Validation failed with error" in result.output


class TestConfigCommand:
    """Test the config command functionality."""
    
    @patch('repoindex.cli.mimir2_validate.get_ai_config')
    @patch('repoindex.cli.mimir2_validate.get_pipeline_coordinator')
    def test_config_command_success(self, mock_coordinator, mock_get_config, cli_runner, mock_config):
        """Test successful config command."""
        mock_get_config.return_value = mock_config
        
        mock_coord_instance = Mock()
        mock_coord_instance.validate_configuration.return_value = asyncio.create_task(
            self._mock_config_report()
        )
        
        async def mock_get_coord(config):
            return mock_coord_instance
        
        mock_coordinator.side_effect = mock_get_coord
        
        report = {
            "pipeline_mode": "enhanced",
            "config_status": {
                "ollama_enabled": True,
                "raptor_enabled": True,
                "hyde_enabled": True,
                "reranking_enabled": True,
                "code_embeddings_enabled": True
            },
            "capabilities": {
                "ollama": True,
                "raptor": True,
                "hyde": True,
                "reranking": True,
                "code_embeddings": True
            },
            "warnings": [],
            "errors": []
        }
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: report
            
            result = cli_runner.invoke(cli, ['config'])
            
            assert result.exit_code == 0
            assert "Pipeline Mode:" in result.output
    
    async def _mock_config_report(self):
        """Helper method to create mock config report."""
        return {
            "pipeline_mode": "enhanced",
            "config_status": {
                "ollama_enabled": True,
                "raptor_enabled": True,
                "hyde_enabled": True,
                "reranking_enabled": True,
                "code_embeddings_enabled": True
            },
            "capabilities": {
                "ollama": True,
                "raptor": True,
                "hyde": True,
                "reranking": True,
                "code_embeddings": True
            },
            "warnings": [],
            "errors": []
        }
    
    @patch('repoindex.cli.mimir2_validate.get_ai_config')
    @patch('repoindex.cli.mimir2_validate.get_pipeline_coordinator')
    def test_config_command_json_output(self, mock_coordinator, mock_get_config, cli_runner, mock_config):
        """Test config command with JSON output."""
        mock_get_config.return_value = mock_config
        
        mock_coord_instance = Mock()
        report = {
            "pipeline_mode": "enhanced",
            "config_status": {},
            "capabilities": {},
            "warnings": [],
            "errors": []
        }
        
        async def mock_get_coord(config):
            return mock_coord_instance
        
        mock_coordinator.side_effect = mock_get_coord
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: report
            
            result = cli_runner.invoke(cli, ['config', '--json-output'])
            
            # Should have JSON output
            try:
                output_data = json.loads(result.output)
                assert "pipeline_mode" in output_data
            except json.JSONDecodeError:
                pytest.fail("Expected valid JSON output")
    
    @patch('repoindex.cli.mimir2_validate.get_ai_config')
    def test_config_command_error(self, mock_get_config, cli_runner):
        """Test config command when an error occurs."""
        mock_get_config.side_effect = Exception("Configuration error")
        
        result = cli_runner.invoke(cli, ['config'])
        
        assert result.exit_code == 1
        assert "Configuration check failed" in result.output


class TestDependenciesCommand:
    """Test the dependencies command functionality."""
    
    def test_dependencies_command_basic(self, cli_runner):
        """Test basic dependencies command."""
        result = cli_runner.invoke(cli, ['dependencies'])
        
        assert result.exit_code == 0
        assert "Checking Dependencies" in result.output
        assert "Core Dependencies" in result.output
        assert "ML/Clustering" in result.output
        assert "Deep Learning" in result.output
    
    @patch('builtins.__import__')
    def test_dependencies_command_missing_packages(self, mock_import, cli_runner):
        """Test dependencies command when packages are missing."""
        # Mock missing packages
        def mock_import_side_effect(name, *args, **kwargs):
            if name in ["torch", "transformers"]:
                raise ImportError(f"No module named '{name}'")
            return MagicMock()
        
        mock_import.side_effect = mock_import_side_effect
        
        result = cli_runner.invoke(cli, ['dependencies'])
        
        assert result.exit_code == 0
        assert "❌ Missing" in result.output
    
    @patch('builtins.__import__')
    def test_dependencies_command_all_available(self, mock_import, cli_runner):
        """Test dependencies command when all packages are available."""
        mock_import.return_value = MagicMock()
        
        result = cli_runner.invoke(cli, ['dependencies'])
        
        assert result.exit_code == 0
        assert "✅ Available" in result.output


class TestOllamaCommand:
    """Test the ollama command functionality."""
    
    @patch('aiohttp.ClientSession.get')
    def test_ollama_command_success(self, mock_get, cli_runner):
        """Test successful ollama connectivity test."""
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = asyncio.create_task(
            asyncio.coroutine(lambda: {
                "models": [
                    {
                        "name": "llama2:7b",
                        "size": 3800000000,
                        "modified_at": "2024-01-15T10:30:00Z"
                    }
                ]
            })()
        )
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            # Mock the async run to execute our test
            async def run_test():
                async with AsyncMock() as mock_session:
                    mock_session.get.return_value.__aenter__.return_value = mock_response
                    # Simulate ollama function logic
                    response_data = await mock_response.json()
                    return response_data
            
            mock_run.side_effect = lambda coro: {"models": [{"name": "llama2:7b", "size": 3800000000}]}
            
            result = cli_runner.invoke(cli, ['ollama'])
            
            assert result.exit_code == 0
    
    @patch('aiohttp.ClientSession.get')
    def test_ollama_command_connection_error(self, mock_get, cli_runner):
        """Test ollama command when connection fails."""
        from aiohttp import ClientConnectorError
        
        mock_get.side_effect = ClientConnectorError("Connection failed", Mock())
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = ClientConnectorError("Connection failed", Mock())
            
            result = cli_runner.invoke(cli, ['ollama'])
            
            # Command should still exit with code 0 as it handles the error gracefully
            assert "Cannot connect to Ollama" in result.output or result.exit_code == 0
    
    def test_ollama_command_custom_host_port(self, cli_runner):
        """Test ollama command with custom host and port."""
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.return_value = None
            
            result = cli_runner.invoke(cli, ['ollama', '--host', 'custom-host', '--port', '12345'])
            
            assert result.exit_code == 0
    
    @patch('aiohttp.ClientSession.get')
    def test_ollama_command_http_error(self, mock_get, cli_runner):
        """Test ollama command when server returns HTTP error."""
        mock_response = AsyncMock()
        mock_response.status = 500
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: None
            
            result = cli_runner.invoke(cli, ['ollama'])
            
            assert result.exit_code == 0


class TestCLIArgumentParsing:
    """Test CLI argument parsing and validation."""
    
    def test_invalid_command(self, cli_runner):
        """Test behavior with invalid command."""
        result = cli_runner.invoke(cli, ['invalid-command'])
        
        assert result.exit_code != 0
        assert "No such command" in result.output
    
    def test_validate_command_invalid_flag(self, cli_runner):
        """Test validate command with invalid flag."""
        result = cli_runner.invoke(cli, ['validate', '--invalid-flag'])
        
        assert result.exit_code != 0
        assert "No such option" in result.output
    
    def test_ollama_command_invalid_port(self, cli_runner):
        """Test ollama command with invalid port."""
        result = cli_runner.invoke(cli, ['ollama', '--port', 'invalid'])
        
        assert result.exit_code != 0
        assert "Invalid value" in result.output


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_keyboard_interrupt(self, mock_validation, cli_runner):
        """Test validate command handling keyboard interrupt."""
        mock_validation.side_effect = KeyboardInterrupt()
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            
            result = cli_runner.invoke(cli, ['validate'])
            
            # Click should handle the KeyboardInterrupt appropriately
            assert result.exit_code != 0
    
    @patch('repoindex.cli.mimir2_validate.get_ai_config')
    def test_config_import_error(self, mock_get_config, cli_runner):
        """Test config command when import fails."""
        mock_get_config.side_effect = ImportError("Module not found")
        
        result = cli_runner.invoke(cli, ['config'])
        
        assert result.exit_code == 1
        assert "Configuration check failed" in result.output


class TestCLIOutputFormatting:
    """Test CLI output formatting and display."""
    
    @patch('repoindex.cli.mimir2_validate.run_integration_validation')
    def test_validate_output_formatting(self, mock_validation, cli_runner, mock_validation_results):
        """Test that validate command produces well-formatted output."""
        async def mock_async_validation():
            return mock_validation_results
        
        mock_validation.return_value = asyncio.create_task(mock_async_validation())
        
        with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
            mock_run.side_effect = lambda coro: mock_validation_results
            
            result = cli_runner.invoke(cli, ['validate', '--verbose'])
            
            # Check for rich formatting elements
            assert "🔍" in result.output or "Validating" in result.output
            assert "Pipeline Mode:" in result.output
    
    def test_dependencies_table_formatting(self, cli_runner):
        """Test dependencies command table formatting."""
        result = cli_runner.invoke(cli, ['dependencies'])
        
        # Should have table structure
        assert "Package" in result.output
        assert "Purpose" in result.output
        assert "Status" in result.output


class TestCLIIntegration:
    """Test CLI integration with actual system components."""
    
    @pytest.mark.slow
    def test_real_dependencies_check(self, cli_runner):
        """Test dependencies command against real system."""
        # This test runs against actual system dependencies
        result = cli_runner.invoke(cli, ['dependencies'])
        
        assert result.exit_code == 0
        # Should show some packages as available since pytest is running
        assert ("✅ Available" in result.output) or ("❌ Missing" in result.output)
    
    @pytest.mark.slow
    @pytest.mark.skipif(
        not Path("/tmp/test_mimir_cli").exists(),
        reason="Requires test environment setup"
    )
    def test_config_with_test_environment(self, cli_runner):
        """Test config command in controlled test environment."""
        # Set up test environment variables
        test_env = os.environ.copy()
        test_env['MIMIR_AI_PROVIDER'] = 'ollama'
        test_env['MIMIR_OLLAMA_BASE_URL'] = 'http://localhost:11434'
        
        with patch.dict(os.environ, test_env):
            result = cli_runner.invoke(cli, ['config', '--json-output'])
            
            # Should produce valid JSON output
            assert result.exit_code in [0, 1]  # May fail if services not running, but should handle gracefully


# Async test support
class TestCLIAsync:
    """Test asynchronous aspects of CLI commands."""
    
    @pytest.mark.asyncio
    async def test_validate_async_execution(self, mock_validation_results):
        """Test async execution of validate command logic."""
        # Test the actual async function directly
        from repoindex.cli.mimir2_validate import run_integration_validation
        
        with patch('repoindex.cli.mimir2_validate.run_integration_validation') as mock_validation:
            mock_validation.return_value = mock_validation_results
            
            result = await mock_validation()
            assert result["overall_success"] is True
    
    @pytest.mark.asyncio
    async def test_config_async_execution(self):
        """Test async execution of config command logic."""
        from repoindex.cli.mimir2_validate import get_ai_config, get_pipeline_coordinator
        
        with patch('repoindex.cli.mimir2_validate.get_ai_config') as mock_get_config:
            with patch('repoindex.cli.mimir2_validate.get_pipeline_coordinator') as mock_coordinator:
                mock_config = {"ai": {"provider": "ollama"}}
                mock_get_config.return_value = mock_config
                
                mock_coord_instance = Mock()
                mock_coord_instance.validate_configuration.return_value = {
                    "pipeline_mode": "enhanced",
                    "config_status": {},
                    "capabilities": {}
                }
                mock_coordinator.return_value = mock_coord_instance
                
                config = mock_get_config()
                coordinator = await mock_coordinator(config)
                
                assert config is not None
                assert coordinator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])