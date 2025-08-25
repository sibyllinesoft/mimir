"""
Comprehensive test runner for CLI and integration tests.

Provides orchestrated test execution, parallel testing capabilities,
and detailed reporting for integration and CLI test suites.
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

import pytest


@dataclass
class TestResult:
    """Individual test result."""
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    output: str = ""
    error_message: str = ""
    traceback: str = ""
    
    
@dataclass
class TestSuiteResult:
    """Test suite execution result."""
    suite_name: str
    total_tests: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    test_results: List[TestResult]
    start_time: datetime
    end_time: datetime
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed / self.total_tests) * 100


class TestExecutor:
    """Execute tests with different strategies."""
    
    def __init__(self, base_path: Path, max_workers: int = 4):
        self.base_path = base_path
        self.max_workers = max_workers
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for test execution."""
        logger = logging.getLogger("TestExecutor")
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    async def run_test_file(self, test_file: Path, pytest_args: List[str] = None) -> TestSuiteResult:
        """Run a single test file."""
        if pytest_args is None:
            pytest_args = ["-v", "--tb=short"]
        
        start_time = datetime.now()
        self.logger.info(f"Starting test file: {test_file.name}")
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest", 
            str(test_file),
            "--json-report",
            "--json-report-file=/tmp/test_report.json"
        ] + pytest_args
        
        try:
            # Run tests
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.base_path
            )
            
            stdout, stderr = await process.communicate()
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Parse results
            test_results = self._parse_pytest_output(stdout.decode(), stderr.decode())
            
            # Calculate statistics
            total_tests = len(test_results)
            passed = sum(1 for r in test_results if r.status == "passed")
            failed = sum(1 for r in test_results if r.status == "failed")
            skipped = sum(1 for r in test_results if r.status == "skipped")
            errors = sum(1 for r in test_results if r.status == "error")
            
            suite_result = TestSuiteResult(
                suite_name=test_file.name,
                total_tests=total_tests,
                passed=passed,
                failed=failed,
                skipped=skipped,
                errors=errors,
                duration=duration,
                test_results=test_results,
                start_time=start_time,
                end_time=end_time
            )
            
            self.logger.info(f"Completed {test_file.name}: {passed}/{total_tests} passed")
            return suite_result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            self.logger.error(f"Error running {test_file.name}: {e}")
            
            return TestSuiteResult(
                suite_name=test_file.name,
                total_tests=0,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=duration,
                test_results=[TestResult(
                    test_name=f"{test_file.name}::execution_error",
                    status="error",
                    duration=duration,
                    error_message=str(e),
                    traceback=""
                )],
                start_time=start_time,
                end_time=end_time
            )
    
    def _parse_pytest_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest output to extract test results."""
        # Try to parse JSON report first
        json_report_path = Path("/tmp/test_report.json")
        if json_report_path.exists():
            try:
                with open(json_report_path) as f:
                    report_data = json.load(f)
                return self._parse_json_report(report_data)
            except Exception as e:
                self.logger.warning(f"Failed to parse JSON report: {e}")
        
        # Fallback to parsing text output
        return self._parse_text_output(stdout, stderr)
    
    def _parse_json_report(self, report_data: Dict) -> List[TestResult]:
        """Parse pytest JSON report."""
        results = []
        
        for test in report_data.get("tests", []):
            nodeid = test.get("nodeid", "unknown")
            outcome = test.get("outcome", "unknown")
            duration = test.get("duration", 0.0)
            
            # Map pytest outcomes to our statuses
            status_map = {
                "passed": "passed",
                "failed": "failed",
                "skipped": "skipped",
                "error": "error",
                "xfail": "skipped",
                "xpass": "passed"
            }
            
            status = status_map.get(outcome, "error")
            
            # Extract error information
            error_message = ""
            traceback = ""
            if outcome == "failed":
                call = test.get("call", {})
                error_message = call.get("longrepr", "")
                traceback = call.get("longrepr", "")
            
            results.append(TestResult(
                test_name=nodeid,
                status=status,
                duration=duration,
                error_message=error_message,
                traceback=traceback
            ))
        
        return results
    
    def _parse_text_output(self, stdout: str, stderr: str) -> List[TestResult]:
        """Parse pytest text output as fallback."""
        results = []
        lines = stdout.split('\n')
        
        current_test = None
        for line in lines:
            line = line.strip()
            
            # Look for test results
            if "::" in line and any(status in line for status in ["PASSED", "FAILED", "SKIPPED", "ERROR"]):
                parts = line.split()
                if len(parts) >= 2:
                    test_name = parts[0]
                    status_part = parts[1]
                    
                    if "PASSED" in status_part:
                        status = "passed"
                    elif "FAILED" in status_part:
                        status = "failed"
                    elif "SKIPPED" in status_part:
                        status = "skipped"
                    elif "ERROR" in status_part:
                        status = "error"
                    else:
                        continue
                    
                    results.append(TestResult(
                        test_name=test_name,
                        status=status,
                        duration=0.0  # Duration not available in text output
                    ))
        
        return results
    
    async def run_test_suite_parallel(self, test_files: List[Path], pytest_args: List[str] = None) -> List[TestSuiteResult]:
        """Run multiple test files in parallel."""
        self.logger.info(f"Running {len(test_files)} test files in parallel (max_workers={self.max_workers})")
        
        # Create semaphore to limit concurrent tests
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def run_single_test(test_file: Path) -> TestSuiteResult:
            async with semaphore:
                return await self.run_test_file(test_file, pytest_args)
        
        # Run all tests concurrently
        tasks = [run_single_test(test_file) for test_file in test_files]
        results = await asyncio.gather(*tasks)
        
        return results
    
    async def run_test_suite_sequential(self, test_files: List[Path], pytest_args: List[str] = None) -> List[TestSuiteResult]:
        """Run test files sequentially."""
        self.logger.info(f"Running {len(test_files)} test files sequentially")
        
        results = []
        for test_file in test_files:
            result = await self.run_test_file(test_file, pytest_args)
            results.append(result)
            
            # Log intermediate results
            self.logger.info(f"Progress: {len(results)}/{len(test_files)} completed")
        
        return results


class CLITestRunner:
    """Specialized runner for CLI tests."""
    
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self.logger = logging.getLogger("CLITestRunner")
        
    async def run_cli_validation_tests(self) -> TestSuiteResult:
        """Run CLI validation tests."""
        start_time = datetime.now()
        
        # Test basic CLI functionality
        cli_tests = [
            self._test_cli_help,
            self._test_validate_command,
            self._test_config_command,
            self._test_dependencies_command,
            self._test_ollama_command,
            self._test_invalid_commands,
            self._test_cli_error_handling
        ]
        
        test_results = []
        
        for test_func in cli_tests:
            try:
                result = await test_func()
                test_results.append(result)
            except Exception as e:
                test_results.append(TestResult(
                    test_name=test_func.__name__,
                    status="error",
                    duration=0.0,
                    error_message=str(e)
                ))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate statistics
        passed = sum(1 for r in test_results if r.status == "passed")
        failed = sum(1 for r in test_results if r.status == "failed")
        errors = sum(1 for r in test_results if r.status == "error")
        
        return TestSuiteResult(
            suite_name="CLI Validation Tests",
            total_tests=len(test_results),
            passed=passed,
            failed=failed,
            skipped=0,
            errors=errors,
            duration=duration,
            test_results=test_results,
            start_time=start_time,
            end_time=end_time
        )
    
    async def _test_cli_help(self) -> TestResult:
        """Test CLI help command."""
        start = time.time()
        
        try:
            # Import CLI module
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(cli, ['--help'])
            
            duration = time.time() - start
            
            if result.exit_code == 0 and "Mimir 2.0 validation" in result.output:
                return TestResult("test_cli_help", "passed", duration)
            else:
                return TestResult(
                    "test_cli_help", "failed", duration,
                    error_message=f"Exit code: {result.exit_code}, Output: {result.output}"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_cli_help", "error", duration, error_message=str(e))
    
    async def _test_validate_command(self) -> TestResult:
        """Test validate command."""
        start = time.time()
        
        try:
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            from unittest.mock import patch
            
            runner = CliRunner()
            
            # Mock the validation function
            with patch('repoindex.cli.mimir2_validate.run_integration_validation') as mock_validate:
                with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
                    mock_run.return_value = {"overall_success": True, "pipeline_mode": "test"}
                    
                    result = runner.invoke(cli, ['validate', '--json-output'])
            
            duration = time.time() - start
            
            # Should not crash and should have some output
            if result.exit_code in [0, 1]:  # 0 for success, 1 for validation failure
                return TestResult("test_validate_command", "passed", duration)
            else:
                return TestResult(
                    "test_validate_command", "failed", duration,
                    error_message=f"Exit code: {result.exit_code}, Output: {result.output}"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_validate_command", "error", duration, error_message=str(e))
    
    async def _test_config_command(self) -> TestResult:
        """Test config command."""
        start = time.time()
        
        try:
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            from unittest.mock import patch
            
            runner = CliRunner()
            
            # Mock the configuration functions
            with patch('repoindex.cli.mimir2_validate.get_ai_config') as mock_config:
                with patch('repoindex.cli.mimir2_validate.get_pipeline_coordinator') as mock_coordinator:
                    with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
                        mock_config.return_value = {"ai": {"provider": "test"}}
                        mock_run.return_value = {"pipeline_mode": "test"}
                        
                        result = runner.invoke(cli, ['config', '--json-output'])
            
            duration = time.time() - start
            
            if result.exit_code in [0, 1]:
                return TestResult("test_config_command", "passed", duration)
            else:
                return TestResult(
                    "test_config_command", "failed", duration,
                    error_message=f"Exit code: {result.exit_code}, Output: {result.output}"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_config_command", "error", duration, error_message=str(e))
    
    async def _test_dependencies_command(self) -> TestResult:
        """Test dependencies command."""
        start = time.time()
        
        try:
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(cli, ['dependencies'])
            
            duration = time.time() - start
            
            if result.exit_code == 0:
                return TestResult("test_dependencies_command", "passed", duration)
            else:
                return TestResult(
                    "test_dependencies_command", "failed", duration,
                    error_message=f"Exit code: {result.exit_code}, Output: {result.output}"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_dependencies_command", "error", duration, error_message=str(e))
    
    async def _test_ollama_command(self) -> TestResult:
        """Test ollama command."""
        start = time.time()
        
        try:
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            from unittest.mock import patch
            
            runner = CliRunner()
            
            # Mock aiohttp to avoid actual network calls
            with patch('repoindex.cli.mimir2_validate.asyncio.run') as mock_run:
                mock_run.return_value = None
                result = runner.invoke(cli, ['ollama', '--host', 'localhost', '--port', '11434'])
            
            duration = time.time() - start
            
            # Command should execute without crashing
            if result.exit_code == 0:
                return TestResult("test_ollama_command", "passed", duration)
            else:
                return TestResult(
                    "test_ollama_command", "failed", duration,
                    error_message=f"Exit code: {result.exit_code}, Output: {result.output}"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_ollama_command", "error", duration, error_message=str(e))
    
    async def _test_invalid_commands(self) -> TestResult:
        """Test handling of invalid commands."""
        start = time.time()
        
        try:
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            
            runner = CliRunner()
            result = runner.invoke(cli, ['invalid-command'])
            
            duration = time.time() - start
            
            # Should fail with non-zero exit code
            if result.exit_code != 0:
                return TestResult("test_invalid_commands", "passed", duration)
            else:
                return TestResult(
                    "test_invalid_commands", "failed", duration,
                    error_message="Invalid command should have failed"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_invalid_commands", "error", duration, error_message=str(e))
    
    async def _test_cli_error_handling(self) -> TestResult:
        """Test CLI error handling."""
        start = time.time()
        
        try:
            sys.path.insert(0, str(self.base_path / "src"))
            from repoindex.cli.mimir2_validate import cli
            from click.testing import CliRunner
            from unittest.mock import patch
            
            runner = CliRunner()
            
            # Test with invalid flag
            result = runner.invoke(cli, ['validate', '--invalid-flag'])
            
            duration = time.time() - start
            
            # Should fail with non-zero exit code
            if result.exit_code != 0:
                return TestResult("test_cli_error_handling", "passed", duration)
            else:
                return TestResult(
                    "test_cli_error_handling", "failed", duration,
                    error_message="Invalid flag should have caused error"
                )
                
        except Exception as e:
            duration = time.time() - start
            return TestResult("test_cli_error_handling", "error", duration, error_message=str(e))


class TestReporter:
    """Generate comprehensive test reports."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_console_report(self, suite_results: List[TestSuiteResult]) -> str:
        """Generate console-friendly report."""
        lines = []
        lines.append("="*80)
        lines.append("MIMIR INTEGRATION AND CLI TEST REPORT")
        lines.append("="*80)
        lines.append("")
        
        total_tests = sum(r.total_tests for r in suite_results)
        total_passed = sum(r.passed for r in suite_results)
        total_failed = sum(r.failed for r in suite_results)
        total_errors = sum(r.errors for r in suite_results)
        total_skipped = sum(r.skipped for r in suite_results)
        total_duration = sum(r.duration for r in suite_results)
        
        lines.append(f"SUMMARY:")
        lines.append(f"  Total Tests: {total_tests}")
        lines.append(f"  Passed: {total_passed} ({(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%)")
        lines.append(f"  Failed: {total_failed} ({(total_failed/total_tests*100) if total_tests > 0 else 0:.1f}%)")
        lines.append(f"  Errors: {total_errors} ({(total_errors/total_tests*100) if total_tests > 0 else 0:.1f}%)")
        lines.append(f"  Skipped: {total_skipped} ({(total_skipped/total_tests*100) if total_tests > 0 else 0:.1f}%)")
        lines.append(f"  Total Duration: {total_duration:.2f}s")
        lines.append("")
        
        # Suite details
        lines.append("SUITE DETAILS:")
        lines.append("")
        
        for suite in suite_results:
            lines.append(f"📁 {suite.suite_name}")
            lines.append(f"   Tests: {suite.passed}/{suite.total_tests} passed ({suite.success_rate:.1f}%)")
            lines.append(f"   Duration: {suite.duration:.2f}s")
            
            if suite.failed > 0 or suite.errors > 0:
                lines.append(f"   ❌ Issues: {suite.failed} failed, {suite.errors} errors")
                
                # Show failed tests
                for test in suite.test_results:
                    if test.status in ["failed", "error"]:
                        lines.append(f"      • {test.test_name}: {test.error_message[:100]}...")
            
            lines.append("")
        
        # Performance analysis
        if suite_results:
            lines.append("PERFORMANCE ANALYSIS:")
            lines.append("")
            
            # Slowest test suites
            sorted_suites = sorted(suite_results, key=lambda x: x.duration, reverse=True)
            lines.append("  Slowest Test Suites:")
            for suite in sorted_suites[:5]:
                lines.append(f"    {suite.suite_name}: {suite.duration:.2f}s")
            lines.append("")
            
            # Most failing suites
            failing_suites = [s for s in suite_results if s.failed > 0 or s.errors > 0]
            if failing_suites:
                lines.append("  Suites with Issues:")
                for suite in sorted(failing_suites, key=lambda x: x.failed + x.errors, reverse=True):
                    lines.append(f"    {suite.suite_name}: {suite.failed + suite.errors} issues")
                lines.append("")
        
        return "\n".join(lines)
    
    def generate_json_report(self, suite_results: List[TestSuiteResult]) -> Dict[str, Any]:
        """Generate JSON report."""
        return {
            "summary": {
                "total_tests": sum(r.total_tests for r in suite_results),
                "passed": sum(r.passed for r in suite_results),
                "failed": sum(r.failed for r in suite_results),
                "errors": sum(r.errors for r in suite_results),
                "skipped": sum(r.skipped for r in suite_results),
                "total_duration": sum(r.duration for r in suite_results),
                "success_rate": (sum(r.passed for r in suite_results) / sum(r.total_tests for r in suite_results)) * 100 if sum(r.total_tests for r in suite_results) > 0 else 0
            },
            "suites": [asdict(suite) for suite in suite_results],
            "generated_at": datetime.now().isoformat(),
            "environment": {
                "python_version": sys.version,
                "platform": os.name,
                "working_directory": str(Path.cwd())
            }
        }
    
    def generate_html_report(self, suite_results: List[TestSuiteResult]) -> str:
        """Generate HTML report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Mimir Integration Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .summary { background: #f5f5f5; padding: 20px; border-radius: 5px; margin-bottom: 20px; }
                .suite { border: 1px solid #ddd; margin-bottom: 10px; border-radius: 5px; }
                .suite-header { background: #e9e9e9; padding: 10px; font-weight: bold; }
                .suite-content { padding: 10px; }
                .passed { color: green; }
                .failed { color: red; }
                .error { color: orange; }
                .skipped { color: blue; }
                table { width: 100%; border-collapse: collapse; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Mimir Integration and CLI Test Report</h1>
        """
        
        # Summary
        total_tests = sum(r.total_tests for r in suite_results)
        total_passed = sum(r.passed for r in suite_results)
        total_failed = sum(r.failed for r in suite_results)
        total_errors = sum(r.errors for r in suite_results)
        total_skipped = sum(r.skipped for r in suite_results)
        
        html += f"""
            <div class="summary">
                <h2>Summary</h2>
                <p><strong>Total Tests:</strong> {total_tests}</p>
                <p><strong>Passed:</strong> <span class="passed">{total_passed}</span></p>
                <p><strong>Failed:</strong> <span class="failed">{total_failed}</span></p>
                <p><strong>Errors:</strong> <span class="error">{total_errors}</span></p>
                <p><strong>Skipped:</strong> <span class="skipped">{total_skipped}</span></p>
                <p><strong>Success Rate:</strong> {(total_passed/total_tests*100) if total_tests > 0 else 0:.1f}%</p>
            </div>
        """
        
        # Test suites
        for suite in suite_results:
            status_class = "passed" if suite.failed == 0 and suite.errors == 0 else "failed"
            
            html += f"""
                <div class="suite">
                    <div class="suite-header {status_class}">
                        {suite.suite_name} - {suite.passed}/{suite.total_tests} passed ({suite.success_rate:.1f}%)
                    </div>
                    <div class="suite-content">
                        <p><strong>Duration:</strong> {suite.duration:.2f}s</p>
            """
            
            if suite.test_results:
                html += """
                        <table>
                            <tr>
                                <th>Test</th>
                                <th>Status</th>
                                <th>Duration</th>
                                <th>Error</th>
                            </tr>
                """
                
                for test in suite.test_results:
                    html += f"""
                            <tr>
                                <td>{test.test_name}</td>
                                <td class="{test.status}">{test.status.upper()}</td>
                                <td>{test.duration:.2f}s</td>
                                <td>{test.error_message[:100]}{'...' if len(test.error_message) > 100 else ''}</td>
                            </tr>
                    """
                
                html += "</table>"
            
            html += "</div></div>"
        
        html += """
        </body>
        </html>
        """
        
        return html
    
    def save_reports(self, suite_results: List[TestSuiteResult]):
        """Save all report formats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Console report
        console_report = self.generate_console_report(suite_results)
        console_path = self.output_dir / f"test_report_{timestamp}.txt"
        console_path.write_text(console_report)
        
        # JSON report
        json_report = self.generate_json_report(suite_results)
        json_path = self.output_dir / f"test_report_{timestamp}.json"
        json_path.write_text(json.dumps(json_report, indent=2))
        
        # HTML report
        html_report = self.generate_html_report(suite_results)
        html_path = self.output_dir / f"test_report_{timestamp}.html"
        html_path.write_text(html_report)
        
        print(f"\nReports saved:")
        print(f"  Console: {console_path}")
        print(f"  JSON: {json_path}")
        print(f"  HTML: {html_path}")
        
        return console_path, json_path, html_path


async def main():
    """Main test runner entry point."""
    base_path = Path(__file__).parent.parent.parent
    
    # Set up test discovery
    integration_test_files = list((base_path / "tests" / "integration").glob("test_*.py"))
    
    # Filter out this runner file
    integration_test_files = [f for f in integration_test_files if f.name != "test_runner_comprehensive.py"]
    
    print(f"Discovered {len(integration_test_files)} integration test files")
    for test_file in integration_test_files:
        print(f"  - {test_file.name}")
    
    # Create test executor and reporter
    executor = TestExecutor(base_path, max_workers=2)
    cli_runner = CLITestRunner(base_path)
    reporter = TestReporter(base_path / "test_reports")
    
    all_results = []
    
    # Run CLI tests first
    print("\n" + "="*80)
    print("RUNNING CLI TESTS")
    print("="*80)
    
    cli_result = await cli_runner.run_cli_validation_tests()
    all_results.append(cli_result)
    
    # Run integration tests
    print("\n" + "="*80)
    print("RUNNING INTEGRATION TESTS")
    print("="*80)
    
    if integration_test_files:
        # Run tests in parallel for speed
        integration_results = await executor.run_test_suite_parallel(
            integration_test_files,
            pytest_args=["-v", "--tb=short", "--asyncio-mode=auto"]
        )
        all_results.extend(integration_results)
    
    # Generate and display reports
    print("\n" + "="*80)
    print("GENERATING REPORTS")
    print("="*80)
    
    reporter.save_reports(all_results)
    
    # Display console report
    console_report = reporter.generate_console_report(all_results)
    print(console_report)
    
    # Exit with appropriate code
    total_failed = sum(r.failed + r.errors for r in all_results)
    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())