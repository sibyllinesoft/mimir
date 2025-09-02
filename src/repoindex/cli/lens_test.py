#!/usr/bin/env python3
"""
Lens Integration Test CLI.

Command-line tool for testing and validating Mimir-Lens integration.
"""

import asyncio
import sys
import json
from pathlib import Path

# Add the parent directory to sys.path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.repoindex.pipeline.lens_integration_helpers import (
    validate_lens_connection,
    test_lens_performance,
    diagnose_lens_issues,
    print_lens_status,
    run_lens_validation_suite,
)
from src.repoindex.config import get_lens_config
from src.repoindex.util.log import get_logger

logger = get_logger(__name__)


async def main():
    """Main CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test and validate Mimir-Lens integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Show current configuration status
  %(prog)s validate                  # Test connectivity and basic functionality
  %(prog)s performance --duration 60 # Run performance test for 60 seconds
  %(prog)s diagnose                  # Diagnose configuration and connectivity issues
  %(prog)s suite                     # Run comprehensive validation suite
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Show current Lens integration status')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate Lens connectivity')
    validate_parser.add_argument('--json', action='store_true', 
                                help='Output results as JSON')
    
    # Performance command
    perf_parser = subparsers.add_parser('performance', help='Run performance test')
    perf_parser.add_argument('--duration', type=int, default=30,
                            help='Test duration in seconds (default: 30)')
    perf_parser.add_argument('--json', action='store_true',
                            help='Output results as JSON')
    
    # Diagnose command
    diagnose_parser = subparsers.add_parser('diagnose', help='Diagnose integration issues')
    diagnose_parser.add_argument('--json', action='store_true',
                                help='Output results as JSON')
    
    # Suite command
    suite_parser = subparsers.add_parser('suite', help='Run comprehensive validation suite')
    suite_parser.add_argument('--json', action='store_true',
                             help='Output results as JSON')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'status':
            print_lens_status()
            return 0
        
        elif args.command == 'validate':
            result = await validate_lens_connection()
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_validation_result(result)
            
            return 0 if result["overall_status"] in ["healthy", "warning"] else 1
        
        elif args.command == 'performance':
            print(f"🚀 Running performance test for {args.duration} seconds...")
            result = await test_lens_performance(args.duration)
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_performance_result(result)
            
            return 0 if result.get("statistics", {}).get("error_rate", 1.0) < 0.1 else 1
        
        elif args.command == 'diagnose':
            result = await diagnose_lens_issues()
            
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                print_diagnosis_result(result)
            
            return 0 if result["overall_health"] in ["healthy", "warning"] else 1
        
        elif args.command == 'suite':
            result = await run_lens_validation_suite()
            
            if args.json:
                print(json.dumps(result, indent=2))
            
            return 0 if result["overall_status"] in ["healthy", "warning"] else 1
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Command failed: {str(e)}")
        print(f"❌ Error: {str(e)}")
        return 1


def print_validation_result(result):
    """Print validation result in human-readable format."""
    print("\n🔍 LENS CONNECTIVITY VALIDATION")
    print("=" * 50)
    
    print(f"Overall Status: {get_status_emoji(result['overall_status'])} {result['overall_status'].upper()}")
    print(f"Timestamp: {result['timestamp']}")
    print(f"Lens URL: {result['lens_url']}")
    print()
    
    if result["health_check"]:
        hc = result["health_check"]
        print("Health Check:")
        print(f"  Status: {hc['status']}")
        print(f"  Response Time: {hc['response_time_ms']:.1f}ms")
        if hc["version"]:
            print(f"  Version: {hc['version']}")
        if hc["uptime_seconds"]:
            print(f"  Uptime: {hc['uptime_seconds']}s")
        if hc["error"]:
            print(f"  Error: {hc['error']}")
        print()
    
    if result["connectivity"]:
        conn = result["connectivity"]
        print("API Connectivity:")
        print(f"  Accessible: {'✅' if conn['api_accessible'] else '❌'}")
        if conn.get("response_time_ms"):
            print(f"  Response Time: {conn['response_time_ms']:.1f}ms")
        if conn.get("error"):
            print(f"  Error: {conn['error']}")
        print()
    
    print("Features:")
    for feature, status in result["features"].items():
        if isinstance(status, dict):
            enabled = status.get("enabled", False)
            emoji = "✅" if enabled else "❌"
            print(f"  {feature.title()}: {emoji} {'Enabled' if enabled else 'Disabled'}")
            if not status.get("config_valid", True):
                print(f"    ⚠️  Configuration issue: {status.get('error', 'Unknown')}")
        else:
            emoji = "✅" if status else "❌"
            print(f"  {feature.title()}: {emoji} {'Enabled' if status else 'Disabled'}")
    print()
    
    if result["errors"]:
        print("Errors:")
        for error in result["errors"]:
            print(f"  ❌ {error}")
        print()
    
    if result["warnings"]:
        print("Warnings:")
        for warning in result["warnings"]:
            print(f"  ⚠️  {warning}")
        print()
    
    print("Performance:")
    perf = result["performance"]
    if perf["health_check_ms"]:
        print(f"  Health Check: {perf['health_check_ms']:.1f}ms")
    if perf["api_response_ms"]:
        print(f"  API Response: {perf['api_response_ms']:.1f}ms")


def print_performance_result(result):
    """Print performance test result in human-readable format."""
    print("\n🚀 LENS PERFORMANCE TEST RESULTS")
    print("=" * 50)
    
    if result.get("error"):
        print(f"❌ {result['error']}")
        return
    
    print(f"Duration: {result['duration_seconds']}s")
    print(f"Timestamp: {result['timestamp']}")
    print()
    
    stats = result["statistics"]
    print("Test Statistics:")
    print(f"  Health Checks: {stats['successful_health_checks']}/{stats['total_health_checks']} successful")
    print(f"  API Calls: {stats['successful_api_calls']}/{stats['total_api_calls']} successful")
    print(f"  Error Rate: {stats['error_rate']:.1%}")
    print()
    
    print("Response Times:")
    print(f"  Average Health Check: {stats['avg_health_check_ms']:.1f}ms")
    print(f"  Average API Call: {stats['avg_api_call_ms']:.1f}ms")
    print(f"  Maximum: {stats['max_response_time_ms']:.1f}ms")
    print(f"  Minimum: {stats['min_response_time_ms']:.1f}ms")
    print()
    
    if result["errors"]:
        print("Errors Encountered:")
        for error in result["errors"][:5]:  # Show first 5 errors
            print(f"  ❌ {error}")
        if len(result["errors"]) > 5:
            print(f"  ... and {len(result['errors']) - 5} more errors")


def print_diagnosis_result(result):
    """Print diagnosis result in human-readable format."""
    print("\n🔧 LENS INTEGRATION DIAGNOSIS")
    print("=" * 50)
    
    print(f"Overall Health: {get_status_emoji(result['overall_health'])} {result['overall_health'].upper()}")
    print(f"Timestamp: {result['timestamp']}")
    print()
    
    config = result["configuration"]
    print("Configuration:")
    print(f"  Enabled: {'✅' if config['enabled'] else '❌'}")
    print(f"  Base URL: {config['base_url']}")
    print(f"  Timeout: {config['timeout']}s")
    print(f"  Max Retries: {config['max_retries']}")
    print(f"  Health Checks: {'✅' if config['health_check_enabled'] else '❌'}")
    print(f"  Fallback: {'✅' if config['fallback_enabled'] else '❌'}")
    print()
    
    if result["issues"]:
        print("Issues Found:")
        for issue in result["issues"]:
            print(f"  ❌ {issue}")
        print()
    
    if result["recommendations"]:
        print("Recommendations:")
        for rec in result["recommendations"]:
            print(f"  💡 {rec}")
        print()
    
    if result.get("connectivity_test"):
        conn_test = result["connectivity_test"]
        print(f"Connectivity Test: {get_status_emoji(conn_test['overall_status'])} {conn_test['overall_status']}")


def get_status_emoji(status):
    """Get emoji for status."""
    emoji_map = {
        "healthy": "✅",
        "warning": "⚠️",
        "error": "❌",
        "unhealthy": "❌",
        "disabled": "🚫",
        "unknown": "❓"
    }
    return emoji_map.get(status, "❓")


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)