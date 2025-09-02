"""
Lens Integration Helper Functions.

Utility functions for validating and testing Mimir-Lens integration.
"""

import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime

from .lens_client import (
    LensIntegrationClient,
    LensHealthStatus,
    LensIndexRequest,
    LensSearchRequest,
    init_lens_client,
)
from ..config import get_lens_config
from ..util.log import get_logger

logger = get_logger(__name__)


async def validate_lens_connection() -> Dict[str, Any]:
    """
    Validate connection to Lens service and test basic functionality.
    
    Returns:
        Dict with validation results and diagnostics
    """
    config = get_lens_config()
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "lens_enabled": config.enabled,
        "lens_url": config.base_url,
        "health_check": None,
        "connectivity": None,
        "features": {
            "indexing": None,
            "search": None,
            "embeddings": None
        },
        "performance": {
            "health_check_ms": None,
            "api_response_ms": None
        },
        "errors": [],
        "warnings": [],
        "overall_status": "unknown"
    }
    
    if not config.enabled:
        result["warnings"].append("Lens integration is disabled in configuration")
        result["overall_status"] = "disabled"
        return result
    
    try:
        async with LensIntegrationClient(config) as client:
            # Test 1: Health Check
            logger.info("Testing Lens service health check...")
            health_check = await client.check_health()
            
            result["health_check"] = {
                "status": health_check.status.value,
                "response_time_ms": health_check.response_time_ms,
                "version": health_check.version,
                "uptime_seconds": health_check.uptime_seconds,
                "error": health_check.error
            }
            result["performance"]["health_check_ms"] = health_check.response_time_ms
            
            if health_check.status != LensHealthStatus.HEALTHY:
                result["errors"].append(f"Lens health check failed: {health_check.error}")
                result["overall_status"] = "unhealthy"
                return result
            
            # Test 2: Basic API Connectivity
            logger.info("Testing basic API connectivity...")
            try:
                # Test getting status for a non-existent repository (should return gracefully)
                status_response = await client.get_repository_status("test-connectivity-check")
                
                result["connectivity"] = {
                    "api_accessible": True,
                    "status_code": status_response.status_code,
                    "response_time_ms": status_response.response_time_ms
                }
                result["performance"]["api_response_ms"] = status_response.response_time_ms
                
            except Exception as e:
                result["errors"].append(f"API connectivity test failed: {str(e)}")
                result["connectivity"] = {"api_accessible": False, "error": str(e)}
            
            # Test 3: Feature Availability
            logger.info("Testing feature availability...")
            
            # Test indexing capability (dry run)
            if config.enable_indexing:
                try:
                    # This is a minimal test - we don't actually index anything
                    test_request = LensIndexRequest(
                        repository_path="/tmp/test",
                        repository_id="connectivity-test",
                        branch="main"
                    )
                    # Don't actually send the request, just validate it can be created
                    result["features"]["indexing"] = {
                        "enabled": True,
                        "config_valid": True
                    }
                except Exception as e:
                    result["features"]["indexing"] = {
                        "enabled": True,
                        "config_valid": False,
                        "error": str(e)
                    }
                    result["warnings"].append(f"Indexing configuration issue: {str(e)}")
            else:
                result["features"]["indexing"] = {"enabled": False}
            
            # Test search capability (dry run)
            if config.enable_search:
                try:
                    test_request = LensSearchRequest(
                        query="test connectivity",
                        max_results=1
                    )
                    # Again, just validate request creation
                    result["features"]["search"] = {
                        "enabled": True,
                        "config_valid": True
                    }
                except Exception as e:
                    result["features"]["search"] = {
                        "enabled": True,
                        "config_valid": False,
                        "error": str(e)
                    }
                    result["warnings"].append(f"Search configuration issue: {str(e)}")
            else:
                result["features"]["search"] = {"enabled": False}
            
            # Test embeddings capability
            if config.enable_embeddings:
                result["features"]["embeddings"] = {"enabled": True}
            else:
                result["features"]["embeddings"] = {"enabled": False}
            
            # Determine overall status
            if result["errors"]:
                result["overall_status"] = "error"
            elif result["warnings"]:
                result["overall_status"] = "warning"
            else:
                result["overall_status"] = "healthy"
            
            logger.info(f"Lens validation completed with status: {result['overall_status']}")
            
    except Exception as e:
        result["errors"].append(f"Lens client initialization failed: {str(e)}")
        result["overall_status"] = "error"
        logger.error(f"Lens validation failed: {str(e)}")
    
    return result


async def test_lens_performance(duration_seconds: int = 30) -> Dict[str, Any]:
    """
    Test Lens service performance over a specified duration.
    
    Args:
        duration_seconds: How long to run the performance test
        
    Returns:
        Dict with performance metrics and statistics
    """
    config = get_lens_config()
    
    if not config.enabled:
        return {
            "error": "Lens integration is disabled",
            "duration_seconds": 0,
            "tests_run": 0
        }
    
    result = {
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": duration_seconds,
        "health_checks": [],
        "api_calls": [],
        "statistics": {
            "total_health_checks": 0,
            "successful_health_checks": 0,
            "total_api_calls": 0,
            "successful_api_calls": 0,
            "avg_health_check_ms": 0.0,
            "avg_api_call_ms": 0.0,
            "max_response_time_ms": 0.0,
            "min_response_time_ms": float('inf'),
            "error_rate": 0.0
        },
        "errors": []
    }
    
    start_time = asyncio.get_event_loop().time()
    end_time = start_time + duration_seconds
    
    try:
        async with LensIntegrationClient(config) as client:
            while asyncio.get_event_loop().time() < end_time:
                # Health check test
                try:
                    health_check = await client.check_health()
                    result["health_checks"].append({
                        "timestamp": datetime.now().isoformat(),
                        "status": health_check.status.value,
                        "response_time_ms": health_check.response_time_ms,
                        "success": health_check.status == LensHealthStatus.HEALTHY
                    })
                    result["statistics"]["total_health_checks"] += 1
                    if health_check.status == LensHealthStatus.HEALTHY:
                        result["statistics"]["successful_health_checks"] += 1
                    
                    # Update response time stats
                    rt = health_check.response_time_ms
                    result["statistics"]["max_response_time_ms"] = max(
                        result["statistics"]["max_response_time_ms"], rt
                    )
                    result["statistics"]["min_response_time_ms"] = min(
                        result["statistics"]["min_response_time_ms"], rt
                    )
                    
                except Exception as e:
                    result["errors"].append(f"Health check failed: {str(e)}")
                
                # API call test
                try:
                    api_response = await client.get_repository_status("perf-test")
                    result["api_calls"].append({
                        "timestamp": datetime.now().isoformat(),
                        "response_time_ms": api_response.response_time_ms,
                        "success": api_response.success,
                        "status_code": api_response.status_code
                    })
                    result["statistics"]["total_api_calls"] += 1
                    if api_response.success:
                        result["statistics"]["successful_api_calls"] += 1
                    
                    # Update response time stats
                    rt = api_response.response_time_ms
                    result["statistics"]["max_response_time_ms"] = max(
                        result["statistics"]["max_response_time_ms"], rt
                    )
                    result["statistics"]["min_response_time_ms"] = min(
                        result["statistics"]["min_response_time_ms"], rt
                    )
                    
                except Exception as e:
                    result["errors"].append(f"API call failed: {str(e)}")
                
                # Small delay between tests
                await asyncio.sleep(0.5)
        
        # Calculate final statistics
        total_calls = result["statistics"]["total_health_checks"] + result["statistics"]["total_api_calls"]
        successful_calls = result["statistics"]["successful_health_checks"] + result["statistics"]["successful_api_calls"]
        
        if total_calls > 0:
            result["statistics"]["error_rate"] = 1.0 - (successful_calls / total_calls)
        
        # Calculate average response times
        if result["health_checks"]:
            health_times = [hc["response_time_ms"] for hc in result["health_checks"]]
            result["statistics"]["avg_health_check_ms"] = sum(health_times) / len(health_times)
        
        if result["api_calls"]:
            api_times = [ac["response_time_ms"] for ac in result["api_calls"]]
            result["statistics"]["avg_api_call_ms"] = sum(api_times) / len(api_times)
        
        # Handle min response time edge case
        if result["statistics"]["min_response_time_ms"] == float('inf'):
            result["statistics"]["min_response_time_ms"] = 0.0
        
    except Exception as e:
        result["errors"].append(f"Performance test failed: {str(e)}")
    
    return result


async def diagnose_lens_issues() -> Dict[str, Any]:
    """
    Diagnose common Lens integration issues and provide recommendations.
    
    Returns:
        Dict with diagnostic information and recommendations
    """
    config = get_lens_config()
    
    diagnosis = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "enabled": config.enabled,
            "base_url": config.base_url,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "health_check_enabled": config.health_check_enabled,
            "fallback_enabled": config.fallback_enabled
        },
        "issues": [],
        "recommendations": [],
        "connectivity_test": None,
        "overall_health": "unknown"
    }
    
    # Configuration checks
    if not config.enabled:
        diagnosis["issues"].append("Lens integration is disabled")
        diagnosis["recommendations"].append("Set LENS_ENABLED=true to enable Lens integration")
    
    if not config.health_check_enabled:
        diagnosis["issues"].append("Health checks are disabled")
        diagnosis["recommendations"].append("Enable health checks for better monitoring")
    
    if config.timeout < 10:
        diagnosis["issues"].append("Timeout setting is very low")
        diagnosis["recommendations"].append("Consider increasing LENS_TIMEOUT to at least 10 seconds")
    
    if not config.fallback_enabled:
        diagnosis["issues"].append("Fallback mechanisms are disabled")
        diagnosis["recommendations"].append("Enable fallback to improve resilience")
    
    # Connectivity test
    if config.enabled:
        try:
            validation_result = await validate_lens_connection()
            diagnosis["connectivity_test"] = validation_result
            
            if validation_result["overall_status"] == "healthy":
                diagnosis["overall_health"] = "healthy"
            elif validation_result["overall_status"] == "warning":
                diagnosis["overall_health"] = "warning"
                diagnosis["issues"].extend(validation_result["warnings"])
            else:
                diagnosis["overall_health"] = "unhealthy"
                diagnosis["issues"].extend(validation_result["errors"])
                
                # Add specific recommendations based on errors
                for error in validation_result["errors"]:
                    if "connection" in error.lower():
                        diagnosis["recommendations"].append("Check that Lens service is running and accessible")
                        diagnosis["recommendations"].append(f"Verify LENS_BASE_URL is correct: {config.base_url}")
                    elif "timeout" in error.lower():
                        diagnosis["recommendations"].append("Increase timeout settings or check network latency")
                    elif "auth" in error.lower():
                        diagnosis["recommendations"].append("Verify LENS_API_KEY is correct if authentication is required")
        
        except Exception as e:
            diagnosis["issues"].append(f"Connectivity test failed: {str(e)}")
            diagnosis["overall_health"] = "error"
            diagnosis["recommendations"].append("Check logs for detailed error information")
    
    return diagnosis


def print_lens_status():
    """Print a summary of current Lens integration status."""
    config = get_lens_config()
    
    print("\n" + "="*50)
    print("LENS INTEGRATION STATUS")
    print("="*50)
    print(f"Enabled: {config.enabled}")
    print(f"Base URL: {config.base_url}")
    print(f"Timeout: {config.timeout}s")
    print(f"Max Retries: {config.max_retries}")
    print(f"Health Checks: {config.health_check_enabled}")
    print(f"Fallback: {config.fallback_enabled}")
    print()
    print("Features:")
    print(f"  Indexing: {config.enable_indexing}")
    print(f"  Search: {config.enable_search}")
    print(f"  Embeddings: {config.enable_embeddings}")
    print()
    
    if not config.enabled:
        print("⚠️  Lens integration is DISABLED")
        print("   Set LENS_ENABLED=true to enable")
    else:
        print("✅ Lens integration is ENABLED")
        print("   Run validation test to check connectivity")
    
    print("="*50)


async def run_lens_validation_suite() -> Dict[str, Any]:
    """
    Run comprehensive Lens integration validation suite.
    
    Returns:
        Dict with complete validation results
    """
    print("🔍 Running Lens Integration Validation Suite...")
    print("-" * 50)
    
    # Step 1: Configuration validation
    print("1. Configuration Validation...")
    config = get_lens_config()
    config_valid = True
    config_issues = []
    
    if not config.enabled:
        config_issues.append("Integration disabled")
        config_valid = False
    
    if not config.base_url:
        config_issues.append("No base URL configured")
        config_valid = False
    
    print(f"   Configuration: {'✅ Valid' if config_valid else '❌ Issues found'}")
    if config_issues:
        for issue in config_issues:
            print(f"   - {issue}")
    
    # Step 2: Connectivity test
    print("\n2. Connectivity Test...")
    connectivity_result = await validate_lens_connection()
    
    status_emoji = {
        "healthy": "✅",
        "warning": "⚠️",
        "error": "❌",
        "disabled": "🚫",
        "unknown": "❓"
    }
    
    emoji = status_emoji.get(connectivity_result["overall_status"], "❓")
    print(f"   Connectivity: {emoji} {connectivity_result['overall_status'].upper()}")
    
    if connectivity_result["health_check"]:
        hc = connectivity_result["health_check"]
        print(f"   Health Check: {hc['status']} ({hc['response_time_ms']:.1f}ms)")
        if hc["version"]:
            print(f"   Lens Version: {hc['version']}")
    
    # Step 3: Performance test (brief)
    if connectivity_result["overall_status"] == "healthy":
        print("\n3. Performance Test (10s)...")
        perf_result = await test_lens_performance(10)
        
        stats = perf_result["statistics"]
        print(f"   Health Checks: {stats['successful_health_checks']}/{stats['total_health_checks']}")
        print(f"   API Calls: {stats['successful_api_calls']}/{stats['total_api_calls']}")
        print(f"   Avg Response: {stats['avg_health_check_ms']:.1f}ms")
        print(f"   Error Rate: {stats['error_rate']:.1%}")
    else:
        print("\n3. Performance Test: Skipped (connectivity issues)")
        perf_result = {"skipped": True, "reason": "connectivity_issues"}
    
    # Step 4: Diagnosis
    print("\n4. Issue Diagnosis...")
    diagnosis = await diagnose_lens_issues()
    
    if diagnosis["issues"]:
        print("   Issues Found:")
        for issue in diagnosis["issues"][:5]:  # Show first 5 issues
            print(f"   - {issue}")
        if len(diagnosis["issues"]) > 5:
            print(f"   ... and {len(diagnosis['issues']) - 5} more")
    else:
        print("   No issues detected ✅")
    
    if diagnosis["recommendations"]:
        print("\n   Recommendations:")
        for rec in diagnosis["recommendations"][:3]:  # Show first 3 recommendations
            print(f"   - {rec}")
        if len(diagnosis["recommendations"]) > 3:
            print(f"   ... and {len(diagnosis['recommendations']) - 3} more")
    
    print("\n" + "-" * 50)
    overall_status = diagnosis["overall_health"]
    emoji = status_emoji.get(overall_status, "❓")
    print(f"🎯 Overall Status: {emoji} {overall_status.upper()}")
    
    return {
        "configuration": {
            "valid": config_valid,
            "issues": config_issues
        },
        "connectivity": connectivity_result,
        "performance": perf_result,
        "diagnosis": diagnosis,
        "overall_status": overall_status
    }