#!/usr/bin/env python3
"""
Simple test script for Lens integration foundation.

This script tests the basic functionality of the Lens integration
without requiring complex imports or external services.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_lens_config():
    """Test Lens configuration."""
    print("🔧 Testing Lens Configuration...")
    
    try:
        from repoindex.config import LensConfig, get_lens_config, MimirConfig
        
        # Test default configuration
        config = LensConfig()
        print(f"✅ Default config created: enabled={config.enabled}, base_url={config.base_url}")
        
        # Test configuration validation
        config = LensConfig(
            enabled=True,
            base_url="http://localhost:3001",
            timeout=30,
            max_retries=3,
            health_check_enabled=True
        )
        print(f"✅ Custom config created: timeout={config.timeout}, retries={config.max_retries}")
        
        # Test integrated configuration
        main_config = MimirConfig()
        lens_config = main_config.lens
        print(f"✅ Integrated config: enabled={lens_config.enabled}")
        
        # Test validation
        errors = main_config.validate()
        print(f"✅ Configuration validation: {len(errors)} warnings/errors")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False


async def test_lens_client_creation():
    """Test Lens client creation."""
    print("\n🚀 Testing Lens Client Creation...")
    
    try:
        from repoindex.config import LensConfig
        from repoindex.pipeline.lens_client import LensIntegrationClient
        
        # Create test configuration
        config = LensConfig(
            enabled=True,
            base_url="http://localhost:3001",
            timeout=10,
            fallback_enabled=True
        )
        
        # Create client
        client = LensIntegrationClient(config)
        print(f"✅ Client created with base_url: {client.config.base_url}")
        
        # Test session management (without actual connection)
        print(f"✅ Initial session state: {client.session is None}")
        
        # Test health status methods
        print(f"✅ Health status: {client.get_health_status().value}")
        print(f"✅ Is healthy: {client.is_healthy()}")
        
        # Test request models
        from repoindex.pipeline.lens_client import LensIndexRequest, LensSearchRequest
        
        index_req = LensIndexRequest(
            repository_path="/test/path",
            repository_id="test-repo"
        )
        print(f"✅ Index request created: repo_id={index_req.repository_id}")
        
        search_req = LensSearchRequest(
            query="test query",
            max_results=10
        )
        print(f"✅ Search request created: query='{search_req.query}'")
        
        await client.close()
        print("✅ Client closed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Client creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_fallback_behavior():
    """Test fallback behavior without external service."""
    print("\n🛡️ Testing Fallback Behavior...")
    
    try:
        from repoindex.config import LensConfig
        from repoindex.pipeline.lens_client import LensIntegrationClient
        
        # Create client with fallback enabled
        config = LensConfig(
            enabled=True,
            base_url="http://localhost:9999",  # Non-existent service
            timeout=1,  # Short timeout
            max_retries=1,
            fallback_enabled=True
        )
        
        client = LensIntegrationClient(config)
        
        # Test fallback responses (these don't make actual HTTP calls)
        search_fallback = await client._fallback_response("GET", "/api/v1/search")
        print(f"✅ Search fallback: success={search_fallback.success}, fallback={search_fallback.from_fallback}")
        print(f"✅ Search fallback data: {len(search_fallback.data.get('results', []))} results")
        
        index_fallback = await client._fallback_response("POST", "/api/v1/index", {"repo": "test"})
        print(f"✅ Index fallback: success={index_fallback.success}, status={index_fallback.data.get('status')}")
        
        await client.close()
        return True
        
    except Exception as e:
        print(f"❌ Fallback test failed: {e}")
        return False


async def test_integration_helpers():
    """Test integration helper functions."""
    print("\n🔍 Testing Integration Helpers...")
    
    try:
        from repoindex.pipeline.lens_integration_helpers import print_lens_status
        
        # Test status printing (this should work without external service)
        print("✅ Testing status printer:")
        print_lens_status()
        
        return True
        
    except Exception as e:
        print(f"❌ Integration helpers test failed: {e}")
        return False


async def main():
    """Run all foundation tests."""
    print("=" * 60)
    print("🏗️ MIMIR-LENS INTEGRATION FOUNDATION TESTS")
    print("=" * 60)
    
    tests = [
        ("Configuration System", test_lens_config),
        ("Client Creation", test_lens_client_creation),
        ("Fallback Behavior", test_fallback_behavior),
        ("Integration Helpers", test_integration_helpers),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All foundation tests PASSED!")
        print("✅ Mimir-Lens integration foundation is ready for Phase 1")
        return 0
    else:
        print(f"\n⚠️  {len(results) - passed} tests FAILED")
        print("❌ Foundation needs fixes before proceeding")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        sys.exit(1)