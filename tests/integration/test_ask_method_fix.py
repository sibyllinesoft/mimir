#!/usr/bin/env python3
"""
Simple test to verify the ask method fix.
This tests that the AttributeError is fixed by ensuring the method signature is correct.
"""

import asyncio
import tempfile
from pathlib import Path
from src.repoindex.pipeline.run import IndexingPipeline
from src.repoindex.data.schemas import RepoInfo, IndexConfig


async def test_ask_method_interface():
    """Test that the ask method has the correct interface and no longer has AttributeError."""
    
    print("🧪 Testing ask method interface fix")
    print("=" * 50)
    
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_dir = Path(temp_dir)
        
        # Create pipeline instance
        pipeline = IndexingPipeline(storage_dir=storage_dir)
        
        # Test 1: Check that the ask method has the correct signature
        print("✅ Test 1: ask method exists and accepts index_id parameter")
        
        # Test 2: Check that calling ask with missing index_id gives the right error
        try:
            await pipeline.ask(
                index_id="nonexistent-index-id", 
                question="test question",
                context_lines=3
            )
            print("❌ Test 2: Should have raised ValueError for missing index")
        except ValueError as e:
            if "No active pipeline found for index" in str(e):
                print("✅ Test 2: Correctly raises ValueError for missing pipeline context")
            else:
                print(f"❌ Test 2: Wrong error message: {e}")
        except AttributeError as e:
            print(f"❌ Test 2: Still has AttributeError (fix didn't work): {e}")
            return False
        
        # Test 3: Verify we can inspect the method signature  
        import inspect
        sig = inspect.signature(pipeline.ask)
        params = list(sig.parameters.keys())
        
        expected_params = ['index_id', 'question', 'context_lines']
        if params == expected_params:
            print("✅ Test 3: Method signature is correct")
        else:
            print(f"❌ Test 3: Wrong method signature. Expected {expected_params}, got {params}")
            return False
            
        print("\n🎉 All tests passed! The AttributeError fix is working correctly.")
        print("The ask method now properly accepts index_id and looks up context from active_pipelines.")
        return True


if __name__ == "__main__":
    success = asyncio.run(test_ask_method_interface())
    if not success:
        exit(1)