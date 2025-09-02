#!/usr/bin/env python3
"""
Direct test of Mimir indexing pipeline to verify fixes work.
Tests the pipeline bypassing MCP to validate core functionality.
"""

import asyncio
import sys
import tempfile
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from repoindex.pipeline.run import IndexingPipeline
from repoindex.pipeline.query_engine import QueryEngine


async def test_mimir_pipeline():
    """Test the indexing pipeline directly."""
    print("🚀 Starting Mimir Pipeline Direct Test")
    
    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_dir = Path(temp_dir)
        print(f"📁 Using storage dir: {storage_dir}")
        
        # Initialize pipeline
        query_engine = QueryEngine()  # Remove storage_dir parameter
        pipeline = IndexingPipeline(
            storage_dir=storage_dir,
            query_engine=query_engine
        )
        
        # Test with current repository (Python codebase)
        repo_path = str(Path(__file__).parent)
        print(f"🔍 Testing with repo: {repo_path}")
        
        try:
            # Start indexing with Python language configuration
            print("⚙️  Starting indexing pipeline...")
            index_id = await pipeline.start_indexing(
                repo_path=repo_path,
                language="python",  # Test Python language support
                index_opts={
                    "stages": ["acquire", "repomapper", "serena", "leann", "snippets", "bundle"],
                    "file_patterns": ["*.py"],  # Focus on Python files
                }
            )
            
            print(f"📋 Pipeline started with index_id: {index_id}")
            
            # Wait for pipeline to complete
            max_wait = 120  # 2 minutes max for test
            wait_time = 0
            
            while wait_time < max_wait:
                # Check if pipeline is still active
                if index_id not in pipeline.active_pipelines:
                    print("❌ Pipeline no longer in active list")
                    break
                    
                context = pipeline.active_pipelines[index_id]
                
                # Try to read status file
                status_file = context.work_dir / "status.json"
                if status_file.exists():
                    import json
                    with open(status_file) as f:
                        status = json.load(f)
                    print(f"📊 Status: {status.get('state', 'unknown')} - {status.get('message', '')}")
                    
                    # Check if completed
                    if status.get('state') in ['COMPLETED', 'FAILED']:
                        break
                
                await asyncio.sleep(5)
                wait_time += 5
            
            # Check results
            print("\n🔍 Checking pipeline results...")
            
            if index_id in pipeline.active_pipelines:
                context = pipeline.active_pipelines[index_id]
                work_dir = context.work_dir
                
                # Check for expected output files
                expected_files = [
                    "serena_graph.jsonl",
                    "snippets.jsonl", 
                    "bundle.json"
                ]
                
                print(f"📁 Work directory: {work_dir}")
                print(f"📂 Work directory contents: {list(work_dir.glob('*'))}")
                
                for file_name in expected_files:
                    file_path = work_dir / file_name
                    if file_path.exists():
                        print(f"✅ Found {file_name} ({file_path.stat().st_size} bytes)")
                    else:
                        print(f"❌ Missing {file_name}")
                        
                # Check final status
                status_file = work_dir / "status.json"
                if status_file.exists():
                    import json
                    with open(status_file) as f:
                        final_status = json.load(f)
                    print(f"🏁 Final status: {final_status.get('state')} - {final_status.get('message')}")
                    
                    if final_status.get('state') == 'COMPLETED':
                        print("🎉 Pipeline completed successfully!")
                        return True
                    else:
                        print("❌ Pipeline did not complete successfully")
                        return False
                else:
                    print("❌ No status file found")
                    return False
            else:
                print("❌ Pipeline context not found")
                return False
                
        except Exception as e:
            print(f"💥 Error during pipeline test: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == "__main__":
    # Run the test
    result = asyncio.run(test_mimir_pipeline())
    if result:
        print("\n🎯 SUCCESS: Pipeline test passed!")
        sys.exit(0)
    else:
        print("\n💥 FAILURE: Pipeline test failed!")
        sys.exit(1)