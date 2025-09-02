#!/usr/bin/env python3
"""
Simple test to verify Mimir pipeline creates expected output files.
"""

import asyncio
import sys
import tempfile
import time
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from repoindex.pipeline.run import IndexingPipeline
from repoindex.pipeline.query_engine import QueryEngine


async def test_mimir_outputs():
    """Test that pipeline creates expected output files."""
    print("🚀 Testing Mimir Pipeline Output Generation")
    
    # Create temporary storage directory
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_dir = Path(temp_dir)
        print(f"📁 Storage dir: {storage_dir}")
        
        # Initialize pipeline
        query_engine = QueryEngine()
        pipeline = IndexingPipeline(storage_dir=storage_dir, query_engine=query_engine)
        
        # Start indexing
        repo_path = str(Path(__file__).parent)
        print(f"🔍 Indexing repo: {repo_path}")
        
        index_id = await pipeline.start_indexing(
            repo_path=repo_path,
            language="python",
            index_opts={"stages": ["acquire", "repomapper", "serena", "snippets", "bundle"]}
        )
        
        print(f"📋 Started pipeline: {index_id}")
        
        # Wait for completion (pipeline completes in ~1-2 seconds based on logs)
        await asyncio.sleep(5)
        
        # Check results
        if index_id in pipeline.active_pipelines:
            context = pipeline.active_pipelines[index_id]
            work_dir = context.work_dir
            
            print(f"📂 Work directory: {work_dir}")
            
            # List all files created
            all_files = list(work_dir.glob('*'))
            print(f"📋 Created files: {[f.name for f in all_files]}")
            
            # Check specific expected files
            expected_files = {
                "serena_graph.jsonl": "Serena symbol graph", 
                "snippets.jsonl": "Code snippets",
                "bundle.json": "Final bundle",
                "status.json": "Pipeline status"
            }
            
            results = {}
            for filename, description in expected_files.items():
                filepath = work_dir / filename
                if filepath.exists():
                    size = filepath.stat().st_size
                    print(f"✅ {description}: {filename} ({size} bytes)")
                    results[filename] = size
                else:
                    print(f"❌ {description}: {filename} MISSING")
                    results[filename] = None
            
            # Check final status
            status_file = work_dir / "status.json"
            if status_file.exists():
                import json
                with open(status_file) as f:
                    status = json.load(f)
                print(f"🏁 Final status: {status.get('state')} - {status.get('message')}")
                
                if status.get('state') == 'COMPLETED':
                    success = all(results.values())  # All files exist (non-None size)
                    if success:
                        print("🎉 SUCCESS: Pipeline completed and all files generated!")
                        return True
                    else:
                        print("❌ PARTIAL SUCCESS: Pipeline completed but some files missing")
                        return False
            
            print("❌ No status information available")
            return False
        else:
            print("❌ Pipeline context not found")
            return False


if __name__ == "__main__":
    result = asyncio.run(test_mimir_outputs())
    sys.exit(0 if result else 1)