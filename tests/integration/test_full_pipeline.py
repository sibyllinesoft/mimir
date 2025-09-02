#!/usr/bin/env python3
"""
Quick test script to verify the full MCP pipeline works with our CLI tools.
"""

import asyncio
import json
import tempfile
import subprocess
from pathlib import Path

from src.repoindex.mcp.server import MCPServer


async def test_full_pipeline():
    """Test the complete pipeline with our CLI tools."""
    
    # Create a simple test repository
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo = Path(tmp_dir)
        
        # Create some test Python files
        (repo / "main.py").write_text('''
def main():
    print("Hello World")
    
def helper_function():
    return "helper"

if __name__ == "__main__":
    main()
''')
        
        (repo / "utils.py").write_text('''
def utility_function():
    return "utility"
    
class UtilityClass:
    def method(self):
        pass
''')
        
        # Initialize as git repository
        subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo, check=True, capture_output=True)
        
        # Create MCP server
        with tempfile.TemporaryDirectory() as storage_dir:
            mcp_server = MCPServer(storage_dir=Path(storage_dir))
            
            print(f"🧪 Testing with repository: {repo}")
            
            # Start indexing
            result = await mcp_server._ensure_repo_index({
                "path": str(repo),
                "index_opts": {
                    "languages": ["py"],
                    "excludes": ["__pycache__/", ".git/", ".pytest_cache/"]
                }
            })
            
            if result.isError:
                print(f"❌ Indexing failed: {result.content}")
                return
            
            response_data = json.loads(result.content[0].text)
            index_id = response_data["index_id"]
            print(f"✅ Started pipeline with index ID: {index_id}")
            
            # Wait longer for pipeline to complete
            print("⏳ Waiting for pipeline to complete...")
            
            for i in range(30):  # Wait up to 30 seconds
                await asyncio.sleep(1)
                
                # Check status
                status_uri = "repo://status"
                status_result = await mcp_server._read_resource(status_uri)
                
                if hasattr(status_result, 'contents'):
                    status_data = json.loads(status_result.contents[0].text)
                    active_count = len(status_data.get('active_pipelines', []))
                    
                    if active_count == 0:
                        print(f"🎉 Pipeline completed after {i+1} seconds")
                        break
                    else:
                        if (i + 1) % 5 == 0:  # Progress update every 5 seconds
                            print(f"⏱️  Still processing... ({i+1}s elapsed)")
                
            # Check final results
            print("🔍 Checking pipeline results...")
            
            # Check logs
            logs_uri = f"repo://logs/{index_id}"
            logs_result = await mcp_server._read_resource(logs_uri)
            
            if hasattr(logs_result, 'contents'):
                log_content = logs_result.contents[0].text
                print(f"\n📄 Pipeline logs:")
                print(log_content[-1000:])  # Show last 1000 chars
                
                # Analyze log content
                if "repomapper" in log_content:
                    print("✅ RepoMapper stage executed")
                if "serena" in log_content:
                    print("✅ Serena stage executed")
                if "leann" in log_content:
                    print("✅ LEANN stage executed")
                    
                if "DONE" in log_content:
                    print("🎯 Pipeline completed successfully!")
                else:
                    print("⚠️ Pipeline may not have completed successfully")
            
            # Check manifest
            try:
                manifest_uri = f"repo://manifest/{index_id}"
                manifest_result = await mcp_server._read_resource(manifest_uri)
                
                if hasattr(manifest_result, 'contents'):
                    manifest_data = json.loads(manifest_result.contents[0].text)
                    print(f"\n📋 Manifest contains:")
                    print(f"  - Files: {len(manifest_data.get('files', []))}")
                    print(f"  - Has RepoMap: {'repomap_data' in manifest_data}")
                    print(f"  - Has Serena Graph: {'serena_graph' in manifest_data}")
                    print(f"  - Has Vector Index: {'vector_index' in manifest_data}")
                else:
                    print("❌ Manifest not available")
                    
            except Exception as e:
                print(f"⚠️ Could not read manifest: {e}")

if __name__ == "__main__":
    asyncio.run(test_full_pipeline())