"""
FastAPI server for optional management UI.

Provides local-only web interface for monitoring pipeline execution,
exploring search results, and managing index artifacts.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from ..data.schemas import (
    IndexManifest,
    PipelineStatus,
    SearchRepoRequest,
    SearchResponse,
    AskIndexRequest,
    AskResponse
)
from ..pipeline.run import IndexingPipeline
from ..util.fs import get_index_directory


class UIServer:
    """
    FastAPI server for management interface.
    
    Provides REST API and real-time updates for pipeline monitoring
    and interactive search functionality.
    """
    
    def __init__(self, storage_dir: Path, host: str = "127.0.0.1", port: int = 8080):
        """Initialize UI server."""
        self.storage_dir = storage_dir
        self.indexes_dir = storage_dir / "indexes"
        self.host = host
        self.port = port
        
        # Create FastAPI app
        self.app = FastAPI(
            title="Mimir Repository Index Manager",
            description="Management interface for Mimir deep code research system",
            version="0.1.0"
        )
        
        # Add CORS middleware (local-only)
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=[f"http://{host}:{port}", "http://localhost:8080"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # WebSocket connections for real-time updates
        self.connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Setup static file serving
        static_dir = Path(__file__).parent / "static"
        if static_dir.exists():
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    def _setup_routes(self):
        """Setup all API routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Serve main UI page."""
            return self._get_index_html()
        
        @self.app.get("/api/runs")
        async def list_runs() -> List[Dict]:
            """List all indexing runs."""
            runs = []
            
            if self.indexes_dir.exists():
                for index_dir in self.indexes_dir.iterdir():
                    if index_dir.is_dir():
                        try:
                            run_info = await self._get_run_info(index_dir)
                            runs.append(run_info)
                        except Exception:
                            continue
            
            # Sort by creation time (newest first)
            runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
            return runs
        
        @self.app.get("/api/runs/{run_id}")
        async def get_run(run_id: str) -> Dict:
            """Get detailed information about a specific run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            
            if not index_dir.exists():
                raise HTTPException(status_code=404, detail="Run not found")
            
            return await self._get_detailed_run_info(index_dir)
        
        @self.app.get("/api/runs/{run_id}/status")
        async def get_run_status(run_id: str) -> Dict:
            """Get current status of a run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            status_file = index_dir / "status.json"
            
            if not status_file.exists():
                raise HTTPException(status_code=404, detail="Status not found")
            
            with open(status_file) as f:
                return json.load(f)
        
        @self.app.get("/api/runs/{run_id}/manifest")
        async def get_run_manifest(run_id: str) -> Dict:
            """Get manifest for a run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            manifest_file = index_dir / "manifest.json"
            
            if not manifest_file.exists():
                raise HTTPException(status_code=404, detail="Manifest not found")
            
            with open(manifest_file) as f:
                return json.load(f)
        
        @self.app.get("/api/runs/{run_id}/log")
        async def get_run_log(run_id: str) -> Dict:
            """Get log for a run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            log_file = index_dir / "log.md"
            
            if not log_file.exists():
                raise HTTPException(status_code=404, detail="Log not found")
            
            content = log_file.read_text()
            return {"content": content, "format": "markdown"}
        
        @self.app.get("/api/runs/{run_id}/download")
        async def download_bundle(run_id: str):
            """Download bundle file for a run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            bundle_file = index_dir / "bundle.tar.zst"
            
            if not bundle_file.exists():
                raise HTTPException(status_code=404, detail="Bundle not found")
            
            return FileResponse(
                bundle_file,
                media_type="application/zstd",
                filename=f"mimir-index-{run_id}.tar.zst"
            )
        
        @self.app.post("/api/search")
        async def search_endpoint(request: SearchRepoRequest) -> SearchResponse:
            """Interactive search interface."""
            # This would integrate with the actual search pipeline
            # For now, return a placeholder response
            return SearchResponse(
                query=request.query,
                results=[],
                total_count=0,
                features_used=request.features,
                execution_time_ms=0.0,
                index_id=request.index_id
            )
        
        @self.app.post("/api/ask")
        async def ask_endpoint(request: AskIndexRequest) -> AskResponse:
            """Interactive question answering."""
            # This would integrate with the actual ask pipeline
            # For now, return a placeholder response
            return AskResponse(
                question=request.question,
                answer="This is a placeholder answer. The ask functionality would integrate with the actual symbol graph navigator.",
                citations=[],
                execution_time_ms=0.0,
                index_id=request.index_id
            )
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connections.remove(websocket)
        
        @self.app.get("/api/graph/{run_id}")
        async def get_graph_data(run_id: str) -> Dict:
            """Get graph visualization data."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            
            # Try to load graph data from various sources
            graph_data = {"nodes": [], "edges": []}
            
            # Load RepoMapper data if available
            repomap_file = index_dir / "repomap.json"
            if repomap_file.exists():
                with open(repomap_file) as f:
                    repomap_data = json.load(f)
                    graph_data.update(self._convert_repomap_to_graph(repomap_data))
            
            return graph_data
        
        @self.app.get("/api/coverage/{run_id}")
        async def get_coverage_data(run_id: str) -> Dict:
            """Get code coverage visualization data."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            
            coverage_data = {
                "files": [],
                "summary": {"total_files": 0, "indexed_files": 0, "coverage_percent": 0}
            }
            
            # Load manifest for file information
            manifest_file = index_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    coverage_data["summary"] = {
                        "total_files": manifest.get("counts", {}).get("files_total", 0),
                        "indexed_files": manifest.get("counts", {}).get("files_indexed", 0),
                        "coverage_percent": 100  # Simplified - all tracked files are indexed
                    }
            
            return coverage_data
    
    async def _get_run_info(self, index_dir: Path) -> Dict:
        """Get basic run information."""
        run_info = {
            "id": index_dir.name,
            "status": "unknown",
            "created_at": "",
            "repo_path": "",
            "progress": 0
        }
        
        # Load status
        status_file = index_dir / "status.json"
        if status_file.exists():
            with open(status_file) as f:
                status_data = json.load(f)
                run_info["status"] = status_data.get("state", "unknown")
                run_info["progress"] = status_data.get("progress", 0)
        
        # Load manifest for more details
        manifest_file = index_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest_data = json.load(f)
                run_info["created_at"] = manifest_data.get("created_at", "")
                run_info["repo_path"] = manifest_data.get("repo", {}).get("root", "")
        
        return run_info
    
    async def _get_detailed_run_info(self, index_dir: Path) -> Dict:
        """Get detailed run information."""
        run_info = await self._get_run_info(index_dir)
        
        # Add file listing
        files = []
        for file_path in index_dir.iterdir():
            if file_path.is_file():
                files.append({
                    "name": file_path.name,
                    "size": file_path.stat().st_size,
                    "modified": file_path.stat().st_mtime
                })
        
        run_info["files"] = files
        
        # Add statistics if manifest exists
        manifest_file = index_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest_data = json.load(f)
                run_info["statistics"] = manifest_data.get("counts", {})
                run_info["config"] = manifest_data.get("config", {})
        
        return run_info
    
    def _convert_repomap_to_graph(self, repomap_data: Dict) -> Dict:
        """Convert RepoMapper data to graph visualization format."""
        nodes = []
        edges = []
        
        # Add file nodes
        for file_rank in repomap_data.get("file_ranks", []):
            nodes.append({
                "id": file_rank["path"],
                "label": Path(file_rank["path"]).name,
                "group": "file",
                "rank": file_rank.get("rank", 0),
                "centrality": file_rank.get("centrality", 0)
            })
        
        # Add dependency edges
        for edge in repomap_data.get("edges", []):
            edges.append({
                "from": edge["source"],
                "to": edge["target"],
                "weight": edge.get("weight", 1),
                "type": edge.get("edge_type", "dependency")
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _get_index_html(self) -> str:
        """Generate main UI HTML page."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimir Repository Index Manager</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
            border-bottom: 2px solid #007acc;
            padding-bottom: 10px;
        }
        .status {
            padding: 20px;
            margin: 20px 0;
            border-radius: 6px;
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        .run-list {
            margin-top: 30px;
        }
        .run-item {
            padding: 15px;
            margin: 10px 0;
            border: 1px solid #ddd;
            border-radius: 6px;
            background: #fafafa;
        }
        .search-section {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 6px;
        }
        input, button {
            padding: 10px;
            margin: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        button {
            background: #007acc;
            color: white;
            cursor: pointer;
        }
        button:hover {
            background: #005c99;
        }
        .loading {
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧠 Mimir Repository Index Manager</h1>
        
        <div class="status">
            <h3>System Status</h3>
            <p>Mimir deep code research system is running on <strong>localhost:8080</strong></p>
            <p>WebSocket connection: <span id="ws-status" class="loading">Connecting...</span></p>
        </div>
        
        <div class="search-section">
            <h3>Interactive Search</h3>
            <div>
                <input type="text" id="search-query" placeholder="Search the indexed repository..." style="width: 300px;">
                <button onclick="performSearch()">Search</button>
            </div>
            <div>
                <input type="text" id="ask-question" placeholder="Ask a question about the code..." style="width: 300px;">
                <button onclick="askQuestion()">Ask</button>
            </div>
            <div id="search-results" style="margin-top: 20px;"></div>
        </div>
        
        <div class="run-list">
            <h3>Recent Indexing Runs</h3>
            <div id="runs" class="loading">Loading runs...</div>
        </div>
    </div>
    
    <script>
        // WebSocket connection for real-time updates
        let ws;
        
        function connectWebSocket() {
            ws = new WebSocket('ws://127.0.0.1:8080/ws');
            
            ws.onopen = function() {
                document.getElementById('ws-status').textContent = 'Connected';
                document.getElementById('ws-status').style.color = 'green';
            };
            
            ws.onclose = function() {
                document.getElementById('ws-status').textContent = 'Disconnected';
                document.getElementById('ws-status').style.color = 'red';
                // Reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
            ws.onmessage = function(event) {
                const data = JSON.parse(event.data);
                if (data.type === 'status_update') {
                    loadRuns(); // Refresh runs list
                }
            };
        }
        
        // Load runs list
        async function loadRuns() {
            try {
                const response = await fetch('/api/runs');
                const runs = await response.json();
                
                const runsDiv = document.getElementById('runs');
                if (runs.length === 0) {
                    runsDiv.innerHTML = '<p>No indexing runs found.</p>';
                    return;
                }
                
                let html = '';
                runs.forEach(run => {
                    html += `
                        <div class="run-item">
                            <strong>ID:</strong> ${run.id}<br>
                            <strong>Status:</strong> ${run.status}<br>
                            <strong>Repository:</strong> ${run.repo_path}<br>
                            <strong>Progress:</strong> ${run.progress}%<br>
                            <strong>Created:</strong> ${new Date(run.created_at).toLocaleString()}<br>
                            <button onclick="viewRun('${run.id}')">View Details</button>
                            <button onclick="downloadBundle('${run.id}')">Download Bundle</button>
                        </div>
                    `;
                });
                
                runsDiv.innerHTML = html;
            } catch (error) {
                document.getElementById('runs').innerHTML = '<p>Error loading runs.</p>';
            }
        }
        
        // Search functionality
        async function performSearch() {
            const query = document.getElementById('search-query').value;
            if (!query.trim()) return;
            
            const resultsDiv = document.getElementById('search-results');
            resultsDiv.innerHTML = '<p class="loading">Searching...</p>';
            
            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        index_id: 'latest',
                        query: query,
                        k: 20,
                        features: {vector: true, symbol: true, graph: true}
                    })
                });
                
                const results = await response.json();
                
                if (results.results.length === 0) {
                    resultsDiv.innerHTML = '<p>No results found.</p>';
                } else {
                    let html = `<h4>Search Results (${results.total_count})</h4>`;
                    results.results.forEach(result => {
                        html += `
                            <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
                                <strong>${result.path}</strong> (Score: ${result.score.toFixed(2)})<br>
                                <code>${result.content.text}</code>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                }
            } catch (error) {
                resultsDiv.innerHTML = '<p>Search failed.</p>';
            }
        }
        
        // Ask functionality
        async function askQuestion() {
            const question = document.getElementById('ask-question').value;
            if (!question.trim()) return;
            
            const resultsDiv = document.getElementById('search-results');
            resultsDiv.innerHTML = '<p class="loading">Processing question...</p>';
            
            try {
                const response = await fetch('/api/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        index_id: 'latest',
                        question: question
                    })
                });
                
                const result = await response.json();
                
                let html = `<h4>Answer</h4><p>${result.answer}</p>`;
                if (result.citations.length > 0) {
                    html += `<h5>Citations (${result.citations.length})</h5>`;
                    result.citations.forEach(citation => {
                        html += `<div style="margin: 5px 0;"><code>${citation.path}</code></div>`;
                    });
                }
                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = '<p>Question processing failed.</p>';
            }
        }
        
        // Utility functions
        function viewRun(runId) {
            window.open(`/api/runs/${runId}`, '_blank');
        }
        
        function downloadBundle(runId) {
            window.open(`/api/runs/${runId}/download`, '_blank');
        }
        
        // Initialize
        connectWebSocket();
        loadRuns();
        
        // Handle Enter key for search
        document.getElementById('search-query').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') performSearch();
        });
        
        document.getElementById('ask-question').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askQuestion();
        });
    </script>
</body>
</html>
        """
    
    async def broadcast_update(self, message: Dict):
        """Broadcast update to all connected WebSocket clients."""
        if self.connections:
            message_str = json.dumps(message)
            disconnected = []
            
            for websocket in self.connections:
                try:
                    await websocket.send_text(message_str)
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected clients
            for ws in disconnected:
                self.connections.remove(ws)
    
    async def start(self):
        """Start the UI server."""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


async def main():
    """Main entry point for UI server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Mimir Repository Index Manager UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--storage-dir", type=Path, help="Storage directory for indexes")
    
    args = parser.parse_args()
    
    storage_dir = args.storage_dir or Path.home() / ".cache" / "mimir"
    
    ui_server = UIServer(storage_dir, args.host, args.port)
    
    print(f"Starting Mimir UI server on http://{args.host}:{args.port}")
    await ui_server.start()


if __name__ == "__main__":
    asyncio.run(main())