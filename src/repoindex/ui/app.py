"""
FastAPI server for optional management UI.

Provides local-only web interface for monitoring pipeline execution,
exploring search results, and managing index artifacts.
"""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from ..data.schemas import (
    AskIndexRequest,
    AskResponse,
    SearchRepoRequest,
    SearchResponse,
)
from ..util.fs import get_index_directory
from ..util.log import get_logger

logger = get_logger(__name__)


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
            version="1.0.0",
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
        self.connections: list[WebSocket] = []

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
        async def list_runs() -> list[dict]:
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
        async def get_run(run_id: str) -> dict:
            """Get detailed information about a specific run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)

            if not index_dir.exists():
                raise HTTPException(status_code=404, detail="Run not found")

            return await self._get_detailed_run_info(index_dir)

        @self.app.get("/api/runs/{run_id}/status")
        async def get_run_status(run_id: str) -> dict:
            """Get current status of a run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            status_file = index_dir / "status.json"

            if not status_file.exists():
                raise HTTPException(status_code=404, detail="Status not found")

            with open(status_file) as f:
                return json.load(f)

        @self.app.get("/api/runs/{run_id}/manifest")
        async def get_run_manifest(run_id: str) -> dict:
            """Get manifest for a run."""
            index_dir = get_index_directory(self.indexes_dir, run_id)
            manifest_file = index_dir / "manifest.json"

            if not manifest_file.exists():
                raise HTTPException(status_code=404, detail="Manifest not found")

            with open(manifest_file) as f:
                return json.load(f)

        @self.app.get("/api/runs/{run_id}/log")
        async def get_run_log(run_id: str) -> dict:
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
                bundle_file, media_type="application/zstd", filename=f"mimir-index-{run_id}.tar.zst"
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
                index_id=request.index_id,
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
                index_id=request.index_id,
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
        async def get_graph_data(run_id: str) -> dict:
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
        async def get_coverage_data(run_id: str) -> dict:
            """Get code coverage visualization data."""
            index_dir = get_index_directory(self.indexes_dir, run_id)

            coverage_data = {
                "files": [],
                "summary": {"total_files": 0, "indexed_files": 0, "coverage_percent": 0},
            }

            # Load manifest for file information
            manifest_file = index_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    coverage_data["summary"] = {
                        "total_files": manifest.get("counts", {}).get("files_total", 0),
                        "indexed_files": manifest.get("counts", {}).get("files_indexed", 0),
                        "coverage_percent": 100,  # Simplified - all tracked files are indexed
                    }

            return coverage_data

    async def _get_run_info(self, index_dir: Path) -> dict:
        """Get basic run information."""
        run_info = {
            "id": index_dir.name,
            "status": "unknown",
            "created_at": "",
            "repo_path": "",
            "progress": 0,
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

    async def _get_detailed_run_info(self, index_dir: Path) -> dict:
        """Get detailed run information."""
        run_info = await self._get_run_info(index_dir)

        # Add file listing
        files = []
        for file_path in index_dir.iterdir():
            if file_path.is_file():
                files.append(
                    {
                        "name": file_path.name,
                        "size": file_path.stat().st_size,
                        "modified": file_path.stat().st_mtime,
                    }
                )

        run_info["files"] = files

        # Add statistics if manifest exists
        manifest_file = index_dir / "manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest_data = json.load(f)
                run_info["statistics"] = manifest_data.get("counts", {})
                run_info["config"] = manifest_data.get("config", {})

        return run_info

    def _convert_repomap_to_graph(self, repomap_data: dict) -> dict:
        """Convert RepoMapper data to graph visualization format."""
        nodes = []
        edges = []

        # Add file nodes
        for file_rank in repomap_data.get("file_ranks", []):
            nodes.append(
                {
                    "id": file_rank["path"],
                    "label": Path(file_rank["path"]).name,
                    "group": "file",
                    "rank": file_rank.get("rank", 0),
                    "centrality": file_rank.get("centrality", 0),
                }
            )

        # Add dependency edges
        for edge in repomap_data.get("edges", []):
            edges.append(
                {
                    "from": edge["source"],
                    "to": edge["target"],
                    "weight": edge.get("weight", 1),
                    "type": edge.get("edge_type", "dependency"),
                }
            )

        return {"nodes": nodes, "edges": edges}

    def _get_index_html(self) -> str:
        """Generate main UI HTML page with enhanced styling and UX."""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimir Repository Index Manager</title>
    <style>
        :root {
            --primary-color: #2563eb;
            --primary-hover: #1d4ed8;
            --secondary-color: #64748b;
            --success-color: #059669;
            --warning-color: #d97706;
            --error-color: #dc2626;
            --background-color: #f8fafc;
            --surface-color: #ffffff;
            --border-color: #e2e8f0;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
            color: var(--text-primary);
            line-height: 1.6;
            min-height: 100vh;
        }

        .header {
            background: linear-gradient(135deg, var(--primary-color), var(--primary-hover));
            color: white;
            padding: 2rem 0;
            margin-bottom: 2rem;
            box-shadow: var(--shadow-lg);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem;
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        .header p {
            font-size: 1.1rem;
            opacity: 0.9;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 2rem 2rem;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
        }

        .card {
            background: var(--surface-color);
            border-radius: 12px;
            box-shadow: var(--shadow);
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }

        .card-header {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            background: linear-gradient(135deg, #f8fafc, #f1f5f9);
        }

        .card-header h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .card-content {
            padding: 1.5rem;
        }

        .status-card {
            grid-column: 1 / -1;
        }

        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .status-item {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 1rem;
            background: #f8fafc;
            border-radius: 8px;
            border: 1px solid var(--border-color);
        }

        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: var(--success-color);
            box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.2);
            animation: pulse 2s infinite;
        }

        .status-indicator.connecting {
            background: var(--warning-color);
            box-shadow: 0 0 0 3px rgba(217, 119, 6, 0.2);
        }

        .status-indicator.error {
            background: var(--error-color);
            box-shadow: 0 0 0 3px rgba(220, 38, 38, 0.2);
        }

        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }

        .search-form {
            display: flex;
            flex-direction: column;
            gap: 1rem;
        }

        .input-group {
            display: flex;
            gap: 0.5rem;
        }

        input[type="text"] {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 2px solid var(--border-color);
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.2s, box-shadow 0.2s;
            background: white;
        }

        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
        }

        .btn {
            padding: 0.75rem 1.5rem;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            text-decoration: none;
            justify-content: center;
        }

        .btn-primary {
            background: var(--primary-color);
            color: white;
        }

        .btn-primary:hover {
            background: var(--primary-hover);
            transform: translateY(-1px);
        }

        .btn-secondary {
            background: var(--secondary-color);
            color: white;
        }

        .btn-secondary:hover {
            background: #475569;
            transform: translateY(-1px);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none !important;
        }

        .run-item {
            padding: 1.5rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            background: white;
            transition: all 0.2s;
        }

        .run-item:hover {
            border-color: var(--primary-color);
            box-shadow: var(--shadow);
        }

        .run-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 1rem;
        }

        .run-id {
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, monospace;
            background: #f1f5f9;
            padding: 0.25rem 0.5rem;
            border-radius: 4px;
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .run-status {
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            font-size: 0.875rem;
            font-weight: 500;
            text-transform: capitalize;
        }

        .run-status.running {
            background: rgba(217, 119, 6, 0.1);
            color: var(--warning-color);
        }

        .run-status.completed {
            background: rgba(5, 150, 105, 0.1);
            color: var(--success-color);
        }

        .run-status.failed {
            background: rgba(220, 38, 38, 0.1);
            color: var(--error-color);
        }

        .run-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }

        .run-detail {
            display: flex;
            flex-direction: column;
        }

        .run-detail label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 0.25rem;
        }

        .run-detail value {
            font-weight: 500;
            color: var(--text-primary);
        }

        .run-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: 1rem;
        }

        .progress-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }

        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--primary-color), var(--primary-hover));
            transition: width 0.3s ease;
        }

        .search-results {
            margin-top: 1.5rem;
            max-height: 600px;
            overflow-y: auto;
        }

        .search-result {
            padding: 1rem;
            margin-bottom: 1rem;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            background: white;
            transition: border-color 0.2s;
        }

        .search-result:hover {
            border-color: var(--primary-color);
        }

        .search-result-header {
            display: flex;
            justify-content: between;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .search-result-path {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.875rem;
            color: var(--primary-color);
            font-weight: 500;
        }

        .search-result-score {
            font-size: 0.875rem;
            color: var(--text-secondary);
            background: #f1f5f9;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
        }

        .search-result-content {
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.875rem;
            background: #f8fafc;
            padding: 1rem;
            border-radius: 6px;
            border-left: 3px solid var(--primary-color);
            white-space: pre-wrap;
            overflow-x: auto;
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
            color: var(--text-secondary);
        }

        .loading::before {
            content: '';
            width: 20px;
            height: 20px;
            border: 2px solid var(--border-color);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 0.5rem;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            color: var(--text-secondary);
        }

        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }

        @media (max-width: 768px) {
            .container {
                grid-template-columns: 1fr;
                padding: 0 1rem 2rem;
            }

            .header-content {
                padding: 0 1rem;
            }

            .header h1 {
                font-size: 2rem;
            }

            .input-group {
                flex-direction: column;
            }

            .run-details {
                grid-template-columns: 1fr;
            }
        }

        /* Dark theme styles for code blocks */
        .code-block {
            background: #1e293b;
            color: #e2e8f0;
            padding: 1rem;
            border-radius: 8px;
            font-family: 'SF Mono', Monaco, monospace;
            font-size: 0.875rem;
            overflow-x: auto;
            border: 1px solid #334155;
        }

        /* Smooth scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #f1f5f9;
        }

        ::-webkit-scrollbar-thumb {
            background: #cbd5e1;
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #94a3b8;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <h1>🧠 Mimir Repository Index Manager</h1>
            <p>Deep code research system with intelligent indexing and search capabilities</p>
        </div>
    </div>

    <div class="container">
        <div class="card status-card">
            <div class="card-header">
                <h3>📊 System Status</h3>
            </div>
            <div class="card-content">
                <div class="status-grid">
                    <div class="status-item">
                        <div class="status-indicator" id="system-status"></div>
                        <div>
                            <strong>System</strong><br>
                            <span id="system-text">Running on localhost:8080</span>
                        </div>
                    </div>
                    <div class="status-item">
                        <div class="status-indicator connecting" id="ws-status-indicator"></div>
                        <div>
                            <strong>WebSocket</strong><br>
                            <span id="ws-status">Connecting...</span>
                        </div>
                    </div>
                    <div class="status-item">
                        <div class="status-indicator" id="pipeline-status"></div>
                        <div>
                            <strong>Pipeline</strong><br>
                            <span id="pipeline-text">Ready</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h3>🔍 Interactive Search</h3>
            </div>
            <div class="card-content">
                <div class="search-form">
                    <div class="input-group">
                        <input type="text" id="search-query" placeholder="Search the indexed repository...">
                        <button class="btn btn-primary" onclick="performSearch()">🔍 Search</button>
                    </div>
                    <div class="input-group">
                        <input type="text" id="ask-question" placeholder="Ask a question about the code...">
                        <button class="btn btn-secondary" onclick="askQuestion()">💭 Ask</button>
                    </div>
                </div>
                <div id="search-results" class="search-results"></div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h3>📋 Recent Indexing Runs</h3>
            </div>
            <div class="card-content">
                <div id="runs" class="loading">Loading runs...</div>
            </div>
        </div>
    </div>

    <script>
        // WebSocket connection for real-time updates
        let ws;

        function connectWebSocket() {
            ws = new WebSocket('ws://127.0.0.1:8080/ws');

            ws.onopen = function() {
                document.getElementById('ws-status').textContent = 'Connected';
                const indicator = document.getElementById('ws-status-indicator');
                indicator.className = 'status-indicator';
            };

            ws.onclose = function() {
                document.getElementById('ws-status').textContent = 'Disconnected';
                const indicator = document.getElementById('ws-status-indicator');
                indicator.className = 'status-indicator error';
                // Reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };

            ws.onerror = function() {
                document.getElementById('ws-status').textContent = 'Connection Error';
                const indicator = document.getElementById('ws-status-indicator');
                indicator.className = 'status-indicator error';
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
                    runsDiv.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">📋</div>
                            <p>No indexing runs found</p>
                            <small>Start indexing a repository to see runs here</small>
                        </div>
                    `;
                    return;
                }

                let html = '';
                runs.forEach(run => {
                    const statusClass = run.status.toLowerCase();
                    const progress = run.progress || 0;
                    const createdDate = new Date(run.created_at).toLocaleString();

                    html += `
                        <div class="run-item">
                            <div class="run-header">
                                <div class="run-id">${run.id}</div>
                                <span class="run-status ${statusClass}">${run.status}</span>
                            </div>
                            <div class="run-details">
                                <div class="run-detail">
                                    <label>Repository</label>
                                    <value>${run.repo_path || 'Unknown'}</value>
                                </div>
                                <div class="run-detail">
                                    <label>Created</label>
                                    <value>${createdDate}</value>
                                </div>
                                <div class="run-detail">
                                    <label>Progress</label>
                                    <value>${progress}%</value>
                                </div>
                            </div>
                            <div class="progress-bar">
                                <div class="progress-fill" style="width: ${progress}%"></div>
                            </div>
                            <div class="run-actions">
                                <button class="btn btn-secondary" onclick="viewRun('${run.id}')">
                                    📋 View Details
                                </button>
                                <button class="btn btn-primary" onclick="downloadBundle('${run.id}')">
                                    ⬇️ Download Bundle
                                </button>
                            </div>
                        </div>
                    `;
                });

                runsDiv.innerHTML = html;
            } catch (error) {
                document.getElementById('runs').innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">⚠️</div>
                        <p>Error loading runs</p>
                        <small>Please check the server connection</small>
                    </div>
                `;
            }
        }

        // Search functionality
        async function performSearch() {
            const query = document.getElementById('search-query').value;
            if (!query.trim()) return;

            const resultsDiv = document.getElementById('search-results');
            resultsDiv.innerHTML = '<div class="loading">Searching...</div>';

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
                    resultsDiv.innerHTML = `
                        <div class="empty-state">
                            <div class="empty-state-icon">🔍</div>
                            <p>No results found</p>
                            <small>Try different search terms or check if the repository is indexed</small>
                        </div>
                    `;
                } else {
                    let html = `<h4 style="margin-bottom: 1rem; color: var(--text-primary);">Search Results (${results.total_count})</h4>`;
                    results.results.forEach(result => {
                        html += `
                            <div class="search-result">
                                <div class="search-result-header">
                                    <div class="search-result-path">${result.path}</div>
                                    <div class="search-result-score">Score: ${result.score.toFixed(3)}</div>
                                </div>
                                <div class="search-result-content">${escapeHtml(result.content.text)}</div>
                            </div>
                        `;
                    });
                    resultsDiv.innerHTML = html;
                }
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">⚠️</div>
                        <p>Search failed</p>
                        <small>Please check the server connection and try again</small>
                    </div>
                `;
            }
        }

        // Ask functionality
        async function askQuestion() {
            const question = document.getElementById('ask-question').value;
            if (!question.trim()) return;

            const resultsDiv = document.getElementById('search-results');
            resultsDiv.innerHTML = '<div class="loading">Processing question...</div>';

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

                let html = `
                    <div class="search-result">
                        <div class="search-result-header">
                            <div class="search-result-path">💭 Answer</div>
                        </div>
                        <div class="search-result-content">${escapeHtml(result.answer)}</div>
                    </div>
                `;

                if (result.citations && result.citations.length > 0) {
                    html += `<h5 style="margin: 1.5rem 0 1rem; color: var(--text-primary);">📚 Citations (${result.citations.length})</h5>`;
                    result.citations.forEach(citation => {
                        html += `
                            <div class="search-result">
                                <div class="search-result-header">
                                    <div class="search-result-path">${citation.path}</div>
                                </div>
                            </div>
                        `;
                    });
                }

                resultsDiv.innerHTML = html;
            } catch (error) {
                resultsDiv.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">⚠️</div>
                        <p>Question processing failed</p>
                        <small>Please check the server connection and try again</small>
                    </div>
                `;
            }
        }

        // Utility functions
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
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

    async def broadcast_update(self, message: dict):
        """Broadcast update to all connected WebSocket clients."""
        if self.connections:
            message_str = json.dumps(message)
            disconnected = []

            for websocket in self.connections:
                try:
                    await websocket.send_text(message_str)
                except Exception:
                    disconnected.append(websocket)

            # Remove disconnected clients
            for ws in disconnected:
                self.connections.remove(ws)

    async def start(self):
        """Start the UI server."""
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
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

    logger.info(f"Starting Mimir UI server on http://{args.host}:{args.port}")
    await ui_server.start()


if __name__ == "__main__":
    asyncio.run(main())
