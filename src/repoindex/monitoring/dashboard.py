"""
Grafana dashboard generation for Mimir Deep Code Research System.

Provides automated generation of comprehensive monitoring dashboards
for pipeline performance, system health, and business metrics.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..util.logging_config import get_logger

logger = get_logger("monitoring.dashboard")


@dataclass
class Panel:
    """Grafana panel definition."""

    title: str
    panel_type: str  # graph, stat, table, heatmap, etc.
    targets: list[dict[str, Any]]
    grid_pos: dict[str, int] = field(default_factory=dict)
    options: dict[str, Any] = field(default_factory=dict)
    field_config: dict[str, Any] = field(default_factory=dict)
    id: int | None = None


@dataclass
class Dashboard:
    """Grafana dashboard definition."""

    title: str
    description: str
    panels: list[Panel] = field(default_factory=list)
    time_range: dict[str, str] = field(default_factory=lambda: {"from": "now-1h", "to": "now"})
    refresh: str = "30s"
    tags: list[str] = field(default_factory=list)
    variables: list[dict[str, Any]] = field(default_factory=list)


class DashboardGenerator:
    """
    Generator for Grafana dashboards tailored to Mimir monitoring.

    Creates comprehensive dashboards for:
    - Pipeline execution monitoring
    - Performance and latency tracking
    - System resource utilization
    - Error tracking and debugging
    - Business metrics visualization
    """

    def __init__(self, datasource_name: str = "Prometheus"):
        """Initialize dashboard generator."""
        self.datasource_name = datasource_name
        self.panel_id_counter = 1

    def _next_panel_id(self) -> int:
        """Get next panel ID."""
        panel_id = self.panel_id_counter
        self.panel_id_counter += 1
        return panel_id

    def _create_target(self, expr: str, legend: str = "", interval: str = ""):
        """Create Prometheus query target."""
        return {
            "expr": expr,
            "legendFormat": legend,
            "interval": interval,
            "datasource": {"type": "prometheus", "uid": self.datasource_name},
        }

    def _create_grid_pos(self, x: int, y: int, w: int = 12, h: int = 8):
        """Create panel grid position."""
        return {"x": x, "y": y, "w": w, "h": h}

    def generate_pipeline_dashboard(self) -> Dashboard:
        """Generate comprehensive pipeline monitoring dashboard."""
        dashboard = Dashboard(
            title="Mimir Pipeline Monitoring",
            description="Comprehensive monitoring of the 6-stage repository indexing pipeline",
            tags=["mimir", "pipeline", "indexing"],
        )

        # Pipeline Success Rate
        success_rate_panel = Panel(
            title="Pipeline Success Rate",
            panel_type="stat",
            targets=[
                self._create_target(
                    'rate(mimir_pipeline_executions_total{status="success"}[5m]) / rate(mimir_pipeline_executions_total[5m]) * 100',
                    "Success Rate %",
                )
            ],
            grid_pos=self._create_grid_pos(0, 0, 6, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "orientation": "auto",
                "textMode": "auto",
                "colorMode": "background",
            },
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "thresholds": {
                        "steps": [
                            {"color": "red", "value": 0},
                            {"color": "yellow", "value": 80},
                            {"color": "green", "value": 95},
                        ]
                    },
                }
            },
        )
        dashboard.panels.append(success_rate_panel)

        # Active Pipelines
        active_pipelines_panel = Panel(
            title="Active Pipelines",
            panel_type="stat",
            targets=[self._create_target("mimir_pipeline_active_count", "Active")],
            grid_pos=self._create_grid_pos(6, 0, 3, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "background",
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 5},
                            {"color": "red", "value": 10},
                        ]
                    },
                }
            },
        )
        dashboard.panels.append(active_pipelines_panel)

        # Queue Length
        queue_panel = Panel(
            title="Pipeline Queue",
            panel_type="stat",
            targets=[self._create_target("mimir_pipeline_queue_length", "Queued")],
            grid_pos=self._create_grid_pos(9, 0, 3, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "background",
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 3},
                            {"color": "red", "value": 8},
                        ]
                    },
                }
            },
        )
        dashboard.panels.append(queue_panel)

        # Pipeline Duration by Stage
        duration_panel = Panel(
            title="Pipeline Duration by Stage",
            panel_type="graph",
            targets=[
                self._create_target(
                    'histogram_quantile(0.95, rate(mimir_pipeline_duration_seconds_bucket{stage!=""}[5m]))',
                    "P95 - {{stage}}",
                ),
                self._create_target(
                    'histogram_quantile(0.50, rate(mimir_pipeline_duration_seconds_bucket{stage!=""}[5m]))',
                    "P50 - {{stage}}",
                ),
            ],
            grid_pos=self._create_grid_pos(0, 4, 12, 8),
            id=self._next_panel_id(),
            options={
                "legend": {"displayMode": "table", "placement": "bottom"},
                "tooltip": {"mode": "multi", "sort": "desc"},
            },
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(duration_panel)

        # Stage Success Rates
        stage_success_panel = Panel(
            title="Stage Success Rates",
            panel_type="graph",
            targets=[
                self._create_target(
                    'rate(mimir_pipeline_executions_total{status="success",stage!=""}[5m]) / rate(mimir_pipeline_executions_total{stage!=""}[5m]) * 100',
                    "{{stage}}",
                )
            ],
            grid_pos=self._create_grid_pos(0, 12, 6, 8),
            id=self._next_panel_id(),
            options={"legend": {"displayMode": "list", "placement": "bottom"}},
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(stage_success_panel)

        # Files Processed
        files_panel = Panel(
            title="Files Processed per Stage",
            panel_type="graph",
            targets=[
                self._create_target(
                    'histogram_quantile(0.95, rate(mimir_pipeline_files_processed_bucket{stage!=""}[5m]))',
                    "P95 - {{stage}}",
                ),
                self._create_target(
                    'histogram_quantile(0.50, rate(mimir_pipeline_files_processed_bucket{stage!=""}[5m]))',
                    "P50 - {{stage}}",
                ),
            ],
            grid_pos=self._create_grid_pos(6, 12, 6, 8),
            id=self._next_panel_id(),
            options={"legend": {"displayMode": "list", "placement": "bottom"}},
            field_config={
                "defaults": {
                    "unit": "short",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(files_panel)

        # Error Rate by Stage
        error_panel = Panel(
            title="Error Rate by Stage",
            panel_type="graph",
            targets=[
                self._create_target(
                    'rate(mimir_pipeline_errors_total{stage!=""}[5m])', "{{stage}} - {{error_type}}"
                )
            ],
            grid_pos=self._create_grid_pos(0, 20, 12, 8),
            id=self._next_panel_id(),
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(error_panel)

        return dashboard

    def generate_performance_dashboard(self) -> Dashboard:
        """Generate performance monitoring dashboard."""
        dashboard = Dashboard(
            title="Mimir Performance Monitoring",
            description="System performance, search latency, and resource utilization",
            tags=["mimir", "performance", "latency"],
        )

        # Search Latency
        search_latency_panel = Panel(
            title="Search Request Latency",
            panel_type="graph",
            targets=[
                self._create_target(
                    "histogram_quantile(0.95, rate(mimir_search_duration_seconds_bucket[5m]))",
                    "P95",
                ),
                self._create_target(
                    "histogram_quantile(0.50, rate(mimir_search_duration_seconds_bucket[5m]))",
                    "P50",
                ),
                self._create_target(
                    "histogram_quantile(0.99, rate(mimir_search_duration_seconds_bucket[5m]))",
                    "P99",
                ),
            ],
            grid_pos=self._create_grid_pos(0, 0, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(search_latency_panel)

        # Search Request Rate
        search_rate_panel = Panel(
            title="Search Request Rate",
            panel_type="graph",
            targets=[
                self._create_target(
                    "rate(mimir_search_requests_total[5m])", "{{search_type}} - {{status}}"
                )
            ],
            grid_pos=self._create_grid_pos(6, 0, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(search_rate_panel)

        # Memory Usage
        memory_panel = Panel(
            title="Memory Usage",
            panel_type="graph",
            targets=[
                self._create_target(
                    'mimir_memory_usage_bytes{type="rss"} / 1024 / 1024', "RSS Memory (MB)"
                ),
                self._create_target(
                    'mimir_memory_usage_bytes{type="vms"} / 1024 / 1024', "Virtual Memory (MB)"
                ),
            ],
            grid_pos=self._create_grid_pos(0, 8, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "decbytes",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(memory_panel)

        # CPU Usage
        cpu_panel = Panel(
            title="CPU Usage",
            panel_type="graph",
            targets=[self._create_target("mimir_cpu_usage_percent", "CPU %")],
            grid_pos=self._create_grid_pos(6, 8, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(cpu_panel)

        # Vector Similarity Scores
        similarity_panel = Panel(
            title="Vector Similarity Score Distribution",
            panel_type="heatmap",
            targets=[
                self._create_target("increase(mimir_vector_similarity_scores_bucket[5m])", "{{le}}")
            ],
            grid_pos=self._create_grid_pos(0, 16, 12, 8),
            id=self._next_panel_id(),
            options={"calculate": True, "yAxis": {"unit": "short"}},
        )
        dashboard.panels.append(similarity_panel)

        # Symbol Lookup Performance
        symbol_panel = Panel(
            title="Symbol Lookup Duration",
            panel_type="graph",
            targets=[
                self._create_target(
                    "histogram_quantile(0.95, rate(mimir_symbol_lookup_duration_seconds_bucket[5m]))",
                    "P95",
                ),
                self._create_target(
                    "histogram_quantile(0.50, rate(mimir_symbol_lookup_duration_seconds_bucket[5m]))",
                    "P50",
                ),
            ],
            grid_pos=self._create_grid_pos(0, 24, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "s",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(symbol_panel)

        # Disk Usage
        disk_panel = Panel(
            title="Disk Usage",
            panel_type="graph",
            targets=[
                self._create_target(
                    'mimir_disk_usage_bytes{type="used"} / mimir_disk_usage_bytes{type="total"} * 100',
                    "{{path}} Used %",
                )
            ],
            grid_pos=self._create_grid_pos(6, 24, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(disk_panel)

        return dashboard

    def generate_error_dashboard(self) -> Dashboard:
        """Generate error tracking and debugging dashboard."""
        dashboard = Dashboard(
            title="Mimir Error Tracking",
            description="Error rates, failure analysis, and system health monitoring",
            tags=["mimir", "errors", "debugging"],
        )

        # Overall Error Rate
        error_rate_panel = Panel(
            title="Overall Error Rate",
            panel_type="stat",
            targets=[
                self._create_target(
                    'rate(mimir_pipeline_errors_total[5m]) + rate(mimir_search_requests_total{status="error"}[5m]) + rate(mimir_mcp_requests_total{status="error"}[5m])',
                    "Error Rate",
                )
            ],
            grid_pos=self._create_grid_pos(0, 0, 4, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "background",
            },
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "thresholds": {
                        "steps": [
                            {"color": "green", "value": 0},
                            {"color": "yellow", "value": 0.1},
                            {"color": "red", "value": 1.0},
                        ]
                    },
                }
            },
        )
        dashboard.panels.append(error_rate_panel)

        # Health Status
        health_panel = Panel(
            title="Health Check Status",
            panel_type="stat",
            targets=[self._create_target("mimir_health_check_status", "{{check_type}}")],
            grid_pos=self._create_grid_pos(4, 0, 4, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "background",
            },
            field_config={
                "defaults": {
                    "unit": "short",
                    "mappings": [
                        {"options": {"0": {"text": "Unhealthy", "color": "red"}}},
                        {"options": {"1": {"text": "Healthy", "color": "green"}}},
                    ],
                }
            },
        )
        dashboard.panels.append(health_panel)

        # Uptime
        uptime_panel = Panel(
            title="System Uptime",
            panel_type="stat",
            targets=[self._create_target("mimir_uptime_seconds / 3600", "Hours")],
            grid_pos=self._create_grid_pos(8, 0, 4, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "value",
            },
            field_config={"defaults": {"unit": "h", "decimals": 1}},
        )
        dashboard.panels.append(uptime_panel)

        # Error Breakdown by Component
        error_breakdown_panel = Panel(
            title="Errors by Component",
            panel_type="graph",
            targets=[
                self._create_target(
                    "rate(mimir_pipeline_errors_total[5m])", "Pipeline - {{stage}} - {{error_type}}"
                ),
                self._create_target(
                    'rate(mimir_search_requests_total{status="error"}[5m])',
                    "Search - {{search_type}}",
                ),
                self._create_target(
                    'rate(mimir_mcp_requests_total{status="error"}[5m])', "MCP - {{method}}"
                ),
            ],
            grid_pos=self._create_grid_pos(0, 4, 12, 8),
            id=self._next_panel_id(),
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(error_breakdown_panel)

        # Pipeline Error Severity
        severity_panel = Panel(
            title="Pipeline Error Severity",
            panel_type="piechart",
            targets=[
                self._create_target(
                    "sum by(severity) (rate(mimir_pipeline_errors_total[5m]))", "{{severity}}"
                )
            ],
            grid_pos=self._create_grid_pos(0, 12, 6, 8),
            id=self._next_panel_id(),
            options={"pieType": "pie", "legend": {"displayMode": "table", "placement": "bottom"}},
        )
        dashboard.panels.append(severity_panel)

        # MCP Connection Status
        mcp_panel = Panel(
            title="MCP Active Connections",
            panel_type="graph",
            targets=[self._create_target("mimir_mcp_active_connections", "Active Connections")],
            grid_pos=self._create_grid_pos(6, 12, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "short",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(mcp_panel)

        # Cache Performance
        cache_panel = Panel(
            title="Cache Hit Rate",
            panel_type="graph",
            targets=[
                self._create_target(
                    "rate(mimir_cache_hits_total[5m]) / (rate(mimir_cache_hits_total[5m]) + rate(mimir_cache_misses_total[5m])) * 100",
                    "Hit Rate % - {{cache_type}}",
                )
            ],
            grid_pos=self._create_grid_pos(0, 20, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "percent",
                    "min": 0,
                    "max": 100,
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(cache_panel)

        # File Descriptors
        fd_panel = Panel(
            title="Open File Descriptors",
            panel_type="graph",
            targets=[self._create_target("mimir_file_descriptors_open", "Open FDs")],
            grid_pos=self._create_grid_pos(6, 20, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "short",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(fd_panel)

        return dashboard

    def generate_business_dashboard(self) -> Dashboard:
        """Generate business metrics dashboard."""
        dashboard = Dashboard(
            title="Mimir Business Metrics",
            description="Repository indexing, symbol extraction, and research query metrics",
            tags=["mimir", "business", "metrics"],
        )

        # Repositories Indexed
        repos_panel = Panel(
            title="Repositories Indexed",
            panel_type="stat",
            targets=[
                self._create_target(
                    'increase(mimir_repositories_indexed_total{status="success"}[24h])', "Last 24h"
                )
            ],
            grid_pos=self._create_grid_pos(0, 0, 3, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "value",
            },
            field_config={"defaults": {"unit": "short"}},
        )
        dashboard.panels.append(repos_panel)

        # Symbols Extracted
        symbols_panel = Panel(
            title="Symbols Extracted",
            panel_type="stat",
            targets=[
                self._create_target("increase(mimir_code_symbols_extracted_total[24h])", "Last 24h")
            ],
            grid_pos=self._create_grid_pos(3, 0, 3, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "value",
            },
            field_config={"defaults": {"unit": "short"}},
        )
        dashboard.panels.append(symbols_panel)

        # Embeddings Created
        embeddings_panel = Panel(
            title="Embeddings Created",
            panel_type="stat",
            targets=[
                self._create_target(
                    "increase(mimir_vector_embeddings_created_total[24h])", "Last 24h"
                )
            ],
            grid_pos=self._create_grid_pos(6, 0, 3, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "value",
            },
            field_config={"defaults": {"unit": "short"}},
        )
        dashboard.panels.append(embeddings_panel)

        # Research Queries
        queries_panel = Panel(
            title="Research Queries",
            panel_type="stat",
            targets=[
                self._create_target(
                    'increase(mimir_research_queries_total{success="true"}[24h])', "Last 24h"
                )
            ],
            grid_pos=self._create_grid_pos(9, 0, 3, 4),
            id=self._next_panel_id(),
            options={
                "reduceOptions": {"values": False, "calcs": ["lastNotNull"]},
                "colorMode": "value",
            },
            field_config={"defaults": {"unit": "short"}},
        )
        dashboard.panels.append(queries_panel)

        # Language Breakdown
        language_panel = Panel(
            title="Repositories by Language",
            panel_type="piechart",
            targets=[
                self._create_target(
                    'sum by(language) (increase(mimir_repositories_indexed_total{status="success"}[7d]))',
                    "{{language}}",
                )
            ],
            grid_pos=self._create_grid_pos(0, 4, 6, 8),
            id=self._next_panel_id(),
            options={"pieType": "pie", "legend": {"displayMode": "table", "placement": "bottom"}},
        )
        dashboard.panels.append(language_panel)

        # Symbol Types
        symbol_types_panel = Panel(
            title="Symbol Types Extracted",
            panel_type="graph",
            targets=[
                self._create_target(
                    "rate(mimir_code_symbols_extracted_total[5m])", "{{symbol_type}} - {{language}}"
                )
            ],
            grid_pos=self._create_grid_pos(6, 4, 6, 8),
            id=self._next_panel_id(),
            options={"legend": {"displayMode": "table", "placement": "bottom"}},
            field_config={
                "defaults": {
                    "unit": "ops",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(symbol_types_panel)

        # Query Complexity
        complexity_panel = Panel(
            title="Research Query Complexity",
            panel_type="graph",
            targets=[
                self._create_target(
                    "rate(mimir_research_queries_total[5m])", "{{complexity}} - {{success}}"
                )
            ],
            grid_pos=self._create_grid_pos(0, 12, 6, 8),
            id=self._next_panel_id(),
            field_config={
                "defaults": {
                    "unit": "reqps",
                    "custom": {"drawStyle": "line", "lineInterpolation": "linear"},
                }
            },
        )
        dashboard.panels.append(complexity_panel)

        # Search Results Distribution
        results_panel = Panel(
            title="Search Results Count Distribution",
            panel_type="heatmap",
            targets=[
                self._create_target("increase(mimir_search_results_count_bucket[5m])", "{{le}}")
            ],
            grid_pos=self._create_grid_pos(6, 12, 6, 8),
            id=self._next_panel_id(),
            options={"calculate": True, "yAxis": {"unit": "short"}},
        )
        dashboard.panels.append(results_panel)

        return dashboard

    def export_dashboard(self, dashboard: Dashboard, output_path: Path) -> None:
        """Export dashboard to JSON file."""
        dashboard_json = {
            "dashboard": {
                "id": None,
                "title": dashboard.title,
                "description": dashboard.description,
                "tags": dashboard.tags,
                "timezone": "browser",
                "panels": [self._panel_to_dict(panel) for panel in dashboard.panels],
                "time": dashboard.time_range,
                "timepicker": {},
                "templating": {"list": dashboard.variables},
                "annotations": {"list": []},
                "refresh": dashboard.refresh,
                "schemaVersion": 30,
                "style": "dark",
                "version": 1,
                "weekStart": "",
            },
            "overwrite": True,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dashboard_json, f, indent=2)

        logger.info(f"Dashboard exported to {output_path}")

    def _panel_to_dict(self, panel: Panel) -> dict[str, Any]:
        """Convert panel to dictionary format."""
        return {
            "id": panel.id,
            "title": panel.title,
            "type": panel.panel_type,
            "targets": panel.targets,
            "gridPos": panel.grid_pos,
            "options": panel.options,
            "fieldConfig": panel.field_config,
            "datasource": {"type": "prometheus", "uid": self.datasource_name},
        }


# =========================
# CONVENIENCE FUNCTIONS
# =========================


def generate_pipeline_dashboard(output_path: Path | None = None) -> Dashboard:
    """Generate and optionally export pipeline dashboard."""
    generator = DashboardGenerator()
    dashboard = generator.generate_pipeline_dashboard()

    if output_path:
        generator.export_dashboard(dashboard, output_path)

    return dashboard


def generate_performance_dashboard(output_path: Path | None = None) -> Dashboard:
    """Generate and optionally export performance dashboard."""
    generator = DashboardGenerator()
    dashboard = generator.generate_performance_dashboard()

    if output_path:
        generator.export_dashboard(dashboard, output_path)

    return dashboard


def generate_error_dashboard(output_path: Path | None = None) -> Dashboard:
    """Generate and optionally export error dashboard."""
    generator = DashboardGenerator()
    dashboard = generator.generate_error_dashboard()

    if output_path:
        generator.export_dashboard(dashboard, output_path)

    return dashboard


def generate_all_dashboards(output_dir: Path) -> list[Dashboard]:
    """Generate all dashboards and export them."""
    generator = DashboardGenerator()
    dashboards = []

    # Pipeline dashboard
    pipeline_dashboard = generator.generate_pipeline_dashboard()
    generator.export_dashboard(pipeline_dashboard, output_dir / "pipeline-dashboard.json")
    dashboards.append(pipeline_dashboard)

    # Performance dashboard
    performance_dashboard = generator.generate_performance_dashboard()
    generator.export_dashboard(performance_dashboard, output_dir / "performance-dashboard.json")
    dashboards.append(performance_dashboard)

    # Error dashboard
    error_dashboard = generator.generate_error_dashboard()
    generator.export_dashboard(error_dashboard, output_dir / "error-dashboard.json")
    dashboards.append(error_dashboard)

    # Business dashboard
    business_dashboard = generator.generate_business_dashboard()
    generator.export_dashboard(business_dashboard, output_dir / "business-dashboard.json")
    dashboards.append(business_dashboard)

    logger.info(f"Generated {len(dashboards)} dashboards in {output_dir}")
    return dashboards
