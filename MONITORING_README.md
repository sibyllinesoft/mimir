# Mimir Deep Research Monitoring

This document describes how to use Mimir's advanced monitoring capabilities powered by **Skald** and **NATS JetStream** for real-time visibility into deep code research operations.

## 🎯 Overview

The monitoring system provides:

- **🔍 Real-time trace emission** - Every tool execution is traced with detailed metadata
- **🧠 Deep research analysis** - Question complexity analysis, evidence tracking, confidence scoring
- **📊 Agent coordination patterns** - Visibility into how Mimir's agents work together
- **⚡ Performance insights** - Execution timing, bottleneck identification, optimization opportunities
- **🚨 Error tracking** - Comprehensive error capture with context and stack traces
- **📈 Session analytics** - Research session flows and pattern recognition

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install monitoring dependencies
python scripts/setup_monitoring.py --install-deps

# Or manually:
pip install sibylline-skald>=0.2.0 nats-py>=2.10.0
```

### 2. Start Monitoring Infrastructure

```bash
# Start NATS JetStream + monitored server
python scripts/setup_monitoring.py --start-stack

# Or using Docker Compose directly:
docker-compose -f docker-compose.monitoring.yml up -d
```

### 3. Use Monitored Server

The monitored server is a drop-in replacement for the standard server:

```bash
# Instead of:
mimir-server

# Use:
mimir-monitored-server
```

### 4. View Live Traces

```bash
# View traces in real-time
docker logs -f mimir-trace-viewer

# Or run trace viewer locally:
python scripts/trace_viewer.py
```

## 📋 Configuration

The monitoring system is configured via `monitoring_config.yaml`:

```yaml
# Enable/disable monitoring features
monitoring:
  enabled: true
  session_tracking: true
  deep_research_analysis: true

# NATS JetStream configuration
nats:
  url: "nats://localhost:4222"
  stream_name: "MIMIR_TRACES"
  enabled: true

# Skald monitoring settings
skald:
  feedback_strength: "detailed"
  capture_output: true
  context: "mimir_deep_research"
```

## 🔧 Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   Claude Code   │───▶│    Mimir     │───▶│ NATS JetStream  │
│                 │    │  Monitored   │    │                 │
└─────────────────┘    │   Server     │    └─────────────────┘
                       │              │             │
                       │  + Skald     │             ▼
                       │  Monitoring  │    ┌─────────────────┐
                       └──────────────┘    │  Trace Viewer   │
                                          │                 │
                                          └─────────────────┘
```

### Components

1. **Monitored MCP Server** (`MonitoredMCPServer`)
   - Wraps all tool executions with Skald monitoring
   - Emits detailed traces to NATS JetStream
   - Maintains session state and research analytics

2. **NATS JetStream**
   - High-performance message streaming
   - Persistent trace storage (configurable retention)
   - Real-time trace distribution to viewers

3. **Trace Viewer**
   - Real-time trace display with color coding
   - Performance metrics and error highlighting
   - Research session tracking

4. **Skald Integration**
   - Universal execution monitoring
   - Structured trace data collection
   - Performance and error analytics

## 📊 Trace Types

### Tool Execution Traces

Every MCP tool call generates comprehensive traces:

```json
{
  "call_id": "uuid",
  "session_id": "uuid", 
  "tool_name": "search_repo",
  "trace_type": "tool_execution_start",
  "timestamp": 1699123456.789,
  "operation": "repo_search",
  "arguments": {
    "query": "authentication flow",
    "k": 20
  }
}
```

### Deep Research Traces

`ask_index` operations include intelligence analysis:

```json
{
  "operation": "deep_research",
  "question": "How is user authentication implemented?",
  "question_complexity": "medium",
  "evidence_count": 5,
  "answer_confidence": "high",
  "research_depth": 5
}
```

### Performance Traces

All operations include timing and performance data:

```json
{
  "execution_time": 2.456,
  "trace_type": "tool_execution_complete",
  "success": true,
  "result_length": 1024
}
```

## 🎛️ Monitoring Features

### Research Session Tracking

The system tracks research sessions across multiple operations:

- **Session lifecycle** - Index creation to completion
- **Operation sequences** - Search → ask → refine patterns
- **Knowledge building** - How queries build on each other
- **Complexity evolution** - How questions become more sophisticated

### Question Complexity Analysis

Deep research questions are automatically analyzed for complexity:

- **Low complexity**: Simple factual queries ("what", "where", "when")
- **Medium complexity**: Explanatory queries ("how", "explain", "design")  
- **High complexity**: Analytical queries ("why", "analyze", "compare", "architecture")

### Performance Monitoring

- **Execution timing** - Per-tool and end-to-end timing
- **Bottleneck identification** - Slow operations highlighted
- **Resource utilization** - Memory and CPU tracking
- **Throughput analysis** - Operations per second

### Error Intelligence

- **Comprehensive error capture** - Full stack traces and context
- **Error categorization** - By type, frequency, and impact
- **Recovery tracking** - How errors are resolved
- **Pattern recognition** - Common failure modes

## 🔍 Using the Trace Viewer

The trace viewer provides real-time monitoring with color-coded output:

```bash
🚀 12:34:56.789 SEARCH_REPO          tool_execution_start
  🔍 Query: 'authentication flow implementation' (35 chars)
  📊 Results: 15/20 requested
  🎛️  Features: vector, symbol, graph
  ⏱️  Execution time: 1.234s

✅ 12:34:58.123 SEARCH_REPO          tool_execution_complete
```

### Color Coding

- 🚀 **Blue** - Operation start
- ✅ **Green** - Successful completion  
- ❌ **Red** - Errors and failures
- ⚠️ **Yellow** - Warnings and slow operations
- 📊 **Cyan** - Information and metrics

## 📈 Monitoring Dashboard (Future)

Planned dashboard features:

- **Real-time metrics** - Active sessions, throughput, errors
- **Research intelligence** - Question complexity trends, topic analysis
- **Performance analytics** - Response time distributions, bottlenecks
- **Agent coordination** - How different research modes work together
- **Knowledge graphs** - Visual representation of research sessions

## 🛠️ Development

### Running Tests

```bash
# Test the monitored server
python -m pytest tests/integration/test_monitored_mcp.py

# Test NATS integration
python -m pytest tests/integration/test_nats_traces.py
```

### Adding Custom Traces

You can add custom trace points using Skald decorators:

```python
from skald.monitor import trace_function

@trace_function(context="custom_research", capture_output=True)
async def my_research_function(query: str) -> str:
    # Your research logic here
    return result
```

### Extending Trace Analysis

Add custom trace analysis in `MonitoredMCPServer`:

```python
def _analyze_custom_pattern(self, trace_data: dict) -> dict:
    """Add custom analysis to traces."""
    # Your analysis logic
    return enhanced_trace_data
```

## 🚨 Troubleshooting

### NATS Connection Issues

```bash
# Check NATS is running
docker ps | grep nats

# Check NATS health
curl http://localhost:8222/varz

# View NATS logs
docker logs mimir-nats
```

### Trace Viewer Not Showing Data

```bash
# Check if traces are being emitted
curl http://localhost:8222/streaming/channelsz

# Check trace viewer logs
docker logs mimir-trace-viewer

# Manually test trace emission
python -c "import asyncio; from scripts.trace_viewer import MimirTraceViewer; asyncio.run(MimirTraceViewer().connect())"
```

### Performance Issues

- **High trace volume**: Adjust `monitoring_config.yaml` to reduce trace frequency
- **NATS memory usage**: Configure stream limits in Docker Compose
- **Slow operations**: Use trace viewer to identify bottlenecks

## 📚 Related Documentation

- [Skald Documentation](https://github.com/sibyllinesoft/skald)
- [NATS JetStream Documentation](https://docs.nats.io/jetstream)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Mimir Architecture](./ARCHITECTURE.md)

## 🎯 Next Steps

1. **Start monitoring** your Mimir usage with the trace viewer
2. **Analyze patterns** in your research sessions  
3. **Optimize queries** based on performance insights
4. **Build custom dashboards** for your specific needs
5. **Contribute improvements** to the monitoring system

Happy deep research monitoring! 🚀🔍