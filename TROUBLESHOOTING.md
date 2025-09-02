# Mimir Troubleshooting Guide

## Overview

This guide provides systematic troubleshooting procedures for common issues with the Mimir Deep Code Research System. Issues are organized by component and include step-by-step resolution procedures.

## Quick Diagnostics

### Health Check Commands

```bash
# Basic system health
./scripts/health-check.sh

# Detailed health with metrics
./scripts/health-check.sh -c detailed -f json

# Check MCP server status
echo '{"jsonrpc":"2.0","method":"resources/list","id":1}' | python -m repoindex.main mcp

# Verify external tools
repomapper --version
serena --version
python -c "import leann; print('LEANN installed')"
```

### System Information Collection

```bash
# Environment information
python --version
uv --version
git --version
docker --version

# Mimir-specific information
python -c "
import sys, platform
from pathlib import Path
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Home: {Path.home()}')
print(f'Cache: {Path.home() / \".cache\" / \"mimir\"}')
"

# Resource availability
df -h
free -h
ps aux | grep -E "(python|mimir)" | head -10
```

## Installation Issues

### Python Environment Problems

#### Issue: "Python version not supported"
```bash
# Check Python version
python --version

# If < 3.11, install newer Python
# Ubuntu/Debian
sudo apt update && sudo apt install python3.11 python3.11-venv

# macOS with Homebrew
brew install python@3.11

# Create new environment with correct Python
uv venv --python 3.11
```

#### Issue: "uv command not found"
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip
pip install uv

# Reload shell
source ~/.bashrc  # or ~/.zshrc
```

#### Issue: "Virtual environment not activating"
```bash
# Remove existing environment
rm -rf .venv

# Create new environment
uv venv

# Activate manually
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows

# Verify activation
which python
python --version
```

### Dependency Installation Issues

#### Issue: "Failed to install dependencies"
```bash
# Clear uv cache
uv cache clean

# Update uv
uv self update

# Install with verbose output
uv sync -v

# Install specific failing packages manually
uv add package-name

# Check for dependency conflicts
uv tree
```

#### Issue: "External tools not found"
```bash
# Install RepoMapper
pip install repomapper
repomapper --version

# Install Serena (requires Node.js)
# First install Node.js if needed
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Then install Serena
npm install -g @context-labs/serena
serena --version

# Install LEANN
pip install leann
python -c "import leann; print('LEANN available')"

# Verify PATH
echo $PATH
which repomapper serena python
```

## MCP Server Issues

### Server Startup Problems

#### Issue: "Server fails to start"
```bash
# Check for port conflicts
netstat -tlnp | grep :8000

# Start with debug logging
MIMIR_LOG_LEVEL=DEBUG python -m repoindex.main mcp

# Check for permission issues
ls -la ~/.cache/mimir/
mkdir -p ~/.cache/mimir/indexes
chmod 755 ~/.cache/mimir/indexes

# Start with minimal configuration
MIMIR_CONCURRENCY_IO=1 MIMIR_CONCURRENCY_CPU=1 python -m repoindex.main mcp
```

#### Issue: "MCP protocol errors"
```bash
# Test MCP communication manually
cat > test_mcp.json << EOF
{"jsonrpc":"2.0","method":"tools/list","id":1}
EOF

cat test_mcp.json | python -m repoindex.main mcp

# Validate JSON-RPC format
python -c "
import json
with open('test_mcp.json') as f:
    data = json.load(f)
    print('Valid JSON:', data)
"

# Check MCP client compatibility
python -c "
from mcp import __version__
print(f'MCP version: {__version__}')
"
```

#### Issue: "Server crashes during operation"
```bash
# Check system resources
free -h
df -h /tmp ~/.cache

# Monitor memory usage
ps aux --sort=-%mem | head -10

# Check for memory leaks
valgrind --tool=memcheck python -m repoindex.main mcp

# Enable core dumps for debugging
ulimit -c unlimited
python -m repoindex.main mcp
```

### Tool Execution Failures

#### Issue: "ensure_repo_index fails"
```bash
# Verify repository path
ls -la /path/to/repository
git -C /path/to/repository status

# Check git repository status
cd /path/to/repository
git rev-parse --show-toplevel
git rev-parse HEAD

# Test with minimal options
python -c "
import asyncio
from repoindex.mcp.server import MCPServer

async def test():
    server = MCPServer()
    result = await server._ensure_repo_index({
        'path': '/path/to/small/repo',
        'language': 'js'
    })
    print(result)

asyncio.run(test())
"

# Check external tool execution
cd /path/to/repository
repomapper --help
serena analyze . --help
```

#### Issue: "search_repo returns no results"
```bash
# Check index status
python -c "
import json
from pathlib import Path

index_dir = Path('~/.cache/mimir/indexes').expanduser()
for idx in index_dir.iterdir():
    if idx.is_dir():
        manifest = idx / 'manifest.json'
        if manifest.exists():
            data = json.loads(manifest.read_text())
            print(f'Index {idx.name}:')
            print(f'  Files: {data[\"counts\"][\"files_indexed\"]}')
            print(f'  Symbols: {data[\"counts\"][\"symbols_defs\"]}')
            print(f'  Vectors: {data[\"counts\"][\"vectors\"]}')
"

# Test individual search modalities
python -c "
import asyncio
from repoindex.pipeline.hybrid_search import HybridSearchEngine

async def test_search():
    engine = HybridSearchEngine(storage_dir=Path('~/.cache/mimir').expanduser())
    
    # Test vector search
    vector_results = await engine._vector_search('test query', k=5)
    print(f'Vector results: {len(vector_results)}')
    
    # Test symbol search  
    symbol_results = await engine._symbol_search('test')
    print(f'Symbol results: {len(symbol_results)}')

asyncio.run(test_search())
"

# Verify search artifacts exist
ls -la ~/.cache/mimir/indexes/*/leann.index
ls -la ~/.cache/mimir/indexes/*/serena_graph.jsonl
```

## Pipeline Execution Issues

### Stage-Specific Failures

#### Issue: "Acquire stage fails"
```bash
# Check git repository
cd /path/to/repository
git status
git ls-files | head -10

# Test file discovery manually
python -c "
from repoindex.pipeline.discover import GitDiscovery
discovery = GitDiscovery()
files = discovery.discover_files('/path/to/repository')
print(f'Discovered files: {len(files)}')
print('First 10 files:', files[:10])
"

# Check for large files
find /path/to/repository -type f -size +100M

# Check permissions
find /path/to/repository -type f ! -readable
```

#### Issue: "RepoMapper stage fails"
```bash
# Test RepoMapper directly
cd /path/to/repository
repomapper . --output repomapper_output.json

# Check output
ls -la repomapper_output.json
head -20 repomapper_output.json

# Verify file types are supported
find . -name "*.ts" -o -name "*.js" -o -name "*.py" | head -10

# Check for syntax errors
python -c "
import json
with open('repomapper_output.json') as f:
    data = json.load(f)
    print(f'RepoMapper found {len(data.get(\"files\", []))} files')
"
```

#### Issue: "Serena stage fails"
```bash
# Test Serena directly
cd /path/to/repository
serena analyze . --output serena_output.jsonl

# Check TypeScript configuration
ls -la tsconfig.json package.json
cat tsconfig.json

# Test with specific files
serena analyze src/ --include "*.ts,*.tsx"

# Check Serena output
head -10 serena_output.jsonl
wc -l serena_output.jsonl

# Verify JSON Lines format
python -c "
import json
with open('serena_output.jsonl') as f:
    for i, line in enumerate(f):
        try:
            data = json.loads(line)
            print(f'Line {i}: {data.get(\"type\", \"unknown\")}')
        except json.JSONDecodeError as e:
            print(f'Invalid JSON on line {i}: {e}')
        if i >= 10:
            break
"
```

#### Issue: "LEANN stage fails"
```bash
# Test LEANN directly
python -c "
import leann
import numpy as np

# Test basic functionality
texts = ['hello world', 'test document']
embeddings = leann.embed(texts)
print(f'Generated embeddings: {embeddings.shape}')

# Test index creation
index = leann.create_index(embeddings)
print(f'Index created: {type(index)}')

# Test search
query_embedding = leann.embed(['hello'])
results = leann.search(index, query_embedding, k=1)
print(f'Search results: {results}')
"

# Check available memory
free -m

# Test with smaller dataset
python -c "
from repoindex.pipeline.leann import LEANNAdapter
import asyncio

async def test_leann():
    adapter = LEANNAdapter()
    test_files = ['/path/to/small/file.py']
    
    try:
        result = await adapter.build_index(test_files, test_files)
        print(f'LEANN index built successfully: {result}')
    except Exception as e:
        print(f'LEANN error: {e}')

asyncio.run(test_leann())
"
```

#### Issue: "Bundle stage fails"
```bash
# Check disk space
df -h ~/.cache/mimir/

# Test compression manually
cd ~/.cache/mimir/indexes/your_index_id/
tar -czf test_bundle.tar.gz *.json *.jsonl *.index *.bin

# Check for file corruption
python -c "
import json
from pathlib import Path

index_dir = Path('~/.cache/mimir/indexes/your_index_id').expanduser()

# Check each artifact
for artifact in ['manifest.json', 'repomap.json', 'serena_graph.jsonl']:
    file_path = index_dir / artifact
    if file_path.exists():
        try:
            if artifact.endswith('.json'):
                data = json.loads(file_path.read_text())
                print(f'{artifact}: Valid JSON, {len(str(data))} chars')
            else:
                lines = file_path.read_text().split('\n')
                print(f'{artifact}: {len(lines)} lines')
        except Exception as e:
            print(f'{artifact}: Error - {e}')
    else:
        print(f'{artifact}: Missing')
"

# Test zstandard compression
python -c "
import zstandard as zstd
compressor = zstd.ZstdCompressor()
test_data = b'test data' * 1000
compressed = compressor.compress(test_data)
print(f'Compression test: {len(test_data)} -> {len(compressed)} bytes')
"
```

### Performance Issues

#### Issue: "Pipeline execution too slow"
```bash
# Monitor resource usage during execution
top -p $(pgrep -f "repoindex")

# Check I/O wait
iostat -x 1

# Profile pipeline execution
python -m cProfile -o pipeline_profile.prof -c "
import asyncio
from repoindex.pipeline.run import IndexingPipeline

async def profile_pipeline():
    pipeline = IndexingPipeline()
    await pipeline.start_indexing('/path/to/repo')

asyncio.run(profile_pipeline())
"

# Analyze profile
python -c "
import pstats
stats = pstats.Stats('pipeline_profile.prof')
stats.sort_stats('cumulative').print_stats(20)
"

# Check for memory issues
python -c "
import tracemalloc
tracemalloc.start()

# Run your pipeline code here

current, peak = tracemalloc.get_traced_memory()
print(f'Current memory: {current / 1024 / 1024:.1f} MB')
print(f'Peak memory: {peak / 1024 / 1024:.1f} MB')
"
```

#### Issue: "Search queries too slow"
```bash
# Benchmark search operations
python -c "
import asyncio
import time
from repoindex.pipeline.hybrid_search import HybridSearchEngine

async def benchmark_search():
    engine = HybridSearchEngine()
    queries = ['test query', 'function', 'class definition']
    
    for query in queries:
        start = time.perf_counter()
        results = await engine.search(query, k=10)
        duration = time.perf_counter() - start
        print(f'Query \"{query}\": {duration:.2f}s, {len(results)} results')

asyncio.run(benchmark_search())
"

# Check index sizes
ls -lh ~/.cache/mimir/indexes/*/leann.index
ls -lh ~/.cache/mimir/indexes/*/vectors.bin

# Monitor search performance
python -c "
from repoindex.monitoring.metrics import get_metrics_collector
collector = get_metrics_collector()

# Enable metrics collection
collector.start()

# Run your search operations

# View metrics
metrics = collector.get_metrics()
for name, value in metrics.items():
    print(f'{name}: {value}')
"
```

#### Issue: "Memory consumption too high"
```bash
# Check memory usage patterns
python -m memory_profiler -m repoindex.main mcp

# Monitor memory growth
python -c "
import psutil
import time
import os

process = psutil.Process(os.getpid())
for i in range(60):  # Monitor for 1 minute
    memory = process.memory_info().rss / 1024 / 1024
    print(f'Memory: {memory:.1f} MB')
    time.sleep(1)
"

# Check for memory leaks
python -c "
import gc
import tracemalloc

tracemalloc.start()

# Run your code that might leak memory

# Force garbage collection
gc.collect()

# Check for unreferenced objects
current, peak = tracemalloc.get_traced_memory()
print(f'Current: {current / 1024 / 1024:.1f} MB')
print(f'Peak: {peak / 1024 / 1024:.1f} MB')

# Show top memory consumers
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)
"
```

## Docker and Deployment Issues

### Container Problems

#### Issue: "Docker container fails to start"
```bash
# Check container logs
docker-compose logs mimir-server

# Check resource limits
docker stats

# Test container build
docker build -t mimir-test .

# Run container interactively
docker run -it mimir-test /bin/bash

# Check port conflicts
netstat -tlnp | grep :8000
docker-compose ps

# Verify Docker Compose configuration
docker-compose config
```

#### Issue: "Container out of memory"
```bash
# Check memory limits
docker inspect mimir-server | grep -i memory

# Monitor container memory usage
docker stats mimir-server

# Increase memory limit in docker-compose.yml
cat >> docker-compose.override.yml << EOF
version: '3.8'
services:
  mimir-server:
    mem_limit: 4g
    memswap_limit: 4g
EOF

# Restart with new limits
docker-compose down
docker-compose up -d
```

#### Issue: "Volume mount issues"
```bash
# Check volume permissions
ls -la data/ cache/ logs/

# Fix permissions
sudo chown -R $(id -u):$(id -g) data/ cache/ logs/
chmod -R 755 data/ cache/ logs/

# Test volume mounts
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/cache:/app/cache mimir-test ls -la /app/

# Check disk space
df -h
```

### Monitoring Stack Issues

#### Issue: "Prometheus not collecting metrics"
```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:8000/metrics

# Check Prometheus configuration
docker exec prometheus cat /etc/prometheus/prometheus.yml

# Test metric collection
curl -s http://localhost:8000/metrics | grep mimir_

# Check Prometheus logs
docker-compose logs prometheus
```

#### Issue: "Grafana dashboards not loading"
```bash
# Check Grafana logs
docker-compose logs grafana

# Verify data source connection
curl -u admin:admin http://localhost:3000/api/datasources

# Test dashboard API
curl -u admin:admin http://localhost:3000/api/dashboards/home

# Check dashboard files
ls -la ops/grafana/dashboards/
```

#### Issue: "Log aggregation not working"
```bash
# Check Loki logs
docker-compose logs loki

# Test log ingestion
curl -X POST http://localhost:3100/loki/api/v1/push \
  -H "Content-Type: application/json" \
  -d '{"streams": [{"stream": {"job": "test"}, "values": [["'$(date +%s%N)'", "test log message"]]}]}'

# Check Promtail configuration
docker exec promtail cat /etc/promtail/config.yml

# Verify log file permissions
ls -la logs/
```

## Security Issues

### Authentication Problems

#### Issue: "API key authentication failing"
```bash
# Check API key configuration
echo $MIMIR_API_KEY

# Generate new API key
python -c "
import secrets
api_key = secrets.token_urlsafe(32)
print(f'New API key: {api_key}')
"

# Test authentication
curl -H "Authorization: Bearer $MIMIR_API_KEY" http://localhost:8000/health

# Check authentication logs
grep -i "auth" ~/.cache/mimir/security/audit.log
```

#### Issue: "Rate limiting too aggressive"
```bash
# Check rate limit configuration
grep -i rate ~/.cache/mimir/security/security_config.json

# View rate limit violations
grep "RATE_LIMIT_EXCEEDED" ~/.cache/mimir/security/audit.log

# Adjust rate limits
python -c "
import json
from pathlib import Path

config_path = Path('~/.cache/mimir/security/security_config.json').expanduser()
config = json.loads(config_path.read_text())

# Increase limits
config['global_rate_limit'] = 2000
config['ip_rate_limit'] = 200

config_path.write_text(json.dumps(config, indent=2))
print('Rate limits increased')
"

# Restart server to apply changes
```

#### Issue: "Sandboxing prevents tool execution"
```bash
# Check sandbox configuration
grep -i sandbox ~/.cache/mimir/security/security_config.json

# View sandbox violations
grep "SANDBOX_VIOLATION" ~/.cache/mimir/security/audit.log

# Test without sandboxing (development only)
MIMIR_ENABLE_SANDBOXING=false python -m repoindex.main_secure mcp

# Adjust sandbox limits
python -c "
import json
from pathlib import Path

config_path = Path('~/.cache/mimir/security/security_config.json').expanduser()
config = json.loads(config_path.read_text())

# Increase resource limits
config['max_memory_mb'] = 2048
config['max_cpu_time_seconds'] = 600

config_path.write_text(json.dumps(config, indent=2))
print('Sandbox limits increased')
"
```

### Encryption Issues

#### Issue: "Index encryption failing"
```bash
# Check encryption key
echo $MIMIR_MASTER_KEY | base64 -d | wc -c

# Generate new encryption key
python -c "
import secrets
import base64
key = secrets.token_bytes(32)
encoded_key = base64.b64encode(key).decode()
print(f'New master key: {encoded_key}')
"

# Test encryption/decryption
python -c "
from repoindex.security.crypto import CryptoManager
crypto = CryptoManager()

test_data = b'test encryption data'
encrypted = crypto.encrypt(test_data)
decrypted = crypto.decrypt(encrypted)

print(f'Original: {test_data}')
print(f'Encrypted: {encrypted[:50]}...')
print(f'Decrypted: {decrypted}')
print(f'Success: {test_data == decrypted}')
"

# Check encryption logs
grep -i encrypt ~/.cache/mimir/security/audit.log
```

## Data Corruption Issues

### Index Corruption

#### Issue: "Index appears corrupted"
```bash
# Verify index integrity
python -c "
import json
import tarfile
from pathlib import Path

index_id = 'your_index_id'
index_dir = Path(f'~/.cache/mimir/indexes/{index_id}').expanduser()

# Check manifest
manifest_path = index_dir / 'manifest.json'
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())
        print('Manifest: Valid')
        print(f'Files indexed: {manifest[\"counts\"][\"files_indexed\"]}')
    except Exception as e:
        print(f'Manifest corrupted: {e}')

# Check bundle
bundle_path = index_dir / 'bundle.tar.zst'
if bundle_path.exists():
    try:
        with tarfile.open(bundle_path, 'r:*') as tar:
            members = tar.getnames()
            print(f'Bundle: Valid, {len(members)} files')
    except Exception as e:
        print(f'Bundle corrupted: {e}')

# Check individual artifacts
for artifact in ['repomap.json', 'serena_graph.jsonl', 'leann.index']:
    artifact_path = index_dir / artifact
    if artifact_path.exists():
        try:
            size = artifact_path.stat().st_size
            print(f'{artifact}: {size} bytes')
        except Exception as e:
            print(f'{artifact}: Error - {e}')
"

# Rebuild corrupted index
python -c "
import asyncio
from repoindex.mcp.server import MCPServer

async def rebuild_index():
    server = MCPServer()
    
    # Cancel existing pipeline
    await server._cancel({'index_id': 'your_index_id'})
    
    # Start fresh indexing
    result = await server._ensure_repo_index({
        'path': '/path/to/repository',
        'language': 'ts'
    })
    
    print(f'Rebuilding index: {result}')

asyncio.run(rebuild_index())
"
```

#### Issue: "Vector index corrupted"
```bash
# Check LEANN index file
python -c "
import numpy as np
from pathlib import Path

index_path = Path('~/.cache/mimir/indexes/your_index_id/leann.index').expanduser()
vectors_path = Path('~/.cache/mimir/indexes/your_index_id/vectors.bin').expanduser()

try:
    # Check if files exist and are readable
    if index_path.exists():
        size = index_path.stat().st_size
        print(f'LEANN index: {size} bytes')
    
    if vectors_path.exists():
        vectors = np.fromfile(vectors_path, dtype=np.float32)
        print(f'Vectors: {len(vectors)} elements')
        
        # Check for NaN or infinite values
        if np.any(np.isnan(vectors)):
            print('WARNING: NaN values found in vectors')
        if np.any(np.isinf(vectors)):
            print('WARNING: Infinite values found in vectors')
    
except Exception as e:
    print(f'Vector index error: {e}')
"

# Rebuild vector index only
python -c "
import asyncio
from repoindex.pipeline.leann import LEANNAdapter

async def rebuild_vectors():
    adapter = LEANNAdapter()
    
    # Get file list from manifest
    import json
    from pathlib import Path
    
    manifest_path = Path('~/.cache/mimir/indexes/your_index_id/manifest.json').expanduser()
    manifest = json.loads(manifest_path.read_text())
    
    # Rebuild LEANN index
    files = [...]  # Extract from manifest
    result = await adapter.build_index(files, files)
    print(f'Vector index rebuilt: {result}')

asyncio.run(rebuild_vectors())
"
```

### Repository State Issues

#### Issue: "Repository state inconsistent"
```bash
# Check repository status
cd /path/to/repository
git status --porcelain
git rev-parse HEAD

# Check for uncommitted changes
git diff --name-only
git diff --cached --name-only

# Clean working directory (if safe)
git clean -fd
git checkout -- .

# Verify git integrity
git fsck --full

# Re-index with clean state
python -c "
import asyncio
from repoindex.mcp.server import MCPServer

async def reindex_clean():
    server = MCPServer()
    result = await server._ensure_repo_index({
        'path': '/path/to/repository',
        'rev': 'HEAD',  # Force specific revision
        'language': 'ts'
    })
    print(result)

asyncio.run(reindex_clean())
"
```

## Network and Connectivity Issues

### External Service Problems

#### Issue: "Cannot reach external services"
```bash
# Test network connectivity
ping -c 3 github.com
curl -I https://api.github.com

# Check DNS resolution
nslookup github.com
dig github.com

# Test with proxy if needed
export http_proxy=http://proxy.company.com:8080
export https_proxy=http://proxy.company.com:8080

# Test external tool downloads
curl -L https://github.com/tree-sitter/tree-sitter/releases/latest

# Check firewall rules
sudo ufw status
iptables -L INPUT -n
```

#### Issue: "Slow external tool execution"
```bash
# Test tool performance individually
time repomapper /path/to/repository
time serena analyze /path/to/repository

# Check network latency
ping -c 10 registry.npmjs.org
traceroute registry.npmjs.org

# Monitor network usage during indexing
iftop -i eth0

# Use local mirrors if available
npm config set registry https://local.mirror.com/npm/
```

## Advanced Debugging

### Memory Debugging

```bash
# Run with memory debugging
valgrind --tool=memcheck --leak-check=full python -m repoindex.main mcp

# Use memory profiler
python -m memory_profiler -m repoindex.main mcp

# Track object creation
python -c "
import gc
import sys
from collections import defaultdict

def track_objects():
    objects = defaultdict(int)
    for obj in gc.get_objects():
        objects[type(obj).__name__] += 1
    
    return dict(sorted(objects.items(), key=lambda x: x[1], reverse=True)[:20])

print('Object counts:')
for obj_type, count in track_objects().items():
    print(f'{obj_type}: {count}')
"
```

### Concurrency Debugging

```bash
# Monitor async tasks
python -c "
import asyncio

async def monitor_tasks():
    while True:
        tasks = asyncio.all_tasks()
        print(f'Active tasks: {len(tasks)}')
        
        for task in tasks:
            if not task.done():
                print(f'  - {task.get_name()}: {task.get_stack()[-1] if task.get_stack() else \"No stack\"}')
        
        await asyncio.sleep(5)

asyncio.run(monitor_tasks())
"

# Check for deadlocks
python -c "
import threading
import time

def check_deadlocks():
    while True:
        for thread in threading.enumerate():
            if thread != threading.current_thread():
                frame = sys._current_frames()[thread.ident]
                print(f'Thread {thread.name}: {frame.f_code.co_filename}:{frame.f_lineno}')
        
        time.sleep(10)

check_deadlocks()
"
```

### Performance Profiling

```bash
# CPU profiling
python -m cProfile -o cpu_profile.prof -m repoindex.main mcp

# Analyze CPU profile
python -c "
import pstats
stats = pstats.Stats('cpu_profile.prof')
stats.sort_stats('cumulative').print_stats(30)
stats.sort_stats('time').print_stats(30)
"

# I/O profiling
strace -f -e trace=read,write,openat -o io_trace.log python -m repoindex.main mcp

# Analyze I/O trace
grep -E "(read|write)" io_trace.log | head -20

# Network profiling
sudo tcpdump -i any -w network_trace.pcap host github.com
```

## Getting Help

### Information to Include

When reporting issues, include:

1. **System Information**
   ```bash
   ./scripts/health-check.sh -c detailed > system_info.txt
   ```

2. **Error Logs**
   ```bash
   # MCP server logs
   MIMIR_LOG_LEVEL=DEBUG python -m repoindex.main mcp 2>&1 | tee mcp_debug.log
   
   # Security logs (if enabled)
   cat ~/.cache/mimir/security/audit.log
   ```

3. **Configuration**
   ```bash
   env | grep MIMIR_
   cat .env
   ```

4. **Index Information**
   ```bash
   ls -la ~/.cache/mimir/indexes/
   cat ~/.cache/mimir/indexes/*/manifest.json
   ```

### Creating Minimal Reproduction

```python
"""Minimal reproduction script for issue reporting."""

import asyncio
import tempfile
from pathlib import Path
from repoindex.mcp.server import MCPServer

async def reproduce_issue():
    """Reproduce the issue with minimal setup."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create minimal test repository
        test_repo = temp_path / "test_repo"
        test_repo.mkdir()
        
        # Add minimal test files
        (test_repo / "test.py").write_text("def hello(): pass")
        (test_repo / "README.md").write_text("# Test")
        
        # Initialize git repository
        import subprocess
        subprocess.run(["git", "init"], cwd=test_repo, check=True)
        subprocess.run(["git", "add", "."], cwd=test_repo, check=True)
        subprocess.run(["git", "commit", "-m", "Initial"], cwd=test_repo, check=True)
        
        # Try to reproduce issue
        server = MCPServer(storage_dir=temp_path / "storage")
        
        try:
            result = await server._ensure_repo_index({
                "path": str(test_repo),
                "language": "py"
            })
            print(f"Success: {result}")
            
        except Exception as e:
            print(f"Error reproduced: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(reproduce_issue())
```

### Community Resources

- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share solutions
- **Documentation**: Check latest documentation updates
- **Examples**: Browse example usage patterns

This troubleshooting guide covers the most common issues with Mimir. For complex problems not covered here, create a detailed issue report with the information gathering steps provided above.