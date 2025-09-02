# 🚀 Mimir Installation Guide

**Get your AI-powered code research system running in under 5 minutes**

> Transform your development workflow with research-backed code intelligence. Mimir brings Stanford's RAPTOR algorithms and Microsoft's HyDE techniques to your fingertips through Claude Desktop integration.

---

## ⚡ Choose Your Adventure

### 🎯 Option 1: Claude Desktop Integration (Most Popular)

**Perfect for:** Individual developers who want instant code intelligence in Claude Desktop

```bash
# 1. Clone and set up
git clone https://github.com/your-org/mimir.git && cd mimir
python setup.py

# 2. Configure Claude Desktop
# Add to claude_desktop_config.json:
# File: ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
#       %APPDATA%\Claude\claude_desktop_config.json (Windows)
{
  "mcpServers": {
    "mimir": {
      "command": "uv",
      "args": ["run", "python", "-m", "repoindex.mcp.server"],
      "cwd": "/path/to/mimir"
    }
  }
}

# 3. Restart Claude Desktop → Start asking code questions! 🎉
```

**✅ You get:** Native Claude integration, zero-config setup, instant code search

### 🛠️ Option 2: Development & API Server 

**Perfect for:** Developers who want API access, web UI, or integration with other tools

```bash
# Clone and set up development environment
git clone https://github.com/your-org/mimir.git && cd mimir
python setup.py

# Start MCP server (for programmatic access)
uv run python -m repoindex.mcp.server

# OR start web UI (for interactive exploration)  
uv run python -m repoindex.ui.app

# Access your tools
# 🔧 MCP Server: stdio interface for programmatic access
# 🌐 Web UI: http://localhost:8000 for interactive exploration  
# 📡 API: http://localhost:8000/api for custom integrations
```

**✅ You get:** Web interface, REST API, development tools, monitoring

### 🐳 Option 3: Docker (Production Ready)

**Perfect for:** Teams, production deployments, or isolated environments

```bash
# Quick start with monitoring
git clone https://github.com/your-org/mimir.git && cd mimir
docker-compose up -d

# Production with full observability stack
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Access your services
# 📊 Mimir: http://localhost:8000
# 📈 Grafana: http://localhost:3000 (admin/admin)
# ⚡ Prometheus: http://localhost:9090
```

**✅ You get:** Production monitoring, auto-scaling, health checks, security

### 🚀 Option 4: One-Click Installers

**Perfect for:** Quick evaluation or non-technical users

**Linux/macOS:**
```bash
# Smart installer detects your system and preferences
curl -sSL https://install.mimir.dev | bash

# What it does:
# ✅ Detects Python version and package manager
# ✅ Configures Claude Desktop automatically  
# ✅ Sets up optimal system settings
# ✅ Runs health checks and validation
```

**Windows:**
```powershell
# PowerShell installer with GUI options
iwr -useb https://install.mimir.dev/windows.ps1 | iex

# Features:
# ✅ Windows-specific optimizations
# ✅ PATH configuration 
# ✅ Claude Desktop auto-detection
# ✅ Visual setup wizard
```

**✅ You get:** Zero-configuration setup, system optimization, automatic validation

## Detailed Installation Methods

### 1. Docker Installation

Docker provides the easiest and most consistent deployment method.

#### Prerequisites
- Docker (20.10+)
- Docker Compose (1.29+)

#### Basic Docker Setup

```bash
# 1. Clone repository
git clone https://github.com/your-username/mimir.git
cd mimir

# 2. Configure environment
cp .env.example .env
# Edit .env as needed

# 3. Start services
docker-compose -f docker-compose.install.yml up -d

# 4. Check status
docker-compose -f docker-compose.install.yml ps

# 5. View logs
docker-compose -f docker-compose.install.yml logs -f mimir-server
```

#### Docker with UI and Monitoring

```bash
# Start with UI and monitoring
docker-compose -f docker-compose.install.yml --profile ui --profile monitoring up -d

# Access services
# - Mimir Server: http://localhost:8000
# - Mimir UI: http://localhost:8080
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

#### Docker Management Commands

```bash
# Stop services
docker-compose -f docker-compose.install.yml down

# Restart services
docker-compose -f docker-compose.install.yml restart

# Update to latest version
docker-compose -f docker-compose.install.yml pull
docker-compose -f docker-compose.install.yml up -d

# View service logs
docker-compose -f docker-compose.install.yml logs -f [service-name]

# Execute commands in container
docker-compose -f docker-compose.install.yml exec mimir-server bash
```

### 2. Python Package Installation

Install Mimir as a Python package for development or integration.

> **For MCP usage with Claude Desktop**, see [MCP_CONFIGURATION.md](MCP_CONFIGURATION.md) for detailed setup instructions.

#### Prerequisites
- Python 3.11+
- pip
- Git

#### From PyPI

```bash
# Basic installation
pip install mimir

# With UI support
pip install mimir[ui]

# With development dependencies
pip install mimir[dev,ui,test]

# Start server
mimir-server --help
mimir-server
```

#### From Wheel File

```bash
# Download wheel from releases
wget https://github.com/your-username/mimir/releases/latest/download/mimir-1.0.0-py3-none-any.whl

# Install
pip install mimir-1.0.0-py3-none-any.whl

# Verify installation
mimir-server --version
```

#### From Source

```bash
# Clone repository
git clone https://github.com/your-username/mimir.git
cd mimir

# Install in development mode
pip install -e .[dev,ui,test]

# Or build and install
python -m build
pip install dist/mimir-1.0.0-py3-none-any.whl
```

### 3. Standalone Executable

Use the standalone executable for systems without Python.

#### Linux/macOS

```bash
# Download executable
wget https://github.com/your-username/mimir/releases/latest/download/mimir-server-linux-x86_64.zip

# Extract and install
unzip mimir-server-linux-x86_64.zip
sudo mv mimir-server /usr/local/bin/
chmod +x /usr/local/bin/mimir-server

# Run
mimir-server --help
```

#### Windows

```cmd
# Download executable
powershell -Command "Invoke-WebRequest -Uri 'https://github.com/your-username/mimir/releases/latest/download/mimir-server-windows-x86_64.zip' -OutFile 'mimir-server.zip'"

# Extract
powershell -Command "Expand-Archive -Path 'mimir-server.zip' -DestinationPath '.'"

# Run
mimir-server.exe --help
```

### 4. Automated Installation Scripts

Use our installation scripts for guided setup.

#### Linux/macOS Script

```bash
# Basic installation from PyPI
./scripts/install.sh

# Install from wheel file
./scripts/install.sh --method wheel --path ./dist/mimir-1.0.0-py3-none-any.whl

# Install from source with development dependencies
./scripts/install.sh --method source --path . --dev

# Install standalone executable
./scripts/install.sh --method standalone --path ./mimir-server

# Docker setup
./scripts/install.sh --method docker
```

#### Windows Script

```cmd
# Basic installation from PyPI
scripts\install.bat

# Install from wheel file
scripts\install.bat /method wheel /path mimir-1.0.0-py3-none-any.whl

# Install from source with development dependencies
scripts\install.bat /method source /path . /dev

# Install standalone executable
scripts\install.bat /method standalone /path mimir-server.exe

# Docker setup
scripts\install.bat /method docker
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
# Core settings
MIMIR_LOG_LEVEL=INFO
MIMIR_MAX_WORKERS=4
MIMIR_PORT=8000

# Storage
MIMIR_DATA_DIR=./data
MIMIR_CACHE_DIR=./cache

# AI Integration (optional)
GOOGLE_API_KEY=your-api-key

# Security
MIMIR_ENABLE_SECURITY=true
MIMIR_CORS_ORIGINS=*
```

### Command Line Options

```bash
# Show all options
mimir-server --help

# Common usage
mimir-server --port 8080 --log-level DEBUG
mimir-server --config /path/to/config.yml
mimir-server --data-dir /custom/data/path
```

### Configuration File

Create `config.yml` for advanced configuration:

```yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4

storage:
  data_dir: "./data"
  cache_dir: "./cache"
  max_size: "10GB"

logging:
  level: "INFO"
  format: "json"
  
security:
  enabled: true
  api_key: "your-secure-key"
  cors_origins: ["*"]

ai:
  provider: "gemini"
  api_key: "${GOOGLE_API_KEY}"
  model: "gemini-1.5-flash"
```

## Verification

### Health Checks

```bash
# Check server status
curl http://localhost:8000/health

# Check MCP server
curl http://localhost:8000/mcp/status

# Docker health check
docker-compose -f docker-compose.install.yml ps
```

### Testing Installation

```bash
# Test MCP server functionality
curl -X POST http://localhost:8000/mcp/tools \
  -H "Content-Type: application/json" \
  -d '{"method": "list_tools"}'

# Test repository indexing
curl -X POST http://localhost:8000/mcp/tools \
  -H "Content-Type: application/json" \
  -d '{
    "method": "index_repository",
    "params": {
      "repo_path": "/path/to/repository"
    }
  }'
```

## Upgrading

### Docker

```bash
# Pull latest images
docker-compose -f docker-compose.install.yml pull

# Restart with new images
docker-compose -f docker-compose.install.yml up -d
```

### Python Package

```bash
# Upgrade from PyPI
pip install --upgrade mimir[ui]

# Upgrade from wheel
pip install --upgrade --force-reinstall mimir-1.1.0-py3-none-any.whl
```

### Standalone Executable

```bash
# Download new version
wget https://github.com/your-username/mimir/releases/latest/download/mimir-server-linux-x86_64.zip

# Replace existing
unzip -o mimir-server-linux-x86_64.zip
sudo mv mimir-server /usr/local/bin/
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Change port in .env or command line
   MIMIR_PORT=8001 mimir-server
   # or
   mimir-server --port 8001
   ```

2. **Permission Denied**
   ```bash
   # Fix file permissions
   chmod +x scripts/install.sh
   sudo chown -R $(whoami) ./data ./cache ./logs
   ```

3. **Docker Issues**
   ```bash
   # Restart Docker daemon
   sudo systemctl restart docker
   
   # Clean up containers
   docker-compose -f docker-compose.install.yml down -v
   docker system prune -f
   ```

4. **Python Dependencies**
   ```bash
   # Update pip
   python -m pip install --upgrade pip
   
   # Clear pip cache
   pip cache purge
   
   # Reinstall dependencies
   pip install --force-reinstall mimir[ui]
   ```

### Getting Help

1. **Logs**: Check application logs in `./logs/` or via Docker:
   ```bash
   docker-compose -f docker-compose.install.yml logs -f mimir-server
   ```

2. **Health Check**: Visit http://localhost:8000/health for status

3. **GitHub Issues**: Report problems at https://github.com/your-username/mimir/issues

4. **Documentation**: Full docs at https://your-username.github.io/mimir/

---

## 🎯 Verify Your Installation

### Quick Health Check
```bash
# Test MCP server
mimir-server --version

# Test Claude Desktop integration (after config)
# In Claude: "Tell me about Mimir's capabilities"

# Test web UI (if enabled)
curl http://localhost:8080/health
```

### 🚀 Your First Index
```bash
# Index a sample repository
mimir-index ./my-project --language typescript

# Or via Claude Desktop:
# "Please index my TypeScript project at /path/to/project"
```

---

## 📈 What Success Looks Like

### ✅ Installation Validated
- **MCP Server:** Responds to version check
- **Claude Integration:** Shows Mimir tools in Claude Desktop
- **Dependencies:** RepoMapper, Serena, LEANN installed successfully
- **Performance:** Index completion in under 5 minutes for medium projects

### 🎯 Usage Milestones

**Week 1:** Basic code search and exploration
```
"Find authentication code in this React project"
"Show me the most complex functions"
```

**Week 2:** Advanced analysis and reasoning
```  
"How does the payment flow work end-to-end?"
"What would break if I refactor this component?"
```

**Month 1:** Team productivity gains
- 40% faster code reviews
- 60% reduction in "where is this used?" questions
- 25% improvement in bug investigation time

---

## 🚀 Next Steps

### 🎬 Get Started
1. **📚 [Follow the MCP Guide](./MCP_CONFIGURATION.md)** - Complete Claude Desktop setup
2. **🔍 Index your first repo** - Start with a small TypeScript/Python project
3. **🧠 Try complex queries** - Test RAPTOR reasoning with architectural questions
4. **📊 Explore the web UI** - Visual code exploration and metrics

### ⚡ Level Up
1. **🔧 Enable monitoring** - Set up Grafana dashboards for production insights
2. **🤝 Team integration** - Share index bundles and collaborate on analysis
3. **🎯 Custom queries** - Learn advanced search patterns and filters
4. **📈 Performance tuning** - Optimize for your codebase size and patterns

### 🌟 Join the Community
- **⭐ [Star us on GitHub](https://github.com/your-org/mimir)** - Help others discover Mimir
- **💬 [Join Discussions](https://github.com/your-org/mimir/discussions)** - Connect with users and contributors
- **📝 [Read the blog](https://mimir.dev/blog)** - Learn about new features and research
- **🐦 [Follow updates](https://twitter.com/MimirDev)** - Stay current with releases

---

## 🛡️ Production Considerations

### Security Checklist
- **🔐 API Keys:** Store securely, rotate regularly
- **🌐 Network:** Use HTTPS, restrict CORS origins  
- **📁 File Access:** Limit repository access, audit file operations
- **🔄 Updates:** Enable automatic security patch updates
- **📊 Monitoring:** Set up security event alerting

### Performance Tuning
- **💾 Memory:** 8GB+ RAM for large repositories
- **⚡ CPU:** Multi-core systems benefit from parallel processing
- **💽 Storage:** SSD recommended for index storage
- **🌐 Network:** Local deployment reduces latency

### Enterprise Features
- **👥 Multi-tenant:** Separate index spaces per team/project
- **📊 Analytics:** Usage metrics and performance dashboards  
- **🔗 Integrations:** LDAP/SSO, Slack/Teams notifications
- **📈 Scaling:** Horizontal scaling for large organizations

**Ready for enterprise deployment? [Contact us](mailto:enterprise@mimir.dev) for dedicated support.**

---

## 🎉 Welcome to the Future of Code Understanding!

> *You're now equipped with research-grade code intelligence. Start with simple queries and watch as Mimir transforms how you understand, navigate, and work with code.*

**[🚀 Start Your First Index](#-your-first-index)** | **[📖 Read the Guide](./MCP_CONFIGURATION.md)** | **[💬 Get Help](https://github.com/your-org/mimir/discussions)**