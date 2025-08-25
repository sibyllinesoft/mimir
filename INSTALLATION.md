# Mimir Installation Guide

This guide provides multiple methods to install and deploy the Mimir Deep Code Research System.

## Quick Start

### Option 1: Docker (Recommended for most users)

```bash
# Clone the repository
git clone https://github.com/your-username/mimir.git
cd mimir

# Copy environment configuration
cp .env.example .env

# Edit .env with your settings (optional)
nano .env

# Start with Docker Compose
docker-compose -f docker-compose.install.yml up -d

# Access the service
# Server: http://localhost:8000
# UI (if enabled): http://localhost:8080
```

### Option 2: Python Package (PyPI)

```bash
# Install from PyPI
pip install repoindex[ui]

# Start the server
mimir-server

# Or start the UI
mimir-ui
```

### Option 3: Installation Script

**Linux/macOS:**
```bash
# Download and run installer
curl -sSL https://raw.githubusercontent.com/your-username/mimir/main/scripts/install.sh | bash

# Or manually
wget https://github.com/your-username/mimir/releases/latest/download/install.sh
chmod +x install.sh
./install.sh
```

**Windows:**
```cmd
# Download and run installer
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/your-username/mimir/main/scripts/install.bat' -OutFile 'install.bat'"
install.bat
```

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

#### Prerequisites
- Python 3.11+
- pip
- Git

#### From PyPI

```bash
# Basic installation
pip install repoindex

# With UI support
pip install repoindex[ui]

# With development dependencies
pip install repoindex[dev,ui,test]

# Start server
mimir-server --help
mimir-server
```

#### From Wheel File

```bash
# Download wheel from releases
wget https://github.com/your-username/mimir/releases/latest/download/repoindex-1.0.0-py3-none-any.whl

# Install
pip install repoindex-1.0.0-py3-none-any.whl

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
pip install dist/repoindex-1.0.0-py3-none-any.whl
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
./scripts/install.sh --method wheel --path ./dist/repoindex-1.0.0-py3-none-any.whl

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
scripts\install.bat /method wheel /path repoindex-1.0.0-py3-none-any.whl

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
pip install --upgrade repoindex[ui]

# Upgrade from wheel
pip install --upgrade --force-reinstall repoindex-1.1.0-py3-none-any.whl
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
   pip install --force-reinstall repoindex[ui]
   ```

### Getting Help

1. **Logs**: Check application logs in `./logs/` or via Docker:
   ```bash
   docker-compose -f docker-compose.install.yml logs -f mimir-server
   ```

2. **Health Check**: Visit http://localhost:8000/health for status

3. **GitHub Issues**: Report problems at https://github.com/your-username/mimir/issues

4. **Documentation**: Full docs at https://your-username.github.io/mimir/

## Next Steps

After installation:

1. **Index a Repository**: Use the MCP tools to index your first repository
2. **Configure AI**: Set up Gemini API for enhanced features
3. **Set Up Monitoring**: Enable Prometheus/Grafana for production use
4. **Explore Documentation**: Read the full documentation for advanced features
5. **Join Community**: Connect with other users and contributors

## Security Notes

- Change default passwords in production
- Use HTTPS with proper SSL certificates
- Restrict CORS origins to trusted domains
- Keep API keys secure and rotate regularly
- Enable authentication for production deployments
- Regularly update to latest versions for security patches