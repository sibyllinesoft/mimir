#!/bin/bash
# Mimir Installation Script for Unix Systems (Linux/macOS)
# Provides multiple installation methods with user choice

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_NAME="mimir"
REPO_URL="https://github.com/your-username/mimir"  # Update with actual repo
PYTHON_MIN_VERSION="3.11"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Version comparison function
version_ge() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Check Python version
check_python() {
    local python_cmd=""
    
    # Try different Python commands
    for cmd in python3 python python3.11 python3.12; do
        if command_exists "$cmd"; then
            local version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1)
            if version_ge "$version" "$PYTHON_MIN_VERSION"; then
                python_cmd="$cmd"
                log_success "Found compatible Python: $cmd ($version)"
                break
            else
                log_warning "Found Python $cmd ($version) but requires >= $PYTHON_MIN_VERSION"
            fi
        fi
    done
    
    if [ -z "$python_cmd" ]; then
        log_error "Python >= $PYTHON_MIN_VERSION not found"
        log_info "Please install Python >= $PYTHON_MIN_VERSION and try again"
        log_info "Visit: https://www.python.org/downloads/"
        exit 1
    fi
    
    echo "$python_cmd"
}

# Check system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."
    
    local missing_deps=()
    
    # Required for Git operations
    if ! command_exists git; then
        missing_deps+=("git")
    fi
    
    # Required for compilation
    if ! command_exists gcc && ! command_exists clang; then
        missing_deps+=("gcc or clang")
    fi
    
    # Required for some packages
    if ! pkg-config --exists libffi; then
        missing_deps+=("libffi-dev")
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing system dependencies: ${missing_deps[*]}"
        log_info "Install them using your system package manager:"
        
        # Provide installation commands for common systems
        if command_exists apt-get; then
            log_info "  sudo apt-get update && sudo apt-get install -y git build-essential libffi-dev"
        elif command_exists yum; then
            log_info "  sudo yum install -y git gcc gcc-c++ libffi-devel"
        elif command_exists brew; then
            log_info "  brew install git libffi"
        else
            log_info "  Please install: ${missing_deps[*]}"
        fi
        
        exit 1
    fi
    
    log_success "All system dependencies found"
}

# Install via pip (from PyPI)
install_from_pypi() {
    log_info "Installing from PyPI..."
    
    local python_cmd=$1
    local install_dev=$2
    
    # Upgrade pip
    $python_cmd -m pip install --upgrade pip
    
    if [ "$install_dev" = "true" ]; then
        $python_cmd -m pip install "repoindex[dev,ui,test]"
    else
        $python_cmd -m pip install "repoindex[ui]"
    fi
    
    log_success "Installed from PyPI"
}

# Install from wheel file
install_from_wheel() {
    log_info "Installing from wheel file..."
    
    local python_cmd=$1
    local wheel_path=$2
    
    if [ ! -f "$wheel_path" ]; then
        log_error "Wheel file not found: $wheel_path"
        exit 1
    fi
    
    # Upgrade pip
    $python_cmd -m pip install --upgrade pip
    
    # Install wheel
    $python_cmd -m pip install "$wheel_path"
    
    log_success "Installed from wheel: $wheel_path"
}

# Install from source
install_from_source() {
    log_info "Installing from source..."
    
    local python_cmd=$1
    local source_path=$2
    local install_dev=$3
    
    if [ ! -d "$source_path" ]; then
        log_error "Source directory not found: $source_path"
        exit 1
    fi
    
    cd "$source_path"
    
    # Upgrade pip and install build tools
    $python_cmd -m pip install --upgrade pip build
    
    if [ "$install_dev" = "true" ]; then
        # Development installation
        $python_cmd -m pip install -e ".[dev,ui,test]"
    else
        # Regular installation
        $python_cmd -m pip install -e ".[ui]"
    fi
    
    log_success "Installed from source: $source_path"
}

# Install using standalone executable
install_standalone() {
    log_info "Installing standalone executable..."
    
    local exe_path=$1
    local install_dir="${HOME}/.local/bin"
    
    if [ ! -f "$exe_path" ]; then
        log_error "Executable not found: $exe_path"
        exit 1
    fi
    
    # Create install directory
    mkdir -p "$install_dir"
    
    # Copy executable
    cp "$exe_path" "$install_dir/mimir-server"
    chmod +x "$install_dir/mimir-server"
    
    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$install_dir:"* ]]; then
        echo "export PATH=\"\$PATH:$install_dir\"" >> ~/.bashrc
        log_info "Added $install_dir to PATH in ~/.bashrc"
        log_info "Run 'source ~/.bashrc' or restart your shell to use mimir-server"
    fi
    
    log_success "Installed standalone executable to: $install_dir/mimir-server"
}

# Docker installation
install_docker() {
    log_info "Setting up Docker installation..."
    
    if ! command_exists docker; then
        log_error "Docker not found"
        log_info "Please install Docker first: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Create docker-compose file
    local compose_file="docker-compose.mimir.yml"
    
    cat > "$compose_file" << 'EOF'
version: '3.8'

services:
  mimir-server:
    image: mimir-server:latest
    container_name: mimir-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - mimir-data:/app/data
      - mimir-cache:/app/cache
      - mimir-logs:/app/logs
    environment:
      - MIMIR_LOG_LEVEL=INFO
      - MIMIR_MAX_WORKERS=4
    healthcheck:
      test: ["CMD", "python", "-c", "import asyncio; from src.repoindex.mcp.server import MCPServer; print('Health check passed')"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  mimir-data:
  mimir-cache:
  mimir-logs:
EOF
    
    log_success "Created Docker Compose file: $compose_file"
    log_info "To start Mimir: docker-compose -f $compose_file up -d"
    log_info "To stop Mimir: docker-compose -f $compose_file down"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    local method=$1
    
    case $method in
        "pip"|"wheel"|"source")
            if command_exists mimir-server; then
                local version=$(mimir-server --version 2>/dev/null || echo "unknown")
                log_success "mimir-server installed successfully (version: $version)"
            else
                log_error "mimir-server command not found in PATH"
                return 1
            fi
            ;;
        "standalone")
            if [ -x "${HOME}/.local/bin/mimir-server" ]; then
                log_success "Standalone executable installed successfully"
            else
                log_error "Standalone executable not found"
                return 1
            fi
            ;;
        "docker")
            if docker image inspect mimir-server:latest >/dev/null 2>&1; then
                log_success "Docker image available"
            else
                log_warning "Docker image not found locally"
                log_info "You may need to build or pull the Docker image"
            fi
            ;;
    esac
}

# Create desktop entry (Linux)
create_desktop_entry() {
    if [ "$OSTYPE" = "linux-gnu"* ] && command_exists desktop-file-install; then
        log_info "Creating desktop entry..."
        
        local desktop_file="mimir-server.desktop"
        
        cat > "$desktop_file" << EOF
[Desktop Entry]
Name=Mimir Server
Comment=Deep Code Research System - MCP Server
Exec=mimir-server
Icon=text-x-script
Terminal=true
Type=Application
Categories=Development;Programming;
StartupNotify=false
EOF
        
        if desktop-file-install --dir="${HOME}/.local/share/applications" "$desktop_file" 2>/dev/null; then
            log_success "Desktop entry created"
        else
            log_warning "Could not create desktop entry"
        fi
        
        rm -f "$desktop_file"
    fi
}

# Show usage information
show_usage() {
    cat << EOF
Mimir Installation Script

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -m, --method METHOD     Installation method (pip|wheel|source|standalone|docker)
    -p, --path PATH         Path to wheel file, source directory, or executable
    -d, --dev               Install development dependencies
    -h, --help              Show this help message

INSTALLATION METHODS:
    pip                     Install from PyPI (default)
    wheel                   Install from wheel file (requires --path)
    source                  Install from source directory (requires --path)
    standalone              Install standalone executable (requires --path)
    docker                  Create Docker Compose setup

EXAMPLES:
    $0                                          # Install from PyPI
    $0 --method wheel --path mimir-1.0.0.whl   # Install from wheel
    $0 --method source --path ./mimir --dev    # Install from source with dev deps
    $0 --method standalone --path mimir-server # Install standalone executable
    $0 --method docker                         # Set up Docker installation

EOF
}

# Main installation function
main() {
    local method="pip"
    local path=""
    local install_dev="false"
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -m|--method)
                method="$2"
                shift 2
                ;;
            -p|--path)
                path="$2"
                shift 2
                ;;
            -d|--dev)
                install_dev="true"
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate method
    case $method in
        pip|wheel|source|standalone|docker)
            ;;
        *)
            log_error "Invalid installation method: $method"
            show_usage
            exit 1
            ;;
    esac
    
    # Check if path is required
    if [[ "$method" =~ ^(wheel|source|standalone)$ ]] && [ -z "$path" ]; then
        log_error "Path is required for $method installation"
        show_usage
        exit 1
    fi
    
    log_info "Starting Mimir installation..."
    log_info "Installation method: $method"
    
    # Skip dependency checks for Docker
    if [ "$method" != "docker" ]; then
        check_dependencies
        python_cmd=$(check_python)
    fi
    
    # Install based on method
    case $method in
        pip)
            install_from_pypi "$python_cmd" "$install_dev"
            ;;
        wheel)
            install_from_wheel "$python_cmd" "$path"
            ;;
        source)
            install_from_source "$python_cmd" "$path" "$install_dev"
            ;;
        standalone)
            install_standalone "$path"
            ;;
        docker)
            install_docker
            ;;
    esac
    
    # Verify installation
    if verify_installation "$method"; then
        log_success "Mimir installation completed successfully!"
        
        # Create desktop entry for Linux
        if [ "$method" != "docker" ]; then
            create_desktop_entry
        fi
        
        # Show next steps
        log_info ""
        log_info "Next steps:"
        case $method in
            docker)
                log_info "1. Start Mimir: docker-compose -f docker-compose.mimir.yml up -d"
                log_info "2. Check status: docker-compose -f docker-compose.mimir.yml ps"
                log_info "3. View logs: docker-compose -f docker-compose.mimir.yml logs -f"
                ;;
            *)
                log_info "1. Run 'mimir-server --help' to see available options"
                log_info "2. Start the server: mimir-server"
                log_info "3. Check the documentation for configuration options"
                ;;
        esac
    else
        log_error "Installation verification failed"
        exit 1
    fi
}

# Run main function
main "$@"