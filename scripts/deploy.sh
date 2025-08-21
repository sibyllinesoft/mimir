#!/bin/bash
# Mimir Deep Code Research System - Deployment Script
# Production-ready deployment automation with safety checks

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${PROJECT_ROOT}/.env"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
PROD_COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.prod.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
ENVIRONMENT="development"
BUILD_CACHE="true"
PULL_IMAGES="true"
RUN_TESTS="false"
BACKUP_DATA="false"
FORCE_RECREATE="false"

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Mimir Deep Code Research System Deployment Script

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --environment ENV    Deployment environment (development|production) [default: development]
    -t, --test              Run tests before deployment
    -b, --backup            Backup existing data before deployment
    -f, --force             Force recreate containers
    --no-cache              Build without Docker cache
    --no-pull               Don't pull latest images
    -h, --help              Show this help message

EXAMPLES:
    $0                                      # Deploy development environment
    $0 -e production -t -b                  # Deploy production with tests and backup
    $0 --force --no-cache                   # Force rebuild without cache

ENVIRONMENT VARIABLES:
    MIMIR_DATA_PATH         Custom data directory path
    MIMIR_CACHE_PATH        Custom cache directory path  
    MIMIR_LOGS_PATH         Custom logs directory path
    DOCKER_REGISTRY         Custom Docker registry URL

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--test)
                RUN_TESTS="true"
                shift
                ;;
            -b|--backup)
                BACKUP_DATA="true"
                shift
                ;;
            -f|--force)
                FORCE_RECREATE="true"
                shift
                ;;
            --no-cache)
                BUILD_CACHE="false"
                shift
                ;;
            --no-pull)
                PULL_IMAGES="false"
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    log_info "Validating deployment environment: $ENVIRONMENT"
    
    if [[ "$ENVIRONMENT" != "development" && "$ENVIRONMENT" != "production" ]]; then
        log_error "Invalid environment: $ENVIRONMENT. Must be 'development' or 'production'"
        exit 1
    fi
    
    # Check Docker availability
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed or not in PATH"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed or not in PATH"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Setup environment file
setup_environment() {
    log_info "Setting up environment configuration"
    
    if [[ ! -f "$ENV_FILE" ]]; then
        if [[ -f "${ENV_FILE}.example" ]]; then
            log_info "Creating .env file from example"
            cp "${ENV_FILE}.example" "$ENV_FILE"
        else
            log_error "No .env file found and no .env.example to copy from"
            exit 1
        fi
    fi
    
    # Ensure required directories exist
    local data_dir="${MIMIR_DATA_PATH:-${PROJECT_ROOT}/data}"
    local cache_dir="${MIMIR_CACHE_PATH:-${PROJECT_ROOT}/cache}"
    local logs_dir="${MIMIR_LOGS_PATH:-${PROJECT_ROOT}/logs}"
    
    mkdir -p "$data_dir" "$cache_dir" "$logs_dir"
    
    # Set appropriate permissions
    chmod 755 "$data_dir" "$cache_dir" "$logs_dir"
    
    log_success "Environment setup completed"
}

# Run tests
run_tests() {
    if [[ "$RUN_TESTS" == "true" ]]; then
        log_info "Running test suite before deployment"
        
        # Build test image
        docker-compose -f "$COMPOSE_FILE" build mimir-server
        
        # Run tests in container
        docker-compose -f "$COMPOSE_FILE" run --rm mimir-server python -m pytest tests/ -v
        
        if [[ $? -eq 0 ]]; then
            log_success "All tests passed"
        else
            log_error "Tests failed. Aborting deployment"
            exit 1
        fi
    fi
}

# Backup existing data
backup_data() {
    if [[ "$BACKUP_DATA" == "true" ]]; then
        log_info "Creating backup of existing data"
        
        local backup_dir="${PROJECT_ROOT}/backups/$(date +%Y%m%d_%H%M%S)"
        local data_dir="${MIMIR_DATA_PATH:-${PROJECT_ROOT}/data}"
        
        if [[ -d "$data_dir" ]]; then
            mkdir -p "$backup_dir"
            log_info "Backing up data to: $backup_dir"
            cp -r "$data_dir" "$backup_dir/"
            log_success "Data backup completed"
        else
            log_warn "No existing data directory found, skipping backup"
        fi
    fi
}

# Build and pull images
prepare_images() {
    log_info "Preparing Docker images"
    
    local compose_files=("-f" "$COMPOSE_FILE")
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_files+=("-f" "$PROD_COMPOSE_FILE")
    fi
    
    # Pull latest images if requested
    if [[ "$PULL_IMAGES" == "true" ]]; then
        log_info "Pulling latest base images"
        docker-compose "${compose_files[@]}" pull --ignore-pull-failures
    fi
    
    # Build images
    local build_args=()
    if [[ "$BUILD_CACHE" == "false" ]]; then
        build_args+=("--no-cache")
    fi
    
    log_info "Building application images"
    docker-compose "${compose_files[@]}" build "${build_args[@]}"
    
    log_success "Image preparation completed"
}

# Deploy services
deploy_services() {
    log_info "Deploying Mimir services"
    
    local compose_files=("-f" "$COMPOSE_FILE")
    local deploy_args=()
    
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_files+=("-f" "$PROD_COMPOSE_FILE")
    fi
    
    if [[ "$FORCE_RECREATE" == "true" ]]; then
        deploy_args+=("--force-recreate")
    fi
    
    # Stop existing services
    log_info "Stopping existing services"
    docker-compose "${compose_files[@]}" down
    
    # Start services
    log_info "Starting services in $ENVIRONMENT mode"
    docker-compose "${compose_files[@]}" up -d "${deploy_args[@]}"
    
    log_success "Services deployment completed"
}

# Health check
wait_for_health() {
    log_info "Waiting for services to become healthy"
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        log_info "Health check attempt $attempt/$max_attempts"
        
        # Check if main service is healthy
        if docker-compose ps | grep -q "mimir-server.*healthy"; then
            log_success "Mimir server is healthy"
            break
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            log_error "Services failed to become healthy within timeout"
            docker-compose logs mimir-server
            exit 1
        fi
        
        sleep 10
        ((attempt++))
    done
}

# Show deployment status
show_status() {
    log_info "Deployment Status"
    echo "==================="
    
    # Show running containers
    docker-compose ps
    
    echo ""
    log_info "Service URLs:"
    
    if [[ "$ENVIRONMENT" == "development" ]]; then
        echo "  UI: http://localhost:8000"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana: http://localhost:3000 (admin/admin)"
    else
        echo "  UI: http://localhost (via nginx)"
        echo "  Grafana: http://localhost:3000"
        echo "  Prometheus: http://localhost:9090"
    fi
    
    echo ""
    log_info "Useful commands:"
    echo "  View logs:    docker-compose logs -f"
    echo "  Stop:         docker-compose down"
    echo "  Restart:      docker-compose restart"
    echo "  Update:       $0 --force"
}

# Cleanup on exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        log_error "Deployment failed. Check the logs above for details."
        log_info "You can view service logs with: docker-compose logs"
    fi
}

# Main deployment function
main() {
    log_info "Starting Mimir Deep Code Research System deployment"
    
    parse_args "$@"
    validate_environment
    setup_environment
    run_tests
    backup_data
    prepare_images
    deploy_services
    wait_for_health
    show_status
    
    log_success "Deployment completed successfully!"
}

# Set trap for cleanup
trap cleanup EXIT

# Change to project directory
cd "$PROJECT_ROOT"

# Run main function
main "$@"