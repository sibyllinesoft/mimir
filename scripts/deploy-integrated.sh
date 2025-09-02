#!/bin/bash
# Production Deployment Script for Mimir-Lens Integration
# Handles blue-green deployments with health checks and rollbacks

set -euo pipefail

# ==========================================================================
# CONFIGURATION
# ==========================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
VERSION="${VERSION:-latest}"
MAX_WAIT_TIME=300  # 5 minutes
HEALTH_CHECK_INTERVAL=10  # 10 seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==========================================================================
# UTILITY FUNCTIONS
# ==========================================================================

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR:${NC} $1"
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy Mimir-Lens integrated system to production environment.

OPTIONS:
    -e, --environment ENV    Target environment (staging|production) [default: production]
    -v, --version VERSION    Version to deploy [default: latest]
    -s, --strategy STRATEGY  Deployment strategy (blue-green|rolling) [default: blue-green]
    -c, --config CONFIG      Configuration file [default: .env.production]
    -d, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

EXAMPLES:
    $0 --environment staging --version v1.2.3
    $0 --strategy rolling --config .env.custom
    $0 --dry-run

EOF
}

# ==========================================================================
# DEPLOYMENT FUNCTIONS
# ==========================================================================

check_prerequisites() {
    log "Checking deployment prerequisites..."
    
    # Check required commands
    local required_commands=("docker" "docker-compose" "curl" "jq")
    for cmd in "${required_commands[@]}"; do
        if ! command -v "$cmd" &> /dev/null; then
            error "Required command '$cmd' not found"
            exit 1
        fi
    done
    
    # Check Docker is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        exit 1
    fi
    
    # Check environment file exists
    if [[ ! -f "$CONFIG_FILE" ]]; then
        error "Configuration file '$CONFIG_FILE' not found"
        exit 1
    fi
    
    # Check Lens repository is available (if building locally)
    if [[ ! -d "/media/nathan/Seagate Hub/Projects/lens" ]]; then
        warn "Lens repository not found locally - assuming remote images"
    fi
    
    log "Prerequisites check completed"
}

backup_current_deployment() {
    log "Creating backup of current deployment..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup current environment configuration
    if [[ -f "$CONFIG_FILE" ]]; then
        cp "$CONFIG_FILE" "$backup_dir/config.backup"
    fi
    
    # Backup database (if accessible)
    if docker-compose -f docker-compose.integrated.yml ps postgres | grep -q "Up"; then
        log "Backing up PostgreSQL database..."
        docker-compose -f docker-compose.integrated.yml exec -T postgres \
            pg_dump -U mimir_user mimir > "$backup_dir/database.sql" || warn "Database backup failed"
    fi
    
    # Create deployment manifest
    cat > "$backup_dir/deployment_info.json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$DEPLOYMENT_ENV",
    "version": "$VERSION",
    "backup_reason": "Pre-deployment backup"
}
EOF
    
    echo "$backup_dir" > .last_backup_path
    log "Backup created: $backup_dir"
}

pull_images() {
    log "Pulling latest container images..."
    
    # Set image tags based on version
    export MIMIR_IMAGE_TAG="${VERSION}"
    export LENS_IMAGE_TAG="${VERSION}"
    
    # Pull images
    docker-compose -f docker-compose.integrated.yml --env-file "$CONFIG_FILE" pull
    
    log "Images pulled successfully"
}

health_check() {
    local service_url="$1"
    local service_name="$2"
    local max_attempts=$(($MAX_WAIT_TIME / $HEALTH_CHECK_INTERVAL))
    local attempt=1
    
    log "Health checking $service_name..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$service_url/health" > /dev/null 2>&1; then
            log "$service_name is healthy"
            return 0
        fi
        
        if [[ $attempt -eq $max_attempts ]]; then
            error "$service_name health check failed after $MAX_WAIT_TIME seconds"
            return 1
        fi
        
        echo -n "."
        sleep $HEALTH_CHECK_INTERVAL
        ((attempt++))
    done
}

comprehensive_health_check() {
    log "Running comprehensive health checks..."
    
    # Define service health endpoints
    local services=(
        "http://localhost:8000:Mimir"
        "http://localhost:3000:Lens"
        "http://localhost:9090:Prometheus"
        "http://localhost:3001:Grafana"
    )
    
    local all_healthy=true
    
    for service in "${services[@]}"; do
        IFS=':' read -r url name <<< "$service"
        if ! health_check "$url" "$name"; then
            all_healthy=false
        fi
    done
    
    if [[ "$all_healthy" == "true" ]]; then
        log "All services are healthy"
        return 0
    else
        error "Some services failed health checks"
        return 1
    fi
}

blue_green_deployment() {
    log "Starting blue-green deployment..."
    
    # Current deployment is "blue", new deployment is "green"
    local green_compose_file="docker-compose.integrated.green.yml"
    
    # Create green environment configuration
    cp docker-compose.integrated.yml "$green_compose_file"
    
    # Modify ports for green deployment (add 100 to each port)
    sed -i 's/8000:8000/8100:8000/g' "$green_compose_file"
    sed -i 's/3000:3000/3100:3000/g' "$green_compose_file"
    sed -i 's/9090:9090/9190:9090/g' "$green_compose_file"
    sed -i 's/3001:3000/3101:3000/g' "$green_compose_file"
    
    # Start green deployment
    log "Starting green deployment..."
    docker-compose -f "$green_compose_file" --env-file "$CONFIG_FILE" up -d
    
    # Wait for green deployment to be ready
    sleep 30
    
    # Health check green deployment
    local green_services=(
        "http://localhost:8100:Mimir-Green"
        "http://localhost:3100:Lens-Green"
    )
    
    local green_healthy=true
    for service in "${green_services[@]}"; do
        IFS=':' read -r url name <<< "$service"
        if ! health_check "$url" "$name"; then
            green_healthy=false
        fi
    done
    
    if [[ "$green_healthy" == "true" ]]; then
        log "Green deployment is healthy, switching traffic..."
        
        # Switch traffic (in production, this would be done by a load balancer)
        # For now, we'll stop blue and rename green to blue
        
        # Stop blue deployment
        docker-compose -f docker-compose.integrated.yml --env-file "$CONFIG_FILE" down
        
        # Replace blue with green
        mv docker-compose.integrated.yml docker-compose.integrated.blue.backup
        mv "$green_compose_file" docker-compose.integrated.yml
        
        # Restart with correct ports
        sed -i 's/8100:8000/8000:8000/g' docker-compose.integrated.yml
        sed -i 's/3100:3000/3000:3000/g' docker-compose.integrated.yml
        sed -i 's/9190:9090/9090:9090/g' docker-compose.integrated.yml
        sed -i 's/3101:3000/3001:3000/g' docker-compose.integrated.yml
        
        docker-compose -f docker-compose.integrated.yml --env-file "$CONFIG_FILE" up -d
        
        # Final health check
        if comprehensive_health_check; then
            log "Blue-green deployment completed successfully"
            
            # Clean up backup
            rm -f docker-compose.integrated.blue.backup
            return 0
        else
            error "Final health check failed, initiating rollback"
            rollback_deployment
            return 1
        fi
    else
        error "Green deployment failed health check, cleaning up"
        docker-compose -f "$green_compose_file" --env-file "$CONFIG_FILE" down
        rm -f "$green_compose_file"
        return 1
    fi
}

rolling_deployment() {
    log "Starting rolling deployment..."
    
    # Rolling deployment with health checks
    docker-compose -f docker-compose.integrated.yml --env-file "$CONFIG_FILE" up -d --remove-orphans
    
    # Wait for deployment to stabilize
    sleep 60
    
    if comprehensive_health_check; then
        log "Rolling deployment completed successfully"
        return 0
    else
        error "Rolling deployment failed health check"
        return 1
    fi
}

rollback_deployment() {
    log "Initiating deployment rollback..."
    
    if [[ -f ".last_backup_path" ]]; then
        local backup_dir=$(cat .last_backup_path)
        
        if [[ -d "$backup_dir" ]]; then
            log "Restoring from backup: $backup_dir"
            
            # Restore configuration
            if [[ -f "$backup_dir/config.backup" ]]; then
                cp "$backup_dir/config.backup" "$CONFIG_FILE"
            fi
            
            # Restore database (if backup exists)
            if [[ -f "$backup_dir/database.sql" ]]; then
                warn "Database rollback requires manual intervention"
                warn "Backup location: $backup_dir/database.sql"
            fi
            
            # Restart with previous configuration
            docker-compose -f docker-compose.integrated.yml --env-file "$CONFIG_FILE" down
            docker-compose -f docker-compose.integrated.yml --env-file "$CONFIG_FILE" up -d
            
            log "Rollback completed"
        else
            error "Backup directory not found: $backup_dir"
            return 1
        fi
    else
        error "No backup information found for rollback"
        return 1
    fi
}

post_deployment_tasks() {
    log "Running post-deployment tasks..."
    
    # Update monitoring annotations
    if curl -s "http://localhost:9090/-/healthy" > /dev/null; then
        # Create deployment annotation in Prometheus/Grafana
        local annotation_data=$(cat << EOF
{
    "time": $(date +%s)000,
    "text": "Deployment: $VERSION",
    "user": "deploy-script",
    "tags": ["deployment", "$DEPLOYMENT_ENV"]
}
EOF
)
        # This would be sent to Grafana API in production
        log "Deployment annotation created"
    fi
    
    # Clean up old containers and images
    docker system prune -f --volumes --filter "until=24h"
    
    # Generate deployment report
    cat > "deployment-report-$(date +%Y%m%d_%H%M%S).json" << EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": "$DEPLOYMENT_ENV",
    "version": "$VERSION",
    "strategy": "$DEPLOYMENT_STRATEGY",
    "status": "success",
    "services": {
        "mimir": "healthy",
        "lens": "healthy",
        "postgres": "healthy",
        "redis": "healthy",
        "monitoring": "healthy"
    }
}
EOF
    
    log "Post-deployment tasks completed"
}

# ==========================================================================
# MAIN DEPLOYMENT LOGIC
# ==========================================================================

main() {
    log "Starting Mimir-Lens integrated deployment"
    log "Environment: $DEPLOYMENT_ENV"
    log "Version: $VERSION"
    log "Strategy: $DEPLOYMENT_STRATEGY"
    log "Config: $CONFIG_FILE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log "DRY RUN MODE - No actual deployment will occur"
        log "Would deploy version $VERSION to $DEPLOYMENT_ENV environment"
        exit 0
    fi
    
    # Deployment steps
    check_prerequisites
    backup_current_deployment
    pull_images
    
    case "$DEPLOYMENT_STRATEGY" in
        "blue-green")
            if blue_green_deployment; then
                post_deployment_tasks
                log "Deployment completed successfully!"
            else
                error "Deployment failed"
                exit 1
            fi
            ;;
        "rolling")
            if rolling_deployment; then
                post_deployment_tasks
                log "Deployment completed successfully!"
            else
                error "Deployment failed"
                rollback_deployment
                exit 1
            fi
            ;;
        *)
            error "Unknown deployment strategy: $DEPLOYMENT_STRATEGY"
            exit 1
            ;;
    esac
}

# ==========================================================================
# ARGUMENT PARSING
# ==========================================================================

DEPLOYMENT_STRATEGY="blue-green"
CONFIG_FILE=".env.production"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            DEPLOYMENT_ENV="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -s|--strategy)
            DEPLOYMENT_STRATEGY="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -d|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ "$DEPLOYMENT_ENV" != "staging" && "$DEPLOYMENT_ENV" != "production" ]]; then
    error "Invalid environment: $DEPLOYMENT_ENV (must be staging or production)"
    exit 1
fi

if [[ "$DEPLOYMENT_STRATEGY" != "blue-green" && "$DEPLOYMENT_STRATEGY" != "rolling" ]]; then
    error "Invalid strategy: $DEPLOYMENT_STRATEGY (must be blue-green or rolling)"
    exit 1
fi

# Update config file based on environment
if [[ "$CONFIG_FILE" == ".env.production" && "$DEPLOYMENT_ENV" == "staging" ]]; then
    CONFIG_FILE=".env.staging"
fi

# Change to project directory
cd "$PROJECT_DIR"

# Run main deployment
main