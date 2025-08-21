#!/bin/bash
# Mimir Health Check Script
# Comprehensive health monitoring for production deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Default values
TIMEOUT=30
INTERVAL=5
MAX_RETRIES=6
VERBOSE=false
OUTPUT_FORMAT="text"
CHECK_TYPE="all"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    if [[ "$VERBOSE" == "true" ]] || [[ "$OUTPUT_FORMAT" == "text" ]]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
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
Mimir Health Check Script

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --timeout SECONDS   Timeout for each check [default: 30]
    -i, --interval SECONDS  Interval between retries [default: 5]
    -r, --retries COUNT     Maximum retry attempts [default: 6]
    -v, --verbose           Verbose output
    -f, --format FORMAT     Output format (text|json) [default: text]
    -c, --check TYPE        Check type (all|liveness|readiness|detailed) [default: all]
    -h, --help              Show this help message

EXAMPLES:
    $0                      # Run all health checks
    $0 -c liveness          # Check only basic liveness
    $0 -f json              # Output in JSON format
    $0 -v -t 60 -r 10       # Verbose with custom timeout and retries

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--timeout)
                TIMEOUT="$2"
                shift 2
                ;;
            -i|--interval)
                INTERVAL="$2"
                shift 2
                ;;
            -r|--retries)
                MAX_RETRIES="$2"
                shift 2
                ;;
            -v|--verbose)
                VERBOSE="true"
                shift
                ;;
            -f|--format)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -c|--check)
                CHECK_TYPE="$2"
                shift 2
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

# Check if Docker containers are running
check_containers() {
    log_info "Checking container status"
    
    local containers=("mimir-server" "mimir-ui" "redis")
    local all_healthy=true
    local results=()
    
    for container in "${containers[@]}"; do
        if docker ps --format "table {{.Names}}\t{{.Status}}" | grep -q "$container"; then
            local status=$(docker ps --format "{{.Status}}" --filter "name=$container")
            if [[ "$status" == *"healthy"* ]] || [[ "$status" == *"Up"* ]]; then
                log_success "Container $container is running"
                results+=("\"$container\": {\"status\": \"healthy\", \"details\": \"$status\"}")
            else
                log_error "Container $container is unhealthy: $status"
                results+=("\"$container\": {\"status\": \"unhealthy\", \"details\": \"$status\"}")
                all_healthy=false
            fi
        else
            log_error "Container $container is not running"
            results+=("\"$container\": {\"status\": \"not_running\", \"details\": \"Container not found\"}")
            all_healthy=false
        fi
    done
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "\"containers\": {$(IFS=,; echo "${results[*]}")}"
    fi
    
    return $([ "$all_healthy" == "true" ] && echo 0 || echo 1)
}

# Check service endpoints
check_endpoints() {
    log_info "Checking service endpoints"
    
    local endpoints=(
        "mimir-ui:8000:/health:UI Health"
        "mimir-server:9100:/metrics:Server Metrics"
        "redis:6379::Redis"
    )
    
    local all_healthy=true
    local results=()
    
    for endpoint_info in "${endpoints[@]}"; do
        IFS=':' read -r host port path description <<< "$endpoint_info"
        
        if [[ -n "$path" ]]; then
            # HTTP endpoint check
            local url="http://$host:$port$path"
            log_info "Checking $description at $url"
            
            if curl -sf --max-time "$TIMEOUT" "$url" > /dev/null 2>&1; then
                log_success "$description is responsive"
                results+=("\"$description\": {\"status\": \"healthy\", \"url\": \"$url\"}")
            else
                log_error "$description is not responsive"
                results+=("\"$description\": {\"status\": \"unhealthy\", \"url\": \"$url\"}")
                all_healthy=false
            fi
        else
            # TCP port check
            log_info "Checking $description TCP connection at $host:$port"
            
            if nc -z "$host" "$port" 2>/dev/null; then
                log_success "$description TCP connection is healthy"
                results+=("\"$description\": {\"status\": \"healthy\", \"endpoint\": \"$host:$port\"}")
            else
                log_error "$description TCP connection failed"
                results+=("\"$description\": {\"status\": \"unhealthy\", \"endpoint\": \"$host:$port\"}")
                all_healthy=false
            fi
        fi
    done
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "\"endpoints\": {$(IFS=,; echo "${results[*]}")}"
    fi
    
    return $([ "$all_healthy" == "true" ] && echo 0 || echo 1)
}

# Check application health via Python health module
check_application_health() {
    log_info "Checking application health"
    
    local health_output
    local exit_code=0
    
    # Execute health check inside the container
    health_output=$(docker exec mimir-server python -c "
import asyncio
import json
import sys
from src.repoindex.health import get_health_checker

async def main():
    checker = get_health_checker()
    try:
        health = await checker.detailed_health()
        print(json.dumps(health, indent=2))
        sys.exit(0 if health['status'] in ['ready', 'alive'] else 1)
    except Exception as e:
        print(json.dumps({'status': 'error', 'error': str(e)}, indent=2))
        sys.exit(1)

asyncio.run(main())
" 2>&1) || exit_code=$?
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "Application health check passed"
        
        if [[ "$OUTPUT_FORMAT" == "json" ]]; then
            echo "\"application\": $health_output"
        elif [[ "$VERBOSE" == "true" ]]; then
            echo "$health_output"
        fi
    else
        log_error "Application health check failed"
        
        if [[ "$OUTPUT_FORMAT" == "json" ]]; then
            echo "\"application\": {\"status\": \"unhealthy\", \"error\": \"Health check failed\", \"output\": \"$health_output\"}"
        else
            echo "Health check output: $health_output"
        fi
    fi
    
    return $exit_code
}

# Check system resources
check_system_resources() {
    log_info "Checking system resources"
    
    local results=()
    local all_healthy=true
    
    # Memory check
    local memory_usage=$(docker exec mimir-server python -c "import psutil; print(psutil.virtual_memory().percent)")
    if (( $(echo "$memory_usage > 90" | bc -l) )); then
        log_warn "High memory usage: ${memory_usage}%"
        results+=("\"memory\": {\"status\": \"warning\", \"usage_percent\": $memory_usage}")
    else
        log_success "Memory usage is healthy: ${memory_usage}%"
        results+=("\"memory\": {\"status\": \"healthy\", \"usage_percent\": $memory_usage}")
    fi
    
    # CPU check
    local cpu_usage=$(docker exec mimir-server python -c "import psutil; print(psutil.cpu_percent(interval=1))")
    if (( $(echo "$cpu_usage > 80" | bc -l) )); then
        log_warn "High CPU usage: ${cpu_usage}%"
        results+=("\"cpu\": {\"status\": \"warning\", \"usage_percent\": $cpu_usage}")
    else
        log_success "CPU usage is healthy: ${cpu_usage}%"
        results+=("\"cpu\": {\"status\": \"healthy\", \"usage_percent\": $cpu_usage}")
    fi
    
    # Disk space check
    local disk_usage=$(docker exec mimir-server df /app/data | awk 'NR==2 {print $5}' | sed 's/%//')
    if [[ $disk_usage -gt 90 ]]; then
        log_error "Low disk space: ${disk_usage}% used"
        results+=("\"disk\": {\"status\": \"critical\", \"usage_percent\": $disk_usage}")
        all_healthy=false
    elif [[ $disk_usage -gt 80 ]]; then
        log_warn "High disk usage: ${disk_usage}% used"
        results+=("\"disk\": {\"status\": \"warning\", \"usage_percent\": $disk_usage}")
    else
        log_success "Disk usage is healthy: ${disk_usage}% used"
        results+=("\"disk\": {\"status\": \"healthy\", \"usage_percent\": $disk_usage}")
    fi
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "\"system_resources\": {$(IFS=,; echo "${results[*]}")}"
    fi
    
    return $([ "$all_healthy" == "true" ] && echo 0 || echo 1)
}

# Perform liveness check
liveness_check() {
    log_info "Performing liveness check"
    
    local results=()
    local overall_status="healthy"
    
    # Check if containers are running
    if check_containers; then
        results+=("containers:healthy")
    else
        results+=("containers:unhealthy")
        overall_status="unhealthy"
    fi
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "{\"check_type\": \"liveness\", \"status\": \"$overall_status\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", $(IFS=,; echo "${results[*]}")}"
    else
        log_info "Liveness check completed with status: $overall_status"
    fi
    
    return $([ "$overall_status" == "healthy" ] && echo 0 || echo 1)
}

# Perform readiness check
readiness_check() {
    log_info "Performing readiness check"
    
    local results=()
    local overall_status="ready"
    
    # Check containers
    if check_containers; then
        results+=("containers:healthy")
    else
        results+=("containers:unhealthy")
        overall_status="not_ready"
    fi
    
    # Check endpoints
    if check_endpoints; then
        results+=("endpoints:healthy")
    else
        results+=("endpoints:unhealthy")
        overall_status="not_ready"
    fi
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "{\"check_type\": \"readiness\", \"status\": \"$overall_status\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", $(IFS=,; echo "${results[*]}")}"
    else
        log_info "Readiness check completed with status: $overall_status"
    fi
    
    return $([ "$overall_status" == "ready" ] && echo 0 || echo 1)
}

# Perform detailed health check
detailed_check() {
    log_info "Performing detailed health check"
    
    local results=()
    local overall_status="healthy"
    
    # All previous checks
    if ! check_containers; then
        overall_status="unhealthy"
    fi
    
    if ! check_endpoints; then
        overall_status="unhealthy"
    fi
    
    if ! check_application_health; then
        overall_status="unhealthy"
    fi
    
    if ! check_system_resources; then
        if [[ "$overall_status" == "healthy" ]]; then
            overall_status="warning"
        fi
    fi
    
    if [[ "$OUTPUT_FORMAT" == "json" ]]; then
        echo "{\"check_type\": \"detailed\", \"status\": \"$overall_status\", \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", $(IFS=,; echo "${results[*]}")}"
    else
        log_info "Detailed health check completed with status: $overall_status"
    fi
    
    return $([ "$overall_status" == "healthy" ] && echo 0 || echo 1)
}

# Main health check function with retries
run_health_check() {
    local attempt=1
    local success=false
    
    while [[ $attempt -le $MAX_RETRIES ]]; do
        log_info "Health check attempt $attempt/$MAX_RETRIES"
        
        case "$CHECK_TYPE" in
            "liveness")
                if liveness_check; then
                    success=true
                    break
                fi
                ;;
            "readiness")
                if readiness_check; then
                    success=true
                    break
                fi
                ;;
            "detailed")
                if detailed_check; then
                    success=true
                    break
                fi
                ;;
            "all")
                if liveness_check && readiness_check && detailed_check; then
                    success=true
                    break
                fi
                ;;
            *)
                log_error "Invalid check type: $CHECK_TYPE"
                exit 1
                ;;
        esac
        
        if [[ $attempt -lt $MAX_RETRIES ]]; then
            log_warn "Health check failed, retrying in ${INTERVAL}s..."
            sleep "$INTERVAL"
        fi
        
        ((attempt++))
    done
    
    if [[ "$success" == "true" ]]; then
        log_success "Health check passed"
        exit 0
    else
        log_error "Health check failed after $MAX_RETRIES attempts"
        exit 1
    fi
}

# Main function
main() {
    parse_args "$@"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Check if Docker Compose services are available
    if ! docker-compose ps > /dev/null 2>&1; then
        log_error "Docker Compose services not found. Are the services running?"
        exit 1
    fi
    
    run_health_check
}

# Run main function
main "$@"