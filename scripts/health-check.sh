#!/bin/bash
# Comprehensive Health Check Script for Mimir-Lens Integration
# Validates all service components and integration points

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ERROR:${NC} $1"
}

check_service() {
    local name="$1"
    local url="$2"
    local description="$3"
    
    if curl -f -s --max-time 10 "$url" > /dev/null 2>&1; then
        printf "✅ %-30s %s\n" "$description" "OK"
        return 0
    else
        printf "❌ %-30s %s\n" "$description" "FAILED"
        return 1
    fi
}

check_port() {
    local service="$1"
    local host="$2"
    local port="$3"
    
    if timeout 5 bash -c "</dev/tcp/$host/$port" 2>/dev/null; then
        printf "✅ %-30s %s\n" "$service" "OK"
        return 0
    else
        printf "❌ %-30s %s\n" "$service" "FAILED"
        return 1
    fi
}

main() {
    echo "🏥 Mimir-Lens Health Check"
    echo "=================================================="
    echo ""
    
    local failed=0
    
    # Check web services
    check_service "mimir" "http://localhost:8000/health" "Mimir MCP Server" || ((failed++))
    check_service "lens" "http://localhost:3000/health" "Lens Search Engine" || ((failed++))
    check_service "prometheus" "http://localhost:9090/-/healthy" "Prometheus" || ((failed++))
    check_service "grafana" "http://localhost:3001/api/health" "Grafana" || ((failed++))
    
    # Check database ports
    check_port "PostgreSQL Database" "localhost" "5432" || ((failed++))
    check_port "Redis Cache" "localhost" "6379" || ((failed++))
    check_port "NATS Queue" "localhost" "4222" || ((failed++))
    
    echo ""
    if [[ $failed -eq 0 ]]; then
        log "All services are healthy ✅"
        exit 0
    else
        error "$failed service(s) are unhealthy ❌"
        exit 1
    fi
}

main "$@"