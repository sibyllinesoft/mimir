# Mimir Deep Code Research System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Mimir Deep Code Research System in both development and production environments. The system is containerized using Docker and orchestrated with Docker Compose for reliable, scalable deployments.

## Architecture

The deployment consists of several key components:

- **MCP Server**: Core AsyncIO-based Python server handling repository indexing
- **UI Service**: Optional web interface for interacting with the system
- **Redis**: Caching layer for improved performance
- **NGINX**: Reverse proxy for production deployments
- **Monitoring Stack**: Prometheus, Grafana, and Loki for observability

## Prerequisites

### System Requirements

- Docker Engine 20.10+
- Docker Compose 2.0+
- 4GB+ RAM (8GB+ recommended for production)
- 20GB+ available disk space
- Linux/macOS (Windows with WSL2)

### Network Requirements

- Outbound internet access for image pulls
- Ports 8000, 3000, 9090 available (development)
- Ports 80, 443, 3000, 9090 available (production)

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd mimir
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` file with your settings:

```bash
# Basic configuration
MIMIR_LOG_LEVEL=INFO
MIMIR_MAX_WORKERS=4
MIMIR_DATA_PATH=./data
MIMIR_CACHE_PATH=./cache
MIMIR_LOGS_PATH=./logs
```

### 3. Deploy Development Environment

```bash
# Using the deployment script (recommended)
./scripts/deploy.sh

# Or using Docker Compose directly
docker-compose up -d
```

### 4. Verify Deployment

```bash
# Check service status
docker-compose ps

# Run health checks
./scripts/health-check.sh

# View logs
docker-compose logs -f mimir-server
```

## Production Deployment

### 1. Environment Setup

Create production environment file:

```bash
cp .env.example .env.production
```

Configure production settings in `.env.production`:

```bash
# Production settings
MIMIR_LOG_LEVEL=WARNING
MIMIR_MAX_WORKERS=8
MIMIR_ENABLE_METRICS=true
GRAFANA_ADMIN_PASSWORD=secure-password-here

# Storage paths (absolute paths for production)
MIMIR_DATA_PATH=/var/lib/mimir/data
MIMIR_CACHE_PATH=/var/lib/mimir/cache
MIMIR_LOGS_PATH=/var/log/mimir

# Security
MIMIR_API_KEY=your-secure-api-key
MIMIR_JWT_SECRET=your-jwt-secret
```

### 2. Production Deployment

```bash
# Deploy with production configuration
./scripts/deploy.sh -e production -t -b

# Or manually
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### 3. SSL/TLS Configuration (Optional)

For HTTPS in production:

```bash
# Create SSL directory
mkdir -p ops/ssl

# Copy your certificates
cp your-domain.crt ops/ssl/mimir.crt
cp your-domain.key ops/ssl/mimir.key

# Update nginx configuration to enable HTTPS server block
```

## Service Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MIMIR_LOG_LEVEL` | Logging level (DEBUG/INFO/WARNING/ERROR) | INFO | No |
| `MIMIR_MAX_WORKERS` | Maximum concurrent workers | 4 | No |
| `MIMIR_TIMEOUT` | Request timeout in seconds | 300 | No |
| `MIMIR_STORAGE_DIR` | Data storage directory | /app/data | No |
| `MIMIR_ENABLE_METRICS` | Enable Prometheus metrics | false | No |
| `MIMIR_API_KEY` | API authentication key | - | Production |

### Resource Limits

#### Development Environment
- **Memory**: 2GB limit, 512MB reserved
- **CPU**: 2.0 cores limit, 0.5 cores reserved

#### Production Environment
- **Memory**: 4GB limit, 1GB reserved
- **CPU**: 4.0 cores limit, 1.0 cores reserved

### Storage Configuration

The system requires persistent storage for:

- **Data Directory**: Repository indexes and metadata
- **Cache Directory**: Temporary processing files and caches
- **Logs Directory**: Application and access logs

```yaml
# Volume configuration in docker-compose.yml
volumes:
  - ${MIMIR_DATA_PATH:-./data}:/app/data
  - ${MIMIR_CACHE_PATH:-./cache}:/app/cache
  - ${MIMIR_LOGS_PATH:-./logs}:/app/logs
```

## Health Checks

### Automated Health Monitoring

The system includes comprehensive health checks:

```bash
# Basic liveness check
./scripts/health-check.sh -c liveness

# Readiness check for load balancers
./scripts/health-check.sh -c readiness

# Detailed health with metrics
./scripts/health-check.sh -c detailed -f json
```

### Health Check Endpoints

- **Liveness**: Container-level health check
- **Readiness**: Service-level readiness verification
- **Detailed**: Comprehensive system metrics

### Monitoring Integration

Health checks integrate with:
- Docker health checks
- Kubernetes liveness/readiness probes
- Load balancer health endpoints
- Prometheus monitoring

## Monitoring and Observability

### Metrics Collection

Prometheus metrics are available at:
- **Development**: http://localhost:9090
- **Production**: http://your-domain:9090

Key metrics include:
- Pipeline execution times
- Request rates and latencies
- System resource utilization
- Error rates and failure modes

### Log Aggregation

Loki aggregates logs from all services:
- **Endpoint**: http://localhost:3100
- **Retention**: 31 days (configurable)
- **Integration**: Grafana dashboards

### Dashboards

Grafana dashboards available at:
- **Development**: http://localhost:3000 (admin/admin)
- **Production**: http://your-domain:3000

Included dashboards:
- System overview and health
- Pipeline performance metrics
- Resource utilization trends
- Error tracking and alerting

## Backup and Recovery

### Data Backup

```bash
# Manual backup
./scripts/deploy.sh -b

# Automated backup (add to cron)
0 2 * * * /path/to/mimir/scripts/deploy.sh -b
```

### Recovery Procedures

```bash
# Restore from backup
docker-compose down
cp -r backups/YYYYMMDD_HHMMSS/data/* data/
docker-compose up -d
```

## Scaling and Performance

### Horizontal Scaling

```bash
# Scale specific services
docker-compose up -d --scale mimir-server=3

# Load balancer configuration required for multiple instances
```

### Performance Tuning

Key parameters for optimization:

```bash
# Worker processes
MIMIR_MAX_WORKERS=8

# Memory allocation
MIMIR_MAX_MEMORY_MB=2048

# I/O configuration
ASYNCIO_MAX_WORKERS=20
FILE_READ_BUFFER_SIZE=16384
```

### Resource Monitoring

Monitor these metrics for scaling decisions:
- CPU utilization (target: <70%)
- Memory usage (target: <80%)
- Request queue depth
- Pipeline execution times

## Security

### Container Security

- Non-root user execution
- Read-only filesystem where possible
- Minimal attack surface (distroless base images)
- Security scanning with Trivy

### Network Security

- Internal Docker network isolation
- NGINX reverse proxy for external access
- Rate limiting and DDoS protection
- SSL/TLS termination

### Secrets Management

```bash
# Environment-based secrets
MIMIR_API_KEY=your-secret-key
MIMIR_JWT_SECRET=your-jwt-secret

# Docker secrets (production)
echo "your-secret" | docker secret create mimir-api-key -
```

## Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check logs
docker-compose logs mimir-server

# Check resource usage
docker stats

# Verify configuration
docker-compose config
```

#### Performance Issues
```bash
# Monitor resource usage
./scripts/health-check.sh -c detailed

# Check pipeline metrics
curl http://localhost:9100/metrics

# Profile application
docker exec -it mimir-server python -m cProfile -o profile.prof
```

#### Storage Issues
```bash
# Check disk space
df -h

# Check permissions
ls -la data/ cache/ logs/

# Clean up old data
docker-compose exec mimir-server find /app/cache -mtime +7 -delete
```

### Log Analysis

```bash
# View all logs
docker-compose logs

# Filter by service
docker-compose logs mimir-server

# Follow logs in real-time
docker-compose logs -f --tail=100

# Export logs for analysis
docker-compose logs > mimir-logs-$(date +%Y%m%d).txt
```

### Performance Profiling

```bash
# CPU profiling
docker exec mimir-server py-spy record -o profile.svg -d 60 -p 1

# Memory profiling
docker exec mimir-server python -m memory_profiler mcp/server.py

# I/O monitoring
docker exec mimir-server iotop -p $(pgrep python)
```

## Maintenance

### Regular Maintenance Tasks

```bash
# Update base images
docker-compose pull
docker-compose up -d

# Clean up old containers
docker system prune -a

# Backup data
./scripts/deploy.sh -b

# Check security updates
docker scan mimir-server:latest
```

### Log Rotation

Configure log rotation in production:

```bash
# Add to /etc/logrotate.d/mimir
/var/log/mimir/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    copytruncate
}
```

## API Documentation

### MCP Protocol Endpoints

The server implements the Model Context Protocol (MCP) with these tools:

- `ensure_repo_index`: Start repository indexing
- `search_repo`: Search indexed repositories
- `ask_index`: Query with natural language
- `get_repo_bundle`: Retrieve complete index
- `cancel`: Cancel ongoing operations

### HTTP Health Endpoints

```bash
# Basic health check
GET /health
Response: {"status": "healthy", "timestamp": "..."}

# Detailed health with metrics
GET /health/detailed
Response: {"status": "ready", "checks": {...}}
```

## Support and Contributing

### Getting Help

1. Check the logs: `docker-compose logs`
2. Run health checks: `./scripts/health-check.sh`
3. Review monitoring dashboards
4. Create GitHub issue with system info

### Development Setup

```bash
# Development environment
./scripts/deploy.sh -e development -t

# With code reloading
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
```

This deployment guide provides a comprehensive foundation for running Mimir in any environment. For specific use cases or advanced configurations, refer to the individual service documentation or create an issue for support.