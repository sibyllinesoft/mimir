# Mimir-Lens Production Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying and operating the Mimir-Lens integrated system in production environments. The integration combines Mimir's deep code analysis capabilities with Lens's high-performance search engine for enterprise-grade code intelligence.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Prerequisites](#prerequisites)
3. [Configuration Management](#configuration-management)
4. [Deployment Strategies](#deployment-strategies)
5. [Monitoring & Observability](#monitoring--observability)
6. [Scaling & Performance](#scaling--performance)
7. [Security Considerations](#security-considerations)
8. [Backup & Recovery](#backup--recovery)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance Procedures](#maintenance-procedures)

---

## Architecture Overview

### System Components

The integrated system consists of:

**Core Services:**
- **Mimir MCP Server** (Port 8000): Python-based code analysis engine
- **Lens Search Engine** (Port 3000): Node.js-based high-performance search
- **PostgreSQL Database** (Port 5432): Persistent data storage
- **Redis Cache** (Port 6379): Session and query caching
- **NATS Message Queue** (Port 4222): Distributed processing coordination

**Monitoring Stack:**
- **Prometheus** (Port 9090): Metrics collection
- **Grafana** (Port 3001): Visualization and dashboards
- **Jaeger** (Port 16686): Distributed tracing
- **OpenTelemetry Collector** (Port 4317/4318): Observability data processing
- **Loki** (Port 3100): Log aggregation

**Infrastructure:**
- **Nginx** (Ports 80/443): Reverse proxy and load balancer

### Network Architecture

```
Internet
    ↓
[Nginx Reverse Proxy]
    ↓
┌─────────────────────────────────────┐
│        Application Network          │
│  ┌─────────┐    ┌─────────────┐     │
│  │  Mimir  │←→  │    Lens     │     │
│  │  :8000  │    │   :3000     │     │
│  └─────────┘    └─────────────┘     │
│       ↓              ↓              │
│  ┌─────────┐    ┌─────────────┐     │
│  │PostgreSQL│    │    NATS     │     │
│  │  :5432   │    │   :4222     │     │
│  └─────────┘    └─────────────┘     │
│       ↓                             │
│  ┌─────────┐                        │
│  │  Redis  │                        │
│  │  :6379  │                        │
│  └─────────┘                        │
└─────────────────────────────────────┘
         ↓
┌─────────────────────────────────────┐
│       Monitoring Network            │
│ [Prometheus][Grafana][Jaeger][Loki] │
└─────────────────────────────────────┘
```

---

## Prerequisites

### System Requirements

**Minimum Production Requirements:**
- **CPU**: 8 cores (16 vCPU recommended)
- **Memory**: 16GB RAM (32GB recommended)
- **Storage**: 200GB SSD (1TB recommended)
- **Network**: 1Gbps connection

**Operating System:**
- Ubuntu 22.04 LTS or later
- CentOS 8+ / RHEL 8+
- Docker-compatible Linux distribution

### Software Dependencies

```bash
# Docker and Docker Compose
sudo apt update
sudo apt install docker.io docker-compose-v2
sudo systemctl enable docker
sudo usermod -aG docker $USER

# Additional utilities
sudo apt install curl jq htop iotop netstat-nat
```

### Port Requirements

Ensure the following ports are available:
- **80/443**: HTTP/HTTPS (Nginx)
- **8000**: Mimir API
- **3000**: Lens API
- **3001**: Grafana Dashboard
- **9090**: Prometheus
- **16686**: Jaeger UI
- **5432**: PostgreSQL (internal)
- **6379**: Redis (internal)
- **4222**: NATS (internal)

---

## Configuration Management

### Environment Configurations

Three environment configurations are provided:

1. **Development** (`.env.development`): Local development with debugging
2. **Staging** (`.env.staging`): Pre-production testing environment  
3. **Production** (`.env.production`): Full production deployment

### Production Configuration Setup

1. **Copy and customize production environment:**

```bash
cp .env.production .env
```

2. **Update critical settings in `.env`:**

```bash
# Security - MUST BE CHANGED
POSTGRES_PASSWORD=YOUR_SECURE_DB_PASSWORD
REDIS_PASSWORD=YOUR_SECURE_REDIS_PASSWORD
MIMIR_API_KEY=YOUR_SECURE_MIMIR_KEY
GRAFANA_ADMIN_PASSWORD=YOUR_SECURE_GRAFANA_PASSWORD
JWT_SECRET=YOUR_SECURE_JWT_SECRET

# Domain configuration
CORS_ORIGINS=https://your-domain.com
SSL_CERT_PATH=/etc/ssl/certs/your-domain.crt
SSL_KEY_PATH=/etc/ssl/private/your-domain.key

# External integrations
GOOGLE_API_KEY=your-gemini-api-key
SLACK_WEBHOOK_URL=your-slack-webhook
EMAIL_ALERTS_TO=alerts@your-domain.com
```

3. **Set resource limits based on your infrastructure:**

```bash
# Container resource limits
DOCKER_MEMORY_LIMIT_MIMIR=4g
DOCKER_CPU_LIMIT_MIMIR=4.0
DOCKER_MEMORY_LIMIT_LENS=6g
DOCKER_CPU_LIMIT_LENS=6.0
```

### SSL/TLS Configuration

**Option 1: Let's Encrypt (Recommended)**

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Obtain certificate
sudo certbot certonly --standalone -d your-domain.com

# Configure paths in .env
SSL_CERT_PATH=/etc/letsencrypt/live/your-domain.com/fullchain.pem
SSL_KEY_PATH=/etc/letsencrypt/live/your-domain.com/privkey.pem
```

**Option 2: Custom Certificate**

Place your certificate files in `ops/nginx/ssl/` and update the paths in your configuration.

---

## Deployment Strategies

### Option 1: Automated Deployment Script (Recommended)

```bash
# Production deployment with blue-green strategy
./scripts/deploy-integrated.sh --environment production --version v1.0.0 --strategy blue-green

# Staging deployment
./scripts/deploy-integrated.sh --environment staging --version latest --strategy rolling

# Dry run (test without executing)
./scripts/deploy-integrated.sh --dry-run --environment production --version v1.0.0
```

### Option 2: Manual Docker Compose Deployment

```bash
# Pull latest images
docker-compose -f docker-compose.integrated.yml pull

# Start services
docker-compose -f docker-compose.integrated.yml --env-file .env up -d

# Verify deployment
docker-compose -f docker-compose.integrated.yml ps
```

### Deployment Verification

After deployment, verify all services:

```bash
# Health checks
curl -f http://localhost:8000/health  # Mimir
curl -f http://localhost:3000/health  # Lens
curl -f http://localhost:9090/-/healthy  # Prometheus

# Service connectivity test
python3 tests/integration/test_integrated_system.py
```

---

## Monitoring & Observability

### Dashboard Access

- **Grafana**: `https://your-domain.com/grafana/` (admin/your-password)
- **Prometheus**: `https://your-domain.com/prometheus/`
- **Jaeger**: `https://your-domain.com/jaeger/`

### Key Metrics to Monitor

**Application Metrics:**
- Request rate and latency (p95, p99)
- Error rates by service
- Integration success rates (Mimir ↔ Lens)
- Query performance and cache hit rates

**Infrastructure Metrics:**
- CPU and memory utilization
- Disk I/O and space usage
- Network throughput
- Container health and restart rates

**Business Metrics:**
- Active users and sessions
- Code repositories indexed
- Search queries executed
- API usage by endpoint

### Alert Thresholds

Production alerts are configured for:
- Service downtime (>1 minute)
- High error rate (>5%)
- High response time (p95 >5s for Mimir, >2s for Lens)
- Resource utilization (>80% CPU, >85% memory)
- Disk space (>90% usage)

### Log Aggregation

Logs are automatically collected and stored in Loki:

```bash
# Query logs using LogQL
curl -G -s "http://localhost:3100/loki/api/v1/query" \
  --data-urlencode 'query={service="mimir"} |= "ERROR"'

# Or use Grafana Explore interface
```

---

## Scaling & Performance

### Horizontal Scaling

**Scale Mimir instances:**

```yaml
# docker-compose.integrated.yml
services:
  mimir:
    deploy:
      replicas: 3  # Multiple instances
  
  # Add load balancer configuration
  nginx:
    # Update upstream configuration for multiple backends
```

**Scale Lens instances:**

```yaml
services:
  lens:
    deploy:
      replicas: 2  # Multiple instances
```

### Vertical Scaling

Update resource limits in `.env`:

```bash
# Increase memory limits
DOCKER_MEMORY_LIMIT_MIMIR=8g
DOCKER_MEMORY_LIMIT_LENS=12g

# Increase CPU limits  
DOCKER_CPU_LIMIT_MIMIR=8.0
DOCKER_CPU_LIMIT_LENS=12.0
```

### Database Scaling

**PostgreSQL optimization:**

```bash
# Update ops/postgres/tuning.conf for your hardware
shared_buffers = 2GB      # 25% of total RAM
effective_cache_size = 6GB # 75% of total RAM
work_mem = 32MB           # Based on concurrent connections
```

**Redis optimization:**

```bash
# Update ops/redis/redis.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
```

### Performance Tuning Checklist

- [ ] Database connection pooling optimized
- [ ] Redis cache hit rate >80%
- [ ] Nginx gzip compression enabled
- [ ] Container resource limits appropriate
- [ ] Disk I/O patterns optimized
- [ ] Network latency minimized
- [ ] JIT compilation enabled (where applicable)

---

## Security Considerations

### Network Security

```bash
# Firewall configuration (UFW example)
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw enable
```

### Container Security

Security measures implemented:
- Non-root container execution
- Read-only root filesystems where possible
- Resource limits to prevent DoS
- Secret management through environment variables
- Network segmentation between services

### API Security

- JWT-based authentication
- API rate limiting (100 requests/minute)
- Input validation and sanitization
- CORS restrictions to allowed domains
- TLS encryption for all communications

### Database Security

- Encrypted connections (TLS)
- Least-privilege user accounts
- Regular security updates
- Audit logging enabled
- Backup encryption

### Monitoring Security Events

Security events are automatically monitored:
- Failed authentication attempts
- Unusual request patterns
- Resource exhaustion attempts
- Container restart anomalies

---

## Backup & Recovery

### Automated Backups

Backups are created automatically:

```bash
# Database backup (daily at 2 AM)
0 2 * * * /opt/mimir-lens/scripts/backup-database.sh

# Configuration backup (weekly)
0 0 * * 0 /opt/mimir-lens/scripts/backup-config.sh

# Volume backup (daily)
0 3 * * * /opt/mimir-lens/scripts/backup-volumes.sh
```

### Manual Backup

```bash
# Create immediate backup
./scripts/deploy-integrated.sh --backup-only

# Database backup
docker-compose exec postgres pg_dump -U mimir_user mimir > backup_$(date +%Y%m%d).sql

# Configuration backup  
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env* ops/ docker-compose*.yml
```

### Recovery Procedures

**Full system recovery:**

```bash
# 1. Stop services
docker-compose -f docker-compose.integrated.yml down

# 2. Restore configuration
tar -xzf config_backup_YYYYMMDD.tar.gz

# 3. Restore database
docker-compose -f docker-compose.integrated.yml up -d postgres
docker-compose exec postgres psql -U mimir_user mimir < backup_YYYYMMDD.sql

# 4. Restart all services
docker-compose -f docker-compose.integrated.yml up -d

# 5. Verify recovery
./tests/integration/test_integrated_system.py
```

**Point-in-time recovery:**

PostgreSQL supports point-in-time recovery if Write-Ahead Logging (WAL) archiving is configured.

---

## Troubleshooting

### Common Issues

**Services won't start:**

```bash
# Check Docker daemon
sudo systemctl status docker

# Check container logs
docker-compose -f docker-compose.integrated.yml logs mimir
docker-compose -f docker-compose.integrated.yml logs lens

# Check resource usage
docker stats

# Check disk space
df -h
```

**Database connection issues:**

```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready -U mimir_user

# Check connection from application
docker-compose exec mimir python -c "import asyncpg; print('DB accessible')"

# Review connection pool settings
grep -E "POSTGRES.*CONNECTION" .env
```

**High memory usage:**

```bash
# Monitor container memory
docker stats --no-stream

# Check for memory leaks
docker-compose exec mimir ps aux | sort -nrk 4

# Restart services if necessary
docker-compose restart mimir lens
```

**Integration connectivity issues:**

```bash
# Test Mimir -> Lens connectivity
docker-compose exec mimir curl -f http://lens:3000/health

# Check NATS connectivity
docker-compose exec lens nats-cli ping

# Review integration logs
docker-compose logs | grep -i "lens\|integration"
```

### Debugging Tools

**Log analysis:**

```bash
# Real-time log following
docker-compose -f docker-compose.integrated.yml logs -f mimir lens

# Error log filtering
docker-compose logs | grep -i error | tail -n 50

# Performance log analysis
docker-compose logs | grep -E "took|duration|latency"
```

**Performance analysis:**

```bash
# Container resource usage
docker exec <container_id> top -p 1

# Network connectivity
docker exec <container_id> netstat -tulpn

# Disk I/O
docker exec <container_id> iotop -o
```

### Health Check Commands

```bash
# Quick health check script
#!/bin/bash
services=("mimir:8000" "lens:3000" "prometheus:9090" "grafana:3000")

for service in "${services[@]}"; do
    name=${service%:*}
    port=${service#*:}
    if curl -f -s "http://localhost:$port/health" > /dev/null; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is unhealthy"
    fi
done
```

---

## Maintenance Procedures

### Regular Maintenance Tasks

**Daily:**
- [ ] Check service health status
- [ ] Review error logs
- [ ] Monitor disk usage
- [ ] Verify backup completion

**Weekly:**
- [ ] Update container images
- [ ] Clean up old logs and backups
- [ ] Review performance metrics
- [ ] Security scan results

**Monthly:**
- [ ] Database maintenance (VACUUM, ANALYZE)
- [ ] Certificate renewal check
- [ ] Capacity planning review
- [ ] Disaster recovery test

### Update Procedures

**Security updates:**

```bash
# Update base images
docker-compose pull

# Apply security patches
sudo apt update && sudo apt upgrade -y

# Restart services with rolling update
./scripts/deploy-integrated.sh --strategy rolling --version latest
```

**Feature updates:**

```bash
# Deploy new version
./scripts/deploy-integrated.sh --environment production --version v1.2.0

# Run integration tests
python3 tests/integration/test_integrated_system.py

# Monitor for issues
watch 'docker-compose ps && echo && curl -s http://localhost:8000/health'
```

### Database Maintenance

```bash
# PostgreSQL maintenance (run during low-traffic periods)
docker-compose exec postgres psql -U mimir_user -d mimir -c "VACUUM ANALYZE;"

# Index maintenance
docker-compose exec postgres psql -U mimir_user -d mimir -c "REINDEX DATABASE mimir;"

# Statistics update
docker-compose exec postgres psql -U mimir_user -d mimir -c "ANALYZE;"
```

### Log Rotation

```bash
# Configure logrotate for Docker logs
sudo cat > /etc/logrotate.d/docker-containers << EOF
/var/lib/docker/containers/*/*.log {
    rotate 7
    daily
    compress
    size 10M
    missingok
    delaycompress
    copytruncate
}
EOF
```

---

## Support & Escalation

### Monitoring Alerts

Alerts are configured to notify the team via:
- **Slack**: `#mimir-lens-alerts` channel
- **Email**: `alerts@your-domain.com`
- **PagerDuty**: Critical production issues

### Escalation Procedures

**Level 1 (Warning):** High resource usage, minor performance degradation
- Automated monitoring alerts
- Self-healing where possible
- Team notification

**Level 2 (Critical):** Service downtime, data loss risk
- Immediate team notification
- Automated rollback if possible
- Incident response initiated

**Level 3 (Emergency):** Security breach, total system failure
- Executive notification
- All-hands response
- External vendor engagement if needed

### Contact Information

- **On-call Engineer**: `+1-555-ON-CALL`
- **DevOps Team**: `devops@your-domain.com`
- **Security Team**: `security@your-domain.com`
- **Management**: `management@your-domain.com`

---

## Additional Resources

- **System Architecture**: `docs/ARCHITECTURE.md`
- **API Documentation**: `API_REFERENCE.md`
- **Security Guidelines**: `SECURITY.md`
- **Contributing Guidelines**: `CONTRIBUTING.md`
- **Troubleshooting**: `TROUBLESHOOTING.md`

---

*This documentation is maintained by the DevOps team and updated with each release. Last updated: $(date)*