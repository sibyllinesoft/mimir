# Mimir Maintenance Commands

## Quick Reference for Repository Maintenance

### Daily Development
```bash
# Setup pre-commit hooks (one-time)
pre-commit install

# Run quality checks before committing
pre-commit run --all-files

# Run test suite
pytest tests/ -v

# Type check
mypy src/

# Security scan
bandit -r src/
```

### Monitoring Stack
```bash
# Start monitoring infrastructure
docker-compose -f docker-compose.monitoring.yml up -d

# View live traces
python scripts/trace_viewer.py

# Stop monitoring
docker-compose -f docker-compose.monitoring.yml down

# Check NATS status
docker-compose -f docker-compose.monitoring.yml logs nats
```

### CI/CD Validation
```bash
# Test CI workflow locally (if act is installed)
act -W .github/workflows/cleanup-enforcement.yml

# Check workflow status
gh workflow list
gh workflow run cleanup-enforcement.yml
```

### Repository Health
```bash
# Check repository structure compliance
find tests/ -name "test_*.py" | wc -l  # Should be >0
ls -la | grep -E "(CODEOWNERS|CONTRIBUTING)" # Should exist
pre-commit run --all-files  # Should pass

# Validate monitoring integration
python -c "from repoindex.mcp.monitored_server import MonitoredMCPServer; print('✅ Monitoring ready')"

# Check security compliance
bandit -r src/ -f json | jq '.results | length'  # Should be 0
```

### Performance Monitoring
```bash
# Run performance benchmarks
pytest tests/benchmarks/ -v

# Profile memory usage
python -m memory_profiler scripts/profile_server.py

# Check for performance regressions
pytest --benchmark-only tests/
```

### Dependency Management
```bash
# Update dependencies
uv sync --upgrade

# Security audit
pip-audit
safety check

# Check for unused dependencies
pip-check
```

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Commit with proper format
git commit -m "feat(scope): description"

# Pre-merge validation
git checkout main
git pull origin main
git checkout feature/your-feature-name
git rebase main
pre-commit run --all-files
pytest tests/ -v
```

### Rollback Procedures
```bash
# Emergency rollback to cleanup baseline
git checkout main
git reset --hard cleanup-backup
git push --force-with-lease origin main

# Rollback specific changes
git revert <commit-hash>

# Disable monitoring temporarily
# Edit pyproject.toml, comment out sibylline-skald
# Use mimir-server instead of mimir-monitored-server
```

### Troubleshooting

#### Pre-commit Issues
```bash
# Reinstall hooks
pre-commit uninstall
pre-commit install

# Run specific hook
pre-commit run black --all-files
pre-commit run ruff --all-files
```

#### Monitoring Issues
```bash
# Check NATS connection
curl -s http://localhost:8222/connz | jq .

# Restart monitoring stack
docker-compose -f docker-compose.monitoring.yml restart

# Check trace emission
tail -f logs/traces.log
```

#### CI/CD Issues
```bash
# Check workflow syntax
gh workflow view cleanup-enforcement.yml

# Re-run failed workflow
gh workflow run cleanup-enforcement.yml

# Debug action logs
gh run list --workflow=cleanup-enforcement.yml
gh run view <run-id>
```

### Quality Metrics
```bash
# Test coverage report
pytest --cov=src/ --cov-report=html tests/

# Code complexity
radon cc src/ -a

# Documentation coverage
interrogate src/

# Security scan summary
bandit -r src/ -f txt
```

---

## Automation Scripts

### setup-dev-env.sh
```bash
#!/bin/bash
# One-time development setup
uv sync --extra dev --extra test
pre-commit install
docker-compose -f docker-compose.monitoring.yml up -d
echo "✅ Development environment ready"
```

### daily-health-check.sh
```bash
#!/bin/bash
# Daily repository health validation
echo "🔍 Running daily health check..."
pre-commit run --all-files || exit 1
pytest tests/unit/ -q || exit 1
bandit -r src/ -f json > /dev/null || exit 1
echo "✅ All health checks passed"
```

### deploy-monitoring.sh
```bash
#!/bin/bash
# Deploy monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d
sleep 10
python -c "import requests; r=requests.get('http://localhost:8222/varz'); exit(0 if r.status_code==200 else 1)"
echo "✅ Monitoring stack deployed and healthy"
```

---

## Integration Points

### Claude Code Integration
- Use `mimir-monitored-server` for full tracing
- Traces appear in real-time via NATS JetStream
- Performance metrics captured automatically

### GitHub Actions Integration
- All PRs run cleanup enforcement workflow
- Security scans block merges on vulnerabilities
- Test coverage tracked and reported

### Development Workflow Integration
- Pre-commit hooks ensure code quality
- Type checking prevents runtime errors
- Security scanning prevents vulnerabilities

---

**Keep this file updated as maintenance procedures evolve.**