---
name: 🐛 Bug Report
about: Create a report to help us improve Mimir
title: '[BUG] '
labels: ['bug', 'needs-triage']
assignees: ''
---

# 🐛 Bug Report

## 📋 Summary

A clear and concise description of what the bug is.

## 🔍 Current Behavior

What is actually happening? Please be as specific as possible.

## ✅ Expected Behavior

What should be happening instead?

## 🔄 Steps to Reproduce

Steps to reproduce the behavior:

1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Minimal reproduction example:**
```python
# Provide a minimal code example that reproduces the issue
from repoindex.pipeline.run import main

# Your reproduction code here
```

## 🌍 Environment

**System Information:**
- OS: [e.g. Ubuntu 22.04, macOS 13, Windows 11]
- Python Version: [e.g. 3.11.5]
- Mimir Version: [e.g. v1.0.0]
- Container Runtime: [e.g. Docker 24.0.6, Podman 4.6.1]

**Dependencies:**
```bash
# Output of: pip list | grep -E "(mcp|fastapi|pydantic|tree-sitter)"
mcp==1.0.0
fastapi==0.104.0
# ... other relevant dependencies
```

**Environment Variables:**
```bash
# Relevant environment variables (remove sensitive values)
MIMIR_LOG_LEVEL=INFO
MIMIR_MAX_WORKERS=4
# ... other relevant env vars
```

## 📸 Screenshots/Logs

If applicable, add screenshots or log output to help explain your problem.

**Error Logs:**
```
# Paste relevant error logs here
# Include timestamps and full stack traces if available
```

**Console Output:**
```
# Paste relevant console output here
```

## 🧪 Additional Context

Add any other context about the problem here.

### Related Issues
- Links to related issues
- Similar bugs or patterns

### Attempted Solutions
- What have you tried to fix this issue?
- Any workarounds that work?

### Impact Assessment
- [ ] Blocks development
- [ ] Blocks testing
- [ ] Blocks deployment
- [ ] Affects performance
- [ ] Affects security
- [ ] User experience impact
- [ ] Data integrity concern

## 🔧 Debugging Information

### MCP Server Information
```bash
# Output of health check or server status
python -c "
import asyncio
from repoindex.mcp.server import MCPServer
# Add any diagnostic code
"
```

### Pipeline Information
```bash
# If related to pipeline processing
python -c "
from repoindex.pipeline.run import main
# Add diagnostic information
"
```

### Container Information (if applicable)
```bash
# Container logs
docker logs mimir-container

# Container inspect
docker inspect mimir-container
```

## 🎯 Acceptance Criteria

What needs to be done to consider this bug fixed?

- [ ] Bug is reproducible
- [ ] Root cause identified
- [ ] Fix implemented
- [ ] Tests added to prevent regression
- [ ] Documentation updated (if needed)
- [ ] Fix verified in target environment

---

**Labels to add:**
- Priority: `priority/low` | `priority/medium` | `priority/high` | `priority/critical`
- Component: `component/mcp-server` | `component/pipeline` | `component/security` | `component/ui` | `component/monitoring`
- Type: `type/functionality` | `type/performance` | `type/security` | `type/usability`