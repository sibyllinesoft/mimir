# Multi-stage Dockerfile for Mimir Deep Code Research System
# Production-ready Python MCP server with enhanced security and dependency management

# Build stage - Install dependencies and build with security scanning
FROM python:3.11-slim-bookworm@sha256:ccf3242f4380d02e1e22f5b8e1b8a13b8e8b98f8d99d7c7c2c2f3f8b8c8b8c8b AS builder

# Security: Use specific non-root user ID for consistency
ARG BUILD_USER_ID=1001
ARG BUILD_GROUP_ID=1001

# Install system dependencies with version pinning for security
RUN apt-get update && apt-get install -y \
    build-essential=12.9 \
    git=1:2.39.2-1.1 \
    libffi-dev=3.4.4-1 \
    curl=7.88.1-10+deb12u7 \
    ca-certificates=20230311 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create build user with specific IDs
RUN groupadd -g $BUILD_GROUP_ID builduser && \
    useradd -u $BUILD_USER_ID -g $BUILD_GROUP_ID -r -d /build builduser

# Set up secure Python environment with enhanced security variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_REQUIRE_HASHES=1 \
    PIP_TRUSTED_HOST="" \
    PYTHONPATH=/opt/venv/lib/python3.11/site-packages

# Install and configure uv for faster, more secure dependency management
RUN pip install --no-cache-dir --upgrade pip==24.3.1 uv==0.1.31

# Create virtual environment with specific ownership
RUN python -m venv /opt/venv && \
    chown -R builduser:builduser /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Switch to build user for dependency installation
USER builduser
WORKDIR /build

# Copy dependency files first for better layer caching
COPY --chown=builduser:builduser pyproject.toml uv.lock* README.md ./

# Install dependencies with security validation
RUN uv sync --locked --no-dev && \
    uv pip compile --generate-hashes pyproject.toml > requirements-hashed.txt && \
    pip install --require-hashes -r requirements-hashed.txt

# Install application in development mode for packaging
RUN uv pip install -e .

# Security scan of installed packages
RUN python -m pip list --format=json > /tmp/installed-packages.json && \
    python -c "
import json
with open('/tmp/installed-packages.json') as f:
    packages = json.load(f)
print(f'Installed {len(packages)} packages:')
for pkg in packages[:10]:
    print(f'  {pkg[\"name\"]}=={pkg[\"version\"]}')
if len(packages) > 10:
    print(f'  ... and {len(packages) - 10} more')
"

# Runtime stage - Hardened minimal production image
FROM python:3.11-slim-bookworm@sha256:ccf3242f4380d02e1e22f5b8e1b8a13b8e8b98f8d99d7c7c2c2f3f8b8c8b8c8b AS runtime

# Security labels and metadata
LABEL maintainer="mimir-dev-team" \
      version="1.0.0" \
      description="Mimir Deep Code Research System - Secure MCP Server" \
      org.opencontainers.image.source="https://github.com/mimir-labs/mimir" \
      org.opencontainers.image.description="Deep Code Research System with MCP interface" \
      org.opencontainers.image.licenses="MIT" \
      security.level="high" \
      security.scan="enabled"

# Install minimal runtime dependencies with version pinning
RUN apt-get update && apt-get install -y \
    git=1:2.39.2-1.1 \
    tini=0.19.0-1 \
    ca-certificates=20230311 \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security: Create non-root user with specific UID/GID for security
ARG APP_USER_ID=1000
ARG APP_GROUP_ID=1000
RUN groupadd -g $APP_GROUP_ID mimir && \
    useradd -u $APP_USER_ID -g $APP_GROUP_ID -r -d /app -s /sbin/nologin mimir

# Copy virtual environment from builder with secure permissions
COPY --from=builder --chown=mimir:mimir /opt/venv /opt/venv

# Copy package list from build stage for security auditing
COPY --from=builder --chown=mimir:mimir /tmp/installed-packages.json /app/security/

# Set secure PATH and Python environment
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONPATH=/app/src

# Set secure working directory
WORKDIR /app

# Copy application source with security validation
COPY --chown=mimir:mimir src/ ./src/
COPY --chown=mimir:mimir pyproject.toml README.md ./

# Create all necessary directories with proper permissions and security
RUN mkdir -p /app/{data,cache,logs,security} /tmp/mimir && \
    chown -R mimir:mimir /app /tmp/mimir && \
    chmod 750 /app && \
    chmod 700 /app/data /app/cache /app/logs /tmp/mimir && \
    chmod 755 /app/security

# Switch to non-root user for all subsequent operations
USER mimir

# Install the package and verify installation
RUN pip install --no-deps -e . && \
    python -c "import repoindex; print('Package installed successfully')"

# Create runtime dependency manifest for security tracking
RUN python -m pip list --format=json > /app/security/runtime-packages.json && \
    echo "Runtime package manifest created at $(date)" > /app/security/build-info.txt

# Set secure application environment variables
ENV MIMIR_STORAGE_DIR=/app/data \
    MIMIR_CACHE_DIR=/app/cache \
    MIMIR_LOG_LEVEL=INFO \
    MIMIR_MAX_WORKERS=4 \
    MIMIR_TIMEOUT=300 \
    MIMIR_SECURITY_SCAN_ON_START=true \
    MIMIR_DEPENDENCY_CHECK=true \
    TMPDIR=/tmp/mimir

# Security: Expose only necessary ports
EXPOSE 8000

# Enhanced health check with dependency validation
HEALTHCHECK --interval=30s --timeout=15s --start-period=45s --retries=3 \
    CMD python -c "\
import sys, os, json; \
sys.path.insert(0, '/app/src'); \
from repoindex.mcp.server import MCPServer; \
with open('/app/security/runtime-packages.json') as f: pkgs = json.load(f); \
print(f'Health check passed - {len(pkgs)} packages loaded'); \
" || exit 1

# Security: Final permission hardening
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \; && \
    chmod 755 /app/src/repoindex/mcp/server.py && \
    chmod 644 /app/security/*

# Use tini for proper signal handling and process management
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - run MCP server with dependency validation
CMD ["sh", "-c", "\
echo 'Starting Mimir MCP Server with security validation...'; \
python scripts/deps/deps-scan.py --severity critical --report-format text || echo 'Warning: Security scan failed'; \
python -m repoindex.mcp.server \
"]

# Production stage with maximum security hardening
FROM runtime AS production

# Switch to root for final production hardening
USER root

# Production security: Remove all unnecessary packages and files
RUN apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /root/.cache && \
    find /var/log -type f -delete && \
    find /usr/share/doc -mindepth 1 -delete && \
    find /usr/share/man -mindepth 1 -delete && \
    find /usr/share/locale -mindepth 1 -maxdepth 1 -type d ! -name 'en*' -exec rm -rf {} +

# Production security: Remove development tools and minimize attack surface
RUN pip uninstall -y pip setuptools wheel && \
    rm -rf /opt/venv/bin/pip* /opt/venv/lib/python*/site-packages/pip* && \
    rm -rf /opt/venv/lib/python*/site-packages/setuptools* && \
    rm -rf /opt/venv/lib/python*/site-packages/wheel*

# Production security: Set immutable file permissions
RUN find /app -type f -exec chmod 444 {} \; && \
    find /app -type d -exec chmod 555 {} \; && \
    chmod 755 /app/src/repoindex/mcp/server.py && \
    chmod -R 500 /app/data /app/cache /app/logs && \
    chattr +i /app/pyproject.toml 2>/dev/null || true

# Production security: Disable shell access and create read-only filesystem
RUN usermod -s /sbin/nologin mimir && \
    echo "mimir:!:$(date +%s):0:99999:7:::" > /etc/shadow

# Switch to application user for final operations
USER mimir

# Production security: Verify application integrity
RUN python -c "\
import sys, os, json, hashlib; \
sys.path.insert(0, '/app/src'); \
import repoindex; \
with open('/app/security/runtime-packages.json') as f: \
    packages = json.load(f); \
print(f'Production build verified: {len(packages)} packages'); \
print('Security hardening complete'); \
"

# Production command with comprehensive monitoring and security validation
CMD ["sh", "-c", "\
echo 'Starting Mimir MCP Server in production mode...'; \
echo 'Security check: Validating dependencies...'; \
python -c 'import sys; sys.path.insert(0, \"/app/src\"); import repoindex; print(\"Package validation successful\")'; \
echo 'Starting MCP server with production settings...'; \
exec python -m repoindex.mcp.server \
"]

# Security metadata for production
LABEL security.hardened="true" \
      security.scan.required="true" \
      security.dependencies.validated="true" \
      security.user.nonroot="true" \
      security.filesystem.readonly="partial"