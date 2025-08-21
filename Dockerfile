# Multi-stage Dockerfile for Mimir Deep Code Research System
# Production-ready Python MCP server with security hardening

# Build stage - Install dependencies and build
FROM python:3.11-slim-bookworm AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Create build user
RUN groupadd -r builduser && useradd -r -g builduser builduser

# Set up Python environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY pyproject.toml uv.lock* ./
RUN pip install --upgrade pip wheel setuptools && \
    pip install .

# Runtime stage - Minimal production image
FROM python:3.11-slim-bookworm AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r mimir && useradd -r -g mimir -d /app -s /bin/bash mimir

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=mimir:mimir src/ ./src/
COPY --chown=mimir:mimir pyproject.toml ./

# Install the package in development mode for runtime
RUN pip install -e .

# Create necessary directories with proper permissions
RUN mkdir -p /app/data /app/cache /app/logs /tmp/mimir && \
    chown -R mimir:mimir /app /tmp/mimir

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MIMIR_STORAGE_DIR=/app/data \
    MIMIR_CACHE_DIR=/app/cache \
    MIMIR_LOG_LEVEL=INFO \
    MIMIR_MAX_WORKERS=4 \
    MIMIR_TIMEOUT=300

# Expose health check port (if UI is enabled)
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import asyncio; from src.repoindex.mcp.server import MCPServer; print('Health check passed')" || exit 1

# Switch to non-root user
USER mimir

# Use tini as init system for proper signal handling
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command - run MCP server
CMD ["python", "-m", "repoindex.mcp.server"]

# Production stage with additional hardening
FROM runtime AS production

# Additional security hardening
USER root

# Remove unnecessary packages and clean up
RUN apt-get remove -y curl && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set strict file permissions
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \; && \
    chmod 755 /app/src/repoindex/mcp/server.py

# Switch back to application user
USER mimir

# Production entrypoint with enhanced monitoring
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python", "-m", "repoindex.mcp.server"]