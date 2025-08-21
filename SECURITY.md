# Mimir Security Implementation

Comprehensive security hardening for the Mimir Deep Code Research System, providing defense-in-depth protection for production deployments processing untrusted code repositories.

## Overview

This security implementation provides multiple layers of protection against malicious inputs, privilege escalation, data exfiltration, and system compromise. The security framework is designed to assume that code repositories may contain malicious content and applies appropriate controls throughout the indexing pipeline.

## Security Features

### 🛡️ Input Validation & Sanitization
- **Path Traversal Prevention**: Validates all file paths to prevent directory traversal attacks
- **File Type Validation**: Restricts processing to allowed file types and extensions
- **Content Size Limits**: Enforces maximum file sizes to prevent resource exhaustion
- **Schema Validation**: Validates all API inputs against strict JSON schemas
- **Dangerous Pattern Detection**: Scans for potentially harmful content patterns

### 🔒 Sandboxing & Isolation
- **Process Isolation**: Executes external tools in isolated processes with restricted capabilities
- **File System Restrictions**: Limits file access to approved directories only
- **Network Controls**: Restricts network access for sandboxed processes
- **Resource Limits**: Enforces CPU, memory, and file descriptor limits
- **Capability Dropping**: Removes unnecessary process capabilities

### 🔐 Code Execution Safety
- **Safe Parsing**: Parses code files without executing any contained code
- **AST-Based Analysis**: Uses Abstract Syntax Tree analysis instead of code execution
- **Timeout Enforcement**: Applies strict timeouts to all processing operations
- **Memory Bounds**: Enforces memory limits to prevent DoS attacks

### 🔑 Data Protection
- **Secure Handling**: Treats all repository data as potentially sensitive
- **Encryption**: Encrypts vector embeddings and metadata at rest
- **Secrets Management**: Detects and prevents indexing of credential data
- **PII Detection**: Identifies and handles personally identifiable information

### 🔓 Authentication & Authorization
- **API Key Validation**: Secure API key generation and validation using HMAC
- **Rate Limiting**: Token bucket rate limiting per IP and API key
- **Request Validation**: Validates all incoming requests against schemas
- **Audit Logging**: Comprehensive logging of all security events

### 🐳 Container Security
- **Non-Root Execution**: Runs all processes as non-privileged users
- **Minimal Attack Surface**: Uses minimal base images and dependencies
- **Security Scanning**: Automated vulnerability scanning in CI/CD
- **Secrets Management**: Secure handling of encryption keys and credentials

## Architecture

### Security Components

```
├── src/repoindex/security/
│   ├── __init__.py              # Security module initialization
│   ├── validation.py            # Input validation and sanitization
│   ├── sandbox.py              # Process isolation and resource limiting
│   ├── auth.py                 # Authentication and authorization
│   ├── secrets.py              # Credential scanning and secrets management
│   ├── audit.py                # Security event logging and monitoring
│   ├── crypto.py               # Cryptographic operations and encryption
│   ├── config.py               # Centralized security configuration
│   ├── server_middleware.py    # Security middleware for MCP server
│   ├── pipeline_integration.py # Pipeline security integration
│   └── testing.py              # Security testing framework
```

### Integration Points

The security framework integrates with existing Mimir components:

- **MCP Server**: `src/repoindex/mcp/secure_server.py`
- **Pipeline Runner**: `src/repoindex/pipeline/secure_run.py`
- **File Discovery**: Enhanced with credential scanning
- **External Tools**: Sandboxed execution for RepoMapper, Serena, LEANN
- **Data Storage**: Encrypted vector indexes and metadata

## Quick Start

### 1. Security Setup

Run the automated security setup script:

```bash
# Development setup with all features
python setup_security.py --dev --all-features

# Production setup with high security
python setup_security.py --production

# Quick setup with minimal security
python setup_security.py --quick
```

This generates:
- Master encryption key
- API keys for authentication
- Security configuration file
- Environment variables file
- Docker and systemd configurations

### 2. Environment Configuration

Source the generated environment file:

```bash
source ~/.cache/mimir/security/mimir_security.env
```

Or set environment variables manually:

```bash
export MIMIR_MASTER_KEY="your-base64-master-key"
export MIMIR_REQUIRE_AUTH=true
export MIMIR_ENABLE_SANDBOXING=true
export MIMIR_ENABLE_ENCRYPTION=true
export MIMIR_ENABLE_AUDIT_LOGGING=true
```

### 3. Start Secure Server

```bash
# Using environment variables
python -m repoindex.main_secure mcp

# Using configuration file
python -m repoindex.main_secure mcp --config ~/.cache/mimir/security/security_config.json

# Disable security for development (NOT recommended for production)
python -m repoindex.main_secure mcp --no-security
```

### 4. Secure Repository Indexing

```bash
# Index repository with security
python -m repoindex.main_secure index /path/to/repo --language ts

# Search with authentication
python -m repoindex.main_secure search <index-id> "search query"
```

## Configuration

### Security Configuration File

The security configuration file (`security_config.json`) controls all security features:

```json
{
  "require_authentication": true,
  "enable_sandboxing": true,
  "enable_index_encryption": true,
  "enable_credential_scanning": true,
  "enable_audit_logging": true,
  "max_file_size": 104857600,
  "max_query_length": 10000,
  "global_rate_limit": 1000,
  "ip_rate_limit": 100,
  "max_memory_mb": 1024,
  "max_cpu_time_seconds": 300
}
```

### Environment Variables

All configuration options can be set via environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `MIMIR_MASTER_KEY` | Base64-encoded encryption key | Required for encryption |
| `MIMIR_REQUIRE_AUTH` | Enable API authentication | `true` |
| `MIMIR_ENABLE_SANDBOXING` | Enable process sandboxing | `true` |
| `MIMIR_ENABLE_ENCRYPTION` | Enable index encryption | `false` |
| `MIMIR_ENABLE_CREDENTIAL_SCANNING` | Enable credential detection | `true` |
| `MIMIR_ALLOWED_BASE_PATHS` | Comma-separated allowed paths | Empty (all paths) |
| `MIMIR_MAX_FILE_SIZE` | Maximum file size in bytes | `104857600` (100MB) |
| `MIMIR_MAX_QUERY_LENGTH` | Maximum query length | `10000` |
| `MIMIR_GLOBAL_RATE_LIMIT` | Global requests per minute | `1000` |
| `MIMIR_IP_RATE_LIMIT` | IP requests per minute | `100` |
| `MIMIR_API_KEY_RATE_LIMIT` | API key requests per minute | `200` |
| `MIMIR_MAX_MEMORY_MB` | Sandbox memory limit (MB) | `1024` |
| `MIMIR_MAX_CPU_TIME` | Sandbox CPU time limit (seconds) | `300` |
| `MIMIR_AUDIT_LOG_FILE` | Audit log file path | `~/.cache/mimir/security/audit.log` |

## Security Controls

### Input Validation

All user inputs are validated against strict schemas:

```python
# Path validation
path_validator = PathValidator(
    allowed_base_paths=["/safe/path"],
    max_path_length=4096,
    max_filename_length=255
)

# Content validation
content_validator = ContentValidator(
    max_file_size=100 * 1024 * 1024,  # 100MB
    allowed_extensions=[".ts", ".js", ".py"],
    dangerous_patterns=["eval(", "exec(", "__import__"]
)

# Schema validation
schema_validator = SchemaValidator()
result = schema_validator.validate_data(data, schema)
```

### Process Sandboxing

External tools are executed in isolated environments:

```python
# Resource limits
resource_limiter = ResourceLimiter(
    max_memory=1024 * 1024 * 1024,  # 1GB
    max_cpu_time=300,  # 5 minutes
    max_wall_time=600,  # 10 minutes
    max_open_files=1024,
    max_processes=32
)

# Process isolation
isolator = ProcessIsolator(
    allowed_paths=["/safe/directory"],
    resource_limits=resource_limiter,
    timeout=300
)

result = await isolator.execute_isolated(
    command=["external-tool", "args"],
    work_dir=Path("/safe/directory")
)
```

### Credential Scanning

Automatic detection of secrets and credentials:

```python
scanner = CredentialScanner()

# Scan text content
credentials = scanner.scan_text(file_content)

# Scan file
credentials = await scanner.scan_file_async(file_path)

# Scan directory
results = await scanner.scan_directory_async(directory_path)
```

Detects patterns for:
- API keys (AWS, GitHub, OpenAI, etc.)
- Private keys (RSA, SSH, etc.)
- Database credentials
- Authentication tokens
- Cloud service credentials

### Encryption

Data is encrypted at rest using AES-256-GCM:

```python
crypto_manager = CryptoManager()
index_encryption = IndexEncryption(crypto_manager)

# Encrypt vector embeddings
encrypted_embeddings = index_encryption.encrypt_embeddings(
    embeddings, 
    metadata={"source": "leann", "timestamp": "2025-01-19"}
)

# Encrypt metadata
encrypted_metadata = index_encryption.encrypt_metadata(
    metadata,
    metadata={"type": "pipeline_data"}
)
```

### Authentication & Rate Limiting

API key authentication with rate limiting:

```python
# API key validation
api_key_validator = APIKeyValidator()
is_valid = await api_key_validator.validate_key(api_key)

# Rate limiting
rate_limiter = RateLimiter(
    global_limit=1000,  # requests per minute
    ip_limit=100,       # requests per minute per IP
    key_limit=200       # requests per minute per API key
)

allowed = await rate_limiter.check_rate_limit("user_id", "ip_address")
```

### Audit Logging

Comprehensive security event logging:

```python
security_auditor = get_security_auditor()

# Record security events
security_auditor.record_event(SecurityEvent(
    event_type=SecurityEventType.FILE_ACCESS,
    component="file_discovery",
    severity="info",
    message="File accessed for indexing",
    metadata={
        "file_path": "/path/to/file",
        "user_id": "user123",
        "ip_address": "192.168.1.1"
    }
))

# Generate security report
report = security_auditor.generate_report()
```

## Production Deployment

### Docker Deployment

Use the generated Docker Compose configuration:

```bash
# Start secure container
docker-compose -f ~/.cache/mimir/security/docker-compose.security.yml up -d

# View logs
docker-compose logs -f mimir-secure

# Stop container
docker-compose down
```

The Docker configuration includes:
- Non-root user execution
- Read-only root filesystem
- Dropped capabilities
- Resource limits
- Security profiles

### Systemd Service

Install as a systemd service for production:

```bash
# Copy service file
sudo cp ~/.cache/mimir/security/mimir-secure.service /etc/systemd/system/

# Enable and start service
sudo systemctl enable mimir-secure
sudo systemctl start mimir-secure

# Check status
sudo systemctl status mimir-secure

# View logs
sudo journalctl -u mimir-secure -f
```

### Security Monitoring

Monitor security events and metrics:

```bash
# Generate security audit report
python -m repoindex.main_secure audit --report --output security_report.json

# View audit log
tail -f ~/.cache/mimir/security/audit.log

# Monitor rate limiting
grep "RATE_LIMIT_EXCEEDED" ~/.cache/mimir/security/audit.log

# Check for security violations
grep "SECURITY_VIOLATION" ~/.cache/mimir/security/audit.log
```

## Security Testing

### Automated Security Tests

Run the comprehensive security test suite:

```python
from src.repoindex.security.testing import SecurityTestSuite

# Run all security tests
test_suite = SecurityTestSuite()
results = await test_suite.run_all_tests()

# Run specific test categories
validation_results = await test_suite.test_input_validation()
auth_results = await test_suite.test_authentication()
sandbox_results = await test_suite.test_sandboxing()
```

### Manual Security Testing

Test security controls manually:

```bash
# Test path traversal protection
python -m repoindex.main_secure index "../../../etc/passwd"

# Test file size limits
python -m repoindex.main_secure index /path/to/large/file

# Test rate limiting
for i in {1..200}; do curl -X POST http://localhost:8000/api/search; done

# Test authentication
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'

# Test with API key
curl -X POST http://localhost:8000/api/search \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}'
```

### Penetration Testing

For comprehensive security assessment:

1. **Static Analysis**: Use `bandit`, `semgrep`, and `safety` for code analysis
2. **Dynamic Analysis**: Use `OWASP ZAP` for API security testing
3. **Container Scanning**: Use `trivy` or `clair` for container vulnerabilities
4. **Dependency Scanning**: Use `snyk` or `safety` for dependency vulnerabilities

## Threat Model

### Threat Actors

1. **Malicious Repository Owners**: Attempting to compromise the indexing system
2. **External Attackers**: Trying to gain unauthorized access to the system
3. **Insider Threats**: Legitimate users attempting to escalate privileges
4. **Automated Attacks**: Bots attempting to exploit vulnerabilities

### Attack Vectors

1. **Code Injection**: Malicious code in repository files
2. **Path Traversal**: Attempts to access files outside allowed directories
3. **Resource Exhaustion**: Large files or complex processing to DoS the system
4. **Credential Theft**: Attempts to extract API keys or encryption keys
5. **Privilege Escalation**: Attempts to gain higher system privileges
6. **Data Exfiltration**: Attempts to access other users' data

### Mitigations

Each attack vector is addressed by multiple security controls:

- **Defense in Depth**: Multiple layers of security controls
- **Principle of Least Privilege**: Minimal permissions for all operations
- **Input Validation**: Strict validation of all user inputs
- **Process Isolation**: Sandboxed execution of external tools
- **Audit Logging**: Comprehensive logging for detection and investigation
- **Rate Limiting**: Protection against automated attacks
- **Encryption**: Protection of data at rest and in transit

## Compliance

The security implementation supports compliance with:

- **SOC 2 Type II**: Security controls and audit logging
- **ISO 27001**: Information security management
- **GDPR**: Data protection and privacy controls
- **HIPAA**: Healthcare data protection (when applicable)
- **FedRAMP**: Federal security requirements (with additional controls)

## Security Roadmap

### Implemented ✅
- Input validation and sanitization
- Process sandboxing and isolation
- Credential scanning and detection
- Data encryption at rest
- API authentication and rate limiting
- Comprehensive audit logging
- Security testing framework

### In Progress 🚧
- Integration with existing pipeline stages
- Performance optimization of security controls
- Advanced threat detection algorithms
- Container security hardening

### Planned 📋
- Network-level security controls
- Advanced behavioral analysis
- Machine learning-based anomaly detection
- Integration with external security tools
- Compliance automation and reporting

## Support

### Security Issues

Report security vulnerabilities privately to: security@mimir-research.com

**Do not create public GitHub issues for security vulnerabilities.**

### Documentation

- Security implementation details: `src/repoindex/security/`
- Configuration reference: `SECURITY.md` (this file)
- API documentation: `docs/api/security.md`
- Deployment guides: `docs/deployment/`

### Community

- Security discussions: GitHub Discussions (Security category)
- Implementation questions: GitHub Issues (with security label)
- Feature requests: GitHub Issues (enhancement label)

---

**Remember**: Security is an ongoing process, not a one-time implementation. Regularly review and update security controls, monitor for new threats, and keep dependencies updated.