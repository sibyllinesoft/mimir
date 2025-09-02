# Security Model Diagram

## Overview
This diagram illustrates the comprehensive security architecture of Mimir, including authentication, authorization, input validation, sandboxing, and audit logging.

```mermaid
graph TB
    %% Client Layer
    subgraph "Client Layer"
        C1[AI Agent]
        C2[Developer CLI]
        C3[IDE Extension]
    end

    %% Authentication Layer
    subgraph "Authentication Layer"
        A1[API Key Validation]
        A2[Token Management]
        A3[Session Control]
        A4[Rate Limiting]
    end

    %% Authorization Layer
    subgraph "Authorization Layer"
        Z1[Permission Matrix]
        Z2[Resource Access Control]
        Z3[Operation Scoping]
        Z4[Privilege Escalation Prevention]
    end

    %% Input Validation Layer
    subgraph "Input Validation"
        V1[Schema Validation]
        V2[Path Sanitization]
        V3[Injection Prevention]
        V4[Type Safety Enforcement]
    end

    %% Sandbox Execution
    subgraph "Execution Sandbox"
        S1[Process Isolation]
        S2[File System Restrictions]
        S3[Network Isolation]
        S4[Resource Limits]
        S5[Subprocess Control]
    end

    %% Secure MCP Server
    subgraph "Secure MCP Server"
        M1[Security Middleware]
        M2[Request Interceptor]
        M3[Response Sanitizer]
        M4[Error Handler]
    end

    %% Pipeline Security
    subgraph "Pipeline Security"
        P1[Stage Isolation]
        P2[Tool Sandboxing]
        P3[Data Encryption]
        P4[Secure IPC]
    end

    %% External Tool Security
    subgraph "External Tool Security"
        E1[Git Command Sanitization]
        E2[RepoMapper Isolation]
        E3[Serena Process Control]
        E4[LEANN Memory Protection]
    end

    %% Data Security
    subgraph "Data Security"
        D1[Encryption at Rest]
        D2[Encryption in Transit]
        D3[PII Detection & Redaction]
        D4[Secure Cache Management]
    end

    %% Audit & Monitoring
    subgraph "Audit & Monitoring"
        L1[Security Event Logging]
        L2[Access Attempt Tracking]
        L3[Anomaly Detection]
        L4[Compliance Reporting]
    end

    %% Threat Detection
    subgraph "Threat Detection"
        T1[Malicious Input Detection]
        T2[Code Injection Prevention]
        T3[Path Traversal Protection]
        T4[Resource Exhaustion Prevention]
    end

    %% Data Flow with Security Checkpoints
    C1 --> A1
    C2 --> A1
    C3 --> A1
    
    A1 --> A2
    A2 --> A3
    A3 --> A4
    A4 --> Z1
    
    Z1 --> Z2
    Z2 --> Z3
    Z3 --> Z4
    Z4 --> V1
    
    V1 --> V2
    V2 --> V3
    V3 --> V4
    V4 --> M1
    
    M1 --> M2
    M2 --> M3
    M3 --> M4
    M4 --> S1
    
    S1 --> S2
    S2 --> S3
    S3 --> S4
    S4 --> S5
    S5 --> P1
    
    P1 --> P2
    P2 --> P3
    P3 --> P4
    P4 --> E1
    
    E1 --> E2
    E2 --> E3
    E3 --> E4
    E4 --> D1
    
    D1 --> D2
    D2 --> D3
    D3 --> D4
    
    %% Security Monitoring Flows
    A1 -.-> L1
    V1 -.-> L2
    M1 -.-> L3
    P1 -.-> L4
    
    %% Threat Detection Flows
    V1 --> T1
    V2 --> T2
    V3 --> T3
    S4 --> T4

    %% Styling
    classDef client fill:#e3f2fd
    classDef auth fill:#e8f5e8
    classDef authz fill:#fff3e0
    classDef validation fill:#f3e5f5
    classDef sandbox fill:#ffebee
    classDef server fill:#fafafa
    classDef pipeline fill:#e1f5fe
    classDef external fill:#f1f8e9
    classDef data fill:#f9fbe7
    classDef audit fill:#ede7f6
    classDef threat fill:#fce4ec

    class C1,C2,C3 client
    class A1,A2,A3,A4 auth
    class Z1,Z2,Z3,Z4 authz
    class V1,V2,V3,V4 validation
    class S1,S2,S3,S4,S5 sandbox
    class M1,M2,M3,M4 server
    class P1,P2,P3,P4 pipeline
    class E1,E2,E3,E4 external
    class D1,D2,D3,D4 data
    class L1,L2,L3,L4 audit
    class T1,T2,T3,T4 threat
```

## Security Components Detail

### Authentication Layer
- **API Key Validation**: Cryptographically secure API key verification
- **Token Management**: JWT-based session tokens with expiration
- **Session Control**: Active session monitoring and revocation capabilities
- **Rate Limiting**: Per-client request throttling to prevent abuse

### Authorization Layer
- **Permission Matrix**: Role-based access control with granular permissions
- **Resource Access Control**: Path-based and operation-specific authorization
- **Operation Scoping**: Limiting operations to authorized repository scopes
- **Privilege Escalation Prevention**: Continuous privilege validation

### Input Validation
- **Schema Validation**: Pydantic-based strict type and format validation
- **Path Sanitization**: Comprehensive path traversal attack prevention
- **Injection Prevention**: SQL, command, and code injection protection
- **Type Safety Enforcement**: Runtime type checking for all inputs

### Execution Sandbox
- **Process Isolation**: Separate processes for each external tool execution
- **File System Restrictions**: Chroot-like restrictions on file access
- **Network Isolation**: Prevented external network access during execution
- **Resource Limits**: Memory, CPU, and disk usage limitations
- **Subprocess Control**: Strict control over spawned processes

### Secure MCP Server
- **Security Middleware**: Request/response security processing
- **Request Interceptor**: Pre-processing security checks
- **Response Sanitizer**: Output sanitization and PII redaction
- **Error Handler**: Secure error reporting without information leakage

## Security Threat Model

### Identified Threats & Mitigations

#### 1. Path Traversal Attacks
**Threat**: Malicious clients attempting to access files outside authorized scope
**Mitigation**:
- Comprehensive path sanitization
- Chroot-style sandbox execution
- Whitelist-based file access control
- Realpath resolution to prevent symbolic link attacks

#### 2. Code Injection
**Threat**: Injection of malicious code through repository content or parameters
**Mitigation**:
- Strict input validation and sanitization
- Process isolation for all external tool execution
- Static analysis of processed content
- Secure subprocess execution with argument validation

#### 3. Information Disclosure
**Threat**: Accidental exposure of sensitive information in responses
**Mitigation**:
- PII detection and automatic redaction
- Response sanitization before client delivery
- Secure error handling without stack trace exposure
- Audit logging of all data access

#### 4. Denial of Service
**Threat**: Resource exhaustion attacks through expensive operations
**Mitigation**:
- Configurable resource limits per operation
- Request rate limiting per client
- Timeout controls on all external tool invocations
- Memory and CPU usage monitoring

#### 5. Privilege Escalation
**Threat**: Unauthorized access to system resources or elevated permissions
**Mitigation**:
- Minimal privilege principle for all operations
- Continuous permission validation
- Process isolation and sandboxing
- Regular security audit logging

## Security Configuration

### API Key Management
```python
class SecurityConfig:
    api_key_length: int = 64
    api_key_rotation_days: int = 90
    max_failed_attempts: int = 5
    lockout_duration: int = 3600  # seconds
```

### Sandbox Limits
```python
class SandboxConfig:
    max_memory_mb: int = 512
    max_cpu_percent: int = 50
    max_execution_time: int = 300  # seconds
    max_open_files: int = 100
    allowed_network_access: bool = False
```

### Input Validation Rules
```python
class ValidationConfig:
    max_path_length: int = 4096
    allowed_path_chars: str = "a-zA-Z0-9._/-"
    max_query_length: int = 1000
    allowed_file_extensions: list = [".py", ".ts", ".js", ".md", ...]
```

## Compliance & Audit

### Security Logging
- All authentication attempts (success/failure)
- Authorization decisions and access control violations
- Input validation failures and potential attacks
- Resource usage and limit violations
- External tool execution and sandbox events

### Audit Trail Requirements
- Immutable log storage with integrity verification
- Centralized log aggregation for security monitoring
- Retention policies for compliance requirements
- Automated anomaly detection and alerting

### Compliance Standards
- **OWASP Top 10**: Protection against common web application vulnerabilities
- **CWE/SANS Top 25**: Mitigation of most dangerous software errors
- **NIST Cybersecurity Framework**: Implementation of security controls
- **SOC 2 Type II**: Controls for security, availability, and confidentiality

## Security Testing Strategy

### Static Analysis
- **Semgrep**: Custom rules for security vulnerability detection
- **Bandit**: Python-specific security issue identification
- **Safety**: Dependency vulnerability scanning
- **CodeQL**: Advanced security pattern matching

### Dynamic Testing
- **Penetration Testing**: Regular security assessments
- **Fuzzing**: Input validation stress testing
- **Load Testing**: DoS resistance validation
- **Integration Testing**: End-to-end security workflow verification

### Security Metrics
- **Mean Time to Detect (MTTD)**: Security incident detection speed
- **Mean Time to Respond (MTTR)**: Security incident response time
- **False Positive Rate**: Security alert accuracy
- **Coverage**: Percentage of code covered by security tests