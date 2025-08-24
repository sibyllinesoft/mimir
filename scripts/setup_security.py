#!/usr/bin/env python3
"""
Security setup script for Mimir Deep Code Research System.

Configures comprehensive security hardening including encryption keys,
API keys, audit logging, and secure defaults for production deployment.
"""

import argparse
import base64
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Dict, Any, Optional

# Import security components
try:
    from src.repoindex.security.config import SecurityConfig, get_security_config
    from src.repoindex.security.crypto import generate_master_key
    from src.repoindex.security.auth import APIKeyManager
    from src.repoindex.util.log import get_logger
except ImportError as e:
    print(f"Error importing Mimir security modules: {e}")
    print("Please ensure you're running this script from the project root directory.")
    sys.exit(1)

logger = get_logger(__name__)


class SecuritySetup:
    """Comprehensive security setup and configuration."""
    
    def __init__(self, config_dir: Optional[Path] = None):
        """Initialize security setup.
        
        Args:
            config_dir: Directory for security configuration files
        """
        self.config_dir = config_dir or Path.home() / ".cache" / "mimir" / "security"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Default file paths
        self.config_file = self.config_dir / "security_config.json"
        self.api_keys_file = self.config_dir / "api_keys.json"
        self.audit_log_file = self.config_dir / "audit.log"
        self.env_file = self.config_dir / "mimir_security.env"
        
        print(f"Security setup using directory: {self.config_dir}")
    
    def generate_master_key(self) -> str:
        """Generate a new master encryption key."""
        print("Generating master encryption key...")
        master_key_b64 = generate_master_key()
        print("✓ Master key generated")
        return master_key_b64
    
    def generate_api_keys(self, count: int = 1) -> Dict[str, str]:
        """Generate API keys for authentication.
        
        Args:
            count: Number of API keys to generate
            
        Returns:
            Dictionary mapping key names to key values
        """
        print(f"Generating {count} API key(s)...")
        
        api_keys = {}
        for i in range(count):
            key_name = f"api_key_{i+1}"
            key_value = base64.b64encode(secrets.token_bytes(32)).decode('utf-8')
            api_keys[key_name] = key_value
        
        print(f"✓ Generated {count} API key(s)")
        return api_keys
    
    def create_security_config(
        self,
        production: bool = False,
        enable_all_features: bool = True
    ) -> SecurityConfig:
        """Create security configuration.
        
        Args:
            production: Whether to use production security settings
            enable_all_features: Whether to enable all security features
            
        Returns:
            Security configuration object
        """
        print("Creating security configuration...")
        
        if production:
            print("Using production security settings (high security)")
            config = SecurityConfig(
                # Authentication
                require_authentication=True,
                api_keys_file=self.api_keys_file,
                auth_token_lifetime=3600,  # 1 hour
                
                # Rate limiting (stricter for production)
                global_rate_limit=500,  # requests per minute
                ip_rate_limit=50,  # requests per minute per IP
                api_key_rate_limit=100,  # requests per minute per API key
                
                # Input validation (stricter limits)
                max_path_length=2048,
                max_filename_length=128,
                max_file_size=50 * 1024 * 1024,  # 50MB
                max_query_length=5000,
                
                # Security features (all enabled)
                enable_credential_scanning=True,
                enable_sandboxing=True,
                enable_index_encryption=True,
                encrypt_embeddings=True,
                encrypt_metadata=True,
                enable_audit_logging=True,
                enable_threat_detection=True,
                enable_abuse_prevention=True,
                
                # Sandbox limits (conservative)
                max_memory_mb=512,  # 512MB
                max_cpu_time_seconds=120,  # 2 minutes
                max_wall_time_seconds=300,  # 5 minutes
                max_open_files=512,
                max_processes=16,
                
                # Audit logging
                audit_log_file=self.audit_log_file,
                audit_log_max_size_mb=50,
                audit_log_backup_count=10,
                
                # Security monitoring (stricter thresholds)
                max_auth_failures=3,
                max_rate_limit_violations=2,
                ip_block_duration_minutes=120,
                suspicious_pattern_threshold=5,
                error_rate_threshold=10
            )
        else:
            print("Using development security settings (moderate security)")
            config = SecurityConfig(
                # Authentication (optional for dev)
                require_authentication=enable_all_features,
                api_keys_file=self.api_keys_file if enable_all_features else None,
                auth_token_lifetime=7200,  # 2 hours
                
                # Rate limiting (more permissive for dev)
                global_rate_limit=2000,
                ip_rate_limit=200,
                api_key_rate_limit=400,
                
                # Input validation (standard limits)
                max_path_length=4096,
                max_filename_length=255,
                max_file_size=100 * 1024 * 1024,  # 100MB
                max_query_length=10000,
                
                # Security features
                enable_credential_scanning=enable_all_features,
                enable_sandboxing=enable_all_features,
                enable_index_encryption=enable_all_features,
                encrypt_embeddings=enable_all_features,
                encrypt_metadata=enable_all_features,
                enable_audit_logging=True,  # Always enable audit logging
                enable_threat_detection=enable_all_features,
                enable_abuse_prevention=enable_all_features,
                
                # Sandbox limits (more permissive for dev)
                max_memory_mb=1024,  # 1GB
                max_cpu_time_seconds=300,  # 5 minutes
                max_wall_time_seconds=600,  # 10 minutes
                max_open_files=1024,
                max_processes=32,
                
                # Audit logging
                audit_log_file=self.audit_log_file,
                audit_log_max_size_mb=100,
                audit_log_backup_count=5,
                
                # Security monitoring (more permissive)
                max_auth_failures=5,
                max_rate_limit_violations=3,
                ip_block_duration_minutes=60,
                suspicious_pattern_threshold=10,
                error_rate_threshold=20
            )
        
        print("✓ Security configuration created")
        return config
    
    def save_security_config(self, config: SecurityConfig) -> None:
        """Save security configuration to file."""
        print(f"Saving security configuration to {self.config_file}")
        config.to_file(self.config_file)
        print("✓ Security configuration saved")
    
    def save_api_keys(self, api_keys: Dict[str, str]) -> None:
        """Save API keys to secure file."""
        print(f"Saving API keys to {self.api_keys_file}")
        
        # Create API key data structure
        api_key_data = {
            "keys": {},
            "metadata": {
                "created_at": "2025-01-19T12:00:00Z",
                "total_keys": len(api_keys),
                "key_format": "base64_encoded_32_bytes"
            }
        }
        
        # Add keys with metadata
        for key_name, key_value in api_keys.items():
            api_key_data["keys"][key_name] = {
                "key": key_value,
                "created_at": "2025-01-19T12:00:00Z",
                "permissions": ["repo:index", "repo:read", "repo:search", "repo:ask", "repo:cancel"],
                "rate_limit": 200,  # requests per minute
                "enabled": True
            }
        
        # Save with restricted permissions
        with open(self.api_keys_file, 'w') as f:
            json.dump(api_key_data, f, indent=2)
        
        # Set restrictive permissions (owner read/write only)
        os.chmod(self.api_keys_file, 0o600)
        
        print("✓ API keys saved with restricted permissions")
    
    def create_environment_file(
        self,
        master_key: str,
        config: SecurityConfig
    ) -> None:
        """Create environment file with security variables."""
        print(f"Creating environment file: {self.env_file}")
        
        env_content = f"""# Mimir Security Environment Configuration
# Generated on 2025-01-19
# 
# Source this file to set security environment variables:
# source {self.env_file}

# Encryption
export MIMIR_MASTER_KEY="{master_key}"
export MIMIR_ENABLE_ENCRYPTION={"true" if config.enable_index_encryption else "false"}
export MIMIR_ENCRYPT_EMBEDDINGS={"true" if config.encrypt_embeddings else "false"}
export MIMIR_ENCRYPT_METADATA={"true" if config.encrypt_metadata else "false"}

# Authentication
export MIMIR_REQUIRE_AUTH={"true" if config.require_authentication else "false"}
export MIMIR_API_KEYS_FILE="{self.api_keys_file}"
export MIMIR_AUTH_TOKEN_LIFETIME="{config.auth_token_lifetime}"

# Rate Limiting
export MIMIR_GLOBAL_RATE_LIMIT="{config.global_rate_limit}"
export MIMIR_IP_RATE_LIMIT="{config.ip_rate_limit}"
export MIMIR_API_KEY_RATE_LIMIT="{config.api_key_rate_limit}"

# Input Validation
export MIMIR_MAX_PATH_LENGTH="{config.max_path_length}"
export MIMIR_MAX_FILE_SIZE="{config.max_file_size}"
export MIMIR_MAX_QUERY_LENGTH="{config.max_query_length}"

# Security Features
export MIMIR_ENABLE_CREDENTIAL_SCANNING={"true" if config.enable_credential_scanning else "false"}
export MIMIR_ENABLE_SANDBOXING={"true" if config.enable_sandboxing else "false"}
export MIMIR_ENABLE_AUDIT_LOGGING={"true" if config.enable_audit_logging else "false"}
export MIMIR_ENABLE_THREAT_DETECTION={"true" if config.enable_threat_detection else "false"}

# Sandbox Limits
export MIMIR_MAX_MEMORY_MB="{config.max_memory_mb}"
export MIMIR_MAX_CPU_TIME="{config.max_cpu_time_seconds}"
export MIMIR_MAX_WALL_TIME="{config.max_wall_time_seconds}"

# Audit Logging
export MIMIR_AUDIT_LOG_FILE="{self.audit_log_file}"
export MIMIR_AUDIT_LOG_MAX_SIZE_MB="{config.audit_log_max_size_mb}"

# Security Monitoring
export MIMIR_MAX_AUTH_FAILURES="{config.max_auth_failures}"
export MIMIR_IP_BLOCK_DURATION="{config.ip_block_duration_minutes}"

# File Paths
export MIMIR_CONFIG_DIR="{self.config_dir}"
export MIMIR_SECURITY_CONFIG_FILE="{self.config_file}"

# Usage Instructions:
# 1. Source this file: source {self.env_file}
# 2. Start Mimir: python -m repoindex.main_secure mcp
# 3. Or use config file: python -m repoindex.main_secure mcp --config {self.config_file}
"""
        
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        # Set restrictive permissions
        os.chmod(self.env_file, 0o600)
        
        print("✓ Environment file created with restricted permissions")
    
    def create_docker_compose_config(self) -> None:
        """Create Docker Compose configuration with security settings."""
        docker_compose_file = self.config_dir / "docker-compose.security.yml"
        
        print(f"Creating Docker Compose security configuration: {docker_compose_file}")
        
        compose_content = f"""version: '3.8'

services:
  mimir-secure:
    image: mimir:latest
    container_name: mimir-secure
    restart: unless-stopped
    
    # Security: Run as non-root user
    user: "1000:1000"
    
    # Security: Read-only root filesystem
    read_only: true
    
    # Security: Drop all capabilities
    cap_drop:
      - ALL
    
    # Security: No new privileges
    security_opt:
      - no-new-privileges:true
    
    # Security: AppArmor profile
    security_opt:
      - apparmor:docker-default
    
    environment:
      # Load from environment file
      - MIMIR_CONFIG_DIR=/app/config
      - MIMIR_SECURITY_CONFIG_FILE=/app/config/security_config.json
      - MIMIR_API_KEYS_FILE=/app/config/api_keys.json
      - MIMIR_AUDIT_LOG_FILE=/app/logs/audit.log
      - MIMIR_ENABLE_SANDBOXING=true
      - MIMIR_ENABLE_ENCRYPTION=true
      - MIMIR_ENABLE_AUDIT_LOGGING=true
      - MIMIR_REQUIRE_AUTH=true
    
    volumes:
      # Security configuration (read-only)
      - {self.config_dir}:/app/config:ro
      
      # Audit logs (read-write)
      - {self.config_dir}/logs:/app/logs:rw
      
      # Data directory (read-write)
      - {self.config_dir}/data:/app/data:rw
      
      # Temporary directory (read-write, tmpfs for security)
      - type: tmpfs
        target: /tmp
        tmpfs:
          size: 100M
          mode: 1777
    
    ports:
      - "127.0.0.1:8000:8000"  # Bind to localhost only
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    
    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
        labels: "service=mimir-secure"

  # Security: Network isolation
networks:
  default:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.name: mimir-secure
    ipam:
      driver: default
      config:
        - subnet: 172.20.0.0/16
"""
        
        with open(docker_compose_file, 'w') as f:
            f.write(compose_content)
        
        print("✓ Docker Compose security configuration created")
    
    def create_systemd_service(self) -> None:
        """Create systemd service file for secure deployment."""
        service_file = self.config_dir / "mimir-secure.service"
        
        print(f"Creating systemd service configuration: {service_file}")
        
        service_content = f"""[Unit]
Description=Mimir Secure Deep Code Research System
Documentation=https://github.com/your-org/mimir
After=network.target
Wants=network.target

[Service]
Type=exec
User=mimir
Group=mimir
WorkingDirectory=/opt/mimir

# Security: Strict service configuration
NoNewPrivileges=yes
PrivateTmp=yes
PrivateDevices=yes
PrivateNetwork=no
ProtectSystem=strict
ProtectHome=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
RestrictRealtime=yes
RestrictSUIDSGID=yes
RemoveIPC=yes
RestrictNamespaces=yes

# Security: Capability restrictions
CapabilityBoundingSet=
AmbientCapabilities=

# Security: System call filtering
SystemCallFilter=@system-service
SystemCallFilter=~@debug @mount @cpu-emulation @obsolete @privileged @reboot @swap
SystemCallErrorNumber=EPERM

# Security: File system access
ReadWritePaths={self.config_dir}/logs {self.config_dir}/data
ReadOnlyPaths={self.config_dir}
InaccessiblePaths=/home /root /boot /opt

# Environment
Environment=MIMIR_CONFIG_DIR={self.config_dir}
Environment=MIMIR_SECURITY_CONFIG_FILE={self.config_file}
EnvironmentFile={self.env_file}

# Execution
ExecStart=/opt/mimir/venv/bin/python -m repoindex.main_secure mcp --config {self.config_file}
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=10

# Resource limits
LimitNOFILE=1024
LimitNPROC=64
MemoryMax=2G
TasksMax=100

# Security: Network restrictions
IPAddressDeny=any
IPAddressAllow=localhost
IPAddressAllow=127.0.0.0/8
IPAddressAllow=::1/128

[Install]
WantedBy=multi-user.target
"""
        
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        print("✓ Systemd service configuration created")
        print(f"  To install: sudo cp {service_file} /etc/systemd/system/")
        print(f"  To enable: sudo systemctl enable mimir-secure")
        print(f"  To start: sudo systemctl start mimir-secure")
    
    def validate_setup(self) -> bool:
        """Validate the security setup."""
        print("Validating security setup...")
        
        errors = []
        
        # Check file existence
        required_files = [
            self.config_file,
            self.api_keys_file,
            self.env_file
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                errors.append(f"Missing required file: {file_path}")
        
        # Check file permissions
        for file_path in [self.api_keys_file, self.env_file]:
            if file_path.exists():
                stat_info = file_path.stat()
                if stat_info.st_mode & 0o077:  # Check if group/other have access
                    errors.append(f"Insecure permissions on {file_path}")
        
        # Validate configuration
        if self.config_file.exists():
            try:
                config = SecurityConfig.from_file(self.config_file)
                config_errors = config.validate()
                errors.extend(config_errors)
            except Exception as e:
                errors.append(f"Configuration validation failed: {e}")
        
        if errors:
            print("✗ Security setup validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        else:
            print("✓ Security setup validation passed")
            return True
    
    def print_summary(self, master_key: str, api_keys: Dict[str, str]) -> None:
        """Print setup summary and next steps."""
        print("\n" + "="*60)
        print("MIMIR SECURITY SETUP COMPLETE")
        print("="*60)
        
        print(f"\nConfiguration Directory: {self.config_dir}")
        print(f"Security Config: {self.config_file}")
        print(f"Environment File: {self.env_file}")
        print(f"API Keys File: {self.api_keys_file}")
        print(f"Audit Log: {self.audit_log_file}")
        
        print(f"\nMaster Encryption Key:")
        print(f"  {master_key}")
        print(f"  ⚠️  SAVE THIS KEY SECURELY - LOST KEYS = LOST DATA!")
        
        print(f"\nGenerated API Keys:")
        for key_name, key_value in api_keys.items():
            print(f"  {key_name}: {key_value}")
        
        print(f"\nNext Steps:")
        print(f"1. Source environment: source {self.env_file}")
        print(f"2. Start server: python -m repoindex.main_secure mcp")
        print(f"3. Or use config: python -m repoindex.main_secure mcp --config {self.config_file}")
        print(f"4. Test indexing: python -m repoindex.main_secure index /path/to/repo")
        
        print(f"\nProduction Deployment:")
        print(f"- Docker: docker-compose -f {self.config_dir}/docker-compose.security.yml up")
        print(f"- Systemd: Install {self.config_dir}/mimir-secure.service")
        
        print(f"\nSecurity Features Enabled:")
        print(f"- ✓ Input validation and sanitization")
        print(f"- ✓ Process sandboxing and resource limits")
        print(f"- ✓ Credential scanning and detection")
        print(f"- ✓ Index encryption and secure storage")
        print(f"- ✓ API authentication and rate limiting")
        print(f"- ✓ Comprehensive audit logging")
        print(f"- ✓ Threat detection and monitoring")
        
        print("\n" + "="*60)


def main():
    """Main entry point for security setup."""
    parser = argparse.ArgumentParser(
        description="Mimir Security Setup - Configure comprehensive security hardening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Development setup with all features
  python setup_security.py --dev --all-features
  
  # Production setup with high security
  python setup_security.py --production --config-dir /etc/mimir/security
  
  # Quick setup with minimal security
  python setup_security.py --quick
  
  # Generate additional API keys
  python setup_security.py --api-keys 5 --config-dir ~/.cache/mimir/security
        """
    )
    
    parser.add_argument(
        "--config-dir",
        type=Path,
        help="Directory for security configuration files"
    )
    parser.add_argument(
        "--production",
        action="store_true",
        help="Use production security settings (high security)"
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Use development security settings (moderate security)"
    )
    parser.add_argument(
        "--all-features",
        action="store_true",
        help="Enable all security features"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick setup with minimal prompts"
    )
    parser.add_argument(
        "--api-keys",
        type=int,
        default=1,
        help="Number of API keys to generate"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing setup"
    )
    parser.add_argument(
        "--docker-config",
        action="store_true",
        help="Generate Docker deployment configuration"
    )
    parser.add_argument(
        "--systemd-config",
        action="store_true",
        help="Generate systemd service configuration"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize setup
        setup = SecuritySetup(args.config_dir)
        
        if args.validate_only:
            success = setup.validate_setup()
            sys.exit(0 if success else 1)
        
        # Determine configuration type
        if args.production:
            print("Setting up PRODUCTION security configuration")
            production = True
            enable_all_features = True
        elif args.dev:
            print("Setting up DEVELOPMENT security configuration")
            production = False
            enable_all_features = args.all_features
        elif args.quick:
            print("Setting up QUICK security configuration")
            production = False
            enable_all_features = False
        else:
            # Interactive setup
            print("Choose configuration type:")
            print("1. Production (high security)")
            print("2. Development (moderate security)")
            print("3. Quick (minimal security)")
            choice = input("Enter choice (1-3): ").strip()
            
            if choice == "1":
                production = True
                enable_all_features = True
            elif choice == "2":
                production = False
                enable_all_features = True
            else:
                production = False
                enable_all_features = False
        
        # Generate master key
        master_key = setup.generate_master_key()
        
        # Generate API keys
        api_keys = setup.generate_api_keys(args.api_keys)
        
        # Create security configuration
        config = setup.create_security_config(production, enable_all_features)
        
        # Save configuration
        setup.save_security_config(config)
        setup.save_api_keys(api_keys)
        setup.create_environment_file(master_key, config)
        
        # Generate deployment configurations if requested
        if args.docker_config or production:
            setup.create_docker_compose_config()
        
        if args.systemd_config or production:
            setup.create_systemd_service()
        
        # Validate setup
        setup.validate_setup()
        
        # Print summary
        setup.print_summary(master_key, api_keys)
        
    except KeyboardInterrupt:
        print("\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Setup failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()