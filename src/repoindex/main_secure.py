"""
Security-hardened main entry point for Mimir repository indexing.

Provides command-line interface with comprehensive security controls including
input validation, sandboxed execution, credential scanning, and audit logging.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .mcp.secure_server import MimirSecureServer
from .pipeline.secure_run import SecureIndexingPipeline
from .security.config import get_security_config, SecurityConfig, load_security_config_file
from .security.audit import get_security_auditor, SecurityEvent, SecurityEventType
from .util.log import get_logger

logger = get_logger(__name__)


class SecureMimirCLI:
    """Security-enhanced command-line interface for Mimir."""
    
    def __init__(self):
        self.security_config: Optional[SecurityConfig] = None
        self.security_auditor = get_security_auditor()
    
    def create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser with security options."""
        parser = argparse.ArgumentParser(
            description="Mimir - Secure Deep Code Research System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Security Features:
  - Input validation and sanitization
  - Sandboxed external tool execution
  - Credential scanning and detection
  - Encrypted index storage
  - Comprehensive audit logging
  - Rate limiting and abuse prevention

Examples:
  # Start secure MCP server
  python -m repoindex.main_secure mcp --storage-dir ./data

  # Index repository with security
  python -m repoindex.main_secure index /path/to/repo --language ts

  # Generate security configuration
  python -m repoindex.main_secure config --generate --output security.json

Security Environment Variables:
  MIMIR_REQUIRE_AUTH=true          # Enable authentication
  MIMIR_ENABLE_SANDBOXING=true     # Enable process sandboxing
  MIMIR_ENABLE_ENCRYPTION=true     # Enable index encryption
  MIMIR_MASTER_KEY=<base64-key>     # Encryption master key
  MIMIR_ALLOWED_BASE_PATHS=/path   # Allowed file paths
  MIMIR_API_KEYS_FILE=/path/keys   # API keys storage
  MIMIR_AUDIT_LOG_FILE=/path/log   # Audit log location
            """
        )
        
        # Add global options
        parser.add_argument(
            "--storage-dir",
            type=Path,
            default=Path.home() / ".cache" / "mimir",
            help="Storage directory for indexes and cache"
        )
        parser.add_argument(
            "--config",
            type=Path,
            help="Security configuration file"
        )
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--no-security",
            action="store_true",
            help="Disable security features (NOT recommended for production)"
        )
        
        # Create subcommands
        subparsers = parser.add_subparsers(dest="command", help="Available commands")
        
        # MCP server command
        mcp_parser = subparsers.add_parser("mcp", help="Start secure MCP server")
        mcp_parser.add_argument(
            "--no-auth",
            action="store_true",
            help="Disable authentication (NOT recommended for production)"
        )
        mcp_parser.add_argument(
            "--generate-config",
            action="store_true",
            help="Generate default configuration and exit"
        )
        
        # Index command
        index_parser = subparsers.add_parser("index", help="Index a repository")
        index_parser.add_argument(
            "repo_path",
            help="Path to repository to index"
        )
        index_parser.add_argument(
            "--language",
            default="ts",
            help="Primary language (default: ts)"
        )
        index_parser.add_argument(
            "--rev",
            help="Git revision to index (default: HEAD)"
        )
        index_parser.add_argument(
            "--output",
            type=Path,
            help="Output directory for index"
        )
        
        # Search command
        search_parser = subparsers.add_parser("search", help="Search an index")
        search_parser.add_argument(
            "index_id",
            help="Index identifier to search"
        )
        search_parser.add_argument(
            "query",
            help="Search query"
        )
        search_parser.add_argument(
            "--k",
            type=int,
            default=20,
            help="Number of results to return"
        )
        
        # Configuration command
        config_parser = subparsers.add_parser("config", help="Manage security configuration")
        config_parser.add_argument(
            "--generate",
            action="store_true",
            help="Generate default configuration"
        )
        config_parser.add_argument(
            "--output",
            type=Path,
            help="Output file for configuration"
        )
        config_parser.add_argument(
            "--validate",
            type=Path,
            help="Validate configuration file"
        )
        
        # Audit command
        audit_parser = subparsers.add_parser("audit", help="Security audit operations")
        audit_parser.add_argument(
            "--report",
            action="store_true",
            help="Generate security audit report"
        )
        audit_parser.add_argument(
            "--output",
            type=Path,
            help="Output file for report"
        )
        
        return parser
    
    def load_security_config(self, config_file: Optional[Path], no_security: bool) -> SecurityConfig:
        """Load security configuration with validation."""
        try:
            if no_security:
                logger.warning("Security features disabled - NOT recommended for production")
                config = SecurityConfig(
                    require_authentication=False,
                    enable_sandboxing=False,
                    enable_index_encryption=False,
                    enable_credential_scanning=False,
                    enable_audit_logging=False,
                    enable_threat_detection=False,
                    enable_abuse_prevention=False
                )
            elif config_file:
                config = load_security_config_file(config_file)
                logger.info(f"Security configuration loaded from: {config_file}")
            else:
                config = get_security_config()
                logger.info("Security configuration loaded from environment")
            
            # Validate configuration
            errors = config.validate()
            if errors:
                logger.warning("Security configuration validation errors:")
                for error in errors:
                    logger.warning(f"  - {error}")
            
            # Record configuration load event
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.SYSTEM_START,
                component="main_secure",
                message="Security configuration loaded",
                metadata={
                    "config_source": "file" if config_file else "environment",
                    "security_disabled": no_security,
                    "validation_errors": len(errors)
                }
            ))
            
            return config
            
        except Exception as e:
            logger.error(f"Failed to load security configuration: {e}")
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.SECURITY_ERROR,
                component="main_secure",
                severity="high",
                message=f"Security configuration load failed: {str(e)}",
                metadata={"error": str(e)}
            ))
            raise
    
    async def run_mcp_server(self, args) -> None:
        """Run secure MCP server."""
        try:
            logger.info("Starting Mimir secure MCP server")
            
            # Override authentication if requested
            if args.no_auth:
                logger.warning("Authentication disabled - NOT recommended for production")
                self.security_config.require_authentication = False
            
            # Generate config if requested
            if args.generate_config:
                config_file = args.output or self.security_config.get_default_paths()["config_file"]
                self.security_config.to_file(config_file)
                print(f"Default security configuration saved to: {config_file}")
                return
            
            # Create and run server
            server = MimirSecureServer(
                storage_dir=args.storage_dir,
                security_config=self.security_config
            )
            
            await server.run()
            
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        except Exception as e:
            logger.error(f"MCP server error: {e}")
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.SYSTEM_ERROR,
                component="mcp_server",
                severity="high",
                message=f"MCP server failed: {str(e)}",
                metadata={"error": str(e)}
            ))
            raise
    
    async def run_index_command(self, args) -> None:
        """Run secure repository indexing."""
        try:
            logger.info(f"Starting secure indexing of repository: {args.repo_path}")
            
            # Create secure pipeline
            pipeline = SecureIndexingPipeline(
                storage_dir=args.storage_dir,
                security_config=self.security_config
            )
            
            # Start indexing
            index_id = await pipeline.start_indexing(
                repo_path=args.repo_path,
                rev=args.rev,
                language=args.language
            )
            
            print(f"Secure indexing started: {index_id}")
            
            # Wait for completion
            # TODO: Implement status polling
            
        except Exception as e:
            logger.error(f"Indexing failed: {e}")
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.PIPELINE_ERROR,
                component="index_command",
                severity="high",
                message=f"Secure indexing failed: {str(e)}",
                metadata={
                    "repo_path": args.repo_path,
                    "error": str(e)
                }
            ))
            raise
    
    async def run_search_command(self, args) -> None:
        """Run secure search operation."""
        try:
            logger.info(f"Searching index {args.index_id} for: {args.query}")
            
            # TODO: Implement secure search with validation
            print(f"Search results for '{args.query}' in index {args.index_id}:")
            print("(Search implementation pending)")
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    def run_config_command(self, args) -> None:
        """Run configuration management."""
        try:
            if args.generate:
                config = get_security_config()
                output_file = args.output or config.get_default_paths()["config_file"]
                config.to_file(output_file)
                print(f"Default security configuration generated: {output_file}")
                
            elif args.validate:
                config = SecurityConfig.from_file(args.validate)
                errors = config.validate()
                if errors:
                    print(f"Configuration validation failed ({len(errors)} errors):")
                    for error in errors:
                        print(f"  - {error}")
                    sys.exit(1)
                else:
                    print("Configuration validation passed")
                    
        except Exception as e:
            logger.error(f"Configuration command failed: {e}")
            raise
    
    def run_audit_command(self, args) -> None:
        """Run security audit operations."""
        try:
            if args.report:
                # Generate security audit report
                report_data = self.security_auditor.generate_report()
                
                output_file = args.output or Path("security_audit_report.json")
                
                import json
                with open(output_file, 'w') as f:
                    json.dump(report_data, f, indent=2)
                
                print(f"Security audit report generated: {output_file}")
                
                # Print summary
                print("\nSecurity Audit Summary:")
                print(f"  Events logged: {report_data.get('total_events', 0)}")
                print(f"  Security violations: {report_data.get('security_violations', 0)}")
                print(f"  High severity events: {report_data.get('high_severity_events', 0)}")
                
        except Exception as e:
            logger.error(f"Audit command failed: {e}")
            raise
    
    async def run(self) -> None:
        """Main entry point."""
        parser = self.create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return
        
        try:
            # Load security configuration
            self.security_config = self.load_security_config(args.config, args.no_security)
            
            # Set up logging
            if args.verbose:
                import logging
                logging.getLogger().setLevel(logging.DEBUG)
            
            # Record startup
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.SYSTEM_START,
                component="main_secure",
                message=f"Mimir secure CLI started: {args.command}",
                metadata={
                    "command": args.command,
                    "security_enabled": not args.no_security
                }
            ))
            
            # Route to command handlers
            if args.command == "mcp":
                await self.run_mcp_server(args)
            elif args.command == "index":
                await self.run_index_command(args)
            elif args.command == "search":
                await self.run_search_command(args)
            elif args.command == "config":
                self.run_config_command(args)
            elif args.command == "audit":
                self.run_audit_command(args)
            else:
                parser.print_help()
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Command failed: {e}")
            self.security_auditor.record_event(SecurityEvent(
                event_type=SecurityEventType.SYSTEM_ERROR,
                component="main_secure",
                severity="high",
                message=f"CLI command failed: {str(e)}",
                metadata={
                    "command": args.command if 'args' in locals() else "unknown",
                    "error": str(e)
                }
            ))
            sys.exit(1)
        
        finally:
            # Cleanup
            if hasattr(self, 'security_auditor'):
                self.security_auditor.close()


def main():
    """Main entry point for secure Mimir CLI."""
    cli = SecureMimirCLI()
    asyncio.run(cli.run())


if __name__ == "__main__":
    main()