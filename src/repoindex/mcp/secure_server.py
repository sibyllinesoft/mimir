"""
Security-hardened MCP server implementation.

Extends the base MCP server with comprehensive security hardening including
authentication, authorization, input validation, rate limiting, and audit logging.
"""

import asyncio
from pathlib import Path
from typing import Any

from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from ..security.audit import configure_security_auditor
from ..security.config import SecurityConfig, get_security_config
from ..security.crypto import configure_crypto, generate_master_key
from ..security.sandbox import configure_sandbox
from ..security.server_middleware import create_security_middleware
from ..util.fs import ensure_directory
from ..util.log import get_logger
from .server import MCPServer  # Import base server

logger = get_logger(__name__)


class SecureMCPServer(MCPServer):
    """Security-hardened MCP server with comprehensive protection."""

    def __init__(
        self, storage_dir: Path | None = None, security_config: SecurityConfig | None = None
    ):
        """Initialize secure MCP server.

        Args:
            storage_dir: Optional storage directory
            security_config: Optional security configuration
        """
        # Load security configuration
        self.security_config = security_config or get_security_config()

        # Initialize base server
        super().__init__(storage_dir)

        # Initialize security components
        self._initialize_security()

        logger.info(
            "Secure MCP server initialized",
            require_auth=self.security_config.require_authentication,
            sandboxing_enabled=self.security_config.enable_sandboxing,
            encryption_enabled=self.security_config.enable_index_encryption,
            audit_logging_enabled=self.security_config.enable_audit_logging,
        )

    def _initialize_security(self) -> None:
        """Initialize all security components."""
        # Configure sandbox
        if self.security_config.enable_sandboxing:
            configure_sandbox(
                base_path=self.storage_dir, custom_limits=self.security_config.get_resource_limits()
            )

        # Configure encryption
        if self.security_config.enable_index_encryption:
            master_key = self._get_or_generate_master_key()
            configure_crypto(master_key)

        # Configure audit logging
        if self.security_config.enable_audit_logging:
            configure_security_auditor(self.security_config.audit_log_file)

        # Create security middleware
        self.security_middleware = create_security_middleware(
            require_auth=self.security_config.require_authentication,
            enable_credential_scanning=self.security_config.enable_credential_scanning,
            allowed_base_paths=self.security_config.allowed_base_paths,
            api_keys_file=self.security_config.api_keys_file,
        )

        # Ensure security directories exist
        if self.security_config.api_keys_file:
            ensure_directory(self.security_config.api_keys_file.parent)

        if self.security_config.audit_log_file:
            ensure_directory(self.security_config.audit_log_file.parent)

    def _get_or_generate_master_key(self) -> bytes | None:
        """Get or generate master key for encryption.

        Returns:
            Master key bytes or None if encryption disabled
        """
        import base64
        import os

        master_key_env = os.environ.get(self.security_config.master_key_env_var)

        if master_key_env:
            try:
                return base64.b64decode(master_key_env)
            except Exception as e:
                logger.error("Failed to decode master key from environment", error=str(e))
                return None

        # Generate new master key
        master_key_b64 = generate_master_key()
        logger.warning(
            "Generated new master key for encryption",
            env_var=self.security_config.master_key_env_var,
            note="Set this as environment variable for production use",
        )

        print("\n=== MIMIR ENCRYPTION SETUP ===")
        print("Generated Master Key (set as environment variable):")
        print(f"export {self.security_config.master_key_env_var}={master_key_b64}")
        print("WARNING: Save this key securely - lost keys = lost data!")
        print("===============================\n")

        return base64.b64decode(master_key_b64)

    def _register_tools(self) -> None:
        """Register all MCP tools with security middleware."""

        @self.server.list_tools()
        async def list_tools() -> list[Tool]:
            """List available indexing tools."""
            return [
                Tool(
                    name="ensure_repo_index",
                    description="Ensure repository is indexed with full pipeline execution",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to repository root"},
                            "rev": {
                                "type": "string",
                                "description": "Git revision (defaults to HEAD)",
                            },
                            "language": {
                                "type": "string",
                                "description": "Primary language (defaults to 'ts')",
                                "default": "ts",
                            },
                            "index_opts": {
                                "type": "object",
                                "description": "Additional indexing options",
                            },
                        },
                        "required": ["path"],
                    },
                ),
                Tool(
                    name="get_repo_bundle",
                    description="Retrieve complete index bundle for a repository",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {"type": "string", "description": "Index identifier"}
                        },
                        "required": ["index_id"],
                    },
                ),
                Tool(
                    name="search_repo",
                    description="Search repository using hybrid vector + symbol + graph approach",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {"type": "string", "description": "Index identifier"},
                            "query": {"type": "string", "description": "Search query"},
                            "k": {
                                "type": "integer",
                                "description": "Number of results to return",
                                "default": 20,
                                "minimum": 1,
                                "maximum": 100,
                            },
                            "features": {
                                "type": "object",
                                "description": "Feature flags for search modalities",
                                "properties": {
                                    "vector": {"type": "boolean", "default": True},
                                    "symbol": {"type": "boolean", "default": True},
                                    "graph": {"type": "boolean", "default": True},
                                },
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Lines of context around matches",
                                "default": 5,
                                "minimum": 0,
                                "maximum": 20,
                            },
                        },
                        "required": ["index_id", "query"],
                    },
                ),
                Tool(
                    name="ask_index",
                    description="Ask complex questions using multi-hop symbol graph reasoning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {"type": "string", "description": "Index identifier"},
                            "question": {
                                "type": "string",
                                "description": "Question to ask about the codebase",
                            },
                            "context_lines": {
                                "type": "integer",
                                "description": "Lines of context for evidence",
                                "default": 5,
                                "minimum": 0,
                                "maximum": 20,
                            },
                        },
                        "required": ["index_id", "question"],
                    },
                ),
                Tool(
                    name="cancel",
                    description="Cancel an ongoing indexing operation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "index_id": {
                                "type": "string",
                                "description": "Index identifier to cancel",
                            }
                        },
                        "required": ["index_id"],
                    },
                ),
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            """Handle tool calls with security middleware."""
            try:
                if name == "ensure_repo_index":
                    return await self._secure_ensure_repo_index(arguments)
                elif name == "get_repo_bundle":
                    return await self._secure_get_repo_bundle(arguments)
                elif name == "search_repo":
                    return await self._secure_search_repo(arguments)
                elif name == "ask_index":
                    return await self._secure_ask_index(arguments)
                elif name == "cancel":
                    return await self._secure_cancel(arguments)
                else:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                        isError=True,
                    )
            except Exception as e:
                logger.exception(f"Error in secure tool {name}: {e}")
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Security error: {str(e)}")],
                    isError=True,
                )

    @property
    def _secure_ensure_repo_index(self):
        """Secure version of ensure_repo_index."""
        return self.security_middleware.secure_tool(
            permissions=["repo:index"], validate_input=True, scan_credentials=True
        )(self._ensure_repo_index)

    @property
    def _secure_get_repo_bundle(self):
        """Secure version of get_repo_bundle."""
        return self.security_middleware.secure_tool(
            permissions=["repo:read"], validate_input=True, scan_credentials=False
        )(self._get_repo_bundle)

    @property
    def _secure_search_repo(self):
        """Secure version of search_repo."""
        return self.security_middleware.secure_tool(
            permissions=["repo:search"], validate_input=True, scan_credentials=False
        )(self._search_repo)

    @property
    def _secure_ask_index(self):
        """Secure version of ask_index."""
        return self.security_middleware.secure_tool(
            permissions=["repo:ask"], validate_input=True, scan_credentials=False
        )(self._ask_index)

    @property
    def _secure_cancel(self):
        """Secure version of cancel."""
        return self.security_middleware.secure_tool(
            permissions=["repo:cancel"], validate_input=True, scan_credentials=False
        )(self._cancel)


class MimirSecureServer:
    """Main server class for running the secure MCP server."""

    def __init__(
        self, storage_dir: Path | None = None, security_config: SecurityConfig | None = None
    ):
        """Initialize the secure server.

        Args:
            storage_dir: Storage directory for indexes
            security_config: Security configuration
        """
        self.storage_dir = storage_dir or Path.home() / ".cache" / "mimir"
        self.security_config = security_config or get_security_config()
        self.mcp_server: SecureMCPServer | None = None

    async def run(self) -> None:
        """Run the secure MCP server."""
        logger.info("Starting Mimir secure MCP server")

        # Initialize secure MCP server
        self.mcp_server = SecureMCPServer(
            storage_dir=self.storage_dir, security_config=self.security_config
        )

        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await self.mcp_server.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="mimir-secure-repoindex",
                    server_version="1.0.0",
                    capabilities=self.mcp_server.server.get_capabilities(
                        tools=True, resources=True
                    ),
                ),
            )

    def cleanup(self) -> None:
        """Clean up server resources."""
        if self.mcp_server:
            # Clean up security components
            self.mcp_server.security_middleware.security_auditor.close()

        logger.info("Mimir secure MCP server shutdown complete")


async def main() -> None:
    """Main entry point for the secure server."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Mimir Secure MCP Server")
    parser.add_argument("--storage-dir", type=Path, help="Storage directory for indexes")
    parser.add_argument("--config-file", type=Path, help="Security configuration file")
    parser.add_argument(
        "--no-auth",
        action="store_true",
        help="Disable authentication (NOT recommended for production)",
    )
    parser.add_argument(
        "--generate-config",
        action="store_true",
        help="Generate default security configuration file and exit",
    )

    args = parser.parse_args()

    # Generate config if requested
    if args.generate_config:
        config = get_security_config()
        config_file = args.config_file or config.get_default_paths()["config_file"]
        config.to_file(config_file)
        print(f"Default security configuration saved to: {config_file}")
        return

    # Load security configuration
    security_config = None
    if args.config_file:
        from ..security.config import load_security_config_file

        security_config = load_security_config_file(args.config_file)
    else:
        security_config = get_security_config()

    # Override authentication if requested
    if args.no_auth:
        logger.warning("Authentication disabled - NOT recommended for production")
        security_config.require_authentication = False

    # Create and run server
    server = MimirSecureServer(storage_dir=args.storage_dir, security_config=security_config)

    try:
        await server.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error("Server error", error=str(e))
        sys.exit(1)
    finally:
        server.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
