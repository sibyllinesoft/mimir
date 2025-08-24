"""
Secrets management and credential scanning for security hardening.

Provides secure credential management and detection of sensitive
information in code repositories.
"""

import base64
import json
import os
import re
import secrets
import struct
import time
from pathlib import Path
from re import Pattern
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from ..util.errors import SecurityError
from ..util.log import get_logger

logger = get_logger(__name__)


class CredentialDetected(SecurityError):
    """Raised when credentials are detected in code."""

    pass


class SecretsError(SecurityError):
    """Raised when secrets management operations fail."""

    pass


class CredentialPattern:
    """Represents a pattern for detecting credentials."""

    def __init__(
        self,
        name: str,
        pattern: str,
        severity: str = "high",
        description: str = "",
        examples: list[str] | None = None,
    ):
        """Initialize credential pattern.

        Args:
            name: Name of the credential type
            pattern: Regular expression pattern
            severity: Severity level (low, medium, high, critical)
            description: Description of the credential type
            examples: Example patterns that should match
        """
        self.name = name
        self.pattern_string = pattern
        self.pattern = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        self.severity = severity
        self.description = description
        self.examples = examples or []


class CredentialScanner:
    """Scans code for embedded credentials and sensitive information."""

    def __init__(self):
        """Initialize credential scanner with default patterns."""
        self.patterns = self._load_default_patterns()
        self.whitelist_patterns = self._load_whitelist_patterns()

    def _load_default_patterns(self) -> list[CredentialPattern]:
        """Load default credential detection patterns.

        Returns:
            List of credential patterns
        """
        patterns = [
            # API Keys
            CredentialPattern(
                name="aws_access_key",
                pattern=r"AKIA[0-9A-Z]{16}",
                severity="critical",
                description="AWS Access Key ID",
            ),
            CredentialPattern(
                name="aws_secret_key",
                pattern=r"aws.{0,20}['\"][0-9a-zA-Z/+]{40}['\"]",
                severity="critical",
                description="AWS Secret Access Key",
            ),
            CredentialPattern(
                name="github_token",
                pattern=r"gh[pousr]_[A-Za-z0-9_]{36,255}",
                severity="high",
                description="GitHub Personal Access Token",
            ),
            CredentialPattern(
                name="slack_token",
                pattern=r"xox[baprs]-([0-9a-zA-Z]{10,48})",
                severity="high",
                description="Slack Token",
            ),
            CredentialPattern(
                name="google_api_key",
                pattern=r"AIza[0-9A-Za-z\-_]{35}",
                severity="high",
                description="Google API Key",
            ),
            CredentialPattern(
                name="openai_api_key",
                pattern=r"sk-[a-zA-Z0-9]{48}",
                severity="high",
                description="OpenAI API Key",
            ),
            CredentialPattern(
                name="anthropic_api_key",
                pattern=r"sk-ant-api03-[a-zA-Z0-9\-_]{95}",
                severity="high",
                description="Anthropic API Key",
            ),
            # Private Keys
            CredentialPattern(
                name="rsa_private_key",
                pattern=r"-----BEGIN RSA PRIVATE KEY-----",
                severity="critical",
                description="RSA Private Key",
            ),
            CredentialPattern(
                name="openssh_private_key",
                pattern=r"-----BEGIN OPENSSH PRIVATE KEY-----",
                severity="critical",
                description="OpenSSH Private Key",
            ),
            CredentialPattern(
                name="ec_private_key",
                pattern=r"-----BEGIN EC PRIVATE KEY-----",
                severity="critical",
                description="EC Private Key",
            ),
            # Database Credentials
            CredentialPattern(
                name="mysql_connection",
                pattern=r"mysql://[^:]+:[^@]+@[^/]+",
                severity="high",
                description="MySQL Connection String",
            ),
            CredentialPattern(
                name="postgres_connection",
                pattern=r"postgres(?:ql)?://[^:]+:[^@]+@[^/]+",
                severity="high",
                description="PostgreSQL Connection String",
            ),
            CredentialPattern(
                name="mongodb_connection",
                pattern=r"mongodb://[^:]+:[^@]+@[^/]+",
                severity="high",
                description="MongoDB Connection String",
            ),
            # Generic Patterns
            CredentialPattern(
                name="password_field",
                pattern=r"['\"]?password['\"]?\s*[:=]\s*['\"][^'\"]{8,}['\"]",
                severity="medium",
                description="Password Assignment",
            ),
            CredentialPattern(
                name="secret_field",
                pattern=r"['\"]?secret['\"]?\s*[:=]\s*['\"][^'\"]{16,}['\"]",
                severity="medium",
                description="Secret Assignment",
            ),
            CredentialPattern(
                name="token_field",
                pattern=r"['\"]?token['\"]?\s*[:=]\s*['\"][^'\"]{20,}['\"]",
                severity="medium",
                description="Token Assignment",
            ),
            CredentialPattern(
                name="api_key_field",
                pattern=r"['\"]?api[_-]?key['\"]?\s*[:=]\s*['\"][^'\"]{16,}['\"]",
                severity="high",
                description="API Key Assignment",
            ),
            # Cloud Provider Patterns
            CredentialPattern(
                name="azure_client_secret",
                pattern=r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
                severity="high",
                description="Azure Client Secret (GUID format)",
            ),
            CredentialPattern(
                name="gcp_service_account",
                pattern=r"\{[^}]*\"type\":\s*\"service_account\"[^}]*\}",
                severity="critical",
                description="GCP Service Account JSON",
            ),
            # JWT Tokens
            CredentialPattern(
                name="jwt_token",
                pattern=r"eyJ[A-Za-z0-9_/+-]*\.eyJ[A-Za-z0-9_/+-]*\.[A-Za-z0-9_/+-]*",
                severity="medium",
                description="JWT Token",
            ),
            # Base64 Encoded Secrets (potential)
            CredentialPattern(
                name="base64_secret",
                pattern=r"(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?",
                severity="low",
                description="Base64 Encoded Data (potential secret)",
            ),
        ]

        return patterns

    def _load_whitelist_patterns(self) -> list[Pattern]:
        """Load whitelist patterns to exclude false positives.

        Returns:
            List of compiled regex patterns for whitelisting
        """
        whitelist_strings = [
            # Common test/example values
            r"password.*example",
            r"secret.*test",
            r"token.*demo",
            r"key.*sample",
            r"YOUR_API_KEY",
            r"YOUR_SECRET",
            r"YOUR_PASSWORD",
            r"INSERT_API_KEY_HERE",
            r"REPLACE_WITH_YOUR",
            r"TODO.*secret",
            r"FIXME.*password",
            # Common placeholders
            r"xxxxxxxxxx",
            r"1234567890",
            r"abcdef",
            r"secret123",
            r"password123",
            r"test_token",
            # Documentation examples
            r"sk-.*example",
            r"ghp_.*example",
            # Common false positives
            r"Bearer\s+\$\{",  # Template variables
            r"process\.env\.",  # Environment variables
            r"\$\{[^}]+\}",  # Template substitutions
        ]

        return [re.compile(pattern, re.IGNORECASE) for pattern in whitelist_strings]

    def add_pattern(self, pattern: CredentialPattern) -> None:
        """Add a custom credential pattern.

        Args:
            pattern: Credential pattern to add
        """
        self.patterns.append(pattern)
        logger.debug("Added credential pattern: %s", pattern.name)

    def scan_text(self, text: str, file_path: str | None = None) -> list[dict[str, Any]]:
        """Scan text for embedded credentials.

        Args:
            text: Text content to scan
            file_path: Optional file path for context

        Returns:
            List of detected credentials
        """
        detected_credentials = []

        for pattern in self.patterns:
            matches = pattern.pattern.finditer(text)

            for match in matches:
                match_text = match.group(0)

                # Check against whitelist
                is_whitelisted = any(
                    whitelist_pattern.search(match_text)
                    for whitelist_pattern in self.whitelist_patterns
                )

                if is_whitelisted:
                    continue

                # Get line number and context
                lines_before = text[: match.start()].count("\n")
                line_number = lines_before + 1

                # Get surrounding context
                start_line = max(0, lines_before - 2)
                end_line = min(len(text.split("\n")), lines_before + 3)
                context_lines = text.split("\n")[start_line:end_line]

                credential_info = {
                    "pattern_name": pattern.name,
                    "severity": pattern.severity,
                    "description": pattern.description,
                    "match": match_text[:100],  # Truncate for security
                    "file_path": file_path,
                    "line_number": line_number,
                    "start_pos": match.start(),
                    "end_pos": match.end(),
                    "context": context_lines,
                }

                detected_credentials.append(credential_info)

                logger.warning(
                    "Credential detected: %s (severity=%s) in %s at line %d",
                    pattern.name,
                    pattern.severity,
                    file_path or "<string>",
                    line_number,
                )

        return detected_credentials

    def scan_file(self, file_path: Path) -> list[dict[str, Any]]:
        """Scan a file for embedded credentials.

        Args:
            file_path: Path to file to scan

        Returns:
            List of detected credentials

        Raises:
            SecurityError: If file cannot be scanned
        """
        try:
            # Check file size
            file_size = file_path.stat().st_size
            max_size = 10 * 1024 * 1024  # 10MB limit for scanning

            if file_size > max_size:
                logger.warning(
                    "File too large for credential scanning: %s (%.1f MB)",
                    str(file_path),
                    file_size / (1024 * 1024),
                )
                return []

            # Read file content
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                content = f.read()

            return self.scan_text(content, str(file_path))

        except Exception as e:
            logger.error(
                "Failed to scan file for credentials %s: %s", str(file_path), str(e)
            )
            return []

    def scan_directory(
        self, directory: Path, extensions: set[str] | None = None, max_files: int = 1000
    ) -> dict[str, list[dict[str, Any]]]:
        """Scan a directory for embedded credentials.

        Args:
            directory: Directory to scan
            extensions: File extensions to scan (if None, scans common code files)
            max_files: Maximum number of files to scan

        Returns:
            Dictionary mapping file paths to detected credentials
        """
        if extensions is None:
            extensions = {
                ".py",
                ".js",
                ".ts",
                ".java",
                ".cpp",
                ".c",
                ".h",
                ".cs",
                ".go",
                ".rs",
                ".php",
                ".rb",
                ".kt",
                ".swift",
                ".yml",
                ".yaml",
                ".json",
                ".xml",
                ".properties",
                ".env",
                ".config",
                ".ini",
                ".conf",
            }

        results = {}
        files_scanned = 0

        try:
            for file_path in directory.rglob("*"):
                if files_scanned >= max_files:
                    logger.warning(
                        "Maximum file limit reached for credential scanning: %d files", max_files
                    )
                    break

                if not file_path.is_file():
                    continue

                if file_path.suffix.lower() not in extensions:
                    continue

                # Skip binary files and large files
                try:
                    if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                        continue
                except OSError:
                    continue

                credentials = self.scan_file(file_path)
                if credentials:
                    results[str(file_path)] = credentials

                files_scanned += 1

        except Exception as e:
            logger.error(
                "Failed to scan directory for credentials %s: %s", str(directory), str(e)
            )

        logger.info(
            "Directory credential scan completed: %s (%d files scanned, %d with credentials)",
            str(directory),
            files_scanned,
            len(results),
        )

        return results


class SecretsManager:
    """Manages encrypted storage of secrets and configuration.
    
    This implementation uses a secure salt storage format:
    - Each secrets file has its own randomly generated salt
    - Salt is stored in the file header for secure key derivation
    - Backward compatibility is maintained through migration
    """

    # File format constants
    MAGIC_HEADER = b"MIMIR_SECRETS_V2"
    LEGACY_SALT = b"mimir_salt_v1"
    SALT_SIZE = 32
    HEADER_SIZE = len(MAGIC_HEADER) + 4 + SALT_SIZE  # magic + salt_len + salt

    def __init__(self, secrets_file: Path | None = None, password: str | None = None):
        """Initialize secrets manager.

        Args:
            secrets_file: Path to encrypted secrets file
            password: Password for encryption (if None, uses environment variable)
        """
        self.secrets_file = secrets_file
        self.password = password or os.environ.get("MIMIR_SECRETS_PASSWORD")
        self.fernet = None
        self.secrets_cache: dict[str, Any] = {}
        self.salt: bytes | None = None

        if self.password:
            self._initialize_encryption()

    def _initialize_encryption(self) -> None:
        """Initialize encryption with password-based key derivation.
        
        Uses secure salt generation and storage:
        - New files get randomly generated salts
        - Existing files are migrated from legacy format if needed
        - Salt is stored securely in file header
        """
        if not self.password:
            raise SecretsError("No password provided for secrets encryption")

        # Load salt from existing file or generate new one
        if self.secrets_file and self.secrets_file.exists():
            # Check if file has actual content
            if self.secrets_file.stat().st_size > 0:
                self.salt = self._load_salt_from_file()
                if self.salt is None:
                    # Legacy file detected - use legacy salt for compatibility
                    self.salt = self.LEGACY_SALT
                    logger.warning(
                        "Legacy secrets file detected - will migrate to secure format on next save: %s",
                        str(self.secrets_file)
                    )
                else:
                    logger.debug("Loaded salt from existing secure file")
            else:
                # Empty file - treat as new
                self.salt = secrets.token_bytes(self.SALT_SIZE)
                logger.info("Generated new random salt for empty secrets file")
        else:
            # Generate new random salt for new files
            self.salt = secrets.token_bytes(self.SALT_SIZE)
            logger.info("Generated new random salt for new secrets file")

        # Derive encryption key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
        self.fernet = Fernet(key)

        # Load existing secrets if file exists
        if self.secrets_file and self.secrets_file.exists():
            self._load_secrets()

    def _load_salt_from_file(self) -> bytes | None:
        """Load salt from the secrets file header.
        
        Returns:
            Salt bytes if new format file, None if legacy format
        """
        if not self.secrets_file or not self.secrets_file.exists():
            return None
            
        try:
            with open(self.secrets_file, "rb") as f:
                # Check for new format header
                header_data = f.read(len(self.MAGIC_HEADER))
                if header_data == self.MAGIC_HEADER:
                    # New format - read salt
                    salt_len_data = f.read(4)
                    if len(salt_len_data) != 4:
                        return None
                    salt_len = struct.unpack("<I", salt_len_data)[0]
                    if salt_len != self.SALT_SIZE:
                        logger.warning("Unexpected salt size in secrets file: %d", salt_len)
                        return None
                    salt = f.read(salt_len)
                    if len(salt) != salt_len:
                        return None
                    return salt
                else:
                    # Legacy format - no salt header
                    return None
        except Exception as e:
            logger.error("Failed to load salt from file: %s", str(e))
            return None

    def _load_secrets(self) -> None:
        """Load and decrypt secrets from file.
        
        Supports both new format (with salt header) and legacy format.
        """
        if not self.secrets_file or not self.secrets_file.exists():
            return

        try:
            with open(self.secrets_file, "rb") as f:
                file_data = f.read()

            if not file_data:
                return

            # Determine if this is new format or legacy format
            encrypted_data = file_data
            if file_data.startswith(self.MAGIC_HEADER):
                # New format - skip header to get encrypted data
                encrypted_data = file_data[self.HEADER_SIZE:]
                logger.debug("Loading secrets from new format file")
            else:
                # Legacy format - entire file is encrypted data
                logger.debug("Loading secrets from legacy format file")

            decrypted_data = self.fernet.decrypt(encrypted_data)
            self.secrets_cache = json.loads(decrypted_data.decode("utf-8"))

            logger.info("Secrets loaded: %d secrets", len(self.secrets_cache))

        except Exception as e:
            logger.error("Failed to load secrets: %s", str(e))
            raise SecretsError(f"Failed to load secrets: {e}")

    def _save_secrets(self) -> None:
        """Encrypt and save secrets to file using secure format.
        
        New format includes:
        - Magic header for format identification
        - Random salt for secure key derivation
        - Encrypted secrets data
        """
        if not self.secrets_file or not self.fernet or not self.salt:
            return

        try:
            # Ensure directory exists
            self.secrets_file.parent.mkdir(parents=True, exist_ok=True)

            # Encrypt secrets data
            json_data = json.dumps(self.secrets_cache, indent=2)
            encrypted_data = self.fernet.encrypt(json_data.encode("utf-8"))

            # Write atomically with new secure format
            temp_file = self.secrets_file.with_suffix(".tmp")
            with open(temp_file, "wb") as f:
                # Write header with salt
                f.write(self.MAGIC_HEADER)
                f.write(struct.pack("<I", len(self.salt)))
                f.write(self.salt)
                
                # Write encrypted data
                f.write(encrypted_data)

            temp_file.replace(self.secrets_file)

            # If we were using legacy salt, generate new salt for future operations
            if self.salt == self.LEGACY_SALT:
                logger.info("Migrated legacy secrets file to secure format")
                self.salt = secrets.token_bytes(self.SALT_SIZE)
                # Re-derive key with new salt (don't re-initialize to avoid recursion)
                from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
                from cryptography.hazmat.primitives import hashes
                
                kdf = PBKDF2HMAC(
                    algorithm=hashes.SHA256(),
                    length=32,
                    salt=self.salt,
                    iterations=100000,
                )
                key = base64.urlsafe_b64encode(kdf.derive(self.password.encode()))
                self.fernet = Fernet(key)

            logger.debug("Secrets saved: %d secrets", len(self.secrets_cache))

        except Exception as e:
            logger.error("Failed to save secrets: %s", str(e))
            raise SecretsError(f"Failed to save secrets: {e}")

    def set_secret(self, key: str, value: Any) -> None:
        """Set a secret value.

        Args:
            key: Secret key
            value: Secret value
        """
        if not self.fernet:
            raise SecretsError("Secrets manager not initialized with password")

        self.secrets_cache[key] = value
        self._save_secrets()

        logger.debug("Secret set: %s", key)

    def get_secret(self, key: str, default: Any = None) -> Any:
        """Get a secret value.

        Args:
            key: Secret key
            default: Default value if key not found

        Returns:
            Secret value or default
        """
        return self.secrets_cache.get(key, default)

    def delete_secret(self, key: str) -> bool:
        """Delete a secret.

        Args:
            key: Secret key to delete

        Returns:
            True if secret was deleted, False if not found
        """
        if key in self.secrets_cache:
            del self.secrets_cache[key]
            self._save_secrets()
            logger.debug("Secret deleted: %s", key)
            return True
        return False

    def list_secrets(self) -> list[str]:
        """List all secret keys.

        Returns:
            List of secret keys
        """
        return list(self.secrets_cache.keys())

    def has_secret(self, key: str) -> bool:
        """Check if a secret exists.

        Args:
            key: Secret key to check

        Returns:
            True if secret exists
        """
        return key in self.secrets_cache

    def rotate_encryption_key(self, new_password: str) -> None:
        """Rotate the encryption key with a new password.
        
        This also generates a new random salt for enhanced security.

        Args:
            new_password: New password for encryption
        """
        # Decrypt with old key
        old_secrets = self.secrets_cache.copy()

        # Generate new salt and initialize with new password
        self.salt = secrets.token_bytes(self.SALT_SIZE)
        self.password = new_password
        self._initialize_encryption()

        # Re-encrypt with new key and salt
        self.secrets_cache = old_secrets
        self._save_secrets()

        logger.info("Encryption key and salt rotated")

    def export_secrets(self, export_path: Path, include_metadata: bool = False) -> None:
        """Export secrets to a file (for backup/migration).

        Args:
            export_path: Path to export file
            include_metadata: Whether to include metadata
        """
        export_data = {
            "secrets": self.secrets_cache,
            "metadata": {"exported_at": time.time(), "version": "1.0"} if include_metadata else {},
        }

        # Encrypt export
        json_data = json.dumps(export_data, indent=2)
        encrypted_data = self.fernet.encrypt(json_data.encode("utf-8"))

        with open(export_path, "wb") as f:
            f.write(encrypted_data)

        logger.info("Secrets exported to: %s", str(export_path))

    def import_secrets(self, import_path: Path, merge: bool = False) -> None:
        """Import secrets from a file.

        Args:
            import_path: Path to import file
            merge: Whether to merge with existing secrets or replace
        """
        with open(import_path, "rb") as f:
            encrypted_data = f.read()

        decrypted_data = self.fernet.decrypt(encrypted_data)
        import_data = json.loads(decrypted_data.decode("utf-8"))

        imported_secrets = import_data.get("secrets", {})

        if merge:
            self.secrets_cache.update(imported_secrets)
        else:
            self.secrets_cache = imported_secrets

        self._save_secrets()

        logger.info("Secrets imported from %s (merge=%s)", str(import_path), merge)

    def migrate_legacy_file(self) -> bool:
        """Migrate a legacy secrets file to the new secure format.
        
        Returns:
            True if migration was performed, False if not needed
        """
        if not self.secrets_file or not self.secrets_file.exists():
            return False
            
        # Check if file is already in new format
        if self._load_salt_from_file() is not None:
            logger.debug("Secrets file already in secure format")
            return False
            
        logger.info("Migrating legacy secrets file to secure format")
        
        # Load secrets with legacy format
        old_secrets = self.secrets_cache.copy()
        
        # Generate new salt and re-initialize
        self.salt = secrets.token_bytes(self.SALT_SIZE)
        self._initialize_encryption()
        
        # Restore secrets and save in new format
        self.secrets_cache = old_secrets
        self._save_secrets()
        
        logger.info("Legacy secrets file migrated successfully")
        return True

    def verify_file_integrity(self) -> bool:
        """Verify the integrity of the secrets file.
        
        Returns:
            True if file is valid and can be decrypted
        """
        if not self.secrets_file or not self.secrets_file.exists():
            return False
            
        try:
            # Try to load and decrypt the file
            original_cache = self.secrets_cache.copy()
            self._load_secrets()
            
            # Restore original cache
            self.secrets_cache = original_cache
            return True
            
        except Exception as e:
            logger.error("Secrets file integrity check failed: %s", str(e))
            return False

    def get_file_info(self) -> dict[str, Any]:
        """Get information about the secrets file format and security.
        
        Returns:
            Dictionary with file format information
        """
        if not self.secrets_file or not self.secrets_file.exists():
            return {"exists": False}
            
        try:
            with open(self.secrets_file, "rb") as f:
                header = f.read(len(self.MAGIC_HEADER))
                file_size = f.seek(0, 2)  # Get file size
                
            is_new_format = header == self.MAGIC_HEADER
            salt_info = "random" if is_new_format else "legacy_hardcoded"
            
            return {
                "exists": True,
                "format_version": "v2" if is_new_format else "v1_legacy",
                "salt_type": salt_info,
                "security_level": "high" if is_new_format else "low",
                "file_size": file_size,
                "migration_needed": not is_new_format,
            }
            
        except Exception as e:
            logger.error("Failed to get file info: %s", str(e))
            return {"exists": True, "error": str(e)}
