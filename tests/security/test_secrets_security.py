"""
Comprehensive security tests for the fixed SecretsManager implementation.

Tests the security improvements including:
- Random salt generation
- Legacy file migration
- Cryptographic security properties
- Attack resistance
"""

import json
import os
import secrets
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.repoindex.security.secrets import SecretsError, SecretsManager


class TestSecretsManagerSecurity:
    """Test security properties of the fixed SecretsManager."""

    def test_random_salt_generation(self):
        """Test that new files get unique random salts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple secrets managers with same password
            managers = []
            for i in range(5):
                secrets_file = Path(temp_dir) / f"secrets_{i}.enc"
                manager = SecretsManager(secrets_file=secrets_file, password="test_password")
                manager.set_secret(f"test_key_{i}", f"test_value_{i}")
                managers.append(manager)

            # Verify each manager has a unique salt
            salts = [manager.salt for manager in managers]
            assert len(set(salts)) == 5, "All salts should be unique"
            
            # Verify salt length
            for salt in salts:
                assert len(salt) == SecretsManager.SALT_SIZE
                assert salt != SecretsManager.LEGACY_SALT

    def test_legacy_migration_detection(self):
        """Test detection and migration of legacy files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "legacy_secrets.enc"
            
            # Create a legacy format file manually
            manager = SecretsManager(password="test_password")
            manager.salt = SecretsManager.LEGACY_SALT
            manager._initialize_encryption()
            manager.secrets_cache = {"legacy_key": "legacy_value"}
            
            # Save in legacy format (simulate old implementation)
            json_data = json.dumps(manager.secrets_cache, indent=2)
            encrypted_data = manager.fernet.encrypt(json_data.encode("utf-8"))
            with open(secrets_file, "wb") as f:
                f.write(encrypted_data)  # No header, just encrypted data
                
            # Create new manager and verify legacy detection
            new_manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            file_info = new_manager.get_file_info()
            
            assert file_info["format_version"] == "v1_legacy"
            assert file_info["salt_type"] == "legacy_hardcoded"
            assert file_info["security_level"] == "low"
            assert file_info["migration_needed"] is True
            
            # Verify legacy data can be read
            assert new_manager.get_secret("legacy_key") == "legacy_value"

    def test_automatic_migration_on_save(self):
        """Test that legacy files are automatically migrated on save."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "migration_test.enc"
            
            # Create legacy file
            manager = SecretsManager(password="test_password")
            manager.salt = SecretsManager.LEGACY_SALT
            manager._initialize_encryption()
            manager.secrets_cache = {"old_key": "old_value"}
            
            # Save legacy format
            json_data = json.dumps(manager.secrets_cache, indent=2)
            encrypted_data = manager.fernet.encrypt(json_data.encode("utf-8"))
            with open(secrets_file, "wb") as f:
                f.write(encrypted_data)
                
            # Load with new manager and trigger migration by saving
            new_manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            assert new_manager.salt == SecretsManager.LEGACY_SALT  # Initially legacy
            
            new_manager.set_secret("new_key", "new_value")  # This triggers save and migration
            
            # Verify file is now in new format
            file_info = new_manager.get_file_info()
            assert file_info["format_version"] == "v2"
            assert file_info["salt_type"] == "random"
            assert file_info["security_level"] == "high"
            assert file_info["migration_needed"] is False

    def test_salt_uniqueness_across_key_rotations(self):
        """Test that key rotation generates new unique salts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "rotation_test.enc"
            
            manager = SecretsManager(secrets_file=secrets_file, password="password1")
            manager.set_secret("test_key", "test_value")
            original_salt = manager.salt
            
            # Rotate key and verify new salt
            manager.rotate_encryption_key("password2")
            new_salt = manager.salt
            
            assert new_salt != original_salt
            assert len(new_salt) == SecretsManager.SALT_SIZE
            
            # Verify data is still accessible
            assert manager.get_secret("test_key") == "test_value"
            
            # Rotate again
            manager.rotate_encryption_key("password3")
            third_salt = manager.salt
            
            assert third_salt != new_salt
            assert third_salt != original_salt

    def test_cryptographic_strength(self):
        """Test cryptographic properties of the implementation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "crypto_test.enc"
            
            # Create manager and add secrets
            manager = SecretsManager(secrets_file=secrets_file, password="strong_password")
            test_data = {
                "api_key": "sk-very-secret-api-key-12345",
                "db_password": "super_secure_db_password_67890",
                "jwt_secret": "jwt-signing-secret-abcdef"
            }
            
            for key, value in test_data.items():
                manager.set_secret(key, value)
                
            # Read raw file and verify encryption
            with open(secrets_file, "rb") as f:
                file_content = f.read()
                
            # Verify no plaintext secrets in file
            file_str = file_content.decode("latin-1", errors="ignore")
            for value in test_data.values():
                assert value not in file_str, f"Plaintext secret found: {value}"
                
            # Verify file structure
            assert file_content.startswith(SecretsManager.MAGIC_HEADER)
            
            # Extract and verify salt
            salt_len = struct.unpack("<I", file_content[len(SecretsManager.MAGIC_HEADER):len(SecretsManager.MAGIC_HEADER)+4])[0]
            assert salt_len == SecretsManager.SALT_SIZE
            
            extracted_salt = file_content[len(SecretsManager.MAGIC_HEADER)+4:len(SecretsManager.MAGIC_HEADER)+4+salt_len]
            assert extracted_salt == manager.salt
            assert len(extracted_salt) == SecretsManager.SALT_SIZE

    def test_password_resistance(self):
        """Test that different passwords produce different results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "password_test.enc"
            
            # Create file with first password
            manager1 = SecretsManager(secrets_file=secrets_file, password="password1")
            manager1.set_secret("test_key", "test_value")
            
            # Try to read with wrong password
            manager2 = SecretsManager(secrets_file=secrets_file, password="wrong_password")
            with pytest.raises(SecretsError):
                _ = manager2.get_secret("test_key")

    def test_salt_entropy(self):
        """Test that generated salts have high entropy."""
        # Generate multiple salts and verify they're sufficiently random
        salts = []
        for _ in range(100):
            with tempfile.TemporaryDirectory() as temp_dir:
                secrets_file = Path(temp_dir) / "entropy_test.enc"
                manager = SecretsManager(secrets_file=secrets_file, password="test_password")
                salts.append(manager.salt)
        
        # Basic entropy check - all salts should be unique
        assert len(set(salts)) == 100, "All salts should be unique"
        
        # Check that salts don't have obvious patterns
        for salt in salts[:10]:  # Check first 10
            # Salt should not be all zeros
            assert salt != b"\x00" * SecretsManager.SALT_SIZE
            # Salt should not be repeating byte
            assert not all(b == salt[0] for b in salt)

    def test_file_integrity_verification(self):
        """Test file integrity verification functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "integrity_test.enc"
            
            # Create valid file
            manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            manager.set_secret("test_key", "test_value")
            
            # Verify integrity
            assert manager.verify_file_integrity() is True
            
            # Corrupt file and verify integrity fails
            with open(secrets_file, "r+b") as f:
                f.seek(-10, 2)  # Go to near end
                f.write(b"corruption")
            
            # Create new manager and test integrity
            corrupted_manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            assert corrupted_manager.verify_file_integrity() is False

    def test_secure_cleanup(self):
        """Test that sensitive data is properly cleaned up."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "cleanup_test.enc"
            
            # Create manager with sensitive data
            manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            manager.set_secret("sensitive_key", "very_sensitive_data")
            
            # Store references to check cleanup
            original_salt = manager.salt
            original_cache = manager.secrets_cache.copy()
            
            # Rotate key (which should cleanup old data)
            manager.rotate_encryption_key("new_password")
            
            # Verify new salt is different (old one should be replaced)
            assert manager.salt != original_salt
            
            # Verify secrets are still accessible
            assert manager.get_secret("sensitive_key") == "very_sensitive_data"

    def test_concurrent_access_safety(self):
        """Test thread safety of secrets operations."""
        import threading
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "concurrent_test.enc"
            manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            
            results = {}
            errors = []
            
            def worker(worker_id: int):
                try:
                    # Each worker sets and gets its own secret
                    key = f"worker_{worker_id}"
                    value = f"value_{worker_id}_{secrets.token_hex(8)}"
                    
                    manager.set_secret(key, value)
                    time.sleep(0.01)  # Small delay
                    retrieved_value = manager.get_secret(key)
                    
                    results[worker_id] = (value, retrieved_value)
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")
            
            # Start multiple threads
            threads = []
            for i in range(10):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for completion
            for thread in threads:
                thread.join()
            
            # Verify results
            assert len(errors) == 0, f"Concurrent access errors: {errors}"
            assert len(results) == 10
            
            for worker_id, (original, retrieved) in results.items():
                assert original == retrieved, f"Data mismatch for worker {worker_id}"


class TestSecurityRegression:
    """Regression tests to ensure the vulnerability is fixed."""

    def test_no_hardcoded_salt_in_new_files(self):
        """Ensure new files never use hardcoded salt."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create multiple files
            for i in range(10):
                secrets_file = Path(temp_dir) / f"test_{i}.enc"
                manager = SecretsManager(secrets_file=secrets_file, password="test_password")
                manager.set_secret("test_key", "test_value")
                
                # Verify salt is not the legacy hardcoded salt
                assert manager.salt != SecretsManager.LEGACY_SALT
                assert len(manager.salt) == SecretsManager.SALT_SIZE

    def test_legacy_salt_only_for_compatibility(self):
        """Ensure legacy salt is only used for reading existing files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "legacy_test.enc"
            
            # Create legacy format file
            manager = SecretsManager(password="test_password")
            manager.salt = SecretsManager.LEGACY_SALT
            manager._initialize_encryption()
            manager.secrets_cache = {"legacy_key": "legacy_value"}
            
            # Save in old format
            json_data = json.dumps(manager.secrets_cache, indent=2)
            encrypted_data = manager.fernet.encrypt(json_data.encode("utf-8"))
            with open(secrets_file, "wb") as f:
                f.write(encrypted_data)
            
            # Load with new manager
            new_manager = SecretsManager(secrets_file=secrets_file, password="test_password")
            assert new_manager.salt == SecretsManager.LEGACY_SALT
            
            # Add new secret (triggers migration)
            new_manager.set_secret("new_key", "new_value")
            
            # Verify migration happened
            assert new_manager.salt != SecretsManager.LEGACY_SALT

    def test_salt_randomness_statistical(self):
        """Statistical test for salt randomness."""
        salts = []
        
        # Generate many salts
        for _ in range(1000):
            with tempfile.TemporaryDirectory() as temp_dir:
                secrets_file = Path(temp_dir) / "random_test.enc"
                manager = SecretsManager(secrets_file=secrets_file, password="test_password")
                salts.append(manager.salt)
        
        # Convert to integers for statistical analysis
        salt_ints = [int.from_bytes(salt, byteorder='big') for salt in salts]
        
        # Basic statistical tests
        mean = sum(salt_ints) / len(salt_ints)
        variance = sum((x - mean) ** 2 for x in salt_ints) / len(salt_ints)
        
        # For truly random data, we expect high variance
        # This is a simple test - in production you'd use more sophisticated tests
        max_possible = 2 ** (SecretsManager.SALT_SIZE * 8) - 1
        expected_mean = max_possible / 2
        
        # Allow for reasonable deviation (this is not a rigorous statistical test)
        assert abs(mean - expected_mean) < expected_mean * 0.1, "Salt distribution seems non-random"
        assert variance > expected_mean ** 2 * 0.01, "Salt variance too low for random data"

    @patch.dict(os.environ, {"MIMIR_SECRETS_PASSWORD": "env_password"})
    def test_environment_variable_security(self):
        """Test that environment variable usage doesn't compromise security."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_file = Path(temp_dir) / "env_test.enc"
            
            # Create manager using environment variable
            manager = SecretsManager(secrets_file=secrets_file)
            manager.set_secret("env_test_key", "env_test_value")
            
            # Verify random salt is still generated
            assert manager.salt != SecretsManager.LEGACY_SALT
            assert len(manager.salt) == SecretsManager.SALT_SIZE
            
            # Verify secrets work correctly
            assert manager.get_secret("env_test_key") == "env_test_value"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])