"""
Cryptographic operations and data encryption for security hardening.

Provides encryption for stored embeddings, indexes, and sensitive data
to protect against unauthorized access.
"""

import hashlib
import hmac
import os
import secrets
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import json
import gzip
import base64

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from ..util.errors import SecurityError
from ..util.log import get_logger

logger = get_logger(__name__)


class CryptoError(SecurityError):
    """Raised when cryptographic operations fail."""
    pass


class KeyDerivationError(CryptoError):
    """Raised when key derivation fails."""
    pass


class EncryptionError(CryptoError):
    """Raised when encryption operations fail."""
    pass


class DecryptionError(CryptoError):
    """Raised when decryption operations fail."""
    pass


class CryptoManager:
    """Main cryptographic operations manager."""
    
    def __init__(self, master_key: Optional[bytes] = None):
        """Initialize crypto manager.
        
        Args:
            master_key: Optional master key for encryption (32 bytes)
        """
        self.master_key = master_key
        self.key_cache: Dict[str, bytes] = {}
        
        if self.master_key and len(self.master_key) != 32:
            raise CryptoError("Master key must be exactly 32 bytes")
    
    def generate_key(self) -> bytes:
        """Generate a new cryptographic key.
        
        Returns:
            32-byte cryptographic key
        """
        return secrets.token_bytes(32)
    
    def generate_salt(self, length: int = 32) -> bytes:
        """Generate a new random salt.
        
        Args:
            length: Length of salt in bytes
            
        Returns:
            Random salt bytes
        """
        return secrets.token_bytes(length)
    
    def derive_key_pbkdf2(
        self,
        password: Union[str, bytes],
        salt: bytes,
        iterations: int = 100000,
        key_length: int = 32
    ) -> bytes:
        """Derive key from password using PBKDF2.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            iterations: Number of iterations
            key_length: Length of derived key
            
        Returns:
            Derived key
            
        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            if isinstance(password, str):
                password = password.encode('utf-8')
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                iterations=iterations,
                backend=default_backend()
            )
            
            return kdf.derive(password)
            
        except Exception as e:
            raise KeyDerivationError(f"PBKDF2 key derivation failed: {e}")
    
    def derive_key_scrypt(
        self,
        password: Union[str, bytes],
        salt: bytes,
        n: int = 2**14,  # CPU/memory cost
        r: int = 8,      # Block size
        p: int = 1,      # Parallelization
        key_length: int = 32
    ) -> bytes:
        """Derive key from password using Scrypt.
        
        Args:
            password: Password to derive key from
            salt: Salt for key derivation
            n: CPU/memory cost parameter
            r: Block size parameter
            p: Parallelization parameter
            key_length: Length of derived key
            
        Returns:
            Derived key
            
        Raises:
            KeyDerivationError: If key derivation fails
        """
        try:
            if isinstance(password, str):
                password = password.encode('utf-8')
            
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=key_length,
                salt=salt,
                n=n,
                r=r,
                p=p,
                backend=default_backend()
            )
            
            return kdf.derive(password)
            
        except Exception as e:
            raise KeyDerivationError(f"Scrypt key derivation failed: {e}")
    
    def compute_hash(self, data: bytes, algorithm: str = "sha256") -> str:
        """Compute hash of data.
        
        Args:
            data: Data to hash
            algorithm: Hash algorithm (sha256, sha384, sha512)
            
        Returns:
            Hex-encoded hash
        """
        hash_funcs = {
            "sha256": hashlib.sha256,
            "sha384": hashlib.sha384,
            "sha512": hashlib.sha512
        }
        
        if algorithm not in hash_funcs:
            raise CryptoError(f"Unsupported hash algorithm: {algorithm}")
        
        hasher = hash_funcs[algorithm]()
        hasher.update(data)
        return hasher.hexdigest()
    
    def compute_hmac(self, data: bytes, key: bytes, algorithm: str = "sha256") -> str:
        """Compute HMAC of data.
        
        Args:
            data: Data to authenticate
            key: HMAC key
            algorithm: Hash algorithm
            
        Returns:
            Hex-encoded HMAC
        """
        hash_funcs = {
            "sha256": hashlib.sha256,
            "sha384": hashlib.sha384,
            "sha512": hashlib.sha512
        }
        
        if algorithm not in hash_funcs:
            raise CryptoError(f"Unsupported hash algorithm: {algorithm}")
        
        return hmac.new(key, data, hash_funcs[algorithm]).hexdigest()
    
    def verify_hmac(
        self,
        data: bytes,
        expected_hmac: str,
        key: bytes,
        algorithm: str = "sha256"
    ) -> bool:
        """Verify HMAC of data.
        
        Args:
            data: Data to verify
            expected_hmac: Expected HMAC value
            key: HMAC key
            algorithm: Hash algorithm
            
        Returns:
            True if HMAC is valid
        """
        try:
            computed_hmac = self.compute_hmac(data, key, algorithm)
            return hmac.compare_digest(computed_hmac, expected_hmac)
        except Exception:
            return False


class FileEncryption:
    """Handles encryption and decryption of files."""
    
    def __init__(self, crypto_manager: CryptoManager):
        """Initialize file encryption.
        
        Args:
            crypto_manager: Crypto manager instance
        """
        self.crypto_manager = crypto_manager
    
    def encrypt_file(
        self,
        input_path: Path,
        output_path: Path,
        password: Optional[str] = None,
        compress: bool = True
    ) -> Dict[str, Any]:
        """Encrypt a file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output encrypted file
            password: Optional password for encryption
            compress: Whether to compress before encryption
            
        Returns:
            Dictionary with encryption metadata
            
        Raises:
            EncryptionError: If encryption fails
        """
        try:
            # Read input file
            with open(input_path, 'rb') as f:
                data = f.read()
            
            # Compress if requested
            if compress:
                data = gzip.compress(data)
            
            # Generate encryption parameters
            salt = self.crypto_manager.generate_salt()
            iv = secrets.token_bytes(16)  # AES block size
            
            # Derive encryption key
            if password:
                key = self.crypto_manager.derive_key_scrypt(password, salt)
            elif self.crypto_manager.master_key:
                key = self.crypto_manager.master_key
            else:
                raise EncryptionError("No password or master key provided")
            
            # Encrypt data using AES-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            auth_tag = encryptor.tag
            
            # Create encrypted file header
            header = {
                "version": 1,
                "algorithm": "AES-256-GCM",
                "compressed": compress,
                "salt": base64.b64encode(salt).decode('ascii'),
                "iv": base64.b64encode(iv).decode('ascii'),
                "auth_tag": base64.b64encode(auth_tag).decode('ascii'),
                "original_size": len(data) if not compress else os.path.getsize(input_path),
                "file_hash": self.crypto_manager.compute_hash(
                    data if not compress else open(input_path, 'rb').read()
                )
            }
            
            # Write encrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                # Write header length and header
                header_bytes = json.dumps(header).encode('utf-8')
                f.write(len(header_bytes).to_bytes(4, 'big'))
                f.write(header_bytes)
                
                # Write encrypted data
                f.write(ciphertext)
            
            logger.info(
                "File encrypted successfully",
                input_path=str(input_path),
                output_path=str(output_path),
                compressed=compress,
                original_size=header["original_size"],
                encrypted_size=len(ciphertext)
            )
            
            return {
                "success": True,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "original_size": header["original_size"],
                "encrypted_size": len(ciphertext) + len(header_bytes) + 4,
                "compressed": compress,
                "algorithm": header["algorithm"]
            }
            
        except Exception as e:
            logger.error("File encryption failed", input_path=str(input_path), error=str(e))
            raise EncryptionError(f"File encryption failed: {e}")
    
    def decrypt_file(
        self,
        input_path: Path,
        output_path: Path,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Decrypt a file.
        
        Args:
            input_path: Path to encrypted file
            output_path: Path to output decrypted file
            password: Optional password for decryption
            
        Returns:
            Dictionary with decryption metadata
            
        Raises:
            DecryptionError: If decryption fails
        """
        try:
            # Read encrypted file
            with open(input_path, 'rb') as f:
                # Read header
                header_length = int.from_bytes(f.read(4), 'big')
                header_bytes = f.read(header_length)
                header = json.loads(header_bytes.decode('utf-8'))
                
                # Read encrypted data
                ciphertext = f.read()
            
            # Validate header
            if header.get("version") != 1:
                raise DecryptionError(f"Unsupported file version: {header.get('version')}")
            
            if header.get("algorithm") != "AES-256-GCM":
                raise DecryptionError(f"Unsupported algorithm: {header.get('algorithm')}")
            
            # Extract encryption parameters
            salt = base64.b64decode(header["salt"])
            iv = base64.b64decode(header["iv"])
            auth_tag = base64.b64decode(header["auth_tag"])
            
            # Derive decryption key
            if password:
                key = self.crypto_manager.derive_key_scrypt(password, salt)
            elif self.crypto_manager.master_key:
                key = self.crypto_manager.master_key
            else:
                raise DecryptionError("No password or master key provided")
            
            # Decrypt data using AES-GCM
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Decompress if needed
            if header.get("compressed", False):
                data = gzip.decompress(data)
            
            # Verify file integrity
            computed_hash = self.crypto_manager.compute_hash(data)
            expected_hash = header.get("file_hash")
            
            if expected_hash and computed_hash != expected_hash:
                raise DecryptionError("File integrity check failed")
            
            # Write decrypted file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'wb') as f:
                f.write(data)
            
            logger.info(
                "File decrypted successfully",
                input_path=str(input_path),
                output_path=str(output_path),
                original_size=len(data),
                algorithm=header["algorithm"]
            )
            
            return {
                "success": True,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "decrypted_size": len(data),
                "compressed": header.get("compressed", False),
                "algorithm": header["algorithm"]
            }
            
        except Exception as e:
            logger.error("File decryption failed", input_path=str(input_path), error=str(e))
            raise DecryptionError(f"File decryption failed: {e}")
    
    def encrypt_data(self, data: bytes, password: Optional[str] = None) -> Tuple[bytes, Dict[str, Any]]:
        """Encrypt data in memory.
        
        Args:
            data: Data to encrypt
            password: Optional password for encryption
            
        Returns:
            Tuple of (encrypted_data, metadata)
        """
        try:
            # Generate encryption parameters
            salt = self.crypto_manager.generate_salt()
            iv = secrets.token_bytes(16)
            
            # Derive encryption key
            if password:
                key = self.crypto_manager.derive_key_scrypt(password, salt)
            elif self.crypto_manager.master_key:
                key = self.crypto_manager.master_key
            else:
                raise EncryptionError("No password or master key provided")
            
            # Encrypt data
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv),
                backend=default_backend()
            )
            encryptor = cipher.encryptor()
            
            ciphertext = encryptor.update(data) + encryptor.finalize()
            auth_tag = encryptor.tag
            
            # Create metadata
            metadata = {
                "algorithm": "AES-256-GCM",
                "salt": base64.b64encode(salt).decode('ascii'),
                "iv": base64.b64encode(iv).decode('ascii'),
                "auth_tag": base64.b64encode(auth_tag).decode('ascii'),
                "original_size": len(data)
            }
            
            return ciphertext, metadata
            
        except Exception as e:
            raise EncryptionError(f"Data encryption failed: {e}")
    
    def decrypt_data(
        self,
        encrypted_data: bytes,
        metadata: Dict[str, Any],
        password: Optional[str] = None
    ) -> bytes:
        """Decrypt data in memory.
        
        Args:
            encrypted_data: Encrypted data
            metadata: Encryption metadata
            password: Optional password for decryption
            
        Returns:
            Decrypted data
        """
        try:
            # Extract encryption parameters
            salt = base64.b64decode(metadata["salt"])
            iv = base64.b64decode(metadata["iv"])
            auth_tag = base64.b64decode(metadata["auth_tag"])
            
            # Derive decryption key
            if password:
                key = self.crypto_manager.derive_key_scrypt(password, salt)
            elif self.crypto_manager.master_key:
                key = self.crypto_manager.master_key
            else:
                raise DecryptionError("No password or master key provided")
            
            # Decrypt data
            cipher = Cipher(
                algorithms.AES(key),
                modes.GCM(iv, auth_tag),
                backend=default_backend()
            )
            decryptor = cipher.decryptor()
            
            data = decryptor.update(encrypted_data) + decryptor.finalize()
            
            return data
            
        except Exception as e:
            raise DecryptionError(f"Data decryption failed: {e}")


class IndexEncryption:
    """Handles encryption of vector indexes and embeddings."""
    
    def __init__(self, file_encryption: FileEncryption):
        """Initialize index encryption.
        
        Args:
            file_encryption: File encryption instance
        """
        self.file_encryption = file_encryption
    
    def encrypt_vector_index(
        self,
        index_dir: Path,
        password: Optional[str] = None,
        encrypt_embeddings: bool = True,
        encrypt_metadata: bool = True
    ) -> Dict[str, Any]:
        """Encrypt a vector index directory.
        
        Args:
            index_dir: Directory containing vector index
            password: Optional password for encryption
            encrypt_embeddings: Whether to encrypt embedding files
            encrypt_metadata: Whether to encrypt metadata files
            
        Returns:
            Dictionary with encryption results
        """
        results = {
            "encrypted_files": [],
            "skipped_files": [],
            "errors": []
        }
        
        try:
            for file_path in index_dir.rglob('*'):
                if not file_path.is_file():
                    continue
                
                # Determine if file should be encrypted
                should_encrypt = False
                
                if encrypt_embeddings and file_path.suffix in ['.npy', '.pkl', '.bin']:
                    should_encrypt = True
                
                if encrypt_metadata and file_path.suffix in ['.json', '.yaml', '.yml']:
                    should_encrypt = True
                
                if not should_encrypt:
                    results["skipped_files"].append(str(file_path))
                    continue
                
                # Encrypt file
                try:
                    encrypted_path = file_path.with_suffix(file_path.suffix + '.enc')
                    
                    encryption_result = self.file_encryption.encrypt_file(
                        input_path=file_path,
                        output_path=encrypted_path,
                        password=password,
                        compress=True
                    )
                    
                    results["encrypted_files"].append({
                        "original_path": str(file_path),
                        "encrypted_path": str(encrypted_path),
                        "original_size": encryption_result["original_size"],
                        "encrypted_size": encryption_result["encrypted_size"]
                    })
                    
                    # Optionally remove original file
                    # file_path.unlink()
                    
                except Exception as e:
                    error_msg = f"Failed to encrypt {file_path}: {e}"
                    results["errors"].append(error_msg)
                    logger.error("Index file encryption failed", file_path=str(file_path), error=str(e))
            
            logger.info(
                "Vector index encryption completed",
                index_dir=str(index_dir),
                encrypted_count=len(results["encrypted_files"]),
                skipped_count=len(results["skipped_files"]),
                error_count=len(results["errors"])
            )
            
        except Exception as e:
            error_msg = f"Index encryption failed: {e}"
            results["errors"].append(error_msg)
            logger.error("Vector index encryption failed", index_dir=str(index_dir), error=str(e))
        
        return results
    
    def decrypt_vector_index(
        self,
        index_dir: Path,
        password: Optional[str] = None
    ) -> Dict[str, Any]:
        """Decrypt a vector index directory.
        
        Args:
            index_dir: Directory containing encrypted vector index
            password: Optional password for decryption
            
        Returns:
            Dictionary with decryption results
        """
        results = {
            "decrypted_files": [],
            "errors": []
        }
        
        try:
            for encrypted_file in index_dir.rglob('*.enc'):
                try:
                    # Determine original file path
                    original_path = encrypted_file.with_suffix('')
                    
                    decryption_result = self.file_encryption.decrypt_file(
                        input_path=encrypted_file,
                        output_path=original_path,
                        password=password
                    )
                    
                    results["decrypted_files"].append({
                        "encrypted_path": str(encrypted_file),
                        "decrypted_path": str(original_path),
                        "decrypted_size": decryption_result["decrypted_size"]
                    })
                    
                    # Optionally remove encrypted file
                    # encrypted_file.unlink()
                    
                except Exception as e:
                    error_msg = f"Failed to decrypt {encrypted_file}: {e}"
                    results["errors"].append(error_msg)
                    logger.error("Index file decryption failed", file_path=str(encrypted_file), error=str(e))
            
            logger.info(
                "Vector index decryption completed",
                index_dir=str(index_dir),
                decrypted_count=len(results["decrypted_files"]),
                error_count=len(results["errors"])
            )
            
        except Exception as e:
            error_msg = f"Index decryption failed: {e}"
            results["errors"].append(error_msg)
            logger.error("Vector index decryption failed", index_dir=str(index_dir), error=str(e))
        
        return results


# Global crypto manager instance
_crypto_manager: Optional[CryptoManager] = None
_file_encryption: Optional[FileEncryption] = None
_index_encryption: Optional[IndexEncryption] = None


def get_crypto_manager() -> CryptoManager:
    """Get the global crypto manager instance.
    
    Returns:
        Global crypto manager instance
    """
    global _crypto_manager
    if _crypto_manager is None:
        # Try to get master key from environment
        master_key_env = os.environ.get('MIMIR_MASTER_KEY')
        master_key = None
        
        if master_key_env:
            try:
                master_key = base64.b64decode(master_key_env)
            except Exception:
                logger.warning("Invalid master key in environment variable")
        
        _crypto_manager = CryptoManager(master_key)
    
    return _crypto_manager


def get_file_encryption() -> FileEncryption:
    """Get the global file encryption instance.
    
    Returns:
        Global file encryption instance
    """
    global _file_encryption
    if _file_encryption is None:
        _file_encryption = FileEncryption(get_crypto_manager())
    return _file_encryption


def get_index_encryption() -> IndexEncryption:
    """Get the global index encryption instance.
    
    Returns:
        Global index encryption instance
    """
    global _index_encryption
    if _index_encryption is None:
        _index_encryption = IndexEncryption(get_file_encryption())
    return _index_encryption


def configure_crypto(master_key: Optional[bytes] = None) -> None:
    """Configure the global crypto manager.
    
    Args:
        master_key: Master key for encryption
    """
    global _crypto_manager, _file_encryption, _index_encryption
    
    _crypto_manager = CryptoManager(master_key)
    _file_encryption = FileEncryption(_crypto_manager)
    _index_encryption = IndexEncryption(_file_encryption)


def generate_master_key() -> str:
    """Generate a new master key for encryption.
    
    Returns:
        Base64-encoded master key
    """
    key = secrets.token_bytes(32)
    return base64.b64encode(key).decode('ascii')