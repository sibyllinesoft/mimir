# Security Migration Guide: Hardcoded Salt Fix

## 🚨 Critical Security Update

A critical security vulnerability has been identified and fixed in the Mimir cryptographic implementation. **Hardcoded cryptographic salts** were being used in the `SecretsManager` class, which significantly weakened the security of encrypted secrets.

## Vulnerability Details

**Issue**: The `SecretsManager` class used a hardcoded salt (`b"mimir_salt_v1"`) for key derivation, which:
- Made all encrypted secrets files use the same salt
- Enabled rainbow table attacks
- Violated cryptographic best practices
- Exposed the salt value in source code

**Severity**: **CRITICAL**
**CVSS Score**: 8.1 (High)

## Security Fix

The vulnerability has been completely resolved with the following improvements:

### ✅ What's Fixed

1. **Random Salt Generation**: Each secrets file now gets a unique, cryptographically random salt
2. **Secure Storage Format**: Salt is stored securely in file header, not hardcoded
3. **Backward Compatibility**: Legacy files are automatically detected and migrated
4. **Enhanced Security**: Key rotation now also rotates salts for maximum security

### 🔧 Technical Changes

- **New File Format**: Includes magic header and random salt storage
- **Automatic Migration**: Legacy files are migrated on first save operation
- **Enhanced Security Functions**: Added integrity verification and security auditing
- **Comprehensive Testing**: Full security test suite validates all changes

## Migration Process

### Automatic Migration (Recommended)

The system automatically handles migration when you use existing encrypted files:

```python
from src.repoindex.security.secrets import SecretsManager

# Loading an old file automatically detects legacy format
manager = SecretsManager(secrets_file="old_secrets.enc", password="your_password")

# First save operation will automatically migrate to secure format
manager.set_secret("new_key", "new_value")  # Triggers migration

# Verify migration completed
file_info = manager.get_file_info()
print(f"Security level: {file_info['security_level']}")  # Should show "high"
```

### Manual Migration

For explicit control over the migration process:

```python
from src.repoindex.security.secrets import SecretsManager

manager = SecretsManager(secrets_file="legacy_file.enc", password="your_password")

# Check if migration is needed
file_info = manager.get_file_info()
if file_info.get('migration_needed', False):
    print("Legacy file detected - performing migration...")
    
    # Trigger explicit migration
    migrated = manager.migrate_legacy_file()
    if migrated:
        print("✅ Migration completed successfully")
    else:
        print("ℹ️  File already in secure format")

# Verify new security status
file_info = manager.get_file_info()
print(f"Format: {file_info['format_version']}")
print(f"Security: {file_info['security_level']}")
```

### Batch Migration Script

For multiple secrets files:

```python
#!/usr/bin/env python3
"""Migrate all legacy secrets files in a directory."""

from pathlib import Path
from src.repoindex.security.secrets import SecretsManager

def migrate_directory(secrets_dir: Path, password: str):
    """Migrate all .enc files in directory."""
    migrated_count = 0
    error_count = 0
    
    for secrets_file in secrets_dir.glob("*.enc"):
        try:
            print(f"Processing {secrets_file.name}...")
            manager = SecretsManager(secrets_file=secrets_file, password=password)
            
            if manager.migrate_legacy_file():
                print(f"✅ Migrated {secrets_file.name}")
                migrated_count += 1
            else:
                print(f"ℹ️  {secrets_file.name} already secure")
                
        except Exception as e:
            print(f"❌ Error migrating {secrets_file.name}: {e}")
            error_count += 1
    
    print(f"Migration complete: {migrated_count} migrated, {error_count} errors")

# Usage
if __name__ == "__main__":
    migrate_directory(Path("./secrets"), "your_master_password")
```

## Security Validation

### 1. Run Security Audit

Verify your installation is secure:

```bash
python scripts/security_audit.py
```

Expected output should show:
```
✅ HARDCODED SALT VULNERABILITY SUCCESSFULLY FIXED
✅ SecretsManager: Uses cryptographically secure salt generation
✅ SecretsManager: Hardcoded salt vulnerability FIXED
```

### 2. Test File Security

Verify your secrets files are using the secure format:

```python
from src.repoindex.security.secrets import SecretsManager

manager = SecretsManager(secrets_file="your_secrets.enc", password="your_password")
file_info = manager.get_file_info()

print("Security Status:")
print(f"  Format Version: {file_info['format_version']}")  # Should be "v2"
print(f"  Salt Type: {file_info['salt_type']}")           # Should be "random"
print(f"  Security Level: {file_info['security_level']}")  # Should be "high"
print(f"  Migration Needed: {file_info['migration_needed']}")  # Should be False
```

### 3. Verify Salt Uniqueness

Confirm that new files get unique salts:

```python
from src.repoindex.security.secrets import SecretsManager
import tempfile
from pathlib import Path

# Create multiple managers and verify unique salts
salts = []
for i in range(5):
    with tempfile.NamedTemporaryFile(suffix='.enc', delete=False) as tmp:
        manager = SecretsManager(secrets_file=Path(tmp.name), password="test")
        salts.append(manager.salt)

# All salts should be unique
assert len(set(salts)) == len(salts), "Salts are not unique!"
print("✅ Salt generation verified - all unique")
```

## Production Deployment

### Pre-Deployment Checklist

- [ ] Run security audit: `python scripts/security_audit.py`
- [ ] Test migration process on staging environment
- [ ] Backup existing secrets files
- [ ] Update deployment scripts to handle migration
- [ ] Verify environment variables are secure
- [ ] Update documentation for operations team

### Deployment Steps

1. **Backup Current Files**:
   ```bash
   # Create backup of all secrets files
   find /path/to/secrets -name "*.enc" -exec cp {} {}.backup \;
   ```

2. **Deploy Updated Code**:
   ```bash
   # Deploy new version with security fixes
   git checkout security/fix-hardcoded-salt
   # Deploy using your standard process
   ```

3. **Verify Migration**:
   ```bash
   # Run security audit
   python scripts/security_audit.py
   
   # Check specific files if needed
   python -c "from src.repoindex.security.secrets import SecretsManager; 
              m = SecretsManager(secrets_file='production.enc', password='$PASSWORD');
              print(m.get_file_info())"
   ```

4. **Monitor for Issues**:
   - Check application logs for migration warnings
   - Verify all secrets are accessible
   - Monitor performance impact (should be minimal)

## Security Best Practices

### Enhanced Security Measures

1. **Key Rotation**: Regularly rotate encryption keys and passwords:
   ```python
   manager.rotate_encryption_key("new_strong_password")
   ```

2. **Environment Variables**: Use environment variables for passwords:
   ```bash
   export MIMIR_SECRETS_PASSWORD="your_secure_password"
   ```

3. **File Permissions**: Restrict access to secrets files:
   ```bash
   chmod 600 *.enc  # Owner read/write only
   ```

4. **Regular Audits**: Run security audits regularly:
   ```bash
   # Add to cron or CI/CD pipeline
   python scripts/security_audit.py
   ```

### Monitoring and Alerting

Set up monitoring for:
- Failed decryption attempts
- Legacy file access attempts
- Unusual secrets access patterns
- Security audit failures

## FAQ

### Q: Will this break my existing encrypted files?
**A**: No! The fix maintains full backward compatibility. Legacy files are automatically detected and can be read normally. They will be migrated to the secure format on the next save operation.

### Q: Do I need to change my passwords?
**A**: Not required, but recommended as a security best practice. Use `rotate_encryption_key()` to change passwords and generate new salts.

### Q: How can I verify the migration worked?
**A**: Use `manager.get_file_info()` to check the security status, or run the security audit script.

### Q: What if migration fails?
**A**: The original file is preserved. Check the error message and ensure you have the correct password. Contact support if issues persist.

### Q: Is there any performance impact?
**A**: Minimal. The new format adds a small header to files but uses the same encryption. Migration happens only once per file.

### Q: Can I rollback if needed?
**A**: Yes, but you'll lose the security improvements. Keep backups of original files if rollback capability is critical.

## Support

If you encounter issues during migration:

1. Check the logs for detailed error messages
2. Run the security audit script for diagnostic information
3. Verify file permissions and password correctness
4. Create an issue with detailed error information

## Changelog

### v2.0.0 - Security Fix
- **SECURITY**: Fixed critical hardcoded salt vulnerability
- **NEW**: Random salt generation for each secrets file
- **NEW**: Secure file format with salt storage
- **NEW**: Automatic legacy file migration
- **NEW**: Enhanced security validation and auditing
- **NEW**: Key rotation with salt regeneration
- **IMPROVED**: Comprehensive security testing

---

**Important**: This is a critical security update. Please prioritize deployment and migration of existing systems.