/**
 * Unit tests for filesystem utilities.
 * 
 * Tests atomic write operations, file hashing, path validation,
 * and directory management functions.
 */

import { describe, expect, it, beforeEach, afterEach } from 'bun:test';
import { 
  atomicWriteText, 
  computeFileHash,
  ensureDir,
  pathExists,
  isFile,
  isDirectory,
  getFileSize,
  joinPaths,
  normalizePath,
  getBaseName,
  getExtension,
  isWithinPath,
  sanitizePath
} from '@/utils/filesystem';
import { writeFileSync, unlinkSync, mkdirSync, rmSync, readFileSync } from 'fs';
import { tmpdir } from 'os';
import { join, resolve } from 'path';
import { createHash } from 'crypto';

describe('Filesystem Utilities', () => {
  let tempDir: string;

  beforeEach(() => {
    // Create temp directory for tests
    tempDir = join(tmpdir(), `mimir-fs-test-${Date.now()}`);
    mkdirSync(tempDir, { recursive: true });
  });

  afterEach(() => {
    // Clean up temp files
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  describe('Atomic Write Operations', () => {
    it('should write file atomically', async () => {
      const filePath = join(tempDir, 'test.txt');
      const content = 'Hello, World!';

      await atomicWriteText(filePath, content);

      expect(await pathExists(filePath)).toBe(true);
      const written = readFileSync(filePath, 'utf-8');
      expect(written).toBe(content);
    });

    it('should overwrite existing file atomically', async () => {
      const filePath = join(tempDir, 'overwrite.txt');
      const originalContent = 'Original content';
      const newContent = 'New content';

      // Write original
      await atomicWriteText(filePath, originalContent);
      expect(readFileSync(filePath, 'utf-8')).toBe(originalContent);

      // Overwrite
      await atomicWriteText(filePath, newContent);
      expect(readFileSync(filePath, 'utf-8')).toBe(newContent);
    });

    it('should handle nested directory creation', async () => {
      const nestedPath = join(tempDir, 'deep', 'nested', 'path', 'file.txt');
      const content = 'Deep content';

      await atomicWriteText(nestedPath, content);

      expect(await pathExists(nestedPath)).toBe(true);
      expect(readFileSync(nestedPath, 'utf-8')).toBe(content);
    });

    it('should handle empty content', async () => {
      const filePath = join(tempDir, 'empty.txt');
      
      await atomicWriteText(filePath, '');

      expect(await pathExists(filePath)).toBe(true);
      expect(readFileSync(filePath, 'utf-8')).toBe('');
    });

    it('should handle Unicode content', async () => {
      const filePath = join(tempDir, 'unicode.txt');
      const content = '🚀 Hello, 世界! 🌟';

      await atomicWriteText(filePath, content);

      expect(readFileSync(filePath, 'utf-8')).toBe(content);
    });
  });

  describe('File Hashing', () => {
    it('should compute correct SHA-256 hash', async () => {
      const filePath = join(tempDir, 'hash-test.txt');
      const content = 'Test content for hashing';
      
      writeFileSync(filePath, content);

      const hash = await computeFileHash(filePath);
      const expectedHash = createHash('sha256').update(content).digest('hex');
      
      expect(hash).toBe(expectedHash);
    });

    it('should handle empty files', async () => {
      const filePath = join(tempDir, 'empty-hash.txt');
      writeFileSync(filePath, '');

      const hash = await computeFileHash(filePath);
      const expectedHash = createHash('sha256').update('').digest('hex');
      
      expect(hash).toBe(expectedHash);
    });

    it('should handle binary files', async () => {
      const filePath = join(tempDir, 'binary-test.bin');
      const binaryData = Buffer.from([0x00, 0x01, 0x02, 0xFF, 0xFE]);
      
      writeFileSync(filePath, binaryData);

      const hash = await computeFileHash(filePath);
      const expectedHash = createHash('sha256').update(binaryData).digest('hex');
      
      expect(hash).toBe(expectedHash);
    });

    it('should throw error for non-existent file', async () => {
      const nonExistentPath = join(tempDir, 'does-not-exist.txt');
      
      await expect(computeFileHash(nonExistentPath)).rejects.toThrow();
    });

    it('should produce different hashes for different content', async () => {
      const file1 = join(tempDir, 'file1.txt');
      const file2 = join(tempDir, 'file2.txt');
      
      writeFileSync(file1, 'Content 1');
      writeFileSync(file2, 'Content 2');

      const hash1 = await computeFileHash(file1);
      const hash2 = await computeFileHash(file2);
      
      expect(hash1).not.toBe(hash2);
    });

    it('should produce same hash for identical content', async () => {
      const file1 = join(tempDir, 'identical1.txt');
      const file2 = join(tempDir, 'identical2.txt');
      const content = 'Identical content';
      
      writeFileSync(file1, content);
      writeFileSync(file2, content);

      const hash1 = await computeFileHash(file1);
      const hash2 = await computeFileHash(file2);
      
      expect(hash1).toBe(hash2);
    });
  });

  describe('Directory Operations', () => {
    it('should create directory if it does not exist', async () => {
      const dirPath = join(tempDir, 'new-dir');
      
      await ensureDir(dirPath);
      
      expect(await pathExists(dirPath)).toBe(true);
      expect(await isDirectory(dirPath)).toBe(true);
    });

    it('should not fail if directory already exists', async () => {
      const dirPath = join(tempDir, 'existing-dir');
      mkdirSync(dirPath);
      
      // Should not throw
      await ensureDir(dirPath);
      
      expect(await isDirectory(dirPath)).toBe(true);
    });

    it('should create nested directories', async () => {
      const nestedPath = join(tempDir, 'level1', 'level2', 'level3');
      
      await ensureDir(nestedPath);
      
      expect(await pathExists(nestedPath)).toBe(true);
      expect(await isDirectory(nestedPath)).toBe(true);
    });
  });

  describe('Path Validation', () => {
    it('should detect existing paths', async () => {
      const filePath = join(tempDir, 'exists.txt');
      writeFileSync(filePath, 'content');

      expect(await pathExists(filePath)).toBe(true);
      expect(await pathExists(tempDir)).toBe(true);
      expect(await pathExists(join(tempDir, 'does-not-exist'))).toBe(false);
    });

    it('should distinguish files from directories', async () => {
      const filePath = join(tempDir, 'is-file.txt');
      const dirPath = join(tempDir, 'is-dir');
      
      writeFileSync(filePath, 'content');
      mkdirSync(dirPath);

      expect(await isFile(filePath)).toBe(true);
      expect(await isFile(dirPath)).toBe(false);
      
      expect(await isDirectory(dirPath)).toBe(true);
      expect(await isDirectory(filePath)).toBe(false);
    });

    it('should get correct file size', async () => {
      const filePath = join(tempDir, 'size-test.txt');
      const content = 'Test content'; // 12 bytes
      
      writeFileSync(filePath, content);

      const size = await getFileSize(filePath);
      expect(size).toBe(Buffer.byteLength(content, 'utf8'));
    });
  });

  describe('Path Manipulation', () => {
    it('should join paths correctly', () => {
      expect(joinPaths('/home', 'user', 'documents')).toBe('/home/user/documents');
      expect(joinPaths('relative', 'path')).toBe('relative/path');
      expect(joinPaths('/absolute', '../relative')).toBe('/relative');
    });

    it('should normalize paths', () => {
      expect(normalizePath('/home/user/../user/documents')).toBe('/home/user/documents');
      expect(normalizePath('relative/../other/./path')).toBe('other/path');
      expect(normalizePath('/path//double//slash')).toBe('/path/double/slash');
    });

    it('should get base name correctly', () => {
      expect(getBaseName('/path/to/file.txt')).toBe('file.txt');
      expect(getBaseName('/path/to/directory/')).toBe('directory');
      expect(getBaseName('simple-file.txt')).toBe('simple-file.txt');
      expect(getBaseName('/')).toBe('');
    });

    it('should get file extension', () => {
      expect(getExtension('file.txt')).toBe('.txt');
      expect(getExtension('document.pdf')).toBe('.pdf');
      expect(getExtension('script.min.js')).toBe('.js');
      expect(getExtension('no-extension')).toBe('');
      expect(getExtension('.hidden')).toBe('');
      expect(getExtension('.hidden.txt')).toBe('.txt');
    });
  });

  describe('Path Security', () => {
    it('should detect path traversal attempts', () => {
      const basePath = '/safe/base/path';
      
      expect(isWithinPath('/safe/base/path/file.txt', basePath)).toBe(true);
      expect(isWithinPath('/safe/base/path/subdir/file.txt', basePath)).toBe(true);
      expect(isWithinPath('/safe/base/path/../outside.txt', basePath)).toBe(false);
      expect(isWithinPath('/completely/different/path', basePath)).toBe(false);
    });

    it('should sanitize dangerous paths', () => {
      expect(sanitizePath('../../../etc/passwd')).toBe('etc/passwd');
      expect(sanitizePath('/absolute/path')).toBe('absolute/path');
      expect(sanitizePath('safe/relative/path')).toBe('safe/relative/path');
      expect(sanitizePath('path/with/../traversal')).toBe('path/traversal');
      expect(sanitizePath('path//double//slash')).toBe('path/double/slash');
    });

    it('should handle malicious path components', () => {
      expect(sanitizePath('path/with/../../traversal')).toBe('traversal');
      expect(sanitizePath('../../../root')).toBe('root');
      expect(sanitizePath('normal/../../path')).toBe('path');
    });
  });

  describe('Error Handling', () => {
    it('should handle permission errors gracefully', async () => {
      // This test may be platform specific
      if (process.platform !== 'win32') {
        const restrictedPath = '/root/cannot-write.txt';
        
        // Should throw appropriate error, not crash
        await expect(atomicWriteText(restrictedPath, 'content')).rejects.toThrow();
      }
    });

    it('should handle invalid paths', async () => {
      // Null byte in path (should be rejected by OS)
      const invalidPath = join(tempDir, 'invalid\0path.txt');
      
      await expect(atomicWriteText(invalidPath, 'content')).rejects.toThrow();
    });

    it('should handle extremely long paths', async () => {
      // Create a very long path
      const longName = 'a'.repeat(255);
      const longPath = join(tempDir, longName);
      
      // Should handle gracefully (may succeed or fail depending on OS limits)
      try {
        await atomicWriteText(longPath, 'content');
        expect(await pathExists(longPath)).toBe(true);
      } catch (error) {
        // Expected to fail on some systems
        expect(error).toBeInstanceOf(Error);
      }
    });
  });
});