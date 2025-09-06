/**
 * Unit tests for filesystem utilities.
 * 
 * Tests atomic write operations, file hashing, path validation,
 * and directory management functions.
 */

import { describe, expect, it, beforeEach, afterEach } from 'bun:test';
import { 
  atomicWriteText, 
  atomicWriteBytes,
  atomicWriteJson,
  computeFileHash,
  computeContentHash,
  readTextWithHash,
  readBytesWithHash,
  safeCopy,
  safeMove,
  cleanupDirectory,
  getDirectorySize,
  createTempDirectory,
  extractFileSpan,
  validateFileIntegrity,
  getFileMetadata,
  getIndexDirectory,
  ensureDirectory,
  ensureDir,
  pathExists,
  fileExists,
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

  describe('Content Hash Functions', () => {
    it('should compute content hash from string', () => {
      const content = 'Test content for hashing';
      const hash = computeContentHash(content);
      const expectedHash = createHash('sha256').update(content, 'utf8').digest('hex');
      
      expect(hash).toBe(expectedHash);
    });

    it('should compute content hash from buffer', () => {
      const content = Buffer.from([0x01, 0x02, 0x03, 0x04]);
      const hash = computeContentHash(content);
      const expectedHash = createHash('sha256').update(content).digest('hex');
      
      expect(hash).toBe(expectedHash);
    });

    it('should use custom algorithm for content hash', () => {
      const content = 'Test content';
      const hash = computeContentHash(content, 'md5');
      const expectedHash = createHash('md5').update(content, 'utf8').digest('hex');
      
      expect(hash).toBe(expectedHash);
    });
  });

  describe('Additional Atomic Write Operations', () => {
    it('should write bytes atomically', async () => {
      const filePath = join(tempDir, 'bytes-test.bin');
      const content = Buffer.from([0x48, 0x65, 0x6c, 0x6c, 0x6f]); // "Hello"

      await atomicWriteBytes(filePath, content);

      expect(await pathExists(filePath)).toBe(true);
      const written = readFileSync(filePath);
      expect(written).toEqual(content);
    });

    it('should write JSON atomically', async () => {
      const filePath = join(tempDir, 'data.json');
      const data = { name: 'test', value: 42, nested: { flag: true } };

      await atomicWriteJson(filePath, data);

      expect(await pathExists(filePath)).toBe(true);
      const written = JSON.parse(readFileSync(filePath, 'utf-8'));
      expect(written).toEqual(data);
    });

    it('should write JSON with custom spacing', async () => {
      const filePath = join(tempDir, 'pretty.json');
      const data = { key: 'value' };

      await atomicWriteJson(filePath, data, 2);

      expect(await pathExists(filePath)).toBe(true);
      const content = readFileSync(filePath, 'utf-8');
      expect(content).toContain('  "key": "value"'); // Check for pretty printing
    });
  });

  describe('Read with Hash Operations', () => {
    it('should read text with hash', async () => {
      const filePath = join(tempDir, 'read-with-hash.txt');
      const content = 'Content to read with hash verification';
      writeFileSync(filePath, content);

      const result = await readTextWithHash(filePath);

      expect(result.content).toBe(content);
      expect(result.hash).toBe(computeContentHash(content));
    });

    it('should read text with hash', async () => {
      const filePath = join(tempDir, 'read-text-hash.txt');
      const content = 'Content with hash for reading';
      writeFileSync(filePath, content);

      const result = await readTextWithHash(filePath);

      expect(result.content).toBe(content);
      expect(result.hash).toBe(computeContentHash(content));
    });

    it('should read bytes with hash', async () => {
      const filePath = join(tempDir, 'read-bytes.bin');
      const content = Buffer.from([0x10, 0x20, 0x30, 0x40]);
      writeFileSync(filePath, content);

      const result = await readBytesWithHash(filePath);

      expect(result.content).toEqual(content);
      expect(result.hash).toBe(computeContentHash(content));
    });
  });

  describe('File Copy and Move Operations', () => {
    it('should copy file safely', async () => {
      const srcPath = join(tempDir, 'source.txt');
      const destPath = join(tempDir, 'dest.txt');
      const content = 'Content to copy';
      writeFileSync(srcPath, content);

      await safeCopy(srcPath, destPath);

      expect(await pathExists(destPath)).toBe(true);
      expect(readFileSync(destPath, 'utf-8')).toBe(content);
      expect(await pathExists(srcPath)).toBe(true); // Source should still exist
    });

    it('should move file safely', async () => {
      const srcPath = join(tempDir, 'source-move.txt');
      const destPath = join(tempDir, 'dest-move.txt');
      const content = 'Content to move';
      writeFileSync(srcPath, content);

      await safeMove(srcPath, destPath);

      expect(await pathExists(destPath)).toBe(true);
      expect(readFileSync(destPath, 'utf-8')).toBe(content);
      expect(await pathExists(srcPath)).toBe(false); // Source should be removed
    });

    it('should create destination directory for copy', async () => {
      const srcPath = join(tempDir, 'source-nested.txt');
      const destPath = join(tempDir, 'nested', 'directory', 'dest.txt');
      const content = 'Content for nested copy';
      writeFileSync(srcPath, content);

      await safeCopy(srcPath, destPath);

      expect(await pathExists(destPath)).toBe(true);
      expect(readFileSync(destPath, 'utf-8')).toBe(content);
    });

    it('should create destination directory for move', async () => {
      const srcPath = join(tempDir, 'source-nested-move.txt');
      const destPath = join(tempDir, 'nested', 'move', 'dest.txt');
      const content = 'Content for nested move';
      writeFileSync(srcPath, content);

      await safeMove(srcPath, destPath);

      expect(await pathExists(destPath)).toBe(true);
      expect(readFileSync(destPath, 'utf-8')).toBe(content);
      expect(await pathExists(srcPath)).toBe(false);
    });
  });

  describe('Directory Cleanup Operations', () => {
    it('should cleanup directory removing all files by default', async () => {
      const cleanupDir = join(tempDir, 'cleanup-test');
      mkdirSync(cleanupDir, { recursive: true });
      
      const file1 = join(cleanupDir, 'file1.txt');
      const file2 = join(cleanupDir, 'file2.txt');
      writeFileSync(file1, 'content 1');
      writeFileSync(file2, 'content 2');

      await cleanupDirectory(cleanupDir);

      expect(await pathExists(file1)).toBe(false);
      expect(await pathExists(file2)).toBe(false);
    });

    it('should cleanup directory with pattern exclusions', async () => {
      const cleanupDir = join(tempDir, 'cleanup-pattern');
      mkdirSync(cleanupDir, { recursive: true });
      
      const logFile = join(cleanupDir, 'keep.log');
      const txtFile = join(cleanupDir, 'delete.txt');
      writeFileSync(logFile, 'keep this log file');
      writeFileSync(txtFile, 'delete this text file');

      await cleanupDirectory(cleanupDir, ['*.log']);

      expect(await pathExists(logFile)).toBe(true); // Should be kept
      expect(await pathExists(txtFile)).toBe(false); // Should be deleted
    });

    it('should cleanup directory with multiple patterns', async () => {
      const cleanupDir = join(tempDir, 'cleanup-multi');
      mkdirSync(cleanupDir, { recursive: true });
      
      const logFile = join(cleanupDir, 'app.log');
      const configFile = join(cleanupDir, 'config.json');
      const tempFile = join(cleanupDir, 'temp.tmp');
      writeFileSync(logFile, 'log content');
      writeFileSync(configFile, 'config content');
      writeFileSync(tempFile, 'temp content');

      await cleanupDirectory(cleanupDir, ['*.log', '*.json']);

      expect(await pathExists(logFile)).toBe(true); // Should be kept
      expect(await pathExists(configFile)).toBe(true); // Should be kept
      expect(await pathExists(tempFile)).toBe(false); // Should be deleted
    });

    it('should calculate directory size', async () => {
      const sizeDir = join(tempDir, 'size-test');
      mkdirSync(sizeDir, { recursive: true });
      
      const file1 = join(sizeDir, 'file1.txt');
      const file2 = join(sizeDir, 'file2.txt');
      writeFileSync(file1, 'x'.repeat(100));
      writeFileSync(file2, 'y'.repeat(200));

      const size = await getDirectorySize(sizeDir);

      expect(size).toBe(300); // 100 + 200 bytes
    });
  });

  describe('Temporary Directory Operations', () => {
    it('should create temporary directory', async () => {
      const tempResult = await createTempDirectory();

      expect(await pathExists(tempResult)).toBe(true);
      expect(await isDirectory(tempResult)).toBe(true);
      
      // Cleanup
      rmSync(tempResult, { recursive: true });
    });

    it('should create temporary directory with prefix', async () => {
      const prefix = 'test-prefix';
      const tempResult = await createTempDirectory(prefix);

      expect(await pathExists(tempResult)).toBe(true);
      expect(tempResult).toContain(prefix);
      
      // Cleanup
      rmSync(tempResult, { recursive: true });
    });
  });

  describe('File Content Extraction', () => {
    it('should extract file span with byte ranges', async () => {
      const filePath = join(tempDir, 'extract-test.txt');
      const content = 'line 1\nline 2\nline 3\nline 4\nline 5';
      writeFileSync(filePath, content);

      // Extract bytes 7-13 ('line 2') with 1 line context
      const result = await extractFileSpan(filePath, 7, 13, 1);

      expect(result.span).toBe('line 2');
      expect(result.pre).toContain('line 1');
      expect(result.post).toContain('line 3');
    });

    it('should extract file span without context', async () => {
      const filePath = join(tempDir, 'extract-no-context.txt');
      const content = 'hello world test';
      writeFileSync(filePath, content);

      // Extract bytes 6-11 ('world') with no context
      const result = await extractFileSpan(filePath, 6, 11, 0);

      expect(result.span).toBe('world');
      expect(result.pre).toBe('');
      expect(result.post).toBe('');
    });

    it('should handle file span at beginning', async () => {
      const filePath = join(tempDir, 'extract-beginning.txt');
      const content = 'start\nmiddle\nend';
      writeFileSync(filePath, content);

      // Extract bytes 0-5 ('start') with context
      const result = await extractFileSpan(filePath, 0, 5, 1);

      expect(result.span).toBe('start');
      expect(result.pre).toBe('');
      expect(result.post).toContain('middle');
    });

    it('should handle file span at end', async () => {
      const filePath = join(tempDir, 'extract-end.txt');
      const content = 'start\nmiddle\nend';
      writeFileSync(filePath, content);

      // Extract 'end' (last 3 bytes) with context  
      const result = await extractFileSpan(filePath, content.length - 3, content.length, 1);

      expect(result.span).toBe('end');
      expect(result.pre).toContain('middle');
      expect(result.post).toBe('');
    });
  });

  describe('File Integrity and Metadata', () => {
    it('should validate file integrity', async () => {
      const filePath = join(tempDir, 'integrity.txt');
      const content = 'Content for integrity check';
      writeFileSync(filePath, content);
      const hash = computeContentHash(content);

      const isValid = await validateFileIntegrity(filePath, hash);

      expect(isValid).toBe(true);
    });

    it('should detect corrupted file', async () => {
      const filePath = join(tempDir, 'corrupted.txt');
      writeFileSync(filePath, 'Original content');
      const originalHash = computeContentHash('Original content');
      
      // Corrupt the file
      writeFileSync(filePath, 'Corrupted content');

      const isValid = await validateFileIntegrity(filePath, originalHash);

      expect(isValid).toBe(false);
    });

    it('should get file metadata', async () => {
      const filePath = join(tempDir, 'metadata.txt');
      const content = 'Metadata test content';
      writeFileSync(filePath, content);

      const metadata = await getFileMetadata(filePath);

      expect(metadata.path).toBe(resolve(filePath));
      expect(metadata.size).toBe(content.length);
      expect(metadata.hash).toBe(computeContentHash(content));
      expect(typeof metadata.created).toBe('number');
      expect(typeof metadata.modified).toBe('number');
      expect(metadata.created).toBeGreaterThan(0);
      expect(metadata.modified).toBeGreaterThan(0);
      expect(metadata.isFile).toBe(true);
      expect(metadata.isDir).toBe(false);
      expect(metadata.exists).toBe(true);
    });
  });

  describe('Directory Management Extensions', () => {
    it('should create index directory', async () => {
      const baseDir = tempDir;
      const indexId = 'test-index-123';
      
      const result = await getIndexDirectory(baseDir, indexId);
      
      expect(result).toBe(resolve(join(baseDir, indexId)));
      expect(await pathExists(result)).toBe(true);
      expect(await isDirectory(result)).toBe(true);
    });

    it('should use ensureDirectory directly', async () => {
      const testDir = join(tempDir, 'ensure-direct');
      
      const result = await ensureDirectory(testDir);
      
      expect(result).toBe(resolve(testDir));
      expect(await pathExists(testDir)).toBe(true);
    });
  });

  describe('File Existence Checks', () => {
    it('should check file existence with fileExists', async () => {
      const existingFile = join(tempDir, 'exists-check.txt');
      const nonExistentFile = join(tempDir, 'not-exists.txt');
      writeFileSync(existingFile, 'exists');

      expect(await fileExists(existingFile)).toBe(true);
      expect(await fileExists(nonExistentFile)).toBe(false);
    });

    it('should distinguish pathExists vs fileExists', async () => {
      const filePath = join(tempDir, 'file-vs-path.txt');
      const dirPath = join(tempDir, 'dir-vs-path');
      writeFileSync(filePath, 'content');
      mkdirSync(dirPath);

      // Both should work for pathExists
      expect(await pathExists(filePath)).toBe(true);
      expect(await pathExists(dirPath)).toBe(true);
      
      // fileExists should work the same way
      expect(await fileExists(filePath)).toBe(true);
      expect(await fileExists(dirPath)).toBe(true);
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