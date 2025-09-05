/**
 * File system utilities for atomic operations and content hashing
 * 
 * Provides safe file operations with content verification and atomic writes
 * to prevent corruption during pipeline execution.
 */

import { createHash } from 'crypto';
import { promises as fs, constants } from 'fs';
import { resolve, dirname, join } from 'path';
import { tmpdir } from 'os';
import { randomBytes } from 'crypto';

// =============================================================================
// Directory Management
// =============================================================================

export async function ensureDirectory(path: string): Promise<string> {
  const resolvedPath = resolve(path);
  try {
    await fs.mkdir(resolvedPath, { recursive: true });
    return resolvedPath;
  } catch (error: any) {
    if (error.code !== 'EEXIST') {
      throw new Error(`Failed to create directory ${resolvedPath}: ${error.message}`);
    }
    return resolvedPath;
  }
}

// Alias for compatibility
export const ensureDir = ensureDirectory;

export async function getIndexDirectory(baseDir: string, indexId: string): Promise<string> {
  const indexDir = join(baseDir, indexId);
  return ensureDirectory(indexDir);
}

// =============================================================================
// Hash Computation
// =============================================================================

export async function computeFileHash(
  filePath: string, 
  algorithm: string = 'sha256'
): Promise<string> {
  const hash = createHash(algorithm);
  const buffer = await fs.readFile(filePath);
  hash.update(buffer);
  return hash.digest('hex');
}

export function computeContentHash(
  content: string | Buffer, 
  algorithm: string = 'sha256'
): string {
  const hash = createHash(algorithm);
  
  if (typeof content === 'string') {
    content = Buffer.from(content, 'utf-8');
  }
  
  hash.update(content);
  return hash.digest('hex');
}

// =============================================================================
// Atomic File Operations
// =============================================================================

export async function atomicWriteText(
  filePath: string, 
  content: string, 
  encoding: BufferEncoding = 'utf-8'
): Promise<void> {
  const resolvedPath = resolve(filePath);
  const dir = dirname(resolvedPath);
  
  // Ensure parent directory exists
  await ensureDirectory(dir);
  
  // Generate temporary file name
  const tempName = `${resolvedPath}.tmp.${randomBytes(8).toString('hex')}`;
  
  try {
    // Write to temporary file
    await fs.writeFile(tempName, content, encoding);
    
    // Atomic move to final location
    await fs.rename(tempName, resolvedPath);
  } catch (error) {
    // Clean up temporary file if it exists
    try {
      await fs.unlink(tempName);
    } catch {
      // Ignore cleanup errors
    }
    throw error;
  }
}

export async function atomicWriteBytes(
  filePath: string, 
  content: Buffer
): Promise<void> {
  const resolvedPath = resolve(filePath);
  const dir = dirname(resolvedPath);
  
  // Ensure parent directory exists
  await ensureDirectory(dir);
  
  // Generate temporary file name
  const tempName = `${resolvedPath}.tmp.${randomBytes(8).toString('hex')}`;
  
  try {
    // Write to temporary file
    await fs.writeFile(tempName, content);
    
    // Atomic move to final location
    await fs.rename(tempName, resolvedPath);
  } catch (error) {
    // Clean up temporary file if it exists
    try {
      await fs.unlink(tempName);
    } catch {
      // Ignore cleanup errors
    }
    throw error;
  }
}

export async function atomicWriteJson(
  filePath: string, 
  data: any, 
  indent: number = 2
): Promise<void> {
  const content = JSON.stringify(data, null, indent);
  await atomicWriteText(filePath, content);
}

// =============================================================================
// File Reading with Validation
// =============================================================================

export async function readTextWithHash(
  filePath: string
): Promise<{ content: string; hash: string }> {
  const content = await fs.readFile(filePath, 'utf-8');
  const hash = computeContentHash(content);
  return { content, hash };
}

export async function readBytesWithHash(
  filePath: string
): Promise<{ content: Buffer; hash: string }> {
  const content = await fs.readFile(filePath);
  const hash = computeContentHash(content);
  return { content, hash };
}

// =============================================================================
// Safe File Operations
// =============================================================================

export async function safeCopy(src: string, dst: string): Promise<void> {
  const srcPath = resolve(src);
  const dstPath = resolve(dst);
  
  // Ensure destination directory exists
  await ensureDirectory(dirname(dstPath));
  
  // Copy file
  await fs.copyFile(srcPath, dstPath);
  
  // Verify copy integrity
  const srcHash = await computeFileHash(srcPath);
  const dstHash = await computeFileHash(dstPath);
  
  if (srcHash !== dstHash) {
    await fs.unlink(dstPath); // Remove corrupted copy
    throw new Error(`Copy verification failed: ${srcPath} -> ${dstPath}`);
  }
}

export async function safeMove(src: string, dst: string): Promise<void> {
  const srcPath = resolve(src);
  const dstPath = resolve(dst);
  
  // Ensure destination directory exists
  await ensureDirectory(dirname(dstPath));
  
  // Compute source hash before move
  const srcHash = await computeFileHash(srcPath);
  
  // Move file
  await fs.rename(srcPath, dstPath);
  
  // Verify move integrity
  const dstHash = await computeFileHash(dstPath);
  
  if (srcHash !== dstHash) {
    throw new Error(`Move verification failed: ${srcPath} -> ${dstPath}`);
  }
}

// =============================================================================
// Directory Operations
// =============================================================================

export async function cleanupDirectory(
  directory: string, 
  keepPatterns: string[] = []
): Promise<void> {
  const dirPath = resolve(directory);
  
  try {
    await fs.access(dirPath, constants.F_OK);
  } catch {
    return; // Directory doesn't exist
  }
  
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  
  for (const entry of entries) {
    const entryPath = join(dirPath, entry.name);
    
    // Check if entry matches any keep pattern
    const shouldKeep = keepPatterns.some(pattern => {
      const regex = new RegExp(pattern.replace(/\*/g, '.*'));
      return regex.test(entry.name);
    });
    
    if (!shouldKeep) {
      if (entry.isFile()) {
        await fs.unlink(entryPath);
      } else if (entry.isDirectory()) {
        await fs.rmdir(entryPath, { recursive: true });
      }
    }
  }
}

export async function getDirectorySize(directory: string): Promise<number> {
  const dirPath = resolve(directory);
  let totalSize = 0;
  
  async function calculateSize(path: string): Promise<void> {
    const entries = await fs.readdir(path, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = join(path, entry.name);
      
      if (entry.isFile()) {
        const stats = await fs.stat(fullPath);
        totalSize += stats.size;
      } else if (entry.isDirectory()) {
        await calculateSize(fullPath);
      }
    }
  }
  
  await calculateSize(dirPath);
  return totalSize;
}

// =============================================================================
// Temporary Directory Management
// =============================================================================

export async function createTempDirectory(
  prefix: string = 'mimir_', 
  parent?: string
): Promise<string> {
  const baseDir = parent || tmpdir();
  const tempName = `${prefix}${randomBytes(8).toString('hex')}`;
  const tempPath = join(baseDir, tempName);
  
  await ensureDirectory(tempPath);
  return tempPath;
}

export class TemporaryDirectory {
  private path: string | null = null;
  
  constructor(
    private prefix: string = 'mimir_',
    private parent?: string
  ) {}
  
  async create(): Promise<string> {
    if (!this.path) {
      this.path = await createTempDirectory(this.prefix, this.parent);
    }
    return this.path;
  }
  
  async cleanup(): Promise<void> {
    if (this.path) {
      try {
        await fs.rmdir(this.path, { recursive: true });
      } catch (error) {
        // Ignore cleanup errors
      }
      this.path = null;
    }
  }
  
  getPath(): string | null {
    return this.path;
  }
}

// =============================================================================
// File Content Extraction
// =============================================================================

export async function extractFileSpan(
  filePath: string,
  startByte: number,
  endByte: number,
  contextLines: number = 5
): Promise<{ pre: string; span: string; post: string }> {
  const content = await fs.readFile(filePath, 'utf-8');
  
  // Extract the span
  const span = content.slice(startByte, endByte);
  
  // Find line boundaries for context
  const lines = content.split('\n');
  let currentByte = 0;
  let startLine = 0;
  let endLine = 0;
  
  for (let i = 0; i < lines.length; i++) {
    const lineEnd = currentByte + lines[i].length + 1; // +1 for newline
    
    if (currentByte <= startByte && startByte < lineEnd) {
      startLine = i;
    }
    
    if (currentByte <= endByte && endByte <= lineEnd) {
      endLine = i;
      break;
    }
    
    currentByte = lineEnd;
  }
  
  // Extract context lines
  const preStart = Math.max(0, startLine - contextLines);
  const postEnd = Math.min(lines.length, endLine + contextLines + 1);
  
  const pre = lines.slice(preStart, startLine).join('\n');
  const post = lines.slice(endLine + 1, postEnd).join('\n');
  
  return { pre, span, post };
}

// =============================================================================
// File Validation
// =============================================================================

export async function validateFileIntegrity(
  filePath: string,
  expectedHash: string,
  algorithm: string = 'sha256'
): Promise<boolean> {
  try {
    const actualHash = await computeFileHash(filePath, algorithm);
    return actualHash === expectedHash;
  } catch {
    return false;
  }
}

export async function getFileMetadata(filePath: string): Promise<{
  path: string;
  size: number;
  modified: number;
  created: number;
  mode: string;
  isFile: boolean;
  isDir: boolean;
  exists: boolean;
  hash?: string;
}> {
  const resolvedPath = resolve(filePath);
  
  try {
    const stats = await fs.stat(resolvedPath);
    
    return {
      path: resolvedPath,
      size: stats.size,
      modified: stats.mtime.getTime(),
      created: stats.birthtime?.getTime() || stats.ctime.getTime(),
      mode: `0${(stats.mode & parseInt('777', 8)).toString(8)}`,
      isFile: stats.isFile(),
      isDir: stats.isDirectory(),
      exists: true,
      hash: stats.isFile() ? await computeFileHash(resolvedPath) : undefined,
    };
  } catch (error: any) {
    if (error.code === 'ENOENT') {
      return {
        path: resolvedPath,
        size: 0,
        modified: 0,
        created: 0,
        mode: '000',
        isFile: false,
        isDir: false,
        exists: false,
      };
    }
    throw error;
  }
}

// =============================================================================
// File System Utilities
// =============================================================================

export async function fileExists(filePath: string): Promise<boolean> {
  try {
    await fs.access(filePath, constants.F_OK);
    return true;
  } catch {
    return false;
  }
}

export async function isFile(filePath: string): Promise<boolean> {
  try {
    const stats = await fs.stat(filePath);
    return stats.isFile();
  } catch {
    return false;
  }
}

export async function isDirectory(filePath: string): Promise<boolean> {
  try {
    const stats = await fs.stat(filePath);
    return stats.isDirectory();
  } catch {
    return false;
  }
}

// =============================================================================
// Additional Path Utilities for Tests
// =============================================================================

export const pathExists = fileExists;
export const getFileSize = async (filePath: string): Promise<number> => {
  const stats = await fs.stat(filePath);
  return stats.size;
};

export const joinPaths = (...paths: string[]): string => {
  return join(...paths);
};

export const normalizePath = (path: string): string => {
  return resolve(path);
};

export const getBaseName = (path: string): string => {
  const parts = path.split('/');
  const name = parts[parts.length - 1];
  return name === '' && parts.length > 1 ? parts[parts.length - 2] : name;
};

export const getExtension = (path: string): string => {
  const name = getBaseName(path);
  const dotIndex = name.lastIndexOf('.');
  return dotIndex > 0 ? name.substring(dotIndex) : '';
};

export const isWithinPath = (filePath: string, basePath: string): boolean => {
  const resolvedFile = resolve(filePath);
  const resolvedBase = resolve(basePath);
  return resolvedFile.startsWith(resolvedBase);
};

export const sanitizePath = (path: string): string => {
  // Remove leading slashes and resolve relative path components
  const normalized = resolve(path.replace(/^\/+/, ''));
  // Remove the current working directory prefix to get relative path
  const cwd = process.cwd();
  if (normalized.startsWith(cwd)) {
    return normalized.slice(cwd.length + 1);
  }
  return normalized;
};
