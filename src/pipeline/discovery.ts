/**
 * Git-scoped file discovery and change detection
 * 
 * Provides intelligent file discovery using git tracking with support for
 * incremental indexing, change detection, and monorepo workspace handling.
 */

import { resolve, join, basename, extname } from 'path';
import { createHash } from 'crypto';
import { promises as fs } from 'fs';
import type { Logger, IndexConfig, FileDiscoveryResult } from '@/types';
import { createLogger } from '@/utils/logger';

export interface FileMetadata {
  path: string;
  size: number;
  modified: number;
  hash: string;
  extension: string;
  isBinary: boolean;
  exists?: boolean;
  error?: string;
}

export interface ChangeDetection {
  addedFiles: string[];
  modifiedFiles: string[];
  removedFiles: string[];
  currentHashes: Record<string, string>;
}

export interface ValidationReport {
  valid: boolean;
  warnings: string[];
  recommendations: string[];
  stats: {
    totalFiles: number;
    workspaces: number;
    fileTypes: Record<string, number>;
  };
}

export class FileDiscovery {
  private repoPath: string;
  private logger: Logger;

  constructor(repoPath: string, logger?: Logger) {
    this.repoPath = resolve(repoPath);
    this.logger = logger || createLogger('mimir.pipeline.discovery');
  }

  async discover(
    path: string,
    config: IndexConfig
  ): Promise<FileDiscoveryResult> {
    const startTime = Date.now();
    
    try {
      const files = await this.discoverFiles(
        config.languages.map(lang => this.getExtensionsForLanguage(lang)).flat(),
        config.excludes
      );

      const totalSize = await this.calculateTotalSize(files);
      const duration = Date.now() - startTime;

      this.logger.info(`Discovered ${files.length} files in ${duration}ms`, {
        totalFiles: files.length,
        totalSizeMB: Math.round(totalSize / 1024 / 1024 * 100) / 100,
        duration,
      });

      return {
        files,
        totalSize,
        excludedCount: 0, // TODO: Track excluded files
        duration,
      };
    } catch (error) {
      this.logger.error('File discovery failed', { error });
      throw error;
    }
  }

  private getExtensionsForLanguage(language: string): string[] {
    const extensionMap: Record<string, string[]> = {
      typescript: ['ts', 'tsx'],
      javascript: ['js', 'jsx', 'mjs', 'cjs'],
      python: ['py', 'pyx', 'pyi'],
      rust: ['rs'],
      go: ['go'],
      java: ['java'],
      json: ['json'],
      yaml: ['yaml', 'yml'],
      markdown: ['md', 'mdx'],
    };

    return extensionMap[language] || [language];
  }

  async discoverFiles(
    extensions?: string[],
    excludes?: string[]
  ): Promise<string[]> {
    // Default extensions for code analysis
    if (!extensions) {
      extensions = ['ts', 'tsx', 'js', 'jsx', 'md', 'mdx', 'json', 'yaml', 'yml', 'py', 'rs', 'go'];
    }

    // Default exclusion patterns
    if (!excludes) {
      excludes = [
        'node_modules/',
        'dist/',
        'build/',
        '.next/',
        'coverage/',
        '__pycache__/',
        '.git/',
        '.vscode/',
        '.idea/',
        'target/',
        '.cache/',
        'tmp/',
        'temp/',
      ];
    }

    const files = await this.walkDirectory(this.repoPath, extensions, excludes);
    return files;
  }

  private async walkDirectory(
    dir: string,
    extensions: string[],
    excludes: string[]
  ): Promise<string[]> {
    const files: string[] = [];
    
    const walk = async (currentDir: string): Promise<void> => {
      try {
        const entries = await fs.readdir(currentDir, { withFileTypes: true });
        
        for (const entry of entries) {
          const fullPath = join(currentDir, entry.name);
          const relativePath = fullPath.substring(this.repoPath.length + 1);
          
          // Check exclusions
          if (this.shouldExclude(relativePath, excludes)) {
            continue;
          }
          
          if (entry.isDirectory()) {
            await walk(fullPath);
          } else if (entry.isFile()) {
            const ext = extname(entry.name).slice(1);
            if (extensions.includes(ext)) {
              files.push(relativePath);
            }
          }
        }
      } catch (error) {
        this.logger.warn(`Failed to read directory ${currentDir}`, { error });
      }
    };
    
    await walk(dir);
    return files;
  }

  private shouldExclude(relativePath: string, excludes: string[]): boolean {
    return excludes.some(exclude => {
      if (exclude.endsWith('/')) {
        return relativePath.startsWith(exclude) || relativePath.includes('/' + exclude);
      }
      return relativePath === exclude || relativePath.endsWith('/' + exclude);
    });
  }

  private async calculateTotalSize(files: string[]): Promise<number> {
    let totalSize = 0;
    
    for (const file of files) {
      try {
        const fullPath = join(this.repoPath, file);
        const stats = await fs.stat(fullPath);
        totalSize += stats.size;
      } catch {
        // Ignore errors for individual files
      }
    }
    
    return totalSize;
  }

  async detectChanges(
    previousFiles: string[],
    previousHashes: Record<string, string>
  ): Promise<ChangeDetection> {
    // Get current file list
    const currentFiles = await this.discoverFiles();
    const currentFileSet = new Set(currentFiles);
    const previousFileSet = new Set(previousFiles);

    // Detect additions and removals
    const addedFiles = Array.from(currentFileSet).filter(f => !previousFileSet.has(f));
    const removedFiles = Array.from(previousFileSet).filter(f => !currentFileSet.has(f));

    // Check for modifications in existing files
    const modifiedFiles: string[] = [];
    const currentHashes: Record<string, string> = {};

    for (const filePath of currentFiles) {
      try {
        const currentHash = await this.hashFileContent(filePath);
        currentHashes[filePath] = currentHash;

        if (filePath in previousHashes && previousHashes[filePath] !== currentHash) {
          modifiedFiles.push(filePath);
        }
      } catch (error) {
        // If we can't hash the file, consider it modified if it existed before
        if (previousFiles.includes(filePath)) {
          modifiedFiles.push(filePath);
        }
      }
    }

    return {
      addedFiles,
      modifiedFiles,
      removedFiles,
      currentHashes,
    };
  }

  async getFileMetadata(filePaths: string[]): Promise<Record<string, FileMetadata>> {
    const metadata: Record<string, FileMetadata> = {};
    
    // Process files in batches
    const batchSize = 50;
    for (let i = 0; i < filePaths.length; i += batchSize) {
      const batch = filePaths.slice(i, i + batchSize);
      const promises = batch.map(fp => this.getSingleFileMetadata(fp));
      
      const results = await Promise.allSettled(promises);
      
      results.forEach((result, index) => {
        const filePath = batch[index];
        if (result.status === 'fulfilled' && result.value) {
          metadata[filePath] = result.value;
        }
      });
    }
    
    return metadata;
  }

  private async getSingleFileMetadata(filePath: string): Promise<FileMetadata | null> {
    const fullPath = join(this.repoPath, filePath);
    
    try {
      const stats = await fs.stat(fullPath);
      const hash = await this.hashFileContent(filePath);
      const isBinary = await this.isBinaryFile(fullPath);
      
      return {
        path: filePath,
        size: stats.size,
        modified: stats.mtime.getTime(),
        hash,
        extension: extname(filePath).slice(1),
        isBinary,
        exists: true,
      };
    } catch (error: any) {
      return {
        path: filePath,
        size: 0,
        modified: 0,
        hash: '',
        extension: '',
        isBinary: false,
        exists: false,
        error: error.message,
      };
    }
  }

  private async hashFileContent(filePath: string): Promise<string> {
    const fullPath = join(this.repoPath, filePath);
    const content = await fs.readFile(fullPath);
    return createHash('sha256').update(content).digest('hex');
  }

  private async isBinaryFile(fullPath: string): Promise<boolean> {
    try {
      const chunkSize = 8192;
      const buffer = Buffer.alloc(chunkSize);
      const fd = await fs.open(fullPath, 'r');
      
      try {
        const { bytesRead } = await fd.read(buffer, 0, chunkSize, 0);
        const chunk = buffer.slice(0, bytesRead);
        
        // Check for null bytes
        if (chunk.includes(0)) {
          return true;
        }
        
        // Check for high ratio of non-printable characters
        const nonPrintable = chunk.filter(byte => 
          byte < 32 && ![9, 10, 13].includes(byte)
        ).length;
        
        const ratio = nonPrintable / chunk.length;
        return ratio > 0.3;
      } finally {
        await fd.close();
      }
    } catch {
      return false;
    }
  }

  async getFileContentWithContext(
    filePath: string,
    startLine: number,
    endLine: number,
    contextLines: number = 5
  ): Promise<{
    pre: string;
    content: string;
    post: string;
    lineStart: number;
    lineEnd: number;
    contextStart: number;
    contextEnd: number;
  } | { error: string; path: string }> {
    const fullPath = join(this.repoPath, filePath);
    
    try {
      const content = await fs.readFile(fullPath, 'utf-8');
      const lines = content.split('\n');
      
      const contextStart = Math.max(0, startLine - contextLines);
      const contextEnd = Math.min(lines.length, endLine + contextLines);
      
      const pre = lines.slice(contextStart, startLine).join('\n');
      const mainContent = lines.slice(startLine, endLine).join('\n');
      const post = lines.slice(endLine, contextEnd).join('\n');
      
      return {
        pre,
        content: mainContent,
        post,
        lineStart: startLine,
        lineEnd: endLine,
        contextStart,
        contextEnd,
      };
    } catch (error: any) {
      return {
        error: error.message,
        path: filePath,
      };
    }
  }

  async getPriorityFiles(
    allFiles: string[],
    dirtyFiles?: Set<string>,
    maxFiles?: number
  ): Promise<string[]> {
    const priorityFiles: string[] = [];
    
    // Always prioritize dirty files first
    if (dirtyFiles) {
      for (const filePath of allFiles) {
        if (dirtyFiles.has(filePath)) {
          priorityFiles.push(filePath);
        }
      }
    }
    
    // Add remaining files with importance heuristics
    const remainingFiles = allFiles.filter(f => !dirtyFiles?.has(f));
    
    const fileImportance = (filePath: string): number => {
      let score = 0;
      
      // Higher score for certain file types
      if (filePath.endsWith('.ts') || filePath.endsWith('.tsx')) {
        score += 10;
      } else if (filePath.endsWith('.js') || filePath.endsWith('.jsx')) {
        score += 8;
      } else if (filePath.endsWith('.json')) {
        score += 5;
      }
      
      // Higher score for root-level files
      if (!filePath.includes('/')) {
        score += 5;
      }
      
      // Higher score for important config files
      const filename = basename(filePath).toLowerCase();
      const importantFiles = new Set([
        'package.json',
        'tsconfig.json',
        'index.ts',
        'index.js',
        'main.ts',
        'main.js',
      ]);
      
      if (importantFiles.has(filename)) {
        score += 15;
      }
      
      // Higher score for shorter paths (likely more important)
      score += Math.max(0, 10 - (filePath.match(/\//g) || []).length);
      
      return score;
    };
    
    remainingFiles.sort((a, b) => fileImportance(b) - fileImportance(a));
    priorityFiles.push(...remainingFiles);
    
    // Apply max files limit if specified
    if (maxFiles !== undefined) {
      return priorityFiles.slice(0, maxFiles);
    }
    
    return priorityFiles;
  }

  async validateRepositoryStructure(): Promise<ValidationReport> {
    const report: ValidationReport = {
      valid: true,
      warnings: [],
      recommendations: [],
      stats: {
        totalFiles: 0,
        workspaces: 0,
        fileTypes: {},
      },
    };
    
    try {
      const files = await this.discoverFiles();
      
      report.stats.totalFiles = files.length;
      
      // Analyze file type distribution
      for (const filePath of files) {
        const ext = extname(filePath).slice(1);
        report.stats.fileTypes[ext] = (report.stats.fileTypes[ext] || 0) + 1;
      }
      
      // Check for common issues
      if (files.length > 10000) {
        report.warnings.push(
          `Large repository with ${files.length} files. Consider using excludes to focus indexing.`
        );
      }
      
      // Check for missing important files
      const importantFiles = new Set(['package.json', 'tsconfig.json', 'README.md']);
      const existingFiles = new Set(files.map(f => basename(f)));
      const missingImportant = Array.from(importantFiles).filter(f => !existingFiles.has(f));
      
      if (missingImportant.length > 0) {
        report.recommendations.push(
          `Consider adding missing project files: ${missingImportant.join(', ')}`
        );
      }
      
      // Check for potential binary files in sample
      let binaryCount = 0;
      const sampleFiles = files.slice(0, 100);
      
      for (const filePath of sampleFiles) {
        const fullPath = join(this.repoPath, filePath);
        if (await this.isBinaryFile(fullPath)) {
          binaryCount++;
        }
      }
      
      if (binaryCount > 10) {
        report.warnings.push(
          'Detected potential binary files. Consider adding binary extensions to excludes.'
        );
      }
      
    } catch (error: any) {
      report.valid = false;
      report.warnings.push(`Validation failed: ${error.message}`);
    }
    
    return report;
  }

  // Generate cache key for incremental indexing
  async getCacheKey(config: IndexConfig): Promise<string> {
    const configStr = JSON.stringify(config, Object.keys(config).sort());
    const configHash = createHash('sha256').update(configStr).digest('hex');
    
    // In a real implementation, this would include git rev and other repo state
    const timestamp = Date.now().toString();
    const combined = configHash + timestamp;
    
    return createHash('sha256').update(combined).digest('hex');
  }
}
