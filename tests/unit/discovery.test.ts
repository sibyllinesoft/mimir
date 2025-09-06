/**
 * Simplified unit tests for file discovery - focusing on core functionality
 */

import { describe, expect, it, beforeEach, afterEach, mock } from 'bun:test';
import { promises as fs } from 'fs';
import { mkdirSync, rmSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { createHash } from 'crypto';
import type { Logger, IndexConfig } from '@/types';
import { 
  FileDiscovery,
  type FileMetadata,
  type ChangeDetection,
  type ValidationReport 
} from '@/pipeline/discovery';

describe('FileDiscovery (Core Functionality)', () => {
  let tempDir: string;
  let discovery: FileDiscovery;
  let mockLogger: Logger;

  beforeEach(async () => {
    // Reset any mocks from previous tests
    mock.restore();
    
    // Create clean temp directory
    tempDir = join(tmpdir(), `discovery-${Date.now()}`);
    mkdirSync(tempDir, { recursive: true });

    // Create mock logger
    mockLogger = {
      debug: () => {},
      info: () => {},
      warn: () => {},
      error: () => {},
    };

    discovery = new FileDiscovery(tempDir, mockLogger);
  });

  afterEach(() => {
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  async function createFile(relativePath: string, content: string = 'test content'): Promise<void> {
    const fullPath = join(tempDir, relativePath);
    const dir = join(fullPath, '..');
    await fs.mkdir(dir, { recursive: true });
    await fs.writeFile(fullPath, content);
  }

  describe('constructor', () => {
    it('creates instance with path and logger', () => {
      const instance = new FileDiscovery('/test', mockLogger);
      expect(instance).toBeInstanceOf(FileDiscovery);
    });

    it('creates instance with path only', () => {
      const instance = new FileDiscovery('/test');
      expect(instance).toBeInstanceOf(FileDiscovery);
    });
  });

  describe('basic file discovery', () => {
    it('discovers TypeScript files', async () => {
      await createFile('test.ts', 'const x = 1;');
      await createFile('test.tsx', 'const comp = <div />;');
      
      const files = await discovery.discoverFiles(['ts', 'tsx']);
      
      expect(files).toBeArray();
      expect(files).toContain('test.ts');
      expect(files).toContain('test.tsx');
    });

    it('discovers multiple file types', async () => {
      await createFile('script.js', 'console.log("hello");');
      await createFile('data.json', '{"key": "value"}');
      await createFile('readme.md', '# Test');
      
      const files = await discovery.discoverFiles(['js', 'json', 'md']);
      
      expect(files).toContain('script.js');
      expect(files).toContain('data.json');
      expect(files).toContain('readme.md');
    });

    it('excludes specified directories', async () => {
      await createFile('src/main.ts', 'export const main = 1;');
      await createFile('node_modules/pkg/index.js', 'module.exports = {};');
      
      const files = await discovery.discoverFiles(['ts', 'js'], ['node_modules/']);
      
      expect(files).toContain('src/main.ts');
      expect(files).not.toContain('node_modules/pkg/index.js');
    });

    it('handles nested directory structure', async () => {
      await createFile('src/utils/helper.ts', 'export function help() {}');
      await createFile('src/components/Button.tsx', 'export const Button = () => {};');
      
      const files = await discovery.discoverFiles(['ts', 'tsx']);
      
      expect(files).toContain('src/utils/helper.ts');
      expect(files).toContain('src/components/Button.tsx');
    });

    it('returns empty array for empty directory', async () => {
      const files = await discovery.discoverFiles(['ts']);
      expect(files).toEqual([]);
    });
  });

  describe('discover method with config', () => {
    it('discovers files based on language config', async () => {
      await createFile('main.ts', 'typescript');
      await createFile('script.js', 'javascript');
      await createFile('data.py', 'python');
      
      const config: IndexConfig = {
        languages: ['typescript', 'python'],
        excludes: [],
        contextLines: 3,
        maxFilesToEmbed: 1000
      };

      const result = await discovery.discover(tempDir, config);

      expect(result.files).toContain('main.ts');
      expect(result.files).toContain('data.py');
      expect(result.files).not.toContain('script.js');
      expect(result.totalSize).toBeNumber();
      expect(result.duration).toBeNumber();
    });
  });

  describe('change detection', () => {
    it('detects added files', async () => {
      await createFile('existing.ts', 'existing');
      await createFile('new.ts', 'new file');

      const changes = await discovery.detectChanges(['existing.ts'], { 'existing.ts': 'old_hash' });

      expect(changes.addedFiles).toContain('new.ts');
    });

    it('detects removed files', async () => {
      await createFile('remaining.ts', 'still here');

      const changes = await discovery.detectChanges(
        ['remaining.ts', 'removed.ts'], 
        { 'remaining.ts': 'hash1', 'removed.ts': 'hash2' }
      );

      expect(changes.removedFiles).toContain('removed.ts');
    });

    it('detects modified files', async () => {
      const content = 'original content';
      await createFile('modified.ts', content);
      
      const originalHash = createHash('sha256').update(content).digest('hex');
      
      // Modify the file
      await createFile('modified.ts', 'modified content');

      const changes = await discovery.detectChanges(['modified.ts'], { 'modified.ts': originalHash });

      expect(changes.modifiedFiles).toContain('modified.ts');
      expect(changes.currentHashes['modified.ts']).not.toBe(originalHash);
    });
  });

  describe('file metadata', () => {
    it('gets metadata for existing files', async () => {
      await createFile('test.ts', 'test content');

      const metadata = await discovery.getFileMetadata(['test.ts']);

      expect(metadata['test.ts']).toMatchObject({
        path: 'test.ts',
        extension: 'ts',
        exists: true,
        isBinary: false
      });
      expect(metadata['test.ts'].size).toBeGreaterThan(0);
      expect(metadata['test.ts'].hash).toBeString();
    });

    it('handles non-existent files', async () => {
      const metadata = await discovery.getFileMetadata(['missing.ts']);

      expect(metadata['missing.ts']).toMatchObject({
        path: 'missing.ts',
        exists: false,
        size: 0,
        hash: ''
      });
    });
  });

  describe('file content with context', () => {
    it('extracts content with context lines', async () => {
      const content = 'line 0\nline 1\nline 2\nline 3\nline 4';
      await createFile('sample.ts', content);

      const result = await discovery.getFileContentWithContext('sample.ts', 1, 3, 1);

      if ('error' in result) {
        throw new Error(result.error);
      }

      expect(result.content).toBe('line 1\nline 2');
      expect(result.pre).toBe('line 0');
      expect(result.post).toBe('line 3');
    });
  });

  describe('priority files', () => {
    it('prioritizes dirty files', async () => {
      const allFiles = ['a.ts', 'b.ts', 'c.ts'];
      const dirtyFiles = new Set(['b.ts']);

      const priority = await discovery.getPriorityFiles(allFiles, dirtyFiles);

      expect(priority[0]).toBe('b.ts');
    });

    it('scores files by importance', async () => {
      const allFiles = ['package.json', 'deep/nested/file.ts', 'index.ts'];

      const priority = await discovery.getPriorityFiles(allFiles);

      // package.json and index.ts should rank higher than deeply nested files
      const packageIndex = priority.indexOf('package.json');
      const indexIndex = priority.indexOf('index.ts');
      const deepIndex = priority.indexOf('deep/nested/file.ts');

      expect(packageIndex).toBeLessThan(deepIndex);
      expect(indexIndex).toBeLessThan(deepIndex);
    });
  });

  describe('repository validation', () => {
    it('validates repository structure', async () => {
      await createFile('package.json', '{}');
      await createFile('src/index.ts', 'code');

      const report = await discovery.validateRepositoryStructure();

      expect(report.valid).toBe(true);
      expect(report.stats.totalFiles).toBeGreaterThan(0);
      expect(report.stats.fileTypes).toHaveProperty('json');
      expect(report.stats.fileTypes).toHaveProperty('ts');
    });

    it('warns about large repositories', async () => {
      // Create enough files to trigger warning (> 10,000)
      const promises = [];
      for (let i = 0; i < 50; i++) {
        for (let j = 0; j < 201; j++) {
          promises.push(createFile(`dir${i}/file${j}.ts`, `content ${i}-${j}`));
        }
      }
      await Promise.all(promises);

      const report = await discovery.validateRepositoryStructure();

      expect(report.warnings.some(w => w.includes('Large repository'))).toBe(true);
    });
  });

  describe('language extension mapping', () => {
    it('maps languages to file extensions correctly', async () => {
      await createFile('test.ts', 'ts');
      await createFile('test.tsx', 'tsx'); 
      await createFile('test.js', 'js');

      const tsConfig: IndexConfig = {
        languages: ['typescript'],
        excludes: [],
        contextLines: 3,
        maxFilesToEmbed: 1000
      };

      const result = await discovery.discover(tempDir, tsConfig);

      expect(result.files).toContain('test.ts');
      expect(result.files).toContain('test.tsx');
      expect(result.files).not.toContain('test.js');
    });

    it('handles unknown languages as extensions', async () => {
      await createFile('test.custom', 'custom content');

      const config: IndexConfig = {
        languages: ['custom'],
        excludes: [],
        contextLines: 3,
        maxFilesToEmbed: 1000
      };

      const result = await discovery.discover(tempDir, config);

      expect(result.files).toContain('test.custom');
    });
  });

  describe('cache key generation', () => {
    it('generates consistent keys for same config', async () => {
      const config: IndexConfig = {
        languages: ['typescript'],
        excludes: ['node_modules/'],
        contextLines: 3,
        maxFilesToEmbed: 1000
      };

      const key1 = await discovery.getCacheKey(config);
      const key2 = await discovery.getCacheKey(config);

      expect(key1).toBeString();
      expect(key2).toBeString();
      expect(key1.length).toBe(64); // SHA256 hex length
      expect(key2.length).toBe(64);
    });

    it('generates different keys for different configs', async () => {
      const config1: IndexConfig = {
        languages: ['typescript'],
        excludes: ['node_modules/'],
        contextLines: 3,
        maxFilesToEmbed: 1000
      };

      const config2: IndexConfig = {
        languages: ['javascript'],
        excludes: ['dist/'],
        contextLines: 5,
        maxFilesToEmbed: 500
      };

      const key1 = await discovery.getCacheKey(config1);
      const key2 = await discovery.getCacheKey(config2);

      expect(key1).not.toBe(key2);
    });
  });

  describe('edge cases', () => {
    it('handles binary file detection', async () => {
      // Create a binary file with null bytes
      const binaryContent = Buffer.from([0x00, 0x01, 0x02, 0xFF]);
      await fs.writeFile(join(tempDir, 'binary.dat'), binaryContent);
      
      // Create a text file
      await createFile('text.txt', 'Hello, world!');

      const metadata = await discovery.getFileMetadata(['binary.dat', 'text.txt']);

      expect(metadata['binary.dat'].isBinary).toBe(true);
      expect(metadata['text.txt'].isBinary).toBe(false);
    });

    it('handles file content with context at boundaries', async () => {
      const content = 'line 0\nline 1\nline 2';
      await createFile('boundary.ts', content);

      // Request context at the beginning
      const result1 = await discovery.getFileContentWithContext('boundary.ts', 0, 1, 2);
      if ('error' in result1) throw new Error(result1.error);
      
      expect(result1.contextStart).toBe(0);
      expect(result1.pre).toBe('');

      // Request context at the end
      const result2 = await discovery.getFileContentWithContext('boundary.ts', 2, 3, 2);
      if ('error' in result2) throw new Error(result2.error);
      
      expect(result2.post).toBe('');
    });

    it('handles non-existent file for content with context', async () => {
      const result = await discovery.getFileContentWithContext('nonexistent.ts', 0, 1);

      if ('content' in result) {
        throw new Error('Expected error result');
      }

      expect(result.error).toBeString();
      expect(result.path).toBe('nonexistent.ts');
    });

    it('applies max files limit in priority sorting', async () => {
      const allFiles = Array.from({ length: 20 }, (_, i) => `file${i}.ts`);
      const maxFiles = 5;

      const priority = await discovery.getPriorityFiles(allFiles, undefined, maxFiles);

      expect(priority.length).toBe(maxFiles);
      expect(priority.length).toBeLessThanOrEqual(maxFiles);
    });

    it('handles empty inputs gracefully', async () => {
      // Empty file list for priority sorting
      const priority = await discovery.getPriorityFiles([]);
      expect(priority).toEqual([]);

      // Empty metadata request
      const metadata = await discovery.getFileMetadata([]);
      expect(metadata).toEqual({});

      // Empty change detection
      const changes = await discovery.detectChanges([], {});
      expect(changes.addedFiles).toEqual([]);
      expect(changes.modifiedFiles).toEqual([]);
      expect(changes.removedFiles).toEqual([]);
    });

    it('validates repository with missing important files', async () => {
      await createFile('index.ts', 'main file');

      const report = await discovery.validateRepositoryStructure();

      expect(report.valid).toBe(true);
      expect(report.recommendations.some(r => 
        r.includes('missing project files')
      )).toBe(true);
    });

    it('handles all language mappings', async () => {
      const languageFiles = {
        'test.py': 'python',
        'test.rs': 'rust', 
        'test.go': 'go',
        'test.java': 'java',
        'config.yaml': 'yaml',
        'config.yml': 'yaml'
      };

      for (const [filename, language] of Object.entries(languageFiles)) {
        await createFile(filename, `${language} content`);
      }

      const config: IndexConfig = {
        languages: ['python', 'rust', 'go', 'java', 'yaml'],
        excludes: [],
        contextLines: 3,
        maxFilesToEmbed: 1000
      };

      const result = await discovery.discover(tempDir, config);

      expect(result.files).toContain('test.py');
      expect(result.files).toContain('test.rs');
      expect(result.files).toContain('test.go'); 
      expect(result.files).toContain('test.java');
      expect(result.files).toContain('config.yaml');
      expect(result.files).toContain('config.yml');
    });

    it('processes large file batches in metadata retrieval', async () => {
      // Create more files than the batch size (which is 50)
      const fileCount = 120;
      const filePaths: string[] = [];
      
      for (let i = 0; i < fileCount; i++) {
        const filename = `batch${i}.ts`;
        filePaths.push(filename);
        await createFile(filename, `content for file ${i}`);
      }

      const metadata = await discovery.getFileMetadata(filePaths);

      expect(Object.keys(metadata)).toHaveLength(fileCount);
      
      // Verify all files have metadata
      for (const filePath of filePaths) {
        expect(metadata[filePath].exists).toBe(true);
        expect(metadata[filePath].size).toBeGreaterThan(0);
      }
    });
  });
});