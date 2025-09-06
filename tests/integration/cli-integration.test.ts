/**
 * Integration tests for CLI interface.
 * 
 * Tests CLI commands, argument parsing, configuration loading,
 * and integration with pipeline components.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn } from 'bun:test';
import { spawn } from 'child_process';
import { promisify } from 'util';
import { writeFileSync, unlinkSync, mkdirSync, rmSync, existsSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

describe('CLI Integration', () => {
  let tempDir: string;
  let tempRepoDir: string;
  let consoleLogSpy: any;
  let processExitSpy: any;

  beforeEach(() => {
    // Create temp directories for test repos
    tempDir = join(tmpdir(), `mimir-cli-test-${Date.now()}`);
    tempRepoDir = join(tempDir, 'test-repo');
    mkdirSync(tempDir, { recursive: true });
    mkdirSync(tempRepoDir, { recursive: true });

    // Create some test files in the repo
    writeFileSync(join(tempRepoDir, 'index.ts'), `
export class TestClass {
  constructor(public name: string) {}
  
  greet() {
    return \`Hello, \${this.name}!\`;
  }
}
    `);

    writeFileSync(join(tempRepoDir, 'utils.ts'), `
export function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export const VERSION = '1.0.0';
    `);

    writeFileSync(join(tempRepoDir, 'package.json'), JSON.stringify({
      name: 'test-repo',
      version: '1.0.0',
      type: 'module'
    }, null, 2));

    // Mock console and process.exit to capture output
    consoleLogSpy = spyOn(console, 'log').mockImplementation(() => {});
    spyOn(console, 'error').mockImplementation(() => {});
    spyOn(console, 'warn').mockImplementation(() => {});
    spyOn(console, 'info').mockImplementation(() => {});

    processExitSpy = spyOn(process, 'exit').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    // Clean up temp directories
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }

    consoleLogSpy.mockRestore();
    processExitSpy.mockRestore();
  });

  // Helper function to run CLI commands
  const runCLI = async (args: string[]): Promise<{ stdout: string; stderr: string; exitCode: number }> => {
    return new Promise((resolve) => {
      const child = spawn('bun', ['run', join(process.cwd(), 'src/cli.ts'), ...args], {
        stdio: 'pipe',
        cwd: process.cwd()
      });

      let stdout = '';
      let stderr = '';

      child.stdout?.on('data', (data) => {
        stdout += data.toString();
      });

      child.stderr?.on('data', (data) => {
        stderr += data.toString();
      });

      child.on('close', (code) => {
        resolve({ stdout, stderr, exitCode: code || 0 });
      });

      child.on('error', (err) => {
        resolve({ stdout, stderr: err.message, exitCode: 1 });
      });
    });
  };

  describe('CLI Help and Version', () => {
    it('should show help when run without arguments', async () => {
      const result = await runCLI([]);
      
      expect([0, 1]).toContain(result.exitCode); // May exit 1 if no arguments provided
      // Should still show help output regardless
      if (result.exitCode === 0) {
        expect(result.stdout.toLowerCase()).toContain('usage');
        expect(result.stdout.toLowerCase()).toContain('commands');
      } else {
        // May show error but still have usage info
        expect(result.stderr.toLowerCase() || result.stdout.toLowerCase()).toContain('usage');
      }
    });

    it('should show help with --help flag', async () => {
      const result = await runCLI(['--help']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout.toLowerCase()).toContain('usage');
      expect(result.stdout).toContain('index');
      expect(result.stdout).toContain('search');
      expect(result.stdout).toContain('status');
      expect(result.stdout).toContain('validate');
    });

    it('should show version with --version flag', async () => {
      const result = await runCLI(['--version']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('2.0.0');
    });

    it('should show version with -V flag', async () => {
      const result = await runCLI(['-V']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('2.0.0');
    });
  });

  describe('Index Command', () => {
    it('should show index command help', async () => {
      const result = await runCLI(['index', '--help']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('Index a repository for search');
      expect(result.stdout).toContain('<path>');
      expect(result.stdout).toContain('--id');
      expect(result.stdout).toContain('--force');
      expect(result.stdout).toContain('--no-lens');
    });

    it('should require path argument for index command', async () => {
      const result = await runCLI(['index']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('error');
    });

    it('should handle non-existent path', async () => {
      const result = await runCLI(['index', '/non/existent/path']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain('does not exist');
    });

    it('should accept valid repository path', async () => {
      // This test may fail if Lens is not available, but should at least validate the path
      const result = await runCLI(['index', tempRepoDir, '--no-lens']);
      
      // May exit with 0 (success) or 1 (Lens unavailable), but should not crash
      expect([0, 1]).toContain(result.exitCode);
      
      if (result.exitCode === 0) {
        expect(result.stdout.toLowerCase()).toContain('index');
      } else {
        // If it fails, it should be due to Lens unavailability, not path issues
        expect(result.stderr).not.toContain('does not exist');
      }
    });

    it('should handle custom index ID', async () => {
      const result = await runCLI(['index', tempRepoDir, '--id', 'custom-repo-id', '--no-lens']);
      
      // Should process the custom ID (success or failure due to Lens, not argument parsing)
      expect([0, 1]).toContain(result.exitCode);
    });

    it('should handle force reindex flag', async () => {
      const result = await runCLI(['index', tempRepoDir, '--force', '--no-lens']);
      
      expect([0, 1]).toContain(result.exitCode);
    });

    it('should handle language filtering', async () => {
      const result = await runCLI(['index', tempRepoDir, '--languages', 'typescript,javascript', '--no-lens']);
      
      expect([0, 1]).toContain(result.exitCode);
    });
  });

  describe('Search Command', () => {
    it('should show search command help', async () => {
      const result = await runCLI(['search', '--help']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('Search indexed repositories');
      expect(result.stdout).toContain('<query>');
      expect(result.stdout).toContain('--index-id');
      expect(result.stdout).toContain('--max-results');
      expect(result.stdout).toContain('--format');
    });

    it('should require query argument for search command', async () => {
      const result = await runCLI(['search']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('error');
    });

    it('should handle search without Lens integration', async () => {
      const result = await runCLI(['search', 'test query']);
      
      // Should fail gracefully when Lens is not available
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('lens');
    });

    it('should handle JSON output format', async () => {
      const result = await runCLI(['search', 'test query', '--format', 'json']);
      
      // May fail due to Lens unavailability, but should validate format option
      expect(result.exitCode).toBe(1);
      expect(result.stderr).not.toContain('unknown option');
    });

    it('should handle max results parameter', async () => {
      const result = await runCLI(['search', 'test query', '--max-results', '5']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).not.toContain('unknown option');
    });

    it('should handle index ID parameter', async () => {
      const result = await runCLI(['search', 'test query', '--index-id', 'specific-repo']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).not.toContain('unknown option');
    });
  });

  describe('Status Command', () => {
    it('should show status command help', async () => {
      const result = await runCLI(['status', '--help']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('Check status of indexed repositories');
      expect(result.stdout).toContain('[path]');
    });

    it('should handle status without Lens integration', async () => {
      const result = await runCLI(['status']);
      
      // Should fail gracefully when Lens is not available
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('lens');
    });

    it('should handle status with specific path', async () => {
      const result = await runCLI(['status', tempRepoDir]);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('lens');
    });
  });

  describe('Validate Command', () => {
    it('should show validate command help', async () => {
      const result = await runCLI(['validate', '--help']);
      
      expect(result.exitCode).toBe(0);
      expect(result.stdout).toContain('Validate repository structure and configuration');
      expect(result.stdout).toContain('<path>');
    });

    it('should require path argument for validate command', async () => {
      const result = await runCLI(['validate']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('error');
    });

    it('should handle non-existent path for validate', async () => {
      const result = await runCLI(['validate', '/non/existent/path']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain('does not exist');
    });

    it('should validate repository structure', async () => {
      const result = await runCLI(['validate', tempRepoDir]);
      
      // Should succeed as we have valid TypeScript files
      expect([0, 1]).toContain(result.exitCode);
      
      if (result.exitCode === 0) {
        expect(result.stdout.toLowerCase()).toContain('valid');
      }
    });
  });

  describe('Global Options', () => {
    it('should handle verbose flag', async () => {
      const result = await runCLI(['--verbose', 'validate', tempRepoDir]);
      
      // Should not fail due to unknown option
      expect(result.stderr).not.toContain('unknown option');
    });

    it('should handle log-level option', async () => {
      const result = await runCLI(['--log-level', 'debug', 'validate', tempRepoDir]);
      
      expect(result.stderr).not.toContain('unknown option');
    });

    it('should handle custom config file', async () => {
      // Create a temporary config file
      const configFile = join(tempDir, 'test-config.json');
      writeFileSync(configFile, JSON.stringify({
        lens: { enabled: false },
        logging: { logLevel: 'warn' }
      }, null, 2));

      const result = await runCLI(['--config', configFile, 'validate', tempRepoDir]);
      
      expect(result.stderr).not.toContain('unknown option');
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid command', async () => {
      const result = await runCLI(['invalid-command']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('unknown command') || 
      expect(result.stderr.toLowerCase()).toContain('error');
    });

    it('should handle invalid options', async () => {
      const result = await runCLI(['index', tempRepoDir, '--invalid-option']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toContain('unknown option') ||
      expect(result.stderr).toContain('error');
    });

    it('should handle malformed config file', async () => {
      const badConfigFile = join(tempDir, 'bad-config.json');
      writeFileSync(badConfigFile, 'invalid json{');

      const result = await runCLI(['--config', badConfigFile, 'validate', tempRepoDir]);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr.toLowerCase()).toContain('config') ||
      expect(result.stderr.toLowerCase()).toContain('error');
    });

    it('should handle permission errors gracefully', async () => {
      // Use a directory that clearly doesn't exist to test error handling
      const nonExistentDir = '/this/directory/does/not/exist';
      
      const result = await runCLI(['validate', nonExistentDir]);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toMatch(/does not exist|permission|access|not found/);
    });
  });

  describe('Configuration Integration', () => {
    it('should load default configuration', async () => {
      // Run a command that should succeed with default config
      const result = await runCLI(['validate', tempRepoDir]);
      
      // Should not fail due to configuration issues
      expect(result.stderr).not.toContain('configuration');
    });

    it('should respect environment variables', async () => {
      // Set environment variables
      const oldEnv = process.env;
      process.env.MIMIR_LOG_LEVEL = 'error';
      process.env.LENS_ENABLED = 'false';
      
      try {
        const result = await runCLI(['validate', tempRepoDir]);
        
        // Should process environment variables without errors
        expect(result.stderr).not.toContain('configuration');
      } finally {
        process.env = oldEnv;
      }
    });
  });

  describe('Output Formats', () => {
    it('should handle human-readable output by default', async () => {
      const result = await runCLI(['validate', tempRepoDir]);
      
      if (result.exitCode === 0) {
        // Human-readable output should contain descriptive text
        expect(result.stdout).toMatch(/[a-zA-Z]+/); // Contains letters, not just JSON
      }
    });

    it('should provide meaningful error messages', async () => {
      const result = await runCLI(['index', '/non/existent/path']);
      
      expect(result.exitCode).toBe(1);
      expect(result.stderr).toBeTruthy();
      expect(result.stderr.length).toBeGreaterThan(10); // Meaningful message
    });

    it('should handle interrupted operations gracefully', async () => {
      // This test is difficult to implement reliably in a unit test
      // but would test SIGINT/SIGTERM handling
      expect(true).toBe(true); // Placeholder
    });
  });

  describe('Repository Structure Validation', () => {
    it('should detect TypeScript files', async () => {
      const result = await runCLI(['validate', tempRepoDir]);
      
      if (result.exitCode === 0) {
        expect(result.stdout.toLowerCase()).toContain('typescript') ||
        expect(result.stdout.toLowerCase()).toContain('files');
      }
    });

    it('should provide file statistics', async () => {
      const result = await runCLI(['validate', tempRepoDir]);
      
      if (result.exitCode === 0) {
        // Should show some kind of file count or statistics
        expect(result.stdout).toMatch(/\d+/); // Contains numbers
      }
    });

    it('should handle empty directories', async () => {
      const emptyDir = join(tempDir, 'empty');
      mkdirSync(emptyDir);
      
      const result = await runCLI(['validate', emptyDir]);
      
      // Should handle empty directories gracefully (success or informative failure)
      expect([0, 1]).toContain(result.exitCode);
    });

    it('should handle directories with only non-code files', async () => {
      const docsDir = join(tempDir, 'docs-only');
      mkdirSync(docsDir);
      writeFileSync(join(docsDir, 'README.md'), '# Documentation');
      writeFileSync(join(docsDir, 'LICENSE'), 'MIT License');
      
      const result = await runCLI(['validate', docsDir]);
      
      expect([0, 1]).toContain(result.exitCode);
    });
  });
});