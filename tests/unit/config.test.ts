/**
 * Unit tests for centralized configuration management.
 * 
 * Tests configuration loading, validation, environment variable parsing,
 * and configuration validation functions.
 */

import { describe, expect, it, beforeEach, afterEach, mock } from 'bun:test';
import { 
  loadConfig,
  loadConfigFromFile,
  validateConfig,
  MimirConfigSchema,
  LensConfigSchema,
  ServerConfigSchema,
  StorageConfigSchema,
  PipelineConfigSchema,
  AIConfigSchema,
  MonitoringConfigSchema,
  LoggingConfigSchema,
  PerformanceConfigSchema
} from '@/config/config';
import type { MimirConfig } from '@/types';
import { writeFileSync, unlinkSync, mkdirSync, rmSync } from 'fs';
import { tmpdir } from 'os';
import { join } from 'path';

describe('Configuration Management', () => {
  const originalEnv = process.env;
  let tempDir: string;
  let tempConfigFile: string;

  beforeEach(() => {
    // Reset any module mocks from other test files
    mock.restore();
    
    // Reset environment
    process.env = { ...originalEnv };
    
    // Create temp directory for test files
    tempDir = join(tmpdir(), `mimir-test-${Date.now()}`);
    mkdirSync(tempDir, { recursive: true });
    tempConfigFile = join(tempDir, 'test-config.json');
  });

  afterEach(() => {
    // Restore environment
    process.env = originalEnv;
    
    // Clean up temp files
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch (error) {
      // Ignore cleanup errors
    }
  });

  describe('Schema Validation', () => {
    it('should validate default server config', () => {
      const config = ServerConfigSchema.parse({});
      expect(config.host).toBe('0.0.0.0');
      expect(config.port).toBe(8000);
      expect(config.timeout).toBe(300);
      expect(config.maxWorkers).toBe(4);
      expect(config.maxMemoryMb).toBe(1024);
    });

    it('should validate lens config with new defaults', () => {
      const config = LensConfigSchema.parse({});
      expect(config.enabled).toBe(true);
      expect(config.baseUrl).toBe('http://localhost:5678');
      expect(config.timeout).toBe(30);
      expect(config.maxRetries).toBe(3);
      expect(config.healthCheckEnabled).toBe(true);
    });

    it('should validate storage config', () => {
      const config = StorageConfigSchema.parse({});
      expect(config.dataPath).toBe('./data');
      expect(config.cachePath).toBe('./cache');
      expect(config.logsPath).toBe('./logs');
    });

    it('should validate pipeline config', () => {
      const config = PipelineConfigSchema.parse({});
      expect(config.enableAcquire).toBe(true);
      expect(config.enableRepomapper).toBe(true);
      expect(config.maxFileSizeMb).toBe(10);
      expect(config.maxRepoSizeMb).toBe(1000);
      expect(config.treeSitterLanguages).toContain('typescript');
      expect(config.treeSitterLanguages).toContain('javascript');
      expect(config.treeSitterLanguages).toContain('python');
    });

    it('should reject invalid server config', () => {
      expect(() => ServerConfigSchema.parse({ port: -1 })).toThrow();
      expect(() => ServerConfigSchema.parse({ port: 70000 })).toThrow();
      expect(() => ServerConfigSchema.parse({ timeout: 0 })).toThrow();
      expect(() => ServerConfigSchema.parse({ maxWorkers: 0 })).toThrow();
    });

    it('should reject invalid lens config', () => {
      expect(() => LensConfigSchema.parse({ timeout: 0 })).toThrow();
      expect(() => LensConfigSchema.parse({ maxRetries: -1 })).toThrow();
      expect(() => LensConfigSchema.parse({ healthCheckTimeout: 0 })).toThrow();
    });
  });

  describe('Environment Variable Loading', () => {
    it('should load server config from environment', () => {
      process.env.MIMIR_UI_HOST = '127.0.0.1';
      process.env.MIMIR_UI_PORT = '9000';
      process.env.MIMIR_MAX_WORKERS = '8';

      const config = loadConfig();
      expect(config.server.host).toBe('127.0.0.1');
      expect(config.server.port).toBe(9000);
      expect(config.server.maxWorkers).toBe(8);
    });

    it('should load lens config from environment', () => {
      process.env.LENS_ENABLED = 'false';
      process.env.LENS_BASE_URL = 'http://localhost:9999';
      process.env.LENS_API_KEY = 'test-key';
      process.env.LENS_TIMEOUT = '60';

      const config = loadConfig();
      expect(config.lens.enabled).toBe(false);
      expect(config.lens.baseUrl).toBe('http://localhost:9999');
      expect(config.lens.apiKey).toBe('test-key');
      expect(config.lens.timeout).toBe(60);
    });

    it('should load boolean values from environment', () => {
      process.env.MIMIR_UI_BEHIND_PROXY = 'true';
      process.env.PIPELINE_ENABLE_ACQUIRE = 'false';
      process.env.LENS_HEALTH_CHECK_ENABLED = 'false';

      const config = loadConfig();
      expect(config.server.behindProxy).toBe(true);
      expect(config.pipeline.enableAcquire).toBe(false);
      expect(config.lens.healthCheckEnabled).toBe(false);
    });

    it('should load array values from environment', () => {
      process.env.TREE_SITTER_LANGUAGES = 'typescript,javascript,python,rust';
      process.env.MIMIR_CORS_ORIGINS = 'http://localhost:3000,http://localhost:8080';

      const config = loadConfig();
      expect(config.pipeline.treeSitterLanguages).toEqual(['typescript', 'javascript', 'python', 'rust']);
      expect(config.server.corsOrigins).toEqual(['http://localhost:3000', 'http://localhost:8080']);
    });
  });

  describe('File Configuration Loading', () => {
    it('should load config from JSON file', () => {
      const fileConfig = {
        server: {
          host: '192.168.1.100',
          port: 3000,
          maxWorkers: 6
        },
        lens: {
          enabled: false,
          baseUrl: 'http://custom-lens:5000',
          timeout: 45
        }
      };

      writeFileSync(tempConfigFile, JSON.stringify(fileConfig, null, 2));

      const config = loadConfigFromFile(tempConfigFile);
      expect(config.server.host).toBe('192.168.1.100');
      expect(config.server.port).toBe(3000);
      expect(config.server.maxWorkers).toBe(6);
      expect(config.lens.enabled).toBe(false);
      expect(config.lens.baseUrl).toBe('http://custom-lens:5000');
      expect(config.lens.timeout).toBe(45);
    });

    it('should throw error for invalid JSON file', () => {
      writeFileSync(tempConfigFile, 'invalid json');
      expect(() => loadConfigFromFile(tempConfigFile)).toThrow();
    });

    it('should throw error for non-existent file', () => {
      expect(() => loadConfigFromFile('/non/existent/file.json')).toThrow();
    });
  });

  describe('Configuration Validation', () => {
    it('should validate valid configuration without warnings', () => {
      const config: MimirConfig = {
        server: {
          host: '0.0.0.0',
          port: 8000,
          behindProxy: false,
          proxyHeaders: false,
          timeout: 300,
          maxWorkers: 4,
          maxMemoryMb: 1024,
          corsOrigins: ['http://localhost:3000']
        },
        storage: {
          dataPath: './data',
          cachePath: './cache',
          logsPath: './logs',
          storageDir: '/app/data',
          cacheDir: '/app/cache'
        },
        pipeline: {
          enableAcquire: true,
          enableRepomapper: true,
          enableSerena: true,
          enableLeann: true,
          enableSnippets: true,
          enableBundle: true,
          timeoutAcquire: 60,
          timeoutRepomapper: 120,
          timeoutSerena: 180,
          timeoutLeann: 240,
          timeoutSnippets: 60,
          timeoutBundle: 30,
          maxFileSizeMb: 10,
          maxRepoSizeMb: 1000,
          treeSitterLanguages: ['typescript', 'javascript'],
          gitDefaultBranch: 'main',
          gitCloneTimeout: 300
        },
        ai: {
          ollamaHost: 'localhost',
          ollamaPort: 11434,
          ollamaModel: 'llama3.2:3b',
          ollamaMaxTokens: 8192,
          ollamaTemperature: 0.1,
          ollamaTimeout: 120,
          enableGemini: false,
          geminiFallback: true,
          enableOllama: true,
          defaultLlmProvider: 'ollama',
          embeddingModel: 'sentence-transformers/all-MiniLM-L6-v2',
          codeEmbeddingModel: 'microsoft/codebert-base',
          enableCodeEmbeddings: true,
          geminiModel: 'gemini-1.5-flash',
          geminiMaxTokens: 8192,
          geminiTemperature: 0.1
        },
        monitoring: {
          enableMetrics: true,
          metricsPort: 9100,
          enableProfiling: false,
          serviceName: 'mimir-repoindex',
          traceConsole: false,
          traceSampleRate: 1.0,
          healthcheckInterval: '30s',
          healthcheckTimeout: '10s',
          healthcheckRetries: 3
        },
        logging: {
          logLevel: 'info',
          logFormat: 'json',
          logMaxSize: '50m',
          logMaxFiles: 10,
          logIncludeTimestamp: true,
          logIncludeLevel: true,
          logIncludeLogger: true,
          logIncludeThread: false
        },
        performance: {
          asyncioMaxWorkers: 10,
          asyncioSemaphoreLimit: 20,
          fileReadBufferSize: 8192,
          fileWriteBufferSize: 8192
        },
        lens: {
          enabled: true,
          baseUrl: 'http://localhost:5678',
          timeout: 30,
          maxRetries: 3,
          retryDelay: 1.0,
          healthCheckEnabled: true,
          healthCheckInterval: 60,
          healthCheckTimeout: 10,
          fallbackEnabled: true,
          fallbackToLocal: true,
          connectionPoolSize: 10,
          keepAliveTimeout: 30,
          enableIndexing: true,
          enableSearch: true,
          enableEmbeddings: true
        }
      };

      const warnings = validateConfig(config);
      expect(warnings).toHaveLength(0);
    });

    it('should warn about server max workers exceeding asyncio max workers', () => {
      const config = loadConfig();
      config.server.maxWorkers = 20;
      config.performance.asyncioMaxWorkers = 10;

      const warnings = validateConfig(config);
      expect(warnings).toContain('Server max workers exceeds asyncio max workers - may cause resource contention');
    });

    it('should warn about metrics enabled without port', () => {
      const config = loadConfig();
      config.monitoring.enableMetrics = true;
      config.monitoring.metricsPort = 0;

      const warnings = validateConfig(config);
      expect(warnings.some(w => w.includes('metrics port'))).toBe(true);
    });

    it('should warn about Gemini enabled without API key', () => {
      const config = loadConfig();
      config.ai.enableGemini = true;
      config.ai.googleApiKey = undefined;
      config.ai.geminiApiKey = undefined;

      const warnings = validateConfig(config);
      expect(warnings).toContain('Gemini enabled but no API key configured');
    });

    it('should warn about Lens enabled without base URL', () => {
      const config = loadConfig();
      config.lens.enabled = true;
      config.lens.baseUrl = '';

      const warnings = validateConfig(config);
      expect(warnings).toContain('Lens integration enabled but no base URL configured');
    });
  });

  describe('Zod Schema Integration', () => {
    it('should validate complete config with Zod', () => {
      const config = loadConfig();
      expect(() => MimirConfigSchema.parse(config)).not.toThrow();
    });

    it('should provide defaults through Zod schemas', () => {
      const parsed = MimirConfigSchema.parse({});
      expect(parsed.server.host).toBe('0.0.0.0');
      expect(parsed.server.port).toBe(8000);
      expect(parsed.lens.enabled).toBe(true);
      expect(parsed.lens.baseUrl).toBe('http://localhost:5678');
    });

    it('should handle partial config objects', () => {
      const partialConfig = {
        server: {
          port: 9000
        },
        lens: {
          timeout: 60
        }
      };

      const parsed = MimirConfigSchema.parse(partialConfig);
      expect(parsed.server.port).toBe(9000);
      expect(parsed.server.host).toBe('0.0.0.0'); // Default
      expect(parsed.lens.timeout).toBe(60);
      expect(parsed.lens.enabled).toBe(true); // Default
    });
  });
});