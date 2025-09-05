/**
 * Configuration management for Mimir TypeScript implementation
 * 
 * Provides type-safe configuration handling using Zod schemas with
 * comprehensive validation, environment variable support, and 12-factor app principles.
 */

import { z } from 'zod';
import { readFileSync } from 'fs';
import { resolve } from 'path';
import type { Logger } from '@/types';
import { createLogger } from '@/utils/logger';

// =============================================================================
// Configuration Schemas
// =============================================================================

export const ServerConfigSchema = z.object({
  host: z.string().default('0.0.0.0'),
  port: z.number().int().min(1).max(65535).default(8000),
  behindProxy: z.boolean().default(false),
  proxyHeaders: z.boolean().default(false),
  timeout: z.number().int().min(1).max(3600).default(300),
  maxWorkers: z.number().int().min(1).max(32).default(4),
  maxMemoryMb: z.number().int().min(128).default(1024),
  corsOrigins: z.array(z.string()).default(['http://localhost:3000', 'http://localhost:8000']),
});

export const StorageConfigSchema = z.object({
  dataPath: z.string().default('./data'),
  cachePath: z.string().default('./cache'),
  logsPath: z.string().default('./logs'),
  storageDir: z.string().default('/app/data'),
  cacheDir: z.string().default('/app/cache'),
});

export const PipelineConfigSchema = z.object({
  enableAcquire: z.boolean().default(true),
  enableRepomapper: z.boolean().default(true),
  enableSerena: z.boolean().default(true),
  enableLeann: z.boolean().default(true),
  enableSnippets: z.boolean().default(true),
  enableBundle: z.boolean().default(true),
  timeoutAcquire: z.number().int().min(1).max(3600).default(60),
  timeoutRepomapper: z.number().int().min(1).max(3600).default(120),
  timeoutSerena: z.number().int().min(1).max(3600).default(180),
  timeoutLeann: z.number().int().min(1).max(3600).default(240),
  timeoutSnippets: z.number().int().min(1).max(3600).default(60),
  timeoutBundle: z.number().int().min(1).max(3600).default(30),
  maxFileSizeMb: z.number().int().min(1).default(10),
  maxRepoSizeMb: z.number().int().min(1).default(1000),
  treeSitterLanguages: z.array(z.string()).default(['typescript', 'javascript', 'python', 'rust', 'go', 'java']),
  gitDefaultBranch: z.string().default('main'),
  gitCloneTimeout: z.number().int().min(1).max(3600).default(300),
});

export const AIConfigSchema = z.object({
  googleApiKey: z.string().optional(),
  geminiApiKey: z.string().optional(),
  geminiModel: z.string().default('gemini-1.5-flash'),
  geminiMaxTokens: z.number().int().min(1).max(32768).default(8192),
  geminiTemperature: z.number().min(0).max(2).default(0.1),
  ollamaHost: z.string().default('localhost'),
  ollamaPort: z.number().int().min(1).max(65535).default(11434),
  ollamaModel: z.string().default('llama3.2:3b'),
  ollamaMaxTokens: z.number().int().min(1).max(32768).default(8192),
  ollamaTemperature: z.number().min(0).max(2).default(0.1),
  ollamaTimeout: z.number().int().min(1).default(120),
  enableGemini: z.boolean().default(true),
  geminiFallback: z.boolean().default(true),
  enableOllama: z.boolean().default(true),
  defaultLlmProvider: z.enum(['ollama', 'gemini', 'mock']).default('ollama'),
  embeddingServiceUrl: z.string().optional(),
  embeddingModel: z.string().default('sentence-transformers/all-MiniLM-L6-v2'),
  codeEmbeddingModel: z.string().default('microsoft/codebert-base'),
  enableCodeEmbeddings: z.boolean().default(true),
});

export const MonitoringConfigSchema = z.object({
  enableMetrics: z.boolean().default(false),
  metricsPort: z.number().int().min(1024).max(65535).default(9100),
  enableProfiling: z.boolean().default(false),
  serviceName: z.string().default('mimir-repoindex'),
  jaegerEndpoint: z.string().optional(),
  otlpEndpoint: z.string().optional(),
  traceConsole: z.boolean().default(false),
  traceSampleRate: z.number().min(0).max(1).default(1.0),
  healthcheckInterval: z.string().default('30s'),
  healthcheckTimeout: z.string().default('10s'),
  healthcheckRetries: z.number().int().min(1).default(3),
});

export const LoggingConfigSchema = z.object({
  logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  logFormat: z.enum(['json', 'text', 'human']).default('json'),
  logFile: z.string().optional(),
  logMaxSize: z.string().default('50m'),
  logMaxFiles: z.number().int().min(1).default(10),
  logIncludeTimestamp: z.boolean().default(true),
  logIncludeLevel: z.boolean().default(true),
  logIncludeLogger: z.boolean().default(true),
  logIncludeThread: z.boolean().default(false),
});

export const PerformanceConfigSchema = z.object({
  asyncioMaxWorkers: z.number().int().min(1).default(10),
  asyncioSemaphoreLimit: z.number().int().min(1).default(20),
  fileReadBufferSize: z.number().int().min(1024).max(1024 * 1024).default(8192),
  fileWriteBufferSize: z.number().int().min(1024).max(1024 * 1024).default(8192),
});

export const LensConfigSchema = z.object({
  enabled: z.boolean().default(true),
  baseUrl: z.string().default('http://localhost:5678'),
  apiKey: z.string().optional(),
  timeout: z.number().int().min(1).max(300).default(30),
  maxRetries: z.number().int().min(0).max(10).default(3),
  retryDelay: z.number().min(0.1).max(60).default(1.0),
  healthCheckEnabled: z.boolean().default(true),
  healthCheckInterval: z.number().int().min(1).default(60),
  healthCheckTimeout: z.number().int().min(1).max(300).default(10),
  fallbackEnabled: z.boolean().default(true),
  fallbackToLocal: z.boolean().default(true),
  connectionPoolSize: z.number().int().min(1).max(100).default(10),
  keepAliveTimeout: z.number().int().min(1).default(30),
  enableIndexing: z.boolean().default(true),
  enableSearch: z.boolean().default(true),
  enableEmbeddings: z.boolean().default(true),
});

export const MimirConfigSchema = z.object({
  server: ServerConfigSchema.default({}),
  storage: StorageConfigSchema.default({}),
  pipeline: PipelineConfigSchema.default({}),
  ai: AIConfigSchema.default({}),
  monitoring: MonitoringConfigSchema.default({}),
  logging: LoggingConfigSchema.default({}),
  performance: PerformanceConfigSchema.default({}),
  lens: LensConfigSchema.default({}),
});

export type ServerConfig = z.infer<typeof ServerConfigSchema>;
export type StorageConfig = z.infer<typeof StorageConfigSchema>;
export type PipelineConfig = z.infer<typeof PipelineConfigSchema>;
export type AIConfig = z.infer<typeof AIConfigSchema>;
export type MonitoringConfig = z.infer<typeof MonitoringConfigSchema>;
export type LoggingConfig = z.infer<typeof LoggingConfigSchema>;
export type PerformanceConfig = z.infer<typeof PerformanceConfigSchema>;
export type LensConfig = z.infer<typeof LensConfigSchema>;
export type MimirConfig = z.infer<typeof MimirConfigSchema>;

// =============================================================================
// Environment Variable Loading
// =============================================================================

function parseEnvValue(value: string | undefined, defaultValue: any): any {
  if (!value) return defaultValue;
  
  // Handle boolean values
  if (typeof defaultValue === 'boolean') {
    return value.toLowerCase() === 'true';
  }
  
  // Handle number values
  if (typeof defaultValue === 'number') {
    const parsed = Number(value);
    return isNaN(parsed) ? defaultValue : parsed;
  }
  
  // Handle array values (comma-separated)
  if (Array.isArray(defaultValue)) {
    return value.split(',').map(v => v.trim());
  }
  
  return value;
}

function loadConfigFromEnv(): Partial<MimirConfig> {
  const env = process.env;
  
  return {
    server: {
      host: parseEnvValue(env.MIMIR_UI_HOST, '0.0.0.0'),
      port: parseEnvValue(env.MIMIR_UI_PORT, 8000),
      behindProxy: parseEnvValue(env.MIMIR_UI_BEHIND_PROXY, false),
      proxyHeaders: parseEnvValue(env.MIMIR_UI_PROXY_HEADERS, false),
      timeout: parseEnvValue(env.MIMIR_TIMEOUT, 300),
      maxWorkers: parseEnvValue(env.MIMIR_MAX_WORKERS, 4),
      maxMemoryMb: parseEnvValue(env.MIMIR_MAX_MEMORY_MB, 1024),
      corsOrigins: parseEnvValue(env.MIMIR_CORS_ORIGINS, ['http://localhost:3000', 'http://localhost:8000']),
    },
    storage: {
      dataPath: parseEnvValue(env.MIMIR_DATA_PATH, './data'),
      cachePath: parseEnvValue(env.MIMIR_CACHE_PATH, './cache'),
      logsPath: parseEnvValue(env.MIMIR_LOGS_PATH, './logs'),
      storageDir: parseEnvValue(env.MIMIR_STORAGE_DIR, '/app/data'),
      cacheDir: parseEnvValue(env.MIMIR_CACHE_DIR, '/app/cache'),
    },
    pipeline: {
      enableAcquire: parseEnvValue(env.PIPELINE_ENABLE_ACQUIRE, true),
      enableRepomapper: parseEnvValue(env.PIPELINE_ENABLE_REPOMAPPER, true),
      enableSerena: parseEnvValue(env.PIPELINE_ENABLE_SERENA, true),
      enableLeann: parseEnvValue(env.PIPELINE_ENABLE_LEANN, true),
      enableSnippets: parseEnvValue(env.PIPELINE_ENABLE_SNIPPETS, true),
      enableBundle: parseEnvValue(env.PIPELINE_ENABLE_BUNDLE, true),
      timeoutAcquire: parseEnvValue(env.PIPELINE_TIMEOUT_ACQUIRE, 60),
      timeoutRepomapper: parseEnvValue(env.PIPELINE_TIMEOUT_REPOMAPPER, 120),
      timeoutSerena: parseEnvValue(env.PIPELINE_TIMEOUT_SERENA, 180),
      timeoutLeann: parseEnvValue(env.PIPELINE_TIMEOUT_LEANN, 240),
      timeoutSnippets: parseEnvValue(env.PIPELINE_TIMEOUT_SNIPPETS, 60),
      timeoutBundle: parseEnvValue(env.PIPELINE_TIMEOUT_BUNDLE, 30),
      maxFileSizeMb: parseEnvValue(env.PIPELINE_MAX_FILE_SIZE, 10),
      maxRepoSizeMb: parseEnvValue(env.PIPELINE_MAX_REPO_SIZE, 1000),
      treeSitterLanguages: parseEnvValue(env.TREE_SITTER_LANGUAGES, ['typescript', 'javascript', 'python', 'rust', 'go', 'java']),
      gitDefaultBranch: parseEnvValue(env.GIT_DEFAULT_BRANCH, 'main'),
      gitCloneTimeout: parseEnvValue(env.GIT_CLONE_TIMEOUT, 300),
    },
    ai: {
      googleApiKey: env.GOOGLE_API_KEY,
      geminiApiKey: env.GEMINI_API_KEY,
      geminiModel: parseEnvValue(env.GEMINI_MODEL, 'gemini-1.5-flash'),
      geminiMaxTokens: parseEnvValue(env.GEMINI_MAX_TOKENS, 8192),
      geminiTemperature: parseEnvValue(env.GEMINI_TEMPERATURE, 0.1),
      ollamaHost: parseEnvValue(env.OLLAMA_HOST, 'localhost'),
      ollamaPort: parseEnvValue(env.OLLAMA_PORT, 11434),
      ollamaModel: parseEnvValue(env.OLLAMA_MODEL, 'llama3.2:3b'),
      ollamaMaxTokens: parseEnvValue(env.OLLAMA_MAX_TOKENS, 8192),
      ollamaTemperature: parseEnvValue(env.OLLAMA_TEMPERATURE, 0.1),
      ollamaTimeout: parseEnvValue(env.OLLAMA_TIMEOUT, 120),
      enableGemini: parseEnvValue(env.MIMIR_ENABLE_GEMINI, true),
      geminiFallback: parseEnvValue(env.MIMIR_GEMINI_FALLBACK, true),
      enableOllama: parseEnvValue(env.MIMIR_ENABLE_OLLAMA, true),
      defaultLlmProvider: parseEnvValue(env.MIMIR_DEFAULT_LLM_PROVIDER, 'ollama'),
      embeddingServiceUrl: env.EMBEDDING_SERVICE_URL,
      embeddingModel: parseEnvValue(env.EMBEDDING_MODEL, 'sentence-transformers/all-MiniLM-L6-v2'),
      codeEmbeddingModel: parseEnvValue(env.CODE_EMBEDDING_MODEL, 'microsoft/codebert-base'),
      enableCodeEmbeddings: parseEnvValue(env.MIMIR_ENABLE_CODE_EMBEDDINGS, true),
    },
    monitoring: {
      enableMetrics: parseEnvValue(env.MIMIR_ENABLE_METRICS, false),
      metricsPort: parseEnvValue(env.MIMIR_METRICS_PORT, 9100),
      enableProfiling: parseEnvValue(env.MIMIR_ENABLE_PROFILING, false),
      serviceName: parseEnvValue(env.MIMIR_SERVICE_NAME, 'mimir-repoindex'),
      jaegerEndpoint: env.JAEGER_ENDPOINT,
      otlpEndpoint: env.OTEL_EXPORTER_OTLP_ENDPOINT,
      traceConsole: parseEnvValue(env.MIMIR_TRACE_CONSOLE, false),
      traceSampleRate: parseEnvValue(env.MIMIR_TRACE_SAMPLE_RATE, 1.0),
    },
    logging: {
      logLevel: parseEnvValue(env.MIMIR_LOG_LEVEL, 'info')?.toLowerCase(),
      logFormat: parseEnvValue(env.LOG_FORMAT, 'json'),
      logFile: env.MIMIR_LOG_FILE,
    },
    performance: {
      asyncioMaxWorkers: parseEnvValue(env.ASYNCIO_MAX_WORKERS, 10),
      asyncioSemaphoreLimit: parseEnvValue(env.ASYNCIO_SEMAPHORE_LIMIT, 20),
      fileReadBufferSize: parseEnvValue(env.FILE_READ_BUFFER_SIZE, 8192),
      fileWriteBufferSize: parseEnvValue(env.FILE_WRITE_BUFFER_SIZE, 8192),
    },
    lens: {
      enabled: parseEnvValue(env.LENS_ENABLED, true),
      baseUrl: parseEnvValue(env.LENS_BASE_URL, 'http://localhost:5678'),
      apiKey: env.LENS_API_KEY,
      timeout: parseEnvValue(env.LENS_TIMEOUT, 30),
      maxRetries: parseEnvValue(env.LENS_MAX_RETRIES, 3),
      retryDelay: parseEnvValue(env.LENS_RETRY_DELAY, 1.0),
      healthCheckEnabled: parseEnvValue(env.LENS_HEALTH_CHECK_ENABLED, true),
      healthCheckInterval: parseEnvValue(env.LENS_HEALTH_CHECK_INTERVAL, 60),
      healthCheckTimeout: parseEnvValue(env.LENS_HEALTH_CHECK_TIMEOUT, 10),
      fallbackEnabled: parseEnvValue(env.LENS_FALLBACK_ENABLED, true),
      fallbackToLocal: parseEnvValue(env.LENS_FALLBACK_TO_LOCAL, true),
      connectionPoolSize: parseEnvValue(env.LENS_CONNECTION_POOL_SIZE, 10),
      keepAliveTimeout: parseEnvValue(env.LENS_KEEP_ALIVE_TIMEOUT, 30),
      enableIndexing: parseEnvValue(env.LENS_ENABLE_INDEXING, true),
      enableSearch: parseEnvValue(env.LENS_ENABLE_SEARCH, true),
      enableEmbeddings: parseEnvValue(env.LENS_ENABLE_EMBEDDINGS, true),
    },
  };
}

// =============================================================================
// Configuration Loading and Management
// =============================================================================

let globalConfig: MimirConfig | null = null;

export function loadConfig(): MimirConfig {
  try {
    // Load from environment variables
    const envConfig = loadConfigFromEnv();
    
    // Parse and validate with Zod
    const config = MimirConfigSchema.parse(envConfig);
    
    return config;
  } catch (error) {
    throw new Error(`Failed to load configuration: ${error}`);
  }
}

export function loadConfigFromFile(filePath: string): MimirConfig {
  try {
    const fileContent = readFileSync(resolve(filePath), 'utf-8');
    const fileConfig = JSON.parse(fileContent);
    
    // Parse and validate with Zod
    const config = MimirConfigSchema.parse(fileConfig);
    
    return config;
  } catch (error) {
    throw new Error(`Failed to load configuration from file ${filePath}: ${error}`);
  }
}

export function getConfig(): MimirConfig {
  if (!globalConfig) {
    globalConfig = loadConfig();
  }
  return globalConfig;
}

export function setConfig(config: MimirConfig): void {
  globalConfig = config;
}

export function reloadConfig(): MimirConfig {
  globalConfig = loadConfig();
  return globalConfig;
}

// =============================================================================
// Configuration Validation and Helpers
// =============================================================================

export function validateConfig(config: MimirConfig): string[] {
  const warnings: string[] = [];
  
  // Cross-configuration validation
  if (config.server.maxWorkers > config.performance.asyncioMaxWorkers) {
    warnings.push('Server max workers exceeds asyncio max workers - may cause resource contention');
  }
  
  if (config.monitoring.enableMetrics && !config.monitoring.metricsPort) {
    warnings.push('Metrics enabled but no metrics port configured');
  }
  
  if (config.ai.enableGemini && !config.ai.googleApiKey && !config.ai.geminiApiKey) {
    warnings.push('Gemini enabled but no API key configured');
  }
  
  if (config.lens.enabled && !config.lens.baseUrl) {
    warnings.push('Lens integration enabled but no base URL configured');
  }
  
  return warnings;
}

export function getApiKey(config: AIConfig): string | undefined {
  return config.googleApiKey || config.geminiApiKey;
}

export function getOllamaBaseUrl(config: AIConfig): string {
  return `http://${config.ollamaHost}:${config.ollamaPort}`;
}

// Convenience functions for accessing specific configuration sections
export function getServerConfig(): ServerConfig {
  return getConfig().server;
}

export function getStorageConfig(): StorageConfig {
  return getConfig().storage;
}

export function getPipelineConfig(): PipelineConfig {
  return getConfig().pipeline;
}

export function getAIConfig(): AIConfig {
  return getConfig().ai;
}

export function getMonitoringConfig(): MonitoringConfig {
  return getConfig().monitoring;
}

export function getLoggingConfig(): LoggingConfig {
  return getConfig().logging;
}

export function getPerformanceConfig(): PerformanceConfig {
  return getConfig().performance;
}

export function getLensConfig(): LensConfig {
  return getConfig().lens;
}
