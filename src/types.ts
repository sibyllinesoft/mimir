/**
 * Core type definitions for Mimir TypeScript implementation
 */
import { z } from 'zod';

// =============================================================================
// MCP Protocol Types
// =============================================================================

export const MCPRequestSchema = z.object({
  jsonrpc: z.literal('2.0'),
  id: z.union([z.string(), z.number()]),
  method: z.string(),
  params: z.record(z.any()).optional(),
});

export const MCPResponseSchema = z.object({
  jsonrpc: z.literal('2.0'),
  id: z.union([z.string(), z.number()]),
  result: z.any().optional(),
  error: z.object({
    code: z.number(),
    message: z.string(),
    data: z.any().optional(),
  }).optional(),
});

export type MCPRequest = z.infer<typeof MCPRequestSchema>;
export type MCPResponse = z.infer<typeof MCPResponseSchema>;

export interface MCPTool {
  name: string;
  description: string;
  inputSchema: {
    type: 'object';
    properties: Record<string, any>;
    required?: string[];
  };
}

// =============================================================================
// Configuration Types
// =============================================================================

export const LensConfigSchema = z.object({
  enabled: z.boolean().default(false),
  baseUrl: z.string().default('http://localhost:3001'),
  apiKey: z.string().optional(),
  timeout: z.number().default(30000),
  maxRetries: z.number().default(3),
  retryDelay: z.number().default(1000),
  healthCheckEnabled: z.boolean().default(true),
  fallbackEnabled: z.boolean().default(true),
});

export const MimirConfigSchema = z.object({
  storageDir: z.string().default('./data'),
  cacheDir: z.string().default('./cache'),
  logLevel: z.enum(['debug', 'info', 'warn', 'error']).default('info'),
  maxWorkers: z.number().default(4),
  timeout: z.number().default(300000),
  lens: LensConfigSchema.default({}),
});

export type LensConfig = z.infer<typeof LensConfigSchema>;
export type MimirConfig = z.infer<typeof MimirConfigSchema>;

// =============================================================================
// Repository & Index Types  
// =============================================================================

export const RepoInfoSchema = z.object({
  root: z.string(),
  rev: z.string(),
  worktreeDirty: z.boolean(),
});

export const IndexConfigSchema = z.object({
  languages: z.array(z.string()).default(['ts', 'js', 'py', 'rs', 'go']),
  excludes: z.array(z.string()).default([
    'node_modules/', 'dist/', 'build/', '.git/', '__pycache__/'
  ]),
  contextLines: z.number().default(3),
  maxFilesToEmbed: z.number().default(1000),
});

export const SymbolEntrySchema = z.object({
  type: z.enum(['def', 'ref', 'call']),
  path: z.string(),
  span: z.tuple([z.number(), z.number()]),
  symbol: z.string().optional(),
  sig: z.string().optional(),
  caller: z.string().optional(),
  callee: z.string().optional(),
});

export const CodeSnippetSchema = z.object({
  path: z.string(),
  span: z.tuple([z.number(), z.number()]),
  hash: z.string(),
  pre: z.string(),
  text: z.string(),
  post: z.string(),
  lineStart: z.number(),
  lineEnd: z.number(),
});

export const SearchResultSchema = z.object({
  path: z.string(),
  span: z.tuple([z.number(), z.number()]),
  score: z.number(),
  scores: z.object({
    vector: z.number(),
    symbol: z.number(), 
    graph: z.number(),
  }),
  content: CodeSnippetSchema,
  citation: z.object({
    repoRoot: z.string(),
    rev: z.string(),
    path: z.string(),
    span: z.tuple([z.number(), z.number()]),
    contentSha: z.string(),
  }),
});

export type RepoInfo = z.infer<typeof RepoInfoSchema>;
export type IndexConfig = z.infer<typeof IndexConfigSchema>;
export type SymbolEntry = z.infer<typeof SymbolEntrySchema>;
export type CodeSnippet = z.infer<typeof CodeSnippetSchema>;
export type SearchResult = z.infer<typeof SearchResultSchema>;

// =============================================================================
// Lens Integration Types
// =============================================================================

export interface LensResponse<T = any> {
  success: boolean;
  data?: T;
  error?: string;
  statusCode: number;
  responseTimeMs: number;
  fromFallback?: boolean;
}

export interface LensIndexRequest {
  repositoryPath: string;
  repositoryId: string;
  branch?: string;
  forceReindex?: boolean;
  includeEmbeddings?: boolean;
  metadata?: Record<string, any>;
}

export interface LensSearchRequest {
  query: string;
  repositoryId?: string;
  maxResults?: number;
  includeEmbeddings?: boolean;
  filters?: Record<string, any>;
}

// =============================================================================
// Pipeline Types
// =============================================================================

export interface PipelineContext {
  indexId: string;
  repoPath: string;
  repoInfo: RepoInfo;
  config: IndexConfig;
  storageDir: string;
  cacheDir: string;
  lensCollectionId?: string;
}

export interface PipelineStage<T = any> {
  name: string;
  execute(context: PipelineContext): Promise<T>;
  cleanup?(): Promise<void>;
}

export type ProgressCallback = (stage: string, progress: number, message?: string) => void;

// =============================================================================
// Utility Types
// =============================================================================

export interface Logger {
  debug(message: string, meta?: any): void;
  info(message: string, meta?: any): void;  
  warn(message: string, meta?: any): void;
  error(message: string, meta?: any): void;
}

export interface FileDiscoveryResult {
  files: string[];
  totalSize: number;
  excludedCount: number;
  duration: number;
}

export interface VectorChunk {
  chunkId: string;
  path: string; 
  span: [number, number];
  content: string;
  embedding: number[];
  tokenCount: number;
}

// =============================================================================
// Error Types
// =============================================================================

export class MimirError extends Error {
  constructor(
    message: string,
    public code: string = 'MIMIR_ERROR',
    public statusCode: number = 500
  ) {
    super(message);
    this.name = 'MimirError';
  }
}

export class LensIntegrationError extends MimirError {
  constructor(message: string, statusCode: number = 503) {
    super(message, 'LENS_ERROR', statusCode);
    this.name = 'LensIntegrationError';
  }
}

export class ConfigurationError extends MimirError {
  constructor(message: string) {
    super(message, 'CONFIG_ERROR', 500);
    this.name = 'ConfigurationError';
  }
}