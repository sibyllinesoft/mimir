/**
 * MCP Server Implementation for Mimir in TypeScript
 * 
 * High-performance MCP server built with Bun that provides deep code research
 * capabilities for Claude Code integration.
 */

import { z } from 'zod';
import type { 
  MCPRequest, 
  MCPResponse, 
  MCPTool, 
  MimirConfig,
  PipelineContext,
  SearchResult,
  Logger,
} from '@/types';

import { createLogger } from '@/utils/logger';
import { loadConfig } from '@/config/config';
import { LensClient } from '@/pipeline/lens-client';
import { FileDiscovery } from '@/pipeline/discovery';
import { SymbolAnalysis } from '@/pipeline/symbols';

export class MimirMCPServer {
  private config: MimirConfig;
  private logger: Logger;
  private lensClient: LensClient;
  private fileDiscovery: FileDiscovery;
  private symbolAnalysis: SymbolAnalysis;
  private isInitialized = false;

  constructor() {
    this.config = loadConfig();
    this.logger = createLogger(this.config.logLevel);
    this.lensClient = new LensClient(this.config.lens, this.logger);
    // FileDiscovery will be instantiated per-request with specific repo path
    this.fileDiscovery = new FileDiscovery(process.cwd(), this.logger);
    this.symbolAnalysis = new SymbolAnalysis(this.logger);
  }

  async initialize(): Promise<void> {
    try {
      this.logger.info('Initializing Mimir MCP Server...');
      
      // Initialize Lens client if enabled
      if (this.config.lens.enabled) {
        await this.lensClient.initialize();
        this.logger.info('Lens integration initialized');
      }

      this.isInitialized = true;
      this.logger.info('Mimir MCP Server initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize MCP server', { error });
      throw error;
    }
  }

  /**
   * Get available MCP tools
   */
  getTools(): MCPTool[] {
    return [
      {
        name: 'ensure_repo_index',
        description: 'Index a repository for deep code analysis using hybrid Mimir-Lens pipeline',
        inputSchema: {
          type: 'object',
          properties: {
            path: { 
              type: 'string', 
              description: 'Path to the repository to index' 
            },
            useLens: { 
              type: 'boolean', 
              default: true,
              description: 'Whether to use Lens for high-performance indexing' 
            },
            forceReindex: {
              type: 'boolean',
              default: false,
              description: 'Force re-indexing even if index exists'
            },
            languages: {
              type: 'array',
              items: { type: 'string' },
              default: ['ts', 'js', 'py', 'rs', 'go'],
              description: 'Programming languages to include'
            },
          },
          required: ['path'],
        },
      },
      {
        name: 'hybrid_search',
        description: 'Search indexed repositories using combined Lens performance + Mimir intelligence',
        inputSchema: {
          type: 'object',
          properties: {
            query: { 
              type: 'string', 
              description: 'Search query' 
            },
            indexId: { 
              type: 'string', 
              description: 'Repository index identifier' 
            },
            maxResults: {
              type: 'number',
              default: 20,
              description: 'Maximum number of results to return'
            },
            lensWeight: {
              type: 'number',
              default: 0.4,
              description: 'Weight for Lens vector similarity scores'
            },
            mimirWeight: {
              type: 'number', 
              default: 0.6,
              description: 'Weight for Mimir semantic analysis scores'
            },
          },
          required: ['query', 'indexId'],
        },
      },
      {
        name: 'deep_code_analysis',
        description: 'Perform advanced code analysis using Mimir specialized capabilities',
        inputSchema: {
          type: 'object',
          properties: {
            target: { 
              type: 'string', 
              description: 'Code target to analyze (file path, symbol, etc.)' 
            },
            analysisDepth: {
              type: 'integer',
              default: 2,
              description: 'Depth of analysis (1=basic, 2=detailed, 3=comprehensive)'
            },
            includeLensContext: {
              type: 'boolean',
              default: true,
              description: 'Include Lens search context in analysis'
            },
            indexId: {
              type: 'string',
              description: 'Repository index identifier'
            },
          },
          required: ['target', 'indexId'],
        },
      },
      {
        name: 'get_repository_status',
        description: 'Get the current indexing status and metadata for a repository',
        inputSchema: {
          type: 'object',
          properties: {
            path: { 
              type: 'string', 
              description: 'Repository path or index ID' 
            },
          },
          required: ['path'],
        },
      },
    ];
  }

  /**
   * Handle MCP request
   */
  async handleRequest(request: MCPRequest): Promise<MCPResponse> {
    try {
      if (!this.isInitialized) {
        await this.initialize();
      }

      const { method, params = {} } = request;

      switch (method) {
        case 'tools/list':
          return this.createSuccessResponse(request.id, {
            tools: this.getTools(),
          });

        case 'tools/call':
          return this.handleToolCall(request);

        case 'ping':
          return this.createSuccessResponse(request.id, { message: 'pong' });

        default:
          return this.createErrorResponse(request.id, -32601, `Method not found: ${method}`);
      }
    } catch (error) {
      this.logger.error('Error handling MCP request', { error, request });
      return this.createErrorResponse(
        request.id, 
        -32603, 
        `Internal error: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle tool call requests
   */
  private async handleToolCall(request: MCPRequest): Promise<MCPResponse> {
    const { name, arguments: args = {} } = request.params || {};

    switch (name) {
      case 'ensure_repo_index':
        return this.handleIndexRepo(request.id, args);
      
      case 'hybrid_search':
        return this.handleHybridSearch(request.id, args);
      
      case 'deep_code_analysis':
        return this.handleDeepAnalysis(request.id, args);
      
      case 'get_repository_status':
        return this.handleGetStatus(request.id, args);

      default:
        return this.createErrorResponse(request.id, -32601, `Tool not found: ${name}`);
    }
  }

  /**
   * Handle repository indexing
   */
  private async handleIndexRepo(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { path, useLens = true, forceReindex = false, languages } = args;
      
      this.logger.info('Starting repository indexing', { path, useLens });

      // Create pipeline context
      const context: PipelineContext = {
        indexId: this.generateIndexId(path),
        repoPath: path,
        repoInfo: {
          root: path,
          rev: 'main', // TODO: Get actual git rev
          worktreeDirty: false, // TODO: Check git status
        },
        config: {
          languages: languages || this.config.lens.enabled ? ['ts', 'js', 'py', 'rs', 'go'] : ['ts', 'js'],
          excludes: ['node_modules/', 'dist/', 'build/', '.git/', '__pycache__/'],
          contextLines: 3,
          maxFilesToEmbed: 1000,
        },
        storageDir: this.config.storageDir,
        cacheDir: this.config.cacheDir,
      };

      // Discover files
      const discoveryResult = await this.fileDiscovery.discover(path, context.config);
      
      let result: any = {
        indexId: context.indexId,
        filesIndexed: discoveryResult.files.length,
        totalSize: discoveryResult.totalSize,
        usedLens: false,
      };

      // Use Lens for bulk indexing if enabled
      if (useLens && this.config.lens.enabled) {
        const lensResponse = await this.lensClient.indexRepository({
          repositoryPath: path,
          repositoryId: context.indexId,
          branch: context.repoInfo.rev,
          forceReindex,
          includeEmbeddings: true,
          metadata: { mimirVersion: '2.0.0', languages: context.config.languages },
        });

        if (lensResponse.success) {
          context.lensCollectionId = lensResponse.data?.collectionId;
          result.usedLens = true;
          result.lensCollectionId = context.lensCollectionId;
        } else {
          this.logger.warn('Lens indexing failed, falling back to local', { error: lensResponse.error });
        }
      }

      // Perform symbol analysis
      const symbolResults = await this.symbolAnalysis.analyze(discoveryResult.files, context);
      result.symbolsFound = symbolResults.symbols.length;

      this.logger.info('Repository indexing completed', result);

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: `Repository indexed successfully!
          
Index ID: ${result.indexId}
Files indexed: ${result.filesIndexed}
Total size: ${Math.round(result.totalSize / 1024)} KB
Symbols found: ${result.symbolsFound}
Used Lens: ${result.usedLens}
${result.lensCollectionId ? `Lens Collection: ${result.lensCollectionId}` : ''}

The repository is now ready for deep code analysis and search.`,
        }],
      });

    } catch (error) {
      this.logger.error('Failed to index repository', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Failed to index repository: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle hybrid search requests
   */
  private async handleHybridSearch(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { query, indexId, maxResults = 20, lensWeight = 0.4, mimirWeight = 0.6 } = args;

      this.logger.info('Performing hybrid search', { query, indexId, maxResults });

      let results: SearchResult[] = [];

      // Stage 1: Lens vector search (if enabled)
      if (this.config.lens.enabled) {
        const lensResponse = await this.lensClient.searchRepository({
          query,
          repositoryId: indexId,
          maxResults: maxResults * 3, // Over-retrieve for filtering
        });

        if (lensResponse.success && lensResponse.data?.results) {
          // Convert Lens results to our format
          results = lensResponse.data.results.map((result: any) => ({
            path: result.path,
            span: result.span || [0, 0],
            score: result.score * lensWeight,
            scores: {
              vector: result.score,
              symbol: 0,
              graph: 0,
            },
            content: {
              path: result.path,
              span: result.span || [0, 0],
              hash: result.contentHash || '',
              pre: result.context?.pre || '',
              text: result.content || '',
              post: result.context?.post || '',
              lineStart: result.span?.[0] || 0,
              lineEnd: result.span?.[1] || 0,
            },
            citation: {
              repoRoot: result.repoRoot || '',
              rev: result.rev || 'main',
              path: result.path,
              span: result.span || [0, 0],
              contentSha: result.contentHash || '',
            },
          }));
        }
      }

      // Stage 2: Mimir semantic analysis (enhance results)
      if (results.length > 0) {
        for (const result of results) {
          const symbolScore = await this.symbolAnalysis.computeRelevance(query, result);
          result.scores.symbol = symbolScore;
          result.score = (lensWeight * result.scores.vector) + (mimirWeight * symbolScore);
        }

        // Re-sort by hybrid score
        results.sort((a, b) => b.score - a.score);
        results = results.slice(0, maxResults);
      }

      const searchSummary = `Found ${results.length} results for "${query}"`;

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text', 
          text: `${searchSummary}

${results.map((result, i) => `${i + 1}. ${result.path} (score: ${result.score.toFixed(3)})
   ${result.content.text.slice(0, 100)}...`).join('\n\n')}`,
        }],
      });

    } catch (error) {
      this.logger.error('Failed to perform hybrid search', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Search failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle deep code analysis requests  
   */
  private async handleDeepAnalysis(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { target, analysisDepth = 2, includeLensContext = true, indexId } = args;

      this.logger.info('Performing deep code analysis', { target, analysisDepth, indexId });

      const analysis = await this.symbolAnalysis.deepAnalyze(target, {
        depth: analysisDepth,
        includeLensContext,
        indexId,
      });

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: `Deep Code Analysis for "${target}"

${analysis.summary}

Key Findings:
${analysis.findings.map((f: any) => `• ${f}`).join('\n')}

Dependencies: ${analysis.dependencies?.length || 0}
Complexity Score: ${analysis.complexity || 'N/A'}
${analysis.recommendations ? `\nRecommendations:\n${analysis.recommendations.map((r: any) => `• ${r}`).join('\n')}` : ''}`,
        }],
      });

    } catch (error) {
      this.logger.error('Failed to perform deep analysis', { error, args });
      return this.createErrorResponse(
        requestId, 
        -32603,
        `Analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle repository status requests
   */
  private async handleGetStatus(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { path } = args;
      const indexId = this.generateIndexId(path);

      // Check Lens status if enabled
      let lensStatus = null;
      if (this.config.lens.enabled) {
        const response = await this.lensClient.getRepositoryStatus(indexId);
        if (response.success) {
          lensStatus = response.data;
        }
      }

      const status = {
        indexId,
        path,
        indexed: lensStatus?.indexed || false,
        lastUpdated: lensStatus?.lastUpdated || 'Never',
        filesCount: lensStatus?.filesCount || 0,
        symbolsCount: lensStatus?.symbolsCount || 0,
        lensEnabled: this.config.lens.enabled,
        lensStatus: lensStatus?.status || 'Unknown',
      };

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: `Repository Status: ${path}

Index ID: ${status.indexId}  
Indexed: ${status.indexed ? 'Yes' : 'No'}
Last Updated: ${status.lastUpdated}
Files: ${status.filesCount}
Symbols: ${status.symbolsCount}
Lens Enabled: ${status.lensEnabled ? 'Yes' : 'No'}
${status.lensEnabled ? `Lens Status: ${status.lensStatus}` : ''}`,
        }],
      });

    } catch (error) {
      this.logger.error('Failed to get repository status', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603, 
        `Status check failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Helper methods
   */
  private generateIndexId(path: string): string {
    // Simple hash-based ID generation
    return Buffer.from(path).toString('base64').replace(/[/+=]/g, '').slice(0, 16);
  }

  private createSuccessResponse(id: string | number, result: any): MCPResponse {
    return {
      jsonrpc: '2.0',
      id,
      result,
    };
  }

  private createErrorResponse(id: string | number, code: number, message: string): MCPResponse {
    return {
      jsonrpc: '2.0',
      id,
      error: { code, message },
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    this.logger.info('Cleaning up MCP server resources');
    if (this.lensClient) {
      await this.lensClient.cleanup();
    }
  }
}