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
import { LoadoutManager, LoadoutManagerConfig } from '@/research/loadout-manager';
import { IntelligenceEngine, IntelligenceEngineConfig } from '@/research/intelligence-engine';
import { SwarmOptimizedReporter, SwarmReportConfig } from '@/research/swarm-reporter';
import { LoadoutMetricsSystem } from '@/research/metrics-system';
import type { ResearchLoadout, ConfigurableResearchResult } from '@/research/types';

export class MimirMCPServer {
  private config: MimirConfig;
  private logger: Logger;
  private lensClient: LensClient;
  private fileDiscovery: FileDiscovery;
  private symbolAnalysis: SymbolAnalysis;
  private loadoutManager: LoadoutManager;
  private intelligenceEngine: IntelligenceEngine;
  private swarmReporter: SwarmOptimizedReporter;
  private metricsSystem: LoadoutMetricsSystem;
  private isInitialized = false;

  constructor() {
    this.config = loadConfig();
    this.logger = createLogger(this.config.logLevel);
    this.lensClient = new LensClient(this.config.lens, this.logger);
    // FileDiscovery will be instantiated per-request with specific repo path
    this.fileDiscovery = new FileDiscovery(process.cwd(), this.logger);
    this.symbolAnalysis = new SymbolAnalysis(this.logger);
    
    // Initialize research system components
    const loadoutManagerConfig: LoadoutManagerConfig = {
      loadoutDirectories: [
        './loadouts/core',
        './loadouts/specialized', 
        './loadouts/experimental'
      ],
      enableHotReload: true,
      cacheEnabled: true,
      validationStrict: false,
      defaultLoadout: 'Deep Intelligence Gathering'
    };
    
    this.loadoutManager = new LoadoutManager(loadoutManagerConfig);
    
    const intelligenceEngineConfig: IntelligenceEngineConfig = {
      maxConcurrentAgents: 10,
      defaultTimeout: 300000,
      enableMetrics: true,
      enableCaching: true,
      cacheDirectory: './cache/intelligence'
    };
    
    this.intelligenceEngine = new IntelligenceEngine(
      intelligenceEngineConfig,
      this.symbolAnalysis,
      this.lensClient
    );
    
    this.swarmReporter = new SwarmOptimizedReporter();
    this.metricsSystem = new LoadoutMetricsSystem(this.loadoutManager);
  }

  async initialize(): Promise<void> {
    try {
      this.logger.info('Initializing Mimir MCP Server...');
      
      // Initialize Lens client if enabled
      if (this.config.lens.enabled) {
        await this.lensClient.initialize();
        this.logger.info('Lens integration initialized');
      }

      // Initialize research system components
      await this.loadoutManager.initialize();
      this.logger.info('Loadout manager initialized');

      await this.metricsSystem.initialize();
      this.logger.info('Metrics system initialized');

      this.isInitialized = true;
      this.logger.info('Mimir MCP Server initialized successfully', {
        loadouts: this.loadoutManager.getValidLoadoutNames().length,
        lensEnabled: this.config.lens.enabled
      });
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
      {
        name: 'configurable_intelligence_research',
        description: 'Execute configurable intelligence-gathering research using loadout configurations',
        inputSchema: {
          type: 'object',
          properties: {
            target: {
              type: 'string',
              description: 'Research target (file, directory, or general description)'
            },
            loadout: {
              type: 'string',
              description: 'Research loadout configuration name',
              default: 'Deep Intelligence Gathering'
            },
            customConfig: {
              type: 'object',
              description: 'Custom configuration overrides for the loadout',
              properties: {
                densityTarget: { type: 'number', description: 'Target token density for output' },
                maxIterations: { type: 'number', description: 'Maximum research iterations' },
                parallelismLevel: { type: 'number', description: 'Level of parallel processing' }
              }
            },
            indexId: {
              type: 'string',
              description: 'Repository index identifier (if available)'
            }
          },
          required: ['target']
        }
      },
      {
        name: 'compare_research_loadouts',
        description: 'Compare multiple research loadouts on the same target for optimization',
        inputSchema: {
          type: 'object',
          properties: {
            target: {
              type: 'string',
              description: 'Research target to analyze with multiple loadouts'
            },
            loadouts: {
              type: 'array',
              items: { type: 'string' },
              description: 'List of loadout names to compare',
              minItems: 2,
              maxItems: 4
            },
            indexId: {
              type: 'string',
              description: 'Repository index identifier (if available)'
            },
            comparisonMetrics: {
              type: 'array',
              items: { 
                type: 'string',
                enum: ['quality', 'speed', 'comprehensiveness', 'actionability']
              },
              description: 'Metrics to focus on in comparison',
              default: ['quality', 'comprehensiveness']
            }
          },
          required: ['target', 'loadouts']
        }
      },
      {
        name: 'list_research_loadouts',
        description: 'List available research loadout configurations with their descriptions',
        inputSchema: {
          type: 'object',
          properties: {
            category: {
              type: 'string',
              enum: ['core', 'specialized', 'experimental', 'all'],
              default: 'all',
              description: 'Filter loadouts by category'
            },
            includeDetails: {
              type: 'boolean',
              default: false,
              description: 'Include detailed configuration information'
            }
          }
        }
      },
      {
        name: 'loadout_metrics_analysis',
        description: 'Get comprehensive metrics and analysis for research loadouts',
        inputSchema: {
          type: 'object',
          properties: {
            loadouts: {
              type: 'array',
              items: { type: 'string' },
              description: 'List of loadout names to analyze (empty for all)',
              default: []
            },
            includeComparison: {
              type: 'boolean',
              default: false,
              description: 'Include comparative analysis between loadouts'
            },
            generateReport: {
              type: 'boolean',
              default: false,
              description: 'Generate a detailed metrics report'
            }
          }
        }
      },
      {
        name: 'benchmark_loadout_performance',
        description: 'Benchmark loadout performance on a specific test case',
        inputSchema: {
          type: 'object',
          properties: {
            loadout: {
              type: 'string',
              description: 'Loadout name to benchmark'
            },
            testCase: {
              type: 'string',
              description: 'Description of the benchmark test case'
            },
            target: {
              type: 'string',
              description: 'Target for benchmarking (file, directory, or description)'
            },
            indexId: {
              type: 'string',
              description: 'Repository index identifier (if available)'
            }
          },
          required: ['loadout', 'testCase', 'target']
        }
      },
      {
        name: 'recommend_optimal_loadout',
        description: 'Get loadout recommendations based on specific requirements',
        inputSchema: {
          type: 'object',
          properties: {
            requirements: {
              type: 'object',
              properties: {
                speed: {
                  type: 'string',
                  enum: ['very-fast', 'fast', 'moderate', 'slow', 'very-slow'],
                  description: 'Required speed level'
                },
                quality: {
                  type: 'string',
                  enum: ['basic', 'good', 'high', 'excellent'],
                  description: 'Required quality level'
                },
                domain: {
                  type: 'array',
                  items: { type: 'string' },
                  description: 'Domain expertise required (e.g., [\"typescript\", \"security\"])'
                },
                resourceBudget: {
                  type: 'string',
                  enum: ['low', 'moderate', 'high', 'unlimited'],
                  description: 'Available resource budget'
                },
                useCase: {
                  type: 'string',
                  description: 'Specific use case or problem type'
                }
              }
            },
            maxRecommendations: {
              type: 'integer',
              default: 3,
              minimum: 1,
              maximum: 5,
              description: 'Maximum number of recommendations to return'
            }
          },
          required: ['requirements']
        }
      },
      {
        name: 'intelligence_to_swarm_handoff',
        description: 'Generate comprehensive intelligence brief then prepare for refinement swarm handoff',
        inputSchema: {
          type: 'object',
          properties: {
            target: {
              type: 'string',
              description: 'Problem or codebase to analyze'
            },
            intelligenceLoadout: {
              type: 'string',
              description: 'Intelligence gathering configuration',
              default: 'Deep Intelligence Gathering'
            },
            swarmOptimization: {
              type: 'object',
              properties: {
                debateStructure: {
                  type: 'string',
                  enum: ['shallow', 'medium', 'deep'],
                  default: 'medium',
                  description: 'Depth of debate framework structuring'
                },
                densityTarget: {
                  type: 'number',
                  default: 5000,
                  description: 'Target token density for swarm consumption'
                },
                includeDebateTopics: {
                  type: 'boolean',
                  default: true,
                  description: 'Include structured debate topics for swarm'
                }
              }
            },
            indexId: {
              type: 'string',
              description: 'Repository index identifier (if available)'
            }
          },
          required: ['target']
        }
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

      // Validate request exists and has required structure
      if (!request || typeof request !== 'object') {
        return this.createErrorResponse(null, -32600, 'Invalid Request');
      }

      // Validate JSON-RPC format
      if (!request.jsonrpc || request.jsonrpc !== '2.0') {
        return this.createErrorResponse(request.id || null, -32600, 'Invalid Request');
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
        request?.id || null, 
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
      
      case 'configurable_intelligence_research':
        return this.handleConfigurableResearch(request.id, args);
      
      case 'compare_research_loadouts':
        return this.handleCompareLoadouts(request.id, args);
      
      case 'list_research_loadouts':
        return this.handleListLoadouts(request.id, args);
      
      case 'intelligence_to_swarm_handoff':
        return this.handleIntelligenceSwarmHandoff(request.id, args);
        
      case 'loadout_metrics_analysis':
        return this.handleLoadoutMetricsAnalysis(args, request.id);
        
      case 'benchmark_loadout_performance':
        return this.handleBenchmarkLoadoutPerformance(args, request.id);
        
      case 'recommend_optimal_loadout':
        return this.handleRecommendOptimalLoadout(args, request.id);

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
      
      // Validate required parameters
      if (!path || typeof path !== 'string') {
        return this.createErrorResponse(
          requestId,
          -32602,
          'Invalid parameters: path is required and must be a string'
        );
      }
      
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
          // If useLens was explicitly requested and fails, consider it a failure
          if (useLens) {
            return this.createSuccessResponse(requestId, {
              content: [{
                type: 'text',
                text: `Repository indexing failed: Lens service unavailable (${lensResponse.error}). 
                
Please ensure the Lens service is running or disable Lens integration.`,
              }],
            });
          }
          result.lensError = lensResponse.error;
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
${result.lensError ? `Lens service error: ${result.lensError}` : ''}

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

      // Validate required parameters
      if (!query || typeof query !== 'string') {
        return this.createErrorResponse(
          requestId,
          -32602,
          'Invalid parameters: query is required and must be a string'
        );
      }

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
        } else if (!lensResponse.success) {
          // If Lens search fails, report the error
          return this.createSuccessResponse(requestId, {
            content: [{
              type: 'text',
              text: `Search failed: Lens service error (${lensResponse.error}). Please ensure the Lens service is running.`,
            }],
          });
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

      // Validate required parameters
      if (!target || typeof target !== 'string') {
        return this.createErrorResponse(
          requestId,
          -32602,
          'Invalid parameters: target is required and must be a string'
        );
      }

      this.logger.info('Performing deep code analysis', { target, analysisDepth, indexId });

      // Check if deepAnalyze method exists
      if (typeof this.symbolAnalysis.deepAnalyze !== 'function') {
        return this.createSuccessResponse(requestId, {
          content: [{
            type: 'text',
            text: `Deep Code Analysis for "${target}"

Analysis method not yet implemented. This feature is under development.

Basic information:
• Target: ${target}
• Analysis Depth: ${analysisDepth}
• Index ID: ${indexId || 'Not specified'}
• Lens Context: ${includeLensContext ? 'Enabled' : 'Disabled'}`,
          }],
        });
      }

      const analysis = await this.symbolAnalysis.deepAnalyze(target, {
        depth: analysisDepth,
        includeLensContext,
        indexId,
      });

      // Check if analysis was successful
      if (!analysis) {
        return this.createSuccessResponse(requestId, {
          content: [{
            type: 'text',
            text: `Deep Code Analysis for "${target}"

Analysis could not be completed - target not found or analysis failed.`,
          }],
        });
      }

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: `Deep Code Analysis for "${target}"

${analysis.summary || 'Analysis completed'}

Key Findings:
${analysis.findings ? analysis.findings.map((f: any) => `• ${f}`).join('\n') : '• No findings available'}

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
      
      // Validate required parameters
      if (!path || typeof path !== 'string') {
        return this.createErrorResponse(
          requestId,
          -32602,
          'Invalid parameters: path is required and must be a string'
        );
      }
      
      const indexId = this.generateIndexId(path);

      // Check Lens status if enabled
      let lensStatus = null;
      if (this.config.lens.enabled) {
        const response = await this.lensClient.getRepositoryStatus(indexId);
        if (response.success) {
          lensStatus = response.data;
        } else {
          // If Lens status check fails, report the error
          return this.createSuccessResponse(requestId, {
            content: [{
              type: 'text',
              text: `Repository status check failed: Lens service error (${response.error}). Please ensure the Lens service is running.`,
            }],
          });
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
  /**
   * Handle configurable intelligence research
   */
  private async handleConfigurableResearch(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { target, loadout = 'Deep Intelligence Gathering', customConfig, indexId } = args;
      
      this.logger.info('Starting configurable intelligence research', { target, loadout });

      // Get the requested loadout
      const loadoutConfig = this.loadoutManager.getLoadout(loadout);
      if (!loadoutConfig) {
        return this.createErrorResponse(requestId, -32602, `Loadout not found: ${loadout}`);
      }

      // Apply custom configuration overrides if provided
      const finalLoadout = customConfig ? this.applyCustomConfig(loadoutConfig, customConfig) : loadoutConfig;

      // Create pipeline context
      const context: PipelineContext = {
        indexId: indexId || this.generateIndexId(target),
        repoPath: typeof target === 'string' && target.startsWith('/') ? target : process.cwd(),
        repoInfo: {
          root: process.cwd(),
          rev: 'HEAD',
          worktreeDirty: false,
        },
        config: {
          languages: ['ts', 'js', 'py', 'rs', 'go'],
          excludes: ['node_modules/', 'dist/', 'build/', '.git/', '__pycache__/'],
          contextLines: 3,
          maxFilesToEmbed: 1000,
        },
        storageDir: this.config.storageDir,
        cacheDir: this.config.cacheDir,
      };

      // Execute the research
      const result = await this.intelligenceEngine.executeResearch(target, finalLoadout, context);

      if (!result.success) {
        return this.createErrorResponse(requestId, -32603, result.error || 'Research execution failed');
      }

      // Generate swarm-optimized report
      const reportConfig: SwarmReportConfig = {
        densityTarget: finalLoadout.output.densityTarget,
        maxSectionTokens: Math.floor(finalLoadout.output.densityTarget / 8),
        includeDebateTopics: finalLoadout.output.includeDebateTopics,
        includeMetrics: finalLoadout.output.includeMetrics,
        includeCitations: finalLoadout.output.includeCitations,
        optimizeFor: finalLoadout.output.optimizeFor,
        debateStructureDepth: 'medium'
      };

      const report = await this.swarmReporter.generateIntelligenceBrief(result.intelligence!, reportConfig);

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: report
        }],
        metadata: {
          loadout: finalLoadout.name,
          executionTime: result.duration,
          verificationScore: result.verificationScore,
          qualityMetrics: result.metrics?.qualityMetrics
        }
      });

    } catch (error) {
      this.logger.error('Failed to execute configurable research', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Research failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle loadout comparison
   */
  private async handleCompareLoadouts(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { target, loadouts, indexId, comparisonMetrics = ['quality', 'comprehensiveness'] } = args;
      
      this.logger.info('Comparing research loadouts', { target, loadouts: loadouts.length });

      // Validate loadouts exist
      for (const loadoutName of loadouts) {
        if (!this.loadoutManager.hasLoadout(loadoutName)) {
          return this.createErrorResponse(requestId, -32602, `Loadout not found: ${loadoutName}`);
        }
      }

      // Execute research with each loadout (this would be implemented in intelligence engine)
      const results: ConfigurableResearchResult[] = [];
      
      for (const loadoutName of loadouts) {
        const loadout = this.loadoutManager.getLoadout(loadoutName)!;
        const context: PipelineContext = {
          indexId: indexId || this.generateIndexId(target),
          repoPath: typeof target === 'string' && target.startsWith('/') ? target : process.cwd(),
          repoInfo: { root: process.cwd(), rev: 'HEAD', worktreeDirty: false },
          config: {
            languages: ['ts', 'js', 'py'],
            excludes: ['node_modules/', 'dist/', 'build/'],
            contextLines: 3,
            maxFilesToEmbed: 1000,
          },
          storageDir: this.config.storageDir,
          cacheDir: this.config.cacheDir,
        };

        const result = await this.intelligenceEngine.executeResearch(target, loadout, context);
        results.push(result);
      }

      // Generate comparison report
      const comparison = this.generateLoadoutComparison(results, comparisonMetrics);

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: comparison
        }]
      });

    } catch (error) {
      this.logger.error('Failed to compare loadouts', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Comparison failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle list loadouts request
   */
  private async handleListLoadouts(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { category = 'all', includeDetails = false } = args;

      const allLoadouts = this.loadoutManager.getAllLoadoutInfo();
      
      // Filter by category if requested
      let filteredLoadouts = allLoadouts;
      if (category !== 'all') {
        filteredLoadouts = allLoadouts.filter(info => 
          info.filePath.includes(`/${category}/`) || info.filePath.includes(`\\${category}\\`)
        );
      }

      // Generate response
      let response = `# Available Research Loadouts\n\n`;
      
      if (includeDetails) {
        for (const info of filteredLoadouts) {
          const loadout = this.loadoutManager.getLoadout(info.name);
          response += `## ${info.name} (v${info.version})\n`;
          response += `- **Description:** ${loadout?.description || 'No description'}\n`;
          response += `- **Valid:** ${info.isValid ? '✓' : '✗'}\n`;
          response += `- **Focus:** ${loadout?.pipeline.focus || 'Unknown'}\n`;
          response += `- **Complexity:** ${loadout?.performance.cpuIntensive ? 'High' : 'Low'}\n`;
          response += `- **Agents:** ${loadout?.agents.filter(a => a.enabled).length || 0}\n`;
          response += `- **Stages:** ${loadout?.stages.filter(s => s.enabled).length || 0}\n\n`;
        }
      } else {
        response += filteredLoadouts.map(info => 
          `- **${info.name}** (v${info.version}): ${this.loadoutManager.getLoadout(info.name)?.description || 'No description'}`
        ).join('\n');
      }

      response += `\n\n**Total loadouts:** ${filteredLoadouts.length}`;
      response += `\n**Categories:** core, specialized, experimental`;

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: response
        }]
      });

    } catch (error) {
      this.logger.error('Failed to list loadouts', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `List failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle intelligence to swarm handoff
   */
  private async handleIntelligenceSwarmHandoff(requestId: string | number, args: any): Promise<MCPResponse> {
    try {
      const { 
        target, 
        intelligenceLoadout = 'Deep Intelligence Gathering', 
        swarmOptimization = {},
        indexId 
      } = args;
      
      this.logger.info('Executing intelligence to swarm handoff', { target, intelligenceLoadout });

      // Get the intelligence loadout
      const loadout = this.loadoutManager.getLoadout(intelligenceLoadout);
      if (!loadout) {
        return this.createErrorResponse(requestId, -32602, `Intelligence loadout not found: ${intelligenceLoadout}`);
      }

      // Execute intelligence gathering
      const context: PipelineContext = {
        indexId: indexId || this.generateIndexId(target),
        repoPath: typeof target === 'string' && target.startsWith('/') ? target : process.cwd(),
        repoInfo: { root: process.cwd(), rev: 'HEAD', worktreeDirty: false },
        config: {
          languages: ['ts', 'js', 'py', 'rs', 'go'],
          excludes: ['node_modules/', 'dist/', 'build/', '.git/'],
          contextLines: 3,
          maxFilesToEmbed: 1000,
        },
        storageDir: this.config.storageDir,
        cacheDir: this.config.cacheDir,
      };

      const researchResult = await this.intelligenceEngine.executeResearch(target, loadout, context);

      if (!researchResult.success) {
        return this.createErrorResponse(requestId, -32603, researchResult.error || 'Intelligence gathering failed');
      }

      // Generate swarm-optimized report
      const reportConfig: SwarmReportConfig = {
        densityTarget: swarmOptimization.densityTarget || 5000,
        maxSectionTokens: Math.floor((swarmOptimization.densityTarget || 5000) / 8),
        includeDebateTopics: swarmOptimization.includeDebateTopics !== false,
        includeMetrics: true,
        includeCitations: true,
        optimizeFor: 'swarm_consumption',
        debateStructureDepth: swarmOptimization.debateStructure || 'medium'
      };

      const intelligenceBrief = await this.swarmReporter.generateIntelligenceBrief(
        researchResult.intelligence!,
        reportConfig
      );

      const response = `# INTELLIGENCE BRIEF FOR REFINEMENT SWARM

${intelligenceBrief}

---

## HANDOFF INSTRUCTIONS FOR REFINEMENT SWARM

This intelligence brief contains comprehensive analysis optimized for debate-style refinement. 

**Next Steps:**
1. Review the debate framework and prioritized topics
2. Validate all hard constraints before solution generation
3. Focus on trade-off analysis rather than absolute solutions
4. Use the provided evidence base to support positions
5. Apply the risk mitigation strategies to any proposed solutions

**Debate Quality Targets:**
- Convergence: ${loadout.pipeline.convergenceThreshold * 100}% agreement on final approach
- Evidence threshold: Each position must cite supporting evidence
- Time allocation: See individual debate topic estimates

The refinement swarm should now begin structured debate using this intelligence foundation.`;

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: response
        }],
        metadata: {
          intelligenceLoadout: loadout.name,
          executionTime: researchResult.duration,
          verificationScore: researchResult.verificationScore,
          debateTopicsCount: researchResult.intelligence?.debateFramework.length || 0,
          swarmOptimized: true
        }
      });

    } catch (error) {
      this.logger.error('Failed to execute intelligence to swarm handoff', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Handoff failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle loadout metrics analysis request
   */
  private async handleLoadoutMetricsAnalysis(args: any, requestId: string): Promise<MCPResponse> {
    try {
      const { loadouts = [], includeComparison = false, generateReport = false } = args;
      
      // Get metrics for specified loadouts or all loadouts
      const targetLoadouts = loadouts.length > 0 ? loadouts : Array.from(this.metricsSystem.getAllMetrics().keys());
      const metricsData = new Map();
      
      for (const loadoutName of targetLoadouts) {
        const metrics = this.metricsSystem.getMetrics(loadoutName);
        if (metrics) {
          metricsData.set(loadoutName, metrics);
        }
      }

      if (metricsData.size === 0) {
        return this.createErrorResponse(requestId, -32600, 'No valid loadouts found for metrics analysis');
      }

      let response = '';
      
      if (generateReport) {
        response = this.metricsSystem.generateReport(Array.from(metricsData.keys()));
      } else {
        response = this.formatMetricsResponse(metricsData, includeComparison);
      }

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: response
        }],
        metadata: {
          loadoutsAnalyzed: metricsData.size,
          includeComparison,
          generateReport
        }
      });

    } catch (error) {
      this.logger.error('Failed to analyze loadout metrics', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Metrics analysis failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle loadout performance benchmarking
   */
  private async handleBenchmarkLoadoutPerformance(args: any, requestId: string): Promise<MCPResponse> {
    try {
      const { loadout, testCase, target, indexId } = args;
      
      if (!this.loadoutManager.hasLoadout(loadout)) {
        return this.createErrorResponse(requestId, -32600, `Loadout '${loadout}' not found`);
      }

      this.logger.info('Starting loadout benchmark', { loadout, testCase, target });

      // Execute the research with timing
      const startTime = Date.now();
      const researchResult = await this.intelligenceEngine.executeResearch(
        target,
        this.loadoutManager.getLoadout(loadout)!,
        { indexId }
      );
      const endTime = Date.now();

      // Calculate quality metrics based on result
      const qualityScore = this.calculateQualityScore(researchResult);
      const completenessScore = this.calculateCompletenessScore(researchResult);

      // Record benchmark
      const benchmark = {
        testCase,
        metrics: {
          actualRuntime: endTime - startTime,
          tokenUsage: researchResult.intelligence?.metadata?.totalTokens || 0,
          qualityScore,
          completenessScore
        },
        timestamp: new Date()
      };

      this.metricsSystem.recordBenchmark(loadout, benchmark);

      const response = `# LOADOUT PERFORMANCE BENCHMARK

**Loadout:** ${loadout}
**Test Case:** ${testCase}
**Target:** ${target}
**Execution Time:** ${endTime - startTime}ms

## Performance Metrics

- **Runtime:** ${(endTime - startTime) / 1000} seconds
- **Token Usage:** ~${Math.round((researchResult.intelligence?.metadata?.totalTokens || 0) / 1000)}k tokens
- **Quality Score:** ${qualityScore.toFixed(1)}/100
- **Completeness:** ${completenessScore.toFixed(1)}/100
- **Success:** ${researchResult.success ? '✅' : '❌'}

## Results Summary

${researchResult.success ? 
  'Benchmark completed successfully. Results have been recorded for comparison with other loadouts.' :
  'Benchmark failed. Check loadout configuration and retry.'
}

${researchResult.intelligence?.executiveSummary ? `

## Executive Summary
${researchResult.intelligence.executiveSummary}

` : ''}`;

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: response
        }],
        metadata: {
          loadout,
          testCase,
          benchmark,
          success: researchResult.success
        }
      });

    } catch (error) {
      this.logger.error('Failed to benchmark loadout performance', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Benchmarking failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Handle optimal loadout recommendation
   */
  private async handleRecommendOptimalLoadout(args: any, requestId: string): Promise<MCPResponse> {
    try {
      const { requirements, maxRecommendations = 3 } = args;
      
      const allMetrics = this.metricsSystem.getAllMetrics();
      const recommendations = this.generateLoadoutRecommendations(allMetrics, requirements, maxRecommendations);

      if (recommendations.length === 0) {
        return this.createErrorResponse(requestId, -32600, 'No loadouts match the specified requirements');
      }

      let response = `# OPTIMAL LOADOUT RECOMMENDATIONS

Based on your requirements:
${Object.entries(requirements).map(([key, value]) => 
  `- **${key}:** ${Array.isArray(value) ? value.join(', ') : value}`
).join('\n')}

## Recommended Loadouts

`;

      recommendations.forEach((rec, index) => {
        const metrics = allMetrics.get(rec.loadout);
        response += `### ${index + 1}. ${rec.loadout}

**Confidence:** ${rec.confidenceScore}%
**Reason:** ${rec.reason}

**Key Metrics:**
- Performance: ${metrics?.performance.expectedSpeed || 'unknown'}
- Quality Score: ${Math.round(metrics?.quality.thoroughnessScore || 0)}/100
- Efficiency: ${Math.round(metrics?.efficiency.costBenefitScore || 0)}/100
- Specialization: ${metrics?.specialization.expertiseLevel || 'unknown'}

${rec.tradeoffs.length > 0 ? `**Trade-offs:**
${rec.tradeoffs.map(t => `- ${t}`).join('\n')}

` : ''}`;
      });

      return this.createSuccessResponse(requestId, {
        content: [{
          type: 'text',
          text: response
        }],
        metadata: {
          requirements,
          recommendationsCount: recommendations.length,
          recommendations: recommendations.map(r => ({
            loadout: r.loadout,
            confidence: r.confidenceScore,
            category: r.category
          }))
        }
      });

    } catch (error) {
      this.logger.error('Failed to recommend optimal loadout', { error, args });
      return this.createErrorResponse(
        requestId,
        -32603,
        `Recommendation failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Helper methods
   */
  private applyCustomConfig(loadout: ResearchLoadout, customConfig: any): ResearchLoadout {
    const modifiedLoadout = JSON.parse(JSON.stringify(loadout)); // Deep clone
    
    if (customConfig.densityTarget) {
      modifiedLoadout.output.densityTarget = customConfig.densityTarget;
    }
    if (customConfig.maxIterations) {
      modifiedLoadout.pipeline.maxIterations = customConfig.maxIterations;
    }
    if (customConfig.parallelismLevel) {
      modifiedLoadout.pipeline.parallelismLevel = customConfig.parallelismLevel;
    }
    
    return modifiedLoadout;
  }

  private generateLoadoutComparison(results: ConfigurableResearchResult[], metrics: string[]): string {
    let comparison = `# RESEARCH LOADOUT COMPARISON\n\n`;
    
    comparison += `**Target:** ${results[0]?.target || 'Unknown'}\n`;
    comparison += `**Loadouts Compared:** ${results.length}\n`;
    comparison += `**Comparison Metrics:** ${metrics.join(', ')}\n\n`;

    // Performance comparison
    comparison += `## Performance Comparison\n\n`;
    comparison += `| Loadout | Duration (ms) | Verification Score | Success |\n`;
    comparison += `|---------|---------------|-------------------|----------|\n`;
    
    for (const result of results) {
      comparison += `| ${result.loadoutName} | ${result.duration} | ${(result.verificationScore || 0).toFixed(2)} | ${result.success ? '✓' : '✗'} |\n`;
    }

    // Quality comparison
    if (metrics.includes('quality')) {
      comparison += `\n## Quality Assessment\n\n`;
      const successfulResults = results.filter(r => r.success);
      
      if (successfulResults.length > 0) {
        const best = successfulResults.reduce((best, current) => 
          (current.verificationScore || 0) > (best.verificationScore || 0) ? current : best
        );
        comparison += `**Highest Quality:** ${best.loadoutName} (${(best.verificationScore || 0).toFixed(2)} verification score)\n`;
      }
    }

    // Speed comparison
    if (metrics.includes('speed')) {
      comparison += `\n## Speed Assessment\n\n`;
      const fastest = results.reduce((fastest, current) => 
        current.duration < fastest.duration ? current : fastest
      );
      comparison += `**Fastest:** ${fastest.loadoutName} (${fastest.duration}ms)\n`;
    }

    // Recommendations
    comparison += `\n## Recommendations\n\n`;
    const successfulResults = results.filter(r => r.success);
    
    if (successfulResults.length > 1) {
      const balanced = successfulResults.reduce((best, current) => {
        const bestScore = (best.verificationScore || 0) / (best.duration / 1000);
        const currentScore = (current.verificationScore || 0) / (current.duration / 1000);
        return currentScore > bestScore ? current : best;
      });
      
      comparison += `- **For balanced performance:** ${balanced.loadoutName}\n`;
    }
    
    if (successfulResults.length > 0) {
      const highest = successfulResults.reduce((best, current) => 
        (current.verificationScore || 0) > (best.verificationScore || 0) ? current : best
      );
      comparison += `- **For highest quality:** ${highest.loadoutName}\n`;
      
      const fastest = successfulResults.reduce((fastest, current) => 
        current.duration < fastest.duration ? current : fastest
      );
      comparison += `- **For speed:** ${fastest.loadoutName}\n`;
    }

    return comparison;
  }

  private generateIndexId(path: string): string {
    // Simple hash-based ID generation
    return Buffer.from(path).toString('base64').replace(/[/+=]/g, '').slice(0, 16);
  }

  private createSuccessResponse(id: string | number | null | undefined, result: any): MCPResponse {
    return {
      jsonrpc: '2.0',
      id: id ?? null,
      result,
    };
  }

  private createErrorResponse(id: string | number | null | undefined, code: number, message: string): MCPResponse {
    return {
      jsonrpc: '2.0',
      id: id ?? null,
      error: { code, message },
    };
  }

  private formatMetricsResponse(metricsData: Map<string, any>, includeComparison: boolean): string {
    let response = '# LOADOUT METRICS ANALYSIS\n\n';
    
    response += '## Summary\n\n';
    response += '| Loadout | Benchmark Score | Quality | Performance | Efficiency |\n';
    response += '|---------|----------------|---------|-------------|------------|\n';
    
    metricsData.forEach((metrics, name) => {
      response += `| ${name} | ${Math.round(metrics.benchmarkScore)} | ${Math.round(metrics.quality.thoroughnessScore)} | ${metrics.performance.expectedSpeed} | ${Math.round(metrics.efficiency.costBenefitScore)} |\n`;
    });

    if (includeComparison && metricsData.size > 1) {
      response += '\n## Comparative Analysis\n\n';
      const comparison = this.metricsSystem.compareLoadouts(Array.from(metricsData.keys()));
      
      response += '**Best by Category:**\n';
      Object.entries(comparison.winnerByCategory).forEach(([category, winner]) => {
        response += `- **${category}**: ${winner}\n`;
      });

      response += '\n**Recommendations:**\n';
      comparison.recommendations.slice(0, 3).forEach(rec => {
        response += `- **${rec.category}**: ${rec.loadout} (${rec.confidenceScore}% confidence) - ${rec.reason}\n`;
      });
    }

    return response;
  }

  private calculateQualityScore(result: any): number {
    // Basic quality scoring based on result completeness
    let score = 0;
    
    if (result.success) score += 40;
    if (result.intelligence?.executiveSummary) score += 20;
    if (result.intelligence?.keyFindings && result.intelligence.keyFindings.length > 0) score += 20;
    if (result.intelligence?.recommendations && result.intelligence.recommendations.length > 0) score += 10;
    if (result.verificationScore) score += Math.min(10, result.verificationScore * 10);
    
    return Math.min(100, score);
  }

  private calculateCompletenessScore(result: any): number {
    // Score based on completeness of sections and content
    let score = 0;
    
    if (result.intelligence?.executiveSummary) score += 25;
    if (result.intelligence?.keyFindings && result.intelligence.keyFindings.length >= 3) score += 25;
    if (result.intelligence?.recommendations && result.intelligence.recommendations.length >= 2) score += 20;
    if (result.intelligence?.debateFramework && result.intelligence.debateFramework.length > 0) score += 15;
    if (result.intelligence?.metadata?.sectionsGenerated >= 5) score += 15;
    
    return Math.min(100, score);
  }

  private generateLoadoutRecommendations(allMetrics: Map<string, any>, requirements: any, maxRecommendations: number): any[] {
    const loadouts = Array.from(allMetrics.entries());
    const recommendations: any[] = [];

    // Score each loadout based on requirements
    const scored = loadouts.map(([name, metrics]) => {
      let score = 0;
      let reasons: string[] = [];

      // Speed requirement
      if (requirements.speed) {
        const speedMap = { 'very-fast': 5, 'fast': 4, 'moderate': 3, 'slow': 2, 'very-slow': 1 };
        const requiredScore = speedMap[requirements.speed as keyof typeof speedMap] || 3;
        const actualScore = speedMap[metrics.performance.expectedSpeed as keyof typeof speedMap] || 3;
        
        if (actualScore >= requiredScore) {
          score += 20;
          reasons.push(`Meets speed requirement (${metrics.performance.expectedSpeed})`);
        } else {
          score -= 10;
        }
      }

      // Quality requirement
      if (requirements.quality) {
        const qualityMap = { 'basic': 50, 'good': 70, 'high': 85, 'excellent': 95 };
        const required = qualityMap[requirements.quality as keyof typeof qualityMap] || 70;
        
        if (metrics.quality.thoroughnessScore >= required) {
          score += 25;
          reasons.push(`Exceeds quality requirement (${Math.round(metrics.quality.thoroughnessScore)}/100)`);
        } else {
          score -= 15;
        }
      }

      // Domain expertise
      if (requirements.domain && requirements.domain.length > 0) {
        const matchingDomains = requirements.domain.filter((d: string) => 
          metrics.specialization.focusAreas.some((area: string) => 
            area.toLowerCase().includes(d.toLowerCase()) || d.toLowerCase().includes(area.toLowerCase())
          )
        );
        
        if (matchingDomains.length > 0) {
          score += 20 * (matchingDomains.length / requirements.domain.length);
          reasons.push(`Domain expertise in ${matchingDomains.join(', ')}`);
        }
      }

      // Resource budget
      if (requirements.resourceBudget) {
        const budgetMap = { 'low': 1, 'moderate': 2, 'high': 3, 'unlimited': 4 };
        const budget = budgetMap[requirements.resourceBudget as keyof typeof budgetMap] || 2;
        const intensity = { 'low': 1, 'moderate': 2, 'high': 3, 'very-high': 4 };
        const loadoutIntensity = intensity[metrics.resourceProfile.computeIntensity.level as keyof typeof intensity] || 2;
        
        if (loadoutIntensity <= budget) {
          score += 15;
          reasons.push(`Fits resource budget (${metrics.resourceProfile.computeIntensity.level})`);
        } else {
          score -= 10;
        }
      }

      // Baseline score from metrics
      score += metrics.benchmarkScore * 0.3;

      return {
        name,
        score,
        reasons,
        metrics,
        tradeoffs: metrics.efficiency.optimizationSuggestions.slice(0, 2)
      };
    });

    // Sort by score and take top recommendations
    scored.sort((a, b) => b.score - a.score);
    
    return scored.slice(0, maxRecommendations).map((item, index) => ({
      loadout: item.name,
      category: index === 0 ? 'best-match' : 'alternative',
      reason: item.reasons.join(', '),
      confidenceScore: Math.min(95, Math.max(60, Math.round(item.score))),
      tradeoffs: item.tradeoffs
    }));
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