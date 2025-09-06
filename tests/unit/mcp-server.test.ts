/**
 * Unit tests for MCP server implementation.
 * 
 * Tests MCP protocol handling, tool implementations, request/response validation,
 * and integration with Lens client.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn, mock } from 'bun:test';
import { MimirMCPServer } from '@/mcp/server';
import { LensClient } from '@/pipeline/lens-client';
import { FileDiscovery } from '@/pipeline/discovery';
import { SymbolAnalysis } from '@/pipeline/symbols';
import type { MCPRequest, MCPResponse, LensResponse } from '@/types';
import { createLogger } from '@/utils/logger';

// Mock the config module to enable Lens but allow environment variable overrides
mock.module('@/config/config', () => {
  const originalModule = require('/media/nathan/Seagate Hub/Projects/mimir/src/config/config.ts');
  
  return {
    ...originalModule,
    loadConfig: () => {
      // Start with default values
      const baseConfig = {
        lens: {
          enabled: process.env.LENS_ENABLED === 'false' ? false : true,
          baseUrl: process.env.LENS_BASE_URL || 'http://localhost:8080',
          apiKey: process.env.LENS_API_KEY,
          enableIndexing: true,
          enableSearch: true,
          enableEmbeddings: true,
          maxRetries: 3,
          retryDelay: 1,
          timeout: process.env.LENS_TIMEOUT ? parseInt(process.env.LENS_TIMEOUT) : 30,
          fallbackEnabled: false,
          healthCheckEnabled: process.env.LENS_HEALTH_CHECK_ENABLED === 'true',
          healthCheckTimeout: 10,
        },
        server: {
          host: process.env.MIMIR_UI_HOST || '127.0.0.1',
          port: process.env.MIMIR_UI_PORT ? parseInt(process.env.MIMIR_UI_PORT) : 8000,
          maxWorkers: process.env.MIMIR_MAX_WORKERS ? parseInt(process.env.MIMIR_MAX_WORKERS) : 4,
          behindProxy: process.env.MIMIR_UI_BEHIND_PROXY === 'true',
          corsOrigins: process.env.MIMIR_CORS_ORIGINS ? process.env.MIMIR_CORS_ORIGINS.split(',') : []
        },
        monitoring: {
          enableMetrics: false,
          metricsPort: 9090
        },
        ai: {
          enableGemini: false,
          googleApiKey: process.env.GOOGLE_API_KEY,
          geminiApiKey: process.env.GEMINI_API_KEY
        },
        pipeline: {
          treeSitterLanguages: process.env.TREE_SITTER_LANGUAGES ? process.env.TREE_SITTER_LANGUAGES.split(',') : ['typescript', 'javascript', 'python'],
          enableAcquire: process.env.PIPELINE_ENABLE_ACQUIRE === 'false' ? false : true
        },
        performance: {
          asyncioMaxWorkers: 10
        },
        storage: {
          dataPath: './tmp/storage',
          cachePath: './tmp/cache'
        },
        logLevel: 'error',
        storageDir: './tmp/storage',
        cacheDir: './tmp/cache'
      };
      
      return baseConfig;
    }
  };
});

describe('Mimir MCP Server', () => {
  let mcpServer: MimirMCPServer;
  let lensClientSpy: any;
  let fileDiscoverySpy: any;
  let symbolAnalysisSpy: any;
  let consoleLogSpy: any;

  beforeEach(async () => {
    // Ensure Lens is enabled for MCP server tests
    delete process.env.LENS_ENABLED; // Remove any existing setting to use default (true)
    
    // Mock console to suppress log output
    consoleLogSpy = spyOn(console, 'log').mockImplementation(() => {});
    spyOn(console, 'error').mockImplementation(() => {});
    spyOn(console, 'warn').mockImplementation(() => {});
    spyOn(console, 'info').mockImplementation(() => {});

    // Create MCP server instance
    mcpServer = new MimirMCPServer();
    await mcpServer.initialize();

    // Mock Lens client methods
    lensClientSpy = spyOn(LensClient.prototype, 'indexRepository').mockResolvedValue({
      success: true,
      data: { indexed: true, collectionId: 'test-collection' },
      statusCode: 200,
      responseTimeMs: 150
    });

    spyOn(LensClient.prototype, 'searchRepository').mockResolvedValue({
      success: true,
      data: {
        results: [
          { path: '/test/file1.ts', score: 0.9, content: 'test content 1' },
          { path: '/test/file2.ts', score: 0.8, content: 'test content 2' }
        ],
        total: 2,
        queryTimeMs: 100
      },
      statusCode: 200,
      responseTimeMs: 120
    });

    spyOn(LensClient.prototype, 'getRepositoryStatus').mockResolvedValue({
      success: true,
      data: {
        indexed: true,
        lastUpdated: '2024-01-01T00:00:00Z',
        filesCount: 100,
        symbolsCount: 500
      },
      statusCode: 200,
      responseTimeMs: 50
    });

    spyOn(LensClient.prototype, 'initialize').mockResolvedValue(undefined);
    spyOn(LensClient.prototype, 'cleanup').mockResolvedValue(undefined);

    // Mock FileDiscovery
    fileDiscoverySpy = spyOn(FileDiscovery.prototype, 'discover').mockResolvedValue({
      files: ['/test/file1.ts', '/test/file2.ts'],
      totalSize: 2048,
      duration: 50,
      gitInfo: {
        branch: 'main',
        commit: 'abc123',
        isDirty: false
      }
    });

    // Mock SymbolAnalysis
    symbolAnalysisSpy = spyOn(SymbolAnalysis.prototype, 'analyze').mockResolvedValue({
      symbols: [
        { name: 'TestClass', type: 'class', path: '/test/file1.ts', line: 10 },
        { name: 'testFunction', type: 'function', path: '/test/file2.ts', line: 5 }
      ],
      symbolCount: 2,
      duration: 75
    });

    spyOn(SymbolAnalysis.prototype, 'deepAnalyze').mockResolvedValue({
      target: '/test/file1.ts',
      analysis: 'Deep analysis result',
      symbols: [
        { name: 'TestClass', type: 'class', path: '/test/file1.ts', line: 10 }
      ],
      dependencies: ['/test/file2.ts'],
      exports: ['TestClass'],
      imports: ['lodash'],
      complexity: 5,
      linesOfCode: 50,
      documentation: 'Class documentation',
      suggestions: ['Consider adding error handling']
    });
  });

  afterEach(async () => {
    await mcpServer.cleanup();
    consoleLogSpy.mockRestore();
  });

  describe('MCP Protocol', () => {
    it('should handle list_tools request', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'test-1',
        method: 'tools/list'
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('test-1');
      expect(response.result).toBeDefined();
      expect(response.result.tools).toBeDefined();
      expect(response.result.tools).toHaveLength(11);
      
      const toolNames = response.result.tools.map((tool: any) => tool.name);
      expect(toolNames).toContain('ensure_repo_index');
      expect(toolNames).toContain('hybrid_search');
      expect(toolNames).toContain('deep_code_analysis');
      expect(toolNames).toContain('get_repository_status');
      expect(toolNames).toContain('configurable_intelligence_research');
      expect(toolNames).toContain('compare_research_loadouts');
      expect(toolNames).toContain('list_research_loadouts');
      expect(toolNames).toContain('loadout_metrics_analysis');
      expect(toolNames).toContain('benchmark_loadout_performance');
      expect(toolNames).toContain('recommend_optimal_loadout');
      expect(toolNames).toContain('intelligence_to_swarm_handoff');
    });

    it('should handle unsupported method', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'test-2',
        method: 'unsupported/method'
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('test-2');
      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32601);
      expect(response.error?.message).toBe('Method not found: unsupported/method');
    });

    it('should handle invalid JSON-RPC format', async () => {
      const request = {
        id: 'test-3',
        method: 'tools/list'
        // Missing jsonrpc field
      } as MCPRequest;

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('test-3');
      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32600);
      expect(response.error?.message).toBe('Invalid Request');
    });

    it('should handle request without ID', async () => {
      const request = {
        jsonrpc: '2.0',
        method: 'tools/list'
        // Missing id field
      } as MCPRequest;

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBeNull();
    });
  });

  describe('Tool: ensure_repo_index', () => {
    it('should index repository successfully', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'index-1',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index',
          arguments: {
            path: '/test/repo',
            forceReindex: false,
            useLens: true
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('index-1');
      expect(response.result).toBeDefined();
      expect(response.result.content[0].type).toBe('text');
      
      const content = response.result.content[0].text;
      expect(content).toContain('Repository indexed successfully');
      expect(content).toContain('Index ID:');
      
      expect(lensClientSpy).toHaveBeenCalledWith(expect.objectContaining({
        repositoryPath: '/test/repo',
        forceReindex: false,
        includeEmbeddings: true
      }));
    });

    it('should handle indexing with force reindex', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'index-2',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index',
          arguments: {
            path: '/test/repo',
            forceReindex: true,
            useLens: true
          }
        }
      };

      await mcpServer.handleRequest(request);

      expect(lensClientSpy).toHaveBeenCalledWith(expect.objectContaining({
        repositoryPath: '/test/repo',
        forceReindex: true,
        includeEmbeddings: true
      }));
    });

    it('should handle missing path argument', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'index-3',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index',
          arguments: {
            forceReindex: false,
            useLens: true
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.message).toMatch(/path|required|string/);
    });

    it('should handle Lens client errors', async () => {
      lensClientSpy.mockResolvedValueOnce({
        success: false,
        error: 'Lens service unavailable',
        statusCode: 503,
        responseTimeMs: 100
      });

      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'index-4',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index',
          arguments: {
            path: '/test/repo',
            forceReindex: false,
            useLens: true
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.result).toBeDefined();
      const content = response.result.content[0].text;
      expect(content).toMatch(/failed|error/);
      expect(content).toContain('service') || expect(content).toContain('unavailable');
    });
  });

  describe('Tool: hybrid_search', () => {
    it('should search repositories successfully', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'search-1',
        method: 'tools/call',
        params: {
          name: 'hybrid_search',
          arguments: {
            query: 'test query',
            indexId: 'test-repo',
            maxResults: 10
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('search-1');
      expect(response.result).toBeDefined();
      expect(response.result.content[0].type).toBe('text');
      
      const content = response.result.content[0].text;
      expect(content).toContain('Found');
      expect(content).toContain('results');
    });

    it('should handle search without index ID', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'search-2',
        method: 'tools/call',
        params: {
          name: 'hybrid_search',
          arguments: {
            query: 'global search',
            maxResults: 5
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.result).toBeDefined();
      const content = response.result.content[0].text;
      expect(content).toContain('Found') && expect(content).toContain('results');
    });

    it('should handle missing query parameter', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'search-3',
        method: 'tools/call',
        params: {
          name: 'hybrid_search',
          arguments: {
            maxResults: 10
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.message).toMatch(/query|required|search/);
    });

    it('should handle search errors', async () => {
      spyOn(LensClient.prototype, 'searchRepository').mockResolvedValueOnce({
        success: false,
        error: 'Search service down',
        statusCode: 500,
        responseTimeMs: 200
      });

      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'search-4',
        method: 'tools/call',
        params: {
          name: 'hybrid_search',
          arguments: {
            query: 'test query',
            maxResults: 10
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.result).toBeDefined();
      const content = response.result.content[0].text;
      expect(content).toMatch(/failed|error/);
      expect(content).toContain('Search service down');
    });
  });

  describe('Tool: deep_code_analysis', () => {
    it('should perform deep analysis successfully', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'analysis-1',
        method: 'tools/call',
        params: {
          name: 'deep_code_analysis',
          arguments: {
            target: '/test/file1.ts',
            indexId: 'test-repo',
            analysisDepth: 2,
            includeLensContext: true
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('analysis-1');
      expect(response.result).toBeDefined();
      expect(response.result.content[0].type).toBe('text');
      
      const content = response.result.content[0].text;
      expect(content).toMatch(/Analysis completed|successfully/);
      expect(content).toContain('test/file1.ts');
      expect(content).toContain('Dependencies');
      expect(content).toContain('Complexity Score');
      
      // Should have called deepAnalyze with the right parameters
      const deepAnalyzeSpy = SymbolAnalysis.prototype.deepAnalyze as any;
      expect(deepAnalyzeSpy).toHaveBeenCalledWith(
        '/test/file1.ts',
        expect.objectContaining({
          depth: 2,
          includeLensContext: true
        })
      );
    });

    it('should handle analysis with default parameters', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'analysis-2',
        method: 'tools/call',
        params: {
          name: 'deep_code_analysis',
          arguments: {
            target: '/test/file2.ts'
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.result).toBeDefined();
      const content = response.result.content[0].text;
      expect(content).toMatch(/Analysis completed|successfully/);
      expect(content).toContain('test/file2.ts');
    });

    it('should handle missing target parameter', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'analysis-3',
        method: 'tools/call',
        params: {
          name: 'deep_code_analysis',
          arguments: {
            analysisDepth: 3
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.message).toMatch(/target|required|analysis/);
    });

    it('should handle analysis errors', async () => {
      // Mock the deepAnalyze method to return null (analysis failed)
      spyOn(SymbolAnalysis.prototype, 'deepAnalyze').mockResolvedValueOnce(null);

      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'analysis-4',
        method: 'tools/call',
        params: {
          name: 'deep_code_analysis',
          arguments: {
            target: '/nonexistent/file.ts'
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.result).toBeDefined();
      const content = response.result.content[0].text;
      expect(content).toMatch(/failed|error|could not be completed/);
      expect(content).toContain('not found') || content.includes('failed');
    });
  });

  describe('Tool: get_repository_status', () => {
    it('should get repository status successfully', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'status-1',
        method: 'tools/call',
        params: {
          name: 'get_repository_status',
          arguments: {
            path: '/test/repo'
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBe('status-1');
      expect(response.result).toBeDefined();
      expect(response.result.content[0].type).toBe('text');
      
      const content = response.result.content[0].text;
      expect(content).toMatch(/Repository Status|indexed|status/);
      // The content is a text string, not an object with properties
      expect(content).toContain('2024-01-01T00:00:00Z');
      expect(content).toContain('100');
      expect(content).toContain('500');
    });

    it('should handle missing path parameter', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'status-2',
        method: 'tools/call',
        params: {
          name: 'get_repository_status',
          arguments: {}
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.message).toMatch(/path|required|string/);
    });

    it('should handle status check errors', async () => {
      spyOn(LensClient.prototype, 'getRepositoryStatus').mockResolvedValueOnce({
        success: false,
        error: 'Repository not found',
        statusCode: 404,
        responseTimeMs: 50
      });

      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'status-3',
        method: 'tools/call',
        params: {
          name: 'get_repository_status',
          arguments: {
            path: '/nonexistent/repo'
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.result).toBeDefined();
      const content = response.result.content[0].text;
      expect(content).toMatch(/failed|error/);
      expect(content).toContain('Repository not found') || expect(content).toContain('not found');
    });
  });

  describe('Tool Parameter Validation', () => {
    it('should handle invalid tool name', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'tool-1',
        method: 'tools/call',
        params: {
          name: 'nonexistent_tool',
          arguments: {}
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32601);
      expect(response.error?.message).toBe('Tool not found: nonexistent_tool');
    });

    it('should handle missing tool name', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'tool-2',
        method: 'tools/call',
        params: {
          arguments: {}
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32601);
      expect(response.error?.message).toContain('Tool not found');
    });

    it('should handle missing arguments', async () => {
      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'tool-3',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index'
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32602); // Invalid params, not internal error
    });
  });

  describe('Error Handling', () => {
    it('should handle internal server errors', async () => {
      // Force an error by mocking a method to throw
      spyOn(LensClient.prototype, 'indexRepository').mockRejectedValueOnce(new Error('Unexpected error'));

      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'error-1',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index',
          arguments: {
            path: '/test/repo'
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32603);
      expect(response.error?.message).toMatch(/Failed to index repository|Unexpected error/);
    });

    it('should handle malformed requests', async () => {
      const malformedRequest = null as any;

      const response = await mcpServer.handleRequest(malformedRequest);

      expect(response.jsonrpc).toBe('2.0');
      expect(response.id).toBeNull();
      expect(response.error).toBeDefined();
      expect(response.error?.code).toBe(-32600);
      expect(response.error?.message).toBe('Invalid Request');
    });

    it('should handle JSON parsing errors in responses', async () => {
      // Mock a response that can't be serialized
      const circularRef: any = { a: 1 };
      circularRef.self = circularRef;

      spyOn(LensClient.prototype, 'indexRepository').mockResolvedValueOnce({
        success: true,
        data: circularRef,
        statusCode: 200,
        responseTimeMs: 100
      });

      const request: MCPRequest = {
        jsonrpc: '2.0',
        id: 'json-error-1',
        method: 'tools/call',
        params: {
          name: 'ensure_repo_index',
          arguments: {
            path: '/test/repo'
          }
        }
      };

      const response = await mcpServer.handleRequest(request);

      // Should handle the serialization error gracefully
      expect(response.error || response.result).toBeDefined();
    });
  });

  describe('Server Lifecycle', () => {
    it('should initialize successfully', async () => {
      const newServer = new MimirMCPServer();
      // Should not throw
      await newServer.initialize();
      await newServer.cleanup();
      expect(true).toBe(true); // Test passes if we reach here
    });

    it('should cleanup successfully', async () => {
      const newServer = new MimirMCPServer();
      await newServer.initialize();
      // Should not throw
      await newServer.cleanup();
      expect(true).toBe(true); // Test passes if we reach here
    });

    it('should handle multiple cleanup calls', async () => {
      const newServer = new MimirMCPServer();
      await newServer.initialize();
      await newServer.cleanup();
      // Should not throw on second cleanup
      await newServer.cleanup();
      expect(true).toBe(true); // Test passes if we reach here
    });
  });
});