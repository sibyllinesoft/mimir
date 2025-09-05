/**
 * Unit tests for MCP server implementation.
 * 
 * Tests MCP protocol handling, tool implementations, request/response validation,
 * and integration with Lens client.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn } from 'bun:test';
import { MimirMCPServer } from '@/mcp/server';
import { LensClient } from '@/pipeline/lens-client';
import { FileDiscovery } from '@/pipeline/discovery';
import { SymbolAnalysis } from '@/pipeline/symbols';
import type { MCPRequest, MCPResponse, LensResponse } from '@/types';
import { createLogger } from '@/utils/logger';

describe('Mimir MCP Server', () => {
  let mcpServer: MimirMCPServer;
  let lensClientSpy: any;
  let fileDiscoverySpy: any;
  let symbolAnalysisSpy: any;
  let consoleLogSpy: any;

  beforeEach(async () => {
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
      expect(response.result.tools).toHaveLength(4);
      
      const toolNames = response.result.tools.map((tool: any) => tool.name);
      expect(toolNames).toContain('ensure_repo_index');
      expect(toolNames).toContain('hybrid_search');
      expect(toolNames).toContain('deep_code_analysis');
      expect(toolNames).toContain('get_repository_status');
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
      expect(response.error?.message).toBe('Method not found');
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
      
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(true);
      expect(content.indexed).toBe(true);
      expect(content.collectionId).toBe('test-collection');
      
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
      expect(response.error?.message).toContain('Missing required parameter: path');
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
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(false);
      expect(content.error).toBe('Lens service unavailable');
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
      
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(true);
      expect(content.results).toHaveLength(2);
      expect(content.total).toBe(2);
      expect(content.queryTimeMs).toBe(100);
      
      expect(content.results[0].path).toBe('/test/file1.ts');
      expect(content.results[0].score).toBe(0.9);
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
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(true);
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
      expect(response.error?.message).toContain('Missing required parameter: query');
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
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(false);
      expect(content.error).toBe('Search service down');
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
      
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(true);
      expect(content.target).toBe('/test/file1.ts');
      expect(content.analysis).toBe('Deep analysis result');
      expect(content.symbols).toHaveLength(1);
      expect(content.dependencies).toContain('/test/file2.ts');
      expect(content.complexity).toBe(5);
      expect(content.linesOfCode).toBe(50);
      
      expect(symbolAnalysisSpy).toHaveBeenCalledWith(
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
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(true);
      expect(content.target).toBe('/test/file2.ts');
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
      expect(response.error?.message).toContain('Missing required parameter: target');
    });

    it('should handle analysis errors', async () => {
      symbolAnalysisSpy.mockRejectedValueOnce(new Error('File not found'));

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
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(false);
      expect(content.error).toContain('File not found');
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
      
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(true);
      expect(content.indexed).toBe(true);
      expect(content.lastUpdated).toBe('2024-01-01T00:00:00Z');
      expect(content.filesCount).toBe(100);
      expect(content.symbolsCount).toBe(500);
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
      expect(response.error?.message).toContain('Missing required parameter: path');
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
      const content = JSON.parse(response.result.content[0].text);
      expect(content.success).toBe(false);
      expect(content.error).toBe('Repository not found');
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
      expect(response.error?.code).toBe(-32602);
      expect(response.error?.message).toContain('Missing tool name');
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
      expect(response.error?.code).toBe(-32602);
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
      expect(response.error?.message).toBe('Internal error');
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
      await expect(newServer.initialize()).resolves.not.toThrow();
      await newServer.cleanup();
    });

    it('should cleanup successfully', async () => {
      const newServer = new MimirMCPServer();
      await newServer.initialize();
      await expect(newServer.cleanup()).resolves.not.toThrow();
    });

    it('should handle multiple cleanup calls', async () => {
      const newServer = new MimirMCPServer();
      await newServer.initialize();
      await newServer.cleanup();
      await expect(newServer.cleanup()).resolves.not.toThrow();
    });
  });
});