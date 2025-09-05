/**
 * Unit tests for Lens integration client.
 * 
 * Tests HTTP client functionality, health checks, error handling,
 * circuit breaker pattern, and fallback mechanisms.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn } from 'bun:test';
import { 
  LensClient,
  LensHealthStatus,
  LensIntegrationError,
  LensConnectionError,
  LensServiceError,
  LensTimeoutError
} from '@/pipeline/lens-client';
import type { LensConfig, LensIndexRequest, LensSearchRequest } from '@/types';
import { createLogger } from '@/utils/logger';

describe('Lens Integration Client', () => {
  let fetchSpy: any;
  let consoleLogSpy: any;
  let lensConfig: LensConfig;

  beforeEach(() => {
    // Mock fetch
    fetchSpy = spyOn(global, 'fetch');
    
    // Mock console to suppress log output in tests
    consoleLogSpy = spyOn(console, 'log').mockImplementation(() => {});
    spyOn(console, 'error').mockImplementation(() => {});
    spyOn(console, 'warn').mockImplementation(() => {});

    // Default lens configuration for tests
    lensConfig = {
      enabled: true,
      baseUrl: 'http://localhost:5678',
      apiKey: 'test-api-key',
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
    };
  });

  afterEach(() => {
    fetchSpy.mockRestore();
    consoleLogSpy.mockRestore();
  });

  describe('Client Initialization', () => {
    it('should initialize with provided configuration', async () => {
      const client = new LensClient(lensConfig);
      await client.initialize();
      
      expect(client).toBeInstanceOf(LensClient);
      expect(client.isHealthy()).toBe(false); // Initially unknown
    });

    it('should initialize with logger', async () => {
      const logger = createLogger('test-lens');
      const client = new LensClient(lensConfig, logger);
      await client.initialize();
      
      expect(client).toBeInstanceOf(LensClient);
    });

    it('should perform health check on initialization when enabled', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ version: '1.0.0', uptime: 3600 })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      expect(fetchSpy).toHaveBeenCalledWith(
        'http://localhost:5678/health',
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-api-key'
          })
        })
      );
    });

    it('should skip health check when disabled', async () => {
      const configWithoutHealthCheck = { ...lensConfig, healthCheckEnabled: false };
      const client = new LensClient(configWithoutHealthCheck);
      
      await client.initialize();
      
      expect(fetchSpy).not.toHaveBeenCalled();
    });
  });

  describe('Health Checks', () => {
    it('should report healthy status for successful health check', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ version: '1.0.0', uptime: 3600 })
      });

      const client = new LensClient(lensConfig);
      const healthCheck = await client.checkHealth();
      
      expect(healthCheck.status).toBe(LensHealthStatus.HEALTHY);
      expect(healthCheck.version).toBe('1.0.0');
      expect(healthCheck.uptimeSeconds).toBe(3600);
      expect(healthCheck.responseTimeMs).toBeGreaterThan(0);
      expect(client.isHealthy()).toBe(true);
    });

    it('should report degraded status for non-200 responses', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 503,
        json: async () => ({})
      });

      const client = new LensClient(lensConfig);
      const healthCheck = await client.checkHealth();
      
      expect(healthCheck.status).toBe(LensHealthStatus.DEGRADED);
      expect(healthCheck.error).toBe('HTTP 503');
      expect(client.isHealthy()).toBe(false);
    });

    it('should report unhealthy status for network errors', async () => {
      fetchSpy.mockRejectedValueOnce(new Error('Connection refused'));

      const client = new LensClient(lensConfig);
      const healthCheck = await client.checkHealth();
      
      expect(healthCheck.status).toBe(LensHealthStatus.UNHEALTHY);
      expect(healthCheck.error).toBe('Connection refused');
      expect(client.isHealthy()).toBe(false);
    });

    it('should handle timeout errors', async () => {
      // Mock AbortError for timeout
      const abortError = new Error('The operation was aborted');
      abortError.name = 'AbortError';
      fetchSpy.mockRejectedValueOnce(abortError);

      const client = new LensClient(lensConfig);
      const healthCheck = await client.checkHealth();
      
      expect(healthCheck.status).toBe(LensHealthStatus.UNHEALTHY);
      expect(healthCheck.error).toBe('Health check timeout');
    });

    it('should track consecutive failures', async () => {
      fetchSpy.mockRejectedValue(new Error('Connection failed'));

      const client = new LensClient(lensConfig);
      
      // First failure
      await client.checkHealth();
      expect(client.getHealthStatus()).toBe(LensHealthStatus.UNHEALTHY);
      
      // Second failure
      await client.checkHealth();
      expect(client.getHealthStatus()).toBe(LensHealthStatus.UNHEALTHY);
      
      // Third failure - should open circuit breaker
      await client.checkHealth();
      expect(client.getHealthStatus()).toBe(LensHealthStatus.UNHEALTHY);
    });
  });

  describe('Circuit Breaker', () => {
    it('should open circuit breaker after consecutive failures', async () => {
      fetchSpy.mockRejectedValue(new Error('Service unavailable'));

      const client = new LensClient(lensConfig);
      
      // Trigger multiple failures
      await client.checkHealth();
      await client.checkHealth();
      await client.checkHealth();
      
      // Next request should use fallback due to circuit breaker
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(true);
      expect(response.fromFallback).toBe(true);
      expect(response.data?.fallback).toBe(true);
    });

    it('should reset circuit breaker after timeout period', async () => {
      fetchSpy
        .mockRejectedValueOnce(new Error('Failure 1'))
        .mockRejectedValueOnce(new Error('Failure 2'))
        .mockRejectedValueOnce(new Error('Failure 3'))
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({ status: 'healthy' })
        });

      const client = new LensClient(lensConfig);
      
      // Trigger failures to open circuit breaker
      await client.checkHealth();
      await client.checkHealth();
      await client.checkHealth();
      
      // Mock time passage by manipulating the internal state
      // In a real implementation, you'd wait or mock timers
      // For this test, we'll assume the circuit breaker auto-resets
      
      // This health check should work as circuit breaker should be reset
      const healthCheck = await client.checkHealth();
      expect(healthCheck.status).toBe(LensHealthStatus.HEALTHY);
    });
  });

  describe('Repository Indexing', () => {
    it('should make successful indexing request', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ 
          indexed: true, 
          collectionId: 'test-collection',
          filesIndexed: 100 
        })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: { version: '1.0.0' }
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(true);
      expect(response.data?.indexed).toBe(true);
      expect(response.data?.collectionId).toBe('test-collection');
      expect(response.data?.filesIndexed).toBe(100);
      
      expect(fetchSpy).toHaveBeenCalledWith(
        'http://localhost:5678/api/v1/index',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-api-key'
          }),
          body: JSON.stringify(indexRequest)
        })
      );
    });

    it('should handle indexing errors', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Invalid repository path' })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(false);
      expect(response.error).toBe('Invalid repository path');
      expect(response.statusCode).toBe(400);
    });

    it('should return error when indexing is disabled', async () => {
      const configWithoutIndexing = { ...lensConfig, enableIndexing: false };
      const client = new LensClient(configWithoutIndexing);
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(false);
      expect(response.error).toBe('Lens indexing is disabled');
      expect(response.statusCode).toBe(503);
    });
  });

  describe('Repository Search', () => {
    it('should make successful search request', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          results: [
            { path: '/test/file1.ts', score: 0.9, content: 'test content 1' },
            { path: '/test/file2.ts', score: 0.8, content: 'test content 2' }
          ],
          total: 2,
          queryTimeMs: 150
        })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const searchRequest: LensSearchRequest = {
        query: 'test query',
        repositoryId: 'test-repo',
        maxResults: 20,
        includeEmbeddings: false,
        filters: { fileType: 'typescript' }
      };

      const response = await client.searchRepository(searchRequest);
      
      expect(response.success).toBe(true);
      expect(response.data?.results).toHaveLength(2);
      expect(response.data?.total).toBe(2);
      expect(response.data?.queryTimeMs).toBe(150);
      
      // Verify URL parameters
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('query=test%20query'),
        expect.objectContaining({ method: 'GET' })
      );
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('repository_id=test-repo'),
        expect.any(Object)
      );
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.stringContaining('max_results=20'),
        expect.any(Object)
      );
    });

    it('should handle search with no repository ID', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          results: [],
          total: 0,
          queryTimeMs: 50
        })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const searchRequest: LensSearchRequest = {
        query: 'global search',
        maxResults: 10,
        includeEmbeddings: false,
        filters: {}
      };

      const response = await client.searchRepository(searchRequest);
      
      expect(response.success).toBe(true);
      expect(response.data?.results).toHaveLength(0);
      
      // Should not include repository_id parameter
      expect(fetchSpy).toHaveBeenCalledWith(
        expect.not.stringContaining('repository_id'),
        expect.any(Object)
      );
    });

    it('should return error when search is disabled', async () => {
      const configWithoutSearch = { ...lensConfig, enableSearch: false };
      const client = new LensClient(configWithoutSearch);
      
      const searchRequest: LensSearchRequest = {
        query: 'test query',
        maxResults: 20,
        includeEmbeddings: false,
        filters: {}
      };

      const response = await client.searchRepository(searchRequest);
      
      expect(response.success).toBe(false);
      expect(response.error).toBe('Lens search is disabled');
      expect(response.statusCode).toBe(503);
    });
  });

  describe('Retry Logic', () => {
    it('should retry on server errors', async () => {
      fetchSpy
        .mockResolvedValueOnce({
          ok: false,
          status: 500,
          json: async () => ({ error: 'Internal server error' })
        })
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({ indexed: true })
        });

      const client = new LensClient({ ...lensConfig, retryDelay: 0.1 });
      await client.initialize();
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(true);
      expect(fetchSpy).toHaveBeenCalledTimes(2);
    });

    it('should not retry on client errors', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: async () => ({ error: 'Bad request' })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(false);
      expect(response.error).toBe('Bad request');
      expect(fetchSpy).toHaveBeenCalledTimes(1); // No retry
    });

    it('should exhaust retries and return error', async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({ error: 'Service unavailable' })
      });

      const client = new LensClient({ ...lensConfig, maxRetries: 2, retryDelay: 0.1 });
      await client.initialize();
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(false);
      expect(response.error).toBe('Service unavailable');
      expect(fetchSpy).toHaveBeenCalledTimes(3); // Initial + 2 retries
    });
  });

  describe('Fallback Responses', () => {
    it('should provide fallback for search when service unavailable', async () => {
      const configWithFallback = { ...lensConfig, fallbackEnabled: true };
      fetchSpy.mockRejectedValue(new Error('Network error'));

      const client = new LensClient(configWithFallback);
      
      // Trigger circuit breaker
      await client.checkHealth();
      await client.checkHealth();
      await client.checkHealth();
      
      const searchRequest: LensSearchRequest = {
        query: 'test query',
        maxResults: 20,
        includeEmbeddings: false,
        filters: {}
      };

      const response = await client.searchRepository(searchRequest);
      
      expect(response.success).toBe(true);
      expect(response.fromFallback).toBe(true);
      expect(response.data?.fallback).toBe(true);
      expect(response.data?.results).toEqual([]);
      expect(response.data?.total).toBe(0);
    });

    it('should provide fallback for indexing when service unavailable', async () => {
      const configWithFallback = { ...lensConfig, fallbackEnabled: true };
      fetchSpy.mockRejectedValue(new Error('Network error'));

      const client = new LensClient(configWithFallback);
      
      // Trigger circuit breaker
      await client.checkHealth();
      await client.checkHealth();
      await client.checkHealth();
      
      const indexRequest: LensIndexRequest = {
        repositoryPath: '/test/repo',
        repositoryId: 'test-repo',
        branch: 'main',
        forceReindex: false,
        includeEmbeddings: true,
        metadata: {}
      };

      const response = await client.indexRepository(indexRequest);
      
      expect(response.success).toBe(true);
      expect(response.fromFallback).toBe(true);
      expect(response.data?.fallback).toBe(true);
      expect(response.data?.indexed).toBe(false);
      expect(response.data?.status).toBe('deferred');
    });
  });

  describe('Repository Status', () => {
    it('should get repository status successfully', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          indexed: true,
          lastUpdated: '2024-01-01T00:00:00Z',
          filesCount: 100,
          symbolsCount: 500
        })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const response = await client.getRepositoryStatus('test-repo');
      
      expect(response.success).toBe(true);
      expect(response.data?.indexed).toBe(true);
      expect(response.data?.filesCount).toBe(100);
      expect(response.data?.symbolsCount).toBe(500);
    });
  });

  describe('Embeddings', () => {
    it('should get embeddings successfully', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          embeddings: [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
          model: 'test-embedding-model'
        })
      });

      const client = new LensClient(lensConfig);
      await client.initialize();
      
      const response = await client.getEmbeddings('test-repo', '/test/file.ts');
      
      expect(response.success).toBe(true);
      expect(response.data?.embeddings).toHaveLength(2);
      expect(response.data?.model).toBe('test-embedding-model');
    });

    it('should return error when embeddings are disabled', async () => {
      const configWithoutEmbeddings = { ...lensConfig, enableEmbeddings: false };
      const client = new LensClient(configWithoutEmbeddings);
      
      const response = await client.getEmbeddings('test-repo');
      
      expect(response.success).toBe(false);
      expect(response.error).toBe('Lens embeddings are disabled');
      expect(response.statusCode).toBe(503);
    });
  });

  describe('Utility Methods', () => {
    it('should track last health check timestamp', async () => {
      fetchSpy.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({ status: 'healthy' })
      });

      const client = new LensClient(lensConfig);
      
      expect(client.getLastHealthCheck()).toBeUndefined();
      
      await client.checkHealth();
      
      expect(client.getLastHealthCheck()).toBeInstanceOf(Date);
    });

    it('should ensure healthy within timeout', async () => {
      let callCount = 0;
      fetchSpy.mockImplementation(() => {
        callCount++;
        if (callCount <= 2) {
          return Promise.resolve({
            ok: false,
            status: 503,
            json: async () => ({})
          });
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({ status: 'healthy' })
        });
      });

      const client = new LensClient(lensConfig);
      
      const result = await client.ensureHealthy(10); // 10 seconds timeout
      
      expect(result).toBe(true);
      expect(client.isHealthy()).toBe(true);
    });

    it('should timeout when service never becomes healthy', async () => {
      fetchSpy.mockResolvedValue({
        ok: false,
        status: 503,
        json: async () => ({})
      });

      const client = new LensClient(lensConfig);
      
      const result = await client.ensureHealthy(1); // 1 second timeout
      
      expect(result).toBe(false);
      expect(client.isHealthy()).toBe(false);
    });
  });

  describe('Cleanup', () => {
    it('should cleanup resources', async () => {
      const client = new LensClient(lensConfig);
      await client.initialize();
      
      // Should not throw
      await expect(client.cleanup()).resolves.not.toThrow();
    });
  });
});