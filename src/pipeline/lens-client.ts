/**
 * Lens Integration Client for Mimir-Lens Communication
 * 
 * Provides HTTP client for communication with Lens indexing service,
 * including health checks, error handling, and fallback mechanisms.
 */

import type { Logger, LensConfig, LensResponse, LensIndexRequest, LensSearchRequest } from '@/types';
import { createLogger } from '@/utils/logger';

export enum LensHealthStatus {
  HEALTHY = 'healthy',
  UNHEALTHY = 'unhealthy',
  DEGRADED = 'degraded',
  UNKNOWN = 'unknown',
}

export interface LensHealthCheck {
  status: LensHealthStatus;
  timestamp: Date;
  responseTimeMs: number;
  version?: string;
  uptimeSeconds?: number;
  error?: string;
}

export class LensIntegrationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LensIntegrationError';
  }
}

export class LensConnectionError extends LensIntegrationError {
  constructor(message: string) {
    super(message);
    this.name = 'LensConnectionError';
  }
}

export class LensServiceError extends LensIntegrationError {
  constructor(message: string) {
    super(message);
    this.name = 'LensServiceError';
  }
}

export class LensTimeoutError extends LensIntegrationError {
  constructor(message: string) {
    super(message);
    this.name = 'LensTimeoutError';
  }
}

export class LensClient {
  private config: LensConfig;
  private logger: Logger;
  private healthStatus: LensHealthStatus = LensHealthStatus.UNKNOWN;
  private lastHealthCheck?: Date;
  private consecutiveFailures = 0;
  private circuitBreakerOpenedAt?: Date;

  constructor(config: LensConfig, logger?: Logger) {
    this.config = config;
    this.logger = logger || createLogger('mimir.pipeline.lens-client');
    this.logger.info(`Initializing Lens client with base_url: ${this.config.baseUrl}`);
  }

  async initialize(): Promise<void> {
    this.logger.info('Lens client initialized');
    
    // Perform initial health check if enabled
    if (this.config.healthCheckEnabled) {
      await this.checkHealth();
    }
  }

  async cleanup(): Promise<void> {
    this.logger.info('Lens client cleaned up');
  }

  async checkHealth(): Promise<LensHealthCheck> {
    const startTime = Date.now();
    
    try {
      const healthUrl = `${this.config.baseUrl}/health`;
      
      const controller = new AbortController();
      const timeoutId = setTimeout(
        () => controller.abort(),
        this.config.healthCheckTimeout * 1000
      );
      
      const response = await fetch(healthUrl, {
        signal: controller.signal,
        headers: this.getHeaders(),
      });
      
      clearTimeout(timeoutId);
      const responseTime = Date.now() - startTime;
      
      if (response.ok) {
        const data = await response.json().catch(() => ({}));
        
        const healthCheck: LensHealthCheck = {
          status: LensHealthStatus.HEALTHY,
          timestamp: new Date(),
          responseTimeMs: responseTime,
          version: data.version,
          uptimeSeconds: data.uptime,
        };
        
        this.healthStatus = LensHealthStatus.HEALTHY;
        this.consecutiveFailures = 0;
        this.circuitBreakerOpenedAt = undefined;
        
        return healthCheck;
      } else {
        const healthCheck: LensHealthCheck = {
          status: LensHealthStatus.DEGRADED,
          timestamp: new Date(),
          responseTimeMs: responseTime,
          error: `HTTP ${response.status}`,
        };
        
        this.healthStatus = LensHealthStatus.DEGRADED;
        return healthCheck;
      }
    } catch (error: any) {
      const responseTime = Date.now() - startTime;
      const healthCheck: LensHealthCheck = {
        status: LensHealthStatus.UNHEALTHY,
        timestamp: new Date(),
        responseTimeMs: responseTime,
        error: error.name === 'AbortError' ? 'Health check timeout' : error.message,
      };
      
      this.healthStatus = LensHealthStatus.UNHEALTHY;
      this.consecutiveFailures += 1;
      
      // Open circuit breaker if too many failures
      if (this.consecutiveFailures >= 3 && !this.circuitBreakerOpenedAt) {
        this.circuitBreakerOpenedAt = new Date();
        this.logger.warn(`Lens circuit breaker opened after ${this.consecutiveFailures} failures`);
      }
      
      return healthCheck;
    } finally {
      this.lastHealthCheck = new Date();
    }
  }

  private isCircuitBreakerOpen(): boolean {
    if (!this.circuitBreakerOpenedAt) {
      return false;
    }
    
    // Auto-reset after 60 seconds
    const now = new Date();
    const timeSinceOpened = now.getTime() - this.circuitBreakerOpenedAt.getTime();
    
    if (timeSinceOpened > 60000) {
      this.circuitBreakerOpenedAt = undefined;
      this.consecutiveFailures = 0;
      this.logger.info('Lens circuit breaker auto-reset');
      return false;
    }
    
    return true;
  }

  private getHeaders(): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'User-Agent': 'Mimir-Lens-Client/1.0',
    };
    
    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }
    
    return headers;
  }

  private async makeRequest(
    method: string,
    endpoint: string,
    data?: any,
    params?: Record<string, string>
  ): Promise<LensResponse> {
    // Check circuit breaker
    if (this.isCircuitBreakerOpen()) {
      if (this.config.fallbackEnabled) {
        this.logger.warn('Circuit breaker open, using fallback');
        return this.createFallbackResponse(method, endpoint, data);
      } else {
        throw new LensConnectionError('Circuit breaker open, service unavailable');
      }
    }

    const url = new URL(endpoint.startsWith('/') ? endpoint.slice(1) : endpoint, this.config.baseUrl);
    if (params) {
      Object.entries(params).forEach(([key, value]) => {
        url.searchParams.set(key, value);
      });
    }

    const startTime = Date.now();

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(
          () => controller.abort(),
          this.config.timeout * 1000
        );

        const requestOptions: RequestInit = {
          method: method.toUpperCase(),
          headers: this.getHeaders(),
          signal: controller.signal,
        };

        if (data && (method.toUpperCase() === 'POST' || method.toUpperCase() === 'PUT')) {
          requestOptions.body = JSON.stringify(data);
        }

        const response = await fetch(url.toString(), requestOptions);
        clearTimeout(timeoutId);
        
        const responseTime = Date.now() - startTime;

        if (response.ok) {
          let resultData: any;
          try {
            resultData = await response.json();
          } catch {
            resultData = await response.text();
          }

          this.consecutiveFailures = 0;

          return {
            success: true,
            data: resultData,
            statusCode: response.status,
            responseTimeMs: responseTime,
          };
        } else {
          let errorMsg: string;
          try {
            const errorData = await response.json();
            errorMsg = errorData.error || `HTTP ${response.status}`;
          } catch {
            errorMsg = `HTTP ${response.status}`;
          }

          // Don't retry client errors (4xx)
          if (response.status >= 400 && response.status < 500) {
            return {
              success: false,
              error: errorMsg,
              statusCode: response.status,
              responseTimeMs: responseTime,
            };
          }

          // Retry server errors (5xx)
          if (attempt < this.config.maxRetries) {
            await this.delay(this.config.retryDelay * 1000 * (attempt + 1));
            continue;
          } else {
            this.consecutiveFailures += 1;
            return {
              success: false,
              error: errorMsg,
              statusCode: response.status,
              responseTimeMs: responseTime,
            };
          }
        }
      } catch (error: any) {
        if (error.name === 'AbortError') {
          if (attempt < this.config.maxRetries) {
            await this.delay(this.config.retryDelay * 1000 * (attempt + 1));
            continue;
          } else {
            this.consecutiveFailures += 1;
            const responseTime = Date.now() - startTime;

            if (this.config.fallbackEnabled) {
              this.logger.warn(`Request timeout after ${this.config.maxRetries} retries, using fallback`);
              return this.createFallbackResponse(method, endpoint, data);
            } else {
              throw new LensTimeoutError(`Request timeout after ${this.config.maxRetries} retries`);
            }
          }
        }

        if (attempt < this.config.maxRetries) {
          this.logger.warn(`Request attempt ${attempt + 1} failed: ${error.message}`);
          await this.delay(this.config.retryDelay * 1000 * (attempt + 1));
          continue;
        } else {
          this.consecutiveFailures += 1;

          if (this.config.fallbackEnabled) {
            this.logger.error(`Request failed after ${this.config.maxRetries} retries, using fallback: ${error.message}`);
            return this.createFallbackResponse(method, endpoint, data);
          } else {
            throw new LensConnectionError(`Request failed after ${this.config.maxRetries} retries: ${error.message}`);
          }
        }
      }
    }

    // This shouldn't be reached, but typescript requires it
    throw new Error('Unexpected end of retry loop');
  }

  private createFallbackResponse(
    method: string,
    endpoint: string,
    data?: any
  ): LensResponse {
    const fallbackData: any = {
      message: 'Lens service unavailable, using local fallback',
      fallback: true,
      originalEndpoint: endpoint,
      timestamp: new Date().toISOString(),
    };

    // For search requests, return empty results
    if (endpoint.toLowerCase().includes('search')) {
      fallbackData.results = [];
      fallbackData.total = 0;
      fallbackData.queryTimeMs = 0;
    }

    // For indexing requests, simulate success
    if (endpoint.toLowerCase().includes('index')) {
      fallbackData.indexed = false;
      fallbackData.status = 'deferred';
      fallbackData.message = 'Indexing deferred until Lens service is available';
    }

    return {
      success: true,
      data: fallbackData,
      statusCode: 200,
      responseTimeMs: 1.0,
      fromFallback: true,
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // API Methods

  async indexRepository(request: LensIndexRequest): Promise<LensResponse> {
    if (!this.config.enabled || !this.config.enableIndexing) {
      return {
        success: false,
        error: 'Lens indexing is disabled',
        statusCode: 503,
        responseTimeMs: 0,
      };
    }

    return this.makeRequest('POST', '/api/v1/index', request);
  }

  async searchRepository(request: LensSearchRequest): Promise<LensResponse> {
    if (!this.config.enabled || !this.config.enableSearch) {
      return {
        success: false,
        error: 'Lens search is disabled',
        statusCode: 503,
        responseTimeMs: 0,
      };
    }

    const params: Record<string, string> = {
      query: request.query,
      ...(request.repositoryId && { repository_id: request.repositoryId }),
      max_results: request.maxResults?.toString() || '20',
      include_embeddings: request.includeEmbeddings?.toString() || 'false',
    };

    if (request.filters) {
      Object.entries(request.filters).forEach(([key, value]) => {
        params[key] = String(value);
      });
    }

    return this.makeRequest('GET', '/api/v1/search', undefined, params);
  }

  async getEmbeddings(repositoryId: string, filePath?: string): Promise<LensResponse> {
    if (!this.config.enabled || !this.config.enableEmbeddings) {
      return {
        success: false,
        error: 'Lens embeddings are disabled',
        statusCode: 503,
        responseTimeMs: 0,
      };
    }

    const params: Record<string, string> = {
      repository_id: repositoryId,
      ...(filePath && { file_path: filePath }),
    };

    return this.makeRequest('GET', '/api/v1/embeddings', undefined, params);
  }

  async getRepositoryStatus(repositoryId: string): Promise<LensResponse> {
    const params = { repository_id: repositoryId };
    return this.makeRequest('GET', '/api/v1/status', undefined, params);
  }

  // Utility Methods

  isHealthy(): boolean {
    return this.healthStatus === LensHealthStatus.HEALTHY;
  }

  getHealthStatus(): LensHealthStatus {
    return this.healthStatus;
  }

  getLastHealthCheck(): Date | undefined {
    return this.lastHealthCheck;
  }

  async ensureHealthy(maxWaitSeconds: number = 30): Promise<boolean> {
    const startTime = Date.now();
    const maxWaitMs = maxWaitSeconds * 1000;

    while (Date.now() - startTime < maxWaitMs) {
      const health = await this.checkHealth();

      if (health.status === LensHealthStatus.HEALTHY) {
        return true;
      }

      // Wait before next check
      const remainingTime = maxWaitMs - (Date.now() - startTime);
      const waitTime = Math.min(5000, remainingTime);
      
      if (waitTime > 0) {
        await this.delay(waitTime);
      }
    }

    return false;
  }
}

// Global client instance
let globalLensClient: LensClient | null = null;

export function getLensClient(config: LensConfig): LensClient {
  if (!globalLensClient) {
    globalLensClient = new LensClient(config);
  }
  return globalLensClient;
}

export async function initLensClient(config: LensConfig): Promise<LensClient> {
  const client = getLensClient(config);
  await client.initialize();
  return client;
}
