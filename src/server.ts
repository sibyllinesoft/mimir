#!/usr/bin/env bun
/**
 * Mimir Server - HTTP server for MCP and web interface
 * 
 * Provides both MCP server functionality and web interface for repository
 * indexing and search operations.
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { logger as honoLogger } from 'hono/logger';
import { serve } from 'bun';
import { loadConfig, validateConfig } from '@/config/config';
import { createLogger, setupLogging } from '@/utils/logger';
import { MimirMCPServer } from '@/mcp/server';

const app = new Hono();
const logger = createLogger('mimir.server');

// Load and validate configuration
const config = loadConfig();
const warnings = validateConfig(config);

if (warnings.length > 0) {
  warnings.forEach(warning => logger.warn(warning));
}

// Setup logging
setupLogging(config.logging);

// Initialize MCP server
const mcpServer = new MimirMCPServer();

// Middleware
app.use('*', cors({
  origin: config.server.corsOrigins,
  allowHeaders: ['Content-Type', 'Authorization'],
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
}));

app.use('*', honoLogger((message, ...rest) => {
  logger.info(message, ...rest);
}));

// Health check endpoint
app.get('/health', (c) => {
  return c.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: '2.0.0',
    uptime: process.uptime(),
  });
});

// MCP endpoint for handling MCP requests
app.post('/mcp', async (c) => {
  try {
    const body = await c.req.json();
    
    logger.debug('Received MCP request', { 
      method: body.method, 
      id: body.id,
      params: body.params ? Object.keys(body.params) : [],
    });
    
    const response = await mcpServer.handleRequest(body);
    
    logger.debug('Sending MCP response', { 
      id: response.id,
      success: !response.error,
      error: response.error?.message,
    });
    
    return c.json(response);
  } catch (error: any) {
    logger.error('MCP request failed', { error: error.message });
    
    return c.json({
      jsonrpc: '2.0',
      id: null,
      error: {
        code: -32603,
        message: 'Internal error',
        data: error.message,
      },
    }, 500);
  }
});

// Repository indexing endpoint
app.post('/api/v1/index', async (c) => {
  try {
    const { path, id, force = false } = await c.req.json();
    
    if (!path) {
      return c.json({ error: 'Repository path is required' }, 400);
    }
    
    logger.info('Indexing repository via HTTP API', { path, id, force });
    
    // Use MCP server to handle indexing
    const mcpRequest = {
      jsonrpc: '2.0' as const,
      id: 'http-index-' + Date.now(),
      method: 'tools/call',
      params: {
        name: 'ensure_repo_index',
        arguments: {
          path,
          forceReindex: force,
          useLens: true,
        },
      },
    };
    
    const mcpResponse = await mcpServer.handleRequest(mcpRequest);
    
    if (mcpResponse.error) {
      return c.json({ error: mcpResponse.error.message }, 500);
    }
    
    return c.json({ success: true, data: mcpResponse.result });
  } catch (error: any) {
    logger.error('HTTP indexing failed', { error: error.message });
    return c.json({ error: 'Indexing failed: ' + error.message }, 500);
  }
});

// Repository search endpoint
app.get('/api/v1/search', async (c) => {
  try {
    const query = c.req.query('query');
    const indexId = c.req.query('index_id');
    const maxResults = parseInt(c.req.query('max_results') || '20');
    
    if (!query) {
      return c.json({ error: 'Query parameter is required' }, 400);
    }
    
    logger.info('Searching repositories via HTTP API', { query, indexId, maxResults });
    
    // Use MCP server to handle search
    const mcpRequest = {
      jsonrpc: '2.0' as const,
      id: 'http-search-' + Date.now(),
      method: 'tools/call',
      params: {
        name: 'hybrid_search',
        arguments: {
          query,
          indexId,
          maxResults,
        },
      },
    };
    
    const mcpResponse = await mcpServer.handleRequest(mcpRequest);
    
    if (mcpResponse.error) {
      return c.json({ error: mcpResponse.error.message }, 500);
    }
    
    return c.json({ success: true, data: mcpResponse.result });
  } catch (error: any) {
    logger.error('HTTP search failed', { error: error.message });
    return c.json({ error: 'Search failed: ' + error.message }, 500);
  }
});

// Repository status endpoint
app.get('/api/v1/status', async (c) => {
  try {
    const repositoryId = c.req.query('repository_id');
    const path = c.req.query('path');
    
    if (!repositoryId && !path) {
      return c.json({ error: 'Repository ID or path is required' }, 400);
    }
    
    logger.info('Getting repository status via HTTP API', { repositoryId, path });
    
    // Use MCP server to handle status check
    const mcpRequest = {
      jsonrpc: '2.0' as const,
      id: 'http-status-' + Date.now(),
      method: 'tools/call',
      params: {
        name: 'get_repository_status',
        arguments: {
          path: path || repositoryId,
        },
      },
    };
    
    const mcpResponse = await mcpServer.handleRequest(mcpRequest);
    
    if (mcpResponse.error) {
      return c.json({ error: mcpResponse.error.message }, 500);
    }
    
    return c.json({ success: true, data: mcpResponse.result });
  } catch (error: any) {
    logger.error('HTTP status check failed', { error: error.message });
    return c.json({ error: 'Status check failed: ' + error.message }, 500);
  }
});

// Code analysis endpoint
app.post('/api/v1/analyze', async (c) => {
  try {
    const { target, indexId, depth = 2 } = await c.req.json();
    
    if (!target || !indexId) {
      return c.json({ error: 'Target and index ID are required' }, 400);
    }
    
    logger.info('Analyzing code via HTTP API', { target, indexId, depth });
    
    // Use MCP server to handle analysis
    const mcpRequest = {
      jsonrpc: '2.0' as const,
      id: 'http-analyze-' + Date.now(),
      method: 'tools/call',
      params: {
        name: 'deep_code_analysis',
        arguments: {
          target,
          indexId,
          analysisDepth: depth,
          includeLensContext: true,
        },
      },
    };
    
    const mcpResponse = await mcpServer.handleRequest(mcpRequest);
    
    if (mcpResponse.error) {
      return c.json({ error: mcpResponse.error.message }, 500);
    }
    
    return c.json({ success: true, data: mcpResponse.result });
  } catch (error: any) {
    logger.error('HTTP analysis failed', { error: error.message });
    return c.json({ error: 'Analysis failed: ' + error.message }, 500);
  }
});

// Basic web interface
app.get('/', (c) => {
  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mimir - AI-Powered Code Research</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
        }
        .header {
            text-align: center;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .api-section {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }
        .endpoint {
            background: white;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #007acc;
        }
        code {
            background: #f1f1f1;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Monaco', 'Menlo', monospace;
        }
        .method {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
            color: white;
        }
        .get { background: #61affe; }
        .post { background: #49cc90; }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Mimir</h1>
        <p>AI-Powered Code Research System</p>
        <p><strong>Version:</strong> 2.0.0 | <strong>Status:</strong> Running</p>
    </div>
    
    <div class="api-section">
        <h2>Available Endpoints</h2>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/health</code>
            <p>Check server health status</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/mcp</code>
            <p>MCP protocol endpoint for Claude Code integration</p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/index</code>
            <p>Index a repository for search</p>
            <p><strong>Body:</strong> <code>{"path": "/repo/path", "id": "optional-id", "force": false}</code></p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/search</code>
            <p>Search indexed repositories</p>
            <p><strong>Query params:</strong> <code>query</code>, <code>index_id</code>, <code>max_results</code></p>
        </div>
        
        <div class="endpoint">
            <span class="method get">GET</span> <code>/api/v1/status</code>
            <p>Get repository indexing status</p>
            <p><strong>Query params:</strong> <code>repository_id</code> or <code>path</code></p>
        </div>
        
        <div class="endpoint">
            <span class="method post">POST</span> <code>/api/v1/analyze</code>
            <p>Perform deep code analysis</p>
            <p><strong>Body:</strong> <code>{"target": "file.ts", "indexId": "repo-id", "depth": 2}</code></p>
        </div>
    </div>
    
    <div class="api-section">
        <h2>Configuration</h2>
        <p><strong>Lens Enabled:</strong> ${config.lens.enabled ? 'Yes' : 'No'}</p>
        <p><strong>Lens URL:</strong> <code>${config.lens.baseUrl}</code></p>
        <p><strong>Storage Path:</strong> <code>${config.storage.dataPath}</code></p>
        <p><strong>Cache Path:</strong> <code>${config.storage.cachePath}</code></p>
    </div>
</body>
</html>
`;
  
  return c.html(html);
});

// Error handling
app.onError((err, c) => {
  logger.error('Server error', { error: err.message, path: c.req.path });
  return c.json({ error: 'Internal server error' }, 500);
});

// 404 handler
app.notFound((c) => {
  return c.json({ error: 'Not found' }, 404);
});

// Start server
const startServer = async () => {
  try {
    // Initialize MCP server
    await mcpServer.initialize();
    logger.info('MCP server initialized');
    
    // Start HTTP server
    const server = serve({
      port: config.server.port,
      hostname: config.server.host,
      fetch: app.fetch,
    });
    
    logger.info(`Mimir server started`, {
      host: config.server.host,
      port: config.server.port,
      url: `http://${config.server.host}:${config.server.port}`,
      lensEnabled: config.lens.enabled,
    });
    
    // Graceful shutdown
    const shutdown = async () => {
      logger.info('Shutting down server...');
      
      try {
        await mcpServer.cleanup();
        logger.info('MCP server cleaned up');
      } catch (error) {
        logger.error('Error during MCP server cleanup', { error });
      }
      
      server.stop();
      logger.info('Server shutdown complete');
      process.exit(0);
    };
    
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
    
  } catch (error) {
    logger.error('Failed to start server', { error });
    process.exit(1);
  }
};

startServer();
