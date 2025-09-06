#!/usr/bin/env bun
/**
 * Mimir CLI - Command line interface for repository indexing and search
 * 
 * Provides commands for indexing repositories and searching indexed content.
 */

import { Command } from 'commander';
import { resolve } from 'path';
import { existsSync } from 'fs';
import { loadConfig, loadConfigFromFile, validateConfig } from '@/config/config';
import { createLogger, setupLogging } from '@/utils/logger';
import { FileDiscovery } from '@/pipeline/discovery';
import { SymbolAnalysis } from '@/pipeline/symbols';
import { LensClient } from '@/pipeline/lens-client';
import type { PipelineContext } from '@/types';

const program = new Command();
const logger = createLogger('mimir.cli');

// Helper function to load config based on global options
function getConfig(): any {
  const globalOpts = program.opts();
  
  try {
    if (globalOpts.config) {
      return loadConfigFromFile(globalOpts.config);
    } else {
      return loadConfig();
    }
  } catch (error) {
    console.error(`Error loading configuration: ${error.message}`);
    process.exit(1);
  }
}

program
  .name('mimir')
  .description('AI-Powered Code Research System')
  .version('2.0.0');

// Global options
program
  .option('-c, --config <path>', 'Configuration file path')
  .option('-v, --verbose', 'Enable verbose logging')
  .option('--log-level <level>', 'Set log level (debug, info, warn, error)', 'info')
  .hook('preAction', (thisCommand) => {
    const opts = thisCommand.opts();
    
    // Setup logging
    setupLogging({
      logLevel: opts.verbose ? 'debug' : (opts.logLevel || 'info'),
      logFormat: 'human',
      logIncludeTimestamp: true,
      logIncludeLevel: true,
      logIncludeLogger: false,
      logIncludeThread: false,
      logFile: undefined,
      logMaxSize: '50m',
      logMaxFiles: 10,
    });
  });

// Index command
program
  .command('index')
  .description('Index a repository for search')
  .argument('<path>', 'Repository path to index')
  .option('--id <id>', 'Custom index ID')
  .option('--force', 'Force re-indexing')
  .option('--no-lens', 'Disable Lens integration')
  .option('--languages <langs>', 'Comma-separated list of languages to include')
  .action(async (repoPath, options) => {
    try {
      logger.info('Starting repository indexing', { path: repoPath, options });
      
      const config = getConfig();
      const warnings = validateConfig(config);
      
      if (warnings.length > 0) {
        warnings.forEach(warning => logger.warn(warning));
      }

      // Resolve repository path
      const resolvedPath = resolve(repoPath);
      if (!existsSync(resolvedPath)) {
        logger.error('Repository path does not exist', { path: resolvedPath });
        process.exit(1);
      }

      // Generate index ID
      const indexId = options.id || Buffer.from(resolvedPath).toString('base64').replace(/[/+=]/g, '').slice(0, 16);
      
      // Create pipeline context
      const context: PipelineContext = {
        indexId,
        repoPath: resolvedPath,
        repoInfo: {
          root: resolvedPath,
          rev: 'main',
          worktreeDirty: false,
        },
        config: {
          languages: options.languages ? options.languages.split(',') : config.pipeline.treeSitterLanguages,
          excludes: ['node_modules/', 'dist/', 'build/', '.git/', '__pycache__/'],
          contextLines: 3,
          maxFilesToEmbed: 1000,
        },
        storageDir: config.storage.dataPath,
        cacheDir: config.storage.cachePath,
      };

      // Initialize components
      const fileDiscovery = new FileDiscovery(resolvedPath, logger);
      const symbolAnalysis = new SymbolAnalysis(logger);
      
      // Discover files
      logger.info('Discovering files...');
      const discoveryResult = await fileDiscovery.discover(resolvedPath, context.config);
      logger.info(`Discovered ${discoveryResult.files.length} files`);
      
      // Analyze symbols
      logger.info('Analyzing symbols...');
      const symbolResult = await symbolAnalysis.analyze(discoveryResult.files, context);
      logger.info(`Found ${symbolResult.symbolCount} unique symbols`);
      
      // Index with Lens if enabled
      if (options.lens !== false && config.lens.enabled) {
        logger.info('Indexing with Lens...');
        
        const lensClient = new LensClient(config.lens, logger);
        await lensClient.initialize();
        
        const lensResponse = await lensClient.indexRepository({
          repositoryPath: resolvedPath,
          repositoryId: indexId,
          branch: context.repoInfo.rev,
          forceReindex: options.force || false,
          includeEmbeddings: true,
          metadata: { 
            mimirVersion: '2.0.0', 
            languages: context.config.languages,
            fileCount: discoveryResult.files.length,
            symbolCount: symbolResult.symbolCount,
          },
        });
        
        if (lensResponse.success) {
          logger.info('Repository indexed successfully with Lens', {
            indexId,
            filesIndexed: discoveryResult.files.length,
            symbolsFound: symbolResult.symbolCount,
            lensCollectionId: lensResponse.data?.collectionId,
          });
        } else {
          logger.warn('Lens indexing failed, using local index', { error: lensResponse.error });
        }
        
        await lensClient.cleanup();
      }
      
      logger.info('Repository indexing completed successfully', {
        indexId,
        filesIndexed: discoveryResult.files.length,
        totalSize: Math.round(discoveryResult.totalSize / 1024) + ' KB',
        symbolsFound: symbolResult.symbolCount,
        duration: discoveryResult.duration + 'ms',
      });
      
    } catch (error) {
      logger.error('Repository indexing failed', { error });
      process.exit(1);
    }
  });

// Search command
program
  .command('search')
  .description('Search indexed repositories')
  .argument('<query>', 'Search query')
  .option('--index-id <id>', 'Specific index to search')
  .option('--max-results <num>', 'Maximum results to return', '20')
  .option('--format <format>', 'Output format (json|text)', 'text')
  .action(async (query, options) => {
    try {
      logger.info('Starting search', { query, options });
      
      const config = getConfig();
      
      if (!config.lens.enabled) {
        logger.error('Search requires Lens integration to be enabled');
        process.exit(1);
      }
      
      const lensClient = new LensClient(config.lens, logger);
      await lensClient.initialize();
      
      const searchResponse = await lensClient.searchRepository({
        query,
        repositoryId: options.indexId,
        maxResults: parseInt(options.maxResults),
        includeEmbeddings: false,
      });
      
      if (searchResponse.success && searchResponse.data) {
        const results = searchResponse.data.results || [];
        
        if (options.format === 'json') {
          console.log(JSON.stringify({
            query,
            totalResults: results.length,
            results,
            responseTimeMs: searchResponse.responseTimeMs,
          }, null, 2));
        } else {
          console.log(`Found ${results.length} results for "${query}"\n`);
          
          results.forEach((result: any, index: number) => {
            console.log(`${index + 1}. ${result.path} (score: ${result.score?.toFixed(3) || 'N/A'})`);
            console.log(`   ${(result.content || result.text || '').slice(0, 100)}...`);
            console.log();
          });
          
          console.log(`Search completed in ${searchResponse.responseTimeMs}ms`);
        }
      } else {
        logger.error('Search failed', { error: searchResponse.error });
        process.exit(1);
      }
      
      await lensClient.cleanup();
      
    } catch (error) {
      logger.error('Search failed', { error });
      process.exit(1);
    }
  });

// Status command
program
  .command('status')
  .description('Check status of indexed repositories')
  .argument('[path]', 'Repository path or index ID')
  .action(async (pathOrId) => {
    try {
      const config = getConfig();
      
      if (!config.lens.enabled) {
        logger.error('Status requires Lens integration to be enabled');
        process.exit(1);
      }
      
      const lensClient = new LensClient(config.lens, logger);
      await lensClient.initialize();
      
      // Check Lens health
      const health = await lensClient.checkHealth();
      console.log(`Lens Service Status: ${health.status}`);
      console.log(`Response Time: ${health.responseTimeMs}ms`);
      
      if (health.version) {
        console.log(`Version: ${health.version}`);
      }
      
      if (pathOrId) {
        // Generate index ID if path provided
        const indexId = existsSync(pathOrId) 
          ? Buffer.from(resolve(pathOrId)).toString('base64').replace(/[/+=]/g, '').slice(0, 16)
          : pathOrId;
        
        const statusResponse = await lensClient.getRepositoryStatus(indexId);
        
        if (statusResponse.success && statusResponse.data) {
          const status = statusResponse.data;
          console.log(`\nRepository Status (${indexId}):`);
          console.log(`  Indexed: ${status.indexed ? 'Yes' : 'No'}`);
          console.log(`  Last Updated: ${status.lastUpdated || 'Never'}`);
          console.log(`  Files: ${status.filesCount || 0}`);
          console.log(`  Symbols: ${status.symbolsCount || 0}`);
        } else {
          logger.warn('Could not get repository status', { error: statusResponse.error });
        }
      }
      
      await lensClient.cleanup();
      
    } catch (error) {
      logger.error('Status check failed', { error });
      process.exit(1);
    }
  });

// Validate command
program
  .command('validate')
  .description('Validate repository structure and configuration')
  .argument('<path>', 'Repository path to validate')
  .action(async (repoPath) => {
    try {
      logger.info('Validating repository', { path: repoPath });
      
      // Load config to trigger validation of config file
      const config = getConfig();
      
      const resolvedPath = resolve(repoPath);
      if (!existsSync(resolvedPath)) {
        logger.error('Repository path does not exist', { path: resolvedPath });
        process.exit(1);
      }
      
      const fileDiscovery = new FileDiscovery(resolvedPath, logger);
      const symbolAnalysis = new SymbolAnalysis(logger);
      
      // Validate repository structure
      const structureReport = await fileDiscovery.validateRepositoryStructure();
      console.log('Repository Structure Validation:');
      console.log(`  Valid: ${structureReport.valid ? 'Yes' : 'No'}`);
      console.log(`  Total Files: ${structureReport.stats.totalFiles}`);
      console.log(`  File Types:`);
      
      Object.entries(structureReport.stats.fileTypes).forEach(([ext, count]) => {
        console.log(`    .${ext}: ${count}`);
      });
      
      if (structureReport.warnings.length > 0) {
        console.log('\n  Warnings:');
        structureReport.warnings.forEach(warning => console.log(`    - ${warning}`));
      }
      
      if (structureReport.recommendations.length > 0) {
        console.log('\n  Recommendations:');
        structureReport.recommendations.forEach(rec => console.log(`    - ${rec}`));
      }
      
      // Validate project for symbol analysis
      const projectValidation = await symbolAnalysis.validateProject(resolvedPath);
      console.log('\nSymbol Analysis Validation:');
      console.log(`  Valid: ${projectValidation.valid ? 'Yes' : 'No'}`);
      console.log(`  TypeScript Files: ${projectValidation.stats.typeScriptFiles}`);
      console.log(`  JavaScript Files: ${projectValidation.stats.javascriptFiles}`);
      console.log(`  Python Files: ${projectValidation.stats.pythonFiles}`);
      
      if (projectValidation.issues.length > 0) {
        console.log('\n  Issues:');
        projectValidation.issues.forEach(issue => console.log(`    - ${issue}`));
      }
      
      if (projectValidation.recommendations.length > 0) {
        console.log('\n  Recommendations:');
        projectValidation.recommendations.forEach(rec => console.log(`    - ${rec}`));
      }
      
      const overallValid = structureReport.valid && projectValidation.valid;
      console.log(`\nOverall Status: ${overallValid ? 'Valid' : 'Issues Found'}`);
      
      if (!overallValid) {
        process.exit(1);
      }
      
    } catch (error) {
      logger.error('Validation failed', { error });
      process.exit(1);
    }
  });

// Parse command line arguments
program.parse();
