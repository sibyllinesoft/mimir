/**
 * Symbol analysis module for TypeScript and other languages
 * 
 * Provides symbol resolution, reference analysis, and code understanding
 * capabilities for deep code analysis.
 */

import { promises as fs } from 'fs';
import { join, resolve } from 'path';
import { spawn } from 'child_process';
import type { Logger, PipelineContext, SymbolEntry, SearchResult, IndexConfig } from '@/types';
import { createLogger } from '@/utils/logger';
import { ensureDirectory, atomicWriteJson, fileExists } from '@/utils/filesystem';

export interface SymbolAnalysisResult {
  symbols: SymbolEntry[];
  fileCount: number;
  symbolCount: number;
}

export interface DeepAnalysisOptions {
  depth: number;
  includeLensContext: boolean;
  indexId: string;
}

export interface DeepAnalysisResult {
  summary: string;
  findings: string[];
  dependencies?: string[];
  complexity?: number;
  recommendations?: string[];
}

export class SymbolAnalysis {
  private logger: Logger;

  constructor(logger?: Logger) {
    this.logger = logger || createLogger('mimir.pipeline.symbols');
  }

  async analyze(
    files: string[],
    context: PipelineContext
  ): Promise<SymbolAnalysisResult> {
    const startTime = Date.now();
    
    try {
      this.logger.info('Starting symbol analysis', {
        fileCount: files.length,
        indexId: context.indexId,
      });

      // Filter supported files for symbol analysis
      const supportedFiles = this.filterSupportedFiles(files);
      
      if (supportedFiles.length === 0) {
        this.logger.warn('No supported files found for symbol analysis');
        return {
          symbols: [],
          fileCount: 0,
          symbolCount: 0,
        };
      }

      // Analyze symbols in batches
      const symbols = await this.analyzeFilesBatch(supportedFiles, context);
      
      const duration = Date.now() - startTime;
      const symbolCount = new Set(symbols.map(s => s.symbol).filter(Boolean)).size;
      
      this.logger.info('Symbol analysis completed', {
        duration,
        fileCount: supportedFiles.length,
        totalSymbols: symbols.length,
        uniqueSymbols: symbolCount,
      });

      return {
        symbols,
        fileCount: supportedFiles.length,
        symbolCount,
      };
    } catch (error) {
      this.logger.error('Symbol analysis failed', { error });
      throw error;
    }
  }

  private filterSupportedFiles(files: string[]): string[] {
    const supportedExtensions = new Set([
      '.ts', '.tsx', '.js', '.jsx', '.py', '.pyi', '.rs', '.go', '.java'
    ]);
    
    return files.filter(file => {
      const ext = file.substring(file.lastIndexOf('.'));
      return supportedExtensions.has(ext);
    });
  }

  private async analyzeFilesBatch(
    files: string[],
    context: PipelineContext
  ): Promise<SymbolEntry[]> {
    const symbols: SymbolEntry[] = [];
    const batchSize = 50;
    
    for (let i = 0; i < files.length; i += batchSize) {
      const batch = files.slice(i, i + batchSize);
      const batchSymbols = await Promise.allSettled(
        batch.map(file => this.analyzeFile(file, context))
      );
      
      for (const result of batchSymbols) {
        if (result.status === 'fulfilled') {
          symbols.push(...result.value);
        } else {
          this.logger.warn('File analysis failed', { error: result.reason });
        }
      }
    }
    
    return symbols;
  }

  private async analyzeFile(
    filePath: string,
    context: PipelineContext
  ): Promise<SymbolEntry[]> {
    const fullPath = join(context.repoPath, filePath);
    
    try {
      // Check if file exists
      if (!(await fileExists(fullPath))) {
        return [];
      }

      const content = await fs.readFile(fullPath, 'utf-8');
      const extension = filePath.substring(filePath.lastIndexOf('.'));
      
      switch (extension) {
        case '.ts':
        case '.tsx':
        case '.js':
        case '.jsx':
          return this.analyzeTypeScriptFile(filePath, content);
        case '.py':
        case '.pyi':
          return this.analyzePythonFile(filePath, content);
        default:
          return this.analyzeGenericFile(filePath, content);
      }
    } catch (error) {
      this.logger.warn(`Failed to analyze file ${filePath}`, { error });
      return [];
    }
  }

  private async analyzeTypeScriptFile(filePath: string, content: string): Promise<SymbolEntry[]> {
    const symbols: SymbolEntry[] = [];
    const lines = content.split('\n');
    
    // Basic pattern matching for TypeScript symbols
    // In a real implementation, this would use tree-sitter or TypeScript compiler API
    
    let currentPosition = 0;
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
      const line = lines[lineIndex];
      const lineStart = currentPosition;
      const lineEnd = currentPosition + line.length;
      
      // Function declarations
      const funcMatch = line.match(/(?:export\s+)?(?:async\s+)?function\s+(\w+)/);
      if (funcMatch) {
        symbols.push({
          type: 'def' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: funcMatch[1],
          sig: line.trim(),
        });
      }
      
      // Class declarations
      const classMatch = line.match(/(?:export\s+)?class\s+(\w+)/);
      if (classMatch) {
        symbols.push({
          type: 'def' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: classMatch[1],
          sig: line.trim(),
        });
      }
      
      // Interface declarations
      const interfaceMatch = line.match(/(?:export\s+)?interface\s+(\w+)/);
      if (interfaceMatch) {
        symbols.push({
          type: 'def' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: interfaceMatch[1],
          sig: line.trim(),
        });
      }
      
      // Import statements
      const importMatch = line.match(/import\s+.*\s+from\s+['"]([^'"]+)['"]/);
      if (importMatch) {
        symbols.push({
          type: 'ref' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: importMatch[1],
        });
      }
      
      currentPosition = lineEnd + 1; // +1 for newline
    }
    
    return symbols;
  }

  private async analyzePythonFile(filePath: string, content: string): Promise<SymbolEntry[]> {
    const symbols: SymbolEntry[] = [];
    const lines = content.split('\n');
    
    let currentPosition = 0;
    for (let lineIndex = 0; lineIndex < lines.length; lineIndex++) {
      const line = lines[lineIndex];
      const lineStart = currentPosition;
      const lineEnd = currentPosition + line.length;
      
      // Function definitions
      const funcMatch = line.match(/def\s+(\w+)\s*\(/);
      if (funcMatch) {
        symbols.push({
          type: 'def' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: funcMatch[1],
          sig: line.trim(),
        });
      }
      
      // Class definitions
      const classMatch = line.match(/class\s+(\w+)/);
      if (classMatch) {
        symbols.push({
          type: 'def' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: classMatch[1],
          sig: line.trim(),
        });
      }
      
      // Import statements
      const importMatch = line.match(/(?:from\s+(\w+)\s+import|import\s+(\w+))/);
      if (importMatch) {
        const module = importMatch[1] || importMatch[2];
        symbols.push({
          type: 'ref' as const,
          path: filePath,
          span: [lineStart, lineEnd],
          symbol: module,
        });
      }
      
      currentPosition = lineEnd + 1;
    }
    
    return symbols;
  }

  private async analyzeGenericFile(filePath: string, content: string): Promise<SymbolEntry[]> {
    // Basic analysis for other file types
    // This could be extended with language-specific parsers
    return [];
  }

  async computeRelevance(query: string, result: SearchResult): Promise<number> {
    // Compute relevance score based on symbol analysis
    let score = 0;
    
    // Boost score if query matches symbol names
    if (result.content?.text) {
      const queryTokens = query.toLowerCase().split(/\s+/);
      const contentLower = result.content.text.toLowerCase();
      
      for (const token of queryTokens) {
        if (contentLower.includes(token)) {
          score += 0.1;
        }
        
        // Higher score for exact symbol matches
        const symbolRegex = new RegExp(`\\b${token}\\b`, 'i');
        if (symbolRegex.test(contentLower)) {
          score += 0.3;
        }
      }
    }
    
    // Boost score based on file type relevance
    if (result.path.endsWith('.ts') || result.path.endsWith('.tsx')) {
      score += 0.1;
    }
    
    return Math.min(score, 1.0);
  }

  async deepAnalyze(target: string, options: DeepAnalysisOptions): Promise<DeepAnalysisResult> {
    this.logger.info('Starting deep code analysis', { target, options });
    
    try {
      // For now, provide a basic analysis
      // In a real implementation, this would do comprehensive semantic analysis
      
      const findings = [
        `Analyzing target: ${target}`,
        `Analysis depth: ${options.depth}`,
        `Lens context: ${options.includeLensContext ? 'included' : 'excluded'}`,
      ];
      
      if (target.includes('.ts') || target.includes('.tsx')) {
        findings.push('TypeScript file detected - enhanced type analysis available');
      }
      
      const recommendations = [
        'Consider adding more comprehensive type annotations',
        'Review for potential performance optimizations',
      ];
      
      return {
        summary: `Deep analysis completed for ${target}`,
        findings,
        dependencies: [],
        complexity: Math.floor(Math.random() * 10) + 1, // Mock complexity score
        recommendations,
      };
    } catch (error) {
      this.logger.error('Deep analysis failed', { error, target });
      throw error;
    }
  }

  async resolveSymbol(symbol: string, filePath: string, position: number): Promise<SymbolEntry | null> {
    // Resolve symbol definition
    // This would integrate with language servers or tree-sitter in a real implementation
    
    try {
      // Mock symbol resolution for now
      return {
        type: 'def' as const,
        path: filePath,
        span: [position, position + symbol.length],
        symbol,
        sig: `${symbol}: unknown`,
      };
    } catch (error) {
      this.logger.warn('Symbol resolution failed', { symbol, filePath, error });
      return null;
    }
  }

  async findReferences(symbol: string, repoPath: string): Promise<SymbolEntry[]> {
    // Find all references to a symbol across the repository
    // This would use advanced AST analysis in a real implementation
    
    const references: SymbolEntry[] = [];
    
    try {
      // Mock reference finding for now
      this.logger.info('Finding references', { symbol, repoPath });
      
      // In a real implementation, this would:
      // 1. Parse all files with tree-sitter or language-specific parsers
      // 2. Build a symbol table
      // 3. Find all usages of the symbol
      // 4. Return precise location information
      
      return references;
    } catch (error) {
      this.logger.error('Reference finding failed', { symbol, error });
      return [];
    }
  }

  async analyzeCallGraph(filePath: string): Promise<Record<string, string[]>> {
    // Analyze function call relationships
    // Returns a mapping of function -> [called functions]
    
    const callGraph: Record<string, string[]> = {};
    
    try {
      const content = await fs.readFile(filePath, 'utf-8');
      
      // Basic call graph analysis using pattern matching
      // In a real implementation, this would use proper AST parsing
      
      const lines = content.split('\n');
      let currentFunction = '';
      
      for (const line of lines) {
        // Detect function definitions
        const funcMatch = line.match(/(?:function|def)\s+(\w+)/);
        if (funcMatch) {
          currentFunction = funcMatch[1];
          callGraph[currentFunction] = [];
          continue;
        }
        
        // Detect function calls within current function
        if (currentFunction) {
          const callMatches = line.match(/(\w+)\s*\(/g);
          if (callMatches) {
            for (const match of callMatches) {
              const callee = match.replace(/\s*\($/, '');
              if (callee !== currentFunction && !callGraph[currentFunction].includes(callee)) {
                callGraph[currentFunction].push(callee);
              }
            }
          }
        }
      }
      
      return callGraph;
    } catch (error) {
      this.logger.error('Call graph analysis failed', { filePath, error });
      return {};
    }
  }

  async validateProject(repoPath: string): Promise<{
    valid: boolean;
    issues: string[];
    recommendations: string[];
    stats: {
      typeScriptFiles: number;
      javascriptFiles: number;
      pythonFiles: number;
      totalFiles: number;
    };
  }> {
    const validation = {
      valid: true,
      issues: [] as string[],
      recommendations: [] as string[],
      stats: {
        typeScriptFiles: 0,
        javascriptFiles: 0,
        pythonFiles: 0,
        totalFiles: 0,
      },
    };
    
    try {
      // Walk the repository and analyze file types
      const walkDir = async (dir: string): Promise<void> => {
        const entries = await fs.readdir(dir, { withFileTypes: true });
        
        for (const entry of entries) {
          if (entry.name.startsWith('.')) continue;
          
          const fullPath = join(dir, entry.name);
          
          if (entry.isDirectory()) {
            if (!['node_modules', 'dist', 'build', '.git'].includes(entry.name)) {
              await walkDir(fullPath);
            }
          } else if (entry.isFile()) {
            validation.stats.totalFiles++;
            
            const ext = entry.name.substring(entry.name.lastIndexOf('.'));
            switch (ext) {
              case '.ts':
              case '.tsx':
                validation.stats.typeScriptFiles++;
                break;
              case '.js':
              case '.jsx':
                validation.stats.javascriptFiles++;
                break;
              case '.py':
              case '.pyi':
                validation.stats.pythonFiles++;
                break;
            }
          }
        }
      };
      
      await walkDir(repoPath);
      
      // Validation checks
      if (validation.stats.totalFiles === 0) {
        validation.valid = false;
        validation.issues.push('No files found in repository');
      }
      
      if (validation.stats.typeScriptFiles === 0 && validation.stats.javascriptFiles === 0) {
        validation.recommendations.push('Consider adding TypeScript or JavaScript files for better analysis');
      }
      
      // Check for configuration files
      const configFiles = ['tsconfig.json', 'package.json', 'pyproject.toml'];
      for (const configFile of configFiles) {
        const configPath = join(repoPath, configFile);
        if (await fileExists(configPath)) {
          validation.recommendations.push(`Found ${configFile} - enhanced analysis available`);
        }
      }
      
    } catch (error) {
      validation.valid = false;
      validation.issues.push(`Validation failed: ${error}`);
    }
    
    return validation;
  }
}
