import { describe, it, expect, beforeEach, mock, spyOn } from 'bun:test';
import { LoadoutMetricsSystem } from '@/research/metrics-system';
import { createLogger } from '@/utils/logger';
import type { ResearchLoadout } from '@/research/types';
import type { Logger } from '@/types';

// Mock logger
const mockLogger: Logger = {
  debug: mock(() => {}),
  info: mock(() => {}),
  warn: mock(() => {}),
  error: mock(() => {}),
  child: mock(() => ({
    debug: mock(() => {}),
    info: mock(() => {}),
    warn: mock(() => {}),
    error: mock(() => {}),
    child: mock(() => {}),
  })),
};

// Mock LoadoutManager
const mockLoadoutManager = {
  getValidLoadouts: mock(() => []),
};

// Test loadout fixture
const createMockLoadout = (name: string = 'test-loadout'): ResearchLoadout => ({
  name,
  description: 'Test loadout',
  version: '1.0.0',
  pipeline: {
    maxIterations: 3,
    convergenceThreshold: 0.8,
    parallelismLevel: 2,
    timeoutMs: 30000,
    focus: 'intelligence_gathering',
    outputOptimization: 'swarm_handoff',
  },
  agents: [
    {
      type: 'intelligence_analyst',
      name: 'analyst-1',
      enabled: true,
      weight: 0.8,
      config: { specialization: ['typescript', 'security'] },
      dependencies: [],
    },
    {
      type: 'strategy_mapper',
      name: 'mapper-1',
      enabled: true,
      weight: 0.6,
      config: {},
      dependencies: ['analyst-1'],
    },
  ],
  stages: [
    {
      name: 'domain-analysis',
      type: 'domain_analysis',
      enabled: true,
      parallel: false,
      weight: 1.0,
      config: { focusAreas: ['refactoring'] },
    },
    {
      name: 'tech-analysis',
      type: 'technical_analysis',
      enabled: true,
      parallel: true,
      weight: 0.8,
      config: {},
    },
  ],
  verification: {
    syntactic: { enabled: true, weight: 0.3, config: {} },
    semantic: { enabled: true, weight: 0.4, config: {} },
    cross_reference: { 
      enabled: true, 
      weight: 0.3, 
      config: { checkFactualConsistency: true } 
    },
    logical_consistency: { enabled: false, weight: 0.0, config: {} },
  },
  output: {
    format: 'comprehensive',
    sections: ['executive-summary', 'technical-analysis', 'risk-assessment'],
    densityTarget: 3000,
    optimizeFor: 'swarm_consumption',
    includeDebateTopics: true,
    includeMetrics: true,
    includeCitations: false,
  },
  performance: {
    cacheResults: true,
    cacheTtlMinutes: 30,
    maxMemoryMb: 512,
    cpuIntensive: false,
  },
});

describe('LoadoutMetricsSystem', () => {
  let metricsSystem: LoadoutMetricsSystem;
  let createLoggerSpy: any;

  beforeEach(() => {
    // Mock createLogger to return our mockLogger
    createLoggerSpy = spyOn(require('@/utils/logger'), 'createLogger').mockReturnValue(mockLogger);
    mockLoadoutManager.getValidLoadouts.mockClear();
    metricsSystem = new LoadoutMetricsSystem(mockLoadoutManager as any);
  });

  describe('constructor', () => {
    it('should create logger with correct name', () => {
      expect(createLoggerSpy).toHaveBeenCalledWith('mimir.research.metrics');
    });

    it('should initialize empty metrics and benchmarks maps', () => {
      expect(metricsSystem.getAllMetrics().size).toBe(0);
      expect(metricsSystem.getAllBenchmarks().size).toBe(0);
    });
  });

  describe('initialize', () => {
    it('should process valid loadouts on initialization', async () => {
      const loadout = createMockLoadout();
      mockLoadoutManager.getValidLoadouts.mockReturnValue([loadout]);

      await metricsSystem.initialize();

      expect(mockLoadoutManager.getValidLoadouts).toHaveBeenCalled();
      expect(mockLogger.info).toHaveBeenCalledWith('Initializing loadout metrics system');
      expect(mockLogger.info).toHaveBeenCalledWith('Metrics system initialized', {
        loadoutsAnalyzed: 1,
        metricsGenerated: 1
      });
    });

    it('should handle empty loadout list', async () => {
      // Clear previous calls and explicitly set empty return
      mockLogger.info.mockClear();
      mockLoadoutManager.getValidLoadouts.mockReturnValue([]);
      
      await metricsSystem.initialize();

      expect(mockLogger.info).toHaveBeenCalledWith('Metrics system initialized', {
        loadoutsAnalyzed: 0,
        metricsGenerated: 0
      });
    });
  });

  describe('calculateMetrics', () => {
    it('should calculate comprehensive metrics for loadout', () => {
      const loadout = createMockLoadout();
      const metrics = metricsSystem.calculateMetrics(loadout);

      expect(metrics).toBeDefined();
      expect(metrics.name).toBe('test-loadout');
      expect(metrics.version).toBe('1.0.0');
      expect(metrics.complexity).toBeDefined();
      expect(metrics.performance).toBeDefined();
      expect(metrics.quality).toBeDefined();
      expect(metrics.resourceProfile).toBeDefined();
      expect(metrics.efficiency).toBeDefined();
      expect(metrics.outputProfile).toBeDefined();
      expect(metrics.specialization).toBeDefined();
      expect(typeof metrics.benchmarkScore).toBe('number');
    });

    it('should store metrics in internal map', () => {
      const loadout = createMockLoadout();
      metricsSystem.calculateMetrics(loadout);

      expect(metricsSystem.getMetrics('test-loadout')).toBeDefined();
    });

    it('should calculate complexity score correctly', () => {
      const loadout = createMockLoadout();
      const metrics = metricsSystem.calculateMetrics(loadout);

      expect(metrics.complexity.overall).toBeGreaterThan(0);
      expect(metrics.complexity.overall).toBeLessThanOrEqual(100);
      expect(metrics.complexity.breakdown.enabledAgents).toBe(2);
      expect(metrics.complexity.breakdown.enabledStages).toBe(2);
      expect(metrics.complexity.breakdown.verificationLayers).toBe(3);
    });

    it('should calculate performance profile correctly', () => {
      const loadout = createMockLoadout();
      const metrics = metricsSystem.calculateMetrics(loadout);

      expect(metrics.performance.parallelismScore).toBe(2);
      expect(metrics.performance.expectedSpeed).toBe('moderate'); // 3 iterations, 2 parallelism = moderate
      expect(metrics.performance.predictedMetrics.estimatedRuntimeMinutes).toBeGreaterThan(0);
      expect(metrics.performance.predictedMetrics.estimatedTokenUsage).toBeGreaterThan(0);
    });

    it('should calculate quality profile correctly', () => {
      const loadout = createMockLoadout();
      const metrics = metricsSystem.calculateMetrics(loadout);

      expect(metrics.quality.thoroughnessScore).toBeGreaterThan(0);
      expect(metrics.quality.accuracyPotential).toBeGreaterThan(0);
      expect(metrics.quality.qualityIndicators.verificationLayers).toBe(3);
      expect(metrics.quality.qualityIndicators.crossReferenceEnabled).toBe(true);
      expect(metrics.quality.qualityIndicators.factualVerification).toBe(true);
    });
  });

  describe('getMetrics', () => {
    it('should return metrics for existing loadout', () => {
      const loadout = createMockLoadout();
      metricsSystem.calculateMetrics(loadout);

      const metrics = metricsSystem.getMetrics('test-loadout');
      expect(metrics).toBeDefined();
      expect(metrics?.name).toBe('test-loadout');
    });

    it('should return null for non-existent loadout', () => {
      const metrics = metricsSystem.getMetrics('non-existent');
      expect(metrics).toBeNull();
    });
  });

  describe('getAllMetrics', () => {
    it('should return all calculated metrics', () => {
      const loadout1 = createMockLoadout('loadout-1');
      const loadout2 = createMockLoadout('loadout-2');
      
      metricsSystem.calculateMetrics(loadout1);
      metricsSystem.calculateMetrics(loadout2);

      const allMetrics = metricsSystem.getAllMetrics();
      expect(allMetrics.size).toBe(2);
      expect(allMetrics.has('loadout-1')).toBe(true);
      expect(allMetrics.has('loadout-2')).toBe(true);
    });
  });

  describe('compareLoadouts', () => {
    it('should compare multiple loadouts', () => {
      const loadout1 = createMockLoadout('loadout-1');
      const loadout2 = createMockLoadout('loadout-2');
      
      metricsSystem.calculateMetrics(loadout1);
      metricsSystem.calculateMetrics(loadout2);

      const comparison = metricsSystem.compareLoadouts(['loadout-1', 'loadout-2']);
      
      expect(comparison.loadouts).toEqual(['loadout-1', 'loadout-2']);
      expect(comparison.comparisonMatrix).toBeDefined();
      expect(comparison.recommendations).toBeDefined();
      expect(comparison.winnerByCategory).toBeDefined();
    });

    it('should handle empty loadout list', () => {
      const comparison = metricsSystem.compareLoadouts([]);
      
      expect(comparison.loadouts).toEqual([]);
      expect(comparison.comparisonMatrix).toEqual({});
      expect(comparison.recommendations).toEqual([]);
      expect(comparison.winnerByCategory).toEqual({});
    });

    it('should filter out non-existent loadouts', () => {
      const loadout = createMockLoadout('existing');
      metricsSystem.calculateMetrics(loadout);

      const comparison = metricsSystem.compareLoadouts(['existing', 'non-existent']);
      
      expect(Object.keys(comparison.comparisonMatrix)).toContain('existing');
      expect(Object.keys(comparison.comparisonMatrix)).not.toContain('non-existent');
    });

    it('should generate recommendations for different categories', () => {
      const loadout1 = createMockLoadout('fast-loadout');
      const loadout2 = createMockLoadout('quality-loadout');
      
      metricsSystem.calculateMetrics(loadout1);
      metricsSystem.calculateMetrics(loadout2);

      const comparison = metricsSystem.compareLoadouts(['fast-loadout', 'quality-loadout']);
      
      expect(comparison.recommendations.length).toBeGreaterThan(0);
      const categories = comparison.recommendations.map(r => r.category);
      expect(categories).toContain('speed');
      expect(categories).toContain('quality');
      expect(categories).toContain('general');
    });
  });

  describe('recordBenchmark', () => {
    it('should record benchmark for loadout', () => {
      const benchmark = {
        testCase: 'performance-test',
        metrics: {
          actualRuntime: 120,
          tokenUsage: 5000,
          qualityScore: 85,
          completenessScore: 90,
        },
        timestamp: new Date(),
      };

      metricsSystem.recordBenchmark('test-loadout', benchmark);
      
      const benchmarks = metricsSystem.getBenchmarks('test-loadout');
      expect(benchmarks).toHaveLength(1);
      expect(benchmarks[0].name).toBe('test-loadout');
      expect(benchmarks[0].testCase).toBe('performance-test');
      
      expect(mockLogger.info).toHaveBeenCalledWith('Recorded benchmark for loadout', {
        loadout: 'test-loadout',
        testCase: 'performance-test',
        qualityScore: 85,
      });
    });

    it('should accumulate multiple benchmarks', () => {
      const benchmark1 = {
        testCase: 'test-1',
        metrics: { actualRuntime: 100, tokenUsage: 3000, qualityScore: 80, completenessScore: 85 },
        timestamp: new Date(),
      };
      const benchmark2 = {
        testCase: 'test-2',
        metrics: { actualRuntime: 150, tokenUsage: 4000, qualityScore: 90, completenessScore: 95 },
        timestamp: new Date(),
      };

      metricsSystem.recordBenchmark('test-loadout', benchmark1);
      metricsSystem.recordBenchmark('test-loadout', benchmark2);
      
      const benchmarks = metricsSystem.getBenchmarks('test-loadout');
      expect(benchmarks).toHaveLength(2);
    });
  });

  describe('getBenchmarks', () => {
    it('should return empty array for loadout with no benchmarks', () => {
      const benchmarks = metricsSystem.getBenchmarks('non-existent');
      expect(benchmarks).toEqual([]);
    });
  });

  describe('getAllBenchmarks', () => {
    it('should return all recorded benchmarks', () => {
      const benchmark = {
        testCase: 'test',
        metrics: { actualRuntime: 100, tokenUsage: 3000, qualityScore: 80, completenessScore: 85 },
        timestamp: new Date(),
      };

      metricsSystem.recordBenchmark('loadout-1', benchmark);
      metricsSystem.recordBenchmark('loadout-2', benchmark);
      
      const allBenchmarks = metricsSystem.getAllBenchmarks();
      expect(allBenchmarks.size).toBe(2);
      expect(allBenchmarks.has('loadout-1')).toBe(true);
      expect(allBenchmarks.has('loadout-2')).toBe(true);
    });
  });

  describe('generateReport', () => {
    it('should generate report for specified loadouts', () => {
      const loadout = createMockLoadout();
      metricsSystem.calculateMetrics(loadout);

      const report = metricsSystem.generateReport(['test-loadout']);
      
      expect(report).toContain('# Loadout Metrics Report');
      expect(report).toContain('test-loadout');
      expect(report).toContain('Benchmark Score');
      expect(report).toContain('Performance Profile');
      expect(report).toContain('Quality Profile');
    });

    it('should generate report for all loadouts when none specified', () => {
      const loadout1 = createMockLoadout('loadout-1');
      const loadout2 = createMockLoadout('loadout-2');
      
      metricsSystem.calculateMetrics(loadout1);
      metricsSystem.calculateMetrics(loadout2);

      const report = metricsSystem.generateReport();
      
      expect(report).toContain('loadout-1');
      expect(report).toContain('loadout-2');
    });

    it('should handle no metrics available', () => {
      const report = metricsSystem.generateReport(['non-existent']);
      
      expect(report).toBe('No metrics available for the specified loadouts.');
    });

    it('should include optimization suggestions when available', () => {
      const loadout = createMockLoadout();
      // Add a third agent to ensure >2 agents condition is met
      loadout.agents.push({
        type: 'risk_analyst',
        name: 'risk-1',
        enabled: true,
        weight: 0.5,
        config: {},
        dependencies: [],
      });
      
      // Create a loadout that will trigger optimization suggestions
      loadout.pipeline.maxIterations = 6; // High iterations
      loadout.pipeline.parallelismLevel = 1; // Low parallelism (< 3) with >2 agents
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      
      // Check that optimization suggestions are generated (should trigger parallelism suggestion)
      expect(metrics.efficiency.optimizationSuggestions.length).toBeGreaterThan(0);
      expect(metrics.efficiency.optimizationSuggestions).toContain('Consider increasing parallelismLevel for better performance');
      
      const report = metricsSystem.generateReport(['test-loadout']);
      
      // Should contain optimization section due to suggestions
      expect(report).toContain('Optimization Suggestions:');
    });

    it('should include best use cases and avoid cases', () => {
      const loadout = createMockLoadout();
      metricsSystem.calculateMetrics(loadout);

      const report = metricsSystem.generateReport(['test-loadout']);
      
      expect(report).toContain('Best Use Cases');
      expect(report).toContain('Avoid For');
    });
  });

  describe('performance categorization', () => {
    it('should categorize very-fast performance correctly', () => {
      const loadout = createMockLoadout();
      loadout.pipeline.maxIterations = 2;
      loadout.pipeline.parallelismLevel = 5;
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.performance.expectedSpeed).toBe('very-fast');
    });

    it('should categorize very-slow performance correctly', () => {
      const loadout = createMockLoadout();
      loadout.pipeline.maxIterations = 8;
      loadout.pipeline.parallelismLevel = 1;
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.performance.expectedSpeed).toBe('very-slow');
    });

    it('should calculate resource intensity correctly', () => {
      const loadout = createMockLoadout();
      loadout.agents = loadout.agents.slice(0, 1); // Single agent
      loadout.pipeline.parallelismLevel = 1;
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.performance.resourceIntensity).toBe('low');
    });
  });

  describe('specialization analysis', () => {
    it('should detect software development domain', () => {
      const loadout = createMockLoadout();
      loadout.agents[0].config.specialization = ['typescript', 'javascript'];
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.specialization.domain).toContain('software-development');
    });

    it('should detect security domain', () => {
      const loadout = createMockLoadout();
      loadout.agents[0].config.specialization = ['security', 'vulnerabilities'];
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.specialization.domain).toContain('security');
    });

    it('should default to general domain when no specific domains detected', () => {
      const loadout = createMockLoadout();
      // Remove all specializations and focus areas that would trigger domain detection
      loadout.agents[0].config.specialization = ['unknown', 'other'];
      loadout.stages[0].config = {}; // Remove focusAreas that contained 'refactoring'
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.specialization.domain).toContain('general');
    });

    it('should determine expertise level based on specialized agents ratio', () => {
      const loadout = createMockLoadout();
      // All agents have specialization
      loadout.agents.forEach(agent => {
        agent.config.specialization = ['typescript'];
      });
      
      const metrics = metricsSystem.calculateMetrics(loadout);
      expect(metrics.specialization.expertiseLevel).toBe('niche');
    });
  });
});