/**
 * Loadout Metrics and Analysis System
 * 
 * Provides comprehensive metrics, benchmarking, and analysis capabilities
 * for research loadout configurations. Enables data-driven loadout optimization.
 */

import type { Logger } from '@/types';
import { createLogger } from '@/utils/logger';
import type { 
  ResearchLoadout, 
  IntelligenceGatheringResult, 
  SwarmIntelligenceBrief 
} from './types';
import type { LoadoutManager } from './loadout-manager';

export interface LoadoutMetrics {
  // Basic Configuration Metrics
  name: string;
  version: string;
  complexity: LoadoutComplexityScore;
  performance: LoadoutPerformanceProfile;
  quality: LoadoutQualityProfile;
  
  // Resource and Efficiency Metrics
  resourceProfile: ResourceProfile;
  efficiency: EfficiencyMetrics;
  
  // Content and Output Metrics
  outputProfile: OutputProfile;
  specialization: SpecializationProfile;
  
  // Comparative Metrics
  benchmarkScore: number;
  rankingPosition?: number;
  percentile?: number;
}

export interface LoadoutComplexityScore {
  overall: number; // 1-100 scale
  agentComplexity: number;
  stageComplexity: number;
  verificationComplexity: number;
  configurationComplexity: number;
  breakdown: {
    totalAgents: number;
    enabledAgents: number;
    totalStages: number;
    enabledStages: number;
    verificationLayers: number;
    dependencies: number;
    configurationOptions: number;
  };
}

export interface LoadoutPerformanceProfile {
  expectedSpeed: 'very-fast' | 'fast' | 'moderate' | 'slow' | 'very-slow';
  parallelismScore: number; // 1-10 scale
  iterationEfficiency: number; // iterations vs results ratio
  convergenceSpeed: number; // how quickly it reaches conclusions
  resourceIntensity: 'low' | 'moderate' | 'high' | 'very-high';
  
  predictedMetrics: {
    estimatedRuntimeMinutes: number;
    estimatedTokenUsage: number;
    estimatedMemoryMB: number;
    concurrentOperations: number;
  };
}

export interface LoadoutQualityProfile {
  thoroughnessScore: number; // 1-100 scale
  accuracyPotential: number; // based on verification layers
  comprehensivenessScore: number; // breadth of analysis
  reliabilityScore: number; // consistency of results
  
  qualityIndicators: {
    verificationLayers: number;
    crossReferenceEnabled: boolean;
    logicalConsistencyCheck: boolean;
    semanticValidation: boolean;
    factualVerification: boolean;
  };
}

export interface ResourceProfile {
  tokenBudget: {
    estimated: number;
    perAgent: number;
    perStage: number;
    outputTokens: number;
  };
  
  computeIntensity: {
    level: 'light' | 'moderate' | 'intensive' | 'extreme';
    cpuScore: number;
    parallelWorkload: number;
    memoryRequirements: number;
  };
  
  timeProfile: {
    estimatedMinutes: number;
    fastestPossible: number;
    slowestExpected: number;
    convergenceTime: number;
  };
}

export interface EfficiencyMetrics {
  tokenEfficiency: number; // output value per token spent
  timeEfficiency: number; // output value per minute spent
  qualityEfficiency: number; // quality score per resource unit
  
  costBenefitScore: number; // overall value proposition
  
  optimizationSuggestions: string[];
}

export interface OutputProfile {
  targetDensity: number;
  estimatedSections: number;
  comprehensiveness: number;
  debateReadiness: number; // optimized for swarm consumption
  
  contentTypes: {
    executiveSummary: boolean;
    technicalAnalysis: boolean;
    strategicGuidance: boolean;
    riskAssessment: boolean;
    debateFramework: boolean;
    metrics: boolean;
    citations: boolean;
  };
}

export interface SpecializationProfile {
  domain: string[];
  focusAreas: string[];
  expertiseLevel: 'generalist' | 'specialized' | 'expert' | 'niche';
  
  strengths: string[];
  limitations: string[];
  bestUseCases: string[];
  avoidUseCases: string[];
}

export interface LoadoutBenchmark {
  name: string;
  testCase: string;
  metrics: {
    actualRuntime: number;
    tokenUsage: number;
    qualityScore: number;
    completenessScore: number;
  };
  timestamp: Date;
}

export interface LoadoutComparison {
  loadouts: string[];
  comparisonMatrix: ComparisonMatrix;
  recommendations: LoadoutRecommendation[];
  winnerByCategory: Record<string, string>;
}

export interface ComparisonMatrix {
  [loadoutName: string]: {
    [metric: string]: {
      value: number;
      rank: number;
      percentile: number;
    };
  };
}

export interface LoadoutRecommendation {
  category: 'speed' | 'quality' | 'cost' | 'specialization' | 'general';
  loadout: string;
  reason: string;
  confidenceScore: number;
  tradeoffs: string[];
}

export class LoadoutMetricsSystem {
  private metrics = new Map<string, LoadoutMetrics>();
  private benchmarks = new Map<string, LoadoutBenchmark[]>();
  private logger: Logger;

  constructor(private loadoutManager: LoadoutManager) {
    this.logger = createLogger('mimir.research.metrics');
  }

  async initialize(): Promise<void> {
    this.logger.info('Initializing loadout metrics system');
    
    // Calculate metrics for all valid loadouts
    const validLoadouts = this.loadoutManager.getValidLoadouts();
    for (const loadout of validLoadouts) {
      this.calculateMetrics(loadout);
    }

    this.logger.info('Metrics system initialized', {
      loadoutsAnalyzed: validLoadouts.length,
      metricsGenerated: this.metrics.size
    });
  }

  calculateMetrics(loadout: ResearchLoadout): LoadoutMetrics {
    const complexity = this.calculateComplexityScore(loadout);
    const performance = this.calculatePerformanceProfile(loadout);
    const quality = this.calculateQualityProfile(loadout);
    const resourceProfile = this.calculateResourceProfile(loadout);
    const efficiency = this.calculateEfficiencyMetrics(loadout, performance, quality, resourceProfile);
    const outputProfile = this.calculateOutputProfile(loadout);
    const specialization = this.calculateSpecializationProfile(loadout);
    const benchmarkScore = this.calculateBenchmarkScore(loadout, complexity, performance, quality);

    const metrics: LoadoutMetrics = {
      name: loadout.name,
      version: loadout.version,
      complexity,
      performance,
      quality,
      resourceProfile,
      efficiency,
      outputProfile,
      specialization,
      benchmarkScore,
    };

    this.metrics.set(loadout.name, metrics);
    return metrics;
  }

  private calculateComplexityScore(loadout: ResearchLoadout): LoadoutComplexityScore {
    const enabledAgents = loadout.agents.filter(a => a.enabled);
    const enabledStages = loadout.stages.filter(s => s.enabled);
    const verificationLayers = Object.values(loadout.verification).filter(v => v.enabled).length;
    
    // Count dependencies
    let dependencies = 0;
    enabledAgents.forEach(agent => {
      dependencies += agent.dependencies?.length || 0;
    });

    // Count configuration complexity
    let configOptions = 0;
    enabledAgents.forEach(agent => {
      configOptions += Object.keys(agent.config).length;
    });
    enabledStages.forEach(stage => {
      configOptions += Object.keys(stage.config || {}).length;
    });

    const agentComplexity = Math.min(100, enabledAgents.length * 15 + dependencies * 5);
    const stageComplexity = Math.min(100, enabledStages.length * 20 + (loadout.pipeline.parallelismLevel * 5));
    const verificationComplexity = Math.min(100, verificationLayers * 20);
    const configurationComplexity = Math.min(100, configOptions * 2);

    const overall = Math.min(100, (agentComplexity + stageComplexity + verificationComplexity + configurationComplexity) / 4);

    return {
      overall,
      agentComplexity,
      stageComplexity,
      verificationComplexity,
      configurationComplexity,
      breakdown: {
        totalAgents: loadout.agents.length,
        enabledAgents: enabledAgents.length,
        totalStages: loadout.stages.length,
        enabledStages: enabledStages.length,
        verificationLayers,
        dependencies,
        configurationOptions: configOptions,
      },
    };
  }

  private calculatePerformanceProfile(loadout: ResearchLoadout): LoadoutPerformanceProfile {
    const { pipeline, performance } = loadout;
    
    // Calculate parallelism score (1-10)
    const parallelismScore = Math.min(10, pipeline.parallelismLevel);
    
    // Estimate speed category
    let expectedSpeed: LoadoutPerformanceProfile['expectedSpeed'];
    if (pipeline.maxIterations <= 2 && pipeline.parallelismLevel >= 4) {
      expectedSpeed = 'very-fast';
    } else if (pipeline.maxIterations <= 3 && pipeline.parallelismLevel >= 3) {
      expectedSpeed = 'fast';
    } else if (pipeline.maxIterations <= 4 && pipeline.parallelismLevel >= 2) {
      expectedSpeed = 'moderate';
    } else if (pipeline.maxIterations <= 5) {
      expectedSpeed = 'slow';
    } else {
      expectedSpeed = 'very-slow';
    }

    // Calculate efficiency scores
    const iterationEfficiency = Math.max(1, 10 - pipeline.maxIterations);
    const convergenceSpeed = pipeline.convergenceThreshold * 10;
    
    // Resource intensity
    let resourceIntensity: LoadoutPerformanceProfile['resourceIntensity'];
    const enabledAgents = loadout.agents.filter(a => a.enabled).length;
    if (enabledAgents <= 2 && pipeline.parallelismLevel <= 2) {
      resourceIntensity = 'low';
    } else if (enabledAgents <= 4 && pipeline.parallelismLevel <= 4) {
      resourceIntensity = 'moderate';
    } else if (enabledAgents <= 6 && pipeline.parallelismLevel <= 6) {
      resourceIntensity = 'high';
    } else {
      resourceIntensity = 'very-high';
    }

    // Predicted metrics
    const estimatedRuntimeMinutes = Math.max(1, 
      (pipeline.maxIterations * enabledAgents * 2) / pipeline.parallelismLevel
    );
    const estimatedTokenUsage = enabledAgents * 3000 * pipeline.maxIterations;
    const estimatedMemoryMB = performance?.maxMemoryMb || (enabledAgents * 256);
    const concurrentOperations = pipeline.parallelismLevel;

    return {
      expectedSpeed,
      parallelismScore,
      iterationEfficiency,
      convergenceSpeed,
      resourceIntensity,
      predictedMetrics: {
        estimatedRuntimeMinutes,
        estimatedTokenUsage,
        estimatedMemoryMB,
        concurrentOperations,
      },
    };
  }

  private calculateQualityProfile(loadout: ResearchLoadout): LoadoutQualityProfile {
    const verification = loadout.verification;
    const verificationLayers = Object.values(verification).filter(v => v.enabled).length;
    
    // Quality indicators
    const qualityIndicators = {
      verificationLayers,
      crossReferenceEnabled: verification.cross_reference?.enabled || false,
      logicalConsistencyCheck: verification.logical_consistency?.enabled || false,
      semanticValidation: verification.semantic?.enabled || false,
      factualVerification: verification.cross_reference?.config?.checkFactualConsistency || false,
    };

    // Calculate scores
    const thoroughnessScore = Math.min(100, 
      (loadout.pipeline.maxIterations * 15) + 
      (verificationLayers * 10) +
      (loadout.stages.filter(s => s.enabled).length * 5)
    );

    const accuracyPotential = Math.min(100, verificationLayers * 20 + 20);
    
    const comprehensivenessScore = Math.min(100,
      loadout.agents.filter(a => a.enabled).length * 15 +
      (loadout.output.sections?.length || 0) * 10
    );

    const reliabilityScore = Math.min(100,
      (qualityIndicators.logicalConsistencyCheck ? 25 : 0) +
      (qualityIndicators.crossReferenceEnabled ? 25 : 0) +
      (qualityIndicators.semanticValidation ? 25 : 0) +
      (verificationLayers >= 3 ? 25 : verificationLayers * 8)
    );

    return {
      thoroughnessScore,
      accuracyPotential,
      comprehensivenessScore,
      reliabilityScore,
      qualityIndicators,
    };
  }

  private calculateResourceProfile(loadout: ResearchLoadout): ResourceProfile {
    const enabledAgents = loadout.agents.filter(a => a.enabled);
    const enabledStages = loadout.stages.filter(s => s.enabled);
    
    // Token budget estimation
    const perAgent = 2500; // average tokens per agent
    const perStage = 1000; // average tokens per stage
    const outputTokens = loadout.output.densityTarget || 3000;
    const estimated = (enabledAgents.length * perAgent * loadout.pipeline.maxIterations) + 
                     (enabledStages.length * perStage) + outputTokens;

    // Compute intensity
    let level: ResourceProfile['computeIntensity']['level'];
    const complexity = enabledAgents.length + enabledStages.length + loadout.pipeline.parallelismLevel;
    if (complexity <= 5) level = 'light';
    else if (complexity <= 10) level = 'moderate';
    else if (complexity <= 15) level = 'intensive';
    else level = 'extreme';

    const cpuScore = Math.min(10, complexity);
    const parallelWorkload = loadout.pipeline.parallelismLevel;
    const memoryRequirements = loadout.performance?.maxMemoryMb || (enabledAgents.length * 256);

    // Time profile
    const estimatedMinutes = Math.max(1, 
      (loadout.pipeline.maxIterations * enabledAgents.length * 1.5) / loadout.pipeline.parallelismLevel
    );
    const fastestPossible = estimatedMinutes * 0.7;
    const slowestExpected = estimatedMinutes * 2;
    const convergenceTime = estimatedMinutes * loadout.pipeline.convergenceThreshold;

    return {
      tokenBudget: {
        estimated,
        perAgent,
        perStage,
        outputTokens,
      },
      computeIntensity: {
        level,
        cpuScore,
        parallelWorkload,
        memoryRequirements,
      },
      timeProfile: {
        estimatedMinutes,
        fastestPossible,
        slowestExpected,
        convergenceTime,
      },
    };
  }

  private calculateEfficiencyMetrics(
    loadout: ResearchLoadout,
    performance: LoadoutPerformanceProfile,
    quality: LoadoutQualityProfile,
    resourceProfile: ResourceProfile
  ): EfficiencyMetrics {
    // Calculate efficiency ratios
    const tokenEfficiency = quality.thoroughnessScore / (resourceProfile.tokenBudget.estimated / 1000);
    const timeEfficiency = quality.thoroughnessScore / resourceProfile.timeProfile.estimatedMinutes;
    const qualityEfficiency = (quality.thoroughnessScore + quality.reliabilityScore) / 
                             (resourceProfile.tokenBudget.estimated / 1000 + resourceProfile.timeProfile.estimatedMinutes);

    // Overall cost-benefit score
    const costBenefitScore = Math.min(100, 
      (tokenEfficiency * 0.3) + 
      (timeEfficiency * 0.3) + 
      (qualityEfficiency * 0.4)
    );

    // Generate optimization suggestions
    const optimizationSuggestions: string[] = [];
    
    if (performance.parallelismScore < 3 && loadout.agents.filter(a => a.enabled).length > 2) {
      optimizationSuggestions.push('Consider increasing parallelismLevel for better performance');
    }
    
    if (loadout.pipeline.maxIterations > 4 && quality.reliabilityScore < 70) {
      optimizationSuggestions.push('High iterations with low reliability - review verification settings');
    }
    
    if (resourceProfile.tokenBudget.estimated > 50000 && quality.thoroughnessScore < 80) {
      optimizationSuggestions.push('High token usage with moderate quality - optimize agent configurations');
    }
    
    if (performance.expectedSpeed === 'very-slow' && quality.thoroughnessScore < 90) {
      optimizationSuggestions.push('Slow performance without exceptional quality - consider simpler loadout');
    }

    return {
      tokenEfficiency,
      timeEfficiency,
      qualityEfficiency,
      costBenefitScore,
      optimizationSuggestions,
    };
  }

  private calculateOutputProfile(loadout: ResearchLoadout): OutputProfile {
    const output = loadout.output;
    
    const contentTypes = {
      executiveSummary: output.sections?.some(s => s.includes('executive') || s.includes('summary')) || false,
      technicalAnalysis: output.sections?.some(s => s.includes('technical') || s.includes('analysis')) || false,
      strategicGuidance: output.sections?.some(s => s.includes('strategic') || s.includes('strategy')) || false,
      riskAssessment: output.sections?.some(s => s.includes('risk')) || false,
      debateFramework: output.includeDebateTopics || false,
      metrics: output.includeMetrics || false,
      citations: output.includeCitations || false,
    };

    const comprehensiveness = Math.min(100, (output.sections?.length || 0) * 12);
    const debateReadiness = output.optimizeFor === 'swarm_consumption' ? 100 : 
                           output.includeDebateTopics ? 75 : 25;

    return {
      targetDensity: output.densityTarget || 3000,
      estimatedSections: output.sections?.length || 0,
      comprehensiveness,
      debateReadiness,
      contentTypes,
    };
  }

  private calculateSpecializationProfile(loadout: ResearchLoadout): SpecializationProfile {
    // Extract focus areas from agents and stages
    const focusAreas: string[] = [];
    const domain: string[] = [];
    
    loadout.agents.forEach(agent => {
      if (agent.config.specialization) {
        focusAreas.push(...agent.config.specialization);
      }
    });
    
    loadout.stages.forEach(stage => {
      if (stage.config?.focusAreas) {
        focusAreas.push(...stage.config.focusAreas);
      }
    });

    // Determine domain from loadout description and focus areas
    if (focusAreas.some(area => ['typescript', 'javascript', 'frontend', 'backend'].includes(area.toLowerCase()))) {
      domain.push('software-development');
    }
    if (focusAreas.some(area => ['security', 'vulnerabilities', 'threat'].includes(area.toLowerCase()))) {
      domain.push('security');
    }
    if (focusAreas.some(area => ['refactoring', 'technical_debt', 'code_quality'].includes(area.toLowerCase()))) {
      domain.push('code-quality');
    }
    if (domain.length === 0) {
      domain.push('general');
    }

    // Determine expertise level
    let expertiseLevel: SpecializationProfile['expertiseLevel'];
    const specializedAgents = loadout.agents.filter(a => a.config.specialization?.length).length;
    const totalAgents = loadout.agents.filter(a => a.enabled).length;
    
    if (specializedAgents === 0) expertiseLevel = 'generalist';
    else if (specializedAgents / totalAgents < 0.5) expertiseLevel = 'specialized';
    else if (specializedAgents / totalAgents < 0.8) expertiseLevel = 'expert';
    else expertiseLevel = 'niche';

    // Generate strengths and limitations
    const strengths: string[] = [];
    const limitations: string[] = [];
    const bestUseCases: string[] = [];
    const avoidUseCases: string[] = [];

    if (expertiseLevel === 'niche' || expertiseLevel === 'expert') {
      strengths.push('Deep domain expertise', 'Highly specialized analysis');
      limitations.push('Limited scope', 'May miss cross-domain insights');
      bestUseCases.push('Domain-specific problems', 'Expert-level analysis needed');
      avoidUseCases.push('General research', 'Cross-domain analysis');
    } else if (expertiseLevel === 'generalist') {
      strengths.push('Broad coverage', 'Cross-domain insights');
      limitations.push('Less specialized depth', 'May lack domain expertise');
      bestUseCases.push('General analysis', 'Exploratory research');
      avoidUseCases.push('Highly specialized domains', 'Expert-level analysis');
    }

    return {
      domain,
      focusAreas: [...new Set(focusAreas)],
      expertiseLevel,
      strengths,
      limitations,
      bestUseCases,
      avoidUseCases,
    };
  }

  private calculateBenchmarkScore(
    loadout: ResearchLoadout,
    complexity: LoadoutComplexityScore,
    performance: LoadoutPerformanceProfile,
    quality: LoadoutQualityProfile
  ): number {
    // Weighted composite score
    const weights = {
      quality: 0.4,
      efficiency: 0.3,
      performance: 0.2,
      complexity: 0.1,
    };

    const qualityScore = (quality.thoroughnessScore + quality.reliabilityScore + quality.accuracyPotential) / 3;
    const efficiencyScore = performance.iterationEfficiency * 10;
    const performanceScore = performance.parallelismScore * 10;
    const complexityScore = 100 - (complexity.overall * 0.5); // Lower complexity is better for benchmark

    return Math.min(100,
      (qualityScore * weights.quality) +
      (efficiencyScore * weights.efficiency) +
      (performanceScore * weights.performance) +
      (complexityScore * weights.complexity)
    );
  }

  getMetrics(loadoutName: string): LoadoutMetrics | null {
    return this.metrics.get(loadoutName) || null;
  }

  getAllMetrics(): Map<string, LoadoutMetrics> {
    return new Map(this.metrics);
  }

  compareLoadouts(loadoutNames: string[]): LoadoutComparison {
    const loadoutMetrics = loadoutNames
      .map(name => this.metrics.get(name))
      .filter(Boolean) as LoadoutMetrics[];

    if (loadoutMetrics.length === 0) {
      return {
        loadouts: [],
        comparisonMatrix: {},
        recommendations: [],
        winnerByCategory: {},
      };
    }

    // Build comparison matrix
    const comparisonMatrix: ComparisonMatrix = {};
    const metrics = [
      'benchmarkScore',
      'complexity.overall',
      'quality.thoroughnessScore',
      'performance.parallelismScore',
      'efficiency.costBenefitScore',
    ];

    loadoutMetrics.forEach(metric => {
      comparisonMatrix[metric.name] = {};
    });

    metrics.forEach(metricPath => {
      const values = loadoutMetrics.map(m => this.getNestedValue(m, metricPath));
      const sorted = [...values].sort((a, b) => b - a);
      
      loadoutMetrics.forEach((metric, index) => {
        const value = values[index];
        const rank = sorted.indexOf(value) + 1;
        const percentile = Math.round(((sorted.length - rank + 1) / sorted.length) * 100);
        
        comparisonMatrix[metric.name][metricPath] = {
          value,
          rank,
          percentile,
        };
      });
    });

    // Generate recommendations
    const recommendations = this.generateRecommendations(loadoutMetrics, comparisonMatrix);

    // Determine winners by category
    const winnerByCategory: Record<string, string> = {};
    metrics.forEach(metric => {
      const winner = loadoutMetrics.reduce((best, current) => {
        const bestValue = comparisonMatrix[best.name][metric].value;
        const currentValue = comparisonMatrix[current.name][metric].value;
        return currentValue > bestValue ? current : best;
      });
      winnerByCategory[metric] = winner.name;
    });

    return {
      loadouts: loadoutNames,
      comparisonMatrix,
      recommendations,
      winnerByCategory,
    };
  }

  private getNestedValue(obj: any, path: string): number {
    return path.split('.').reduce((current, key) => current?.[key], obj) || 0;
  }

  private generateRecommendations(
    loadoutMetrics: LoadoutMetrics[],
    comparisonMatrix: ComparisonMatrix
  ): LoadoutRecommendation[] {
    const recommendations: LoadoutRecommendation[] = [];

    // Find best for speed
    const fastest = loadoutMetrics.reduce((best, current) => 
      current.performance.expectedSpeed === 'very-fast' || 
      (current.performance.expectedSpeed === 'fast' && best.performance.expectedSpeed !== 'very-fast')
        ? current : best
    );
    recommendations.push({
      category: 'speed',
      loadout: fastest.name,
      reason: `Optimized for speed with ${fastest.performance.expectedSpeed} performance and ${fastest.performance.parallelismScore}/10 parallelism`,
      confidenceScore: 85,
      tradeoffs: fastest.performance.expectedSpeed === 'very-fast' ? ['May sacrifice some thoroughness for speed'] : [],
    });

    // Find best for quality
    const highestQuality = loadoutMetrics.reduce((best, current) => 
      current.quality.thoroughnessScore > best.quality.thoroughnessScore ? current : best
    );
    recommendations.push({
      category: 'quality',
      loadout: highestQuality.name,
      reason: `Highest quality score: ${Math.round(highestQuality.quality.thoroughnessScore)} thoroughness, ${Math.round(highestQuality.quality.reliabilityScore)} reliability`,
      confidenceScore: 90,
      tradeoffs: highestQuality.performance.expectedSpeed === 'slow' || highestQuality.performance.expectedSpeed === 'very-slow' 
        ? ['Higher quality comes with longer processing time'] : [],
    });

    // Find best cost-efficiency
    const mostEfficient = loadoutMetrics.reduce((best, current) => 
      current.efficiency.costBenefitScore > best.efficiency.costBenefitScore ? current : best
    );
    recommendations.push({
      category: 'cost',
      loadout: mostEfficient.name,
      reason: `Best cost-benefit ratio: ${Math.round(mostEfficient.efficiency.costBenefitScore)}/100 efficiency score`,
      confidenceScore: 80,
      tradeoffs: [],
    });

    // Find best specialist
    const mostSpecialized = loadoutMetrics.reduce((best, current) => 
      current.specialization.expertiseLevel === 'niche' || 
      (current.specialization.expertiseLevel === 'expert' && best.specialization.expertiseLevel !== 'niche')
        ? current : best
    );
    if (mostSpecialized.specialization.expertiseLevel !== 'generalist') {
      recommendations.push({
        category: 'specialization',
        loadout: mostSpecialized.name,
        reason: `${mostSpecialized.specialization.expertiseLevel} expertise in ${mostSpecialized.specialization.domain.join(', ')}`,
        confidenceScore: 75,
        tradeoffs: ['Limited to specific domains', 'May not suit general analysis'],
      });
    }

    // Find best general-purpose
    const bestGeneral = loadoutMetrics.reduce((best, current) => 
      current.benchmarkScore > best.benchmarkScore ? current : best
    );
    recommendations.push({
      category: 'general',
      loadout: bestGeneral.name,
      reason: `Highest overall benchmark score: ${Math.round(bestGeneral.benchmarkScore)}/100`,
      confidenceScore: 95,
      tradeoffs: [],
    });

    return recommendations;
  }

  recordBenchmark(loadoutName: string, benchmark: Omit<LoadoutBenchmark, 'name'>): void {
    if (!this.benchmarks.has(loadoutName)) {
      this.benchmarks.set(loadoutName, []);
    }
    
    this.benchmarks.get(loadoutName)!.push({
      name: loadoutName,
      ...benchmark,
    });

    this.logger.info('Recorded benchmark for loadout', {
      loadout: loadoutName,
      testCase: benchmark.testCase,
      qualityScore: benchmark.metrics.qualityScore,
    });
  }

  getBenchmarks(loadoutName: string): LoadoutBenchmark[] {
    return this.benchmarks.get(loadoutName) || [];
  }

  getAllBenchmarks(): Map<string, LoadoutBenchmark[]> {
    return new Map(this.benchmarks);
  }

  generateReport(loadoutNames?: string[]): string {
    const targetLoadouts = loadoutNames || Array.from(this.metrics.keys());
    const metrics = targetLoadouts.map(name => this.metrics.get(name)).filter(Boolean) as LoadoutMetrics[];

    if (metrics.length === 0) {
      return 'No metrics available for the specified loadouts.';
    }

    let report = '# Loadout Metrics Report\n\n';
    
    // Summary table
    report += '## Summary\n\n';
    report += '| Loadout | Benchmark Score | Quality | Performance | Efficiency | Specialization |\n';
    report += '|---------|----------------|---------|-------------|------------|---------------|\n';
    
    metrics.forEach(metric => {
      report += `| ${metric.name} | ${Math.round(metric.benchmarkScore)} | ${Math.round(metric.quality.thoroughnessScore)} | ${metric.performance.expectedSpeed} | ${Math.round(metric.efficiency.costBenefitScore)} | ${metric.specialization.expertiseLevel} |\n`;
    });

    // Detailed analysis for each loadout
    report += '\n## Detailed Analysis\n\n';
    metrics.forEach(metric => {
      report += `### ${metric.name}\n\n`;
      report += `**Version:** ${metric.version}\n`;
      report += `**Benchmark Score:** ${Math.round(metric.benchmarkScore)}/100\n\n`;
      
      report += '**Performance Profile:**\n';
      report += `- Speed: ${metric.performance.expectedSpeed}\n`;
      report += `- Parallelism: ${metric.performance.parallelismScore}/10\n`;
      report += `- Estimated Runtime: ${Math.round(metric.resourceProfile.timeProfile.estimatedMinutes)} minutes\n`;
      report += `- Token Usage: ~${Math.round(metric.resourceProfile.tokenBudget.estimated / 1000)}k tokens\n\n`;
      
      report += '**Quality Profile:**\n';
      report += `- Thoroughness: ${Math.round(metric.quality.thoroughnessScore)}/100\n`;
      report += `- Reliability: ${Math.round(metric.quality.reliabilityScore)}/100\n`;
      report += `- Verification Layers: ${metric.quality.qualityIndicators.verificationLayers}\n\n`;
      
      if (metric.efficiency.optimizationSuggestions.length > 0) {
        report += '**Optimization Suggestions:**\n';
        metric.efficiency.optimizationSuggestions.forEach(suggestion => {
          report += `- ${suggestion}\n`;
        });
        report += '\n';
      }
      
      report += '**Best Use Cases:**\n';
      metric.specialization.bestUseCases.forEach(useCase => {
        report += `- ${useCase}\n`;
      });
      report += '\n';
      
      if (metric.specialization.avoidUseCases.length > 0) {
        report += '**Avoid For:**\n';
        metric.specialization.avoidUseCases.forEach(avoidCase => {
          report += `- ${avoidCase}\n`;
        });
        report += '\n';
      }
      
      report += '---\n\n';
    });

    return report;
  }
}