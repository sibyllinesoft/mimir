/**
 * Types for configurable intelligence-gathering research pipeline
 * 
 * This system focuses on comprehensive intelligence gathering to feed
 * into refinement swarms, not solution generation itself.
 */

import { z } from 'zod';

// ============================================================================
// CORE RESEARCH PIPELINE TYPES
// ============================================================================

export interface ResearchLoadout {
  name: string;
  description: string;
  version: string;
  
  // Core pipeline configuration
  pipeline: PipelineConfig;
  
  // Agent configuration for intelligence gathering
  agents: AgentConfig[];
  
  // Analysis stage configuration
  stages: StageConfig[];
  
  // Verification layers
  verification: VerificationConfig[];
  
  // Output formatting for swarm handoff
  output: OutputConfig;
  
  // Performance tuning
  performance: PerformanceConfig;
}

export interface PipelineConfig {
  maxIterations: number;
  convergenceThreshold: number;
  parallelismLevel: number;
  timeoutMs: number;
  focus: 'intelligence_gathering' | 'solution_generation';
  outputOptimization: 'swarm_handoff' | 'direct_consumption';
}

export interface AgentConfig {
  type: 'intelligence_analyst' | 'strategy_mapper' | 'risk_analyst' | 'intelligence_synthesizer' | 'specialist';
  name: string;
  enabled: boolean;
  weight: number; // 0.0-1.0 influence in final synthesis
  config: {
    systemPrompt?: string;
    specialization?: string[];
    temperature?: number;
    maxTokens?: number;
    model?: string;
  };
  dependencies: string[]; // Other agents this depends on
}

export interface StageConfig {
  name: string;
  type: 'domain_analysis' | 'technical_analysis' | 'strategy_analysis' | 'risk_analysis' | 'symbol_analysis' | 'dependency_mapping' | 'call_graph' | 'semantic_search' | 'type_analysis';
  enabled: boolean;
  parallel: boolean;
  weight: number;
  config: Record<string, any>;
  conditions?: {
    runIf?: string; // JavaScript expression
    skipIf?: string;
  };
}

export interface VerificationConfig {
  type: 'syntactic' | 'semantic' | 'cross_reference' | 'logical_consistency';
  enabled: boolean;
  weight: number;
  config: Record<string, any>;
}

export interface OutputConfig {
  format: 'intelligence_brief' | 'comprehensive' | 'overview';
  sections: string[];
  densityTarget: number; // target tokens for output
  optimizeFor: 'swarm_consumption' | 'human_readable';
  includeDebateTopics: boolean;
  includeMetrics: boolean;
  includeCitations: boolean;
}

export interface PerformanceConfig {
  cacheResults: boolean;
  cacheTtlMinutes: number;
  maxMemoryMb: number;
  cpuIntensive: boolean;
}

// ============================================================================
// INTELLIGENCE GATHERING RESULT TYPES
// ============================================================================

export interface IntelligenceGatheringResult {
  // Core intelligence for refinement swarm
  findings: StructuredFindings;
  strategicGuidance: StrategyRecommendations;
  implementationConstraints: Constraint[];
  riskFactors: RiskFactor[];
  
  // Supporting intelligence
  domainContext: DomainContext;
  toolingGuidance: ToolingRecommendations;
  precedentAnalysis: PrecedentCase[];
  
  // Meta-intelligence to help swarm work better
  complexityAssessment: ComplexityMetrics;
  solutionSpaceMapping: SolutionSpace;
  debateFramework: DebateTopic[];
  
  // Execution metadata
  loadoutName: string;
  executionMetrics: ExecutionMetrics;
  verificationScore: number;
  generatedAt: string;
}

export interface StructuredFindings {
  keyInsights: KeyInsight[];
  technicalFindings: TechnicalFinding[];
  domainFindings: DomainFinding[];
  architecturalFindings: ArchitecturalFinding[];
}

export interface StrategyRecommendations {
  primaryApproaches: ApproachVector[];
  alternativeStrategies: ApproachVector[];
  hybridOpportunities: HybridStrategy[];
  antiPatterns: AntiPattern[];
  approachTradeoffs: TradeoffMatrix;
}

export interface ApproachVector {
  name: string;
  description: string;
  confidence: number;
  complexity: number;
  riskLevel: number;
  implementationEffort: number;
  supportingEvidence: Evidence[];
  potentialDownsides: string[];
}

export interface HybridStrategy {
  name: string;
  description: string;
  combinesApproaches: string[];
  synergies: string[];
  additionalComplexity: number;
  confidence: number;
}

export interface AntiPattern {
  name: string;
  description: string;
  whyProblematic: string;
  commonTriggers: string[];
  alternatives: string[];
}

export interface TradeoffMatrix {
  dimensions: string[];
  approaches: string[];
  scores: Record<string, Record<string, number>>; // approach -> dimension -> score
  weights: Record<string, number>; // dimension -> importance weight
}

// ============================================================================
// SWARM-OPTIMIZED INTELLIGENCE TYPES
// ============================================================================

export interface SwarmIntelligenceBrief {
  // What the swarm needs to debate
  debateTopics: DebateTopic[];
  
  // Context for informed debate
  domainIntelligence: {
    keyConstraints: Constraint[];
    successCriteria: SuccessCriterion[];
    stakeholderNeeds: StakeholderRequirement[];
    precedentCases: PrecedentCase[];
  };
  
  // Strategic landscape for debate
  approachLandscape: {
    primaryApproaches: ApproachVector[];
    hybridOpportunities: HybridStrategy[];
    approachTradeoffs: TradeoffMatrix;
    eliminatedApproaches: RejectedApproach[];
  };
  
  // Technical intelligence for solution quality
  technicalIntelligence: {
    codebaseConstraints: CodeConstraint[];
    performanceRequirements: PerformanceRequirement[];
    integrationPoints: IntegrationPoint[];
    toolingRecommendations: ToolingGuide[];
  };
  
  // Risk intelligence for solution validation
  riskProfile: {
    technicalRisks: Risk[];
    implementationChallenges: Challenge[];
    maintenanceConcerns: MaintenanceConcern[];
    scalabilityLimitations: ScalabilityLimit[];
  };
}

export interface DebateTopic {
  topic: string;
  importance: number; // 0-1
  category: 'architecture' | 'implementation' | 'technology' | 'process' | 'risk';
  positions: DebatePosition[];
  evidence: Evidence[];
  suggestedDebateStructure: string;
  stakesDescription: string;
  timeEstimate: number; // minutes
}

export interface DebatePosition {
  position: string;
  supportingEvidence: Evidence[];
  counterarguments: string[];
  implementationImplications: string[];
  riskProfile: string;
  confidence: number;
}

// ============================================================================
// DOMAIN INTELLIGENCE TYPES
// ============================================================================

export interface DomainContext {
  problemSpace: ProblemSpace;
  domainExpertise: DomainExpertise;
  businessContext: BusinessContext;
  technicalContext: TechnicalContext;
}

export interface ProblemSpace {
  coreChallenge: string;
  subProblems: SubProblem[];
  dependencies: ProblemDependency[];
  constraints: Constraint[];
  assumptions: Assumption[];
}

export interface DomainExpertise {
  domain: string;
  keyConceptsNeeded: Concept[];
  expertiseLevel: 'novice' | 'intermediate' | 'expert' | 'domain_expert';
  knowledgeGaps: KnowledgeGap[];
  learningResources: LearningResource[];
}

export interface BusinessContext {
  businessValue: string;
  stakeholders: Stakeholder[];
  timeline: TimelineConstraint[];
  budget: BudgetConstraint[];
  successMetrics: BusinessMetric[];
}

export interface TechnicalContext {
  currentArchitecture: ArchitectureDescription;
  technicalDebt: TechnicalDebtItem[];
  performanceRequirements: PerformanceRequirement[];
  scalabilityRequirements: ScalabilityRequirement[];
  integrationRequirements: IntegrationRequirement[];
}

// ============================================================================
// EXECUTION AND METRICS TYPES
// ============================================================================

export interface ExecutionMetrics {
  startTime: string;
  endTime: string;
  duration: number;
  iterationsCompleted: number;
  convergenceScore: number;
  stageResults: StageExecutionResult[];
  agentResults: AgentExecutionResult[];
  resourceUsage: ResourceUsage;
  qualityMetrics: QualityMetrics;
}

export interface StageExecutionResult {
  stageName: string;
  success: boolean;
  duration: number;
  outputSize: number;
  confidence: number;
  error?: string;
}

export interface AgentExecutionResult {
  agentName: string;
  iterations: number;
  totalTokens: number;
  avgConfidence: number;
  contributionScore: number;
  error?: string;
}

export interface ResourceUsage {
  maxMemoryMb: number;
  avgCpuPercent: number;
  networkRequests: number;
  cacheHits: number;
  cacheMisses: number;
}

export interface QualityMetrics {
  completeness: number; // 0-1
  accuracy: number; // 0-1, based on verification
  density: number; // 0-1, information per token
  relevance: number; // 0-1, relevance to query
  actionability: number; // 0-1, how actionable the intelligence is
  overall: number; // 0-1, weighted combination
}

export interface ComplexityMetrics {
  technicalComplexity: number; // 0-10
  domainComplexity: number; // 0-10
  implementationComplexity: number; // 0-10
  riskComplexity: number; // 0-10
  overallComplexity: number; // 0-10, weighted average
  complexityFactors: ComplexityFactor[];
}

export interface ComplexityFactor {
  factor: string;
  impact: number; // 0-10
  description: string;
  mitigationStrategies: string[];
}

// ============================================================================
// SHARED KNOWLEDGE TYPES
// ============================================================================

export interface SharedKnowledgePacket {
  target: string;
  iterations: IterationResult[];
  stageResults: Map<string, StageResult>;
  agentContributions: Map<string, AgentContribution>;
  verificationResults: VerificationResult[];
  synthesisHistory: SynthesisResult[];
  confidence: number;
  convergenceHistory: number[];
}

export interface IterationResult {
  iteration: number;
  findings: Finding[];
  hypotheses: Hypothesis[];
  confidence: number;
  convergenceScore: number;
  agentOutputs: AgentOutput[];
}

export interface AgentContribution {
  agentName: string;
  totalContributions: number;
  avgConfidence: number;
  specializations: string[];
  keyInsights: KeyInsight[];
}

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

export interface KeyInsight {
  insight: string;
  confidence: number;
  supportingEvidence: Evidence[];
  implications: string[];
  source: string;
  category: 'technical' | 'strategic' | 'risk' | 'opportunity';
}

export interface Evidence {
  type: 'code_analysis' | 'precedent' | 'benchmark' | 'documentation' | 'expert_knowledge';
  description: string;
  source: string;
  confidence: number;
  relevance: number;
  citation?: Citation;
}

export interface Citation {
  repoRoot: string;
  rev: string;
  path: string;
  span: [number, number];
  contentSha: string;
}

export interface Constraint {
  type: 'technical' | 'business' | 'regulatory' | 'resource' | 'time';
  name: string;
  description: string;
  severity: 'hard' | 'soft' | 'preference';
  impact: string;
  workarounds: string[];
}

export interface RiskFactor {
  risk: string;
  probability: number; // 0-1
  impact: number; // 0-10
  riskScore: number; // probability * impact
  category: 'technical' | 'business' | 'operational' | 'strategic';
  mitigationStrategies: string[];
  contingencyPlans: string[];
}

export interface ToolingRecommendations {
  recommended: ToolRecommendation[];
  alternatives: ToolRecommendation[];
  integrationGuidance: IntegrationGuide[];
  toolingRisks: ToolingRisk[];
}

export interface ToolRecommendation {
  tool: string;
  purpose: string;
  confidence: number;
  learningCurve: 'low' | 'medium' | 'high';
  integrationComplexity: 'low' | 'medium' | 'high';
  alternatives: string[];
  reasoning: string;
}

export interface PrecedentCase {
  title: string;
  description: string;
  similarity: number; // 0-1, how similar to current situation
  outcome: 'success' | 'partial_success' | 'failure';
  keyLessons: string[];
  applicableStrategies: string[];
  pitfallsToAvoid: string[];
  source: string;
}

export interface SolutionSpace {
  dimensions: SolutionDimension[];
  viableRegions: ViableRegion[];
  impossibleRegions: ImpossibleRegion[];
  unexploredRegions: UnexploredRegion[];
  recommendedExploration: ExplorationStrategy[];
}

export interface SolutionDimension {
  name: string;
  description: string;
  range: [number, number];
  unit: string;
  importance: number; // 0-1
}

export interface ViableRegion {
  name: string;
  description: string;
  coordinates: Record<string, number>; // dimension -> value
  confidence: number;
  advantages: string[];
  disadvantages: string[];
}

// ============================================================================
// CONFIGURATION RESULT TYPES
// ============================================================================

export interface ConfigurableResearchResult {
  target: string;
  loadoutName: string;
  success: boolean;
  intelligence?: IntelligenceGatheringResult;
  report?: string;
  metrics?: ExecutionMetrics;
  duration: number;
  verificationScore?: number;
  error?: string;
}

export interface LoadoutComparisonResult {
  summary: ComparisonSummary;
  metrics: MetricsComparison;
  qualityScores: QualityScore[];
  recommendations: LoadoutRecommendation[];
}

export interface QualityScore {
  loadoutName: string;
  completeness: number; // 0-1, how much of the target was analyzed
  accuracy: number; // 0-1, verification score
  density: number; // 0-1, information density of output
  performance: number; // 0-1, speed vs target time
  overall: number; // 0-1, weighted combination
}

export interface LoadoutRecommendation {
  scenario: string;
  recommendedLoadout: string;
  reasoning: string;
  alternatives: string[];
  tradeoffs: string;
}

// ============================================================================
// ZOD SCHEMAS FOR VALIDATION
// ============================================================================

export const ResearchLoadoutSchema = z.object({
  name: z.string(),
  description: z.string(),
  version: z.string(),
  pipeline: z.object({
    maxIterations: z.number().min(1).max(10),
    convergenceThreshold: z.number().min(0).max(1),
    parallelismLevel: z.number().min(1).max(10),
    timeoutMs: z.number().min(1000),
    focus: z.enum(['intelligence_gathering', 'solution_generation']),
    outputOptimization: z.enum(['swarm_handoff', 'direct_consumption']),
  }),
  agents: z.array(z.object({
    type: z.enum(['intelligence_analyst', 'strategy_mapper', 'risk_analyst', 'intelligence_synthesizer', 'specialist']),
    name: z.string(),
    enabled: z.boolean(),
    weight: z.number().min(0).max(1),
    config: z.object({
      systemPrompt: z.string().optional(),
      specialization: z.array(z.string()).optional(),
      temperature: z.number().min(0).max(2).optional(),
      maxTokens: z.number().min(100).max(8000).optional(),
      model: z.string().optional(),
    }),
    dependencies: z.array(z.string()),
  })),
  stages: z.array(z.object({
    name: z.string(),
    type: z.enum(['domain_analysis', 'technical_analysis', 'strategy_analysis', 'risk_analysis', 'symbol_analysis', 'dependency_mapping', 'call_graph', 'semantic_search', 'type_analysis']),
    enabled: z.boolean(),
    parallel: z.boolean(),
    weight: z.number().min(0).max(1),
    config: z.record(z.any()),
    conditions: z.object({
      runIf: z.string().optional(),
      skipIf: z.string().optional(),
    }).optional(),
  })),
  verification: z.array(z.object({
    type: z.enum(['syntactic', 'semantic', 'cross_reference', 'logical_consistency']),
    enabled: z.boolean(),
    weight: z.number().min(0).max(1),
    config: z.record(z.any()),
  })),
  output: z.object({
    format: z.enum(['intelligence_brief', 'comprehensive', 'overview']),
    sections: z.array(z.string()),
    densityTarget: z.number().min(500).max(10000),
    optimizeFor: z.enum(['swarm_consumption', 'human_readable']),
    includeDebateTopics: z.boolean(),
    includeMetrics: z.boolean(),
    includeCitations: z.boolean(),
  }),
  performance: z.object({
    cacheResults: z.boolean(),
    cacheTtlMinutes: z.number().min(1),
    maxMemoryMb: z.number().min(256),
    cpuIntensive: z.boolean(),
  }),
});

export type ResearchLoadoutType = z.infer<typeof ResearchLoadoutSchema>;

// ============================================================================
// TYPE GUARDS AND UTILITIES
// ============================================================================

export function isIntelligenceGatheringResult(obj: any): obj is IntelligenceGatheringResult {
  return obj && 
    typeof obj.findings === 'object' &&
    typeof obj.strategicGuidance === 'object' &&
    Array.isArray(obj.implementationConstraints) &&
    Array.isArray(obj.riskFactors);
}

export function isValidLoadout(obj: any): obj is ResearchLoadout {
  try {
    ResearchLoadoutSchema.parse(obj);
    return true;
  } catch {
    return false;
  }
}

export function validateLoadout(obj: any): ResearchLoadout {
  return ResearchLoadoutSchema.parse(obj);
}