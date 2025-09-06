/**
 * Unit tests for SwarmOptimizedReporter.
 * 
 * Tests report generation, section optimization, token counting,
 * and all report section generators with comprehensive mocking
 * and realistic test data.
 */

import { describe, expect, it, beforeEach, afterEach, spyOn } from 'bun:test';
import { 
  SwarmOptimizedReporter,
  SwarmReportConfig,
  ReportSection
} from '@/research/swarm-reporter';
import type {
  IntelligenceGatheringResult,
  KeyInsight,
  Evidence,
  RiskFactor,
  ApproachVector,
  HybridStrategy,
  AntiPattern,
  TradeoffMatrix,
  Constraint,
  ComplexityMetrics,
  ComplexityFactor,
  DebateTopic,
  DebatePosition,
  ToolingRecommendations,
  ToolRecommendation,
  ExecutionMetrics,
  StageExecutionResult,
  QualityMetrics,
  ResourceUsage
} from '@/research/types';

describe('SwarmOptimizedReporter', () => {
  let reporter: SwarmOptimizedReporter;
  let loggerSpy: any;
  let consoleLogSpy: any;

  // Mock data
  let mockIntelligence: IntelligenceGatheringResult;
  let mockConfig: SwarmReportConfig;

  beforeEach(() => {
    // Mock console to suppress log output in tests
    consoleLogSpy = spyOn(console, 'log').mockImplementation(() => {});
    
    reporter = new SwarmOptimizedReporter();
    
    // Mock logger calls
    loggerSpy = {
      info: spyOn(reporter as any, 'logger').mockReturnValue({
        info: () => {},
        warn: () => {},
        error: () => {},
        debug: () => {}
      })
    };

    // Create comprehensive mock intelligence data
    mockIntelligence = createMockIntelligenceData();
    mockConfig = createMockConfig();
  });

  afterEach(() => {
    consoleLogSpy?.mockRestore();
  });

  describe('constructor', () => {
    it('should create logger with correct name', () => {
      const newReporter = new SwarmOptimizedReporter();
      expect(newReporter).toBeInstanceOf(SwarmOptimizedReporter);
    });
  });

  describe('generateIntelligenceBrief', () => {
    it('should generate complete intelligence brief', async () => {
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, mockConfig);
      
      expect(result).toContain('# INTELLIGENCE BRIEF FOR REFINEMENT SWARM');
      expect(result).toContain('## TABLE OF CONTENTS');
      expect(result).toContain('## EXECUTIVE INTELLIGENCE SUMMARY');
      expect(result).toContain('## SWARM HANDOFF CHECKLIST');
      expect(result).toContain(mockIntelligence.loadoutName);
      expect(result).toContain(mockIntelligence.verificationScore.toFixed(2));
    });

    it('should handle swarm_consumption optimization', async () => {
      const swarmConfig = {
        ...mockConfig,
        optimizeFor: 'swarm_consumption' as const
      };
      
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, swarmConfig);
      
      expect(result).toContain('swarm_consumption');
    });

    it('should handle human_readable optimization', async () => {
      const humanConfig = {
        ...mockConfig,
        optimizeFor: 'human_readable' as const
      };
      
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, humanConfig);
      
      expect(result).toContain('human_readable');
    });

    it('should handle different debate structure depths', async () => {
      for (const depth of ['shallow', 'medium', 'deep'] as const) {
        const config = { ...mockConfig, debateStructureDepth: depth };
        const result = await reporter.generateIntelligenceBrief(mockIntelligence, config);
        
        expect(result).toBeDefined();
        expect(typeof result).toBe('string');
      }
    });

    it('should include debate topics when configured', async () => {
      const configWithDebate = {
        ...mockConfig,
        includeDebateTopics: true
      };
      
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, configWithDebate);
      
      expect(result).toContain('DEBATE FRAMEWORK');
    });

    it('should exclude debate topics when not configured', async () => {
      const configWithoutDebate = {
        ...mockConfig,
        includeDebateTopics: false
      };
      
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, configWithoutDebate);
      
      expect(result).not.toContain('DEBATE FRAMEWORK');
    });

    it('should include metrics when configured', async () => {
      const configWithMetrics = {
        ...mockConfig,
        includeMetrics: true,
        densityTarget: 8000 // Increase token limit to ensure metrics section is included
      };
      
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, configWithMetrics);
      
      expect(result).toContain('ANALYSIS EXECUTION METRICS');
    });

    it('should exclude metrics when not configured', async () => {
      const configWithoutMetrics = {
        ...mockConfig,
        includeMetrics: false
      };
      
      const result = await reporter.generateIntelligenceBrief(mockIntelligence, configWithoutMetrics);
      
      expect(result).not.toContain('ANALYSIS EXECUTION METRICS');
    });
  });

  describe('generateAllSections', () => {
    it('should generate all required sections', async () => {
      const sections = await (reporter as any).generateAllSections(mockIntelligence, mockConfig);
      
      expect(sections).toBeArray();
      expect(sections.length).toBeGreaterThan(0);
      
      // Check that executive summary is always included
      const execSummary = sections.find((s: ReportSection) => s.category === 'executive');
      expect(execSummary).toBeDefined();
      expect(execSummary.priority).toBe(1.0);
    });

    it('should conditionally include debate framework', async () => {
      const configWithDebate = { ...mockConfig, includeDebateTopics: true };
      const sectionsWithDebate = await (reporter as any).generateAllSections(mockIntelligence, configWithDebate);
      
      const debateSection = sectionsWithDebate.find((s: ReportSection) => s.category === 'debate');
      expect(debateSection).toBeDefined();
      
      const configWithoutDebate = { ...mockConfig, includeDebateTopics: false };
      const sectionsWithoutDebate = await (reporter as any).generateAllSections(mockIntelligence, configWithoutDebate);
      
      const noDebateSection = sectionsWithoutDebate.find((s: ReportSection) => s.category === 'debate');
      expect(noDebateSection).toBeUndefined();
    });

    it('should conditionally include metrics section', async () => {
      const configWithMetrics = { ...mockConfig, includeMetrics: true };
      const sectionsWithMetrics = await (reporter as any).generateAllSections(mockIntelligence, configWithMetrics);
      
      const metricsSection = sectionsWithMetrics.find((s: ReportSection) => 
        s.title === 'Analysis Execution Metrics'
      );
      expect(metricsSection).toBeDefined();
      
      const configWithoutMetrics = { ...mockConfig, includeMetrics: false };
      const sectionsWithoutMetrics = await (reporter as any).generateAllSections(mockIntelligence, configWithoutMetrics);
      
      const noMetricsSection = sectionsWithoutMetrics.find((s: ReportSection) => 
        s.title === 'Analysis Execution Metrics'
      );
      expect(noMetricsSection).toBeUndefined();
    });
  });

  describe('optimizeSectionSelection', () => {
    it('should prioritize sections by priority score', () => {
      const mockSections: ReportSection[] = [
        { title: 'Low Priority', content: 'test', tokenCount: 100, priority: 0.3, category: 'technical' },
        { title: 'High Priority', content: 'test', tokenCount: 100, priority: 0.9, category: 'strategic' },
        { title: 'Executive', content: 'test', tokenCount: 100, priority: 1.0, category: 'executive' }
      ];
      
      const optimized = (reporter as any).optimizeSectionSelection(mockSections, mockConfig);
      
      expect(optimized[0].category).toBe('executive');
      expect(optimized[1].title).toBe('High Priority');
    });

    it('should respect token budget limits', () => {
      const restrictiveConfig = {
        ...mockConfig,
        densityTarget: 300,
        maxSectionTokens: 100
      };
      
      const mockSections: ReportSection[] = [
        { title: 'Executive', content: 'test', tokenCount: 150, priority: 1.0, category: 'executive' },
        { title: 'Large Section', content: 'test', tokenCount: 500, priority: 0.8, category: 'strategic' },
        { title: 'Small Section', content: 'test', tokenCount: 100, priority: 0.7, category: 'technical' }
      ];
      
      const optimized = (reporter as any).optimizeSectionSelection(mockSections, restrictiveConfig);
      
      const totalTokens = optimized.reduce((sum: number, section: ReportSection) => 
        sum + section.tokenCount, 0
      );
      expect(totalTokens).toBeLessThanOrEqual(restrictiveConfig.densityTarget);
    });

    it('should always include executive summary', () => {
      const mockSections: ReportSection[] = [
        { title: 'Other Section', content: 'test', tokenCount: 100, priority: 0.9, category: 'strategic' },
        { title: 'Executive Summary', content: 'test', tokenCount: 200, priority: 1.0, category: 'executive' }
      ];
      
      const optimized = (reporter as any).optimizeSectionSelection(mockSections, {
        ...mockConfig,
        densityTarget: 150
      });
      
      expect(optimized.some((s: ReportSection) => s.category === 'executive')).toBe(true);
    });

    it('should truncate sections when over budget but under 80%', () => {
      const mockSections: ReportSection[] = [
        { title: 'Executive', content: 'test', tokenCount: 300, priority: 1.0, category: 'executive' },
        { title: 'Large Section', content: 'Line 1\nLine 2\nLine 3\nLine 4', tokenCount: 400, priority: 0.8, category: 'strategic' }
      ];
      
      const optimized = (reporter as any).optimizeSectionSelection(mockSections, {
        ...mockConfig,
        densityTarget: 500,
        maxSectionTokens: 200
      });
      
      const truncatedSection = optimized.find((s: ReportSection) => s.title.includes('Truncated'));
      expect(truncatedSection).toBeDefined();
    });
  });

  describe('assembleFinalReport', () => {
    it('should assemble complete report with all components', () => {
      const mockSections: ReportSection[] = [
        { title: 'Test Section', content: '## Test Content', tokenCount: 100, priority: 1.0, category: 'executive' }
      ];
      
      const result = (reporter as any).assembleFinalReport(mockSections, mockIntelligence, mockConfig);
      
      expect(result).toContain('# INTELLIGENCE BRIEF FOR REFINEMENT SWARM');
      expect(result).toContain('## TABLE OF CONTENTS');
      expect(result).toContain('## Test Content');
      expect(result).toContain('## SWARM HANDOFF CHECKLIST');
      expect(result).toContain(mockIntelligence.loadoutName);
    });

    it('should include correct metadata in header', () => {
      const mockSections: ReportSection[] = [];
      const result = (reporter as any).assembleFinalReport(mockSections, mockIntelligence, mockConfig);
      
      expect(result).toContain(`**Loadout:** ${mockIntelligence.loadoutName}`);
      expect(result).toContain(`**Target Density:** ${mockConfig.densityTarget} tokens`);
      expect(result).toContain(`**Optimization:** ${mockConfig.optimizeFor}`);
      expect(result).toContain(`**Verification Score:** ${mockIntelligence.verificationScore.toFixed(2)}`);
    });
  });

  describe('section generators', () => {
    describe('generateExecutiveSummary', () => {
      it('should generate executive summary with key insights', async () => {
        const section = await (reporter as any).generateExecutiveSummary(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Executive Intelligence Summary');
        expect(section.category).toBe('executive');
        expect(section.priority).toBe(1.0);
        expect(section.content).toContain('EXECUTIVE INTELLIGENCE SUMMARY');
        expect(section.content).toContain(mockIntelligence.loadoutName);
        expect(section.content).toContain('Key Strategic Insights');
        expect(section.content).toContain('Primary Strategic Approaches');
        expect(section.content).toContain('Critical Risk Factors');
      });

      it('should handle empty insights gracefully', async () => {
        const emptyIntelligence = {
          ...mockIntelligence,
          findings: {
            ...mockIntelligence.findings,
            keyInsights: []
          }
        };
        
        const section = await (reporter as any).generateExecutiveSummary(emptyIntelligence, mockConfig);
        
        expect(section.content).toBeDefined();
        expect(section.tokenCount).toBeGreaterThan(0);
      });
    });

    describe('generateDebateFramework', () => {
      it('should generate debate framework with topics and positions', async () => {
        const section = await (reporter as any).generateDebateFramework(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Debate Framework');
        expect(section.category).toBe('debate');
        expect(section.content).toContain('DEBATE FRAMEWORK FOR REFINEMENT SWARM');
        expect(section.content).toContain('Recommended Debate Structure');
        expect(section.content).toContain('Priority Debate Topics');
      });

      it('should handle topics without positions', async () => {
        const intelligenceWithEmptyPositions = {
          ...mockIntelligence,
          debateFramework: [
            {
              ...mockIntelligence.debateFramework[0],
              positions: []
            }
          ]
        };
        
        const section = await (reporter as any).generateDebateFramework(intelligenceWithEmptyPositions, mockConfig);
        
        expect(section.content).toContain('No specific positions identified');
      });
    });

    describe('generateApproachLandscape', () => {
      it('should generate approach landscape with strategies', async () => {
        const section = await (reporter as any).generateApproachLandscape(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Strategic Approach Landscape');
        expect(section.category).toBe('strategic');
        expect(section.content).toContain('STRATEGIC APPROACH LANDSCAPE');
        expect(section.content).toContain('Primary Approaches');
        expect(section.content).toContain('Hybrid Strategy Opportunities');
      });

      it('should handle empty hybrid opportunities', async () => {
        const intelligenceWithNoHybrids = {
          ...mockIntelligence,
          strategicGuidance: {
            ...mockIntelligence.strategicGuidance,
            hybridOpportunities: []
          }
        };
        
        const section = await (reporter as any).generateApproachLandscape(intelligenceWithNoHybrids, mockConfig);
        
        expect(section.content).toContain('No viable hybrid approaches identified');
      });
    });

    describe('generateTechnicalConstraints', () => {
      it('should generate technical constraints analysis', async () => {
        const section = await (reporter as any).generateTechnicalConstraints(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Technical Constraint Analysis');
        expect(section.category).toBe('technical');
        expect(section.content).toContain('TECHNICAL CONSTRAINT ANALYSIS');
        expect(section.content).toContain('Hard Constraints');
        expect(section.content).toContain('Soft Constraints');
      });

      it('should handle no constraints gracefully', async () => {
        const intelligenceWithNoConstraints = {
          ...mockIntelligence,
          implementationConstraints: []
        };
        
        const section = await (reporter as any).generateTechnicalConstraints(intelligenceWithNoConstraints, mockConfig);
        
        expect(section.content).toContain('No hard technical constraints identified');
        expect(section.content).toContain('No significant soft constraints identified');
      });

      it('should handle soft constraints', async () => {
        const intelligenceWithSoftConstraints = {
          ...mockIntelligence,
          implementationConstraints: [
            {
              type: 'business',
              name: 'Preferred framework',
              description: 'Team prefers React over Vue',
              severity: 'soft',
              impact: 'Development speed',
              workarounds: ['Training', 'Gradual migration']
            }
          ]
        };
        
        const section = await (reporter as any).generateTechnicalConstraints(intelligenceWithSoftConstraints, mockConfig);
        
        expect(section.content).toContain('Preferred framework');
        expect(section.content).toContain('Training');
      });
    });

    describe('generateRiskProfile', () => {
      it('should generate risk and complexity profile', async () => {
        const section = await (reporter as any).generateRiskProfile(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Risk and Complexity Profile');
        expect(section.category).toBe('risk');
        expect(section.content).toContain('RISK AND COMPLEXITY PROFILE');
        expect(section.content).toContain('Overall Complexity Assessment');
        expect(section.content).toContain('Critical Risks');
      });

      it('should categorize risks correctly', async () => {
        const section = await (reporter as any).generateRiskProfile(mockIntelligence, mockConfig);
        
        expect(section.content).toContain('Risk Analysis by Category');
      });

      it('should handle critical risks properly', async () => {
        const intelligenceWithCriticalRisks = {
          ...mockIntelligence,
          riskFactors: [
            {
              risk: 'Critical system failure',
              probability: 0.8,
              impact: 9,
              riskScore: 7.2,
              category: 'technical',
              mitigationStrategies: ['Backup systems', 'Monitoring'],
              contingencyPlans: ['Emergency rollback', 'Manual failover']
            }
          ]
        };
        
        const section = await (reporter as any).generateRiskProfile(intelligenceWithCriticalRisks, mockConfig);
        
        expect(section.content).toContain('Critical system failure');
        expect(section.content).toContain('Backup systems');
        expect(section.content).toContain('Emergency rollback');
      });

      it('should handle no critical risks', async () => {
        const intelligenceWithLowRisks = {
          ...mockIntelligence,
          riskFactors: [
            {
              risk: 'Minor UI issue',
              probability: 0.3,
              impact: 2,
              riskScore: 0.6,
              category: 'technical',
              mitigationStrategies: ['Testing'],
              contingencyPlans: ['Quick fix']
            }
          ]
        };
        
        const section = await (reporter as any).generateRiskProfile(intelligenceWithLowRisks, mockConfig);
        
        expect(section.content).toContain('No critical risks identified');
      });
    });

    describe('generateImplementationIntelligence', () => {
      it('should generate implementation intelligence map', async () => {
        const section = await (reporter as any).generateImplementationIntelligence(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Implementation Intelligence Map');
        expect(section.category).toBe('implementation');
        expect(section.content).toContain('IMPLEMENTATION INTELLIGENCE MAP');
        expect(section.content).toContain('Implementation Readiness Assessment');
        expect(section.content).toContain('Quality Gates & Validation');
      });
    });

    describe('generateToolingGuidance', () => {
      it('should generate tooling and resource guidance', async () => {
        const section = await (reporter as any).generateToolingGuidance(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Tooling and Resource Guide');
        expect(section.category).toBe('implementation');
        expect(section.content).toContain('TOOLING AND RESOURCE GUIDANCE');
        expect(section.content).toContain('Recommended Tools');
      });

      it('should handle empty tooling recommendations', async () => {
        const intelligenceWithNoTools = {
          ...mockIntelligence,
          toolingGuidance: {
            ...mockIntelligence.toolingGuidance,
            recommended: []
          }
        };
        
        const section = await (reporter as any).generateToolingGuidance(intelligenceWithNoTools, mockConfig);
        
        expect(section.content).toContain('No specific tool recommendations available');
      });
    });

    describe('generateSuccessCriteria', () => {
      it('should generate success criteria definition', async () => {
        const section = await (reporter as any).generateSuccessCriteria(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Success Criteria Definition');
        expect(section.category).toBe('implementation');
        expect(section.content).toContain('SUCCESS CRITERIA DEFINITION');
        expect(section.content).toContain('Primary Success Metrics');
        expect(section.content).toContain('Quality Gates');
      });
    });

    describe('generateMetricsSection', () => {
      it('should generate analysis execution metrics', async () => {
        const section = await (reporter as any).generateMetricsSection(mockIntelligence, mockConfig);
        
        expect(section.title).toBe('Analysis Execution Metrics');
        expect(section.category).toBe('technical');
        expect(section.content).toContain('ANALYSIS EXECUTION METRICS');
        expect(section.content).toContain('Research Execution Summary');
        expect(section.content).toContain('Quality Metrics');
        expect(section.content).toContain('Resource Usage');
      });
    });
  });

  describe('helper methods', () => {
    describe('estimateTokenCount', () => {
      it('should estimate token count correctly', () => {
        const text = 'This is a test string with multiple words';
        const tokenCount = (reporter as any).estimateTokenCount(text);
        
        expect(tokenCount).toBeGreaterThan(0);
        expect(tokenCount).toBe(Math.ceil(text.length / 4));
      });
    });

    describe('countEvidence', () => {
      it('should count evidence from insights and approaches', () => {
        const count = (reporter as any).countEvidence(mockIntelligence);
        
        expect(count).toBeGreaterThan(0);
      });
    });

    describe('groupRisksByCategory', () => {
      it('should group risks by category', () => {
        const grouped = (reporter as any).groupRisksByCategory(mockIntelligence.riskFactors);
        
        expect(typeof grouped).toBe('object');
        expect(Object.keys(grouped).length).toBeGreaterThan(0);
        
        // Verify risks are sorted within categories
        Object.values(grouped).forEach((risks: any[]) => {
          for (let i = 1; i < risks.length; i++) {
            expect(risks[i - 1].riskScore).toBeGreaterThanOrEqual(risks[i].riskScore);
          }
        });
      });
    });

    describe('calculateReadinessScore', () => {
      it('should calculate readiness score based on confidence and risk', () => {
        const approach: ApproachVector = mockIntelligence.strategicGuidance.primaryApproaches[0];
        const score = (reporter as any).calculateReadinessScore(approach);
        
        expect(score).toBeGreaterThanOrEqual(1);
        expect(score).toBeLessThanOrEqual(10);
      });
    });

    describe('truncateSection', () => {
      it('should truncate section content correctly', () => {
        const originalSection: ReportSection = {
          title: 'Test Section',
          content: 'Line 1\nLine 2\nLine 3\nLine 4\nLine 5',
          tokenCount: 100,
          priority: 1.0,
          category: 'technical'
        };
        
        const truncated = (reporter as any).truncateSection(originalSection, 50);
        
        expect(truncated.tokenCount).toBe(50);
        expect(truncated.title).toContain('(Truncated)');
        expect(truncated.content).toContain('[Content truncated for brevity...]');
      });
    });
  });

  describe('error handling', () => {
    it('should handle invalid intelligence data gracefully', async () => {
      const invalidIntelligence = {
        ...mockIntelligence,
        findings: {
          keyInsights: [],
          technicalFindings: [],
          domainFindings: [],
          architecturalFindings: []
        },
        strategicGuidance: {
          primaryApproaches: [],
          alternativeStrategies: [],
          hybridOpportunities: [],
          antiPatterns: [],
          approachTradeoffs: {
            dimensions: [],
            approaches: [],
            scores: {},
            weights: {}
          }
        },
        riskFactors: [],
        implementationConstraints: [],
        debateFramework: []
      };
      
      // Should complete without throwing but may have limited content
      const result = await reporter.generateIntelligenceBrief(invalidIntelligence, mockConfig);
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
    });

    it('should handle empty arrays in intelligence data', async () => {
      const emptyIntelligence = {
        ...mockIntelligence,
        findings: {
          keyInsights: [],
          technicalFindings: [],
          domainFindings: [],
          architecturalFindings: []
        },
        strategicGuidance: {
          primaryApproaches: [],
          alternativeStrategies: [],
          hybridOpportunities: [],
          antiPatterns: [],
          approachTradeoffs: {
            dimensions: [],
            approaches: [],
            scores: {},
            weights: {}
          }
        },
        riskFactors: [],
        implementationConstraints: [],
        debateFramework: []
      };
      
      const result = await reporter.generateIntelligenceBrief(emptyIntelligence, mockConfig);
      
      expect(result).toBeDefined();
      expect(typeof result).toBe('string');
    });
  });

  // Helper function to create comprehensive mock intelligence data
  function createMockIntelligenceData(): IntelligenceGatheringResult {
    const mockEvidence: Evidence[] = [
      {
        type: 'code_analysis',
        description: 'Analysis shows performance bottleneck in data processing',
        source: 'Static analysis tool',
        confidence: 0.85,
        relevance: 0.9
      },
      {
        type: 'precedent',
        description: 'Similar pattern used successfully in related project',
        source: 'Internal knowledge base',
        confidence: 0.75,
        relevance: 0.8
      }
    ];

    const mockKeyInsights: KeyInsight[] = [
      {
        insight: 'Current architecture has scalability limitations',
        confidence: 0.9,
        supportingEvidence: mockEvidence.slice(0, 1),
        implications: ['Need to redesign data layer', 'Consider caching strategy'],
        source: 'technical_analysis',
        category: 'technical'
      },
      {
        insight: 'Multiple viable approaches with different trade-offs',
        confidence: 0.85,
        supportingEvidence: mockEvidence.slice(1, 2),
        implications: ['Requires stakeholder input', 'Consider hybrid approach'],
        source: 'strategy_analysis',
        category: 'strategic'
      }
    ];

    const mockApproaches: ApproachVector[] = [
      {
        name: 'Microservices Architecture',
        description: 'Decompose monolith into domain-specific microservices',
        confidence: 0.8,
        complexity: 7,
        riskLevel: 6,
        implementationEffort: 8,
        supportingEvidence: mockEvidence,
        potentialDownsides: ['Increased operational complexity', 'Network latency concerns']
      },
      {
        name: 'Modular Monolith',
        description: 'Restructure existing code into well-defined modules',
        confidence: 0.85,
        complexity: 5,
        riskLevel: 3,
        implementationEffort: 5,
        supportingEvidence: mockEvidence.slice(0, 1),
        potentialDownsides: ['Still single point of failure', 'Shared database constraints']
      }
    ];

    const mockHybridStrategies: HybridStrategy[] = [
      {
        name: 'Strangler Fig Pattern',
        description: 'Gradually migrate to microservices while maintaining monolith',
        combinesApproaches: ['Microservices Architecture', 'Modular Monolith'],
        synergies: ['Reduced migration risk', 'Incremental value delivery'],
        additionalComplexity: 3,
        confidence: 0.75
      }
    ];

    const mockAntiPatterns: AntiPattern[] = [
      {
        name: 'Big Bang Migration',
        description: 'Attempting to migrate entire system at once',
        whyProblematic: 'High risk of system failure and extended downtime',
        commonTriggers: ['Pressure for quick results', 'Underestimating complexity'],
        alternatives: ['Strangler Fig Pattern', 'Incremental migration']
      }
    ];

    const mockTradeoffs: TradeoffMatrix = {
      dimensions: ['Scalability', 'Complexity', 'Performance', 'Maintainability'],
      approaches: ['Microservices Architecture', 'Modular Monolith'],
      scores: {
        'Microservices Architecture': {
          'Scalability': 9,
          'Complexity': 3,
          'Performance': 6,
          'Maintainability': 7
        },
        'Modular Monolith': {
          'Scalability': 6,
          'Complexity': 7,
          'Performance': 8,
          'Maintainability': 8
        }
      },
      weights: {
        'Scalability': 0.3,
        'Complexity': 0.2,
        'Performance': 0.25,
        'Maintainability': 0.25
      }
    };

    const mockRisks: RiskFactor[] = [
      {
        risk: 'Database performance degradation under load',
        probability: 0.7,
        impact: 8,
        riskScore: 5.6,
        category: 'technical',
        mitigationStrategies: ['Implement connection pooling', 'Add read replicas'],
        contingencyPlans: ['Scale vertically', 'Implement caching layer']
      },
      {
        risk: 'Team lacks microservices experience',
        probability: 0.6,
        impact: 6,
        riskScore: 3.6,
        category: 'operational',
        mitigationStrategies: ['Provide training', 'Hire experienced developers'],
        contingencyPlans: ['Engage external consultants', 'Start with pilot project']
      }
    ];

    const mockConstraints: Constraint[] = [
      {
        type: 'technical',
        name: 'Legacy database compatibility',
        description: 'Must maintain compatibility with existing Oracle database',
        severity: 'hard',
        impact: 'Limits architectural options',
        workarounds: ['Database abstraction layer', 'Gradual migration strategy']
      },
      {
        type: 'business',
        name: 'Zero downtime requirement',
        description: 'System must maintain 99.9% uptime during migration',
        severity: 'hard',
        impact: 'Requires blue-green deployment strategy',
        workarounds: ['Staged migration', 'Feature flagging']
      }
    ];

    const mockDebateTopics: DebateTopic[] = [
      {
        topic: 'Should we adopt microservices or modular monolith?',
        importance: 0.9,
        category: 'architecture',
        positions: [
          {
            position: 'Microservices provide better scalability',
            supportingEvidence: mockEvidence,
            counterarguments: ['Increased operational complexity'],
            implementationImplications: ['Need container orchestration'],
            riskProfile: 'Medium-High',
            confidence: 0.8
          }
        ],
        evidence: mockEvidence,
        suggestedDebateStructure: 'Comparative analysis with weighted scoring',
        stakesDescription: 'Architectural decision affecting long-term maintainability',
        timeEstimate: 45
      }
    ];

    const mockComplexityFactors: ComplexityFactor[] = [
      {
        factor: 'Multiple integration points',
        impact: 7,
        description: 'System integrates with 12 external services',
        mitigationStrategies: ['API versioning', 'Circuit breaker pattern']
      }
    ];

    const mockComplexity: ComplexityMetrics = {
      technicalComplexity: 7.5,
      domainComplexity: 6.0,
      implementationComplexity: 8.0,
      riskComplexity: 6.5,
      overallComplexity: 7.0,
      complexityFactors: mockComplexityFactors
    };

    const mockToolRecommendations: ToolRecommendation[] = [
      {
        tool: 'Docker',
        purpose: 'Containerization for microservices',
        confidence: 0.9,
        learningCurve: 'medium',
        integrationComplexity: 'medium',
        alternatives: ['Podman', 'containerd'],
        reasoning: 'Industry standard with excellent ecosystem support'
      }
    ];

    const mockToolingGuidance: ToolingRecommendations = {
      recommended: mockToolRecommendations,
      alternatives: [],
      integrationGuidance: [{ description: 'Use docker-compose for local development' } as any],
      toolingRisks: [{ tool: 'Docker', risk: 'Image vulnerabilities', mitigation: 'Regular security scanning' } as any]
    };

    const mockStageResults: StageExecutionResult[] = [
      {
        stageName: 'Domain Analysis',
        success: true,
        duration: 1500,
        outputSize: 2048,
        confidence: 0.85
      }
    ];

    const mockResourceUsage: ResourceUsage = {
      maxMemoryMb: 512,
      avgCpuPercent: 45,
      networkRequests: 25,
      cacheHits: 18,
      cacheMisses: 7
    };

    const mockQualityMetrics: QualityMetrics = {
      completeness: 0.85,
      accuracy: 0.9,
      density: 0.75,
      relevance: 0.88,
      actionability: 0.82,
      overall: 0.84
    };

    const mockExecutionMetrics: ExecutionMetrics = {
      startTime: new Date().toISOString(),
      endTime: new Date().toISOString(),
      duration: 5000,
      iterationsCompleted: 3,
      convergenceScore: 0.85,
      stageResults: mockStageResults,
      agentResults: [],
      resourceUsage: mockResourceUsage,
      qualityMetrics: mockQualityMetrics
    };

    return {
      findings: {
        keyInsights: mockKeyInsights,
        technicalFindings: [],
        domainFindings: [],
        architecturalFindings: []
      },
      strategicGuidance: {
        primaryApproaches: mockApproaches,
        alternativeStrategies: [],
        hybridOpportunities: mockHybridStrategies,
        antiPatterns: mockAntiPatterns,
        approachTradeoffs: mockTradeoffs
      },
      implementationConstraints: mockConstraints,
      riskFactors: mockRisks,
      domainContext: {} as any,
      toolingGuidance: mockToolingGuidance,
      precedentAnalysis: [],
      complexityAssessment: mockComplexity,
      solutionSpaceMapping: {} as any,
      debateFramework: mockDebateTopics,
      loadoutName: 'comprehensive-analysis-v1.0',
      executionMetrics: mockExecutionMetrics,
      verificationScore: 0.85,
      generatedAt: new Date().toISOString()
    };
  }

  function createMockConfig(): SwarmReportConfig {
    return {
      densityTarget: 3000,
      maxSectionTokens: 500,
      includeDebateTopics: true,
      includeMetrics: true,
      includeCitations: true,
      optimizeFor: 'swarm_consumption',
      debateStructureDepth: 'medium'
    };
  }
});