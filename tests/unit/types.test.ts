/**
 * Unit tests for research types validation and utilities
 * Focused on increasing overall coverage to >85%
 */

import { describe, it, expect } from 'vitest';
import {
  isIntelligenceGatheringResult,
  isValidLoadout,
  validateLoadout,
  ResearchLoadoutSchema,
  type ResearchLoadout,
  type IntelligenceGatheringResult,
} from '../../src/research/types.js';

describe('Type Guards and Validation', () => {
  describe('isIntelligenceGatheringResult', () => {
    it('should return true for valid IntelligenceGatheringResult', () => {
      const validResult = {
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
        implementationConstraints: [],
        riskFactors: [],
        domainContext: {
          problemSpace: {
            coreChallenge: 'test',
            subProblems: [],
            dependencies: [],
            constraints: [],
            assumptions: []
          },
          domainExpertise: {
            domain: 'test',
            keyConceptsNeeded: [],
            expertiseLevel: 'intermediate' as const,
            knowledgeGaps: [],
            learningResources: []
          },
          businessContext: {
            businessValue: 'test',
            stakeholders: [],
            timeline: [],
            budget: [],
            successMetrics: []
          },
          technicalContext: {
            currentArchitecture: {
              layers: [],
              components: [],
              patterns: []
            },
            technicalDebt: [],
            performanceRequirements: [],
            scalabilityRequirements: [],
            integrationRequirements: []
          }
        },
        toolingGuidance: {
          recommended: [],
          alternatives: [],
          integrationGuidance: [],
          toolingRisks: []
        },
        precedentAnalysis: [],
        complexityAssessment: {
          technicalComplexity: 5,
          domainComplexity: 5,
          implementationComplexity: 5,
          riskComplexity: 5,
          overallComplexity: 5,
          complexityFactors: []
        },
        solutionSpaceMapping: {
          dimensions: [],
          viableRegions: [],
          impossibleRegions: [],
          unexploredRegions: [],
          recommendedExploration: []
        },
        debateFramework: [],
        loadoutName: 'test-loadout',
        executionMetrics: {
          startTime: '2024-01-01T00:00:00Z',
          endTime: '2024-01-01T00:01:00Z',
          duration: 60000,
          iterationsCompleted: 1,
          convergenceScore: 0.8,
          stageResults: [],
          agentResults: [],
          resourceUsage: {
            maxMemoryMb: 512,
            avgCpuPercent: 25,
            networkRequests: 10,
            cacheHits: 5,
            cacheMisses: 2
          },
          qualityMetrics: {
            completeness: 0.9,
            accuracy: 0.85,
            density: 0.8,
            relevance: 0.9,
            actionability: 0.85,
            overall: 0.86
          }
        },
        verificationScore: 0.85,
        generatedAt: '2024-01-01T00:00:00Z'
      };

      expect(isIntelligenceGatheringResult(validResult)).toBe(true);
    });

    it('should return false for invalid objects', () => {
      expect(isIntelligenceGatheringResult(null)).toBeFalsy();
      expect(isIntelligenceGatheringResult(undefined)).toBeFalsy();
      expect(isIntelligenceGatheringResult({})).toBeFalsy();
      expect(isIntelligenceGatheringResult({ findings: 'invalid' })).toBeFalsy();
      expect(isIntelligenceGatheringResult({ 
        findings: {}, 
        strategicGuidance: 'invalid' 
      })).toBeFalsy();
      expect(isIntelligenceGatheringResult({ 
        findings: {}, 
        strategicGuidance: {},
        implementationConstraints: 'invalid'
      })).toBeFalsy();
      expect(isIntelligenceGatheringResult({ 
        findings: {}, 
        strategicGuidance: {},
        implementationConstraints: [],
        riskFactors: 'invalid'
      })).toBeFalsy();
    });
  });

  describe('isValidLoadout', () => {
    const validLoadout: ResearchLoadout = {
      name: 'test-loadout',
      description: 'A test research loadout',
      version: '1.0.0',
      pipeline: {
        maxIterations: 5,
        convergenceThreshold: 0.8,
        parallelismLevel: 2,
        timeoutMs: 30000,
        focus: 'intelligence_gathering',
        outputOptimization: 'swarm_handoff'
      },
      agents: [{
        type: 'intelligence_analyst',
        name: 'test-agent',
        enabled: true,
        weight: 0.5,
        config: {
          systemPrompt: 'test prompt',
          temperature: 0.7,
          maxTokens: 2000
        },
        dependencies: []
      }],
      stages: [{
        name: 'test-stage',
        type: 'domain_analysis',
        enabled: true,
        parallel: false,
        weight: 1.0,
        config: {}
      }],
      verification: [{
        type: 'syntactic',
        enabled: true,
        weight: 1.0,
        config: {}
      }],
      output: {
        format: 'intelligence_brief',
        sections: ['findings', 'recommendations'],
        densityTarget: 2000,
        optimizeFor: 'swarm_consumption',
        includeDebateTopics: true,
        includeMetrics: true,
        includeCitations: false
      },
      performance: {
        cacheResults: true,
        cacheTtlMinutes: 30,
        maxMemoryMb: 1024,
        cpuIntensive: false
      }
    };

    it('should return true for valid loadout', () => {
      expect(isValidLoadout(validLoadout)).toBe(true);
    });

    it('should return false for invalid loadouts', () => {
      expect(isValidLoadout(null)).toBe(false);
      expect(isValidLoadout(undefined)).toBe(false);
      expect(isValidLoadout({})).toBe(false);
      
      // Test missing required fields
      const invalidLoadout = { ...validLoadout };
      delete invalidLoadout.name;
      expect(isValidLoadout(invalidLoadout)).toBe(false);
    });

    it('should return false for invalid pipeline config', () => {
      const invalidPipeline = { 
        ...validLoadout, 
        pipeline: { 
          ...validLoadout.pipeline, 
          maxIterations: 15 // exceeds max of 10
        } 
      };
      expect(isValidLoadout(invalidPipeline)).toBe(false);
    });

    it('should return false for invalid agent config', () => {
      const invalidAgent = { 
        ...validLoadout, 
        agents: [{
          ...validLoadout.agents[0],
          type: 'invalid_type' as any
        }]
      };
      expect(isValidLoadout(invalidAgent)).toBe(false);
    });
  });

  describe('validateLoadout', () => {
    const validLoadout: ResearchLoadout = {
      name: 'test-loadout',
      description: 'A test research loadout',
      version: '1.0.0',
      pipeline: {
        maxIterations: 3,
        convergenceThreshold: 0.9,
        parallelismLevel: 1,
        timeoutMs: 15000,
        focus: 'solution_generation',
        outputOptimization: 'direct_consumption'
      },
      agents: [{
        type: 'strategy_mapper',
        name: 'mapper-agent',
        enabled: true,
        weight: 0.8,
        config: {
          model: 'gpt-4',
          temperature: 0.3
        },
        dependencies: ['intelligence_analyst']
      }],
      stages: [{
        name: 'strategy-stage',
        type: 'strategy_analysis',
        enabled: true,
        parallel: true,
        weight: 0.9,
        config: { depth: 'deep' },
        conditions: {
          runIf: 'hasInput',
          skipIf: 'isEmpty'
        }
      }],
      verification: [{
        type: 'semantic',
        enabled: true,
        weight: 0.7,
        config: { threshold: 0.8 }
      }],
      output: {
        format: 'comprehensive',
        sections: ['analysis', 'recommendations', 'metrics'],
        densityTarget: 5000,
        optimizeFor: 'human_readable',
        includeDebateTopics: false,
        includeMetrics: true,
        includeCitations: true
      },
      performance: {
        cacheResults: false,
        cacheTtlMinutes: 60,
        maxMemoryMb: 2048,
        cpuIntensive: true
      }
    };

    it('should return valid loadout for correct input', () => {
      const result = validateLoadout(validLoadout);
      expect(result).toEqual(validLoadout);
    });

    it('should throw for invalid loadout', () => {
      expect(() => validateLoadout(null)).toThrow();
      expect(() => validateLoadout({})).toThrow();
      
      const invalidLoadout = { ...validLoadout };
      delete invalidLoadout.description;
      expect(() => validateLoadout(invalidLoadout)).toThrow();
    });

    it('should validate nested structures', () => {
      // Test pipeline validation boundaries
      const invalidPipelineMax = {
        ...validLoadout,
        pipeline: { ...validLoadout.pipeline, maxIterations: 0 } // below min of 1
      };
      expect(() => validateLoadout(invalidPipelineMax)).toThrow();

      const invalidTimeout = {
        ...validLoadout,
        pipeline: { ...validLoadout.pipeline, timeoutMs: 500 } // below min of 1000
      };
      expect(() => validateLoadout(invalidTimeout)).toThrow();
    });

    it('should validate agent weight boundaries', () => {
      const invalidWeight = {
        ...validLoadout,
        agents: [{
          ...validLoadout.agents[0],
          weight: 1.5 // above max of 1.0
        }]
      };
      expect(() => validateLoadout(invalidWeight)).toThrow();
    });

    it('should validate output density target', () => {
      const invalidDensity = {
        ...validLoadout,
        output: {
          ...validLoadout.output,
          densityTarget: 400 // below min of 500
        }
      };
      expect(() => validateLoadout(invalidDensity)).toThrow();
    });
  });

  describe('ResearchLoadoutSchema', () => {
    it('should validate enum values correctly', () => {
      const testData = {
        name: 'test',
        description: 'test',
        version: '1.0.0',
        pipeline: {
          maxIterations: 5,
          convergenceThreshold: 0.8,
          parallelismLevel: 2,
          timeoutMs: 30000,
          focus: 'intelligence_gathering' as const,
          outputOptimization: 'swarm_handoff' as const
        },
        agents: [],
        stages: [],
        verification: [],
        output: {
          format: 'intelligence_brief' as const,
          sections: [],
          densityTarget: 1000,
          optimizeFor: 'swarm_consumption' as const,
          includeDebateTopics: false,
          includeMetrics: false,
          includeCitations: false
        },
        performance: {
          cacheResults: true,
          cacheTtlMinutes: 30,
          maxMemoryMb: 512,
          cpuIntensive: false
        }
      };

      // Test valid enum values
      expect(() => ResearchLoadoutSchema.parse(testData)).not.toThrow();

      // Test invalid enum values
      const invalidFocus = {
        ...testData,
        pipeline: { ...testData.pipeline, focus: 'invalid' as any }
      };
      expect(() => ResearchLoadoutSchema.parse(invalidFocus)).toThrow();
    });

    it('should validate array constraints', () => {
      const baseLoadout = {
        name: 'test',
        description: 'test', 
        version: '1.0.0',
        pipeline: {
          maxIterations: 5,
          convergenceThreshold: 0.8,
          parallelismLevel: 2,
          timeoutMs: 30000,
          focus: 'intelligence_gathering' as const,
          outputOptimization: 'swarm_handoff' as const
        },
        agents: [],
        stages: [],
        verification: [],
        output: {
          format: 'overview' as const,
          sections: [],
          densityTarget: 1000,
          optimizeFor: 'human_readable' as const,
          includeDebateTopics: false,
          includeMetrics: false,
          includeCitations: false
        },
        performance: {
          cacheResults: true,
          cacheTtlMinutes: 1, // minimum value
          maxMemoryMb: 256, // minimum value
          cpuIntensive: false
        }
      };

      expect(() => ResearchLoadoutSchema.parse(baseLoadout)).not.toThrow();
    });
  });
});