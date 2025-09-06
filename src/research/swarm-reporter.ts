/**
 * Swarm-Optimized Report Generation System
 * 
 * Generates dense, comprehensive intelligence reports specifically optimized
 * for consumption by refinement swarms. Focus is on actionable intelligence
 * and debate framing rather than solution generation.
 */

import type { Logger } from '@/types';
import { createLogger } from '@/utils/logger';
import {
  IntelligenceGatheringResult,
  SwarmIntelligenceBrief,
  DebateTopic,
  DebatePosition,
  ApproachVector,
  StrategyRecommendations,
  KeyInsight,
  Evidence,
  RiskFactor,
  Constraint,
  ComplexityMetrics,
  ResearchLoadout
} from './types';

export interface SwarmReportConfig {
  densityTarget: number; // target token count
  maxSectionTokens: number;
  includeDebateTopics: boolean;
  includeMetrics: boolean;
  includeCitations: boolean;
  optimizeFor: 'swarm_consumption' | 'human_readable';
  debateStructureDepth: 'shallow' | 'medium' | 'deep';
}

export interface ReportSection {
  title: string;
  content: string;
  tokenCount: number;
  priority: number; // 0-1, higher is more important
  category: 'executive' | 'strategic' | 'technical' | 'risk' | 'debate' | 'implementation';
}

export class SwarmOptimizedReporter {
  private logger: Logger;

  constructor() {
    this.logger = createLogger('mimir.research.swarm-reporter');
  }

  async generateIntelligenceBrief(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<string> {
    this.logger.info('Generating swarm-optimized intelligence brief', {
      densityTarget: config.densityTarget,
      optimization: config.optimizeFor
    });

    // Generate all potential sections
    const sections = await this.generateAllSections(intelligence, config);

    // Optimize section selection and ordering for target density
    const optimizedSections = this.optimizeSectionSelection(sections, config);

    // Generate final report
    const report = this.assembleFinalReport(optimizedSections, intelligence, config);

    this.logger.info('Intelligence brief generated', {
      sectionCount: optimizedSections.length,
      estimatedTokens: optimizedSections.reduce((sum, s) => sum + s.tokenCount, 0)
    });

    return report;
  }

  private async generateAllSections(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection[]> {
    const sections: ReportSection[] = [];

    // Executive Intelligence Summary (highest priority)
    sections.push(await this.generateExecutiveSummary(intelligence, config));

    // Debate Framework (critical for swarm optimization)
    if (config.includeDebateTopics) {
      sections.push(await this.generateDebateFramework(intelligence, config));
    }

    // Strategic Approach Landscape
    sections.push(await this.generateApproachLandscape(intelligence, config));

    // Technical Constraint Analysis
    sections.push(await this.generateTechnicalConstraints(intelligence, config));

    // Risk and Complexity Profile
    sections.push(await this.generateRiskProfile(intelligence, config));

    // Implementation Intelligence
    sections.push(await this.generateImplementationIntelligence(intelligence, config));

    // Tooling and Resource Guide
    sections.push(await this.generateToolingGuidance(intelligence, config));

    // Success Criteria Definition
    sections.push(await this.generateSuccessCriteria(intelligence, config));

    // Execution metrics (if requested)
    if (config.includeMetrics) {
      sections.push(await this.generateMetricsSection(intelligence, config));
    }

    return sections;
  }

  private async generateExecutiveSummary(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const keyInsights = intelligence.findings.keyInsights
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 5);

    const primaryApproaches = intelligence.strategicGuidance.primaryApproaches
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    const topRisks = intelligence.riskFactors
      .sort((a, b) => b.riskScore - a.riskScore)
      .slice(0, 3);

    const content = `## EXECUTIVE INTELLIGENCE SUMMARY

**Target:** ${intelligence.loadoutName} analysis results
**Complexity Assessment:** ${intelligence.complexityAssessment.overallComplexity.toFixed(1)}/10
**Verification Score:** ${intelligence.verificationScore.toFixed(2)}

### Key Strategic Insights:
${keyInsights.map((insight, i) => 
  `${i + 1}. **${insight.insight}** (confidence: ${(insight.confidence * 100).toFixed(0)}%)
     - Category: ${insight.category}
     - Implications: ${insight.implications.slice(0, 2).join(', ')}${insight.implications.length > 2 ? '...' : ''}`
).join('\n\n')}

### Primary Strategic Approaches:
${primaryApproaches.map((approach, i) => 
  `${i + 1}. **${approach.name}** 
     - Confidence: ${(approach.confidence * 100).toFixed(0)}% | Complexity: ${approach.complexity}/10 | Risk: ${approach.riskLevel}/10
     - ${approach.description}`
).join('\n\n')}

### Critical Risk Factors:
${topRisks.map((risk, i) => 
  `${i + 1}. **${risk.risk}** (Risk Score: ${risk.riskScore.toFixed(1)})
     - Probability: ${(risk.probability * 100).toFixed(0)}% | Impact: ${risk.impact}/10
     - Category: ${risk.category}`
).join('\n\n')}

### Swarm Guidance:
- **Debate Focus:** ${intelligence.debateFramework.length} critical topics identified
- **Evidence Base:** ${this.countEvidence(intelligence)} pieces of supporting evidence
- **Constraint Level:** ${intelligence.implementationConstraints.length} hard constraints
- **Complexity Warning:** ${intelligence.complexityAssessment.overallComplexity > 7 ? 'HIGH complexity - expect extended debate cycles' : 'Moderate complexity - standard debate approach suitable'}`;

    return {
      title: 'Executive Intelligence Summary',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 1.0,
      category: 'executive'
    };
  }

  private async generateDebateFramework(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const debateTopics = intelligence.debateFramework
      .sort((a, b) => b.importance - a.importance)
      .slice(0, 5); // Focus on top 5 most important topics

    const topRisks = intelligence.riskFactors
      .sort((a, b) => b.riskScore - a.riskScore)
      .slice(0, 3);

    const content = `## DEBATE FRAMEWORK FOR REFINEMENT SWARM

### Recommended Debate Structure:
1. **Constraint Validation Phase** (5 minutes)
   - Validate all solutions respect hard constraints
   - Eliminate non-viable approaches early
   - Focus: Technical feasibility and business requirements

2. **Strategic Approach Comparison** (15-20 minutes)
   - Compare primary approaches on key dimensions
   - Focus: Trade-off analysis, not absolute right/wrong
   - Weight: Long-term maintainability and evolution

3. **Implementation Feasibility Assessment** (10-15 minutes)
   - Assess real-world implementation challenges
   - Consider team expertise and timeline constraints
   - Focus: Practical execution over theoretical perfection

### Priority Debate Topics:

${debateTopics.map((topic, i) => {
  const positionsList = topic.positions.length > 0 
    ? topic.positions.map((pos, j) => 
        `     ${j + 1}. **${pos.position}** (confidence: ${(pos.confidence * 100).toFixed(0)}%)
        - Risk: ${pos.riskProfile}
        - Key evidence: ${pos.supportingEvidence.slice(0, 2).map(e => e.description).join(', ')}${pos.supportingEvidence.length > 2 ? '...' : ''}`
      ).join('\n\n')
    : '     No specific positions identified - open exploration recommended';

  return `**${i + 1}. ${topic.topic}** (Importance: ${(topic.importance * 100).toFixed(0)}% | Est. ${topic.timeEstimate} min)
   - **Category:** ${topic.category}
   - **Stakes:** ${topic.stakesDescription}
   - **Structure:** ${topic.suggestedDebateStructure}
   
   **Positions to Consider:**
${positionsList}`;
}).join('\n\n')}

### Debate Success Metrics:
- **Convergence Target:** ${intelligence.complexityAssessment.overallComplexity > 7 ? '85%' : '80%'} agreement on final approach
- **Evidence Threshold:** Each position must cite at least 2 pieces of supporting evidence
- **Risk Assessment:** All solutions must address top ${Math.min(topRisks.length, 3)} identified risks
- **Constraint Compliance:** 100% compliance with hard constraints required

### Anti-Patterns to Avoid in Debate:
${intelligence.strategicGuidance.antiPatterns.slice(0, 3).map(pattern => 
  `- **${pattern.name}:** ${pattern.whyProblematic} → Alternative: ${pattern.alternatives.slice(0, 2).join(', ')}`
).join('\n')}`;

    // topRisks already declared above

    return {
      title: 'Debate Framework',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.95,
      category: 'debate'
    };
  }

  private async generateApproachLandscape(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const approaches = intelligence.strategicGuidance.primaryApproaches
      .sort((a, b) => b.confidence - a.confidence);
    
    const hybrids = intelligence.strategicGuidance.hybridOpportunities
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, 3);

    const tradeoffs = intelligence.strategicGuidance.approachTradeoffs;

    const content = `## STRATEGIC APPROACH LANDSCAPE

### Primary Approaches (Ranked by Confidence):

${approaches.map((approach, i) => 
  `**${i + 1}. ${approach.name}**
   - **Confidence:** ${(approach.confidence * 100).toFixed(0)}%
   - **Complexity:** ${approach.complexity}/10 | **Risk Level:** ${approach.riskLevel}/10 | **Implementation Effort:** ${approach.implementationEffort}/10
   - **Description:** ${approach.description}
   - **Supporting Evidence:** ${approach.supportingEvidence.slice(0, 2).map(e => e.description).join('; ')}${approach.supportingEvidence.length > 2 ? '...' : ''}
   - **Potential Downsides:** ${approach.potentialDownsides.slice(0, 2).join('; ')}${approach.potentialDownsides.length > 2 ? '...' : ''}
   `
).join('\n')}

### Hybrid Strategy Opportunities:

${hybrids.length > 0 ? hybrids.map((hybrid, i) => 
  `**${i + 1}. ${hybrid.name}**
   - **Combines:** ${hybrid.combinesApproaches.join(' + ')}
   - **Confidence:** ${(hybrid.confidence * 100).toFixed(0)}%
   - **Key Synergies:** ${hybrid.synergies.slice(0, 2).join(', ')}${hybrid.synergies.length > 2 ? '...' : ''}
   - **Additional Complexity:** +${hybrid.additionalComplexity}/10 over individual approaches
   - **Description:** ${hybrid.description}`
).join('\n\n') : 'No viable hybrid approaches identified at this analysis depth.'}

### Approach Trade-off Matrix:

${tradeoffs.dimensions.length > 0 ? `
**Evaluation Dimensions:** ${tradeoffs.dimensions.join(', ')}
**Dimension Weights:** ${tradeoffs.dimensions.map(dim => `${dim}: ${((tradeoffs.weights[dim] || 0) * 100).toFixed(0)}%`).join(', ')}

${approaches.slice(0, 4).map(approach => {
  const scores = tradeoffs.scores[approach.name] || {};
  const scoreStr = tradeoffs.dimensions.map(dim => `${dim}: ${(scores[dim] || 0).toFixed(1)}`).join(' | ');
  return `- **${approach.name}:** ${scoreStr}`;
}).join('\n')}
` : 'Detailed trade-off analysis not available - recommend manual evaluation during debate.'}

### Eliminated/Problematic Approaches:

${intelligence.strategicGuidance.antiPatterns.map(pattern => 
  `- **${pattern.name}:** ${pattern.whyProblematic}
    - Common triggers: ${pattern.commonTriggers.slice(0, 2).join(', ')}
    - Alternatives: ${pattern.alternatives.slice(0, 2).join(', ')}`
).join('\n\n')}`;

    return {
      title: 'Strategic Approach Landscape',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.9,
      category: 'strategic'
    };
  }

  private async generateTechnicalConstraints(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const hardConstraints = intelligence.implementationConstraints
      .filter(c => c.severity === 'hard')
      .sort((a, b) => (b.impact?.length || 0) - (a.impact?.length || 0));

    const softConstraints = intelligence.implementationConstraints
      .filter(c => c.severity === 'soft')
      .slice(0, 5);

    const content = `## TECHNICAL CONSTRAINT ANALYSIS

### Hard Constraints (Non-Negotiable):

${hardConstraints.length > 0 ? hardConstraints.map((constraint, i) => 
  `**${i + 1}. ${constraint.name}** (${constraint.type})
   - **Description:** ${constraint.description}
   - **Impact:** ${constraint.impact}
   - **Workarounds:** ${constraint.workarounds.length > 0 ? constraint.workarounds.slice(0, 2).join('; ') : 'None identified'}
   `
).join('\n') : 'No hard technical constraints identified.'}

### Soft Constraints (Preferences):

${softConstraints.length > 0 ? softConstraints.map((constraint, i) => 
  `**${i + 1}. ${constraint.name}** (${constraint.type})
   - **Description:** ${constraint.description}
   - **Workarounds:** ${constraint.workarounds.slice(0, 2).join('; ')}`
).join('\n\n') : 'No significant soft constraints identified.'}

### Constraint Impact on Approach Selection:

${this.analyzeConstraintImpact(intelligence.implementationConstraints, intelligence.strategicGuidance.primaryApproaches)}

### Integration Points & Dependencies:

${intelligence.domainContext && intelligence.domainContext.technicalContext ? 
  this.analyzeTechnicalIntegrations(intelligence.domainContext.technicalContext) : 
  'Technical integration analysis not available - recommend manual assessment of system dependencies.'}`;

    return {
      title: 'Technical Constraint Analysis',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.85,
      category: 'technical'
    };
  }

  private async generateRiskProfile(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const criticalRisks = intelligence.riskFactors
      .filter(r => r.riskScore >= 6)
      .sort((a, b) => b.riskScore - a.riskScore);

    const moderateRisks = intelligence.riskFactors
      .filter(r => r.riskScore >= 3 && r.riskScore < 6)
      .sort((a, b) => b.riskScore - a.riskScore)
      .slice(0, 5);

    const riskByCategory = this.groupRisksByCategory(intelligence.riskFactors);

    const content = `## RISK AND COMPLEXITY PROFILE

### Overall Complexity Assessment:
- **Technical Complexity:** ${intelligence.complexityAssessment.technicalComplexity.toFixed(1)}/10
- **Domain Complexity:** ${intelligence.complexityAssessment.domainComplexity.toFixed(1)}/10
- **Implementation Complexity:** ${intelligence.complexityAssessment.implementationComplexity.toFixed(1)}/10
- **Risk Complexity:** ${intelligence.complexityAssessment.riskComplexity.toFixed(1)}/10
- **Overall:** ${intelligence.complexityAssessment.overallComplexity.toFixed(1)}/10

### Critical Risks (Risk Score ≥ 6):

${criticalRisks.length > 0 ? criticalRisks.map((risk, i) => 
  `**${i + 1}. ${risk.risk}** (Score: ${risk.riskScore.toFixed(1)})
   - **Probability:** ${(risk.probability * 100).toFixed(0)}% | **Impact:** ${risk.impact}/10
   - **Category:** ${risk.category}
   - **Mitigation Strategies:**
     ${risk.mitigationStrategies.slice(0, 2).map(strategy => `• ${strategy}`).join('\n     ')}
   - **Contingency Plans:**
     ${risk.contingencyPlans.slice(0, 2).map(plan => `• ${plan}`).join('\n     ')}`
).join('\n\n') : 'No critical risks identified.'}

### Moderate Risks (Score 3-6):

${moderateRisks.length > 0 ? moderateRisks.map(risk => 
  `- **${risk.risk}** (${risk.riskScore.toFixed(1)}) - ${risk.category}: ${risk.mitigationStrategies[0] || 'No mitigation identified'}`
).join('\n') : 'No moderate risks identified.'}

### Risk Analysis by Category:

${Object.entries(riskByCategory).map(([category, risks]) => 
  `**${category.toUpperCase()}** (${risks.length} risks, avg score: ${(risks.reduce((sum, r) => sum + r.riskScore, 0) / risks.length).toFixed(1)})
   - Top concern: ${risks[0]?.risk || 'None'}
   - Mitigation focus: ${this.extractTopMitigation(risks)}`
).join('\n\n')}

### Complexity Factors:

${intelligence.complexityAssessment.complexityFactors.slice(0, 5).map((factor, i) => 
  `**${i + 1}. ${factor.factor}** (Impact: ${factor.impact}/10)
   - ${factor.description}
   - Mitigations: ${factor.mitigationStrategies.slice(0, 2).join(', ')}`
).join('\n\n')}

### Risk Monitoring Recommendations:

- **High Priority:** Monitor ${criticalRisks.length} critical risks with ${criticalRisks.length > 2 ? 'weekly' : 'bi-weekly'} check-ins
- **Escalation Triggers:** Any critical risk probability increase >20% or new risks with score >7
- **Mitigation Budget:** Recommend ${criticalRisks.length * 15}% additional effort for risk mitigation
- **Contingency Readiness:** Have fallback plans ready for top ${Math.min(criticalRisks.length, 3)} risks`;

    return {
      title: 'Risk and Complexity Profile',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.8,
      category: 'risk'
    };
  }

  private async generateImplementationIntelligence(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const approaches = intelligence.strategicGuidance.primaryApproaches.slice(0, 3);
    
    const content = `## IMPLEMENTATION INTELLIGENCE MAP

### Implementation Readiness Assessment:

${approaches.map((approach, i) => 
  `**${approach.name}**
   - **Readiness Score:** ${this.calculateReadinessScore(approach)}/10
   - **Implementation Effort:** ${approach.implementationEffort}/10
   - **Key Implementation Steps:**
     ${this.generateImplementationSteps(approach)}
   - **Success Factors:**
     ${this.extractSuccessFactors(approach)}
   - **Failure Points:**
     ${approach.potentialDownsides.slice(0, 2).map(downside => `• ${downside}`).join('\n     ')}`
).join('\n\n')}

### Resource Requirements:

${this.generateResourceAnalysis(intelligence)}

### Timeline Considerations:

${this.generateTimelineGuidance(intelligence)}

### Team Expertise Requirements:

${this.generateExpertiseRequirements(intelligence)}

### Integration Complexity:

${intelligence.domainContext ? this.analyzeIntegrationComplexity(intelligence.domainContext) : 'Integration analysis not available - recommend manual assessment.'}

### Quality Gates & Validation:

- **Architecture Review:** Required before implementation start
- **Prototype Validation:** ${intelligence.complexityAssessment.overallComplexity > 7 ? 'Mandatory' : 'Recommended'} for complex approaches  
- **Performance Baseline:** Establish before changes, validate during implementation
- **Security Review:** ${this.requiresSecurityReview(intelligence) ? 'Mandatory' : 'Standard process'}
- **Rollback Readiness:** ${this.calculateRollbackComplexity(intelligence)} complexity rollback plan required`;

    return {
      title: 'Implementation Intelligence Map',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.75,
      category: 'implementation'
    };
  }

  private async generateToolingGuidance(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const tooling = intelligence.toolingGuidance;

    const content = `## TOOLING AND RESOURCE GUIDANCE

### Recommended Tools:

${tooling.recommended.length > 0 ? tooling.recommended.slice(0, 5).map((tool, i) => 
  `**${i + 1}. ${tool.tool}**
   - **Purpose:** ${tool.purpose}
   - **Confidence:** ${(tool.confidence * 100).toFixed(0)}%
   - **Learning Curve:** ${tool.learningCurve} | **Integration Complexity:** ${tool.integrationComplexity}
   - **Reasoning:** ${tool.reasoning}
   - **Alternatives:** ${tool.alternatives.slice(0, 2).join(', ')}`
).join('\n\n') : 'No specific tool recommendations available.'}

### Integration Considerations:

${tooling.integrationGuidance.length > 0 ? tooling.integrationGuidance.slice(0, 3).map(guide => 
  `- ${guide.description || 'Integration guidance available'}`
).join('\n') : 'Standard integration practices apply.'}

### Tooling Risks:

${tooling.toolingRisks.length > 0 ? tooling.toolingRisks.slice(0, 3).map(risk => 
  `- **${risk.tool || 'General'}:** ${risk.risk || risk.description} (Mitigation: ${risk.mitigation || 'Standard practices'})`
).join('\n') : 'No significant tooling risks identified.'}

### Development Environment Setup:

${this.generateDevEnvironmentGuidance(intelligence)}

### Monitoring and Observability:

${this.generateMonitoringGuidance(intelligence)}`;

    return {
      title: 'Tooling and Resource Guide',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.6,
      category: 'implementation'
    };
  }

  private async generateSuccessCriteria(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const content = `## SUCCESS CRITERIA DEFINITION

### Primary Success Metrics:

1. **Functional Success:**
   - All core requirements met with ${intelligence.verificationScore >= 0.9 ? 'high' : 'acceptable'} confidence
   - No critical defects in production within first 30 days
   - User acceptance criteria satisfied

2. **Technical Success:**
   - Performance requirements met or exceeded
   - Security standards maintained
   - Maintainability score ≥ ${intelligence.complexityAssessment.overallComplexity > 7 ? '7' : '6'}/10

3. **Process Success:**
   - Implementation completed within ${this.estimateTimelineBuffer(intelligence)}% of estimated timeline
   - Budget variance ≤ 20%
   - Team knowledge transfer completed

### Quality Gates:

**Phase 1: Design Validation**
- [ ] Architecture review passed
- [ ] Security design review completed
- [ ] Performance requirements validated
- [ ] All hard constraints addressed

**Phase 2: Implementation Validation**
- [ ] Unit test coverage ≥ 90%
- [ ] Integration tests passing
- [ ] Performance baseline maintained
- [ ] Security scan clean

**Phase 3: Deployment Validation**
- [ ] Rollback plan tested
- [ ] Monitoring and alerting configured
- [ ] Documentation complete
- [ ] Team training completed

### Acceptance Criteria:

${this.generateAcceptanceCriteria(intelligence)}

### Risk Tolerance:

- **Critical Risk Threshold:** No more than ${this.calculateCriticalRiskThreshold(intelligence)} critical risks active during implementation
- **Performance Degradation:** <5% acceptable during transition
- **Downtime:** Maximum ${this.calculateDowntimeTolerance(intelligence)} during deployment

### Success Validation Timeline:

- **Immediate (Day 1):** Core functionality operational
- **Short-term (Week 1):** Performance within acceptable range
- **Medium-term (Month 1):** No critical issues, user adoption on track  
- **Long-term (Quarter 1):** Maintainability and evolution capability demonstrated`;

    return {
      title: 'Success Criteria Definition',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.7,
      category: 'implementation'
    };
  }

  private async generateMetricsSection(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): Promise<ReportSection> {
    const metrics = intelligence.executionMetrics;

    const content = `## ANALYSIS EXECUTION METRICS

### Research Execution Summary:
- **Duration:** ${metrics.duration}ms (${(metrics.duration / 1000).toFixed(1)}s)
- **Iterations Completed:** ${metrics.iterationsCompleted}
- **Convergence Score:** ${metrics.convergenceScore.toFixed(2)}
- **Verification Score:** ${intelligence.verificationScore.toFixed(2)}

### Stage Performance:
${metrics.stageResults.map(stage => 
  `- **${stage.stageName}:** ${stage.success ? '✓' : '✗'} (${stage.duration}ms, confidence: ${stage.confidence.toFixed(2)})`
).join('\n')}

### Quality Metrics:
- **Completeness:** ${(metrics.qualityMetrics.completeness * 100).toFixed(0)}%
- **Accuracy:** ${(metrics.qualityMetrics.accuracy * 100).toFixed(0)}%  
- **Density:** ${(metrics.qualityMetrics.density * 100).toFixed(0)}%
- **Relevance:** ${(metrics.qualityMetrics.relevance * 100).toFixed(0)}%
- **Actionability:** ${(metrics.qualityMetrics.actionability * 100).toFixed(0)}%
- **Overall Quality:** ${(metrics.qualityMetrics.overall * 100).toFixed(0)}%

### Resource Usage:
- **Peak Memory:** ${metrics.resourceUsage.maxMemoryMb}MB
- **Avg CPU:** ${metrics.resourceUsage.avgCpuPercent}%
- **Network Requests:** ${metrics.resourceUsage.networkRequests}
- **Cache Hit Rate:** ${((metrics.resourceUsage.cacheHits / (metrics.resourceUsage.cacheHits + metrics.resourceUsage.cacheMisses)) * 100).toFixed(0)}%`;

    return {
      title: 'Analysis Execution Metrics',
      content,
      tokenCount: this.estimateTokenCount(content),
      priority: 0.3,
      category: 'technical'
    };
  }

  private optimizeSectionSelection(
    sections: ReportSection[],
    config: SwarmReportConfig
  ): ReportSection[] {
    // Sort by priority
    sections.sort((a, b) => b.priority - a.priority);

    let totalTokens = 0;
    const selectedSections: ReportSection[] = [];

    // Always include executive summary
    const execSummary = sections.find(s => s.category === 'executive');
    if (execSummary) {
      selectedSections.push(execSummary);
      totalTokens += execSummary.tokenCount;
    }

    // Add other sections based on priority and token budget
    for (const section of sections) {
      if (section.category === 'executive') continue; // Already added

      if (totalTokens + section.tokenCount <= config.densityTarget) {
        selectedSections.push(section);
        totalTokens += section.tokenCount;
      } else if (totalTokens < config.densityTarget * 0.8) {
        // Try to include truncated version if we're under 80% of target
        const remainingTokens = config.densityTarget - totalTokens;
        if (remainingTokens > config.maxSectionTokens * 0.3) {
          const truncatedSection = this.truncateSection(section, remainingTokens);
          selectedSections.push(truncatedSection);
          totalTokens += truncatedSection.tokenCount;
          break;
        }
      }
    }

    this.logger.info('Section optimization completed', {
      originalSections: sections.length,
      selectedSections: selectedSections.length,
      targetTokens: config.densityTarget,
      actualTokens: totalTokens
    });

    return selectedSections;
  }

  private truncateSection(section: ReportSection, maxTokens: number): ReportSection {
    // Simple truncation - could be made more sophisticated
    const lines = section.content.split('\n');
    const avgTokensPerLine = section.tokenCount / lines.length;
    const maxLines = Math.floor(maxTokens / avgTokensPerLine);

    const truncatedContent = lines.slice(0, maxLines).join('\n') + '\n\n[Content truncated for brevity...]';

    return {
      ...section,
      content: truncatedContent,
      tokenCount: maxTokens,
      title: section.title + ' (Truncated)'
    };
  }

  private assembleFinalReport(
    sections: ReportSection[],
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): string {
    const header = this.generateReportHeader(intelligence, config);
    const toc = this.generateTableOfContents(sections);
    const sectionContent = sections.map(section => section.content).join('\n\n---\n\n');
    const footer = this.generateReportFooter(intelligence, config);

    return `${header}\n\n${toc}\n\n---\n\n${sectionContent}\n\n---\n\n${footer}`;
  }

  private generateReportHeader(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): string {
    return `# INTELLIGENCE BRIEF FOR REFINEMENT SWARM

**Generated:** ${new Date().toISOString()}
**Loadout:** ${intelligence.loadoutName}
**Target Density:** ${config.densityTarget} tokens
**Optimization:** ${config.optimizeFor}
**Verification Score:** ${intelligence.verificationScore.toFixed(2)}

> **SWARM GUIDANCE:** This brief is optimized for debate-style refinement. Focus on strategic approaches and trade-offs rather than implementation details. Evidence and constraints are provided for informed debate.`;
  }

  private generateTableOfContents(sections: ReportSection[]): string {
    return `## TABLE OF CONTENTS\n\n` + sections.map((section, i) => 
      `${i + 1}. ${section.title} (${section.category})`
    ).join('\n');
  }

  private generateReportFooter(
    intelligence: IntelligenceGatheringResult,
    config: SwarmReportConfig
  ): string {
    return `## SWARM HANDOFF CHECKLIST

- [ ] **Constraint Validation:** All hard constraints identified and documented
- [ ] **Approach Evaluation:** ${intelligence.strategicGuidance.primaryApproaches.length} primary approaches ready for debate
- [ ] **Risk Assessment:** ${intelligence.riskFactors.length} risks identified and mitigation strategies provided
- [ ] **Evidence Base:** ${this.countEvidence(intelligence)} pieces of supporting evidence available
- [ ] **Success Criteria:** Clear definition of acceptable outcomes established
- [ ] **Debate Framework:** ${intelligence.debateFramework.length} structured debate topics prepared

**Next Steps:** Begin structured debate with constraint validation, proceed through approach comparison, conclude with implementation feasibility assessment.

*Generated by Mimir Intelligence Engine v${intelligence.loadoutName} - ${new Date().toISOString()}*`;
  }

  // Helper methods
  private estimateTokenCount(text: string): number {
    // Rough estimation: ~4 characters per token
    return Math.ceil(text.length / 4);
  }

  private countEvidence(intelligence: IntelligenceGatheringResult): number {
    return intelligence.findings.keyInsights.reduce((sum, insight) => 
      sum + insight.supportingEvidence.length, 0
    ) + intelligence.strategicGuidance.primaryApproaches.reduce((sum, approach) =>
      sum + approach.supportingEvidence.length, 0
    );
  }

  private analyzeConstraintImpact(constraints: Constraint[], approaches: ApproachVector[]): string {
    if (constraints.length === 0 || approaches.length === 0) {
      return 'No constraint-approach impact analysis available.';
    }

    const hardConstraints = constraints.filter(c => c.severity === 'hard');
    return `${hardConstraints.length} hard constraints will significantly impact approach selection. Key considerations:\n` +
      hardConstraints.slice(0, 3).map(c => `- ${c.name}: May eliminate approaches that ${c.impact}`).join('\n');
  }

  private analyzeTechnicalIntegrations(techContext: any): string {
    return 'Technical integration analysis requires manual assessment based on current architecture.';
  }

  private groupRisksByCategory(risks: RiskFactor[]): Record<string, RiskFactor[]> {
    const grouped: Record<string, RiskFactor[]> = {};
    
    for (const risk of risks) {
      if (!grouped[risk.category]) {
        grouped[risk.category] = [];
      }
      grouped[risk.category].push(risk);
    }

    // Sort risks within each category by score
    Object.keys(grouped).forEach(category => {
      grouped[category].sort((a, b) => b.riskScore - a.riskScore);
    });

    return grouped;
  }

  private extractTopMitigation(risks: RiskFactor[]): string {
    const allMitigations = risks.flatMap(r => r.mitigationStrategies);
    const mitigationCounts = new Map<string, number>();
    
    allMitigations.forEach(mitigation => {
      mitigationCounts.set(mitigation, (mitigationCounts.get(mitigation) || 0) + 1);
    });

    const topMitigation = Array.from(mitigationCounts.entries())
      .sort(([, a], [, b]) => b - a)[0];

    return topMitigation ? topMitigation[0] : 'Standard risk management practices';
  }

  private calculateReadinessScore(approach: ApproachVector): number {
    // Simple readiness calculation based on confidence and risk
    return Math.max(1, Math.min(10, (approach.confidence * 10) - (approach.riskLevel * 0.5)));
  }

  private generateImplementationSteps(approach: ApproachVector): string {
    // Generate basic implementation steps based on approach characteristics
    const steps = [
      '1. Architecture validation and planning',
      '2. Core component implementation',
      '3. Integration and testing phase',
      '4. Performance optimization and validation'
    ];

    if (approach.riskLevel > 6) {
      steps.splice(1, 0, '1.5. Risk mitigation prototype');
    }

    return steps.map(step => `     ${step}`).join('\n');
  }

  private extractSuccessFactors(approach: ApproachVector): string {
    return approach.supportingEvidence
      .slice(0, 2)
      .map(evidence => `• ${evidence.description}`)
      .join('\n     ') || '     • Evidence-based approach validation required';
  }

  private generateResourceAnalysis(intelligence: IntelligenceGatheringResult): string {
    const complexity = intelligence.complexityAssessment.overallComplexity;
    
    if (complexity > 7) {
      return `High complexity project requiring:
- Senior developer time: 60-80% of implementation effort
- Architecture review cycles: 2-3 iterations expected
- Testing effort: 40-50% of development time
- Documentation effort: 15-20% of total project time`;
    } else if (complexity > 4) {
      return `Medium complexity project requiring:
- Mixed skill level team acceptable
- Architecture review: 1-2 iterations expected  
- Testing effort: 30-40% of development time
- Documentation effort: 10-15% of total project time`;
    } else {
      return `Low complexity project:
- Junior-friendly with senior oversight
- Standard development practices sufficient
- Testing effort: 20-30% of development time`;
    }
  }

  private generateTimelineGuidance(intelligence: IntelligenceGatheringResult): string {
    const riskCount = intelligence.riskFactors.filter(r => r.riskScore >= 6).length;
    const complexity = intelligence.complexityAssessment.overallComplexity;
    
    let buffer = 20; // base 20% buffer
    if (complexity > 7) buffer += 30;
    if (riskCount > 2) buffer += 20;
    
    return `Recommended timeline buffer: ${buffer}% over base estimates
- Risk mitigation: ${riskCount * 10}% additional effort
- Complexity overhead: ${complexity > 7 ? 30 : 15}% additional effort
- Integration testing: ${complexity > 5 ? 'Extended' : 'Standard'} testing cycles`;
  }

  private generateExpertiseRequirements(intelligence: IntelligenceGatheringResult): string {
    const complexity = intelligence.complexityAssessment.overallComplexity;
    const techFindings = intelligence.findings.technicalFindings;
    
    return `Required expertise level: ${complexity > 7 ? 'Senior' : complexity > 4 ? 'Intermediate' : 'Junior'}-level
Key skill areas: ${techFindings.slice(0, 3).map(f => f.category || 'general').join(', ')}
Team composition: ${complexity > 7 ? '1 senior + 2-3 developers' : '1-2 developers with senior consultation'}`;
  }

  private analyzeIntegrationComplexity(domainContext: any): string {
    return 'Integration complexity analysis requires domain-specific assessment.';
  }

  private requiresSecurityReview(intelligence: IntelligenceGatheringResult): boolean {
    return intelligence.riskFactors.some(r => r.category === 'security') ||
           intelligence.complexityAssessment.overallComplexity > 6;
  }

  private calculateRollbackComplexity(intelligence: IntelligenceGatheringResult): string {
    const complexity = intelligence.complexityAssessment.overallComplexity;
    return complexity > 7 ? 'High' : complexity > 4 ? 'Medium' : 'Low';
  }

  private generateDevEnvironmentGuidance(intelligence: IntelligenceGatheringResult): string {
    return `Development environment considerations:
- Local setup complexity: ${intelligence.complexityAssessment.technicalComplexity > 6 ? 'High - containerized environment recommended' : 'Standard'}
- Dependencies: Review tooling recommendations above
- Testing environment: ${intelligence.riskFactors.length > 5 ? 'Dedicated staging required' : 'Standard test environment sufficient'}`;
  }

  private generateMonitoringGuidance(intelligence: IntelligenceGatheringResult): string {
    return `Monitoring requirements:
- Error tracking: ${intelligence.riskFactors.length > 3 ? 'Enhanced' : 'Standard'} error monitoring required
- Performance monitoring: ${intelligence.complexityAssessment.overallComplexity > 6 ? 'Real-time' : 'Standard'} performance tracking
- Business metrics: ${intelligence.findings.keyInsights.length > 3 ? 'Custom dashboards recommended' : 'Standard metrics sufficient'}`;
  }

  private generateAcceptanceCriteria(intelligence: IntelligenceGatheringResult): string {
    return `Core acceptance criteria:
1. All ${intelligence.implementationConstraints.filter(c => c.severity === 'hard').length} hard constraints satisfied
2. Performance within acceptable parameters  
3. Security requirements met
4. ${intelligence.complexityAssessment.overallComplexity > 7 ? 'Comprehensive' : 'Standard'} documentation provided
5. Team knowledge transfer completed`;
  }

  private calculateCriticalRiskThreshold(intelligence: IntelligenceGatheringResult): number {
    const criticalRisks = intelligence.riskFactors.filter(r => r.riskScore >= 6).length;
    return Math.max(1, Math.floor(criticalRisks * 0.3)); // Allow 30% of critical risks to remain active
  }

  private calculateDowntimeTolerance(intelligence: IntelligenceGatheringResult): string {
    const riskLevel = intelligence.riskFactors.reduce((sum, r) => sum + r.riskScore, 0) / intelligence.riskFactors.length;
    return riskLevel > 6 ? '4 hours planned maintenance window' : '2 hours planned maintenance window';
  }

  private estimateTimelineBuffer(intelligence: IntelligenceGatheringResult): number {
    const complexity = intelligence.complexityAssessment.overallComplexity;
    const riskCount = intelligence.riskFactors.filter(r => r.riskScore >= 6).length;
    
    return Math.min(50, 20 + (complexity * 3) + (riskCount * 5)); // Cap at 50% buffer
  }
}