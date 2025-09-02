"""
Intelligent Result Synthesis and Ranking Algorithms.

Provides advanced algorithms for combining, scoring, and ranking search results
from multiple systems with context-aware relevance scoring.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..data.schemas import SearchResult, SearchScores
from ..util.log import get_logger
from .advanced_query_processor import ProcessedQuery, QueryIntent, CodeLanguage

logger = get_logger(__name__)


class RankingStrategy(Enum):
    """Different ranking strategies for result synthesis."""
    RELEVANCE_FIRST = "relevance_first"      # Pure relevance scoring
    CONSENSUS_BOOST = "consensus_boost"      # Boost results found by multiple systems
    DIVERSITY_AWARE = "diversity_aware"      # Promote result diversity
    CONTEXT_ADAPTIVE = "context_adaptive"   # Adapt to query context and intent


class ScoreComponent(Enum):
    """Individual components of relevance scoring."""
    VECTOR_SIMILARITY = "vector_similarity"
    SEMANTIC_MATCH = "semantic_match"
    SYMBOL_RELEVANCE = "symbol_relevance"
    PATH_RELEVANCE = "path_relevance"
    CONTENT_QUALITY = "content_quality"
    SYSTEM_CONSENSUS = "system_consensus"
    FRESHNESS = "freshness"
    AUTHORITY = "authority"


@dataclass
class RankingContext:
    """Context information for intelligent ranking."""
    processed_query: ProcessedQuery
    user_preferences: Dict[str, float] = field(default_factory=dict)
    domain_weights: Dict[str, float] = field(default_factory=dict)
    historical_feedback: Dict[str, float] = field(default_factory=dict)
    performance_budget_ms: int = 1000


@dataclass
class DetailedScore:
    """Detailed scoring breakdown for a search result."""
    total_score: float
    component_scores: Dict[ScoreComponent, float] = field(default_factory=dict)
    explanation: List[str] = field(default_factory=list)
    confidence: float = 0.0
    ranking_factors: Dict[str, float] = field(default_factory=dict)


@dataclass
class SynthesizedResult:
    """A search result with enhanced ranking information."""
    search_result: SearchResult
    detailed_score: DetailedScore
    source_systems: List[str]
    synthesis_metadata: Dict[str, any] = field(default_factory=dict)


class RelevanceScorer:
    """Advanced relevance scoring with multiple factors."""
    
    def __init__(self):
        """Initialize relevance scorer."""
        self.component_weights = self._initialize_component_weights()
        self.language_specific_weights = self._initialize_language_weights()
        
    def _initialize_component_weights(self) -> Dict[ScoreComponent, float]:
        """Initialize default component weights."""
        return {
            ScoreComponent.VECTOR_SIMILARITY: 0.25,
            ScoreComponent.SEMANTIC_MATCH: 0.20,
            ScoreComponent.SYMBOL_RELEVANCE: 0.15,
            ScoreComponent.PATH_RELEVANCE: 0.10,
            ScoreComponent.CONTENT_QUALITY: 0.10,
            ScoreComponent.SYSTEM_CONSENSUS: 0.10,
            ScoreComponent.FRESHNESS: 0.05,
            ScoreComponent.AUTHORITY: 0.05,
        }
    
    def _initialize_language_weights(self) -> Dict[CodeLanguage, Dict[ScoreComponent, float]]:
        """Initialize language-specific component weights."""
        return {
            CodeLanguage.PYTHON: {
                ScoreComponent.SYMBOL_RELEVANCE: 1.2,  # Python has clear naming conventions
                ScoreComponent.PATH_RELEVANCE: 1.1,    # Module structure is important
            },
            CodeLanguage.JAVASCRIPT: {
                ScoreComponent.VECTOR_SIMILARITY: 1.1,  # Dynamic nature benefits from vector search
                ScoreComponent.CONTENT_QUALITY: 0.9,    # More variability in code quality
            },
            CodeLanguage.TYPESCRIPT: {
                ScoreComponent.SEMANTIC_MATCH: 1.2,     # Rich type information
                ScoreComponent.SYMBOL_RELEVANCE: 1.1,   # Strong typing helps symbol matching
            },
            CodeLanguage.RUST: {
                ScoreComponent.SYMBOL_RELEVANCE: 1.3,   # Very explicit naming
                ScoreComponent.CONTENT_QUALITY: 1.2,    # Generally high code quality
            },
            CodeLanguage.GO: {
                ScoreComponent.SYMBOL_RELEVANCE: 1.1,   # Simple, clear naming
                ScoreComponent.PATH_RELEVANCE: 1.2,     # Package structure is critical
            },
        }
    
    def score_result(
        self, 
        result: SearchResult, 
        context: RankingContext,
        source_systems: List[str]
    ) -> DetailedScore:
        """Calculate detailed relevance score for a search result."""
        component_scores = {}
        explanation = []
        
        # Calculate individual component scores
        component_scores[ScoreComponent.VECTOR_SIMILARITY] = self._score_vector_similarity(
            result, context
        )
        
        component_scores[ScoreComponent.SEMANTIC_MATCH] = self._score_semantic_match(
            result, context
        )
        
        component_scores[ScoreComponent.SYMBOL_RELEVANCE] = self._score_symbol_relevance(
            result, context
        )
        
        component_scores[ScoreComponent.PATH_RELEVANCE] = self._score_path_relevance(
            result, context
        )
        
        component_scores[ScoreComponent.CONTENT_QUALITY] = self._score_content_quality(
            result, context
        )
        
        component_scores[ScoreComponent.SYSTEM_CONSENSUS] = self._score_system_consensus(
            source_systems, context
        )
        
        component_scores[ScoreComponent.FRESHNESS] = self._score_freshness(
            result, context
        )
        
        component_scores[ScoreComponent.AUTHORITY] = self._score_authority(
            result, context
        )
        
        # Apply language-specific adjustments
        primary_language = self._infer_primary_language(result, context)
        if primary_language and primary_language in self.language_specific_weights:
            language_adjustments = self.language_specific_weights[primary_language]
            for component, adjustment in language_adjustments.items():
                if component in component_scores:
                    component_scores[component] *= adjustment
                    explanation.append(f"Applied {primary_language.value} adjustment to {component.value}")
        
        # Apply intent-specific adjustments
        component_scores = self._apply_intent_adjustments(
            component_scores, context, explanation
        )
        
        # Calculate weighted total score
        total_score = 0.0
        active_weights = self._get_active_weights(context)
        
        for component, score in component_scores.items():
            weight = active_weights.get(component, 0.0)
            weighted_score = score * weight
            total_score += weighted_score
            
            if weighted_score > 0.1:  # Only explain significant contributions
                explanation.append(f"{component.value}: {score:.2f} (weight: {weight:.2f})")
        
        # Calculate confidence based on score distribution
        confidence = self._calculate_scoring_confidence(component_scores, source_systems)
        
        # Create ranking factors summary
        ranking_factors = {
            "total_score": total_score,
            "primary_language": primary_language.value if primary_language else "unknown",
            "source_count": len(source_systems),
            "top_component": max(component_scores.items(), key=lambda x: x[1])[0].value,
        }
        
        return DetailedScore(
            total_score=total_score,
            component_scores=component_scores,
            explanation=explanation,
            confidence=confidence,
            ranking_factors=ranking_factors
        )
    
    def _score_vector_similarity(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on vector similarity."""
        return result.scores.vector if result.scores.vector > 0 else 0.0
    
    def _score_semantic_match(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on semantic understanding."""
        # Use symbol score as a proxy for semantic relevance
        # In practice, this could use more sophisticated semantic analysis
        base_score = result.scores.symbol if result.scores.symbol > 0 else 0.0
        
        # Boost for semantic query intents
        if context.processed_query.intent in [
            QueryIntent.UNDERSTAND_CODE, 
            QueryIntent.FIND_IMPLEMENTATION,
            QueryIntent.FIND_PATTERN
        ]:
            base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def _score_symbol_relevance(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on symbol name matching."""
        symbol_score = result.scores.symbol if result.scores.symbol > 0 else 0.0
        
        # Boost for queries with detected entities that match result path/content
        query_entities = [entity.text.lower() for entity in context.processed_query.entities]
        result_text = f"{result.path} {result.content.text}".lower()
        
        entity_matches = sum(1 for entity in query_entities if entity in result_text)
        if entity_matches > 0:
            entity_boost = min(entity_matches * 0.2, 0.6)  # Cap boost at 0.6
            symbol_score += entity_boost
        
        return min(symbol_score, 1.0)
    
    def _score_path_relevance(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on file path relevance."""
        path_lower = result.path.lower()
        query_lower = context.processed_query.normalized_query
        
        # Basic path scoring
        path_score = 0.0
        
        # Exact path component matches
        query_tokens = set(query_lower.split())
        path_tokens = set(result.path.replace('/', ' ').replace('_', ' ').replace('.', ' ').lower().split())
        
        token_overlap = len(query_tokens & path_tokens)
        if token_overlap > 0:
            path_score += token_overlap / len(query_tokens) * 0.8
        
        # File extension relevance
        if any(pattern.pattern_type == "file_extension" for pattern in context.processed_query.code_patterns):
            # Query mentions specific file types
            for pattern in context.processed_query.code_patterns:
                if pattern.pattern_type == "file_extension" and pattern.value.lower() in path_lower:
                    path_score += 0.3
        
        # Directory structure relevance
        for keyword in context.processed_query.semantic_keywords:
            if keyword in path_lower:
                path_score += 0.1
        
        return min(path_score, 1.0)
    
    def _score_content_quality(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on content quality indicators."""
        # Simplified content quality scoring
        # In practice, this would analyze code complexity, documentation, etc.
        
        content_score = 0.5  # Baseline score
        
        # Code length heuristic (moderate length is often good)
        content_length = len(result.content.text)
        if 50 <= content_length <= 500:
            content_score += 0.2
        elif content_length > 1000:
            content_score -= 0.1
        
        # Presence of documentation/comments
        if any(indicator in result.content.text.lower() for indicator in ["#", "//", "/*", "\"\"\"", "'''"]):
            content_score += 0.2
        
        # Code structure indicators
        if any(indicator in result.content.text for indicator in ["def ", "class ", "function ", "const ", "let "]):
            content_score += 0.1
        
        return min(content_score, 1.0)
    
    def _score_system_consensus(self, source_systems: List[str], context: RankingContext) -> float:
        """Score based on agreement between different systems."""
        if len(source_systems) <= 1:
            return 0.0
        
        # Higher score for more systems finding the same result
        consensus_score = min(len(source_systems) / 3.0, 1.0)  # Normalize to max 3 systems
        
        # Specific system combinations that are particularly valuable
        valuable_combinations = {
            ("lens", "mimir"): 1.0,  # Vector + semantic is ideal
            ("lens", "mimir", "hybrid"): 1.2,  # All systems agree
        }
        
        system_key = tuple(sorted(source_systems))
        if system_key in valuable_combinations:
            consensus_score *= valuable_combinations[system_key]
        
        return min(consensus_score, 1.0)
    
    def _score_freshness(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on content freshness (simplified)."""
        # Placeholder implementation - would use actual timestamps in practice
        return 0.8  # Assume most content is reasonably fresh
    
    def _score_authority(self, result: SearchResult, context: RankingContext) -> float:
        """Score based on code authority/importance indicators."""
        # Simplified authority scoring
        authority_score = 0.5
        
        # Main/core files often have higher authority
        path_lower = result.path.lower()
        if any(indicator in path_lower for indicator in ["main", "core", "base", "index", "__init__"]):
            authority_score += 0.3
        
        # Test files have lower authority for non-testing queries
        if "test" in path_lower and context.processed_query.intent != QueryIntent.FIND_BUG:
            authority_score -= 0.2
        
        # Public/exposed APIs have higher authority
        if any(indicator in result.content.text.lower() for indicator in ["public", "export", "api"]):
            authority_score += 0.2
        
        return min(max(authority_score, 0.0), 1.0)
    
    def _infer_primary_language(self, result: SearchResult, context: RankingContext) -> Optional[CodeLanguage]:
        """Infer the primary programming language of the result."""
        # Check file extension first
        path_lower = result.path.lower()
        extension_mapping = {
            ".py": CodeLanguage.PYTHON,
            ".js": CodeLanguage.JAVASCRIPT,
            ".ts": CodeLanguage.TYPESCRIPT,
            ".tsx": CodeLanguage.TYPESCRIPT,
            ".jsx": CodeLanguage.JAVASCRIPT,
            ".java": CodeLanguage.JAVA,
            ".rs": CodeLanguage.RUST,
            ".go": CodeLanguage.GO,
            ".cpp": CodeLanguage.CPP,
            ".c": CodeLanguage.CPP,
            ".h": CodeLanguage.CPP,
        }
        
        for ext, lang in extension_mapping.items():
            if path_lower.endswith(ext):
                return lang
        
        # Fall back to query language hints
        if context.processed_query.language_hints:
            return context.processed_query.language_hints[0]
        
        return None
    
    def _apply_intent_adjustments(
        self,
        component_scores: Dict[ScoreComponent, float],
        context: RankingContext,
        explanation: List[str]
    ) -> Dict[ScoreComponent, float]:
        """Apply query intent-specific score adjustments."""
        intent = context.processed_query.intent
        adjustments = {}
        
        if intent == QueryIntent.FIND_FUNCTION:
            adjustments[ScoreComponent.SYMBOL_RELEVANCE] = 1.3
            adjustments[ScoreComponent.SEMANTIC_MATCH] = 1.1
            explanation.append("Boosted symbol and semantic scores for function search")
        
        elif intent == QueryIntent.FIND_CLASS:
            adjustments[ScoreComponent.SYMBOL_RELEVANCE] = 1.4
            adjustments[ScoreComponent.PATH_RELEVANCE] = 1.2
            explanation.append("Boosted symbol and path scores for class search")
        
        elif intent == QueryIntent.FIND_IMPLEMENTATION:
            adjustments[ScoreComponent.SEMANTIC_MATCH] = 1.4
            adjustments[ScoreComponent.CONTENT_QUALITY] = 1.2
            explanation.append("Boosted semantic and content quality for implementation search")
        
        elif intent == QueryIntent.UNDERSTAND_CODE:
            adjustments[ScoreComponent.SEMANTIC_MATCH] = 1.3
            adjustments[ScoreComponent.AUTHORITY] = 1.2
            adjustments[ScoreComponent.CONTENT_QUALITY] = 1.1
            explanation.append("Boosted semantic, authority, and quality for understanding")
        
        elif intent == QueryIntent.FIND_SIMILAR:
            adjustments[ScoreComponent.VECTOR_SIMILARITY] = 1.3
            adjustments[ScoreComponent.PATH_RELEVANCE] = 1.1
            explanation.append("Boosted vector similarity for finding similar code")
        
        # Apply adjustments
        for component, adjustment in adjustments.items():
            if component in component_scores:
                component_scores[component] *= adjustment
        
        return component_scores
    
    def _get_active_weights(self, context: RankingContext) -> Dict[ScoreComponent, float]:
        """Get component weights adjusted for current context."""
        weights = dict(self.component_weights)
        
        # Apply user preferences if available
        if context.user_preferences:
            for component_str, preference in context.user_preferences.items():
                try:
                    component = ScoreComponent(component_str)
                    weights[component] *= (1.0 + preference)  # preference is -0.5 to +0.5
                except ValueError:
                    pass  # Invalid component name
        
        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        
        return weights
    
    def _calculate_scoring_confidence(
        self,
        component_scores: Dict[ScoreComponent, float],
        source_systems: List[str]
    ) -> float:
        """Calculate confidence in the scoring."""
        factors = []
        
        # More source systems = higher confidence
        source_factor = min(len(source_systems) / 2.0, 1.0)
        factors.append(source_factor * 0.4)
        
        # Higher and more consistent scores = higher confidence
        scores = list(component_scores.values())
        if scores:
            avg_score = sum(scores) / len(scores)
            score_variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency_factor = 1.0 - min(score_variance, 1.0)
            factors.append(avg_score * 0.3)
            factors.append(consistency_factor * 0.3)
        
        return sum(factors) if factors else 0.5


class ResultSynthesizer:
    """Synthesizes and ranks results from multiple search systems."""
    
    def __init__(self):
        """Initialize result synthesizer."""
        self.relevance_scorer = RelevanceScorer()
        self.deduplication_threshold = 0.9  # Similarity threshold for deduplication
        
    def synthesize_results(
        self,
        results_by_system: Dict[str, List[SearchResult]],
        context: RankingContext,
        strategy: RankingStrategy = RankingStrategy.CONTEXT_ADAPTIVE
    ) -> List[SynthesizedResult]:
        """
        Synthesize and rank results from multiple systems.
        
        Args:
            results_by_system: Results grouped by source system
            context: Ranking context with query analysis
            strategy: Ranking strategy to use
            
        Returns:
            List of synthesized and ranked results
        """
        logger.debug(f"Synthesizing results from {len(results_by_system)} systems using {strategy.value} strategy")
        
        # Step 1: Deduplicate results across systems
        unified_results = self._deduplicate_results(results_by_system)
        
        # Step 2: Score each result using intelligent ranking
        synthesized_results = []
        for result_data in unified_results:
            detailed_score = self.relevance_scorer.score_result(
                result_data["result"],
                context,
                result_data["sources"]
            )
            
            synthesized_result = SynthesizedResult(
                search_result=result_data["result"],
                detailed_score=detailed_score,
                source_systems=result_data["sources"],
                synthesis_metadata={
                    "deduplication_key": result_data["dedup_key"],
                    "original_scores": result_data["original_scores"]
                }
            )
            synthesized_results.append(synthesized_result)
        
        # Step 3: Apply ranking strategy
        ranked_results = self._apply_ranking_strategy(synthesized_results, context, strategy)
        
        # Step 4: Apply diversity and quality filters
        final_results = self._apply_post_processing(ranked_results, context)
        
        logger.debug(f"Synthesized {len(final_results)} results from {len(unified_results)} candidates")
        return final_results
    
    def _deduplicate_results(
        self, 
        results_by_system: Dict[str, List[SearchResult]]
    ) -> List[Dict[str, any]]:
        """Deduplicate results across systems, preserving source information."""
        unified_results = {}  # dedup_key -> result_data
        
        for system_name, results in results_by_system.items():
            for result in results:
                # Create deduplication key based on path and span
                dedup_key = f"{result.path}:{result.span[0]}-{result.span[1]}"
                
                if dedup_key in unified_results:
                    # Merge with existing result
                    existing = unified_results[dedup_key]
                    existing["sources"].append(system_name)
                    existing["original_scores"][system_name] = result.score
                    
                    # Use the result with higher score as the canonical version
                    if result.score > existing["result"].score:
                        existing["result"] = result
                else:
                    # New result
                    unified_results[dedup_key] = {
                        "result": result,
                        "sources": [system_name],
                        "dedup_key": dedup_key,
                        "original_scores": {system_name: result.score}
                    }
        
        return list(unified_results.values())
    
    def _apply_ranking_strategy(
        self,
        results: List[SynthesizedResult],
        context: RankingContext,
        strategy: RankingStrategy
    ) -> List[SynthesizedResult]:
        """Apply the specified ranking strategy."""
        if strategy == RankingStrategy.RELEVANCE_FIRST:
            return self._rank_by_relevance(results)
        
        elif strategy == RankingStrategy.CONSENSUS_BOOST:
            return self._rank_by_consensus(results)
        
        elif strategy == RankingStrategy.DIVERSITY_AWARE:
            return self._rank_by_diversity(results, context)
        
        elif strategy == RankingStrategy.CONTEXT_ADAPTIVE:
            return self._rank_adaptively(results, context)
        
        else:
            # Default to relevance
            return self._rank_by_relevance(results)
    
    def _rank_by_relevance(self, results: List[SynthesizedResult]) -> List[SynthesizedResult]:
        """Simple relevance-based ranking."""
        return sorted(results, key=lambda r: r.detailed_score.total_score, reverse=True)
    
    def _rank_by_consensus(self, results: List[SynthesizedResult]) -> List[SynthesizedResult]:
        """Ranking that boosts multi-system consensus."""
        def consensus_score(result: SynthesizedResult) -> float:
            base_score = result.detailed_score.total_score
            consensus_boost = len(result.source_systems) * 0.1
            return base_score + consensus_boost
        
        return sorted(results, key=consensus_score, reverse=True)
    
    def _rank_by_diversity(
        self, 
        results: List[SynthesizedResult], 
        context: RankingContext
    ) -> List[SynthesizedResult]:
        """Ranking that promotes result diversity."""
        # Group results by file path to ensure diversity
        path_groups = {}
        for result in results:
            path = result.search_result.path
            if path not in path_groups:
                path_groups[path] = []
            path_groups[path].append(result)
        
        # Select best result from each path, then fill remaining slots
        diverse_results = []
        remaining_results = []
        
        for path, path_results in path_groups.items():
            # Sort by score within each path
            path_results.sort(key=lambda r: r.detailed_score.total_score, reverse=True)
            diverse_results.append(path_results[0])  # Best from this path
            remaining_results.extend(path_results[1:])  # Rest for later consideration
        
        # Sort diverse results by score
        diverse_results.sort(key=lambda r: r.detailed_score.total_score, reverse=True)
        
        # Add remaining results to fill out the list
        remaining_results.sort(key=lambda r: r.detailed_score.total_score, reverse=True)
        diverse_results.extend(remaining_results)
        
        return diverse_results
    
    def _rank_adaptively(
        self, 
        results: List[SynthesizedResult], 
        context: RankingContext
    ) -> List[SynthesizedResult]:
        """Context-adaptive ranking based on query characteristics."""
        intent = context.processed_query.intent
        
        # Choose strategy based on query intent
        if intent in [QueryIntent.FIND_FUNCTION, QueryIntent.FIND_CLASS]:
            # For specific searches, prioritize relevance
            return self._rank_by_relevance(results)
        
        elif intent == QueryIntent.UNDERSTAND_CODE:
            # For understanding, boost consensus and authority
            return self._rank_by_consensus(results)
        
        elif intent in [QueryIntent.FIND_PATTERN, QueryIntent.FIND_SIMILAR]:
            # For pattern searches, promote diversity
            return self._rank_by_diversity(results, context)
        
        else:
            # For general searches, use consensus
            return self._rank_by_consensus(results)
    
    def _apply_post_processing(
        self, 
        results: List[SynthesizedResult], 
        context: RankingContext
    ) -> List[SynthesizedResult]:
        """Apply final post-processing filters and adjustments."""
        processed_results = []
        
        # Apply confidence threshold
        min_confidence = context.user_preferences.get("min_confidence", 0.3)
        
        for result in results:
            if result.detailed_score.confidence >= min_confidence:
                processed_results.append(result)
            else:
                logger.debug(f"Filtered out result with low confidence: {result.detailed_score.confidence}")
        
        # Apply score threshold
        min_score = context.user_preferences.get("min_score", 0.1)
        processed_results = [
            r for r in processed_results 
            if r.detailed_score.total_score >= min_score
        ]
        
        return processed_results


# Factory functions for easy instantiation
def create_relevance_scorer() -> RelevanceScorer:
    """Create and return a RelevanceScorer instance."""
    return RelevanceScorer()


def create_result_synthesizer() -> ResultSynthesizer:
    """Create and return a ResultSynthesizer instance."""
    return ResultSynthesizer()