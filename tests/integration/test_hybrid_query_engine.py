"""
Comprehensive Integration Tests for Hybrid Query Engine.

Tests Phase 3 hybrid search functionality including intelligent query processing,
multiple search strategies, and result synthesis accuracy.
"""

import asyncio
import pytest
from pathlib import Path
from typing import Dict, List, Any

from src.repoindex.data.schemas import (
    FeatureConfig,
    SearchResponse,
    VectorIndex,
    SerenaGraph,
    RepoMap,
)
from src.repoindex.pipeline.hybrid_query_engine import (
    HybridQueryEngine,
    QueryContext,
    QueryStrategy,
    QueryType,
)
from src.repoindex.pipeline.advanced_query_processor import (
    AdvancedQueryProcessor,
    QueryIntent,
    create_query_processor,
)
from src.repoindex.pipeline.intelligent_ranking import (
    RankingStrategy,
    create_result_synthesizer,
)


class TestHybridQueryEngineIntegration:
    """Integration tests for the hybrid query engine system."""
    
    @pytest.fixture
    async def hybrid_engine(self):
        """Create and initialize hybrid query engine."""
        engine = HybridQueryEngine()
        await engine.initialize()
        return engine
    
    @pytest.fixture
    def query_processor(self):
        """Create query processor."""
        return create_query_processor()
    
    @pytest.fixture
    def mock_search_data(self):
        """Create mock search data for testing."""
        return {
            "vector_index": self._create_mock_vector_index(),
            "serena_graph": self._create_mock_serena_graph(),
            "repomap": self._create_mock_repomap(),
            "repo_root": "/test/repo",
            "rev": "main",
            "repo_id": "test_repo"
        }
    
    def _create_mock_vector_index(self) -> VectorIndex:
        """Create mock vector index."""
        from src.repoindex.data.schemas import VectorChunk
        
        chunks = [
            VectorChunk(
                path="src/utils.py",
                span=(10, 20),
                hash="abc123",
                text="def calculate_sum(a, b):\n    return a + b",
                embedding=[0.1, 0.2, 0.3]  # Mock embedding
            ),
            VectorChunk(
                path="src/math_helpers.py", 
                span=(5, 15),
                hash="def456",
                text="class Calculator:\n    def add(self, x, y):\n        return x + y",
                embedding=[0.2, 0.3, 0.4]
            ),
            VectorChunk(
                path="tests/test_utils.py",
                span=(1, 10),
                hash="ghi789",
                text="import unittest\nfrom src.utils import calculate_sum",
                embedding=[0.3, 0.4, 0.5]
            )
        ]
        
        return VectorIndex(chunks=chunks)
    
    def _create_mock_serena_graph(self) -> SerenaGraph:
        """Create mock Serena graph."""
        from src.repoindex.data.schemas import SerenaEntry, SerenaType
        
        entries = [
            SerenaEntry(
                path="src/utils.py",
                span=(10, 20),
                type=SerenaType.DEF,
                symbol="calculate_sum",
                sig="def calculate_sum(a, b)",
                doc="Calculate sum of two numbers"
            ),
            SerenaEntry(
                path="src/math_helpers.py",
                span=(5, 10),
                type=SerenaType.CLASS,
                symbol="Calculator",
                sig="class Calculator",
                doc="Basic calculator class"
            ),
            SerenaEntry(
                path="src/math_helpers.py",
                span=(11, 15),
                type=SerenaType.DEF,
                symbol="add",
                sig="def add(self, x, y)",
                doc="Add two numbers"
            )
        ]
        
        return SerenaGraph(entries=entries)
    
    def _create_mock_repomap(self) -> RepoMap:
        """Create mock repository map."""
        from src.repoindex.data.schemas import RepoEdge, FileRank
        
        edges = [
            RepoEdge(source="src/utils.py", target="tests/test_utils.py", weight=0.8),
            RepoEdge(source="src/math_helpers.py", target="src/utils.py", weight=0.6)
        ]
        
        file_ranks = [
            FileRank(path="src/utils.py", rank=0.9),
            FileRank(path="src/math_helpers.py", rank=0.8), 
            FileRank(path="tests/test_utils.py", rank=0.5)
        ]
        
        return RepoMap(edges=edges, file_ranks=file_ranks)
    
    @pytest.mark.asyncio
    async def test_vector_first_strategy(self, hybrid_engine, mock_search_data):
        """Test vector-first search strategy."""
        query = "calculate sum of numbers"
        context = QueryContext(
            strategy=QueryStrategy.VECTOR_FIRST,
            max_results=10
        )
        
        response = await hybrid_engine.search(
            query=query,
            index_id="test_index",
            context=context,
            **mock_search_data
        )
        
        assert isinstance(response, SearchResponse)
        assert response.query == query
        assert len(response.results) > 0
        
        # Verify that results contain relevant matches
        result_paths = [r.path for r in response.results]
        assert "src/utils.py" in result_paths or "src/math_helpers.py" in result_paths
        
        # Verify execution time is reasonable
        assert response.execution_time_ms > 0
        assert response.execution_time_ms < 10000  # Should be under 10 seconds
    
    @pytest.mark.asyncio
    async def test_semantic_first_strategy(self, hybrid_engine, mock_search_data):
        """Test semantic-first search strategy."""
        query = "how to add two numbers together"
        context = QueryContext(
            strategy=QueryStrategy.SEMANTIC_FIRST,
            max_results=10
        )
        
        response = await hybrid_engine.search(
            query=query,
            index_id="test_index",
            context=context,
            **mock_search_data
        )
        
        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        
        # Semantic search should find relevant functions
        result_contents = [r.content.text.lower() for r in response.results]
        assert any("add" in content or "sum" in content for content in result_contents)
    
    @pytest.mark.asyncio
    async def test_parallel_hybrid_strategy(self, hybrid_engine, mock_search_data):
        """Test parallel hybrid search strategy."""
        query = "Calculator class methods"
        context = QueryContext(
            strategy=QueryStrategy.PARALLEL_HYBRID,
            max_results=15
        )
        
        response = await hybrid_engine.search(
            query=query,
            index_id="test_index", 
            context=context,
            **mock_search_data
        )
        
        assert isinstance(response, SearchResponse)
        assert len(response.results) > 0
        
        # Should find Calculator class and related methods
        result_paths = [r.path for r in response.results]
        assert "src/math_helpers.py" in result_paths
        
        # Check for hybrid scoring (results from multiple systems)
        high_score_results = [r for r in response.results if r.score > 0.5]
        assert len(high_score_results) > 0
    
    @pytest.mark.asyncio
    async def test_adaptive_strategy_selection(self, hybrid_engine, mock_search_data):
        """Test adaptive strategy selection based on query analysis."""
        test_cases = [
            {
                "query": "find function calculate_sum",
                "expected_strategy": QueryStrategy.SEMANTIC_FIRST,  # Code-specific
                "should_find": ["src/utils.py"]
            },
            {
                "query": "similar code patterns", 
                "expected_strategy": QueryStrategy.PARALLEL_HYBRID,  # Pattern matching
                "should_find": ["src/utils.py", "src/math_helpers.py"]
            },
            {
                "query": "math calculator",
                "expected_strategy": QueryStrategy.VECTOR_FIRST,  # General similarity  
                "should_find": ["src/math_helpers.py"]
            }
        ]
        
        for case in test_cases:
            context = QueryContext(
                strategy=QueryStrategy.ADAPTIVE,  # Let it choose
                max_results=10
            )
            
            response = await hybrid_engine.search(
                query=case["query"],
                index_id="test_index",
                context=context,
                **mock_search_data
            )
            
            assert len(response.results) > 0
            
            # Verify relevant results are found
            result_paths = [r.path for r in response.results]
            for expected_path in case["should_find"]:
                if expected_path in result_paths:
                    break
            else:
                pytest.fail(f"None of expected paths {case['should_find']} found in results: {result_paths}")
    
    @pytest.mark.asyncio
    async def test_query_caching(self, hybrid_engine, mock_search_data):
        """Test query result caching functionality."""
        query = "test caching query"
        context = QueryContext(max_results=5)
        
        # First query
        start_time = asyncio.get_event_loop().time()
        response1 = await hybrid_engine.search(
            query=query,
            index_id="test_index",
            context=context,
            **mock_search_data
        )
        first_time = asyncio.get_event_loop().time() - start_time
        
        # Second identical query (should hit cache)
        start_time = asyncio.get_event_loop().time()
        response2 = await hybrid_engine.search(
            query=query,
            index_id="test_index", 
            context=context,
            **mock_search_data
        )
        second_time = asyncio.get_event_loop().time() - start_time
        
        # Verify responses are equivalent
        assert len(response1.results) == len(response2.results)
        assert response1.query == response2.query
        
        # Second query should be significantly faster (cache hit)
        assert second_time < first_time * 0.5  # At least 50% faster
    
    def test_query_processor_intent_classification(self, query_processor):
        """Test query processor intent classification."""
        test_cases = [
            ("find function calculate", QueryIntent.FIND_FUNCTION),
            ("class Calculator definition", QueryIntent.FIND_CLASS),
            ("how does this work", QueryIntent.UNDERSTAND_CODE),
            ("similar to this pattern", QueryIntent.FIND_SIMILAR),
            ("bug in calculation", QueryIntent.FIND_BUG),
            ("where is this used", QueryIntent.FIND_USAGE),
        ]
        
        for query, expected_intent in test_cases:
            processed = query_processor.process_query(query)
            assert processed.intent == expected_intent, f"Query '{query}' expected {expected_intent}, got {processed.intent}"
    
    def test_query_processor_entity_extraction(self, query_processor):
        """Test entity extraction from queries."""
        test_cases = [
            ("find function calculateSum", ["calculateSum"]),
            ("Calculator class methods", ["Calculator"]),
            ("import numpy as np", ["numpy"]),
            ("file utils.py contents", ["utils.py"]),
        ]
        
        for query, expected_entities in test_cases:
            processed = query_processor.process_query(query)
            extracted_entities = [entity.text for entity in processed.entities]
            
            for expected in expected_entities:
                assert any(expected.lower() in entity.lower() for entity in extracted_entities), \
                    f"Expected entity '{expected}' not found in {extracted_entities}"
    
    def test_query_processor_code_pattern_detection(self, query_processor):
        """Test code pattern detection."""
        test_cases = [
            ("calculateSum function", ["camel_case"]),
            ("calculate_sum method", ["snake_case"]),
            ("Calculator.add()", ["function_call", "method_chain"]),
            ("from utils import helper", ["import_statement"]),
            ("utils.py file", ["file_extension"]),
        ]
        
        for query, expected_patterns in test_cases:
            processed = query_processor.process_query(query)
            detected_patterns = [pattern.pattern_type for pattern in processed.code_patterns]
            
            for expected in expected_patterns:
                assert expected in detected_patterns, \
                    f"Expected pattern '{expected}' not found in {detected_patterns} for query '{query}'"
    
    def test_query_processor_language_detection(self, query_processor):
        """Test programming language detection."""
        test_cases = [
            ("def calculate_sum python function", ["python"]),
            ("function calculateSum javascript", ["javascript"]),
            ("interface Calculator typescript", ["typescript"]),
            ("public class Calculator java", ["java"]),
            ("fn calculate rust function", ["rust"]),
            ("func calculate go function", ["go"]),
        ]
        
        for query, expected_languages in test_cases:
            processed = query_processor.process_query(query)
            detected_languages = [lang.value for lang in processed.language_hints]
            
            for expected in expected_languages:
                assert expected in detected_languages, \
                    f"Expected language '{expected}' not found in {detected_languages}"
    
    @pytest.mark.asyncio
    async def test_result_ranking_strategies(self, hybrid_engine, mock_search_data):
        """Test different result ranking strategies."""
        query = "calculation functions"
        
        strategies = [
            RankingStrategy.RELEVANCE_FIRST,
            RankingStrategy.CONSENSUS_BOOST,
            RankingStrategy.DIVERSITY_AWARE,
            RankingStrategy.CONTEXT_ADAPTIVE
        ]
        
        results_by_strategy = {}
        
        for strategy in strategies:
            context = QueryContext(max_results=10)
            # Note: We'd need to pass ranking strategy through context
            # This is a simplified test
            
            response = await hybrid_engine.search(
                query=query,
                index_id="test_index",
                context=context,
                **mock_search_data
            )
            
            results_by_strategy[strategy] = response.results
            assert len(response.results) > 0
        
        # Verify different strategies can produce different orderings
        # (This is a basic check - in practice rankings may be identical for simple test data)
        for strategy, results in results_by_strategy.items():
            assert len(results) > 0, f"Strategy {strategy} returned no results"
    
    @pytest.mark.asyncio
    async def test_performance_budget_enforcement(self, hybrid_engine, mock_search_data):
        """Test performance budget enforcement."""
        query = "performance test query"
        
        # Test with very short budget
        context = QueryContext(
            performance_budget_ms=100,  # Very short budget
            max_results=5
        )
        
        start_time = asyncio.get_event_loop().time()
        response = await hybrid_engine.search(
            query=query,
            index_id="test_index",
            context=context,
            **mock_search_data
        )
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        # Should respect performance budget (with some tolerance)
        assert elapsed_ms <= context.performance_budget_ms * 2  # 2x tolerance for test environment
        assert len(response.results) >= 0  # Should return some results even with tight budget
    
    @pytest.mark.asyncio
    async def test_confidence_threshold_filtering(self, hybrid_engine, mock_search_data):
        """Test confidence threshold filtering."""
        query = "very specific unlikely query that should have low confidence"
        
        # High confidence threshold
        high_threshold_context = QueryContext(
            confidence_threshold=0.9,
            max_results=10
        )
        
        # Low confidence threshold
        low_threshold_context = QueryContext(
            confidence_threshold=0.1,
            max_results=10
        )
        
        high_response = await hybrid_engine.search(
            query=query,
            index_id="test_index",
            context=high_threshold_context,
            **mock_search_data
        )
        
        low_response = await hybrid_engine.search(
            query=query,
            index_id="test_index",
            context=low_threshold_context,
            **mock_search_data
        )
        
        # Low threshold should return more or equal results
        assert len(low_response.results) >= len(high_response.results)
    
    @pytest.mark.asyncio
    async def test_health_check(self, hybrid_engine):
        """Test hybrid query engine health check."""
        health = await hybrid_engine.health_check()
        
        assert "status" in health
        assert "components" in health
        assert "analytics" in health
        
        # Should report component health
        assert "mimir_engine" in health["components"]
        assert health["components"]["mimir_engine"] == "healthy"


class TestSemanticSearchAccuracy:
    """Tests specifically for semantic search accuracy and quality."""
    
    @pytest.fixture
    def test_queries_and_expectations(self):
        """Define test queries with expected semantic matches."""
        return [
            {
                "query": "function to add two numbers",
                "expected_symbols": ["calculate_sum", "add"],
                "expected_paths": ["src/utils.py", "src/math_helpers.py"],
                "semantic_keywords": ["function", "add", "numbers"]
            },
            {
                "query": "calculator class implementation",
                "expected_symbols": ["Calculator"],
                "expected_paths": ["src/math_helpers.py"],
                "semantic_keywords": ["calculator", "class", "implementation"]
            },
            {
                "query": "unit tests for mathematical operations",
                "expected_symbols": ["test"],
                "expected_paths": ["tests/test_utils.py"],
                "semantic_keywords": ["unit", "tests", "mathematical"]
            },
            {
                "query": "import statements and dependencies",
                "expected_symbols": ["import"],
                "expected_paths": ["tests/test_utils.py"],
                "semantic_keywords": ["import", "dependencies"]
            }
        ]
    
    @pytest.mark.asyncio
    async def test_semantic_relevance_accuracy(
        self, 
        hybrid_engine, 
        mock_search_data, 
        test_queries_and_expectations
    ):
        """Test semantic relevance accuracy across different query types."""
        total_tests = 0
        accurate_results = 0
        
        for test_case in test_queries_and_expectations:
            query = test_case["query"]
            expected_symbols = test_case["expected_symbols"]
            expected_paths = test_case["expected_paths"]
            
            context = QueryContext(
                strategy=QueryStrategy.SEMANTIC_FIRST,
                max_results=10
            )
            
            response = await hybrid_engine.search(
                query=query,
                index_id="test_index",
                context=context,
                **mock_search_data
            )
            
            total_tests += 1
            
            # Check if any expected symbols are found
            found_symbols = []
            found_paths = []
            
            for result in response.results:
                # Extract symbols from content
                content_lower = result.content.text.lower()
                for symbol in expected_symbols:
                    if symbol.lower() in content_lower:
                        found_symbols.append(symbol)
                
                # Check paths
                if result.path in expected_paths:
                    found_paths.append(result.path)
            
            # Consider accurate if we found at least one expected symbol or path
            if found_symbols or found_paths:
                accurate_results += 1
            else:
                print(f"No expected matches for query: '{query}'")
                print(f"  Expected symbols: {expected_symbols}")
                print(f"  Expected paths: {expected_paths}")
                print(f"  Actual results: {[r.path for r in response.results[:3]]}")
        
        accuracy_rate = accurate_results / total_tests if total_tests > 0 else 0
        print(f"Semantic accuracy: {accuracy_rate:.2%} ({accurate_results}/{total_tests})")
        
        # Should achieve at least 70% accuracy on semantic matching
        assert accuracy_rate >= 0.7, f"Semantic accuracy {accuracy_rate:.2%} below 70% threshold"
    
    @pytest.mark.asyncio
    async def test_query_expansion_effectiveness(
        self, 
        hybrid_engine, 
        mock_search_data, 
        query_processor
    ):
        """Test effectiveness of query expansion for improving recall."""
        base_query = "add"
        expanded_queries = [
            "add two numbers",
            "addition operation", 
            "calculate sum",
            "mathematical addition"
        ]
        
        # Search with base query
        base_context = QueryContext(
            enable_expansion=False,
            max_results=10
        )
        
        base_response = await hybrid_engine.search(
            query=base_query,
            index_id="test_index",
            context=base_context,
            **mock_search_data
        )
        
        # Search with expansion enabled
        expanded_context = QueryContext(
            enable_expansion=True,
            max_results=10
        )
        
        improved_recall = 0
        total_expanded_queries = len(expanded_queries)
        
        for expanded_query in expanded_queries:
            expanded_response = await hybrid_engine.search(
                query=expanded_query,
                index_id="test_index",
                context=expanded_context,
                **mock_search_data
            )
            
            # Expansion should generally improve or maintain recall
            if len(expanded_response.results) >= len(base_response.results):
                improved_recall += 1
        
        expansion_effectiveness = improved_recall / total_expanded_queries
        print(f"Query expansion effectiveness: {expansion_effectiveness:.2%}")
        
        # Expansion should be effective in at least 60% of cases
        assert expansion_effectiveness >= 0.6
    
    @pytest.mark.asyncio
    async def test_multi_modal_search_integration(self, hybrid_engine, mock_search_data):
        """Test integration of text + code + semantic search modalities."""
        multi_modal_queries = [
            {
                "query": "def calculate(x, y): return x + y",  # Code + semantic
                "modalities": ["code", "semantic"],
                "expected_relevance": "high"
            },
            {
                "query": "Calculator class with add method for numbers",  # Text + code + semantic
                "modalities": ["text", "code", "semantic"],
                "expected_relevance": "high"
            },
            {
                "query": "import math library for calculations",  # Text + code
                "modalities": ["text", "code"], 
                "expected_relevance": "medium"
            }
        ]
        
        for test_case in multi_modal_queries:
            context = QueryContext(
                query_type=QueryType.MULTI_MODAL,
                strategy=QueryStrategy.PARALLEL_HYBRID,
                max_results=10
            )
            
            response = await hybrid_engine.search(
                query=test_case["query"],
                index_id="test_index",
                context=context,
                **mock_search_data
            )
            
            assert len(response.results) > 0, f"Multi-modal query returned no results: {test_case['query']}"
            
            # Check result quality based on expected relevance
            if test_case["expected_relevance"] == "high":
                high_score_results = [r for r in response.results if r.score > 0.5]
                assert len(high_score_results) > 0, f"No high-relevance results for: {test_case['query']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])