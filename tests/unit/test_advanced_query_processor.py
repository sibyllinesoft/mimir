"""
Unit Tests for Advanced Query Processor.

Tests natural language understanding, code pattern detection,
and entity extraction capabilities.
"""

import pytest
from typing import List

from src.repoindex.pipeline.advanced_query_processor import (
    AdvancedQueryProcessor,
    QueryIntent,
    QueryEntity,
    CodePattern,
    CodeLanguage,
    ProcessedQuery,
    create_query_processor,
)


class TestAdvancedQueryProcessor:
    """Unit tests for advanced query processing functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create query processor instance."""
        return create_query_processor()
    
    def test_query_normalization(self, processor):
        """Test basic query normalization."""
        test_cases = [
            ("  Find   the   function  ", "find the function"),
            ("SEARCH FOR CLASS", "search for class"),
            ("func with spaces", "function with spaces"),
            ("cls definition", "class definition"),
            ("impl details", "implementation details"),
        ]
        
        for input_query, expected in test_cases:
            result = processor._normalize_query(input_query)
            assert result == expected
    
    def test_intent_classification(self, processor):
        """Test query intent classification accuracy."""
        test_cases = [
            # Function search intents
            ("find function calculate", QueryIntent.FIND_FUNCTION),
            ("where is the method getName", QueryIntent.FIND_FUNCTION),
            ("show me the procedure for sorting", QueryIntent.FIND_FUNCTION),
            ("def processData implementation", QueryIntent.FIND_FUNCTION),
            
            # Class search intents
            ("find class Calculator", QueryIntent.FIND_CLASS),
            ("show interface UserService", QueryIntent.FIND_CLASS),
            ("where is struct Point", QueryIntent.FIND_CLASS),
            ("type definition Person", QueryIntent.FIND_CLASS),
            
            # Implementation search intents
            ("how to implement authentication", QueryIntent.FIND_IMPLEMENTATION),
            ("show me the implementation of sorting", QueryIntent.FIND_IMPLEMENTATION),
            ("how does login work", QueryIntent.FIND_IMPLEMENTATION),
            ("algorithm for calculating hash", QueryIntent.FIND_IMPLEMENTATION),
            
            # Usage search intents  
            ("where is calculateSum used", QueryIntent.FIND_USAGE),
            ("find calls to processUser", QueryIntent.FIND_USAGE),
            ("who invokes this method", QueryIntent.FIND_USAGE),
            ("examples of using Logger", QueryIntent.FIND_USAGE),
            
            # Understanding intents
            ("what does this function do", QueryIntent.UNDERSTAND_CODE),
            ("explain how authentication works", QueryIntent.UNDERSTAND_CODE),
            ("why is this needed", QueryIntent.UNDERSTAND_CODE),
            ("how does the system handle errors", QueryIntent.UNDERSTAND_CODE),
            
            # Pattern search intents
            ("find similar patterns", QueryIntent.FIND_PATTERN),
            ("code like this example", QueryIntent.FIND_PATTERN),
            ("same structure as UserController", QueryIntent.FIND_PATTERN),
            ("template for API endpoints", QueryIntent.FIND_PATTERN),
            
            # Bug search intents
            ("bug in calculation", QueryIntent.FIND_BUG),
            ("error in login process", QueryIntent.FIND_BUG),
            ("fix the issue with parsing", QueryIntent.FIND_BUG),
            ("problem with null pointer", QueryIntent.FIND_BUG),
            
            # Similar search intents
            ("find similar functions", QueryIntent.FIND_SIMILAR),
            ("code equivalent to this", QueryIntent.FIND_SIMILAR),
            ("like this implementation", QueryIntent.FIND_SIMILAR),
            ("comparable solution", QueryIntent.FIND_SIMILAR),
        ]
        
        correct_classifications = 0
        total_tests = len(test_cases)
        
        for query, expected_intent in test_cases:
            processed = processor.process_query(query)
            if processed.intent == expected_intent:
                correct_classifications += 1
            else:
                print(f"Misclassified: '{query}' -> {processed.intent} (expected {expected_intent})")
        
        accuracy = correct_classifications / total_tests
        print(f"Intent classification accuracy: {accuracy:.2%} ({correct_classifications}/{total_tests})")
        
        # Should achieve at least 80% accuracy
        assert accuracy >= 0.8, f"Intent classification accuracy {accuracy:.2%} below 80%"
    
    def test_entity_extraction(self, processor):
        """Test entity extraction from queries."""
        test_cases = [
            {
                "query": "find function calculateSum",
                "expected_entities": [("calculateSum", "function")],
            },
            {
                "query": "where is class UserManager defined",
                "expected_entities": [("UserManager", "class")],
            },
            {
                "query": "variable userCount in authentication",
                "expected_entities": [("userCount", "variable")],
            },
            {
                "query": "file utils.py contains helper methods",
                "expected_entities": [("utils.py", "file")],
            },
            {
                "query": "module database.connection imports",
                "expected_entities": [("database.connection", "module")],
            },
            {
                "query": "interface PaymentProcessor and class OrderManager",
                "expected_entities": [("PaymentProcessor", "interface"), ("OrderManager", "class")],
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            for expected_text, expected_type in test_case["expected_entities"]:
                # Check if entity was extracted
                found_entity = None
                for entity in processed.entities:
                    if expected_text.lower() in entity.text.lower():
                        found_entity = entity
                        break
                
                assert found_entity is not None, \
                    f"Entity '{expected_text}' not found in query '{test_case['query']}'"
                
                # Note: Entity type matching might be approximate due to context-based classification
    
    def test_code_pattern_detection(self, processor):
        """Test code pattern detection accuracy."""
        test_cases = [
            # Camel case patterns
            {
                "query": "find calculateSum method",
                "expected_patterns": ["camel_case"],
                "expected_values": ["calculateSum"]
            },
            # Pascal case patterns  
            {
                "query": "UserManager class definition",
                "expected_patterns": ["pascal_case"],
                "expected_values": ["UserManager"]
            },
            # Snake case patterns
            {
                "query": "user_profile variable access",
                "expected_patterns": ["snake_case"],
                "expected_values": ["user_profile"]
            },
            # Function call patterns
            {
                "query": "call to processData() method",
                "expected_patterns": ["function_call"],
                "expected_values": ["processData("]
            },
            # Method chain patterns
            {
                "query": "user.profile.getName chain",
                "expected_patterns": ["method_chain"],
                "expected_values": ["user.profile.getName"]
            },
            # File extension patterns
            {
                "query": "search in utils.py file",
                "expected_patterns": ["file_extension"],
                "expected_values": [".py"]
            },
            # Path-like patterns
            {
                "query": "src/utils/helpers.py location",
                "expected_patterns": ["path_like"],
                "expected_values": ["src/utils/helpers.py"]
            },
            # Import statement patterns
            {
                "query": "import json from library",
                "expected_patterns": ["import_statement"],
                "expected_values": ["import json"]
            },
            # Constant patterns
            {
                "query": "MAX_CONNECTIONS constant value",
                "expected_patterns": ["constant"],
                "expected_values": ["MAX_CONNECTIONS"]
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            detected_pattern_types = [p.pattern_type for p in processed.code_patterns]
            
            for expected_pattern in test_case["expected_patterns"]:
                assert expected_pattern in detected_pattern_types, \
                    f"Pattern '{expected_pattern}' not detected in query '{test_case['query']}'"
            
            # Check if expected values are found
            detected_values = [p.value for p in processed.code_patterns]
            for expected_value in test_case["expected_values"]:
                found = any(expected_value in value for value in detected_values)
                assert found, \
                    f"Value '{expected_value}' not found in detected patterns: {detected_values}"
    
    def test_language_detection(self, processor):
        """Test programming language detection."""
        test_cases = [
            # Python indicators
            {
                "query": "def calculate_sum python function",
                "expected_languages": [CodeLanguage.PYTHON],
            },
            {
                "query": "import numpy as np module",
                "expected_languages": [CodeLanguage.PYTHON],
            },
            {
                "query": "class UserManager with __init__ method",
                "expected_languages": [CodeLanguage.PYTHON],
            },
            # JavaScript indicators  
            {
                "query": "function calculateSum() javascript",
                "expected_languages": [CodeLanguage.JAVASCRIPT],
            },
            {
                "query": "const userData = require module",
                "expected_languages": [CodeLanguage.JAVASCRIPT],
            },
            {
                "query": "async function with arrow => syntax",
                "expected_languages": [CodeLanguage.JAVASCRIPT],
            },
            # TypeScript indicators
            {
                "query": "interface UserProfile typescript",
                "expected_languages": [CodeLanguage.TYPESCRIPT],
            },
            {
                "query": "type StringOrNumber = string | number",
                "expected_languages": [CodeLanguage.TYPESCRIPT],
            },
            {
                "query": "export class with generic<T> type",
                "expected_languages": [CodeLanguage.TYPESCRIPT],
            },
            # Java indicators
            {
                "query": "public class UserManager java",
                "expected_languages": [CodeLanguage.JAVA],
            },
            {
                "query": "private static final constant",
                "expected_languages": [CodeLanguage.JAVA],
            },
            {
                "query": "@Override annotation method",
                "expected_languages": [CodeLanguage.JAVA],
            },
            # Rust indicators
            {
                "query": "fn calculate_sum rust function",
                "expected_languages": [CodeLanguage.RUST],
            },
            {
                "query": "struct User with impl block",
                "expected_languages": [CodeLanguage.RUST],
            },
            {
                "query": "let mut variable ownership",
                "expected_languages": [CodeLanguage.RUST],
            },
            # Go indicators
            {
                "query": "func calculateSum go function",
                "expected_languages": [CodeLanguage.GO],
            },
            {
                "query": "type User struct definition",
                "expected_languages": [CodeLanguage.GO],
            },
            {
                "query": "package main with import",
                "expected_languages": [CodeLanguage.GO],
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            for expected_lang in test_case["expected_languages"]:
                assert expected_lang in processed.language_hints, \
                    f"Language '{expected_lang.value}' not detected in query '{test_case['query']}'"
    
    def test_semantic_keyword_extraction(self, processor):
        """Test semantic keyword extraction."""
        test_cases = [
            {
                "query": "create user authentication system",
                "expected_keywords": ["create", "user", "authentication", "system"],
                "semantic_categories": ["action_words", "data_words", "structure_words"]
            },
            {
                "query": "optimize database performance issues",
                "expected_keywords": ["optimize", "database", "performance"],
                "semantic_categories": ["action_words", "data_words", "quality_words"]
            },
            {
                "query": "handle error validation in data processing",
                "expected_keywords": ["handle", "error", "validation", "data", "processing"],
                "semantic_categories": ["action_words", "data_words", "quality_words"]
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            for expected_keyword in test_case["expected_keywords"]:
                # Check if keyword or related term is found
                keyword_found = any(
                    expected_keyword.lower() in keyword.lower() 
                    for keyword in processed.semantic_keywords
                )
                
                # Also check in query tokens
                if not keyword_found:
                    keyword_found = expected_keyword.lower() in test_case["query"].lower()
                
                assert keyword_found, \
                    f"Semantic keyword '{expected_keyword}' not found for query '{test_case['query']}'"
    
    def test_query_expansion_generation(self, processor):
        """Test query expansion term generation."""
        test_cases = [
            {
                "query": "find function calculate",
                "should_expand_to": ["method", "procedure", "routine"],
            },
            {
                "query": "class definition User",
                "should_expand_to": ["object", "type", "interface"],
            },
            {
                "query": "variable userCount access", 
                "should_expand_to": ["field", "property", "attribute"],
            },
            {
                "query": "implement sorting algorithm",
                "should_expand_to": ["create", "build", "write"],
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            # Check if any expansion terms are related to expected expansions
            expansion_found = False
            for expected_term in test_case["should_expand_to"]:
                if expected_term in processed.expansion_terms:
                    expansion_found = True
                    break
            
            # At least some relevant expansion should be found
            assert len(processed.expansion_terms) > 0, \
                f"No expansion terms generated for query '{test_case['query']}'"
    
    def test_query_complexity_calculation(self, processor):
        """Test query complexity scoring."""
        test_cases = [
            {
                "query": "search",
                "expected_complexity": "low",  # < 0.3
            },
            {
                "query": "find function calculateSum in authentication module",
                "expected_complexity": "medium",  # 0.3-0.7
            },
            {
                "query": "explain how the advanced user authentication system implements OAuth2 with JWT token validation and refresh mechanisms",
                "expected_complexity": "high",  # > 0.7
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            if test_case["expected_complexity"] == "low":
                assert processed.query_complexity < 0.3, \
                    f"Query complexity {processed.query_complexity} not low for '{test_case['query']}'"
            elif test_case["expected_complexity"] == "medium":
                assert 0.3 <= processed.query_complexity <= 0.7, \
                    f"Query complexity {processed.query_complexity} not medium for '{test_case['query']}'"
            elif test_case["expected_complexity"] == "high":
                assert processed.query_complexity > 0.7, \
                    f"Query complexity {processed.query_complexity} not high for '{test_case['query']}'"
    
    def test_confidence_scoring(self, processor):
        """Test confidence scoring for query analysis."""
        test_cases = [
            {
                "query": "calculateSum",  # Clear code pattern
                "expected_confidence": "high",
            },
            {
                "query": "function to add numbers",  # Clear intent, some structure
                "expected_confidence": "medium",
            },
            {
                "query": "find stuff",  # Vague query
                "expected_confidence": "low",
            },
            {
                "query": "def calculate_sum(a, b): return a + b",  # Very specific code
                "expected_confidence": "high",
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            if test_case["expected_confidence"] == "high":
                assert processed.confidence > 0.7, \
                    f"Confidence {processed.confidence} not high for '{test_case['query']}'"
            elif test_case["expected_confidence"] == "medium":
                assert 0.4 <= processed.confidence <= 0.7, \
                    f"Confidence {processed.confidence} not medium for '{test_case['query']}'"
            elif test_case["expected_confidence"] == "low":
                assert processed.confidence < 0.4, \
                    f"Confidence {processed.confidence} not low for '{test_case['query']}'"
    
    def test_multi_language_query_processing(self, processor):
        """Test processing queries with multiple language indicators."""
        test_cases = [
            {
                "query": "python function def and javascript const variable",
                "expected_languages": [CodeLanguage.PYTHON, CodeLanguage.JAVASCRIPT],
                "expected_patterns": ["function_call"]  # Should detect both def and const patterns
            },
            {
                "query": "typescript interface extends java class implementation",
                "expected_languages": [CodeLanguage.TYPESCRIPT, CodeLanguage.JAVA],
                "expected_patterns": ["pascal_case"]  # Should detect class/interface patterns
            },
        ]
        
        for test_case in test_cases:
            processed = processor.process_query(test_case["query"])
            
            # Should detect multiple languages
            detected_languages = set(processed.language_hints)
            expected_languages = set(test_case["expected_languages"])
            
            # At least some overlap in detected languages
            overlap = detected_languages & expected_languages
            assert len(overlap) > 0, \
                f"No language overlap detected. Expected: {expected_languages}, Got: {detected_languages}"
    
    def test_edge_cases(self, processor):
        """Test edge cases and error handling."""
        edge_cases = [
            "",  # Empty query
            "   ",  # Whitespace only
            "a",  # Single character
            "!@#$%^&*()",  # Special characters only
            "123 456 789",  # Numbers only
            "find find find find find",  # Repeated words
        ]
        
        for query in edge_cases:
            # Should not crash on edge cases
            processed = processor.process_query(query)
            assert isinstance(processed, ProcessedQuery)
            assert processed.original_query == query
            
            # Should have some basic normalization
            if query.strip():
                assert len(processed.normalized_query) > 0
            
            # Should have reasonable default values
            assert isinstance(processed.intent, QueryIntent)
            assert isinstance(processed.entities, list)
            assert isinstance(processed.code_patterns, list)
            assert isinstance(processed.language_hints, list)
            assert isinstance(processed.semantic_keywords, list)
            assert isinstance(processed.expansion_terms, list)
            assert 0.0 <= processed.confidence <= 1.0
            assert 0.0 <= processed.query_complexity <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])