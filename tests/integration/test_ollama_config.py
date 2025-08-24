"""
Configuration and utilities for Ollama integration testing.

Provides shared configuration, test data, and helper functions for
testing MCP server integration with Ollama models.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional

# Ollama server configuration
OLLAMA_CONFIG = {
    "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    "embedding_model": os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text"),
    "chat_model": os.getenv("OLLAMA_CHAT_MODEL", "llama3.1"),
    "timeout": float(os.getenv("OLLAMA_TIMEOUT", "30.0")),
    "max_retries": int(os.getenv("OLLAMA_MAX_RETRIES", "3")),
}

# Test repository configurations
TEST_REPO_CONFIGS = {
    "python_ml": {
        "languages": ["py"],
        "excludes": ["__pycache__/", ".git/", "*.pyc", ".pytest_cache/"],
        "context_lines": 5,
        "max_files_to_embed": 100,
        "embedding_provider": "ollama",
        "ollama_config": OLLAMA_CONFIG,
    },
    "typescript_web": {
        "languages": ["ts", "tsx", "js", "jsx"],
        "excludes": ["node_modules/", "dist/", ".git/", "*.d.ts"],
        "context_lines": 3,
        "max_files_to_embed": 75,
        "embedding_provider": "ollama",
        "ollama_config": OLLAMA_CONFIG,
    },
    "mixed_project": {
        "languages": ["py", "ts", "js", "tsx", "rs"],
        "excludes": ["target/", "node_modules/", "__pycache__/", ".git/"],
        "context_lines": 4,
        "max_files_to_embed": 150,
        "embedding_provider": "ollama",
        "ollama_config": OLLAMA_CONFIG,
    }
}

# Expected embedding dimensions for different models
EMBEDDING_DIMENSIONS = {
    "nomic-embed-text": 768,
    "all-minilm": 384,
    "bge-large": 1024,
    "gte-large": 1024,
}

# Test queries for different domains
TEST_QUERIES = {
    "ml_python": [
        "machine learning data preprocessing",
        "model training and evaluation",
        "linear regression sklearn",
        "data loading pandas csv",
        "feature normalization StandardScaler"
    ],
    "web_typescript": [
        "authentication controller login",
        "user service database",
        "express route handler",
        "JWT token generation",
        "REST API endpoint"
    ],
    "general": [
        "error handling try catch",
        "class constructor method",
        "import export modules",
        "function parameter types",
        "async await promise"
    ]
}

# Expected search result validation criteria
SEARCH_VALIDATION_CRITERIA = {
    "min_results": 1,
    "max_results": 50,
    "required_fields": ["path", "content", "score", "type"],
    "score_range": (0.0, 1.0),
    "content_min_length": 10,
}

# Question answering validation criteria
QA_VALIDATION_CRITERIA = {
    "min_answer_length": 30,
    "max_answer_length": 2000,
    "required_context_terms": ["class", "method", "function"],  # At least one should appear
    "forbidden_terms": ["I don't know", "cannot determine", "unclear"],
}

def get_ollama_test_config() -> Dict[str, Any]:
    """Get complete Ollama test configuration."""
    return {
        "server": OLLAMA_CONFIG,
        "repo_configs": TEST_REPO_CONFIGS,
        "embedding_dimensions": EMBEDDING_DIMENSIONS,
        "test_queries": TEST_QUERIES,
        "validation": {
            "search": SEARCH_VALIDATION_CRITERIA,
            "qa": QA_VALIDATION_CRITERIA,
        }
    }

def get_embedding_model_config(model_name: Optional[str] = None) -> Dict[str, Any]:
    """Get configuration for specific embedding model."""
    model_name = model_name or OLLAMA_CONFIG["embedding_model"]
    
    return {
        "model_name": model_name,
        "dimensions": EMBEDDING_DIMENSIONS.get(model_name, 768),
        "base_url": OLLAMA_CONFIG["base_url"],
        "timeout": OLLAMA_CONFIG["timeout"],
    }

def get_test_queries_for_domain(domain: str) -> list[str]:
    """Get test queries for a specific domain."""
    return TEST_QUERIES.get(domain, TEST_QUERIES["general"])

def validate_search_results(results: list[Dict[str, Any]]) -> tuple[bool, list[str]]:
    """Validate search results meet expected criteria."""
    errors = []
    
    if len(results) < SEARCH_VALIDATION_CRITERIA["min_results"]:
        errors.append(f"Too few results: {len(results)} < {SEARCH_VALIDATION_CRITERIA['min_results']}")
    
    if len(results) > SEARCH_VALIDATION_CRITERIA["max_results"]:
        errors.append(f"Too many results: {len(results)} > {SEARCH_VALIDATION_CRITERIA['max_results']}")
    
    for i, result in enumerate(results):
        # Check required fields
        for field in SEARCH_VALIDATION_CRITERIA["required_fields"]:
            if field not in result:
                errors.append(f"Result {i} missing field: {field}")
        
        # Check score range
        if "score" in result:
            score = result["score"]
            min_score, max_score = SEARCH_VALIDATION_CRITERIA["score_range"]
            if not (min_score <= score <= max_score):
                errors.append(f"Result {i} score out of range: {score} not in [{min_score}, {max_score}]")
        
        # Check content length
        if "content" in result:
            content = result["content"]
            if len(content) < SEARCH_VALIDATION_CRITERIA["content_min_length"]:
                errors.append(f"Result {i} content too short: {len(content)} < {SEARCH_VALIDATION_CRITERIA['content_min_length']}")
    
    return len(errors) == 0, errors

def validate_qa_answer(answer: str, question: str) -> tuple[bool, list[str]]:
    """Validate question answering response."""
    errors = []
    
    # Check answer length
    if len(answer) < QA_VALIDATION_CRITERIA["min_answer_length"]:
        errors.append(f"Answer too short: {len(answer)} < {QA_VALIDATION_CRITERIA['min_answer_length']}")
    
    if len(answer) > QA_VALIDATION_CRITERIA["max_answer_length"]:
        errors.append(f"Answer too long: {len(answer)} > {QA_VALIDATION_CRITERIA['max_answer_length']}")
    
    # Check for required context terms
    answer_lower = answer.lower()
    has_context_term = any(term in answer_lower for term in QA_VALIDATION_CRITERIA["required_context_terms"])
    if not has_context_term:
        errors.append(f"Answer lacks programming context terms: {QA_VALIDATION_CRITERIA['required_context_terms']}")
    
    # Check for forbidden terms that indicate failure
    has_forbidden_term = any(term in answer_lower for term in QA_VALIDATION_CRITERIA["forbidden_terms"])
    if has_forbidden_term:
        errors.append(f"Answer contains failure indicators: {[term for term in QA_VALIDATION_CRITERIA['forbidden_terms'] if term in answer_lower]}")
    
    return len(errors) == 0, errors

def get_model_requirements() -> Dict[str, str]:
    """Get minimum model requirements for testing."""
    return {
        "embedding_model": OLLAMA_CONFIG["embedding_model"],
        "chat_model": OLLAMA_CONFIG["chat_model"],
        "min_embedding_dims": str(EMBEDDING_DIMENSIONS.get(OLLAMA_CONFIG["embedding_model"], 768)),
    }

# Test environment checks
def check_test_environment() -> Dict[str, Any]:
    """Check if test environment is properly configured."""
    checks = {
        "ollama_url_set": bool(os.getenv("OLLAMA_BASE_URL")),
        "embedding_model_set": bool(os.getenv("OLLAMA_EMBEDDING_MODEL")), 
        "chat_model_set": bool(os.getenv("OLLAMA_CHAT_MODEL")),
        "timeout_configured": bool(os.getenv("OLLAMA_TIMEOUT")),
        "config": get_ollama_test_config(),
    }
    
    return checks