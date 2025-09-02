"""
Advanced Query Processing for Hybrid Search.

Provides natural language understanding, code-specific patterns,
and multi-modal query processing capabilities.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

from ..util.log import get_logger

logger = get_logger(__name__)


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript" 
    TYPESCRIPT = "typescript"
    JAVA = "java"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    UNKNOWN = "unknown"


class QueryIntent(Enum):
    """Classification of user query intent."""
    FIND_FUNCTION = "find_function"
    FIND_CLASS = "find_class"
    FIND_IMPLEMENTATION = "find_implementation"
    FIND_USAGE = "find_usage"
    UNDERSTAND_CODE = "understand_code"
    FIND_PATTERN = "find_pattern"
    FIND_BUG = "find_bug"
    FIND_SIMILAR = "find_similar"
    GENERAL_SEARCH = "general_search"


@dataclass
class QueryEntity:
    """Extracted entity from query."""
    text: str
    entity_type: str  # "function", "class", "variable", "file", "concept"
    confidence: float
    span: Tuple[int, int]  # Character positions in query


@dataclass
class CodePattern:
    """Detected code pattern in query."""
    pattern_type: str  # "camel_case", "snake_case", "function_call", "import_statement"
    value: str
    language_hint: Optional[CodeLanguage] = None
    confidence: float = 0.0


@dataclass
class ProcessedQuery:
    """Result of advanced query processing."""
    original_query: str
    normalized_query: str
    intent: QueryIntent
    entities: List[QueryEntity] = field(default_factory=list)
    code_patterns: List[CodePattern] = field(default_factory=list)
    language_hints: List[CodeLanguage] = field(default_factory=list)
    semantic_keywords: List[str] = field(default_factory=list)
    expansion_terms: List[str] = field(default_factory=list)
    confidence: float = 0.0
    query_complexity: float = 0.0  # 0=simple, 1=complex


class AdvancedQueryProcessor:
    """
    Advanced query processing for intelligent search routing.
    
    Features:
    - Natural language understanding
    - Code-specific pattern recognition
    - Multi-modal query analysis
    - Query expansion and refinement
    - Language detection and hints
    """
    
    def __init__(self):
        """Initialize advanced query processor."""
        self._initialize_patterns()
        self._initialize_vocabularies()
        
    def _initialize_patterns(self) -> None:
        """Initialize code pattern recognition."""
        self.code_patterns = {
            "camel_case": re.compile(r'\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b'),
            "pascal_case": re.compile(r'\b[A-Z][a-zA-Z0-9]*\b'),
            "snake_case": re.compile(r'\b[a-z][a-z0-9]*_[a-z0-9_]*\b'),
            "constant": re.compile(r'\b[A-Z][A-Z0-9]*(_[A-Z0-9]+)*\b'),
            "function_call": re.compile(r'\b\w+\s*\('),
            "method_chain": re.compile(r'\w+\.\w+(\.\w+)*'),
            "file_extension": re.compile(r'\.\w{1,4}$'),
            "path_like": re.compile(r'[\w\-\.]+/[\w\-\./]+'),
            "import_statement": re.compile(r'(import|from|require|include)\s+[\w\.\-/]+'),
        }
        
        # Language-specific patterns
        self.language_patterns = {
            CodeLanguage.PYTHON: [
                re.compile(r'\bdef\s+\w+'),
                re.compile(r'\bclass\s+\w+'),
                re.compile(r'\bimport\s+\w+'),
                re.compile(r'\bfrom\s+\w+\s+import'),
                re.compile(r'__\w+__'),
                re.compile(r'self\.\w+'),
            ],
            CodeLanguage.JAVASCRIPT: [
                re.compile(r'\bfunction\s+\w+'),
                re.compile(r'\bconst\s+\w+'),
                re.compile(r'\blet\s+\w+'),
                re.compile(r'\bvar\s+\w+'),
                re.compile(r'\basync\s+function'),
                re.compile(r'=>'),
            ],
            CodeLanguage.TYPESCRIPT: [
                re.compile(r'\binterface\s+\w+'),
                re.compile(r'\btype\s+\w+'),
                re.compile(r'\bexport\s+\w+'),
                re.compile(r':\s*\w+(\[\])?'),
                re.compile(r'<[\w\s,<>]+>'),
            ],
            CodeLanguage.JAVA: [
                re.compile(r'\bpublic\s+class'),
                re.compile(r'\bpublic\s+static'),
                re.compile(r'\bprivate\s+\w+'),
                re.compile(r'@\w+'),
                re.compile(r'\bSystem\.out\.println'),
            ],
            CodeLanguage.RUST: [
                re.compile(r'\bfn\s+\w+'),
                re.compile(r'\bstruct\s+\w+'),
                re.compile(r'\benum\s+\w+'),
                re.compile(r'\bimpl\s+\w+'),
                re.compile(r'\bmut\s+\w+'),
                re.compile(r'&\w+'),
            ],
            CodeLanguage.GO: [
                re.compile(r'\bfunc\s+\w+'),
                re.compile(r'\btype\s+\w+\s+struct'),
                re.compile(r'\bpackage\s+\w+'),
                re.compile(r'\bimport\s*\('),
                re.compile(r':='),
            ],
        }
    
    def _initialize_vocabularies(self) -> None:
        """Initialize vocabulary for semantic understanding."""
        self.intent_keywords = {
            QueryIntent.FIND_FUNCTION: [
                "function", "method", "procedure", "routine", "def", "fn",
                "implement", "implementation", "logic", "algorithm"
            ],
            QueryIntent.FIND_CLASS: [
                "class", "struct", "interface", "type", "object", "entity",
                "model", "component", "service"
            ],
            QueryIntent.FIND_IMPLEMENTATION: [
                "implementation", "implement", "how", "logic", "algorithm",
                "solution", "approach", "code", "write", "build"
            ],
            QueryIntent.FIND_USAGE: [
                "usage", "use", "used", "call", "invoke", "reference", 
                "example", "where", "who", "calls"
            ],
            QueryIntent.UNDERSTAND_CODE: [
                "understand", "explain", "what", "why", "how", "purpose",
                "meaning", "does", "work", "functionality"
            ],
            QueryIntent.FIND_PATTERN: [
                "pattern", "similar", "like", "template", "structure",
                "design", "architecture", "approach"
            ],
            QueryIntent.FIND_BUG: [
                "bug", "error", "issue", "problem", "fix", "broken",
                "exception", "failure", "crash", "wrong"
            ],
            QueryIntent.FIND_SIMILAR: [
                "similar", "like", "same", "equivalent", "related",
                "comparable", "analogous", "matching"
            ],
        }
        
        self.semantic_keywords = {
            "action_words": [
                "create", "build", "implement", "generate", "make",
                "process", "handle", "manage", "control", "execute"
            ],
            "data_words": [
                "data", "information", "content", "value", "parameter",
                "variable", "field", "property", "attribute", "member"
            ],
            "structure_words": [
                "structure", "architecture", "design", "pattern", "framework",
                "organization", "hierarchy", "relationship", "connection"
            ],
            "quality_words": [
                "performance", "optimization", "efficiency", "speed",
                "memory", "security", "validation", "testing", "quality"
            ],
        }
        
        self.code_entity_types = {
            "function_words": ["function", "method", "procedure", "routine"],
            "class_words": ["class", "struct", "interface", "type", "object"],
            "variable_words": ["variable", "field", "property", "attribute"],
            "file_words": ["file", "module", "package", "library"],
            "concept_words": ["concept", "idea", "approach", "pattern", "design"],
        }
    
    def process_query(self, query: str) -> ProcessedQuery:
        """
        Process query with advanced NLP and code analysis.
        
        Args:
            query: Raw user query string
            
        Returns:
            ProcessedQuery with analysis results
        """
        logger.debug(f"Processing query: '{query}'")
        
        # Basic preprocessing
        normalized_query = self._normalize_query(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Detect code patterns
        code_patterns = self._detect_code_patterns(query)
        
        # Detect language hints
        language_hints = self._detect_language_hints(query, code_patterns)
        
        # Classify intent
        intent = self._classify_intent(query, entities, code_patterns)
        
        # Extract semantic keywords
        semantic_keywords = self._extract_semantic_keywords(query)
        
        # Generate expansion terms
        expansion_terms = self._generate_expansion_terms(
            query, entities, code_patterns, semantic_keywords
        )
        
        # Calculate complexity and confidence
        complexity = self._calculate_query_complexity(
            query, entities, code_patterns, intent
        )
        confidence = self._calculate_confidence(
            entities, code_patterns, intent
        )
        
        processed_query = ProcessedQuery(
            original_query=query,
            normalized_query=normalized_query,
            intent=intent,
            entities=entities,
            code_patterns=code_patterns,
            language_hints=language_hints,
            semantic_keywords=semantic_keywords,
            expansion_terms=expansion_terms,
            confidence=confidence,
            query_complexity=complexity
        )
        
        logger.debug(f"Query processed - Intent: {intent.value}, Complexity: {complexity:.2f}")
        return processed_query
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for processing."""
        # Basic normalization
        normalized = query.strip().lower()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Handle common abbreviations
        abbreviations = {
            'func': 'function',
            'fn': 'function', 
            'cls': 'class',
            'var': 'variable',
            'impl': 'implementation',
            'def': 'definition',
        }
        
        for abbrev, full in abbreviations.items():
            normalized = re.sub(r'\b' + re.escape(abbrev) + r'\b', full, normalized)
        
        return normalized
    
    def _extract_entities(self, query: str) -> List[QueryEntity]:
        """Extract named entities from query."""
        entities = []
        query_lower = query.lower()
        
        # Extract code identifiers
        for pattern_name, pattern in self.code_patterns.items():
            matches = pattern.finditer(query)
            for match in matches:
                entity_type = self._pattern_to_entity_type(pattern_name)
                confidence = self._calculate_pattern_confidence(pattern_name, match.group())
                
                entities.append(QueryEntity(
                    text=match.group(),
                    entity_type=entity_type,
                    confidence=confidence,
                    span=match.span()
                ))
        
        # Extract contextual entities
        for entity_category, keywords in self.code_entity_types.items():
            entity_type = entity_category.replace('_words', '')
            for keyword in keywords:
                if keyword in query_lower:
                    # Look for identifiers near these keywords
                    pattern = rf'{re.escape(keyword)}\s+(\w+)'
                    matches = re.finditer(pattern, query, re.IGNORECASE)
                    for match in matches:
                        entities.append(QueryEntity(
                            text=match.group(1),
                            entity_type=entity_type,
                            confidence=0.8,
                            span=match.span(1)
                        ))
        
        return entities
    
    def _pattern_to_entity_type(self, pattern_name: str) -> str:
        """Map code pattern to entity type."""
        mapping = {
            "camel_case": "variable",
            "pascal_case": "class",
            "snake_case": "variable",
            "constant": "constant",
            "function_call": "function",
            "method_chain": "method",
            "file_extension": "file",
            "path_like": "file",
            "import_statement": "module",
        }
        return mapping.get(pattern_name, "identifier")
    
    def _calculate_pattern_confidence(self, pattern_name: str, text: str) -> float:
        """Calculate confidence score for pattern match."""
        base_confidence = {
            "camel_case": 0.8,
            "pascal_case": 0.9,
            "snake_case": 0.8,
            "constant": 0.9,
            "function_call": 0.95,
            "method_chain": 0.9,
            "file_extension": 0.95,
            "path_like": 0.85,
            "import_statement": 0.95,
        }
        
        confidence = base_confidence.get(pattern_name, 0.5)
        
        # Boost confidence for longer, more specific patterns
        if len(text) > 10:
            confidence *= 1.1
        elif len(text) < 3:
            confidence *= 0.8
        
        return min(confidence, 1.0)
    
    def _detect_code_patterns(self, query: str) -> List[CodePattern]:
        """Detect code-specific patterns in query."""
        patterns = []
        
        for pattern_name, pattern_regex in self.code_patterns.items():
            matches = pattern_regex.finditer(query)
            for match in matches:
                # Detect potential language hints from pattern
                language_hint = self._infer_language_from_pattern(pattern_name, match.group())
                confidence = self._calculate_pattern_confidence(pattern_name, match.group())
                
                patterns.append(CodePattern(
                    pattern_type=pattern_name,
                    value=match.group(),
                    language_hint=language_hint,
                    confidence=confidence
                ))
        
        return patterns
    
    def _infer_language_from_pattern(self, pattern_name: str, text: str) -> Optional[CodeLanguage]:
        """Infer programming language from code pattern."""
        text_lower = text.lower()
        
        # Strong language indicators
        if pattern_name == "import_statement":
            if "import" in text_lower and "from" in text_lower:
                return CodeLanguage.PYTHON
            elif "require(" in text_lower:
                return CodeLanguage.JAVASCRIPT
        
        # Snake case suggests Python
        if pattern_name == "snake_case" and "_" in text:
            return CodeLanguage.PYTHON
        
        # Certain function names suggest languages
        if pattern_name == "function_call":
            if text_lower.startswith("console.log"):
                return CodeLanguage.JAVASCRIPT
            elif text_lower.startswith("print("):
                return CodeLanguage.PYTHON
        
        return None
    
    def _detect_language_hints(
        self, 
        query: str, 
        code_patterns: List[CodePattern]
    ) -> List[CodeLanguage]:
        """Detect programming language hints from query."""
        language_scores = {lang: 0.0 for lang in CodeLanguage if lang != CodeLanguage.UNKNOWN}
        
        # Score based on language-specific patterns
        for lang, patterns in self.language_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(query)
                language_scores[lang] += len(matches) * 1.0
        
        # Score based on detected code patterns
        for code_pattern in code_patterns:
            if code_pattern.language_hint:
                language_scores[code_pattern.language_hint] += code_pattern.confidence
        
        # Score based on explicit language mentions
        query_lower = query.lower()
        language_names = {
            CodeLanguage.PYTHON: ["python", "py", "django", "flask", "pandas"],
            CodeLanguage.JAVASCRIPT: ["javascript", "js", "node", "react", "vue"],
            CodeLanguage.TYPESCRIPT: ["typescript", "ts", "angular", "nestjs"],
            CodeLanguage.JAVA: ["java", "spring", "android", "jvm"],
            CodeLanguage.RUST: ["rust", "cargo", "rustc"],
            CodeLanguage.GO: ["go", "golang", "goroutine"],
        }
        
        for lang, names in language_names.items():
            for name in names:
                if name in query_lower:
                    language_scores[lang] += 2.0
        
        # Return languages with significant scores
        significant_languages = [
            lang for lang, score in language_scores.items() 
            if score > 0.5
        ]
        
        # Sort by score
        significant_languages.sort(
            key=lambda lang: language_scores[lang], 
            reverse=True
        )
        
        return significant_languages[:3]  # Top 3 language hints
    
    def _classify_intent(
        self, 
        query: str, 
        entities: List[QueryEntity],
        code_patterns: List[CodePattern]
    ) -> QueryIntent:
        """Classify user intent from query analysis."""
        query_lower = query.lower()
        intent_scores = {intent: 0.0 for intent in QueryIntent}
        
        # Score based on intent keywords
        for intent, keywords in self.intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    intent_scores[intent] += 1.0
        
        # Boost based on entities and patterns
        for entity in entities:
            if entity.entity_type == "function":
                intent_scores[QueryIntent.FIND_FUNCTION] += 0.5
            elif entity.entity_type == "class":
                intent_scores[QueryIntent.FIND_CLASS] += 0.5
        
        for pattern in code_patterns:
            if pattern.pattern_type == "function_call":
                intent_scores[QueryIntent.FIND_FUNCTION] += 0.3
                intent_scores[QueryIntent.FIND_USAGE] += 0.2
        
        # Question words suggest understanding intent
        question_words = ["what", "how", "why", "when", "where", "which"]
        if any(word in query_lower for word in question_words):
            intent_scores[QueryIntent.UNDERSTAND_CODE] += 0.5
        
        # Find intent with highest score
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        
        return best_intent[0] if best_intent[1] > 0 else QueryIntent.GENERAL_SEARCH
    
    def _extract_semantic_keywords(self, query: str) -> List[str]:
        """Extract semantic keywords for query expansion."""
        query_lower = query.lower()
        keywords = set()
        
        # Extract from predefined semantic categories
        for category, words in self.semantic_keywords.items():
            for word in words:
                if word in query_lower:
                    keywords.add(word)
        
        # Extract meaningful nouns and verbs (simplified)
        # In practice, this would use proper NLP libraries
        tokens = query_lower.split()
        meaningful_words = [
            token for token in tokens 
            if len(token) > 3 and token not in [
                "the", "and", "for", "with", "that", "this", "from", "they", "have"
            ]
        ]
        
        keywords.update(meaningful_words)
        
        return list(keywords)
    
    def _generate_expansion_terms(
        self,
        query: str,
        entities: List[QueryEntity],
        code_patterns: List[CodePattern],
        semantic_keywords: List[str]
    ) -> List[str]:
        """Generate query expansion terms."""
        expansion_terms = set()
        
        # Add entity text
        for entity in entities:
            expansion_terms.add(entity.text)
            
            # Add related terms based on entity type
            if entity.entity_type == "function":
                expansion_terms.update(["method", "procedure", "routine"])
            elif entity.entity_type == "class":
                expansion_terms.update(["object", "type", "interface"])
        
        # Add pattern values
        for pattern in code_patterns:
            expansion_terms.add(pattern.value)
            
            # Add pattern-specific expansions
            if pattern.pattern_type == "snake_case":
                # Convert snake_case to space-separated words
                words = pattern.value.split('_')
                expansion_terms.update(words)
        
        # Add semantic synonyms
        synonyms = {
            "function": ["method", "procedure", "routine"],
            "class": ["object", "type", "interface", "struct"],
            "variable": ["field", "property", "attribute"],
            "implement": ["create", "build", "write"],
            "find": ["search", "locate", "discover"],
        }
        
        for keyword in semantic_keywords:
            if keyword in synonyms:
                expansion_terms.update(synonyms[keyword])
        
        # Remove original query words to avoid duplication
        query_words = set(query.lower().split())
        expansion_terms -= query_words
        
        # Filter short or common words
        filtered_terms = [
            term for term in expansion_terms 
            if len(term) > 2 and term not in ["the", "and", "for", "with"]
        ]
        
        return filtered_terms[:10]  # Limit expansion terms
    
    def _calculate_query_complexity(
        self,
        query: str,
        entities: List[QueryEntity],
        code_patterns: List[CodePattern],
        intent: QueryIntent
    ) -> float:
        """Calculate query complexity score (0=simple, 1=complex)."""
        complexity_factors = []
        
        # Query length factor
        word_count = len(query.split())
        length_factor = min(word_count / 20.0, 1.0)  # Normalize to max 20 words
        complexity_factors.append(length_factor * 0.2)
        
        # Entity complexity
        entity_factor = min(len(entities) / 5.0, 1.0)  # Normalize to max 5 entities
        complexity_factors.append(entity_factor * 0.3)
        
        # Code pattern complexity
        pattern_factor = min(len(code_patterns) / 3.0, 1.0)  # Normalize to max 3 patterns
        complexity_factors.append(pattern_factor * 0.2)
        
        # Intent complexity
        intent_complexity = {
            QueryIntent.GENERAL_SEARCH: 0.1,
            QueryIntent.FIND_FUNCTION: 0.3,
            QueryIntent.FIND_CLASS: 0.3,
            QueryIntent.FIND_USAGE: 0.4,
            QueryIntent.FIND_IMPLEMENTATION: 0.6,
            QueryIntent.UNDERSTAND_CODE: 0.8,
            QueryIntent.FIND_PATTERN: 0.7,
            QueryIntent.FIND_BUG: 0.6,
            QueryIntent.FIND_SIMILAR: 0.5,
        }
        complexity_factors.append(intent_complexity.get(intent, 0.3) * 0.3)
        
        return sum(complexity_factors)
    
    def _calculate_confidence(
        self,
        entities: List[QueryEntity],
        code_patterns: List[CodePattern],
        intent: QueryIntent
    ) -> float:
        """Calculate overall confidence in query analysis."""
        confidence_factors = []
        
        # Entity confidence
        if entities:
            avg_entity_confidence = sum(e.confidence for e in entities) / len(entities)
            confidence_factors.append(avg_entity_confidence * 0.4)
        
        # Pattern confidence
        if code_patterns:
            avg_pattern_confidence = sum(p.confidence for p in code_patterns) / len(code_patterns)
            confidence_factors.append(avg_pattern_confidence * 0.3)
        
        # Intent confidence (higher for specific intents)
        intent_confidence = {
            QueryIntent.GENERAL_SEARCH: 0.5,
            QueryIntent.FIND_FUNCTION: 0.8,
            QueryIntent.FIND_CLASS: 0.8,
            QueryIntent.FIND_USAGE: 0.7,
            QueryIntent.FIND_IMPLEMENTATION: 0.6,
            QueryIntent.UNDERSTAND_CODE: 0.7,
            QueryIntent.FIND_PATTERN: 0.6,
            QueryIntent.FIND_BUG: 0.7,
            QueryIntent.FIND_SIMILAR: 0.6,
        }
        confidence_factors.append(intent_confidence.get(intent, 0.5) * 0.3)
        
        # Baseline confidence
        if not confidence_factors:
            return 0.5
        
        return sum(confidence_factors)


# Factory function for easy instantiation
def create_query_processor() -> AdvancedQueryProcessor:
    """Create and return an AdvancedQueryProcessor instance."""
    return AdvancedQueryProcessor()