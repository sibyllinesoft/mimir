"""
Abstract LLM adapter interface for multi-provider AI integration.

This module defines the standard interface for LLM providers, enabling
support for multiple AI services (Gemini, OpenAI, Claude, local models)
with consistent error handling and response formatting.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

from ..data.schemas import CodeSnippet
from ..util.errors import MimirError, ErrorSeverity, ErrorCategory, RecoveryStrategy


class LLMCapability(Enum):
    """Enumeration of LLM capabilities."""
    CODE_ANALYSIS = "code_analysis"
    QUESTION_ANSWERING = "question_answering"
    CODE_EXPLANATION = "code_explanation"
    IMPROVEMENT_SUGGESTIONS = "improvement_suggestions"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    FUNCTION_CALLING = "function_calling"
    STRUCTURED_OUTPUT = "structured_output"


@dataclass
class LLMModelInfo:
    """Information about an LLM model."""
    model_name: str
    provider: str
    max_tokens: int
    temperature: float
    capabilities: List[LLMCapability]
    supports_streaming: bool = False
    context_window: int = 4096
    cost_per_1k_tokens: Optional[float] = None


@dataclass
class LLMRequest:
    """Standardized LLM request structure."""
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class LLMResponse:
    """Standardized LLM response structure."""
    text: str
    model_name: str
    tokens_used: Optional[int] = None
    finish_reason: str = "completed"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LLMError(MimirError):
    """Specialized error for LLM operations."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        model: str,
        error_type: str = "unknown",
        retryable: bool = False,
        **kwargs
    ):
        super().__init__(
            message=f"LLM Error ({provider}/{model}): {message}",
            severity=ErrorSeverity.MEDIUM,
            category=ErrorCategory.EXTERNAL_SERVICE,
            recovery_strategy=RecoveryStrategy.RETRY if retryable else RecoveryStrategy.ESCALATE,
            **kwargs
        )
        self.provider = provider
        self.model = model
        self.error_type = error_type
        self.retryable = retryable


class LLMAdapter(ABC):
    """
    Abstract base class for LLM adapters.
    
    Defines the standard interface for AI providers, enabling seamless
    integration of multiple LLM services with consistent behavior and
    error handling patterns.
    """

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.1,
        **kwargs
    ):
        """Initialize LLM adapter with configuration."""
        self.model_name = model_name
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.config = kwargs

    @abstractmethod
    async def synthesize_answer(
        self,
        question: str,
        evidence_snippets: List[CodeSnippet],
        intents: List[Dict[str, Any]],
        repo_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Synthesize an intelligent answer using the LLM based on code evidence.

        Args:
            question: The user's question
            evidence_snippets: Relevant code snippets as evidence
            intents: Parsed intent information from the question
            repo_info: Optional repository context information

        Returns:
            Comprehensive answer synthesized by the LLM

        Raises:
            LLMError: If synthesis fails
        """
        pass

    @abstractmethod
    async def explain_code_snippet(
        self, 
        snippet: CodeSnippet, 
        context: Optional[str] = None
    ) -> str:
        """
        Provide detailed explanation of a specific code snippet.

        Args:
            snippet: Code snippet to explain
            context: Optional additional context

        Returns:
            Detailed explanation of the code snippet

        Raises:
            LLMError: If explanation fails
        """
        pass

    @abstractmethod
    async def suggest_improvements(
        self, 
        evidence_snippets: List[CodeSnippet], 
        focus_area: Optional[str] = None
    ) -> str:
        """
        Suggest improvements for the provided code evidence.

        Args:
            evidence_snippets: Code snippets to analyze for improvements
            focus_area: Optional specific area to focus on

        Returns:
            Improvement suggestions based on the code analysis

        Raises:
            LLMError: If suggestion generation fails
        """
        pass

    @abstractmethod
    def get_model_info(self) -> LLMModelInfo:
        """
        Get information about the configured LLM model.

        Returns:
            Model information including capabilities and limits
        """
        pass

    async def generate_response(self, request: LLMRequest) -> LLMResponse:
        """
        Generate a response for a structured LLM request.
        
        This is a high-level interface for custom prompts.

        Args:
            request: Structured LLM request

        Returns:
            Structured LLM response

        Raises:
            LLMError: If response generation fails
        """
        try:
            # Default implementation - subclasses should override for efficiency
            return await self._generate_raw_response(request)
        except Exception as e:
            raise LLMError(
                message=f"Failed to generate response: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                cause=e,
            )

    @abstractmethod
    async def _generate_raw_response(self, request: LLMRequest) -> LLMResponse:
        """
        Internal method to generate raw response.
        
        Subclasses must implement this method with provider-specific logic.

        Args:
            request: Structured LLM request

        Returns:
            Structured LLM response
        """
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Get the name of the LLM provider.

        Returns:
            Provider name (e.g., "google", "openai", "anthropic")
        """
        pass

    def validate_request(self, request: LLMRequest) -> None:
        """
        Validate LLM request parameters.

        Args:
            request: Request to validate

        Raises:
            LLMError: If request is invalid
        """
        if not request.prompt:
            raise LLMError(
                message="Prompt cannot be empty",
                provider=self.get_provider_name(),
                model=self.model_name,
                error_type="validation",
            )

        if request.max_tokens and request.max_tokens > self.max_tokens:
            raise LLMError(
                message=f"Requested tokens ({request.max_tokens}) exceed model limit ({self.max_tokens})",
                provider=self.get_provider_name(),
                model=self.model_name,
                error_type="validation",
            )

    def _is_retryable_error(self, error: Exception) -> bool:
        """
        Determine if an error is retryable.
        
        Override in subclasses for provider-specific logic.

        Args:
            error: The error to check

        Returns:
            True if the error is retryable
        """
        # Default implementation - consider network/timeout errors retryable
        error_str = str(error).lower()
        retryable_patterns = [
            "timeout", "connection", "network", "rate limit", 
            "temporary", "unavailable", "overloaded"
        ]
        return any(pattern in error_str for pattern in retryable_patterns)

    def get_token_estimate(self, text: str) -> int:
        """
        Estimate token count for text.
        
        Override in subclasses for more accurate estimates.

        Args:
            text: Text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 characters per token
        return len(text) // 4

    def _create_system_prompt(self, task_type: str) -> str:
        """
        Create system prompt for different task types.

        Args:
            task_type: Type of task (analysis, explanation, etc.)

        Returns:
            System prompt string
        """
        base_prompt = """You are an expert code analyst helping developers understand their codebase. 
Provide accurate, helpful, and well-structured responses based on the provided evidence."""

        task_prompts = {
            "analysis": f"{base_prompt}\n\nFocus on comprehensive code analysis with specific references to the evidence provided.",
            "explanation": f"{base_prompt}\n\nProvide clear, detailed explanations that help developers understand the code structure and purpose.",
            "improvement": f"{base_prompt}\n\nSuggest concrete, actionable improvements with clear reasoning and examples.",
        }

        return task_prompts.get(task_type, base_prompt)

    def _format_evidence_context(
        self, 
        evidence_snippets: List[CodeSnippet], 
        max_snippets: int = 10
    ) -> str:
        """
        Format evidence snippets into context string.

        Args:
            evidence_snippets: Code snippets to format
            max_snippets: Maximum number of snippets to include

        Returns:
            Formatted context string
        """
        if not evidence_snippets:
            return "No code evidence available."

        context_parts = ["## Code Evidence\n"]
        
        for i, snippet in enumerate(evidence_snippets[:max_snippets]):
            context_parts.append(f"### Evidence {i + 1}: `{snippet.path}`")
            context_parts.append(f"**Location**: Lines {snippet.line_start}-{snippet.line_end}\n")

            if snippet.pre:
                context_parts.append("**Context Before:**")
                context_parts.append("```")
                context_parts.append(snippet.pre)
                context_parts.append("```\n")

            context_parts.append("**Main Code:**")
            context_parts.append("```")
            context_parts.append(snippet.text)
            context_parts.append("```\n")

            if snippet.post:
                context_parts.append("**Context After:**")
                context_parts.append("```")
                context_parts.append(snippet.post)
                context_parts.append("```\n")

            context_parts.append("")  # Empty line between snippets

        return "\n".join(context_parts)


class MockLLMAdapter(LLMAdapter):
    """Mock LLM adapter for testing and fallback scenarios."""

    def __init__(self, **kwargs):
        super().__init__(
            model_name="mock-model",
            max_tokens=8192,
            temperature=0.1,
            **kwargs
        )

    async def synthesize_answer(
        self,
        question: str,
        evidence_snippets: List[CodeSnippet],
        intents: List[Dict[str, Any]],
        repo_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a mock answer for testing."""
        return f"Mock answer for question: {question} (based on {len(evidence_snippets)} code snippets)"

    async def explain_code_snippet(
        self, 
        snippet: CodeSnippet, 
        context: Optional[str] = None
    ) -> str:
        """Return a mock explanation for testing."""
        return f"Mock explanation for code snippet in {snippet.path} (lines {snippet.line_start}-{snippet.line_end})"

    async def suggest_improvements(
        self, 
        evidence_snippets: List[CodeSnippet], 
        focus_area: Optional[str] = None
    ) -> str:
        """Return mock improvement suggestions for testing."""
        focus = f" focusing on {focus_area}" if focus_area else ""
        return f"Mock improvement suggestions for {len(evidence_snippets)} code snippets{focus}"

    def get_model_info(self) -> LLMModelInfo:
        """Return mock model information."""
        return LLMModelInfo(
            model_name="mock-model",
            provider="mock",
            max_tokens=8192,
            temperature=0.1,
            capabilities=[
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.QUESTION_ANSWERING,
                LLMCapability.CODE_EXPLANATION,
                LLMCapability.IMPROVEMENT_SUGGESTIONS,
            ],
            supports_streaming=False,
            context_window=8192,
        )

    async def _generate_raw_response(self, request: LLMRequest) -> LLMResponse:
        """Generate mock response."""
        return LLMResponse(
            text=f"Mock response to: {request.prompt[:100]}...",
            model_name="mock-model",
            tokens_used=100,
            finish_reason="completed",
        )

    def get_provider_name(self) -> str:
        """Return mock provider name."""
        return "mock"