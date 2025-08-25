"""
Comprehensive tests for LLM adapter interfaces and implementations.

Tests cover abstract base class behavior, mock adapter implementation,
request/response handling, error scenarios, and model capabilities.
"""

import pytest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from src.repoindex.pipeline.llm_adapter import (
    LLMAdapter,
    MockLLMAdapter,
    LLMCapability,
    LLMModelInfo,
    LLMRequest,
    LLMResponse,
    LLMError,
)
from src.repoindex.data.schemas import CodeSnippet
from src.repoindex.util.errors import ErrorSeverity, ErrorCategory, RecoveryStrategy


class ConcreteLLMAdapter(LLMAdapter):
    """Concrete implementation for testing abstract methods."""

    def __init__(self, **kwargs):
        super().__init__(
            model_name="test-model",
            max_tokens=4096,
            temperature=0.2,
            **kwargs
        )
        self.call_history = []

    async def synthesize_answer(
        self,
        question: str,
        evidence_snippets: List[CodeSnippet],
        intents: List[Dict[str, Any]],
        repo_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Mock implementation that tracks calls."""
        self.call_history.append(("synthesize_answer", question, len(evidence_snippets)))
        return f"Synthesized answer for: {question} with {len(evidence_snippets)} snippets"

    async def explain_code_snippet(
        self, 
        snippet: CodeSnippet, 
        context: Optional[str] = None
    ) -> str:
        """Mock implementation that tracks calls."""
        self.call_history.append(("explain_code_snippet", snippet.path, context))
        return f"Explanation for {snippet.path}: {snippet.text[:50]}..."

    async def suggest_improvements(
        self, 
        evidence_snippets: List[CodeSnippet], 
        focus_area: Optional[str] = None
    ) -> str:
        """Mock implementation that tracks calls."""
        self.call_history.append(("suggest_improvements", len(evidence_snippets), focus_area))
        return f"Improvements for {len(evidence_snippets)} snippets focusing on {focus_area}"

    def get_model_info(self) -> LLMModelInfo:
        """Return test model info."""
        return LLMModelInfo(
            model_name="test-model",
            provider="test",
            max_tokens=4096,
            temperature=0.2,
            capabilities=[
                LLMCapability.TEXT_GENERATION,
                LLMCapability.CODE_ANALYSIS,
            ],
            supports_streaming=False,
            context_window=4096,
        )

    async def _generate_raw_response(self, request: LLMRequest) -> LLMResponse:
        """Mock raw response generation."""
        self.call_history.append(("_generate_raw_response", request.prompt[:20]))
        return LLMResponse(
            text=f"Test response to: {request.prompt[:50]}...",
            model_name="test-model",
            tokens_used=100,
            finish_reason="completed",
        )

    def get_provider_name(self) -> str:
        """Return test provider name."""
        return "test"


class TestLLMCapability:
    """Test LLM capability enumeration."""

    def test_all_capabilities_defined(self):
        """Test that all expected capabilities are defined."""
        expected_capabilities = [
            "TEXT_GENERATION",
            "CODE_ANALYSIS", 
            "QUESTION_ANSWERING",
            "CODE_EXPLANATION",
            "IMPROVEMENT_SUGGESTIONS",
            "MULTI_HOP_REASONING",
            "FUNCTION_CALLING",
            "STRUCTURED_OUTPUT",
        ]
        
        actual_capabilities = [cap.name for cap in LLMCapability]
        
        for capability in expected_capabilities:
            assert capability in actual_capabilities

    def test_capability_values(self):
        """Test that capability values are as expected."""
        assert LLMCapability.TEXT_GENERATION.value == "text_generation"
        assert LLMCapability.CODE_ANALYSIS.value == "code_analysis"
        assert LLMCapability.QUESTION_ANSWERING.value == "question_answering"


class TestLLMModelInfo:
    """Test LLM model information dataclass."""

    def test_required_fields(self):
        """Test model info with required fields only."""
        model_info = LLMModelInfo(
            model_name="gpt-4",
            provider="openai",
            max_tokens=8192,
            temperature=0.1,
            capabilities=[LLMCapability.TEXT_GENERATION],
        )
        
        assert model_info.model_name == "gpt-4"
        assert model_info.provider == "openai"
        assert model_info.max_tokens == 8192
        assert model_info.temperature == 0.1
        assert model_info.capabilities == [LLMCapability.TEXT_GENERATION]
        
        # Test default values
        assert model_info.supports_streaming is False
        assert model_info.context_window == 4096
        assert model_info.cost_per_1k_tokens is None

    def test_all_fields(self):
        """Test model info with all fields specified."""
        capabilities = [
            LLMCapability.TEXT_GENERATION,
            LLMCapability.CODE_ANALYSIS,
            LLMCapability.FUNCTION_CALLING,
        ]
        
        model_info = LLMModelInfo(
            model_name="claude-3",
            provider="anthropic",
            max_tokens=4096,
            temperature=0.0,
            capabilities=capabilities,
            supports_streaming=True,
            context_window=200000,
            cost_per_1k_tokens=0.03,
        )
        
        assert model_info.model_name == "claude-3"
        assert model_info.provider == "anthropic"
        assert model_info.supports_streaming is True
        assert model_info.context_window == 200000
        assert model_info.cost_per_1k_tokens == 0.03
        assert len(model_info.capabilities) == 3


class TestLLMRequest:
    """Test LLM request dataclass."""

    def test_minimal_request(self):
        """Test request with minimal fields."""
        request = LLMRequest(prompt="Test prompt")
        
        assert request.prompt == "Test prompt"
        assert request.max_tokens is None
        assert request.temperature is None
        assert request.system_prompt is None
        assert request.metadata == {}

    def test_complete_request(self):
        """Test request with all fields."""
        metadata = {"task_type": "code_analysis", "priority": "high"}
        
        request = LLMRequest(
            prompt="Analyze this code",
            max_tokens=2048,
            temperature=0.3,
            system_prompt="You are a code expert",
            metadata=metadata,
        )
        
        assert request.prompt == "Analyze this code"
        assert request.max_tokens == 2048
        assert request.temperature == 0.3
        assert request.system_prompt == "You are a code expert"
        assert request.metadata == metadata

    def test_metadata_initialization(self):
        """Test that metadata is properly initialized."""
        # Test with None metadata
        request1 = LLMRequest(prompt="Test", metadata=None)
        assert request1.metadata == {}
        
        # Test without metadata parameter
        request2 = LLMRequest(prompt="Test")
        assert request2.metadata == {}
        
        # Verify they are separate dictionaries
        request1.metadata["key"] = "value1"
        request2.metadata["key"] = "value2"
        assert request1.metadata["key"] != request2.metadata["key"]


class TestLLMResponse:
    """Test LLM response dataclass."""

    def test_minimal_response(self):
        """Test response with minimal fields."""
        response = LLMResponse(
            text="Generated response",
            model_name="test-model",
        )
        
        assert response.text == "Generated response"
        assert response.model_name == "test-model"
        assert response.tokens_used is None
        assert response.finish_reason == "completed"
        assert response.metadata == {}

    def test_complete_response(self):
        """Test response with all fields."""
        metadata = {"execution_time": 1.5, "confidence": 0.9}
        
        response = LLMResponse(
            text="Complete response text",
            model_name="advanced-model",
            tokens_used=512,
            finish_reason="max_tokens",
            metadata=metadata,
        )
        
        assert response.text == "Complete response text"
        assert response.model_name == "advanced-model"
        assert response.tokens_used == 512
        assert response.finish_reason == "max_tokens"
        assert response.metadata == metadata


class TestLLMError:
    """Test LLM-specific error class."""

    def test_basic_llm_error(self):
        """Test basic LLM error creation."""
        error = LLMError(
            message="API rate limit exceeded",
            provider="openai",
            model="gpt-4",
        )
        
        assert "LLM Error (openai/gpt-4)" in str(error)
        assert "API rate limit exceeded" in str(error)
        assert error.provider == "openai"
        assert error.model == "gpt-4"
        assert error.error_type == "unknown"
        assert error.retryable is False
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.category == ErrorCategory.EXTERNAL_TOOL

    def test_retryable_llm_error(self):
        """Test retryable LLM error."""
        error = LLMError(
            message="Temporary service unavailable",
            provider="anthropic",
            model="claude-3",
            error_type="service_unavailable",
            retryable=True,
        )
        
        assert error.retryable is True
        assert error.recovery_strategy == RecoveryStrategy.RETRY
        assert error.error_type == "service_unavailable"

    def test_non_retryable_llm_error(self):
        """Test non-retryable LLM error."""
        error = LLMError(
            message="Invalid API key",
            provider="openai",
            model="gpt-3.5",
            error_type="authentication",
            retryable=False,
        )
        
        assert error.retryable is False
        assert error.recovery_strategy == RecoveryStrategy.ESCALATE
        assert error.error_type == "authentication"


class TestLLMAdapter:
    """Test abstract LLM adapter base class."""

    @pytest.fixture
    def adapter(self):
        """Create concrete adapter for testing."""
        return ConcreteLLMAdapter(
            api_key="test-key",
            custom_param="custom_value",
        )

    @pytest.fixture
    def sample_snippet(self):
        """Create sample code snippet for testing."""
        return CodeSnippet(
            path="test.py",
            line_start=10,
            line_end=15,
            text="def hello():\n    return 'Hello, World!'",
            pre="# Context before",
            post="# Context after",
        )

    def test_adapter_initialization(self, adapter):
        """Test adapter initialization with parameters."""
        assert adapter.model_name == "test-model"
        assert adapter.api_key == "test-key"
        assert adapter.max_tokens == 4096
        assert adapter.temperature == 0.2
        assert adapter.config["custom_param"] == "custom_value"

    def test_default_initialization(self):
        """Test adapter with default parameters."""
        adapter = ConcreteLLMAdapter(model_name="default-model")
        
        assert adapter.model_name == "default-model"
        assert adapter.api_key is None
        assert adapter.max_tokens == 8192  # Class default
        assert adapter.temperature == 0.1  # Class default

    @pytest.mark.asyncio
    async def test_synthesize_answer(self, adapter, sample_snippet):
        """Test synthesize_answer method."""
        question = "What does this function do?"
        evidence_snippets = [sample_snippet]
        intents = [{"type": "function_explanation"}]
        
        answer = await adapter.synthesize_answer(
            question, evidence_snippets, intents
        )
        
        assert "Synthesized answer for: What does this function do?" in answer
        assert "with 1 snippets" in answer
        assert ("synthesize_answer", question, 1) in adapter.call_history

    @pytest.mark.asyncio
    async def test_synthesize_answer_with_repo_info(self, adapter, sample_snippet):
        """Test synthesize_answer with repository information."""
        repo_info = {"name": "test-repo", "language": "Python"}
        
        answer = await adapter.synthesize_answer(
            "Test question?",
            [sample_snippet],
            [],
            repo_info=repo_info,
        )
        
        assert "Synthesized answer" in answer
        assert len(adapter.call_history) == 1

    @pytest.mark.asyncio
    async def test_explain_code_snippet(self, adapter, sample_snippet):
        """Test explain_code_snippet method."""
        explanation = await adapter.explain_code_snippet(
            sample_snippet,
            context="This is a greeting function",
        )
        
        assert f"Explanation for {sample_snippet.path}" in explanation
        assert ("explain_code_snippet", sample_snippet.path, "This is a greeting function") in adapter.call_history

    @pytest.mark.asyncio
    async def test_explain_code_snippet_no_context(self, adapter, sample_snippet):
        """Test explain_code_snippet without context."""
        explanation = await adapter.explain_code_snippet(sample_snippet)
        
        assert f"Explanation for {sample_snippet.path}" in explanation
        assert ("explain_code_snippet", sample_snippet.path, None) in adapter.call_history

    @pytest.mark.asyncio
    async def test_suggest_improvements(self, adapter, sample_snippet):
        """Test suggest_improvements method."""
        improvements = await adapter.suggest_improvements(
            [sample_snippet],
            focus_area="performance",
        )
        
        assert "Improvements for 1 snippets" in improvements
        assert "focusing on performance" in improvements
        assert ("suggest_improvements", 1, "performance") in adapter.call_history

    @pytest.mark.asyncio
    async def test_suggest_improvements_no_focus(self, adapter, sample_snippet):
        """Test suggest_improvements without focus area."""
        improvements = await adapter.suggest_improvements([sample_snippet])
        
        assert "focusing on None" in improvements
        assert ("suggest_improvements", 1, None) in adapter.call_history

    def test_get_model_info(self, adapter):
        """Test get_model_info method."""
        model_info = adapter.get_model_info()
        
        assert isinstance(model_info, LLMModelInfo)
        assert model_info.model_name == "test-model"
        assert model_info.provider == "test"
        assert model_info.max_tokens == 4096
        assert LLMCapability.TEXT_GENERATION in model_info.capabilities

    @pytest.mark.asyncio
    async def test_generate_response(self, adapter):
        """Test generate_response method."""
        request = LLMRequest(
            prompt="Generate a response",
            max_tokens=512,
            temperature=0.5,
        )
        
        response = await adapter.generate_response(request)
        
        assert isinstance(response, LLMResponse)
        assert response.model_name == "test-model"
        assert "Test response to: Generate a response..." in response.text
        assert response.tokens_used == 100
        assert ("_generate_raw_response", "Generate a response") in adapter.call_history

    @pytest.mark.asyncio
    async def test_generate_response_error_handling(self, adapter):
        """Test generate_response error handling."""
        # Mock _generate_raw_response to raise an exception
        original_method = adapter._generate_raw_response
        
        async def failing_method(request):
            raise ValueError("Mock failure")
        
        adapter._generate_raw_response = failing_method
        
        request = LLMRequest(prompt="Failing request")
        
        with pytest.raises(LLMError) as exc_info:
            await adapter.generate_response(request)
        
        error = exc_info.value
        assert "Failed to generate response" in str(error)
        assert error.provider == "test"
        assert error.model == "test-model"
        
        # Restore original method
        adapter._generate_raw_response = original_method

    def test_validate_request_success(self, adapter):
        """Test successful request validation."""
        request = LLMRequest(
            prompt="Valid prompt",
            max_tokens=2048,  # Within limit
        )
        
        # Should not raise any exception
        adapter.validate_request(request)

    def test_validate_request_empty_prompt(self, adapter):
        """Test validation failure for empty prompt."""
        request = LLMRequest(prompt="")
        
        with pytest.raises(LLMError) as exc_info:
            adapter.validate_request(request)
        
        error = exc_info.value
        assert "Prompt cannot be empty" in str(error)
        assert error.error_type == "validation"

    def test_validate_request_excessive_tokens(self, adapter):
        """Test validation failure for excessive token request."""
        request = LLMRequest(
            prompt="Valid prompt",
            max_tokens=10000,  # Exceeds limit of 4096
        )
        
        with pytest.raises(LLMError) as exc_info:
            adapter.validate_request(request)
        
        error = exc_info.value
        assert "exceed model limit" in str(error)
        assert "10000" in str(error)
        assert "4096" in str(error)
        assert error.error_type == "validation"

    def test_is_retryable_error_timeout(self, adapter):
        """Test retryable error detection for timeout."""
        timeout_error = Exception("Connection timeout occurred")
        assert adapter._is_retryable_error(timeout_error) is True

    def test_is_retryable_error_rate_limit(self, adapter):
        """Test retryable error detection for rate limit."""
        rate_limit_error = Exception("Rate limit exceeded")
        assert adapter._is_retryable_error(rate_limit_error) is True

    def test_is_retryable_error_non_retryable(self, adapter):
        """Test non-retryable error detection."""
        auth_error = Exception("Invalid API key")
        assert adapter._is_retryable_error(auth_error) is False

    def test_get_token_estimate(self, adapter):
        """Test token estimation."""
        short_text = "Hello"
        long_text = "This is a much longer text that should have more tokens"
        
        short_estimate = adapter.get_token_estimate(short_text)
        long_estimate = adapter.get_token_estimate(long_text)
        
        assert short_estimate == len(short_text) // 4
        assert long_estimate == len(long_text) // 4
        assert long_estimate > short_estimate

    def test_create_system_prompt_default(self, adapter):
        """Test system prompt creation with default task type."""
        prompt = adapter._create_system_prompt("unknown_task")
        
        assert "expert code analyst" in prompt.lower()
        assert "accurate, helpful" in prompt.lower()

    def test_create_system_prompt_analysis(self, adapter):
        """Test system prompt creation for analysis task."""
        prompt = adapter._create_system_prompt("analysis")
        
        assert "expert code analyst" in prompt.lower()
        assert "comprehensive code analysis" in prompt.lower()

    def test_create_system_prompt_explanation(self, adapter):
        """Test system prompt creation for explanation task."""
        prompt = adapter._create_system_prompt("explanation")
        
        assert "expert code analyst" in prompt.lower()
        assert "clear, detailed explanations" in prompt.lower()

    def test_create_system_prompt_improvement(self, adapter):
        """Test system prompt creation for improvement task."""
        prompt = adapter._create_system_prompt("improvement")
        
        assert "expert code analyst" in prompt.lower()
        assert "concrete, actionable improvements" in prompt.lower()

    def test_format_evidence_context_empty(self, adapter):
        """Test evidence formatting with empty list."""
        context = adapter._format_evidence_context([])
        
        assert "No code evidence available" in context

    def test_format_evidence_context_single_snippet(self, adapter, sample_snippet):
        """Test evidence formatting with single snippet."""
        context = adapter._format_evidence_context([sample_snippet])
        
        assert "## Code Evidence" in context
        assert f"### Evidence 1: `{sample_snippet.path}`" in context
        assert f"Lines {sample_snippet.line_start}-{sample_snippet.line_end}" in context
        assert "**Context Before:**" in context
        assert "**Main Code:**" in context
        assert "**Context After:**" in context
        assert sample_snippet.text in context

    def test_format_evidence_context_multiple_snippets(self, adapter):
        """Test evidence formatting with multiple snippets."""
        snippets = [
            CodeSnippet(path="file1.py", line_start=1, line_end=5, text="code1"),
            CodeSnippet(path="file2.py", line_start=10, line_end=15, text="code2"),
            CodeSnippet(path="file3.py", line_start=20, line_end=25, text="code3"),
        ]
        
        context = adapter._format_evidence_context(snippets)
        
        assert "### Evidence 1: `file1.py`" in context
        assert "### Evidence 2: `file2.py`" in context
        assert "### Evidence 3: `file3.py`" in context
        assert "code1" in context
        assert "code2" in context
        assert "code3" in context

    def test_format_evidence_context_max_snippets(self, adapter):
        """Test evidence formatting respects max_snippets limit."""
        snippets = [
            CodeSnippet(path=f"file{i}.py", line_start=i, line_end=i+5, text=f"code{i}")
            for i in range(15)  # Create 15 snippets
        ]
        
        context = adapter._format_evidence_context(snippets, max_snippets=5)
        
        # Should only see first 5 snippets
        assert "### Evidence 1:" in context
        assert "### Evidence 5:" in context
        assert "### Evidence 6:" not in context


class TestMockLLMAdapter:
    """Test the mock LLM adapter implementation."""

    @pytest.fixture
    def mock_adapter(self):
        """Create mock adapter for testing."""
        return MockLLMAdapter()

    @pytest.fixture
    def sample_snippet(self):
        """Create sample code snippet."""
        return CodeSnippet(
            path="mock_test.py",
            line_start=1,
            line_end=5,
            text="def mock_function():\n    pass",
        )

    def test_mock_adapter_initialization(self, mock_adapter):
        """Test mock adapter initialization."""
        assert mock_adapter.model_name == "mock-model"
        assert mock_adapter.max_tokens == 8192
        assert mock_adapter.temperature == 0.1

    @pytest.mark.asyncio
    async def test_mock_synthesize_answer(self, mock_adapter, sample_snippet):
        """Test mock synthesize_answer implementation."""
        question = "Mock question?"
        evidence = [sample_snippet]
        intents = [{"type": "test"}]
        
        answer = await mock_adapter.synthesize_answer(question, evidence, intents)
        
        assert "Mock answer for question: Mock question?" in answer
        assert "based on 1 code snippets" in answer

    @pytest.mark.asyncio
    async def test_mock_explain_code_snippet(self, mock_adapter, sample_snippet):
        """Test mock explain_code_snippet implementation."""
        explanation = await mock_adapter.explain_code_snippet(sample_snippet)
        
        assert "Mock explanation for code snippet in mock_test.py" in explanation
        assert "lines 1-5" in explanation

    @pytest.mark.asyncio
    async def test_mock_suggest_improvements(self, mock_adapter, sample_snippet):
        """Test mock suggest_improvements implementation."""
        improvements = await mock_adapter.suggest_improvements([sample_snippet])
        
        assert "Mock improvement suggestions for 1 code snippets" in improvements

    @pytest.mark.asyncio
    async def test_mock_suggest_improvements_with_focus(self, mock_adapter, sample_snippet):
        """Test mock suggest_improvements with focus area."""
        improvements = await mock_adapter.suggest_improvements(
            [sample_snippet], 
            focus_area="performance"
        )
        
        assert "focusing on performance" in improvements

    def test_mock_get_model_info(self, mock_adapter):
        """Test mock get_model_info implementation."""
        model_info = mock_adapter.get_model_info()
        
        assert model_info.model_name == "mock-model"
        assert model_info.provider == "mock"
        assert model_info.max_tokens == 8192
        assert model_info.temperature == 0.1
        assert model_info.supports_streaming is False
        assert model_info.context_window == 8192
        
        # Check capabilities
        expected_capabilities = [
            LLMCapability.CODE_ANALYSIS,
            LLMCapability.QUESTION_ANSWERING,
            LLMCapability.CODE_EXPLANATION,
            LLMCapability.IMPROVEMENT_SUGGESTIONS,
        ]
        
        for capability in expected_capabilities:
            assert capability in model_info.capabilities

    @pytest.mark.asyncio
    async def test_mock_generate_raw_response(self, mock_adapter):
        """Test mock _generate_raw_response implementation."""
        request = LLMRequest(prompt="Test prompt for mock response")
        
        response = await mock_adapter._generate_raw_response(request)
        
        assert isinstance(response, LLMResponse)
        assert response.model_name == "mock-model"
        assert "Mock response to: Test prompt for mock response" in response.text
        assert response.tokens_used == 100
        assert response.finish_reason == "completed"

    def test_mock_get_provider_name(self, mock_adapter):
        """Test mock get_provider_name implementation."""
        assert mock_adapter.get_provider_name() == "mock"

    @pytest.mark.asyncio
    async def test_mock_adapter_integration(self, mock_adapter, sample_snippet):
        """Test full integration of mock adapter methods."""
        # Test synthesize answer
        answer = await mock_adapter.synthesize_answer(
            "Integration test question?",
            [sample_snippet],
            [{"intent": "integration"}],
        )
        assert "Mock answer" in answer
        
        # Test explanation
        explanation = await mock_adapter.explain_code_snippet(
            sample_snippet,
            context="Integration test context",
        )
        assert "Mock explanation" in explanation
        
        # Test improvements
        improvements = await mock_adapter.suggest_improvements([sample_snippet])
        assert "Mock improvement" in improvements
        
        # Test generate response
        request = LLMRequest(prompt="Integration test prompt")
        response = await mock_adapter.generate_response(request)
        assert "Mock response" in response.text
        
        # All operations should succeed without errors