"""
Gemini AI integration for knowledge management and code analysis.

Provides intelligent question answering, code explanation, and reasoning
capabilities using Google's Gemini models.
"""

import asyncio
import os
from pathlib import Path
from typing import Any, Optional

from ..data.schemas import AskResponse, Citation, CodeSnippet
from ..util.errors import ExternalToolError, create_error_context
from .llm_adapter import LLMAdapter, LLMError, LLMModelInfo, LLMRequest, LLMResponse

try:
    import google.generativeai as genai
    from google.generativeai.types import GenerationConfig, HarmBlockThreshold, HarmCategory
except ImportError:
    genai = None


class GeminiError(Exception):
    """Gemini-specific errors."""

    pass


class GeminiAdapter(LLMAdapter):
    """
    Gemini AI adapter for intelligent code analysis and question answering.

    Provides structured reasoning over code snippets and symbols to generate
    comprehensive answers with supporting evidence and citations.
    """

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ):
        """Initialize Gemini adapter."""
        if not genai:
            raise GeminiError(
                "google-generativeai package not installed. Run: pip install google-generativeai"
            )

        # Try to get configuration from centralized config first, fall back to parameters/environment
        try:
            from ..config import get_ai_config
            from ..config_migration import migration_tracker
            
            config = get_ai_config()
            resolved_model_name = model_name or config.gemini_model
            resolved_max_tokens = max_tokens or config.gemini_max_tokens
            resolved_temperature = temperature or config.gemini_temperature
            
            # Configure API key from centralized config
            resolved_api_key = api_key or config.api_key
            if not resolved_api_key:
                raise GeminiError(
                    "Gemini API key not found in centralized configuration. "
                    "Please configure ai.google_api_key or ai.gemini_api_key."
                )
            
            # Mark this file as migrated
            migration_tracker.mark_migrated(__file__, ["ai"])
            
        except ImportError:
            # Centralized config not available, fall back to parameters and environment
            resolved_model_name = model_name or "gemini-1.5-flash"
            resolved_max_tokens = max_tokens or 8192
            resolved_temperature = temperature or 0.1
            
            # Configure API key from environment
            resolved_api_key = api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if not resolved_api_key:
                raise GeminiError(
                    "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
                )

        # Initialize parent class
        super().__init__(
            model_name=resolved_model_name,
            api_key=resolved_api_key,
            max_tokens=resolved_max_tokens,
            temperature=resolved_temperature,
        )

        genai.configure(api_key=resolved_api_key)

        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=resolved_model_name,
            generation_config=GenerationConfig(
                max_output_tokens=resolved_max_tokens,
                temperature=resolved_temperature,
                candidate_count=1,
            ),
            safety_settings={
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            },
        )

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "google"

    async def synthesize_answer(
        self,
        question: str,
        evidence_snippets: list[CodeSnippet],
        intents: list[dict[str, Any]],
        repo_info: dict[str, Any] | None = None,
    ) -> str:
        """
        Synthesize an intelligent answer using Gemini based on code evidence.

        Args:
            question: The user's question
            evidence_snippets: Relevant code snippets as evidence
            intents: Parsed intent information from the question
            repo_info: Optional repository context information

        Returns:
            Comprehensive answer synthesized by Gemini

        Raises:
            LLMError: If synthesis fails
        """
        if not evidence_snippets:
            return "I couldn't find enough information in the codebase to answer your question."

        try:
            # Build context from evidence
            context_parts = []
            context_parts.append("# Code Repository Analysis")
            context_parts.append(f"**Question**: {question}")
            context_parts.append("")

            if repo_info:
                context_parts.append(f"**Repository**: {repo_info.get('root', 'Unknown')}")
                context_parts.append(f"**Revision**: {repo_info.get('rev', 'Unknown')}")
                context_parts.append("")

            # Add intent information
            if intents:
                intent_desc = []
                for intent in intents:
                    intent_type = intent.get("intent_type", "general")
                    targets = intent.get("targets", [])
                    if targets:
                        intent_desc.append(f"{intent_type}: {', '.join(targets)}")

                if intent_desc:
                    context_parts.append(f"**Detected Intent**: {' | '.join(intent_desc)}")
                    context_parts.append("")

            # Use parent class method for formatting evidence
            evidence_context = self._format_evidence_context(evidence_snippets, max_snippets=10)
            context_parts.append(evidence_context)

            context = "\n".join(context_parts)

            # Create the analysis prompt
            prompt = self._create_analysis_prompt(question, context, intents)

            # Generate response using Gemini
            response = await asyncio.to_thread(self.model.generate_content, prompt)

            if response.candidates and response.candidates[0].content.parts:
                answer = response.candidates[0].content.parts[0].text
                return answer.strip()
            else:
                return "I was unable to generate a comprehensive answer. Please try rephrasing your question."

        except Exception as e:
            error_context = create_error_context(
                component="gemini",
                operation="synthesize_answer",
                parameters={
                    "question": question,
                    "evidence_count": len(evidence_snippets),
                    "model": self.model_name,
                },
            )

            raise LLMError(
                message=f"Failed to synthesize answer: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                context=error_context,
                cause=e,
                suggestions=[
                    "Check your Gemini API key is valid",
                    "Verify network connectivity",
                    "Try reducing the question complexity",
                    "Check if the model supports the input size",
                ],
            )

    def _create_analysis_prompt(
        self, question: str, context: str, intents: list[dict[str, Any]]
    ) -> str:
        """Create a structured prompt for Gemini analysis."""

        # Determine the analysis focus based on intents
        analysis_focus = "general code analysis"
        if intents:
            intent_types = [i.get("intent_type", "") for i in intents]
            if "definition" in intent_types:
                analysis_focus = "symbol definitions and their purpose"
            elif "usage" in intent_types:
                analysis_focus = "usage patterns and call relationships"
            elif "flow" in intent_types:
                analysis_focus = "execution flow and control flow"
            elif "dependency" in intent_types:
                analysis_focus = "dependencies and relationships"

        prompt = f"""You are an expert code analyst helping developers understand their codebase. Analyze the provided code evidence and answer the user's question comprehensively.

## Analysis Instructions:

1. **Focus**: Your analysis should emphasize {analysis_focus}
2. **Evidence-Based**: Base your answer strictly on the provided code evidence
3. **Structure**: Provide a clear, well-organized response
4. **Citations**: Reference specific files and line numbers when making claims
5. **Accuracy**: Only make statements you can support with the provided evidence

## Guidelines:

- **Be Specific**: Point to exact code locations when explaining concepts
- **Explain Context**: Help the user understand not just what the code does, but why
- **Identify Patterns**: Look for design patterns, architectural decisions, and coding conventions
- **Highlight Relationships**: Explain how different parts of the code interact
- **Be Helpful**: Provide actionable insights and suggestions when appropriate

## User's Question:
{question}

## Code Evidence and Context:
{context}

## Your Analysis:

Please provide a comprehensive answer that:
1. Directly addresses the user's question
2. Explains relevant code functionality with specific references
3. Identifies important patterns or relationships
4. Provides context for why the code is structured this way
5. Suggests areas for further exploration if relevant

Answer:"""

        return prompt

    async def explain_code_snippet(
        self, snippet: CodeSnippet, context: str | None = None
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
        try:
            prompt_parts = []
            prompt_parts.append("Analyze and explain this code snippet in detail:")
            prompt_parts.append("")
            prompt_parts.append(f"**File**: `{snippet.path}`")
            prompt_parts.append(f"**Lines**: {snippet.line_start}-{snippet.line_end}")
            prompt_parts.append("")

            if snippet.pre:
                prompt_parts.append("**Context Before:**")
                prompt_parts.append("```")
                prompt_parts.append(snippet.pre)
                prompt_parts.append("```")
                prompt_parts.append("")

            prompt_parts.append("**Main Code:**")
            prompt_parts.append("```")
            prompt_parts.append(snippet.text)
            prompt_parts.append("```")
            prompt_parts.append("")

            if snippet.post:
                prompt_parts.append("**Context After:**")
                prompt_parts.append("```")
                prompt_parts.append(snippet.post)
                prompt_parts.append("```")
                prompt_parts.append("")

            if context:
                prompt_parts.append(f"**Additional Context**: {context}")
                prompt_parts.append("")

            prompt_parts.append("Please explain:")
            prompt_parts.append("1. What this code does")
            prompt_parts.append("2. How it works")
            prompt_parts.append("3. Its role in the larger system")
            prompt_parts.append("4. Any notable patterns or design decisions")
            prompt_parts.append("5. Potential improvements or concerns")

            prompt = "\n".join(prompt_parts)

            response = await asyncio.to_thread(self.model.generate_content, prompt)

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return f"Unable to explain the code snippet in {snippet.path}."

        except Exception as e:
            raise LLMError(
                message=f"Error explaining code snippet: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                cause=e,
            )

    async def suggest_improvements(
        self, evidence_snippets: list[CodeSnippet], focus_area: str | None = None
    ) -> str:
        """
        Suggest improvements for the provided code evidence.

        Args:
            evidence_snippets: Code snippets to analyze for improvements
            focus_area: Optional specific area to focus on (performance, readability, etc.)

        Returns:
            Improvement suggestions based on the code analysis

        Raises:
            LLMError: If suggestion generation fails
        """
        if not evidence_snippets:
            return "No code evidence provided for improvement analysis."

        try:
            context_parts = []
            context_parts.append("# Code Improvement Analysis")
            context_parts.append("")

            if focus_area:
                context_parts.append(f"**Focus Area**: {focus_area}")
                context_parts.append("")

            # Use parent class method for formatting evidence
            evidence_context = self._format_evidence_context(evidence_snippets, max_snippets=5)
            context_parts.append(evidence_context)

            context = "\n".join(context_parts)

            focus_prompt = f" with special attention to {focus_area}" if focus_area else ""

            prompt = f"""Analyze the provided code{focus_prompt} and suggest specific improvements.

{context}

Please provide:
1. **Identified Issues**: Specific problems or areas for improvement
2. **Recommended Changes**: Concrete suggestions with examples
3. **Priority**: Order suggestions by impact and effort
4. **Best Practices**: Relevant coding standards and conventions
5. **Potential Risks**: Any concerns about the suggested changes

Focus on actionable, specific recommendations that will improve code quality, maintainability, performance, or readability."""

            response = await asyncio.to_thread(self.model.generate_content, prompt)

            if response.candidates and response.candidates[0].content.parts:
                return response.candidates[0].content.parts[0].text.strip()
            else:
                return "Unable to generate improvement suggestions for the provided code."

        except Exception as e:
            raise LLMError(
                message=f"Error generating improvement suggestions: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                cause=e,
            )

    def get_model_info(self) -> LLMModelInfo:
        """Get information about the configured Gemini model."""
        from .llm_adapter import LLMModelInfo, LLMCapability
        
        return LLMModelInfo(
            model_name=self.model_name,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            provider="google",
            capabilities=[
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.QUESTION_ANSWERING,
                LLMCapability.CODE_EXPLANATION,
                LLMCapability.IMPROVEMENT_SUGGESTIONS,
                LLMCapability.MULTI_HOP_REASONING,
            ],
            supports_streaming=False,
            context_window=self.max_tokens,
        )

    async def _generate_raw_response(self, request) -> "LLMResponse":
        """Generate raw response for the LLMAdapter interface."""
        from .llm_adapter import LLMResponse
        
        try:
            # Validate request
            self.validate_request(request)
            
            # Build prompt with system prompt if provided
            full_prompt = request.prompt
            if request.system_prompt:
                full_prompt = f"{request.system_prompt}\n\n{request.prompt}"
            
            # Configure generation parameters
            generation_config = GenerationConfig(
                max_output_tokens=request.max_tokens or self.max_tokens,
                temperature=request.temperature or self.temperature,
                candidate_count=1,
            )
            
            # Generate response
            response = await asyncio.to_thread(
                self.model.generate_content,
                full_prompt,
                generation_config=generation_config
            )
            
            if response.candidates and response.candidates[0].content.parts:
                text = response.candidates[0].content.parts[0].text.strip()
                tokens_used = self.get_token_estimate(text)  # Rough estimate
                
                return LLMResponse(
                    text=text,
                    model_name=self.model_name,
                    tokens_used=tokens_used,
                    finish_reason="completed",
                    metadata=request.metadata,
                )
            else:
                raise LLMError(
                    message="No response generated from Gemini",
                    provider=self.get_provider_name(),
                    model=self.model_name,
                    error_type="no_response",
                )
                
        except Exception as e:
            if isinstance(e, LLMError):
                raise
                
            raise LLMError(
                message=f"Failed to generate response: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                cause=e,
            )
