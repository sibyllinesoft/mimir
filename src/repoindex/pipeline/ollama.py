"""
Ollama LLM adapter for Mimir.

Provides integration with Ollama for local LLM inference, supporting
the standard LLMAdapter interface for seamless integration with
the Mimir pipeline.
"""

import asyncio
import aiohttp
import json
import logging
from typing import Any, Dict, List, Optional, AsyncIterator
from dataclasses import dataclass

from .llm_adapter import (
    LLMAdapter, 
    LLMRequest, 
    LLMResponse, 
    LLMError, 
    LLMModelInfo,
    LLMCapability,
    CodeSnippet
)

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelInfo:
    """Information about an Ollama model."""
    name: str
    size: int
    modified_at: str
    digest: str
    details: Dict[str, Any]


class OllamaAdapter(LLMAdapter):
    """
    Ollama LLM adapter for local LLM inference.
    
    Provides integration with Ollama API for text generation,
    code analysis, and other AI tasks using local models.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "llama3.2:3b",
        max_tokens: int = 8192,
        temperature: float = 0.1,
        timeout: int = 120,
        **kwargs
    ):
        """
        Initialize Ollama adapter.
        
        Args:
            base_url: Ollama server base URL
            model_name: Name of the Ollama model to use
            max_tokens: Maximum tokens for generation
            temperature: Sampling temperature
            timeout: Request timeout in seconds
            **kwargs: Additional configuration
        """
        super().__init__(
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    def get_provider_name(self) -> str:
        """Get the provider name."""
        return "ollama"

    async def is_available(self) -> bool:
        """Check if Ollama server is available."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama server not available: {e}")
            return False

    async def list_models(self) -> List[OllamaModelInfo]:
        """List available Ollama models."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tags") as response:
                if response.status != 200:
                    raise LLMError(
                        message=f"Failed to list models: HTTP {response.status}",
                        provider=self.get_provider_name(),
                        model=self.model_name
                    )
                
                data = await response.json()
                models = []
                for model_data in data.get("models", []):
                    models.append(OllamaModelInfo(
                        name=model_data["name"],
                        size=model_data["size"],
                        modified_at=model_data["modified_at"],
                        digest=model_data["digest"],
                        details=model_data.get("details", {})
                    ))
                return models
                
        except Exception as e:
            raise LLMError(
                message=f"Failed to list models: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                cause=e
            )

    async def pull_model(self, model_name: str) -> None:
        """Pull/download a model from Ollama registry."""
        try:
            session = await self._get_session()
            payload = {"name": model_name}
            
            async with session.post(
                f"{self.base_url}/api/pull",
                json=payload
            ) as response:
                if response.status != 200:
                    raise LLMError(
                        message=f"Failed to pull model {model_name}: HTTP {response.status}",
                        provider=self.get_provider_name(),
                        model=model_name
                    )
                
                # Stream the pull progress
                async for line in response.content:
                    if line:
                        try:
                            progress = json.loads(line.decode())
                            if progress.get("status"):
                                logger.info(f"Pull progress: {progress['status']}")
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            raise LLMError(
                message=f"Failed to pull model {model_name}: {str(e)}",
                provider=self.get_provider_name(),
                model=model_name,
                retryable=self._is_retryable_error(e),
                cause=e
            )

    async def generate_text(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        try:
            session = await self._get_session()
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": kwargs.get("temperature", self.temperature),
                    "num_predict": kwargs.get("max_tokens", self.max_tokens),
                    "top_p": kwargs.get("top_p", 0.9),
                    "top_k": kwargs.get("top_k", 40),
                }
            }
            
            if system_prompt:
                payload["system"] = system_prompt
            
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise LLMError(
                        message=f"Generation failed: HTTP {response.status} - {error_text}",
                        provider=self.get_provider_name(),
                        model=self.model_name
                    )
                
                if stream:
                    # Handle streaming response
                    full_response = ""
                    async for line in response.content:
                        if line:
                            try:
                                chunk = json.loads(line.decode())
                                if chunk.get("response"):
                                    full_response += chunk["response"]
                                if chunk.get("done", False):
                                    break
                            except json.JSONDecodeError:
                                continue
                    return full_response
                else:
                    # Handle non-streaming response
                    data = await response.json()
                    return data.get("response", "")
                    
        except Exception as e:
            raise LLMError(
                message=f"Text generation failed: {str(e)}",
                provider=self.get_provider_name(),
                model=self.model_name,
                retryable=self._is_retryable_error(e),
                cause=e
            )

    async def _generate_raw_response(self, request: LLMRequest) -> LLMResponse:
        """Generate raw response for LLMRequest."""
        self.validate_request(request)
        
        # Create system prompt if needed
        system_prompt = None
        if request.system_prompt:
            system_prompt = request.system_prompt
        
        # Generate response
        response_text = await self.generate_text(
            prompt=request.prompt,
            system_prompt=system_prompt,
            temperature=request.temperature or self.temperature,
            max_tokens=request.max_tokens or self.max_tokens,
            stream=request.stream or False
        )
        
        return LLMResponse(
            content=response_text,
            model=self.model_name,
            provider=self.get_provider_name(),
            token_count=self.get_token_estimate(response_text),
            metadata=request.metadata or {}
        )

    async def synthesize_answer(
        self,
        question: str,
        evidence_snippets: List[CodeSnippet],
        intents: List[Dict[str, Any]],
        repo_info: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Synthesize an intelligent answer using Ollama based on code evidence."""
        
        # Create system prompt for code analysis
        system_prompt = self._create_system_prompt("analysis")
        
        # Format evidence context
        evidence_context = self._format_evidence_context(evidence_snippets)
        
        # Create comprehensive prompt
        prompt_parts = [
            f"## Question\n{question}\n",
            evidence_context,
        ]
        
        if intents:
            intent_text = "\n".join([f"- {intent.get('type', 'unknown')}: {intent.get('description', '')}" for intent in intents])
            prompt_parts.append(f"## User Intent\n{intent_text}\n")
        
        if repo_info:
            repo_context = f"## Repository Context\n"
            repo_context += f"- Name: {repo_info.get('name', 'Unknown')}\n"
            repo_context += f"- Language: {repo_info.get('language', 'Unknown')}\n"
            repo_context += f"- Description: {repo_info.get('description', 'No description')}\n"
            prompt_parts.append(repo_context)
        
        prompt_parts.append(
            "\n## Instructions\n"
            "Based on the code evidence provided, give a comprehensive answer to the question. "
            "Reference specific code snippets and explain how they relate to the question. "
            "Provide concrete examples and be precise in your analysis.\n"
        )
        
        full_prompt = "\n".join(prompt_parts)
        
        return await self.generate_text(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

    async def explain_code_snippet(
        self, 
        snippet: CodeSnippet, 
        context: Optional[str] = None
    ) -> str:
        """Provide detailed explanation of a specific code snippet."""
        
        system_prompt = self._create_system_prompt("explanation")
        
        prompt_parts = [
            f"## Code to Explain\n",
            f"**File**: `{snippet.path}`\n",
            f"**Lines**: {snippet.line_start}-{snippet.line_end}\n\n",
            "```\n",
            snippet.text,
            "\n```\n"
        ]
        
        if context:
            prompt_parts.append(f"## Additional Context\n{context}\n")
        
        prompt_parts.append(
            "\n## Instructions\n"
            "Provide a clear, detailed explanation of this code snippet. "
            "Explain what it does, how it works, and any important patterns or techniques used.\n"
        )
        
        full_prompt = "\n".join(prompt_parts)
        
        return await self.generate_text(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

    async def suggest_improvements(
        self, 
        evidence_snippets: List[CodeSnippet], 
        focus_area: Optional[str] = None
    ) -> str:
        """Suggest improvements for the provided code evidence."""
        
        system_prompt = self._create_system_prompt("improvement")
        
        evidence_context = self._format_evidence_context(evidence_snippets)
        
        prompt_parts = [
            "## Code for Improvement Analysis\n",
            evidence_context
        ]
        
        if focus_area:
            prompt_parts.append(f"## Focus Area\n{focus_area}\n")
        
        prompt_parts.append(
            "\n## Instructions\n"
            "Analyze the provided code and suggest specific improvements. "
            "Consider performance, readability, maintainability, security, and best practices. "
            "Provide concrete examples and explain the reasoning for each suggestion.\n"
        )
        
        full_prompt = "\n".join(prompt_parts)
        
        return await self.generate_text(
            prompt=full_prompt,
            system_prompt=system_prompt
        )

    def get_model_info(self) -> LLMModelInfo:
        """Get information about the configured Ollama model."""
        return LLMModelInfo(
            model_name=self.model_name,
            provider=self.get_provider_name(),
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            capabilities=[
                LLMCapability.TEXT_GENERATION,
                LLMCapability.CODE_ANALYSIS,
                LLMCapability.QUESTION_ANSWERING,
            ],
            supports_streaming=True,
            context_window=self.max_tokens,
        )

    def _is_retryable_error(self, error: Exception) -> bool:
        """Determine if an error is retryable for Ollama."""
        error_str = str(error).lower()
        
        # Ollama-specific retryable errors
        ollama_retryable = [
            "connection refused",
            "connection timeout", 
            "read timeout",
            "server error",
            "model not found",
            "temporarily unavailable"
        ]
        
        # Check for Ollama-specific patterns
        if any(pattern in error_str for pattern in ollama_retryable):
            return True
            
        # Fall back to parent class logic
        return super()._is_retryable_error(error)