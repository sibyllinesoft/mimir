"""
HyDE (Hypothetical Document Embeddings) query transformation.

Implements the HyDE technique where queries are transformed by generating
hypothetical documents that would answer the query, improving retrieval quality.
"""

import asyncio
import time
from typing import Dict, List, Optional

from ..config import get_ai_config
from ..util.errors import MimirError, create_error_context
from ..util.logging_config import get_logger
from ..monitoring import get_metrics_collector
from .llm_adapter import LLMAdapter
from .llm_adapter_factory import LLMAdapterFactory


logger = get_logger(__name__)


class HyDEError(MimirError):
    """Errors related to HyDE query transformation."""
    pass


class HyDETransformer:
    """
    HyDE query transformer for enhanced semantic search.
    
    Uses LLMs to generate hypothetical documents that would answer
    the user's query, then combines these with the original query
    for better embedding-based retrieval.
    """
    
    def __init__(self, llm_adapter: Optional[LLMAdapter] = None):
        """
        Initialize HyDE transformer.
        
        Args:
            llm_adapter: LLM adapter to use, defaults to configured provider
        """
        self.config = get_ai_config()
        self.llm_adapter = llm_adapter
        self.metrics_collector = get_metrics_collector()
        
        # HyDE generation settings
        self.max_hypothetical_length = 500
        self.num_hypotheticals = 1  # Can be increased for better quality
        self.temperature = 0.7
        
        # Query type detection patterns
        self.code_patterns = [
            "function", "method", "class", "implementation", "algorithm",
            "bug", "error", "exception", "code", "snippet", "example"
        ]
        
        self.explanation_patterns = [
            "what", "how", "why", "explain", "describe", "understand",
            "purpose", "meaning", "concept", "difference"
        ]
        
        logger.info("Initialized HyDE transformer")
    
    async def initialize(self) -> None:
        """Initialize the LLM adapter if not provided."""
        if self.llm_adapter is None:
            if not self.config.query.enable_hyde:
                logger.info("HyDE is disabled in configuration")
                return
                
            factory = LLMAdapterFactory()
            self.llm_adapter = await factory.create_adapter(
                provider=self.config.query.transformer_provider,
                model_name=self.config.query.transformer_model
            )
            logger.info(f"Initialized HyDE with {self.config.query.transformer_provider} provider")
    
    async def transform_query(self, query: str) -> str:
        """
        Transform a query using HyDE technique.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query combining original and hypothetical content
        """
        if not self.config.query.enable_hyde:
            logger.debug("HyDE disabled, returning original query")
            return query
            
        await self.initialize()
        
        if self.llm_adapter is None:
            logger.warning("HyDE LLM adapter not available, returning original query")
            return query
        
        start_time = time.time()
        
        try:
            # Detect query type to use appropriate prompt
            query_type = self._detect_query_type(query)
            
            # Generate hypothetical document(s)
            hypothetical_docs = await self._generate_hypothetical_documents(query, query_type)
            
            # Combine original query with hypothetical content
            enhanced_query = self._combine_query_and_hypotheticals(query, hypothetical_docs)
            
            execution_time = time.time() - start_time
            
            # Record metrics
            self.metrics_collector.record_query_transformation(
                "hyde",
                execution_time,
                len(enhanced_query),
                len(hypothetical_docs)
            )
            
            logger.debug(f"HyDE transformation completed in {execution_time:.3f}s")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"HyDE transformation failed: {e}")
            # Fallback to original query
            return query
    
    async def _generate_hypothetical_documents(
        self, 
        query: str, 
        query_type: str
    ) -> List[str]:
        """Generate hypothetical documents for the query."""
        prompts = self._build_prompts(query, query_type)
        hypothetical_docs = []
        
        for prompt in prompts:
            try:
                response = await self.llm_adapter.generate_response(
                    prompt,
                    max_tokens=self.max_hypothetical_length,
                    temperature=self.temperature
                )
                
                if response.success and response.text.strip():
                    hypothetical_docs.append(response.text.strip())
                    
            except Exception as e:
                logger.warning(f"Failed to generate hypothetical document: {e}")
                continue
        
        return hypothetical_docs
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to use appropriate prompts.
        
        Args:
            query: User query
            
        Returns:
            Query type: 'code', 'explanation', or 'general'
        """
        query_lower = query.lower()
        
        # Check for code-related patterns
        if any(pattern in query_lower for pattern in self.code_patterns):
            return "code"
            
        # Check for explanation patterns
        if any(pattern in query_lower for pattern in self.explanation_patterns):
            return "explanation"
            
        return "general"
    
    def _build_prompts(self, query: str, query_type: str) -> List[str]:
        """
        Build appropriate prompts for hypothetical document generation.
        
        Args:
            query: Original user query
            query_type: Detected type of query
            
        Returns:
            List of prompts to use for generation
        """
        prompts = []
        
        if query_type == "code":
            prompts.extend([
                f"""You are a senior software engineer. A developer asks: "{query}"
                
Write a high-quality, detailed code example or implementation that would perfectly answer this question. Include relevant comments and follow best practices.

Code example:""",
                
                f"""Given this programming question: "{query}"
                
Provide a comprehensive code solution with explanation. Show the implementation that demonstrates the answer clearly.

Implementation:"""
            ])
            
        elif query_type == "explanation":
            prompts.extend([
                f"""You are an expert software engineer writing documentation. A developer asks: "{query}"
                
Write a clear, comprehensive explanation that would fully answer this question. Include technical details, context, and examples where helpful.

Explanation:""",
                
                f"""Question: "{query}"
                
Provide a detailed technical explanation that covers all aspects of this question. Include the key concepts, how they work, and practical considerations.

Answer:"""
            ])
            
        else:  # general
            prompts.append(
                f"""You are a knowledgeable software engineer. Answer this question thoroughly: "{query}"
                
Provide a complete, detailed response that would fully satisfy this query. Include relevant technical information and practical insights.

Answer:"""
            )
        
        return prompts[:self.num_hypotheticals]
    
    def _combine_query_and_hypotheticals(
        self, 
        original_query: str, 
        hypothetical_docs: List[str]
    ) -> str:
        """
        Combine original query with hypothetical documents.
        
        Args:
            original_query: Original user query
            hypothetical_docs: Generated hypothetical documents
            
        Returns:
            Combined query for embedding
        """
        if not hypothetical_docs:
            return original_query
        
        # Start with original query (weighted heavily)
        combined_parts = [f"Query: {original_query}"]
        
        # Add hypothetical documents
        for i, doc in enumerate(hypothetical_docs):
            # Truncate very long hypothetical docs
            if len(doc) > self.max_hypothetical_length:
                doc = doc[:self.max_hypothetical_length] + "..."
            combined_parts.append(f"Example: {doc}")
        
        combined_query = "\n\n".join(combined_parts)
        
        # Ensure the combined query isn't too long
        max_combined_length = 2000  # Reasonable limit for embeddings
        if len(combined_query) > max_combined_length:
            # Truncate hypothetical parts proportionally
            available_space = max_combined_length - len(f"Query: {original_query}\n\n")
            if available_space > 100:
                truncated_hypotheticals = []
                space_per_doc = available_space // len(hypothetical_docs)
                
                for doc in hypothetical_docs:
                    if len(doc) > space_per_doc:
                        truncated_doc = doc[:space_per_doc-3] + "..."
                    else:
                        truncated_doc = doc
                    truncated_hypotheticals.append(f"Example: {truncated_doc}")
                
                combined_query = f"Query: {original_query}\n\n" + "\n\n".join(truncated_hypotheticals)
            else:
                # Fall back to original query if no space for hypotheticals
                combined_query = original_query
        
        return combined_query
    
    async def batch_transform_queries(self, queries: List[str]) -> List[str]:
        """
        Transform multiple queries efficiently.
        
        Args:
            queries: List of queries to transform
            
        Returns:
            List of transformed queries
        """
        if not self.config.query.enable_hyde or not queries:
            return queries
        
        await self.initialize()
        
        if self.llm_adapter is None:
            logger.warning("HyDE LLM adapter not available")
            return queries
        
        # Process queries concurrently with reasonable limit
        semaphore = asyncio.Semaphore(3)  # Limit concurrent requests
        
        async def transform_with_semaphore(query: str) -> str:
            async with semaphore:
                return await self.transform_query(query)
        
        tasks = [transform_with_semaphore(query) for query in queries]
        transformed_queries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions by falling back to original queries
        results = []
        for i, result in enumerate(transformed_queries):
            if isinstance(result, Exception):
                logger.warning(f"Failed to transform query {i}: {result}")
                results.append(queries[i])
            else:
                results.append(result)
        
        return results
    
    def get_configuration(self) -> Dict[str, any]:
        """Get current HyDE configuration."""
        return {
            "enabled": self.config.query.enable_hyde,
            "provider": self.config.query.transformer_provider,
            "model": self.config.query.transformer_model,
            "max_length": self.max_hypothetical_length,
            "num_hypotheticals": self.num_hypotheticals,
            "temperature": self.temperature,
            "adapter_initialized": self.llm_adapter is not None,
        }