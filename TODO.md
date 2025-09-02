## Mimir Enhancement Implementation Plan: "Mimir 2.0"

This plan integrates all discussed features: **RAPTOR**, **Ollama Adapter**, **Code-Specific Embeddings**, **Query Transformations (HyDE)**, and a **configurable Reranking stage**.

### Guiding Principles
1.  **Modularity:** Each new feature will be implemented as a distinct, pluggable component.
2.  **Configurability:** All new features will be optional and configurable via `MimirConfig`.
3.  **Backward Compatibility:** The system must function without these new features enabled, matching its current behavior.
4.  **Performance Awareness:** Local models (Ollama) will be leveraged for speed-critical, low-latency tasks.

---

### **Part 1: Foundational - Ollama & Pluggable LLM Adapters**

This is the prerequisite for making RAPTOR and HyDE feasible with local models.

**1.1. New `OllamaAdapter`**
*   **File:** `src/repoindex/pipeline/ollama.py`
*   **Class:** `OllamaAdapter(LLMAdapter)`
*   **Dependencies:** Add `httpx` to `pyproject.toml` if it's not already a core dependency.
*   **Configuration:**
    *   In `src/repoindex/config.py`, add `OllamaConfig` inside `AIConfig`.
        ```python
        class OllamaConfig(BaseSettings):
            host: str = Field(default="http://localhost:11434", env="OLLAMA_HOST")
            default_model: str = Field(default="gemma2:9b", env="OLLAMA_DEFAULT_MODEL")
            request_timeout: int = Field(default=120, env="OLLAMA_TIMEOUT")
        
        class AIConfig(BaseSettings):
            # ... existing fields ...
            ollama: OllamaConfig = Field(default_factory=OllamaConfig)
        ```
*   **Implementation Details:**
    *   The `__init__` method will take an `OllamaConfig` object.
    *   Implement the abstract `_generate_raw_response` method.
        *   It will use `httpx.AsyncClient` to make a `POST` request to `f"{self.config.host}/api/generate"`.
        *   The JSON payload will include `model`, `prompt`, and `"stream": False`.
        *   Handle potential `httpx.HTTPStatusError` and `httpx.TimeoutException`, wrapping them in a standard `LLMError`.
    *   Implement `get_provider_name` to return `"ollama"`.
    *   Implement `get_model_info` to return generic info about the Ollama-proxied model.

**1.2. LLM Adapter Factory**
*   **File:** `src/repoindex/pipeline/llm_adapter_factory.py` (New File)
*   **Purpose:** To dynamically create the correct LLM adapter based on configuration. This decouples the rest of the application from specific adapter implementations.
*   **Function:** `create_llm_adapter(provider: str, model_name: str | None = None) -> LLMAdapter:`
    ```python
    from ..config import get_ai_config
    from .gemini import GeminiAdapter
    from .ollama import OllamaAdapter # New import
    
    def create_llm_adapter(provider, model_name=None):
        ai_config = get_ai_config()
        if provider == "gemini":
            return GeminiAdapter(model_name=model_name or ai_config.gemini_model)
        elif provider == "ollama":
            return OllamaAdapter(model_name=model_name or ai_config.ollama.default_model)
        # Add other adapters like Anthropic, OpenAI later
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
    ```

---

### **Part 2: RAPTOR - Hierarchical Indexing**

This adds the core hierarchical summarization capability.

**2.1. Configuration**
*   **File:** `src/repoindex/config.py`
*   **New Class:** `RaptorConfig(BaseSettings)` inside `PipelineConfig`.
    ```python
    class RaptorConfig(BaseSettings):
        enabled: bool = Field(default=False, env="PIPELINE_RAPTOR_ENABLED")
        summarizer_provider: Literal["gemini", "ollama"] = Field(default="ollama", env="RAPTOR_SUMMARIZER_PROVIDER")
        summarizer_model: str | None = Field(default=None, env="RAPTOR_SUMMARIZER_MODEL") # e.g., "gemma2:9b"
        clustering_algorithm: Literal["hdbscan", "kmeans"] = Field(default="hdbscan", env="RAPTOR_CLUSTERING_ALGORITHM")
        hdbscan_min_cluster_size: int = Field(default=5, env="RAPTOR_HDBSCAN_MIN_CLUSTER_SIZE")
        max_levels: int = Field(default=5, env="RAPTOR_MAX_LEVELS")
        # Add more tuning params as needed
    
    class PipelineConfig(BaseSettings):
        # ...
        raptor: RaptorConfig = Field(default_factory=RaptorConfig)
    ```

**2.2. RAPTOR Stage Implementation**
*   **File:** `src/repoindex/pipeline/raptor.py` (New File)
*   **Dependencies:** Add `scikit-learn` and `hdbscan` to `pyproject.toml`. `hdbscan` might require `cython`.
*   **Class:** `RaptorProcessor`
    *   `__init__(self, config: RaptorConfig)`: Initializes the processor, including creating the summarizer LLM adapter using the factory from Part 1.
    *   `async def build_tree(self, l0_chunks: List[VectorChunk]) -> RaptorTree:`: The main entry point.
        1.  Initializes `current_level_chunks = l0_chunks`.
        2.  Enters a `for` loop up to `config.max_levels`.
        3.  **Cluster:** Calls `_cluster_chunks(current_level_chunks)`.
        4.  **Check Termination:** If clustering yields too few new clusters, break the loop.
        5.  **Summarize:** Asynchronously calls `_summarize_cluster` for each new cluster using `asyncio.gather`.
        6.  **Handle Noise:** Pass noise chunks up to the next level.
        7.  **Update:** Sets `current_level_chunks` to the list of new summary chunks + noise chunks.
    *   `_cluster_chunks(self, chunks: List[VectorChunk]) -> Dict[int, List[VectorChunk]]:`:
        1.  Extracts embeddings from chunks into a NumPy array.
        2.  Initializes `hdbscan.HDBSCAN(min_cluster_size=...)`.
        3.  Fits the model to the embeddings.
        4.  Returns a dictionary mapping cluster labels (including `-1` for noise) to the list of chunks in that cluster.
    *   `async def _summarize_cluster(self, cluster_chunks: List[VectorChunk]) -> VectorChunk:`:
        1.  Concatenates the `content` of all chunks in the cluster.
        2.  Constructs a detailed prompt for the LLM: `"You are a senior software engineer. Summarize the following code snippets, focusing on their collective purpose and functionality. Snippets:\n\n{concatenated_code}"`.
        3.  Calls `self.summarizer_adapter.generate_response()`.
        4.  Creates a new `VectorChunk` with the summary as its content, an empty span, and a unique `chunk_id`.
        5.  Calls `LEANNAdapter` to generate an embedding for this new summary chunk.
        6.  Returns the new summary `VectorChunk`.

**2.3. Data Structures**
*   **File:** `src/repoindex/data/schemas.py`
*   **Modify `VectorChunk`:** Add `parent_id: Optional[str] = None` and `level: int = 0`.
*   **New Class:** `RaptorTree`: This will likely be represented by modifying `VectorIndex` to be a flat list of all chunks from all levels, with parent/child relationships encoded in the chunks themselves.
    ```python
    class VectorIndex(BaseModel):
        # ... existing fields ...
        # The chunks list will now contain L0, L1, L2... chunks
        chunks: List[VectorChunk] = []
        
        def get_children(self, chunk_id: str) -> List[VectorChunk]:
            # Method to traverse the tree down
            ...
        def get_parent(self, chunk_id: str) -> Optional[VectorChunk]:
            # Method to traverse up
            ...
    ```

**2.4. Pipeline Integration**
*   **File:** `src/repoindex/pipeline/stages.py`
*   **New Class:** `RaptorStage(AsyncPipelineStage)`
    *   The `execute` method will check `context.config.pipeline.raptor.enabled`.
    *   It will instantiate `RaptorProcessor`.
    *   It will take `context.vector_index.chunks` as input.
    *   The result (the full list of L0, L1, L2... chunks) will be used to *replace* `context.vector_index.chunks`.

---

### **Part 3: Embeddings & Search Enhancements**

**3.1. Code-Specific Embedding Model**
*   **File:** `src/repoindex/config.py`
*   **Modify `AIConfig`:** Change the default for `embedding_model` from a generic one to a code-specific one.
    ```python
    embedding_model: str = Field(default="BAAI/bge-large-en-v1.5", env="EMBEDDING_MODEL") 
    # Or another good code model like unixcoder, etc.
    ```
*   **File:** `src/repoindex/pipeline/leann.py`
*   **Modify `LEANNAdapter`:** Ensure it correctly loads the specified model. Since you're using a stub, this just means passing the `model_name` correctly. The real implementation would use a library like `sentence-transformers` that can download from Hugging Face.

**3.2. Query Transformation (HyDE)**
*   **File:** `src/repoindex/config.py`
*   **New Class:** `QueryConfig(BaseSettings)` within `AIConfig`.
    ```python
    class QueryConfig(BaseSettings):
        enable_hyde: bool = Field(default=True, env="QUERY_ENABLE_HYDE")
        transformer_provider: Literal["gemini", "ollama"] = Field(default="ollama", env="QUERY_TRANSFORMER_PROVIDER")
        transformer_model: str | None = Field(default=None, env="QUERY_TRANSFORMER_MODEL")
    
    class AIConfig(BaseSettings):
        # ...
        query: QueryConfig = Field(default_factory=QueryConfig)
    ```
*   **File:** `src/repoindex/pipeline/query_engine.py`
*   **Modify `QueryEngine.search()` and `ask()`:**
    1.  At the beginning of the methods, check `ai_config.query.enable_hyde`.
    2.  If true, create a transformer adapter: `transformer = create_llm_adapter(ai_config.query.transformer_provider, ...)`
    3.  Call `_transform_query_with_hyde(query, transformer)`.
    4.  Use the transformed query for the vector search step.
*   **New Method:** `async def _transform_query_with_hyde(self, query: str, transformer: LLMAdapter) -> str:`
    1.  Construct the HyDE prompt: `"You are a senior software engineer. Given the user's question about a codebase, write a detailed, high-quality code snippet that would be the perfect answer. Question: {query}"`.
    2.  Call `transformer.generate_response()` with this prompt.
    3.  Return the original query concatenated with the hypothetical document: `f"{query}\n\n{hypothetical_answer.text}"`. This combined text is then embedded for the search.

**3.3. Configurable Reranking**
*   **File:** `src/repoindex/config.py`
*   **New Class:** `RerankerConfig(BaseSettings)` within `AIConfig`.
    ```python
    class RerankerConfig(BaseSettings):
        enabled: bool = Field(default=True, env="QUERY_RERANKER_ENABLED")
        model: str = Field(default="BAAI/bge-reranker-large", env="QUERY_RERANKER_MODEL")
        top_k: int = Field(default=20, env="QUERY_RERANKER_TOP_K") # How many results to keep after reranking
        initial_retrieval_k: int = Field(default=100, env="QUERY_INITIAL_K")
    
    class AIConfig(BaseSettings):
        # ...
        reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    ```
*   **File:** `src/repoindex/pipeline/reranker.py` (New File)
*   **Dependencies:** Add `sentence-transformers` to `pyproject.toml`.
*   **Class:** `CrossEncoderReranker`
    *   `__init__(self, config: RerankerConfig)`: Loads the cross-encoder model from sentence-transformers onto the GPU (if available).
    *   `rerank(self, query: str, chunks: List[VectorChunk]) -> List[VectorChunk]:`
        1.  Creates pairs of `(query, chunk.content)` for each input chunk.
        2.  Calls `self.model.predict(pairs)` to get relevance scores.
        3.  Adds the score to each chunk object.
        4.  Sorts the chunks by the new score in descending order.
        5.  Returns the sorted list of chunks.
*   **File:** `src/repoindex/pipeline/hybrid_search.py`
*   **Modify `HybridSearchEngine.search()`:**
    1.  Use `ai_config.reranker.initial_retrieval_k` when calling `_vector_search` and `_symbol_search` to retrieve a large number of initial candidates.
    2.  Before the final merge and scoring, check `ai_config.reranker.enabled`.
    3.  If true:
        *   Instantiate `CrossEncoderReranker`.
        *   Call `reranker.rerank(query, all_candidate_chunks)`.
        *   Take the top `ai_config.reranker.top_k` results from the reranked list to pass to the final LLM.
    4.  If false, proceed with the existing merge/score logic.

---

### **Part 4: Testing & Validation Plan**

For each new feature, create a corresponding test file.

1.  **`tests/unit/pipeline/test_ollama_adapter.py`:** Mock the `httpx` client to test that the adapter correctly formats requests and parses responses for the Ollama API.
2.  **`tests/unit/pipeline/test_raptor.py`:**
    *   Test `_cluster_chunks` with mock embeddings to ensure it correctly groups chunks and identifies noise. Use a small, predictable dataset.
    *   Test `_summarize_cluster` by mocking the `LLMAdapter` to ensure it concatenates text and constructs the prompt correctly.
    *   Test the main `build_tree` loop to ensure it terminates correctly and builds a valid tree structure.
3.  **`tests/integration/test_full_pipeline_with_raptor.py`:** A slow integration test that runs the entire pipeline with RAPTOR enabled on a small, sample repository.
4.  **`tests/unit/pipeline/test_query_engine_hyde.py`:** Test that `QueryEngine.search()` calls the transformer adapter and modifies the query before searching, when HyDE is enabled.
5.  **`tests/unit/pipeline/test_reranker.py`:** Test the `CrossEncoderReranker` with a small set of documents to ensure it reorders them correctly based on a query.
6.  **`tests/integration/test_search_with_reranker.py`:** Test the end-to-end search flow, verifying that the number of documents passed to the final LLM respects the `top_k` setting when the reranker is enabled.

This detailed plan provides a clear roadmap to significantly enhance Mimir's capabilities, transforming it into a highly flexible, powerful, and production-ready system for deep code research.