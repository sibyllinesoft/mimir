"""
Hybrid search engine combining vector, symbol, and graph-based search.

Provides multi-modal search with configurable feature weights and
result merging for comprehensive code search capabilities.
"""

import asyncio
import re

from ..data.schemas import (
    Citation,
    CodeSnippet,
    FeatureConfig,
    RepoMap,
    SearchResponse,
    SearchResult,
    SearchScores,
    SerenaGraph,
    VectorIndex,
)
from ..monitoring import get_metrics_collector, get_trace_manager, search_metrics


class HybridSearchEngine:
    """
    Multi-modal search engine for code repositories.

    Combines vector similarity, symbol matching, and graph-based expansion
    to provide comprehensive search results with relevance scoring.
    """

    def __init__(self):
        """Initialize hybrid search engine."""
        # Feature weights for result scoring
        self.feature_weights = {"vector": 1.0, "symbol": 0.9, "graph": 0.3}

        # Search parameters
        self.max_results_per_feature = 100
        self.min_similarity_threshold = 0.1

        # Performance optimizations
        self._vector_cache = {}  # Query -> results cache
        self._cache_max_size = 1000
        self._early_termination_threshold = 0.8  # Stop search when high confidence results found
        self._batch_size = 50  # Process chunks in batches for better memory management

        # Initialize monitoring
        self.metrics_collector = get_metrics_collector()
        self.trace_manager = get_trace_manager()

    @search_metrics("hybrid")
    async def search(
        self,
        query: str,
        vector_index: VectorIndex | None,
        serena_graph: SerenaGraph | None,
        repomap: RepoMap | None,
        repo_root: str,
        rev: str,
        features: FeatureConfig,
        k: int = 20,
        context_lines: int = 5,
    ) -> SearchResponse:
        """
        Execute hybrid search across all enabled modalities.

        Returns ranked search results with citations and content.
        """
        import time

        start_time = time.time()

        # Create trace context
        async with self.trace_manager.trace_search_request(
            "hybrid", query, k=k, features_enabled=features.dict()
        ) as span:

            # Initialize result candidates
            vector_candidates = []
            symbol_candidates = []
            graph_candidates = []

            # Execute searches in parallel where possible
            search_tasks = []

            if features.vector and vector_index:
                search_tasks.append(self._vector_search(query, vector_index))

            if features.symbol and serena_graph:
                search_tasks.append(self._symbol_search(query, serena_graph))

            # Execute searches concurrently
            if search_tasks:
                search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

                result_idx = 0
                if features.vector and vector_index:
                    if not isinstance(search_results[result_idx], Exception):
                        vector_candidates = search_results[result_idx]
                    result_idx += 1

                if features.symbol and serena_graph:
                    if not isinstance(search_results[result_idx], Exception):
                        symbol_candidates = search_results[result_idx]

            # Apply graph expansion if enabled
            if features.graph and repomap:
                all_candidates = vector_candidates + symbol_candidates
                graph_candidates = await self._graph_expansion(all_candidates, repomap)

            # Merge and score results
            merged_results = await self._merge_and_score_results(
                vector_candidates,
                symbol_candidates,
                graph_candidates,
                repo_root,
                rev,
                context_lines,
            )

            # Sort by combined score and limit results
            merged_results.sort(key=lambda r: r.score, reverse=True)
            final_results = merged_results[:k]

            execution_time_ms = (time.time() - start_time) * 1000
            execution_time_seconds = execution_time_ms / 1000

            # Record search metrics
            self.metrics_collector.record_search_request(
                "hybrid", execution_time_seconds, len(final_results), "success"
            )

            # Record vector similarity scores for monitoring
            for candidate in vector_candidates:
                if len(candidate) >= 3:  # (path, span, score)
                    self.metrics_collector.record_vector_similarity(candidate[2])

            # Add span attributes
            span.set_attribute("search.results_count", len(final_results))
            span.set_attribute("search.total_candidates", len(merged_results))
            span.set_attribute("search.execution_time_ms", execution_time_ms)
            span.set_attribute("search.vector_candidates", len(vector_candidates))
            span.set_attribute("search.symbol_candidates", len(symbol_candidates))
            span.set_attribute("search.graph_candidates", len(graph_candidates))

            return SearchResponse(
                query=query,
                results=final_results,
                total_count=len(final_results),
                features_used=features,
                execution_time_ms=execution_time_ms,
                index_id="",  # Will be set by caller
            )

    async def _vector_search(
        self, query: str, vector_index: VectorIndex
    ) -> list[tuple[str, tuple[int, int], float]]:
        """
        Execute vector similarity search with caching and early termination.

        Returns list of (file_path, span, score) tuples.
        """
        # Check cache first
        cache_key = f"{query}:{len(vector_index.chunks)}"
        if cache_key in self._vector_cache:
            self.metrics_collector.record_cache_hit("vector_search")
            return self._vector_cache[cache_key]

        # Cache miss
        self.metrics_collector.record_cache_miss("vector_search")

        from .leann import LEANNAdapter

        if not vector_index.chunks:
            return []

        try:
            leann = LEANNAdapter()
            similar_chunks = await leann.search_similar(
                query, vector_index, k=self.max_results_per_feature
            )

            candidates = []
            high_confidence_count = 0

            for chunk, similarity in similar_chunks:
                if similarity >= self.min_similarity_threshold:
                    candidates.append((chunk.path, chunk.span, similarity))

                    # Record similarity scores for monitoring
                    self.metrics_collector.record_vector_similarity(similarity)

                    # Early termination optimization
                    if similarity >= self._early_termination_threshold:
                        high_confidence_count += 1
                        # If we have enough high-confidence results, stop searching
                        if high_confidence_count >= 20:
                            break

            # Cache results (with LRU eviction)
            if len(self._vector_cache) >= self._cache_max_size:
                # Remove oldest entry (simple FIFO for now)
                oldest_key = next(iter(self._vector_cache))
                del self._vector_cache[oldest_key]

            self._vector_cache[cache_key] = candidates

            return candidates

        except Exception as e:
            print(f"Vector search failed: {e}")
            return []

    async def _symbol_search(
        self, query: str, serena_graph: SerenaGraph
    ) -> list[tuple[str, tuple[int, int], float]]:
        """
        Execute symbol name and signature matching.

        Returns list of (file_path, span, score) tuples.
        """
        if not serena_graph.entries:
            return []

        candidates = []
        query_lower = query.lower()
        query_tokens = set(re.findall(r"\w+", query_lower))

        for entry in serena_graph.entries:
            score = 0.0

            # Symbol name matching
            if entry.symbol:
                symbol_lower = entry.symbol.lower()

                # Exact match gets highest score
                if query_lower == symbol_lower:
                    score += 1.0
                # Substring match
                elif query_lower in symbol_lower:
                    score += 0.8
                # Token overlap
                else:
                    symbol_tokens = set(re.findall(r"\w+", symbol_lower))
                    token_overlap = len(query_tokens & symbol_tokens)
                    if token_overlap > 0:
                        score += 0.4 * (token_overlap / len(query_tokens))

            # Signature matching for definitions
            if entry.sig and entry.type.value == "def":
                sig_lower = entry.sig.lower()
                if any(token in sig_lower for token in query_tokens):
                    score += 0.3

            # Path matching
            path_lower = entry.path.lower()
            if any(token in path_lower for token in query_tokens):
                score += 0.2

            if score >= self.min_similarity_threshold:
                candidates.append((entry.path, entry.span, score))

        # Sort by score and limit results
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[: self.max_results_per_feature]

    async def _graph_expansion(
        self, candidates: list[tuple[str, tuple[int, int], float]], repomap: RepoMap
    ) -> list[tuple[str, tuple[int, int], float]]:
        """
        Expand search results using repository graph structure.

        Finds related files through dependency relationships.
        """
        if not candidates or not repomap.edges:
            return []

        # Get files from current candidates
        candidate_files = {path for path, _, _ in candidates}

        # Build adjacency graph
        graph = {}
        for edge in repomap.edges:
            if edge.source not in graph:
                graph[edge.source] = []
            graph[edge.source].append((edge.target, edge.weight))

        # Find related files
        related_files = set()
        for file_path in candidate_files:
            if file_path in graph:
                for target, weight in graph[file_path]:
                    if weight > 0.1:  # Only consider significant relationships
                        related_files.add(target)

        # Create graph expansion candidates
        expansion_candidates = []
        for file_path in related_files:
            if file_path not in candidate_files:
                # Assign graph score based on file importance
                file_rank = 0.0
                for rank_info in repomap.file_ranks:
                    if rank_info.path == file_path:
                        file_rank = rank_info.rank
                        break

                # Create a representative span for the file (beginning)
                span = (0, 100)
                score = file_rank * 0.5  # Graph expansion gets lower base score

                if score >= self.min_similarity_threshold:
                    expansion_candidates.append((file_path, span, score))

        return expansion_candidates[:20]  # Limit graph expansion results

    async def _merge_and_score_results(
        self,
        vector_candidates: list[tuple[str, tuple[int, int], float]],
        symbol_candidates: list[tuple[str, tuple[int, int], float]],
        graph_candidates: list[tuple[str, tuple[int, int], float]],
        repo_root: str,
        rev: str,
        context_lines: int,
    ) -> list[SearchResult]:
        """
        Merge candidates from all search modalities and create final results.

        Handles span deduplication and score combination.
        """
        # Track all candidates with their source modality
        all_candidates = {}

        # Add vector candidates
        for path, span, score in vector_candidates:
            key = (path, span)
            if key not in all_candidates:
                all_candidates[key] = SearchScores()
            all_candidates[key].vector = score

        # Add symbol candidates
        for path, span, score in symbol_candidates:
            key = (path, span)
            if key not in all_candidates:
                all_candidates[key] = SearchScores()
            all_candidates[key].symbol = score

        # Add graph candidates
        for path, span, score in graph_candidates:
            key = (path, span)
            if key not in all_candidates:
                all_candidates[key] = SearchScores()
            all_candidates[key].graph = score

        # Create search results
        search_results = []

        for (path, span), scores in all_candidates.items():
            # Calculate combined score
            combined_score = (
                self.feature_weights["vector"] * scores.vector
                + self.feature_weights["symbol"] * scores.symbol
                + self.feature_weights["graph"] * scores.graph
            )

            # Create placeholder content (would be extracted from file in practice)
            content = CodeSnippet(
                path=path,
                span=span,
                hash="",  # Would compute actual hash
                pre="",  # Would extract actual context
                text=f"Content at {path}:{span[0]}-{span[1]}",
                post="",
                line_start=1,
                line_end=1,
            )

            # Create citation
            citation = Citation(
                repo_root=repo_root,
                rev=rev,
                path=path,
                span=span,
                content_sha="",  # Would compute actual hash
            )

            result = SearchResult(
                path=path,
                span=span,
                score=combined_score,
                scores=scores,
                content=content,
                citation=citation,
            )

            search_results.append(result)

        return search_results

    async def search_by_symbol(
        self, symbol_name: str, serena_graph: SerenaGraph, exact_match: bool = False
    ) -> list[SearchResult]:
        """
        Search for specific symbol across the codebase.

        Useful for "go to definition" and "find references" functionality.
        """
        if not serena_graph.entries:
            return []

        results = []

        for entry in serena_graph.entries:
            if not entry.symbol:
                continue

            match = False
            if exact_match:
                match = entry.symbol == symbol_name
            else:
                match = symbol_name.lower() in entry.symbol.lower()

            if match:
                # Create high-confidence result
                scores = SearchScores(symbol=1.0)

                content = CodeSnippet(
                    path=entry.path,
                    span=entry.span,
                    hash="",
                    pre="",
                    text=entry.symbol,
                    post="",
                    line_start=1,
                    line_end=1,
                )

                citation = Citation(
                    repo_root="", rev="", path=entry.path, span=entry.span, content_sha=""
                )

                result = SearchResult(
                    path=entry.path,
                    span=entry.span,
                    score=1.0,
                    scores=scores,
                    content=content,
                    citation=citation,
                )

                results.append(result)

        # Sort by symbol type (definitions first)
        def sort_key(r):
            # This would examine the entry type if we had access to it
            return (1, r.path, r.span[0])  # Simplified sorting

        results.sort(key=sort_key)
        return results

    def configure_weights(
        self, vector_weight: float = 1.0, symbol_weight: float = 0.9, graph_weight: float = 0.3
    ) -> None:
        """Configure feature weights for result scoring."""
        self.feature_weights = {
            "vector": vector_weight,
            "symbol": symbol_weight,
            "graph": graph_weight,
        }
