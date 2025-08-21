"""
Multi-hop symbol graph navigation for complex queries.

Provides intelligent question answering through symbol graph traversal,
intent parsing, and evidence gathering for code understanding.
"""

import re

from ..data.schemas import AskResponse, Citation, CodeSnippet, SerenaGraph, SymbolEntry, SymbolType


class Intent:
    """Represents a parsed query intent."""

    def __init__(self, intent_type: str, targets: list[str], context: str = ""):
        self.intent_type = intent_type
        self.targets = targets
        self.context = context


class SymbolGraphNavigator:
    """
    Multi-hop reasoning engine for code questions.

    Parses natural language queries, navigates symbol relationships,
    and synthesizes answers with supporting evidence citations.
    """

    def __init__(self):
        """Initialize symbol graph navigator."""
        self.max_hops = 3
        self.max_evidence_items = 10

        # Performance optimizations
        self._symbol_index = {}  # symbol -> List[SymbolEntry]
        self._indexed_graph = None  # Track which graph is indexed

        # Intent patterns for query classification
        self.intent_patterns = {
            "definition": [
                r"what is (.*)",
                r"define (.*)",
                r"how is (.*) defined",
                r"show me (.*) definition",
            ],
            "usage": [
                r"how is (.*) used",
                r"where is (.*) called",
                r"find usages of (.*)",
                r"who calls (.*)",
            ],
            "relationship": [
                r"how does (.*) relate to (.*)",
                r"what's the relationship between (.*) and (.*)",
                r"how are (.*) and (.*) connected",
            ],
            "flow": [
                r"how does (.*) work",
                r"trace the flow of (.*)",
                r"walk through (.*)",
                r"explain the process of (.*)",
            ],
            "dependency": [
                r"what does (.*) depend on",
                r"what depends on (.*)",
                r"show dependencies of (.*)",
                r"find imports of (.*)",
            ],
        }

    def _build_symbol_index(self, serena_graph: SerenaGraph) -> None:
        """Build optimized symbol index for fast lookups."""
        if self._indexed_graph is serena_graph:
            return  # Already indexed

        self._symbol_index.clear()

        for entry in serena_graph.entries:
            if entry.symbol:
                symbol_lower = entry.symbol.lower()
                if symbol_lower not in self._symbol_index:
                    self._symbol_index[symbol_lower] = []
                self._symbol_index[symbol_lower].append(entry)

        self._indexed_graph = serena_graph

    async def ask(
        self,
        question: str,
        serena_graph: SerenaGraph,
        repo_root: str,
        rev: str,
        context_lines: int = 5,
    ) -> AskResponse:
        """
        Answer a question about the codebase using symbol graph navigation.

        Returns structured answer with supporting citations.
        """
        import time

        start_time = time.time()

        # Parse question intent
        intents = await self._parse_intent(question)

        if not intents:
            return AskResponse(
                question=question,
                answer="I couldn't understand the question. Try asking about specific symbols, functions, or relationships in the code.",
                citations=[],
                execution_time_ms=(time.time() - start_time) * 1000,
                index_id="",
            )

        # Plan symbol navigation based on intents
        seed_symbols = await self._plan_symbols(intents, serena_graph)

        if not seed_symbols:
            return AskResponse(
                question=question,
                answer="I couldn't find the symbols or concepts mentioned in your question.",
                citations=[],
                execution_time_ms=(time.time() - start_time) * 1000,
                index_id="",
            )

        # Navigate symbol graph
        evidence_symbols = await self._walk_symbol_graph(seed_symbols, serena_graph, intents)

        # Gather evidence snippets
        evidence_snippets = await self._gather_evidence(evidence_symbols, context_lines)

        # Synthesize answer
        answer = await self._synthesize_answer(question, intents, evidence_snippets)

        # Create citations
        citations = [
            Citation(
                repo_root=repo_root,
                rev=rev,
                path=snippet.path,
                span=snippet.span,
                content_sha=snippet.hash,
            )
            for snippet in evidence_snippets
        ]

        execution_time_ms = (time.time() - start_time) * 1000

        return AskResponse(
            question=question,
            answer=answer,
            citations=citations,
            execution_time_ms=execution_time_ms,
            index_id="",
        )

    async def _parse_intent(self, question: str) -> list[Intent]:
        """Parse question to extract intents and targets."""
        question_lower = question.lower().strip()
        intents = []

        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, question_lower)
                if match:
                    targets = []

                    # Extract targets from match groups
                    for group in match.groups():
                        if group:
                            # Clean up the target
                            target = group.strip()
                            # Remove common words
                            target = re.sub(r"\b(the|a|an|this|that)\b", "", target).strip()
                            if target:
                                targets.append(target)

                    if targets:
                        intents.append(Intent(intent_type, targets, question))
                        break

        # If no pattern matched, try to extract symbol names
        if not intents:
            # Look for identifiers in the question
            identifiers = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", question)
            # Filter out common English words
            common_words = {
                "what",
                "is",
                "how",
                "does",
                "the",
                "a",
                "an",
                "this",
                "that",
                "where",
                "when",
                "why",
                "who",
                "which",
                "can",
                "could",
                "would",
                "should",
                "will",
                "do",
                "did",
                "have",
                "has",
                "had",
                "be",
                "am",
                "are",
                "was",
                "were",
                "been",
                "being",
                "to",
                "of",
                "in",
                "for",
                "on",
                "with",
                "by",
                "from",
                "at",
                "about",
                "into",
                "through",
                "during",
                "before",
                "after",
                "above",
                "below",
                "up",
                "down",
                "out",
                "off",
                "over",
                "under",
                "again",
                "further",
                "then",
                "once",
            }

            targets = [id for id in identifiers if id.lower() not in common_words]

            if targets:
                # Default to definition intent for generic questions
                intents.append(Intent("definition", targets, question))

        return intents

    async def _plan_symbols(
        self, intents: list[Intent], serena_graph: SerenaGraph
    ) -> list[SymbolEntry]:
        """Plan which symbols to start navigation from."""
        seed_symbols = []

        # Build symbol lookup
        symbol_lookup = {}
        for entry in serena_graph.entries:
            if entry.symbol:
                symbol_lookup[entry.symbol.lower()] = entry

        # Find matching symbols for each intent
        for intent in intents:
            for target in intent.targets:
                target_lower = target.lower()

                # Exact match
                if target_lower in symbol_lookup:
                    seed_symbols.append(symbol_lookup[target_lower])
                    continue

                # Partial match
                matches = [
                    entry
                    for symbol, entry in symbol_lookup.items()
                    if target_lower in symbol or symbol in target_lower
                ]

                # Prefer definitions over references
                definitions = [m for m in matches if m.type == SymbolType.DEF]
                if definitions:
                    seed_symbols.extend(definitions[:3])  # Limit matches
                elif matches:
                    seed_symbols.extend(matches[:3])

        return list(set(seed_symbols))  # Remove duplicates

    async def _walk_symbol_graph(
        self, seed_symbols: list[SymbolEntry], serena_graph: SerenaGraph, intents: list[Intent]
    ) -> list[SymbolEntry]:
        """Walk the symbol graph to gather related symbols."""
        visited = set()
        evidence_symbols = []

        # Build relationship maps
        calls_to = {}  # symbol -> list of symbols it calls
        called_by = {}  # symbol -> list of symbols that call it
        references = {}  # symbol -> list of reference locations

        for entry in serena_graph.entries:
            if entry.type == SymbolType.CALL and entry.caller and entry.callee:
                if entry.caller not in calls_to:
                    calls_to[entry.caller] = []
                calls_to[entry.caller].append(entry.callee)

                if entry.callee not in called_by:
                    called_by[entry.callee] = []
                called_by[entry.callee].append(entry.caller)

            elif entry.type == SymbolType.REF and entry.symbol:
                if entry.symbol not in references:
                    references[entry.symbol] = []
                references[entry.symbol].append(entry)

        # Start with seed symbols
        current_level = seed_symbols
        evidence_symbols.extend(current_level)

        for _hop in range(self.max_hops):
            next_level = []

            for symbol_entry in current_level:
                if not symbol_entry.symbol or symbol_entry.symbol in visited:
                    continue

                visited.add(symbol_entry.symbol)

                # Determine which relationships to follow based on intent
                for intent in intents:
                    if intent.intent_type == "usage":
                        # Follow references and calls
                        if symbol_entry.symbol in references:
                            next_level.extend(references[symbol_entry.symbol])
                        if symbol_entry.symbol in called_by:
                            for caller in called_by[symbol_entry.symbol]:
                                caller_entries = [
                                    e
                                    for e in serena_graph.entries
                                    if e.symbol == caller and e.type == SymbolType.DEF
                                ]
                                next_level.extend(caller_entries)

                    elif intent.intent_type == "dependency":
                        # Follow calls and imports
                        if symbol_entry.symbol in calls_to:
                            for callee in calls_to[symbol_entry.symbol]:
                                callee_entries = [
                                    e
                                    for e in serena_graph.entries
                                    if e.symbol == callee and e.type == SymbolType.DEF
                                ]
                                next_level.extend(callee_entries)

                    elif intent.intent_type == "flow":
                        # Follow both directions for flow understanding
                        if symbol_entry.symbol in calls_to:
                            for callee in calls_to[symbol_entry.symbol]:
                                callee_entries = [
                                    e
                                    for e in serena_graph.entries
                                    if e.symbol == callee and e.type == SymbolType.DEF
                                ]
                                next_level.extend(callee_entries)

                        if symbol_entry.symbol in called_by:
                            for caller in called_by[symbol_entry.symbol]:
                                caller_entries = [
                                    e
                                    for e in serena_graph.entries
                                    if e.symbol == caller and e.type == SymbolType.DEF
                                ]
                                next_level.extend(caller_entries)

            # Limit growth and update for next iteration
            if next_level:
                next_level = list(set(next_level))[:10]  # Limit per hop
                evidence_symbols.extend(next_level)
                current_level = next_level
            else:
                break

        return evidence_symbols[: self.max_evidence_items]

    async def _gather_evidence(
        self, evidence_symbols: list[SymbolEntry], context_lines: int
    ) -> list[CodeSnippet]:
        """Gather code snippets for evidence symbols."""
        evidence_snippets = []

        for entry in evidence_symbols:
            # Create placeholder snippet (would extract from actual file)
            snippet = CodeSnippet(
                path=entry.path,
                span=entry.span,
                hash="",  # Would compute actual hash
                pre="",  # Would extract actual context
                text=entry.symbol or f"Symbol at {entry.path}:{entry.span[0]}-{entry.span[1]}",
                post="",
                line_start=1,
                line_end=1,
            )

            evidence_snippets.append(snippet)

        return evidence_snippets

    async def _synthesize_answer(
        self, question: str, intents: list[Intent], evidence_snippets: list[CodeSnippet]
    ) -> str:
        """Synthesize answer from collected evidence."""
        if not evidence_snippets:
            return "I couldn't find enough information to answer your question."

        answer_parts = []

        # Analyze the primary intent
        primary_intent = intents[0] if intents else None

        if primary_intent:
            if primary_intent.intent_type == "definition":
                answer_parts.append(
                    f"Based on the code analysis, here's what I found about {', '.join(primary_intent.targets)}:"
                )

                # Group evidence by file
                file_groups = {}
                for snippet in evidence_snippets:
                    if snippet.path not in file_groups:
                        file_groups[snippet.path] = []
                    file_groups[snippet.path].append(snippet)

                for file_path, snippets in file_groups.items():
                    answer_parts.append(f"\nIn `{file_path}`:")
                    for snippet in snippets[:3]:  # Limit per file
                        answer_parts.append(f"  - {snippet.text}")

            elif primary_intent.intent_type == "usage":
                answer_parts.append(
                    f"Here are the usage patterns I found for {', '.join(primary_intent.targets)}:"
                )

                usage_count = len(evidence_snippets)
                answer_parts.append(f"\nFound {usage_count} usage locations:")

                for snippet in evidence_snippets[:5]:  # Show top 5 usages
                    answer_parts.append(f"  - `{snippet.path}` at line {snippet.line_start}")

            elif primary_intent.intent_type == "dependency":
                answer_parts.append(
                    f"Here are the dependencies I found for {', '.join(primary_intent.targets)}:"
                )

                # Group by relationship type
                deps = set()
                for snippet in evidence_snippets:
                    deps.add(snippet.path)

                answer_parts.append(f"\nDependent files ({len(deps)}):")
                for dep in sorted(deps)[:10]:
                    answer_parts.append(f"  - `{dep}`")

            elif primary_intent.intent_type == "flow":
                answer_parts.append(
                    f"Here's the execution flow for {', '.join(primary_intent.targets)}:"
                )

                # Try to show flow sequence
                answer_parts.append(f"\nFlow involves {len(evidence_snippets)} components:")
                for i, snippet in enumerate(evidence_snippets[:7]):
                    answer_parts.append(f"  {i+1}. `{snippet.path}` - {snippet.text}")

            else:
                answer_parts.append("Based on the code analysis:")
                for snippet in evidence_snippets[:5]:
                    answer_parts.append(f"  - `{snippet.path}`: {snippet.text}")

        # Add summary
        if len(evidence_snippets) > 5:
            answer_parts.append(
                f"\n... and {len(evidence_snippets) - 5} more locations (see citations for complete list)"
            )

        return "\n".join(answer_parts)

    async def find_symbol_definition(
        self, symbol_name: str, serena_graph: SerenaGraph
    ) -> SymbolEntry | None:
        """Find the definition of a specific symbol using optimized index."""
        # Build index if needed
        self._build_symbol_index(serena_graph)

        # Fast lookup using index
        symbol_lower = symbol_name.lower()
        if symbol_lower in self._symbol_index:
            for entry in self._symbol_index[symbol_lower]:
                if entry.type == SymbolType.DEF:
                    return entry
        return None

    async def find_symbol_references(
        self, symbol_name: str, serena_graph: SerenaGraph
    ) -> list[SymbolEntry]:
        """Find all references to a specific symbol using optimized index."""
        # Build index if needed
        self._build_symbol_index(serena_graph)

        # Fast lookup using index
        symbol_lower = symbol_name.lower()
        references = []
        if symbol_lower in self._symbol_index:
            for entry in self._symbol_index[symbol_lower]:
                if entry.type == SymbolType.REF:
                    references.append(entry)
        return references
