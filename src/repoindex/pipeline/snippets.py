"""
Code snippet extraction for search results.

Extracts stable code snippets with context lines based on symbol graph
locations, providing consistent byte spans and content hashes.
"""

import asyncio
import hashlib
from pathlib import Path
from typing import Callable, List, Optional

from ..data.schemas import SerenaGraph, SymbolEntry, CodeSnippet, SnippetCollection
from ..util.fs import extract_file_span


class SnippetExtractor:
    """
    Extracts code snippets with context from symbol graph locations.
    
    Provides stable byte-span extraction for search result presentation
    with configurable context lines and content verification.
    """
    
    def __init__(self):
        """Initialize snippet extractor."""
        pass
    
    async def extract_snippets(
        self,
        repo_root: Path,
        serena_graph: SerenaGraph,
        work_dir: Path,
        context_lines: int = 5,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> SnippetCollection:
        """
        Extract all snippets from Serena graph symbol locations.
        
        Returns collection of code snippets with context.
        """
        if progress_callback:
            progress_callback(10)
        
        if not serena_graph or not serena_graph.entries:
            return SnippetCollection(snippets=[], total_count=0)
        
        # Group symbol entries by file for efficient processing
        file_entries = {}
        for entry in serena_graph.entries:
            if entry.path not in file_entries:
                file_entries[entry.path] = []
            file_entries[entry.path].append(entry)
        
        if progress_callback:
            progress_callback(20)
        
        # Extract snippets for each file
        all_snippets = []
        processed_files = 0
        
        for file_path, entries in file_entries.items():
            try:
                file_snippets = await self._extract_file_snippets(
                    repo_root, file_path, entries, context_lines
                )
                all_snippets.extend(file_snippets)
                
                processed_files += 1
                if progress_callback:
                    progress = 20 + int((processed_files / len(file_entries)) * 70)
                    progress_callback(progress)
                    
            except Exception as e:
                print(f"Warning: Failed to extract snippets from {file_path}: {e}")
                continue
        
        # Create snippet collection
        snippet_collection = SnippetCollection(
            snippets=all_snippets,
            total_count=len(all_snippets)
        )
        
        # Save snippets to work directory
        snippet_collection.save_to_jsonl(work_dir / "snippets.jsonl")
        
        if progress_callback:
            progress_callback(100)
        
        return snippet_collection
    
    async def _extract_file_snippets(
        self,
        repo_root: Path,
        file_path: str,
        entries: List[SymbolEntry],
        context_lines: int
    ) -> List[CodeSnippet]:
        """Extract snippets from a single file."""
        full_path = repo_root / file_path
        
        if not full_path.exists():
            return []
        
        try:
            # Read file content
            content = await asyncio.to_thread(
                full_path.read_text, encoding='utf-8', errors='ignore'
            )
        except Exception:
            return []
        
        # Sort entries by span to process in order
        sorted_entries = sorted(entries, key=lambda e: e.span[0])
        
        snippets = []
        processed_spans = set()
        
        for entry in sorted_entries:
            start_byte, end_byte = entry.span
            
            # Skip if we've already processed this exact span
            span_key = (start_byte, end_byte)
            if span_key in processed_spans:
                continue
            processed_spans.add(span_key)
            
            # Extract snippet with context
            try:
                snippet = await self._extract_single_snippet(
                    content, file_path, start_byte, end_byte, context_lines
                )
                if snippet:
                    snippets.append(snippet)
            except Exception:
                continue
        
        return snippets
    
    async def _extract_single_snippet(
        self,
        content: str,
        file_path: str,
        start_byte: int,
        end_byte: int,
        context_lines: int
    ) -> Optional[CodeSnippet]:
        """Extract a single code snippet with context."""
        # Validate span
        if start_byte < 0 or end_byte > len(content) or start_byte >= end_byte:
            return None
        
        # Find line boundaries
        lines = content.split('\n')
        
        # Find which lines contain the span
        current_byte = 0
        start_line = 0
        end_line = 0
        
        for i, line in enumerate(lines):
            line_end = current_byte + len(line) + 1  # +1 for newline
            
            if current_byte <= start_byte < line_end:
                start_line = i
            
            if current_byte <= end_byte <= line_end:
                end_line = i
                break
            
            current_byte = line_end
        
        # Calculate context boundaries
        context_start = max(0, start_line - context_lines)
        context_end = min(len(lines), end_line + context_lines + 1)
        
        # Extract content sections
        pre_context = '\n'.join(lines[context_start:start_line])
        main_text = content[start_byte:end_byte]
        post_context = '\n'.join(lines[end_line + 1:context_end])
        
        # Compute content hash for verification
        content_hash = hashlib.sha256(main_text.encode('utf-8')).hexdigest()
        
        return CodeSnippet(
            path=file_path,
            span=(start_byte, end_byte),
            hash=content_hash,
            pre=pre_context,
            text=main_text,
            post=post_context,
            line_start=start_line + 1,  # 1-indexed for display
            line_end=end_line + 1
        )
    
    async def get_snippet_by_location(
        self,
        repo_root: Path,
        file_path: str,
        start_byte: int,
        end_byte: int,
        context_lines: int = 5
    ) -> Optional[CodeSnippet]:
        """
        Get a single snippet by file location.
        
        Useful for on-demand snippet extraction.
        """
        full_path = repo_root / file_path
        
        if not full_path.exists():
            return None
        
        try:
            content = await asyncio.to_thread(
                full_path.read_text, encoding='utf-8', errors='ignore'
            )
            
            return await self._extract_single_snippet(
                content, file_path, start_byte, end_byte, context_lines
            )
        except Exception:
            return None
    
    async def verify_snippet_integrity(
        self,
        repo_root: Path,
        snippet: CodeSnippet
    ) -> bool:
        """
        Verify snippet integrity against current file content.
        
        Checks if the content hash still matches the file content.
        """
        full_path = repo_root / snippet.path
        
        if not full_path.exists():
            return False
        
        try:
            content = await asyncio.to_thread(
                full_path.read_text, encoding='utf-8', errors='ignore'
            )
            
            # Extract current content at span
            start_byte, end_byte = snippet.span
            if start_byte < 0 or end_byte > len(content):
                return False
            
            current_text = content[start_byte:end_byte]
            current_hash = hashlib.sha256(current_text.encode('utf-8')).hexdigest()
            
            return current_hash == snippet.hash
            
        except Exception:
            return False
    
    async def update_snippet_content(
        self,
        repo_root: Path,
        snippet: CodeSnippet,
        context_lines: int = 5
    ) -> Optional[CodeSnippet]:
        """
        Update snippet with current file content.
        
        Returns updated snippet or None if location is invalid.
        """
        return await self.get_snippet_by_location(
            repo_root, snippet.path, snippet.span[0], snippet.span[1], context_lines
        )
    
    async def merge_overlapping_snippets(
        self,
        snippets: List[CodeSnippet],
        max_gap: int = 100
    ) -> List[CodeSnippet]:
        """
        Merge overlapping or nearby snippets to reduce redundancy.
        
        Useful for cleaning up snippet collections.
        """
        if not snippets:
            return []
        
        # Group by file
        file_snippets = {}
        for snippet in snippets:
            if snippet.path not in file_snippets:
                file_snippets[snippet.path] = []
            file_snippets[snippet.path].append(snippet)
        
        merged_snippets = []
        
        for file_path, file_snippet_list in file_snippets.items():
            # Sort by start position
            file_snippet_list.sort(key=lambda s: s.span[0])
            
            merged_file_snippets = []
            current_snippet = file_snippet_list[0]
            
            for next_snippet in file_snippet_list[1:]:
                # Check if snippets should be merged
                if (next_snippet.span[0] - current_snippet.span[1]) <= max_gap:
                    # Merge snippets
                    merged_span = (current_snippet.span[0], next_snippet.span[1])
                    merged_text = current_snippet.text + next_snippet.text
                    merged_hash = hashlib.sha256(merged_text.encode('utf-8')).hexdigest()
                    
                    current_snippet = CodeSnippet(
                        path=file_path,
                        span=merged_span,
                        hash=merged_hash,
                        pre=current_snippet.pre,
                        text=merged_text,
                        post=next_snippet.post,
                        line_start=current_snippet.line_start,
                        line_end=next_snippet.line_end
                    )
                else:
                    # Add current snippet and start new one
                    merged_file_snippets.append(current_snippet)
                    current_snippet = next_snippet
            
            # Add final snippet
            merged_file_snippets.append(current_snippet)
            merged_snippets.extend(merged_file_snippets)
        
        return merged_snippets