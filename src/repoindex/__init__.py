"""
Mimir Deep Code Research System

A full-Python MCP server for intelligent repository indexing and search.
Provides zero-prompt local indexing with defensible citations.
"""

__version__ = "0.1.0"
__author__ = "Mimir Team"
__description__ = "Deep Code Research System - MCP server for intelligent repository indexing"

from .mcp.server import MCPServer
from .pipeline.run import IndexingPipeline

__all__ = ["IndexingPipeline", "MCPServer"]
