"""
Logging utilities for structured event tracking.

Provides structured logging with JSON event streams and human-readable
markdown logs for pipeline progress monitoring.
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ..data.schemas import PipelineStage, IndexState


class JSONFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add extra fields if present
        if hasattr(record, 'index_id'):
            log_entry['index_id'] = record.index_id
        if hasattr(record, 'stage'):
            log_entry['stage'] = record.stage
        if hasattr(record, 'progress'):
            log_entry['progress'] = record.progress
        if hasattr(record, 'duration_ms'):
            log_entry['duration_ms'] = record.duration_ms
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)


class MarkdownFormatter(logging.Formatter):
    """Formatter that outputs human-readable markdown logs."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as markdown."""
        timestamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        level_emoji = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }.get(record.levelname, "📝")
        
        # Base message
        message = f"{level_emoji} **{timestamp}** {record.getMessage()}"
        
        # Add stage and progress if available
        if hasattr(record, 'stage') and hasattr(record, 'progress'):
            stage_emoji = {
                PipelineStage.ACQUIRE: "📂",
                PipelineStage.REPOMAPPER: "🗺️",
                PipelineStage.SERENA: "🔗",
                PipelineStage.LEANN: "🧠",
                PipelineStage.SNIPPETS: "✂️",
                PipelineStage.BUNDLE: "📦"
            }.get(record.stage, "⚙️")
            
            message += f" {stage_emoji} **{record.stage.upper()}** ({record.progress}%)"
        
        # Add duration if available
        if hasattr(record, 'duration_ms'):
            message += f" ⏱️ {record.duration_ms:.1f}ms"
        
        # Add exception if present
        if record.exc_info:
            exception_text = self.formatException(record.exc_info)
            message += f"\n\n```\n{exception_text}\n```"
        
        return message


class PipelineLogger:
    """
    Specialized logger for pipeline operations.
    
    Maintains both structured JSON event logs and human-readable markdown logs.
    """
    
    def __init__(self, index_id: str, log_dir: Path):
        """Initialize pipeline logger."""
        self.index_id = index_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up loggers
        self.json_logger = logging.getLogger(f"mimir.pipeline.{index_id}.json")
        self.md_logger = logging.getLogger(f"mimir.pipeline.{index_id}.md")
        
        # Configure JSON logger
        json_handler = logging.FileHandler(self.log_dir / "events.jsonl")
        json_handler.setFormatter(JSONFormatter())
        self.json_logger.addHandler(json_handler)
        self.json_logger.setLevel(logging.DEBUG)
        
        # Configure Markdown logger
        md_handler = logging.FileHandler(self.log_dir / "log.md")
        md_handler.setFormatter(MarkdownFormatter())
        self.md_logger.addHandler(md_handler)
        self.md_logger.setLevel(logging.INFO)
        
        # Initialize markdown log
        self._write_header()
    
    def _write_header(self) -> None:
        """Write markdown log header."""
        header = f"""# Pipeline Log - {self.index_id}

Started: {datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}

---

"""
        with open(self.log_dir / "log.md", "w") as f:
            f.write(header)
    
    def log_stage_start(
        self,
        stage: PipelineStage,
        message: str = "",
        **kwargs: Any
    ) -> None:
        """Log the start of a pipeline stage."""
        extra = {
            "index_id": self.index_id,
            "stage": stage,
            "progress": 0,
            **kwargs
        }
        
        log_message = message or f"Starting {stage.value} stage"
        
        self.json_logger.info(log_message, extra=extra)
        self.md_logger.info(log_message, extra=extra)
    
    def log_stage_progress(
        self,
        stage: PipelineStage,
        progress: int,
        message: str = "",
        **kwargs: Any
    ) -> None:
        """Log progress within a pipeline stage."""
        extra = {
            "index_id": self.index_id,
            "stage": stage,
            "progress": progress,
            **kwargs
        }
        
        log_message = message or f"{stage.value} progress: {progress}%"
        
        self.json_logger.info(log_message, extra=extra)
        if progress % 10 == 0:  # Only log every 10% for markdown
            self.md_logger.info(log_message, extra=extra)
    
    def log_stage_complete(
        self,
        stage: PipelineStage,
        duration_ms: float,
        message: str = "",
        **kwargs: Any
    ) -> None:
        """Log completion of a pipeline stage."""
        extra = {
            "index_id": self.index_id,
            "stage": stage,
            "progress": 100,
            "duration_ms": duration_ms,
            **kwargs
        }
        
        log_message = message or f"Completed {stage.value} stage"
        
        self.json_logger.info(log_message, extra=extra)
        self.md_logger.info(log_message, extra=extra)
    
    def log_stage_error(
        self,
        stage: PipelineStage,
        error: Exception,
        message: str = "",
        **kwargs: Any
    ) -> None:
        """Log an error during a pipeline stage."""
        extra = {
            "index_id": self.index_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "error_message": str(error),
            **kwargs
        }
        
        log_message = message or f"Error in {stage.value} stage: {error}"
        
        self.json_logger.error(log_message, extra=extra, exc_info=True)
        self.md_logger.error(log_message, extra=extra, exc_info=True)
    
    def log_info(
        self,
        message: str,
        stage: Optional[PipelineStage] = None,
        **kwargs: Any
    ) -> None:
        """Log general information."""
        extra = {"index_id": self.index_id, **kwargs}
        if stage:
            extra["stage"] = stage
        
        self.json_logger.info(message, extra=extra)
        self.md_logger.info(message, extra=extra)
    
    def log_warning(
        self,
        message: str,
        stage: Optional[PipelineStage] = None,
        **kwargs: Any
    ) -> None:
        """Log a warning."""
        extra = {"index_id": self.index_id, **kwargs}
        if stage:
            extra["stage"] = stage
        
        self.json_logger.warning(message, extra=extra)
        self.md_logger.warning(message, extra=extra)
    
    def log_error(
        self,
        message: str,
        error: Optional[Exception] = None,
        stage: Optional[PipelineStage] = None,
        **kwargs: Any
    ) -> None:
        """Log an error."""
        extra = {"index_id": self.index_id, **kwargs}
        if stage:
            extra["stage"] = stage
        
        self.json_logger.error(message, extra=extra, exc_info=error)
        self.md_logger.error(message, extra=extra, exc_info=error)
    
    def close(self) -> None:
        """Close log handlers."""
        for handler in self.json_logger.handlers:
            handler.close()
            self.json_logger.removeHandler(handler)
        
        for handler in self.md_logger.handlers:
            handler.close()
            self.md_logger.removeHandler(handler)


def setup_logging(
    level: str = "INFO",
    format_type: str = "standard"
) -> None:
    """Set up global logging configuration."""
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stderr)
    
    if format_type == "json":
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
    
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_pipeline_logger(index_id: str, log_dir: Union[str, Path]) -> PipelineLogger:
    """Get a pipeline logger instance."""
    return PipelineLogger(index_id, Path(log_dir))


def get_logger(name: str) -> logging.Logger:
    """Get a standard logger instance for the given name."""
    return logging.getLogger(name)


class LogContext:
    """Context manager for adding structured context to logs."""
    
    def __init__(self, logger: logging.Logger, **context: Any):
        self.logger = logger
        self.context = context
        self.old_filters = []
    
    def __enter__(self):
        # Add context filter
        def add_context(record):
            for key, value in self.context.items():
                setattr(record, key, value)
            return True
        
        self.logger.addFilter(add_context)
        self.old_filters.append(add_context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove context filters
        for filter_func in self.old_filters:
            self.logger.removeFilter(filter_func)