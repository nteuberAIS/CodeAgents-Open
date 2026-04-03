"""Logging configuration for the agent system.

Provides dual-output logging: human-readable console + structured JSON log file.
Call setup_logging() once at application startup (from main.py).
"""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Build the set of default LogRecord attribute names so we can extract
# user-supplied extras by exclusion.
_DUMMY = logging.LogRecord("", 0, "", 0, "", (), None)
_DEFAULT_RECORD_KEYS = frozenset(_DUMMY.__dict__.keys()) | {
    "message",
    "asctime",
    "stack_info",
    "exc_info",
    "exc_text",
    "taskName",
}


class JSONLineFormatter(logging.Formatter):
    """Format log records as single-line JSON objects (JSON Lines / .jsonl)."""

    def format(self, record: logging.LogRecord) -> str:
        # Let the base class populate record.message
        record.getMessage()

        entry: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Extract user-supplied extra fields
        for key, value in record.__dict__.items():
            if key not in _DEFAULT_RECORD_KEYS:
                try:
                    json.dumps(value)  # ensure serializable
                    entry[key] = value
                except (TypeError, ValueError):
                    entry[key] = str(value)

        if record.exc_info and record.exc_info[0] is not None:
            entry["traceback"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(entry, default=str)


def setup_logging(
    level: str = "INFO",
    log_file: str | None = "data/logs/codeagents.jsonl",
    json_output: bool = True,
) -> None:
    """Configure root logger with console and optional JSON file handlers.

    Args:
        level: Log level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Path to JSON Lines log file. None disables file logging.
        json_output: Whether to write the JSON file log (requires log_file).
    """
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers to avoid duplication on repeated calls
    root.handlers.clear()

    # -- Console handler (human-readable to stderr) --
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(console)

    # -- File handler (JSON Lines) --
    if log_file and json_output:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(JSONLineFormatter())
        root.addHandler(file_handler)

    # Quiet noisy third-party loggers
    for name in ("httpx", "chromadb", "urllib3", "langchain", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def add_cascade_handler(sprint_id: str) -> logging.FileHandler:
    """Add a per-sprint JSON Lines log file handler.

    Returns the handler so the caller can remove it when the cascade ends:
        handler = add_cascade_handler("sprint-8")
        try:
            ...
        finally:
            logging.getLogger().removeHandler(handler)
            handler.close()
    """
    log_path = Path(f"data/logs/cascade-{sprint_id}.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(JSONLineFormatter())
    logging.getLogger().addHandler(handler)
    return handler
