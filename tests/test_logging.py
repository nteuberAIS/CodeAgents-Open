"""Tests for config/logging_config.py — JSON formatter, setup, cascade handler."""

from __future__ import annotations

import json
import logging
import sys

import pytest

from config.logging_config import (
    JSONLineFormatter,
    add_cascade_handler,
    setup_logging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    message: str = "test message",
    level: int = logging.INFO,
    name: str = "test.logger",
    extra: dict | None = None,
    exc_info: tuple | None = None,
) -> logging.LogRecord:
    """Create a LogRecord for testing."""
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname="test.py",
        lineno=1,
        msg=message,
        args=(),
        exc_info=exc_info,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Ensure root logger is clean before/after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    # Restore
    for h in root.handlers[:]:
        root.removeHandler(h)
        if hasattr(h, "close"):
            h.close()
    for h in original_handlers:
        root.addHandler(h)
    root.setLevel(original_level)


# ---------------------------------------------------------------------------
# JSONLineFormatter
# ---------------------------------------------------------------------------

class TestJSONLineFormatter:
    def test_basic_fields(self):
        fmt = JSONLineFormatter()
        record = _make_record("hello world", level=logging.WARNING, name="my.mod")
        line = fmt.format(record)
        data = json.loads(line)

        assert data["message"] == "hello world"
        assert data["level"] == "WARNING"
        assert data["logger"] == "my.mod"
        assert "timestamp" in data

    def test_extra_fields_included(self):
        fmt = JSONLineFormatter()
        record = _make_record(
            "task done",
            extra={"sprint_id": "s-8", "task_count": 5},
        )
        line = fmt.format(record)
        data = json.loads(line)

        assert data["sprint_id"] == "s-8"
        assert data["task_count"] == 5

    def test_default_record_keys_excluded(self):
        fmt = JSONLineFormatter()
        record = _make_record("msg")
        line = fmt.format(record)
        data = json.loads(line)

        # Standard LogRecord attrs should NOT leak into JSON
        for key in ("pathname", "lineno", "funcName", "args", "levelno"):
            assert key not in data

    def test_exception_traceback(self):
        fmt = JSONLineFormatter()
        try:
            raise ValueError("boom")
        except ValueError:
            exc_info = sys.exc_info()

        record = _make_record("error occurred", level=logging.ERROR, exc_info=exc_info)
        line = fmt.format(record)
        data = json.loads(line)

        assert "traceback" in data
        assert "ValueError: boom" in data["traceback"]

    def test_non_serializable_extra(self):
        """Non-JSON-serializable extras are converted to str."""
        fmt = JSONLineFormatter()
        record = _make_record("msg", extra={"obj": object()})
        line = fmt.format(record)
        data = json.loads(line)

        assert "obj" in data
        assert isinstance(data["obj"], str)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------

class TestSetupLogging:
    def test_console_only(self):
        setup_logging("DEBUG", log_file=None, json_output=False)
        root = logging.getLogger()
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0], logging.StreamHandler)

    def test_console_and_file(self, tmp_path):
        log_file = tmp_path / "test.jsonl"
        setup_logging("INFO", log_file=str(log_file), json_output=True)
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_creates_directory(self, tmp_path):
        log_file = tmp_path / "subdir" / "nested" / "test.jsonl"
        setup_logging("INFO", log_file=str(log_file), json_output=True)
        assert log_file.parent.exists()

    def test_sets_level(self):
        setup_logging("DEBUG", log_file=None, json_output=False)
        assert logging.getLogger().level == logging.DEBUG

        setup_logging("ERROR", log_file=None, json_output=False)
        assert logging.getLogger().level == logging.ERROR

    def test_idempotent(self, tmp_path):
        """Calling setup_logging twice doesn't duplicate handlers."""
        log_file = tmp_path / "test.jsonl"
        setup_logging("INFO", log_file=str(log_file), json_output=True)
        setup_logging("INFO", log_file=str(log_file), json_output=True)
        assert len(logging.getLogger().handlers) == 2

    def test_quiets_third_party(self):
        setup_logging("DEBUG", log_file=None, json_output=False)
        assert logging.getLogger("httpx").level >= logging.WARNING
        assert logging.getLogger("chromadb").level >= logging.WARNING

    def test_file_handler_writes_json(self, tmp_path):
        log_file = tmp_path / "out.jsonl"
        setup_logging("DEBUG", log_file=str(log_file), json_output=True)

        test_logger = logging.getLogger("test.write")
        test_logger.info("hello", extra={"key": "value"})

        # Flush handlers
        for h in logging.getLogger().handlers:
            h.flush()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        data = json.loads(lines[-1])
        assert data["message"] == "hello"
        assert data["key"] == "value"


# ---------------------------------------------------------------------------
# add_cascade_handler
# ---------------------------------------------------------------------------

class TestAddCascadeHandler:
    def test_adds_and_removes_handler(self, tmp_path, monkeypatch):
        # Point log path to tmp_path
        monkeypatch.chdir(tmp_path)
        (tmp_path / "data" / "logs").mkdir(parents=True)

        handler = add_cascade_handler("sprint-8")
        root = logging.getLogger()
        assert handler in root.handlers

        root.removeHandler(handler)
        handler.close()
        assert handler not in root.handlers

    def test_writes_valid_jsonl(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        logging.getLogger().setLevel(logging.DEBUG)

        handler = add_cascade_handler("s-1")
        test_logger = logging.getLogger("test.cascade")
        test_logger.info("task done", extra={"task_id": "t-1"})
        handler.flush()

        log_file = tmp_path / "data" / "logs" / "cascade-s-1.jsonl"
        assert log_file.exists()

        lines = log_file.read_text(encoding="utf-8").strip().splitlines()
        data = json.loads(lines[0])
        assert data["task_id"] == "t-1"

        logging.getLogger().removeHandler(handler)
        handler.close()
