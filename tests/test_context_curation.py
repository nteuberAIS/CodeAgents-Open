"""Tests for BaseAgent.curate_context() and SprintPlannerAgent curation overrides."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from agents.base import BaseAgent
from agents.sprint_planner import SprintPlannerAgent


# -- Helpers --


class _StubAgent(BaseAgent):
    """Concrete agent for testing BaseAgent classmethods."""

    name = "stub"

    def run(self, user_input: str) -> dict:
        return {}


class _SmallAgent(BaseAgent):
    """Agent with restrictive curation limits for testing."""

    name = "small"
    MAX_CONTENT_ITEMS = 2
    MAX_CONTENT_CHARS = 500
    CONTENT_STATUSES = ["Ready", "In Progress"]

    def run(self, user_input: str) -> dict:
        return {}


def _make_context(work_items=None, sprints=None):
    """Build a minimal context dict."""
    return {
        "work_items": work_items or [],
        "sprints": sprints or [],
        "docs": [],
        "decisions": [],
        "risks": [],
    }


def _make_work_item(notion_id, name, status="Ready", has_content=True, sprint_id=None):
    return {
        "notion_id": notion_id,
        "name": name,
        "status": status,
        "has_content": has_content,
        "sprint_id": sprint_id,
        "type": "Task",
        "priority": "P1",
    }


def _write_content(content_dir, notion_id, text):
    """Write a markdown content file for a given notion_id."""
    (content_dir / f"{notion_id}.md").write_text(text, encoding="utf-8")


# -- Tests --


class TestCurateContextBasics:
    """Basic curate_context behavior."""

    def test_none_input_returns_none(self):
        assert _StubAgent.curate_context(None) is None

    def test_none_input_with_content_dir_returns_none(self, tmp_path):
        assert _StubAgent.curate_context(None, content_dir=tmp_path) is None

    def test_no_content_dir_returns_empty_page_content(self):
        ctx = _make_context()
        result = _StubAgent.curate_context(ctx)
        assert result is not None
        assert result["page_content"] == {}

    def test_nonexistent_content_dir_returns_empty_page_content(self, tmp_path):
        ctx = _make_context()
        result = _StubAgent.curate_context(ctx, content_dir=tmp_path / "nope")
        assert result["page_content"] == {}

    def test_does_not_mutate_original_context(self, tmp_path):
        ctx = _make_context()
        original_keys = set(ctx.keys())
        _StubAgent.curate_context(ctx, content_dir=tmp_path)
        assert set(ctx.keys()) == original_keys  # no "page_content" added to original


class TestCurateContextContentLoading:
    """Content loading from disk."""

    def test_loads_content_for_entities_with_has_content(self, tmp_path):
        items = [_make_work_item("wi-001", "Task A", has_content=True)]
        ctx = _make_context(work_items=items)
        _write_content(tmp_path, "wi-001", "# Task A\nSome content.")

        result = _StubAgent.curate_context(ctx, content_dir=tmp_path)

        assert "wi-001" in result["page_content"]
        assert "Some content." in result["page_content"]["wi-001"]

    def test_skips_entities_without_has_content(self, tmp_path):
        items = [_make_work_item("wi-001", "Task A", has_content=False)]
        ctx = _make_context(work_items=items)
        _write_content(tmp_path, "wi-001", "# Content")

        result = _StubAgent.curate_context(ctx, content_dir=tmp_path)

        assert result["page_content"] == {}

    def test_skips_missing_content_files(self, tmp_path):
        items = [_make_work_item("wi-001", "Task A", has_content=True)]
        ctx = _make_context(work_items=items)
        # Don't create the file

        result = _StubAgent.curate_context(ctx, content_dir=tmp_path)

        assert result["page_content"] == {}

    def test_loads_across_entity_types(self, tmp_path):
        ctx = _make_context(
            work_items=[_make_work_item("wi-001", "Task A")],
        )
        ctx["docs"] = [{"notion_id": "doc-001", "name": "Doc A", "status": "Ready", "has_content": True}]
        _write_content(tmp_path, "wi-001", "WI content")
        _write_content(tmp_path, "doc-001", "Doc content")

        result = _StubAgent.curate_context(ctx, content_dir=tmp_path)

        assert "wi-001" in result["page_content"]
        assert "doc-001" in result["page_content"]


class TestCurateContextFiltering:
    """Status filtering and limits."""

    def test_filters_by_content_statuses(self, tmp_path):
        items = [
            _make_work_item("wi-001", "Ready item", status="Ready", has_content=True),
            _make_work_item("wi-002", "Done item", status="Done", has_content=True),
        ]
        ctx = _make_context(work_items=items)
        _write_content(tmp_path, "wi-001", "Ready content")
        _write_content(tmp_path, "wi-002", "Done content")

        result = _SmallAgent.curate_context(ctx, content_dir=tmp_path)

        assert "wi-001" in result["page_content"]
        assert "wi-002" not in result["page_content"]  # "Done" not in CONTENT_STATUSES

    def test_respects_max_content_items(self, tmp_path):
        items = [
            _make_work_item(f"wi-{i:03d}", f"Task {i}", has_content=True)
            for i in range(5)
        ]
        ctx = _make_context(work_items=items)
        for i in range(5):
            _write_content(tmp_path, f"wi-{i:03d}", f"Content {i}")

        result = _SmallAgent.curate_context(ctx, content_dir=tmp_path)

        assert len(result["page_content"]) == 2  # MAX_CONTENT_ITEMS = 2

    def test_respects_max_content_chars(self, tmp_path):
        items = [
            _make_work_item("wi-001", "Task A", has_content=True),
            _make_work_item("wi-002", "Task B", has_content=True),
        ]
        ctx = _make_context(work_items=items)
        _write_content(tmp_path, "wi-001", "A" * 400)
        _write_content(tmp_path, "wi-002", "B" * 400)

        result = _SmallAgent.curate_context(ctx, content_dir=tmp_path)

        total = sum(len(v) for v in result["page_content"].values())
        assert total <= 500  # MAX_CONTENT_CHARS

    def test_truncates_content_with_marker(self, tmp_path):
        items = [_make_work_item("wi-001", "Task A", has_content=True)]
        ctx = _make_context(work_items=items)
        _write_content(tmp_path, "wi-001", "X" * 1000)

        result = _SmallAgent.curate_context(ctx, content_dir=tmp_path)

        content = result["page_content"]["wi-001"]
        assert content.endswith("... (truncated)")
        assert len(content) <= 520  # 500 + truncation marker

    def test_sorts_by_status_priority(self, tmp_path):
        items = [
            _make_work_item("wi-backlog", "Backlog item", status="Backlog", has_content=True),
            _make_work_item("wi-ready", "Ready item", status="Ready", has_content=True),
            _make_work_item("wi-inprog", "In Progress item", status="In Progress", has_content=True),
        ]
        ctx = _make_context(work_items=items)
        for item in items:
            _write_content(tmp_path, item["notion_id"], f"Content for {item['name']}")

        # _SmallAgent has CONTENT_STATUSES = ["Ready", "In Progress"]
        # MAX_CONTENT_ITEMS = 2, so only top 2 by priority get loaded
        result = _SmallAgent.curate_context(ctx, content_dir=tmp_path)

        assert "wi-ready" in result["page_content"]
        assert "wi-inprog" in result["page_content"]
        assert "wi-backlog" not in result["page_content"]  # "Backlog" not in statuses


class TestSprintPlannerCurationOverrides:
    """Verify SprintPlannerAgent class attribute values."""

    def test_max_content_items(self):
        assert SprintPlannerAgent.MAX_CONTENT_ITEMS == 5

    def test_max_content_chars(self):
        assert SprintPlannerAgent.MAX_CONTENT_CHARS == 6000

    def test_content_statuses(self):
        assert "Ready" in SprintPlannerAgent.CONTENT_STATUSES
        assert "In Progress" in SprintPlannerAgent.CONTENT_STATUSES
        assert "Backlog" in SprintPlannerAgent.CONTENT_STATUSES
