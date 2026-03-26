"""Tests for main.py — CLI tool binding and context loading."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from main import _get_agent_tools, _load_notion_context, _snapshot_to_context


# -- Helpers --


def _make_settings():
    """Create a minimal Settings-like object."""
    settings = MagicMock()
    settings.data_dir = "data"
    settings.notion_api_key = "test-key"
    return settings


def _make_snapshot():
    """Create a mock NotionSnapshot with model_dump() support."""
    snapshot = MagicMock()

    def _make_items(n, prefix):
        items = []
        for i in range(n):
            item = MagicMock()
            item.model_dump.return_value = {"name": f"{prefix}-{i}"}
            items.append(item)
        return items

    snapshot.work_items = _make_items(3, "wi")
    snapshot.sprints = _make_items(1, "sprint")
    snapshot.docs = _make_items(2, "doc")
    snapshot.decisions = _make_items(1, "decision")
    snapshot.risks = _make_items(0, "risk")
    return snapshot


# -- Tests for _get_agent_tools --


class TestGetAgentTools:
    """Tests for _get_agent_tools()."""

    def test_combines_required_and_optional(self):
        agent = MagicMock()
        agent.REQUIRED_TOOLS = ["notion_write"]
        agent.OPTIONAL_TOOLS = ["github", "azdevops"]

        result = _get_agent_tools(agent)

        assert result == ["notion_write", "github", "azdevops"]

    def test_empty_list_for_agent_without_declarations(self):
        agent = MagicMock(spec=[])  # No REQUIRED_TOOLS or OPTIONAL_TOOLS attrs

        result = _get_agent_tools(agent)

        assert result == []

    def test_only_required_tools(self):
        agent = MagicMock()
        agent.REQUIRED_TOOLS = ["notion_write"]
        agent.OPTIONAL_TOOLS = []

        result = _get_agent_tools(agent)

        assert result == ["notion_write"]

    def test_only_optional_tools(self):
        agent = MagicMock()
        agent.REQUIRED_TOOLS = []
        agent.OPTIONAL_TOOLS = ["github"]

        result = _get_agent_tools(agent)

        assert result == ["github"]


# -- Tests for _snapshot_to_context --


class TestSnapshotToContext:
    """Tests for _snapshot_to_context()."""

    def test_converts_snapshot_correctly(self):
        snapshot = _make_snapshot()

        context = _snapshot_to_context(snapshot)

        assert len(context["work_items"]) == 3
        assert len(context["sprints"]) == 1
        assert len(context["docs"]) == 2
        assert len(context["decisions"]) == 1
        assert len(context["risks"]) == 0

    def test_calls_model_dump_on_each_item(self):
        snapshot = _make_snapshot()

        context = _snapshot_to_context(snapshot)

        # Verify model_dump() was called
        for item in snapshot.work_items:
            item.model_dump.assert_called_once()

    def test_context_values_are_dicts(self):
        snapshot = _make_snapshot()

        context = _snapshot_to_context(snapshot)

        assert context["work_items"][0] == {"name": "wi-0"}


# -- Tests for _load_notion_context --


class TestLoadNotionContext:
    """Tests for _load_notion_context()."""

    @patch("main.resolve_tool_class")
    def test_prefers_local_snapshot(self, mock_resolve):
        """Should try local snapshot first via NotionWriteTool."""
        snapshot = _make_snapshot()
        write_tool = MagicMock()
        write_tool.load_local_snapshot.return_value = snapshot
        write_tool_cls = MagicMock(return_value=write_tool)
        mock_resolve.return_value = write_tool_cls
        settings = _make_settings()

        context = _load_notion_context(settings)

        assert context is not None
        assert len(context["work_items"]) == 3
        # Should have been called with "notion_write" first
        mock_resolve.assert_called_with("notion_write", settings)

    @patch("main.resolve_tool_class")
    def test_falls_back_to_cloud_snapshot(self, mock_resolve):
        """If local snapshot fails, try cloud snapshot."""
        snapshot = _make_snapshot()
        cloud_tool = MagicMock()
        cloud_tool.load_snapshot.return_value = snapshot
        cloud_tool_cls = MagicMock(return_value=cloud_tool)

        write_tool_cls = MagicMock(side_effect=Exception("No local data"))

        def resolve(name, settings):
            if name == "notion_write":
                return write_tool_cls
            return cloud_tool_cls

        mock_resolve.side_effect = resolve
        settings = _make_settings()

        context = _load_notion_context(settings)

        assert context is not None
        assert len(context["work_items"]) == 3

    @patch("main.resolve_tool_class")
    def test_returns_none_when_no_data(self, mock_resolve):
        """Returns None when both local and cloud fail."""
        mock_resolve.side_effect = Exception("Tool not available")
        settings = _make_settings()

        context = _load_notion_context(settings)

        assert context is None

    @patch("main.resolve_tool_class")
    def test_returns_none_when_snapshot_is_none(self, mock_resolve):
        """Returns None when local snapshot returns None and cloud has no key."""
        write_tool = MagicMock()
        write_tool.load_local_snapshot.return_value = None
        write_tool_cls = MagicMock(return_value=write_tool)

        cloud_tool = MagicMock()
        cloud_tool.load_snapshot.return_value = None
        cloud_tool_cls = MagicMock(return_value=cloud_tool)

        def resolve(name, settings):
            if name == "notion_write":
                return write_tool_cls
            return cloud_tool_cls

        mock_resolve.side_effect = resolve
        settings = _make_settings()

        context = _load_notion_context(settings)

        assert context is None
