"""Tests for agents/base.py tool binding."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from agents.base import BaseAgent


# -- Concrete subclass for testing (BaseAgent is abstract) --


class _StubAgent(BaseAgent):
    """Minimal concrete agent for testing base class methods."""

    name = "stub"

    def run(self, user_input: str) -> dict:
        return {"echo": user_input}


# -- Helpers --


def _make_settings(**overrides):
    """Create a minimal Settings-like object for testing."""
    defaults = {
        "data_dir": "data",
        "notion_api_key": "test-key",
        "github_repo_dir": ".",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _make_tool_cls(accepts_dry_run: bool = True, raises: Exception | None = None):
    """Create a mock tool class with configurable constructor behavior."""
    tool_instance = MagicMock()

    if raises:
        tool_cls = MagicMock(side_effect=raises)
    elif accepts_dry_run:
        tool_cls = MagicMock(return_value=tool_instance)
    else:
        # Simulate tools that don't accept dry_run — first call raises TypeError,
        # second call (without dry_run) succeeds
        def side_effect(**kwargs):
            if "dry_run" in kwargs:
                raise TypeError("unexpected keyword argument 'dry_run'")
            return tool_instance

        tool_cls = MagicMock(side_effect=side_effect)

    return tool_cls, tool_instance


# -- Tests --


class TestBindTools:
    """Tests for BaseAgent.bind_tools()."""

    @patch("config.settings.resolve_tool_class")
    def test_bind_single_tool_success(self, mock_resolve):
        tool_cls, tool_instance = _make_tool_cls()
        mock_resolve.return_value = tool_cls
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        results = agent.bind_tools(["github"], settings)

        assert results == {"github": True}
        assert agent.tools["github"] is tool_instance

    @patch("config.settings.resolve_tool_class")
    def test_bind_multiple_tools(self, mock_resolve):
        tool_cls_a, inst_a = _make_tool_cls()
        tool_cls_b, inst_b = _make_tool_cls()

        mock_resolve.side_effect = [tool_cls_a, tool_cls_b]
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        results = agent.bind_tools(["github", "notion_write"], settings)

        assert results == {"github": True, "notion_write": True}
        assert agent.tools["github"] is inst_a
        assert agent.tools["notion_write"] is inst_b

    @patch("config.settings.resolve_tool_class")
    def test_bind_returns_false_on_failure(self, mock_resolve):
        mock_resolve.side_effect = ValueError("Tool not found")
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        results = agent.bind_tools(["nonexistent"], settings)

        assert results == {"nonexistent": False}

    @patch("config.settings.resolve_tool_class")
    def test_bind_does_not_raise_on_failure(self, mock_resolve):
        mock_resolve.side_effect = RuntimeError("CLI not found")
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        # Should NOT raise
        results = agent.bind_tools(["github"], settings)
        assert results["github"] is False

    @patch("config.settings.resolve_tool_class")
    def test_bind_passes_dry_run_to_tool(self, mock_resolve):
        tool_cls, _ = _make_tool_cls()
        mock_resolve.return_value = tool_cls
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        agent.bind_tools(["github"], settings, dry_run=True)

        tool_cls.assert_called_once_with(settings=settings, dry_run=True)

    @patch("config.settings.resolve_tool_class")
    def test_bind_falls_back_without_dry_run(self, mock_resolve):
        tool_cls, tool_instance = _make_tool_cls(accepts_dry_run=False)
        mock_resolve.return_value = tool_cls
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        results = agent.bind_tools(["notion_write"], settings, dry_run=True)

        assert results == {"notion_write": True}
        assert agent.tools["notion_write"] is tool_instance

    @patch("config.settings.resolve_tool_class")
    def test_bind_partial_failure(self, mock_resolve):
        """One tool succeeds, another fails — both reported."""
        good_cls, good_inst = _make_tool_cls()

        def resolve(name, settings):
            if name == "github":
                return good_cls
            raise ValueError("azdevops not configured")

        mock_resolve.side_effect = resolve
        settings = _make_settings()

        agent = _StubAgent(llm=MagicMock())
        results = agent.bind_tools(["github", "azdevops"], settings)

        assert results == {"github": True, "azdevops": False}
        assert agent.tools["github"] is good_inst
        assert agent.tools["azdevops"] is None


class TestGetTool:
    """Tests for BaseAgent.get_tool()."""

    def test_get_bound_tool(self):
        agent = _StubAgent(llm=MagicMock())
        mock_tool = MagicMock()
        agent.tools["notion_write"] = mock_tool

        assert agent.get_tool("notion_write") is mock_tool

    def test_get_unbound_tool_returns_none(self):
        agent = _StubAgent(llm=MagicMock())

        assert agent.get_tool("github") is None

    def test_get_failed_tool_returns_none(self):
        agent = _StubAgent(llm=MagicMock())
        agent.tools["github"] = None  # Failed bind stores None

        assert agent.get_tool("github") is None


class TestHasTool:
    """Tests for BaseAgent.has_tool()."""

    def test_has_bound_tool(self):
        agent = _StubAgent(llm=MagicMock())
        agent.tools["notion_write"] = MagicMock()

        assert agent.has_tool("notion_write") is True

    def test_has_unbound_tool(self):
        agent = _StubAgent(llm=MagicMock())

        assert agent.has_tool("github") is False

    def test_has_failed_tool(self):
        agent = _StubAgent(llm=MagicMock())
        agent.tools["github"] = None

        assert agent.has_tool("github") is False


class TestToolsInitialization:
    """Tests for tool dict initialization."""

    def test_tools_empty_by_default(self):
        agent = _StubAgent(llm=MagicMock())
        assert agent.tools == {}
