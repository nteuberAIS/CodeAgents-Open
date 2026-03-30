"""Tests for main.py — CLI tool binding, context loading, and cascade CLI."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from main import _get_agent_tools, _load_notion_context, _snapshot_to_context, cmd_cascade


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


# -- Tests for per-agent model overrides --


class TestGetLlmModelOverrides:
    """Tests for get_llm() per-agent model overrides."""

    def test_default_model_when_no_agent_name(self):
        from config.settings import get_llm

        settings = _make_settings()
        settings.ollama_model = "qwen2.5-coder:7b"
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_temperature = 0.2
        settings.agent_model_overrides = {}

        llm = get_llm(settings)

        assert llm.model == "qwen2.5-coder:7b"

    def test_agent_override_picks_up_specific_model(self):
        from config.settings import get_llm

        settings = _make_settings()
        settings.ollama_model = "qwen2.5-coder:7b"
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_temperature = 0.2
        settings.agent_model_overrides = {"sprint_planner": "mistral:7b"}

        llm = get_llm(settings, agent_name="sprint_planner")

        assert llm.model == "mistral:7b"

    def test_falls_back_to_global_when_agent_not_in_overrides(self):
        from config.settings import get_llm

        settings = _make_settings()
        settings.ollama_model = "qwen2.5-coder:7b"
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_temperature = 0.2
        settings.agent_model_overrides = {"coder": "deepseek:7b"}

        llm = get_llm(settings, agent_name="sprint_planner")

        assert llm.model == "qwen2.5-coder:7b"

    def test_cli_model_overrides_per_agent_setting(self):
        """Simulates --model CLI flag: sets ollama_model and clears override."""
        from config.settings import get_llm

        settings = _make_settings()
        settings.ollama_base_url = "http://localhost:11434"
        settings.ollama_temperature = 0.2
        settings.agent_model_overrides = {"sprint_planner": "mistral:7b"}

        # CLI --model flag sets global and clears agent override
        settings.ollama_model = "llama3:8b"
        settings.agent_model_overrides.pop("sprint_planner", None)

        llm = get_llm(settings, agent_name="sprint_planner")

        assert llm.model == "llama3:8b"


# -- Tests for cmd_cascade --


def _cascade_args(**overrides) -> Namespace:
    """Create a Namespace mimicking parsed cascade CLI args."""
    defaults = {
        "prompt": "Deploy SHIR and bronze layer",
        "sprint_id": None,
        "dry_run": False,
        "max_tasks": None,
        "abort_threshold": 0.5,
        "sync": False,
        "model": None,
        "list": False,
        "show": None,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestCmdCascade:
    """Tests for cmd_cascade()."""

    @patch("orchestration.CascadeRunner")
    @patch("main.get_settings")
    def test_happy_path(self, mock_get_settings, mock_runner_cls, tmp_path):
        """CascadeRunner is called and state is saved."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        final_state = {
            "sprint_id": "sprint-8",
            "status": "completed",
            "tasks": [{"id": "T1"}],
            "task_results": {},
            "errors": [],
            "failed_task_ids": [],
            "plan": {},
            "current_task_index": 1,
            "iteration_counts": {},
            "abort_threshold": 0.5,
            "max_tasks": 0,
        }
        runner = MagicMock()
        runner.run.return_value = final_state
        mock_runner_cls.return_value = runner

        args = _cascade_args(sprint_id="sprint-8")
        cmd_cascade(args)

        runner.run.assert_called_once_with(
            sprint_id="sprint-8",
            goal="Deploy SHIR and bronze layer",
            abort_threshold=0.5,
            max_tasks=0,
        )
        state_path = tmp_path / "cascade" / "sprint-8.json"
        assert state_path.exists()
        saved = json.loads(state_path.read_text())
        assert saved["status"] == "completed"

    @patch("orchestration.CascadeRunner")
    @patch("main.get_settings")
    def test_max_tasks_passed(self, mock_get_settings, mock_runner_cls, tmp_path):
        """--max-tasks is forwarded to runner."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        runner = MagicMock()
        runner.run.return_value = {"sprint_id": "s1", "status": "completed",
                                    "tasks": [], "task_results": {},
                                    "errors": [], "failed_task_ids": [],
                                    "plan": {}, "current_task_index": 0,
                                    "iteration_counts": {},
                                    "abort_threshold": 0.5, "max_tasks": 2}
        mock_runner_cls.return_value = runner

        args = _cascade_args(sprint_id="s1", max_tasks=2)
        cmd_cascade(args)

        runner.run.assert_called_once_with(
            sprint_id="s1",
            goal="Deploy SHIR and bronze layer",
            abort_threshold=0.5,
            max_tasks=2,
        )

    @patch("main.get_settings")
    def test_dry_run(self, mock_get_settings, capsys):
        """--dry-run prints info without executing."""
        settings = _make_settings()
        mock_get_settings.return_value = settings

        args = _cascade_args(dry_run=True, sprint_id="sprint-9")
        cmd_cascade(args)

        output = capsys.readouterr().out
        assert "Dry run" in output
        assert "sprint-9" in output

    @patch("main.get_settings")
    def test_list_empty(self, mock_get_settings, tmp_path, capsys):
        """--list with no saved runs prints 'No saved runs'."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        args = _cascade_args(prompt=None, **{"list": True})
        cmd_cascade(args)

        output = capsys.readouterr().out
        assert "No saved runs" in output

    @patch("main.get_settings")
    def test_list_shows_saved_runs(self, mock_get_settings, tmp_path, capsys):
        """--list displays saved cascade run files."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        cascade_dir = tmp_path / "cascade"
        cascade_dir.mkdir()
        state = {"status": "completed", "tasks": [{"id": "T1"}, {"id": "T2"}]}
        (cascade_dir / "sprint-8.json").write_text(json.dumps(state))

        args = _cascade_args(prompt=None, **{"list": True})
        cmd_cascade(args)

        output = capsys.readouterr().out
        assert "sprint-8" in output
        assert "completed" in output
        assert "tasks=2" in output

    @patch("main.get_settings")
    def test_show_displays_state(self, mock_get_settings, tmp_path, capsys):
        """--show reads and prints saved state JSON."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        cascade_dir = tmp_path / "cascade"
        cascade_dir.mkdir()
        state = {"status": "completed", "sprint_id": "sprint-8"}
        (cascade_dir / "sprint-8.json").write_text(json.dumps(state))

        args = _cascade_args(prompt=None, show="sprint-8")
        cmd_cascade(args)

        output = capsys.readouterr().out
        assert '"sprint_id": "sprint-8"' in output

    @patch("main.get_settings")
    def test_show_missing_exits(self, mock_get_settings, tmp_path):
        """--show with nonexistent sprint ID exits with error."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        args = _cascade_args(prompt=None, show="nonexistent")
        with pytest.raises(SystemExit):
            cmd_cascade(args)

    @patch("main.resolve_tool_class")
    @patch("orchestration.CascadeRunner")
    @patch("main.get_settings")
    def test_sync_flag(self, mock_get_settings, mock_runner_cls, mock_resolve, tmp_path):
        """--sync calls Notion sync before cascade."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        tool = MagicMock()
        tool.sync.return_value = MagicMock(counts={"work_items": 5})
        tool_cls = MagicMock(return_value=tool)
        mock_resolve.return_value = tool_cls

        runner = MagicMock()
        runner.run.return_value = {"sprint_id": "s1", "status": "completed",
                                    "tasks": [], "task_results": {},
                                    "errors": [], "failed_task_ids": [],
                                    "plan": {}, "current_task_index": 0,
                                    "iteration_counts": {},
                                    "abort_threshold": 0.5, "max_tasks": 0}
        mock_runner_cls.return_value = runner

        args = _cascade_args(sprint_id="s1", sync=True)
        cmd_cascade(args)

        tool.sync.assert_called_once()
        runner.run.assert_called_once()

    @patch("orchestration.CascadeRunner")
    @patch("main.get_settings")
    def test_default_sprint_id(self, mock_get_settings, mock_runner_cls, tmp_path):
        """Sprint ID defaults to timestamp-based format when not provided."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        mock_get_settings.return_value = settings

        runner = MagicMock()
        runner.run.return_value = {"sprint_id": "auto", "status": "completed",
                                    "tasks": [], "task_results": {},
                                    "errors": [], "failed_task_ids": [],
                                    "plan": {}, "current_task_index": 0,
                                    "iteration_counts": {},
                                    "abort_threshold": 0.5, "max_tasks": 0}
        mock_runner_cls.return_value = runner

        args = _cascade_args()  # no sprint_id
        cmd_cascade(args)

        call_kwargs = runner.run.call_args[1]
        assert call_kwargs["sprint_id"].startswith("sprint-")

    @patch("main.get_settings")
    def test_no_prompt_no_list_no_show_exits(self, mock_get_settings):
        """Missing prompt without --list or --show exits with error."""
        settings = _make_settings()
        mock_get_settings.return_value = settings

        args = _cascade_args(prompt=None)
        with pytest.raises(SystemExit):
            cmd_cascade(args)

    @patch("orchestration.CascadeRunner")
    @patch("main.get_settings")
    def test_model_override(self, mock_get_settings, mock_runner_cls, tmp_path):
        """--model overrides the Ollama model in settings."""
        settings = _make_settings()
        settings.data_dir = str(tmp_path)
        settings.ollama_model = "qwen2.5-coder:7b"
        mock_get_settings.return_value = settings

        runner = MagicMock()
        runner.run.return_value = {"sprint_id": "s1", "status": "completed",
                                    "tasks": [], "task_results": {},
                                    "errors": [], "failed_task_ids": [],
                                    "plan": {}, "current_task_index": 0,
                                    "iteration_counts": {},
                                    "abort_threshold": 0.5, "max_tasks": 0}
        mock_runner_cls.return_value = runner

        args = _cascade_args(sprint_id="s1", model="mistral:7b")
        cmd_cascade(args)

        assert settings.ollama_model == "mistral:7b"
