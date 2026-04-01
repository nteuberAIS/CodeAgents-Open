"""Tests for the Aider tool.

All tests mock subprocess — no real Aider CLI calls are executed.
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from schemas.aider_models import AiderResult
from tools.aider_tool import AiderTool, AiderToolError


# ------------------------------------------------------------------ #
#  Test helpers                                                       #
# ------------------------------------------------------------------ #


def _make_settings(**overrides):
    """Create a minimal Settings-like object for testing."""
    defaults = {
        "aider_binary": "aider",
        "aider_model": None,
        "aider_timeout": 120,
        "aider_repo_dir": "/tmp/repo",
        "ollama_model": "qwen2.5-coder:7b",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _mock_subprocess_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
):
    """Create a mock subprocess.CompletedProcess."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


# ------------------------------------------------------------------ #
#  AiderResult model                                                  #
# ------------------------------------------------------------------ #


class TestAiderResult:
    def test_defaults(self):
        r = AiderResult(command="aider --version", success=True)
        assert r.output is None
        assert r.error is None
        assert r.modified_files == []
        assert r.dry_run is False

    def test_all_fields(self):
        r = AiderResult(
            command="aider --message 'fix' --file a.py",
            success=True,
            output="Wrote a.py",
            error=None,
            modified_files=["a.py"],
            dry_run=False,
        )
        assert r.modified_files == ["a.py"]

    def test_round_trip(self):
        r = AiderResult(
            command="aider -m 'test'",
            success=True,
            output="done",
            modified_files=["x.py", "y.py"],
            dry_run=True,
        )
        data = r.model_dump()
        r2 = AiderResult(**data)
        assert r2 == r


# ------------------------------------------------------------------ #
#  AiderTool init & validation                                        #
# ------------------------------------------------------------------ #


class TestAiderToolInit:
    def test_skips_validation_in_dry_run(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        assert tool.dry_run is True
        assert tool.binary == "aider"

    @patch("tools.aider_tool.subprocess.run")
    def test_validates_cli_on_init(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="aider v0.50.0", returncode=0
        )
        tool = AiderTool(_make_settings(), dry_run=False)
        assert tool.binary == "aider"
        mock_run.assert_called_once()
        cmd = mock_run.call_args.args[0]
        assert cmd == ["aider", "--version"]

    @patch("tools.aider_tool.subprocess.run")
    def test_raises_when_binary_not_found(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with pytest.raises(AiderToolError, match="not found"):
            AiderTool(_make_settings(), dry_run=False)

    @patch("tools.aider_tool.subprocess.run")
    def test_raises_when_version_fails(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="error", returncode=1
        )
        with pytest.raises(AiderToolError, match="non-zero exit code"):
            AiderTool(_make_settings(), dry_run=False)

    @patch("tools.aider_tool.subprocess.run")
    def test_raises_when_validation_times_out(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="aider", timeout=10)
        with pytest.raises(AiderToolError, match="timed out"):
            AiderTool(_make_settings(), dry_run=False)

    def test_extracts_settings(self):
        tool = AiderTool(
            _make_settings(
                aider_binary="/usr/bin/aider",
                aider_timeout=300,
                aider_repo_dir="/home/user/project",
            ),
            dry_run=True,
        )
        assert tool.binary == "/usr/bin/aider"
        assert tool.timeout == 300
        assert tool.repo_dir == "/home/user/project"

    def test_auto_prefixes_ollama_model(self):
        tool = AiderTool(
            _make_settings(aider_model=None, ollama_model="qwen2.5-coder:7b"),
            dry_run=True,
        )
        assert tool.model == "ollama/qwen2.5-coder:7b"

    def test_uses_explicit_aider_model(self):
        tool = AiderTool(
            _make_settings(aider_model="openai/gpt-4"),
            dry_run=True,
        )
        assert tool.model == "openai/gpt-4"


# ------------------------------------------------------------------ #
#  Dry-run mode                                                       #
# ------------------------------------------------------------------ #


class TestAiderToolDryRun:
    def test_edit_returns_dry_run_result(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        result = tool.edit(
            instruction="Add error handling",
            files=["src/pipeline.py"],
        )
        assert result.dry_run is True
        assert result.success is True
        assert result.output is None
        assert result.modified_files == []

    def test_edit_dry_run_populates_command(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        result = tool.edit(
            instruction="Fix the bug",
            files=["a.py", "b.py"],
        )
        assert "--message" in result.command
        assert "Fix the bug" in result.command
        assert "--file a.py" in result.command
        assert "--file b.py" in result.command


# ------------------------------------------------------------------ #
#  Happy path                                                         #
# ------------------------------------------------------------------ #


class TestAiderToolEdit:
    @patch("tools.aider_tool.subprocess.run")
    def test_edit_success(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="Applied edit.\nWrote src/pipeline.py\n",
            returncode=0,
        )
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(), dry_run=False)
        result = tool.edit(
            instruction="Add error handling",
            files=["src/pipeline.py"],
        )
        assert result.success is True
        assert result.dry_run is False
        assert "src/pipeline.py" in result.modified_files
        mock_run.assert_called_once()

    @patch("tools.aider_tool.subprocess.run")
    def test_edit_uses_repo_dir_from_settings(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(returncode=0)
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(
                _make_settings(aider_repo_dir="/opt/project"), dry_run=False
            )
        tool.edit(instruction="Fix it", files=["a.py"])
        assert mock_run.call_args.kwargs["cwd"] == "/opt/project"

    @patch("tools.aider_tool.subprocess.run")
    def test_edit_repo_dir_param_overrides_settings(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(returncode=0)
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(
                _make_settings(aider_repo_dir="/opt/project"), dry_run=False
            )
        tool.edit(instruction="Fix it", files=["a.py"], repo_dir="/tmp/other")
        assert mock_run.call_args.kwargs["cwd"] == "/tmp/other"

    @patch("tools.aider_tool.subprocess.run")
    def test_edit_multiple_modified_files(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="Wrote src/a.py\nWrote src/b.py\nWrote src/c.py\n",
            returncode=0,
        )
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(), dry_run=False)
        result = tool.edit(instruction="Refactor", files=["src/a.py", "src/b.py"])
        assert len(result.modified_files) == 3
        assert "src/a.py" in result.modified_files
        assert "src/c.py" in result.modified_files


# ------------------------------------------------------------------ #
#  Failure cases                                                      #
# ------------------------------------------------------------------ #


class TestAiderToolFailures:
    @patch("tools.aider_tool.subprocess.run")
    def test_edit_nonzero_exit(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="Error: model not found", returncode=1
        )
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(), dry_run=False)
        result = tool.edit(instruction="Do something", files=["a.py"])
        assert result.success is False
        assert result.error == "Error: model not found"

    @patch("tools.aider_tool.subprocess.run")
    def test_edit_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="aider", timeout=120)
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(), dry_run=False)
        result = tool.edit(instruction="Slow task", files=["a.py"])
        assert result.success is False
        assert "timed out" in result.error

    @patch("tools.aider_tool.subprocess.run")
    def test_edit_binary_not_found_at_runtime(self, mock_run):
        mock_run.side_effect = FileNotFoundError()
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(), dry_run=False)
        result = tool.edit(instruction="Do something", files=["a.py"])
        assert result.success is False
        assert "not found" in result.error


# ------------------------------------------------------------------ #
#  Command construction                                               #
# ------------------------------------------------------------------ #


class TestAiderToolCommandConstruction:
    def test_required_flags_present(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        result = tool.edit(instruction="Fix bug", files=["main.py"])
        assert "--message" in result.command
        assert "--yes-always" in result.command
        assert "--no-auto-commits" in result.command
        assert "--no-pretty" in result.command
        assert "--no-stream" in result.command
        assert "--no-show-model-warnings" in result.command
        assert "--no-gitignore" in result.command
        assert "--model" in result.command

    def test_model_in_command(self):
        tool = AiderTool(
            _make_settings(aider_model=None, ollama_model="qwen2.5-coder:7b"),
            dry_run=True,
        )
        result = tool.edit(instruction="Fix", files=["a.py"])
        assert "ollama/qwen2.5-coder:7b" in result.command

    def test_multiple_files_produce_multiple_flags(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        result = tool.edit(
            instruction="Refactor",
            files=["a.py", "b.py", "c.py"],
        )
        assert result.command.count("--file") == 3

    def test_read_files_flags(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        result = tool.edit(
            instruction="Implement spec",
            files=["main.py"],
            read_files=["docs/spec.md", "docs/design.md"],
        )
        assert result.command.count("--read") == 2
        assert "docs/spec.md" in result.command
        assert "docs/design.md" in result.command

    def test_no_read_flags_when_none(self):
        tool = AiderTool(_make_settings(), dry_run=True)
        result = tool.edit(instruction="Fix", files=["a.py"])
        assert "--read" not in result.command

    @patch("tools.aider_tool.subprocess.run")
    def test_subprocess_cwd_set(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(returncode=0)
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(), dry_run=False)
        tool.edit(instruction="Fix", files=["a.py"], repo_dir="/my/repo")
        assert mock_run.call_args.kwargs["cwd"] == "/my/repo"

    @patch("tools.aider_tool.subprocess.run")
    def test_subprocess_timeout_set(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(returncode=0)
        with patch.object(AiderTool, "_validate_cli"):
            tool = AiderTool(_make_settings(aider_timeout=60), dry_run=False)
        tool.edit(instruction="Fix", files=["a.py"])
        assert mock_run.call_args.kwargs["timeout"] == 60


# ------------------------------------------------------------------ #
#  Output parsing                                                     #
# ------------------------------------------------------------------ #


class TestParseModifiedFiles:
    def test_parses_wrote_lines(self):
        output = "Thinking...\nWrote src/main.py\nWrote src/utils.py\nDone."
        result = AiderTool._parse_modified_files(output)
        assert result == ["src/main.py", "src/utils.py"]

    def test_empty_output(self):
        assert AiderTool._parse_modified_files("") == []

    def test_none_output(self):
        assert AiderTool._parse_modified_files(None) == []

    def test_no_wrote_lines(self):
        output = "Thinking...\nNo changes needed.\nDone."
        assert AiderTool._parse_modified_files(output) == []
