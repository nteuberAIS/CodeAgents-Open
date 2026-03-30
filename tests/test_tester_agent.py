"""Tests for agents/tester.py.

All tests mock subprocess.run — no real pytest execution occurs.
The LLM is injected per BaseAgent contract but never invoked.
"""

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from agents.tester import TesterAgent


# -- Test constants --

TASK_INPUT = json.dumps({
    "task_id": "SP8-001",
    "task_title": "Add error handling",
    "repo_dir": "/tmp/test-repo",
})

TASK_INPUT_NO_REPO = json.dumps({
    "task_id": "SP8-002",
    "task_title": "Fix tests",
})

TASK_INPUT_CUSTOM_CMD = json.dumps({
    "task_id": "SP8-003",
    "task_title": "Run custom tests",
    "repo_dir": "/tmp/test-repo",
    "test_command": "python -m pytest tests/",
})


# -- Helpers --

def _make_llm() -> MagicMock:
    """Create a mock ChatOllama (injected but never called by TesterAgent)."""
    return MagicMock()


def _make_subprocess_result(
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
) -> subprocess.CompletedProcess:
    """Create a CompletedProcess for mocking."""
    return subprocess.CompletedProcess(
        args=["pytest"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def _make_settings(**overrides):
    """Create a mock Settings object."""
    defaults = {
        "test_command": "pytest",
        "test_timeout": 300,
        "test_repo_dir": None,
        "aider_repo_dir": None,
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


# -- Tests: Class attributes --

class TestTesterInit:
    def test_class_attributes(self):
        llm = _make_llm()
        agent = TesterAgent(llm=llm)
        assert agent.name == "tester"
        assert agent.MAX_ITERATIONS == 1

    def test_required_tools_empty(self):
        assert TesterAgent.REQUIRED_TOOLS == []

    def test_optional_tools_empty(self):
        assert TesterAgent.OPTIONAL_TOOLS == []


# -- Tests: Happy path --

class TestRunHappyPath:
    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_all_tests_pass(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.return_value = _make_subprocess_result(
            stdout="test_foo.py::test_one PASSED\n=== 5 passed in 0.12s ===\n",
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["error_type"] is None
        po = result["partial_output"]
        assert po["test_passed"] is True
        assert po["passed_count"] == 5
        assert po["failed_count"] == 0
        assert po["error_count"] == 0
        assert po["task_id"] == "SP8-001"
        assert "5 passed" in po["test_output"]

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_custom_test_command(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 3 passed in 0.10s ===\n",
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT_CUSTOM_CMD)

        assert result["success"] is True
        mock_run.assert_called_once()
        call_args = mock_run.call_args
        assert call_args[0][0] == ["python", "-m", "pytest", "tests/"]


# -- Tests: Test failures reported as data --

class TestRunWithFailures:
    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_some_failures(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 3 passed, 2 failed in 0.34s ===\n",
            returncode=1,
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT)

        # Test failures are DATA, not agent errors
        assert result["success"] is True
        assert result["error_type"] is None
        po = result["partial_output"]
        assert po["test_passed"] is False
        assert po["passed_count"] == 3
        assert po["failed_count"] == 2
        assert po["error_count"] == 0

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_failures_and_errors(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 1 passed, 1 failed, 1 error in 0.56s ===\n",
            returncode=1,
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["test_passed"] is False
        assert po["passed_count"] == 1
        assert po["failed_count"] == 1
        assert po["error_count"] == 1

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_all_failed(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 3 failed in 0.20s ===\n",
            returncode=1,
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["test_passed"] is False
        assert po["passed_count"] == 0
        assert po["failed_count"] == 3


# -- Tests: Subprocess errors --

class TestSubprocessErrors:
    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_subprocess_timeout(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.side_effect = subprocess.TimeoutExpired(
            cmd="pytest", timeout=300,
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "timeout"
        assert "timed out" in result["error_message"]
        assert result["partial_output"]["task_id"] == "SP8-001"

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_subprocess_crash(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.side_effect = OSError("No such file or directory: 'pytest'")

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "infra"
        assert "Failed to execute" in result["error_message"]

    @patch("agents.tester.get_settings")
    def test_no_repo_dir(self, mock_settings):
        mock_settings.return_value = _make_settings()

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(TASK_INPUT_NO_REPO)

        assert result["success"] is False
        assert result["error_type"] == "logic"
        assert "repo_dir" in result["error_message"]


# -- Tests: Input validation --

class TestInputValidation:
    @patch("agents.tester.get_settings")
    def test_invalid_json_input(self, mock_settings):
        mock_settings.return_value = _make_settings()

        agent = TesterAgent(llm=_make_llm())
        result = agent.run("not valid json {{{")

        assert result["success"] is False
        assert result["error_type"] == "logic"
        assert "Invalid JSON" in result["error_message"]

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_missing_task_id_defaults_to_unknown(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings()
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 1 passed in 0.01s ===\n",
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(json.dumps({"repo_dir": "/tmp/repo"}))

        assert result["success"] is True
        assert result["partial_output"]["task_id"] == "unknown"


# -- Tests: Pytest output parsing --

class TestParsePytestSummary:
    def test_all_passed(self):
        output = "=== 5 passed in 0.12s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (5, 0, 0)

    def test_passed_and_failed(self):
        output = "=== 3 passed, 2 failed in 0.34s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (3, 2, 0)

    def test_passed_failed_error(self):
        output = "=== 1 passed, 1 failed, 1 error in 0.56s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (1, 1, 1)

    def test_only_failed(self):
        output = "=== 3 failed in 0.20s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (0, 3, 0)

    def test_only_errors(self):
        output = "=== 2 error in 0.10s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (0, 0, 2)

    def test_large_counts(self):
        output = "=== 340 passed, 5 failed in 12.34s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (340, 5, 0)

    def test_no_summary_line(self):
        output = "Some random output without a summary"
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (0, 0, 0)

    def test_empty_output(self):
        p, f, e = TesterAgent._parse_pytest_summary("")
        assert (p, f, e) == (0, 0, 0)

    def test_summary_in_stderr(self):
        output = "ERRORS\n=== 1 passed, 2 error in 0.30s ==="
        p, f, e = TesterAgent._parse_pytest_summary(output)
        assert (p, f, e) == (1, 0, 2)


# -- Tests: Settings fallback --

class TestSettingsFallback:
    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_falls_back_to_settings_repo_dir(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings(
            test_repo_dir="/configured/repo",
        )
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 1 passed in 0.01s ===\n",
        )

        agent = TesterAgent(llm=_make_llm())
        # Input without repo_dir
        result = agent.run(json.dumps({"task_id": "SP8-010"}))

        assert result["success"] is True
        call_args = mock_run.call_args
        assert call_args[1]["cwd"] == "/configured/repo"

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_falls_back_to_aider_repo_dir(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings(
            aider_repo_dir="/aider/repo",
        )
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 1 passed in 0.01s ===\n",
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(json.dumps({"task_id": "SP8-010"}))

        assert result["success"] is True
        call_args = mock_run.call_args
        assert call_args[1]["cwd"] == "/aider/repo"

    @patch("agents.tester.get_settings")
    @patch("agents.tester.subprocess.run")
    def test_input_repo_dir_overrides_settings(self, mock_run, mock_settings):
        mock_settings.return_value = _make_settings(
            test_repo_dir="/configured/repo",
        )
        mock_run.return_value = _make_subprocess_result(
            stdout="=== 1 passed in 0.01s ===\n",
        )

        agent = TesterAgent(llm=_make_llm())
        result = agent.run(json.dumps({
            "task_id": "SP8-010",
            "repo_dir": "/override/repo",
        }))

        assert result["success"] is True
        call_args = mock_run.call_args
        assert call_args[1]["cwd"] == "/override/repo"


# -- Tests: Output truncation --

class TestTruncate:
    def test_short_text_unchanged(self):
        assert TesterAgent._truncate("short", 100) == "short"

    def test_long_text_truncated(self):
        text = "a" * 5000
        result = TesterAgent._truncate(text, 100)
        assert len(result) < 200
        assert result.endswith("... (truncated)")

    def test_none_returns_none(self):
        assert TesterAgent._truncate(None, 100) is None
