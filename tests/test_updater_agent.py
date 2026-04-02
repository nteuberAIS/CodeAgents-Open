"""Tests for agents/updater.py.

All tests mock the LLM, git tool, and notion_write tool —
no real API or subprocess calls are made.
"""

import json
from unittest.mock import MagicMock

import pytest

from agents.updater import UpdaterAgent
from schemas.git_models import GitCommandResult


# -- Test constants --

TASK_INPUT = json.dumps({
    "task_id": "SP8-001",
    "task_title": "Add error handling",
    "source_branch": "task/sprint-8/SP8-001",
    "target_branch": "sprint-8",
    "notion_id": "wi-001",
    "test_summary": "5 passed, 0 failed",
    "task_description": "Add try/except to ingest() function",
    "modified_files": ["src/pipeline.py"],
})

TASK_INPUT_NO_NOTION = json.dumps({
    "task_id": "SP8-002",
    "task_title": "Fix tests",
    "source_branch": "task/sprint-8/SP8-002",
    "target_branch": "sprint-8",
})

TASK_INPUT_MINIMAL = json.dumps({
    "task_id": "SP8-003",
    "task_title": "Simple change",
})


# -- Helpers --

def _make_llm(response_content: str | None = None) -> MagicMock:
    """Create a mock ChatOllama that returns a fixed response."""
    llm = MagicMock()
    if response_content is None:
        response_content = json.dumps({
            "pr_title": "Add error handling to ingest pipeline",
            "pr_description": "Adds try/except blocks to the ingest() function.",
        })
    mock_response = MagicMock()
    mock_response.content = response_content
    llm.invoke.return_value = mock_response
    return llm


def _make_git_result(
    success: bool = True,
    output: str | None = "https://dev.azure.com/org/project/_git/repo/pullrequest/42",
    error: str | None = None,
    dry_run: bool = False,
) -> GitCommandResult:
    """Create a GitCommandResult for mocking."""
    return GitCommandResult(
        command="az repos pr create ...",
        success=success,
        output=output,
        error=error,
        dry_run=dry_run,
    )


def _make_git_tool(result: GitCommandResult | None = None, side_effect=None) -> MagicMock:
    """Create a mock git tool."""
    tool = MagicMock()
    if side_effect is not None:
        tool.create_pull_request.side_effect = side_effect
    elif result is not None:
        tool.create_pull_request.return_value = result
    else:
        tool.create_pull_request.return_value = _make_git_result()
    return tool


def _make_notion_tool(side_effect=None) -> MagicMock:
    """Create a mock notion_write tool."""
    tool = MagicMock()
    if side_effect is not None:
        tool.update_work_item.side_effect = side_effect
    return tool


def _make_agent_with_tools(
    llm: MagicMock | None = None,
    git_tool: MagicMock | None = None,
    notion_tool: MagicMock | None = None,
) -> UpdaterAgent:
    """Create an UpdaterAgent with optional mock tools bound."""
    agent = UpdaterAgent(llm=llm or _make_llm())
    if git_tool is not None:
        agent.tools["azdevops"] = git_tool
    if notion_tool is not None:
        agent.tools["notion_write"] = notion_tool
    return agent


# -- Tests: Class attributes --

class TestUpdaterInit:
    def test_class_attributes(self):
        agent = UpdaterAgent(llm=_make_llm())
        assert agent.name == "updater"
        assert agent.MAX_ITERATIONS == 2

    def test_required_tools_empty(self):
        assert UpdaterAgent.REQUIRED_TOOLS == []

    def test_optional_tools(self):
        assert "notion_write" in UpdaterAgent.OPTIONAL_TOOLS
        assert "azdevops" in UpdaterAgent.OPTIONAL_TOOLS
        assert "github" in UpdaterAgent.OPTIONAL_TOOLS


# -- Tests: Happy path --

class TestRunHappyPath:
    def test_both_tools_succeed(self):
        git_tool = _make_git_tool()
        notion_tool = _make_notion_tool()
        agent = _make_agent_with_tools(
            git_tool=git_tool, notion_tool=notion_tool,
        )

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["error_type"] is None
        po = result["partial_output"]
        assert po["pr_created"] is True
        assert po["pr_url"] is not None
        assert po["pr_title"] != ""
        assert po["notion_updated"] is True
        assert po["notion_status"] == "In Review"
        assert po["task_id"] == "SP8-001"

    def test_pr_created_sets_notion_in_review(self):
        git_tool = _make_git_tool()
        notion_tool = _make_notion_tool()
        agent = _make_agent_with_tools(
            git_tool=git_tool, notion_tool=notion_tool,
        )

        agent.run(TASK_INPUT)

        notion_tool.update_work_item.assert_called_once_with(
            "wi-001", status="In Review",
        )


# -- Tests: Graceful degradation --

class TestGracefulDegradation:
    def test_no_git_tool(self):
        notion_tool = _make_notion_tool()
        agent = _make_agent_with_tools(notion_tool=notion_tool)

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is False
        assert po["notion_updated"] is True
        # Without PR, Notion should still update but with "In Progress"
        assert po["notion_status"] == "In Progress"

    def test_no_notion_tool(self):
        git_tool = _make_git_tool()
        agent = _make_agent_with_tools(git_tool=git_tool)

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is True
        assert po["notion_updated"] is False

    def test_both_tools_missing(self):
        agent = UpdaterAgent(llm=_make_llm())

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is False
        assert po["notion_updated"] is False
        assert po["task_id"] == "SP8-001"

    def test_no_notion_id_skips_update(self):
        git_tool = _make_git_tool()
        notion_tool = _make_notion_tool()
        agent = _make_agent_with_tools(
            git_tool=git_tool, notion_tool=notion_tool,
        )

        result = agent.run(TASK_INPUT_NO_NOTION)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is True
        assert po["notion_updated"] is False
        notion_tool.update_work_item.assert_not_called()

    def test_no_branches_skips_pr(self):
        git_tool = _make_git_tool()
        agent = _make_agent_with_tools(git_tool=git_tool)

        result = agent.run(TASK_INPUT_MINIMAL)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is False


# -- Tests: Retry logic --

class TestRetryLogic:
    def test_pr_creation_retry_succeeds(self):
        """First PR attempt fails, second succeeds."""
        git_tool = _make_git_tool(side_effect=[
            _make_git_result(success=False, error="Transient error"),
            _make_git_result(success=True),
        ])
        agent = _make_agent_with_tools(git_tool=git_tool)

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is True
        assert git_tool.create_pull_request.call_count == 2

    def test_pr_creation_both_retries_fail(self):
        """Both PR attempts fail — agent reports failure."""
        git_tool = _make_git_tool(side_effect=[
            _make_git_result(success=False, error="Permanent error"),
            _make_git_result(success=False, error="Permanent error"),
        ])
        notion_tool = _make_notion_tool()
        agent = _make_agent_with_tools(
            git_tool=git_tool, notion_tool=notion_tool,
        )

        result = agent.run(TASK_INPUT)

        po = result["partial_output"]
        assert po["pr_created"] is False
        assert git_tool.create_pull_request.call_count == 2
        # Notion still updated despite PR failure
        assert po["notion_updated"] is True

    def test_notion_update_retry_succeeds(self):
        """First Notion update fails, second succeeds."""
        notion_tool = _make_notion_tool(side_effect=[
            Exception("Transient error"),
            MagicMock(),  # Success on second attempt
        ])
        agent = _make_agent_with_tools(notion_tool=notion_tool)

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["notion_updated"] is True
        assert notion_tool.update_work_item.call_count == 2

    def test_all_operations_fail_returns_failure(self):
        """Both PR and Notion fail after retries."""
        git_tool = _make_git_tool(side_effect=[
            _make_git_result(success=False, error="Failed"),
            _make_git_result(success=False, error="Failed"),
        ])
        notion_tool = _make_notion_tool(side_effect=[
            Exception("Failed"),
            Exception("Failed"),
        ])
        agent = _make_agent_with_tools(
            git_tool=git_tool, notion_tool=notion_tool,
        )

        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "tool"

    def test_pr_exception_retried(self):
        """PR creation raises exception, retried."""
        git_tool = _make_git_tool(side_effect=[
            Exception("Network error"),
            _make_git_result(success=True),
        ])
        agent = _make_agent_with_tools(git_tool=git_tool)

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is True


# -- Tests: LLM interaction --

class TestLLMInteraction:
    def test_llm_invoked_for_pr_description(self):
        llm = _make_llm()
        agent = UpdaterAgent(llm=llm)

        agent.run(TASK_INPUT)

        llm.invoke.assert_called_once()

    def test_llm_failure_uses_fallback(self):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("LLM crashed")

        agent = UpdaterAgent(llm=llm)
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        # Fallback: task title used as PR title
        assert po["pr_title"] == "Add error handling"

    def test_llm_invalid_json_uses_fallback(self):
        llm = _make_llm("not json at all")

        agent = UpdaterAgent(llm=llm)
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_title"] == "Add error handling"

    def test_llm_output_used_in_pr(self):
        llm = _make_llm(json.dumps({
            "pr_title": "feat: Add retry logic to ingest pipeline",
            "pr_description": "Implements exponential backoff retry.",
        }))
        git_tool = _make_git_tool()
        agent = _make_agent_with_tools(llm=llm, git_tool=git_tool)

        agent.run(TASK_INPUT)

        call_args = git_tool.create_pull_request.call_args
        assert call_args[1]["title"] == "feat: Add retry logic to ingest pipeline"
        assert call_args[1]["description"] == "Implements exponential backoff retry."


# -- Tests: Dry-run mode --

class TestDryRun:
    def test_git_dry_run(self):
        git_tool = _make_git_tool(
            result=_make_git_result(success=False, dry_run=True),
        )
        agent = _make_agent_with_tools(git_tool=git_tool)

        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        po = result["partial_output"]
        assert po["pr_created"] is True


# -- Tests: Input validation --

class TestInputValidation:
    def test_invalid_json_input(self):
        agent = UpdaterAgent(llm=_make_llm())
        result = agent.run("not valid json {{{")

        assert result["success"] is False
        assert result["error_type"] == "logic"
        assert "Invalid JSON" in result["error_message"]


# -- Tests: PR URL extraction --

class TestExtractPrUrl:
    def test_extracts_url_from_az_json(self):
        """Realistic az repos pr create JSON — constructs web URL."""
        az_output = json.dumps({
            "pullRequestId": 154,
            "title": "Add error handling",
            "url": "https://dev.azure.com/org/proj/_apis/git/repositories/repo/pullRequests/154",
            "createdBy": {
                "displayName": "Agent",
                "imageUrl": "https://dev.azure.com/org/_apis/GraphProfile/MemberAvatars/abc",
            },
            "repository": {
                "webUrl": "https://dev.azure.com/org/proj/_git/repo",
            },
        })
        url = UpdaterAgent._extract_pr_url(az_output)
        assert url == "https://dev.azure.com/org/proj/_git/repo/pullrequest/154"

    def test_falls_back_to_api_url(self):
        """JSON with pullRequestId but no repository.webUrl — falls back to url field."""
        az_output = json.dumps({
            "pullRequestId": 42,
            "url": "https://dev.azure.com/org/proj/_apis/git/repositories/repo/pullRequests/42",
        })
        url = UpdaterAgent._extract_pr_url(az_output)
        assert url == "https://dev.azure.com/org/proj/_apis/git/repositories/repo/pullRequests/42"

    def test_no_url_returns_none(self):
        assert UpdaterAgent._extract_pr_url("No URL here") is None

    def test_none_input_returns_none(self):
        assert UpdaterAgent._extract_pr_url(None) is None

    def test_github_plain_url_fallback(self):
        """Non-JSON output (GitHub CLI) — regex fallback."""
        url = UpdaterAgent._extract_pr_url(
            "https://github.com/org/repo/pull/123"
        )
        assert url == "https://github.com/org/repo/pull/123"
