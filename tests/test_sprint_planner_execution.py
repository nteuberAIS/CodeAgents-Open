"""Tests for SprintPlannerAgent plan execution (Phase 2d).

Separate from test_sprint_planner.py to maintain clear separation between
LLM planning tests and tool execution tests.
"""

from __future__ import annotations

import json

from unittest.mock import MagicMock

import pytest

from agents.sprint_planner import SprintPlannerAgent


# -- Helpers --


VALID_PLAN = {
    "sprint": 8,
    "goal": "Complete Phase 2d",
    "tasks": [
        {
            "id": "SP8-001",
            "title": "Wire tools into agents",
            "description": "Connect tools to agent run loop",
            "assignee": None,
            "estimate_hrs": 4,
            "status": "todo",
        },
        {
            "id": "SP8-002",
            "title": "Add execution tests",
            "description": "Test plan execution with mock tools",
            "assignee": None,
            "estimate_hrs": 3,
            "status": "todo",
        },
    ],
    "dependencies": [
        {"from": "SP8-001", "to": "SP8-002", "type": "blocks"},
    ],
}


def _make_llm(response_content: str) -> MagicMock:
    """Create a mock ChatOllama that returns a fixed response."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    llm.invoke.return_value = mock_response
    return llm


def _make_mock_notion_write():
    """Mock NotionWriteTool that tracks calls."""
    tool = MagicMock()
    counter = {"n": 0}

    def create_work_item(**kwargs):
        counter["n"] += 1
        item = MagicMock()
        item.notion_id = f"local-mock-{counter['n']:03d}"
        item.name = kwargs.get("name", "Mock Task")
        return item

    tool.create_work_item.side_effect = create_work_item
    return tool


def _make_mock_git_tool(dry_run: bool = False):
    """Mock GitHubTool that returns success results."""
    tool = MagicMock()
    tool.sprint_branch_name.side_effect = lambda n: f"sprint-{n}"
    tool.task_branch_name.side_effect = lambda n, t: f"sprint-{n}/{t}"

    def create_branch(name, from_ref="main"):
        result = MagicMock()
        result.success = True
        result.dry_run = dry_run
        return result

    tool.create_branch.side_effect = create_branch
    return tool


def _make_agent_with_plan(plan: dict = None, **tools) -> SprintPlannerAgent:
    """Create a SprintPlannerAgent with mocked LLM and optional tools."""
    plan = plan or VALID_PLAN
    llm = _make_llm(json.dumps(plan))
    agent = SprintPlannerAgent(llm=llm)

    for name, tool in tools.items():
        agent.tools[name] = tool

    return agent


# -- Tests --


class TestExecutePlan:
    """Tests for SprintPlannerAgent._execute_plan()."""

    def test_creates_notion_work_items(self):
        notion_write = _make_mock_notion_write()
        agent = _make_agent_with_plan(notion_write=notion_write)

        execution = agent._execute_plan(VALID_PLAN)

        assert len(execution["notion_items_created"]) == 2
        assert execution["notion_items_created"][0]["task_id"] == "SP8-001"
        assert execution["notion_items_created"][0]["notion_id"] == "local-mock-001"
        assert execution["notion_items_created"][1]["task_id"] == "SP8-002"

    def test_creates_sprint_and_task_branches(self):
        git_tool = _make_mock_git_tool()
        agent = _make_agent_with_plan(github=git_tool)

        execution = agent._execute_plan(VALID_PLAN)

        # 1 sprint branch + 2 task branches
        assert len(execution["branches_created"]) == 3
        assert execution["branches_created"][0]["branch"] == "sprint-8"
        assert execution["branches_created"][0]["task_id"] is None
        assert execution["branches_created"][1]["branch"] == "sprint-8/SP8-001"
        assert execution["branches_created"][2]["branch"] == "sprint-8/SP8-002"

    def test_captures_notion_errors_without_raising(self):
        notion_write = MagicMock()
        notion_write.create_work_item.side_effect = RuntimeError("Disk full")
        agent = _make_agent_with_plan(notion_write=notion_write)

        execution = agent._execute_plan(VALID_PLAN)

        assert len(execution["errors"]) == 2
        assert "SP8-001" in execution["errors"][0]
        assert "Disk full" in execution["errors"][0]
        assert execution["notion_items_created"] == []

    def test_captures_git_errors_without_raising(self):
        git_tool = MagicMock()
        git_tool.sprint_branch_name.side_effect = lambda n: f"sprint-{n}"
        git_tool.create_branch.side_effect = RuntimeError("git not found")
        agent = _make_agent_with_plan(github=git_tool)

        execution = agent._execute_plan(VALID_PLAN)

        assert len(execution["errors"]) >= 1
        assert "Sprint branch creation failed" in execution["errors"][0]

    def test_empty_result_when_no_tools(self):
        agent = _make_agent_with_plan()  # No tools bound

        execution = agent._execute_plan(VALID_PLAN)

        assert execution["notion_items_created"] == []
        assert execution["branches_created"] == []
        assert execution["errors"] == []

    def test_both_notion_and_git_tools(self):
        notion_write = _make_mock_notion_write()
        git_tool = _make_mock_git_tool()
        agent = _make_agent_with_plan(notion_write=notion_write, github=git_tool)

        execution = agent._execute_plan(VALID_PLAN)

        assert len(execution["notion_items_created"]) == 2
        assert len(execution["branches_created"]) == 3
        assert execution["errors"] == []

    def test_azdevops_tool_used_when_github_not_bound(self):
        git_tool = _make_mock_git_tool()
        agent = _make_agent_with_plan(azdevops=git_tool)

        execution = agent._execute_plan(VALID_PLAN)

        assert len(execution["branches_created"]) == 3
        assert execution["branches_created"][0]["branch"] == "sprint-8"

    def test_skips_branches_when_sprint_number_missing(self):
        plan_no_sprint = {**VALID_PLAN, "sprint": None}
        git_tool = _make_mock_git_tool()
        agent = _make_agent_with_plan(plan=plan_no_sprint, github=git_tool)

        execution = agent._execute_plan(plan_no_sprint)

        assert execution["branches_created"] == []

    def test_dry_run_git_tool(self):
        git_tool = _make_mock_git_tool(dry_run=True)
        agent = _make_agent_with_plan(github=git_tool)

        execution = agent._execute_plan(VALID_PLAN)

        for branch in execution["branches_created"]:
            assert branch["dry_run"] is True


class TestRunWithTools:
    """Tests for run() with tools bound."""

    def test_run_includes_execution_key(self):
        notion_write = _make_mock_notion_write()
        agent = _make_agent_with_plan(notion_write=notion_write)

        result = agent.run("Plan sprint 8")

        assert "execution" in result
        assert len(result["execution"]["notion_items_created"]) == 2

    def test_run_omits_execution_when_no_tools(self):
        agent = _make_agent_with_plan()  # No tools

        result = agent.run("Plan sprint 8")

        assert "execution" not in result

    def test_run_omits_execution_on_parse_error(self):
        llm = _make_llm("not valid json at all")
        agent = SprintPlannerAgent(llm=llm)
        agent.tools["notion_write"] = _make_mock_notion_write()

        result = agent.run("Plan sprint 8")

        assert "parse_error" in result
        assert "execution" not in result

    def test_run_returns_plan_even_with_execution_errors(self):
        notion_write = MagicMock()
        notion_write.create_work_item.side_effect = RuntimeError("fail")
        agent = _make_agent_with_plan(notion_write=notion_write)

        result = agent.run("Plan sprint 8")

        assert result["sprint"] == 8
        assert "execution" in result
        assert len(result["execution"]["errors"]) > 0


class TestRunWithoutTools:
    """Tests for backward compatibility — run() without tools."""

    def test_produces_same_output_as_before(self):
        llm = _make_llm(json.dumps(VALID_PLAN))
        agent = SprintPlannerAgent(llm=llm)

        result = agent.run("Plan sprint 8")

        assert result["sprint"] == 8
        assert result["goal"] == "Complete Phase 2d"
        assert len(result["tasks"]) == 2
        assert "execution" not in result

    def test_tool_declarations_exist(self):
        """Verify agent declares its tool requirements."""
        assert SprintPlannerAgent.REQUIRED_TOOLS == []
        assert "notion_write" in SprintPlannerAgent.OPTIONAL_TOOLS
        assert "github" in SprintPlannerAgent.OPTIONAL_TOOLS
        assert "azdevops" in SprintPlannerAgent.OPTIONAL_TOOLS
