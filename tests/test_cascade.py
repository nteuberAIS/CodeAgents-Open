"""Tests for the LangGraph cascade orchestrator.

All agents are mocked — these tests verify graph traversal logic,
conditional routing, retry behavior, and state transitions.

plan_node validates pre-loaded tasks (no data loading or LLM).
coder/tester/updater use their respective agents.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orchestration.cascade import (
    MAX_OUTER_RETRIES,
    build_cascade_graph,
    commit_push_node,
    route_after_check,
    route_after_plan,
    route_after_test,
)
from orchestration.runner import CascadeRunner
from schemas.sprint_state import SprintState


# ---------------------------------------------------------------------------
# Helpers — result factories
# ---------------------------------------------------------------------------

def _make_notion_context(num_tasks: int = 2, sprint_id: str = "sprint-8") -> dict:
    """Mock Notion context with a sprint and linked work items."""
    sprint_notion_id = "test-sprint-id"
    return {
        "sprints": [
            {
                "notion_id": sprint_notion_id,
                "name": f"Sprint {sprint_id.lstrip('s-').lstrip('sprint-')}",
                "status": "Not started",
                "sprint_number": 8,
                "goal": "Test sprint goal",
                "start_date": "2026-04-01",
                "end_date": "2026-04-07",
                "work_item_ids": [f"wi-{i}" for i in range(1, num_tasks + 1)],
                "risk_ids": [],
                "has_content": False,
            }
        ],
        "work_items": [
            {
                "notion_id": f"wi-{i}",
                "name": f"Task {i}",
                "type": "Task",
                "status": "Ready",
                "priority": "P1",
                "estimate_hrs": 2,
                "sprint_id": sprint_notion_id,
                "owner": None,
                "has_content": False,
            }
            for i in range(1, num_tasks + 1)
        ],
        "docs": [],
        "decisions": [],
        "risks": [],
    }


def _make_coder_result(success: bool = True) -> dict:
    """CoderAgent result envelope."""
    if not success:
        return {
            "success": False,
            "error_type": "tool",
            "error_message": "Aider failed after retries",
            "partial_output": {
                "instruction": "Add error handling",
                "modified_files": [],
                "aider_output": "",
                "iterations_used": 5,
            },
        }
    return {
        "success": True,
        "error_type": None,
        "error_message": None,
        "partial_output": {
            "instruction": "Add error handling to pipeline.py",
            "modified_files": ["src/pipeline.py"],
            "aider_output": "Applied changes to src/pipeline.py",
            "iterations_used": 1,
        },
    }


def _make_tester_result(
    test_passed: bool = True,
    success: bool = True,
) -> dict:
    """TesterAgent result envelope."""
    if not success:
        return {
            "success": False,
            "error_type": "infra",
            "error_message": "subprocess timed out",
            "partial_output": {
                "test_passed": False,
                "passed_count": 0,
                "failed_count": 0,
                "error_count": 0,
                "test_output": "",
                "task_id": "",
            },
        }
    return {
        "success": True,
        "error_type": None,
        "error_message": None,
        "partial_output": {
            "test_passed": test_passed,
            "passed_count": 5 if test_passed else 3,
            "failed_count": 0 if test_passed else 2,
            "error_count": 0,
            "test_output": "5 passed" if test_passed else "3 passed, 2 failed",
            "task_id": "",
        },
    }


def _make_updater_result(success: bool = True) -> dict:
    """UpdaterAgent result envelope."""
    if not success:
        return {
            "success": False,
            "error_type": "tool",
            "error_message": "PR creation failed",
            "partial_output": {
                "pr_created": False,
                "pr_url": None,
                "notion_updated": False,
                "task_id": "",
                "dry_run": False,
            },
        }
    return {
        "success": True,
        "error_type": None,
        "error_message": None,
        "partial_output": {
            "pr_created": True,
            "pr_url": "https://github.com/org/repo/pull/42",
            "pr_title": "wi-1: Add error handling",
            "notion_updated": True,
            "notion_status": "In Review",
            "task_id": "",
            "dry_run": False,
        },
    }


# ---------------------------------------------------------------------------
# Helpers — mock agent factory
# ---------------------------------------------------------------------------

def _mock_agent_class(run_returns):
    """Create a mock agent class whose instances return predetermined results.

    Args:
        run_returns: A single dict or list of dicts.  If a list, successive
                     ``run()`` calls return the next item (side_effect).
    """
    class MockAgent:
        REQUIRED_TOOLS: list[str] = []
        OPTIONAL_TOOLS: list[str] = []

        def __init__(self, llm=None, context=None, **kwargs):
            self.llm = llm
            self.tools: dict = {}

        def bind_tools(self, tool_names, settings, dry_run=False):
            return {name: True for name in tool_names}

        def run(self, user_input: str) -> dict:
            ...  # replaced below

    mock_instance = MagicMock()
    if isinstance(run_returns, list):
        mock_instance.run.side_effect = list(run_returns)
    else:
        mock_instance.run.return_value = run_returns

    # bind_tools should be a no-op
    mock_instance.bind_tools.return_value = {}

    # Make the class return our mock instance
    MockAgentClass = MagicMock(return_value=mock_instance)
    MockAgentClass.REQUIRED_TOOLS = []
    MockAgentClass.OPTIONAL_TOOLS = []
    return MockAgentClass


def _build_agent_map(
    coder_result=None,
    tester_result=None,
    updater_result=None,
):
    """Build a dict of agent_name -> mock class for patching resolve_agent_class."""
    agents = {}
    agents["coder"] = _mock_agent_class(
        coder_result if coder_result is not None else _make_coder_result()
    )
    agents["tester"] = _mock_agent_class(
        tester_result if tester_result is not None else _make_tester_result()
    )
    agents["updater"] = _mock_agent_class(
        updater_result if updater_result is not None else _make_updater_result()
    )
    return agents


def _tasks_from_context(notion_context: dict, sprint_id: str) -> list[dict]:
    """Extract task dicts from a Notion context, matching the CLI layer logic."""
    # Find the sprint
    sprint_notion_id = None
    for s in notion_context.get("sprints", []):
        sprint_notion_id = s["notion_id"]
        break  # test contexts have exactly one sprint

    if sprint_notion_id is None:
        return []

    return [
        {
            "id": wi["notion_id"],
            "notion_id": wi["notion_id"],
            "title": wi.get("name", "Untitled"),
            "description": "",
            "status": wi.get("status", "Ready"),
            "priority": wi.get("priority", ""),
            "estimate_hrs": wi.get("estimate_hrs", 0),
            "type": wi.get("type", "Task"),
        }
        for wi in notion_context.get("work_items", [])
        if wi.get("sprint_id") == sprint_notion_id
    ]


def _patch_and_run(
    agent_map: dict,
    sprint_id: str = "sprint-8",
    goal: str = "Test goal",
    abort_threshold: float = 0.5,
    dry_run: bool = False,
    num_tasks: int = 2,
    tasks: list[dict] | None = None,
) -> SprintState:
    """Patch agents and run the cascade with pre-loaded tasks."""
    def mock_resolve(name, settings=None):
        if name in agent_map:
            return agent_map[name]
        raise ValueError(f"Unknown agent: {name}")

    # Build tasks from notion context if not explicitly provided
    if tasks is None:
        context = _make_notion_context(num_tasks=num_tasks, sprint_id=sprint_id)
        tasks = _tasks_from_context(context, sprint_id)

    with (
        patch("orchestration.cascade.resolve_agent_class", side_effect=mock_resolve),
        patch("orchestration.cascade.get_llm", return_value=MagicMock()),
        patch("orchestration.cascade.resolve_tool_class", side_effect=ValueError("no git")),
    ):
        settings = MagicMock()
        settings.test_repo_dir = None
        settings.aider_repo_dir = None
        settings.data_dir = "data"
        graph = build_cascade_graph(settings, dry_run=dry_run)
        initial_state: SprintState = {
            "sprint_id": sprint_id,
            "plan": {"goal": goal},
            "tasks": tasks,
            "current_task_index": 0,
            "task_results": {},
            "errors": [],
            "iteration_counts": {},
            "status": "planning",
            "failed_task_ids": [],
            "abort_threshold": abort_threshold,
        }
        return graph.invoke(initial_state, config={"recursion_limit": 200})


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_two_tasks_all_succeed(self):
        """plan_node loads 2 tasks from Notion, all agents succeed → completed."""
        agents = _build_agent_map()
        state = _patch_and_run(agents, num_tasks=2)

        assert state["status"] == "completed"
        assert len(state["failed_task_ids"]) == 0
        assert len(state["tasks"]) == 2

        # Both tasks have coder + tester + updater results
        for task in state["tasks"]:
            tid = task["id"]
            assert tid in state["task_results"]
            assert "coder" in state["task_results"][tid]
            assert "tester" in state["task_results"][tid]
            assert "updater" in state["task_results"][tid]

    def test_single_task_succeeds(self):
        """Simplest case: 1 task, all pass."""
        agents = _build_agent_map()
        state = _patch_and_run(agents, num_tasks=1)

        assert state["status"] == "completed"
        assert len(state["failed_task_ids"]) == 0


# ---------------------------------------------------------------------------
# Tests — plan failure (Notion data issues)
# ---------------------------------------------------------------------------

class TestPlanFailure:
    def test_no_tasks_completes_with_no_work(self):
        """Empty task list → sprint completed with no work done."""
        agents = _build_agent_map()
        state = _patch_and_run(agents, tasks=[])

        assert state["status"] == "completed"
        assert len(state["task_results"]) == 0
        assert len(state["failed_task_ids"]) == 0

    def test_empty_sprint_completes(self):
        """Sprint has 0 work items → completed with no work done."""
        agents = _build_agent_map()
        state = _patch_and_run(agents, num_tasks=0)

        assert state["status"] == "completed"
        assert len(state["task_results"]) == 0
        assert len(state["failed_task_ids"]) == 0


# ---------------------------------------------------------------------------
# Tests — task failure and skip
# ---------------------------------------------------------------------------

class TestTaskFailure:
    def test_single_task_failure_skip_continue(self):
        """3 tasks, coder fails on task 2 → task 2 failed, 1+3 complete."""
        coder_results = [
            _make_coder_result(success=True),    # Task 1
            _make_coder_result(success=False),   # Task 2 — fail
            _make_coder_result(success=True),    # Task 3
        ]
        agents = _build_agent_map(
            coder_result=coder_results,
        )
        state = _patch_and_run(agents, num_tasks=3)

        assert state["status"] == "completed"
        assert "wi-2" in state["failed_task_ids"]
        assert len(state["failed_task_ids"]) == 1
        # Tasks 1 and 3 have updater results
        assert "updater" in state["task_results"]["wi-1"]
        assert "updater" in state["task_results"]["wi-3"]

    def test_abort_threshold_exceeded(self):
        """4 tasks, 3 coder failures → >50% → aborted."""
        coder_results = [
            _make_coder_result(success=False),   # Task 1
            _make_coder_result(success=False),   # Task 2
            _make_coder_result(success=False),   # Task 3
            _make_coder_result(success=True),    # Task 4 (never reached)
        ]
        agents = _build_agent_map(
            coder_result=coder_results,
        )
        state = _patch_and_run(agents, num_tasks=4)

        assert state["status"] == "aborted"
        assert len(state["failed_task_ids"]) >= 3


# ---------------------------------------------------------------------------
# Tests — outer retry (test failure → reroute to coder)
# ---------------------------------------------------------------------------

class TestOuterRetry:
    def test_test_failure_triggers_outer_retry(self):
        """Tests fail once, pass on retry → task succeeds."""
        # Coder always succeeds
        coder_results = [
            _make_coder_result(),  # Initial
            _make_coder_result(),  # Retry after test failure
        ]
        # Tester: fail first, pass second
        tester_results = [
            _make_tester_result(test_passed=False),
            _make_tester_result(test_passed=True),
        ]
        agents = _build_agent_map(
            coder_result=coder_results,
            tester_result=tester_results,
        )
        state = _patch_and_run(agents, num_tasks=1)

        assert state["status"] == "completed"
        assert len(state["failed_task_ids"]) == 0
        # Verify outer retry was tracked
        assert state["iteration_counts"].get("outer_coder:wi-1", 0) >= 1
        # Updater should have run
        assert "updater" in state["task_results"]["wi-1"]

    def test_test_failure_exhausts_outer_retries(self):
        """Tests always fail → after MAX_OUTER_RETRIES, task marked failed."""
        # Coder succeeds every time (1 initial + MAX_OUTER_RETRIES retries)
        num_coder_calls = 1 + MAX_OUTER_RETRIES
        coder_results = [_make_coder_result() for _ in range(num_coder_calls)]
        # Tester always fails
        tester_results = [
            _make_tester_result(test_passed=False) for _ in range(num_coder_calls)
        ]
        agents = _build_agent_map(
            coder_result=coder_results,
            tester_result=tester_results,
        )
        state = _patch_and_run(agents, num_tasks=1)

        assert "wi-1" in state["failed_task_ids"]
        assert state["iteration_counts"].get("outer_coder:wi-1", 0) == MAX_OUTER_RETRIES

    def test_tester_infra_failure_skips_task(self):
        """TesterAgent success=False (infra error) → skip, no retry."""
        agents = _build_agent_map(
            tester_result=_make_tester_result(success=False),
        )
        state = _patch_and_run(agents, num_tasks=1)

        assert "wi-1" in state["failed_task_ids"]
        # No outer retries — infra failure goes straight to check
        assert state["iteration_counts"].get("outer_coder:wi-1", 0) == 0


# ---------------------------------------------------------------------------
# Tests — outer retry with test feedback
# ---------------------------------------------------------------------------

class TestTestFeedback:
    def test_code_node_includes_test_feedback_on_retry(self):
        """On outer retry, coder input should contain test_feedback."""
        captured_inputs: list[str] = []

        def capture_coder_run(user_input: str) -> dict:
            captured_inputs.append(user_input)
            return _make_coder_result()

        # Build mock coder that captures input
        coder_cls = MagicMock()
        coder_instance = MagicMock()
        coder_instance.run.side_effect = capture_coder_run
        coder_instance.bind_tools.return_value = {}
        coder_cls.return_value = coder_instance
        coder_cls.REQUIRED_TOOLS = ["aider"]
        coder_cls.OPTIONAL_TOOLS = []

        # Tester: fail first, pass second
        tester_results = [
            _make_tester_result(test_passed=False),
            _make_tester_result(test_passed=True),
        ]

        agent_map = _build_agent_map(
            tester_result=tester_results,
        )
        agent_map["coder"] = coder_cls

        state = _patch_and_run(agent_map, num_tasks=1)

        assert state["status"] == "completed"
        # Second coder call should have test_feedback
        assert len(captured_inputs) == 2
        second_input = json.loads(captured_inputs[1])
        assert "test_feedback" in second_input
        assert "failed" in second_input["test_feedback"]

# ---------------------------------------------------------------------------
# Tests — dry-run mode
# ---------------------------------------------------------------------------

class TestDryRun:
    def test_dry_run_completes(self):
        """Graph completes with dry_run=True."""
        agents = _build_agent_map()
        state = _patch_and_run(agents, dry_run=True, num_tasks=1)

        assert state["status"] == "completed"
        assert len(state["failed_task_ids"]) == 0


# ---------------------------------------------------------------------------
# Tests — routing functions (unit tests)
# ---------------------------------------------------------------------------

class TestRouting:
    def test_route_after_plan_aborted(self):
        from langgraph.graph import END
        state = {"status": "aborted", "tasks": []}
        assert route_after_plan(state) == END

    def test_route_after_plan_no_tasks(self):
        from langgraph.graph import END
        state = {"status": "running", "tasks": []}
        assert route_after_plan(state) == END

    def test_route_after_plan_has_tasks(self):
        state = {"status": "running", "tasks": [{"id": "wi-1"}]}
        assert route_after_plan(state) == "setup_task_node"

    def test_route_after_test_passed(self):
        state = {
            "tasks": [{"id": "wi-1"}],
            "current_task_index": 0,
            "failed_task_ids": [],
            "task_results": {
                "wi-1": {
                    "tester": {
                        "success": True,
                        "partial_output": {"test_passed": True},
                    }
                }
            },
            "iteration_counts": {},
        }
        assert route_after_test(state) == "update_node"

    def test_route_after_test_failed_with_retries(self):
        state = {
            "tasks": [{"id": "wi-1"}],
            "current_task_index": 0,
            "failed_task_ids": [],
            "task_results": {
                "wi-1": {
                    "tester": {
                        "success": True,
                        "partial_output": {"test_passed": False},
                    }
                }
            },
            "iteration_counts": {},
        }
        assert route_after_test(state) == "code_node"

    def test_route_after_test_failed_no_retries(self):
        state = {
            "tasks": [{"id": "wi-1"}],
            "current_task_index": 0,
            "failed_task_ids": [],
            "task_results": {
                "wi-1": {
                    "tester": {
                        "success": True,
                        "partial_output": {"test_passed": False},
                    }
                }
            },
            "iteration_counts": {"outer_coder:wi-1": MAX_OUTER_RETRIES},
        }
        assert route_after_test(state) == "check_node"

    def test_route_after_check_completed(self):
        from langgraph.graph import END
        state = {"status": "completed"}
        assert route_after_check(state) == END

    def test_route_after_check_aborted(self):
        from langgraph.graph import END
        state = {"status": "aborted"}
        assert route_after_check(state) == END

    def test_route_after_check_running(self):
        state = {"status": "running"}
        assert route_after_check(state) == "setup_task_node"


# ---------------------------------------------------------------------------
# Tests — escalation format
# ---------------------------------------------------------------------------

class TestEscalation:
    def test_format_escalation_has_required_sections(self):
        result = {
            "success": False,
            "error_type": "tool",
            "error_message": "Aider timed out",
            "partial_output": {"instruction": "Fix the thing"},
        }
        output = CascadeRunner.format_escalation(
            agent_name="coder",
            task_id="wi-1",
            task_title="Add error handling",
            result=result,
            iteration=3,
            max_iterations=5,
        )
        assert "ESCALATION: coder failed on wi-1" in output
        assert "Error type:  tool" in output
        assert "Message:     Aider timed out" in output
        assert "Iteration:   3 / 5" in output
        assert "wi-1 - Add error handling" in output
        assert "Partial Output" in output
        assert "Suggested Actions" in output

    def test_format_escalation_truncates_large_output(self):
        result = {
            "success": False,
            "error_type": "llm",
            "error_message": "bad output",
            "partial_output": {"data": "x" * 5000},
        }
        output = CascadeRunner.format_escalation(
            "coder", "wi-1", "Task", result, 1, 5,
        )
        assert "truncated" in output


# ---------------------------------------------------------------------------
# commit_push_node tests
# ---------------------------------------------------------------------------


class TestCommitPushNode:
    """Tests for the commit_push_node function."""

    def _make_state(self, task_id="wi-1", modified_files=None):
        """Build a minimal SprintState for commit_push_node."""
        if modified_files is None:
            modified_files = ["src/pipeline.py"]
        return {
            "sprint_id": "s-8",
            "plan": {"sprint": "s-8", "goal": "", "tasks": []},
            "tasks": [{"id": task_id, "title": "Add error handling"}],
            "current_task_index": 0,
            "task_results": {
                task_id: {
                    "coder": {
                        "success": True,
                        "partial_output": {
                            "modified_files": modified_files,
                        },
                    },
                },
            },
            "errors": [],
            "iteration_counts": {},
            "status": "running",
            "failed_task_ids": [],
            "abort_threshold": 0.5,
            "max_tasks": 0,
        }

    @patch("orchestration.cascade.resolve_tool_class")
    def test_commits_and_pushes_with_modified_files(self, mock_resolve):
        mock_git = MagicMock()
        mock_git.commit.return_value = MagicMock(success=True, dry_run=False)
        mock_git.push.return_value = MagicMock(success=True, dry_run=False)
        mock_git.task_branch_name.return_value = "task/sprint-8/wi-1"

        mock_tool_cls = MagicMock(return_value=mock_git)
        mock_resolve.return_value = mock_tool_cls

        settings = MagicMock()
        state = self._make_state(modified_files=["src/pipeline.py"])

        result = commit_push_node(state, settings=settings, dry_run=False)

        mock_git.commit.assert_called_once_with(
            "[cascade] wi-1: Add error handling", ["src/pipeline.py"]
        )
        mock_git.push.assert_called_once_with("task/sprint-8/wi-1")
        assert result["failed_task_ids"] == []
        assert result["task_results"]["wi-1"]["commit_push"]["committed"] is True
        assert result["task_results"]["wi-1"]["commit_push"]["pushed"] is True

    @patch("orchestration.cascade.resolve_tool_class")
    def test_commits_all_when_no_modified_files(self, mock_resolve):
        mock_git = MagicMock()
        mock_git.commit.return_value = MagicMock(success=True, dry_run=False)
        mock_git.push.return_value = MagicMock(success=True, dry_run=False)
        mock_git.task_branch_name.return_value = "task/sprint-8/wi-1"

        mock_tool_cls = MagicMock(return_value=mock_git)
        mock_resolve.return_value = mock_tool_cls

        settings = MagicMock()
        state = self._make_state(modified_files=[])

        result = commit_push_node(state, settings=settings, dry_run=False)

        # Should call commit without file list (git add -A)
        mock_git.commit.assert_called_once_with("[cascade] wi-1: Add error handling")
        assert result["failed_task_ids"] == []

    @patch("orchestration.cascade.resolve_tool_class")
    def test_commit_failure_does_not_fail_task(self, mock_resolve):
        mock_git = MagicMock()
        mock_git.commit.return_value = MagicMock(
            success=False, dry_run=False, error="nothing to commit"
        )

        mock_tool_cls = MagicMock(return_value=mock_git)
        mock_resolve.return_value = mock_tool_cls

        settings = MagicMock()
        state = self._make_state()

        result = commit_push_node(state, settings=settings, dry_run=False)

        assert result["failed_task_ids"] == []
        assert result["task_results"]["wi-1"]["commit_push"]["committed"] is False

    @patch("orchestration.cascade.resolve_tool_class")
    def test_no_git_provider_skips_gracefully(self, mock_resolve):
        mock_resolve.side_effect = Exception("no provider")

        settings = MagicMock()
        state = self._make_state()

        result = commit_push_node(state, settings=settings, dry_run=False)

        assert result["failed_task_ids"] == []
        assert result["errors"] == []


# ---------------------------------------------------------------------------
# Tests — RAG / snapshot passthrough
# ---------------------------------------------------------------------------

class TestRAGSnapshotPassthrough:
    """Verify that rag and snapshot are passed through to agent constructors."""

    def test_code_node_passes_rag_snapshot(self):
        """code_node should forward rag and snapshot kwargs to agent constructor."""
        agents = _build_agent_map()
        mock_rag = MagicMock()
        mock_snapshot = MagicMock()

        context = _make_notion_context(num_tasks=1)
        tasks = _tasks_from_context(context, "sprint-8")

        def mock_resolve(name, settings=None):
            if name in agents:
                return agents[name]
            raise ValueError(f"Unknown agent: {name}")

        with (
            patch("orchestration.cascade.resolve_agent_class", side_effect=mock_resolve),
            patch("orchestration.cascade.get_llm", return_value=MagicMock()),
            patch("orchestration.cascade.resolve_tool_class", side_effect=ValueError("no git")),
        ):
            settings = MagicMock()
            settings.test_repo_dir = None
            settings.aider_repo_dir = None
            settings.data_dir = "data"
            graph = build_cascade_graph(
                settings, dry_run=False, rag=mock_rag, snapshot=mock_snapshot
            )
            initial_state: SprintState = {
                "sprint_id": "sprint-8",
                "plan": {"goal": "Test goal"},
                "tasks": tasks,
                "current_task_index": 0,
                "task_results": {},
                "errors": [],
                "iteration_counts": {},
                "status": "planning",
                "failed_task_ids": [],
                "abort_threshold": 0.5,
            }
            graph.invoke(initial_state, config={"recursion_limit": 200})

        # Verify coder agent was instantiated with rag and snapshot
        agents["coder"].assert_called()
        call_kwargs = agents["coder"].call_args
        assert call_kwargs.kwargs.get("rag") is mock_rag
        assert call_kwargs.kwargs.get("snapshot") is mock_snapshot

    def test_cascade_runner_passes_rag_snapshot(self):
        """CascadeRunner should thread rag/snapshot to build_cascade_graph."""
        mock_rag = MagicMock()
        mock_snapshot = MagicMock()

        with patch("orchestration.runner.build_cascade_graph") as mock_build:
            mock_build.return_value = MagicMock()
            CascadeRunner(
                settings=MagicMock(),
                dry_run=False,
                rag=mock_rag,
                snapshot=mock_snapshot,
            )

        mock_build.assert_called_once()
        call_kwargs = mock_build.call_args
        assert call_kwargs.kwargs.get("rag") is mock_rag
        assert call_kwargs.kwargs.get("snapshot") is mock_snapshot
