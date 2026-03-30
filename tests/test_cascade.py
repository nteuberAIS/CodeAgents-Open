"""Tests for the LangGraph cascade orchestrator.

All agents are mocked — these tests verify graph traversal logic,
conditional routing, retry behavior, and state transitions.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from orchestration.cascade import (
    MAX_OUTER_RETRIES,
    build_cascade_graph,
    route_after_check,
    route_after_plan,
    route_after_test,
)
from orchestration.runner import CascadeRunner
from schemas.sprint_state import SprintState


# ---------------------------------------------------------------------------
# Helpers — result factories
# ---------------------------------------------------------------------------

def _make_plan_result(num_tasks: int = 2, success: bool = True) -> dict:
    """SprintPlannerAgent result envelope."""
    if not success:
        return {
            "success": False,
            "error_type": "llm",
            "error_message": "LLM produced invalid JSON",
            "partial_output": {"raw_output": "bad json"},
        }
    return {
        "success": True,
        "error_type": None,
        "error_message": None,
        "partial_output": {
            "sprint": 8,
            "goal": "Test sprint",
            "tasks": [
                {
                    "id": f"SP8-{i:03d}",
                    "title": f"Task {i}",
                    "description": f"Do thing {i}",
                    "estimate_hrs": 2,
                    "status": "todo",
                }
                for i in range(1, num_tasks + 1)
            ],
            "dependencies": [],
        },
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
            },
        }
    return {
        "success": True,
        "error_type": None,
        "error_message": None,
        "partial_output": {
            "pr_created": True,
            "pr_url": "https://github.com/org/repo/pull/42",
            "pr_title": "SP8-001: Add error handling",
            "notion_updated": True,
            "notion_status": "In Review",
            "task_id": "",
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

        def __init__(self, llm=None, context=None):
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
    plan_result=None,
    coder_result=None,
    tester_result=None,
    updater_result=None,
):
    """Build a dict of agent_name -> mock class for patching resolve_agent_class."""
    agents = {}
    agents["sprint_planner"] = _mock_agent_class(
        plan_result if plan_result is not None else _make_plan_result()
    )
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


def _patch_and_run(
    agent_map: dict,
    sprint_id: str = "sprint-8",
    goal: str = "Test goal",
    abort_threshold: float = 0.5,
    dry_run: bool = False,
) -> SprintState:
    """Patch resolve_agent_class, get_llm, resolve_tool_class and run the cascade."""
    def mock_resolve(name, settings=None):
        if name in agent_map:
            return agent_map[name]
        raise ValueError(f"Unknown agent: {name}")

    with (
        patch("orchestration.cascade.resolve_agent_class", side_effect=mock_resolve),
        patch("orchestration.cascade.get_llm", return_value=MagicMock()),
        patch("orchestration.cascade.resolve_tool_class", side_effect=ValueError("no git")),
    ):
        settings = MagicMock()
        settings.test_repo_dir = None
        settings.aider_repo_dir = None
        graph = build_cascade_graph(settings, dry_run=dry_run)
        initial_state: SprintState = {
            "sprint_id": sprint_id,
            "plan": {"goal": goal},
            "tasks": [],
            "current_task_index": 0,
            "task_results": {},
            "errors": [],
            "iteration_counts": {},
            "status": "planning",
            "failed_task_ids": [],
            "abort_threshold": abort_threshold,
        }
        return graph.invoke(initial_state, config={"recursion_limit": 100})


# ---------------------------------------------------------------------------
# Tests — happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_two_tasks_all_succeed(self):
        """Planner returns 2 tasks, all agents succeed → completed."""
        agents = _build_agent_map(plan_result=_make_plan_result(num_tasks=2))
        state = _patch_and_run(agents)

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
        agents = _build_agent_map(plan_result=_make_plan_result(num_tasks=1))
        state = _patch_and_run(agents)

        assert state["status"] == "completed"
        assert len(state["failed_task_ids"]) == 0


# ---------------------------------------------------------------------------
# Tests — plan failure
# ---------------------------------------------------------------------------

class TestPlanFailure:
    def test_plan_failure_aborts_immediately(self):
        """Planner returns success=False → sprint aborted, no tasks processed."""
        agents = _build_agent_map(plan_result=_make_plan_result(success=False))
        state = _patch_and_run(agents)

        assert state["status"] == "aborted"
        assert len(state["task_results"]) == 0
        assert any("SprintPlanner failed" in e for e in state["errors"])

    def test_empty_plan_completes(self):
        """Planner succeeds with 0 tasks → completed with no work done."""
        agents = _build_agent_map(plan_result=_make_plan_result(num_tasks=0))
        state = _patch_and_run(agents)

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
            plan_result=_make_plan_result(num_tasks=3),
            coder_result=coder_results,
        )
        state = _patch_and_run(agents)

        assert state["status"] == "completed"
        assert "SP8-002" in state["failed_task_ids"]
        assert len(state["failed_task_ids"]) == 1
        # Tasks 1 and 3 have updater results
        assert "updater" in state["task_results"]["SP8-001"]
        assert "updater" in state["task_results"]["SP8-003"]

    def test_abort_threshold_exceeded(self):
        """4 tasks, 3 coder failures → >50% → aborted."""
        coder_results = [
            _make_coder_result(success=False),   # Task 1
            _make_coder_result(success=False),   # Task 2
            _make_coder_result(success=False),   # Task 3
            _make_coder_result(success=True),    # Task 4 (never reached)
        ]
        agents = _build_agent_map(
            plan_result=_make_plan_result(num_tasks=4),
            coder_result=coder_results,
        )
        state = _patch_and_run(agents)

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
            plan_result=_make_plan_result(num_tasks=1),
            coder_result=coder_results,
            tester_result=tester_results,
        )
        state = _patch_and_run(agents)

        assert state["status"] == "completed"
        assert len(state["failed_task_ids"]) == 0
        # Verify outer retry was tracked
        assert state["iteration_counts"].get("outer_coder:SP8-001", 0) >= 1
        # Updater should have run
        assert "updater" in state["task_results"]["SP8-001"]

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
            plan_result=_make_plan_result(num_tasks=1),
            coder_result=coder_results,
            tester_result=tester_results,
        )
        state = _patch_and_run(agents)

        assert "SP8-001" in state["failed_task_ids"]
        assert state["iteration_counts"].get("outer_coder:SP8-001", 0) == MAX_OUTER_RETRIES

    def test_tester_infra_failure_skips_task(self):
        """TesterAgent success=False (infra error) → skip, no retry."""
        agents = _build_agent_map(
            plan_result=_make_plan_result(num_tasks=1),
            tester_result=_make_tester_result(success=False),
        )
        state = _patch_and_run(agents)

        assert "SP8-001" in state["failed_task_ids"]
        # No outer retries — infra failure goes straight to check
        assert state["iteration_counts"].get("outer_coder:SP8-001", 0) == 0


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
            plan_result=_make_plan_result(num_tasks=1),
            tester_result=tester_results,
        )
        agent_map["coder"] = coder_cls

        state = _patch_and_run(agent_map)

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
        agents = _build_agent_map(plan_result=_make_plan_result(num_tasks=1))
        state = _patch_and_run(agents, dry_run=True)

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
        state = {"status": "running", "tasks": [{"id": "SP8-001"}]}
        assert route_after_plan(state) == "setup_task_node"

    def test_route_after_test_passed(self):
        state = {
            "tasks": [{"id": "SP8-001"}],
            "current_task_index": 0,
            "failed_task_ids": [],
            "task_results": {
                "SP8-001": {
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
            "tasks": [{"id": "SP8-001"}],
            "current_task_index": 0,
            "failed_task_ids": [],
            "task_results": {
                "SP8-001": {
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
            "tasks": [{"id": "SP8-001"}],
            "current_task_index": 0,
            "failed_task_ids": [],
            "task_results": {
                "SP8-001": {
                    "tester": {
                        "success": True,
                        "partial_output": {"test_passed": False},
                    }
                }
            },
            "iteration_counts": {"outer_coder:SP8-001": MAX_OUTER_RETRIES},
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
            task_id="SP8-001",
            task_title="Add error handling",
            result=result,
            iteration=3,
            max_iterations=5,
        )
        assert "ESCALATION: coder failed on SP8-001" in output
        assert "Error type:  tool" in output
        assert "Message:     Aider timed out" in output
        assert "Iteration:   3 / 5" in output
        assert "SP8-001 - Add error handling" in output
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
            "coder", "SP8-001", "Task", result, 1, 5,
        )
        assert "truncated" in output
