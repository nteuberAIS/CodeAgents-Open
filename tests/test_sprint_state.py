"""Tests for SprintState TypedDict and pure helper functions."""

from __future__ import annotations

import pytest

from schemas.sprint_state import (
    SprintState,
    advance_task,
    create_initial_state,
    get_current_task,
    get_iteration_count,
    increment_iteration,
    mark_task_failed,
    record_task_result,
    should_abort,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(num_tasks: int = 3) -> dict:
    """Return a minimal plan dict matching SprintPlanner's partial_output."""
    return {
        "sprint": 8,
        "goal": "Implement cascade orchestrator",
        "tasks": [
            {"id": f"SP8-{i:03d}", "title": f"Task {i}", "status": "Ready"}
            for i in range(1, num_tasks + 1)
        ],
        "dependencies": [],
    }


def _seed(num_tasks: int = 3, **overrides) -> SprintState:
    """Shortcut: create initial state and apply any field overrides."""
    state = create_initial_state(_make_plan(num_tasks), "sprint-8")
    if overrides:
        state = {**state, **overrides}
    return state


# ---------------------------------------------------------------------------
# create_initial_state
# ---------------------------------------------------------------------------

class TestCreateInitialState:
    def test_valid_plan(self):
        plan = _make_plan(3)
        state = create_initial_state(plan, "sprint-8")

        assert state["sprint_id"] == "sprint-8"
        assert state["plan"] is plan
        assert len(state["tasks"]) == 3
        assert state["current_task_index"] == 0
        assert state["task_results"] == {}
        assert state["errors"] == []
        assert state["iteration_counts"] == {}
        assert state["status"] == "running"
        assert state["failed_task_ids"] == []
        assert state["abort_threshold"] == 0.5

    def test_zero_tasks(self):
        plan = _make_plan(0)
        state = create_initial_state(plan, "sprint-0")

        assert state["tasks"] == []
        assert state["status"] == "running"

    def test_custom_abort_threshold(self):
        state = create_initial_state(_make_plan(), "sprint-8", abort_threshold=0.75)
        assert state["abort_threshold"] == 0.75

    def test_tasks_are_copied(self):
        """Mutating the returned tasks list must not affect the plan."""
        plan = _make_plan(2)
        state = create_initial_state(plan, "s")
        state["tasks"].append({"id": "extra"})
        assert len(plan["tasks"]) == 2


# ---------------------------------------------------------------------------
# get_current_task
# ---------------------------------------------------------------------------

class TestGetCurrentTask:
    def test_returns_first_task(self):
        state = _seed()
        task = get_current_task(state)
        assert task is not None
        assert task["id"] == "SP8-001"

    def test_skips_failed_tasks(self):
        state = _seed(failed_task_ids=["SP8-001"])
        task = get_current_task(state)
        assert task is not None
        assert task["id"] == "SP8-002"

    def test_returns_none_when_all_processed(self):
        state = _seed(current_task_index=3)
        assert get_current_task(state) is None

    def test_returns_none_when_all_failed(self):
        state = _seed(failed_task_ids=["SP8-001", "SP8-002", "SP8-003"])
        assert get_current_task(state) is None

    def test_skips_multiple_consecutive_failed(self):
        state = _seed(failed_task_ids=["SP8-001", "SP8-002"])
        task = get_current_task(state)
        assert task is not None
        assert task["id"] == "SP8-003"


# ---------------------------------------------------------------------------
# advance_task
# ---------------------------------------------------------------------------

class TestAdvanceTask:
    def test_increments_index(self):
        state = _seed()
        new = advance_task(state)
        assert new["current_task_index"] == 1
        # Original unchanged
        assert state["current_task_index"] == 0

    def test_does_not_clamp(self):
        """Index can go past the end — get_current_task returns None."""
        state = _seed(current_task_index=2)
        new = advance_task(state)
        assert new["current_task_index"] == 3
        assert get_current_task(new) is None


# ---------------------------------------------------------------------------
# mark_task_failed
# ---------------------------------------------------------------------------

class TestMarkTaskFailed:
    def test_adds_to_failed_and_errors(self):
        state = _seed()
        new = mark_task_failed(state, "SP8-001", "Code gen failed")
        assert "SP8-001" in new["failed_task_ids"]
        assert "Code gen failed" in new["errors"]
        # Original unchanged
        assert state["failed_task_ids"] == []
        assert state["errors"] == []

    def test_idempotent_marking(self):
        state = _seed()
        s1 = mark_task_failed(state, "SP8-001", "err1")
        s2 = mark_task_failed(s1, "SP8-001", "err2")
        # Both entries present (no dedup — that's the orchestrator's job)
        assert s2["failed_task_ids"].count("SP8-001") == 2
        assert len(s2["errors"]) == 2


# ---------------------------------------------------------------------------
# should_abort
# ---------------------------------------------------------------------------

class TestShouldAbort:
    def test_below_threshold(self):
        state = _seed(4, failed_task_ids=["SP8-001"])  # 1/4 = 25%
        assert should_abort(state) is False

    def test_above_threshold(self):
        state = _seed(4, failed_task_ids=["SP8-001", "SP8-002", "SP8-003"])  # 75%
        assert should_abort(state) is True

    def test_at_threshold_does_not_abort(self):
        """Exactly 50% is *not* above the 0.5 threshold (strict >)."""
        state = _seed(4, failed_task_ids=["SP8-001", "SP8-002"])  # 50%
        assert should_abort(state) is False

    def test_empty_tasks(self):
        state = _seed(0)
        assert should_abort(state) is False

    def test_single_task_fails(self):
        """1/1 = 100% > 50% → abort."""
        state = _seed(1, failed_task_ids=["SP8-001"])
        assert should_abort(state) is True

    def test_custom_threshold(self):
        state = _seed(4, abort_threshold=0.75, failed_task_ids=["SP8-001", "SP8-002", "SP8-003"])
        assert should_abort(state) is False  # 75% is not > 75%


# ---------------------------------------------------------------------------
# record_task_result
# ---------------------------------------------------------------------------

class TestRecordTaskResult:
    def test_first_result(self):
        state = _seed()
        result = {"success": True, "partial_output": {"branch": "feature/SP8-001"}}
        new = record_task_result(state, "SP8-001", "coder", result)

        assert "SP8-001" in new["task_results"]
        assert new["task_results"]["SP8-001"]["coder"] == result
        # Original unchanged
        assert state["task_results"] == {}

    def test_second_agent_merges(self):
        state = _seed()
        s1 = record_task_result(state, "SP8-001", "coder", {"success": True})
        s2 = record_task_result(s1, "SP8-001", "tester", {"success": True, "tests_passed": 5})

        assert "coder" in s2["task_results"]["SP8-001"]
        assert "tester" in s2["task_results"]["SP8-001"]
        assert s2["task_results"]["SP8-001"]["tester"]["tests_passed"] == 5

    def test_different_tasks(self):
        state = _seed()
        s1 = record_task_result(state, "SP8-001", "coder", {"a": 1})
        s2 = record_task_result(s1, "SP8-002", "coder", {"b": 2})

        assert s2["task_results"]["SP8-001"]["coder"] == {"a": 1}
        assert s2["task_results"]["SP8-002"]["coder"] == {"b": 2}


# ---------------------------------------------------------------------------
# get_iteration_count / increment_iteration
# ---------------------------------------------------------------------------

class TestIterationTracking:
    def test_unseen_returns_zero(self):
        state = _seed()
        assert get_iteration_count(state, "coder", "SP8-001") == 0

    def test_increment_from_zero(self):
        state = _seed()
        new = increment_iteration(state, "coder", "SP8-001")
        assert get_iteration_count(new, "coder", "SP8-001") == 1
        # Original unchanged
        assert get_iteration_count(state, "coder", "SP8-001") == 0

    def test_increment_from_one(self):
        state = _seed()
        s1 = increment_iteration(state, "coder", "SP8-001")
        s2 = increment_iteration(s1, "coder", "SP8-001")
        assert get_iteration_count(s2, "coder", "SP8-001") == 2

    def test_different_agents_independent(self):
        state = _seed()
        s1 = increment_iteration(state, "coder", "SP8-001")
        s2 = increment_iteration(s1, "tester", "SP8-001")
        assert get_iteration_count(s2, "coder", "SP8-001") == 1
        assert get_iteration_count(s2, "tester", "SP8-001") == 1

    def test_different_tasks_independent(self):
        state = _seed()
        s1 = increment_iteration(state, "coder", "SP8-001")
        s2 = increment_iteration(s1, "coder", "SP8-002")
        assert get_iteration_count(s2, "coder", "SP8-001") == 1
        assert get_iteration_count(s2, "coder", "SP8-002") == 1
