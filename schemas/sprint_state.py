"""Shared state schema for the LangGraph cascade orchestrator.

SprintState is the TypedDict that flows between agent nodes
(Planner → Coder → Tester → Updater), accumulating results and
tracking failures.  Helper functions are pure — they take state in
and return new state out, with no side-effects.

TypedDict (not Pydantic) is used because LangGraph nodes return partial
dicts that the graph framework merges; TypedDict aligns naturally with
this pattern.  Agent outputs are already validated through AgentResult
before entering state.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


# ---------------------------------------------------------------------------
# Sprint state
# ---------------------------------------------------------------------------

class SprintState(TypedDict):
    """Shared state passed through every node in the cascade graph."""

    sprint_id: str
    plan: dict                                          # Full SprintPlanner partial_output
    tasks: list[dict]                                   # Individual task dicts from plan["tasks"]
    current_task_index: int                             # Index into tasks for current task
    task_results: dict[str, dict]                       # task_id → {agent_name: result, …}
    errors: Annotated[list[str], operator.add]           # Accumulated error messages
    iteration_counts: dict[str, int]                    # "agent:task_id" → retry count
    status: str                                         # running | completed | aborted | escalated
    failed_task_ids: Annotated[list[str], operator.add]  # Task IDs that were skipped
    abort_threshold: float                              # Fraction of tasks that can fail (default 0.5)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

def create_initial_state(
    plan_output: dict,
    sprint_id: str,
    abort_threshold: float = 0.5,
) -> SprintState:
    """Seed state from SprintPlanner's ``partial_output``."""
    return SprintState(
        sprint_id=sprint_id,
        plan=plan_output,
        tasks=list(plan_output.get("tasks", [])),
        current_task_index=0,
        task_results={},
        errors=[],
        iteration_counts={},
        status="running",
        failed_task_ids=[],
        abort_threshold=abort_threshold,
    )


def get_current_task(state: SprintState) -> dict | None:
    """Return the current task, skipping failed ones.  ``None`` when done."""
    tasks = state["tasks"]
    failed = set(state["failed_task_ids"])
    idx = state["current_task_index"]
    while idx < len(tasks):
        task = tasks[idx]
        if task.get("id") not in failed:
            return task
        idx += 1
    return None


def advance_task(state: SprintState) -> SprintState:
    """Return state with ``current_task_index`` incremented by one."""
    new_index = state["current_task_index"] + 1
    return {**state, "current_task_index": new_index}


def mark_task_failed(
    state: SprintState, task_id: str, error: str,
) -> SprintState:
    """Add *task_id* to failures and append *error* to the error log."""
    return {
        **state,
        "failed_task_ids": [*state["failed_task_ids"], task_id],
        "errors": [*state["errors"], error],
    }


def should_abort(state: SprintState) -> bool:
    """``True`` when the fraction of failed tasks exceeds the threshold."""
    total = len(state["tasks"])
    if total == 0:
        return False
    return len(state["failed_task_ids"]) / total > state["abort_threshold"]


def record_task_result(
    state: SprintState,
    task_id: str,
    agent_name: str,
    result: dict,
) -> SprintState:
    """Merge *result* into ``task_results[task_id]`` under *agent_name*."""
    existing = dict(state["task_results"])
    task_entry = dict(existing.get(task_id, {}))
    task_entry[agent_name] = result
    existing[task_id] = task_entry
    return {**state, "task_results": existing}


def get_iteration_count(
    state: SprintState, agent_name: str, task_id: str,
) -> int:
    """Return current retry count for this agent + task (0 if unseen)."""
    key = f"{agent_name}:{task_id}"
    return state["iteration_counts"].get(key, 0)


def increment_iteration(
    state: SprintState, agent_name: str, task_id: str,
) -> SprintState:
    """Return state with the retry count for *agent_name*:*task_id* bumped."""
    key = f"{agent_name}:{task_id}"
    counts = dict(state["iteration_counts"])
    counts[key] = counts.get(key, 0) + 1
    return {**state, "iteration_counts": counts}
