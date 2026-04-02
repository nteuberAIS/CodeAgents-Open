"""LangGraph StateGraph definition for the sprint cascade.

The cascade processes tasks sequentially:
    plan_node → (for each task) setup → code → commit_push → test → update → check → next

Two reflection layers:
    Inner  — CoderAgent retries Aider internally (up to MAX_ITERATIONS=5).
    Outer  — Orchestrator routes test failures back to code_node (up to
             MAX_OUTER_RETRIES=2), giving CoderAgent the test output as context.
"""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from config.settings import get_llm, get_settings, resolve_agent_class, resolve_tool_class
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

logger = logging.getLogger(__name__)

MAX_OUTER_RETRIES = 2


def _sprint_branch_prefix(sprint_id: str) -> str:
    """Convert CLI sprint_id to branch prefix.

    Examples: 's-1.4' -> '1.4', 'sprint-8' -> '8', 'sprint-20260401' -> '20260401'
    """
    sid = sprint_id
    for prefix in ("s-", "sprint-"):
        if sid.startswith(prefix):
            sid = sid[len(prefix):]
            break
    return sid


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def plan_node(state: SprintState, *, settings: Any, dry_run: bool) -> dict:
    """Validate pre-loaded tasks and seed the cascade state.

    Tasks are loaded by the CLI layer (main.py) and passed via initial state.
    This node validates the input, applies max_tasks truncation, and builds
    the plan dict for downstream nodes.
    """
    sprint_id = state["sprint_id"]
    tasks = state.get("tasks", [])
    print(f"[cascade] plan: {len(tasks)} tasks for sprint {sprint_id}")

    if not tasks:
        return {
            "status": "completed",
            "plan": {"sprint": sprint_id, "goal": state.get("plan", {}).get("goal", ""), "tasks": []},
            "tasks": [],
            "current_task_index": 0,
            "task_results": {},
            "iteration_counts": {},
            "errors": [],
            "failed_task_ids": [],
        }

    # Truncate if max_tasks is set
    max_tasks = state.get("max_tasks", 0)
    if max_tasks > 0:
        tasks = tasks[:max_tasks]

    plan_output = {
        "sprint": sprint_id,
        "goal": state.get("plan", {}).get("goal", ""),
        "tasks": tasks,
        "dependencies": [],
    }

    new_state = create_initial_state(
        plan_output,
        sprint_id,
        state.get("abort_threshold", 0.5),
    )
    if max_tasks > 0:
        new_state["tasks"] = new_state["tasks"][:max_tasks]

    status = "completed" if not new_state["tasks"] else new_state["status"]
    return {
        "plan": new_state["plan"],
        "tasks": new_state["tasks"],
        "current_task_index": new_state["current_task_index"],
        "task_results": new_state["task_results"],
        "iteration_counts": new_state["iteration_counts"],
        "status": status,
        "errors": [],
        "failed_task_ids": [],
    }


def setup_task_node(state: SprintState, *, settings: Any, dry_run: bool) -> dict:
    """Prepare git branch for the current task."""
    task = get_current_task(state)
    if task is None:
        return {"errors": [], "failed_task_ids": []}

    sprint_prefix = _sprint_branch_prefix(state["sprint_id"])
    task_id = task["id"]
    print(f"[cascade] setup: preparing branch for {task_id} — {task.get('title', '')}")

    # Try to create and checkout a task branch (non-fatal on failure)
    for provider in ("github", "azdevops"):
        try:
            tool_cls = resolve_tool_class(provider, settings)
            git_tool = tool_cls(settings=settings, dry_run=dry_run)
            branch_name = git_tool.task_branch_name(sprint_prefix, task_id)
            create_result = git_tool.create_branch(branch_name, from_ref="main")
            if not create_result.success and not create_result.dry_run:
                # Branch might already exist — try checkout
                git_tool.checkout_branch(branch_name)
            logger.info("Checked out branch %s for task %s", branch_name, task_id)
            return {"errors": [], "failed_task_ids": []}
        except Exception as exc:  # noqa: BLE001
            logger.debug("Git provider %s unavailable: %s", provider, exc)
            continue

    logger.warning("No git provider available — coding on current branch")
    return {"errors": [], "failed_task_ids": []}


def code_node(state: SprintState, *, settings: Any, dry_run: bool, rag: Any = None, snapshot: Any = None) -> dict:
    """Run CoderAgent on the current task, with optional test-failure feedback."""
    task = get_current_task(state)
    if task is None:
        return {"errors": [], "failed_task_ids": []}

    task_id = task["id"]
    print(f"[cascade] code: running CoderAgent on {task_id} — {task.get('title', '')}")

    # Build task input
    task_input = dict(task)
    existing_results = state.get("task_results", {}).get(task_id, {})

    # Detect outer retry: if tester results already exist for this task
    tester_result = existing_results.get("tester")
    updated_counts = dict(state.get("iteration_counts", {}))

    if tester_result is not None:
        # This is an outer retry — append test feedback
        test_output = tester_result.get("partial_output", {}).get("test_output", "")
        task_input["test_feedback"] = test_output
        updated_counts = dict(
            increment_iteration(state, "outer_coder", task_id)["iteration_counts"]
        )

    agent_cls = resolve_agent_class("coder", settings)
    llm = get_llm(settings, agent_name="coder")
    agent = agent_cls(llm=llm, rag=rag, snapshot=snapshot)
    tool_names = getattr(agent_cls, "REQUIRED_TOOLS", []) + getattr(agent_cls, "OPTIONAL_TOOLS", [])
    if tool_names:
        agent.bind_tools(tool_names, settings, dry_run=dry_run)

    result = agent.run(json.dumps(task_input))

    # Record result
    updated_results = dict(state.get("task_results", {}))
    task_entry = dict(updated_results.get(task_id, {}))
    task_entry["coder"] = result
    updated_results[task_id] = task_entry

    if not result.get("success"):
        error_msg = f"CoderAgent failed on {task_id}: {result.get('error_message', 'unknown')}"
        logger.error(error_msg)
        return {
            "task_results": updated_results,
            "iteration_counts": updated_counts,
            "errors": [error_msg],
            "failed_task_ids": [task_id],
        }

    return {
        "task_results": updated_results,
        "iteration_counts": updated_counts,
        "errors": [],
        "failed_task_ids": [],
    }


def commit_push_node(state: SprintState, *, settings: Any, dry_run: bool) -> dict:
    """Commit Aider's file changes and push the task branch to the remote."""
    task = get_current_task(state)
    if task is None:
        return {"errors": [], "failed_task_ids": []}

    task_id = task["id"]
    task_title = task.get("title", "")
    sprint_prefix = _sprint_branch_prefix(state["sprint_id"])

    # Get modified files from coder result
    coder_result = state.get("task_results", {}).get(task_id, {}).get("coder", {})
    modified_files = coder_result.get("partial_output", {}).get("modified_files", [])

    # Resolve a git tool (same pattern as setup_task_node)
    git_tool = None
    for provider in ("azdevops", "github"):
        try:
            tool_cls = resolve_tool_class(provider, settings)
            git_tool = tool_cls(settings=settings, dry_run=dry_run)
            break
        except Exception:  # noqa: BLE001
            continue

    if git_tool is None:
        logger.warning("No git provider available — skipping commit+push")
        return {"errors": [], "failed_task_ids": []}

    # Commit
    message = f"[cascade] {task_id}: {task_title}"
    try:
        if modified_files:
            commit_result = git_tool.commit(message, modified_files)
        else:
            commit_result = git_tool.commit(message)

        if not commit_result.success and not commit_result.dry_run:
            logger.warning("Commit failed for %s: %s", task_id, commit_result.error)
            # Record error but don't fail the task
            updated_results = dict(state.get("task_results", {}))
            task_entry = dict(updated_results.get(task_id, {}))
            task_entry["commit_push"] = {"error": commit_result.error, "committed": False, "pushed": False}
            updated_results[task_id] = task_entry
            return {"task_results": updated_results, "errors": [], "failed_task_ids": []}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Commit exception for %s: %s", task_id, exc)
        updated_results = dict(state.get("task_results", {}))
        task_entry = dict(updated_results.get(task_id, {}))
        task_entry["commit_push"] = {"error": str(exc), "committed": False, "pushed": False}
        updated_results[task_id] = task_entry
        return {"task_results": updated_results, "errors": [], "failed_task_ids": []}

    # Push
    branch_name = git_tool.task_branch_name(sprint_prefix, task_id)
    try:
        push_result = git_tool.push(branch_name)
        if not push_result.success and not push_result.dry_run:
            logger.warning("Push failed for %s: %s", task_id, push_result.error)
            updated_results = dict(state.get("task_results", {}))
            task_entry = dict(updated_results.get(task_id, {}))
            task_entry["commit_push"] = {"error": push_result.error, "committed": True, "pushed": False}
            updated_results[task_id] = task_entry
            return {"task_results": updated_results, "errors": [], "failed_task_ids": []}
    except Exception as exc:  # noqa: BLE001
        logger.warning("Push exception for %s: %s", task_id, exc)
        updated_results = dict(state.get("task_results", {}))
        task_entry = dict(updated_results.get(task_id, {}))
        task_entry["commit_push"] = {"error": str(exc), "committed": True, "pushed": False}
        updated_results[task_id] = task_entry
        return {"task_results": updated_results, "errors": [], "failed_task_ids": []}

    print(f"[cascade] commit_push: committed and pushed {branch_name}")

    # Record success
    updated_results = dict(state.get("task_results", {}))
    task_entry = dict(updated_results.get(task_id, {}))
    task_entry["commit_push"] = {"committed": True, "pushed": True, "branch": branch_name}
    updated_results[task_id] = task_entry

    return {"task_results": updated_results, "errors": [], "failed_task_ids": []}


def test_node(state: SprintState, *, settings: Any, dry_run: bool, rag: Any = None, snapshot: Any = None) -> dict:
    """Run TesterAgent on the current task."""
    task = get_current_task(state)
    if task is None:
        return {"errors": [], "failed_task_ids": []}

    task_id = task["id"]
    print(f"[cascade] test: running TesterAgent on {task_id}")
    test_input = {
        "task_id": task_id,
        "task_title": task.get("title", ""),
    }
    # Pass repo_dir from settings if available
    repo_dir = getattr(settings, "test_repo_dir", None) or getattr(settings, "aider_repo_dir", None)
    if repo_dir:
        test_input["repo_dir"] = repo_dir

    agent_cls = resolve_agent_class("tester", settings)
    llm = get_llm(settings, agent_name="tester")
    agent = agent_cls(llm=llm, rag=rag, snapshot=snapshot)

    result = agent.run(json.dumps(test_input))

    # Record result
    updated_results = dict(state.get("task_results", {}))
    task_entry = dict(updated_results.get(task_id, {}))
    task_entry["tester"] = result
    updated_results[task_id] = task_entry

    if not result.get("success"):
        error_msg = f"TesterAgent infra failure on {task_id}: {result.get('error_message', 'unknown')}"
        logger.error(error_msg)
        return {
            "task_results": updated_results,
            "errors": [error_msg],
            "failed_task_ids": [],
        }

    return {
        "task_results": updated_results,
        "errors": [],
        "failed_task_ids": [],
    }


def update_node(state: SprintState, *, settings: Any, dry_run: bool, rag: Any = None, snapshot: Any = None) -> dict:
    """Run UpdaterAgent — create PR, update Notion status."""
    task = get_current_task(state)
    if task is None:
        return {"errors": [], "failed_task_ids": []}

    task_id = task["id"]
    sprint_prefix = _sprint_branch_prefix(state["sprint_id"])
    print(f"[cascade] update: creating PR for {task_id} — {task.get('title', '')}")

    # Build updater input from task + accumulated results
    coder_result = state.get("task_results", {}).get(task_id, {}).get("coder", {})
    tester_result = state.get("task_results", {}).get(task_id, {}).get("tester", {})

    update_input = {
        "task_id": task_id,
        "task_title": task.get("title", ""),
        "task_description": task.get("description", ""),
        "source_branch": f"task/sprint-{sprint_prefix}/{task_id}",
        "target_branch": f"sprint-{sprint_prefix}",
        "modified_files": coder_result.get("partial_output", {}).get("modified_files", []),
    }

    # Add test summary
    tester_po = tester_result.get("partial_output", {})
    if tester_po:
        passed = tester_po.get("passed_count", 0)
        failed = tester_po.get("failed_count", 0)
        update_input["test_summary"] = f"{passed} passed, {failed} failed"

    # Add notion_id if task has one
    notion_id = task.get("notion_id")
    if notion_id:
        update_input["notion_id"] = notion_id

    agent_cls = resolve_agent_class("updater", settings)
    llm = get_llm(settings, agent_name="updater")
    agent = agent_cls(llm=llm, rag=rag, snapshot=snapshot)
    tool_names = getattr(agent_cls, "REQUIRED_TOOLS", []) + getattr(agent_cls, "OPTIONAL_TOOLS", [])
    if tool_names:
        agent.bind_tools(tool_names, settings, dry_run=dry_run)

    result = agent.run(json.dumps(update_input))

    # Record result
    updated_results = dict(state.get("task_results", {}))
    task_entry = dict(updated_results.get(task_id, {}))
    task_entry["updater"] = result
    updated_results[task_id] = task_entry

    if not result.get("success"):
        error_msg = f"UpdaterAgent failed on {task_id}: {result.get('error_message', 'unknown')}"
        logger.warning(error_msg)
        return {
            "task_results": updated_results,
            "errors": [error_msg],
            "failed_task_ids": [],
        }

    return {
        "task_results": updated_results,
        "errors": [],
        "failed_task_ids": [],
    }


def check_node(state: SprintState, *, settings: Any, dry_run: bool) -> dict:
    """Advance to next task, check abort threshold, update status."""
    task = get_current_task(state)
    if task:
        print(f"[cascade] check: task {task['id']} done, advancing")
    new_errors: list[str] = []
    new_failed: list[str] = []

    if task is not None:
        task_id = task["id"]
        # Check if current task actually passed
        tester_result = state.get("task_results", {}).get(task_id, {}).get("tester", {})
        test_passed = tester_result.get("partial_output", {}).get("test_passed", False)
        tester_success = tester_result.get("success", False)

        # If task didn't reach updater (routed here from failed test), mark failed
        updater_result = state.get("task_results", {}).get(task_id, {}).get("updater")
        if updater_result is None and task_id not in state.get("failed_task_ids", []):
            if not test_passed or not tester_success:
                error = f"Task {task_id} failed: tests did not pass after retries"
                new_errors.append(error)
                new_failed.append(task_id)

        # Completion validation — warn if no meaningful work was done
        coder_result = state.get("task_results", {}).get(task_id, {}).get("coder", {})
        modified_files = coder_result.get("partial_output", {}).get("modified_files", [])
        pr_created = (
            updater_result.get("partial_output", {}).get("pr_created", False)
            if updater_result else False
        )
        if not modified_files and not pr_created and task_id not in state.get("failed_task_ids", []):
            new_errors.append(
                f"Task {task_id} completed with no file changes and no PR"
                " — may need manual review"
            )

    # Advance to next task
    new_index = state["current_task_index"] + 1

    # Build a temporary state view to check abort
    temp_failed = list(state.get("failed_task_ids", [])) + new_failed
    tasks = state.get("tasks", [])
    total = len(tasks)

    if total > 0 and len(temp_failed) / total > state.get("abort_threshold", 0.5):
        return {
            "current_task_index": new_index,
            "status": "aborted",
            "errors": new_errors,
            "failed_task_ids": new_failed,
        }

    # Check if all tasks are done
    done = True
    failed_set = set(temp_failed)
    idx = new_index
    while idx < len(tasks):
        if tasks[idx].get("id") not in failed_set:
            done = False
            break
        idx += 1

    status = "completed" if done else "running"
    return {
        "current_task_index": new_index,
        "status": status,
        "errors": new_errors,
        "failed_task_ids": new_failed,
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_plan(state: SprintState) -> str:
    """Route after plan_node: proceed to tasks or end."""
    if state.get("status") == "aborted":
        return END
    if not state.get("tasks"):
        return END
    return "setup_task_node"


def route_after_test(state: SprintState) -> str:
    """Route after test_node: update, retry coder, or skip task."""
    task = get_current_task(state)
    if task is None:
        return "check_node"

    task_id = task["id"]

    # If task already failed (coder failure), skip to check
    if task_id in state.get("failed_task_ids", []):
        return "check_node"

    tester_result = state.get("task_results", {}).get(task_id, {}).get("tester", {})

    # TesterAgent infrastructure failure — skip task
    if not tester_result.get("success", False):
        return "check_node"

    test_passed = tester_result.get("partial_output", {}).get("test_passed", False)
    if test_passed:
        return "update_node"

    # Tests failed — check outer retries
    outer_count = get_iteration_count(state, "outer_coder", task_id)
    if outer_count < MAX_OUTER_RETRIES:
        return "code_node"

    # Exhausted retries — skip task
    return "check_node"


def route_after_check(state: SprintState) -> str:
    """Route after check_node: next task, abort, or finish."""
    if state.get("status") in ("aborted", "completed"):
        return END
    return "setup_task_node"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_cascade_graph(settings: Any, dry_run: bool = False, rag: Any = None, snapshot: Any = None):
    """Build and compile the cascade StateGraph.

    Args:
        settings: Application settings (passed to agent/tool factories).
        dry_run: If True, tools operate in dry-run mode.
        rag: Optional RAGRetriever instance for semantic search.
        snapshot: Optional SnapshotLookup instance for relational queries.

    Returns:
        A compiled LangGraph StateGraph ready for ``invoke()``.
    """
    graph = StateGraph(SprintState)

    # Bind settings/dry_run/rag/snapshot into node functions
    _plan = partial(plan_node, settings=settings, dry_run=dry_run)
    _setup = partial(setup_task_node, settings=settings, dry_run=dry_run)
    _code = partial(code_node, settings=settings, dry_run=dry_run, rag=rag, snapshot=snapshot)
    _commit_push = partial(commit_push_node, settings=settings, dry_run=dry_run)
    _test = partial(test_node, settings=settings, dry_run=dry_run, rag=rag, snapshot=snapshot)
    _update = partial(update_node, settings=settings, dry_run=dry_run, rag=rag, snapshot=snapshot)
    _check = partial(check_node, settings=settings, dry_run=dry_run)

    graph.add_node("plan_node", _plan)
    graph.add_node("setup_task_node", _setup)
    graph.add_node("code_node", _code)
    graph.add_node("commit_push_node", _commit_push)
    graph.add_node("test_node", _test)
    graph.add_node("update_node", _update)
    graph.add_node("check_node", _check)

    # Edges
    graph.add_edge(START, "plan_node")
    graph.add_conditional_edges("plan_node", route_after_plan)
    graph.add_edge("setup_task_node", "code_node")
    graph.add_edge("code_node", "commit_push_node")
    graph.add_edge("commit_push_node", "test_node")
    graph.add_conditional_edges("test_node", route_after_test)
    graph.add_edge("update_node", "check_node")
    graph.add_conditional_edges("check_node", route_after_check)

    return graph.compile()
