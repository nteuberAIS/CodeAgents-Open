"""High-level runner for the cascade orchestrator.

Usage:
    runner = CascadeRunner(settings, dry_run=True)
    final_state = runner.run("sprint-8", goal="Implement feature X")
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config.settings import get_settings
from orchestration.cascade import build_cascade_graph
from schemas.sprint_state import SprintState

logger = logging.getLogger(__name__)


class CascadeRunner:
    """Wraps the LangGraph cascade for convenient invocation."""

    def __init__(
        self,
        settings: Any | None = None,
        dry_run: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.dry_run = dry_run
        self.graph = build_cascade_graph(self.settings, dry_run=dry_run)

    def run(
        self,
        sprint_id: str,
        goal: str = "",
        abort_threshold: float = 0.5,
    ) -> SprintState:
        """Execute the full cascade and return the final state.

        Args:
            sprint_id: Identifier for this sprint (e.g. "sprint-8").
            goal: Sprint goal passed to SprintPlannerAgent.
            abort_threshold: Fraction of tasks that can fail before aborting.

        Returns:
            The final SprintState after all nodes have executed.
        """
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

        final_state = self.graph.invoke(
            initial_state,
            config={"recursion_limit": 100},
        )
        self.print_summary(final_state)
        return final_state

    @staticmethod
    def format_escalation(
        agent_name: str,
        task_id: str,
        task_title: str,
        result: dict,
        iteration: int,
        max_iterations: int,
    ) -> str:
        """Format an escalation message per failure-modes.md spec."""
        error_type = result.get("error_type", "unknown")
        error_message = result.get("error_message", "no message")
        partial_output = result.get("partial_output", {})

        partial_json = json.dumps(partial_output, indent=2, default=str)
        if len(partial_json) > 2000:
            partial_json = partial_json[:2000] + "\n... (truncated)"

        return (
            f"=== ESCALATION: {agent_name} failed on {task_id} ===\n"
            f"Error type:  {error_type}\n"
            f"Message:     {error_message}\n"
            f"Iteration:   {iteration} / {max_iterations}\n"
            f"Task:        {task_id} - {task_title}\n"
            f"\n"
            f"--- Partial Output ---\n"
            f"{partial_json}\n"
            f"\n"
            f"--- Suggested Actions ---\n"
            f"1. Review the error and fix manually\n"
            f"2. Re-run: python main.py run --resume {task_id}\n"
            f"3. Skip:   python main.py run --skip {task_id}\n"
        )

    @staticmethod
    def print_summary(state: SprintState) -> None:
        """Print a human-readable execution summary."""
        status = state.get("status", "unknown")
        tasks = state.get("tasks", [])
        failed = state.get("failed_task_ids", [])
        errors = state.get("errors", [])
        results = state.get("task_results", {})

        total = len(tasks)
        completed = sum(
            1 for tid, r in results.items()
            if "updater" in r and tid not in failed
        )

        logger.info("=" * 50)
        logger.info("CASCADE SUMMARY")
        logger.info("=" * 50)
        logger.info("Sprint:    %s", state.get("sprint_id", "?"))
        logger.info("Status:    %s", status)
        logger.info("Tasks:     %d total, %d completed, %d failed", total, completed, len(failed))

        if failed:
            logger.info("Failed:    %s", ", ".join(failed))

        # Count PRs created
        pr_count = sum(
            1 for r in results.values()
            if r.get("updater", {}).get("partial_output", {}).get("pr_created", False)
        )
        if pr_count:
            logger.info("PRs:       %d created", pr_count)

        if errors:
            logger.info("Errors:")
            for err in errors:
                logger.info("  - %s", err)

        logger.info("=" * 50)
