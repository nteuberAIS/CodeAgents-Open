"""Evaluation suite for UpdaterAgent output quality.

Tests the LLM's ability to generate PR descriptions and the agent's
graceful degradation when tools are unavailable.
"""

from __future__ import annotations

import json

from evals.base_eval import BaseEval, EvalCase, EvalScore


_NO_TOOLS_TASK = json.dumps({
    "task_id": "SP8-001",
    "task_title": "Add retry logic to data pipeline",
    "source_branch": "sprint-8/SP8-001",
    "target_branch": "sprint-8",
})

_FULL_TASK = json.dumps({
    "task_id": "SP8-002",
    "task_title": "Create bronze layer Parquet schemas",
    "task_description": "Define Pydantic models for HR and finance bronze data sources",
    "source_branch": "sprint-8/SP8-002",
    "target_branch": "sprint-8",
    "notion_id": "wi-001",
    "test_summary": "5 passed, 0 failed",
    "modified_files": ["src/schemas/bronze.py", "tests/test_bronze.py"],
})


class UpdaterEval(BaseEval):
    """Eval suite for updater agent PR creation and status updates."""

    agent_name = "updater"

    def get_cases(self) -> list[EvalCase]:
        return [
            EvalCase(
                name="basic_update_no_tools",
                description="No tools bound — verify graceful degradation",
                prompt=_NO_TOOLS_TASK,
                context=None,
            ),
            EvalCase(
                name="update_with_task_info",
                description="Task info provided — verify LLM generates PR description",
                prompt=_FULL_TASK,
                context=None,
            ),
        ]

    def score(self, case: EvalCase, output: dict) -> list[EvalScore]:
        scores = [
            self._score_json_valid(output),
        ]
        if case.name == "update_with_task_info":
            scores.append(self._score_has_pr_description(output))
            scores.append(self._score_description_references_task(case, output))
        return scores

    def _score_json_valid(self, output: dict) -> EvalScore:
        """Check output is a valid agent result envelope."""
        success = output.get("success", False)
        has_output = bool(output.get("partial_output"))
        passed = success or has_output
        return EvalScore(
            name="json_valid",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="" if passed else output.get("error_message", "unknown"),
        )

    def _score_has_pr_description(self, output: dict) -> EvalScore:
        """Check that partial_output has a non-empty PR title."""
        po = output.get("partial_output", {})
        pr_title = po.get("pr_title", "")
        passed = bool(pr_title and len(pr_title.strip()) > 0)
        return EvalScore(
            name="has_pr_description",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=f"title: '{pr_title[:50]}'" if pr_title else "empty PR title",
        )

    def _score_description_references_task(
        self, case: EvalCase, output: dict,
    ) -> EvalScore:
        """Check that PR title or description references the task."""
        po = output.get("partial_output", {})
        pr_title = po.get("pr_title", "").lower()

        task_info = json.loads(case.prompt)
        task_title = task_info.get("task_title", "").lower()

        # Check if any significant words from task_title appear in pr_title
        task_words = {w for w in task_title.split() if len(w) > 3}
        matches = sum(1 for w in task_words if w in pr_title)
        target = max(1, len(task_words))
        score = min(1.0, matches / target)

        return EvalScore(
            name="description_references_task",
            passed=score >= 0.3,
            score=score,
            detail=f"matched {matches}/{len(task_words)} task words in PR title",
        )
