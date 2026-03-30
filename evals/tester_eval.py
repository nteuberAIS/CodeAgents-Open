"""Evaluation suite for TesterAgent output quality.

Tests that the agent correctly runs tests and reports structured results.
Since TesterAgent doesn't use the LLM, evals focus on correct output
structure and success-flag semantics (test failures are data, not errors).
"""

from __future__ import annotations

import json

from evals.base_eval import BaseEval, EvalCase, EvalScore


_BASIC_TASK = json.dumps({
    "task_id": "SP8-001",
    "task_title": "Add retry logic to data pipeline",
    "repo_dir": "/tmp/test-repo",
})

_FAILURE_TASK = json.dumps({
    "task_id": "SP8-002",
    "task_title": "Fix broken tests",
    "repo_dir": "/tmp/test-repo",
})


class TesterEval(BaseEval):
    """Eval suite for tester agent test execution and reporting."""

    agent_name = "tester"

    def get_cases(self) -> list[EvalCase]:
        return [
            EvalCase(
                name="basic_test_run",
                description="Run tests on a simple task, verify structured results",
                prompt=_BASIC_TASK,
                context=None,
            ),
            EvalCase(
                name="test_with_failures",
                description="Run tests with failures, verify counts reported correctly",
                prompt=_FAILURE_TASK,
                context=None,
            ),
        ]

    def score(self, case: EvalCase, output: dict) -> list[EvalScore]:
        return [
            self._score_has_test_results(output),
            self._score_success_flag_correct(output),
        ]

    def _score_has_test_results(self, output: dict) -> EvalScore:
        """Check that partial_output contains test result counts."""
        po = output.get("partial_output", {})
        has_counts = all(
            key in po
            for key in ("passed_count", "failed_count", "error_count", "test_passed")
        )
        return EvalScore(
            name="has_test_results",
            passed=has_counts,
            score=1.0 if has_counts else 0.0,
            detail="all count fields present" if has_counts else "missing count fields",
        )

    def _score_success_flag_correct(self, output: dict) -> EvalScore:
        """Check that success=True (test failures are data, not errors)."""
        success = output.get("success", False)
        return EvalScore(
            name="success_flag_correct",
            passed=success,
            score=1.0 if success else 0.0,
            detail="" if success else "success should be True even with test failures",
        )
