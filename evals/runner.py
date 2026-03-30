"""Eval runner — loads eval suites and runs them against agents."""

from __future__ import annotations

import importlib
import time
from typing import Any, Callable

from evals.base_eval import BaseEval, EvalResult


# Registry of eval suites — maps agent name to eval class import path
EVAL_REGISTRY: dict[str, str] = {
    "sprint_planner": "evals.sprint_planner_eval.SprintPlannerEval",
    "coder": "evals.coder_eval.CoderEval",
}


def resolve_eval_class(agent_name: str) -> type[BaseEval]:
    """Dynamically import and return an eval class by agent name."""
    dotted_path = EVAL_REGISTRY.get(agent_name)
    if not dotted_path:
        available = ", ".join(EVAL_REGISTRY.keys())
        raise ValueError(
            f"No eval suite for agent '{agent_name}'. Available: {available}"
        )
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class EvalRunner:
    """Runs evaluation suites against agents."""

    def __init__(self, eval_suite: BaseEval) -> None:
        self.eval_suite = eval_suite

    def run_all(
        self, agent_factory: Callable[[dict | None], Any],
    ) -> list[EvalResult]:
        """Run all eval cases.

        Args:
            agent_factory: Callable(context) -> agent instance.
                The runner calls agent.run(case.prompt) for each case.

        Returns:
            List of EvalResult objects, one per case.
        """
        results: list[EvalResult] = []
        for case in self.eval_suite.get_cases():
            t0 = time.monotonic()
            try:
                agent = agent_factory(case.context)
                output = agent.run(case.prompt)
            except Exception as e:
                output = {
                    "success": False,
                    "error_type": "infra",
                    "error_message": str(e),
                    "partial_output": {},
                }
            elapsed = time.monotonic() - t0

            scores = self.eval_suite.score(case, output)
            overall_score = (
                sum(s.score for s in scores) / len(scores) if scores else 0.0
            )
            overall_pass = all(s.passed for s in scores)

            results.append(
                EvalResult(
                    case_name=case.name,
                    agent_output=output,
                    scores=scores,
                    overall_pass=overall_pass,
                    overall_score=overall_score,
                )
            )
        return results

    def print_report(self, results: list[EvalResult]) -> None:
        """Print a human-readable eval report to stdout."""
        total_pass = sum(1 for r in results if r.overall_pass)
        print(f"\n{'=' * 60}")
        print(
            f"Eval: {self.eval_suite.agent_name}"
            f" — {total_pass}/{len(results)} cases passed"
        )
        print(f"{'=' * 60}\n")

        for result in results:
            status = "PASS" if result.overall_pass else "FAIL"
            print(
                f"[{status}] {result.case_name}"
                f" (score: {result.overall_score:.2f})"
            )
            for score in result.scores:
                mark = "+" if score.passed else "-"
                print(f"  [{mark}] {score.name}: {score.score:.2f} — {score.detail}")
            print()
