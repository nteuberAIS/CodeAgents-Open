"""Evaluation suite for SprintPlannerAgent output quality."""

from __future__ import annotations

import json
import re

from evals.base_eval import BaseEval, EvalCase, EvalScore


# -- Sample context for eval cases --

_ACTIVE_SPRINT = {
    "notion_id": "spr-eval-001",
    "name": "Sprint 8",
    "status": "Active",
    "goal": "Deploy SHIR and bronze layer",
    "start_date": "2026-03-01",
    "end_date": "2026-03-14",
    "work_item_ids": ["wi-eval-001", "wi-eval-002"],
    "has_content": False,
}

_WORK_ITEMS = [
    {
        "notion_id": "wi-eval-001",
        "name": "Deploy Self-Hosted Integration Runtime",
        "type": "Task",
        "status": "Ready",
        "priority": "P1",
        "estimate_hrs": 4.0,
        "sprint_id": "spr-eval-001",
        "has_content": False,
    },
    {
        "notion_id": "wi-eval-002",
        "name": "Create bronze layer Parquet schemas",
        "type": "Task",
        "status": "Ready",
        "priority": "P2",
        "estimate_hrs": 6.0,
        "sprint_id": "spr-eval-001",
        "has_content": False,
    },
    {
        "notion_id": "wi-eval-003",
        "name": "Write integration tests for data pipeline",
        "type": "Task",
        "status": "Backlog",
        "priority": "P2",
        "estimate_hrs": 3.0,
        "sprint_id": "spr-eval-001",
        "has_content": False,
    },
]

_OPEN_RISK = {
    "notion_id": "rsk-eval-001",
    "name": "SHIR VM VRAM limit may block GPU workloads",
    "type": "Risk",
    "status": "Open",
    "severity": "High",
    "has_content": False,
}


def _sample_context() -> dict:
    return {
        "work_items": _WORK_ITEMS,
        "sprints": [_ACTIVE_SPRINT],
        "docs": [],
        "decisions": [],
        "risks": [],
        "page_content": {},
    }


def _sample_context_with_risks() -> dict:
    ctx = _sample_context()
    ctx["risks"] = [_OPEN_RISK]
    return ctx


class SprintPlannerEval(BaseEval):
    """Eval suite for sprint planner output quality."""

    agent_name = "sprint_planner"

    def get_cases(self) -> list[EvalCase]:
        return [
            EvalCase(
                name="basic_plan_no_context",
                description="Generate a plan without backlog context",
                prompt="Plan sprint 8",
                context=None,
            ),
            EvalCase(
                name="plan_with_backlog",
                description="Generate a plan using backlog context",
                prompt="Plan sprint 8 focusing on SHIR deployment",
                context=_sample_context(),
            ),
            EvalCase(
                name="plan_with_risks",
                description="Plan should acknowledge open risks",
                prompt="Plan sprint 9",
                context=_sample_context_with_risks(),
            ),
        ]

    def score(self, case: EvalCase, output: dict) -> list[EvalScore]:
        scores = [
            self._score_json_valid(output),
            self._score_schema_compliance(output),
            self._score_task_count(output),
            self._score_task_completeness(output),
            self._score_dependency_validity(output),
            self._score_id_convention(output),
        ]
        if case.context:
            scores.append(self._score_context_usage(case, output))
        return scores

    # -- Scoring criteria --

    def _score_json_valid(self, output: dict) -> EvalScore:
        passed = "parse_error" not in output
        return EvalScore(
            name="json_valid",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="" if passed else output.get("parse_error", "unknown"),
        )

    def _score_schema_compliance(self, output: dict) -> EvalScore:
        required = {"sprint", "goal", "tasks", "dependencies"}
        present = required.intersection(output.keys())
        score = len(present) / len(required) if required else 1.0
        missing = required - present
        return EvalScore(
            name="schema_compliance",
            passed=score == 1.0,
            score=score,
            detail=f"missing: {missing}" if missing else "all keys present",
        )

    def _score_task_count(self, output: dict) -> EvalScore:
        tasks = output.get("tasks", [])
        count = len(tasks)
        passed = 2 <= count <= 8
        return EvalScore(
            name="task_count",
            passed=passed,
            score=1.0 if passed else 0.5,
            detail=f"{count} tasks (expected 2-8)",
        )

    def _score_task_completeness(self, output: dict) -> EvalScore:
        tasks = output.get("tasks", [])
        if not tasks:
            return EvalScore(
                name="task_completeness",
                passed=False,
                score=0.0,
                detail="no tasks to check",
            )
        required_fields = {"id", "title", "description", "estimate_hrs", "status"}
        complete = 0
        for task in tasks:
            if required_fields.issubset(task.keys()):
                complete += 1
        score = complete / len(tasks)
        return EvalScore(
            name="task_completeness",
            passed=score == 1.0,
            score=score,
            detail=f"{complete}/{len(tasks)} tasks have all required fields",
        )

    def _score_dependency_validity(self, output: dict) -> EvalScore:
        task_ids = {t.get("id") for t in output.get("tasks", [])}
        deps = output.get("dependencies", [])
        if not deps:
            return EvalScore(
                name="dependency_validity",
                passed=True,
                score=1.0,
                detail="no dependencies to validate",
            )
        valid = sum(
            1
            for d in deps
            if d.get("from") in task_ids and d.get("to") in task_ids
        )
        score = valid / len(deps)
        return EvalScore(
            name="dependency_validity",
            passed=score == 1.0,
            score=score,
            detail=f"{valid}/{len(deps)} valid references",
        )

    def _score_id_convention(self, output: dict) -> EvalScore:
        tasks = output.get("tasks", [])
        if not tasks:
            return EvalScore(
                name="id_convention",
                passed=False,
                score=0.0,
                detail="no tasks",
            )
        pattern = re.compile(r"^SP\d+-\d{3}$")
        matches = sum(1 for t in tasks if pattern.match(t.get("id", "")))
        score = matches / len(tasks)
        return EvalScore(
            name="id_convention",
            passed=score == 1.0,
            score=score,
            detail=f"{matches}/{len(tasks)} IDs match SP<N>-<NNN>",
        )

    def _score_context_usage(self, case: EvalCase, output: dict) -> EvalScore:
        context_names = set()
        for wi in (case.context or {}).get("work_items", []):
            context_names.add(wi.get("name", "").lower())

        output_text = json.dumps(output).lower()
        matches = sum(1 for name in context_names if name in output_text)
        target = max(1, len(context_names) * 0.3)
        score = min(1.0, matches / target)
        return EvalScore(
            name="context_usage",
            passed=score >= 0.5,
            score=score,
            detail=f"referenced {matches}/{len(context_names)} context items",
        )
