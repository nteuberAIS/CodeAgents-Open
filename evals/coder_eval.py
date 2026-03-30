"""Evaluation suite for CoderAgent output quality.

Tests the LLM's ability to generate precise Aider-compatible instructions
and file lists from task descriptions. Evals run with dry-run tools or
--no-tools mode, so they test instruction generation only (not Aider execution).
"""

from __future__ import annotations

import json

from evals.base_eval import BaseEval, EvalCase, EvalScore


# -- Sample data for eval cases --

_SIMPLE_TASK = json.dumps({
    "id": "SP8-001",
    "title": "Add retry logic to data pipeline",
    "description": "Add exponential backoff retry to the ingest() function",
    "estimate_hrs": 3,
    "status": "todo",
})

_CONTEXT_TASK = json.dumps({
    "id": "SP8-002",
    "title": "Create bronze layer Parquet schemas",
    "description": "Define Pydantic models for HR and finance bronze data sources",
    "estimate_hrs": 4,
    "status": "todo",
})

_MULTI_FILE_TASK = json.dumps({
    "id": "SP8-003",
    "title": "Add API endpoint with tests",
    "description": (
        "Create a new REST endpoint for data pipeline status checks "
        "and add corresponding pytest tests"
    ),
    "estimate_hrs": 5,
    "status": "todo",
})

_WORK_ITEMS = [
    {
        "notion_id": "wi-eval-001",
        "name": "Build ingestion pipeline",
        "type": "Task",
        "status": "Ready",
        "priority": "P1",
        "estimate_hrs": 4.0,
        "description": "Ingest HR data from on-prem SQL Server via SHIR",
    },
    {
        "notion_id": "wi-eval-002",
        "name": "Define bronze schemas",
        "type": "Task",
        "status": "Ready",
        "priority": "P2",
        "estimate_hrs": 6.0,
        "description": "Parquet schemas for raw data layer",
    },
]


def _sample_context() -> dict:
    return {
        "work_items": _WORK_ITEMS,
        "sprints": [],
        "docs": [],
        "decisions": [],
        "risks": [],
        "page_content": {},
    }


class CoderEval(BaseEval):
    """Eval suite for coder agent instruction generation quality."""

    agent_name = "coder"

    def get_cases(self) -> list[EvalCase]:
        return [
            EvalCase(
                name="basic_task_no_context",
                description="Generate instruction for a simple task without context",
                prompt=_SIMPLE_TASK,
                context=None,
            ),
            EvalCase(
                name="task_with_context",
                description="Generate instruction with backlog context available",
                prompt=_CONTEXT_TASK,
                context=_sample_context(),
            ),
            EvalCase(
                name="task_with_complex_files",
                description="Multi-file task requiring multiple entries in files list",
                prompt=_MULTI_FILE_TASK,
                context=None,
            ),
        ]

    def score(self, case: EvalCase, output: dict) -> list[EvalScore]:
        scores = [
            self._score_json_valid(output),
            self._score_has_instruction(output),
            self._score_has_files(output),
            self._score_instruction_specificity(output),
        ]
        if case.context:
            scores.append(self._score_context_usage(case, output))
        return scores

    # -- Scoring criteria --

    def _score_json_valid(self, output: dict) -> EvalScore:
        passed = output.get("success", False)
        return EvalScore(
            name="json_valid",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail="" if passed else output.get("error_message", "unknown"),
        )

    def _score_has_instruction(self, output: dict) -> EvalScore:
        po = output.get("partial_output", {})
        instruction = po.get("instruction", "")
        passed = bool(instruction and len(instruction.strip()) > 0)
        return EvalScore(
            name="has_instruction",
            passed=passed,
            score=1.0 if passed else 0.0,
            detail=f"instruction length: {len(instruction)}" if instruction else "empty",
        )

    def _score_has_files(self, output: dict) -> EvalScore:
        po = output.get("partial_output", {})
        # In dry-run mode, modified_files may be empty; check instruction instead
        modified = po.get("modified_files", [])
        instruction = po.get("instruction", "")
        has_files = bool(modified) or ("/" in instruction or ".py" in instruction)
        return EvalScore(
            name="has_files",
            passed=has_files,
            score=1.0 if has_files else 0.0,
            detail=f"modified: {len(modified)} files" if modified else "file refs in instruction",
        )

    def _score_instruction_specificity(self, output: dict) -> EvalScore:
        po = output.get("partial_output", {})
        instruction = po.get("instruction", "")
        # Instruction should be reasonably detailed (>20 chars) and reference
        # identifiers like file paths, function names, or class names
        long_enough = len(instruction) > 20
        has_identifiers = any(
            marker in instruction
            for marker in ("/", ".py", "()", "def ", "class ", "import ")
        )
        passed = long_enough and has_identifiers
        score = (0.5 if long_enough else 0.0) + (0.5 if has_identifiers else 0.0)
        return EvalScore(
            name="instruction_specificity",
            passed=passed,
            score=score,
            detail=f"len={len(instruction)}, has_ids={has_identifiers}",
        )

    def _score_context_usage(self, case: EvalCase, output: dict) -> EvalScore:
        context_names = set()
        for wi in (case.context or {}).get("work_items", []):
            context_names.add(wi.get("name", "").lower())

        po = output.get("partial_output", {})
        output_text = json.dumps(po).lower()
        matches = sum(1 for name in context_names if name in output_text)
        target = max(1, len(context_names))
        score = min(1.0, matches / target)
        return EvalScore(
            name="context_usage",
            passed=score >= 0.3,
            score=score,
            detail=f"referenced {matches}/{len(context_names)} context items",
        )
