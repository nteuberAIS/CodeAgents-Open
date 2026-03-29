"""Tests for the eval framework — all with mocked LLM, no real Ollama."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from evals.base_eval import BaseEval, EvalCase, EvalResult, EvalScore
from evals.runner import EVAL_REGISTRY, EvalRunner, resolve_eval_class
from evals.sprint_planner_eval import SprintPlannerEval


# -- Helpers --


VALID_PLAN = {
    "sprint": 8,
    "goal": "Deploy SHIR and establish medallion bronze layer",
    "tasks": [
        {
            "id": "SP8-001",
            "title": "Deploy Self-Hosted Integration Runtime",
            "description": "Install and configure SHIR on the data gateway VM",
            "assignee": None,
            "estimate_hrs": 4,
            "status": "todo",
        },
        {
            "id": "SP8-002",
            "title": "Create bronze layer Parquet schemas",
            "description": "Define Parquet schemas for raw data ingestion",
            "assignee": None,
            "estimate_hrs": 6,
            "status": "todo",
        },
        {
            "id": "SP8-003",
            "title": "Build ADF pipeline for HR source",
            "description": "Create Data Factory pipeline for HR data into bronze",
            "assignee": None,
            "estimate_hrs": 5,
            "status": "todo",
        },
        {
            "id": "SP8-004",
            "title": "Add pipeline integration tests",
            "description": "Write tests validating bronze layer outputs",
            "assignee": None,
            "estimate_hrs": 3,
            "status": "todo",
        },
    ],
    "dependencies": [
        {"from": "SP8-003", "to": "SP8-001", "type": "blocks"},
        {"from": "SP8-004", "to": "SP8-003", "type": "blocks"},
    ],
}


def _wrap_envelope(plan_dict):
    """Wrap a plan dict in the standard agent result envelope."""
    return {
        "success": True,
        "error_type": None,
        "error_message": None,
        "partial_output": plan_dict,
    }


def _make_mock_agent(plan_dict):
    """Create a mock agent that returns a fixed plan in envelope format."""
    agent = MagicMock()
    agent.run.return_value = _wrap_envelope(plan_dict)
    return agent


# -- Model tests --


class TestEvalModels:
    """Pydantic model validation for eval types."""

    def test_eval_case_model(self):
        case = EvalCase(
            name="test",
            description="A test case",
            prompt="Plan sprint 1",
        )
        assert case.name == "test"
        assert case.context is None

    def test_eval_case_with_context(self):
        case = EvalCase(
            name="test",
            description="With context",
            prompt="Plan",
            context={"work_items": []},
        )
        assert case.context == {"work_items": []}

    def test_eval_score_model(self):
        score = EvalScore(name="test", passed=True, score=1.0, detail="ok")
        assert score.passed is True

    def test_eval_result_model(self):
        result = EvalResult(
            case_name="test",
            agent_output=_wrap_envelope({"sprint": 1}),
            scores=[EvalScore(name="s1", passed=True, score=1.0)],
            overall_pass=True,
            overall_score=1.0,
        )
        assert result.overall_pass is True


# -- SprintPlannerEval tests --


class TestSprintPlannerEvalCases:
    def test_get_cases_returns_non_empty(self):
        suite = SprintPlannerEval()
        cases = suite.get_cases()
        assert len(cases) >= 3

    def test_all_cases_have_required_fields(self):
        suite = SprintPlannerEval()
        for case in suite.get_cases():
            assert case.name
            assert case.description
            assert case.prompt


class TestSprintPlannerEvalScoring:
    def setup_method(self):
        self.suite = SprintPlannerEval()

    def test_score_valid_output_all_pass(self):
        case = EvalCase(name="test", description="test", prompt="Plan sprint 8")
        scores = self.suite.score(case, _wrap_envelope(VALID_PLAN))
        assert all(s.passed for s in scores), [
            (s.name, s.detail) for s in scores if not s.passed
        ]

    def test_score_parse_error_fails_json_valid(self):
        output = {
            "success": False,
            "error_type": "llm",
            "error_message": "not JSON",
            "partial_output": {"raw_output": "garbage"},
        }
        case = EvalCase(name="test", description="test", prompt="Plan sprint 8")
        scores = self.suite.score(case, output)
        json_score = next(s for s in scores if s.name == "json_valid")
        assert not json_score.passed
        assert json_score.score == 0.0

    def test_score_missing_tasks_fails_schema(self):
        output = _wrap_envelope({"sprint": 8, "goal": "test"})  # missing tasks, dependencies
        case = EvalCase(name="test", description="test", prompt="Plan")
        scores = self.suite.score(case, output)
        schema_score = next(s for s in scores if s.name == "schema_compliance")
        assert not schema_score.passed
        assert schema_score.score < 1.0

    def test_score_invalid_dependency_refs(self):
        bad_plan = dict(VALID_PLAN)
        bad_plan["dependencies"] = [
            {"from": "FAKE-001", "to": "FAKE-002", "type": "blocks"}
        ]
        case = EvalCase(name="test", description="test", prompt="Plan")
        scores = self.suite.score(case, _wrap_envelope(bad_plan))
        dep_score = next(s for s in scores if s.name == "dependency_validity")
        assert not dep_score.passed
        assert dep_score.score == 0.0

    def test_score_bad_id_convention(self):
        bad_plan = dict(VALID_PLAN)
        bad_plan["tasks"] = [
            {**VALID_PLAN["tasks"][0], "id": "task-1"},  # wrong format
            {**VALID_PLAN["tasks"][1], "id": "task-2"},
        ]
        case = EvalCase(name="test", description="test", prompt="Plan")
        scores = self.suite.score(case, _wrap_envelope(bad_plan))
        id_score = next(s for s in scores if s.name == "id_convention")
        assert not id_score.passed

    def test_score_context_usage_with_matching_output(self):
        case = EvalCase(
            name="test",
            description="test",
            prompt="Plan sprint 8",
            context={
                "work_items": [
                    {"name": "Deploy Self-Hosted Integration Runtime"},
                ],
                "sprints": [],
                "docs": [],
                "decisions": [],
                "risks": [],
                "page_content": {},
            },
        )
        scores = self.suite.score(case, _wrap_envelope(VALID_PLAN))
        ctx_score = next(s for s in scores if s.name == "context_usage")
        assert ctx_score.passed
        assert ctx_score.score > 0.0


# -- EvalRunner tests --


class TestEvalRunner:
    def test_run_all_with_mock_agent(self):
        suite = SprintPlannerEval()
        runner = EvalRunner(suite)

        def agent_factory(context):
            return _make_mock_agent(VALID_PLAN)

        results = runner.run_all(agent_factory)

        assert len(results) == len(suite.get_cases())
        for result in results:
            assert isinstance(result, EvalResult)
            assert len(result.scores) > 0

    def test_run_all_handles_agent_exception(self):
        suite = SprintPlannerEval()
        runner = EvalRunner(suite)

        def agent_factory(context):
            agent = MagicMock()
            agent.run.side_effect = RuntimeError("LLM exploded")
            return agent

        results = runner.run_all(agent_factory)

        # Should not raise — errors captured in output
        assert len(results) == len(suite.get_cases())
        for result in results:
            assert result.agent_output["success"] is False
            assert result.agent_output["error_type"] == "infra"

    def test_print_report_runs_without_error(self, capsys):
        suite = SprintPlannerEval()
        runner = EvalRunner(suite)

        results = [
            EvalResult(
                case_name="test",
                agent_output=_wrap_envelope(VALID_PLAN),
                scores=[EvalScore(name="s1", passed=True, score=1.0, detail="ok")],
                overall_pass=True,
                overall_score=1.0,
            ),
        ]
        runner.print_report(results)

        captured = capsys.readouterr()
        assert "sprint_planner" in captured.out
        assert "PASS" in captured.out


# -- Registry tests --


class TestEvalRegistry:
    def test_sprint_planner_in_registry(self):
        assert "sprint_planner" in EVAL_REGISTRY

    def test_resolve_eval_class(self):
        cls = resolve_eval_class("sprint_planner")
        assert cls is SprintPlannerEval

    def test_resolve_unknown_raises(self):
        with pytest.raises(ValueError, match="No eval suite"):
            resolve_eval_class("nonexistent_agent")
