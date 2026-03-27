"""Tests for agents/sprint_planner.py.

All tests mock the LLM — no Ollama calls are made.
"""

import json
from unittest.mock import MagicMock

import pytest

from agents.sprint_planner import SprintPlannerAgent

VALID_PLAN_JSON = json.dumps({
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
    ],
    "dependencies": [
        {"from": "SP8-002", "to": "SP8-001", "type": "blocks"},
    ],
})


def _make_llm(response_content: str) -> MagicMock:
    """Create a mock ChatOllama that returns a fixed response."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    llm.invoke.return_value = mock_response
    return llm


def _make_context(
    sprints=None, work_items=None, risks=None, docs=None, decisions=None,
):
    """Build a context dict with sensible defaults."""
    return {
        "sprints": sprints or [],
        "work_items": work_items or [],
        "risks": risks or [],
        "docs": docs or [],
        "decisions": decisions or [],
    }


ACTIVE_SPRINT = {
    "notion_id": "spr-001",
    "name": "Sprint 8",
    "status": "Active",
    "goal": "Deploy SHIR and bronze layer",
    "start_date": "2026-03-01",
    "end_date": "2026-03-14",
    "work_item_ids": ["wi-001", "wi-002"],
}

WORK_ITEMS = [
    {
        "notion_id": "wi-001",
        "name": "Deploy VM",
        "type": "Task",
        "status": "Ready",
        "priority": "P1",
        "estimate_hrs": 4.0,
        "sprint_id": "spr-001",
    },
    {
        "notion_id": "wi-002",
        "name": "Configure VNet",
        "type": "Task",
        "status": "In Progress",
        "priority": "P2",
        "estimate_hrs": 2.0,
        "sprint_id": "spr-001",
    },
    {
        "notion_id": "wi-003",
        "name": "Write docs",
        "type": "Story",
        "status": "Backlog",
        "priority": "P3",
        "estimate_hrs": 3.0,
        "sprint_id": "spr-other",
    },
]

OPEN_RISK = {
    "notion_id": "rsk-001",
    "name": "VRAM limit",
    "type": "Risk",
    "status": "Open",
    "severity": "High",
}


class TestSprintPlannerInit:
    def test_name_attribute(self):
        agent = SprintPlannerAgent(llm=_make_llm(""))
        assert agent.name == "sprint_planner"

    def test_stores_llm(self):
        llm = _make_llm("")
        agent = SprintPlannerAgent(llm=llm)
        assert agent.llm is llm

    def test_stores_context(self):
        ctx = _make_context()
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        assert agent.context is ctx

    def test_context_defaults_to_none(self):
        agent = SprintPlannerAgent(llm=_make_llm(""))
        assert agent.context is None


class TestParseResponse:
    def setup_method(self):
        self.agent = SprintPlannerAgent(llm=_make_llm(""))

    def test_valid_json(self):
        result = self.agent._parse_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_fenced_with_language_tag(self):
        raw = '```json\n{"key": "value"}\n```'
        result = self.agent._parse_response(raw)
        assert result == {"key": "value"}

    def test_json_fenced_without_language_tag(self):
        raw = '```\n{"key": "value"}\n```'
        result = self.agent._parse_response(raw)
        assert result == {"key": "value"}

    def test_invalid_json(self):
        result = self.agent._parse_response("not json at all")
        assert "raw_output" in result
        assert "parse_error" in result
        assert result["raw_output"] == "not json at all"

    def test_empty_string(self):
        result = self.agent._parse_response("")
        assert "raw_output" in result
        assert "parse_error" in result

    def test_json_with_whitespace(self):
        raw = '  \n  {"key": "value"}  \n  '
        result = self.agent._parse_response(raw)
        assert result == {"key": "value"}


class TestFormatContext:
    def test_no_context(self):
        agent = SprintPlannerAgent(llm=_make_llm(""), context=None)
        assert agent._format_context() == ""

    def test_empty_context(self):
        agent = SprintPlannerAgent(llm=_make_llm(""), context=_make_context())
        result = agent._format_context()
        assert "No active sprint" in result
        assert "No work items found" in result

    def test_active_sprint_fields(self):
        ctx = _make_context(sprints=[ACTIVE_SPRINT])
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "Sprint 8" in result
        assert "Deploy SHIR and bronze layer" in result
        assert "2026-03-01" in result
        assert "2026-03-14" in result

    def test_work_items_filtered_by_sprint(self):
        ctx = _make_context(sprints=[ACTIVE_SPRINT], work_items=WORK_ITEMS)
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "Deploy VM" in result
        assert "Configure VNet" in result
        assert "Write docs" not in result  # different sprint_id

    def test_work_items_no_active_sprint(self):
        closed_sprint = {**ACTIVE_SPRINT, "status": "Closed"}
        ctx = _make_context(sprints=[closed_sprint], work_items=WORK_ITEMS)
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        # All items shown when no active sprint
        assert "Deploy VM" in result
        assert "Configure VNet" in result
        assert "Write docs" in result

    def test_open_risks_section(self):
        ctx = _make_context(risks=[OPEN_RISK])
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "Open Risks:" in result
        assert "[High]" in result
        assert "VRAM limit" in result

    def test_no_open_risks(self):
        closed_risk = {**OPEN_RISK, "status": "Closed"}
        ctx = _make_context(risks=[closed_risk])
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "Open Risks:" not in result

    def test_work_item_line_format(self):
        ctx = _make_context(sprints=[ACTIVE_SPRINT], work_items=[WORK_ITEMS[0]])
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "- [Ready] [P1] Deploy VM (Task, 4.0h)" in result

    def test_missing_fields_fallback(self):
        bare_item = {"notion_id": "wi-x", "name": "Bare item", "sprint_id": "spr-001"}
        ctx = _make_context(sprints=[ACTIVE_SPRINT], work_items=[bare_item])
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "[?]" in result
        assert "Bare item" in result


class TestRun:
    def test_run_without_context(self):
        llm = _make_llm(VALID_PLAN_JSON)
        agent = SprintPlannerAgent(llm=llm)
        agent.run("Plan sprint 8")

        messages = llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert "Sprint Planner" in messages[0].content
        # No backlog context appended
        assert "CURRENT BACKLOG" not in messages[0].content

    def test_run_with_context(self):
        llm = _make_llm(VALID_PLAN_JSON)
        ctx = _make_context(sprints=[ACTIVE_SPRINT], work_items=WORK_ITEMS)
        agent = SprintPlannerAgent(llm=llm, context=ctx)
        agent.run("Plan sprint 8")

        messages = llm.invoke.call_args[0][0]
        assert "CURRENT BACKLOG" in messages[0].content

    def test_user_input_as_human_message(self):
        llm = _make_llm(VALID_PLAN_JSON)
        agent = SprintPlannerAgent(llm=llm)
        agent.run("Plan sprint 8")

        messages = llm.invoke.call_args[0][0]
        assert messages[1].content == "Plan sprint 8"

    def test_returns_parsed_json(self):
        agent = SprintPlannerAgent(llm=_make_llm(VALID_PLAN_JSON))
        result = agent.run("Plan sprint 8")
        assert result["sprint"] == 8
        assert result["goal"] == "Deploy SHIR and establish medallion bronze layer"
        assert len(result["tasks"]) == 2

    def test_returns_parse_error_on_garbage(self):
        agent = SprintPlannerAgent(llm=_make_llm("Sure! Here is your plan..."))
        result = agent.run("Plan sprint 8")
        assert "raw_output" in result
        assert "parse_error" in result

    def test_full_happy_path(self):
        agent = SprintPlannerAgent(llm=_make_llm(VALID_PLAN_JSON))
        result = agent.run("Plan sprint 8")
        assert result["sprint"] == 8
        assert result["goal"] == "Deploy SHIR and establish medallion bronze layer"
        assert len(result["tasks"]) == 2
        assert result["tasks"][0]["id"] == "SP8-001"
        assert result["tasks"][1]["id"] == "SP8-002"
        assert len(result["dependencies"]) == 1
        assert result["dependencies"][0]["from"] == "SP8-002"
        assert result["dependencies"][0]["to"] == "SP8-001"


class TestPromptExternalization:
    """Tests for externalized Jinja2 prompt loading."""

    def test_prompts_load_from_files(self):
        from agents.base import BaseAgent

        system = BaseAgent.load_prompt("sprint_planner/system.j2")
        assert len(system) > 0
        assert "Sprint Planner" in system

    def test_context_template_loads(self):
        from agents.base import BaseAgent

        ctx = BaseAgent.load_prompt(
            "sprint_planner/context.j2",
            sprint_name="Sprint 8",
            sprint_goal="Test goal",
            start_date="2026-03-01",
            end_date="2026-03-14",
            work_items_summary="- item 1",
            extra_context="",
            page_content_section="",
        )
        assert "Sprint 8" in ctx
        assert "Test goal" in ctx

    def test_few_shots_in_system_message(self):
        llm = _make_llm(VALID_PLAN_JSON)
        agent = SprintPlannerAgent(llm=llm)
        agent.run("Plan sprint 8")

        system_msg = llm.invoke.call_args[0][0][0].content
        assert "Example" in system_msg
        assert "SP7-001" in system_msg

    def test_system_prompt_has_json_schema(self):
        from agents.base import BaseAgent

        system = BaseAgent.load_prompt("sprint_planner/system.j2")
        assert '"sprint"' in system
        assert '"tasks"' in system
        assert '"dependencies"' in system

    def test_missing_template_raises(self):
        from agents.base import BaseAgent
        from jinja2 import TemplateNotFound

        with pytest.raises(TemplateNotFound):
            BaseAgent.load_prompt("nonexistent/missing.j2")


class TestFormatPageContent:
    """Tests for _format_page_content()."""

    def test_empty_when_no_context(self):
        agent = SprintPlannerAgent(llm=_make_llm(""), context=None)
        assert agent._format_page_content() == ""

    def test_empty_when_no_page_content_key(self):
        ctx = _make_context()
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        assert agent._format_page_content() == ""

    def test_formats_content_with_entity_names(self):
        ctx = _make_context(work_items=[WORK_ITEMS[0]])
        ctx["page_content"] = {"wi-001": "Some markdown content here."}
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_page_content()
        assert "Deploy VM" in result
        assert "Some markdown content here." in result
        assert "ITEM DETAILS" in result

    def test_uses_notion_id_when_name_not_found(self):
        ctx = _make_context()
        ctx["page_content"] = {"unknown-id": "Content for unknown."}
        agent = SprintPlannerAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_page_content()
        assert "unknown-id" in result


class TestRunIntegration:
    def test_end_to_end_with_context(self):
        llm = _make_llm(VALID_PLAN_JSON)
        ctx = _make_context(
            sprints=[ACTIVE_SPRINT],
            work_items=WORK_ITEMS[:3],
            risks=[OPEN_RISK],
        )
        agent = SprintPlannerAgent(llm=llm, context=ctx)
        result = agent.run("Plan sprint 8")

        # Verify system message contains context data
        system_msg = llm.invoke.call_args[0][0][0].content
        assert "Sprint 8" in system_msg
        assert "Deploy VM" in system_msg
        assert "Configure VNet" in system_msg
        assert "Write docs" not in system_msg  # filtered out (different sprint)
        assert "VRAM limit" in system_msg

        # Verify parsed result
        assert result["sprint"] == 8
        assert len(result["tasks"]) == 2
        assert len(result["dependencies"]) == 1
