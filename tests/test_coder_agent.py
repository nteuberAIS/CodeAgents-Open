"""Tests for agents/coder.py.

All tests mock the LLM and AiderTool — no Ollama or Aider calls are made.
"""

import json
from unittest.mock import MagicMock

import pytest

from agents.coder import CoderAgent
from schemas.aider_models import AiderResult


# -- Test constants --

VALID_CODER_JSON = json.dumps({
    "instruction": "Add error handling to src/pipeline.py ingest() function",
    "files": ["src/pipeline.py"],
})

MULTI_FILE_JSON = json.dumps({
    "instruction": "Create schemas/bronze.py and update tests/test_bronze.py",
    "files": ["schemas/bronze.py", "tests/test_bronze.py"],
})

EMPTY_INSTRUCTION_JSON = json.dumps({
    "instruction": "",
    "files": ["src/pipeline.py"],
})

TASK_INPUT = json.dumps({
    "id": "SP8-001",
    "title": "Add error handling",
    "description": "Add try/except to ingest() function",
    "estimate_hrs": 3,
    "status": "todo",
})


# -- Helpers --

def _make_llm(response_content: str) -> MagicMock:
    """Create a mock ChatOllama that returns a fixed response."""
    llm = MagicMock()
    mock_response = MagicMock()
    mock_response.content = response_content
    llm.invoke.return_value = mock_response
    return llm


def _make_llm_sequence(responses: list[str]) -> MagicMock:
    """Create a mock ChatOllama that returns different responses on each call."""
    llm = MagicMock()
    mock_responses = []
    for content in responses:
        resp = MagicMock()
        resp.content = content
        mock_responses.append(resp)
    llm.invoke.side_effect = mock_responses
    return llm


def _make_aider_result(
    success: bool = True,
    output: str | None = "Wrote src/pipeline.py",
    error: str | None = None,
    modified_files: list[str] | None = None,
    dry_run: bool = False,
) -> AiderResult:
    """Create an AiderResult for mocking."""
    return AiderResult(
        command="aider --message ... --model ollama/qwen2.5-coder:7b",
        success=success,
        output=output,
        error=error,
        modified_files=modified_files or (["src/pipeline.py"] if success else []),
        dry_run=dry_run,
    )


def _make_agent_with_aider(
    llm: MagicMock,
    aider_result: AiderResult | None = None,
    aider_side_effect: list[AiderResult] | None = None,
    context: dict | None = None,
) -> CoderAgent:
    """Create a CoderAgent with a mocked aider tool."""
    agent = CoderAgent(llm=llm, context=context)
    mock_aider = MagicMock()
    if aider_side_effect is not None:
        mock_aider.edit.side_effect = aider_side_effect
    elif aider_result is not None:
        mock_aider.edit.return_value = aider_result
    else:
        mock_aider.edit.return_value = _make_aider_result()
    agent.tools["aider"] = mock_aider
    return agent


# -- Test Classes --

class TestCoderInit:
    def test_name_attribute(self):
        agent = CoderAgent(llm=_make_llm(""))
        assert agent.name == "coder"

    def test_max_iterations(self):
        assert CoderAgent.MAX_ITERATIONS == 5

    def test_required_tools(self):
        assert CoderAgent.REQUIRED_TOOLS == ["aider"]

    def test_optional_tools(self):
        assert "azdevops" in CoderAgent.OPTIONAL_TOOLS
        assert "github" in CoderAgent.OPTIONAL_TOOLS

    def test_stores_llm(self):
        llm = _make_llm("")
        agent = CoderAgent(llm=llm)
        assert agent.llm is llm

    def test_stores_context(self):
        ctx = {"work_items": []}
        agent = CoderAgent(llm=_make_llm(""), context=ctx)
        assert agent.context is ctx

    def test_context_defaults_to_none(self):
        agent = CoderAgent(llm=_make_llm(""))
        assert agent.context is None

    def test_content_curation_attrs(self):
        assert CoderAgent.MAX_CONTENT_ITEMS == 3
        assert CoderAgent.MAX_CONTENT_CHARS == 4000


class TestParseResponse:
    def setup_method(self):
        self.agent = CoderAgent(llm=_make_llm(""))

    def test_valid_json(self):
        result = self.agent._parse_response('{"instruction": "do something", "files": ["a.py"]}')
        assert result == {"instruction": "do something", "files": ["a.py"]}

    def test_json_fenced_with_language_tag(self):
        raw = '```json\n{"instruction": "fix it", "files": ["b.py"]}\n```'
        result = self.agent._parse_response(raw)
        assert result == {"instruction": "fix it", "files": ["b.py"]}

    def test_json_fenced_without_language_tag(self):
        raw = '```\n{"instruction": "fix it", "files": ["b.py"]}\n```'
        result = self.agent._parse_response(raw)
        assert result == {"instruction": "fix it", "files": ["b.py"]}

    def test_invalid_json(self):
        result = self.agent._parse_response("not json at all")
        assert "parse_error" in result
        assert "JSON" in result["parse_error"]

    def test_empty_string(self):
        result = self.agent._parse_response("")
        assert "parse_error" in result

    def test_json_with_whitespace(self):
        raw = '  \n  {"instruction": "x", "files": []}  \n  '
        result = self.agent._parse_response(raw)
        assert result == {"instruction": "x", "files": []}


class TestFormatContext:
    def test_no_context(self):
        agent = CoderAgent(llm=_make_llm(""), context=None)
        assert agent._format_context() == ""

    def test_empty_context(self):
        agent = CoderAgent(llm=_make_llm(""), context={})
        assert agent._format_context() == ""

    def test_work_items_formatted(self):
        ctx = {
            "work_items": [
                {"name": "Deploy VM", "status": "Ready", "description": "Install VM"},
                {"name": "Configure VNet", "status": "In Progress", "description": "Set up VNet"},
            ],
        }
        agent = CoderAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "RELATED WORK ITEMS" in result
        assert "Deploy VM" in result
        assert "Configure VNet" in result
        assert "Ready" in result

    def test_page_content_included(self):
        ctx = {
            "work_items": [],
            "page_content": {"wi-001": "Some reference markdown content."},
        }
        agent = CoderAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        assert "REFERENCE CONTENT" in result
        assert "Some reference markdown content." in result

    def test_max_content_items_respected(self):
        ctx = {
            "work_items": [
                {"name": f"Item {i}", "status": "Ready", "description": f"Desc {i}"}
                for i in range(10)
            ],
        }
        agent = CoderAgent(llm=_make_llm(""), context=ctx)
        result = agent._format_context()
        # Only MAX_CONTENT_ITEMS (3) should appear
        assert "Item 0" in result
        assert "Item 2" in result
        assert "Item 3" not in result


class TestRunHappyPath:
    def test_success_returns_correct_envelope(self):
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(),
        )
        result = agent.run(TASK_INPUT)
        assert result["success"] is True
        assert result["error_type"] is None
        assert result["error_message"] is None

    def test_partial_output_keys(self):
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(),
        )
        result = agent.run(TASK_INPUT)
        po = result["partial_output"]
        assert "instruction" in po
        assert "modified_files" in po
        assert "aider_output" in po
        assert "iterations_used" in po
        assert "dry_run" in po

    def test_iterations_used_is_one_on_first_success(self):
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(),
        )
        result = agent.run(TASK_INPUT)
        assert result["partial_output"]["iterations_used"] == 1

    def test_system_prompt_loaded(self):
        llm = _make_llm(VALID_CODER_JSON)
        agent = _make_agent_with_aider(llm=llm)
        agent.run(TASK_INPUT)

        messages = llm.invoke.call_args[0][0]
        assert len(messages) == 2
        assert "Coder Agent" in messages[0].content

    def test_user_input_as_human_message(self):
        llm = _make_llm(VALID_CODER_JSON)
        agent = _make_agent_with_aider(llm=llm)
        agent.run(TASK_INPUT)

        messages = llm.invoke.call_args[0][0]
        assert messages[1].content == TASK_INPUT

    def test_context_appended_to_system_message(self):
        ctx = {
            "work_items": [
                {"name": "Deploy VM", "status": "Ready", "description": "Install VM"},
            ],
        }
        llm = _make_llm(VALID_CODER_JSON)
        agent = _make_agent_with_aider(llm=llm, context=ctx)
        agent.run(TASK_INPUT)

        system_msg = llm.invoke.call_args[0][0][0].content
        assert "Deploy VM" in system_msg

    def test_aider_called_with_correct_args(self):
        llm = _make_llm(VALID_CODER_JSON)
        agent = _make_agent_with_aider(llm=llm)
        agent.run(TASK_INPUT)

        aider = agent.tools["aider"]
        aider.edit.assert_called_once_with(
            instruction="Add error handling to src/pipeline.py ingest() function",
            files=["src/pipeline.py"],
        )

    def test_dry_run_flag_propagated(self):
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(dry_run=True, output=None, modified_files=[]),
        )
        result = agent.run(TASK_INPUT)
        assert result["success"] is True
        assert result["partial_output"]["dry_run"] is True

    def test_multi_file_instruction(self):
        agent = _make_agent_with_aider(
            llm=_make_llm(MULTI_FILE_JSON),
            aider_result=_make_aider_result(
                modified_files=["schemas/bronze.py", "tests/test_bronze.py"]
            ),
        )
        result = agent.run(TASK_INPUT)
        assert result["success"] is True
        aider = agent.tools["aider"]
        call_kwargs = aider.edit.call_args[1] if aider.edit.call_args[1] else {}
        call_args = aider.edit.call_args
        # Verify both files were passed
        assert "schemas/bronze.py" in call_args.kwargs.get("files", call_args[1]["files"] if call_args[1] else []) or \
               "schemas/bronze.py" in (call_args[0][1] if len(call_args[0]) > 1 else [])


class TestRunReflectionLoop:
    def test_parse_error_then_success(self):
        llm = _make_llm_sequence([
            "not valid json",
            VALID_CODER_JSON,
        ])
        agent = _make_agent_with_aider(llm=llm)
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["partial_output"]["iterations_used"] == 2
        assert llm.invoke.call_count == 2

    def test_retry_messages_grow(self):
        """After a parse error, the next LLM call gets 4 messages."""
        llm = _make_llm_sequence([
            "not valid json",
            VALID_CODER_JSON,
        ])
        agent = _make_agent_with_aider(llm=llm)
        agent.run(TASK_INPUT)

        # Second call should have 4 messages: System, Human, AI(bad), Human(feedback)
        second_call_messages = llm.invoke.call_args_list[1][0][0]
        assert len(second_call_messages) == 4

    def test_aider_failure_then_success(self):
        llm = _make_llm_sequence([
            VALID_CODER_JSON,
            VALID_CODER_JSON,
        ])
        agent = _make_agent_with_aider(
            llm=llm,
            aider_side_effect=[
                _make_aider_result(success=False, error="syntax error", output=None, modified_files=[]),
                _make_aider_result(success=True),
            ],
        )
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["partial_output"]["iterations_used"] == 2

    def test_aider_failure_feedback_in_messages(self):
        """After aider failure, error is included in retry prompt."""
        llm = _make_llm_sequence([
            VALID_CODER_JSON,
            VALID_CODER_JSON,
        ])
        agent = _make_agent_with_aider(
            llm=llm,
            aider_side_effect=[
                _make_aider_result(success=False, error="syntax error in line 5", output=None, modified_files=[]),
                _make_aider_result(success=True),
            ],
        )
        agent.run(TASK_INPUT)

        second_call_messages = llm.invoke.call_args_list[1][0][0]
        # Should contain the error message
        feedback = second_call_messages[-1].content
        assert "syntax error in line 5" in feedback

    def test_empty_instruction_then_valid(self):
        llm = _make_llm_sequence([
            EMPTY_INSTRUCTION_JSON,
            VALID_CODER_JSON,
        ])
        agent = _make_agent_with_aider(llm=llm)
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["partial_output"]["iterations_used"] == 2

    def test_max_iterations_exceeded_parse_error(self):
        """All iterations fail with parse errors."""
        llm = _make_llm_sequence(["bad json"] * 5)
        agent = _make_agent_with_aider(llm=llm)
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "llm"
        assert "JSON" in result["error_message"]
        assert result["partial_output"]["iterations_used"] == 5

    def test_max_iterations_exceeded_aider_failure(self):
        """All iterations fail with aider errors."""
        llm = _make_llm_sequence([VALID_CODER_JSON] * 5)
        agent = _make_agent_with_aider(
            llm=llm,
            aider_side_effect=[
                _make_aider_result(success=False, error=f"error {i}", output=None, modified_files=[])
                for i in range(5)
            ],
        )
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "tool"
        assert "5 attempts" in result["error_message"]
        assert result["partial_output"]["iterations_used"] == 5
        assert result["partial_output"]["last_instruction"] is not None
        assert result["partial_output"]["last_error"] is not None

    def test_max_iterations_exceeded_empty_instruction(self):
        """All iterations return empty instructions."""
        llm = _make_llm_sequence([EMPTY_INSTRUCTION_JSON] * 5)
        agent = _make_agent_with_aider(llm=llm)
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "llm"
        assert "empty instruction" in result["error_message"]


class TestRunEdgeCases:
    def test_no_aider_tool(self):
        """Agent without aider tool returns tool error immediately."""
        agent = CoderAgent(llm=_make_llm(VALID_CODER_JSON))
        # No tools bound at all
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "tool"
        assert "aider tool not available" in result["error_message"]

    def test_aider_tool_is_none(self):
        """Aider tool bound as None (failed init) returns tool error."""
        agent = CoderAgent(llm=_make_llm(VALID_CODER_JSON))
        agent.tools["aider"] = None
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert result["error_type"] == "tool"
        assert "aider tool not available" in result["error_message"]

    def test_long_aider_output_truncated(self):
        long_output = "x" * 5000
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(output=long_output),
        )
        result = agent.run(TASK_INPUT)

        aider_output = result["partial_output"]["aider_output"]
        assert len(aider_output) < 5000
        assert "truncated" in aider_output

    def test_none_aider_output_handled(self):
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(output=None),
        )
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["partial_output"]["aider_output"] is None

    def test_aider_success_no_modified_files_retries(self):
        """Regression: Aider exit 0 with no files modified should retry, not succeed."""
        llm = _make_llm_sequence([VALID_CODER_JSON] * 5)
        # Build AiderResult directly to avoid _make_aider_result's falsy-[] default
        empty_result = AiderResult(
            command="aider --message ...",
            success=True,
            output="No changes needed",
            error=None,
            modified_files=[],
            dry_run=False,
        )
        agent = _make_agent_with_aider(
            llm=llm,
            aider_side_effect=[empty_result for _ in range(5)],
        )
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert "no files were modified" in result["partial_output"]["last_error"].lower()
        # LLM should have been called MAX_ITERATIONS times (retries on empty result)
        assert llm.invoke.call_count == 5

    def test_aider_success_no_modified_files_dry_run_still_succeeds(self):
        """Dry-run with no files is expected — should still succeed."""
        agent = _make_agent_with_aider(
            llm=_make_llm(VALID_CODER_JSON),
            aider_result=_make_aider_result(
                success=True,
                output=None,
                modified_files=[],
                dry_run=True,
            ),
        )
        result = agent.run(TASK_INPUT)

        assert result["success"] is True
        assert result["partial_output"]["dry_run"] is True

    def test_aider_error_uses_output_as_fallback(self):
        """When error is None, output is used as the error context."""
        llm = _make_llm_sequence([VALID_CODER_JSON] * 5)
        agent = _make_agent_with_aider(
            llm=llm,
            aider_side_effect=[
                _make_aider_result(
                    success=False,
                    error=None,
                    output="some failure output",
                    modified_files=[],
                )
                for _ in range(5)
            ],
        )
        result = agent.run(TASK_INPUT)

        assert result["success"] is False
        assert "some failure output" in result["partial_output"]["last_error"]


class TestTruncate:
    def test_short_text_unchanged(self):
        assert CoderAgent._truncate("hello", 100) == "hello"

    def test_long_text_truncated(self):
        result = CoderAgent._truncate("a" * 200, 50)
        assert len(result) < 200
        assert result.endswith("... (truncated)")
        assert result.startswith("a" * 50)

    def test_none_returns_none(self):
        assert CoderAgent._truncate(None, 100) is None

    def test_exact_length_unchanged(self):
        text = "a" * 100
        assert CoderAgent._truncate(text, 100) == text


class TestPromptExternalization:
    def test_system_prompt_loads(self):
        from agents.base import BaseAgent

        system = BaseAgent.load_prompt("coder/system.j2")
        assert len(system) > 0
        assert "Coder Agent" in system

    def test_few_shots_included(self):
        from agents.base import BaseAgent

        system = BaseAgent.load_prompt("coder/system.j2")
        assert "Example" in system
        assert "instruction" in system
        assert "files" in system

    def test_system_prompt_has_json_schema(self):
        from agents.base import BaseAgent

        system = BaseAgent.load_prompt("coder/system.j2")
        assert '"instruction"' in system
        assert '"files"' in system

    def test_missing_template_raises(self):
        from agents.base import BaseAgent
        from jinja2 import TemplateNotFound

        with pytest.raises(TemplateNotFound):
            BaseAgent.load_prompt("coder/nonexistent.j2")
