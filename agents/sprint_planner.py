"""Sprint Planner agent — generates a sprint plan from a user prompt.

Phase 1: Calls the LLM with a structured system prompt, parses JSON output.
No real Notion integration yet — console output only.

Future enhancements (marked in comments):
- RAG context injection from local Notion mirror
- Tool calls to create Notion pages and git branches
- LangGraph node registration for multi-agent cascade
- Reflection/validation loop with iteration cap
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent

SYSTEM_PROMPT = """\
You are a Sprint Planner for a software development team.

Given a sprint planning request, produce a JSON object with this exact schema:
{
  "sprint": <sprint number as integer>,
  "goal": "<one-line sprint goal>",
  "tasks": [
    {
      "id": "<short task ID like SP8-001>",
      "title": "<task title>",
      "description": "<brief description>",
      "assignee": null,
      "story_points": <integer 1-8>,
      "status": "todo"
    }
  ],
  "dependencies": [
    {"from": "<task_id>", "to": "<task_id>", "type": "blocks"}
  ]
}

Rules:
- Generate 4-6 realistic tasks for a data platform project.
- Include 1-2 dependencies between tasks.
- Return ONLY the JSON object, no markdown fences, no explanation.
"""

# Future: Inject RAG context here
# RAG_CONTEXT_TEMPLATE = """
# The following context is retrieved from the team's Notion workspace:
# {rag_context}
#
# Use this context to inform your sprint plan.
# """


class SprintPlannerAgent(BaseAgent):
    """Generates a structured sprint plan from a natural language prompt."""

    name = "sprint_planner"

    def run(self, user_input: str) -> dict:
        """Generate a sprint plan.

        Args:
            user_input: e.g. "Plan sprint 8"

        Returns:
            Parsed JSON sprint plan, or {"raw_output": ...} on parse failure.
        """
        # Future: Inject RAG context from Notion mirror
        # rag_context = self._retrieve_context(user_input)
        # system_content = SYSTEM_PROMPT + RAG_CONTEXT_TEMPLATE.format(rag_context=rag_context)
        system_content = SYSTEM_PROMPT

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        response = self.llm.invoke(messages)
        raw = response.content

        return self._parse_response(raw)

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM output as JSON with fallback."""
        # Strip markdown fences if the model wraps output
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_output": raw, "parse_error": "LLM did not return valid JSON"}

    # -- Future extension points --
    #
    # def _retrieve_context(self, query: str) -> str:
    #     """Pull relevant docs from local Notion mirror via RAG retriever."""
    #     pass
    #
    # def _create_notion_tasks(self, plan: dict):
    #     """Push tasks to Notion database via tools.notion_tool."""
    #     pass
    #
    # def _create_git_branches(self, plan: dict):
    #     """Create feature branches in Azure DevOps via tools.git_tool."""
    #     pass
