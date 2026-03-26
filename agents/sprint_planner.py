"""Sprint Planner agent — generates a sprint plan from a user prompt.

When local Notion data is available (via context injection), plans against
real backlog data. Without context, falls back to LLM-generated tasks.

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
You are a Sprint Planner for a solo-developer data platform project.

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
      "estimate_hrs": <number of hours>,
      "status": "todo"
    }
  ],
  "dependencies": [
    {"from": "<task_id>", "to": "<task_id>", "type": "blocks"}
  ]
}

Rules:
- Generate 4-6 realistic tasks for a data platform project.
- Use estimate_hrs (hours) for effort estimates, not story points.
- Include 1-2 dependencies between tasks.
- Return ONLY the JSON object, no markdown fences, no explanation.
"""

CONTEXT_TEMPLATE = """\

--- CURRENT BACKLOG (from Notion) ---

Active Sprint: {sprint_name}
Sprint Goal: {sprint_goal}
Sprint Dates: {start_date} → {end_date}

Work Items in this sprint:
{work_items_summary}

{extra_context}\
Use the above backlog data to inform your sprint plan. Reference existing
work item names where relevant. Prioritize items that are Ready or Backlog.
"""


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
        system_content = SYSTEM_PROMPT
        if self.context:
            system_content += self._format_context()

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        response = self.llm.invoke(messages)
        raw = response.content

        return self._parse_response(raw)

    def _format_context(self) -> str:
        """Format local Notion data as context for the LLM."""
        if not self.context:
            return ""

        sprints = self.context.get("sprints", [])
        work_items = self.context.get("work_items", [])
        risks = self.context.get("risks", [])

        # Find active sprint
        active = None
        for s in sprints:
            if s.get("status") == "Active":
                active = s
                break

        sprint_name = active["name"] if active else "No active sprint"
        sprint_goal = active.get("goal", "N/A") if active else "N/A"
        start_date = active.get("start_date", "N/A") if active else "N/A"
        end_date = active.get("end_date", "N/A") if active else "N/A"

        # Filter work items for active sprint
        if active:
            sprint_id = active.get("notion_id")
            sprint_items = [
                wi for wi in work_items if wi.get("sprint_id") == sprint_id
            ]
        else:
            sprint_items = work_items

        # Format work items summary
        lines = []
        for wi in sprint_items:
            status = wi.get("status", "?")
            priority = wi.get("priority", "?")
            est = wi.get("estimate_hrs")
            est_str = f"{est}h" if est else "?"
            name = wi.get("name", "Untitled")
            wi_type = wi.get("type", "?")
            lines.append(f"- [{status}] [{priority}] {name} ({wi_type}, {est_str})")

        work_items_summary = "\n".join(lines) if lines else "No work items found."

        # Add risks if any are open
        extra_parts = []
        open_risks = [r for r in risks if r.get("status") == "Open"]
        if open_risks:
            risk_lines = []
            for r in open_risks:
                sev = r.get("severity", "?")
                risk_lines.append(f"- [{sev}] {r.get('name', 'Untitled')}")
            extra_parts.append("Open Risks:\n" + "\n".join(risk_lines))

        extra_context = "\n\n".join(extra_parts) + "\n" if extra_parts else ""

        return CONTEXT_TEMPLATE.format(
            sprint_name=sprint_name,
            sprint_goal=sprint_goal,
            start_date=start_date,
            end_date=end_date,
            work_items_summary=work_items_summary,
            extra_context=extra_context,
        )

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
