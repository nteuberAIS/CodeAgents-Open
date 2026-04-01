"""Sprint Planner agent — generates a sprint plan from a user prompt.

When local Notion data is available (via context injection), plans against
real backlog data. Without context, falls back to LLM-generated tasks.

Post-planning execution:
- If notion_write tool is bound: creates work items in local Notion snapshot
- If a git tool is bound: creates sprint + task branches
- Both are optional — agent degrades gracefully without tools
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent

class SprintPlannerAgent(BaseAgent):
    """Generates a structured sprint plan from a natural language prompt."""

    name = "sprint_planner"
    MAX_ITERATIONS = 2  # Near-one-shot: retry once on JSON parse failure

    # Tools this agent can use
    REQUIRED_TOOLS: list[str] = []  # None required — agent works without tools
    OPTIONAL_TOOLS: list[str] = ["notion_write", "github", "azdevops"]

    # Context curation — planner needs broad overview, short snippets
    MAX_CONTENT_ITEMS: int = 5
    MAX_CONTENT_CHARS: int = 6000
    CONTENT_STATUSES: list[str] = ["Ready", "In Progress", "Backlog", "Active"]

    def run(self, user_input: str) -> dict:
        """Generate a sprint plan, optionally execute it.

        Args:
            user_input: e.g. "Plan sprint 8"

        Returns:
            Standard envelope with partial_output containing the plan keys
            (sprint, goal, tasks, dependencies) and optional "execution".
        """
        # Phase 1: Generate plan via LLM
        system_content = self.load_prompt("sprint_planner/system.j2")
        if self.context:
            system_content += self._format_context()

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        response = self.llm.invoke(messages)
        raw = response.content
        plan = self._parse_response(raw)

        # Parse failure — wrap as LLM error
        if "parse_error" in plan:
            return self.wrap_result(
                success=False,
                error_type="llm",
                error_message=plan["parse_error"],
                partial_output={"raw_output": plan["raw_output"]},
            )

        # Phase 2: Execute plan if tools are available
        if self.tools:
            plan["execution"] = self._execute_plan(plan)

        return self.wrap_result(success=True, partial_output=plan)

    def _execute_plan(self, plan: dict) -> dict:
        """Execute a sprint plan by creating Notion items and git branches.

        This is a best-effort operation — individual failures are captured
        in the errors list, not raised.

        Returns:
            {
                "notion_items_created": [...],
                "branches_created": [...],
                "errors": [...]
            }
        """
        execution: dict = {
            "notion_items_created": [],
            "branches_created": [],
            "errors": [],
        }

        sprint_number = plan.get("sprint")
        tasks = plan.get("tasks", [])

        # --- Create Notion work items ---
        notion_write = self.get_tool("notion_write")
        if notion_write:
            for task in tasks:
                try:
                    item = notion_write.create_work_item(
                        name=task["title"],
                        type="Task",
                        status=task.get("status", "todo"),
                        estimate_hrs=task.get("estimate_hrs"),
                    )
                    execution["notion_items_created"].append({
                        "task_id": task["id"],
                        "notion_id": item.notion_id,
                        "title": item.name,
                    })
                except Exception as e:
                    execution["errors"].append(
                        f"Notion create failed for {task.get('id', '?')}: {e}"
                    )

        # --- Create git branches ---
        git_tool = self.get_tool("github") or self.get_tool("azdevops")
        if git_tool and sprint_number:
            # Create sprint branch first
            try:
                sprint_branch = git_tool.sprint_branch_name(sprint_number)
                result = git_tool.create_branch(sprint_branch, from_ref="main")
                execution["branches_created"].append({
                    "task_id": None,
                    "branch": sprint_branch,
                    "success": result.success,
                    "dry_run": result.dry_run,
                })
            except Exception as e:
                execution["errors"].append(f"Sprint branch creation failed: {e}")

            # Create task branches (from main, per project convention)
            for task in tasks:
                try:
                    task_id = task.get("id", "")
                    branch_name = git_tool.task_branch_name(sprint_number, task_id)
                    result = git_tool.create_branch(branch_name, from_ref="main")
                    execution["branches_created"].append({
                        "task_id": task_id,
                        "branch": branch_name,
                        "success": result.success,
                        "dry_run": result.dry_run,
                    })
                except Exception as e:
                    execution["errors"].append(
                        f"Branch creation failed for {task.get('id', '?')}: {e}"
                    )

        return execution

    def _format_context(self) -> str:
        """Format local Notion data as context for the LLM."""
        if not self.context:
            return ""

        sprints = self.context.get("sprints", [])
        work_items = self.context.get("work_items", [])
        risks = self.context.get("risks", [])

        # Find target sprint: prefer "Active", then "Not started", then "In Progress"
        active = None
        for status in ("Active", "Not started", "In Progress"):
            for s in sprints:
                if s.get("status") == status:
                    active = s
                    break
            if active:
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

        page_content_section = self._format_page_content()

        return self.load_prompt(
            "sprint_planner/context.j2",
            sprint_name=sprint_name,
            sprint_goal=sprint_goal,
            start_date=start_date,
            end_date=end_date,
            work_items_summary=work_items_summary,
            extra_context=extra_context,
            page_content_section=page_content_section,
        )

    def _format_page_content(self) -> str:
        """Format loaded page content snippets for the LLM context."""
        if not self.context:
            return ""
        page_content = self.context.get("page_content", {})
        if not page_content:
            return ""

        lines = ["\n--- ITEM DETAILS (selected pages) ---\n"]
        for notion_id, content in page_content.items():
            name = self._resolve_entity_name(notion_id)
            lines.append(f"### {name}")
            lines.append(content)
            lines.append("")
        return "\n".join(lines)

    def _resolve_entity_name(self, notion_id: str) -> str:
        """Look up the display name for a notion_id across all entity types."""
        if not self.context:
            return notion_id
        for key in ("work_items", "sprints", "docs", "decisions", "risks"):
            for entity in self.context.get(key, []):
                if entity.get("notion_id") == notion_id:
                    return entity.get("name", notion_id)
        return notion_id

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM output as JSON with fallback."""
        # Strip markdown fences if the model wraps output
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"raw_output": raw, "parse_error": "LLM did not return valid JSON"}
