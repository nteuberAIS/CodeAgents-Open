"""Coder agent — translates task descriptions into Aider coding instructions.

Takes a sprint task (JSON string from SprintState), uses the LLM to generate
a precise Aider-compatible instruction and file list, then calls AiderTool
to make the code changes. Includes an internal reflection loop: on failure,
error context is appended and the LLM is re-invoked (up to MAX_ITERATIONS).

This is the second agent in the cascade (after SprintPlannerAgent).
Git branch management is the orchestrator's responsibility.
"""

from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agents.base import BaseAgent


class CoderAgent(BaseAgent):
    """Generates Aider coding instructions and executes them."""

    name = "coder"
    MAX_ITERATIONS = 5  # code -> test -> fix cycles (per failure-modes.md)

    # Tools this agent can use
    REQUIRED_TOOLS: list[str] = ["aider"]
    OPTIONAL_TOOLS: list[str] = ["azdevops", "github"]

    # Context curation — coder needs focused context, not broad overview
    MAX_CONTENT_ITEMS: int = 3
    MAX_CONTENT_CHARS: int = 4000
    CONTENT_STATUSES: list[str] = ["Ready", "In Progress"]

    def run(self, user_input: str) -> dict:
        """Generate an Aider instruction and execute it, retrying on failure.

        Args:
            user_input: Task description — either a JSON string of a task dict
                        (e.g. '{"id": "SP8-001", "title": "...", ...}') or a
                        natural language description.

        Returns:
            Standard envelope via wrap_result(). On success, partial_output
            contains: instruction, modified_files, aider_output,
            iterations_used, dry_run.
        """
        # 1. Check aider tool is available
        aider = self.get_tool("aider")
        if aider is None:
            return self.wrap_result(
                success=False,
                partial_output={},
                error_type="tool",
                error_message="aider tool not available",
            )

        # 2. Build initial messages
        system_content = self.load_prompt("coder/system.j2")
        if self.context:
            system_content += "\n\n" + self._format_context()

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=user_input),
        ]

        # 3. Reflection loop
        last_instruction = None
        last_error = None

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            # Invoke LLM
            response = self.llm.invoke(messages)
            raw = response.content
            parsed = self._parse_response(raw)

            # Parse error — retry with feedback
            if "parse_error" in parsed:
                if iteration < self.MAX_ITERATIONS:
                    messages.append(AIMessage(content=raw))
                    messages.append(HumanMessage(
                        content=(
                            "Your response was not valid JSON. "
                            f"Error: {parsed['parse_error']}. "
                            "Please return ONLY a JSON object with "
                            "'instruction' and 'files' keys."
                        ),
                    ))
                    continue
                return self.wrap_result(
                    success=False,
                    partial_output={
                        "raw_output": raw,
                        "iterations_used": iteration,
                    },
                    error_type="llm",
                    error_message=parsed["parse_error"],
                )

            # Validate instruction field
            instruction = parsed.get("instruction", "")
            files = parsed.get("files", [])

            if not instruction:
                if iteration < self.MAX_ITERATIONS:
                    messages.append(AIMessage(content=raw))
                    messages.append(HumanMessage(
                        content=(
                            "The instruction field is empty. "
                            "Please provide a specific coding instruction."
                        ),
                    ))
                    continue
                return self.wrap_result(
                    success=False,
                    partial_output={
                        "raw_output": raw,
                        "iterations_used": iteration,
                    },
                    error_type="llm",
                    error_message="LLM returned empty instruction",
                )

            last_instruction = instruction

            # Call Aider
            result = aider.edit(instruction=instruction, files=files)

            if result.success:
                return self.wrap_result(
                    success=True,
                    partial_output={
                        "instruction": instruction,
                        "modified_files": result.modified_files,
                        "aider_output": self._truncate(result.output, 4000),
                        "iterations_used": iteration,
                        "dry_run": result.dry_run,
                    },
                )

            # Aider failed — retry with error context
            last_error = result.error or result.output or "Unknown Aider error"
            if iteration < self.MAX_ITERATIONS:
                messages.append(AIMessage(content=raw))
                messages.append(HumanMessage(
                    content=(
                        f"Aider failed with error:\n{last_error}\n\n"
                        "Please revise your instruction and file list "
                        "to fix this issue. Return ONLY a JSON object "
                        "with 'instruction' and 'files' keys."
                    ),
                ))

        # Max iterations exceeded
        return self.wrap_result(
            success=False,
            partial_output={
                "last_instruction": last_instruction,
                "last_error": last_error,
                "iterations_used": self.MAX_ITERATIONS,
            },
            error_type="tool",
            error_message=(
                f"Aider failed after {self.MAX_ITERATIONS} attempts: "
                f"{last_error}"
            ),
        )

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM output as JSON, stripping markdown fences."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"parse_error": "LLM did not return valid JSON"}

    def _format_context(self) -> str:
        """Format context data for the LLM system prompt."""
        if not self.context:
            return ""

        parts: list[str] = []

        # Work items section
        work_items = self.context.get("work_items", [])
        if work_items:
            items = work_items[: self.MAX_CONTENT_ITEMS]
            parts.append("--- RELATED WORK ITEMS ---")
            for item in items:
                name = item.get("name", "Untitled")
                status = item.get("status", "unknown")
                desc = item.get("description", "")
                parts.append(f"- {name} ({status}): {desc}")

        # Page content section
        page_content = self.context.get("page_content", {})
        if page_content:
            parts.append("\n--- REFERENCE CONTENT ---")
            for notion_id, content in list(page_content.items())[
                : self.MAX_CONTENT_ITEMS
            ]:
                if len(content) > self.MAX_CONTENT_CHARS:
                    content = content[: self.MAX_CONTENT_CHARS] + "\n... (truncated)"
                parts.append(f"### {notion_id}")
                parts.append(content)
                parts.append("")

        return "\n".join(parts)

    @staticmethod
    def _truncate(text: str | None, max_len: int) -> str | None:
        """Truncate text to max_len, appending suffix if shortened."""
        if text is None:
            return None
        if len(text) <= max_len:
            return text
        return text[: max_len] + "... (truncated)"
