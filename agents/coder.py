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

import logging

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


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
        # 0. Parse task info for RAG/snapshot queries
        self._current_task: dict = {}
        try:
            self._current_task = json.loads(user_input)
        except (json.JSONDecodeError, TypeError):
            pass

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
            logger.info(
                "coder iteration %d: files=%s, instruction=%.100s",
                iteration, files, instruction,
            )
            result = aider.edit(instruction=instruction, files=files)
            logger.info(
                "coder iteration %d: success=%s, modified=%s, output_len=%s",
                iteration, result.success, result.modified_files,
                len(result.output) if result.output else 0,
            )

            if result.success and (result.modified_files or result.dry_run):
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

            # Aider reported success but modified nothing — treat as failure
            if result.success and not result.modified_files:
                last_error = (
                    "Aider exited successfully but no files were modified. "
                    "The instruction may not match any existing files."
                )
                if iteration < self.MAX_ITERATIONS:
                    messages.append(AIMessage(content=raw))
                    messages.append(HumanMessage(
                        content=(
                            f"{last_error}\n\n"
                            "Please revise your instruction and file list. "
                            "Ensure the file paths exist in the repo. "
                            "Return ONLY a JSON object with "
                            "'instruction' and 'files' keys."
                        ),
                    ))
                    continue
            else:
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
                "aider_output": self._truncate(
                    result.output if result else None, 4000
                ),
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
        parts: list[str] = []
        char_budget = self.MAX_CONTENT_CHARS

        # Static context from snapshot
        if self.context:
            work_items = self.context.get("work_items", [])
            if work_items:
                items = work_items[: self.MAX_CONTENT_ITEMS]
                parts.append("--- RELATED WORK ITEMS ---")
                for item in items:
                    name = item.get("name", "Untitled")
                    status = item.get("status", "unknown")
                    desc = item.get("description", "")
                    parts.append(f"- {name} ({status}): {desc}")

            page_content = self.context.get("page_content", {})
            if page_content:
                parts.append("\n--- REFERENCE CONTENT ---")
                for notion_id, content in list(page_content.items())[
                    : self.MAX_CONTENT_ITEMS
                ]:
                    if len(content) > char_budget:
                        content = content[:char_budget] + "\n... (truncated)"
                    parts.append(f"### {notion_id}")
                    parts.append(content)
                    parts.append("")

        # RAG semantic search for the current task
        task_desc = self._current_task.get("description", "") if hasattr(self, "_current_task") else ""
        task_id = self._current_task.get("notion_id", "") if hasattr(self, "_current_task") else ""

        if self.rag and task_desc:
            rag_results = self.retrieve(
                task_desc, entity_types=["doc", "work_item"], top_k=3
            )
            if rag_results:
                rag_section = self.rag.format_results(rag_results, max_chars=2000)
                if rag_section:
                    parts.append("\n--- RETRIEVED CONTEXT (semantically similar) ---")
                    parts.append(rag_section)

        # Snapshot relational lookups for linked docs and decisions
        if self.snapshot and task_id:
            linked_doc_ids = self.snapshot.get_related_ids(task_id, "doc_ids")
            linked_decision_ids = self.snapshot.get_related_ids(task_id, "decision_ids")

            # Composed query: RAG filtered to linked docs
            if linked_doc_ids and self.rag and task_desc:
                linked_results = self.retrieve(
                    task_desc, notion_ids=linked_doc_ids, top_k=3
                )
                if linked_results:
                    linked_section = self.rag.format_results(
                        linked_results, max_chars=1500
                    )
                    if linked_section:
                        parts.append("\n--- LINKED DOCUMENTS ---")
                        parts.append(linked_section)

            # Format linked decisions directly from snapshot
            if linked_decision_ids:
                decisions = [
                    self.snapshot.get_entity(did)
                    for did in linked_decision_ids
                ]
                decisions = [d for d in decisions if d is not None]
                if decisions:
                    parts.append("\n--- LINKED DECISIONS (ADRs) ---")
                    for d in decisions:
                        adr_id = d.get("adr_id", "")
                        title = d.get("title", "Untitled")
                        status_d = d.get("status", "?")
                        prefix = f"[{adr_id}] " if adr_id else ""
                        parts.append(f"- {prefix}{title} ({status_d})")

        return "\n".join(parts)

    @staticmethod
    def _truncate(text: str | None, max_len: int) -> str | None:
        """Truncate text to max_len, appending suffix if shortened."""
        if text is None:
            return None
        if len(text) <= max_len:
            return text
        return text[: max_len] + "... (truncated)"
