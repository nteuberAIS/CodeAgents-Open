"""UpdaterAgent — creates PRs and updates work item status.

The final agent in the cascade: Planner -> Coder -> Tester -> Updater.
Gracefully degrades if git or Notion tools are unavailable.
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agents.base import BaseAgent

logger = logging.getLogger(__name__)


class UpdaterAgent(BaseAgent):
    """Creates pull requests and updates Notion work item status.

    Uses the LLM to generate PR title and description from task info.
    Each tool operation is independent — the agent succeeds if it did
    what it could with available tools.
    """

    name = "updater"
    MAX_ITERATIONS = 2  # Retry once on transient failure (per failure-modes.md)

    REQUIRED_TOOLS: list[str] = []
    OPTIONAL_TOOLS: list[str] = ["notion_write", "azdevops", "github"]

    def run(self, user_input: str) -> dict:
        """Create PR and update work item status.

        Args:
            user_input: JSON string with task and branch info.
                Required: task_id, task_title, source_branch, target_branch
                Optional: notion_id, test_summary, task_description,
                          modified_files

        Returns:
            Standard agent result envelope via wrap_result().
        """
        # Parse input
        try:
            params = json.loads(user_input)
        except (json.JSONDecodeError, TypeError):
            return self.wrap_result(
                success=False,
                partial_output={},
                error_type="logic",
                error_message=f"Invalid JSON input: {user_input[:200]}",
            )

        task_id = params.get("task_id", "unknown")
        task_title = params.get("task_title", "")
        source_branch = params.get("source_branch", "")
        target_branch = params.get("target_branch", "")
        notion_id = params.get("notion_id")
        test_summary = params.get("test_summary", "")
        task_description = params.get("task_description", "")
        modified_files = params.get("modified_files", [])

        # Generate PR details via LLM
        pr_title, pr_description = self._generate_pr_details(
            task_id=task_id,
            task_title=task_title,
            task_description=task_description,
            modified_files=modified_files,
            test_summary=test_summary,
        )

        # Attempt PR creation
        pr_created = False
        pr_url = None
        pr_error = None

        git_tool = self.get_tool("azdevops") or self.get_tool("github")
        is_dry_run = getattr(git_tool, "dry_run", False) if git_tool else False
        if git_tool and source_branch and target_branch:
            pr_created, pr_url, pr_error = self._create_pr_with_retry(
                git_tool=git_tool,
                title=pr_title,
                source_branch=source_branch,
                target_branch=target_branch,
                description=pr_description,
            )

        # Attempt Notion status update
        notion_updated = False
        notion_status = ""

        notion_write = self.get_tool("notion_write")
        if notion_write and notion_id:
            new_status = "In Review" if pr_created else "In Progress"
            notion_updated, notion_status = self._update_notion_with_retry(
                notion_write=notion_write,
                notion_id=notion_id,
                status=new_status,
            )

        # Determine overall success
        # Agent succeeds if it did what it could — both tools missing is still OK
        all_ops_failed = False
        if git_tool and source_branch and target_branch and not pr_created:
            if notion_write and notion_id and not notion_updated:
                all_ops_failed = True

        if all_ops_failed:
            return self.wrap_result(
                success=False,
                partial_output={
                    "pr_created": pr_created,
                    "pr_url": pr_url,
                    "pr_title": pr_title,
                    "notion_updated": notion_updated,
                    "notion_status": notion_status,
                    "task_id": task_id,
                    "dry_run": is_dry_run,
                },
                error_type="tool",
                error_message=pr_error or "All operations failed",
            )

        return self.wrap_result(
            success=True,
            partial_output={
                "pr_created": pr_created,
                "pr_url": pr_url,
                "pr_title": pr_title,
                "notion_updated": notion_updated,
                "notion_status": notion_status,
                "task_id": task_id,
                "dry_run": is_dry_run,
            },
        )

    def _generate_pr_details(
        self,
        task_id: str,
        task_title: str,
        task_description: str = "",
        modified_files: list[str] | None = None,
        test_summary: str = "",
    ) -> tuple[str, str]:
        """Use LLM to generate PR title and description.

        Returns:
            Tuple of (pr_title, pr_description). Falls back to
            task_title if LLM fails.
        """
        try:
            system_content = self.load_prompt(
                "updater/system.j2",
                task_id=task_id,
                task_title=task_title,
                task_description=task_description,
                modified_files=modified_files or [],
                test_summary=test_summary,
            )

            messages = [
                SystemMessage(content=system_content),
                HumanMessage(content=f"Generate PR details for task {task_id}: {task_title}"),
            ]

            response = self.llm.invoke(messages)
            parsed = self._parse_response(response.content)

            if "parse_error" not in parsed:
                return (
                    parsed.get("pr_title", task_title)[:80],
                    parsed.get("pr_description", ""),
                )
        except Exception as e:
            logger.warning("LLM PR generation failed: %s", e)

        # Fallback: use task title directly
        return task_title[:80], f"Task {task_id}: {task_title}"

    def _create_pr_with_retry(
        self,
        git_tool,
        title: str,
        source_branch: str,
        target_branch: str,
        description: str,
    ) -> tuple[bool, str | None, str | None]:
        """Create a PR with one retry on failure.

        Returns:
            Tuple of (success, pr_url, error_message).
        """
        for attempt in range(self.MAX_ITERATIONS):
            try:
                result = git_tool.create_pull_request(
                    title=title,
                    source_branch=source_branch,
                    target_branch=target_branch,
                    description=description,
                )
                if result.success or result.dry_run:
                    # Parse PR URL from output if available
                    pr_url = self._extract_pr_url(result.output)
                    return True, pr_url, None
                error = result.error or result.output or "Unknown error"
                logger.warning(
                    "PR creation attempt %d failed: %s", attempt + 1, error,
                )
            except Exception as e:
                error = str(e)
                logger.warning(
                    "PR creation attempt %d exception: %s", attempt + 1, e,
                )

        return False, None, error

    def _update_notion_with_retry(
        self,
        notion_write,
        notion_id: str,
        status: str,
    ) -> tuple[bool, str]:
        """Update Notion work item status with one retry on failure.

        Returns:
            Tuple of (success, status_set).
        """
        for attempt in range(self.MAX_ITERATIONS):
            try:
                notion_write.update_work_item(notion_id, status=status)
                return True, status
            except Exception as e:
                logger.warning(
                    "Notion update attempt %d failed: %s", attempt + 1, e,
                )

        return False, ""

    def _parse_response(self, raw: str) -> dict:
        """Parse LLM output as JSON, stripping markdown fences."""
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"parse_error": "LLM did not return valid JSON"}

    @staticmethod
    def _extract_pr_url(output: str | None) -> str | None:
        """Best-effort extraction of PR URL from git tool output."""
        if not output:
            return None
        # Match common URL patterns
        match = re.search(r"https?://\S+", output)
        return match.group(0) if match else None
