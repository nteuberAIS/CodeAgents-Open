"""TesterAgent — runs tests on the task branch and reports structured results.

Test failures are DATA (reported in partial_output), not agent errors.
Only subprocess crashes or timeouts are agent failures (success=False).
"""

from __future__ import annotations

import json
import re
import subprocess

from agents.base import BaseAgent
from config.settings import get_settings


class TesterAgent(BaseAgent):
    """Runs pytest on the target repo and reports structured results.

    This agent is purely subprocess-based — the LLM is injected per
    BaseAgent contract but never invoked.
    """

    name = "tester"
    MAX_ITERATIONS = 1  # Single test run, failures are data not errors
    REQUIRED_TOOLS: list[str] = []
    OPTIONAL_TOOLS: list[str] = []

    def run(self, user_input: str) -> dict:
        """Run tests and return structured results.

        Args:
            user_input: JSON string with task info and optional test config.
                Required: task_id
                Optional: task_title, repo_dir, test_command

        Returns:
            Standard agent result envelope via wrap_result().
        """
        # Parse input
        try:
            params = json.loads(user_input)
        except (json.JSONDecodeError, TypeError):
            return self.wrap_result(
                success=False,
                partial_output={"test_output": None},
                error_type="logic",
                error_message=f"Invalid JSON input: {user_input[:200]}",
            )

        task_id = params.get("task_id", "unknown")
        settings = get_settings()

        # Resolve test configuration with fallbacks
        repo_dir = (
            params.get("repo_dir")
            or settings.test_repo_dir
            or settings.aider_repo_dir
        )
        test_command = params.get("test_command") or settings.test_command
        timeout = settings.test_timeout

        if not repo_dir:
            return self.wrap_result(
                success=False,
                partial_output={"task_id": task_id, "test_output": None},
                error_type="logic",
                error_message="No repo_dir configured (set test_repo_dir or aider_repo_dir)",
            )

        # Run tests via subprocess
        try:
            result = subprocess.run(
                test_command.split(),
                cwd=repo_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            raw_output = result.stdout + result.stderr
        except subprocess.TimeoutExpired as e:
            return self.wrap_result(
                success=False,
                partial_output={
                    "task_id": task_id,
                    "test_output": str(e),
                },
                error_type="timeout",
                error_message=f"Test command timed out after {timeout}s",
            )
        except OSError as e:
            return self.wrap_result(
                success=False,
                partial_output={
                    "task_id": task_id,
                    "test_output": None,
                },
                error_type="infra",
                error_message=f"Failed to execute test command: {e}",
            )

        # Parse pytest summary
        passed, failed, errors = self._parse_pytest_summary(raw_output)
        test_passed = failed == 0 and errors == 0

        # Test failures are DATA, not agent errors — always success=True
        return self.wrap_result(
            success=True,
            partial_output={
                "test_passed": test_passed,
                "passed_count": passed,
                "failed_count": failed,
                "error_count": errors,
                "test_output": self._truncate(raw_output, 3000),
                "task_id": task_id,
            },
        )

    @staticmethod
    def _parse_pytest_summary(output: str) -> tuple[int, int, int]:
        """Parse pytest summary line for passed/failed/error counts.

        Handles formats like:
            === 5 passed in 0.12s ===
            === 3 passed, 2 failed in 0.34s ===
            === 1 passed, 1 failed, 1 error in 0.56s ===

        Returns:
            Tuple of (passed, failed, errors). All zero if parsing fails.
        """
        passed = failed = errors = 0
        # Match individual count + label pairs in the summary line
        for match in re.finditer(r"(\d+)\s+(passed|failed|error)", output):
            count = int(match.group(1))
            label = match.group(2)
            if label == "passed":
                passed = count
            elif label == "failed":
                failed = count
            elif label == "error":
                errors = count
        return passed, failed, errors

    @staticmethod
    def _truncate(text: str | None, max_len: int) -> str | None:
        """Truncate text with indicator if too long."""
        if text is None or len(text) <= max_len:
            return text
        return text[:max_len] + "... (truncated)"
