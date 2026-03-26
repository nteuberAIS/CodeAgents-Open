"""Base git tool abstraction.

Provides an ABC for git provider tools (GitHub, Azure DevOps) with:
- Shared CLI execution via _run_command / _run_write_command
- Dry-run support for all write operations
- Human-approval gate for merges to main/master
- Branch naming helpers following project convention
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from typing import Any

from schemas.git_models import Branch, GitCommandResult, PullRequest


class GitToolError(Exception):
    """Raised when a git operation fails."""


class BaseGitTool(ABC):
    """Abstract base for git provider tools.

    Subclasses implement provider-specific CLI calls for PR operations.
    Local git operations (branch, commit, push) are concrete since they
    are identical across providers.

    All write operations support dry_run mode.
    """

    provider: str = "base"

    def __init__(self, settings: Any, dry_run: bool = False) -> None:
        self.settings = settings
        self.dry_run = dry_run
        self._validate_cli()

    # ------------------------------------------------------------------ #
    #  Abstract methods — subclasses must implement                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def _validate_cli(self) -> None:
        """Validate that the provider CLI is installed and authenticated.

        Raise GitToolError if validation fails.
        Should be skipped (no-op) when dry_run is True.
        """

    @abstractmethod
    def list_branches(self, pattern: str | None = None) -> list[Branch]:
        """List branches, optionally filtered by pattern."""

    @abstractmethod
    def list_pull_requests(self, status: str = "open") -> list[PullRequest]:
        """List pull requests with the given status."""

    @abstractmethod
    def get_pull_request(self, pr_id: str) -> PullRequest | None:
        """Get a single pull request by ID."""

    @abstractmethod
    def create_pull_request(
        self,
        title: str,
        source_branch: str,
        target_branch: str,
        description: str = "",
    ) -> GitCommandResult:
        """Create a pull request. Uses _run_write_command internally."""

    @abstractmethod
    def merge_pull_request(self, pr_id: str) -> GitCommandResult:
        """Merge a pull request.

        Subclasses must call requires_approval() before executing and
        raise GitToolError if approval is required.
        """

    # ------------------------------------------------------------------ #
    #  Concrete git operations (provider-agnostic, use local git CLI)     #
    # ------------------------------------------------------------------ #

    def get_current_branch(self) -> str:
        """Return the name of the currently checked-out branch."""
        result = self._run_command(["git", "branch", "--show-current"])
        if not result.success:
            raise GitToolError(
                f"Failed to get current branch: {result.error}"
            )
        return (result.output or "").strip()

    def create_branch(
        self, name: str, from_ref: str = "main"
    ) -> GitCommandResult:
        """Create and checkout a new branch from the given ref."""
        return self._run_write_command(
            ["git", "checkout", "-b", name, from_ref]
        )

    def checkout_branch(self, name: str) -> GitCommandResult:
        """Checkout an existing branch."""
        return self._run_write_command(["git", "checkout", name])

    def commit(
        self, message: str, files: list[str] | None = None
    ) -> GitCommandResult:
        """Stage files and create a commit.

        If files is None, stages all changes (git add -A).
        """
        if files:
            add_result = self._run_write_command(["git", "add"] + files)
        else:
            add_result = self._run_write_command(["git", "add", "-A"])

        if not add_result.success and not add_result.dry_run:
            return add_result

        return self._run_write_command(["git", "commit", "-m", message])

    def push(self, branch: str | None = None) -> GitCommandResult:
        """Push the current branch to origin."""
        cmd = ["git", "push", "-u", "origin"]
        if branch:
            cmd.append(branch)
        return self._run_write_command(cmd)

    # ------------------------------------------------------------------ #
    #  Shared helpers                                                     #
    # ------------------------------------------------------------------ #

    def _run_command(self, cmd: list[str]) -> GitCommandResult:
        """Execute a CLI command and return the result.

        Always executes — used for read operations.
        """
        command_str = " ".join(cmd)
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
            )
            return GitCommandResult(
                command=command_str,
                success=result.returncode == 0,
                output=result.stdout.strip() if result.stdout else None,
                error=(
                    result.stderr.strip()
                    if result.stderr and result.returncode != 0
                    else None
                ),
            )
        except subprocess.TimeoutExpired:
            return GitCommandResult(
                command=command_str,
                success=False,
                error="Command timed out after 30 seconds",
            )

    def _run_write_command(self, cmd: list[str]) -> GitCommandResult:
        """Execute a CLI command for write operations.

        If dry_run is True, returns a result without executing.
        Otherwise delegates to _run_command.
        """
        command_str = " ".join(cmd)

        if self.dry_run:
            return GitCommandResult(
                command=command_str,
                success=True,
                output=None,
                dry_run=True,
            )

        return self._run_command(cmd)

    def requires_approval(self, operation: str, target_branch: str) -> bool:
        """Check if an operation requires human approval.

        Only merges targeting main/master require approval.
        """
        if operation == "merge" and target_branch in ("main", "master"):
            return True
        return False

    # ------------------------------------------------------------------ #
    #  Branching strategy helpers                                         #
    # ------------------------------------------------------------------ #

    def task_branch_name(self, sprint_number: int, task_id: str) -> str:
        """Generate branch name per project convention.

        Convention: sprint-{N}/{task-id}
        Example: sprint-8/SP8-001
        """
        return f"sprint-{sprint_number}/{task_id}"

    def sprint_branch_name(self, sprint_number: int) -> str:
        """Generate sprint branch name.

        Convention: sprint-{N}
        Example: sprint-8
        """
        return f"sprint-{sprint_number}"
