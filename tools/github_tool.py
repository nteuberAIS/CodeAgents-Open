"""GitHub git tool implementation using the gh CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from schemas.git_models import Branch, GitCommandResult, PullRequest
from tools.git_tool import BaseGitTool, GitToolError


class GitHubTool(BaseGitTool):
    """Git tool backed by the GitHub CLI (gh).

    Uses local git commands for branch/commit operations and
    gh CLI for pull request operations.
    """

    provider: str = "github"

    def __init__(self, settings: Any, dry_run: bool = False) -> None:
        self.repo_dir = Path(getattr(settings, "github_repo_dir", None) or ".")
        super().__init__(settings, dry_run)

    def _validate_cli(self) -> None:
        """Check that gh CLI is installed and authenticated."""
        if self.dry_run:
            return
        result = self._run_command(["gh", "auth", "status"])
        if not result.success:
            raise GitToolError(
                "gh CLI not authenticated. Run 'gh auth login' first."
            )

    # ------------------------------------------------------------------ #
    #  Read operations                                                    #
    # ------------------------------------------------------------------ #

    def list_branches(self, pattern: str | None = None) -> list[Branch]:
        """List branches using local git."""
        result = self._run_command(["git", "branch", "-a"])
        if not result.success:
            raise GitToolError(f"Failed to list branches: {result.error}")

        branches: list[Branch] = []
        for line in (result.output or "").splitlines():
            line = line.strip()
            if not line:
                continue

            is_current = line.startswith("* ")
            name = line.lstrip("* ").strip()

            # Parse remote prefix
            remote = None
            if name.startswith("remotes/"):
                parts = name.split("/", 2)  # remotes/origin/branch-name
                if len(parts) >= 3:
                    remote = parts[1]
                    name = parts[2]
                else:
                    continue
                # Skip HEAD pointer
                if " -> " in name:
                    continue

            if pattern and pattern not in name:
                continue

            branches.append(
                Branch(name=name, is_current=is_current, remote=remote)
            )
        return branches

    def list_pull_requests(self, status: str = "open") -> list[PullRequest]:
        """List pull requests using gh CLI."""
        result = self._run_command([
            "gh", "pr", "list",
            "--state", status,
            "--json", "number,title,headRefName,baseRefName,state,url",
        ])
        if not result.success:
            raise GitToolError(
                f"Failed to list pull requests: {result.error}"
            )

        prs: list[PullRequest] = []
        for item in json.loads(result.output or "[]"):
            prs.append(PullRequest(
                id=str(item["number"]),
                title=item["title"],
                source_branch=item["headRefName"],
                target_branch=item["baseRefName"],
                status=item.get("state"),
                url=item.get("url"),
            ))
        return prs

    def get_pull_request(self, pr_id: str) -> PullRequest | None:
        """Get a single pull request by number."""
        result = self._run_command([
            "gh", "pr", "view", pr_id,
            "--json", "number,title,headRefName,baseRefName,state,url",
        ])
        if not result.success:
            return None

        item = json.loads(result.output or "{}")
        if not item:
            return None

        return PullRequest(
            id=str(item["number"]),
            title=item["title"],
            source_branch=item["headRefName"],
            target_branch=item["baseRefName"],
            status=item.get("state"),
            url=item.get("url"),
        )

    # ------------------------------------------------------------------ #
    #  Write operations                                                   #
    # ------------------------------------------------------------------ #

    def create_pull_request(
        self,
        title: str,
        source_branch: str,
        target_branch: str,
        description: str = "",
    ) -> GitCommandResult:
        """Create a pull request using gh CLI."""
        cmd = [
            "gh", "pr", "create",
            "--title", title,
            "--body", description,
            "--head", source_branch,
            "--base", target_branch,
        ]
        return self._run_write_command(cmd)

    def merge_pull_request(self, pr_id: str) -> GitCommandResult:
        """Merge a pull request.

        Raises GitToolError if the PR targets main/master (requires approval).
        """
        # Fetch PR to check target branch
        pr = self.get_pull_request(pr_id)
        if pr and self.requires_approval("merge", pr.target_branch):
            raise GitToolError(
                f"Merge to '{pr.target_branch}' requires human approval. "
                f"PR #{pr_id} targets a protected branch."
            )

        return self._run_write_command([
            "gh", "pr", "merge", pr_id, "--merge",
        ])
