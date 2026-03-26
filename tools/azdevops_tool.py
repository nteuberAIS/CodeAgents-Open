"""Azure DevOps git tool implementation using the az repos CLI."""

from __future__ import annotations

import json
from typing import Any

from schemas.git_models import Branch, GitCommandResult, PullRequest
from tools.git_tool import BaseGitTool, GitToolError


class AzDevOpsTool(BaseGitTool):
    """Git tool backed by the Azure DevOps CLI (az repos).

    Uses local git commands for branch/commit operations and
    az repos CLI for pull request operations.
    """

    provider: str = "azdevops"

    def __init__(self, settings: Any, dry_run: bool = False) -> None:
        self.org_url = getattr(settings, "azure_devops_org_url", None)
        self.project = getattr(settings, "azure_devops_project", None)
        self.repo = getattr(settings, "azure_devops_repo", None)
        if not all([self.org_url, self.project, self.repo]):
            raise GitToolError(
                "Azure DevOps config incomplete. Set AZURE_DEVOPS_ORG_URL, "
                "AZURE_DEVOPS_PROJECT, and AZURE_DEVOPS_REPO in .env"
            )
        super().__init__(settings, dry_run)

    def _validate_cli(self) -> None:
        """Check that az CLI is installed and authenticated."""
        if self.dry_run:
            return
        result = self._run_command(["az", "account", "show"])
        if not result.success:
            raise GitToolError(
                "az CLI not authenticated. Run 'az login' first."
            )

    # ------------------------------------------------------------------ #
    #  Read operations                                                    #
    # ------------------------------------------------------------------ #

    def list_branches(self, pattern: str | None = None) -> list[Branch]:
        """List branches using az repos ref list."""
        cmd = [
            "az", "repos", "ref", "list",
            "--repository", self.repo,
            "--filter", "heads/",
            "--org", self.org_url,
            "--project", self.project,
            "-o", "json",
        ]
        result = self._run_command(cmd)
        if not result.success:
            raise GitToolError(f"Failed to list branches: {result.error}")

        branches: list[Branch] = []
        for item in json.loads(result.output or "[]"):
            # Azure DevOps returns refs like "refs/heads/main"
            full_name = item.get("name", "")
            name = full_name.removeprefix("refs/heads/")

            if pattern and pattern not in name:
                continue

            branches.append(Branch(
                name=name,
                ref=item.get("objectId"),
                remote="origin",
            ))
        return branches

    def list_pull_requests(self, status: str = "open") -> list[PullRequest]:
        """List pull requests using az repos pr list."""
        # Azure DevOps uses "active" instead of "open"
        az_status = "active" if status == "open" else status

        cmd = [
            "az", "repos", "pr", "list",
            "--repository", self.repo,
            "--status", az_status,
            "--org", self.org_url,
            "--project", self.project,
            "-o", "json",
        ]
        result = self._run_command(cmd)
        if not result.success:
            raise GitToolError(
                f"Failed to list pull requests: {result.error}"
            )

        prs: list[PullRequest] = []
        for item in json.loads(result.output or "[]"):
            prs.append(self._parse_az_pr(item))
        return prs

    def get_pull_request(self, pr_id: str) -> PullRequest | None:
        """Get a single pull request by ID."""
        cmd = [
            "az", "repos", "pr", "show",
            "--id", pr_id,
            "--org", self.org_url,
            "-o", "json",
        ]
        result = self._run_command(cmd)
        if not result.success:
            return None

        item = json.loads(result.output or "{}")
        if not item:
            return None

        return self._parse_az_pr(item)

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
        """Create a pull request using az repos pr create."""
        cmd = [
            "az", "repos", "pr", "create",
            "--title", title,
            "--description", description,
            "--source-branch", source_branch,
            "--target-branch", target_branch,
            "--repository", self.repo,
            "--org", self.org_url,
            "--project", self.project,
            "-o", "json",
        ]
        return self._run_write_command(cmd)

    def merge_pull_request(self, pr_id: str) -> GitCommandResult:
        """Merge a pull request by completing it.

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
            "az", "repos", "pr", "update",
            "--id", pr_id,
            "--status", "completed",
            "--org", self.org_url,
            "-o", "json",
        ])

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_az_pr(item: dict) -> PullRequest:
        """Parse an Azure DevOps PR JSON object into a PullRequest model."""
        # Azure DevOps branch refs include "refs/heads/" prefix
        source = item.get("sourceRefName", "")
        target = item.get("targetRefName", "")

        return PullRequest(
            id=str(item.get("pullRequestId", "")),
            title=item.get("title", ""),
            source_branch=source.removeprefix("refs/heads/"),
            target_branch=target.removeprefix("refs/heads/"),
            status=item.get("status"),
            url=item.get("url"),
        )
