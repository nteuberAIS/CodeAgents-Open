"""Pydantic models for git entities.

Used by BaseGitTool and its subclasses (GitHubTool, AzDevOpsTool)
to represent branches, pull requests, commits, and command results.
"""

from __future__ import annotations

from pydantic import BaseModel


class Branch(BaseModel):
    """A git branch."""

    name: str
    ref: str | None = None       # SHA or ref
    is_current: bool = False
    remote: str | None = None    # e.g. "origin"


class PullRequest(BaseModel):
    """A pull request on a git hosting provider."""

    id: str                      # PR number or ID
    title: str
    source_branch: str
    target_branch: str
    status: str | None = None    # open, closed, merged, draft
    url: str | None = None       # Web URL to the PR


class CommitInfo(BaseModel):
    """A git commit."""

    sha: str
    message: str
    branch: str | None = None


class GitCommandResult(BaseModel):
    """Wrapper for CLI command results.

    Useful for dry-run mode and logging.
    """

    command: str                 # The CLI command that was/would be run
    success: bool
    output: str | None = None
    error: str | None = None
    dry_run: bool = False        # True if command was not actually executed
