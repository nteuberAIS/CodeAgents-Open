"""Tests for the git tool suite.

All tests mock subprocess — no real git/gh/az commands are executed.
"""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import pytest

from schemas.git_models import Branch, CommitInfo, GitCommandResult, PullRequest
from tools.git_tool import BaseGitTool, GitToolError
from tools.github_tool import GitHubTool
from tools.azdevops_tool import AzDevOpsTool


# ------------------------------------------------------------------ #
#  Test helpers                                                       #
# ------------------------------------------------------------------ #


def _make_settings(**overrides):
    """Create a minimal Settings-like object for testing."""
    defaults = {
        "github_repo_dir": ".",
        "azure_devops_org_url": "https://dev.azure.com/test-org",
        "azure_devops_project": "test-project",
        "azure_devops_repo": "test-repo",
    }
    defaults.update(overrides)
    settings = MagicMock()
    for k, v in defaults.items():
        setattr(settings, k, v)
    return settings


def _mock_subprocess_result(
    stdout: str = "", stderr: str = "", returncode: int = 0
):
    """Create a mock subprocess.CompletedProcess."""
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


# ------------------------------------------------------------------ #
#  Git Models                                                         #
# ------------------------------------------------------------------ #


class TestGitModels:
    def test_branch_defaults(self):
        b = Branch(name="main")
        assert b.name == "main"
        assert b.ref is None
        assert b.is_current is False
        assert b.remote is None

    def test_branch_all_fields(self):
        b = Branch(name="feature", ref="abc123", is_current=True, remote="origin")
        assert b.name == "feature"
        assert b.ref == "abc123"
        assert b.is_current is True
        assert b.remote == "origin"

    def test_pull_request_defaults(self):
        pr = PullRequest(
            id="42", title="Fix bug", source_branch="fix", target_branch="main"
        )
        assert pr.id == "42"
        assert pr.status is None
        assert pr.url is None

    def test_pull_request_all_fields(self):
        pr = PullRequest(
            id="42",
            title="Fix bug",
            source_branch="fix",
            target_branch="main",
            status="open",
            url="https://github.com/org/repo/pull/42",
        )
        assert pr.status == "open"
        assert pr.url == "https://github.com/org/repo/pull/42"

    def test_commit_info(self):
        c = CommitInfo(sha="abc123", message="Initial commit", branch="main")
        assert c.sha == "abc123"
        assert c.message == "Initial commit"
        assert c.branch == "main"

    def test_commit_info_defaults(self):
        c = CommitInfo(sha="abc123", message="msg")
        assert c.branch is None

    def test_git_command_result_defaults(self):
        r = GitCommandResult(command="git status", success=True)
        assert r.output is None
        assert r.error is None
        assert r.dry_run is False

    def test_git_command_result_round_trip(self):
        r = GitCommandResult(
            command="git push",
            success=True,
            output="Everything up-to-date",
            dry_run=True,
        )
        data = r.model_dump()
        r2 = GitCommandResult(**data)
        assert r2 == r


# ------------------------------------------------------------------ #
#  BaseGitTool                                                        #
# ------------------------------------------------------------------ #


class TestBaseGitTool:
    """Test concrete methods on BaseGitTool via GitHubTool (dry_run)."""

    def test_task_branch_name(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.task_branch_name(8, "SP8-001") == "task/sprint-8/SP8-001"

    def test_sprint_branch_name(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.sprint_branch_name(8) == "sprint-8"

    def test_requires_approval_merge_to_main(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.requires_approval("merge", "main") is True

    def test_requires_approval_merge_to_master(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.requires_approval("merge", "master") is True

    def test_requires_approval_merge_to_feature(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.requires_approval("merge", "sprint-8") is False

    def test_requires_approval_non_merge(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.requires_approval("push", "main") is False

    @patch("tools.git_tool.subprocess.run")
    def test_run_command_success(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="output text", returncode=0
        )
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool._run_command(["git", "status"])
        assert result.success is True
        assert result.output == "output text"
        assert result.error is None
        assert result.dry_run is False
        mock_run.assert_called_once()

    @patch("tools.git_tool.subprocess.run")
    def test_run_command_failure(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="fatal: not a repo", returncode=128
        )
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool._run_command(["git", "status"])
        assert result.success is False
        assert result.error == "fatal: not a repo"

    @patch("tools.git_tool.subprocess.run")
    def test_run_command_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="git", timeout=30)
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool._run_command(["git", "fetch"])
        assert result.success is False
        assert "timed out" in result.error

    def test_run_write_command_dry_run(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool._run_write_command(["git", "push"])
        assert result.success is True
        assert result.dry_run is True
        assert result.command == "git push"
        assert result.output is None

    @patch("tools.git_tool.subprocess.run")
    def test_run_write_command_live(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="pushed", returncode=0
        )
        # Need to also mock the init validation
        with patch.object(GitHubTool, "_validate_cli"):
            tool = GitHubTool(_make_settings(), dry_run=False)
        result = tool._run_write_command(["git", "push"])
        assert result.success is True
        assert result.dry_run is False

    # -- Concrete git operations --

    def test_create_branch_dry_run(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.create_branch("task/sprint-8/SP8-001", "main")
        assert result.dry_run is True
        assert "checkout -b task/sprint-8/SP8-001 main" in result.command

    def test_checkout_branch_dry_run(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.checkout_branch("sprint-8")
        assert result.dry_run is True
        assert "checkout sprint-8" in result.command

    def test_commit_dry_run_with_files(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.commit("fix: update readme", files=["README.md"])
        assert result.dry_run is True
        assert "commit" in result.command

    def test_commit_dry_run_all_files(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.commit("chore: cleanup")
        assert result.dry_run is True

    def test_push_dry_run(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.push("task/sprint-8/SP8-001")
        assert result.dry_run is True
        assert "push -u origin task/sprint-8/SP8-001" in result.command

    def test_push_dry_run_no_branch(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.push()
        assert result.dry_run is True
        assert result.command == "git push -u origin"

    @patch("tools.git_tool.subprocess.run")
    def test_get_current_branch(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(stdout="main\n")
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.get_current_branch() == "main"


# ------------------------------------------------------------------ #
#  GitHubTool                                                         #
# ------------------------------------------------------------------ #


class TestGitHubToolInit:
    @patch("tools.git_tool.subprocess.run")
    def test_validates_gh_auth(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(returncode=0)
        tool = GitHubTool(_make_settings(), dry_run=False)
        assert tool.provider == "github"
        # gh auth status was called
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["gh", "auth", "status"] in calls

    @patch("tools.git_tool.subprocess.run")
    def test_raises_when_not_authenticated(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="not logged in", returncode=1
        )
        with pytest.raises(GitToolError, match="gh CLI not authenticated"):
            GitHubTool(_make_settings(), dry_run=False)

    def test_skips_auth_check_in_dry_run(self):
        # Should not raise even without gh installed
        tool = GitHubTool(_make_settings(), dry_run=True)
        assert tool.provider == "github"


class TestGitHubToolOperations:
    @patch("tools.git_tool.subprocess.run")
    def test_list_branches(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout=(
                "* main\n"
                "  feature/foo\n"
                "  remotes/origin/main\n"
                "  remotes/origin/feature/foo\n"
                "  remotes/origin/HEAD -> origin/main\n"
            )
        )
        tool = GitHubTool(_make_settings(), dry_run=True)
        branches = tool.list_branches()
        names = [b.name for b in branches]
        assert "main" in names
        assert "feature/foo" in names
        # HEAD pointer should be filtered out
        assert not any("HEAD" in b.name for b in branches)

    @patch("tools.git_tool.subprocess.run")
    def test_list_branches_with_pattern(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="* main\n  task/sprint-8/SP8-001\n  task/sprint-8/SP8-002\n"
        )
        tool = GitHubTool(_make_settings(), dry_run=True)
        branches = tool.list_branches(pattern="sprint-8")
        assert len(branches) == 2
        assert all("sprint-8" in b.name for b in branches)

    @patch("tools.git_tool.subprocess.run")
    def test_list_pull_requests(self, mock_run):
        pr_json = json.dumps([
            {
                "number": 42,
                "title": "Add feature",
                "headRefName": "feature/foo",
                "baseRefName": "main",
                "state": "OPEN",
                "url": "https://github.com/org/repo/pull/42",
            },
            {
                "number": 43,
                "title": "Fix bug",
                "headRefName": "fix/bar",
                "baseRefName": "sprint-8",
                "state": "OPEN",
                "url": "https://github.com/org/repo/pull/43",
            },
        ])
        mock_run.return_value = _mock_subprocess_result(stdout=pr_json)
        tool = GitHubTool(_make_settings(), dry_run=True)
        prs = tool.list_pull_requests()
        assert len(prs) == 2
        assert prs[0].id == "42"
        assert prs[0].source_branch == "feature/foo"
        assert prs[1].target_branch == "sprint-8"

    @patch("tools.git_tool.subprocess.run")
    def test_get_pull_request(self, mock_run):
        pr_json = json.dumps({
            "number": 42,
            "title": "Add feature",
            "headRefName": "feature/foo",
            "baseRefName": "main",
            "state": "OPEN",
            "url": "https://github.com/org/repo/pull/42",
        })
        mock_run.return_value = _mock_subprocess_result(stdout=pr_json)
        tool = GitHubTool(_make_settings(), dry_run=True)
        pr = tool.get_pull_request("42")
        assert pr is not None
        assert pr.id == "42"
        assert pr.title == "Add feature"

    @patch("tools.git_tool.subprocess.run")
    def test_get_pull_request_not_found(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="not found", returncode=1
        )
        tool = GitHubTool(_make_settings(), dry_run=True)
        pr = tool.get_pull_request("999")
        assert pr is None

    def test_create_pull_request_dry_run(self):
        tool = GitHubTool(_make_settings(), dry_run=True)
        result = tool.create_pull_request(
            title="Add feature",
            source_branch="feature/foo",
            target_branch="sprint-8",
            description="A new feature",
        )
        assert result.dry_run is True
        assert "gh pr create" in result.command

    @patch("tools.git_tool.subprocess.run")
    def test_create_pull_request_live(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stdout="https://github.com/org/repo/pull/44"
        )
        with patch.object(GitHubTool, "_validate_cli"):
            tool = GitHubTool(_make_settings(), dry_run=False)
        result = tool.create_pull_request(
            title="Add feature",
            source_branch="feature/foo",
            target_branch="sprint-8",
        )
        assert result.success is True
        cmd = mock_run.call_args.args[0]
        assert "gh" in cmd
        assert "pr" in cmd
        assert "create" in cmd

    @patch("tools.git_tool.subprocess.run")
    def test_merge_pull_request_to_sprint(self, mock_run):
        """Merging to a sprint branch should succeed."""
        # First call: get_pull_request (gh pr view)
        # Second call: gh pr merge
        pr_json = json.dumps({
            "number": 42,
            "title": "Fix",
            "headRefName": "task/sprint-8/SP8-001",
            "baseRefName": "sprint-8",
            "state": "OPEN",
            "url": "https://github.com/org/repo/pull/42",
        })
        mock_run.side_effect = [
            _mock_subprocess_result(stdout=pr_json),  # get_pull_request
            _mock_subprocess_result(stdout="merged"),  # merge
        ]
        with patch.object(GitHubTool, "_validate_cli"):
            tool = GitHubTool(_make_settings(), dry_run=False)
        result = tool.merge_pull_request("42")
        assert result.success is True

    @patch("tools.git_tool.subprocess.run")
    def test_merge_pull_request_to_main_raises(self, mock_run):
        """Merging to main should raise GitToolError (requires approval)."""
        pr_json = json.dumps({
            "number": 42,
            "title": "Fix",
            "headRefName": "sprint-8",
            "baseRefName": "main",
            "state": "OPEN",
            "url": "https://github.com/org/repo/pull/42",
        })
        mock_run.return_value = _mock_subprocess_result(stdout=pr_json)
        with patch.object(GitHubTool, "_validate_cli"):
            tool = GitHubTool(_make_settings(), dry_run=False)
        with pytest.raises(GitToolError, match="requires human approval"):
            tool.merge_pull_request("42")


# ------------------------------------------------------------------ #
#  AzDevOpsTool                                                       #
# ------------------------------------------------------------------ #


class TestAzDevOpsToolInit:
    def test_raises_without_org(self):
        with pytest.raises(GitToolError, match="config incomplete"):
            AzDevOpsTool(
                _make_settings(azure_devops_org_url=None), dry_run=True
            )

    def test_raises_without_project(self):
        with pytest.raises(GitToolError, match="config incomplete"):
            AzDevOpsTool(
                _make_settings(azure_devops_project=None), dry_run=True
            )

    def test_raises_without_repo(self):
        with pytest.raises(GitToolError, match="config incomplete"):
            AzDevOpsTool(
                _make_settings(azure_devops_repo=None), dry_run=True
            )

    @patch("tools.git_tool.subprocess.run")
    @patch("tools.azdevops_tool.shutil.which", return_value="az")
    def test_validates_az_auth(self, mock_which, mock_run):
        mock_run.return_value = _mock_subprocess_result(returncode=0)
        tool = AzDevOpsTool(_make_settings(), dry_run=False)
        assert tool.provider == "azdevops"
        calls = [c.args[0] for c in mock_run.call_args_list]
        assert ["az", "account", "show"] in calls

    @patch("tools.git_tool.subprocess.run")
    def test_raises_when_az_not_authenticated(self, mock_run):
        mock_run.return_value = _mock_subprocess_result(
            stderr="not logged in", returncode=1
        )
        with pytest.raises(GitToolError, match="az CLI not authenticated"):
            AzDevOpsTool(_make_settings(), dry_run=False)

    def test_skips_auth_in_dry_run(self):
        tool = AzDevOpsTool(_make_settings(), dry_run=True)
        assert tool.provider == "azdevops"


class TestAzDevOpsToolOperations:
    @patch("tools.git_tool.subprocess.run")
    def test_list_branches(self, mock_run):
        az_json = json.dumps([
            {"name": "refs/heads/main", "objectId": "abc123"},
            {"name": "refs/heads/sprint-8", "objectId": "def456"},
        ])
        mock_run.return_value = _mock_subprocess_result(stdout=az_json)
        tool = AzDevOpsTool(_make_settings(), dry_run=True)
        branches = tool.list_branches()
        assert len(branches) == 2
        assert branches[0].name == "main"
        assert branches[0].ref == "abc123"
        assert branches[1].name == "sprint-8"

    @patch("tools.git_tool.subprocess.run")
    def test_list_pull_requests(self, mock_run):
        az_json = json.dumps([
            {
                "pullRequestId": 101,
                "title": "Add feature",
                "sourceRefName": "refs/heads/feature/foo",
                "targetRefName": "refs/heads/sprint-8",
                "status": "active",
                "url": "https://dev.azure.com/org/project/_git/repo/pullrequest/101",
            },
        ])
        mock_run.return_value = _mock_subprocess_result(stdout=az_json)
        tool = AzDevOpsTool(_make_settings(), dry_run=True)
        prs = tool.list_pull_requests()
        assert len(prs) == 1
        assert prs[0].id == "101"
        assert prs[0].source_branch == "feature/foo"
        assert prs[0].target_branch == "sprint-8"

    @patch("tools.git_tool.subprocess.run")
    def test_get_pull_request(self, mock_run):
        az_json = json.dumps({
            "pullRequestId": 101,
            "title": "Add feature",
            "sourceRefName": "refs/heads/feature/foo",
            "targetRefName": "refs/heads/sprint-8",
            "status": "active",
            "url": "https://dev.azure.com/org/project/_git/repo/pullrequest/101",
        })
        mock_run.return_value = _mock_subprocess_result(stdout=az_json)
        tool = AzDevOpsTool(_make_settings(), dry_run=True)
        pr = tool.get_pull_request("101")
        assert pr is not None
        assert pr.id == "101"

    def test_create_pull_request_dry_run(self):
        tool = AzDevOpsTool(_make_settings(), dry_run=True)
        result = tool.create_pull_request(
            title="Add feature",
            source_branch="feature/foo",
            target_branch="sprint-8",
        )
        assert result.dry_run is True
        assert "az repos pr create" in result.command

    @patch("tools.git_tool.subprocess.run")
    def test_merge_pull_request_to_main_raises(self, mock_run):
        """Merging to main should raise GitToolError."""
        az_json = json.dumps({
            "pullRequestId": 101,
            "title": "Sprint 8",
            "sourceRefName": "refs/heads/sprint-8",
            "targetRefName": "refs/heads/main",
            "status": "active",
        })
        mock_run.return_value = _mock_subprocess_result(stdout=az_json)
        with patch.object(AzDevOpsTool, "_validate_cli"):
            tool = AzDevOpsTool(_make_settings(), dry_run=False)
        with pytest.raises(GitToolError, match="requires human approval"):
            tool.merge_pull_request("101")

    @patch("tools.git_tool.subprocess.run")
    def test_merge_pull_request_to_sprint(self, mock_run):
        """Merging to sprint branch should succeed."""
        az_json = json.dumps({
            "pullRequestId": 101,
            "title": "Task",
            "sourceRefName": "refs/heads/task/sprint-8/SP8-001",
            "targetRefName": "refs/heads/sprint-8",
            "status": "active",
        })
        mock_run.side_effect = [
            _mock_subprocess_result(stdout=az_json),  # get_pull_request
            _mock_subprocess_result(stdout="completed"),  # merge
        ]
        with patch.object(AzDevOpsTool, "_validate_cli"):
            tool = AzDevOpsTool(_make_settings(), dry_run=False)
        result = tool.merge_pull_request("101")
        assert result.success is True
