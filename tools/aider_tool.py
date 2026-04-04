"""Aider CLI subprocess wrapper.

Provides a tool interface for the Aider AI coding assistant,
following the same patterns as BaseGitTool / AzDevOpsTool.
Aider is invoked as an external CLI binary — this tool does NOT
import the aider package directly.
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Any

from schemas.aider_models import AiderResult


class AiderToolError(Exception):
    """Raised when an Aider operation fails."""


class AiderTool:
    """Subprocess wrapper for the Aider CLI.

    Usage:
        tool = AiderTool(settings, dry_run=False)
        result = tool.edit(
            instruction="Add error handling to pipeline.py",
            files=["src/pipeline.py"],
        )
    """

    def __init__(self, settings: Any, dry_run: bool = False) -> None:
        self.dry_run = dry_run

        # Extract aider-specific settings
        self.binary: str = getattr(settings, "aider_binary", "aider")
        self.timeout: int = getattr(settings, "aider_timeout", 120)
        self.repo_dir: str | None = getattr(settings, "aider_repo_dir", None)

        # Model: use aider_model if set, otherwise auto-prefix ollama_model
        aider_model = getattr(settings, "aider_model", None)
        if aider_model:
            self.model = aider_model
        else:
            ollama_model = getattr(settings, "ollama_model", "qwen2.5-coder:7b")
            self.model = f"ollama/{ollama_model}"

        # Edit format: "udiff" works best with smaller models (7b/14b)
        # that naturally produce unified-diff output
        self.edit_format: str | None = getattr(
            settings, "aider_edit_format", "udiff"
        )

        self._validate_cli()

    def _validate_cli(self) -> None:
        """Check that the Aider binary is available on PATH.

        Skipped in dry-run mode so tests don't need the binary installed.
        """
        if self.dry_run:
            return

        try:
            result = subprocess.run(
                [self.binary, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                raise AiderToolError(
                    f"Aider CLI returned non-zero exit code: {result.stderr.strip()}"
                )
        except FileNotFoundError:
            raise AiderToolError(
                f"Aider CLI not found at '{self.binary}'. "
                "Install with: pip install aider-chat"
            )
        except subprocess.TimeoutExpired:
            raise AiderToolError("Aider CLI validation timed out.")

    def _build_command(
        self,
        instruction: str,
        files: list[str],
        read_files: list[str] | None = None,
    ) -> list[str]:
        """Build the Aider CLI command list."""
        cmd = [
            self.binary,
            "--message", instruction,
            "--model", self.model,
            "--yes-always",
            "--no-auto-commits",
            "--no-pretty",
            "--no-stream",
            "--no-show-model-warnings",
            "--no-detect-urls",
            "--map-tokens", "1024",
            "--no-gitignore",
        ]
        if self.edit_format:
            cmd.extend(["--edit-format", self.edit_format])
        for f in files:
            cmd.extend(["--file", f])
        if read_files:
            for f in read_files:
                cmd.extend(["--read", f])
        return cmd

    @staticmethod
    def _parse_modified_files(output: str) -> list[str]:
        """Best-effort parse of modified file paths from Aider output.

        Handles two output formats:
        - "Wrote path/to/file" lines (whole/diff edit formats)
        - "+++ b/path/to/file" lines (udiff edit format)

        Excludes /dev/null (deleted files) and deduplicates results.
        """
        if not output:
            return []
        # Pattern 1: "Wrote <path>" (whole edit format)
        wrote = re.findall(r"^Wrote\s+(.+)$", output, re.MULTILINE)
        # Pattern 2: "+++ b/<path>" (udiff edit format)
        udiff = re.findall(r"^\+\+\+ b/(.+)$", output, re.MULTILINE)
        udiff = [f for f in udiff if f.strip() != "dev/null"]
        # Pattern 3: "Applied edit to <path>" (diff/whole edit formats)
        applied = re.findall(r"^Applied edit to (.+)$", output, re.MULTILINE)
        # Deduplicate, preserve order
        seen: set[str] = set()
        result: list[str] = []
        for f in [m.strip() for m in wrote] + [m.strip() for m in udiff] + [m.strip() for m in applied]:
            if f not in seen:
                seen.add(f)
                result.append(f)
        return result

    def edit(
        self,
        instruction: str,
        files: list[str],
        repo_dir: str | None = None,
        read_files: list[str] | None = None,
    ) -> AiderResult:
        """Run Aider to edit files based on a natural language instruction.

        Args:
            instruction: The coding task description for Aider.
            files: List of file paths to edit.
            repo_dir: Working directory (git repo). Falls back to
                      self.repo_dir from settings, then current dir.
            read_files: Optional read-only context files for Aider.

        Returns:
            AiderResult with command output and status.
        """
        cmd = self._build_command(instruction, files, read_files)
        command_str = " ".join(cmd)
        cwd = repo_dir or self.repo_dir

        # Dry-run: return mock result without executing
        if self.dry_run:
            return AiderResult(
                command=command_str,
                success=True,
                output=None,
                error=None,
                modified_files=[],
                dry_run=True,
            )

        try:
            # Override TERM to avoid prompt_toolkit crash on Windows
            # when the parent shell sets TERM=xterm-256color
            env = os.environ.copy()
            env["TERM"] = "dumb"
            env["PYTHONIOENCODING"] = "utf-8"
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            modified = self._parse_modified_files(result.stdout)
            return AiderResult(
                command=command_str,
                success=result.returncode == 0,
                output=result.stdout or None,
                error=result.stderr or None,
                modified_files=modified,
                dry_run=False,
            )
        except subprocess.TimeoutExpired:
            return AiderResult(
                command=command_str,
                success=False,
                output=None,
                error=f"Aider timed out after {self.timeout}s",
                modified_files=[],
                dry_run=False,
            )
        except FileNotFoundError:
            return AiderResult(
                command=command_str,
                success=False,
                output=None,
                error=f"Aider binary not found at '{self.binary}'",
                modified_files=[],
                dry_run=False,
            )
