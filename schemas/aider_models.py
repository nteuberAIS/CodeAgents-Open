"""Pydantic models for the Aider tool.

Used by AiderTool to represent command results from the Aider CLI.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AiderResult(BaseModel):
    """Wrapper for Aider CLI command results.

    Follows the same pattern as GitCommandResult.
    """

    command: str                               # The full CLI command that was/would be run
    success: bool
    output: str | None = None
    error: str | None = None
    modified_files: list[str] = Field(default_factory=list)  # Best-effort parsed from output
    dry_run: bool = False                      # True if command was not actually executed
