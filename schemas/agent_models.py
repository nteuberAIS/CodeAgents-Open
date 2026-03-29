"""Pydantic models for agent output contracts.

All agent.run() calls return a standardized envelope for the LangGraph
cascade orchestrator to make routing decisions (retry, skip, abort).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, model_validator


ErrorType = Literal["llm", "tool", "logic", "timeout", "infra"]


class AgentResult(BaseModel):
    """Standard envelope for all agent.run() outputs.

    Callers check ``success`` first, then read ``partial_output``
    for agent-specific data.
    """

    success: bool
    error_type: ErrorType | None = None
    error_message: str | None = None
    partial_output: dict = {}

    @model_validator(mode="after")
    def _check_error_fields(self) -> AgentResult:
        if self.success:
            if self.error_type is not None:
                raise ValueError("error_type must be None when success=True")
            if self.error_message is not None:
                raise ValueError("error_message must be None when success=True")
        else:
            if self.error_type is None:
                raise ValueError("error_type is required when success=False")
        return self
