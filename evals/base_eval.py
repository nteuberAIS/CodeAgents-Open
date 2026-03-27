"""Base evaluation framework — models and abstract class.

All eval suites inherit from BaseEval and define agent-specific
eval cases and scoring criteria.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class EvalCase(BaseModel):
    """A single evaluation test case."""

    name: str
    description: str
    prompt: str
    context: dict | None = None
    expected: dict[str, Any] = {}


class EvalScore(BaseModel):
    """Score for a single evaluation criterion."""

    name: str
    passed: bool
    score: float  # 0.0 to 1.0
    detail: str = ""


class EvalResult(BaseModel):
    """Result of running one eval case against an agent."""

    case_name: str
    agent_output: dict
    scores: list[EvalScore]
    overall_pass: bool
    overall_score: float  # Average of individual scores


class BaseEval(ABC):
    """Abstract base for agent evaluation suites.

    Subclasses define eval cases and scoring functions for a specific agent.
    """

    agent_name: str = "base"

    @abstractmethod
    def get_cases(self) -> list[EvalCase]:
        """Return the list of eval cases for this agent."""
        ...

    @abstractmethod
    def score(self, case: EvalCase, output: dict) -> list[EvalScore]:
        """Score an agent's output against an eval case.

        Returns a list of EvalScore objects, one per criterion.
        """
        ...
