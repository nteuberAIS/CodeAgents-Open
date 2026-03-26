"""Base agent class — all agents inherit from this.

Design notes:
- LLM is injected (not created internally) for testability.
- run() returns a dict for structured, composable output.
- Future: add tool binding, memory, reflection loops, LangGraph state node registration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from langchain_ollama import ChatOllama


class BaseAgent(ABC):
    """Abstract base for all agents in the system.

    Subclasses must implement `run(user_input) -> dict`.
    """

    name: str = "base"

    def __init__(self, llm: ChatOllama) -> None:
        self.llm = llm
        # Future: self.tools = []          — bound tool instances
        # Future: self.memory = None       — conversation / retrieval memory
        # Future: self.max_iterations = 5  — reflection loop cap (kill rabbit holes)

    @abstractmethod
    def run(self, user_input: str) -> dict:
        """Execute the agent's task and return structured output.

        Args:
            user_input: The user's prompt / instruction.

        Returns:
            A dict with the agent's result. Schema varies by agent.
        """
        ...

    # -- Future extension points --
    #
    # def as_langgraph_node(self):
    #     """Register this agent as a node in a LangGraph StateGraph.
    #     Each agent becomes a node; edges define the cascade:
    #     Planner -> Coder -> Tester -> Updater
    #     """
    #     pass
    #
    # def bind_tools(self, tools: list):
    #     """Attach tool instances (git, Notion, Aider) to this agent."""
    #     self.tools = tools
    #
    # def reflect(self, result: dict) -> dict:
    #     """Self-critique loop: re-evaluate output, fix issues, cap iterations.
    #     Prevents rabbit holes by enforcing max_iterations.
    #     """
    #     pass
