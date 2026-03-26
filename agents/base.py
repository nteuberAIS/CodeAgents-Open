"""Base agent class — all agents inherit from this.

Design notes:
- LLM is injected (not created internally) for testability.
- run() returns a dict for structured, composable output.
- Tools are resolved from the registry and injected via bind_tools().
- Future: add memory, reflection loops, LangGraph state node registration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from langchain_ollama import ChatOllama


class BaseAgent(ABC):
    """Abstract base for all agents in the system.

    Subclasses must implement `run(user_input) -> dict`.
    """

    name: str = "base"

    def __init__(self, llm: ChatOllama, context: dict | None = None) -> None:
        self.llm = llm
        self.context = context
        self.tools: dict[str, Any] = {}
        # Future: self.memory = None       — conversation / retrieval memory
        # Future: self.max_iterations = 5  — reflection loop cap (kill rabbit holes)

    def bind_tools(
        self,
        tool_names: list[str],
        settings: Any,
        dry_run: bool = False,
    ) -> dict[str, bool]:
        """Resolve and instantiate tools from the registry.

        Args:
            tool_names: Tool names to bind (from tool_registry).
            settings: Settings instance for tool construction.
            dry_run: Passed to tools that support it (git tools).

        Returns:
            Dict of tool_name -> success (bool). Failed tools are logged
            but don't raise — allows graceful degradation.
        """
        from config.settings import resolve_tool_class

        results: dict[str, bool] = {}
        for name in tool_names:
            try:
                tool_cls = resolve_tool_class(name, settings)
                # Tools have different __init__ signatures:
                # - NotionTool(settings) / NotionWriteTool(settings)
                # - GitHubTool(settings, dry_run) / AzDevOpsTool(settings, dry_run)
                # Try (settings, dry_run) first, fall back to (settings,)
                try:
                    tool = tool_cls(settings=settings, dry_run=dry_run)
                except TypeError:
                    tool = tool_cls(settings=settings)

                self.tools[name] = tool
                results[name] = True
            except Exception:
                # Tool failed to init — store None, report failure
                self.tools[name] = None
                results[name] = False
        return results

    def get_tool(self, name: str) -> Any | None:
        """Get a bound tool by name.

        Returns None if tool not bound or failed to initialize.
        """
        tool = self.tools.get(name)
        if tool is None:
            return None
        return tool

    def has_tool(self, name: str) -> bool:
        """Check if a tool is successfully bound."""
        return self.tools.get(name) is not None

    @abstractmethod
    def run(self, user_input: str) -> dict:
        """Execute the agent's task and return structured output.

        Args:
            user_input: The user's prompt / instruction.

        Returns:
            A dict with the agent's result. Schema varies by agent.
        """
        ...

    # -- Future extension points (Phase 3) --
    #
    # def as_langgraph_node(self):
    #     """Register this agent as a node in a LangGraph StateGraph.
    #     Each agent becomes a node; edges define the cascade:
    #     Planner -> Coder -> Tester -> Updater
    #     """
    #     pass
    #
    # def reflect(self, result: dict) -> dict:
    #     """Self-critique loop: re-evaluate output, fix issues, cap iterations.
    #     Prevents rabbit holes by enforcing max_iterations.
    #     """
    #     pass
