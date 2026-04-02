"""Base agent class — all agents inherit from this.

Design notes:
- LLM is injected (not created internally) for testability.
- run() returns a dict for structured, composable output.
- Tools are resolved from the registry and injected via bind_tools().
- Future: add memory, reflection loops, LangGraph state node registration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader
from langchain_ollama import ChatOllama

from schemas.agent_models import AgentResult

# Jinja2 environment for prompt templates — loaded once, shared by all agents
_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    keep_trailing_newline=True,
    undefined=__import__("jinja2").StrictUndefined,
)


class BaseAgent(ABC):
    """Abstract base for all agents in the system.

    Subclasses must implement `run(user_input) -> dict`.
    """

    name: str = "base"

    # Iteration cap — override in subclasses (used by reflection loop / orchestrator)
    MAX_ITERATIONS: int = 1

    # Context curation defaults — override in subclasses to tune
    MAX_CONTENT_ITEMS: int = 10
    MAX_CONTENT_CHARS: int = 8000
    CONTENT_STATUSES: list[str] = ["Ready", "In Progress", "Backlog", "Active"]

    def __init__(self, llm: ChatOllama, context: dict | None = None, *, rag: Any | None = None, snapshot: Any | None = None) -> None:
        self.llm = llm
        self.context = context
        self.rag = rag
        self.snapshot = snapshot  # SnapshotLookup instance
        self.tools: dict[str, Any] = {}

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

    def retrieve(self, query: str, **kwargs: Any) -> list[dict]:
        """Query the RAG retriever if available.

        Returns empty list if no retriever is set.

        Args:
            query: Search text for semantic retrieval.
            **kwargs: Passed through to RAGRetriever.query()
                      (top_k, entity_types, status, sprint_id, score_threshold).

        Returns:
            List of result dicts from RAGRetriever.query(), or [].
        """
        if self.rag is None:
            return []
        return self.rag.query(query, **kwargs)

    def lookup_relations(self, notion_id: str, relation_field: str) -> list[dict]:
        """Relational query via snapshot. Returns [] if no snapshot."""
        if self.snapshot is None:
            return []
        return self.snapshot.get_related(notion_id, relation_field)

    @classmethod
    def curate_context(
        cls,
        raw_context: dict | None,
        content_dir: Path | None = None,
    ) -> dict | None:
        """Filter raw Notion context and attach page content.

        Default implementation:
        1. Passes through all entity lists unchanged.
        2. If content_dir is provided, loads page content for entities
           with has_content=True, filtered by CONTENT_STATUSES.
        3. Adds a "page_content" key: dict mapping notion_id -> markdown.

        Subclasses can override class attributes (MAX_CONTENT_ITEMS, etc.)
        to tune behavior, or override this method entirely.

        Args:
            raw_context: The context dict from _snapshot_to_context().
            content_dir: Path to data/notion/content/ directory.

        Returns:
            Curated context dict with "page_content" key, or None.
        """
        if not raw_context:
            return None

        context = dict(raw_context)  # shallow copy

        if not content_dir or not content_dir.exists():
            context["page_content"] = {}
            return context

        # Collect entities with content, filtered by status
        candidates: list[dict] = []
        for entity_key in ("work_items", "sprints", "docs", "decisions", "risks"):
            for entity in context.get(entity_key, []):
                if not entity.get("has_content", False):
                    continue
                status = entity.get("status", "")
                if cls.CONTENT_STATUSES and status not in cls.CONTENT_STATUSES:
                    continue
                candidates.append(entity)

        # Sort by status priority (position in CONTENT_STATUSES list)
        status_order = {s: i for i, s in enumerate(cls.CONTENT_STATUSES)}
        candidates.sort(key=lambda e: status_order.get(e.get("status", ""), 999))

        # Load content up to limits
        page_content: dict[str, str] = {}
        total_chars = 0
        for entity in candidates[: cls.MAX_CONTENT_ITEMS]:
            notion_id = entity.get("notion_id", "")
            path = content_dir / f"{notion_id}.md"
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8")
            if total_chars + len(content) > cls.MAX_CONTENT_CHARS:
                remaining = cls.MAX_CONTENT_CHARS - total_chars
                if remaining > 200:
                    content = content[:remaining] + "\n... (truncated)"
                else:
                    break
            page_content[notion_id] = content
            total_chars += len(content)

        context["page_content"] = page_content
        return context

    @staticmethod
    def load_prompt(template_path: str, **kwargs: object) -> str:
        """Load and render a Jinja2 prompt template.

        Args:
            template_path: Path relative to prompts/ dir, e.g.
                           "sprint_planner/system.j2".
            **kwargs: Template variables passed to Jinja2 render().

        Returns:
            Rendered prompt string.

        Raises:
            jinja2.TemplateNotFound: If the template file doesn't exist.
        """
        template = _jinja_env.get_template(template_path)
        return template.render(**kwargs)

    def wrap_result(
        self,
        success: bool,
        partial_output: dict,
        error_type: str | None = None,
        error_message: str | None = None,
    ) -> dict:
        """Construct a standardized result envelope.

        Validates via AgentResult Pydantic model, then returns a plain dict
        for compatibility with LangGraph and JSON serialization.

        Args:
            success: True if the agent completed its primary task.
            partial_output: Agent-specific data (full on success, partial on failure).
            error_type: One of "llm", "tool", "logic", "timeout", "infra", or None.
            error_message: Human-readable error description, or None.

        Returns:
            Dict with keys: success, error_type, error_message, partial_output.
        """
        return AgentResult(
            success=success,
            error_type=error_type,
            error_message=error_message,
            partial_output=partial_output,
        ).model_dump()

    @abstractmethod
    def run(self, user_input: str) -> dict:
        """Execute the agent's task and return structured output.

        All implementations must return the standard envelope via
        ``self.wrap_result()``:

        .. code-block:: python

            {
                "success": bool,
                "error_type": str | None,   # "llm", "tool", "logic", "timeout", "infra"
                "error_message": str | None,
                "partial_output": dict,      # agent-specific data
            }

        Args:
            user_input: The user's prompt / instruction.

        Returns:
            Standardized result envelope (see above).
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
    #     """Self-critique loop: re-evaluate output, fix issues.
    #     Enforces MAX_ITERATIONS to prevent rabbit holes.
    #     """
    #     pass
