"""Central configuration for the agent system.

All settings are overridable via environment variables (prefixed with nothing —
just set OLLAMA_BASE_URL, OLLAMA_MODEL, etc.) or via a .env file in the project root.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from pathlib import Path

from langchain_ollama import ChatOllama
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """System-wide settings. Override via env vars or .env file."""

    # -- Ollama LLM --
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_temperature: float = 0.2

    # -- Agent registry --
    # Maps agent name -> dotted import path to its class.
    # Future agents: "coder": "agents.coder.CoderAgent", etc.
    agent_registry: dict[str, str] = {
        "sprint_planner": "agents.sprint_planner.SprintPlannerAgent",
    }

    # -- Tool registry --
    tool_registry: dict[str, str] = {
        "notion": "tools.notion_tool.NotionTool",
    }

    # -- Notion (Phase 2a: read-only sync) --
    notion_api_key: str | None = None
    notion_work_items_db: str = "9500e7b4-a700-49fc-bf69-7585374a1158"
    notion_sprints_db: str = "9d2ceae1-defa-4746-8eec-bedf230935d9"
    notion_docs_db: str = "fb0e9547-7439-409e-81f8-d908a6990eb1"
    notion_decisions_db: str = "391edcf1-0fa3-45da-910e-de4ed04e3a41"
    notion_risks_db: str = "c0d318bd-5b75-4c32-8b30-c1c3ec7712b3"

    # -- Local data directory --
    data_dir: Path = Path("data")

    # -- Future config hooks (uncomment as needed) --
    # rag_retriever_path: str | None = None        # Path to local Notion mirror / vector DB
    # azure_devops_org_url: str | None = None       # https://dev.azure.com/your-org
    # azure_devops_pat: str | None = None           # Personal access token
    # aider_binary: str = "aider"                   # Path to Aider CLI

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


def get_llm(settings: Settings | None = None) -> ChatOllama:
    """Create a ChatOllama instance from settings.

    Accepts an optional Settings override for testing.
    """
    s = settings or get_settings()
    return ChatOllama(
        base_url=s.ollama_base_url,
        model=s.ollama_model,
        temperature=s.ollama_temperature,
    )


def resolve_agent_class(agent_name: str, settings: Settings | None = None):
    """Dynamically import and return an agent class by registry name.

    Example: resolve_agent_class("sprint_planner") -> SprintPlannerAgent class
    """
    s = settings or get_settings()
    dotted_path = s.agent_registry.get(agent_name)
    if not dotted_path:
        available = ", ".join(s.agent_registry.keys())
        raise ValueError(
            f"Unknown agent '{agent_name}'. Available agents: {available}"
        )
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def resolve_tool_class(tool_name: str, settings: Settings | None = None):
    """Dynamically import and return a tool class by registry name.

    Example: resolve_tool_class("notion") -> NotionTool class
    """
    s = settings or get_settings()
    dotted_path = s.tool_registry.get(tool_name)
    if not dotted_path:
        available = ", ".join(s.tool_registry.keys())
        raise ValueError(
            f"Unknown tool '{tool_name}'. Available tools: {available}"
        )
    module_path, class_name = dotted_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
