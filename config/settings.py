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

    # Per-agent model overrides (env: AGENT_MODEL_OVERRIDES='{"sprint_planner":"mistral:7b"}')
    agent_model_overrides: dict[str, str] = {}

    # -- Agent registry --
    # Maps agent name -> dotted import path to its class.
    # Future agents: "coder": "agents.coder.CoderAgent", etc.
    agent_registry: dict[str, str] = {
        "sprint_planner": "agents.sprint_planner.SprintPlannerAgent",
        "coder": "agents.coder.CoderAgent",
        "tester": "agents.tester.TesterAgent",
        "updater": "agents.updater.UpdaterAgent",
    }

    # -- Tool registry --
    tool_registry: dict[str, str] = {
        "notion": "tools.notion_tool.NotionTool",
        "notion_write": "tools.notion_write_tool.NotionWriteTool",
        "github": "tools.github_tool.GitHubTool",
        "azdevops": "tools.azdevops_tool.AzDevOpsTool",
        "aider": "tools.aider_tool.AiderTool",
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

    # -- GitHub (Phase 2b) --
    github_repo_dir: str | None = None  # Path to local repo clone (default: current dir)

    # -- Azure DevOps (Phase 2b) --
    azure_devops_org_url: str | None = None   # https://dev.azure.com/your-org
    azure_devops_project: str | None = None
    azure_devops_repo: str | None = None

    # -- Aider CLI (Phase 3c) --
    aider_binary: str = "aider"                     # Path to Aider CLI binary
    aider_model: str | None = None                  # Full provider-prefixed model (e.g. "ollama/qwen2.5-coder:7b"); auto-derived from ollama_model if None
    aider_timeout: int = 120                        # Subprocess timeout in seconds
    aider_repo_dir: str | None = None               # Path to target repo for Aider edits
    aider_edit_format: str = "udiff"                  # Aider edit format: "udiff" (best for 7b/14b), "diff", "whole"

    # -- TesterAgent (Phase 3e) --
    test_command: str = "pytest"                     # Test command to run
    test_timeout: int = 300                          # Test subprocess timeout in seconds (5 min)
    test_repo_dir: str | None = None                 # Target repo for tests (falls back to aider_repo_dir)

    # -- RAG (Phase 4) --
    chroma_db_path: Path = Path("data/chroma")
    embedding_model: str = "nomic-embed-text"
    rag_chunk_size: int = 4000           # Files under this stay as single document
    rag_chunk_overlap: int = 200         # Overlap for secondary char-limit splits
    rag_max_chunk_size: int = 4000       # Hard cap per chunk (nomic 8K tokens ≈ ~4K chars for dense content)
    rag_top_k: int = 5                   # Default number of retrieval results
    rag_score_threshold: float | None = None  # Min similarity score (None = no filter)

    # -- Logging (Phase 5a) --
    log_level: str = "INFO"
    log_file: str = "data/logs/codeagents.jsonl"
    log_json: bool = True

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache()
def get_settings() -> Settings:
    """Return cached settings instance."""
    return Settings()


def get_llm(settings: Settings | None = None, agent_name: str | None = None) -> ChatOllama:
    """Create a ChatOllama instance from settings.

    If agent_name is provided and has an entry in agent_model_overrides,
    that model is used instead of the default ollama_model.

    Accepts an optional Settings override for testing.
    """
    s = settings or get_settings()
    model = s.ollama_model
    if agent_name and agent_name in s.agent_model_overrides:
        model = s.agent_model_overrides[agent_name]
    return ChatOllama(
        base_url=s.ollama_base_url,
        model=model,
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
