# Architecture

## System Overview

A modular, local-first AI agent system powered by Ollama. Designed for zero-cost
operation with a clear upgrade path to multi-agent workflows.

## Folder Structure

```
CodeAgents-Open/
├── agents/                 # Agent definitions
│   ├── base.py             # BaseAgent ABC (all agents inherit)
│   └── sprint_planner.py   # Phase 1: Sprint planning agent
├── tools/                  # External tool wrappers (Phase 2+)
│   └── __init__.py         # Future: git, Notion, Aider tools
├── config/
│   └── settings.py         # Pydantic settings, LLM factory, registries
├── docs/                   # Project documentation
├── main.py                 # CLI entry point
└── requirements.txt
```

## Core Concepts

### Agent Registry
Agents are registered in `config/settings.py` as a dict mapping names to dotted
import paths. `resolve_agent_class()` dynamically imports them. To add a new agent:

1. Create `agents/my_agent.py` with a class inheriting `BaseAgent`
2. Add `"my_agent": "agents.my_agent.MyAgent"` to `agent_registry`
3. Run with `python main.py --agent my_agent "your prompt"`

### Tool Registry (Phase 2+)
Same pattern as agents. Tools will be loadable classes that agents can bind via
`agent.bind_tools([...])`. Planned tools:
- `git_tool` — Azure DevOps PRs, branch management
- `notion_tool` — Notion API read/write
- `aider_tool` — Aider CLI for code edits
- `continue_tool` — Continue.dev IDE integration

### LLM Factory
`get_llm()` creates a `ChatOllama` instance from settings. All config is
env-var overridable (set `OLLAMA_MODEL=mistral:7b` to swap models).

## Future Architecture: Multi-Agent Cascade

```
User Prompt
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Planner    │────▶│    Coder     │────▶│    Tester    │────▶│   Updater    │
│ (sprint plan)│     │ (code edits) │     │ (run tests)  │     │ (Notion/git) │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │                    │
       ▼                    ▼                    ▼                    ▼
   RAG Context         Aider/Continue       pytest/lint          Notion API
   (Notion mirror)     (code tools)         (validation)         Azure DevOps
```

This cascade will be orchestrated via LangGraph `StateGraph`, where each agent
is a node and edges define the flow. Reflection loops at each node prevent
rabbit holes (capped iterations).
