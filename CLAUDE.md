# CodeAgents-Open

Local, zero-cost AI agent system for sprint automation. Solo developer project.

## Quick Reference
- Python 3.11+, Windows 11 (use Unix shell syntax in Claude Code)
- LLM: Ollama + LangChain (qwen2.5-coder:7b default)
- Config: Pydantic Settings with env-var overrides
- Notion SDK: `notion-client` — uses `data_sources.query()` for pagination

## Project Structure
- `agents/` — Agent classes inheriting `BaseAgent` (ABC with LLM injection)
- `tools/` — External tool wrappers (Notion sync, Notion write-back, Git providers, block renderer)
- `schemas/` — Pydantic models (NOT `models/` — that's gitignored for Ollama cache)
- `rag/` — RAG pipeline (ChromaDB ingestion, semantic retriever, snapshot relational lookup — Phase 4)
- `config/` — Settings, LLM factory, agent/tool registries
- `data/notion/` — Local JSON snapshots from Notion sync
- `prompts/` — Jinja2 prompt templates per agent (e.g., `sprint_planner/system.j2`)
- `evals/` — Eval framework: BaseEval ABC, EvalRunner, per-agent eval suites
- `tests/` — pytest tests with fixtures in `tests/fixtures/`

## Conventions
- **Registry pattern**: Add agents/tools by creating a class and registering in `config/settings.py`
- **LLM injection**: Agents receive `ChatOllama` in `__init__`, never create their own
- **Structured output**: Agent `run()` returns `dict`, not strings
- **Local-first**: Agents read from local JSON snapshots, never write to cloud Notion during execution
- **Human gates**: Destructive actions (git merge to main, Notion cloud write) always require approval
- **Testing**: Mock external services (Notion client, Ollama). Use fixtures in `tests/fixtures/`
- **Prompt templates**: Externalized to `prompts/{agent_name}/` as Jinja2 `.j2` files, loaded via `BaseAgent.load_prompt()`
- **Context curation**: Agents declare `MAX_CONTENT_ITEMS`, `MAX_CONTENT_CHARS`, `CONTENT_STATUSES` class attrs; `BaseAgent.curate_context()` filters page content accordingly
- **Eval framework**: Subclass `BaseEval` in `evals/`, register in `EVAL_REGISTRY` in `evals/runner.py`
- **Per-agent models**: Override via `agent_model_overrides` dict in Settings (env: `AGENT_MODEL_OVERRIDES='{"sprint_planner":"mistral:7b"}'`)

## Key Commands
```bash
python main.py sync                    # Pull Notion DBs + page content to local JSON
python main.py sync --dry-run          # Check without writing
python main.py run "Plan sprint 8"     # Run agent with prompt (auto-binds tools)
python main.py run "..." --sync        # Sync first, then run
python main.py run "..." --no-tools    # LLM planning only, no tool execution
python main.py run "..." --dry-run     # Show what would happen
python main.py ingest                  # Embed Notion content into ChromaDB
python main.py ingest --force          # Re-ingest from scratch (delete + rebuild)
python main.py ingest --dry-run        # Show what would be ingested
python main.py cascade "Deploy SHIR"          # Run full cascade (Planner→Coder→Tester→Updater)
python main.py cascade "..." --dry-run        # Show what would happen
python main.py cascade "..." --max-tasks 2    # Limit tasks processed
python main.py cascade "..." --abort-threshold 0.3  # Custom failure threshold (default: 0.5)
python main.py cascade "..." --sprint-id s-8  # Explicit sprint ID
python main.py cascade "..." --sync           # Sync Notion first, then run
python main.py cascade "..." --model qwen2.5-coder:7b  # Override model
python main.py cascade --list                 # List past cascade runs
python main.py cascade --show sprint-8        # Show saved state for a run
python main.py eval                           # Run all evals (requires Ollama)
python main.py eval --agent sprint_planner    # Run specific agent evals
python main.py eval --model mistral:7b        # Override model for benchmarking
pytest tests/                                 # Run all tests
```

## Project Structure (cont.)
- `orchestration/` — LangGraph cascade: graph definition, runner, state management
- `data/cascade/` — Saved cascade run states (JSON)
- `data/chroma/` — ChromaDB persistent vector storage (gitignored via `data/`)

## Current State
- Phase 1 (Foundation): Complete
- Phase 2a–2e (Notion Sync, Git, Write-Back, Wiring, Content): Complete
- Phase 2.5 (Agent Quality Pass): Complete
- Phase 2.6a–2.6b (Doc Cleanup, Benchmarking): Complete
- Phase 3 (Multi-Agent Cascade): Complete
- Phase 4 (RAG & Context): Complete
