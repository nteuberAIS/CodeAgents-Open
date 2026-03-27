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
- `config/` — Settings, LLM factory, agent/tool registries
- `data/notion/` — Local JSON snapshots from Notion sync
- `tests/` — pytest tests with fixtures in `tests/fixtures/`

## Conventions
- **Registry pattern**: Add agents/tools by creating a class and registering in `config/settings.py`
- **LLM injection**: Agents receive `ChatOllama` in `__init__`, never create their own
- **Structured output**: Agent `run()` returns `dict`, not strings
- **Local-first**: Agents read from local JSON snapshots, never write to cloud Notion during execution
- **Human gates**: Destructive actions (git merge to main, Notion cloud write) always require approval
- **Testing**: Mock external services (Notion client, Ollama). Use fixtures in `tests/fixtures/`

## Key Commands
```bash
python main.py sync                    # Pull Notion DBs + page content to local JSON
python main.py sync --dry-run          # Check without writing
python main.py run "Plan sprint 8"     # Run agent with prompt (auto-binds tools)
python main.py run "..." --sync        # Sync first, then run
python main.py run "..." --no-tools    # LLM planning only, no tool execution
python main.py run "..." --dry-run     # Show what would happen
pytest tests/                          # Run all tests (266 tests)
```

## Current State
- Phase 1 (Foundation): Complete
- Phase 2a (Notion Read-Only Sync): Complete
- Phase 2b (Git Tool): Complete
- Phase 2c (Notion Local Write-Back): Complete
- Phase 2d (Wire Tools into Agents): Complete
- Phase 2e (Page Content Sync): Complete
- Next: Phase 2.5 (Agent Quality Pass) — see docs/roadmap.md
