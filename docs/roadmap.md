# Roadmap

## Phase 1: Foundation (current)
- [x] Project structure with modular folders
- [x] Ollama integration via LangChain
- [x] Config layer with env-var overrides and registries
- [x] BaseAgent ABC with extension points
- [x] SprintPlanner agent (LLM → JSON plan)
- [x] CLI entry point with agent selection

## Phase 2a: Notion Read-Only Sync ✓
- [x] Pydantic models for all 5 Notion databases (`schemas/notion_models.py`)
- [x] NotionTool with paginated sync, property extractors, mappers (`tools/notion_tool.py`)
- [x] CLI `sync` subcommand with `--dry-run` support
- [x] Local JSON snapshot storage (`data/notion/`)
- [x] Test suite with fixtures (`test_notion_models.py`, `test_notion_tool.py`)

## Phase 2b: Git Tool
- [ ] Git tool — Azure DevOps (`az repos`) + GitHub (`gh`) CLI wrappers
- [ ] Branch creation, commit, PR creation with human gates
- [ ] Read-only operations first (list branches, PRs), then write

## Phase 2c: Notion Write Tool
- [ ] Extend NotionTool with create/update page capabilities
- [ ] Human approval gate before cloud writes

## Phase 2d: Wire Tools into Agents
- [ ] Tool auto-loading from registry
- [ ] SprintPlanner binds NotionTool + GitTool
- [ ] SprintPlanner creates Notion pages and git branches as part of planning

## Phase 2.5: Agent Quality Pass
- [ ] Prompt engineering with few-shot examples
- [ ] Context curation (filter/summarize Notion data for agents)
- [ ] Model benchmarking (3-4 models on eval set)
- [ ] Eval harness in `evals/` directory
- [ ] Per-agent model overrides (`OLLAMA_MODEL_CODER` pattern)

## Phase 3: Multi-Agent Cascade
- [ ] CoderAgent — generates code changes from sprint tasks
- [ ] TesterAgent — runs tests, reports results
- [ ] UpdaterAgent — pushes to Notion, creates PRs
- [ ] LangGraph StateGraph orchestration (Planner → Coder → Tester → Updater)
- [ ] Reflection loops with iteration caps (kill rabbit holes)

## Phase 4: RAG & Context
- [ ] Local Notion mirror (export → vector DB)
- [ ] RAG retriever injected into agent prompts
- [ ] Context-aware sprint planning from historical data

## Phase 5: Code Tools
- [ ] Aider CLI integration for AI-assisted edits
- [ ] Continue.dev integration for IDE-level changes
- [ ] Automated code review via TesterAgent

## Phase 6: Production Hardening
- [ ] Logging and observability
- [ ] Error recovery and retry logic
- [ ] Cost tracking (tokens/model usage, even if local)
- [ ] CI/CD pipeline integration
