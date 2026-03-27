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

## Phase 2b: Git Tool ✓
- [x] BaseGitTool ABC with shared CLI execution (`tools/git_tool.py`)
- [x] GitHubTool — `gh` CLI wrapper (`tools/github_tool.py`)
- [x] AzDevOpsTool — `az repos` CLI wrapper (`tools/azdevops_tool.py`)
- [x] Dry-run mode for all write operations
- [x] Human approval gate for merges to main/master
- [x] Branch naming helpers (sprint-N/task-id convention)
- [x] Pydantic models: Branch, PullRequest, CommitInfo, GitCommandResult (`schemas/git_models.py`)

## Phase 2c: Notion Local Write-Back ✓
- [x] NotionWriteTool with create/update for all 5 entity types (`tools/notion_write_tool.py`)
- [x] Append-only pending changelog (`data/notion/pending_changes.json`)
- [x] Local snapshot with merged state (`data/notion/local_snapshot.json`)
- [x] Cloud snapshot files never modified (safety by design)
- [x] Field-level validation with allowlists per entity type
- [x] Cloud push deliberately NOT implemented (requires human approval, separate phase)

## Phase 2d: Wire Tools into Agents ✓
- [x] BaseAgent.bind_tools() with graceful degradation
- [x] Tool auto-loading from REQUIRED_TOOLS + OPTIONAL_TOOLS
- [x] SprintPlanner executes plans: creates Notion items + git branches
- [x] CLI auto-binds tools, `--no-tools` flag for LLM-only mode
- [x] Context loading prefers local snapshot (with pending changes)

## Phase 2e: Page Content Sync ✓
- [x] Notion block-to-markdown renderer supporting 30+ block types (`tools/notion_renderer.py`)
- [x] Page content sync — fetches blocks, renders to markdown, writes to `data/notion/content/`
- [x] Template detection and storage (`data/notion/templates/{db_name}/`)
- [x] Sub-page discovery with depth-limited recursion (max 4 levels, cycle prevention)
- [x] Rate limiting (0.34s/request) for Notion API compliance
- [x] Write tool extended — page content as allowed update field for all entity types
- [x] `has_content` field added to all entity models
- [x] SyncMeta extended with content_counts, template_counts, templates list

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
