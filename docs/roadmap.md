# Roadmap

## Phase 1: Foundation ✓
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

## Phase 2.5: Agent Quality Pass ✓
- [x] Prompt externalization — Jinja2 templates in `prompts/` loaded via `BaseAgent.load_prompt()`
- [x] Few-shot examples in `prompts/sprint_planner/few_shots.j2`
- [x] Context curation — `BaseAgent.curate_context()` with per-agent filtering (status, item count, char budget)
- [x] Page content injected into agent context from `data/notion/content/`
- [x] Eval harness — generic `BaseEval` ABC + `EvalRunner` in `evals/`, SprintPlannerEval (3 cases, 7 criteria)
- [x] Per-agent model overrides via `agent_model_overrides` dict in Settings
- [x] Model benchmarking infrastructure ready (eval framework + per-agent overrides; multi-model comparison deferred)

## Phase 2.6a: Doc Cleanup & Dependency Updates ✓
- [x] Updated all docs to reflect Phases 1–2.6b completion
- [x] Moved Aider CLI integration from Phase 5 into Phase 3
- [x] Designed minimum viable error handling spec (failure-modes.md)
- [x] Added `langgraph` dependency, bumped `jinja2` pin
- [x] Resolved Open Question #5 (git tool abstraction)

## Phase 2.6b: Model Benchmarking ✓
- [x] Benchmarking framework in `evals/benchmarks/` with GPU monitoring
- [x] 7 models tested on RTX 2000 Ada 8GB (3 runs each)
- [x] Top performers: qwen2.5-coder:3b (1.000 avg, 115 tok/s) and qwen2.5-coder:7b (1.000 avg, 52.7 tok/s)
- [x] Results in `evals/benchmarks/RESULTS.md`

## Phase 3a: Structured Error Return ✓
- [x] AgentResult schema with `success`, `error_type`, `error_message`, `partial_output`
- [x] All agents return structured results (no raw strings)
- [x] Error classification: `llm_error`, `tool_error`, `validation_error`, `timeout`

## Phase 3b: SprintState Schema ✓
- [x] SprintState TypedDict for LangGraph cascade state
- [x] Reducer fields for errors and failed_task_ids (LangGraph `Annotated[list, operator.add]`)
- [x] Pure helper functions: `create_initial_state`, `get_current_task`, `advance_task`, `should_abort`

## Phase 3c: Aider Tool ✓
- [x] AiderTool wrapping Aider CLI for AI-assisted code edits
- [x] Dry-run mode, configurable timeout, model passthrough
- [x] Registered in tool_registry

## Phase 3d: CoderAgent ✓
- [x] CoderAgent wraps AiderTool for code generation from sprint tasks
- [x] Accepts test feedback for outer retry loop
- [x] Prompt templates in `prompts/coder/`

## Phase 3e: TesterAgent + UpdaterAgent ✓
- [x] TesterAgent runs pytest, parses results, reports pass/fail
- [x] UpdaterAgent creates PRs and updates Notion status
- [x] Both registered in agent_registry

## Phase 3f: LangGraph Cascade ✓
- [x] StateGraph orchestration: plan → setup → code → test → update → check
- [x] Inner reflection: CoderAgent retries Aider internally (up to 5 iterations)
- [x] Outer reflection: test failures route back to code_node (up to 2 retries)
- [x] Abort threshold: cascade stops if too many tasks fail
- [x] CascadeRunner high-level wrapper with summary reporting

## Phase 3g: CLI Integration + Docs ✓
- [x] `cascade` CLI subcommand with `--dry-run`, `--max-tasks`, `--sprint-id`, `--sync`, `--model`
- [x] State persistence to `data/cascade/{sprint_id}.json`
- [x] `--list` and `--show` for viewing past runs
- [x] Documentation updates (roadmap, architecture, project outline)

## Phase 3.5: Live Validation & Fixes ✓
- [x] First dry-run cascade on Sprint 1.4 (17 Notion tasks loaded, full pipeline exercised)
- [x] First live cascade runs (4 runs against SynDataPlatform Azure DevOps repo)
- [x] Fixed: `TEST_COMMAND=skip` sentinel check in TesterAgent
- [x] Fixed: Windows `az.cmd` subprocess resolution + cmd.exe metacharacter sanitization
- [x] Fixed: Missing `cwd` in git subprocess calls (all git ops ran in wrong repo)
- [x] Added `commit_push_node` to cascade (commits Aider changes, pushes branch to remote)
- [x] Fixed branch naming collision (`task/sprint-{N}/{task-id}` prefix)
- [x] Added Aider CLI flags (`--no-show-model-warnings`, `--no-gitignore`, `--no-detect-urls`, `--edit-format udiff`)
- [x] Added `aider_edit_format` setting (default: `udiff`)
- [x] Increased Aider output capture from 2000 to 4000 chars
- [x] Added completion validation in `check_node` (warns when no files changed or no PR created)
- [x] Moved Notion loading out of cascade graph into CLI layer (`main.py`)
- [x] Removed TaskPlannerAgent (premature, deferred to post-RAG)
- [x] All findings documented in `docs/dev-planning/failure-modes.md`

## Phase 4a: RAG Ingestion ✓
- [x] ChromaDB vector database with Ollama embeddings (nomic-embed-text)
- [x] Hybrid chunking strategy (document-level for short pages, section-level for long)
- [x] Notion markup stripping before embedding (reduces token count)
- [x] Entity metadata attached to chunks (entity_type, status, priority, sprint_id)
- [x] `ingest` CLI subcommand with `--force` and `--dry-run` support
- [x] Cosine similarity distance metric (`hnsw:space: cosine`)

## Phase 4b: RAG Retriever ✓
- [x] RAGRetriever with semantic search, metadata filtering, score thresholding
- [x] `notion_ids` filter for composed queries (snapshot IDs → RAG filter)
- [x] `format_results()` for prompt-ready context strings with char budget
- [x] BaseAgent `rag` parameter and `retrieve()` helper method
- [x] Settings: `rag_top_k`, `rag_score_threshold`

## Phase 4c: Agent Integration & Context-Aware Planning ✓
- [x] SnapshotLookup — relational index over JSON snapshots for O(1) entity lookups
- [x] Composed query pattern: snapshot provides relation IDs → RAG filters by those IDs
- [x] SprintPlannerAgent: RAG for sprint goal context + snapshot for linked risks
- [x] CoderAgent: RAG for task-relevant docs + snapshot for linked docs/ADRs + composed queries
- [x] BaseAgent `snapshot` parameter and `lookup_relations()` helper method
- [x] RAG/snapshot wired through cascade via CascadeRunner → build_cascade_graph → partial() bindings
- [x] Graceful degradation — agents work without RAG/snapshot (fallback to curate_context)
- [x] Prompts updated with conditional retrieved_context and linked_context sections

## Phase 4c.5: Parser Fixes & Live Validation ✓
- [x] Aider `modified_files` parser handles udiff format (`+++ b/path` lines)
- [x] PR URL extraction parses JSON from `az repos pr create -o json` (constructs web URL)
- [x] Live cascade validated: ADLS capacity alert task, end-to-end with RAG context
- [x] RAG quality confirmed: linked docs improved coder instruction quality

## Phase 5: IDE & Review Tools
- [ ] Continue.dev integration for IDE-level changes
- [ ] Automated code review via TesterAgent

## Phase 6: Production Hardening
- [ ] Logging and observability
- [ ] Error recovery and retry logic
- [ ] Cost tracking (tokens/model usage, even if local)
- [ ] CI/CD pipeline integration
