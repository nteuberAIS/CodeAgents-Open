# Roadmap

## Phase 1: Foundation (current)
- [x] Project structure with modular folders
- [x] Ollama integration via LangChain
- [x] Config layer with env-var overrides and registries
- [x] BaseAgent ABC with extension points
- [x] SprintPlanner agent (LLM → JSON plan)
- [x] CLI entry point with agent selection

## Phase 2: Tools & Notion Sync
- [ ] Notion API tool — read/write sprint databases
- [ ] Git tool — Azure DevOps PR creation, branch management
- [ ] Tool registry auto-loading in agents
- [ ] SprintPlanner writes plans to Notion (not just console)

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
