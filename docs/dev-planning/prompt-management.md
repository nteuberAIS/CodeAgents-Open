# Deferred Decision: Prompt Management

> **Status**: Partially resolved — core templating implemented in Phase 2.5, remaining items deferred.
> **Relevant Phase**: Phase 2+ (all agents depend on prompts)
> **Last Updated**: 2026-04-02

---

## Context

Every agent's quality is bounded by its prompt. As the system scales from one
agent to many, prompt management becomes critical — versioning, testing,
templating, and ensuring consistency across the cascade.

This decision also covers related concerns:
- PR review flow templates
- Branch naming conventions
- Commit message formats
- Agent output schemas

## Questions to Resolve

### 1. Prompt Storage & Format ✓ Resolved

- Jinja2 `.j2` files in `prompts/{agent_name}/` — git-versioned, no registry needed.
- No structured metadata for now — git history is sufficient. Eval scores tracked in `evals/`.
- Resolved in Phase 2.5.

### 2. Prompt Templating ✓ Resolved

- `BaseAgent.load_prompt()` renders Jinja2 templates with `StrictUndefined`.
- Multi-section prompts implemented: system (`system.j2`), few-shot examples (`few_shots.j2`).
- Conditional sections and output schema enforcement handled per-agent.
- Resolved in Phase 2.5. Domain terms remain open (see Question 6).

### 3. Prompt Versioning

Beyond git history, do we need:
- Named versions (v1, v2) with ability to pin agents to specific versions?
- A/B testing: run same task with two prompt versions, compare output?
- Eval scores attached to prompt versions?

### 4. PR & Git Templates

These are prompt-adjacent — they define how agents format their git output:

- **Branch naming**: `task/sprint-{N}/{task-id}` (decided, updated Phase 3.5)
- **Commit messages**: Structured format TBD (e.g., `[SP8-001] Add pipeline X`)
- **PR titles**: Template TBD
- **PR descriptions**: Template with task context, changes summary, test results

### 5. Cross-Agent Consistency

How do we ensure the Planner's output format matches what the Coder expects?
- Shared output schemas?
- Contract testing between agents?
- Single source of truth for data structures (e.g., task schema)?

> **Partially resolved (Phase 3a–3b):** AgentResult envelope (`schemas/agent_models.py`)
> provides a standard return contract across all agents. SprintState TypedDict
> (`schemas/sprint_state.py`) defines the cascade state contract. Per-agent output
> validation remains open for Phase 6.

### 6. Domain-Specific Prompt Tuning

The data platform has specific terminology and patterns:
- Medallion architecture (bronze/silver/gold)
- ADF pipeline JSON structure
- Bicep module conventions
- Power BI DAX/M patterns
- Azure SQL naming conventions

How do we inject this domain knowledge into prompts without bloating them?
- RAG handles most of it?
- Static "domain primer" section in each prompt?
- Separate domain knowledge base that agents can query?

## Current State

Phase 2.5 resolved core prompt management:

- Jinja2 templates in `prompts/{agent_name}/` (e.g., `system.j2`, `few_shots.j2`)
- `BaseAgent.load_prompt(template_path, **kwargs)` renders templates with `StrictUndefined`
- Multi-section prompts: system instructions, context injection, few-shot examples
- SprintPlannerAgent returns structured dict (`sprint`, `goal`, `tasks`)
- Eval harness validates prompt quality (`evals/sprint_planner_eval.py`)

## Next Steps

- [x] Design prompt file format → Jinja2 `.j2` files in `prompts/`, loaded via `BaseAgent.load_prompt()`
- [x] Define output schemas for each agent → SprintPlannerAgent returns structured dict
- [x] Cross-agent output schemas / cascade contract → AgentResult envelope + SprintState TypedDict (Phase 3a–3b)
- [ ] Create PR/commit message templates (deferred — UpdaterAgent uses LLM-generated descriptions)
- [ ] Prompt versioning / A/B testing (Phase 6)
- [x] Domain primer for data platform terminology → Addressed via RAG context enrichment (Phase 4c). Agents receive domain-specific content from linked docs and semantic search, replacing the need for a static primer.
