# Deferred Decision: Prompt Management

> **Status**: Deferred — important, needs dedicated time to design properly.
> **Relevant Phase**: Phase 2+ (all agents depend on prompts)
> **Last Updated**: 2026-03-25

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

### 1. Prompt Storage & Format

- Markdown files in `prompts/` (current plan) — simple, git-versioned.
- Should prompts include structured metadata (version, author, last-eval-score)?
- Do we need a prompt registry beyond the filesystem?

### 2. Prompt Templating

- Current: Simple string with `{context}` placeholder for RAG injection.
- Future needs:
  - Multi-section prompts (system, user, few-shot examples)
  - Conditional sections (include git context only for Coder)
  - Output schema enforcement (JSON schema in prompt)
  - Data platform domain terms (Azure SQL, ADF, Medallion, etc.)

### 3. Prompt Versioning

Beyond git history, do we need:
- Named versions (v1, v2) with ability to pin agents to specific versions?
- A/B testing: run same task with two prompt versions, compare output?
- Eval scores attached to prompt versions?

### 4. PR & Git Templates

These are prompt-adjacent — they define how agents format their git output:

- **Branch naming**: `sprint-{N}/{task-id}` (decided)
- **Commit messages**: Structured format TBD (e.g., `[SP8-001] Add pipeline X`)
- **PR titles**: Template TBD
- **PR descriptions**: Template with task context, changes summary, test results

### 5. Cross-Agent Consistency

How do we ensure the Planner's output format matches what the Coder expects?
- Shared output schemas?
- Contract testing between agents?
- Single source of truth for data structures (e.g., task schema)?

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

- `agents/sprint_planner.py` has an inline prompt string.
- No `prompts/` directory exists yet.
- No templating or versioning in place.

## Next Steps

- [ ] Design prompt file format (frontmatter + body)
- [ ] Define output schemas for each agent
- [ ] Create PR/commit message templates
- [ ] Decide on eval-driven prompt iteration workflow
- [ ] Build domain primer for data platform terminology
