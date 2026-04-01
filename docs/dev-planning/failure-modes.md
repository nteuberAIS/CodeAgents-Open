# Deferred Decision: Failure Modes & Error Recovery

> **Status**: Partially resolved — MVP error handling designed, implementation in Phase 3.
> **Relevant Phase**: Phase 3+ (Multi-Agent Cascade)
> **Last Updated**: 2026-03-27

---

## Context

When agents encounter errors during sprint execution, the system needs defined
behavior for how to handle, recover, and escalate. This is especially important
for early iterations where agent reliability is unproven.

## Questions to Resolve

### 1. Failure Categories

What types of failures should the system distinguish?

- **LLM failures**: Model produces invalid output, hallucination, off-topic
- **Tool failures**: Git command fails, Notion API error, file not found
- **Logic failures**: Tests fail, code doesn't match spec, wrong branch state
- **Timeout failures**: Agent exceeds time/iteration cap
- **Infrastructure failures**: Ollama crashes, VRAM exhaustion, disk full

### 2. Recovery Strategies

For each failure category, what's the right recovery approach?

- **Retry with context**: Re-run the agent with error message in prompt
- **Rollback to checkpoint**: Revert to last known good state
- **Skip and continue**: Mark task as failed, continue to next task
- **Escalate to human**: Pause everything, dump context, wait for input
- **Abort sprint**: Stop all work, preserve state for diagnosis

### 3. Cascade Impact

When one agent fails, how does it affect downstream agents?

- If Coder fails on task SP8-003, can Tester still run on SP8-001 and SP8-002?
- Should the Supervisor re-route around failed tasks?
- At what point does a sprint become "too broken" to continue?

### 4. Early Iteration Guardrails

For the first few sprints, what extra safety nets do we need?

- Lower iteration caps?
- More human approval gates?
- Mandatory dry-run before live execution?
- Task-by-task approval instead of batch?

### 5. Git-Specific Failure Modes

Branching strategy creates its own failure modes:

- Task branch from main conflicts with another task branch on sprint branch
- PR auto-review passes but code is subtly wrong
- Sprint branch diverges significantly from main during long sprint
- Merge conflicts between task branches

## Minimum Viable Error Handling (Phase 3)

Design spec for the error handling that will be implemented when Phase 3 agents are built.

### Structured Error Return

All `agent.run()` calls return a dict with these standard keys:

| Key              | Type          | Description                                                  |
|------------------|---------------|--------------------------------------------------------------|
| `success`        | `bool`        | `True` if the agent completed its primary task               |
| `error_type`     | `str \| None` | One of: `llm`, `tool`, `logic`, `timeout`, `infra`. `None` on success |
| `error_message`  | `str \| None` | Human-readable error description                             |
| `partial_output` | `dict`        | Agent-specific results (full on success, partial on failure)  |

- `partial_output` always holds agent-specific data — callers check `success` first, then read `partial_output` regardless.
- The existing `parse_error` pattern in SprintPlannerAgent maps to `error_type: "llm"`.
- `SprintState.errors` accumulates `error_message` strings; `SprintState.iteration_counts` tracks per-agent retries.

### Iteration Caps

Per-task limits within the reflection loop. Set as class attributes (replacing the `self.max_iterations` placeholder in `BaseAgent.__init__`).

| Agent          | Max Iterations | Rationale                                          |
|----------------|----------------|----------------------------------------------------|
| SprintPlanner  | 2              | Near-one-shot. Retry once on JSON parse failure    |
| CoderAgent     | 5              | code → test → fix cycles need room for iteration   |
| TesterAgent    | 1              | Runs tests and reports. Failures are data, not errors |
| UpdaterAgent   | 2              | Create PR / update Notion. Retry once on transient failure |

### Recovery Matrix

Failure category × agent type → action:

| Category | SprintPlanner          | CoderAgent                        | TesterAgent           | UpdaterAgent                |
|----------|------------------------|-----------------------------------|-----------------------|-----------------------------|
| LLM      | retry w/ error context | retry w/ error context            | skip (report raw)     | retry once, then skip       |
| Tool     | skip tool, plan without| retry once, then escalate         | report as test result | retry once, then escalate   |
| Logic    | N/A                    | retry w/ test output (up to cap)  | report as-is          | escalate                    |
| Timeout  | abort sprint           | skip task, continue               | skip task, log        | skip task, continue         |
| Infra    | abort                  | abort                             | abort                 | abort                       |

Actions:
- **retry**: Re-invoke agent with error appended to context. Decrements remaining iterations.
- **skip**: Mark task as failed in `SprintState`, move to next. Log the error.
- **escalate**: Pause execution, dump context for human review.
- **abort**: Stop the entire sprint. Preserve all state for diagnosis.

### Skip-and-Continue Semantics

Tasks within a sprint are independent (each branches from `main`):

- If CoderAgent fails on SP8-003, downstream agents still run on SP8-001 and SP8-002.
- Failed tasks get `status: "failed"` in their checkpoint and are skipped by downstream agents.
- Sprint aborts when **>50% of tasks fail** (configurable threshold).
- **Exception**: SprintPlanner failure aborts the entire sprint (no plan = no tasks).

### Human Escalation Format

When escalating, print structured context to console:

```
=== ESCALATION: [agent_name] failed on [task_id] ===
Error type:  [error_type]
Message:     [error_message]
Iteration:   [current] / [max]
Task:        [task_id] - [task_title]

--- Last Agent Input (truncated to 2000 chars) ---
[the prompt/input that caused the failure]

--- Partial Output ---
[JSON dump of partial_output]

--- Suggested Actions ---
1. Review the error and fix manually
2. Re-run: python main.py run --resume [sprint_id] --task [task_id]
3. Skip:   python main.py run --resume [sprint_id] --skip [task_id]
```

Console output only for Phase 3. Structured file logging comes in Phase 6.

---

## Notes

- This is a "later feature" — the initial system will use simple retry + escalate.
- The branching strategy (fresh branch from main per task) already mitigates
  some compounding failure issues.
- Failure mode handling is closely related to prompt management (how error
  context is fed back to agents).

## Phase 3.5 — First End-to-End Validation Findings

> **Date**: 2026-03-31
> **Sprint tested**: Sprint 1.4 (s-1.4)
> **Mode**: Dry-run (no actual pushes, no real code changes)

### Pre-Run Bugs Found and Fixed (5)

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 1 | `--dry-run` never ran the cascade | `main.py` short-circuited with early return; hardcoded `dry_run=False` | Removed early return, pass flag through |
| 2 | TesterAgent crashes on Windows with no tests | `subprocess.run([])` raises OSError; `echo` is a shell builtin on Windows | Added skip mode: `TEST_COMMAND=skip` returns `test_passed: True` |
| 3 | `.env` missing `AIDER_REPO_DIR`, `TEST_COMMAND` | Only `NOTION_API_KEY` was configured | Added values |
| 4 | `plan_node` had no Notion context | `cascade.py` never loaded snapshot data | Added `_load_notion_context()` + `curate_context()` |
| 5 | Sprint matching only found `status=="Active"` | Sprint 1.4 was "Not started" | Extended to match Not started, In Progress |

### Runtime Issues Found (6)

| # | Issue | Severity | Resolution |
|---|-------|----------|------------|
| 1 | Planner generated synthetic tasks instead of using Notion items | High | Architectural: separated SprintPlanner (sprint planning) from cascade (Notion data loading). New TaskPlannerAgent handles per-task implementation planning. |
| 2 | Branch naming collision (`sprint-1/...` for all Phase 1 sprints) | Medium | Use CLI `sprint_id` (e.g. "1.4") instead of numeric `sprint_number` for branch naming |
| 3 | `pr_created: true` but `pr_url: null` in dry-run | Low | Added `dry_run` field to updater output for clarity |
| 4 | `notion_updated: false` on all tasks | Low | Resolved by using real Notion work items (issue #1 fix) |
| 5 | No console progress output during cascade | Low | Added `print()` per node + changed `print_summary` from `logger.info` to `print` |
| 6 | Aider not installed | Info | Required for live runs: `pip install aider-chat` |

### Architectural Insight: SprintPlanner ≠ TaskPlanner

The SprintPlannerAgent was incorrectly used in the cascade to generate tasks. In reality:
- **Sprint planning** = deciding WHAT to do (already done in Notion)
- **Task planning** = deciding HOW to implement a specific work item

The cascade now:
1. `plan_node` loads existing Notion tasks (no LLM)
2. `task_plan_node` runs TaskPlannerAgent per task (optional, non-fatal on failure)
3. CoderAgent receives enriched context from TaskPlanner

### Failure Scenarios Observed

1. **Missing config values**: Cascade ran but tools silently degraded. Need explicit validation of required config at startup.
2. **Sprint status mismatch**: "Not started" sprints were invisible to the planner. Need flexible matching by name/ID, not just status.
3. **Shell builtins on Windows**: `subprocess.run(["echo", ...])` fails silently. Any subprocess-based tool needs Windows compatibility testing.
4. **Dry-run output ambiguity**: `pr_created: true` when no PR exists is misleading. All dry-run outputs need a `dry_run` flag.
5. **Shared sprint_number across sub-sprints**: Phase 1 sprints 1.1-1.4 all have `sprint_number: 1`. Any identifier derived from this field will collide.

### Remaining Gaps for Phase 4

- Aider integration untested (not installed; dry-run mocked the tool)
- Azure DevOps PR creation untested (no config; tool binding skipped gracefully)
- Notion write-back untested with real IDs (dry-run used local snapshot)
- No structured logging of cascade execution (console print only)

## Next Steps

- [x] Brainstorm failure scenarios from first few live sprint runs
- [x] Define minimum viable error handling for Phase 3
- [ ] Design failure logging format for post-mortem analysis
- [x] Decide on iteration caps per agent type
