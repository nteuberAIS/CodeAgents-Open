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

## Next Steps

- [ ] Brainstorm failure scenarios from first few live sprint runs
- [x] Define minimum viable error handling for Phase 3
- [ ] Design failure logging format for post-mortem analysis
- [x] Decide on iteration caps per agent type
