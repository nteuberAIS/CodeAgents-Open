# Deferred Decision: Failure Modes & Error Recovery

> **Status**: Deferred — needs further brainstorming before implementation.
> **Relevant Phase**: Phase 3+ (Multi-Agent Cascade)
> **Last Updated**: 2026-03-25

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

## Notes

- This is a "later feature" — the initial system will use simple retry + escalate.
- The branching strategy (fresh branch from main per task) already mitigates
  some compounding failure issues.
- Failure mode handling is closely related to prompt management (how error
  context is fed back to agents).

## Next Steps

- [ ] Brainstorm failure scenarios from first few live sprint runs
- [ ] Define minimum viable error handling for Phase 3
- [ ] Design failure logging format for post-mortem analysis
- [ ] Decide on iteration caps per agent type
