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

- [x] Brainstorm failure scenarios from first few live sprint runs
- [x] Define minimum viable error handling for Phase 3
- [ ] Design failure logging format for post-mortem analysis
- [x] Decide on iteration caps per agent type

---

## Phase 3.5 — First Live Run Findings (2026-04-01)

First-ever live cascade run against Sprint 1.4 (SynDataPlatform IaC repo on Azure DevOps).
Target task: #17 — Fix SHIR NSG table stale entries (documentation/Bicep fix).
Four live runs executed with incremental fixes between each.

### Pre-run issues

1. **Aider not installable in project venv** — Python 3.14 in venv, aider-chat requires <3.13.
   All versions above 0.16.0 rejected by pip. Fixed by installing via `uv tool install aider-chat --python 3.12`.

2. **Azure CLI not installed** — `az` not present on system. Installed via `winget install Microsoft.AzureCLI`.

3. **`TEST_COMMAND=skip` not handled** — TesterAgent treated `"skip"` as a literal command to execute
   via `subprocess.run(["skip"], ...)`, causing `[WinError 2] FileNotFoundError`. Every task failed
   at the test node, bypassing update_node entirely. Zero PRs could ever be created.
   **Fix applied**: Added skip sentinel check in `agents/tester.py` before the subprocess call.

4. **Sprint target branch didn't exist** — PRs target `sprint-1.4` but that branch didn't exist on the
   Azure DevOps remote. Created manually before the live run.

### Runtime issues

5. **`az.cmd` not found by subprocess (Windows)** — `subprocess.run(["az", ...])` raises
   `FileNotFoundError` on Windows because the az CLI binary is `az.cmd`, not a plain executable.
   `_validate_cli()` in `AzDevOpsTool` called `self._run_command(["az", "account", "show"])` which
   failed silently (caught by `except Exception` in `setup_task_node`, logged at DEBUG).
   This was invisible during dry runs because `_validate_cli` returns early when `dry_run=True`.
   **Fix applied**: `AzDevOpsTool._run_command()` now resolves the full az binary path via
   `shutil.which("az")` and passes it directly, avoiding shell=True entirely.

6. **`shell=True` + `&` in PR title = shell injection** — Initial fix used `shell=True` with
   `" ".join(cmd)`. PR title "Fix SHIR NSG table stale entries in Network Egress & PE Rules doc"
   contained `&`, which `cmd.exe` interpreted as a command separator. Error:
   `'PE' is not recognized as an internal or external command`.
   **Fix applied**: Switched to `shutil.which()` approach (no shell=True). Added `_sanitize_for_cmd()`
   to strip cmd.exe metacharacters from PR titles/descriptions as defense-in-depth, since `.cmd`
   files still invoke `cmd.exe` internally.

7. **git_tool.py had no `cwd` — all git ops ran in wrong repo** — `BaseGitTool._run_command()` called
   `subprocess.run(cmd, ...)` without `cwd`. All git operations (branch creation, checkout) ran in
   `CodeAgents-Open` instead of `SynDataPlatform`. The task branch `sprint-1.4/31da5e21-...` was
   created in the agent repo, not the target repo. Aider still ran on `main` in SynDataPlatform.
   **Fix applied**: Added `self.repo_dir` to `BaseGitTool.__init__` (from `settings.aider_repo_dir`),
   passed `cwd=self.repo_dir` to all subprocess calls in both `BaseGitTool` and `AzDevOpsTool`.

8. **Git ref naming collision** — The cascade creates task branches as `sprint-1.4/{task-id}` and PRs
   target `sprint-1.4`. But if a local branch named `sprint-1.4` exists, git refuses to create
   `sprint-1.4/{task-id}` because refs can't be both a file and a directory in `.git/refs/heads/`.
   Error: `fatal: cannot lock ref 'refs/heads/sprint-1.4/test-branch': 'refs/heads/sprint-1.4' exists`.
   The cascade caught this silently and proceeded on the current branch (usually `main`).
   **Workaround**: Delete the local `sprint-1.4` branch before running; the remote ref suffices for PRs.
   **Needs design fix**: Change naming convention to avoid collision (e.g., `task/sprint-1.4/{task-id}`).

9. **No commit or push between code_node and update_node** — `aider_tool.py` uses `--no-auto-commits`
   so Aider writes files but does not commit. The cascade's `update_node` creates a PR via
   `az repos pr create` targeting `sprint-1.4/{task-id}` → `sprint-1.4`, but the source branch was
   never committed to or pushed to the remote. PR creation fails with:
   `TF401398: source and/or the target branch no longer exists, or the requested refs are not branches`.
   `git_tool.py` has `commit()` and `push()` methods but neither is called by the cascade.
   **Not yet fixed** — architectural gap requiring a new cascade node or additions to `code_node`/`update_node`.

10. **Aider opens browser tab on every run** — Warning about `OLLAMA_API_BASE` not being set causes
    Aider to open `https://aider.chat/docs/llms/warnings.html` in the default browser. Setting
    `OLLAMA_API_BASE` in `.env` breaks Pydantic Settings validation (`extra = "forbid"`).
    **Needs fix**: Pass `--no-show-model-warnings` flag in `aider_tool.py`, or set `OLLAMA_API_BASE`
    as a subprocess-only env var in the Aider invocation.

11. **Aider modifies `.gitignore` on target repo** — Aider adds `.aider*` to `.gitignore` on every run.
    Combined with `--no-auto-commits`, this leaves uncommitted noise in the working tree.
    **Needs fix**: Pass `--no-gitignore` flag in `aider_tool.py`, or add `.aider*` to the repo's
    `.gitignore` permanently.

12. **LLM instruction quality inconsistent** — CoderAgent's LLM generated instructions telling Aider
    to "Open the Notion doc..." instead of identifying repo files to edit. Aider sometimes interpreted
    this correctly (found `infra/modules/nsg-shir.bicep` on one run) and sometimes didn't (no changes
    on another run). `modified_files: []` in the result even when Aider did modify files, because
    the output was truncated before the "Wrote path" lines appeared.
    **Needs fix**: Improve coder system prompt to emphasize that instructions are for Aider editing
    local files, not Notion documents. Increase `aider_output` truncation limit.

### What worked

- **Ollama + qwen2.5-coder:7b** — LLM responded reliably, no timeouts or VRAM issues
- **TesterAgent skip mode** — After fix, cleanly skipped tests with `success=True`
- **Notion write-back** — UpdaterAgent successfully updated task status to "In Progress" in Notion
- **Cascade state persistence** — `data/cascade/s-1.4.json` saved correctly after each run
- **Abort threshold logic** — Not triggered (only 1 task per run), but routing worked correctly
- **Dry-run mode** — Correctly skipped all write operations and CLI validation
- **Aider invocation** — Aider CLI launched correctly, connected to Ollama, scanned repo,
  and generated edits (when instructions were clear enough)
- **Error accumulation** — Errors recorded in cascade state, printed in summary
- **LangGraph routing** — All nodes executed in correct order, conditional routing worked

### Architectural gaps

1. **Missing commit+push step** (Critical) — The cascade has no mechanism to commit Aider's file
   changes and push the branch to the remote before PR creation. This is the primary blocker for
   end-to-end automation. Options:
   - Add a `commit_and_push_node` between `code_node` and `test_node`
   - Add commit+push logic to the end of `code_node`
   - Add push logic to the beginning of `update_node`

2. **Branch naming collision** (High) — The convention of `sprint-{N}` (target) and `sprint-{N}/{task-id}`
   (source) creates git ref conflicts when a local branch named `sprint-{N}` exists. The cascade
   doesn't detect or recover from this — it silently falls through and Aider runs on `main`.

3. **No cwd in git operations** (Fixed) — Was an architectural gap; all git commands ran in the process
   cwd rather than the target repo.

4. **Windows CLI compatibility** (Partially fixed) — The system was designed for Unix-like environments.
   Windows-specific issues with `.cmd` files, `cmd.exe` metacharacter parsing, and `shell=True` vs
   `shell=False` tradeoffs needed multiple fixes.

5. **Task completion false positives** (Medium) — UpdaterAgent returns `success=True` if Notion update
   succeeds, even when PR creation failed and no code changes were made. The cascade marks the task
   "completed" with 0 files changed and no PR. The `check_node` doesn't validate whether meaningful
   work was actually done.

6. **Aider subprocess environment** (Low) — Aider warnings about `OLLAMA_API_BASE` can't be suppressed
   via `.env` because Pydantic Settings rejects unknown env vars. The env var needs to be set only
   in the Aider subprocess environment, not globally.

### Recommendations (ordered by priority)

1. **Add commit+push step to cascade** — ✅ **RESOLVED** (Phase 3.5): `commit_push_node` added
   between `code_node` and `test_node`. Commits Aider's modified files and pushes to remote.

2. **Fix branch naming convention** — ✅ **RESOLVED** (Phase 3.5): Changed to
   `task/sprint-{N}/{task-id}` to avoid ref collision with the sprint target branch `sprint-{N}`.

3. **Improve CoderAgent prompt** — ⏸️ **DEFERRED**: Skipped intentionally. Root cause was task
   confusion (Notion page content vs repo files), not a prompt issue. Revisit post-RAG when
   context curation improves.

4. **Add Aider CLI flags** — ✅ **RESOLVED** (Phase 3.5): Added `--no-show-model-warnings`,
   `--no-gitignore`, `--no-detect-urls`. Also added `--edit-format udiff` via new
   `aider_edit_format` setting.

5. **Set `OLLAMA_API_BASE` in Aider subprocess env** — ❌ **DROPPED**: Rec 4 flags suppress the
   model warning that prompted this workaround, making the env var unnecessary.

6. **Add completion validation to check_node** — ✅ **RESOLVED** (Phase 3.5): `check_node` now
   warns when no files were changed or no PR was created.

7. **Increase Aider output capture limit** — ✅ **RESOLVED** (Phase 3.5): Increased from 2000 to
   4000 chars to preserve "Wrote path/to/file" lines.

8. **Add structured logging for cascade runs** — 🔲 **OPEN**: Deferred to Phase 6 (Production
   Hardening). Currently using `print()` for progress and `logger.debug()` for errors.
