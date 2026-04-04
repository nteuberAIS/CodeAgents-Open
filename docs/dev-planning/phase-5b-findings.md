# Phase 5b: Live Sprint Validation Findings

**Date:** 2026-04-03
**Sprint:** Sprint 1.4 — Monitoring, Scheduling & Validation (17 tasks)
**Scope:** 3-task validation runs (--max-tasks 3, --abort-threshold 0.7)
**Runs:** 4 total (3 diagnostic, 1 successful)

---

## Summary

First sprint-scale cascade run against Sprint 1.4. Discovered and fixed 5 bugs
across 4 files before achieving a successful end-to-end run. The code task
(ADLS capacity alert) completed in 1 iteration with a PR created on Azure
DevOps. Non-code tasks (doc update, process audit) failed gracefully as expected.

## Bugs Found and Fixed

### Bug 1: TERM environment breaks Aider on Windows
- **File:** `tools/aider_tool.py`
- **Symptom:** Aider crashed immediately: `"Can't initialize prompt toolkit: Found xterm-256color, while expecting a Windows console"`
- **Root cause:** Claude Code's bash shell sets `TERM=xterm-256color`, which breaks Aider's prompt_toolkit on Windows.
- **Fix:** Override `TERM=dumb` and `PYTHONIOENCODING=utf-8` in the subprocess environment.
- **Test:** `TestAiderToolEnvironment::test_edit_sets_term_to_dumb`

### Bug 2: CoderAgent false success on empty Aider results
- **File:** `agents/coder.py`
- **Symptom:** CoderAgent reported `success=True` when Aider exited 0 but modified no files, causing empty PRs.
- **Root cause:** CoderAgent only checked `result.success` (exit code), not `result.modified_files`.
- **Fix:** Treat `success=True` with `modified_files=[]` as a failure and retry. Dry-run is exempted.
- **Tests:** `test_aider_success_no_modified_files_retries`, `test_aider_success_no_modified_files_dry_run_still_succeeds`

### Bug 3: PRs created for empty branches (resolved by Bug 2)
- **Symptom:** 3 PRs created on Azure DevOps with no file changes.
- **Root cause:** Upstream false success from Bug 2 — cascade continued through commit_push and update nodes.
- **Fix:** No separate fix needed. Bug 2 fix prevents the cascade from reaching update_node when no files are modified.

### Bug 4: Code failure routes cascade to wrong task
- **File:** `orchestration/cascade.py`, `schemas/sprint_state.py`
- **Symptom:** When code_node failed on task N, commit_push_node operated on task N+1 instead. Task N+1 then got processed twice.
- **Root cause:** `get_current_task()` skips failed tasks. After code_node adds task N to `failed_task_ids`, subsequent nodes call `get_current_task()` which returns task N+1 (the next non-failed task). The unconditional edge `code_node -> commit_push_node` meant commit_push always ran, even after code failure.
- **Fix:** (a) Added `route_after_code()` conditional edge that skips directly to `check_node` when the current task is in `failed_task_ids`. (b) Changed `check_node` to use `tasks[current_task_index]` directly instead of `get_current_task()`, preventing it from validating the wrong task.
- **Test:** Existing `test_single_task_failure_skip_continue` now passes correctly.

### Bug 5: Modified files parser misses diff format output
- **File:** `tools/aider_tool.py`
- **Symptom:** Aider successfully created files but `modified_files` was always `[]` with `--edit-format diff`.
- **Root cause:** `_parse_modified_files()` only matched `"Wrote ..."` (whole format) and `"+++ b/..."` (udiff format). The `diff` edit format uses `"Applied edit to ..."` which wasn't matched.
- **Fix:** Added Pattern 3: `r"^Applied edit to (.+)$"` to the regex list.
- **Tests:** `test_parses_applied_edit_lines`, `test_mixed_applied_and_wrote`

## Run Results

| Run | Bugs Present | Task 1 (Code) | Task 2 (Doc) | Task 3 (Process) | Status |
|-----|-------------|---------------|--------------|-------------------|--------|
| 1 | 1,2,3 | False success | False success | False success | completed (3/3 "ok") |
| 2 | 2,3,4 | Failed (5 retries) | Wrong-task routing | Wrong-task routing | aborted |
| 3 | 5 | Failed (no files detected) | Failed (expected) | Failed (expected) | aborted (3/3 fail) |
| 4 | None | **Success** (1 iter, PR #185) | Failed (expected) | Failed (expected) | **completed** (1/3 ok) |

## Performance Observations

- **Task 1 (code task):** ~1 min for LLM + Aider (1 iteration)
- **Task 2 (doc task, fail):** ~12 min (5 iterations x ~2.5 min each, includes Aider timeouts)
- **Task 3 (process task, fail):** ~3 min (5 iterations, Aider returns quickly with no changes)
- **Total Run 4 time:** ~17 min for 3 tasks
- **Estimated full sprint (17 tasks):** 30-90 min depending on code vs non-code ratio

## Known Limitations

1. **Non-code tasks exhaust all 5 retries** — CoderAgent doesn't detect that a task is non-automatable. Each retry wastes ~1-5 min of LLM + Aider time. Future improvement: add task-type classification to skip non-code tasks early.

2. **Local main must match remote** — The cascade creates branches from local main. If local main is behind remote, files referenced in task descriptions may not exist for Aider to edit. Prerequisite: `git pull origin main` before running.

3. **Aider udiff format unreliable for new files** — The `diff` format (SEARCH/REPLACE) works but produces 0-byte files when the model hallucinates SEARCH content. The `whole` format is more reliable for file creation but less efficient for edits.

4. **LLM returns empty file lists for non-code tasks** — The LLM correctly identifies tasks as non-code ("This task is complete. No further action required.") but still returns valid JSON with `files=[]`, causing Aider to run with no target files.

## Files Changed

| File | Changes |
|------|---------|
| `tools/aider_tool.py` | TERM=dumb env override, PYTHONIOENCODING=utf-8, "Applied edit to" parser pattern |
| `agents/coder.py` | Empty modified_files check, logging of Aider iterations, aider_output in failure partial_output |
| `orchestration/cascade.py` | `route_after_code()` conditional edge, `check_node` uses direct index instead of `get_current_task()` |

## Test Impact

- **Before:** 687 tests passing
- **After:** 692 tests passing (+5 new regression tests)
- **No regressions**

## Recommendations for Phase 5c (Notion Cloud Push)

1. **Add status filtering** to `_load_cascade_tasks()` — skip tasks with status "Done", "In Review", "Cancelled" to avoid re-processing completed work.
2. **Add task-type classification** — detect non-code tasks (audit, docs, process) and skip them with a clear message instead of exhausting retries.
3. **Ensure local main is up-to-date** before running — add a pre-flight check or document as a required step.
4. **Consider `--edit-format whole`** for new file creation tasks vs `diff` for existing file edits.
5. **Cloud push prerequisites** — the pending_changes.json accumulation model works correctly. Phase 5c should focus on the sync-to-cloud workflow with human approval gates.
