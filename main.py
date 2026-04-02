"""Entry point for the local AI agent system.

Usage:
    python main.py sync                              # Sync Notion databases
    python main.py sync --dry-run                    # Show what would sync
    python main.py run "Plan sprint 1.4"             # Run agent
    python main.py run "Plan sprint 1.4" --sync      # Sync first, then run
    python main.py run "Plan sprint 1.4" --dry-run   # Show what would happen
    python main.py run "Plan sprint 1.4" --no-tools  # LLM planning only
    python main.py cascade "Deploy SHIR"             # Run full cascade
    python main.py cascade "Deploy SHIR" --dry-run   # Show what would happen
    python main.py cascade "Deploy SHIR" --max-tasks 2  # Limit tasks
    python main.py cascade --list                    # List past cascade runs
    python main.py cascade --show sprint-8           # Show saved state
    python main.py benchmark                         # Benchmark models
    python main.py benchmark --models a:7b,b:3b      # Specific models
    python main.py benchmark --runs 1 --dry-run      # Quick preview
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from config.settings import get_llm, get_settings, resolve_agent_class, resolve_tool_class


def _load_notion_context(settings) -> dict | None:
    """Try to load local Notion snapshot as agent context.

    Prefers local snapshot (with pending changes applied) over cloud snapshot.
    """
    # Try local snapshot first (includes pending changes from Phase 2c)
    try:
        write_tool_cls = resolve_tool_class("notion_write", settings)
        write_tool = write_tool_cls(settings=settings)
        snapshot = write_tool.load_local_snapshot()
        if snapshot:
            return _snapshot_to_context(snapshot)
    except Exception:
        pass

    # Fall back to cloud snapshot
    try:
        tool_cls = resolve_tool_class("notion", settings)
        tool = tool_cls(settings=settings)
        snapshot = tool.load_snapshot()
        if snapshot:
            return _snapshot_to_context(snapshot)
    except Exception as e:
        print(f"[Warning] Could not load Notion context: {e}")
    return None


def _snapshot_to_context(snapshot) -> dict:
    """Convert a NotionSnapshot to the context dict format agents expect."""
    return {
        "work_items": [wi.model_dump() for wi in snapshot.work_items],
        "sprints": [s.model_dump() for s in snapshot.sprints],
        "docs": [d.model_dump() for d in snapshot.docs],
        "decisions": [d.model_dump() for d in snapshot.decisions],
        "risks": [r.model_dump() for r in snapshot.risks],
    }


def _load_cascade_tasks(settings, sprint_id: str) -> tuple[list[dict] | None, dict | None]:
    """Load tasks for a cascade run from local Notion snapshot.

    Matches the CLI sprint_id to a sprint in local data, filters work items,
    and enriches task descriptions with page content from disk.

    Returns:
        (tasks, sprint) if sprint found, (None, None) otherwise.
    """
    context = _load_notion_context(settings)
    if not context:
        return None, None

    # Strip prefix to get version string: "s-1.4" -> "1.4", "sprint-8" -> "8"
    version = sprint_id
    for prefix in ("s-", "sprint-"):
        if version.startswith(prefix):
            version = version[len(prefix):]
            break

    # Match sprint by name or notion_id
    sprints = context.get("sprints", [])
    sprint = None
    for s in sprints:
        name = s.get("name", "")
        if f" {version}" in name or name.startswith(version):
            sprint = s
            break
    if sprint is None:
        for s in sprints:
            if s.get("notion_id") == sprint_id:
                sprint = s
                break
    if sprint is None:
        return None, None

    # Filter work items linked to this sprint
    sprint_notion_id = sprint["notion_id"]
    work_items = [
        wi for wi in context.get("work_items", [])
        if wi.get("sprint_id") == sprint_notion_id
    ]

    # Build task dicts
    content_dir = Path(settings.data_dir) / "notion" / "content"
    tasks = []
    for wi in work_items:
        notion_id = wi["notion_id"]
        description = wi.get("definition_of_done") or ""

        # Enrich description with page content from disk
        content_path = content_dir / f"{notion_id}.md"
        if content_path.exists():
            page_content = content_path.read_text(encoding="utf-8").strip()
            if page_content:
                description = f"{description}\n\n{page_content}" if description else page_content
        else:
            print(f"[Cascade] Warning: no page content for task {notion_id} ({wi.get('name', 'Untitled')})")

        tasks.append({
            "id": notion_id,
            "notion_id": notion_id,
            "title": wi.get("name", "Untitled"),
            "description": description,
            "status": wi.get("status", "Ready"),
            "priority": wi.get("priority", ""),
            "estimate_hrs": wi.get("estimate_hrs", 0),
            "type": wi.get("type", "Task"),
        })

    return tasks, sprint


def _get_agent_tools(agent) -> list[str]:
    """Get the list of tools an agent wants.

    Combines REQUIRED_TOOLS + OPTIONAL_TOOLS if the agent declares them.
    Falls back to empty list for agents that don't use tools.
    """
    tools: list[str] = []
    tools.extend(getattr(agent, "REQUIRED_TOOLS", []))
    tools.extend(getattr(agent, "OPTIONAL_TOOLS", []))
    return tools


def cmd_sync(args) -> None:
    """Handle the 'sync' subcommand."""
    settings = get_settings()

    tool_cls = resolve_tool_class("notion", settings)
    tool = tool_cls(settings=settings)

    if args.dry_run:
        print("[Sync] Dry run — connecting to Notion to check databases...")

    meta = tool.sync(dry_run=args.dry_run)

    action = "Would sync" if args.dry_run else "Synced"
    print(f"[Sync] {action} at {meta.synced_at}")
    for name, count in meta.counts.items():
        print(f"  {name}: {count} pages")

    if not args.dry_run:
        print(f"[Sync] Data written to {tool.data_dir}/")


def cmd_run(args) -> None:
    """Handle the 'run' subcommand."""
    settings = get_settings()

    # Allow model override from CLI (takes precedence over per-agent overrides)
    if args.model:
        settings.ollama_model = args.model
        settings.agent_model_overrides.pop(args.agent, None)

    # Optional pre-sync
    if args.sync:
        tool_cls = resolve_tool_class("notion", settings)
        tool = tool_cls(settings=settings)
        meta = tool.sync(dry_run=args.dry_run)
        action = "Would sync" if args.dry_run else "Synced"
        print(f"[Sync] {action}: {sum(meta.counts.values())} pages across {len(meta.counts)} databases")
        if args.dry_run:
            print(f"[Dry-run] Would run agent '{args.agent}' with prompt: {args.prompt}")
            return

    # Load context from local Notion data
    context = _load_notion_context(settings)

    if args.dry_run:
        print(f"[Dry-run] Would run agent '{args.agent}' with prompt: {args.prompt}")
        if context:
            entity_count = sum(
                len(v) for k, v in context.items()
                if k != "page_content" and isinstance(v, list)
            )
            page_count = len(context.get("page_content", {}))
            print(f"[Dry-run] Context loaded: {entity_count} items from local Notion data")
            if page_count:
                print(f"[Dry-run] Page content: {page_count} snippets loaded")
        else:
            print("[Dry-run] No local Notion data available — agent will generate tasks")
        return

    # Resolve agent class and curate context (filter entities, load page content)
    agent_cls = resolve_agent_class(args.agent, settings)
    content_dir = Path(settings.data_dir) / "notion" / "content"
    context = agent_cls.curate_context(context, content_dir=content_dir)

    llm = get_llm(settings, agent_name=args.agent)
    agent = agent_cls(llm=llm, context=context)

    # Auto-bind tools (unless --no-tools)
    if not args.no_tools:
        tool_names = _get_agent_tools(agent)
        if tool_names:
            bind_results = agent.bind_tools(tool_names, settings, dry_run=args.dry_run)
            for tool_name, success in bind_results.items():
                status = "bound" if success else "unavailable"
                print(f"[Tools] {tool_name}: {status}")

    print(f"[Agent: {agent.name}] Running with model: {settings.ollama_model}")
    if context:
        total = sum(len(v) for v in context.values())
        print(f"[Agent: {agent.name}] Context: {total} items from local Notion data")
    print(f"[Agent: {agent.name}] Prompt: {args.prompt}")
    print("-" * 60)

    result = agent.run(args.prompt)
    print(json.dumps(result, indent=2))


def cmd_eval(args) -> None:
    """Handle the 'eval' subcommand."""
    from evals.runner import EvalRunner, resolve_eval_class

    settings = get_settings()
    if args.model:
        settings.ollama_model = args.model

    # Resolve eval suite
    eval_cls = resolve_eval_class(args.agent)
    eval_suite = eval_cls()

    if args.dry_run:
        print(f"[Eval] Agent: {args.agent} — {len(eval_suite.get_cases())} cases")
        for case in eval_suite.get_cases():
            print(f"  - {case.name}: {case.description}")
        return

    # Build agent factory
    agent_cls = resolve_agent_class(args.agent, settings)
    llm = get_llm(settings, agent_name=args.agent)

    def agent_factory(context: dict | None):
        return agent_cls(llm=llm, context=context)

    runner = EvalRunner(eval_suite)
    results = runner.run_all(agent_factory)
    runner.print_report(results)


def cmd_benchmark(args) -> None:
    """Handle the 'benchmark' subcommand."""
    from evals.benchmark import DEFAULT_MODELS, BenchmarkRunner
    from evals.runner import resolve_eval_class

    models = args.models.split(",") if args.models else None

    if args.dry_run:
        model_list = models or DEFAULT_MODELS
        eval_cls = resolve_eval_class(args.agent)
        cases = eval_cls().get_cases()
        total = len(model_list) * args.runs * len(cases)
        print(f"[Benchmark] Agent: {args.agent}")
        print(f"[Benchmark] Models ({len(model_list)}):")
        for m in model_list:
            print(f"  - {m}")
        print(f"[Benchmark] Runs per model: {args.runs}")
        print(f"[Benchmark] Cases per run: {len(cases)}")
        print(f"[Benchmark] Total inferences: {total}")
        return

    runner = BenchmarkRunner(
        agent_name=args.agent,
        models=models,
        num_runs=args.runs,
    )
    benchmark = runner.run()
    runner.save_results(benchmark)
    print()
    runner.print_summary(benchmark)


def cmd_ingest(args) -> None:
    """Handle the 'ingest' subcommand."""
    settings = get_settings()
    content_dir = Path(settings.data_dir) / "notion" / "content"
    snapshot_dir = Path(settings.data_dir) / "notion"

    if not content_dir.exists():
        print("[Ingest] No content directory found. Run 'python main.py sync' first.")
        sys.exit(1)

    md_files = list(content_dir.glob("*.md"))
    non_empty = [f for f in md_files if f.stat().st_size > 0]

    if args.dry_run:
        print(f"[Ingest] Dry run — would ingest {len(non_empty)} documents "
              f"({len(md_files) - len(non_empty)} empty, skipped)")
        print(f"[Ingest] Content dir: {content_dir}")
        print(f"[Ingest] ChromaDB path: {settings.chroma_db_path}")
        print(f"[Ingest] Embedding model: {settings.embedding_model}")
        print(f"[Ingest] Chunk threshold: {settings.rag_chunk_size} chars")
        if args.force:
            print("[Ingest] --force: would delete and recreate collection")
        return

    from rag.ingest import ingest_notion_content

    result = ingest_notion_content(
        settings=settings,
        content_dir=content_dir,
        snapshot_dir=snapshot_dir,
        force=args.force,
    )
    print(f"[Ingest] Done: {result['documents_ingested']} documents, "
          f"{result['chunks_created']} chunks")
    print(f"[Ingest] Collection: {result['collection_name']}")
    print(f"[Ingest] ChromaDB path: {settings.chroma_db_path}")


def cmd_cascade(args) -> None:
    """Handle the 'cascade' subcommand."""
    from datetime import datetime

    from orchestration import CascadeRunner

    settings = get_settings()
    cascade_dir = Path(settings.data_dir) / "cascade"

    # Handle --list: show saved cascade runs
    if args.list:
        if not cascade_dir.exists():
            print("[Cascade] No saved runs.")
            return
        files = sorted(cascade_dir.glob("*.json"))
        if not files:
            print("[Cascade] No saved runs.")
            return
        for f in files:
            try:
                data = json.loads(f.read_text())
                status = data.get("status", "unknown")
                task_count = len(data.get("tasks", []))
                print(f"  {f.stem}  status={status}  tasks={task_count}")
            except Exception:
                print(f"  {f.stem}  (unreadable)")
        return

    # Handle --show: display a specific saved state
    if args.show:
        state_path = cascade_dir / f"{args.show}.json"
        if not state_path.exists():
            print(f"[Cascade] No saved state for '{args.show}'")
            sys.exit(1)
        data = json.loads(state_path.read_text())
        print(json.dumps(data, indent=2))
        return

    # Require prompt for actual cascade runs
    if not args.prompt:
        print("[Cascade] Error: prompt is required (or use --list/--show)")
        sys.exit(1)

    # Allow model override from CLI
    if args.model:
        settings.ollama_model = args.model

    # Optional pre-sync
    if args.sync:
        tool_cls = resolve_tool_class("notion", settings)
        tool = tool_cls(settings=settings)
        meta = tool.sync(dry_run=args.dry_run)
        action = "Would sync" if args.dry_run else "Synced"
        print(f"[Sync] {action}: {sum(meta.counts.values())} pages across {len(meta.counts)} databases")

    # Derive sprint_id
    sprint_id = args.sprint_id or f"sprint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    # Load tasks from local Notion data
    tasks, sprint = _load_cascade_tasks(settings, sprint_id)
    if tasks is None:
        # Try to list available sprints for a helpful error
        context = _load_notion_context(settings)
        available = []
        if context:
            available = [s.get("name", "?") for s in context.get("sprints", [])]
        print(f"[Cascade] Sprint '{sprint_id}' not found in local Notion data.")
        print("[Cascade] Run 'python main.py sync' first, then check the sprint ID.")
        if available:
            print(f"[Cascade] Available sprints: {available}")
        sys.exit(1)

    if not tasks:
        sprint_name = sprint.get("name", sprint_id) if sprint else sprint_id
        print(f"[Cascade] Sprint '{sprint_name}' found but has 0 work items assigned.")
        print("[Cascade] Check Notion: work items may not be linked to this sprint yet.")
        print("[Cascade] Then run 'python main.py sync' to refresh local data.")
        sys.exit(1)

    # Combine sprint goal with CLI prompt
    sprint_goal = sprint.get("goal", "") if sprint else ""
    goal = args.prompt
    if sprint_goal and sprint_goal not in goal:
        goal = f"{goal}\n\nSprint goal: {sprint_goal}"

    runner = CascadeRunner(settings, dry_run=args.dry_run)
    max_tasks = args.max_tasks or 0
    final_state = runner.run(
        sprint_id=sprint_id,
        goal=goal,
        abort_threshold=args.abort_threshold,
        max_tasks=max_tasks,
        tasks=tasks,
    )

    # Save final state
    cascade_dir.mkdir(parents=True, exist_ok=True)
    state_path = cascade_dir / f"{sprint_id}.json"
    with open(state_path, "w") as f:
        json.dump(final_state, f, indent=2, default=str)
    print(f"[Cascade] State saved to {state_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local AI agent system for sprint automation.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # -- sync subcommand --
    sync_parser = subparsers.add_parser(
        "sync",
        help="Sync Notion databases to local storage.",
    )
    sync_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without writing files.",
    )

    # -- run subcommand --
    run_parser = subparsers.add_parser(
        "run",
        help="Run an agent with a prompt.",
    )
    run_parser.add_argument(
        "prompt",
        help="The task prompt to send to the agent.",
    )
    run_parser.add_argument(
        "--agent",
        default="sprint_planner",
        help="Agent to run (from agent_registry). Default: sprint_planner",
    )
    run_parser.add_argument(
        "--model",
        default=None,
        help="Override the Ollama model (e.g. qwen2.5-coder:7b, mistral:7b).",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned actions without executing.",
    )
    run_parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync Notion databases before running the agent.",
    )
    run_parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Run agent without binding tools (LLM planning only).",
    )

    # -- eval subcommand --
    eval_parser = subparsers.add_parser(
        "eval",
        help="Run agent evaluation suite.",
    )
    eval_parser.add_argument(
        "--agent",
        default="sprint_planner",
        help="Agent to evaluate (must have an eval suite). Default: sprint_planner",
    )
    eval_parser.add_argument(
        "--model",
        default=None,
        help="Override the Ollama model for benchmarking.",
    )
    eval_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show eval cases without running them.",
    )

    # -- ingest subcommand --
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Embed Notion content into ChromaDB for RAG.",
    )
    ingest_parser.add_argument(
        "--force",
        action="store_true",
        help="Re-ingest from scratch (delete existing collection).",
    )
    ingest_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be ingested without embedding.",
    )

    # -- cascade subcommand --
    cascade_parser = subparsers.add_parser(
        "cascade",
        help="Run the full agent cascade (Planner → Coder → Tester → Updater).",
    )
    cascade_parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help="Sprint goal / planning instruction.",
    )
    cascade_parser.add_argument(
        "--sprint-id",
        default=None,
        help="Explicit sprint ID (default: sprint-YYYYMMDD-HHMMSS).",
    )
    cascade_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing.",
    )
    cascade_parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Limit number of tasks processed (safety valve).",
    )
    cascade_parser.add_argument(
        "--abort-threshold",
        type=float,
        default=0.5,
        help="Fraction of tasks that can fail before aborting (default: 0.5).",
    )
    cascade_parser.add_argument(
        "--sync",
        action="store_true",
        help="Sync Notion databases before running.",
    )
    cascade_parser.add_argument(
        "--model",
        default=None,
        help="Override the Ollama model.",
    )
    cascade_parser.add_argument(
        "--list",
        action="store_true",
        help="List saved cascade runs.",
    )
    cascade_parser.add_argument(
        "--show",
        default=None,
        help="Show saved state for a specific sprint ID.",
    )

    # -- benchmark subcommand --
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Benchmark multiple models against the eval suite.",
    )
    bench_parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of Ollama model names to benchmark.",
    )
    bench_parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per model (default: 3).",
    )
    bench_parser.add_argument(
        "--agent",
        default="sprint_planner",
        help="Agent to evaluate. Default: sprint_planner",
    )
    bench_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show benchmark plan without running.",
    )

    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "run":
        cmd_run(args)
    elif args.command == "ingest":
        cmd_ingest(args)
    elif args.command == "cascade":
        cmd_cascade(args)
    elif args.command == "eval":
        cmd_eval(args)
    elif args.command == "benchmark":
        cmd_benchmark(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
