"""Entry point for the local AI agent system.

Usage:
    python main.py sync                              # Sync Notion databases
    python main.py sync --dry-run                    # Show what would sync
    python main.py run "Plan sprint 1.4"             # Run agent
    python main.py run "Plan sprint 1.4" --sync      # Sync first, then run
    python main.py run "Plan sprint 1.4" --dry-run   # Show what would happen

Future:
- Agent chaining: Planner -> Coder -> Tester -> Updater
- LangGraph workflow orchestration replaces sequential calls
- Tool auto-loading from tool_registry
"""

from __future__ import annotations

import argparse
import json
import sys

from config.settings import get_llm, get_settings, resolve_agent_class, resolve_tool_class


def _load_notion_context(settings) -> dict | None:
    """Try to load local Notion snapshot as agent context."""
    try:
        tool_cls = resolve_tool_class("notion", settings)
        tool = tool_cls(settings=settings)
        snapshot = tool.load_snapshot()
        if snapshot:
            return {
                "work_items": [wi.model_dump() for wi in snapshot.work_items],
                "sprints": [s.model_dump() for s in snapshot.sprints],
                "docs": [d.model_dump() for d in snapshot.docs],
                "decisions": [d.model_dump() for d in snapshot.decisions],
                "risks": [r.model_dump() for r in snapshot.risks],
            }
    except Exception as e:
        print(f"[Warning] Could not load Notion context: {e}")
    return None


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
        if args.dry_run:
            print(f"[Dry-run] Would run agent '{args.agent}' with prompt: {args.prompt}")
            return

    # Load context from local Notion data
    context = _load_notion_context(settings)

    if args.dry_run:
        print(f"[Dry-run] Would run agent '{args.agent}' with prompt: {args.prompt}")
        if context:
            total = sum(len(v) for v in context.values())
            print(f"[Dry-run] Context loaded: {total} items from local Notion data")
        else:
            print("[Dry-run] No local Notion data available — agent will generate tasks")
        return

    # Resolve and instantiate the agent
    agent_cls = resolve_agent_class(args.agent, settings)
    llm = get_llm(settings)
    agent = agent_cls(llm=llm, context=context)

    print(f"[Agent: {agent.name}] Running with model: {settings.ollama_model}")
    if context:
        total = sum(len(v) for v in context.values())
        print(f"[Agent: {agent.name}] Context: {total} items from local Notion data")
    print(f"[Agent: {agent.name}] Prompt: {args.prompt}")
    print("-" * 60)

    result = agent.run(args.prompt)
    print(json.dumps(result, indent=2))


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

    args = parser.parse_args()

    if args.command == "sync":
        cmd_sync(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
