"""Entry point for the local AI agent system.

Usage:
    python main.py "Plan sprint 8"
    python main.py --agent sprint_planner "Plan sprint 8"

Future:
- Agent chaining: Planner -> Coder -> Tester -> Updater
- LangGraph workflow orchestration replaces sequential calls
- Tool auto-loading from tool_registry
"""

from __future__ import annotations

import argparse
import json
import sys

from config.settings import get_llm, get_settings, resolve_agent_class


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a local AI agent task.",
        epilog="Example: python main.py \"Plan sprint 8\"",
    )
    parser.add_argument(
        "prompt",
        help="The task prompt to send to the agent.",
    )
    parser.add_argument(
        "--agent",
        default="sprint_planner",
        help="Agent to run (from agent_registry). Default: sprint_planner",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the Ollama model (e.g. qwen2.5-coder:7b, mistral:7b).",
    )
    args = parser.parse_args()

    settings = get_settings()

    # Allow model override from CLI
    if args.model:
        settings.ollama_model = args.model

    # Resolve and instantiate the agent
    agent_cls = resolve_agent_class(args.agent, settings)
    llm = get_llm(settings)
    agent = agent_cls(llm=llm)

    print(f"[Agent: {agent.name}] Running with model: {settings.ollama_model}")
    print(f"[Agent: {agent.name}] Prompt: {args.prompt}")
    print("-" * 60)

    # Run the agent
    result = agent.run(args.prompt)

    # Pretty-print JSON result
    print(json.dumps(result, indent=2))

    # Future: Agent chaining / LangGraph orchestration
    # workflow = StateGraph(...)
    # workflow.add_node("planner", planner_agent.as_langgraph_node())
    # workflow.add_node("coder", coder_agent.as_langgraph_node())
    # workflow.add_edge("planner", "coder")
    # result = workflow.compile().invoke({"prompt": args.prompt})


if __name__ == "__main__":
    main()
