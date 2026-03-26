# CodeAgents-Open

A local, zero-cost AI agent system powered by Ollama. Modular foundation for
multi-agent workflows — sprint planning, code generation, testing, and project
management, all running offline.

## Quick Start

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com/) installed and running (`ollama serve`)

### 1. Pull the model

```bash
ollama pull qwen2.5-coder:7b
```

### 2. Set up the project

```bash
cd CodeAgents-Open
python -m venv .venv

# Windows (Git Bash)
source .venv/Scripts/activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run

```bash
python main.py "Plan sprint 8"
```

Options:
```bash
python main.py --agent sprint_planner "Plan sprint 8"   # explicit agent
python main.py --model mistral:7b "Plan sprint 8"       # override model
```

## Project Structure

```
agents/              Agent definitions (BaseAgent ABC + implementations)
tools/               External tool wrappers (git, Notion, Aider — Phase 2+)
config/              Settings, LLM factory, agent/tool registries
docs/                Architecture, Ollama setup guide, roadmap
main.py              CLI entry point
```

## Documentation

- [Architecture](docs/architecture.md) — system design, extension points, future cascade diagram
- [Ollama Setup](docs/ollama-setup.md) — install guide, model recommendations by VRAM tier
- [Roadmap](docs/roadmap.md) — phased plan from foundation to production

## Adding a New Agent

1. Create `agents/my_agent.py` with a class inheriting `BaseAgent`
2. Implement `run(user_input: str) -> dict`
3. Register in `config/settings.py` → `agent_registry`
4. Run: `python main.py --agent my_agent "your prompt"`
