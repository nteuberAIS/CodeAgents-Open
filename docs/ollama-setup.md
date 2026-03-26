# Ollama Setup Guide

## Installation

### Windows
Download from https://ollama.com/download and run the installer.

### Linux
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### macOS
```bash
brew install ollama
```

## Start the Server

```bash
ollama serve
```

The server runs on `http://localhost:11434` by default.

## Pull the Default Model

```bash
ollama pull qwen2.5-coder:7b
```

Verify it's available:
```bash
ollama list
```

## Model Recommendations by VRAM

| VRAM | Model | Command | Notes |
|------|-------|---------|-------|
| 8 GB | Qwen 2.5 Coder 7B | `ollama pull qwen2.5-coder:7b` | Best coding benchmarks at 7B |
| 8 GB | DeepSeek-Coder-V2 Lite | `ollama pull deepseek-coder-v2:lite` | 16B MoE, strong coding |
| 16 GB | Qwen 2.5 Coder 14B | `ollama pull qwen2.5-coder:14b` | Significant quality jump |
| 24 GB | Qwen 2.5 Coder 32B | `ollama pull qwen2.5-coder:32b` | Competitive with GPT-4o |

## Override the Model

Set an environment variable or pass `--model` on the CLI:

```bash
# Environment variable
export OLLAMA_MODEL=mistral:7b
python main.py "Plan sprint 8"

# CLI flag
python main.py --model mistral:7b "Plan sprint 8"
```

## Tips

- Close Chrome or disable GPU acceleration to free VRAM when running models.
- Use `q4_k_m` quantization for the best quality/speed/memory balance.
- If inference is very slow, the model may be spilling to system RAM. Try a
  smaller model or lower quantization.
