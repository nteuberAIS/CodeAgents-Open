"""Model benchmarking runner.

Runs the eval suite against multiple Ollama models, captures timing,
token stats, and VRAM usage, then produces structured results and
a markdown comparison report.

Usage via CLI:
    python main.py benchmark
    python main.py benchmark --models qwen2.5-coder:7b,qwen3:8b --runs 1
    python main.py benchmark --dry-run
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_ollama import ChatOllama

from config.settings import resolve_agent_class
from evals.runner import resolve_eval_class
from schemas.benchmark import (
    BenchmarkRun,
    CaseRunResult,
    HardwareInfo,
    ModelResult,
    SingleRunResult,
    TokenStats,
)

logger = logging.getLogger(__name__)

DEFAULT_MODELS = [
    "qwen2.5-coder:1.5b",
    "qwen2.5-coder:3b",
    "qwen3:4b",
    "qwen2.5-coder:7b",
    "qwen3:8b",
    "qwen3:14b",
    "deepseek-coder-v2:16b",
]

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin HTTP client for the Ollama REST API (stdlib only)."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def _get(self, path: str) -> Any:
        url = f"{self.base_url}{path}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read())

    def _post(self, path: str, body: dict) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())

    def is_running(self) -> bool:
        try:
            self._get("/api/tags")
            return True
        except (urllib.error.URLError, OSError):
            return False

    def list_models(self) -> list[str]:
        data = self._get("/api/tags")
        return [m["name"] for m in data.get("models", [])]

    def pull_model(self, model_name: str) -> None:
        print(f"  Pulling {model_name} ...")
        self._post("/api/pull", {"name": model_name, "stream": False})
        print(f"  {model_name} ready.")

    def get_model_info(self, model_name: str) -> dict:
        """Return parameter_size and quantization_level for a model."""
        try:
            data = self._post("/api/show", {"name": model_name})
            details = data.get("details", {})
            return {
                "parameter_size": details.get("parameter_size"),
                "quantization_level": details.get("quantization_level"),
            }
        except Exception:
            return {"parameter_size": None, "quantization_level": None}


class TokenTracker(BaseCallbackHandler):
    """LangChain callback that captures Ollama token statistics."""

    def __init__(self) -> None:
        super().__init__()
        self._prompt_tokens: int | None = None
        self._completion_tokens: int | None = None
        self._eval_duration_ns: int | None = None

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        try:
            gen_info = response.generations[0][0].generation_info or {}
            self._prompt_tokens = gen_info.get("prompt_eval_count")
            self._completion_tokens = gen_info.get("eval_count")
            self._eval_duration_ns = gen_info.get("eval_duration")
        except (IndexError, AttributeError):
            pass

    def to_token_stats(self) -> TokenStats:
        tps = None
        if self._completion_tokens and self._eval_duration_ns:
            duration_s = self._eval_duration_ns / 1e9
            if duration_s > 0:
                tps = self._completion_tokens / duration_s
        return TokenStats(
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
            tokens_per_second=round(tps, 2) if tps else None,
        )

    def reset(self) -> None:
        self._prompt_tokens = None
        self._completion_tokens = None
        self._eval_duration_ns = None


class VRAMMonitor:
    """Context manager that polls nvidia-smi for peak VRAM usage."""

    def __init__(self, poll_interval: float = 0.5) -> None:
        self.poll_interval = poll_interval
        self.peak_mb: float | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().splitlines():
                        val = float(line.strip())
                        if self.peak_mb is None or val > self.peak_mb:
                            self.peak_mb = val
            except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
                # No GPU or nvidia-smi not available — stop polling
                return
            self._stop.wait(self.poll_interval)

    def __enter__(self) -> VRAMMonitor:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def collect_hardware_info() -> HardwareInfo:
    """Gather hardware info from nvidia-smi and platform."""
    gpu_name = None
    gpu_vram_mb = None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 2:
                gpu_name = parts[0].strip()
                gpu_vram_mb = int(float(parts[1].strip()))
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return HardwareInfo(
        gpu_name=gpu_name,
        gpu_vram_mb=gpu_vram_mb,
        cpu=platform.processor() or None,
        os_version=platform.platform(),
    )


# ---------------------------------------------------------------------------
# Benchmark Runner
# ---------------------------------------------------------------------------


class BenchmarkRunner:
    """Orchestrates model benchmarking against an eval suite."""

    def __init__(
        self,
        agent_name: str = "sprint_planner",
        models: list[str] | None = None,
        num_runs: int = 3,
        base_url: str = "http://localhost:11434",
    ) -> None:
        self.agent_name = agent_name
        self.models = models or list(DEFAULT_MODELS)
        self.num_runs = num_runs
        self.base_url = base_url
        self.client = OllamaClient(base_url)

    def preflight_check(self) -> None:
        """Verify Ollama is running and pull any missing models."""
        if not self.client.is_running():
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve"
            )
        available = self.client.list_models()
        # Normalize names for comparison (Ollama may include :latest)
        available_set = set()
        for name in available:
            available_set.add(name)
            # "qwen3:8b" should match "qwen3:8b" in list
            if ":" in name:
                available_set.add(name.split(":")[0])

        for model in self.models:
            # Check both exact name and base name match
            if model not in available_set:
                self.client.pull_model(model)

    def run(self) -> BenchmarkRun:
        """Run the full benchmark: all models x num_runs."""
        self.preflight_check()
        hardware = collect_hardware_info()

        eval_cls = resolve_eval_class(self.agent_name)
        eval_suite = eval_cls()
        agent_cls = resolve_agent_class(self.agent_name)

        model_results: list[ModelResult] = []

        for model_name in self.models:
            print(f"\n{'=' * 60}")
            print(f"Benchmarking: {model_name}")
            print(f"{'=' * 60}")

            # Get model metadata
            info = self.client.get_model_info(model_name)

            runs: list[SingleRunResult] = []
            for run_idx in range(self.num_runs):
                print(f"  Run {run_idx + 1}/{self.num_runs} ...", end=" ", flush=True)
                try:
                    result = self._run_single(
                        model_name, eval_suite, agent_cls, run_idx,
                    )
                    runs.append(result)
                    avg = sum(c.overall_score for c in result.case_results) / len(result.case_results)
                    print(f"avg_score={avg:.2f}  time={result.total_elapsed_seconds:.1f}s")
                except Exception as e:
                    logger.warning("Run %d for %s failed: %s", run_idx, model_name, e)
                    print(f"FAILED: {e}")
                    # Record an empty failed run
                    runs.append(SingleRunResult(
                        run_index=run_idx,
                        case_results=[],
                        total_elapsed_seconds=0.0,
                    ))

            model_result = ModelResult(
                model_name=model_name,
                parameter_count=info.get("parameter_size"),
                quantization=info.get("quantization_level"),
                runs=runs,
            )
            _compute_aggregates(model_result)
            model_results.append(model_result)

        return BenchmarkRun(
            timestamp=datetime.now(timezone.utc),
            agent_name=self.agent_name,
            num_runs_per_model=self.num_runs,
            hardware=hardware,
            model_results=model_results,
        )

    def _run_single(
        self,
        model_name: str,
        eval_suite: Any,
        agent_cls: type,
        run_index: int,
    ) -> SingleRunResult:
        """Execute one complete pass through all eval cases."""
        tracker = TokenTracker()
        llm = ChatOllama(
            base_url=self.base_url,
            model=model_name,
            temperature=0.2,
            callbacks=[tracker],
        )

        case_results: list[CaseRunResult] = []
        run_start = time.monotonic()

        with VRAMMonitor() as vram:
            for case in eval_suite.get_cases():
                tracker.reset()
                t0 = time.monotonic()
                try:
                    agent = agent_cls(llm=llm, context=case.context)
                    output = agent.run(case.prompt)
                except Exception as e:
                    output = {"parse_error": str(e), "raw_output": ""}
                elapsed = time.monotonic() - t0

                scores = eval_suite.score(case, output)
                overall_score = (
                    sum(s.score for s in scores) / len(scores) if scores else 0.0
                )
                overall_pass = all(s.passed for s in scores)

                case_results.append(CaseRunResult(
                    case_name=case.name,
                    scores={s.name: s.score for s in scores},
                    overall_score=overall_score,
                    overall_pass=overall_pass,
                    elapsed_seconds=round(elapsed, 3),
                    token_stats=tracker.to_token_stats(),
                ))

        return SingleRunResult(
            run_index=run_index,
            case_results=case_results,
            total_elapsed_seconds=round(time.monotonic() - run_start, 3),
            peak_vram_mb=vram.peak_mb,
        )

    def save_results(self, benchmark: BenchmarkRun) -> tuple[Path, Path]:
        """Write JSON results and markdown summary to evals/benchmarks/."""
        BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

        ts = benchmark.timestamp.strftime("%Y%m%d_%H%M%S")
        json_path = BENCHMARKS_DIR / f"benchmark_{ts}.json"
        md_path = BENCHMARKS_DIR / "RESULTS.md"

        json_path.write_text(
            benchmark.model_dump_json(indent=2), encoding="utf-8",
        )
        md_path.write_text(
            _render_markdown(benchmark), encoding="utf-8",
        )

        print(f"\nResults saved:")
        print(f"  JSON: {json_path}")
        print(f"  Summary: {md_path}")
        return json_path, md_path

    def print_summary(self, benchmark: BenchmarkRun) -> None:
        """Print the markdown comparison table to stdout."""
        print(_render_markdown(benchmark))


# ---------------------------------------------------------------------------
# Aggregation & rendering (module-level for testability)
# ---------------------------------------------------------------------------


def _compute_aggregates(result: ModelResult) -> None:
    """Fill in avg/min/max fields on a ModelResult from its runs."""
    valid_runs = [r for r in result.runs if r.case_results]
    if not valid_runs:
        return

    run_scores = []
    for run in valid_runs:
        avg = sum(c.overall_score for c in run.case_results) / len(run.case_results)
        run_scores.append(avg)

    result.avg_overall_score = round(sum(run_scores) / len(run_scores), 4)
    result.min_overall_score = round(min(run_scores), 4)
    result.max_overall_score = round(max(run_scores), 4)

    result.avg_elapsed_seconds = round(
        sum(r.total_elapsed_seconds for r in valid_runs) / len(valid_runs), 2,
    )

    # Tokens per second — average across all cases in all valid runs
    tps_values = []
    for run in valid_runs:
        for case in run.case_results:
            if case.token_stats and case.token_stats.tokens_per_second:
                tps_values.append(case.token_stats.tokens_per_second)
    if tps_values:
        result.avg_tokens_per_second = round(
            sum(tps_values) / len(tps_values), 2,
        )

    # Peak VRAM — average of per-run peaks
    vram_values = [r.peak_vram_mb for r in valid_runs if r.peak_vram_mb is not None]
    if vram_values:
        result.avg_peak_vram_mb = round(sum(vram_values) / len(vram_values), 1)


def _render_markdown(benchmark: BenchmarkRun) -> str:
    """Produce a markdown comparison table sorted by avg score descending."""
    lines = [
        f"# Benchmark Results — {benchmark.agent_name}",
        "",
        f"**Date:** {benchmark.timestamp.strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Runs per model:** {benchmark.num_runs_per_model}  ",
    ]
    hw = benchmark.hardware
    if hw.gpu_name:
        lines.append(f"**GPU:** {hw.gpu_name} ({hw.gpu_vram_mb} MB)  ")
    if hw.cpu:
        lines.append(f"**CPU:** {hw.cpu}  ")
    lines.append("")

    # Sort by avg score descending
    sorted_results = sorted(
        benchmark.model_results,
        key=lambda m: m.avg_overall_score,
        reverse=True,
    )

    # Table header
    lines.append(
        "| Rank | Model | Params | Quant | Avg Score | Min | Max "
        "| Avg Time (s) | Tok/s | Peak VRAM (MB) |"
    )
    lines.append(
        "|------|-------|--------|-------|-----------|-----|-----"
        "|--------------|-------|----------------|"
    )

    for i, m in enumerate(sorted_results, 1):
        tps = f"{m.avg_tokens_per_second:.1f}" if m.avg_tokens_per_second else "—"
        vram = f"{m.avg_peak_vram_mb:.0f}" if m.avg_peak_vram_mb else "—"
        lines.append(
            f"| {i} | {m.model_name} | {m.parameter_count or '—'} "
            f"| {m.quantization or '—'} "
            f"| {m.avg_overall_score:.3f} | {m.min_overall_score:.3f} "
            f"| {m.max_overall_score:.3f} | {m.avg_elapsed_seconds:.1f} "
            f"| {tps} | {vram} |"
        )

    lines.append("")
    return "\n".join(lines)
