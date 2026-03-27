"""Pydantic models for benchmark results.

Used by evals/benchmark.py to store structured benchmark data
across models, runs, and eval cases.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class HardwareInfo(BaseModel):
    """Machine hardware snapshot captured at benchmark start."""

    gpu_name: str | None = None
    gpu_vram_mb: int | None = None
    cpu: str | None = None
    ram_gb: float | None = None
    os_version: str | None = None


class TokenStats(BaseModel):
    """Token-level metrics from a single LLM invocation."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_per_second: float | None = None


class CaseRunResult(BaseModel):
    """Result of a single eval case within a single run."""

    case_name: str
    scores: dict[str, float]          # criterion name -> 0.0-1.0
    overall_score: float
    overall_pass: bool
    elapsed_seconds: float
    token_stats: TokenStats | None = None


class SingleRunResult(BaseModel):
    """One complete pass through all eval cases for a single model."""

    run_index: int
    case_results: list[CaseRunResult]
    total_elapsed_seconds: float
    peak_vram_mb: float | None = None


class ModelResult(BaseModel):
    """Aggregated benchmark results for one model across all runs."""

    model_name: str
    parameter_count: str | None = None    # e.g. "7B"
    quantization: str | None = None       # e.g. "Q4_K_M"
    runs: list[SingleRunResult]
    # Aggregates (computed after all runs complete)
    avg_overall_score: float = 0.0
    min_overall_score: float = 0.0
    max_overall_score: float = 0.0
    avg_elapsed_seconds: float = 0.0
    avg_tokens_per_second: float | None = None
    avg_peak_vram_mb: float | None = None


class BenchmarkRun(BaseModel):
    """Top-level container for a full benchmark session."""

    timestamp: datetime
    agent_name: str
    num_runs_per_model: int
    hardware: HardwareInfo
    model_results: list[ModelResult]
