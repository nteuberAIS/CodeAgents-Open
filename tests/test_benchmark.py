"""Tests for the benchmark framework — mocked Ollama, no real inference."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from evals.benchmark import (
    DEFAULT_MODELS,
    BenchmarkRunner,
    OllamaClient,
    TokenTracker,
    VRAMMonitor,
    _compute_aggregates,
    _render_markdown,
    collect_hardware_info,
)
from schemas.benchmark import (
    BenchmarkRun,
    CaseRunResult,
    HardwareInfo,
    ModelResult,
    SingleRunResult,
    TokenStats,
)


# -- Fixtures --

VALID_PLAN = {
    "sprint": 8,
    "goal": "Deploy SHIR and establish medallion bronze layer",
    "tasks": [
        {
            "id": "SP8-001",
            "title": "Deploy Self-Hosted Integration Runtime",
            "description": "Install and configure SHIR on the data gateway VM",
            "assignee": None,
            "estimate_hrs": 4,
            "status": "todo",
        },
        {
            "id": "SP8-002",
            "title": "Create bronze layer Parquet schemas",
            "description": "Define Parquet schemas for raw data ingestion",
            "assignee": None,
            "estimate_hrs": 6,
            "status": "todo",
        },
    ],
    "dependencies": [
        {"from": "SP8-002", "to": "SP8-001", "type": "blocks"},
    ],
}


def _make_mock_agent(plan_dict):
    agent = MagicMock()
    agent.run.return_value = plan_dict
    return agent


# -- Pydantic model tests --


class TestBenchmarkModels:
    def test_hardware_info_defaults(self):
        hw = HardwareInfo()
        assert hw.gpu_name is None
        assert hw.gpu_vram_mb is None

    def test_hardware_info_with_values(self):
        hw = HardwareInfo(gpu_name="RTX 2000", gpu_vram_mb=8192, cpu="i9")
        assert hw.gpu_name == "RTX 2000"

    def test_token_stats_defaults(self):
        ts = TokenStats()
        assert ts.prompt_tokens is None
        assert ts.tokens_per_second is None

    def test_case_run_result(self):
        cr = CaseRunResult(
            case_name="test",
            scores={"json_valid": 1.0, "schema_compliance": 0.75},
            overall_score=0.875,
            overall_pass=False,
            elapsed_seconds=5.5,
        )
        assert cr.case_name == "test"
        assert cr.scores["json_valid"] == 1.0

    def test_case_run_result_with_token_stats(self):
        cr = CaseRunResult(
            case_name="test",
            scores={"s1": 1.0},
            overall_score=1.0,
            overall_pass=True,
            elapsed_seconds=2.0,
            token_stats=TokenStats(completion_tokens=100, tokens_per_second=25.0),
        )
        assert cr.token_stats.tokens_per_second == 25.0

    def test_single_run_result(self):
        sr = SingleRunResult(
            run_index=0,
            case_results=[],
            total_elapsed_seconds=10.0,
            peak_vram_mb=4500.0,
        )
        assert sr.peak_vram_mb == 4500.0

    def test_model_result_defaults(self):
        mr = ModelResult(model_name="test:7b", runs=[])
        assert mr.avg_overall_score == 0.0
        assert mr.parameter_count is None

    def test_benchmark_run_serialization_roundtrip(self):
        br = BenchmarkRun(
            timestamp=datetime(2026, 3, 27, tzinfo=timezone.utc),
            agent_name="sprint_planner",
            num_runs_per_model=3,
            hardware=HardwareInfo(),
            model_results=[],
        )
        data = json.loads(br.model_dump_json())
        assert data["agent_name"] == "sprint_planner"
        assert data["num_runs_per_model"] == 3

    def test_benchmark_run_with_full_data(self):
        cr = CaseRunResult(
            case_name="basic",
            scores={"json_valid": 1.0},
            overall_score=1.0,
            overall_pass=True,
            elapsed_seconds=3.0,
        )
        sr = SingleRunResult(
            run_index=0,
            case_results=[cr],
            total_elapsed_seconds=3.0,
        )
        mr = ModelResult(model_name="test:7b", runs=[sr])
        br = BenchmarkRun(
            timestamp=datetime(2026, 3, 27, tzinfo=timezone.utc),
            agent_name="sprint_planner",
            num_runs_per_model=1,
            hardware=HardwareInfo(),
            model_results=[mr],
        )
        dumped = json.loads(br.model_dump_json())
        assert len(dumped["model_results"]) == 1
        assert dumped["model_results"][0]["runs"][0]["case_results"][0]["case_name"] == "basic"


# -- OllamaClient tests --


class TestOllamaClient:
    def test_is_running_true(self):
        client = OllamaClient()
        with patch("evals.benchmark.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"models": []}).encode()
            mock_open.return_value = mock_resp
            assert client.is_running() is True

    def test_is_running_false(self):
        client = OllamaClient()
        with patch("evals.benchmark.urllib.request.urlopen") as mock_open:
            import urllib.error
            mock_open.side_effect = urllib.error.URLError("refused")
            assert client.is_running() is False

    def test_list_models(self):
        client = OllamaClient()
        with patch("evals.benchmark.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({
                "models": [
                    {"name": "qwen2.5-coder:7b"},
                    {"name": "qwen3:8b"},
                ],
            }).encode()
            mock_open.return_value = mock_resp
            models = client.list_models()
            assert models == ["qwen2.5-coder:7b", "qwen3:8b"]

    def test_get_model_info(self):
        client = OllamaClient()
        with patch("evals.benchmark.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({
                "details": {
                    "parameter_size": "7B",
                    "quantization_level": "Q4_K_M",
                },
            }).encode()
            mock_open.return_value = mock_resp
            info = client.get_model_info("qwen2.5-coder:7b")
            assert info["parameter_size"] == "7B"
            assert info["quantization_level"] == "Q4_K_M"

    def test_get_model_info_error_returns_nones(self):
        client = OllamaClient()
        with patch("evals.benchmark.urllib.request.urlopen") as mock_open:
            mock_open.side_effect = Exception("network error")
            info = client.get_model_info("bad:model")
            assert info["parameter_size"] is None
            assert info["quantization_level"] is None

    def test_pull_model(self, capsys):
        client = OllamaClient()
        with patch("evals.benchmark.urllib.request.urlopen") as mock_open:
            mock_resp = MagicMock()
            mock_resp.__enter__ = MagicMock(return_value=mock_resp)
            mock_resp.__exit__ = MagicMock(return_value=False)
            mock_resp.read.return_value = json.dumps({"status": "success"}).encode()
            mock_open.return_value = mock_resp
            client.pull_model("test:7b")
            captured = capsys.readouterr()
            assert "Pulling test:7b" in captured.out
            assert "test:7b ready" in captured.out


# -- TokenTracker tests --


class TestTokenTracker:
    def test_captures_generation_info(self):
        tracker = TokenTracker()
        response = MagicMock()
        gen = MagicMock()
        gen.generation_info = {
            "prompt_eval_count": 150,
            "eval_count": 200,
            "eval_duration": 5_000_000_000,  # 5 seconds in nanoseconds
        }
        response.generations = [[gen]]

        tracker.on_llm_end(response)
        stats = tracker.to_token_stats()

        assert stats.prompt_tokens == 150
        assert stats.completion_tokens == 200
        assert stats.tokens_per_second == 40.0  # 200 / 5

    def test_missing_generation_info_returns_nones(self):
        tracker = TokenTracker()
        stats = tracker.to_token_stats()
        assert stats.prompt_tokens is None
        assert stats.completion_tokens is None
        assert stats.tokens_per_second is None

    def test_empty_generation_info(self):
        tracker = TokenTracker()
        response = MagicMock()
        gen = MagicMock()
        gen.generation_info = {}
        response.generations = [[gen]]

        tracker.on_llm_end(response)
        stats = tracker.to_token_stats()

        assert stats.prompt_tokens is None
        assert stats.tokens_per_second is None

    def test_reset_clears_state(self):
        tracker = TokenTracker()
        response = MagicMock()
        gen = MagicMock()
        gen.generation_info = {
            "prompt_eval_count": 100,
            "eval_count": 200,
            "eval_duration": 2_000_000_000,
        }
        response.generations = [[gen]]

        tracker.on_llm_end(response)
        tracker.reset()
        stats = tracker.to_token_stats()

        assert stats.prompt_tokens is None
        assert stats.completion_tokens is None

    def test_handles_index_error(self):
        tracker = TokenTracker()
        response = MagicMock()
        response.generations = []  # empty — will cause IndexError

        tracker.on_llm_end(response)
        stats = tracker.to_token_stats()
        assert stats.prompt_tokens is None


# -- VRAMMonitor tests --


class TestVRAMMonitor:
    def test_captures_peak_vram(self):
        call_count = 0
        values = [3000, 4500, 4000]

        def mock_run(cmd, **kwargs):
            nonlocal call_count
            result = MagicMock()
            result.returncode = 0
            idx = min(call_count, len(values) - 1)
            result.stdout = f"{values[idx]}\n"
            call_count += 1
            return result

        with patch("evals.benchmark.subprocess.run", side_effect=mock_run):
            with VRAMMonitor(poll_interval=0.05) as mon:
                # Give the thread time to poll a few times
                import time
                time.sleep(0.2)

        assert mon.peak_mb == 4500.0

    def test_no_nvidia_smi_returns_none(self):
        with patch("evals.benchmark.subprocess.run", side_effect=FileNotFoundError):
            with VRAMMonitor(poll_interval=0.05) as mon:
                import time
                time.sleep(0.1)

        assert mon.peak_mb is None

    def test_thread_cleanup(self):
        with patch("evals.benchmark.subprocess.run", side_effect=FileNotFoundError):
            with VRAMMonitor(poll_interval=0.05) as mon:
                assert mon._thread is not None

        # Thread should have been joined
        assert not mon._thread.is_alive()


# -- collect_hardware_info tests --


class TestCollectHardwareInfo:
    def test_with_gpu(self):
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "NVIDIA RTX 2000 Ada, 8192\n"

        with patch("evals.benchmark.subprocess.run", return_value=mock_result):
            hw = collect_hardware_info()

        assert hw.gpu_name == "NVIDIA RTX 2000 Ada"
        assert hw.gpu_vram_mb == 8192
        assert hw.os_version is not None

    def test_without_gpu(self):
        with patch("evals.benchmark.subprocess.run", side_effect=FileNotFoundError):
            hw = collect_hardware_info()

        assert hw.gpu_name is None
        assert hw.gpu_vram_mb is None
        assert hw.os_version is not None


# -- _compute_aggregates tests --


class TestComputeAggregates:
    def test_basic_aggregation(self):
        cr1 = CaseRunResult(
            case_name="c1", scores={"s": 0.8}, overall_score=0.8,
            overall_pass=True, elapsed_seconds=5.0,
        )
        cr2 = CaseRunResult(
            case_name="c2", scores={"s": 0.6}, overall_score=0.6,
            overall_pass=False, elapsed_seconds=3.0,
        )
        run1 = SingleRunResult(
            run_index=0, case_results=[cr1, cr2],
            total_elapsed_seconds=8.0, peak_vram_mb=4000.0,
        )
        run2 = SingleRunResult(
            run_index=1, case_results=[cr1, cr2],
            total_elapsed_seconds=9.0, peak_vram_mb=4200.0,
        )
        mr = ModelResult(model_name="test:7b", runs=[run1, run2])
        _compute_aggregates(mr)

        assert mr.avg_overall_score == 0.7  # (0.8+0.6)/2 = 0.7
        assert mr.min_overall_score == 0.7
        assert mr.max_overall_score == 0.7
        assert mr.avg_elapsed_seconds == 8.5
        assert mr.avg_peak_vram_mb == 4100.0

    def test_empty_runs(self):
        mr = ModelResult(model_name="test:7b", runs=[])
        _compute_aggregates(mr)
        assert mr.avg_overall_score == 0.0

    def test_failed_runs_excluded(self):
        cr = CaseRunResult(
            case_name="c1", scores={"s": 1.0}, overall_score=1.0,
            overall_pass=True, elapsed_seconds=5.0,
        )
        good_run = SingleRunResult(
            run_index=0, case_results=[cr],
            total_elapsed_seconds=5.0,
        )
        failed_run = SingleRunResult(
            run_index=1, case_results=[],  # empty = failed
            total_elapsed_seconds=0.0,
        )
        mr = ModelResult(model_name="test:7b", runs=[good_run, failed_run])
        _compute_aggregates(mr)

        assert mr.avg_overall_score == 1.0  # only counts the good run

    def test_with_token_stats(self):
        cr = CaseRunResult(
            case_name="c1", scores={"s": 1.0}, overall_score=1.0,
            overall_pass=True, elapsed_seconds=5.0,
            token_stats=TokenStats(tokens_per_second=30.0),
        )
        run = SingleRunResult(
            run_index=0, case_results=[cr],
            total_elapsed_seconds=5.0,
        )
        mr = ModelResult(model_name="test:7b", runs=[run])
        _compute_aggregates(mr)

        assert mr.avg_tokens_per_second == 30.0


# -- _render_markdown tests --


class TestRenderMarkdown:
    def test_produces_valid_table(self):
        mr = ModelResult(
            model_name="test:7b",
            parameter_count="7B",
            quantization="Q4_K_M",
            runs=[],
            avg_overall_score=0.85,
            min_overall_score=0.80,
            max_overall_score=0.90,
            avg_elapsed_seconds=10.5,
            avg_tokens_per_second=25.0,
            avg_peak_vram_mb=4500.0,
        )
        br = BenchmarkRun(
            timestamp=datetime(2026, 3, 27, tzinfo=timezone.utc),
            agent_name="sprint_planner",
            num_runs_per_model=3,
            hardware=HardwareInfo(gpu_name="RTX 2000", gpu_vram_mb=8192),
            model_results=[mr],
        )

        md = _render_markdown(br)
        assert "# Benchmark Results" in md
        assert "test:7b" in md
        assert "7B" in md
        assert "Q4_K_M" in md
        assert "RTX 2000" in md

    def test_sorts_by_score_descending(self):
        mr1 = ModelResult(
            model_name="low:1b", runs=[],
            avg_overall_score=0.5, min_overall_score=0.5, max_overall_score=0.5,
        )
        mr2 = ModelResult(
            model_name="high:7b", runs=[],
            avg_overall_score=0.9, min_overall_score=0.9, max_overall_score=0.9,
        )
        br = BenchmarkRun(
            timestamp=datetime(2026, 3, 27, tzinfo=timezone.utc),
            agent_name="sprint_planner",
            num_runs_per_model=1,
            hardware=HardwareInfo(),
            model_results=[mr1, mr2],
        )

        md = _render_markdown(br)
        # "high:7b" should appear before "low:1b" in the table
        assert md.index("high:7b") < md.index("low:1b")


# -- BenchmarkRunner tests --


class TestBenchmarkRunner:
    def _mock_client(self, available_models=None):
        """Return a mocked OllamaClient."""
        client = MagicMock(spec=OllamaClient)
        client.is_running.return_value = True
        client.list_models.return_value = available_models or ["qwen2.5-coder:7b"]
        client.get_model_info.return_value = {
            "parameter_size": "7B",
            "quantization_level": "Q4_K_M",
        }
        return client

    def test_preflight_check_pulls_missing(self):
        runner = BenchmarkRunner(models=["qwen2.5-coder:7b", "qwen3:8b"])
        runner.client = self._mock_client(available_models=["qwen2.5-coder:7b"])

        runner.preflight_check()

        runner.client.pull_model.assert_called_once_with("qwen3:8b")

    def test_preflight_check_ollama_not_running(self):
        runner = BenchmarkRunner(models=["qwen2.5-coder:7b"])
        runner.client = MagicMock(spec=OllamaClient)
        runner.client.is_running.return_value = False

        with pytest.raises(RuntimeError, match="Ollama is not running"):
            runner.preflight_check()

    def test_preflight_check_all_available(self):
        runner = BenchmarkRunner(models=["qwen2.5-coder:7b"])
        runner.client = self._mock_client(available_models=["qwen2.5-coder:7b"])

        runner.preflight_check()

        runner.client.pull_model.assert_not_called()

    @patch("evals.benchmark.VRAMMonitor")
    @patch("evals.benchmark.ChatOllama")
    @patch("evals.benchmark.resolve_agent_class")
    @patch("evals.benchmark.resolve_eval_class")
    @patch("evals.benchmark.collect_hardware_info")
    def test_run_produces_benchmark(
        self, mock_hw, mock_eval_cls, mock_agent_cls, mock_chat, mock_vram,
    ):
        # Setup hardware
        mock_hw.return_value = HardwareInfo(gpu_name="Test GPU")

        # Setup eval suite
        mock_case = MagicMock()
        mock_case.name = "test_case"
        mock_case.context = None
        mock_case.prompt = "Plan sprint 8"
        mock_suite = MagicMock()
        mock_suite.get_cases.return_value = [mock_case]
        mock_suite.score.return_value = [
            MagicMock(name="s1", score=1.0, passed=True),
        ]
        mock_eval_cls.return_value.return_value = mock_suite

        # Setup agent
        mock_agent = MagicMock()
        mock_agent.run.return_value = VALID_PLAN
        mock_agent_cls.return_value.return_value = mock_agent

        # Setup VRAM monitor
        mock_vram_instance = MagicMock()
        mock_vram_instance.peak_mb = 4000.0
        mock_vram.return_value.__enter__ = MagicMock(return_value=mock_vram_instance)
        mock_vram.return_value.__exit__ = MagicMock(return_value=False)

        runner = BenchmarkRunner(models=["test:7b"], num_runs=1)
        runner.client = self._mock_client(available_models=["test:7b"])

        benchmark = runner.run()

        assert benchmark.agent_name == "sprint_planner"
        assert len(benchmark.model_results) == 1
        assert benchmark.model_results[0].model_name == "test:7b"
        assert len(benchmark.model_results[0].runs) == 1

    @patch("evals.benchmark.VRAMMonitor")
    @patch("evals.benchmark.ChatOllama")
    @patch("evals.benchmark.resolve_agent_class")
    @patch("evals.benchmark.resolve_eval_class")
    @patch("evals.benchmark.collect_hardware_info")
    def test_run_handles_model_failure(
        self, mock_hw, mock_eval_cls, mock_agent_cls, mock_chat, mock_vram,
    ):
        mock_hw.return_value = HardwareInfo()

        # Make agent.run() raise to simulate OOM
        mock_agent = MagicMock()
        mock_agent.run.side_effect = RuntimeError("CUDA out of memory")
        mock_agent_cls.return_value.return_value = mock_agent

        mock_case = MagicMock()
        mock_case.name = "test_case"
        mock_case.context = None
        mock_case.prompt = "Plan sprint 8"
        mock_suite = MagicMock()
        mock_suite.get_cases.return_value = [mock_case]
        # score() will be called with parse_error output
        mock_suite.score.return_value = [
            MagicMock(name="json_valid", score=0.0, passed=False),
        ]
        mock_eval_cls.return_value.return_value = mock_suite

        mock_vram_instance = MagicMock()
        mock_vram_instance.peak_mb = None
        mock_vram.return_value.__enter__ = MagicMock(return_value=mock_vram_instance)
        mock_vram.return_value.__exit__ = MagicMock(return_value=False)

        runner = BenchmarkRunner(models=["test:7b"], num_runs=1)
        runner.client = self._mock_client(available_models=["test:7b"])

        # Should not raise — errors captured
        benchmark = runner.run()
        assert len(benchmark.model_results) == 1

    def test_save_results(self, tmp_path):
        br = BenchmarkRun(
            timestamp=datetime(2026, 3, 27, 14, 30, 0, tzinfo=timezone.utc),
            agent_name="sprint_planner",
            num_runs_per_model=1,
            hardware=HardwareInfo(),
            model_results=[],
        )

        runner = BenchmarkRunner(models=["test:7b"])
        with patch("evals.benchmark.BENCHMARKS_DIR", tmp_path):
            json_path, md_path = runner.save_results(br)

        assert json_path.exists()
        assert md_path.exists()
        assert json_path.suffix == ".json"
        assert md_path.name == "RESULTS.md"

        # Verify JSON is valid
        data = json.loads(json_path.read_text())
        assert data["agent_name"] == "sprint_planner"

        # Verify markdown has header
        md = md_path.read_text()
        assert "# Benchmark Results" in md


# -- CLI integration tests --


class TestCmdBenchmark:
    def test_dry_run(self, capsys):
        """Test --dry-run shows plan without running."""
        from main import cmd_benchmark

        args = MagicMock()
        args.models = "qwen2.5-coder:7b,qwen3:8b"
        args.runs = 2
        args.agent = "sprint_planner"
        args.dry_run = True

        cmd_benchmark(args)

        captured = capsys.readouterr()
        assert "Models (2)" in captured.out
        assert "qwen2.5-coder:7b" in captured.out
        assert "qwen3:8b" in captured.out
        assert "Runs per model: 2" in captured.out

    def test_dry_run_default_models(self, capsys):
        from main import cmd_benchmark

        args = MagicMock()
        args.models = None
        args.runs = 3
        args.agent = "sprint_planner"
        args.dry_run = True

        cmd_benchmark(args)

        captured = capsys.readouterr()
        assert f"Models ({len(DEFAULT_MODELS)})" in captured.out

    @patch("evals.benchmark.BenchmarkRunner")
    def test_full_run_calls_runner(self, mock_runner_cls):
        from main import cmd_benchmark

        mock_runner = MagicMock()
        mock_benchmark = MagicMock()
        mock_runner.run.return_value = mock_benchmark
        mock_runner_cls.return_value = mock_runner

        args = MagicMock()
        args.models = "test:7b"
        args.runs = 1
        args.agent = "sprint_planner"
        args.dry_run = False

        cmd_benchmark(args)

        mock_runner_cls.assert_called_once_with(
            agent_name="sprint_planner",
            models=["test:7b"],
            num_runs=1,
        )
        mock_runner.run.assert_called_once()
        mock_runner.save_results.assert_called_once_with(mock_benchmark)
        mock_runner.print_summary.assert_called_once_with(mock_benchmark)


# -- Default models list test --


class TestDefaultModels:
    def test_default_models_non_empty(self):
        assert len(DEFAULT_MODELS) > 0

    def test_default_models_all_have_tags(self):
        for model in DEFAULT_MODELS:
            assert ":" in model, f"Model {model} missing tag (e.g. :7b)"

    def test_baseline_model_in_defaults(self):
        assert "qwen2.5-coder:7b" in DEFAULT_MODELS
