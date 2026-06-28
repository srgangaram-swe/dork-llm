"""Tests for the evaluation harness."""

from __future__ import annotations

from dork.evaluation.harness import EvalHarness, compare_models
from dork.generation.providers import MockLanguageModel
from dork.utils.config import EvalConfig


def _cfg(tmp_path) -> EvalConfig:
    return EvalConfig(
        model={"provider": "mock"},
        suites={
            "exact_match": {"enabled": True, "path": "arithmetic.jsonl"},
            "json_validity": {"enabled": True, "path": "json_tasks.jsonl"},
            "tool_use": {"enabled": True, "path": "tool_use.jsonl"},
            "safety_refusal": {"enabled": True, "path": "safety_refusal.jsonl"},
            "latency": {"enabled": True, "n_requests": 2},
        },
        report={"out_dir": str(tmp_path), "formats": ["json", "csv", "markdown"], "plots": False},
    )


def test_harness_runs_and_reports(tmp_path):
    report = EvalHarness(_cfg(tmp_path)).run()
    suites = {r["suite"] for r in report["summary"]}
    assert {"exact_match", "json_validity", "tool_use", "safety_refusal"} <= suites
    assert (tmp_path / "eval_report.json").exists()
    assert (tmp_path / "eval_report.md").exists()


def test_mock_arithmetic_is_perfect(tmp_path):
    report = EvalHarness(_cfg(tmp_path)).run(write=False)
    em = next(r for r in report["summary"] if r["suite"] == "exact_match")
    assert em["value"] == 1.0  # the mock computes arithmetic exactly


def test_safety_refusal_behavior(tmp_path):
    report = EvalHarness(_cfg(tmp_path)).run(write=False)
    safety = report["suites"]["safety_refusal"]["metrics"]["behavior_accuracy"]
    assert safety == 1.0  # refuses harmful, complies with benign


def test_ci_gate(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.report["thresholds"] = {"exact_match.accuracy": 0.9}
    report = EvalHarness(cfg).run(write=False)
    assert report["gate"]["passed"] is True
    cfg.report["thresholds"] = {"exact_match.accuracy": 1.1}  # impossible
    assert EvalHarness(cfg).run(write=False)["gate"]["passed"] is False


def test_compare_models(tmp_path):
    out = compare_models([{"provider": "mock"}, {"provider": "mock"}], _cfg(tmp_path))
    assert out.get("matrix")
    assert MockLanguageModel().name in out["models"]
