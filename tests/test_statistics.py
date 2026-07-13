"""Property and edge-case tests for evaluation statistics and latency reports."""

from __future__ import annotations

import math
import sys
from types import ModuleType, SimpleNamespace

import dork.pipelines as pipelines
import numpy as np
import pytest
from dork.evaluation.statistics import paired_bootstrap, wilson_interval


def test_paired_delta_preserves_pairs_and_is_deterministic() -> None:
    baseline = np.array([2.0, 20.0, 5.0, 11.0, 37.0])
    improved = baseline - 3.5

    first = paired_bootstrap(
        baseline,
        improved,
        statistic="mean_delta",
        n_resamples=2_000,
        seed=718,
    )
    repeated = paired_bootstrap(
        baseline,
        improved,
        statistic="mean_delta",
        n_resamples=2_000,
        seed=718,
    )

    assert first == repeated
    assert first.estimate == pytest.approx(3.5)
    assert first.ci_low == pytest.approx(3.5)
    assert first.ci_high == pytest.approx(3.5)
    assert first.n_pairs == 5
    assert first.seed == 718
    assert first.method == "paired_percentile_bootstrap"


def test_paired_delta_has_expected_antisymmetry() -> None:
    first = [1.0, 2.0, 8.0, 10.0, 15.0]
    second = [1.5, 1.0, 7.0, 13.0, 12.0]
    forward = paired_bootstrap(first, second, n_resamples=4_000, seed=91)
    reverse = paired_bootstrap(second, first, n_resamples=4_000, seed=91)

    assert forward.estimate == pytest.approx(-reverse.estimate)
    assert forward.ci_low == pytest.approx(-reverse.ci_high)
    assert forward.ci_high == pytest.approx(-reverse.ci_low)


def test_ratio_of_means_is_invariant_to_common_positive_scale() -> None:
    numerator = np.array([4.0, 12.0, 10.0, 20.0])
    denominator = np.array([2.0, 5.0, 6.0, 8.0])
    original = paired_bootstrap(
        numerator,
        denominator,
        statistic="ratio_of_means",
        n_resamples=3_000,
        seed=42,
    )
    scaled = paired_bootstrap(
        13.0 * numerator,
        13.0 * denominator,
        statistic="ratio_of_means",
        n_resamples=3_000,
        seed=42,
    )

    assert original.estimate == pytest.approx(np.mean(numerator) / np.mean(denominator))
    assert scaled.estimate == pytest.approx(original.estimate)
    assert scaled.ci_low == pytest.approx(original.ci_low)
    assert scaled.ci_high == pytest.approx(original.ci_high)
    assert original.as_dict()["statistic"] == "ratio_of_means"


@pytest.mark.parametrize(
    ("first", "second", "error"),
    [
        ([], [], "must not be empty"),
        ([1.0], [1.0, 2.0], "same number"),
        ([[1.0]], [[1.0]], "one-dimensional"),
        ([1.0, math.nan], [1.0, 2.0], "finite"),
        (["1"], [1.0], "real numeric"),
    ],
)
def test_paired_bootstrap_rejects_invalid_samples(first, second, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        paired_bootstrap(first, second)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"confidence_level": 0.0}, "confidence_level"),
        ({"confidence_level": float("nan")}, "confidence_level"),
        ({"n_resamples": 0}, "n_resamples"),
        ({"n_resamples": True}, "n_resamples"),
        ({"seed": -1}, "seed"),
        ({"statistic": "median_delta"}, "statistic"),
    ],
)
def test_paired_bootstrap_rejects_invalid_configuration(kwargs, error: str) -> None:
    with pytest.raises(ValueError, match=error):
        paired_bootstrap([1.0, 2.0], [1.0, 2.0], **kwargs)


@pytest.mark.parametrize("denominator", [[0.0, 1.0], [-1.0, 2.0]])
def test_ratio_requires_strictly_positive_denominator(denominator) -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        paired_bootstrap([1.0, 2.0], denominator, statistic="ratio_of_means")


def test_wilson_interval_matches_known_half_proportion() -> None:
    interval = wilson_interval(5, 10)

    assert interval.estimate == 0.5
    assert interval.ci_low == pytest.approx(0.236593, abs=1e-6)
    assert interval.ci_high == pytest.approx(0.763407, abs=1e-6)
    assert interval.method == "wilson_score"
    assert interval.as_dict()["trials"] == 10


def test_wilson_interval_handles_boundaries_and_complement_symmetry() -> None:
    none = wilson_interval(0, 10)
    all_successes = wilson_interval(10, 10)
    low = wilson_interval(2, 10)
    high = wilson_interval(8, 10)

    assert none.ci_low == 0.0
    assert none.ci_high > 0.0
    assert all_successes.ci_high == 1.0
    assert all_successes.ci_low < 1.0
    assert low.ci_low == pytest.approx(1.0 - high.ci_high)
    assert low.ci_high == pytest.approx(1.0 - high.ci_low)


def test_wilson_interval_widens_with_confidence_level() -> None:
    narrow = wilson_interval(37, 100, confidence_level=0.80)
    wide = wilson_interval(37, 100, confidence_level=0.99)

    assert wide.ci_low < narrow.ci_low < narrow.estimate
    assert wide.ci_high > narrow.ci_high > narrow.estimate


@pytest.mark.parametrize(
    ("successes", "trials", "confidence_level"),
    [(-1, 10, 0.95), (11, 10, 0.95), (1, 0, 0.95), (True, 10, 0.95), (1, 10, 1.0)],
)
def test_wilson_interval_rejects_invalid_inputs(successes, trials, confidence_level) -> None:
    with pytest.raises(ValueError):
        wilson_interval(successes, trials, confidence_level=confidence_level)


def test_latency_summary_reports_quantiles_and_sample_variability() -> None:
    summary = pipelines._latency_summary([0.001, 0.002, 0.003, 0.004], new_tokens=5)

    assert summary["mean_ms"] == pytest.approx(2.5)
    assert summary["p50_ms"] == pytest.approx(2.5)
    assert summary["p95_ms"] == pytest.approx(3.85)
    assert summary["std_ms"] == pytest.approx(np.std([1.0, 2.0, 3.0, 4.0], ddof=1))
    assert summary["coefficient_of_variation"] == pytest.approx(summary["std_ms"] / 2.5)
    assert summary["tokens_per_sec"] == pytest.approx(2_000.0)


def test_accelerator_synchronization_dispatch(monkeypatch) -> None:
    calls: list[str] = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(synchronize=lambda: calls.append("cuda")),
        mps=SimpleNamespace(synchronize=lambda: calls.append("mps")),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    pipelines._synchronize_device("cpu")
    pipelines._synchronize_device("cuda:1")
    pipelines._synchronize_device("mps")

    assert calls == ["cuda", "mps"]


def _fake_module(name: str, **attributes) -> ModuleType:
    module = ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def test_benchmark_reports_counterbalanced_paired_estimates(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeGenerator:
        def __init__(self, model, tokenizer, device: str) -> None:
            assert device == "cpu"

        def generate(self, prompt: str, **kwargs) -> str:
            calls.append(kwargs)
            return f"generated-{kwargs['seed']}"

    class FakeModel:
        @staticmethod
        def num_params() -> int:
            return 123

    cfg = SimpleNamespace(
        seed=100,
        training=SimpleNamespace(device="cpu", out_dir="unused"),
        generation=SimpleNamespace(max_new_tokens=4),
        tokenizer=SimpleNamespace(path="unused"),
        tracking=None,
    )
    monkeypatch.setattr(pipelines, "load_tiny_gpt_config", lambda _: cfg)
    monkeypatch.setattr(pipelines, "start_tracker", lambda *args, **kwargs: None)
    monkeypatch.setitem(
        sys.modules,
        "dork.generation.generator",
        _fake_module("dork.generation.generator", Generator=FakeGenerator),
    )
    monkeypatch.setitem(
        sys.modules,
        "dork.tokenizer.factory",
        _fake_module("dork.tokenizer.factory", load_tokenizer=lambda _: object()),
    )
    monkeypatch.setitem(
        sys.modules,
        "dork.training.checkpoint",
        _fake_module(
            "dork.training.checkpoint",
            load_model_from_checkpoint=lambda *args, **kwargs: (FakeModel(), {}),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "dork.training.trainer",
        _fake_module("dork.training.trainer", resolve_device=lambda _: "cpu"),
    )

    report = pipelines.benchmark(
        "unused.yaml",
        n_requests=4,
        warmup_requests=1,
        bootstrap_resamples=200,
        bootstrap_seed=73,
    )

    assert report["params"] == 123
    assert "mean_ms" not in report
    assert {"p50_ms", "p95_ms", "std_ms", "coefficient_of_variation"} <= report["kv_cache"].keys()
    assert len(report["pairs"]) == 4
    assert report["pairs"][0]["order"] == ["kv_cache", "reference"]
    assert report["pairs"][1]["order"] == ["reference", "kv_cache"]
    assert all(pair["outputs_match"] for pair in report["pairs"])
    assert report["paired_speedup"]["n_pairs"] == 4
    assert report["paired_speedup"]["n_resamples"] == 200
    assert report["paired_speedup"]["seed"] == 73
    assert report["paired_latency_savings_ms"]["unit"] == "ms"
    assert report["protocol"]["counterbalanced_order"] is True
    # One warm-up per path plus two measurements per pair.
    assert len(calls) == 10
    for pair_index in range(4):
        first = calls[2 + pair_index * 2]
        second = calls[3 + pair_index * 2]
        assert first["seed"] == second["seed"] == 101 + pair_index
        assert first["use_cache"] is not second["use_cache"]


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_benchmark_rejects_invalid_request_count(value) -> None:
    with pytest.raises(ValueError, match="n_requests"):
        pipelines.benchmark("unused.yaml", n_requests=value)
