"""The evaluation harness: run enabled suites, aggregate, gate, and report.

This is the orchestration layer companies wrap around model releases: given a
config it builds the target model, runs the selected suites, computes a summary,
applies CI thresholds (pass/fail), and writes reports. It also supports running
several models for a side-by-side comparison table.
"""

from __future__ import annotations

from typing import Any

# Importing evaluators triggers their @register decorators (side-effect import).
from dork.evaluation import evaluators as _evaluators  # noqa: F401
from dork.evaluation.base import SuiteResult, get_evaluator
from dork.evaluation.report import now_iso, write_reports
from dork.generation.providers import LanguageModel, build_language_model
from dork.utils.config import EvalConfig
from dork.utils.logging import get_logger
from dork.utils.seed import seed_everything

logger = get_logger(__name__)


class EvalHarness:
    """Run a configured battery of evaluation suites against one model."""

    def __init__(self, cfg: EvalConfig) -> None:
        self.cfg = cfg
        self.datasets_dir = cfg.datasets_dir

    def _enabled_suites(self) -> list[tuple[str, dict[str, Any]]]:
        suites = []
        for name, block in self.cfg.suites.items():
            block = block or {}
            if block.get("enabled", True):
                suites.append((name, block))
        return suites

    def run(self, model: LanguageModel | None = None, write: bool = True) -> dict[str, Any]:
        """Execute all enabled suites and return the assembled report dict."""
        seed_everything(self.cfg.seed)
        model = model or build_language_model(self.cfg.model)
        logger.info("Evaluating model: %s", model.name)

        suite_results: dict[str, SuiteResult] = {}
        for name, block in self._enabled_suites():
            try:
                evaluator = get_evaluator(name)(name, block, self.datasets_dir)
            except KeyError:
                logger.warning("No evaluator registered for suite %r; skipping.", name)
                continue
            logger.info("Running suite: %s", name)
            suite_results[name] = evaluator.run(model)

        report = self._assemble_report(model, suite_results)

        if write:
            report_cfg = self.cfg.report or {}
            write_reports(
                report,
                report_cfg.get("out_dir", "reports"),
                report_cfg.get("formats", ["json", "csv", "markdown"]),
                bool(report_cfg.get("plots", True)),
            )
        return report

    def _assemble_report(
        self, model: LanguageModel, results: dict[str, SuiteResult]
    ) -> dict[str, Any]:
        summary = []
        for name, res in results.items():
            metric, value = res.primary_metric
            summary.append(
                {
                    "suite": name,
                    "category": res.category,
                    "n": res.n,
                    "metric": metric,
                    "value": value,
                }
            )

        gate = self._evaluate_gate(results)
        return {
            "model": {"provider": self.cfg.model.get("provider"), "name": model.name},
            "seed": self.cfg.seed,
            "timestamp": now_iso(),
            "summary": summary,
            "suites": {k: v.to_dict() for k, v in results.items()},
            "gate": gate,
        }

    def _evaluate_gate(self, results: dict[str, SuiteResult]) -> dict[str, Any]:
        """Compare metrics against `report.thresholds` (``suite.metric: min_value``)."""
        thresholds = (self.cfg.report or {}).get("thresholds", {}) or {}
        checks = []
        passed_all = True
        for key, threshold in thresholds.items():
            suite_name, _, metric = key.partition(".")
            suite = results.get(suite_name)
            actual = suite.metrics.get(metric) if suite is not None else None
            ok = actual is not None and actual >= threshold
            passed_all = passed_all and ok
            checks.append(
                {
                    "key": key,
                    "threshold": float(threshold),
                    "actual": actual if actual is not None else float("nan"),
                    "passed": ok,
                }
            )
        return {"passed": passed_all, "checks": checks}


def compare_models(model_cfgs: list[dict[str, Any]], eval_cfg: EvalConfig) -> dict[str, Any]:
    """Run the same suites across multiple models and build a comparison table.

    Returns a dict with per-model reports and a flattened comparison matrix
    (suite/metric rows x model columns) suitable for Markdown/CSV rendering.
    """
    harness = EvalHarness(eval_cfg)
    reports: dict[str, dict[str, Any]] = {}
    for mc in model_cfgs:
        model = build_language_model(mc)
        reports[model.name] = harness.run(model, write=False)

    # Build comparison matrix keyed by "suite.metric".
    matrix: dict[str, dict[str, Any]] = {}
    for model_name, rep in reports.items():
        for suite, res in rep["suites"].items():
            for metric, value in res["metrics"].items():
                matrix.setdefault(f"{suite}.{metric}", {})[model_name] = value

    return {"models": list(reports), "matrix": matrix, "reports": reports}
