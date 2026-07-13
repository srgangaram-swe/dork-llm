"""The AxiomStack evaluation harness.

A reusable, config-driven battery of suites that score any ``LanguageModel`` on
language modeling, reasoning, structured output, instruction following, RAG
faithfulness, tool use, safety and serving performance — then emit JSON/CSV/
Markdown reports with CI pass/fail gating.
"""

from __future__ import annotations

from dork.evaluation.base import CaseResult, Evaluator, SuiteResult, register, registered_suites
from dork.evaluation.harness import EvalHarness, compare_models
from dork.evaluation.statistics import (
    BinomialEstimate,
    PairedBootstrapEstimate,
    paired_bootstrap,
    wilson_interval,
)

__all__ = [
    "BinomialEstimate",
    "CaseResult",
    "EvalHarness",
    "Evaluator",
    "PairedBootstrapEstimate",
    "SuiteResult",
    "compare_models",
    "paired_bootstrap",
    "register",
    "registered_suites",
    "wilson_interval",
]
