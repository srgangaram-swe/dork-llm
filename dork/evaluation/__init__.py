"""The Dork LLM evaluation harness.

A reusable, config-driven battery of suites that score any ``LanguageModel`` on
language modeling, reasoning, structured output, instruction following, RAG
faithfulness, tool use, safety and serving performance — then emit JSON/CSV/
Markdown reports with CI pass/fail gating.
"""

from __future__ import annotations

from dork.evaluation.base import CaseResult, Evaluator, SuiteResult, register, registered_suites
from dork.evaluation.harness import EvalHarness, compare_models

__all__ = [
    "CaseResult",
    "EvalHarness",
    "Evaluator",
    "SuiteResult",
    "compare_models",
    "register",
    "registered_suites",
]
