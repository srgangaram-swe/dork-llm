"""Core abstractions for the evaluation harness: results, the evaluator ABC,
and a name -> evaluator registry.

Every suite produces a uniform :class:`SuiteResult` (aggregate metrics + per-case
detail) so the reporting layer can render any combination of suites without
special-casing. New suites register themselves with ``@register("name")``.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

from dork.generation.providers import LanguageModel


@dataclass
class CaseResult:
    """The outcome of a single evaluation case."""

    case_id: str
    passed: bool
    score: float
    prompt: str = ""
    output: str = ""
    expected: Any = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SuiteResult:
    """Aggregate result for one evaluation suite."""

    suite: str
    metrics: dict[str, float]
    cases: list[CaseResult] = field(default_factory=list)
    category: str = "general"

    @property
    def n(self) -> int:
        return len(self.cases)

    @property
    def primary_metric(self) -> tuple[str, float]:
        """The first metric, treated as the headline number for tables."""
        if not self.metrics:
            return ("n", float(self.n))
        key = next(iter(self.metrics))
        return (key, self.metrics[key])

    def failures(self) -> list[CaseResult]:
        return [c for c in self.cases if not c.passed]

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "category": self.category,
            "n": self.n,
            "metrics": self.metrics,
            "cases": [c.to_dict() for c in self.cases],
        }


class Evaluator(abc.ABC):
    """Base class for all evaluators.

    Args:
        name: Suite name (matches the key in the eval config ``suites``).
        config: The per-suite config block.
        datasets_dir: Directory containing bundled JSONL datasets.
    """

    category: str = "general"

    def __init__(self, name: str, config: dict[str, Any], datasets_dir: str) -> None:
        self.name = name
        self.config = config
        self.datasets_dir = datasets_dir

    @abc.abstractmethod
    def run(self, model: LanguageModel) -> SuiteResult:
        """Evaluate ``model`` and return a :class:`SuiteResult`."""

    # Convenience for evaluators that read a bundled JSONL dataset.
    def _load_jsonl(self, filename: str) -> list[dict[str, Any]]:
        from dork.utils.io import read_jsonl

        return read_jsonl(f"{self.datasets_dir}/{filename}")


# ─────────────────────────── Registry ────────────────────────────────
_REGISTRY: dict[str, type[Evaluator]] = {}


def register(name: str) -> Callable[[type[Evaluator]], type[Evaluator]]:
    """Class decorator: register an evaluator under ``name``."""

    def deco(cls: type[Evaluator]) -> type[Evaluator]:
        _REGISTRY[name] = cls
        return cls

    return deco


def get_evaluator(name: str) -> type[Evaluator]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown evaluator {name!r}. Registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def registered_suites() -> list[str]:
    return sorted(_REGISTRY)
