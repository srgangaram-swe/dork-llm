"""Service layer backing the API and dashboard.

Holds lazily-initialized, shared state (the RAG pipeline, a text-completion model)
and degrades gracefully: if no trained checkpoint exists, generation falls back to
the deterministic mock model so the service is always runnable. Also tracks simple
in-memory metrics surfaced at ``/metrics``.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from dork import __version__
from dork.generation.providers import LanguageModel, MockLanguageModel
from dork.utils.config import load_rag_config
from dork.utils.logging import get_logger

logger = get_logger(__name__)


class Metrics:
    """Thread-safe counters and latency accumulators."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.counts: dict[str, int] = defaultdict(int)
        self.latency_ms_sum: dict[str, float] = defaultdict(float)

    def record(self, endpoint: str, latency_ms: float) -> None:
        with self._lock:
            self.counts[endpoint] += 1
            self.latency_ms_sum[endpoint] += latency_ms

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "requests": dict(self.counts),
                "avg_latency_ms": {
                    k: (self.latency_ms_sum[k] / self.counts[k]) if self.counts[k] else 0.0
                    for k in self.counts
                },
            }


class DorkService:
    """Singleton-style holder for models and the RAG pipeline."""

    def __init__(self, rag_config: str = "configs/rag_default.yaml") -> None:
        self.rag_config_path = rag_config
        self.metrics = Metrics()
        self._lm: LanguageModel | None = None
        self._rag: Any = None
        self._lock = threading.Lock()

    # ── Text model (trained GPT or mock fallback) ────────────────────
    @property
    def language_model(self) -> LanguageModel:
        if self._lm is None:
            with self._lock:
                if self._lm is None:
                    self._lm = self._load_language_model()
        return self._lm

    def _load_language_model(self) -> LanguageModel:
        try:
            from dork.generation.providers import LocalGPTModel

            lm = LocalGPTModel.from_artifacts("artifacts/tiny_gpt", device="cpu")
            logger.info("Loaded trained tiny GPT for serving.")
            return lm
        except Exception as exc:
            logger.warning("No trained checkpoint (%s); serving the mock model.", exc)
            return MockLanguageModel()

    @property
    def model_loaded(self) -> bool:
        return not isinstance(self.language_model, MockLanguageModel)

    # ── RAG pipeline ─────────────────────────────────────────────────
    @property
    def rag(self) -> Any:
        if self._rag is None:
            with self._lock:
                if self._rag is None:
                    from dork.rag.pipeline import RagPipeline

                    cfg = load_rag_config(self.rag_config_path)
                    self._rag = RagPipeline(cfg)
                    if self._rag.store.count() == 0:
                        try:
                            self._rag.ingest()
                        except Exception as exc:  # pragma: no cover
                            logger.warning("Auto-ingest skipped: %s", exc)
        return self._rag

    # ── Operations ───────────────────────────────────────────────────
    def generate(self, prompt: str, **kw: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        completion = self.language_model.complete(
            prompt,
            max_new_tokens=kw.get("max_new_tokens", 128),
            temperature=kw.get("temperature", 0.8),
            top_k=kw.get("top_k"),
            top_p=kw.get("top_p"),
        )
        latency = (time.perf_counter() - t0) * 1000
        self.metrics.record("generate", latency)
        return {
            "prompt": prompt,
            "completion": completion,
            "model": self.language_model.name,
            "latency_ms": latency,
        }

    def rag_ingest(self, source: str | None = None) -> dict[str, Any]:
        t0 = time.perf_counter()
        stats = self.rag.ingest(source).to_dict()
        self.metrics.record("rag_ingest", (time.perf_counter() - t0) * 1000)
        return stats

    def rag_query(self, question: str, top_k: int = 5) -> dict[str, Any]:
        t0 = time.perf_counter()
        ans = self.rag.query(question, top_k=top_k).to_dict()
        self.metrics.record("rag_query", (time.perf_counter() - t0) * 1000)
        return ans

    def run_agent(self, task: str) -> dict[str, Any]:
        from dork.agents.research_agent import ResearchAgent

        t0 = time.perf_counter()
        result = ResearchAgent(self.rag).run(task).to_dict()
        self.metrics.record("agent_run", (time.perf_counter() - t0) * 1000)
        return result

    def evaluate(self, config_path: str = "configs/eval_default.yaml") -> dict[str, Any]:
        from dork.evaluation.harness import EvalHarness
        from dork.utils.config import load_eval_config

        t0 = time.perf_counter()
        report = EvalHarness(load_eval_config(config_path)).run(write=False)
        self.metrics.record("evaluate", (time.perf_counter() - t0) * 1000)
        return {"summary": report["summary"], "gate": report["gate"], "model": report["model"]}

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model_loaded": self.model_loaded,
            "rag_chunks": self.rag.store.count() if self._rag is not None else 0,
        }
