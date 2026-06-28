"""Dork LLM — an end-to-end LLM systems platform.

Three cohesive subsystems share one package:

* ``dork.models`` / ``dork.training`` / ``dork.generation`` — a tiny GPT-style
  decoder-only transformer trained from scratch in PyTorch.
* ``dork.evaluation`` — a reusable evaluation harness (perplexity, exact-match,
  multiple-choice, JSON validity, RAG faithfulness, tool-use, latency).
* ``dork.rag`` / ``dork.agents`` — a cited, source-grounded RAG pipeline and an
  agentic research assistant.

The package is intentionally *local-first*: every subsystem has a zero-dependency
fallback (mock model, hash embedder, in-memory vector store) so the platform runs
and self-tests fully offline, then swaps in heavier backends for real use.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
