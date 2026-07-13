"""AxiomStack — proof, probability, and production for local LLM systems.

Three cohesive subsystems share one package:

* ``dork.models`` / ``dork.training`` / ``dork.generation`` — a tiny GPT-style
  decoder-only transformer trained from scratch in PyTorch.
* ``dork.evaluation`` — a reusable evaluation harness (perplexity, exact-match,
  multiple-choice, JSON validity, RAG faithfulness, tool-use, latency).
* ``dork.rag`` / ``dork.agents`` — a cited, source-grounded RAG pipeline and an
  agentic research assistant.

The package is intentionally local-first. Deterministic test backends remain
explicit so offline verification never disguises a mock as a trained DorkLLM.
"""

from __future__ import annotations

__version__ = "0.2.0"

__all__ = ["__version__"]
