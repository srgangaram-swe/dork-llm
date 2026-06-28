"""Lightweight lexical reranker.

A full cross-encoder reranker (e.g. ``ms-marco-MiniLM``) is the production choice;
to stay local-first this module ships a dependency-free lexical reranker that
blends the retrieval (cosine) score with query/chunk token-overlap F1. It cheaply
improves precision and demonstrates the rerank stage. Swap in a cross-encoder by
implementing the same ``rerank`` signature.
"""

from __future__ import annotations

from dork.evaluation.metrics import token_f1
from dork.rag.schema import ScoredChunk


class LexicalReranker:
    """Re-score chunks by combining cosine similarity with lexical overlap."""

    def __init__(self, alpha: float = 0.5) -> None:
        # alpha weights the original retrieval score vs. the lexical overlap.
        self.alpha = alpha

    def rerank(self, query: str, chunks: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        rescored: list[ScoredChunk] = []
        for sc in chunks:
            overlap = token_f1(query, sc.chunk.text)
            blended = self.alpha * sc.score + (1 - self.alpha) * overlap
            rescored.append(ScoredChunk(sc.chunk, blended))
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:top_n]
