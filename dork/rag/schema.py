"""Data models for the RAG pipeline: documents, chunks, retrievals and answers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A source document loaded from disk."""

    doc_id: str
    source: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a document with provenance for citations."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    index: int
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "index": self.index,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
        }


@dataclass
class ScoredChunk:
    """A chunk paired with a retrieval/rerank relevance score."""

    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, **self.chunk.to_dict()}


@dataclass
class Citation:
    """A numbered citation pointing back to a source chunk."""

    marker: int  # the [n] shown in the answer
    source: str
    chunk_id: str
    snippet: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "snippet": self.snippet,
            "score": self.score,
        }


@dataclass
class RagAnswer:
    """The final grounded answer with its evidence and citations."""

    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    contexts: list[ScoredChunk] = field(default_factory=list)
    refused: bool = False
    model: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "refused": self.refused,
            "model": self.model,
            "citations": [c.to_dict() for c in self.citations],
            "contexts": [c.to_dict() for c in self.contexts],
        }
