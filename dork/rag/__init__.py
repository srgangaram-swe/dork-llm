"""Retrieval-augmented generation: ingestion, indexing, retrieval and grounded QA."""

from __future__ import annotations

from dork.rag.embeddings import EmbeddingBackend, HashEmbedder, build_embedder
from dork.rag.pipeline import IngestStats, RagPipeline, build_pipeline
from dork.rag.schema import Chunk, Citation, Document, RagAnswer, ScoredChunk
from dork.rag.vectorstore import MemoryVectorStore, VectorStore, build_vector_store

__all__ = [
    "Chunk",
    "Citation",
    "Document",
    "EmbeddingBackend",
    "HashEmbedder",
    "IngestStats",
    "MemoryVectorStore",
    "RagAnswer",
    "RagPipeline",
    "ScoredChunk",
    "VectorStore",
    "build_embedder",
    "build_pipeline",
    "build_vector_store",
]
