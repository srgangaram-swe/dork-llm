"""Tests for embeddings, the vector store, and the RAG pipeline."""

from __future__ import annotations

import numpy as np
from dork.rag.embeddings import HashEmbedder
from dork.rag.pipeline import RagPipeline
from dork.rag.schema import Chunk
from dork.rag.vectorstore import MemoryVectorStore


def test_hash_embedder_is_deterministic_and_normalized():
    emb = HashEmbedder(dim=128)
    a = emb.embed(["causal masking prevents future attention"])
    b = emb.embed(["causal masking prevents future attention"])
    assert np.allclose(a, b)
    assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5


def test_hash_embedder_similarity_reflects_overlap():
    emb = HashEmbedder(dim=256)
    v = emb.embed(
        [
            "transformers use self attention",
            "self attention in transformers",
            "a recipe for vegetable soup",
        ]
    )
    sim_related = float(v[0] @ v[1])
    sim_unrelated = float(v[0] @ v[2])
    assert sim_related > sim_unrelated


def test_memory_vector_store_search():
    emb = HashEmbedder(dim=128)
    chunks = [
        Chunk(f"c{i}", "doc", "src.md", text, i, 0, len(text))
        for i, text in enumerate(["alpha beta", "gamma delta", "alpha gamma"])
    ]
    store = MemoryVectorStore()
    store.add(chunks, emb.embed([c.text for c in chunks]))
    hits = store.search(emb.embed_one("alpha"), top_k=2)
    assert len(hits) == 2
    assert hits[0].score >= hits[1].score


def test_rag_pipeline_grounded_answer_with_citation(rag_config):
    pipe = RagPipeline(rag_config)
    stats = pipe.ingest()
    assert stats.chunks > 0
    ans = pipe.query("What does causal masking prevent?")
    assert ans.contexts, "should retrieve context"
    assert ans.citations, "grounded answer should cite a source"
    assert ans.citations[0].source.endswith(".md")


def test_rag_pipeline_refuses_when_no_context(rag_config):
    # Force refusal by setting an impossible score threshold.
    rag_config.retrieval["min_score"] = 1.5
    pipe = RagPipeline(rag_config)
    pipe.ingest()
    ans = pipe.query("Completely unrelated question about quarterly revenue?")
    assert ans.refused is True
