"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from dork.tokenizer.char import CharTokenizer
from dork.utils.config import ModelConfig

SAMPLE_TEXT = (
    "To be, or not to be, that is the question. "
    "All the world's a stage, and all the men and women merely players. "
    "The quality of mercy is not strained. "
) * 4


@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def char_tokenizer() -> CharTokenizer:
    return CharTokenizer.train(SAMPLE_TEXT)


@pytest.fixture
def tiny_model_config() -> ModelConfig:
    return ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


@pytest.fixture
def rag_config():
    from dork.utils.config import RagConfig

    return RagConfig(
        ingest={
            "source_dir": "data/sample_docs",
            "chunking": {"chunk_size": 80, "chunk_overlap": 16, "min_chunk_chars": 20},
        },
        embeddings={"backend": "hash", "dim": 128},
        vector_store={"backend": "memory"},
        retrieval={"top_k": 3, "rerank": True, "rerank_top_n": 2, "min_score": 0.0},
        generation={"provider": "mock", "refuse_when_insufficient": True},
        agent={"max_steps": 4, "allow_code_exec": True},
    )
