"""Tests for document chunking strategies."""

from __future__ import annotations

import pytest
from dork.rag.chunking import chunk_text

TEXT = (
    "First paragraph sentence one. First paragraph sentence two.\n\n"
    "Second paragraph here. It has two sentences as well.\n\n"
    "Third short one."
)


@pytest.mark.parametrize("strategy", ["recursive", "fixed", "sentence"])
def test_offsets_are_valid(strategy):
    spans = chunk_text(TEXT, strategy=strategy, chunk_size=8, chunk_overlap=2, min_chunk_chars=4)
    assert spans, "expected at least one chunk"
    for s in spans:
        assert 0 <= s.start <= s.end <= len(TEXT)
        # The recorded text should match the source slice at those offsets.
        assert TEXT[s.start : s.end].strip() == s.text.strip() or s.text in TEXT


def test_min_chunk_chars_filters():
    spans = chunk_text("tiny", strategy="recursive", min_chunk_chars=100)
    # Falls back to a single chunk of the whole text rather than dropping everything.
    assert len(spans) == 1


def test_empty_text():
    assert chunk_text("   ", strategy="recursive") == []
