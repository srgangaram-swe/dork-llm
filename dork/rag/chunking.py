"""Document chunking strategies.

Chunking is the highest-leverage knob in a RAG system: too large and retrieval
is imprecise, too small and context is lost. Three strategies are provided; the
recursive splitter (paragraph -> sentence -> word) is the sensible default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


@dataclass
class TextSpan:
    """A chunk's text plus its character offsets within the source document."""

    text: str
    start: int
    end: int


def chunk_text(
    text: str,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    min_chunk_chars: int = 64,
) -> list[TextSpan]:
    """Split ``text`` into overlapping spans using the chosen strategy.

    Args:
        text: The document text.
        strategy: ``recursive`` | ``fixed`` | ``sentence``.
        chunk_size: Target chunk size in *whitespace tokens* (approx).
        chunk_overlap: Token overlap between consecutive chunks.
        min_chunk_chars: Drop trailing chunks shorter than this.

    Returns:
        A list of :class:`TextSpan` with character offsets for citations.
    """
    if strategy == "fixed":
        spans = _fixed_window(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        spans = _sentence_pack(text, chunk_size, chunk_overlap)
    else:
        spans = _recursive(text, chunk_size, chunk_overlap)

    return [s for s in spans if len(s.text.strip()) >= min_chunk_chars] or (
        [TextSpan(text.strip(), 0, len(text))] if text.strip() else []
    )


def _approx_tokens(s: str) -> int:
    return len(s.split())


def _fixed_window(text: str, chunk_size: int, overlap: int) -> list[TextSpan]:
    """Sliding window over whitespace tokens with offset tracking."""
    # Tokenize while remembering each token's character span.
    spans: list[TextSpan] = []
    tokens = list(re.finditer(r"\S+", text))
    if not tokens:
        return []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(tokens), step):
        window = tokens[i : i + chunk_size]
        if not window:
            break
        start = window[0].start()
        end = window[-1].end()
        spans.append(TextSpan(text[start:end], start, end))
        if i + chunk_size >= len(tokens):
            break
    return spans


def _sentence_pack(text: str, chunk_size: int, overlap: int) -> list[TextSpan]:
    """Greedily pack whole sentences up to the token budget."""
    spans: list[TextSpan] = []
    pos = 0
    sentences: list[tuple[str, int, int]] = []
    for sent in _SENTENCE_RE.split(text):
        idx = text.find(sent, pos)
        if idx < 0:
            idx = pos
        sentences.append((sent, idx, idx + len(sent)))
        pos = idx + len(sent)

    cur: list[tuple[str, int, int]] = []
    cur_tokens = 0
    for sent in sentences:
        s_tokens = _approx_tokens(sent[0])
        if cur and cur_tokens + s_tokens > chunk_size:
            spans.append(TextSpan(text[cur[0][1] : cur[-1][2]], cur[0][1], cur[-1][2]))
            # Overlap: carry the last sentence into the next chunk.
            cur = cur[-1:] if overlap > 0 else []
            cur_tokens = _approx_tokens(cur[0][0]) if cur else 0
        cur.append(sent)
        cur_tokens += s_tokens
    if cur:
        spans.append(TextSpan(text[cur[0][1] : cur[-1][2]], cur[0][1], cur[-1][2]))
    return spans


def _recursive(text: str, chunk_size: int, overlap: int) -> list[TextSpan]:
    """Split on paragraphs first; further split oversized paragraphs by sentence."""
    spans: list[TextSpan] = []
    pos = 0
    for para in _PARAGRAPH_RE.split(text):
        if not para.strip():
            pos += len(para) + 2
            continue
        start = text.find(para, pos)
        if start < 0:
            start = pos
        pos = start + len(para)
        if _approx_tokens(para) <= chunk_size:
            spans.append(TextSpan(para.strip(), start, start + len(para)))
        else:
            # Recurse into sentence packing for long paragraphs.
            for span in _sentence_pack(para, chunk_size, overlap):
                spans.append(TextSpan(span.text, start + span.start, start + span.end))
    return spans
