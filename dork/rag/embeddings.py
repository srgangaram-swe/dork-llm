"""Embedding backends behind a common interface.

* :class:`HashEmbedder` — a deterministic hashing vectorizer (no model download,
  no network). It produces *real* lexical-overlap similarity, so retrieval demos
  and tests work fully offline and reproducibly.
* :class:`SentenceTransformerEmbedder` — wraps a real sentence-transformers model
  for production-quality semantic search.
"""

from __future__ import annotations

import abc
import hashlib
import re

import numpy as np

from dork.utils.logging import get_logger

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")


class EmbeddingBackend(abc.ABC):
    """Map texts to dense vectors."""

    dim: int

    @abc.abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(len(texts), dim)`` float32 array."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class HashEmbedder(EmbeddingBackend):
    """Hashing vectorizer over word unigrams + character trigrams.

    Each feature is hashed into a fixed-width vector with a signed component, so
    cosine similarity reflects shared vocabulary — enough for meaningful retrieval
    without any external model. Deterministic across runs and machines.
    """

    def __init__(self, dim: int = 256, normalize: bool = True) -> None:
        self.dim = dim
        self.normalize = normalize

    def _features(self, text: str) -> list[str]:
        words = _WORD_RE.findall(text.lower())
        feats = list(words)
        # Character trigrams add robustness to morphology / typos.
        for w in words:
            padded = f"#{w}#"
            feats.extend(padded[i : i + 3] for i in range(len(padded) - 2))
        return feats

    @staticmethod
    def _hash(feature: str, dim: int) -> tuple[int, float]:
        h = hashlib.md5(feature.encode("utf-8")).digest()
        idx = int.from_bytes(h[:4], "little") % dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for feat in self._features(text):
                idx, sign = self._hash(feat, self.dim)
                out[i, idx] += sign
        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out /= norms
        return out


class SentenceTransformerEmbedder(EmbeddingBackend):
    """Wrap a sentence-transformers model (downloads weights on first use)."""

    def __init__(self, model_name: str, normalize: bool = True, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore

        self._model = SentenceTransformer(model_name)
        self.dim = self._model.get_sentence_embedding_dimension()
        self.normalize = normalize
        self.batch_size = batch_size
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(cfg: dict) -> EmbeddingBackend:
    """Construct an embedder from a config dict; fall back to hashing offline."""
    backend = str(cfg.get("backend", "hash")).lower()
    normalize = bool(cfg.get("normalize", True))
    if backend == "sentence_transformers":
        try:
            return SentenceTransformerEmbedder(
                cfg.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
                normalize=normalize,
                batch_size=int(cfg.get("batch_size", 32)),
            )
        except Exception as exc:
            logger.warning("sentence-transformers unavailable (%s); using HashEmbedder.", exc)
    return HashEmbedder(dim=int(cfg.get("dim", 256)), normalize=normalize)
