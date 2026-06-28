"""Abstract tokenizer interface shared by the char and BPE implementations."""

from __future__ import annotations

import abc
from pathlib import Path


class Tokenizer(abc.ABC):
    """Minimal encode/decode contract every Dork tokenizer satisfies."""

    #: Number of distinct token ids (== embedding table rows).
    vocab_size: int

    @abc.abstractmethod
    def encode(self, text: str) -> list[int]:
        """Convert a string into a list of integer token ids."""

    @abc.abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Convert token ids back into a string."""

    @abc.abstractmethod
    def save(self, path: str | Path) -> Path:
        """Persist the tokenizer to ``path``."""

    @classmethod
    @abc.abstractmethod
    def load(cls, path: str | Path) -> Tokenizer:
        """Load a tokenizer previously written by :meth:`save`."""

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        """Encode many strings (override for vectorized backends)."""
        return [self.encode(t) for t in texts]
