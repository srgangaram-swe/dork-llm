"""Character-level tokenizer.

Zero dependencies, zero training cost, fully deterministic — ideal for offline
runs, unit tests and CI. Vocabulary is the sorted set of characters in the
training corpus plus a small set of reserved special tokens.
"""

from __future__ import annotations

import json
from pathlib import Path

from dork.tokenizer.base import Tokenizer
from dork.utils.paths import resolve_path


class CharTokenizer(Tokenizer):
    """Maps each unique character to an integer id."""

    def __init__(self, stoi: dict[str, int], special_tokens: list[str] | None = None) -> None:
        self.stoi = dict(stoi)
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.special_tokens = special_tokens or []
        self.vocab_size = len(self.stoi)
        self.unk_id = self.stoi.get("<|unk|>", 0)

    @classmethod
    def train(cls, text: str, special_tokens: list[str] | None = None) -> CharTokenizer:
        """Build the vocabulary from the unique characters in ``text``."""
        specials = special_tokens or ["<|endoftext|>", "<|pad|>", "<|unk|>"]
        chars = sorted(set(text))
        stoi: dict[str, int] = {}
        # Reserve ids 0..k-1 for specials so they are stable across corpora.
        for tok in specials:
            stoi[tok] = len(stoi)
        for ch in chars:
            if ch not in stoi:
                stoi[ch] = len(stoi)
        return cls(stoi, specials)

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: list[int]) -> str:
        specials = set(self.special_tokens)
        out = []
        for i in ids:
            ch = self.itos.get(int(i), "")
            if ch and ch not in specials:
                out.append(ch)
        return "".join(out)

    def save(self, path: str | Path) -> Path:
        p = resolve_path(path, create_parent=True)
        payload = {
            "type": "char",
            "stoi": self.stoi,
            "special_tokens": self.special_tokens,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        p = resolve_path(path)
        payload = json.loads(p.read_text(encoding="utf-8"))
        if payload.get("type") != "char":
            raise ValueError(f"{p} is not a char tokenizer (type={payload.get('type')!r}).")
        return cls(payload["stoi"], payload.get("special_tokens"))
