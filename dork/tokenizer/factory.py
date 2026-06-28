"""Factory helpers to train and load tokenizers from a validated config.

The factory degrades gracefully: if ``type=bpe`` is requested but the
``tokenizers`` package is unavailable, it logs a warning and falls back to the
character tokenizer so the pipeline still runs offline.
"""

from __future__ import annotations

import json
from pathlib import Path

from dork.tokenizer.base import Tokenizer
from dork.tokenizer.bpe import BPETokenizer
from dork.tokenizer.char import CharTokenizer
from dork.utils.config import TokenizerConfig
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)


def train_tokenizer(cfg: TokenizerConfig, text: str) -> Tokenizer:
    """Train (or build) a tokenizer per ``cfg`` and return it (not yet saved)."""
    if cfg.type == "char":
        return CharTokenizer.train(text, cfg.special_tokens)

    try:
        return BPETokenizer.train(text, cfg.vocab_size, cfg.special_tokens)
    except ImportError:
        logger.warning("Falling back to char tokenizer (tokenizers package missing).")
        return CharTokenizer.train(text, cfg.special_tokens)


def load_tokenizer(path: str | Path) -> Tokenizer:
    """Load a tokenizer, auto-detecting char vs BPE from the file contents."""
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No tokenizer at {p}. Run `make train-tokenizer` first.")

    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        payload = {}

    if payload.get("type") == "char":
        return CharTokenizer.load(p)
    return BPETokenizer.load(p)


def load_or_train_tokenizer(cfg: TokenizerConfig, text: str) -> Tokenizer:
    """Load the tokenizer at ``cfg.path`` if present, else train and save it."""
    p = resolve_path(cfg.path)
    if p.exists():
        logger.info("Loading existing tokenizer from %s", p)
        return load_tokenizer(p)
    tok = train_tokenizer(cfg, text)
    tok.save(p)
    logger.info("Saved newly trained tokenizer to %s", p)
    return tok
