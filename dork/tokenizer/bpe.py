"""Byte-level BPE tokenizer backed by Hugging Face ``tokenizers``.

Byte-level BPE (the GPT-2 family scheme) never emits ``<unk>``: any UTF-8 string
round-trips losslessly because the base alphabet is the 256 bytes. We train a
compact vocabulary on the project corpus for a realistic subword tokenizer.
"""

from __future__ import annotations

from pathlib import Path

from dork.tokenizer.base import Tokenizer
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)


class BPETokenizer(Tokenizer):
    """Thin wrapper over a trained ``tokenizers.Tokenizer`` (byte-level BPE)."""

    def __init__(self, backend: object) -> None:
        self._tk = backend
        self.vocab_size = backend.get_vocab_size()  # type: ignore[attr-defined]

    @classmethod
    def train(
        cls,
        text: str,
        vocab_size: int = 4096,
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Train a byte-level BPE tokenizer on ``text``.

        Raises:
            ImportError: If the ``tokenizers`` package (the ``[train]`` extra)
                is not installed.
        """
        try:
            from tokenizers import Tokenizer as HFTokenizer
            from tokenizers import decoders, models, pre_tokenizers, trainers
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ImportError(
                "BPE training requires the `tokenizers` package. "
                "Install with `pip install -e '.[train]'` or use tokenizer.type=char."
            ) from exc

        specials = special_tokens or ["<|endoftext|>", "<|pad|>", "<|unk|>"]
        tk = HFTokenizer(models.BPE(unk_token=None))
        tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tk.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=specials,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tk.train_from_iterator([text], trainer=trainer)
        logger.info("Trained BPE tokenizer: vocab_size=%d", tk.get_vocab_size())
        return cls(tk)

    def encode(self, text: str) -> list[int]:
        return self._tk.encode(text).ids  # type: ignore[attr-defined]

    def decode(self, ids: list[int]) -> str:
        return self._tk.decode(list(ids))  # type: ignore[attr-defined]

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [enc.ids for enc in self._tk.encode_batch(texts)]  # type: ignore[attr-defined]

    def save(self, path: str | Path) -> Path:
        p = resolve_path(path, create_parent=True)
        self._tk.save(str(p))  # type: ignore[attr-defined]
        return p

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        from tokenizers import Tokenizer as HFTokenizer

        p = resolve_path(path)
        return cls(HFTokenizer.from_file(str(p)))
