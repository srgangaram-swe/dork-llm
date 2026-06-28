"""Binary token datasets and batching for language-model training.

Following the nanoGPT pattern, the corpus is tokenized once into flat ``uint16``
binary files (``train.bin`` / ``val.bin``) and memory-mapped at train time. This
keeps RAM usage flat regardless of corpus size and makes data loading trivially
fast and reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = get_logger(__name__)

EncodeFn = Callable[[str], list[int]]


def build_token_bins(
    text: str,
    encode: EncodeFn,
    out_dir: str | Path,
    val_fraction: float = 0.1,
) -> dict[str, object]:
    """Tokenize ``text`` and write ``train.bin`` / ``val.bin`` as uint16 arrays.

    Args:
        text: The raw corpus.
        encode: A function mapping a string to a list of token ids.
        out_dir: Directory to write the bins and ``meta`` into.
        val_fraction: Fraction of tokens held out for validation.

    Returns:
        A metadata dict with token counts and file paths.
    """
    out = resolve_path(out_dir, create_parent=True)
    out.mkdir(parents=True, exist_ok=True)

    ids = np.asarray(encode(text), dtype=np.uint32)
    if ids.max(initial=0) >= 2**16:
        raise ValueError("Token id exceeds uint16 range; reduce vocab_size or change dtype.")
    ids = ids.astype(np.uint16)

    n_val = max(1, int(len(ids) * val_fraction))
    train_ids, val_ids = ids[:-n_val], ids[-n_val:]

    train_path = out / "train.bin"
    val_path = out / "val.bin"
    train_ids.tofile(train_path)
    val_ids.tofile(val_path)

    meta = {
        "n_tokens": len(ids),
        "n_train": len(train_ids),
        "n_val": len(val_ids),
        "train_bin": str(train_path),
        "val_bin": str(val_path),
        "dtype": "uint16",
    }
    logger.info(
        "Tokenized corpus: %d tokens (train=%d, val=%d) -> %s",
        meta["n_tokens"],
        meta["n_train"],
        meta["n_val"],
        out,
    )
    return meta


@dataclass
class BinTokenDataset:
    """Memory-mapped view over a tokenized split with random-window batching."""

    path: str | Path
    block_size: int

    def __post_init__(self) -> None:
        self.path = resolve_path(self.path)
        self._data = np.memmap(self.path, dtype=np.uint16, mode="r")
        if len(self._data) <= self.block_size + 1:
            raise ValueError(
                f"Split {self.path} has {len(self._data)} tokens, "
                f"too few for block_size={self.block_size}."
            )

    def __len__(self) -> int:
        return len(self._data)

    def get_batch(self, batch_size: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch of (inputs, targets) shifted by one token."""
        return get_batch(self._data, self.block_size, batch_size, device)


def get_batch(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a causal LM batch: ``x`` and ``y`` are ``x`` shifted left by one.

    Args:
        data: A 1-D array of token ids.
        block_size: Context length.
        batch_size: Number of sequences per batch.
        device: Torch device string.

    Returns:
        ``(x, y)`` int64 tensors, each shaped ``(batch_size, block_size)``.
    """
    import torch

    max_start = len(data) - block_size - 1
    ix = np.random.randint(0, max_start, size=batch_size)
    x = np.stack([data[i : i + block_size].astype(np.int64) for i in ix])
    y = np.stack([data[i + 1 : i + 1 + block_size].astype(np.int64) for i in ix])

    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    if device.startswith("cuda"):
        # Pinned memory + non-blocking transfer for a small throughput win.
        xt = xt.pin_memory().to(device, non_blocking=True)
        yt = yt.pin_memory().to(device, non_blocking=True)
    else:
        xt = xt.to(device)
        yt = yt.to(device)
    return xt, yt
