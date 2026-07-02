"""High-level text generation: tie a :class:`TinyGPT` to a tokenizer.

The :class:`Generator` is the bridge between raw token ids and human-readable
text. It is reused by the CLI, the FastAPI service, the dashboard, and the
evaluation harness so generation behaves identically everywhere. ``torch`` is
imported lazily so this module imports without the ``[train]`` extra.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dork.tokenizer.base import Tokenizer
from dork.utils.config import GenerationConfig
from dork.utils.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover
    from dork.models.tiny_gpt import TinyGPT

logger = get_logger(__name__)


@dataclass
class Generator:
    """Generate text from a trained :class:`TinyGPT` and its tokenizer."""

    model: TinyGPT
    tokenizer: Tokenizer
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model.to(self.device)
        self.model.eval()

    def generate(
        self,
        prompt: str,
        cfg: GenerationConfig | None = None,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        use_cache: bool = True,
    ) -> str:
        """Generate a continuation of ``prompt`` and return only the new text.

        ``use_cache`` selects the fast KV-cached decode path (default); set it to
        False to use the O(T²) reference path (useful for benchmarking/debugging).
        """
        import torch

        cfg = cfg or GenerationConfig()
        if seed is not None:
            torch.manual_seed(seed)

        ids = self.tokenizer.encode(prompt) or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens or cfg.max_new_tokens,
                temperature=temperature if temperature is not None else cfg.temperature,
                top_k=top_k if top_k is not None else cfg.top_k,
                top_p=top_p if top_p is not None else cfg.top_p,
                use_cache=use_cache,
            )
        new_ids = out[0, len(ids) :].tolist()
        return self.tokenizer.decode(new_ids)

    def perplexity(self, text: str, stride: int | None = None) -> float:
        """Compute token-level perplexity of ``text`` under the model.

        Uses a sliding window so sequences longer than ``block_size`` are scored
        with as much left-context as possible (standard LM perplexity protocol).
        """
        import torch
        import torch.nn.functional as F

        block = self.model.config.block_size
        stride = stride or block
        ids = self.tokenizer.encode(text)
        if len(ids) < 2:
            return float("nan")

        data = torch.tensor(ids, dtype=torch.long, device=self.device)
        nll_sum, n_tokens = 0.0, 0
        prev_end = 0
        with torch.no_grad():
            for begin in range(0, len(ids) - 1, stride):
                end = min(begin + block, len(ids) - 1)
                x = data[begin:end].unsqueeze(0)
                y = data[begin + 1 : end + 1].clone().unsqueeze(0)
                # Only score tokens not already counted in a previous window.
                trg_len = end - prev_end
                y[:, :-trg_len] = -1
                logits, _ = self.model(x, y)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                    reduction="sum",
                )
                nll_sum += loss.item()
                n_tokens += trg_len
                prev_end = end
                if end >= len(ids) - 1:
                    break
        return math.exp(nll_sum / max(n_tokens, 1))
