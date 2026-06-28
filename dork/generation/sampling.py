"""Token sampling primitives: temperature, top-k, and nucleus (top-p) filtering.

These are pure functions over a logits tensor so they can be unit-tested in
isolation and reused by both the model's ``generate`` loop and the eval harness.
``torch`` is imported lazily inside each function so this module (and the wider
``dork.generation`` package) imports without the ``[train]`` extra installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Mask all but the ``top_k`` highest-logit tokens (per row) to ``-inf``."""
    import torch

    if top_k <= 0:
        return logits
    k = min(top_k, logits.size(-1))
    kth_value = torch.topk(logits, k, dim=-1).values[..., -1, None]
    return logits.masked_fill(logits < kth_value, float("-inf"))


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Nucleus filtering: keep the smallest set of tokens whose cumulative
    probability mass is >= ``top_p``; mask the rest to ``-inf``.
    """
    import torch
    import torch.nn.functional as F

    if not (0.0 < top_p < 1.0):
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens once cumulative mass exceeds top_p, but always keep the top-1.
    remove = cumulative > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    # Scatter back to the original vocabulary order.
    return torch.empty_like(logits).scatter_(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
) -> torch.Tensor:
    """Sample one next-token id per row from final-position ``logits``.

    Args:
        logits: Tensor of shape ``(B, vocab_size)``.
        temperature: 0 -> greedy argmax; >0 -> scaled sampling.
        top_k: Optional top-k truncation.
        top_p: Optional nucleus truncation.

    Returns:
        A ``(B, 1)`` int64 tensor of sampled token ids.
    """
    import torch
    import torch.nn.functional as F

    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / max(temperature, 1e-8)
    if top_k:
        logits = apply_top_k(logits, top_k)
    if top_p:
        logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)
