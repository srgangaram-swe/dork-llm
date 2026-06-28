"""Learning-rate schedule: linear warmup followed by cosine decay to ``min_lr``."""

from __future__ import annotations

import math


def cosine_lr(
    step: int,
    *,
    learning_rate: float,
    min_lr: float,
    warmup_steps: int,
    max_steps: int,
    decay: bool = True,
) -> float:
    """Return the learning rate for ``step``.

    Schedule:
        * ``step < warmup_steps``: linear ramp ``0 -> learning_rate``.
        * ``warmup_steps <= step <= max_steps``: cosine decay to ``min_lr``.
        * ``step > max_steps``: clamp at ``min_lr``.

    With ``decay=False`` the learning rate is constant after warmup.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return learning_rate * (step + 1) / (warmup_steps + 1)
    if not decay:
        return learning_rate
    if step > max_steps:
        return min_lr
    span = max(max_steps - warmup_steps, 1)
    ratio = (step - warmup_steps) / span
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 -> 0
    return min_lr + coeff * (learning_rate - min_lr)
