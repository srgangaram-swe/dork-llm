"""Training: the optimization loop, LR schedule and checkpoint management."""

from __future__ import annotations

from dork.training.checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
    save_checkpoint,
)
from dork.training.scheduler import cosine_lr
from dork.training.sft import SFTTrainer, build_sft_dataset, format_prompt
from dork.training.trainer import Trainer, resolve_device

__all__ = [
    "SFTTrainer",
    "Trainer",
    "build_sft_dataset",
    "cosine_lr",
    "format_prompt",
    "load_checkpoint",
    "load_model_from_checkpoint",
    "resolve_device",
    "save_checkpoint",
]
