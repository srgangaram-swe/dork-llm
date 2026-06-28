"""Checkpoint save/load for the tiny GPT.

A checkpoint is a single ``ckpt.pt`` bundling the model weights, the exact model
config needed to reconstruct the architecture, optimizer state (for resuming),
and training metadata. This makes runs reproducible and models self-describing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from dork.models.tiny_gpt import TinyGPT
from dork.utils.config import ModelConfig
from dork.utils.logging import get_logger
from dork.utils.paths import resolve_path

logger = get_logger(__name__)

CKPT_NAME = "ckpt.pt"


def save_checkpoint(
    out_dir: str | Path,
    model: TinyGPT,
    *,
    step: int,
    best_val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
    tokenizer_path: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a checkpoint to ``out_dir/ckpt.pt`` and return its path."""
    out = resolve_path(out_dir, create_parent=True)
    out.mkdir(parents=True, exist_ok=True)
    path = out / CKPT_NAME
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config.model_dump(),
        "step": step,
        "best_val_loss": best_val_loss,
        "tokenizer_path": tokenizer_path,
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)
    logger.info("Saved checkpoint to %s (step=%d, val_loss=%.4f)", path, step, best_val_loss)
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict[str, Any]:
    """Load the raw checkpoint payload."""
    p = resolve_path(path)
    if p.is_dir():
        p = p / CKPT_NAME
    if not p.exists():
        raise FileNotFoundError(f"No checkpoint at {p}. Train a model first.")
    return torch.load(p, map_location=map_location, weights_only=False)


def load_model_from_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[TinyGPT, dict[str, Any]]:
    """Reconstruct a :class:`TinyGPT` from a checkpoint and move it to ``device``."""
    payload = load_checkpoint(path, map_location=device)
    config = ModelConfig.model_validate(payload["model_config"])
    model = TinyGPT(config)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info("Loaded model from %s (step=%s)", path, payload.get("step"))
    return model, payload
