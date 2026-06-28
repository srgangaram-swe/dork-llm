"""The tiny GPT training loop with validation, checkpointing and LR scheduling.

The :class:`Trainer` is framework-light and synchronous so the full optimization
loop — forward, loss, backward, grad-clip, optimizer step, periodic eval and
best-checkpoint saving — is visible in one place. It supports gradient
accumulation, mixed precision (bf16/fp16) and device auto-selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from dork.data.loader import BinTokenDataset
from dork.models.tiny_gpt import TinyGPT
from dork.training.checkpoint import save_checkpoint
from dork.training.scheduler import cosine_lr
from dork.utils.config import TrainingConfig
from dork.utils.logging import get_logger

logger = get_logger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Pick a concrete device string from ``auto`` (cuda > mps > cpu)."""
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(requested: str, device: str) -> torch.dtype:
    """Choose an autocast dtype, defaulting to bf16 on capable hardware."""
    if requested == "float32":
        return torch.float32
    if requested == "bfloat16":
        return torch.bfloat16
    if requested == "float16":
        return torch.float16
    # auto
    if device == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@dataclass
class Trainer:
    """Train a :class:`TinyGPT` on memory-mapped token bins."""

    model: TinyGPT
    train_ds: BinTokenDataset
    val_ds: BinTokenDataset
    cfg: TrainingConfig
    tokenizer_path: str | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.cfg.device)
        self.device_type = "cuda" if self.device.startswith("cuda") else self.device
        self.dtype = resolve_dtype(self.cfg.dtype, self.device)
        self.model.to(self.device)
        if self.cfg.compile and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile ...")
            self.model = torch.compile(self.model)  # type: ignore[assignment]
        self.optimizer = self.model.configure_optimizers(
            self.cfg.weight_decay,
            self.cfg.learning_rate,
            (self.cfg.beta1, self.cfg.beta2),
            self.device_type,
        )
        # GradScaler only matters for fp16 on CUDA. Use the modern torch.amp API
        # when available, falling back for older torch versions.
        enabled = self.dtype == torch.float16 and self.device_type == "cuda"
        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=enabled)
        except (AttributeError, TypeError):  # pragma: no cover - old torch
            self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)
        logger.info("Trainer ready on device=%s dtype=%s", self.device, self.dtype)

    def _autocast(self):
        if self.device_type in {"cuda", "cpu"} and self.dtype != torch.float32:
            return torch.autocast(device_type=self.device_type, dtype=self.dtype)
        from contextlib import nullcontext

        return nullcontext()

    @torch.no_grad()
    def estimate_loss(self) -> dict[str, float]:
        """Average loss over ``eval_iters`` batches for both splits."""
        self.model.eval()
        out: dict[str, float] = {}
        for name, ds in (("train", self.train_ds), ("val", self.val_ds)):
            losses = torch.zeros(self.cfg.eval_iters)
            for i in range(self.cfg.eval_iters):
                x, y = ds.get_batch(self.cfg.batch_size, self.device)
                with self._autocast():
                    _, loss = self.model(x, y)
                losses[i] = loss.item()
            out[name] = losses.mean().item()
        self.model.train()
        return out

    def train(self) -> list[dict[str, float]]:
        """Run the optimization loop and return the metric history."""
        cfg = self.cfg
        best_val = float("inf")
        self.model.train()
        accum = max(cfg.gradient_accumulation_steps, 1)

        for step in range(cfg.max_steps + 1):
            lr = cosine_lr(
                step,
                learning_rate=cfg.learning_rate,
                min_lr=cfg.min_lr,
                warmup_steps=cfg.warmup_steps,
                max_steps=cfg.max_steps,
                decay=cfg.decay_lr,
            )
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Periodic evaluation + checkpointing.
            if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                metrics = self.estimate_loss()
                metrics.update({"step": step, "lr": lr})
                self.history.append(metrics)
                logger.info(
                    "step %d | train %.4f | val %.4f | lr %.2e",
                    step,
                    metrics["train"],
                    metrics["val"],
                    lr,
                )
                if metrics["val"] < best_val or cfg.always_save_checkpoint:
                    best_val = min(best_val, metrics["val"])
                    save_checkpoint(
                        cfg.out_dir,
                        self.model,
                        step=step,
                        best_val_loss=best_val,
                        optimizer=self.optimizer,
                        tokenizer_path=self.tokenizer_path,
                        extra={"history": self.history},
                    )

            if step == cfg.max_steps:
                break

            # Gradient accumulation micro-steps.
            for _micro in range(accum):
                x, y = self.train_ds.get_batch(cfg.batch_size, self.device)
                with self._autocast():
                    _, loss = self.model(x, y)
                    loss = loss / accum
                self.scaler.scale(loss).backward()

            if cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_interval == 0 and step > 0:
                logger.debug("step %d | loss %.4f | lr %.2e", step, loss.item() * accum, lr)

        logger.info("Training complete. Best val loss: %.4f", best_val)
        return self.history


def latest_checkpoint_path(out_dir: str | Path) -> Path:
    """Return the conventional checkpoint path for ``out_dir``."""
    from dork.training.checkpoint import CKPT_NAME
    from dork.utils.paths import resolve_path

    return resolve_path(out_dir) / CKPT_NAME
