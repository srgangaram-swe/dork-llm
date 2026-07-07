"""Supervised fine-tuning (instruction tuning) for the tiny GPT.

This is the *post-training* half of the pretrain -> fine-tune paradigm. The two
ideas that make SFT different from pretraining are implemented explicitly:

1. **A prompt template** wraps each example so the model learns a consistent
   instruction/response format.
2. **Response-only loss masking**: prompt and padding tokens are set to the
   ignore index (-1) so the loss is computed on the *response* tokens only — the
   model is trained to produce answers, not to re-predict the prompt.

At educational scale this demonstrates the mechanism and the measurable effect
(held-out response perplexity drops), not a large capability gain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch

from dork.models.tiny_gpt import TinyGPT
from dork.tokenizer.base import Tokenizer
from dork.training.checkpoint import save_checkpoint
from dork.training.scheduler import cosine_lr
from dork.training.trainer import resolve_device
from dork.utils.config import TrainingConfig
from dork.utils.logging import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Response:\n"
IGNORE_INDEX = -1


def format_prompt(instruction: str) -> str:
    """Wrap an instruction in the training/inference prompt template."""
    return PROMPT_TEMPLATE.format(instruction=instruction)


def _eot_id(tokenizer: Tokenizer) -> int:
    """Best-effort end-of-text token id, falling back to 0 (pad)."""
    tid = tokenizer.token_to_id("<|endoftext|>")
    return tid if tid is not None else 0


def build_sft_dataset(
    pairs: list[dict[str, str]],
    tokenizer: Tokenizer,
    block_size: int,
    pad_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize instruction/response pairs into padded ``(input_ids, labels)``.

    Labels mask the prompt and padding with :data:`IGNORE_INDEX` so cross-entropy
    is only taken over the response tokens (plus the end-of-text token).

    Returns:
        ``(X, Y)`` int64 tensors of shape ``(N, T)`` where ``T`` is the longest
        example (capped at ``block_size``).
    """
    eot = _eot_id(tokenizer)
    rows: list[tuple[list[int], list[int]]] = []
    max_len = 1
    for pair in pairs:
        prompt_ids = tokenizer.encode(format_prompt(pair["instruction"]))
        resp_ids = [*tokenizer.encode(pair["response"]), eot]
        ids = (prompt_ids + resp_ids)[:block_size]
        n_prompt = min(len(prompt_ids), len(ids))
        labels = [IGNORE_INDEX] * n_prompt + resp_ids
        labels = labels[: len(ids)]
        rows.append((ids, labels))
        max_len = max(max_len, len(ids))

    x = torch.full((len(rows), max_len), pad_id, dtype=torch.long)
    y = torch.full((len(rows), max_len), IGNORE_INDEX, dtype=torch.long)
    for i, (ids, labels) in enumerate(rows):
        x[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        y[i, : len(labels)] = torch.tensor(labels, dtype=torch.long)
    return x, y


@dataclass
class SFTTrainer:
    """Fine-tune a :class:`TinyGPT` on tokenized instruction data."""

    model: TinyGPT
    train_xy: tuple[torch.Tensor, torch.Tensor]
    val_xy: tuple[torch.Tensor, torch.Tensor]
    cfg: TrainingConfig
    tokenizer_path: str | None = None
    tracker: Any | None = None
    history: list[dict[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.device = resolve_device(self.cfg.device)
        self.model.to(self.device)
        self.optimizer = self.model.configure_optimizers(
            self.cfg.weight_decay,
            self.cfg.learning_rate,
            (self.cfg.beta1, self.cfg.beta2),
            "cuda" if self.device.startswith("cuda") else self.device,
        )

    def _batch(self, xy: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = xy
        idx = torch.randint(0, x.shape[0], (min(self.cfg.batch_size, x.shape[0]),))
        return x[idx].to(self.device), y[idx].to(self.device)

    @torch.no_grad()
    def _eval_response_loss(self, xy: tuple[torch.Tensor, torch.Tensor]) -> float:
        """Mean cross-entropy over *response* tokens for the whole split."""
        self.model.eval()
        x, y = xy[0].to(self.device), xy[1].to(self.device)
        logits, _ = self.model(x, y)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=IGNORE_INDEX
        )
        self.model.train()
        return loss.item()

    def train(self) -> list[dict[str, float]]:
        """Run fine-tuning and return the metric history."""
        cfg = self.cfg
        self.model.train()
        best_val = float("inf")
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

            if step % cfg.eval_interval == 0 or step == cfg.max_steps:
                train_loss = self._eval_response_loss(self.train_xy)
                val_loss = self._eval_response_loss(self.val_xy)
                metrics = {
                    "step": step,
                    "train": train_loss,
                    "val": val_loss,
                    "val_ppl": math.exp(min(val_loss, 20)),
                    "lr": lr,
                }
                self.history.append(metrics)
                if self.tracker is not None:
                    self.tracker.log_metrics(metrics, step=step)
                logger.info(
                    "sft step %d | train %.4f | val %.4f | ppl %.2f",
                    step,
                    train_loss,
                    val_loss,
                    math.exp(min(val_loss, 20)),
                )
                if val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(
                        cfg.out_dir,
                        self.model,
                        step=step,
                        best_val_loss=best_val,
                        optimizer=self.optimizer,
                        tokenizer_path=self.tokenizer_path,
                        extra={"stage": "sft", "history": self.history},
                    )

            if step == cfg.max_steps:
                break

            x, y = self._batch(self.train_xy)
            _, loss = self.model(x, y)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()

        logger.info("SFT complete. Best val response-loss: %.4f", best_val)
        return self.history
