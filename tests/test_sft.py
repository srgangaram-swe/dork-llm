"""Tests for supervised fine-tuning (post-training)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dork.data.instructions import split_instructions, synthetic_instructions  # noqa: E402
from dork.models.tiny_gpt import TinyGPT  # noqa: E402
from dork.training.sft import (  # noqa: E402
    IGNORE_INDEX,
    SFTTrainer,
    build_sft_dataset,
    format_prompt,
)
from dork.utils.config import ModelConfig, TrainingConfig  # noqa: E402

pytestmark = pytest.mark.torch


def test_synthetic_instructions_are_wellformed():
    pairs = synthetic_instructions(n_arith=10, seed=0)
    assert len(pairs) > 10
    assert all("instruction" in p and "response" in p for p in pairs)
    train, val = split_instructions(pairs, val_fraction=0.2, seed=0)
    assert train and val and len(train) + len(val) == len(pairs)


def test_prompt_template_contains_markers():
    p = format_prompt("Do X")
    assert "### Instruction:" in p and "### Response:" in p


def test_sft_dataset_masks_prompt_tokens(char_tokenizer):
    pairs = [{"instruction": "Compute 2 + 2.", "response": "4"}]
    x, y = build_sft_dataset(pairs, char_tokenizer, block_size=64)
    assert x.shape == y.shape and x.shape[0] == 1
    # The prompt region must be ignored; at least one response token is supervised.
    assert (y[0] == IGNORE_INDEX).sum() > 0
    assert (y[0] != IGNORE_INDEX).sum() >= 1
    # Wherever a label is supervised, it must match the input token (teacher forcing).
    supervised = y[0] != IGNORE_INDEX
    assert torch.equal(x[0][supervised], y[0][supervised])


def test_sft_reduces_response_loss(char_tokenizer):
    torch.manual_seed(0)
    pairs = synthetic_instructions(n_arith=20, seed=0)
    train, val = split_instructions(pairs, 0.3, 0)
    train_xy = build_sft_dataset(train, char_tokenizer, 64)
    val_xy = build_sft_dataset(val, char_tokenizer, 64)

    model = TinyGPT(
        ModelConfig(
            vocab_size=char_tokenizer.vocab_size,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=64,
            dropout=0.0,
        )
    )
    cfg = TrainingConfig(
        batch_size=8,
        max_steps=60,
        eval_interval=30,
        warmup_steps=5,
        learning_rate=3e-3,
        device="cpu",
        dtype="float32",
    )
    trainer = SFTTrainer(model, train_xy, val_xy, cfg)
    before = trainer._eval_response_loss(train_xy)
    trainer.train()
    after = trainer._eval_response_loss(train_xy)
    assert after < before, f"SFT should reduce train response loss ({after} !< {before})"
