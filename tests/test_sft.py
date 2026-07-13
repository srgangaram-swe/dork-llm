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


def test_sft_dataset_shifts_targets_and_masks_prompt(char_tokenizer):
    pairs = [{"instruction": "Explain attention.", "response": "stage"}]
    x, y = build_sft_dataset(pairs, char_tokenizer, block_size=64)
    assert x.shape == y.shape and x.shape[0] == 1

    prompt_ids = char_tokenizer.encode(format_prompt(pairs[0]["instruction"]))
    response_ids = [
        *char_tokenizer.encode(pairs[0]["response"]),
        char_tokenizer.token_to_id("<|endoftext|>") or 0,
    ]
    sequence = (prompt_ids + response_ids)[:65]
    expected_x = torch.tensor(sequence[:-1])
    expected_y = torch.tensor(sequence[1:])
    expected_y[: len(prompt_ids) - 1] = IGNORE_INDEX

    assert torch.equal(x[0, : len(expected_x)], expected_x)
    assert torch.equal(y[0, : len(expected_y)], expected_y)
    supervised = y[0] != IGNORE_INDEX
    first_supervised = supervised.nonzero()[0].item()
    assert first_supervised == len(prompt_ids) - 1
    assert y[0, first_supervised].item() == response_ids[0]
    assert x[0, first_supervised].item() == prompt_ids[-1]
    assert x[0, first_supervised].item() != y[0, first_supervised].item()


def test_sft_dataset_truncates_prompt_before_dropping_response_targets(char_tokenizer):
    response = "answer"
    x, y = build_sft_dataset(
        [{"instruction": "This instruction is much too long to fit.", "response": response}],
        char_tokenizer,
        block_size=8,
    )
    supervised = y[0][y[0] != IGNORE_INDEX]
    assert x.shape == y.shape == (1, 8)
    assert supervised[0].item() == char_tokenizer.encode(response)[0]
    assert supervised[-1].item() == (char_tokenizer.token_to_id("<|endoftext|>") or 0)


def test_sft_reduces_response_loss_without_touching_project_artifacts(char_tokenizer, tmp_path):
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
        out_dir=str(tmp_path / "sft"),
    )
    trainer = SFTTrainer(model, train_xy, val_xy, cfg)
    before = trainer._eval_response_loss(train_xy)
    trainer.train()
    after = trainer._eval_response_loss(train_xy)
    assert after < before, f"SFT should reduce train response loss ({after} !< {before})"
    assert (tmp_path / "sft" / "ckpt.pt").exists()
