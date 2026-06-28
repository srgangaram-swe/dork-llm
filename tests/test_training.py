"""Integration test: the model can actually learn (loss decreases)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from dork.data.loader import BinTokenDataset, build_token_bins  # noqa: E402
from dork.models.tiny_gpt import TinyGPT  # noqa: E402
from dork.training.checkpoint import load_model_from_checkpoint  # noqa: E402
from dork.training.scheduler import cosine_lr  # noqa: E402
from dork.training.trainer import Trainer  # noqa: E402
from dork.utils.config import ModelConfig, TrainingConfig  # noqa: E402

pytestmark = pytest.mark.torch


def test_cosine_schedule_shape():
    warm = cosine_lr(0, learning_rate=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    peak = cosine_lr(10, learning_rate=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    end = cosine_lr(100, learning_rate=1e-3, min_lr=1e-4, warmup_steps=10, max_steps=100)
    assert warm < peak
    assert abs(peak - 1e-3) < 1e-9
    assert abs(end - 1e-4) < 1e-6


@pytest.mark.slow
def test_training_reduces_loss_and_checkpoints(tmp_path, sample_text, char_tokenizer):
    meta = build_token_bins(sample_text, char_tokenizer.encode, tmp_path / "bins", val_fraction=0.2)
    block = 16
    train_ds = BinTokenDataset(meta["train_bin"], block)
    val_ds = BinTokenDataset(meta["val_bin"], block)

    model = TinyGPT(
        ModelConfig(
            vocab_size=char_tokenizer.vocab_size,
            block_size=block,
            n_layer=2,
            n_head=2,
            n_embd=32,
            dropout=0.0,
        )
    )
    cfg = TrainingConfig(
        batch_size=8,
        max_steps=40,
        eval_interval=20,
        eval_iters=3,
        warmup_steps=5,
        out_dir=str(tmp_path / "ckpt"),
        device="cpu",
        dtype="float32",
        learning_rate=3e-3,
    )
    history = Trainer(model, train_ds, val_ds, cfg, str(tmp_path / "tok.json")).train()

    assert history[-1]["train"] < history[0]["train"], "training loss should decrease"
    # A checkpoint should have been written and be reloadable.
    reloaded, payload = load_model_from_checkpoint(tmp_path / "ckpt", device="cpu")
    assert payload["step"] >= 0
    assert reloaded.config.n_layer == 2
